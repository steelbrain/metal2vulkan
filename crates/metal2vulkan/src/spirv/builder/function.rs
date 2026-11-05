//! The minimal `FunctionBuilder`: parameters, `Function` variables, loads and stores, and the
//! straight-line terminators. The per-family constructors live in `ops/` and the construct openers
//! in `constructs.rs`; they extend this type in their own files.

use crate::spirv::builder::constructs::ConstructState;
use crate::spirv::builder::module::{ModuleBuilder, NarrowFmt, NotedBits, StructOp};
use crate::spirv::error::{BResult, BuildError, ControlRule, Dangling, OperandRule, TypeRule};
use crate::spirv::ids::{BlockId, ConstId, FuncId, TypeId, Value, VarId};
use crate::spirv::ir::{
    Inst, InstNode, LocalVar, MemAccess, Merge, Shape, StorageClass, Terminator, Type, TypeFacts,
    ValueOrigin,
};

/// The most struct constructs, inserts and extracts [`FunctionBuilder::member_source`] follows
/// for one value: a bound on the walk (a longer chain stands for no known member).
const MEMBER_STEPS: usize = 64;

/// Builds one function of a [`ModuleBuilder`]. The cursor is the open block that instructions
/// append to; a terminator closes it. Call [`FunctionBuilder::finish`] when done: a function that
/// is never finished makes `finalize` fail.
#[derive(Debug)]
pub struct FunctionBuilder<'m> {
    m: &'m mut ModuleBuilder,
    func: FuncId,
    cursor: Option<BlockId>,
    /// Open loops and switches (`constructs.rs`).
    pub(super) cs: ConstructState,
}

impl<'m> FunctionBuilder<'m> {
    pub(super) fn new(m: &'m mut ModuleBuilder, func: FuncId, entry: BlockId) -> Self {
        FunctionBuilder {
            m,
            func,
            cursor: Some(entry),
            cs: ConstructState::default(),
        }
    }

    /// This function's id.
    pub fn id(&self) -> FuncId {
        self.func
    }

    /// The module, for interning types and constants mid-function.
    pub fn module(&mut self) -> &mut ModuleBuilder {
        self.m
    }

    /// The module, read-only (for the constructor files).
    pub(super) fn mb(&self) -> &ModuleBuilder {
        self.m
    }

    /// The open block, if any.
    pub fn current_block(&self) -> Option<BlockId> {
        self.cursor
    }

    /// The function's entry block, which dominates every block of the function.
    pub fn entry_block(&self) -> Option<BlockId> {
        self.m
            .functions
            .get(self.func.index())
            .and_then(|f| f.blocks.first().copied())
    }

    /// The type of a value.
    pub fn value_ty(&self, v: Value) -> BResult<TypeId> {
        self.m.value_ty(v)
    }

    /// Note that each lane's low 16 bits of `bits` are the `fmt` bits of that lane of `x`, a
    /// narrow float held widened, and that the bits above them are zero but in the lanes `dirty`
    /// has a bit for. With `everywhere` the note holds wherever `x` is (`bits` dominates `x`: `x`
    /// was computed from it), else only in the open block (`bits` was computed from `x`). A note
    /// that holds everywhere is never replaced by one that holds in a block.
    pub fn note_bits(
        &mut self,
        fmt: NarrowFmt,
        x: Value,
        bits: Value,
        dirty: u32,
        everywhere: bool,
    ) {
        let at = if everywhere { None } else { self.cursor };
        if !everywhere && at.is_none() {
            return;
        }
        let entry = self
            .m
            .bits
            .entry((fmt, x))
            .or_insert(NotedBits { bits, dirty, at });
        if entry.at.is_some() {
            *entry = NotedBits { bits, dirty, at };
        }
    }

    /// The noted `fmt` bits of `x` that hold in the open block, and their dirty lanes.
    pub fn bits_of(&self, fmt: NarrowFmt, x: Value) -> Option<(Value, u32)> {
        self.m
            .bits
            .get(&(fmt, x))
            .filter(|n| n.at.is_none() || n.at == self.cursor)
            .map(|n| (n.bits, n.dirty))
    }

    /// The noted `fmt` bits `x` was decoded from, which hold wherever `x` is (a note made with
    /// `everywhere`: `x` is the decode of `bits`), and their dirty lanes.
    pub fn decoded_from(&self, fmt: NarrowFmt, x: Value) -> Option<(Value, u32)> {
        self.m
            .bits
            .get(&(fmt, x))
            .filter(|n| n.at.is_none())
            .map(|n| (n.bits, n.dirty))
    }

    /// Note that `Function` variable `v` holds one value across the workgroup wherever it is
    /// read, by the emitter's construction (`TypedModule::uniform_vars`). It changes no emitted
    /// word.
    #[cfg(test)]
    pub fn note_uniform_var(&mut self, v: VarId) {
        self.m.uniform_vars.insert(v);
    }

    /// Note that `y` is lane `index` of vector `of`: `of` dominates `y`.
    pub fn note_lane(&mut self, y: Value, of: Value, index: u32) {
        self.m.lanes.insert(y, (of, index));
    }

    /// The vector and lane `y` was noted to read.
    pub fn lane_of(&self, y: Value) -> Option<(Value, u32)> {
        self.m.lanes.get(&y).copied()
    }

    /// Note that `y`, a 32-bit unsigned integer, is the zero extension of 16-bit integer `of`:
    /// `of` dominates `y`.
    pub fn note_widened16(&mut self, y: Value, of: Value) {
        self.m.widened16.insert(y, of);
    }

    /// The 16-bit integer `y` was noted to zero-extend.
    pub fn widened16_of(&self, y: Value) -> Option<Value> {
        self.m.widened16.get(&y).copied()
    }

    /// Note that `y`, a 32-bit float, is the widening of 16-bit float `of`: `of` dominates `y`. A
    /// widening of a narrowing ([`FunctionBuilder::note_narrowed_float`]) is a rounding's second
    /// half, which is not noted, so the pair stays whole.
    pub(super) fn note_widened_half(&mut self, y: Value, of: Value) {
        if !self.m.narrowed_floats.contains(&of) {
            self.m.widened_halves.insert(y, of);
        }
    }

    /// Note that `y` is a conversion of a 32-bit float to 16 bits.
    pub(super) fn note_narrowed_float(&mut self, y: Value) {
        self.m.narrowed_floats.insert(y);
    }

    /// The 16-bit float `y` was noted to widen.
    pub(super) fn widened_half_of(&self, y: Value) -> Option<Value> {
        self.m.widened_halves.get(&y).copied()
    }

    /// Note that vector `y` is vector `v` with lane `index` replaced by `e`: `v` and `e` dominate
    /// `y`.
    pub fn note_insert(&mut self, y: Value, v: Value, e: Value, index: u32) {
        self.m.inserts.insert(y, (v, e, index));
    }

    /// The vector, lane value, and lane index `y` was noted to insert.
    pub fn insert_of(&self, y: Value) -> Option<(Value, Value, u32)> {
        self.m.inserts.get(&y).copied()
    }

    /// Record how struct `y` holds its members, or which member of a struct it reads
    /// ([`FunctionBuilder::member_source`]).
    pub(super) fn note_struct(&mut self, y: Value, op: StructOp) {
        let heap = match &op {
            StructOp::Construct(parts) => std::mem::size_of_val::<[Value]>(parts),
            StructOp::Insert { indices, .. } | StructOp::Extract { indices, .. } => {
                std::mem::size_of_val::<[u32]>(indices)
            }
        };
        self.m.struct_bytes = self.m.struct_bytes.saturating_add(heap);
        self.m.structs.insert(y, op);
    }

    /// The value `y` holds where it is a member of a struct the builder made: `y` extracts the
    /// member at a path of a struct whose constructs and inserts, through further extracts of
    /// structs, supply that member wholly from one value, which dominates `y` (each is an operand
    /// of the instructions that made the struct). `None` where the member is no such value (a
    /// struct the builder did not make, a load or a phi of one, a path into a vector, a later
    /// insert into part of it), or past [`MEMBER_STEPS`] steps.
    pub fn member_source(&self, y: Value) -> Option<Value> {
        let Some(StructOp::Extract { composite, indices }) = self.m.structs.get(&y) else {
            return None;
        };
        let (mut at, mut path) = (*composite, indices.to_vec());
        for _ in 0..MEMBER_STEPS {
            let Some((&first, rest)) = path.split_first() else {
                return Some(at);
            };
            match self.m.structs.get(&at)? {
                StructOp::Construct(parts) => {
                    at = *parts.get(first as usize)?;
                    path = rest.to_vec();
                }
                StructOp::Insert {
                    object,
                    composite,
                    indices,
                } => {
                    if path.starts_with(indices) {
                        at = *object;
                        path.drain(..indices.len());
                    } else if indices.starts_with(&path) {
                        return None;
                    } else {
                        at = *composite;
                    }
                }
                StructOp::Extract { composite, indices } => {
                    let mut full = indices.to_vec();
                    full.extend_from_slice(&path);
                    path = full;
                    at = *composite;
                }
            }
        }
        None
    }

    /// A new group for [`FunctionBuilder::offer`].
    pub fn offer_group(&mut self) -> u32 {
        let g = self.m.offer_groups;
        self.m.offer_groups = g.saturating_add(1);
        g
    }

    /// Offer `to` for `from` in `group`: `from` is a removable instruction's result, `to` holds
    /// what it holds wherever it is, and `to` was emitted before `from`, so `from`'s instruction
    /// may become a copy of `to` if that makes the module smaller.
    pub fn offer(&mut self, group: u32, from: Value, to: Value) {
        self.m
            .offers
            .push(crate::spirv::dead::Offer { group, from, to });
    }

    /// Note `bits`, the encode of narrow vector `v` computed in the open block for its lanes'
    /// stores, with the offer group of those stores.
    pub fn note_encode(&mut self, fmt: NarrowFmt, v: Value, bits: Value, group: u32) {
        if let Some(at) = self.cursor {
            self.m.encodes.insert((fmt, v), (bits, at, group));
        }
    }

    /// The encode of `v` noted in the open block, and its offer group.
    pub fn encode_of(&self, fmt: NarrowFmt, v: Value) -> Option<(Value, u32)> {
        self.m
            .encodes
            .get(&(fmt, v))
            .filter(|e| Some(e.1) == self.cursor)
            .map(|e| (e.0, e.2))
    }

    /// Append a parameter of type `ty`.
    pub fn param(&mut self, ty: TypeId) -> BResult<Value> {
        if matches!(self.m.types().get(ty)?, Type::Void | Type::Function { .. }) {
            return Err(BuildError::Type(TypeRule::FunctionSignature));
        }
        let index = self.m.draft(self.func)?.params.len();
        let index = u32::try_from(index).map_err(|_| BuildError::IdSpaceExhausted)?;
        let v = self.m.new_value(
            ty,
            ValueOrigin::Param {
                func: self.func,
                index,
            },
        )?;
        let d = self.draft_mut()?;
        d.params.push(v);
        d.param_tys.push(ty);
        Ok(v)
    }

    /// Declare a `Function`-storage variable holding `ty`, optionally initialized. It is declared
    /// at the top of the entry block whichever block is open; loop-carried and demoted values live
    /// in such variables.
    pub fn local_var(&mut self, ty: TypeId, init: Option<ConstId>) -> BResult<VarId> {
        if let Some(c) = init {
            if self.m.consts().get(c)?.ty != ty {
                return Err(BuildError::Control(ControlRule::LocalType));
            }
        }
        let ptr_ty = self
            .m
            .ty(Type::Pointer {
                class: StorageClass::Function,
                pointee: ty,
            })
            .map_err(|e| match e {
                BuildError::Type(TypeRule::PointeeForClass(_)) => {
                    BuildError::Control(ControlRule::LocalType)
                }
                e => e,
            })?;
        let id = u32::try_from(self.m.vars.len())
            .map(VarId::new)
            .map_err(|_| BuildError::IdSpaceExhausted)?;
        let value = self.m.new_value(ptr_ty, ValueOrigin::Var(id))?;
        self.m.vars.push(LocalVar {
            func: self.func,
            pointee: ty,
            ptr_ty,
            value,
            init,
        });
        self.draft_mut()?.vars.push(id);
        Ok(id)
    }

    /// The pointer value of a `Function` variable of this function.
    pub fn var_ptr(&self, var: VarId) -> BResult<Value> {
        match self.m.vars.get(var.index()) {
            Some(v) if v.func == self.func => Ok(v.value),
            Some(_) | None => Err(BuildError::Dangling(Dangling::Var(var))),
        }
    }

    /// `OpLoad`: the result has the pointee type.
    pub fn load(&mut self, ptr: Value, access: MemAccess) -> BResult<Value> {
        const OP: crate::spv::Op = crate::spv::Op::Load;
        let (class, pointee, f) = self.pointer(OP, ptr)?;
        not_texel_pointer(OP, class)?;
        check_access(OP, class, access)?;
        if !loadable(&f) {
            return Err(operand(OP, OperandRule::Opaque));
        }
        let v = self.append(Inst::Load { ptr, access }, Some(pointee))?;
        v.ok_or(BuildError::IdSpaceExhausted)
    }

    /// `OpStore`: `value` must have the pointee type, and the storage class must be writable.
    pub fn store(&mut self, ptr: Value, value: Value, access: MemAccess) -> BResult<()> {
        const OP: crate::spv::Op = crate::spv::Op::Store;
        let (class, pointee, f) = self.pointer(OP, ptr)?;
        if matches!(
            class,
            StorageClass::Input
                | StorageClass::UniformConstant
                | StorageClass::PushConstant
                | StorageClass::Uniform
        ) {
            return Err(operand(OP, OperandRule::ReadOnlyClass(class)));
        }
        not_texel_pointer(OP, class)?;
        check_access(OP, class, access)?;
        if f.shape == Shape::Opaque || !loadable(&f) {
            return Err(operand(OP, OperandRule::Opaque));
        }
        if self.m.value_ty(value)? != pointee {
            return Err(operand(OP, OperandRule::StoredType));
        }
        self.append(Inst::Store { ptr, value, access }, None)?;
        Ok(())
    }

    /// `OpReturn` (`v = None`, void function) or `OpReturnValue`.
    pub fn ret(&mut self, v: Option<Value>) -> BResult<()> {
        let ret = self.m.draft(self.func)?.ret;
        let void = matches!(self.m.types().get(ret)?, Type::Void);
        let term = match v {
            None if void => Terminator::Return,
            Some(v) if !void && self.m.value_ty(v)? == ret => Terminator::ReturnValue(v),
            _ => return Err(BuildError::Control(ControlRule::ReturnType)),
        };
        self.terminate(None, term)
    }

    /// `OpUnreachable`.
    pub fn unreachable(&mut self) -> BResult<()> {
        self.terminate(None, Terminator::Unreachable)
    }

    /// Finish the function: every block must be terminated.
    pub fn finish(self) -> BResult<FuncId> {
        if let Some(open) = self.cursor {
            return Err(BuildError::Control(ControlRule::UnterminatedBlock(open)));
        }
        let d = self.m.draft(self.func)?;
        for b in &d.blocks {
            match self.m.blocks.get(b.index()) {
                Some(bd) if bd.term.is_some() => {}
                Some(_) => return Err(BuildError::Control(ControlRule::UnterminatedBlock(*b))),
                None => return Err(BuildError::Dangling(Dangling::Block(*b))),
            }
        }
        let fn_ty = Type::Function {
            ret: d.ret,
            params: d.param_tys.clone().into_boxed_slice(),
        };
        let fn_ty = self.m.ty(fn_ty)?;
        let func = self.func;
        self.m
            .functions
            .get_mut(func.index())
            .ok_or(BuildError::Dangling(Dangling::Func(func)))?
            .fn_ty = Some(fn_ty);
        Ok(func)
    }

    // ---- Plumbing shared with the constructor and construct files ----

    /// The storage class, pointee, and pointee facts of pointer operand `ptr`.
    pub(super) fn pointer(
        &self,
        op: crate::spv::Op,
        ptr: Value,
    ) -> BResult<(StorageClass, TypeId, TypeFacts)> {
        match self.m.types().get(self.m.value_ty(ptr)?)? {
            Type::Pointer { class, pointee } => {
                Ok((*class, *pointee, *self.m.types().facts(*pointee)?))
            }
            _ => Err(operand(op, OperandRule::NotPointer)),
        }
    }

    /// Append `inst` to the open block. A result of type `result` is allocated when given.
    /// Inside a switch-at-use, `inst` must not be tangled.
    pub(super) fn append(&mut self, inst: Inst, result: Option<TypeId>) -> BResult<Option<Value>> {
        let b = self
            .cursor
            .ok_or(BuildError::Control(ControlRule::NoCurrentBlock))?;
        if inst.is_tangled() && self.cs.in_switch_at_use() {
            return Err(BuildError::Control(ControlRule::TangledInSwitchAtUse));
        }
        let ty = result;
        let result = match result {
            Some(ty) => Some(self.m.new_value(ty, ValueOrigin::Inst(b))?),
            None => None,
        };
        if let (Some(r), Some(ty)) = (result, ty) {
            let opaque = matches!(
                self.m.types().get(ty)?,
                Type::Image(_) | Type::Sampler | Type::SampledImage { .. }
            );
            if inst.carries_non_uniform(|v| self.m.non_uniform.contains(&v), opaque) {
                self.m.non_uniform.insert(r);
            }
        }
        let boxed = inst.heap_bytes();
        let insts = &mut self.block_mut(b)?.insts;
        let before = insts.capacity();
        insts.push(InstNode { result, inst });
        let after = insts.capacity();
        self.m.count_block_growth::<InstNode>(before, after);
        self.m.block_bytes = self.m.block_bytes.saturating_add(boxed);
        Ok(result)
    }

    /// Make `b`, a fresh block of this function, the open block and place it next in layout
    /// order. Blocks are laid out when entered, so the layout is the construct preorder (header,
    /// contents, continue, merge) and every block follows its dominators.
    pub(super) fn enter(&mut self, b: BlockId) -> BResult<()> {
        if let Some(open) = self.cursor {
            return Err(BuildError::Control(ControlRule::UnterminatedBlock(open)));
        }
        self.block_mut(b)?;
        self.draft_mut()?.blocks.push(b);
        self.cursor = Some(b);
        Ok(())
    }

    /// Close the open block with `term` (and an optional merge declaration).
    pub(super) fn terminate(&mut self, merge: Option<Merge>, term: Terminator) -> BResult<()> {
        let b = self
            .cursor
            .ok_or(BuildError::Control(ControlRule::NoCurrentBlock))?;
        let bd = self.block_mut(b)?;
        bd.merge = merge;
        bd.term = Some(term);
        self.cursor = None;
        Ok(())
    }

    pub(super) fn block_mut(&mut self, b: BlockId) -> BResult<&mut super::module::BlockDraft> {
        let func = self.func;
        match self.m.blocks.get_mut(b.index()) {
            Some(bd) if bd.func == func => Ok(bd),
            Some(_) => Err(BuildError::Control(ControlRule::ForeignBlock(b))),
            None => Err(BuildError::Dangling(Dangling::Block(b))),
        }
    }

    fn draft_mut(&mut self) -> BResult<&mut super::module::FunctionDraft> {
        let func = self.func;
        self.m
            .functions
            .get_mut(func.index())
            .ok_or(BuildError::Dangling(Dangling::Func(func)))
    }
}

pub(super) fn operand(op: crate::spv::Op, rule: OperandRule) -> BuildError {
    BuildError::Operand { op, rule }
}

/// A whole value of this type can be loaded or stored: data of fixed size (opaque values load but
/// never store; the store check adds that).
pub(super) fn loadable(f: &TypeFacts) -> bool {
    !f.runtime_sized
        && !matches!(
            f.shape,
            Shape::NotData | Shape::DescriptorArray { .. } | Shape::RayQuery
        )
}

/// An `Image`-class pointer comes only from `OpImageTexelPointer` and is valid only as an atomic
/// operand.
pub(super) fn not_texel_pointer(op: crate::spv::Op, class: StorageClass) -> BResult<()> {
    if class == StorageClass::Image {
        Err(operand(op, OperandRule::Class(class)))
    } else {
        Ok(())
    }
}

pub(super) fn check_access(
    op: crate::spv::Op,
    class: StorageClass,
    access: MemAccess,
) -> BResult<()> {
    match access.aligned {
        Some(a) if a.is_power_of_two() => {}
        None if class != StorageClass::PhysicalStorageBuffer => {}
        _ => return Err(operand(op, OperandRule::Alignment)),
    }
    // `MakePointerAvailable` is a store's, `MakePointerVisible` a load's; both need
    // `NonPrivatePointer`, which names only memory other invocations share.
    let (avail, visible) = (access.make_available, access.make_visible);
    let misplaced = (avail.is_some() && op == crate::spv::Op::Load)
        || (visible.is_some() && op == crate::spv::Op::Store)
        || ((avail.is_some() || visible.is_some()) && !access.non_private);
    if misplaced {
        return Err(operand(op, OperandRule::Semantics));
    }
    let shared = matches!(
        class,
        StorageClass::StorageBuffer
            | StorageClass::PhysicalStorageBuffer
            | StorageClass::Uniform
            | StorageClass::Workgroup
    );
    if access.non_private && !shared {
        return Err(operand(op, OperandRule::Class(class)));
    }
    Ok(())
}
