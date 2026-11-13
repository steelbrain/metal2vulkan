//! The output body of an expansion into plain AIR (`matmul2d`, the simdgroup asynchronous copies):
//! blocks opened and closed in order, instructions pushed into the open block, and the typed
//! helpers the expansions build plain AIR with.

use crate::budget::Meter;
use crate::error::Error;
use crate::intrinsics::{self, AirIntrinsic, Bound, DeclClass};
use crate::ir::{
    AddrSpace, Align, BinOp, Block, BlockId, Body, Call, CallConv, Callee, CastOp, Const, Constant,
    FastMath, FnAttrs, FuncId, Function, Inst, InstId, InstKind, IntFlags, IntPredicate, Linkage,
    ModuleParts, Param, ParamAttrs, Sym, TailKind, Type, TypeId, UnnamedAddr, Value, Visibility,
};

use super::rewrite::invariant;

/// The block after `b`.
fn next_block(b: BlockId) -> Result<BlockId, Error> {
    b.raw()
        .checked_add(1)
        .map(BlockId::new)
        .ok_or_else(|| invariant("block ids"))
}

/// The output body under construction.
pub(super) struct Out<'p> {
    pub parts: &'p mut ModuleParts,
    pub body: Body,
    /// The open block: its first instruction and name.
    open: Option<(InstId, Option<Sym>)>,
}

impl<'p> Out<'p> {
    pub fn new(parts: &'p mut ModuleParts) -> Out<'p> {
        Out {
            parts,
            body: Body::default(),
            open: None,
        }
    }

    fn next_inst(&self) -> Result<InstId, Error> {
        u32::try_from(self.body.insts.len())
            .map(InstId::new)
            .map_err(|_| invariant("instruction ids"))
    }

    /// The id of the open block (blocks are numbered in the order they open).
    pub fn current(&self) -> Result<BlockId, Error> {
        if self.open.is_none() {
            return Err(invariant("no open block"));
        }
        u32::try_from(self.body.blocks.len())
            .map(BlockId::new)
            .map_err(|_| invariant("block ids"))
    }

    /// Open the next block; returns its id.
    pub fn open(&mut self, name: Option<Sym>) -> Result<BlockId, Error> {
        if self.open.is_some() {
            return Err(invariant("a block is already open"));
        }
        self.open = Some((self.next_inst()?, name));
        self.current()
    }

    /// Close the open block, which must end in a terminator; returns its id.
    pub fn close(&mut self, m: &mut Meter) -> Result<BlockId, Error> {
        let id = self.current()?;
        let (start, name) = self.open.take().ok_or_else(|| invariant("no open block"))?;
        let end = self.next_inst()?;
        self.body.blocks.push(Block { start, end, name }, m)?;
        Ok(id)
    }

    pub fn push(&mut self, ty: TypeId, kind: InstKind, m: &mut Meter) -> Result<Value, Error> {
        if self.open.is_none() {
            return Err(invariant("an instruction outside a block"));
        }
        let id = self.body.push_inst(Inst { ty, kind }, m)?;
        Ok(Value::Inst(id))
    }

    pub fn push_void(&mut self, kind: InstKind, m: &mut Meter) -> Result<InstId, Error> {
        let void = self.ty(Type::Void, m)?;
        match self.push(void, kind, m)? {
            Value::Inst(id) => Ok(id),
            _ => Err(invariant("a pushed instruction is not an instruction")),
        }
    }

    pub fn inst_mut(&mut self, id: InstId) -> Result<&mut Inst, Error> {
        self.body
            .insts
            .get_mut(id)
            .ok_or_else(|| invariant("instruction out of range"))
    }

    // ---- Types and constants ----

    pub fn ty(&mut self, ty: Type, m: &mut Meter) -> Result<TypeId, Error> {
        self.parts.types.intern(ty, m)
    }

    pub fn int_ty(&mut self, bits: u8, m: &mut Meter) -> Result<TypeId, Error> {
        self.ty(Type::Int(bits), m)
    }

    pub fn ptr_ty(&mut self, space: AddrSpace, m: &mut Meter) -> Result<TypeId, Error> {
        self.ty(Type::Ptr { space }, m)
    }

    pub fn constant(&mut self, ty: TypeId, value: Const, m: &mut Meter) -> Result<Value, Error> {
        Ok(Value::Const(
            self.parts.consts.intern(Constant { ty, value }, m)?,
        ))
    }

    /// The integer constant `v` of `bits` bits (truncated to them).
    pub fn int(&mut self, bits: u8, v: u64, m: &mut Meter) -> Result<Value, Error> {
        let ty = self.int_ty(bits, m)?;
        let v = if bits >= 64 {
            v
        } else {
            v & ((1u64 << bits) - 1)
        };
        self.constant(ty, Const::Int(v), m)
    }

    pub fn k32(&mut self, v: u32, m: &mut Meter) -> Result<Value, Error> {
        self.int(32, u64::from(v), m)
    }

    pub fn k64(&mut self, v: u64, m: &mut Meter) -> Result<Value, Error> {
        self.int(64, v, m)
    }

    pub fn f32k(&mut self, v: f32, m: &mut Meter) -> Result<Value, Error> {
        let ty = self.ty(Type::Float, m)?;
        self.constant(ty, Const::Float(u64::from(v.to_bits())), m)
    }

    // ---- Arithmetic ----

    pub fn bin(
        &mut self,
        ty: TypeId,
        op: BinOp,
        a: Value,
        b: Value,
        m: &mut Meter,
    ) -> Result<Value, Error> {
        let kind = InstKind::BinOp {
            op,
            flags: IntFlags::NONE,
            fmf: FastMath::NONE,
            a,
            b,
        };
        self.push(ty, kind, m)
    }

    /// `a op b` on `i32`.
    pub fn i32op(&mut self, op: BinOp, a: Value, b: Value, m: &mut Meter) -> Result<Value, Error> {
        let ty = self.int_ty(32, m)?;
        self.bin(ty, op, a, b, m)
    }

    /// `a op k` on `i32`.
    pub fn i32k(&mut self, op: BinOp, a: Value, k: u32, m: &mut Meter) -> Result<Value, Error> {
        let k = self.k32(k, m)?;
        self.i32op(op, a, k, m)
    }

    pub fn icmp(
        &mut self,
        pred: IntPredicate,
        a: Value,
        b: Value,
        m: &mut Meter,
    ) -> Result<Value, Error> {
        let ty = self.int_ty(1, m)?;
        self.push(ty, InstKind::ICmp { pred, a, b }, m)
    }

    pub fn select(
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

    pub fn cast(
        &mut self,
        op: CastOp,
        v: Value,
        ty: TypeId,
        m: &mut Meter,
    ) -> Result<Value, Error> {
        self.push(ty, InstKind::Cast { op, v }, m)
    }

    // ---- Memory ----

    /// `base + offset` bytes (`offset` an `i64`), a pointer in `space`.
    pub fn byte_gep(
        &mut self,
        space: AddrSpace,
        base: Value,
        offset: Value,
        m: &mut Meter,
    ) -> Result<Value, Error> {
        let i8t = self.int_ty(8, m)?;
        let ptr = self.ptr_ty(space, m)?;
        let kind = InstKind::Gep {
            src_ty: i8t,
            base,
            indices: Box::new([offset]),
            inbounds: false,
        };
        self.push(ptr, kind, m)
    }

    /// `base + k` bytes.
    pub fn byte_gep_k(
        &mut self,
        space: AddrSpace,
        base: Value,
        k: u64,
        m: &mut Meter,
    ) -> Result<Value, Error> {
        if k == 0 {
            return Ok(base);
        }
        let k = self.k64(k, m)?;
        self.byte_gep(space, base, k, m)
    }

    pub fn load(
        &mut self,
        ty: TypeId,
        ptr: Value,
        align: u64,
        m: &mut Meter,
    ) -> Result<Value, Error> {
        let kind = InstKind::Load {
            ptr,
            align: Align::from_bytes(align),
            volatile: false,
            atomic: None,
            range: None,
        };
        self.push(ty, kind, m)
    }

    pub fn store(
        &mut self,
        val: Value,
        ptr: Value,
        align: u64,
        m: &mut Meter,
    ) -> Result<(), Error> {
        let kind = InstKind::Store {
            val,
            ptr,
            align: Align::from_bytes(align),
            volatile: false,
            atomic: None,
        };
        self.push_void(kind, m)?;
        Ok(())
    }

    // ---- Calls ----

    /// A declaration of `name` with type `ret (params)`: the module's own when it declares one
    /// of that type, else a new one.
    fn declare(
        &mut self,
        name: &str,
        ret: TypeId,
        params: &[TypeId],
        m: &mut Meter,
    ) -> Result<(FuncId, TypeId), Error> {
        let fn_ty = self.ty(
            Type::Function {
                ret,
                params: params.into(),
                vararg: false,
            },
            m,
        )?;
        let found = self.parts.funcs.iter().find(|(_, f)| {
            f.body.is_none()
                && f.ty == fn_ty
                && self.parts.symbols.get(f.name) == Some(name.as_bytes())
        });
        if let Some((id, _)) = found {
            return Ok((id, fn_ty));
        }
        let sym = self.parts.symbols.intern(name.as_bytes(), m)?;
        let mut ps = Vec::new();
        m.reserve(&mut ps, params.len())?;
        ps.resize(params.len(), Param::default());
        let f = Function {
            name: sym,
            ty: fn_ty,
            linkage: Linkage::External,
            visibility: Visibility::Default,
            unnamed_addr: UnnamedAddr::Significant,
            cconv: CallConv::C,
            attrs: FnAttrs::default(),
            ret_attrs: ParamAttrs::default(),
            params: ps.into_boxed_slice(),
            body: None,
            section: None,
            align: None,
        };
        Ok((self.parts.funcs.push(f, m)?, fn_ty))
    }

    /// A call of the `air.*` or `llvm.*` intrinsic `name`, typed `ret (params)`, classified as
    /// decode classifies it.
    pub fn intrinsic(
        &mut self,
        name: &str,
        ret: TypeId,
        params: &[TypeId],
        args: &[Value],
        m: &mut Meter,
    ) -> Result<Value, Error> {
        let (decl, fn_ty) = self.declare(name, ret, params, m)?;
        let sym = self
            .parts
            .funcs
            .get(decl)
            .map(|f| f.name)
            .ok_or_else(|| invariant("a declaration out of range"))?;
        let class = intrinsics::classify(name.as_bytes(), sym, fn_ty, &self.parts.types)
            .ok_or_else(|| invariant("an expansion intrinsic is not an intrinsic"))?;
        if matches!(class, DeclClass::Unknown | DeclClass::Mismatch) {
            return Err(invariant("an expansion intrinsic outside the decoded set"));
        }
        let callee = match intrinsics::bind(class, args, &self.parts.consts)
            .map_err(|_| invariant("an expansion intrinsic's immediates"))?
        {
            Bound::Air(op) => Callee::Air { decl, op },
            Bound::Llvm(op) => Callee::Llvm { decl, op },
            Bound::Unknown => return Err(invariant("an expansion intrinsic bound as unknown")),
        };
        self.call(callee, ret, args, m)
    }

    /// A call of `op`, an `air.*` operation no AIR name spells (so decode makes none), typed
    /// `ret (params)` and declared as `name`, which decode does not classify as `op` either.
    pub fn air_call(
        &mut self,
        name: &str,
        op: AirIntrinsic,
        ret: TypeId,
        params: &[TypeId],
        args: &[Value],
        m: &mut Meter,
    ) -> Result<Value, Error> {
        let (decl, _) = self.declare(name, ret, params, m)?;
        self.call(Callee::Air { decl, op }, ret, args, m)
    }

    fn call(
        &mut self,
        callee: Callee,
        ret: TypeId,
        args: &[Value],
        m: &mut Meter,
    ) -> Result<Value, Error> {
        let call = Call {
            callee,
            args: args.into(),
            fmf: FastMath::NONE,
            tail: TailKind::None,
            attrs: None,
            range: None,
        };
        self.push(ret, InstKind::Call(Box::new(call)), m)
    }

    // ---- Control ----

    pub fn br(&mut self, dest: BlockId, m: &mut Meter) -> Result<(), Error> {
        self.push_void(InstKind::Br { dest, hints: None }, m)?;
        Ok(())
    }

    /// `ret v`, closing the open block.
    pub fn ret(&mut self, v: Option<Value>, m: &mut Meter) -> Result<(), Error> {
        self.push_void(InstKind::Ret(v), m)?;
        self.close(m)?;
        Ok(())
    }

    /// `c ? then() : otherwise`, where `then` runs only when `c` holds (in a block of its own,
    /// so a load it makes is not executed otherwise). Continues in a merge block, left open,
    /// whose PHI is the value.
    pub fn cond_value(
        &mut self,
        c: Value,
        ty: TypeId,
        otherwise: Value,
        m: &mut Meter,
        then: &mut dyn FnMut(&mut Self, &mut Meter) -> Result<Value, Error>,
    ) -> Result<Value, Error> {
        let here = self.current()?;
        let then_block = next_block(here)?;
        let test = self.push_void(
            InstKind::CondBr {
                c,
                t: then_block,
                f: then_block,
                hints: None,
            },
            m,
        )?;
        self.close(m)?;
        self.open(None)?;
        let v = then(self, m)?;
        let end = self.current()?;
        let merge = next_block(end)?;
        self.br(merge, m)?;
        self.close(m)?;
        if self.open(None)? != merge {
            return Err(invariant("a merge block out of order"));
        }
        if let InstKind::CondBr { f, .. } = &mut self.inst_mut(test)?.kind {
            *f = merge;
        }
        let incoming = Box::new([(v, end), (otherwise, here)]);
        self.push(
            ty,
            InstKind::Phi {
                incoming,
                fmf: FastMath::NONE,
            },
            m,
        )
    }

    /// `for i in 0..n` (`n` an `i32`, compared unsigned) with loop-carried values `carried`
    /// (their types and initial values): `body` gets `i` and the carried values and returns
    /// their next values. The loop tests before each iteration, so `n == 0` runs no iteration.
    /// Returns the carried values after the loop, and leaves the exit block open.
    pub fn counted_loop(
        &mut self,
        n: Value,
        carried: &[(TypeId, Value)],
        m: &mut Meter,
        body: &mut dyn FnMut(&mut Self, Value, &[Value], &mut Meter) -> Result<Vec<Value>, Error>,
    ) -> Result<Vec<Value>, Error> {
        let i32t = self.int_ty(32, m)?;
        let zero = self.k32(0, m)?;
        let one = self.k32(1, m)?;
        self.range_loop((i32t, zero, n, one), carried, m, body)
    }

    /// `for (i = start; i < end; i += step)` over integers of type `ty` (compared unsigned) with
    /// loop-carried values `carried`, as [`Out::counted_loop`]: the loop tests before each
    /// iteration, so `start >= end` runs none. Returns the carried values after the loop, and
    /// leaves the exit block open.
    pub fn range_loop(
        &mut self,
        (ty, start, end, step): (TypeId, Value, Value, Value),
        carried: &[(TypeId, Value)],
        m: &mut Meter,
        body: &mut dyn FnMut(&mut Self, Value, &[Value], &mut Meter) -> Result<Vec<Value>, Error>,
    ) -> Result<Vec<Value>, Error> {
        let pre = self.current()?;
        let header = next_block(pre)?;
        self.br(header, m)?;
        self.close(m)?;
        if self.open(None)? != header {
            return Err(invariant("a loop header out of order"));
        }
        let phi = |init: Value| InstKind::Phi {
            incoming: Box::new([(init, pre), (init, pre)]),
            fmf: FastMath::NONE,
        };
        let i = self.push(ty, phi(start), m)?;
        let mut phis = Vec::new();
        m.reserve(&mut phis, carried.len())?;
        for &(ty, init) in carried {
            phis.push(self.push(ty, phi(init), m)?);
        }
        let more = self.icmp(IntPredicate::Ult, i, end, m)?;
        let body_block = next_block(header)?;
        let test = self.push_void(
            InstKind::CondBr {
                c: more,
                t: body_block,
                f: body_block,
                hints: None,
            },
            m,
        )?;
        self.close(m)?;
        self.open(None)?;
        let next = body(self, i, &phis, m)?;
        if next.len() != phis.len() {
            return Err(invariant("a loop body's carried values"));
        }
        let i1 = self.bin(ty, BinOp::Add, i, step, m)?;
        let latch = self.current()?;
        self.br(header, m)?;
        self.close(m)?;
        let exit = self.open(None)?;
        if let InstKind::CondBr { f, .. } = &mut self.inst_mut(test)?.kind {
            *f = exit;
        }
        for (phi, value) in std::iter::once(i)
            .chain(phis.iter().copied())
            .zip(std::iter::once(i1).chain(next.iter().copied()))
        {
            let Value::Inst(id) = phi else {
                return Err(invariant("a loop phi is not an instruction"));
            };
            if let InstKind::Phi { incoming, .. } = &mut self.inst_mut(id)?.kind {
                if let Some(slot) = incoming.get_mut(1) {
                    *slot = (value, latch);
                }
            }
        }
        m.release(phis.capacity() * std::mem::size_of::<Value>());
        Ok(phis)
    }
}
