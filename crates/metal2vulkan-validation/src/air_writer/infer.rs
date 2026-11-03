//! Pointee inference: one term per value, equations from every use.
//!
//! Every instruction result and parameter gets a term from its erased type; constants get a
//! fresh term per use, because the IR interns `null`, `undef`, and aggregates of them once for
//! every pointee they stand for. The equations are the typed-pointer typing rules of each
//! instruction (a load's pointer points at its result type, a GEP's base at its source element
//! type, call arguments are the callee's parameter types, PHI incomings are the PHI's type, ...)
//! plus the recorded pointees of every function's parameters and return value. Pointer casts
//! (`bitcast`, `addrspacecast`, `inttoptr`, `ptrtoint`) relate nothing, because a typed-pointer
//! module changes pointees only through them, with two exceptions from LLVM's constant folder:
//! a constant `addrspacecast` keeps its pointee (LLVM splits one that changes it into a
//! `bitcast` and an `addrspacecast`), and a constant pointer `bitcast` must change its pointee
//! (LLVM folds one that does not away), so an unconstrained result pointee is chosen to differ
//! from the operand's.

use metal2vulkan_air::ir::{
    Callee, CastOp, Const, ConstExpr, ConstId, FuncId, Global, GlobalId, Idx, InstKind, Md, Module,
    Type, TypeId, Value,
};

use super::terms::{Terms, N};
use super::{Pointees, WriteError};

/// A constant at one use: its term and, for aggregates with pointers and for expressions, the
/// uses of its operands.
#[derive(Clone, Debug)]
pub(super) struct CUse {
    pub node: N,
    pub kids: Box<[CUse]>,
}

/// One operand of an instruction, in `InstKind::for_each_value` order.
#[derive(Clone, Debug)]
pub(super) struct Op {
    pub node: N,
    pub cuse: Option<CUse>,
}

/// The terms of one function body.
#[derive(Debug, Default)]
pub(super) struct BodyTyping {
    /// The result term of each instruction (`None` for void instructions).
    pub results: Vec<Option<N>>,
    /// The operand terms of each instruction.
    pub ops: Vec<Vec<Op>>,
    /// For a call, the function term it calls through.
    pub callee: Vec<Option<N>>,
}

/// The solved typing of a module.
pub(super) struct Typing {
    pub funcs: Vec<N>,
    pub globals: Vec<N>,
    pub inits: Vec<Option<CUse>>,
    pub bodies: Vec<Option<BodyTyping>>,
    pub md_values: Vec<Option<CUse>>,
}

struct Infer<'a, 'm> {
    m: &'m Module,
    t: &'a mut Terms<'m>,
    funcs: Vec<N>,
    globals: Vec<N>,
    /// `(result pointee, operand pointee)` of every constant pointer `bitcast`.
    const_casts: Vec<(N, N)>,
    /// Where the current equation comes from, for conflict reports.
    site: String,
}

/// Build and solve the equations of `m`.
pub(super) fn infer<'m>(
    m: &'m Module,
    pointees: &Pointees,
    t: &mut Terms<'m>,
) -> Result<Typing, WriteError> {
    let parts = m.parts();
    let mut cx = Infer {
        m,
        t,
        funcs: Vec::with_capacity(parts.funcs.len()),
        globals: Vec::with_capacity(parts.globals.len()),
        const_casts: Vec::new(),
        site: String::new(),
    };
    for (id, f) in parts.funcs.iter() {
        let node = cx.t.fresh(f.ty)?;
        let (ret, params, _) = cx.t.signature(node)?;
        cx.site = format!("the signature of {}", cx.fn_name(id));
        if let Some(p) = pointees.ret(id) {
            let want = cx.t.fresh(p)?;
            let got = cx.t.pointee(ret)?;
            cx.unify(got, want)?;
        }
        for (i, &param) in params.iter().enumerate() {
            if let Some(p) = pointees.param(id, i) {
                let want = cx.t.fresh(p)?;
                let got = cx.t.pointee(param)?;
                cx.unify(got, want)?;
            }
            let attrs = f.params.get(i).map(|p| &p.attrs);
            for ty in attrs.into_iter().flat_map(|a| a.types()) {
                let want = cx.t.fresh(ty)?;
                let got = cx.t.pointee(param)?;
                cx.unify(got, want)?;
            }
        }
        cx.funcs.push(node);
    }
    for (_, g) in parts.globals.iter() {
        let value = cx.t.fresh(g.value_ty)?;
        let node = cx.t.ptr_to(g.space.raw(), value);
        cx.globals.push(node);
    }
    for (id, g) in parts.globals.iter() {
        cx.site = format!("the type of global {}", id.raw());
        cx.reserved_list(g, cx.globals[id.index()])?;
    }
    let mut inits = Vec::with_capacity(parts.globals.len());
    for (id, g) in parts.globals.iter() {
        cx.site = format!("the initializer of global {}", id.raw());
        inits.push(match g.init {
            Some(c) => {
                let node = cx.global(id)?;
                let value = cx.t.pointee(node)?;
                Some(cx.cuse(c, value)?)
            }
            None => None,
        });
    }
    let mut bodies = Vec::with_capacity(parts.funcs.len());
    for (id, _) in parts.funcs.iter() {
        bodies.push(cx.body(id)?);
    }
    let mut md_values = Vec::with_capacity(parts.metadata.len());
    for (id, md) in parts.metadata.iter() {
        cx.site = format!("metadata {}", id.raw());
        md_values.push(match md {
            Md::Value(c) => Some(cx.fresh_cuse(*c)?),
            _ => None,
        });
    }
    let Infer {
        t,
        funcs,
        globals,
        const_casts,
        ..
    } = cx;
    // A constant `bitcast` between pointers whose two spellings coincide would be folded away by
    // the assembler; an unconstrained result pointee is chosen to differ from the operand's.
    for (result, operand) in const_casts {
        if t.is_var(result) {
            let width = if t.text(operand)? == "i8" { 16 } else { 8 };
            t.bind_int(result, width);
        }
    }
    Ok(Typing {
        funcs,
        globals,
        inits,
        bodies,
        md_values,
    })
}

impl Infer<'_, '_> {
    fn unify(&mut self, a: N, b: N) -> Result<(), WriteError> {
        self.t
            .unify(a, b)
            .map_err(|detail| WriteError::PointeeConflict {
                site: self.site.clone(),
                detail,
            })
    }

    /// LLVM fixes the type of its constructor and destructor lists itself: `llvm.global_ctors`
    /// and `llvm.global_dtors` are `[N x { i32, void ()*, i8* }]` (LangRef, and the verifier's
    /// "wrong type for intrinsic global variable"). Their entries constrain the function pointer
    /// only when there are some, and Metal emits an empty list, so the contract is an equation of
    /// its own. A list of another erased shape is left to the verifier.
    fn reserved_list(&mut self, g: &Global, node: N) -> Result<(), WriteError> {
        let name = self.m.symbols().get(g.name);
        if !matches!(name, Some(b"llvm.global_ctors" | b"llvm.global_dtors")) {
            return Ok(());
        }
        let types = self.m.types();
        let Some(Type::Array { elem, .. }) = types.get(g.value_ty) else {
            return Ok(());
        };
        let Some(Type::Struct { fields, .. }) = types.get(*elem) else {
            return Ok(());
        };
        let is_ptr = |i: usize| {
            matches!(
                fields.get(i).and_then(|&f| types.get(f)),
                Some(Type::Ptr { .. })
            )
        };
        if fields.len() != 3 || !is_ptr(1) || !is_ptr(2) {
            return Ok(());
        }
        let value = self.t.pointee(node)?;
        let entry = self.t.child(value, 0)?;
        let function = self.t.child(entry, 1)?;
        let function = self.t.pointee(function)?;
        let want = self.t.void_fn();
        self.unify(function, want)?;
        let data = self.t.child(entry, 2)?;
        let data = self.t.pointee(data)?;
        let byte = self.t.int(8);
        self.unify(data, byte)
    }

    fn fn_name(&self, f: FuncId) -> String {
        self.m
            .func(f)
            .and_then(|f| self.m.symbols().get(f.name))
            .map(|b| String::from_utf8_lossy(b).into_owned())
            .unwrap_or_else(|| format!("{f:?}"))
    }

    fn func(&self, f: FuncId) -> Result<N, WriteError> {
        self.funcs
            .get(f.index())
            .copied()
            .ok_or_else(|| WriteError::Malformed(format!("function {f:?} out of range")))
    }

    fn global(&self, g: GlobalId) -> Result<N, WriteError> {
        self.globals
            .get(g.index())
            .copied()
            .ok_or_else(|| WriteError::Malformed(format!("global {g:?} out of range")))
    }

    fn const_ty(&self, c: ConstId) -> Result<TypeId, WriteError> {
        self.m
            .consts()
            .ty(c)
            .ok_or_else(|| WriteError::Malformed(format!("constant {c:?} out of range")))
    }

    /// A use of constant `c` at a fresh term of its own type.
    fn fresh_cuse(&mut self, c: ConstId) -> Result<CUse, WriteError> {
        let ty = self.const_ty(c)?;
        let node = self.t.fresh(ty)?;
        self.cuse(c, node)
    }

    /// A use of constant `c` at term `node`.
    fn cuse(&mut self, c: ConstId, node: N) -> Result<CUse, WriteError> {
        let k = self
            .m
            .consts()
            .get(c)
            .ok_or_else(|| WriteError::Malformed(format!("constant {c:?} out of range")))?;
        let leaf = |node| CUse {
            node,
            kids: Box::new([]),
        };
        if !self.t.has_ptr(k.ty) && !matches!(k.value, Const::Expr(_) | Const::Aggregate(_)) {
            return Ok(leaf(node));
        }
        Ok(match &k.value {
            Const::Int(_)
            | Const::Float(_)
            | Const::Null
            | Const::Undef
            | Const::Poison
            | Const::InlineAsm(_) => leaf(node),
            Const::Aggregate(elems) => {
                let mut kids = Vec::with_capacity(elems.len());
                for (i, &e) in elems.iter().enumerate() {
                    let child = self.t.child(node, i as u64)?;
                    kids.push(self.cuse(e, child)?);
                }
                CUse {
                    node,
                    kids: kids.into_boxed_slice(),
                }
            }
            Const::GlobalAddr(g) => {
                let g = self.global(*g)?;
                self.unify(node, g)?;
                leaf(node)
            }
            Const::FuncAddr(f) => {
                let f = self.func(*f)?;
                let pointee = self.t.pointee(node)?;
                self.unify(pointee, f)?;
                leaf(node)
            }
            Const::Expr(ConstExpr::Cast { op, ops: [o] }) => {
                let kid = self.fresh_cuse(*o)?;
                if self.is_ptr(self.const_ty(*o)?) && self.is_ptr(k.ty) {
                    let result = self.t.pointee(node)?;
                    let operand = self.t.pointee(kid.node)?;
                    match op {
                        CastOp::Bitcast => self.const_casts.push((result, operand)),
                        // LLVM folds a typed constant `addrspacecast` that changes the pointee
                        // into a `bitcast` and an `addrspacecast`, so one it kept has equal
                        // pointees (an instruction may change both).
                        CastOp::AddrSpaceCast => self.unify(result, operand)?,
                        _ => {}
                    }
                }
                CUse {
                    node,
                    kids: Box::new([kid]),
                }
            }
            Const::Expr(ConstExpr::BinOp { ops, .. }) => {
                let mut kids = Vec::with_capacity(2);
                for &o in ops {
                    kids.push(self.cuse(o, node)?);
                }
                CUse {
                    node,
                    kids: kids.into_boxed_slice(),
                }
            }
            Const::Expr(ConstExpr::Gep { src_ty, ops, .. }) => {
                let mut kids = Vec::with_capacity(ops.len());
                for &o in ops.iter() {
                    kids.push(self.fresh_cuse(o)?);
                }
                let base = kids
                    .first()
                    .map(|k| k.node)
                    .ok_or_else(|| WriteError::Malformed("a constant GEP without a base".into()))?;
                let indices: Vec<Value> = ops.iter().skip(1).map(|&o| Value::Const(o)).collect();
                self.gep(*src_ty, base, &indices, node)?;
                CUse {
                    node,
                    kids: kids.into_boxed_slice(),
                }
            }
        })
    }

    fn is_ptr(&self, t: TypeId) -> bool {
        matches!(self.m.types().get(t), Some(Type::Ptr { .. }))
    }

    /// The GEP equations: `base` points at `src_ty`, and `result` points at the element the
    /// indices after the first select.
    fn gep(
        &mut self,
        src_ty: TypeId,
        base: N,
        indices: &[Value],
        result: N,
    ) -> Result<(), WriteError> {
        let src = self.t.fresh(src_ty)?;
        let base_pointee = self.t.pointee(base)?;
        self.unify(base_pointee, src)?;
        let mut at = src;
        let mut ty = src_ty;
        for &idx in indices.iter().skip(1) {
            let field = match self.m.types().get(ty) {
                Some(Type::Struct { fields, .. }) => {
                    let i = self.const_int(idx).ok_or_else(|| {
                        WriteError::Malformed("a struct GEP index is not a constant".into())
                    })?;
                    ty = *usize::try_from(i)
                        .ok()
                        .and_then(|i| fields.get(i))
                        .ok_or_else(|| WriteError::Malformed("GEP field out of range".into()))?;
                    i
                }
                Some(Type::Array { elem, .. } | Type::Vector { elem, .. }) => {
                    ty = *elem;
                    0
                }
                _ => return Err(WriteError::Malformed("GEP into a non-aggregate".into())),
            };
            at = self.t.child(at, field)?;
        }
        let result_pointee = self.t.pointee(result)?;
        self.unify(result_pointee, at)
    }

    fn const_int(&self, v: Value) -> Option<u64> {
        let Value::Const(c) = v else {
            return None;
        };
        match self.m.consts().get(c)?.value {
            Const::Int(i) => Some(i),
            _ => None,
        }
    }

    fn body(&mut self, id: FuncId) -> Result<Option<BodyTyping>, WriteError> {
        let Some(f) = self.m.func(id) else {
            return Err(WriteError::Malformed(format!(
                "function {id:?} out of range"
            )));
        };
        let Some(body) = &f.body else {
            return Ok(None);
        };
        let fn_node = self.func(id)?;
        let (fn_ret, fn_params, _) = self.t.signature(fn_node)?;
        let mut typing = BodyTyping::default();
        for inst in body.insts.values() {
            let void = matches!(self.m.types().get(inst.ty), Some(Type::Void));
            typing.results.push(if void {
                None
            } else {
                Some(self.t.fresh(inst.ty)?)
            });
        }
        let name = self.fn_name(id);
        for (i, inst) in body.insts.iter() {
            self.site = format!("{name}, instruction {}", i.raw());
            let res = typing.results.get(i.index()).copied().flatten();
            let mut ops = Vec::new();
            let mut callee = None;
            let op = |cx: &mut Self, v: Value, ops: &mut Vec<Op>| -> Result<N, WriteError> {
                let o = match v {
                    Value::Arg(a) => Op {
                        node: *fn_params.get(a.index()).ok_or_else(|| {
                            WriteError::Malformed(format!("argument {a:?} out of range"))
                        })?,
                        cuse: None,
                    },
                    Value::Inst(d) => Op {
                        node: typing
                            .results
                            .get(d.index())
                            .copied()
                            .flatten()
                            .ok_or_else(|| WriteError::Malformed(format!("{d:?} has no value")))?,
                        cuse: None,
                    },
                    Value::Const(c) => {
                        let cu = cx.fresh_cuse(c)?;
                        Op {
                            node: cu.node,
                            cuse: Some(cu),
                        }
                    }
                };
                let n = o.node;
                ops.push(o);
                Ok(n)
            };
            let res_node = || {
                res.ok_or_else(|| WriteError::Malformed("an instruction without a value".into()))
            };
            match &inst.kind {
                InstKind::BinOp { a, b, .. } => {
                    let (a, b) = (op(self, *a, &mut ops)?, op(self, *b, &mut ops)?);
                    let r = res_node()?;
                    self.unify(a, r)?;
                    self.unify(b, r)?;
                }
                InstKind::FNeg { v, .. } | InstKind::Freeze(v) => {
                    let v = op(self, *v, &mut ops)?;
                    self.unify(v, res_node()?)?;
                }
                InstKind::Cast { v, .. } => {
                    op(self, *v, &mut ops)?;
                }
                InstKind::ICmp { a, b, .. } | InstKind::FCmp { a, b, .. } => {
                    let (a, b) = (op(self, *a, &mut ops)?, op(self, *b, &mut ops)?);
                    self.unify(a, b)?;
                }
                InstKind::Select { c, t, f, .. } => {
                    op(self, *c, &mut ops)?;
                    let (t, f) = (op(self, *t, &mut ops)?, op(self, *f, &mut ops)?);
                    let r = res_node()?;
                    self.unify(t, r)?;
                    self.unify(f, r)?;
                }
                InstKind::ExtractElement { vec, idx } => {
                    let vec = op(self, *vec, &mut ops)?;
                    op(self, *idx, &mut ops)?;
                    let elem = self.t.child(vec, 0)?;
                    self.unify(elem, res_node()?)?;
                }
                InstKind::InsertElement { vec, elt, idx } => {
                    let vec = op(self, *vec, &mut ops)?;
                    let elt = op(self, *elt, &mut ops)?;
                    op(self, *idx, &mut ops)?;
                    let r = res_node()?;
                    self.unify(vec, r)?;
                    let elem = self.t.child(r, 0)?;
                    self.unify(elem, elt)?;
                }
                InstKind::ShuffleVector { a, b, .. } => {
                    let (a, b) = (op(self, *a, &mut ops)?, op(self, *b, &mut ops)?);
                    self.unify(a, b)?;
                    let (ea, er) = (self.t.child(a, 0)?, self.t.child(res_node()?, 0)?);
                    self.unify(ea, er)?;
                }
                InstKind::ExtractValue { agg, indices } => {
                    let mut at = op(self, *agg, &mut ops)?;
                    for &i in indices.iter() {
                        at = self.t.child(at, u64::from(i))?;
                    }
                    self.unify(at, res_node()?)?;
                }
                InstKind::InsertValue { agg, val, indices } => {
                    let agg = op(self, *agg, &mut ops)?;
                    let val = op(self, *val, &mut ops)?;
                    let r = res_node()?;
                    self.unify(agg, r)?;
                    let mut at = r;
                    for &i in indices.iter() {
                        at = self.t.child(at, u64::from(i))?;
                    }
                    self.unify(at, val)?;
                }
                InstKind::Alloca { ty, count, .. } => {
                    op(self, *count, &mut ops)?;
                    let want = self.t.fresh(*ty)?;
                    let got = self.t.pointee(res_node()?)?;
                    self.unify(got, want)?;
                }
                InstKind::Load { ptr, .. } => {
                    let ptr = op(self, *ptr, &mut ops)?;
                    let pointee = self.t.pointee(ptr)?;
                    self.unify(pointee, res_node()?)?;
                }
                InstKind::Store { val, ptr, .. } => {
                    let val = op(self, *val, &mut ops)?;
                    let ptr = op(self, *ptr, &mut ops)?;
                    let pointee = self.t.pointee(ptr)?;
                    self.unify(pointee, val)?;
                }
                InstKind::Gep {
                    src_ty,
                    base,
                    indices,
                    ..
                } => {
                    let base = op(self, *base, &mut ops)?;
                    for &i in indices.iter() {
                        op(self, i, &mut ops)?;
                    }
                    self.gep(*src_ty, base, indices, res_node()?)?;
                }
                InstKind::AtomicRmw { ptr, val, .. } => {
                    let ptr = op(self, *ptr, &mut ops)?;
                    let val = op(self, *val, &mut ops)?;
                    let pointee = self.t.pointee(ptr)?;
                    self.unify(pointee, val)?;
                    self.unify(val, res_node()?)?;
                }
                InstKind::CmpXchg { ptr, cmp, new, .. } => {
                    let ptr = op(self, *ptr, &mut ops)?;
                    let cmp = op(self, *cmp, &mut ops)?;
                    let new = op(self, *new, &mut ops)?;
                    let pointee = self.t.pointee(ptr)?;
                    self.unify(pointee, cmp)?;
                    self.unify(cmp, new)?;
                    let first = self.t.child(res_node()?, 0)?;
                    self.unify(first, cmp)?;
                }
                InstKind::Fence(_) | InstKind::Br { .. } | InstKind::Unreachable => {}
                InstKind::Call(call) => {
                    let f = match &call.callee {
                        Callee::Indirect { callee, fn_ty } => {
                            let callee = op(self, *callee, &mut ops)?;
                            let f = self.t.fresh(*fn_ty)?;
                            let pointee = self.t.pointee(callee)?;
                            self.unify(pointee, f)?;
                            f
                        }
                        other => {
                            let decl = other.func().ok_or_else(|| {
                                WriteError::Malformed("a call without a callee".into())
                            })?;
                            self.func(decl)?
                        }
                    };
                    let (ret, params, _) = self.t.signature(f)?;
                    for (i, &a) in call.args.iter().enumerate() {
                        let a = op(self, a, &mut ops)?;
                        if let Some(&p) = params.get(i) {
                            self.unify(a, p)?;
                        }
                        let attrs = call.attrs.as_ref().and_then(|c| c.params.get(i));
                        for ty in attrs.into_iter().flat_map(|a| a.types()) {
                            let want = self.t.fresh(ty)?;
                            let got = self.t.pointee(a)?;
                            self.unify(got, want)?;
                        }
                    }
                    if let Some(r) = res {
                        self.unify(r, ret)?;
                    }
                    callee = Some(f);
                }
                InstKind::Phi { incoming, .. } => {
                    let r = res_node()?;
                    for &(v, _) in incoming.iter() {
                        let v = op(self, v, &mut ops)?;
                        self.unify(v, r)?;
                    }
                }
                InstKind::Ret(v) => {
                    if let Some(v) = v {
                        let v = op(self, *v, &mut ops)?;
                        self.unify(v, fn_ret)?;
                    }
                }
                InstKind::CondBr { c, .. } => {
                    op(self, *c, &mut ops)?;
                }
                InstKind::Switch { v, .. } => {
                    op(self, *v, &mut ops)?;
                }
            }
            typing.ops.push(ops);
            typing.callee.push(callee);
        }
        Ok(Some(typing))
    }
}
