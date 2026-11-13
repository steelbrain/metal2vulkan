//! The rewrite: new signatures, the entry's status-buffer argument, and the checks spliced into
//! each planned function (see the module documentation of `oracle_fuel`).

use std::collections::{BTreeMap, BTreeSet};

use metal2vulkan_air::bitcode::{EnumAttrTable, EnumAttrs, SiteEnums};
use metal2vulkan_air::budget::Meter;
use metal2vulkan_air::intrinsics::{bind, classify, Bound, DeclClass};
use metal2vulkan_air::ir::{
    AddrSpace, Align, Arena, ArgIdx, BinOp, BlockId, Body, Call, CallConv, Callee, Const, ConstId,
    ConstTable, Constant, FastMath, FnAttrs, FuncId, Function, Idx, InstId, InstKind, IntFlags,
    IntPredicate, Linkage, Md, MdId, Module, ModuleParts, NamedMd, Param, ParamAttrs, Symbols,
    TailKind, Type, TypeId, TypeTable, Value,
};

use super::edit::Editor;
use super::plan::{name, FnPlan, Plan};
use super::status::{
    CHUNK_SHIFT, EXHAUSTED, WORD_CONSUMED_CHUNKS, WORD_INVOCATION_FUEL, WORD_POOL_CHUNKS,
};
use super::{FuelConfig, FuelError, FuelReport, FuelUnsupported, Instrumented};
use crate::air_writer::Pointees;

/// Drop from a function's (or a call site's) attributes what stops being true once it writes the
/// thread's fuel state and the status buffer, through volatile atomics: memory effects that say
/// otherwise, `speculatable` (no side effects), and `nosync` (a volatile access synchronizes).
/// Left in place, they would let the Metal compiler drop, move, or speculate the calls. Every other
/// attribute stays as the front end wrote it. `enums` is the position's entry in the module's
/// enum-attribute table, when it has one.
pub(super) fn writes_state(attrs: &mut FnAttrs, enums: Option<&mut SiteEnums>) {
    attrs.memory = None;
    if let Some(enums) = enums {
        writes_state_kinds(enums);
    }
}

/// The enum kinds [`writes_state`] drops: `speculatable` and `nosync`.
pub(super) fn writes_state_kinds(enums: &mut SiteEnums) {
    enums.func.remove(EnumAttrs::SPECULATABLE);
    enums.func.remove(EnumAttrs::NO_SYNC);
}

/// `air.atomic.*` operands the transform passes: relaxed order, device scope, no `mem_flags` (the
/// AIR 2.9 form's), volatile.
const ORDER_RELAXED: u64 = 0;
const SCOPE_DEVICE: u64 = 2;
const MEM_FLAGS_NONE: u64 = 0;

/// The first AIR version whose atomic intrinsics take a `mem_flags` `i32` between the scope and the
/// volatile flag (`T (ptr, i32 order, i32 scope, i32 flags, i1 volatile)` for a load): what Apple's
/// compiler writes for macOS 27, and what `metal-opt` requires of a module that states it.
const FLAGGED_ATOMICS: (u64, u64) = (2, 9);

/// The `(major, minor)` of the module's `!air.version`, when it states one.
fn air_version(module: &Module) -> Option<(u64, u64)> {
    let list = module
        .named_md()
        .iter()
        .find(|l| module.symbols().get(l.name) == Some(b"air.version".as_slice()))?;
    let [node] = &*list.ops else {
        return None;
    };
    let Some(Md::Node { ops, .. }) = module.md(*node) else {
        return None;
    };
    let int = |at: usize| match module.md((*ops.get(at)?)?)? {
        Md::Value(c) => match module.consts().get(*c)?.value {
            Const::Int(v) => Some(v),
            _ => None,
        },
        _ => None,
    };
    Some((int(0)?, int(1)?))
}

/// Local iterations per pool chunk, minus one.
const CHUNK_MASK: u64 = (1 << CHUNK_SHIFT) - 1;

/// The status argument's name in the stage metadata (diagnostics only).
const STATUS_ARG_NAME: &str = "m2v_oracle_status";

impl Ids {
    /// An atomic call's operands: `head` (the pointer, and an `rmw`'s value), then the order, the
    /// scope, the `mem_flags` where the module's atomics take one, and the volatile flag.
    fn atomic(&self, head: &[Value]) -> Vec<Value> {
        let mut args = head.to_vec();
        args.extend([self.order, self.scope]);
        args.extend(self.flags);
        args.push(self.volatile);
        args
    }
}

/// Interning over the module's tables.
struct Tables<'m, 'b> {
    symbols: &'m mut Symbols,
    types: &'m mut TypeTable,
    consts: &'m mut ConstTable,
    meter: &'b mut Meter,
}

impl Tables<'_, '_> {
    fn ty(&mut self, ty: Type) -> Result<TypeId, FuelError> {
        self.types.intern(ty, self.meter).map_err(internal)
    }

    fn constant(&mut self, ty: TypeId, value: Const) -> Result<ConstId, FuelError> {
        self.consts
            .intern(Constant { ty, value }, self.meter)
            .map_err(internal)
    }

    fn sym(&mut self, s: &str) -> Result<metal2vulkan_air::ir::Sym, FuelError> {
        self.symbols
            .intern(s.as_bytes(), self.meter)
            .map_err(internal)
    }
}

fn internal(e: impl Into<metal2vulkan::error::Error>) -> FuelError {
    FuelError::Internal(e.into().to_string())
}

/// Everything a check refers to, interned before any body is edited.
struct Ids {
    void: TypeId,
    i1: TypeId,
    i32: TypeId,
    state_ty: TypeId,
    thread_ptr: TypeId,
    device_ptr: TypeId,
    zero: Value,
    one: Value,
    chunk_mask: Value,
    exhausted: Value,
    order: Value,
    scope: Value,
    /// The `mem_flags` immediate, when the module's atomics take one ([`FLAGGED_ATOMICS`]).
    flags: Option<Value>,
    volatile: Value,
    word: [Value; 4],
    one_count: Value,
    load: Callee,
    add: Callee,
    or: Callee,
    align: Option<Align>,
}

/// A runtime intrinsic the checks call.
struct Intrinsic {
    name: &'static str,
    /// Operands after the pointer: `rmw` intrinsics also take the value.
    rmw: bool,
}

const LOAD: Intrinsic = Intrinsic {
    name: "air.atomic.global.load.i32",
    rmw: false,
};
const ADD: Intrinsic = Intrinsic {
    name: "air.atomic.global.add.u.i32",
    rmw: true,
};
const OR: Intrinsic = Intrinsic {
    name: "air.atomic.global.or.u.i32",
    rmw: true,
};

/// The stage-metadata node that lists the entry, found before the module is taken apart.
struct EntrySite {
    /// Index into the named metadata, and position within its operands.
    named: usize,
    op: usize,
    distinct: bool,
    ops: Vec<Option<MdId>>,
    args_distinct: bool,
    args: Vec<Option<MdId>>,
}

fn entry_site(module: &Module, entry: FuncId) -> Option<EntrySite> {
    for (named, list) in module.named_md().iter().enumerate() {
        let stage = module.symbols().get(list.name)?;
        if !matches!(stage, b"air.kernel" | b"air.vertex" | b"air.fragment") {
            continue;
        }
        for (op, &node) in list.ops.iter().enumerate() {
            let Some(Md::Node { distinct, ops }) = module.md(node) else {
                continue;
            };
            let names_entry = ops
                .first()
                .copied()
                .flatten()
                .and_then(|m| match module.md(m) {
                    Some(Md::Value(c)) => module.consts().get(*c).map(|c| &c.value),
                    _ => None,
                })
                == Some(&Const::FuncAddr(entry));
            if !names_entry {
                continue;
            }
            let Some(Md::Node {
                distinct: args_distinct,
                ops: args,
            }) = ops.get(2).copied().flatten().and_then(|m| module.md(m))
            else {
                return None;
            };
            return Some(EntrySite {
                named,
                op,
                distinct: *distinct,
                ops: ops.to_vec(),
                args_distinct: *args_distinct,
                args: args.to_vec(),
            });
        }
    }
    None
}

pub(super) fn emit(
    module: Module,
    mut pointees: Pointees,
    mut enums: EnumAttrTable,
    plan: &Plan,
    config: &FuelConfig,
    meter: &mut Meter,
) -> Result<Instrumented, FuelError> {
    // Nothing the entry executes loops (the module the oracle loads may be compiled apart from the
    // one the checker classified, and the Metal compiler folds many loops away): the module runs
    // as it is. The status buffer is still bound, and reads back untouched.
    let exempt = FuelReport {
        exempt_loops: plan.exempt_loops,
        exempt_nest: plan.exempt_nest,
        ..FuelReport::default()
    };
    if !plan.funcs.contains_key(&plan.entry) {
        return Ok(Instrumented {
            module,
            pointees,
            enums,
            report: exempt,
        });
    }
    let entry_name = name(&module, plan.entry);
    let site = entry_site(&module, plan.entry)
        .ok_or(FuelUnsupported::EntryNotAStage { entry: entry_name })?;
    let flagged = air_version(&module).is_some_and(|v| v >= FLAGGED_ATOMICS);
    let ModuleParts {
        mut symbols,
        mut types,
        mut consts,
        globals,
        funcs,
        mut metadata,
        mut named_md,
        triple,
        source_filename,
        data_layout,
    } = module.into_parts();
    let mut funcs = funcs.into_vec();
    let alloca_space = AddrSpace::from_raw(data_layout.alloca_space)
        .map_err(|e| FuelError::Internal(format!("alloca address space: {e:?}")))?;

    let mut t = Tables {
        symbols: &mut symbols,
        types: &mut types,
        consts: &mut consts,
        meter,
    };
    let void = t.ty(Type::Void)?;
    let i1 = t.ty(Type::Int(1))?;
    let i32 = t.ty(Type::Int(32))?;
    let i64 = t.ty(Type::Int(64))?;
    let state_ty = t.ty(Type::Array { elem: i32, len: 2 })?;
    let thread_ptr = t.ty(Type::Ptr {
        space: alloca_space,
    })?;
    let device_ptr = t.ty(Type::Ptr {
        space: AddrSpace::Device,
    })?;
    let int = |t: &mut Tables, ty, v| -> Result<Value, FuelError> {
        Ok(Value::Const(t.constant(ty, Const::Int(v))?))
    };
    let word = [
        int(&mut t, i64, 0)?,
        int(&mut t, i64, WORD_CONSUMED_CHUNKS)?,
        int(&mut t, i64, WORD_POOL_CHUNKS)?,
        int(&mut t, i64, WORD_INVOCATION_FUEL)?,
    ];
    let zero = int(&mut t, i32, 0)?;
    let one = int(&mut t, i32, 1)?;
    let order = int(&mut t, i32, ORDER_RELAXED)?;
    let scope = int(&mut t, i32, SCOPE_DEVICE)?;
    let volatile = int(&mut t, i1, 1)?;
    let flags = if flagged {
        Some(int(&mut t, i32, MEM_FLAGS_NONE)?)
    } else {
        None
    };
    let null_status = Value::Const(t.constant(device_ptr, Const::Null)?);

    // The runtime intrinsics: reuse a declaration of the same name, else declare one, in the form
    // of the module's AIR version.
    let immediates = |head: &[TypeId]| -> Vec<TypeId> {
        let mut params = head.to_vec();
        params.extend([i32, i32]);
        params.extend(flagged.then_some(i32));
        params.push(i1);
        params
    };
    let load_ty = t.ty(Type::Function {
        ret: i32,
        params: immediates(&[device_ptr]).into(),
        vararg: false,
    })?;
    let rmw_ty = t.ty(Type::Function {
        ret: i32,
        params: immediates(&[device_ptr, i32]).into(),
        vararg: false,
    })?;
    // The typed-pointer declaration is `i32 (i32 addrspace(1)*, ...)`, as Metal declares it.
    // The pointer parameter's pointee is recorded rather than left to inference: a module whose
    // loops are all barrier loops calls the load and the add nowhere, and nothing else would
    // constrain it (metal-opt refuses `i8 addrspace(1)*` there).
    let callee =
        |t: &mut Tables, funcs: &mut Vec<Function>, pointees: &mut Pointees, which: &Intrinsic| {
            let fn_ty = if which.rmw { rmw_ty } else { load_ty };
            let sym = t.sym(which.name)?;
            let existing = funcs.iter().position(|f| f.name == sym);
            let decl = match existing {
                Some(at) => {
                    let decl = FuncId::new(at as u32);
                    let pointee = pointees.param(decl, 0);
                    if funcs[at].ty != fn_ty
                        || funcs[at].body.is_some()
                        || pointee.is_some_and(|p| p != i32)
                    {
                        return Err(FuelUnsupported::IntrinsicSignature {
                            name: which.name.into(),
                        }
                        .into());
                    }
                    pointees.set_param(decl, 0, Some(i32));
                    decl
                }
                None => {
                    let params = if which.rmw { 5 } else { 4 } + usize::from(flagged);
                    funcs.push(Function {
                        name: sym,
                        ty: fn_ty,
                        linkage: Linkage::External,
                        visibility: Default::default(),
                        unnamed_addr: Default::default(),
                        cconv: CallConv::C,
                        attrs: FnAttrs::default(),
                        ret_attrs: ParamAttrs::default(),
                        params: vec![Param::default(); params].into(),
                        body: None,
                        section: None,
                        align: None,
                    });
                    let decl = FuncId::new(funcs.len() as u32 - 1);
                    pointees.set_param(decl, 0, Some(i32));
                    decl
                }
            };
            let mut args: Vec<Value> = if which.rmw {
                vec![null_status, one, order, scope]
            } else {
                vec![null_status, order, scope]
            };
            args.extend(flags);
            args.push(volatile);
            let class = classify(which.name.as_bytes(), sym, fn_ty, t.types);
            let bound = match class {
                Some(class @ DeclClass::Air(_)) => bind(class, &args, t.consts)
                    .map_err(|e| FuelError::Internal(format!("{}: {e:?}", which.name)))?,
                _ => {
                    return Err(FuelError::Internal(format!(
                        "{} does not classify as an AIR intrinsic",
                        which.name
                    )))
                }
            };
            match bound {
                Bound::Air(op) => Ok(Callee::Air { decl, op }),
                _ => Err(FuelError::Internal(format!(
                    "{} does not bind as an AIR intrinsic",
                    which.name
                ))),
            }
        };
    let load = callee(&mut t, &mut funcs, &mut pointees, &LOAD)?;
    let add = callee(&mut t, &mut funcs, &mut pointees, &ADD)?;
    let or = callee(&mut t, &mut funcs, &mut pointees, &OR)?;
    let ids = Ids {
        void,
        i1,
        i32,
        state_ty,
        thread_ptr,
        device_ptr,
        zero,
        one,
        chunk_mask: int(&mut t, i32, CHUNK_MASK)?,
        exhausted: int(&mut t, i32, u64::from(EXHAUSTED))?,
        order,
        scope,
        flags,
        volatile,
        word,
        one_count: int(&mut t, i32, 1)?,
        load,
        add,
        or,
        align: Align::from_bytes(4),
    };

    // The prototype every function but the entry grows into (its parameters, then the state and
    // the status buffer), by its original type: an indirect call whose closed callee passes the
    // state calls its targets through their grown type, and the plan gives every target of such a
    // call the state, so one original type grows into one type.
    let mut grown = BTreeMap::<TypeId, TypeId>::new();
    for &f in plan.funcs.keys().filter(|&&f| f != plan.entry) {
        let ty = funcs
            .get(f.index())
            .ok_or_else(|| FuelError::Internal("planned function".into()))?
            .ty;
        let Some(Type::Function {
            ret,
            params,
            vararg,
        }) = t.types.get(ty).cloned()
        else {
            return Err(FuelError::Internal("function type".into()));
        };
        let mut params = params.to_vec();
        params.push(thread_ptr);
        params.push(device_ptr);
        let new = t.ty(Type::Function {
            ret,
            params: params.into(),
            vararg,
        })?;
        grown.insert(ty, new);
    }

    // Signatures, per-function constants, and the bodies.
    let mut report = exempt;
    let mut rebuilt = Vec::with_capacity(funcs.len());
    for (index, function) in funcs.into_iter().enumerate() {
        let f = FuncId::new(index as u32);
        let Some(fp) = plan.funcs.get(&f) else {
            rebuilt.push(function);
            continue;
        };
        let is_entry = f == plan.entry;
        let Some(Type::Function {
            ret,
            params,
            vararg,
        }) = t.types.get(function.ty).cloned()
        else {
            return Err(FuelError::Internal("function type".into()));
        };
        let nparams = params.len() as u32;
        let ty = if is_entry {
            let mut new_params = params.to_vec();
            new_params.push(device_ptr);
            pointees.set_param(f, nparams as usize, Some(i32));
            t.ty(Type::Function {
                ret,
                params: new_params.into(),
                vararg,
            })?
        } else {
            pointees.set_param(f, nparams as usize, Some(i32));
            pointees.set_param(f, nparams as usize + 1, Some(i32));
            grown
                .get(&function.ty)
                .copied()
                .ok_or_else(|| FuelError::Internal("grown prototype".into()))?
        };
        let ret_zero = if ret == void {
            None
        } else {
            Some(Value::Const(t.constant(ret, Const::Null)?))
        };
        let mut budgets = Vec::new();
        for l in &fp.loops {
            budgets.push(match &l.barrier {
                Some(b) => Some(int(&mut t, i32, u64::from(b.budget))?),
                None => None,
            });
        }
        let Function {
            name,
            ty: _,
            linkage,
            visibility,
            unnamed_addr,
            cconv,
            mut attrs,
            ret_attrs,
            params: param_attrs,
            body,
            section,
            align,
        } = function;
        let mut param_attrs = param_attrs.into_vec();
        param_attrs.push(Param::default());
        if !is_entry {
            param_attrs.push(Param::default());
        }
        let body =
            body.ok_or_else(|| FuelError::Internal("planned function has no body".into()))?;
        report.tail_calls += fp
            .fueled_calls
            .iter()
            .filter(|&&call| {
                matches!(body.insts.get(call).map(|inst| &inst.kind),
                    Some(InstKind::Call(c)) if c.tail == TailKind::Tail)
            })
            .count() as u32;
        let (body, renumbered) = Rewrite {
            ids: &ids,
            plan: fp,
            is_entry,
            nparams,
            ret_zero,
            budgets: &budgets,
            grown: &grown,
        }
        .run(body, t.meter)?;
        // The calls that pass the state, under their original ids, then every call's new id.
        for &call in &fp.fueled_calls {
            if let Some(site) = enums.call_mut(f, call) {
                writes_state_kinds(site);
            }
        }
        enums.renumber_calls(f, |call| renumbered.get(call.index()).copied().flatten());
        report.functions += 1;
        for l in &fp.loops {
            if l.barrier.is_some() {
                report.barrier_loops += 1;
            } else {
                report.loops += 1;
            }
        }
        writes_state(&mut attrs, enums.func_mut(f));
        rebuilt.push(Function {
            name,
            ty,
            linkage,
            visibility,
            unnamed_addr,
            cconv,
            attrs,
            ret_attrs,
            params: param_attrs.into(),
            body: Some(body),
            section,
            align,
        });
    }

    // The status buffer in the entry's stage argument list.
    let md = |t: &mut Tables, metadata: &mut Arena<MdId, Md>, m: Md| {
        metadata.push(m, t.meter).map_err(internal)
    };
    let string = |t: &mut Tables, metadata: &mut Arena<MdId, Md>, s: &str| {
        let sym = t.sym(s)?;
        metadata.push(Md::String(sym), t.meter).map_err(internal)
    };
    let mut ops = Vec::new();
    let entry_params = match rebuilt
        .get(plan.entry.index())
        .and_then(|f| t.types.get(f.ty))
    {
        Some(Type::Function { params, .. }) => params.len() as u64 - 1,
        _ => return Err(FuelError::Internal("entry type".into())),
    };
    let int_md = |t: &mut Tables, metadata: &mut Arena<MdId, Md>, v: u64| {
        let c = t.constant(i32, Const::Int(v))?;
        metadata.push(Md::Value(c), t.meter).map_err(internal)
    };
    ops.push(int_md(&mut t, &mut metadata, entry_params)?);
    for key in ["air.buffer", "air.location_index"] {
        ops.push(string(&mut t, &mut metadata, key)?);
    }
    ops.push(int_md(
        &mut t,
        &mut metadata,
        u64::from(config.status_slot),
    )?);
    ops.push(int_md(&mut t, &mut metadata, 1)?);
    for key in ["air.read_write", "air.address_space"] {
        ops.push(string(&mut t, &mut metadata, key)?);
    }
    ops.push(int_md(
        &mut t,
        &mut metadata,
        u64::from(AddrSpace::Device.raw()),
    )?);
    ops.push(string(&mut t, &mut metadata, "air.arg_type_size")?);
    ops.push(int_md(&mut t, &mut metadata, 4)?);
    ops.push(string(&mut t, &mut metadata, "air.arg_type_align_size")?);
    ops.push(int_md(&mut t, &mut metadata, 4)?);
    for key in ["air.arg_type_name", "uint", "air.arg_name", STATUS_ARG_NAME] {
        ops.push(string(&mut t, &mut metadata, key)?);
    }
    let status_node = md(
        &mut t,
        &mut metadata,
        Md::Node {
            distinct: false,
            ops: ops.into_iter().map(Some).collect(),
        },
    )?;
    let mut args = site.args;
    args.push(Some(status_node));
    let args_node = md(
        &mut t,
        &mut metadata,
        Md::Node {
            distinct: site.args_distinct,
            ops: args.into(),
        },
    )?;
    let mut entry_ops = site.ops;
    entry_ops[2] = Some(args_node);
    let entry_node = md(
        &mut t,
        &mut metadata,
        Md::Node {
            distinct: site.distinct,
            ops: entry_ops.into(),
        },
    )?;
    let list = &named_md[site.named];
    let mut list_ops = list.ops.to_vec();
    list_ops[site.op] = entry_node;
    named_md[site.named] = NamedMd {
        name: list.name,
        ops: list_ops.into(),
    };

    let mut func_arena = Arena::new();
    for function in rebuilt {
        func_arena.push(function, meter).map_err(internal)?;
    }
    let parts = ModuleParts {
        symbols,
        types,
        consts,
        globals,
        funcs: func_arena,
        metadata,
        named_md,
        triple,
        source_filename,
        data_layout,
    };
    let module =
        Module::new(parts, meter).map_err(|e| FuelError::Internal(format!("verify: {e}")))?;
    Ok(Instrumented {
        module,
        pointees,
        enums,
        report,
    })
}

/// The rewrite of one planned function.
struct Rewrite<'a> {
    ids: &'a Ids,
    plan: &'a FnPlan,
    is_entry: bool,
    nparams: u32,
    ret_zero: Option<Value>,
    /// Per loop of the plan, the barrier budget constant.
    budgets: &'a [Option<Value>],
    /// Each original prototype of a function that takes the state, and the one it grows into.
    grown: &'a BTreeMap<TypeId, TypeId>,
}

/// Values every check of one function uses.
struct Frame {
    /// The thread's fuel word.
    fuel: Value,
    /// The thread's `forced` word.
    forced: Value,
    status: Value,
    /// Per loop of the plan, its barrier counter.
    counters: Vec<Option<Value>>,
    exit: Option<BlockId>,
}

impl Rewrite<'_> {
    /// The rewritten body, and each original instruction's new id.
    fn run(&self, body: Body, meter: &mut Meter) -> Result<(Body, Vec<Option<InstId>>), FuelError> {
        let ids = self.ids;
        // The block of every planned call, from the original layout.
        let mut calls_by_block = BTreeMap::<BlockId, Vec<InstId>>::new();
        for (b, block) in body.blocks.iter() {
            for i in block.inst_ids() {
                if self.plan.fueled_calls.contains(&i) {
                    calls_by_block.entry(b).or_default().push(i);
                }
            }
        }
        let original = body.blocks.len();
        let mut e = Editor::new(body);

        // The prologue at the top of the entry block.
        let mut run = Vec::new();
        let mut detached = |e: &mut Editor, ty, kind| {
            let (id, v) = e.detached(ty, kind);
            run.push(id);
            v
        };
        let (fuel, forced, status) = if self.is_entry {
            let status = Value::Arg(ArgIdx::new(self.nparams));
            let state = detached(
                &mut e,
                ids.thread_ptr,
                InstKind::Alloca {
                    ty: ids.state_ty,
                    count: ids.one_count,
                    align: ids.align,
                },
            );
            let fuel = detached(
                &mut e,
                ids.thread_ptr,
                self.gep(ids.state_ty, state, &[ids.word[0], ids.word[0]]),
            );
            let forced = detached(
                &mut e,
                ids.thread_ptr,
                self.gep(ids.state_ty, state, &[ids.word[0], ids.word[1]]),
            );
            let word = detached(
                &mut e,
                ids.device_ptr,
                self.gep(ids.i32, status, &[ids.word[3]]),
            );
            let initial = detached(&mut e, ids.i32, self.load(word));
            detached(&mut e, ids.void, self.store(initial, fuel));
            detached(&mut e, ids.void, self.store(ids.zero, forced));
            (fuel, forced, status)
        } else {
            let fuel = Value::Arg(ArgIdx::new(self.nparams));
            let status = Value::Arg(ArgIdx::new(self.nparams + 1));
            let forced = detached(
                &mut e,
                ids.thread_ptr,
                self.gep(ids.i32, fuel, &[ids.word[1]]),
            );
            (fuel, forced, status)
        };
        let mut counters = Vec::new();
        for l in &self.plan.loops {
            let counter = l.barrier.as_ref().map(|_| {
                detached(
                    &mut e,
                    ids.thread_ptr,
                    InstKind::Alloca {
                        ty: ids.i32,
                        count: ids.one_count,
                        align: ids.align,
                    },
                )
            });
            counters.push(counter);
        }
        e.prepend(BlockId::new(0), run);
        let mut frame = Frame {
            fuel,
            forced,
            status,
            counters,
            exit: None,
        };

        // After every fueled call: pass the state (dropping its `tail` promise), and return if the
        // callee was forced out.
        let mut last: Vec<BlockId> = (0..original as u32).map(BlockId::new).collect();
        for (b, calls) in &calls_by_block {
            for &call in calls {
                e.extend_call(call, &[frame.fuel, frame.status], self.grown)
                    .map_err(FuelError::Internal)?;
                let head = last[b.index()];
                let tail = e
                    .split_after(head, call)
                    .ok_or_else(|| FuelError::Internal("call not in its block".into()))?;
                let exit = self.exit(&mut e, &mut frame);
                let was = e.push(head, ids.i32, self.load(frame.forced));
                let hit = e.push(head, ids.i1, self.icmp(IntPredicate::Ne, was, ids.zero));
                e.push(head, ids.void, self.cond_br(hit, exit, tail));
                last[b.index()] = tail;
            }
        }

        // The loop checks and the barrier counters' resets.
        let mut phi_map = BTreeMap::<(BlockId, BlockId), BlockId>::new();
        let mut single = BTreeSet::new();
        for (index, l) in self.plan.loops.iter().enumerate() {
            for &p in &l.latches {
                let (first, join) = match frame.counters[index] {
                    None => self.fuel_check(&mut e, &mut frame),
                    Some(counter) => self.barrier_check(&mut e, &mut frame, counter),
                };
                e.push(join, ids.void, self.br(l.header));
                e.retarget(last[p.index()], l.header, first);
                phi_map.insert((l.header, p), join);
                single.insert(join);
            }
            if let (Some(b), Some(counter), Some(budget)) =
                (&l.barrier, frame.counters[index], self.budgets[index])
            {
                for &p in &b.preheaders {
                    let reset = e.new_block();
                    e.push(reset, ids.void, self.store(budget, counter));
                    e.push(reset, ids.void, self.br(l.header));
                    e.retarget(last[p.index()], l.header, reset);
                    phi_map.insert((l.header, p), reset);
                    single.insert(reset);
                }
            }
        }
        for h in 0..original as u32 {
            let h = BlockId::new(h);
            e.remap_phis(
                h,
                |p| match phi_map.get(&(h, p)) {
                    Some(&j) => j,
                    None => last.get(p.index()).copied().unwrap_or(p),
                },
                |b| single.contains(&b),
            );
        }
        e.finish(meter).map_err(internal)
    }

    /// The function's forced-exit block, built on first use.
    fn exit(&self, e: &mut Editor, frame: &mut Frame) -> BlockId {
        if let Some(exit) = frame.exit {
            return exit;
        }
        let ids = self.ids;
        let exit = e.new_block();
        e.push(
            exit,
            ids.i32,
            self.call(&ids.or, &ids.atomic(&[frame.status, ids.exhausted])),
        );
        e.push(exit, ids.void, self.store(ids.one, frame.forced));
        e.push(exit, ids.void, InstKind::Ret(self.ret_zero));
        frame.exit = Some(exit);
        exit
    }

    /// The check on one latch edge: `(first block, join block)`.
    fn fuel_check(&self, e: &mut Editor, frame: &mut Frame) -> (BlockId, BlockId) {
        let ids = self.ids;
        let exit = self.exit(e, frame);
        let (check, spend, pool, draw, join) = (
            e.new_block(),
            e.new_block(),
            e.new_block(),
            e.new_block(),
            e.new_block(),
        );
        let fuel = e.push(check, ids.i32, self.load(frame.fuel));
        let empty = e.push(check, ids.i1, self.icmp(IntPredicate::Eq, fuel, ids.zero));
        e.push(check, ids.void, self.cond_br(empty, exit, spend));

        let left = e.push(spend, ids.i32, self.bin(BinOp::Sub, fuel, ids.one));
        e.push(spend, ids.void, self.store(left, frame.fuel));
        let local = e.push(spend, ids.i32, self.bin(BinOp::And, left, ids.chunk_mask));
        let chunk = e.push(spend, ids.i1, self.icmp(IntPredicate::Eq, local, ids.zero));
        e.push(spend, ids.void, self.cond_br(chunk, pool, join));

        let bits = e.push(
            pool,
            ids.i32,
            self.call(&ids.load, &ids.atomic(&[frame.status])),
        );
        let bit = e.push(pool, ids.i32, self.bin(BinOp::And, bits, ids.exhausted));
        let done = e.push(pool, ids.i1, self.icmp(IntPredicate::Ne, bit, ids.zero));
        e.push(pool, ids.void, self.cond_br(done, exit, draw));

        let consumed = e.push(
            draw,
            ids.device_ptr,
            self.gep(ids.i32, frame.status, &[ids.word[1]]),
        );
        let taken = e.push(
            draw,
            ids.i32,
            self.call(&ids.add, &ids.atomic(&[consumed, ids.one])),
        );
        let size = e.push(
            draw,
            ids.device_ptr,
            self.gep(ids.i32, frame.status, &[ids.word[2]]),
        );
        let chunks = e.push(draw, ids.i32, self.call(&ids.load, &ids.atomic(&[size])));
        let over = e.push(draw, ids.i1, self.icmp(IntPredicate::Uge, taken, chunks));
        e.push(draw, ids.void, self.cond_br(over, exit, join));
        (check, join)
    }

    /// The barrier-counter check on one latch edge: `(first block, join block)`.
    fn barrier_check(
        &self,
        e: &mut Editor,
        frame: &mut Frame,
        counter: Value,
    ) -> (BlockId, BlockId) {
        let ids = self.ids;
        let exit = self.exit(e, frame);
        let (check, join) = (e.new_block(), e.new_block());
        let left = e.push(check, ids.i32, self.load(counter));
        let empty = e.push(check, ids.i1, self.icmp(IntPredicate::Eq, left, ids.zero));
        e.push(check, ids.void, self.cond_br(empty, exit, join));
        let next = e.push(join, ids.i32, self.bin(BinOp::Sub, left, ids.one));
        e.push(join, ids.void, self.store(next, counter));
        (check, join)
    }

    fn gep(&self, src_ty: TypeId, base: Value, indices: &[Value]) -> InstKind {
        InstKind::Gep {
            src_ty,
            base,
            indices: indices.into(),
            inbounds: true,
        }
    }

    fn load(&self, ptr: Value) -> InstKind {
        InstKind::Load {
            ptr,
            align: self.ids.align,
            volatile: false,
            atomic: None,
            range: None,
        }
    }

    fn store(&self, val: Value, ptr: Value) -> InstKind {
        InstKind::Store {
            val,
            ptr,
            align: self.ids.align,
            volatile: false,
            atomic: None,
        }
    }

    fn bin(&self, op: BinOp, a: Value, b: Value) -> InstKind {
        InstKind::BinOp {
            op,
            flags: IntFlags::NONE,
            fmf: FastMath::NONE,
            a,
            b,
        }
    }

    fn icmp(&self, pred: IntPredicate, a: Value, b: Value) -> InstKind {
        InstKind::ICmp { pred, a, b }
    }

    fn br(&self, dest: BlockId) -> InstKind {
        InstKind::Br { dest, hints: None }
    }

    fn cond_br(&self, c: Value, t: BlockId, f: BlockId) -> InstKind {
        InstKind::CondBr {
            c,
            t,
            f,
            hints: None,
        }
    }

    fn call(&self, callee: &Callee, args: &[Value]) -> InstKind {
        InstKind::Call(Box::new(Call {
            callee: callee.clone(),
            args: args.into(),
            fmf: FastMath::NONE,
            tail: TailKind::None,
            attrs: None,
            range: None,
        }))
    }
}
