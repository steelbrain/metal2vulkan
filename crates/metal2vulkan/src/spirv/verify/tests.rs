//! Verifier tests: builder modules pass; each rule rejects a module tampered to break exactly it
//! (L0 negative tests); profile rules; metering.

use super::*;
use crate::budget::{Budget, Meter};
use crate::device::requirements::{StageCreateFlag, VkFeature, VkProperty};
use crate::device::subgroup::SubgroupRequirement;
use crate::error::{Capability, Limit};
use crate::pipeline::sym::{ResourceKind, ResourceSlot, Stage};
use crate::spirv::builder::constructs::{ArmEnd, ArmFn, LoopCtx, LoopSpec};
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
use crate::spirv::ids::{BlockId, FuncId, GlobalId, TypeId, Value};
use crate::spirv::ir::{
    natural_layout, AtomicOp, BinOp, Block, ConstBits, Constant, ConvOp, DerivOp, ExecMode,
    ExecModel, Inst, InstNode, IoDir, LayoutRule, MemAccess, Merge, Scope, Semantics, StorageClass,
    Terminator, Type, TypedModule, UnOp, ValueDef, ValueOrigin,
};
use crate::spirv::testing::{assert_valid, committed_profiles};

// ---- Fixture ----

#[derive(Clone, Copy)]
struct Tys {
    void: TypeId,
    bool_: TypeId,
    u32_: TypeId,
    f32_: TypeId,
}

fn tys(b: &mut ModuleBuilder) -> Tys {
    Tys {
        void: b.ty(Type::Void).expect("void"),
        bool_: b.ty(Type::Bool).expect("bool"),
        u32_: b
            .ty(Type::Int {
                width: 32,
                signed: false,
            })
            .expect("u32"),
        f32_: b.ty(Type::Float { width: 32 }).expect("f32"),
    }
}

fn konst(f: &mut FunctionBuilder<'_>, ty: TypeId, bits: u64) -> Value {
    let c = f
        .module()
        .constant(Constant::Scalar {
            ty,
            bits: ConstBits(bits),
        })
        .expect("constant");
    f.module().const_value(c).expect("constant value")
}

/// A `Block { u32 data[]; }` storage buffer in slot 0.
fn ssbo(b: &mut ModuleBuilder, u32_: TypeId) -> GlobalId {
    let rta = b
        .ty(Type::RuntimeArray {
            elem: u32_,
            stride: Some(4),
        })
        .expect("runtime array");
    let layout = natural_layout(b.types(), LayoutRule::Std430, true, &[rta]).expect("layout");
    let block = b
        .ty(Type::Struct {
            members: [rta].into(),
            layout: Some(layout),
        })
        .expect("block");
    b.resource(
        ResourceSlot {
            stage: Stage::Kernel,
            kind: ResourceKind::Buffer,
            index: 0,
        },
        StorageClass::StorageBuffer,
        block,
    )
    .expect("resource")
}

fn store_at(f: &mut FunctionBuilder<'_>, out: GlobalId, t: Tys, i: Value, v: Value) {
    let zero = konst(f, t.u32_, 0);
    let base = f.module().global_value(out).expect("global");
    let p = f.access_chain(base, &[zero, i]).expect("access chain");
    f.store(p, v, MemAccess::default()).expect("store");
}

fn arm<'a, C>(
    f: impl FnOnce(&mut FunctionBuilder<'_>, &mut C) -> Result<ArmEnd, crate::spirv::error::BuildError>
        + 'a,
) -> ArmFn<'a, C> {
    Box::new(f)
}

fn empty<'a>() -> ArmFn<'a, ()> {
    arm(|_, _| Ok(ArmEnd::FallThrough(Vec::new())))
}

/// The rich kernel and the values the tests tamper with.
struct Kernel {
    m: TypedModule,
    main: FuncId,
    helper: FuncId,
    /// `gl_GlobalInvocationID.x`.
    i: Value,
    /// `i * 3` and `a + 1`, both in the entry block.
    a: Value,
    b: Value,
    /// `b + 2`, in the first if's then arm.
    t: Value,
}

/// Main's blocks in layout order (construct preorder):
///
/// ```text
///  0 E      if1 header (i, a, b, c)       9 first  inner if header (loop body)
///  1 then1  t = b + 2                    10 brk    break
///  2 else1                               11 ifm    counter += 1
///  3 M1     phi; switch header           12 cont   continue block
///  4 c0     case 0                       13 LM     loop merge; call helper; if2 header
///  5 c1     case 1, 5                    14 then2
///  6 dflt   default                      15 else2
///  7 M2     switch merge; preheader      16 M3     store b; return
///  8 H      loop header
/// ```
fn kernel() -> Kernel {
    kernel_with(None)
}

/// [`kernel`], its plan's subgroup requirement `subgroup` recorded.
fn kernel_with(subgroup: Option<SubgroupRequirement>) -> Kernel {
    let mut b = ModuleBuilder::new();
    let t = tys(&mut b);
    let out = ssbo(&mut b, t.u32_);

    let mut h = b.function(t.void).expect("helper");
    let zero = konst(&mut h, t.u32_, 0);
    let seven = konst(&mut h, t.u32_, 7);
    store_at(&mut h, out, t, zero, seven);
    h.ret(None).expect("ret");
    let helper = h.finish().expect("finish helper");

    let mut f = b.function(t.void).expect("main");
    let u32_ = t.u32_;
    let uvec3 = f
        .module()
        .ty(Type::Vector { elem: u32_, len: 3 })
        .expect("uvec3");
    let g = f
        .module()
        .builtin(crate::spv::BuiltIn::GlobalInvocationId, IoDir::Input, uvec3)
        .expect("builtin");
    let gp = f.module().global_value(g).expect("global");
    let gv = f.load(gp, MemAccess::default()).expect("load");
    let i = f.composite_extract(gv, &[0]).expect("extract");
    let three = konst(&mut f, u32_, 3);
    let one = konst(&mut f, u32_, 1);
    let two = konst(&mut f, u32_, 2);
    let five = konst(&mut f, u32_, 5);
    let four = konst(&mut f, u32_, 4);
    let a = f.binary(BinOp::IMul, i, three).expect("mul");
    let bv = f.binary(BinOp::IAdd, a, one).expect("add");
    let c = f.binary(BinOp::ULessThan, i, five).expect("cmp");
    let mut tv = None;
    let merged = f
        .if_(
            &mut tv,
            c,
            arm(move |fb, tv: &mut Option<Value>| {
                let x = fb.binary(BinOp::IAdd, bv, two)?;
                *tv = Some(x);
                Ok(ArmEnd::FallThrough(vec![x]))
            }),
            Some(arm(move |_, _| Ok(ArmEnd::FallThrough(vec![bv])))),
        )
        .expect("if1");
    store_at(&mut f, out, t, i, merged.0[0]);
    f.switch_(
        &mut (),
        i,
        vec![(vec![0], empty()), (vec![1, 5], empty())],
        empty(),
    )
    .expect("switch");
    let var = f.local_var(u32_, None).expect("var");
    let counter = f.var_ptr(var).expect("ptr");
    let zero = konst(&mut f, u32_, 0);
    f.store(counter, zero, MemAccess::default()).expect("store");
    f.loop_(
        &mut (),
        LoopSpec::default(),
        Box::new(
            move |fb: &mut FunctionBuilder<'_>, _: &mut (), lc: &LoopCtx| {
                let n = fb.load(counter, MemAccess::default())?;
                let done = fb.binary(BinOp::UGreaterThanEqual, n, four)?;
                fb.if_(
                    &mut (),
                    done,
                    arm(|fb, _| {
                        lc.break_(fb)?;
                        Ok(ArmEnd::Exits)
                    }),
                    None,
                )?;
                let n1 = fb.binary(BinOp::IAdd, n, one)?;
                fb.store(counter, n1, MemAccess::default())
            },
        ),
    )
    .expect("loop");
    f.call(helper, &[]).expect("call");
    f.if_(&mut (), c, empty(), Some(empty())).expect("if2");
    store_at(&mut f, out, t, i, bv);
    f.ret(None).expect("ret");
    let main = f.finish().expect("finish main");
    b.entry(
        ExecModel::GLCompute,
        main,
        vec![ExecMode::LocalSize { x: 8, y: 1, z: 1 }],
    )
    .expect("entry");
    if let Some(r) = subgroup {
        b.subgroup(r);
    }
    let _ = t.bool_;
    Kernel {
        m: b.finalize().expect("finalize"),
        main,
        helper,
        i,
        a,
        b: bv,
        t: tv.expect("then arm ran"),
    }
}

impl Kernel {
    /// Main's block at layout position `p`.
    fn at(&self, p: usize) -> BlockId {
        self.m.functions[self.main.index()].blocks[p]
    }

    fn helper_entry(&self) -> BlockId {
        self.m.functions[self.helper.index()].blocks[0]
    }

    fn block(&mut self, b: BlockId) -> &mut Block {
        &mut self.m.blocks[b.index()]
    }
}

fn meter() -> Meter {
    Meter::new(&Budget::default())
}

fn structure(m: &TypedModule) -> Result<ModuleStats, VerifyError> {
    check_structure(m, &mut meter())
}

/// The violation `m` breaks.
#[track_caller]
fn violation(m: &TypedModule) -> Violation {
    match structure(m) {
        Err(VerifyError::Rule(v)) => v,
        other => panic!("expected a broken rule, got {other:?}"),
    }
}

/// Tamper with a fresh kernel and return the violation it breaks.
#[track_caller]
fn tampered(edit: impl FnOnce(&mut Kernel)) -> Violation {
    let mut k = kernel();
    edit(&mut k);
    violation(&k.m)
}

/// The value operand of the (only) store of `b`.
fn store_value(b: &mut Block) -> &mut Value {
    b.insts
        .iter_mut()
        .rev()
        .find_map(|n| match &mut n.inst {
            Inst::Store { value, .. } => Some(value),
            _ => None,
        })
        .expect("a store")
}

// ---- Positive ----

#[test]
fn the_fixture_has_the_documented_layout() {
    let k = kernel();
    let term = |p| &k.m.blocks[k.at(p).index()].term;
    assert!(matches!(term(0), Terminator::CondBranch { .. }));
    assert!(matches!(term(3), Terminator::Switch { .. }));
    assert_eq!(k.m.blocks[k.at(8).index()].merge.map(|_| ()), Some(()));
    assert_eq!(k.m.loops()[0].header, k.at(8));
    assert_eq!(k.m.loops()[0].preheader, k.at(7));
    assert_eq!(k.m.loops()[0].cont, k.at(12));
    assert_eq!(k.m.loops()[0].merge, k.at(13));
    assert_eq!(*term(12), Terminator::Branch(k.at(8)));
    assert!(matches!(term(13), Terminator::CondBranch { .. }));
    assert_eq!(*term(16), Terminator::Return);
    assert_eq!(k.m.functions[k.main.index()].blocks.len(), 17);
}

#[test]
fn a_builder_module_passes_and_reports_what_it_covered() {
    let k = kernel();
    let r = structure(&k.m).expect("valid");
    assert_eq!(r.functions, 2);
    assert_eq!(r.blocks, 18);
    assert_eq!(r.loops, 1);
    // if1, the switch, the loop, the break's if, if2.
    assert_eq!(r.constructs, 5);
    assert_eq!(r.reachable_blocks, 18);
}

#[test]
fn the_kernel_passes_l3_on_every_committed_profile() {
    let profiles = committed_profiles().expect("profiles");
    assert!(profiles.len() >= 10, "{} profiles", profiles.len());
    for (stem, p) in &profiles {
        let r = verify(&kernel().m, p, &mut meter());
        assert!(r.is_ok(), "{stem}: {r:?}");
        assert_valid(kernel().m, p);
    }
}

#[test]
fn the_ledger_is_released_on_success_and_on_failure() {
    let mut m = meter();
    check_structure(&kernel().m, &mut m).expect("valid");
    assert_eq!(m.ledger(), 0);
    assert!(m.ledger_peak() > 0, "tables are charged");
    let mut k = kernel();
    k.m.interface.clear();
    let mut m = meter();
    assert!(check_structure(&k.m, &mut m).is_err());
    assert_eq!(m.ledger(), 0);
}

#[test]
fn work_is_metered() {
    let mut m = Meter::with_caps(&Budget::default(), 4, u64::MAX);
    assert_eq!(
        check_structure(&kernel().m, &mut m),
        Err(VerifyError::Budget(Error::LimitExceeded(Limit::WorkUnits)))
    );
    assert_eq!(m.ledger(), 0);
}

// ---- Ownership ----

#[test]
fn every_function_has_blocks() {
    let v = tampered(|k| k.m.functions[k.helper.index()].blocks.clear());
    assert_eq!(v, Violation::EmptyFunction(kernel().helper));
}

#[test]
fn a_block_belongs_to_the_one_function_it_names() {
    let v = tampered(|k| {
        let b = k.at(16);
        k.m.functions[k.helper.index()].blocks.push(b);
    });
    assert!(matches!(v, Violation::BlockOwnership(_)), "{v:?}");
}

#[test]
fn branches_stay_in_their_function() {
    let v = tampered(|k| {
        let (e, m3) = (k.helper_entry(), k.at(16));
        k.block(e).term = Terminator::Branch(m3);
    });
    assert!(matches!(v, Violation::ForeignTarget { .. }), "{v:?}");
}

#[test]
fn nothing_branches_to_a_function_entry() {
    let v = tampered(|k| {
        let (else1, e) = (k.at(2), k.at(0));
        k.block(else1).term = Terminator::Branch(e);
    });
    assert_eq!(v, Violation::EntryBlockTarget(kernel().at(0)));
}

// ---- SSA ----

#[test]
fn a_use_follows_its_definition_in_the_block() {
    let k0 = kernel();
    let (a, e) = (k0.a, k0.at(0));
    let v = tampered(|k| {
        let insts = &mut k.block(e).insts;
        let pa = insts.iter().position(|n| n.result == Some(a)).expect("a");
        insts.swap(pa, pa + 1);
    });
    assert_eq!(v, Violation::NotDominated { value: a, block: e });
}

#[test]
fn a_use_is_dominated_by_its_definition() {
    let k0 = kernel();
    let (t, m3) = (k0.t, k0.at(16));
    let v = tampered(|k| *store_value(k.block(m3)) = t);
    assert_eq!(
        v,
        Violation::NotDominated {
            value: t,
            block: m3
        }
    );
}

#[test]
fn a_phi_lists_every_predecessor_once() {
    let m1 = kernel().at(3);
    let v = tampered(|k| {
        k.block(m1).phis[0].incoming.pop();
    });
    assert!(
        matches!(v, Violation::PhiIncoming { block, .. } if block == m1),
        "{v:?}"
    );
    let v = tampered(|k| {
        let first = k.block(m1).phis[0].incoming[0];
        k.block(m1).phis[0].incoming.push(first);
    });
    assert!(matches!(v, Violation::PhiIncoming { .. }), "{v:?}");
}

#[test]
fn a_value_is_defined_once_where_its_origin_says() {
    let k0 = kernel();
    let (a, e, then1) = (k0.a, k0.at(0), k0.at(1));
    let v = tampered(|k| {
        let node = k
            .block(e)
            .insts
            .iter()
            .find(|n| n.result == Some(a))
            .cloned()
            .expect("a");
        k.block(e).insts.push(node);
    });
    assert_eq!(v, Violation::DefinedTwice(a));
    let v = tampered(|k| k.m.values[a.index()].origin = ValueOrigin::Inst(then1));
    assert_eq!(v, Violation::ResultOrigin(a));
}

#[test]
fn a_value_of_another_function_is_foreign() {
    let i = kernel().i;
    let v = tampered(|k| {
        let e = k.helper_entry();
        *store_value(k.block(e)) = i;
    });
    assert!(
        matches!(v, Violation::ForeignValue { value, .. } if value == i),
        "{v:?}"
    );
}

#[test]
fn a_use_of_nothing_is_undefined_or_dangling() {
    let m3 = kernel().at(16);
    let v = tampered(|k| {
        let e = k.at(0);
        let ty = k.m.values[k.b.index()].ty;
        k.m.values.push(ValueDef {
            ty,
            origin: ValueOrigin::Inst(e),
        });
        let fresh = Value::new(k.m.values.len() as u32 - 1);
        *store_value(k.block(m3)) = fresh;
    });
    assert!(
        matches!(v, Violation::Undefined { block, .. } if block == m3),
        "{v:?}"
    );
    let v = tampered(|k| *store_value(k.block(m3)) = Value::new(u32::MAX - 7));
    assert_eq!(
        v,
        Violation::Dangling(crate::spirv::error::Dangling::Value(Value::new(
            u32::MAX - 7
        )))
    );
}

// ---- Structured control flow ----

#[test]
fn a_conditional_branch_needs_a_selection_merge() {
    let lm = kernel().at(13);
    let v = tampered(|k| k.block(lm).merge = None);
    assert_eq!(v, Violation::MissingMerge(lm));
}

#[test]
fn a_merge_declaration_fits_its_terminator() {
    let k0 = kernel();
    let (lm, then2, m3) = (k0.at(13), k0.at(14), k0.at(16));
    let v = tampered(|k| {
        k.block(lm).merge = Some(Merge::Loop {
            merge: m3,
            cont: then2,
        })
    });
    assert_eq!(v, Violation::MergeTerminator(lm));
}

#[test]
fn a_block_merges_one_construct() {
    let k0 = kernel();
    let (e, m2) = (k0.at(0), k0.at(7));
    let v = tampered(|k| k.block(e).merge = Some(Merge::Selection { merge: m2 }));
    assert_eq!(v, Violation::SharedMerge(m2));
}

#[test]
fn a_continue_target_lies_inside_its_loop() {
    let k0 = kernel();
    let (h, cont, lm) = (k0.at(8), k0.at(12), k0.at(13));
    let v = tampered(|k| {
        k.block(h).merge = Some(Merge::Loop {
            merge: cont,
            cont: lm,
        })
    });
    assert_eq!(v, Violation::ConstructLayout(h));
}

#[test]
fn constructs_nest() {
    let k0 = kernel();
    let (e, c1, m1) = (k0.at(0), k0.at(5), k0.at(3));
    let v = tampered(|k| k.block(e).merge = Some(Merge::Selection { merge: c1 }));
    assert_eq!(v, Violation::ConstructNesting(m1));
}

#[test]
fn a_block_follows_its_immediate_dominator() {
    let v = tampered(|k| {
        let e = k.helper_entry();
        let helper = k.helper;
        let n = k.m.blocks.len() as u32;
        let (a, b) = (BlockId::new(n), BlockId::new(n + 1));
        let plain = |term| Block {
            func: helper,
            phis: Vec::new(),
            insts: Vec::new(),
            merge: None,
            term,
        };
        k.m.blocks.push(plain(Terminator::Branch(b)));
        k.m.blocks.push(plain(Terminator::Return));
        k.block(e).term = Terminator::Branch(a);
        // Layout E, B, A: B's immediate dominator A comes after it.
        k.m.functions[helper.index()].blocks.extend([b, a]);
    });
    assert!(matches!(v, Violation::LayoutDominance(_)), "{v:?}");
}

#[test]
fn a_header_dominates_its_merge() {
    // The preheader skips the loop and if2 and branches to if2's merge: every edge is still
    // structured, but if2's header no longer dominates its (reachable) merge.
    let k0 = kernel();
    let (m2, lm, m3) = (k0.at(7), k0.at(13), k0.at(16));
    let v = tampered(|k| k.block(m2).term = Terminator::Branch(m3));
    assert_eq!(
        v,
        Violation::HeaderDominance {
            header: lm,
            block: m3
        }
    );
}

#[test]
fn only_a_continue_block_branches_backward() {
    let k0 = kernel();
    let (then2, lm) = (k0.at(14), k0.at(13));
    let v = tampered(|k| k.block(then2).term = Terminator::Branch(lm));
    assert_eq!(
        v,
        Violation::BackEdge {
            from: then2,
            to: lm
        }
    );
}

#[test]
fn a_construct_is_entered_at_its_header() {
    let k0 = kernel();
    let (m2, then2) = (k0.at(7), k0.at(14));
    let v = tampered(|k| k.block(m2).term = Terminator::Branch(then2));
    assert_eq!(
        v,
        Violation::ConstructEntry {
            from: m2,
            to: then2
        }
    );
}

#[test]
fn a_construct_is_left_to_its_merge() {
    let k0 = kernel();
    let (then1, m2) = (k0.at(1), k0.at(7));
    let v = tampered(|k| k.block(then1).term = Terminator::Branch(m2));
    assert_eq!(
        v,
        Violation::ConstructExit {
            from: then1,
            to: m2
        }
    );
}

#[test]
fn a_break_leaves_only_the_innermost_loop() {
    // The break arm of the loop body branches to if2's merge instead of the loop merge.
    let k0 = kernel();
    let (brk, m3) = (k0.at(10), k0.at(16));
    let v = tampered(|k| k.block(brk).term = Terminator::Branch(m3));
    assert_eq!(v, Violation::ConstructExit { from: brk, to: m3 });
}

#[test]
fn switch_literals_are_distinct() {
    let sw = kernel().at(3);
    let v = tampered(|k| {
        if let Terminator::Switch { cases, .. } = &mut k.block(sw).term {
            let first = cases[0];
            cases.push(first);
        }
    });
    assert_eq!(v, Violation::SwitchLiterals(sw));
}

#[test]
fn switch_cases_do_not_fall_through() {
    let k0 = kernel();
    let (sw, c0, c1) = (k0.at(3), k0.at(4), k0.at(5));
    let v = tampered(|k| k.block(c0).term = Terminator::Branch(c1));
    assert_eq!(
        v,
        Violation::CaseFallthrough {
            header: sw,
            case: c1
        }
    );
}

// ---- Loops ----

#[test]
fn every_loop_merge_has_its_loop_meta() {
    let h = kernel().at(8);
    let v = tampered(|k| k.m.loops.clear());
    assert_eq!(v, Violation::LoopMetaMissing(h));
}

#[test]
fn a_loop_meta_agrees_with_its_loop() {
    let v = tampered(|k| k.m.loops[0].preheader = k.at(0));
    assert_eq!(v, Violation::LoopMetaMismatch(0));
    let v = tampered(|k| {
        let dup = k.m.loops[0];
        k.m.loops.push(dup);
    });
    assert_eq!(v, Violation::LoopMetaMismatch(1));
    let v = tampered(|k| k.m.loops[0].header = k.at(0));
    assert_eq!(v, Violation::LoopMetaMismatch(0));
}

#[test]
fn a_loop_has_canonical_shape() {
    let h = kernel().at(8);
    let v = tampered(|k| {
        let cont = k.at(12);
        k.block(cont).term = Terminator::Return;
    });
    assert_eq!(v, Violation::LoopShape(h));
    let v = tampered(|k| {
        let (h, n) = (k.at(8), k.i);
        k.block(h).insts.push(InstNode {
            result: None,
            inst: Inst::Unary {
                op: UnOp::CopyObject,
                a: n,
            },
        });
    });
    assert_eq!(v, Violation::LoopShape(h));
}

// ---- Functions, stages, derived facts ----

#[test]
fn the_call_graph_has_no_cycle() {
    let v = tampered(|k| {
        let (e, helper) = (k.helper_entry(), k.helper);
        k.block(e).insts.push(InstNode {
            result: None,
            inst: Inst::Call {
                func: helper,
                args: Box::new([]),
            },
        });
    });
    assert_eq!(v, Violation::Recursion(kernel().helper));
}

#[test]
fn kills_need_the_fragment_model() {
    let v = tampered(|k| {
        let e = k.helper_entry();
        k.block(e).term = Terminator::Kill;
    });
    assert!(matches!(v, Violation::KillForModel(_)), "{v:?}");
}

#[test]
fn the_recorded_interface_and_requirements_are_current() {
    let v = tampered(|k| {
        k.m.interface.pop();
    });
    assert_eq!(v, Violation::InterfaceStale);
    let v = tampered(|k| {
        k.m.requirements.features.insert(VkFeature::ShaderFloat64);
    });
    assert_eq!(v, Violation::RequirementsStale);
}

/// The requirements' plan facts are exactly what the module's own subgroup requirement needs.
/// `finalize` records them with the requirement; they are stale without it, and the requirement is
/// stale without them or with another stage's.
#[test]
fn the_subgroup_facts_stand_only_with_their_requirement() {
    let pinned = SubgroupRequirement {
        size: Some(32),
        full_subgroups: true,
        local_size_multiple: 32,
        varying: false,
    };
    let k = kernel_with(Some(pinned));
    let r = &k.m.requirements;
    assert!(r.features.contains(&VkFeature::SubgroupSizeControl));
    assert!(r
        .properties
        .contains(&VkProperty::RequiredSubgroupSizeStage(Stage::Kernel)));
    assert!(r
        .stage_flags
        .contains(&StageCreateFlag::RequireFullSubgroups));
    structure(&k.m).expect("the pinned kernel verifies");
    // A module whose plan pins nothing records no plan fact.
    let k = kernel();
    assert!(!k
        .m
        .requirements
        .features
        .contains(&VkFeature::SubgroupSizeControl));
    assert!(k.m.requirements.stage_flags.is_empty());
    // The facts without the requirement.
    let v = tampered(|k| pinned.add_device_facts(Stage::Kernel, &mut k.m.requirements));
    assert_eq!(v, Violation::RequirementsStale);
    // Full subgroups' stage flag alone, without the requirement.
    let v = tampered(|k| {
        k.m.requirements
            .stage_flags
            .insert(StageCreateFlag::RequireFullSubgroups);
    });
    assert_eq!(v, Violation::RequirementsStale);
    // The requirement without its facts.
    let v = tampered(|k| k.m.subgroup = pinned);
    assert_eq!(v, Violation::RequirementsStale);
    // Another stage's facts beside the requirement.
    let mut k = kernel_with(Some(pinned));
    let r = &mut k.m.requirements;
    r.properties
        .remove(&VkProperty::RequiredSubgroupSizeStage(Stage::Kernel));
    r.properties
        .insert(VkProperty::RequiredSubgroupSizeStage(Stage::Fragment));
    assert_eq!(violation(&k.m), Violation::RequirementsStale);
}

/// A fragment module: a source switch holding a mapped derivative, a source switch calling a
/// helper that holds an unmapped one, an (empty) switch-at-use, and an if. Returns the module
/// and the headers `[s1, s2, s3, if]`.
fn fragment(modes: Vec<ExecMode>) -> (TypedModule, [BlockId; 4]) {
    let mut b = ModuleBuilder::new();
    let t = tys(&mut b);
    let mut h = b.function(t.void).expect("helper");
    let y = konst(&mut h, t.f32_, 0);
    h.derivative(DerivOp::DPdy, y).expect("derivative");
    h.ret(None).expect("ret");
    let helper = h.finish().expect("helper");

    let mut f = b.function(t.void).expect("main");
    let x = konst(&mut f, t.f32_, 0x3f80_0000);
    let var = f.local_var(t.u32_, None).expect("var");
    let p = f.var_ptr(var).expect("ptr");
    let sel = f.load(p, MemAccess::default()).expect("sel");
    let cvar = f.local_var(t.bool_, None).expect("var");
    let cp = f.var_ptr(cvar).expect("ptr");
    let c = f.load(cp, MemAccess::default()).expect("c");
    let mut heads = Vec::new();
    for kind in 0..3 {
        heads.push(f.current_block().expect("open"));
        let case: ArmFn<'_, ()> = match kind {
            0 => arm(move |fb, _| {
                fb.derivative(DerivOp::DPdx, x)?;
                Ok(ArmEnd::FallThrough(Vec::new()))
            }),
            1 => arm(move |fb, _| {
                fb.call(helper, &[])?;
                Ok(ArmEnd::FallThrough(Vec::new()))
            }),
            _ => empty(),
        };
        let cases = vec![(vec![1], case)];
        if kind == 2 {
            f.switch_at_use(&mut (), sel, cases, empty())
                .expect("switch at use");
        } else {
            f.switch_(&mut (), sel, cases, empty()).expect("switch");
        }
    }
    heads.push(f.current_block().expect("open"));
    f.if_(&mut (), c, empty(), None).expect("if");
    f.ret(None).expect("ret");
    let main = f.finish().expect("main");
    b.entry(ExecModel::Fragment, main, modes).expect("entry");
    let m = b.finalize().expect("finalize");
    (m, [heads[0], heads[1], heads[2], heads[3]])
}

#[test]
fn the_fragment_fixture_passes() {
    let (m, [_, _, s3, _]) = fragment(Vec::new());
    structure(&m).expect("valid");
    assert_eq!(m.switches_at_use(), &[s3]);
}

#[test]
fn nothing_tangled_sits_inside_a_switch_at_use() {
    let (mut m, [s1, s2, _, i]) = fragment(Vec::new());
    m.switches_at_use.push(s1);
    assert!(
        matches!(violation(&m), Violation::TangledInSwitchAtUse { header, .. } if header == s1)
    );
    let (mut m, _) = fragment(Vec::new());
    m.switches_at_use.push(s2);
    assert_eq!(
        violation(&m),
        Violation::TangledCallInSwitchAtUse {
            header: s2,
            callee: FuncId::new(0)
        }
    );
    let (mut m, _) = fragment(Vec::new());
    m.switches_at_use.push(i);
    assert_eq!(violation(&m), Violation::SwitchAtUseHeader(i));
}

#[test]
fn under_interlock_only_the_entry_kills() {
    let (mut m, _) = fragment(vec![ExecMode::PixelInterlockOrdered]);
    structure(&m).expect("valid");
    let e = m.functions[0].blocks[0];
    m.blocks[e.index()].term = Terminator::Kill;
    assert_eq!(violation(&m), Violation::KillOutsideEntry(FuncId::new(0)));
    // Without an interlock mode a helper may kill.
    let (mut m, _) = fragment(Vec::new());
    m.blocks[e.index()].term = Terminator::Kill;
    structure(&m).expect("valid");
}

/// A compute kernel that stores to `out[0]` from a value produced by `body`.
fn compute(body: impl FnOnce(&mut FunctionBuilder<'_>, Tys) -> Value) -> TypedModule {
    let mut b = ModuleBuilder::new();
    let t = tys(&mut b);
    let out = ssbo(&mut b, t.u32_);
    let mut f = b.function(t.void).expect("main");
    let v = body(&mut f, t);
    let zero = konst(&mut f, t.u32_, 0);
    store_at(&mut f, out, t, zero, v);
    f.ret(None).expect("ret");
    let main = f.finish().expect("main");
    b.entry(
        ExecModel::GLCompute,
        main,
        vec![ExecMode::LocalSize { x: 1, y: 1, z: 1 }],
    )
    .expect("entry");
    b.finalize().expect("finalize")
}

#[test]
fn builtins_belong_to_the_entry_model() {
    let m = compute(|f, t| {
        let vec4 = f
            .module()
            .ty(Type::Vector {
                elem: t.f32_,
                len: 4,
            })
            .expect("vec4");
        let g = f
            .module()
            .builtin(crate::spv::BuiltIn::FragCoord, IoDir::Input, vec4)
            .expect("builtin");
        let p = f.module().global_value(g).expect("global");
        let v = f.load(p, MemAccess::default()).expect("load");
        let x = f.composite_extract(v, &[0]).expect("x");
        f.convert(crate::spirv::ir::ConvOp::ConvertFToU, x, t.u32_)
            .expect("convert")
    });
    assert!(matches!(violation(&m), Violation::BuiltInForModel(_)));
}

// ---- Profile rules ----

fn profile(stem: &str) -> DeviceProfile {
    committed_profiles()
        .expect("profiles")
        .into_iter()
        .find(|(s, _)| s == stem)
        .unwrap_or_else(|| panic!("no committed profile {stem}"))
        .1
}

fn workgroup_initialized() -> TypedModule {
    compute(|f, t| {
        let null = f
            .module()
            .constant(Constant::Null { ty: t.u32_ })
            .expect("null");
        let g = f
            .module()
            .plain_global(StorageClass::Workgroup, t.u32_, Some(null))
            .expect("workgroup");
        let p = f.module().global_value(g).expect("global");
        f.load(p, MemAccess::default()).expect("load")
    })
}

#[test]
fn workgroup_initializers_need_zero_initialize_workgroup_memory() {
    let m = workgroup_initialized();
    structure(&m).expect("structurally valid");
    // The runtime's floor without the feature, which the floor requires: a device below it.
    let floor = crate::pipeline::testing::profile_without(
        "runtime-floor",
        &[VkFeature::ShaderZeroInitializeWorkgroupMemory],
        crate::device::DevicePolicy::RESIDENT_TABLES,
    )
    .expect("profile");
    assert!(!floor
        .fields()
        .enabled
        .features
        .contains(&VkFeature::ShaderZeroInitializeWorkgroupMemory));
    assert_eq!(
        check(&m, &floor, &mut meter()),
        Err(VerifyError::Rule(Violation::WorkgroupInitializer(
            GlobalId::new(1)
        )))
    );
    // A profile with the feature accepts it.
    let with: Vec<_> = committed_profiles()
        .expect("profiles")
        .into_iter()
        .filter(|(_, p)| {
            p.fields()
                .enabled
                .features
                .contains(&VkFeature::ShaderZeroInitializeWorkgroupMemory)
        })
        .collect();
    assert!(!with.is_empty());
    for (stem, p) in with {
        assert_eq!(check(&m, &p, &mut meter()).map(|_| ()), Ok(()), "{stem}");
        assert_valid(workgroup_initialized(), &p);
    }
}

/// A `Workgroup` initializer other than `OpConstantNull` breaks a rule on every profile, one with
/// `shaderZeroInitializeWorkgroupMemory` too (`VUID-StandaloneSpirv-OpVariable-04734`).
#[test]
fn a_workgroup_initializer_is_only_a_null() {
    let m = compute(|f, t| {
        let one = f
            .module()
            .constant(Constant::Scalar {
                ty: t.u32_,
                bits: ConstBits(1),
            })
            .expect("one");
        let g = f
            .module()
            .plain_global(StorageClass::Workgroup, t.u32_, Some(one))
            .expect("workgroup");
        let p = f.module().global_value(g).expect("global");
        f.load(p, MemAccess::default()).expect("load")
    });
    structure(&m).expect("structurally valid");
    let with: Vec<_> = committed_profiles()
        .expect("profiles")
        .into_iter()
        .filter(|(_, p)| {
            p.fields()
                .enabled
                .features
                .contains(&VkFeature::ShaderZeroInitializeWorkgroupMemory)
        })
        .collect();
    assert!(!with.is_empty());
    for (stem, p) in with {
        assert_eq!(
            check(&m, &p, &mut meter()),
            Err(VerifyError::Rule(Violation::WorkgroupInitializer(
                GlobalId::new(1)
            ))),
            "{stem}"
        );
    }
}

#[test]
fn requirements_beyond_the_profile_are_an_internal_error() {
    let m = compute(|f, t| {
        let f64_ = f.module().ty(Type::Float { width: 64 }).expect("f64");
        let one = konst(f, f64_, 1.0f64.to_bits());
        let two = f.binary(BinOp::FAdd, one, one).expect("add");
        f.convert(crate::spirv::ir::ConvOp::ConvertFToU, two, t.u32_)
            .expect("convert")
    });
    structure(&m).expect("structurally valid");
    let mvk = profile("moltenvk-m3");
    assert_eq!(
        check(&m, &mvk, &mut meter()),
        Err(VerifyError::ExceedsProfile(Capability::Feature(
            VkFeature::ShaderFloat64
        )))
    );
    match verify(&m, &mvk, &mut meter()) {
        Err(Error::Internal(e)) => assert_eq!(e.kind, InternalKind::RequirementsExceedProfile),
        other => panic!("{other:?}"),
    }
}

#[test]
fn a_broken_rule_is_an_internal_verify_error() {
    let mut k = kernel();
    k.m.interface.clear();
    match verify(&k.m, &profile("moltenvk-m3"), &mut meter()) {
        Err(Error::Internal(e)) => assert_eq!(e.kind, InternalKind::Verify),
        other => panic!("{other:?}"),
    }
}

// ---- Places ----

/// A kernel whose atomic takes the device place of a word, beside a bare converted pointer to the
/// same word that a plain load reads; and that bare pointer.
fn device_atomic() -> (TypedModule, Value) {
    let mut bare = None;
    let m = compute(|f, t| {
        let u64_ = f
            .module()
            .ty(Type::Int {
                width: 64,
                signed: false,
            })
            .expect("u64");
        let addr = konst(f, u64_, 0x1000);
        let place = f.device_place(addr, t.u32_).expect("place");
        let psb = f
            .module()
            .ty(Type::Pointer {
                class: StorageClass::PhysicalStorageBuffer,
                pointee: t.u32_,
            })
            .expect("pointer");
        let p = f.convert(ConvOp::ConvertUToPtr, addr, psb).expect("bare");
        let aligned = MemAccess {
            aligned: Some(4),
            ..MemAccess::default()
        };
        let seen = f.load(p, aligned).expect("load");
        bare = Some(p);
        let one = konst(f, t.u32_, 1);
        let old = f
            .atomic(
                AtomicOp::IAdd,
                place,
                Scope::Device,
                Semantics::RELAXED,
                one,
            )
            .expect("atomic");
        f.binary(BinOp::IAdd, old, seen).expect("add")
    });
    (m, bare.expect("the bare pointer"))
}

#[test]
fn an_atomic_on_a_device_place_passes() {
    let (m, _) = device_atomic();
    structure(&m).expect("a device place is a place");
    assert_valid(m, &profile("moltenvk-m3"));
}

#[test]
fn an_atomic_on_a_converted_address_is_not_a_place() {
    let (mut m, bare) = device_atomic();
    let (block, ptr) = m
        .blocks
        .iter_mut()
        .enumerate()
        .find_map(|(i, b)| {
            b.insts.iter_mut().find_map(|n| match &mut n.inst {
                Inst::Atomic { ptr, .. } => Some((BlockId::new(i as u32), ptr)),
                _ => None,
            })
        })
        .expect("the atomic");
    *ptr = bare;
    assert_eq!(
        violation(&m),
        Violation::AtomicNotPlace { ptr: bare, block }
    );
}
