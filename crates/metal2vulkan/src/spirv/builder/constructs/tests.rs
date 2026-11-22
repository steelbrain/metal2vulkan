//! L0 tests of the construct openers: shapes, phi merging, loop invariants and `LoopMeta`, loop
//! exits, tangled instructions in switches, and the demote/kill rules.

use crate::spirv::builder::constructs::{
    ArmEnd, ArmFn, KillForm, LoopCtx, LoopFn, LoopSpec, Merged,
};
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
use crate::spirv::error::{BuildError, ControlRule, Dangling, EntryRule, OperandRule};
use crate::spirv::ids::{BlockId, TypeId, Value, VarId};
use crate::spirv::ir::{
    BinOp, Block, ConstBits, Constant, DerivOp, ExecMode, ExecModel, LoopMeta, MemAccess, Merge,
    Terminator, Type, TypedModule, ValueOrigin,
};

use crate::spv::Op;

// ---- Fixture ----

#[derive(Clone, Copy)]
struct Tys {
    void: TypeId,
    bool_: TypeId,
    u32_: TypeId,
    u16_: TypeId,
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
        u16_: b
            .ty(Type::Int {
                width: 16,
                signed: false,
            })
            .expect("u16"),
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

fn truth(f: &mut FunctionBuilder<'_>, t: Tys) -> Value {
    let v = f.local_var(t.bool_, None).expect("var");
    let p = f.var_ptr(v).expect("ptr");
    f.load(p, MemAccess::default()).expect("load")
}

fn compute() -> (ExecModel, Vec<ExecMode>) {
    (
        ExecModel::GLCompute,
        vec![ExecMode::LocalSize { x: 1, y: 1, z: 1 }],
    )
}

/// Build a module whose entry function (of `model`) is `body`, and finalize it.
fn build(
    model: ExecModel,
    modes: Vec<ExecMode>,
    body: impl FnOnce(&mut FunctionBuilder<'_>, Tys),
) -> Result<TypedModule, BuildError> {
    let mut b = ModuleBuilder::new();
    let t = tys(&mut b);
    let mut f = b.function(t.void)?;
    body(&mut f, t);
    if f.current_block().is_some() {
        f.ret(None)?;
    }
    let id = f.finish()?;
    b.entry(model, id, modes)?;
    b.finalize()
}

/// Run `body` inside a void function of a fresh module, without finalizing (for constructions
/// that abort the function).
fn run(body: impl FnOnce(&mut FunctionBuilder<'_>, Tys)) {
    let mut b = ModuleBuilder::new();
    let t = tys(&mut b);
    let mut f = b.function(t.void).expect("function");
    body(&mut f, t);
}

fn build_compute(body: impl FnOnce(&mut FunctionBuilder<'_>, Tys)) -> TypedModule {
    let (model, modes) = compute();
    build(model, modes, body).expect("module")
}

fn blk(m: &TypedModule, b: BlockId) -> &Block {
    m.blocks().get(b.index()).expect("block")
}

fn entry_blocks(m: &TypedModule) -> &[BlockId] {
    &m.functions()
        .get(m.entry().func.index())
        .expect("entry")
        .blocks
}

fn arm<'a, C>(
    f: impl FnOnce(&mut FunctionBuilder<'_>, &mut C) -> Result<ArmEnd, BuildError> + 'a,
) -> ArmFn<'a, C> {
    Box::new(f)
}

fn body<'a, C>(
    f: impl FnOnce(&mut FunctionBuilder<'_>, &mut C, &LoopCtx) -> Result<(), BuildError> + 'a,
) -> LoopFn<'a, C> {
    Box::new(f)
}

fn empty() -> ArmFn<'static, ()> {
    arm(|_, _| Ok(ArmEnd::FallThrough(Vec::new())))
}

// ---- if_ ----

#[test]
fn if_merges_fall_through_values_by_phi_in_arm_order() {
    let mut out = None;
    let m = build_compute(|f, t| {
        let c = truth(f, t);
        let header = f.current_block().expect("open");
        let one = konst(f, t.u32_, 1);
        let two = konst(f, t.u32_, 2);
        let Merged(vals) = f
            .if_(
                &mut (),
                c,
                arm(move |_, _| Ok(ArmEnd::FallThrough(vec![one]))),
                Some(arm(move |_, _| Ok(ArmEnd::FallThrough(vec![two])))),
            )
            .expect("if");
        assert_eq!(vals.len(), 1);
        assert_eq!(f.value_ty(vals[0]).expect("ty"), t.u32_);
        out = Some((
            header,
            vals[0],
            one,
            two,
            f.current_block().expect("merge open"),
        ));
    });
    let (header, phi, one, two, merge) = out.expect("ran");
    let h = blk(&m, header);
    assert_eq!(h.merge, Some(Merge::Selection { merge }));
    let Terminator::CondBranch { t: tb, f: fb, .. } = h.term else {
        panic!("header ends in a conditional branch: {:?}", h.term);
    };
    assert!(tb != fb && tb != merge && fb != merge);
    assert_eq!(blk(&m, tb).term, Terminator::Branch(merge));
    assert_eq!(blk(&m, fb).term, Terminator::Branch(merge));
    let mb = blk(&m, merge);
    assert_eq!(mb.phis.len(), 1);
    assert_eq!(mb.phis[0].result, phi);
    assert_eq!(mb.phis[0].incoming, vec![(one, tb), (two, fb)]);
    assert_eq!(m.value(phi).expect("phi").origin, ValueOrigin::Phi(merge));
    // Layout is the construct preorder: header, then arms, then merge.
    let order = entry_blocks(&m);
    let at = |b| order.iter().position(|x| *x == b).expect("laid out");
    assert!(at(header) < at(tb) && at(tb) < at(fb) && at(fb) < at(merge));
}

#[test]
fn arms_that_exit_add_no_phi_incoming() {
    // One arm returns: the merge's phi has only the other arm's incoming edge.
    let mut out = None;
    let m = build_compute(|f, t| {
        let c = truth(f, t);
        let two = konst(f, t.u32_, 2);
        let Merged(vals) = f
            .if_(
                &mut (),
                c,
                arm(|f, _| {
                    f.ret(None)?;
                    Ok(ArmEnd::Exits)
                }),
                Some(arm(move |_, _| Ok(ArmEnd::FallThrough(vec![two])))),
            )
            .expect("if");
        out = Some((vals, f.current_block().expect("merge open")));
    });
    let (vals, merge) = out.expect("ran");
    assert_eq!(vals.len(), 1);
    let phis = &blk(&m, merge).phis;
    assert_eq!(phis.len(), 1);
    assert_eq!(phis[0].incoming.len(), 1);

    // Every arm exits: no phi at all, the merge is unreachable, and no block is open.
    let m = build_compute(|f, t| {
        let c = truth(f, t);
        let Merged(vals) = f
            .if_(
                &mut (),
                c,
                arm(|f, _| {
                    f.ret(None)?;
                    Ok(ArmEnd::Exits)
                }),
                Some(arm(|f, _| {
                    f.unreachable()?;
                    Ok(ArmEnd::Exits)
                })),
            )
            .expect("if");
        assert!(vals.is_empty());
        assert_eq!(f.current_block(), None);
    });
    assert!(m.blocks().iter().all(|b| b.phis.is_empty()));
    let merges: Vec<_> = m
        .blocks()
        .iter()
        .filter_map(|b| match b.merge {
            Some(Merge::Selection { merge }) => Some(merge),
            _ => None,
        })
        .collect();
    assert_eq!(merges.len(), 1);
    assert_eq!(blk(&m, merges[0]).term, Terminator::Unreachable);
}

#[test]
fn if_without_else_branches_false_to_the_merge() {
    let mut out = None;
    let m = build_compute(|f, t| {
        let c = truth(f, t);
        let header = f.current_block().expect("open");
        let Merged(vals) = f.if_(&mut (), c, empty(), None).expect("if");
        assert!(vals.is_empty());
        out = Some((header, f.current_block().expect("merge")));
    });
    let (header, merge) = out.expect("ran");
    let Terminator::CondBranch { t: tb, f: fb, .. } = blk(&m, header).term else {
        panic!("conditional branch");
    };
    assert_eq!(fb, merge);
    assert_ne!(tb, merge);
    assert!(blk(&m, merge).phis.is_empty());

    // The header's false edge carries no value, so the true arm may not fall through with one.
    run(|f, t| {
        let c = truth(f, t);
        let one = konst(f, t.u32_, 1);
        assert_eq!(
            f.if_(
                &mut (),
                c,
                arm(move |_, _| Ok(ArmEnd::FallThrough(vec![one]))),
                None
            ),
            Err(BuildError::Control(ControlRule::ArmValues))
        );
    });
}

#[test]
fn arm_end_must_match_the_open_block() {
    let (model, modes) = compute();
    let r = build(model, modes.clone(), |f, t| {
        let c = truth(f, t);
        let e = f.if_(
            &mut (),
            c,
            arm(|f, _| {
                f.ret(None)?;
                Ok(ArmEnd::FallThrough(Vec::new()))
            }),
            None,
        );
        assert_eq!(e, Err(BuildError::Control(ControlRule::ArmEnd)));
    });
    assert!(r.is_err(), "an aborted construct cannot be finalized");
    build(model, modes, |f, t| {
        let c = truth(f, t);
        let e = f.if_(&mut (), c, arm(|_, _| Ok(ArmEnd::Exits)), None);
        assert_eq!(e, Err(BuildError::Control(ControlRule::ArmEnd)));
    })
    .ok();
}

#[test]
fn fall_through_values_must_agree_and_be_phi_types() {
    run(|f, t| {
        let c = truth(f, t);
        let one = konst(f, t.u32_, 1);
        let x = konst(f, t.f32_, 0);
        assert_eq!(
            f.if_(
                &mut (),
                c,
                arm(move |_, _| Ok(ArmEnd::FallThrough(vec![one]))),
                Some(arm(move |_, _| Ok(ArmEnd::FallThrough(vec![x])))),
            ),
            Err(BuildError::Control(ControlRule::ArmValues))
        );
    });
    run(|f, t| {
        let c = truth(f, t);
        let one = konst(f, t.u32_, 1);
        assert_eq!(
            f.if_(
                &mut (),
                c,
                arm(move |_, _| Ok(ArmEnd::FallThrough(vec![one, one]))),
                Some(arm(move |_, _| Ok(ArmEnd::FallThrough(vec![one])))),
            ),
            Err(BuildError::Control(ControlRule::ArmValues))
        );
    });
    // A logical pointer cannot be merged by `OpPhi` (no variable pointers).
    run(|f, t| {
        let c = truth(f, t);
        let v = f.local_var(t.u32_, None).expect("var");
        let p = f.var_ptr(v).expect("ptr");
        assert_eq!(
            f.if_(
                &mut (),
                c,
                arm(move |_, _| Ok(ArmEnd::FallThrough(vec![p]))),
                Some(arm(move |_, _| Ok(ArmEnd::FallThrough(vec![p])))),
            ),
            Err(BuildError::Control(ControlRule::PhiType))
        );
    });
}

#[test]
fn a_rejected_opener_leaves_the_function_unchanged() {
    let m = build_compute(|f, t| {
        let x = konst(f, t.u32_, 1);
        let open = f.current_block();
        assert_eq!(
            f.if_(&mut (), x, empty(), None),
            Err(BuildError::Operand {
                op: Op::BranchConditional,
                rule: OperandRule::Category(0)
            })
        );
        let sel = konst(f, t.u16_, 1);
        assert_eq!(
            f.switch_(&mut (), sel, Vec::new(), empty()),
            Err(BuildError::Operand {
                op: Op::Switch,
                rule: OperandRule::Width(0)
            })
        );
        assert_eq!(
            f.switch_(
                &mut (),
                x,
                vec![(vec![1, 2], empty()), (vec![2], empty())],
                empty()
            ),
            Err(BuildError::Control(ControlRule::CaseLiterals))
        );
        assert_eq!(
            f.switch_(&mut (), x, vec![(Vec::new(), empty())], empty()),
            Err(BuildError::Control(ControlRule::CaseLiterals))
        );
        assert_eq!(f.current_block(), open);
    });
    // Nothing but the entry block was created.
    assert_eq!(m.blocks().len(), 1);
}

// ---- switch_ ----

#[test]
fn switch_targets_one_fresh_block_per_case_and_merges_by_phi() {
    let mut out = None;
    let m = build_compute(|f, t| {
        let v = f.local_var(t.u32_, None).expect("var");
        let p = f.var_ptr(v).expect("ptr");
        let sel = f.load(p, MemAccess::default()).expect("load");
        let header = f.current_block().expect("open");
        let a = konst(f, t.u32_, 10);
        let b = konst(f, t.u32_, 20);
        let d = konst(f, t.u32_, 30);
        let Merged(vals) = f
            .switch_(
                &mut (),
                sel,
                vec![
                    (
                        vec![1, 3],
                        arm(move |_, _| Ok(ArmEnd::FallThrough(vec![a]))),
                    ),
                    (vec![7], arm(move |_, _| Ok(ArmEnd::FallThrough(vec![b])))),
                    (
                        vec![9],
                        arm(|f, _| {
                            f.ret(None)?;
                            Ok(ArmEnd::Exits)
                        }),
                    ),
                ],
                arm(move |_, _| Ok(ArmEnd::FallThrough(vec![d]))),
            )
            .expect("switch");
        out = Some((header, vals, f.current_block().expect("merge")));
    });
    let (header, vals, merge) = out.expect("ran");
    let h = blk(&m, header);
    assert_eq!(h.merge, Some(Merge::Selection { merge }));
    let Terminator::Switch { default, cases, .. } = &h.term else {
        panic!("switch");
    };
    let lits: Vec<u64> = cases.iter().map(|(l, _)| *l).collect();
    assert_eq!(lits, vec![1, 3, 7, 9]);
    assert_eq!(
        cases[0].1, cases[1].1,
        "literals of one case share its block"
    );
    let mut targets: Vec<BlockId> = cases.iter().map(|(_, b)| *b).collect();
    targets.push(*default);
    targets.dedup();
    assert_eq!(targets.len(), 4);
    assert!(!targets.contains(&merge));
    let phis = &blk(&m, merge).phis;
    assert_eq!(phis.len(), 1);
    assert_eq!(phis[0].result, vals[0]);
    assert_eq!(
        phis[0].incoming.len(),
        3,
        "the returning case adds no incoming edge"
    );
}

// ---- loop_ ----

fn loop_of(m: &TypedModule, i: usize) -> LoopMeta {
    *m.loops().get(i).expect("loop")
}

#[test]
fn loop_has_the_canonical_shape_and_records_loop_meta() {
    let mut pre = None;
    let m = build_compute(|f, t| {
        let c = truth(f, t);
        pre = f.current_block();
        f.loop_(
            &mut (),
            LoopSpec::default(),
            body(move |f, _, lc| {
                f.if_(
                    &mut (),
                    c,
                    arm(|f, _| {
                        lc.break_(f)?;
                        Ok(ArmEnd::Exits)
                    }),
                    None,
                )?;
                Ok(())
            }),
        )
        .expect("loop");
        assert!(
            f.current_block().is_some(),
            "a loop with a break leaves its merge open"
        );
    });
    assert_eq!(m.loops().len(), 1);
    let l = loop_of(&m, 0);
    assert_eq!(Some(l.preheader), pre);
    assert_eq!(l.exit_sel, None);
    // One entry edge from the preheader.
    assert_eq!(blk(&m, l.preheader).term, Terminator::Branch(l.header));
    // Empty header: no phi, no instruction, the loop merge, and a branch into the body.
    let h = blk(&m, l.header);
    assert!(h.phis.is_empty() && h.insts.is_empty());
    assert_eq!(
        h.merge,
        Some(Merge::Loop {
            merge: l.merge,
            cont: l.cont
        })
    );
    let Terminator::Branch(first) = h.term else {
        panic!("header branches into the body");
    };
    // The single continue block branches back to the header and is the only back edge.
    let c = blk(&m, l.cont);
    assert!(c.insts.is_empty() && c.phis.is_empty() && c.merge.is_none());
    assert_eq!(c.term, Terminator::Branch(l.header));
    let into_header: Vec<BlockId> = entry_blocks(&m)
        .iter()
        .copied()
        .filter(|b| blk(&m, *b).term == Terminator::Branch(l.header))
        .collect();
    assert_eq!(into_header, vec![l.preheader, l.cont]);
    // Loops yield no SSA values: no phi anywhere.
    assert!(m.blocks().iter().all(|b| b.phis.is_empty()));
    // Layout: preheader, header, body, continue, merge.
    let order = entry_blocks(&m);
    let at = |b| order.iter().position(|x| *x == b).expect("laid out");
    assert!(at(l.preheader) < at(l.header));
    assert_eq!(at(first), at(l.header) + 1);
    assert!(order[at(first)..at(l.cont)]
        .iter()
        .all(|b| *b != l.merge && *b != l.preheader));
    assert_eq!(at(l.merge), at(l.cont) + 1);
    // Four distinct blocks.
    let mut ids = vec![l.preheader, l.header, first, l.cont, l.merge];
    ids.sort();
    ids.dedup();
    assert_eq!(ids.len(), 5);
}

#[test]
fn loop_without_break_has_an_unreachable_merge_and_leaves_no_block_open() {
    let m = build_compute(|f, _| {
        f.loop_(&mut (), LoopSpec::default(), body(|_, _, _| Ok(())))
            .expect("loop");
        assert_eq!(f.current_block(), None);
    });
    let l = loop_of(&m, 0);
    assert_eq!(blk(&m, l.merge).term, Terminator::Unreachable);
    // The body fell off its end, which continues.
    let Terminator::Branch(first) = blk(&m, l.header).term else {
        panic!("branch");
    };
    assert_eq!(blk(&m, first).term, Terminator::Branch(l.cont));
}

#[test]
fn break_and_continue_reach_only_the_innermost_loop() {
    build_compute(|f, t| {
        let c = truth(f, t);
        f.loop_(
            &mut (),
            LoopSpec::default(),
            body(move |f, _, outer| {
                f.loop_(
                    &mut (),
                    LoopSpec::default(),
                    body(|f, _, inner| {
                        assert_eq!(
                            outer.break_(f),
                            Err(BuildError::Control(ControlRule::NotInnermostLoop))
                        );
                        assert_eq!(
                            outer.continue_(f),
                            Err(BuildError::Control(ControlRule::NotInnermostLoop))
                        );
                        inner.break_(f)
                    }),
                )?;
                // Inside a selection nested in the loop, both exits are legal.
                f.if_(
                    &mut (),
                    c,
                    arm(|f, _| {
                        outer.continue_(f)?;
                        Ok(ArmEnd::Exits)
                    }),
                    Some(arm(|f, _| {
                        outer.break_(f)?;
                        Ok(ArmEnd::Exits)
                    })),
                )?;
                Ok(())
            }),
        )
        .expect("loops");
    });
}

#[test]
fn exit_selector_is_a_32_bit_function_variable_of_this_function() {
    let mut sel = None;
    let m = build_compute(|f, t| {
        let v = f.local_var(t.u32_, None).expect("var");
        sel = Some(v);
        f.loop_(
            &mut (),
            LoopSpec {
                exit_sel: Some(v),
                ..LoopSpec::default()
            },
            body(|f, _, lc| lc.break_(f)),
        )
        .expect("loop");
        let bad = f.local_var(t.f32_, None).expect("var");
        assert_eq!(
            f.loop_(
                &mut (),
                LoopSpec {
                    exit_sel: Some(bad),
                    ..LoopSpec::default()
                },
                body(|_, _, _| Ok(()))
            ),
            Err(BuildError::Control(ControlRule::ExitSelector))
        );
    });
    assert_eq!(loop_of(&m, 0).exit_sel, sel);

    // A 32-bit variable of another function is refused as `ExitSelector`; an id naming no
    // variable is dangling.
    let mut b = ModuleBuilder::new();
    let t = tys(&mut b);
    let mut g = b.function(t.void).expect("other");
    let foreign = g.local_var(t.u32_, None).expect("var");
    g.ret(None).expect("ret");
    g.finish().expect("finish");
    let mut f = b.function(t.void).expect("function");
    let spin = |f: &mut FunctionBuilder<'_>, v| {
        f.loop_(
            &mut (),
            LoopSpec {
                exit_sel: Some(v),
                ..LoopSpec::default()
            },
            body(|f, _, lc| lc.break_(f)),
        )
    };
    assert_eq!(
        spin(&mut f, foreign),
        Err(BuildError::Control(ControlRule::ExitSelector))
    );
    let missing = VarId::new(99);
    assert_eq!(
        spin(&mut f, missing),
        Err(BuildError::Dangling(Dangling::Var(missing)))
    );
}

#[test]
fn context_is_threaded_through_every_arm_and_body() {
    let mut count = 0u32;
    build_compute(|f, t| {
        let c = truth(f, t);
        f.if_(
            &mut count,
            c,
            arm(|_, n: &mut u32| {
                *n += 1;
                Ok(ArmEnd::FallThrough(Vec::new()))
            }),
            Some(arm(|f, n: &mut u32| {
                *n += 10;
                f.loop_(
                    n,
                    LoopSpec::default(),
                    body(|f, n: &mut u32, lc| {
                        *n += 100;
                        lc.break_(f)
                    }),
                )?;
                Ok(ArmEnd::FallThrough(Vec::new()))
            })),
        )
        .expect("if");
    });
    assert_eq!(count, 111);
}

// ---- Tangled instructions and switches ----

fn selector(f: &mut FunctionBuilder<'_>, t: Tys) -> Value {
    let v = f.local_var(t.u32_, None).expect("var");
    let p = f.var_ptr(v).expect("ptr");
    f.load(p, MemAccess::default()).expect("load")
}

#[test]
fn source_switches_may_hold_tangled_instructions() {
    // A source switch keeps its tangled instructions in their source position, nested or not.
    let m = build(ExecModel::Fragment, Vec::new(), |f, t| {
        let x = konst(f, t.f32_, 0);
        let sel = selector(f, t);
        f.switch_(
            &mut (),
            sel,
            vec![(
                vec![1],
                arm(|f, _| {
                    f.switch_(
                        &mut (),
                        sel,
                        Vec::new(),
                        arm(|f, _| {
                            f.derivative(DerivOp::DPdx, x)?;
                            Ok(ArmEnd::FallThrough(Vec::new()))
                        }),
                    )?;
                    Ok(ArmEnd::FallThrough(Vec::new()))
                }),
            )],
            arm(|f, _| {
                f.derivative(DerivOp::DPdy, x)?;
                Ok(ArmEnd::FallThrough(Vec::new()))
            }),
        )
        .expect("switch");
    })
    .expect("module");
    assert!(m.switches_at_use().is_empty());
}

#[test]
fn a_switch_at_use_refuses_tangled_instructions() {
    type Inner = Box<dyn FnOnce(&mut FunctionBuilder<'_>, Tys, Value) -> Result<(), BuildError>>;
    let cases: Vec<(&str, Inner)> = vec![
        (
            "directly",
            Box::new(|f, _, x| {
                f.derivative(DerivOp::DPdx, x)?;
                Ok(())
            }),
        ),
        (
            "nested in a construct",
            Box::new(|f, t, x| {
                let c = truth(f, t);
                f.if_(
                    &mut (),
                    c,
                    arm(move |f, _| {
                        f.derivative(DerivOp::DPdx, x)?;
                        Ok(ArmEnd::FallThrough(Vec::new()))
                    }),
                    None,
                )?;
                Ok(())
            }),
        ),
    ];
    for (what, inner) in cases {
        run(|f, t| {
            let x = konst(f, t.f32_, 0);
            let sel = selector(f, t);
            let r = f.switch_at_use(
                &mut (),
                sel,
                vec![(vec![0], empty())],
                arm(move |f, _| {
                    inner(f, t, x)?;
                    Ok(ArmEnd::FallThrough(Vec::new()))
                }),
            );
            assert_eq!(
                r,
                Err(BuildError::Control(ControlRule::TangledInSwitchAtUse)),
                "{what}"
            );
        });
    }

    // Non-tangled work inside the arms and a tangled instruction hoisted before the switch are
    // fine.
    let mut out = None;
    let m = build(ExecModel::Fragment, Vec::new(), |f, t| {
        let x = konst(f, t.f32_, 0);
        let d = f.derivative(DerivOp::DPdx, x).expect("hoisted");
        let sel = selector(f, t);
        let header = f.current_block().expect("open");
        f.switch_at_use(
            &mut (),
            sel,
            vec![(
                vec![0],
                arm(move |f, _| {
                    f.binary(BinOp::FAdd, d, x)?;
                    Ok(ArmEnd::FallThrough(Vec::new()))
                }),
            )],
            empty(),
        )
        .expect("switch-at-use");
        // Past the merge, tangled instructions are allowed again.
        f.derivative(DerivOp::DPdy, x).expect("after the switch");
        out = Some(header);
    })
    .expect("module");
    assert_eq!(m.switches_at_use(), &[out.expect("ran")]);
}

#[test]
fn finalize_refuses_a_switch_at_use_that_calls_a_tangled_function() {
    for at_use in [false, true] {
        let mut b = ModuleBuilder::new();
        let t = tys(&mut b);
        let mut g = b.function(t.void).expect("callee");
        let x = konst(&mut g, t.f32_, 0);
        g.derivative(DerivOp::DPdx, x).expect("derivative");
        g.ret(None).expect("ret");
        let callee = g.finish().expect("finish");
        let mut f = b.function(t.void).expect("entry");
        let sel = selector(&mut f, t);
        let call = arm(move |f: &mut FunctionBuilder<'_>, _: &mut ()| {
            f.call(callee, &[])?;
            Ok(ArmEnd::FallThrough(Vec::new()))
        });
        if at_use {
            f.switch_at_use(&mut (), sel, Vec::new(), call)
        } else {
            f.switch_(&mut (), sel, Vec::new(), call)
        }
        .expect("switch");
        f.ret(None).expect("ret");
        let entry = f.finish().expect("finish");
        b.entry(ExecModel::Fragment, entry, Vec::new())
            .expect("entry");
        let r = b.finalize().map(|_| ());
        if at_use {
            assert_eq!(
                r,
                Err(BuildError::Control(ControlRule::TangledCallInSwitchAtUse(
                    callee
                )))
            );
        } else {
            assert_eq!(r, Ok(()), "a source switch may call it");
        }
    }
}

// ---- demote / kill ----

#[test]
fn demote_is_fragment_only() {
    build(ExecModel::Fragment, Vec::new(), |f, _| {
        f.demote().expect("demote")
    })
    .expect("module");
    let (model, modes) = compute();
    assert_eq!(
        build(model, modes, |f, _| f.demote().expect("demote")).err(),
        Some(BuildError::Entry(EntryRule::InstructionForModel(
            Op::DemoteToHelperInvocation
        )))
    );
}

#[test]
fn kill_is_fragment_only_and_entry_only_under_interlock() {
    let m = build(ExecModel::Fragment, Vec::new(), |f, _| {
        f.kill(KillForm::TerminateInvocation).expect("kill");
    })
    .expect("module");
    assert!(m
        .blocks()
        .iter()
        .any(|b| b.term == Terminator::TerminateInvocation));
    let (model, modes) = compute();
    assert_eq!(
        build(model, modes, |f, _| f.kill(KillForm::Kill).expect("kill")).err(),
        Some(BuildError::Entry(EntryRule::InstructionForModel(Op::Kill)))
    );

    // A helper function that kills, called by a fragment entry.
    let helper_module = |modes: Vec<ExecMode>| {
        let mut b = ModuleBuilder::new();
        let t = tys(&mut b);
        let mut g = b.function(t.void).expect("helper");
        g.kill(KillForm::Kill).expect("kill");
        let helper = g.finish().expect("finish");
        let mut f = b.function(t.void).expect("entry");
        f.call(helper, &[]).expect("call");
        f.ret(None).expect("ret");
        let entry = f.finish().expect("finish");
        b.entry(ExecModel::Fragment, entry, modes).expect("entry");
        (b.finalize(), helper)
    };
    assert!(helper_module(Vec::new()).0.is_ok());
    let (r, helper) = helper_module(vec![ExecMode::PixelInterlockOrdered]);
    assert_eq!(
        r.err(),
        Some(BuildError::Control(ControlRule::KillOutsideEntry(helper)))
    );
    // The entry itself may kill under interlock.
    build(
        ExecModel::Fragment,
        vec![ExecMode::PixelInterlockOrdered],
        |f, _| f.kill(KillForm::Kill).expect("kill"),
    )
    .expect("module");
}

#[test]
fn constructs_need_an_open_block() {
    build_compute(|f, t| {
        let c = truth(f, t);
        f.ret(None).expect("ret");
        assert_eq!(
            f.if_(&mut (), c, empty(), None),
            Err(BuildError::Control(ControlRule::NoCurrentBlock))
        );
        assert_eq!(
            f.loop_(&mut (), LoopSpec::default(), body(|_, _, _| Ok(()))),
            Err(BuildError::Control(ControlRule::NoCurrentBlock))
        );
        assert_eq!(
            f.kill(KillForm::Kill),
            Err(BuildError::Control(ControlRule::NoCurrentBlock))
        );
    });
}

// ---- split switches ----

/// The bytes of the working tables `ModuleBuilder::footprint` keeps after building a switch on a
/// loaded selector over `n` one-literal arms falling through with their literal, with `limit`
/// pairs to an `OpSwitch`.
fn split_table_bytes(n: u32, limit: usize) -> (usize, usize) {
    use crate::spirv::builder::constructs::SwitchKind;
    let mut b = ModuleBuilder::new();
    let t = tys(&mut b);
    let mut f = b.function(t.void).expect("function");
    let v = f.local_var(t.u32_, None).expect("var");
    let p = f.var_ptr(v).expect("ptr");
    let sel = f.load(p, MemAccess::default()).expect("load");
    let u32_ = t.u32_;
    let cases: Vec<(Vec<u32>, ArmFn<'_, ()>)> = (0..n)
        .map(|l| {
            let a: ArmFn<'_, ()> =
                arm(move |fb, _| Ok(ArmEnd::FallThrough(vec![konst(fb, u32_, l.into())])));
            (vec![l], a)
        })
        .collect();
    let default = arm(move |fb, _| Ok(ArmEnd::FallThrough(vec![konst(fb, u32_, 0)])));
    let kind = SwitchKind {
        at_use: false,
        limit,
    };
    f.switch_within(&mut (), sel, cases, default, kind)
        .expect("switch");
    f.ret(None).expect("ret");
    f.finish().expect("finish");
    (b.split_bytes, b.footprint())
}

/// A switch split's working tables are freed when it returns, so the builder keeps their bytes in
/// its footprint, which the emitters' ledger polls charge: none for a switch within the limit; for
/// 4,096 one-literal arms at 16 pairs to an `OpSwitch`, at least the table of each literal with its
/// chunk and each arm's chunk, and every group's literals by chunk, and growing with the literals.
/// The footprint holds them.
#[test]
fn a_split_switch_keeps_its_working_tables_in_the_footprint() {
    let (within, _) = split_table_bytes(4_096, 4_096);
    assert_eq!(within, 0);
    let pair = std::mem::size_of::<(u32, u32)>();
    let (small, small_footprint) = split_table_bytes(4_096, 16);
    // The step-1 table (a pair and a chunk per literal), each group's literals by chunk (a pair
    // and a literal per literal), and nested splits of the chunk switch.
    let floor = 4_096 * (pair + std::mem::size_of::<usize>() + pair + 4);
    assert!(small >= floor, "{small} < {floor}");
    assert!(small_footprint > small);
    let (large, _) = split_table_bytes(8_192, 16);
    assert!(large >= 2 * floor, "{large} < {}", 2 * floor);
}
