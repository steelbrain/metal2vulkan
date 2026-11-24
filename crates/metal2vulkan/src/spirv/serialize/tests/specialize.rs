//! `specialize`: the words a driver is handed hold no specialization constant, and are valid.

use super::{compute, cval, elem_ptr, finish_compute, gid_x, int, link_all, scalar, ssbo};
use crate::pipeline::sym::{Axis, SpecKey, Symbol};
use crate::spirv::builder::ModuleBuilder;
use crate::spirv::ir::{ConstBits, ExecModel, MemAccess, SpecOp, StorageClass, Type, TypedModule};
use crate::spirv::serialize::{relocate, serialize};
use crate::spirv::specialize::specialize;
use crate::spirv::testing::{spirv_dis, spirv_val_with, TargetEnv, ValOptions};
use crate::spv::{BuiltIn, Decoration, ExecutionMode, Op};

/// Each instruction of `words` past the header: its opcode and operands.
fn insts(words: &[u32]) -> Vec<(u32, Vec<u32>)> {
    let mut out = Vec::new();
    let mut at = 5;
    while at < words.len() {
        let count = (words[at] >> 16) as usize;
        out.push((words[at] & 0xFFFF, words[at + 1..at + count].to_vec()));
        at += count;
    }
    out
}

/// `words` linked (every symbol numbered by [`link_all`]), with the `SpecId` link gave each key.
fn linked(m: TypedModule) -> (Vec<u32>, Vec<(SpecKey, u32)>, ValOptions) {
    let r = serialize(m).expect("serialize");
    let table = link_all(&r);
    let words = relocate(&r, &table).expect("relocate");
    let ids = (table.iter())
        .filter_map(|(sym, &id)| match sym {
            Symbol::SpecId(key) => Some((*key, id)),
            _ => None,
        })
        .collect();
    (words, ids, ValOptions::for_requirements(&r.requirements))
}

fn assert_valid(words: &[u32], opts: ValOptions) {
    for env in [TargetEnv::Vulkan1_2, TargetEnv::Vulkan1_3] {
        if let Err(e) = spirv_val_with(words, env, opts) {
            panic!(
                "spirv-val {env:?} rejected the module: {e}\n{}",
                spirv_dis(words)
            );
        }
    }
}

/// A kernel shaped as the translator's deep-threadgroup kernels are: `LocalSize` constants the
/// body reads, the `WorkgroupSize` composite of its own constants, a `bool`, and two `Workgroup`
/// arrays whose lengths are `OpSpecConstantOp`s of them: the simdgroup exchange's
/// `(sx·sy·sz + 31) & ~31` and a threadgroup argument's `TgLength + 16`.
fn deep_kernel() -> TypedModule {
    let mut b = ModuleBuilder::new();
    let u32_ = int(&mut b, 32, false);
    let bool_ = b.ty(Type::Bool).expect("bool");
    let [sx, sy, sz] = [Axis::X, Axis::Y, Axis::Z].map(|a| {
        b.spec(SpecKey::LocalSize(a), u32_, ConstBits(1))
            .expect("spec")
    });
    b.spec(SpecKey::DeepThreadgroup, bool_, ConstBits(0))
        .expect("spec");
    let [wx, wy, wz] = [Axis::X, Axis::Y, Axis::Z].map(|a| {
        b.spec(SpecKey::WorkgroupSize(a), u32_, ConstBits(1))
            .expect("spec")
    });
    b.workgroup_size(wx, wy, wz).expect("workgroup");
    let xy = b.spec_op(SpecOp::IMul, u32_, &[sx, sy]).expect("xy");
    let n = b.spec_op(SpecOp::IMul, u32_, &[xy, sz]).expect("n");
    let up = scalar(&mut b, u32_, 31);
    let mask = scalar(&mut b, u32_, !31 & 0xFFFF_FFFF);
    let padded = b.spec_op(SpecOp::IAdd, u32_, &[n, up]).expect("padded");
    let padded = b
        .spec_op(SpecOp::BitwiseAnd, u32_, &[padded, mask])
        .expect("padded");
    let length = b
        .spec(SpecKey::TgLength(0), u32_, ConstBits(0))
        .expect("spec");
    let sixteen = scalar(&mut b, u32_, 16);
    let words = b
        .spec_op(SpecOp::IAdd, u32_, &[length, sixteen])
        .expect("words");
    for len in [padded, words] {
        let ty = b
            .ty(Type::Array {
                elem: u32_,
                len,
                stride: None,
            })
            .expect("array");
        b.plain_global(StorageClass::Workgroup, ty, None)
            .expect("global");
    }
    let out = ssbo(&mut b, 0, u32_, 4);
    let main = compute(&mut b, |f| {
        let i = gid_x(f);
        let dst = elem_ptr(f, out, i);
        let v = f.module().const_value(sx).expect("sx");
        f.store(dst, v, MemAccess::default()).expect("store");
    });
    b.entry(ExecModel::GLCompute, main, Vec::new())
        .expect("entry");
    b.finalize().expect("finalize")
}

/// The words handed a driver for a threadgroup of 8 x 4 x 2 and a 100-byte threadgroup argument
/// hold no specialization constant and no `SpecId`, the workgroup's size is the entry's `LocalSize`
/// mode of those literals with no `BuiltIn WorkgroupSize` beside it, every length is the constant
/// its operation computes, and `spirv-val` accepts them under Vulkan 1.2 and 1.3. With no values
/// every constant keeps its default.
#[test]
fn a_specialized_module_holds_its_values_as_literals() {
    let (words, ids, opts) = linked(deep_kernel());
    assert_valid(&words, opts);
    let value = |key: SpecKey| match key {
        SpecKey::LocalSize(Axis::X) | SpecKey::WorkgroupSize(Axis::X) => 8,
        SpecKey::LocalSize(Axis::Y) | SpecKey::WorkgroupSize(Axis::Y) => 4,
        SpecKey::LocalSize(Axis::Z) | SpecKey::WorkgroupSize(Axis::Z) => 2,
        SpecKey::TgLength(0) => 100,
        _ => 0,
    };
    let values: Vec<(u32, u64)> = ids.iter().map(|&(key, id)| (id, value(key))).collect();
    for (values, size, lengths) in [
        (values, [8, 4, 2], [64, 116]),
        (Vec::new(), [1, 1, 1], [32, 16]),
    ] {
        let frozen = specialize(&words, &values).expect("specialize");
        assert_valid(&frozen, opts);
        let insts = insts(&frozen);
        let spec = [
            Op::SpecConstantTrue,
            Op::SpecConstantFalse,
            Op::SpecConstant,
            Op::SpecConstantComposite,
            Op::SpecConstantOperation,
        ];
        for (op, operands) in &insts {
            assert!(!spec.iter().any(|s| *s as u32 == *op), "{op}: {operands:?}");
            if *op == Op::Decorate as u32 {
                assert_ne!(operands[1], Decoration::SpecId as u32);
                assert_ne!(
                    operands[1..],
                    [Decoration::BuiltIn as u32, BuiltIn::WorkgroupSize as u32]
                );
            }
        }
        let modes: Vec<&Vec<u32>> = (insts.iter())
            .filter(|(op, _)| *op == Op::ExecutionMode as u32)
            .map(|(_, operands)| operands)
            .collect();
        let entry = (insts.iter())
            .find(|(op, _)| *op == Op::EntryPoint as u32)
            .map(|(_, operands)| operands[1])
            .expect("entry point");
        let local = [
            entry,
            ExecutionMode::LocalSize as u32,
            size[0],
            size[1],
            size[2],
        ];
        assert_eq!(modes, [&local.to_vec()]);
        // Each Workgroup array's length is the constant of the length it computes.
        let constant = |id: u32| {
            (insts.iter())
                .find(|(op, operands)| *op == Op::Constant as u32 && operands[1] == id)
                .map(|(_, operands)| operands[2])
        };
        let found: Vec<u32> = (insts.iter())
            .filter(|(op, _)| *op == Op::TypeArray as u32)
            .filter_map(|(_, operands)| constant(operands[2]))
            .collect();
        assert_eq!(found, lengths);
    }
}

/// A module without specialization constants is unchanged; one with a `LocalSize` mode keeps it,
/// its constants fixed at their values.
#[test]
fn a_module_without_a_workgroup_composite_keeps_its_mode() {
    let mut b = ModuleBuilder::new();
    let u32_ = int(&mut b, 32, false);
    let out = ssbo(&mut b, 0, u32_, 4);
    let main = compute(&mut b, |f| {
        let i = gid_x(f);
        let dst = elem_ptr(f, out, i);
        let v = cval(f, u32_, 7);
        f.store(dst, v, MemAccess::default()).expect("store");
    });
    let (plain, _, _) = linked(finish_compute(b, main));
    assert_eq!(specialize(&plain, &[(0, 5)]).expect("specialize"), plain);

    let mut b = ModuleBuilder::new();
    let u32_ = int(&mut b, 32, false);
    let c = b
        .spec(SpecKey::TgLength(0), u32_, ConstBits(3))
        .expect("spec");
    let out = ssbo(&mut b, 0, u32_, 4);
    let main = compute(&mut b, |f| {
        let i = gid_x(f);
        let dst = elem_ptr(f, out, i);
        let v = f.module().const_value(c).expect("c");
        f.store(dst, v, MemAccess::default()).expect("store");
    });
    let (words, ids, opts) = linked(finish_compute(b, main));
    let values: Vec<(u32, u64)> = ids.iter().map(|&(_, id)| (id, 9)).collect();
    let frozen = specialize(&words, &values).expect("specialize");
    assert_valid(&frozen, opts);
    let insts = insts(&frozen);
    assert!(insts
        .iter()
        .any(|(op, operands)| *op == Op::Constant as u32 && operands[2] == 9));
    let modes = (insts.iter())
        .filter(|(op, operands)| {
            *op == Op::ExecutionMode as u32 && operands[1] == ExecutionMode::LocalSize as u32
        })
        .count();
    assert_eq!(modes, 1);
}

/// `words` with `inst` after the first instruction of `after`, its id bound grown by `ids`.
fn inserted(words: &[u32], after: Op, inst: &[u32], ids: u32) -> Vec<u32> {
    let mut at = 5;
    while words[at] & 0xFFFF != after as u32 {
        at += (words[at] >> 16) as usize;
    }
    at += (words[at] >> 16) as usize;
    let mut out = words.to_vec();
    out.splice(at..at, inst.iter().copied());
    out[3] += ids;
    out
}

/// A module holding an instruction the translator does not write, whose constants the pass would
/// leave half specialized, is refused, the refusal naming it: an `OpExecutionModeId` of
/// `LocalSizeId`, an `OpDecorationGroup`, an `OpSpecConstantOp` of `ISub`.
#[test]
fn a_module_of_an_instruction_the_pass_does_not_specialize_is_refused() {
    let (words, ids, _) = linked(deep_kernel());
    let entry = (insts(&words).into_iter())
        .find(|(op, _)| *op == Op::EntryPoint as u32)
        .map(|(_, operands)| operands[1])
        .expect("entry point");
    let local = |axis| {
        (ids.iter())
            .find(|&&(key, _)| key == SpecKey::LocalSize(axis))
            .map(|&(_, id)| id)
            .expect("a LocalSize constant")
    };
    // The LocalSize constants' result ids, by the SpecId decorations that name them.
    let decorated = |spec: u32| {
        (insts(&words).into_iter())
            .find(|(op, operands)| {
                *op == Op::Decorate as u32 && operands[1..] == [Decoration::SpecId as u32, spec]
            })
            .map(|(_, operands)| operands[0])
            .expect("a SpecId decoration")
    };
    let [x, y, z] = [Axis::X, Axis::Y, Axis::Z].map(|axis| decorated(local(axis)));
    let mode_id = [
        (6 << 16) | Op::ExecutionModeId as u32,
        entry,
        ExecutionMode::LocalSizeId as u32,
        x,
        y,
        z,
    ];
    let bound = words[3];
    let group = [(2 << 16) | Op::DecorationGroup as u32, bound];
    let mut subtract = words.clone();
    let mut at = 5;
    while subtract[at] & 0xFFFF != Op::SpecConstantOperation as u32 {
        at += (subtract[at] >> 16) as usize;
    }
    subtract[at + 3] = Op::ISub as u32;
    for (module, named) in [
        (
            inserted(&words, Op::EntryPoint, &mode_id, 0),
            "OpExecutionModeId",
        ),
        (
            inserted(&words, Op::Decorate, &group, 1),
            "OpDecorationGroup",
        ),
        (subtract, "OpSpecConstantOp of opcode 130"),
    ] {
        match specialize(&module, &[]) {
            Err(crate::error::Error::Internal(error)) => {
                assert!(error.detail.contains(named), "{named}: {error}");
            }
            other => panic!("{named}: {:?}", other.map(|w| w.len())),
        }
    }
}

/// A module with a `LocalSize` mode beside its `WorkgroupSize` composite, as `glslang` writes one
/// from `local_size_x_id`: the composite takes precedence, so the one mode takes the composite's
/// values, and `spirv-val` accepts the words. A second mode is refused.
#[test]
fn a_local_size_mode_beside_the_workgroup_composite_takes_its_values() {
    let (words, ids, opts) = linked(deep_kernel());
    let entry = (insts(&words).into_iter())
        .find(|(op, _)| *op == Op::EntryPoint as u32)
        .map(|(_, operands)| operands[1])
        .expect("entry point");
    let mode = [
        (6 << 16) | Op::ExecutionMode as u32,
        entry,
        ExecutionMode::LocalSize as u32,
        1,
        1,
        1,
    ];
    let both = inserted(&words, Op::EntryPoint, &mode, 0);
    assert_valid(&both, opts);
    let value = |key: SpecKey| match key {
        SpecKey::WorkgroupSize(Axis::X) => 8,
        SpecKey::WorkgroupSize(Axis::Y) => 4,
        SpecKey::WorkgroupSize(Axis::Z) => 2,
        _ => 1,
    };
    let values: Vec<(u32, u64)> = ids.iter().map(|&(key, id)| (id, value(key))).collect();
    let frozen = specialize(&both, &values).expect("specialize");
    assert_valid(&frozen, opts);
    let modes: Vec<Vec<u32>> = (insts(&frozen).into_iter())
        .filter(|(op, _)| *op == Op::ExecutionMode as u32)
        .map(|(_, operands)| operands)
        .collect();
    assert_eq!(modes, [vec![entry, ExecutionMode::LocalSize as u32, 8, 4, 2]]);
    let twice = inserted(&both, Op::EntryPoint, &mode, 0);
    match specialize(&twice, &values) {
        Err(crate::error::Error::Internal(error)) => {
            assert!(error.detail.contains("more than one LocalSize"), "{error}");
        }
        other => panic!("two modes: {:?}", other.map(|w| w.len())),
    }
}
