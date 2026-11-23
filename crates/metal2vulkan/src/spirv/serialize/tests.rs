//! Serializer tests: `spirv-val` over every builder family, the parse round trip, relocation, and
//! determinism.

mod builder_properties;
mod derivation;
mod families;
mod families_image;
mod families_memory;
mod families_mesh;
mod families_ray;
mod families_stage;
mod forwarded;
mod lazy;
mod model_rules;

use std::collections::BTreeMap;

use crate::pipeline::sym::{
    AirUserKey, ResourceKind, ResourceSlot, SpecKey, Stage, Symbol, VaryingKey,
};
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
use crate::spirv::error::{BuildError, RelocRule};
use crate::spirv::ids::{ConstId, GlobalId, TypeId, Value};
use crate::spirv::ir::{
    natural_layout, BinOp, ConstBits, Constant, ExecMode, ExecModel, IoDecor, IoDir, LayoutRule,
    MemAccess, StorageClass, TexelMemory, Type, TypedModule, UnOp,
};
use crate::spirv::serialize::{relocate, serialize, RelocatableModule};
use crate::spirv::testing::{spirv_dis, spirv_val_with, TargetEnv, ValOptions};

// ---- Shared fixture ----

pub(super) fn int(b: &mut ModuleBuilder, width: u8, signed: bool) -> TypeId {
    b.ty(Type::Int { width, signed }).expect("int")
}

pub(super) fn float(b: &mut ModuleBuilder, width: u8) -> TypeId {
    b.ty(Type::Float { width }).expect("float")
}

pub(super) fn vec(b: &mut ModuleBuilder, elem: TypeId, len: u8) -> TypeId {
    b.ty(Type::Vector { elem, len }).expect("vector")
}

pub(super) fn scalar(b: &mut ModuleBuilder, ty: TypeId, bits: u64) -> ConstId {
    b.constant(Constant::Scalar {
        ty,
        bits: ConstBits(bits),
    })
    .expect("constant")
}

pub(super) fn cval(f: &mut FunctionBuilder<'_>, ty: TypeId, bits: u64) -> Value {
    let c = scalar(f.module(), ty, bits);
    f.module().const_value(c).expect("constant value")
}

pub(super) fn slot(kind: ResourceKind, index: u32) -> ResourceSlot {
    ResourceSlot {
        stage: Stage::Kernel,
        kind,
        index,
    }
}

pub(super) fn key(name: &str) -> VaryingKey {
    VaryingKey(AirUserKey(name.as_bytes().into()))
}

/// A `Block { elem data[]; }` storage buffer (std430) in `slot`, with its pointer value.
pub(super) fn ssbo(b: &mut ModuleBuilder, index: u32, elem: TypeId, stride: u32) -> GlobalId {
    let rta = b
        .ty(Type::RuntimeArray {
            elem,
            stride: Some(stride),
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
        slot(ResourceKind::Buffer, index),
        StorageClass::StorageBuffer,
        block,
    )
    .expect("resource")
}

/// A pointer to element `i` of storage buffer `g`.
pub(super) fn elem_ptr(f: &mut FunctionBuilder<'_>, g: GlobalId, i: Value) -> Value {
    let u32_ = int(f.module(), 32, false);
    let zero = cval(f, u32_, 0);
    let base = f.module().global_value(g).expect("global");
    f.access_chain(base, &[zero, i]).expect("access chain")
}

/// A link table numbering every symbol of `r` densely per kind (set 0 for every resource).
pub(super) fn link_all(r: &RelocatableModule) -> BTreeMap<Symbol, u32> {
    let mut table = BTreeMap::new();
    let mut next = [0u32; 3];
    for (sym, _) in &r.relocs {
        if table.contains_key(sym) {
            continue;
        }
        let n = match sym {
            Symbol::DescriptorSet(_) => 0,
            Symbol::Binding(_) => bump(&mut next[0]),
            Symbol::Location(_) => bump(&mut next[1]),
            Symbol::Component(_) => 0,
            Symbol::SpecId(_) => bump(&mut next[2]),
        };
        table.insert(sym.clone(), n);
    }
    table
}

fn bump(n: &mut u32) -> u32 {
    let v = *n;
    *n += 1;
    v
}

/// Serialize `m`, link it with [`link_all`], and require `spirv-val` to accept it under Vulkan
/// 1.2 and 1.3 with exactly the relaxations its requirements enable. Returns the linked words.
pub(super) fn assert_spirv_valid(m: TypedModule) -> Vec<u32> {
    let mut meter = crate::budget::Meter::new(&crate::budget::Budget::default());
    if let Err(e) = crate::spirv::verify::check_structure(&m, &mut meter) {
        panic!("the owned verifier rejected a builder module: {e}");
    }
    let r = serialize(m).expect("serialize");
    let words = relocate(&r, &link_all(&r)).expect("relocate");
    if let Err(e) = crate::spirv::requirements::tests::check_against_registry(&r.requirements) {
        panic!(
            "requirements disagree with the registry:\n{e}\nrequirements: {:?}",
            r.requirements
        );
    }
    let opts = ValOptions::for_requirements(&r.requirements);
    for env in [TargetEnv::Vulkan1_2, TargetEnv::Vulkan1_3] {
        if let Err(e) = spirv_val_with(&words, env, opts) {
            panic!(
                "spirv-val {env:?} rejected the module: {e}\nrequirements: {:?}\n{}",
                r.requirements,
                spirv_dis(&words)
            );
        }
    }
    words
}

/// A compute entry around `body`, finished and finalized.
pub(super) fn compute(
    b: &mut ModuleBuilder,
    body: impl FnOnce(&mut FunctionBuilder<'_>),
) -> crate::spirv::ids::FuncId {
    let void = b.ty(Type::Void).expect("void");
    let mut f = b.function(void).expect("function");
    body(&mut f);
    if f.current_block().is_some() {
        f.ret(None).expect("return");
    }
    f.finish().expect("finish")
}

pub(super) fn finish_compute(mut b: ModuleBuilder, main: crate::spirv::ids::FuncId) -> TypedModule {
    b.entry(
        ExecModel::GLCompute,
        main,
        vec![ExecMode::LocalSize { x: 8, y: 1, z: 1 }],
    )
    .expect("entry");
    b.finalize().expect("finalize")
}

/// The invocation index `gl_GlobalInvocationID.x`.
pub(super) fn gid_x(f: &mut FunctionBuilder<'_>) -> Value {
    let u32_ = int(f.module(), 32, false);
    let uvec3 = vec(f.module(), u32_, 3);
    let g = f
        .module()
        .builtin(crate::spv::BuiltIn::GlobalInvocationId, IoDir::Input, uvec3)
        .expect("builtin");
    let p = f.module().global_value(g).expect("global");
    let v = f.load(p, MemAccess::default()).expect("load");
    f.composite_extract(v, &[0]).expect("extract")
}

/// `out[gid] = in[gid] * 3 + 1` over `u32` storage buffers.
fn simple_kernel() -> TypedModule {
    let mut b = ModuleBuilder::new();
    let u32_ = int(&mut b, 32, false);
    let input = ssbo(&mut b, 0, u32_, 4);
    let output = ssbo(&mut b, 1, u32_, 4);
    let main = compute(&mut b, |f| {
        let i = gid_x(f);
        let src = elem_ptr(f, input, i);
        let x = f.load(src, MemAccess::default()).expect("load");
        let three = cval(f, u32_, 3);
        let one = cval(f, u32_, 1);
        let y = f.binary(BinOp::IMul, x, three).expect("mul");
        let y = f.binary(BinOp::IAdd, y, one).expect("add");
        let dst = elem_ptr(f, output, i);
        f.store(dst, y, MemAccess::default()).expect("store");
    });
    finish_compute(b, main)
}

#[test]
fn a_simple_kernel_passes_spirv_val() {
    assert_spirv_valid(simple_kernel());
}

#[test]
fn spirv_val_rejects_an_invalid_module() {
    let r = serialize(simple_kernel()).expect("serialize");
    let mut words = relocate(&r, &link_all(&r)).expect("relocate");
    // Drop the final `OpFunctionEnd`: the function is unterminated.
    words.pop();
    assert!(spirv_val_with(&words, TargetEnv::Vulkan1_2, ValOptions::default()).is_err());
}

#[test]
fn serialization_is_deterministic() {
    let a = serialize(simple_kernel()).expect("serialize");
    let b = serialize(simple_kernel()).expect("serialize");
    assert_eq!(a, b);
}

#[test]
fn serialized_words_parse_back_through_the_grammar() {
    let r = serialize(simple_kernel()).expect("serialize");
    let words = relocate(&r, &link_all(&r)).expect("relocate");
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let parsed = crate::spirv_binary::parse_bytes(&bytes).expect("parse");
    let ops: Vec<crate::spv::Op> = parsed.instructions.iter().map(|i| i.opcode).collect();
    // Logical layout: capabilities, memory model, entry point, mode, annotations, then
    // declarations and one function.
    assert_eq!(ops.first(), Some(&crate::spv::Op::Capability));
    let pos = |op| ops.iter().position(|o| *o == op).expect("present");
    assert!(pos(crate::spv::Op::MemoryModel) < pos(crate::spv::Op::EntryPoint));
    assert!(pos(crate::spv::Op::EntryPoint) < pos(crate::spv::Op::ExecutionMode));
    assert!(pos(crate::spv::Op::ExecutionMode) < pos(crate::spv::Op::Decorate));
    assert!(pos(crate::spv::Op::Decorate) < pos(crate::spv::Op::TypeVoid));
    assert!(pos(crate::spv::Op::TypeVoid) < pos(crate::spv::Op::Function));
    assert_eq!(ops.last(), Some(&crate::spv::Op::FunctionEnd));
    assert_eq!(
        ops.iter()
            .filter(|o| **o == crate::spv::Op::Function)
            .count(),
        1
    );
    // The id bound is tight: every id below it is defined.
    let max = parsed
        .instructions
        .iter()
        .filter_map(|i| i.result_id)
        .max()
        .unwrap_or(0);
    assert_eq!(parsed.bound, max + 1);
}

/// A copy is written as its operand: `out[gid] = copy(copy(in[gid])) + copy(1)` carries no
/// `OpCopyObject`, stores the loaded id's sum, keeps a tight id bound, and passes `spirv-val`.
#[test]
fn copies_are_written_as_their_operands() {
    let mut b = ModuleBuilder::new();
    let u32_ = int(&mut b, 32, false);
    let input = ssbo(&mut b, 0, u32_, 4);
    let output = ssbo(&mut b, 1, u32_, 4);
    let main = compute(&mut b, |f| {
        let i = gid_x(f);
        let src = elem_ptr(f, input, i);
        let x = f.load(src, MemAccess::default()).expect("load");
        let x = f.unary(UnOp::CopyObject, x).expect("copy");
        let x = f.unary(UnOp::CopyObject, x).expect("copy of a copy");
        let one = cval(f, u32_, 1);
        let one = f.unary(UnOp::CopyObject, one).expect("copy of a constant");
        let y = f.binary(BinOp::IAdd, x, one).expect("add");
        let dst = elem_ptr(f, output, i);
        f.store(dst, y, MemAccess::default()).expect("store");
    });
    let words = assert_spirv_valid(finish_compute(b, main));
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let parsed = crate::spirv_binary::parse_bytes(&bytes).expect("parse");
    let find = |op| {
        parsed
            .instructions
            .iter()
            .find(|i| i.opcode == op)
            .expect("present")
    };
    assert!(parsed
        .instructions
        .iter()
        .all(|i| i.opcode != crate::spv::Op::CopyObject));
    // The add reads the loaded id and the constant's id directly.
    let loaded = parsed
        .instructions
        .iter()
        .filter(|i| i.opcode == crate::spv::Op::Load)
        .filter_map(|i| i.result_id)
        .collect::<Vec<_>>();
    let add = find(crate::spv::Op::IAdd);
    let operands: Vec<u32> = add
        .operands
        .iter()
        .filter_map(|o| match o {
            crate::spirv_binary::Operand::IdRef(id) => Some(*id),
            _ => None,
        })
        .collect();
    assert!(operands.first().is_some_and(|x| loaded.contains(x)));
    let constant = parsed
        .instructions
        .iter()
        .find(|i| i.opcode == crate::spv::Op::Constant && i.result_id == operands.get(1).copied());
    assert!(constant.is_some(), "the add reads the constant itself");
    let max = parsed
        .instructions
        .iter()
        .filter_map(|i| i.result_id)
        .max()
        .unwrap_or(0);
    assert_eq!(parsed.bound, max + 1);
}

/// Every `OpUndef` of one type is one module-level `OpUndef`: three undefined values of two types
/// serialize as two `OpUndef`s among the declarations, and the module passes `spirv-val`.
#[test]
fn undefs_are_one_module_level_undef_per_type() {
    let mut b = ModuleBuilder::new();
    let u32_ = int(&mut b, 32, false);
    let uvec2 = vec(&mut b, u32_, 2);
    let output = ssbo(&mut b, 1, u32_, 4);
    let main = compute(&mut b, |f| {
        let i = gid_x(f);
        let a = f.undef(uvec2).expect("undef");
        let a = f.composite_insert(i, a, &[0]).expect("insert");
        let c = f.undef(uvec2).expect("second undef");
        let c = f.composite_insert(i, c, &[1]).expect("insert");
        let x = f.composite_extract(a, &[0]).expect("extract");
        let y = f.composite_extract(c, &[1]).expect("extract");
        let z = f.undef(u32_).expect("scalar undef");
        let y = f.binary(BinOp::BitwiseAnd, y, z).expect("and");
        let s = f.binary(BinOp::IAdd, x, y).expect("add");
        let dst = elem_ptr(f, output, i);
        f.store(dst, s, MemAccess::default()).expect("store");
    });
    let words = assert_spirv_valid(finish_compute(b, main));
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let parsed = crate::spirv_binary::parse_bytes(&bytes).expect("parse");
    let ops: Vec<crate::spv::Op> = parsed.instructions.iter().map(|i| i.opcode).collect();
    let undefs: Vec<usize> = ops
        .iter()
        .enumerate()
        .filter(|(_, o)| **o == crate::spv::Op::Undef)
        .map(|(i, _)| i)
        .collect();
    assert_eq!(undefs.len(), 2);
    let function = ops
        .iter()
        .position(|o| *o == crate::spv::Op::Function)
        .expect("function");
    assert!(undefs.iter().all(|i| *i < function));
}

/// A module with every relocatable symbol kind: two buffers, a packed varying, a spec constant.
fn relocatable_fragment() -> TypedModule {
    let mut b = ModuleBuilder::new();
    let u32_ = int(&mut b, 32, false);
    let f32_ = float(&mut b, 32);
    let vec4 = vec(&mut b, f32_, 4);
    let a = ssbo(&mut b, 0, u32_, 4);
    let c = ssbo(&mut b, 1, u32_, 4);
    let spec = b
        .spec(SpecKey::SampleCount, u32_, ConstBits(1))
        .expect("spec");
    let input = b
        .io(
            key("color"),
            IoDir::Input,
            vec4,
            IoDecor {
                packed: true,
                ..IoDecor::default()
            },
        )
        .expect("io");
    let output = b
        .io(key("target"), IoDir::Output, vec4, IoDecor::default())
        .expect("io");
    let void = b.ty(Type::Void).expect("void");
    let mut f = b.function(void).expect("function");
    let zero = cval(&mut f, u32_, 0);
    let sv = f.module().const_value(spec).expect("spec value");
    let pa = elem_ptr(&mut f, a, zero);
    let x = f.load(pa, MemAccess::default()).expect("load");
    let x = f.binary(BinOp::IAdd, x, sv).expect("add");
    let pc = elem_ptr(&mut f, c, zero);
    f.store(pc, x, MemAccess::default()).expect("store");
    let pin = f.module().global_value(input).expect("in");
    let v = f.load(pin, MemAccess::default()).expect("load");
    let pout = f.module().global_value(output).expect("out");
    f.store(pout, v, MemAccess::default()).expect("store");
    f.ret(None).expect("ret");
    let main = f.finish().expect("finish");
    b.entry(ExecModel::Fragment, main, vec![]).expect("entry");
    b.finalize().expect("finalize")
}

#[test]
fn relocations_cover_every_link_symbol_and_nothing_else() {
    let r = serialize(relocatable_fragment()).expect("serialize");
    let kinds: Vec<u8> = r
        .relocs
        .iter()
        .map(|(s, _)| match s {
            Symbol::Binding(_) => 0,
            Symbol::DescriptorSet(_) => 1,
            Symbol::Location(_) => 2,
            Symbol::Component(_) => 3,
            Symbol::SpecId(_) => 4,
        })
        .collect();
    for k in 0..5u8 {
        assert!(kinds.contains(&k), "symbol kind {k} relocated");
    }
    // Two buffers (set + binding each), two varyings, one component, one spec id.
    assert_eq!(r.relocs.len(), 4 + 2 + 1 + 1);
    for (_, at) in &r.relocs {
        assert_eq!(r.words[*at as usize], 0, "relocation slots are zero");
    }
    assert_spirv_valid(relocatable_fragment());
}

/// Over `rounds` pseudo-random link tables (from `seed`), the words with relocation slots zeroed
/// are identical to the serialized words, and relocation writes exactly those slots.
pub(super) fn assert_relocation_writes_only_literals(
    r: &RelocatableModule,
    seed: u32,
    rounds: u32,
) {
    assert!(!r.relocs.is_empty(), "the module has no relocation slots");
    let slots: std::collections::BTreeSet<usize> =
        r.relocs.iter().map(|(_, at)| *at as usize).collect();
    let mut seed = seed | 1;
    for round in 0..rounds {
        let table: BTreeMap<Symbol, u32> = r
            .relocs
            .iter()
            .map(|(s, _)| {
                seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (s.clone(), (seed >> 8) % 64 + round)
            })
            .collect();
        let words = relocate(r, &table).expect("relocate");
        assert_eq!(words.len(), r.words.len());
        let mut zeroed = words.clone();
        for (i, w) in words.iter().enumerate() {
            if slots.contains(&i) {
                zeroed[i] = 0;
            } else {
                assert_eq!(*w, r.words[i], "word {i} changed outside a relocation slot");
            }
        }
        assert_eq!(zeroed, r.words);
        for (sym, at) in &r.relocs {
            assert_eq!(words[*at as usize], table[sym]);
        }
    }
}

/// Over the fixed module shapes here: one with every relocatable symbol kind and a compute kernel.
/// `builder_properties` runs the same check over every random program.
#[test]
fn relocation_writes_only_decoration_literals() {
    for (i, m) in [relocatable_fragment(), simple_kernel()]
        .into_iter()
        .enumerate()
    {
        let r = serialize(m).expect("serialize");
        assert_relocation_writes_only_literals(&r, 0x9e37_79b9 ^ i as u32, 16);
    }
}

#[test]
fn a_missing_symbol_is_a_typed_error() {
    let r = serialize(relocatable_fragment()).expect("serialize");
    let mut table = link_all(&r);
    let (missing, _) = r.relocs[0].clone();
    table.remove(&missing);
    assert_eq!(
        relocate(&r, &table),
        Err(BuildError::Reloc(RelocRule::MissingSymbol(missing)))
    );
}

#[test]
fn switch_literals_serialize_as_one_word_each() {
    let mut b = ModuleBuilder::new();
    let u32_ = int(&mut b, 32, false);
    let out = ssbo(&mut b, 0, u32_, 4);
    let main = compute(&mut b, |f| {
        let i = gid_x(f);
        let mut ctx = ();
        let arm = |v: u64| -> crate::spirv::builder::constructs::ArmFn<'_, ()> {
            Box::new(move |fb: &mut FunctionBuilder<'_>, _: &mut ()| {
                let u = int(fb.module(), 32, false);
                Ok(crate::spirv::builder::constructs::ArmEnd::FallThrough(
                    vec![cval(fb, u, v)],
                ))
            })
        };
        let merged = f
            .switch_(
                &mut ctx,
                i,
                vec![(vec![0, u32::MAX], arm(10)), (vec![7], arm(20))],
                arm(30),
            )
            .expect("switch");
        let p = elem_ptr(f, out, i);
        f.store(p, merged.0[0], MemAccess::default())
            .expect("store");
    });
    let words = assert_spirv_valid(finish_compute(b, main));
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let parsed = crate::spirv_binary::parse_bytes(&bytes).expect("parse");
    let sw = parsed
        .instructions
        .iter()
        .find(|i| i.opcode == crate::spv::Op::Switch)
        .expect("switch");
    // selector, default, then (literal, label) for 3 literals.
    assert_eq!(sw.operands.len(), 2 + 2 * 3);
}

/// The two constructor families the other sweeps do not reach: a switch-at-use (it shares
/// `switch_impl` with `switch_` but records a different construct) and an explicit
/// `OpUnreachable` terminator, here closing the switch's default arm.
#[test]
fn switch_at_use_and_unreachable_pass_spirv_val() {
    use crate::spirv::builder::constructs::{ArmEnd, ArmFn};
    let mut b = ModuleBuilder::new();
    let u32_ = int(&mut b, 32, false);
    let out = ssbo(&mut b, 0, u32_, 4);
    let main = compute(&mut b, |f| {
        let i = gid_x(f);
        let mut ctx = ();
        let arm = |v: u64| -> ArmFn<'_, ()> {
            Box::new(move |fb: &mut FunctionBuilder<'_>, _: &mut ()| {
                let u = int(fb.module(), 32, false);
                Ok(ArmEnd::FallThrough(vec![cval(fb, u, v)]))
            })
        };
        let unreachable: ArmFn<'_, ()> = Box::new(|fb: &mut FunctionBuilder<'_>, _: &mut ()| {
            fb.unreachable()?;
            Ok(ArmEnd::Exits)
        });
        let merged = f
            .switch_at_use(
                &mut ctx,
                i,
                vec![(vec![0], arm(10)), (vec![1, 2], arm(20))],
                unreachable,
            )
            .expect("switch at use");
        let p = elem_ptr(f, out, i);
        f.store(p, merged.0[0], MemAccess::default())
            .expect("store");
    });
    let words = assert_spirv_valid(finish_compute(b, main));
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let parsed = crate::spirv_binary::parse_bytes(&bytes).expect("parse");
    for op in [crate::spv::Op::Switch, crate::spv::Op::Unreachable] {
        assert!(
            parsed.instructions.iter().any(|i| i.opcode == op),
            "{op:?} missing"
        );
    }
}

/// A compute module whose invocation `gid` switches on `gid` over `cases` (each arm's literals,
/// falling through with the arm's number) of `kind`, storing what it merges.
fn many_case_switch(
    cases: &[Vec<u32>],
    kind: crate::spirv::builder::constructs::SwitchKind,
) -> TypedModule {
    use crate::spirv::builder::constructs::{ArmEnd, ArmFn};
    let mut b = ModuleBuilder::new();
    let u32_ = int(&mut b, 32, false);
    let out = ssbo(&mut b, 0, u32_, 4);
    let main = compute(&mut b, |f| {
        let i = gid_x(f);
        let arm = |v: u64| -> ArmFn<'_, ()> {
            Box::new(move |fb: &mut FunctionBuilder<'_>, _: &mut ()| {
                let u = int(fb.module(), 32, false);
                Ok(ArmEnd::FallThrough(vec![cval(fb, u, v)]))
            })
        };
        let arms = (0u64..)
            .zip(cases)
            .map(|(a, lits)| (lits.clone(), arm(a)))
            .collect();
        let merged = f
            .switch_within(&mut (), i, arms, arm(u64::from(u32::MAX)), kind)
            .expect("switch");
        let p = elem_ptr(f, out, i);
        f.store(p, merged.0[0], MemAccess::default())
            .expect("store");
    });
    finish_compute(b, main)
}

/// The `(literal, label)` pairs of each `OpSwitch` of `words`.
fn switch_pairs(words: &[u32]) -> Vec<usize> {
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let parsed = crate::spirv_binary::parse_bytes(&bytes).expect("parse");
    parsed
        .instructions
        .iter()
        .filter(|i| i.opcode == crate::spv::Op::Switch)
        .map(|i| (i.operands.len() - 2) / 2)
        .collect()
}

/// SPIR-V allows an `OpSwitch` 16,383 `(literal, label)` pairs, and `spirv-val` rejects one more. A
/// switch-at-use of 16,384 one-literal arms, and a source switch of 40,000 literals in arms of up
/// to 7 (several arms' literals interleaved), are split into `OpSwitch`es within the limit and pass
/// `spirv-val`; one of exactly 16,383 is one `OpSwitch`, as before. Built as one `OpSwitch`
/// regardless (no limit), the 16,384 arms are rejected.
#[test]
fn switches_past_the_pair_limit_are_split_and_pass_spirv_val() {
    use crate::spirv::builder::constructs::{SwitchKind, SWITCH_PAIRS};
    let ones = |n: u32| (0..n).map(|l| vec![l]).collect::<Vec<_>>();
    let at_use = |limit| SwitchKind {
        at_use: true,
        limit,
    };
    let words = assert_spirv_valid(many_case_switch(&ones(16_383), at_use(SWITCH_PAIRS)));
    assert_eq!(switch_pairs(&words), vec![16_383]);
    let words = assert_spirv_valid(many_case_switch(&ones(16_384), at_use(SWITCH_PAIRS)));
    let pairs = switch_pairs(&words);
    assert!(
        pairs.len() > 1 && pairs.iter().all(|n| *n <= SWITCH_PAIRS),
        "{pairs:?}"
    );
    // Arms of 1 to 7 literals: arm `a` holds `a, a + n, a + 2n, …`, so arms interleave.
    let arms = 10_000u32;
    let wide: Vec<Vec<u32>> = (0..arms)
        .map(|a| (0..1 + a % 7).map(|k| a + k * arms).collect())
        .collect();
    let total: usize = wide.iter().map(Vec::len).sum();
    assert!(total > 2 * SWITCH_PAIRS);
    let source = SwitchKind {
        at_use: false,
        limit: SWITCH_PAIRS,
    };
    let words = assert_spirv_valid(many_case_switch(&wide, source));
    let pairs = switch_pairs(&words);
    assert!(pairs.iter().all(|n| *n <= SWITCH_PAIRS), "{pairs:?}");
    // One OpSwitch of 16,384 pairs: the limit is spirv-val's.
    let m = many_case_switch(&ones(16_384), at_use(usize::MAX));
    let r = serialize(m).expect("serialize");
    let words = relocate(&r, &link_all(&r)).expect("relocate");
    assert_eq!(switch_pairs(&words), vec![16_384]);
    let opts = ValOptions::for_requirements(&r.requirements);
    let e = spirv_val_with(&words, TargetEnv::Vulkan1_2, opts).expect_err("16,384 pairs");
    assert!(e.contains("exceeds the limit"), "{e}");
}

#[test]
fn an_instruction_longer_than_the_word_count_field_is_a_typed_error() {
    let mut w = super::Writer::default();
    let fits = vec![0u32; 0xFFFE];
    w.inst(crate::spv::Op::CompositeConstruct, &fits)
        .expect("0xFFFF words fit");
    assert_eq!(w.words.len(), 0xFFFF);
    let long = vec![0u32; 0xFFFF];
    assert_eq!(
        w.inst(crate::spv::Op::CompositeConstruct, &long),
        Err(BuildError::InstructionTooLong {
            op: crate::spv::Op::CompositeConstruct,
            words: 0x1_0000
        })
    );
    assert_eq!(w.words.len(), 0xFFFF, "nothing written on refusal");
}

/// The local size as a `WorkgroupSize` spec composite of `sx, sy, sz` is valid under Vulkan 1.2
/// without `maintenance4` (no `LocalSizeId`), carries one `SpecId` relocation per axis, and a
/// `GLCompute` entry needs no local-size mode with it.
#[test]
fn a_workgroup_size_spec_composite_passes_spirv_val_without_maintenance4() {
    use crate::pipeline::sym::Axis;
    let mut b = ModuleBuilder::new();
    let u32_ = int(&mut b, 32, false);
    let axes = [Axis::X, Axis::Y, Axis::Z].map(|a| {
        b.spec(SpecKey::LocalSize(a), u32_, ConstBits(1))
            .expect("spec")
    });
    let wg = b.workgroup_size(axes[0], axes[1], axes[2]).expect("wg");
    // Declaring it again with the same parts is the same constant.
    assert_eq!(b.workgroup_size(axes[0], axes[1], axes[2]), Ok(wg));
    assert_eq!(
        b.workgroup_size(axes[1], axes[0], axes[2]),
        Err(BuildError::Constant(
            crate::spirv::error::ConstRule::CompositeParts
        ))
    );
    let out = ssbo(&mut b, 0, u32_, 4);
    let main = compute(&mut b, |f| {
        let i = gid_x(f);
        let dst = elem_ptr(f, out, i);
        let v = cval(f, u32_, 7);
        f.store(dst, v, MemAccess::default()).expect("store");
    });
    b.entry(ExecModel::GLCompute, main, Vec::new())
        .expect("entry");
    let m = b.finalize().expect("finalize");
    let r = serialize(m.clone()).expect("serialize");
    let specs = r
        .relocs
        .iter()
        .filter(|(s, _)| matches!(s, Symbol::SpecId(SpecKey::LocalSize(_))))
        .count();
    assert_eq!(specs, 3);
    assert!(!r
        .requirements
        .features
        .contains(&crate::device::requirements::VkFeature::Maintenance4));
    let words = assert_spirv_valid(m);
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let parsed = crate::spirv_binary::parse_bytes(&bytes).expect("parse");
    let ops: Vec<crate::spv::Op> = parsed.instructions.iter().map(|i| i.opcode).collect();
    assert!(ops.contains(&crate::spv::Op::SpecConstantComposite));
    assert!(!ops.contains(&crate::spv::Op::ExecutionMode));
    assert!(!ops.contains(&crate::spv::Op::ExecutionModeId));
}

/// A zero local size (a specialization default or a plain part) or a composite without a
/// specialization part is refused; plain positive parts beside a specialization constant are the
/// flattened `(sn, 1, 1)` form and pass `spirv-val`.
#[test]
fn a_workgroup_size_composite_needs_positive_u32_parts_and_a_spec_part() {
    use crate::pipeline::sym::Axis;
    use crate::spirv::error::ConstRule;
    let mut b = ModuleBuilder::new();
    let u32_ = int(&mut b, 32, false);
    let one = b
        .spec(SpecKey::LocalSize(Axis::X), u32_, ConstBits(1))
        .expect("spec");
    let zero = b
        .spec(SpecKey::LocalSize(Axis::Y), u32_, ConstBits(0))
        .expect("spec");
    let plain = scalar(&mut b, u32_, 1);
    assert_eq!(
        b.workgroup_size(one, zero, one),
        Err(BuildError::Constant(ConstRule::CompositeParts))
    );
    let plain_zero = scalar(&mut b, u32_, 0);
    assert_eq!(
        b.workgroup_size(one, plain_zero, plain),
        Err(BuildError::Constant(ConstRule::CompositeParts))
    );
    assert_eq!(
        b.workgroup_size(plain, plain, plain),
        Err(BuildError::Constant(ConstRule::CompositeParts))
    );
    let flat = b
        .spec(SpecKey::FlatLocalSize, u32_, ConstBits(64))
        .expect("spec");
    b.workgroup_size(flat, plain, plain).expect("(sn, 1, 1)");
    let out = ssbo(&mut b, 0, u32_, 64);
    let main = compute(&mut b, |f| {
        let i = gid_x(f);
        let dst = elem_ptr(f, out, i);
        let v = cval(f, u32_, 7);
        f.store(dst, v, MemAccess::default()).expect("store");
    });
    b.entry(ExecModel::GLCompute, main, Vec::new())
        .expect("entry");
    assert_spirv_valid(b.finalize().expect("finalize"));
}

/// A `WorkgroupSize` part that is an `OpSpecConstantOp` is refused, whatever its operation, the
/// deep threadgroup's former parts among them (a select over products of the size constants): a
/// driver may size the workgroup from the parts as declared and read such a part as 1, while the
/// body computes with its value. Plain specialization constants on every axis, the deep form's
/// parts, pass `spirv-val`.
#[test]
fn a_workgroup_size_part_is_never_a_spec_constant_op() {
    use crate::pipeline::sym::Axis;
    use crate::spirv::error::ConstRule;
    use crate::spirv::ir::SpecOp;
    let mut b = ModuleBuilder::new();
    let u32_ = int(&mut b, 32, false);
    let bool_ = b.ty(Type::Bool).expect("bool");
    let [sx, sy, sz] = [Axis::X, Axis::Y, Axis::Z].map(|a| {
        b.spec(SpecKey::LocalSize(a), u32_, ConstBits(1))
            .expect("spec")
    });
    let one_row = b
        .spec(SpecKey::DeepThreadgroup, bool_, ConstBits(0))
        .expect("spec");
    let row = b.spec_op(SpecOp::IMul, u32_, &[sx, sy]).expect("row");
    let sn = b.spec_op(SpecOp::IMul, u32_, &[row, sz]).expect("sn");
    let one = scalar(&mut b, u32_, 1);
    let x = (b.spec_op(SpecOp::Select, u32_, &[one_row, sn, sx])).expect("x");
    let y = (b.spec_op(SpecOp::Select, u32_, &[one_row, one, sy])).expect("y");
    let z = (b.spec_op(SpecOp::Select, u32_, &[one_row, one, sz])).expect("z");
    let op = Err(BuildError::Constant(ConstRule::WorkgroupSizeOp));
    assert_eq!(b.workgroup_size(x, y, z), op);
    assert_eq!(b.workgroup_size(sn, one, one), op);
    assert_eq!(b.workgroup_size(sx, sy, row), op);
    let [wx, wy, wz] = [Axis::X, Axis::Y, Axis::Z].map(|a| {
        b.spec(SpecKey::WorkgroupSize(a), u32_, ConstBits(1))
            .expect("spec")
    });
    b.workgroup_size(wx, wy, wz).expect("the deep form's parts");
    let out = ssbo(&mut b, 0, u32_, 4);
    let main = compute(&mut b, |f| {
        let i = gid_x(f);
        let dst = elem_ptr(f, out, i);
        let v = cval(f, u32_, 7);
        f.store(dst, v, MemAccess::default()).expect("store");
    });
    b.entry(ExecModel::GLCompute, main, Vec::new())
        .expect("entry");
    assert_spirv_valid(b.finalize().expect("finalize"));
}

/// The resources a module declares `NonWritable`, by binding symbol, in relocation order.
fn non_writable_resources(r: &RelocatableModule) -> Vec<ResourceSlot> {
    let decorate = (3 << 16) | crate::spv::Op::Decorate as u32;
    let non_writable = crate::spv::Decoration::NonWritable as u32;
    let ids: std::collections::BTreeSet<u32> = r
        .words
        .windows(3)
        .filter(|w| w[0] == decorate && w[2] == non_writable)
        .map(|w| w[1])
        .collect();
    r.relocs
        .iter()
        .filter_map(|(s, at)| match s {
            Symbol::Binding(slot) => r
                .words
                .get((*at as usize).checked_sub(2)?)
                .filter(|id| ids.contains(id))
                .map(|_| *slot),
            _ => None,
        })
        .collect()
}

/// A graphics stage decorates exactly the storage buffers nothing writes `NonWritable`
/// (`VUID-RuntimeSpirv-NonWritable-06340/06341`); a kernel is left as it was.
#[test]
fn read_only_storage_buffers_of_graphics_stages_are_non_writable() {
    for model in [ExecModel::Vertex, ExecModel::Fragment, ExecModel::GLCompute] {
        let build = || {
            let mut b = ModuleBuilder::new();
            let u32_ = int(&mut b, 32, false);
            let input = ssbo(&mut b, 0, u32_, 4);
            let output = ssbo(&mut b, 1, u32_, 4);
            let main = compute(&mut b, |f| {
                let zero = cval(f, u32_, 0);
                let src = elem_ptr(f, input, zero);
                let x = f.load(src, MemAccess::default()).expect("load");
                let dst = elem_ptr(f, output, zero);
                f.store(dst, x, MemAccess::default()).expect("store");
            });
            let modes = match model {
                ExecModel::GLCompute => vec![ExecMode::LocalSize { x: 1, y: 1, z: 1 }],
                ExecModel::Fragment => vec![ExecMode::OriginUpperLeft],
                _ => vec![],
            };
            b.entry(model, main, modes).expect("entry");
            b.finalize().expect("finalize")
        };
        assert_spirv_valid(build());
        let r = serialize(build()).expect("serialize");
        let want = match model {
            ExecModel::GLCompute => vec![],
            _ => vec![slot(ResourceKind::Buffer, 0)],
        };
        assert_eq!(non_writable_resources(&r), want, "{model:?}");
    }
}

/// The entry modes `model` needs in these tests.
fn modes_for(model: ExecModel) -> Vec<ExecMode> {
    match model {
        ExecModel::GLCompute => vec![ExecMode::LocalSize { x: 1, y: 1, z: 1 }],
        ExecModel::Fragment => vec![ExecMode::OriginUpperLeft],
        _ => vec![],
    }
}

/// The `r32ui` 2D storage image type.
fn r32ui(b: &mut ModuleBuilder) -> TypeId {
    use families_image::{image_ty, Img, SAMPLED_2D};
    let u32_ = int(b, 32, false);
    image_ty(
        b,
        u32_,
        Img {
            usage: crate::spirv::ir::ImageUsage::Storage,
            format: crate::spv::ImageFormat::R32ui,
            ..SAMPLED_2D
        },
    )
}

/// An `r32ui` 2D storage image in texture slot `index`.
fn r32ui_image(b: &mut ModuleBuilder, index: u32) -> GlobalId {
    let ty = r32ui(b);
    families_image::texture(b, index, ty)
}

/// A storage image a graphics stage only reads is `NonWritable`; one written by `OpImageWrite` or
/// by an atomic through `OpImageTexelPointer` is not. A kernel is left as it was.
#[test]
fn read_only_storage_images_of_graphics_stages_are_non_writable() {
    use crate::spirv::ir::{AtomicOp, ImageOperands, Scope, Semantics};
    use families_image::{ivec, load_global};
    for model in [ExecModel::Vertex, ExecModel::Fragment, ExecModel::GLCompute] {
        let build = || {
            let mut b = ModuleBuilder::new();
            let u32_ = int(&mut b, 32, false);
            let read = r32ui_image(&mut b, 0);
            let written = r32ui_image(&mut b, 1);
            let atomic = r32ui_image(&mut b, 2);
            let main = compute(&mut b, |f| {
                let c2 = ivec(f, 2, 0);
                let none = ImageOperands::default();
                let img = load_global(f, read);
                let v = f.read(img, c2, &none, TexelMemory::PRIVATE).expect("read");
                let img = load_global(f, written);
                f.write(img, c2, v, &none, TexelMemory::PRIVATE)
                    .expect("write");
                let i32_ = int(f.module(), 32, true);
                let zero = cval(f, i32_, 0);
                let one = cval(f, u32_, 1);
                let p = f.module().global_value(atomic).expect("image ptr");
                let tp = f.image_texel_pointer(p, c2, zero).expect("texel pointer");
                f.atomic(AtomicOp::IAdd, tp, Scope::Device, Semantics::RELAXED, one)
                    .expect("image atomic");
            });
            b.entry(model, main, modes_for(model)).expect("entry");
            b.finalize().expect("finalize")
        };
        assert_spirv_valid(build());
        let r = serialize(build()).expect("serialize");
        let want = match model {
            ExecModel::GLCompute => vec![],
            _ => vec![slot(ResourceKind::Texture, 0)],
        };
        assert_eq!(non_writable_resources(&r), want, "{model:?}");
    }
}

/// The conservative rule: a write through a value that does not trace to one variable (here a
/// function parameter: an image value written by `OpImageWrite`, or an image pointer used by a
/// texel-pointer atomic) counts as a write to every variable of its kind, so no storage image is
/// `NonWritable`, including one the module only reads; a storage buffer is unaffected. (The
/// builder refuses untraceable logical `StorageBuffer` pointers: parameters, phis, and selects
/// of them need variable pointers; so the buffer kind's fallback has no builder-made witness.)
#[test]
fn an_untraced_write_keeps_every_resource_of_its_kind_writable() {
    use crate::spirv::ir::{AtomicOp, ImageOperands, Scope, Semantics};
    use families_image::{ivec, load_global};
    for through_pointer in [false, true] {
        let build = || {
            let mut b = ModuleBuilder::new();
            let void = b.ty(Type::Void).expect("void");
            let u32_ = int(&mut b, 32, false);
            let uvec4 = vec(&mut b, u32_, 4);
            let buffer = ssbo(&mut b, 0, u32_, 4);
            let images = [r32ui_image(&mut b, 1), r32ui_image(&mut b, 2)];
            let image_ty = r32ui(&mut b);
            let image_ptr = b
                .ty(Type::Pointer {
                    class: StorageClass::UniformConstant,
                    pointee: image_ty,
                })
                .expect("image pointer");
            // A callee writing the image its parameter names.
            let callee = {
                let mut f = b.function(void).expect("callee");
                let c2 = ivec(&mut f, 2, 0);
                let one = cval(&mut f, u32_, 1);
                if through_pointer {
                    let p = f.param(image_ptr).expect("image pointer param");
                    let i32_ = int(f.module(), 32, true);
                    let zero = cval(&mut f, i32_, 0);
                    let tp = f.image_texel_pointer(p, c2, zero).expect("texel pointer");
                    f.atomic(AtomicOp::IAdd, tp, Scope::Device, Semantics::RELAXED, one)
                        .expect("image atomic");
                } else {
                    let img = f.param(image_ty).expect("image param");
                    let texel = f
                        .composite_construct(uvec4, &[one, one, one, one])
                        .expect("texel");
                    f.write(
                        img,
                        c2,
                        texel,
                        &ImageOperands::default(),
                        TexelMemory::PRIVATE,
                    )
                    .expect("write");
                }
                f.ret(None).expect("ret");
                f.finish().expect("finish")
            };
            let main = compute(&mut b, |f| {
                let zero = cval(f, u32_, 0);
                let p = elem_ptr(f, buffer, zero);
                f.load(p, MemAccess::default()).expect("load");
                let c2 = ivec(f, 2, 0);
                for g in images {
                    let img = load_global(f, g);
                    f.read(img, c2, &ImageOperands::default(), TexelMemory::PRIVATE)
                        .expect("read");
                }
                // Only the second image reaches the writing callee.
                let arg = if through_pointer {
                    f.module().global_value(images[1]).expect("image ptr")
                } else {
                    load_global(f, images[1])
                };
                f.call(callee, &[arg]).expect("call");
            });
            b.entry(ExecModel::Vertex, main, vec![]).expect("entry");
            b.finalize().expect("finalize")
        };
        assert_spirv_valid(build());
        let r = serialize(build()).expect("serialize");
        assert_eq!(
            non_writable_resources(&r),
            vec![slot(ResourceKind::Buffer, 0)],
            "through_pointer: {through_pointer}"
        );
    }
}
