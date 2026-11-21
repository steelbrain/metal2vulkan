//! The bits of a specialization constant a module can read (`spirv::spec_bits`): a routine's bit
//! tests of its constant argument leave the other bits unread, a use the analysis does not
//! recognise keeps every bit, and a bit a specialized note tests stays read.

use crate::budget::{Budget, Meter};
use crate::pipeline::artifact::{SpecClause, SpecTest, SpecializedNote};
use crate::pipeline::interface::StageSpec;
use crate::pipeline::sym::{ResourceKind, ResourceSlot, SlotClass, SlotRef, SpecKey, Stage};
use crate::pipeline::PrecisionNote;
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
use crate::spirv::error::BResult;
use crate::spirv::ids::{GlobalId, Value};
use crate::spirv::ir::{
    natural_layout, BinOp, ConstBits, ExecMode, ExecModel, LayoutRule, MemAccess, StorageClass,
    Type, TypedModule,
};
use crate::spirv::spec_bits::mark_unread;

/// The constant the tests read: a texture slot's bicubic texel class.
const KEY: SpecKey = SpecKey::BicubicTexels(SlotRef {
    stage: Stage::Kernel,
    class: SlotClass::Texture,
    index: 0,
});

fn buffer(b: &mut ModuleBuilder) -> BResult<GlobalId> {
    let u32_ = b.ty(Type::Int {
        width: 32,
        signed: false,
    })?;
    let rta = b.ty(Type::RuntimeArray {
        elem: u32_,
        stride: Some(4),
    })?;
    let layout = natural_layout(b.types(), LayoutRule::Std430, true, &[rta])?;
    let block = b.ty(Type::Struct {
        members: [rta].into(),
        layout: Some(layout),
    })?;
    let slot = ResourceSlot {
        stage: Stage::Kernel,
        kind: ResourceKind::Buffer,
        index: 0,
    };
    b.resource(slot, StorageClass::StorageBuffer, block)
}

fn k(fb: &mut FunctionBuilder<'_>, v: u32) -> BResult<Value> {
    let c = fb.u32_const(v)?;
    fb.module().const_value(c)
}

/// A kernel that stores, to word 0 of buffer 0, what routine `f` computes from `KEY`'s
/// constant, or (`call` false) the constant itself, `f` defined and never called.
fn kernel(
    call: bool,
    f: impl FnOnce(&mut FunctionBuilder<'_>, Value) -> BResult<Value>,
) -> BResult<TypedModule> {
    let mut b = ModuleBuilder::new();
    let buf = buffer(&mut b)?;
    let void = b.ty(Type::Void)?;
    let u32_ = b.ty(Type::Int {
        width: 32,
        signed: false,
    })?;
    let spec = b.spec(KEY, u32_, ConstBits(0))?;
    let flags = b.const_value(spec)?;
    let mut fb = b.function(u32_)?;
    let param = fb.param(u32_)?;
    let r = f(&mut fb, param)?;
    fb.ret(Some(r))?;
    let routine = fb.finish()?;
    let mut fb = b.function(void)?;
    let r = match call {
        true => fb
            .call(routine, &[flags])?
            .expect("the routine returns a value"),
        false => flags,
    };
    let base = fb.module().global_value(buf)?;
    let zero = k(&mut fb, 0)?;
    let p = fb.access_chain(base, &[zero, zero])?;
    fb.store(p, r, MemAccess::default())?;
    fb.ret(None)?;
    let id = fb.finish()?;
    b.entry(
        ExecModel::GLCompute,
        id,
        vec![ExecMode::LocalSize { x: 1, y: 1, z: 1 }],
    )?;
    b.finalize()
}

/// `KEY`'s unread bits in `m`, with `notes`.
fn unread(m: &TypedModule, notes: &[SpecializedNote]) -> u64 {
    let mut spec = [StageSpec::new(KEY, 4, 0)];
    let mut meter = Meter::new(&Budget::standard());
    mark_unread(m, &mut spec, notes, &mut meter).expect("analysis");
    spec[0].unread_bits
}

/// A routine's reads of its constant argument: bits under a mask, and a field
/// shifted down and masked. The analysis follows the constant into the routine's parameter.
fn bit_tests(fb: &mut FunctionBuilder<'_>, flags: Value) -> BResult<Value> {
    let m = k(fb, 0x30D)?;
    let low = fb.binary(BinOp::BitwiseAnd, flags, m)?;
    let s = k(fb, 28)?;
    let field = fb.binary(BinOp::ShiftRightLogical, flags, s)?;
    let f = k(fb, 0xF)?;
    let field = fb.binary(BinOp::BitwiseAnd, field, f)?;
    fb.binary(BinOp::BitwiseOr, low, field)
}

/// A routine that tests bits `0x30D` and the field at bits 28 to 31 of its constant argument leaves
/// every other bit unread, so a binder may clear them.
#[test]
fn a_routines_bit_tests_leave_the_other_bits_unread() {
    let m = kernel(true, bit_tests).expect("module");
    assert_eq!(unread(&m, &[]), 0xFFFF_FFFF & !0xF000_030D);
}

/// A use the analysis does not recognise keeps the whole word: the constant stored as it is, added
/// to, chosen by a select, or shifted by an amount that is not a constant.
#[test]
fn an_unrecognised_use_keeps_every_bit() {
    let stored = kernel(false, bit_tests).expect("module");
    assert_eq!(unread(&stored, &[]), 0, "stored");
    let added = kernel(true, |fb, flags| {
        let one = k(fb, 1)?;
        fb.binary(BinOp::IAdd, flags, one)
    })
    .expect("module");
    assert_eq!(unread(&added, &[]), 0, "added");
    let selected = kernel(true, |fb, flags| {
        let low = bit_tests(fb, flags)?;
        let zero = k(fb, 0)?;
        let cond = fb.binary(BinOp::INotEqual, low, zero)?;
        fb.select(cond, flags, zero)
    })
    .expect("module");
    assert_eq!(unread(&selected, &[]), 0, "selected");
    let shifted = kernel(true, |fb, flags| {
        let by = fb.binary(BinOp::BitwiseAnd, flags, flags)?;
        let field = fb.binary(BinOp::ShiftRightLogical, flags, by)?;
        let f = k(fb, 0xF)?;
        fb.binary(BinOp::BitwiseAnd, field, f)
    })
    .expect("module");
    assert_eq!(unread(&shifted, &[]), 0, "shifted by a value");
}

/// A bit a specialized note tests stays read though no instruction reads it: the runtime announces
/// the note from the value the binder hands the driver.
#[test]
fn a_bit_a_note_tests_stays_read() {
    let m = kernel(true, bit_tests).expect("module");
    let note = SpecializedNote {
        note: PrecisionNote::HalfTexelConversion,
        when: vec![SpecClause {
            all: vec![SpecTest::equal(KEY, 1 << 10, 0)],
        }],
    };
    assert_eq!(unread(&m, &[note]), 0xFFFF_FFFF & !0xF000_030D & !(1 << 10));
}
