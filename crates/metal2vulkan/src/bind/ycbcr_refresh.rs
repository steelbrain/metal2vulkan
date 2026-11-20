//! A two-plane YCbCr texture's planes compared with the copy its last conversion read, and the
//! conversion dispatched only where they differ.
//!
//! A YCbCr texture's image holds the texels Metal reads from its surface's planes
//! ([`ycbcr_convert`](super::ycbcr_convert)), converted from a copy of the planes' bytes the
//! consumer keeps. The surface's bytes may change unseen between two uses, so each use converts
//! again; one dispatch of [`ycbcr_refresh_module`] first compares the planes' current words with
//! that copy: each word that differs it stores over the copy, and it then writes the conversion's
//! group count into the state's indirect arguments. The conversion dispatched after it through
//! those arguments (`vkCmdDispatchIndirect`) runs every workgroup where any word changed, and none
//! where none did, so a texture whose planes are as its image was converted from converts nothing.
//! A copy no conversion completed from (the state's [`COMPLETED`](YCbCrRefreshModule::COMPLETED)
//! word 0) takes the group count too. After the dispatch the copy holds the planes' bytes, word for
//! word, so the conversion reads what it reads without it.
//!
//! **The module.** One `GLCompute` entry `main` with local size ([`YCbCrRefreshModule::LOCAL_SIZE`],
//! 1, 1), [`YCbCrRefreshModule::PUSH_BYTES`] of `COMPUTE` push constants at offset 0 (the shape,
//! [`YCbCrRefreshModule::shape`]), and in set 0:
//!
//! | Binding | Resource |
//! |---|---|
//! | [`YCbCrRefreshModule::PLANES_BINDING`] | a storage buffer over the planes' bytes, read |
//! | [`YCbCrRefreshModule::COPY_BINDING`] | a storage buffer over the copy's bytes, read and written |
//! | [`YCbCrRefreshModule::STATE_BINDING`] | the state's words, a storage buffer: the conversion's `VkDispatchIndirectCommand` at word 0, then [`COMPLETED`](YCbCrRefreshModule::COMPLETED) |
//!
//! The shape's words, in order: `firsts`, the planes binding's word of the planes' first word in
//! its low 16 bits and the copy binding's in its high 16; `words`, the planes' words; `row`, the
//! words one row of workgroups covers (a multiple of [`LOCAL_SIZE`](YCbCrRefreshModule::LOCAL_SIZE));
//! `groups`, the conversion's workgroups along x. Invocation `(x, y)` with `w = y · row + x` below
//! `words` compares planes word `first_planes + w` with copy word `first_copy + w`; where they
//! differ it stores the planes' word over the copy's and `groups` into state word 0 (an atomic
//! store: every such invocation stores the same value). Invocation `(0, 0)` stores `groups` there
//! too where the state's [`COMPLETED`](YCbCrRefreshModule::COMPLETED) word is 0. Nothing else is
//! written: no planes word, and no state word but word 0. A dispatch takes `row / LOCAL_SIZE` by
//! `⌈words / row⌉` workgroups ([`YCbCrRefreshModule::groups`]).
//!
//! **Synchronization** (consumer): before the dispatch, the planes' earlier writes →
//! `SHADER_STORAGE_READ`, and the copy's and the state's earlier accesses (the state's words set,
//! word 0 to 0) → `SHADER_STORAGE_READ | SHADER_STORAGE_WRITE`, in `COMPUTE_SHADER`; after it, its
//! copy writes → the conversion's `SHADER_STORAGE_READ` in `COMPUTE_SHADER`, and its state write →
//! `INDIRECT_COMMAND_READ` in `DRAW_INDIRECT`.

use std::sync::Arc;

use crate::budget::{Budget, Meter};
use crate::device::requirements::Requirements;
use crate::device::Device;
use crate::error::Error;
use crate::pipeline::sym::{PushField, ResourceKind, ResourceSlot, Stage, Symbol};
use crate::spirv::builder::constructs::{ArmEnd, ArmFn};
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
use crate::spirv::error::BResult;
use crate::spirv::ids::{GlobalId, TypeId, Value};
use crate::spirv::ir::{
    natural_layout, BinOp, ExecMode, ExecModel, IoDir, LayoutRule, MemAccess, Scope, Semantics,
    StorageClass, Type, TypedModule,
};

/// The crate-owned YCbCr refresh module on one device, validated like any output.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct YCbCrRefreshModule {
    /// SPIR-V 1.5 words, entry point `main`, bindings numbered as below in set 0.
    pub spirv: Arc<[u32]>,
    /// What the module needs from the device (within every Vulkan 1.2 device).
    pub requirements: Requirements,
}

impl YCbCrRefreshModule {
    /// Descriptor set of every binding.
    pub const SET: u32 = 0;
    /// The planes' bytes, a storage buffer the dispatch reads.
    pub const PLANES_BINDING: u32 = 0;
    /// The copy's bytes, a storage buffer the dispatch reads and writes the changed words of.
    pub const COPY_BINDING: u32 = 1;
    /// The state's words, a storage buffer.
    pub const STATE_BINDING: u32 = 2;
    /// The state's word that says a conversion completed from the copy: 0 where none did, which
    /// takes the conversion's group count whatever the words.
    pub const COMPLETED: u32 = 3;
    /// Invocations of one workgroup, along x: one word each.
    pub const LOCAL_SIZE: u32 = 64;
    /// Bytes of the `COMPUTE` push-constant range at offset 0: the shape.
    pub const PUSH_BYTES: u32 = 16;
    /// The words one row of workgroups covers: 1024 workgroups across.
    pub const ROW_WORDS: u32 = Self::LOCAL_SIZE * 1024;

    /// The push bytes of `words` planes words, the first at word `first_planes` of the planes
    /// binding and `first_copy` of the copy's (each at most `0xffff`), for a conversion of
    /// `groups` workgroups across
    /// (the module docs).
    pub fn shape(firsts: [u32; 2], words: u32, groups: u32) -> [u8; 16] {
        shape_of(firsts, words, Self::ROW_WORDS, groups)
    }

    /// The workgroups of a dispatch over `words` words.
    pub fn groups(words: u32) -> [u32; 2] {
        [
            Self::ROW_WORDS / Self::LOCAL_SIZE,
            words.div_ceil(Self::ROW_WORDS).max(1),
        ]
    }
}

/// The shape's bytes (the module docs), its rows `row` words apart.
fn shape_of([first_planes, first_copy]: [u32; 2], words: u32, row: u32, groups: u32) -> [u8; 16] {
    let firsts = (first_planes & 0xffff) | (first_copy << 16);
    let words = [firsts, words, row, groups];
    let mut bytes = [0; 16];
    for (chunk, word) in bytes.as_chunks_mut::<4>().0.iter_mut().zip(words) {
        chunk.copy_from_slice(&word.to_le_bytes());
    }
    bytes
}

/// The binding symbol of the planes' bytes in the typed module.
const PLANES: ResourceSlot = ResourceSlot {
    stage: Stage::Kernel,
    kind: ResourceKind::Buffer,
    index: 0,
};
/// The binding symbol of the copy's bytes in the typed module.
const COPY: ResourceSlot = ResourceSlot {
    stage: Stage::Kernel,
    kind: ResourceKind::Buffer,
    index: 1,
};
/// The binding symbol of the state's words in the typed module.
const STATE: ResourceSlot = ResourceSlot {
    stage: Stage::Kernel,
    kind: ResourceKind::Buffer,
    index: 2,
};

/// The push field the shape is declared under: a crate-owned module's own words, which no
/// selection places.
const SHAPE: PushField = PushField::DispatchRegion;

/// A `Block { u32 words[]; }` storage buffer bound at `sym`.
fn words_buffer(b: &mut ModuleBuilder, sym: ResourceSlot, u32_: TypeId) -> BResult<GlobalId> {
    let rta = b.ty(Type::RuntimeArray {
        elem: u32_,
        stride: Some(4),
    })?;
    let layout = natural_layout(b.types(), LayoutRule::Std430, true, &[rta])?;
    let block = b.ty(Type::Struct {
        members: [rta].into(),
        layout: Some(layout),
    })?;
    b.resource(sym, StorageClass::StorageBuffer, block)
}

fn u32_value(f: &mut FunctionBuilder<'_>, v: u32) -> BResult<Value> {
    let c = f.u32_const(v)?;
    f.module().const_value(c)
}

/// The refresh module as a typed module, before serialization: the module
/// [`ycbcr_refresh_module`] serializes, which the CPU interpreter runs. Its bindings are the
/// symbols `PLANES`, `COPY` and `STATE`, and its push field the shape, a `uvec4` (the module
/// docs).
fn typed_module() -> BResult<TypedModule> {
    let mut b = ModuleBuilder::new();
    let u32_ = b.ty(Type::Int {
        width: 32,
        signed: false,
    })?;
    let uvec3 = b.ty(Type::Vector { elem: u32_, len: 3 })?;
    let uvec4 = b.ty(Type::Vector { elem: u32_, len: 4 })?;
    let m_shape = b.push_member(SHAPE, uvec4, 0)?;
    let push = b.push_global()?;
    let planes = words_buffer(&mut b, PLANES, u32_)?;
    let copy = words_buffer(&mut b, COPY, u32_)?;
    let state = words_buffer(&mut b, STATE, u32_)?;
    let invocation = b.builtin(crate::spv::BuiltIn::GlobalInvocationId, IoDir::Input, uvec3)?;
    let void = b.ty(Type::Void)?;
    let mut f = b.function(void)?;
    let zero = u32_value(&mut f, 0)?;
    let low = u32_value(&mut f, 0xffff)?;
    let sixteen = u32_value(&mut f, 16)?;
    let completed = u32_value(&mut f, YCbCrRefreshModule::COMPLETED)?;
    let invocation = f.module().global_value(invocation)?;
    let invocation = f.load(invocation, MemAccess::default())?;
    let x = f.composite_extract(invocation, &[0])?;
    let y = f.composite_extract(invocation, &[1])?;
    let planes = f.module().global_value(planes)?;
    let copy = f.module().global_value(copy)?;
    let state = f.module().global_value(state)?;
    let push = f.module().global_value(push)?;
    let member = u32_value(&mut f, m_shape.index)?;
    let shape = f.access_chain(push, &[member])?;
    let shape = f.load(shape, MemAccess::default())?;
    let firsts = f.composite_extract(shape, &[0])?;
    let words = f.composite_extract(shape, &[1])?;
    let row = f.composite_extract(shape, &[2])?;
    let groups = f.composite_extract(shape, &[3])?;
    let arguments = f.access_chain(state, &[zero, zero])?;
    let w = f.binary(BinOp::IMul, y, row)?;
    let w = f.binary(BinOp::IAdd, w, x)?;
    let inside = f.binary(BinOp::ULessThan, w, words)?;
    let compare: ArmFn<'_, ()> = Box::new(move |f, _| {
        let first_planes = f.binary(BinOp::BitwiseAnd, firsts, low)?;
        let first_copy = f.binary(BinOp::ShiftRightLogical, firsts, sixteen)?;
        let from = f.binary(BinOp::IAdd, first_planes, w)?;
        let to = f.binary(BinOp::IAdd, first_copy, w)?;
        let from = f.access_chain(planes, &[zero, from])?;
        let word = f.load(from, MemAccess::default())?;
        let to = f.access_chain(copy, &[zero, to])?;
        let kept = f.load(to, MemAccess::default())?;
        let changed = f.binary(BinOp::INotEqual, word, kept)?;
        let store: ArmFn<'_, ()> = Box::new(move |f, _| {
            f.store(to, word, MemAccess::default())?;
            f.atomic_store(arguments, Scope::Device, Semantics::RELAXED, groups)?;
            Ok(ArmEnd::FallThrough(Vec::new()))
        });
        f.if_(&mut (), changed, store, None)?;
        Ok(ArmEnd::FallThrough(Vec::new()))
    });
    f.if_(&mut (), inside, compare, None)?;
    // The first invocation: a copy no conversion completed from converts whatever the words.
    let first = f.binary(BinOp::IEqual, w, zero)?;
    let unfinished: ArmFn<'_, ()> = Box::new(move |f, _| {
        let done = f.access_chain(state, &[zero, completed])?;
        let done = f.load(done, MemAccess::default())?;
        let never = f.binary(BinOp::IEqual, done, zero)?;
        let store: ArmFn<'_, ()> = Box::new(move |f, _| {
            f.atomic_store(arguments, Scope::Device, Semantics::RELAXED, groups)?;
            Ok(ArmEnd::FallThrough(Vec::new()))
        });
        f.if_(&mut (), never, store, None)?;
        Ok(ArmEnd::FallThrough(Vec::new()))
    });
    f.if_(&mut (), first, unfinished, None)?;
    f.ret(None)?;
    let id = f.finish()?;
    b.entry(
        ExecModel::GLCompute,
        id,
        vec![ExecMode::LocalSize {
            x: YCbCrRefreshModule::LOCAL_SIZE,
            y: 1,
            z: 1,
        }],
    )?;
    b.finalize()
}

/// The fixed binding numbers of the module's symbols.
fn binding_of(sym: &Symbol) -> Option<u32> {
    match sym {
        Symbol::DescriptorSet(_) => Some(YCbCrRefreshModule::SET),
        Symbol::Binding(s) if *s == PLANES => Some(YCbCrRefreshModule::PLANES_BINDING),
        Symbol::Binding(s) if *s == COPY => Some(YCbCrRefreshModule::COPY_BINDING),
        Symbol::Binding(s) if *s == STATE => Some(YCbCrRefreshModule::STATE_BINDING),
        _ => None,
    }
}

/// The YCbCr refresh module of `device`: built with the typed builder, checked by the owned
/// verifier against the device's profile (requirements included), serialized, and numbered with
/// the fixed bindings of [`YCbCrRefreshModule`].
pub fn ycbcr_refresh_module(device: &Device) -> Result<YCbCrRefreshModule, Error> {
    let profile = device.profile();
    let m = typed_module()?;
    let mut meter = Meter::new(&Budget::default());
    crate::spirv::verify::verify(&m, profile, &mut meter)?;
    let r = crate::spirv::serialize::serialize(m)?;
    profile.check(&r.requirements).map_err(|c| {
        Error::Internal(crate::error::InternalError::new(
            crate::error::InternalKind::RequirementsExceedProfile,
            format!("the YCbCr refresh needs {c:?}"),
        ))
    })?;
    let words = crate::spirv::serialize::relocate(&r, &binding_of)?;
    Ok(YCbCrRefreshModule {
        spirv: words.into(),
        requirements: r.requirements,
    })
}

#[cfg(test)]
mod tests {
    use super::{shape_of, typed_module, YCbCrRefreshModule, COPY, PLANES, STATE};
    use crate::spirv::interp::{run, Buffer, Config, Order, Resources};
    use crate::spirv::testing::{assert_valid, committed_profiles};

    #[test]
    fn the_ycbcr_refresh_kernel_is_valid_on_every_committed_profile() {
        for (_, p) in committed_profiles().expect("profiles") {
            assert_valid(typed_module().expect("module"), &p);
        }
    }

    fn bytes(words: &[u32]) -> Vec<u8> {
        words.iter().flat_map(|w| w.to_le_bytes()).collect()
    }

    fn words(bytes: &[u8]) -> Vec<u32> {
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| u32::from_le_bytes(*b))
            .collect()
    }

    /// The rows of the test dispatches: one workgroup across.
    const ROW: u32 = YCbCrRefreshModule::LOCAL_SIZE;

    /// The copy's and the state's words after one dispatch over `count` planes words, the first at
    /// word `firsts[0]` of `planes` and `firsts[1]` of `copy`, for a conversion of `groups`
    /// workgroups across, from `state`; in both invocation orders, which agree.
    fn refreshed(
        planes: &[u32],
        copy: &[u32],
        state: [u32; 4],
        firsts: [u32; 2],
        count: u32,
        groups: u32,
    ) -> (Vec<u32>, Vec<u32>) {
        let m = typed_module().expect("module");
        let mut out = Vec::new();
        for order in [Order::Forward, Order::Reverse] {
            let res = Resources {
                buffers: vec![
                    Buffer {
                        slot: Some(PLANES),
                        address: None,
                        bytes: bytes(planes),
                    },
                    Buffer {
                        slot: Some(COPY),
                        address: None,
                        bytes: bytes(copy),
                    },
                    Buffer {
                        slot: Some(STATE),
                        address: None,
                        bytes: bytes(&state),
                    },
                ],
                push: shape_of(firsts, count, ROW, groups).to_vec(),
                spec: Default::default(),
            };
            let cfg = Config {
                order,
                workgroups: [1, count.div_ceil(ROW).max(1), 1],
                ..Config::default()
            };
            let outcome = run(&m, &cfg, res).expect("run");
            assert_eq!(
                words(&outcome.buffers[0]),
                planes,
                "the planes are only read"
            );
            out.push((words(&outcome.buffers[1]), words(&outcome.buffers[2])));
        }
        assert_eq!(out[0], out[1], "order independent");
        out.swap_remove(0)
    }

    /// The planes' words, `count` of them from word 2, and the copy's, from word 5, apart from
    /// them where `changed` says.
    fn surfaces(count: u32, changed: &[u32]) -> (Vec<u32>, Vec<u32>) {
        let planes: Vec<u32> = (0..count + 4).map(|i| 0x5a00_0000 | i).collect();
        let mut copy: Vec<u32> = (0..count + 9).map(|i| 0xa5a5_0000 | i).collect();
        for i in 0..count {
            copy[(5 + i) as usize] = planes[(2 + i) as usize];
        }
        for &i in changed {
            copy[(5 + i) as usize] ^= 1 << (i % 32);
        }
        (planes, copy)
    }

    /// Planes as the copy holds them, after a completed conversion: nothing is written, and the
    /// conversion's group count stays 0, so its indirect dispatch runs no workgroup.
    #[test]
    fn unchanged_planes_leave_the_conversion_no_workgroup() {
        let count = 3 * ROW + 7;
        let (planes, copy) = surfaces(count, &[]);
        let (after, state) = refreshed(&planes, &copy, [0, 9, 1, 1], [2, 5], count, 31);
        assert_eq!(after, copy, "the copy");
        assert_eq!(state, [0, 9, 1, 1], "the state");
    }

    /// Each changed word, the first, one at a row's end, one past a row and the last, lands in
    /// the copy, and the conversion takes its group count; no other copy word changes, and no
    /// state word but the first.
    #[test]
    fn each_changed_word_lands_in_the_copy_and_takes_the_conversion() {
        let count = 3 * ROW + 7;
        for changed in [
            vec![0],
            vec![ROW - 1],
            vec![ROW],
            vec![count - 1],
            vec![1, ROW + 3, 2 * ROW + 5],
        ] {
            let (planes, copy) = surfaces(count, &changed);
            let (after, state) = refreshed(&planes, &copy, [0, 9, 1, 1], [2, 5], count, 31);
            let mut want = copy.clone();
            for i in 0..count {
                want[(5 + i) as usize] = planes[(2 + i) as usize];
            }
            assert_eq!(after, want, "the copy, {changed:?} changed");
            assert_eq!(state, [31, 9, 1, 1], "the state, {changed:?} changed");
        }
    }

    /// A copy no conversion completed from (its state's completed word 0) takes the conversion's
    /// group count with no word changed.
    #[test]
    fn a_copy_no_conversion_completed_from_takes_the_conversion() {
        let count = ROW + 1;
        let (planes, copy) = surfaces(count, &[]);
        let (after, state) = refreshed(&planes, &copy, [0, 9, 1, 0], [2, 5], count, 31);
        assert_eq!(after, copy);
        assert_eq!(state, [31, 9, 1, 0]);
    }

    /// The first words' two halves: a first word past 255 in either binding is read from its
    /// place; and the shape's public form takes the row of 1024 workgroups.
    #[test]
    fn the_first_words_reach_far_words() {
        let planes: Vec<u32> = (0..400).collect();
        let mut copy = vec![0u32; 700];
        copy[600..602].copy_from_slice(&planes[300..302]);
        copy[601] = 7;
        let (after, state) = refreshed(&planes, &copy, [0, 1, 1, 1], [300, 600], 2, 5);
        let mut want = copy.clone();
        want[601] = 301;
        assert_eq!(after, want);
        assert_eq!(state[0], 5);
        let shape = YCbCrRefreshModule::shape([1, 2], 3, 4);
        assert_eq!(
            words(&shape),
            [1 | 2 << 16, 3, YCbCrRefreshModule::ROW_WORDS, 4]
        );
        assert_eq!(
            YCbCrRefreshModule::groups(YCbCrRefreshModule::ROW_WORDS + 1),
            [1024, 2]
        );
    }
}
