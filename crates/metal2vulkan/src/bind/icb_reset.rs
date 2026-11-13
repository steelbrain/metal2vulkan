//! A run of indirect commands reset by a compute dispatch that writes only the records whose
//! written flag is set (`indirect_commands::written_flag_offset`), clearing each flag with its
//! record.
//!
//! A reset of a run written as one fill stores every byte of every record of the run, written or
//! not. Where the records lie in host memory across a discrete GPU's link, that is the cost of
//! the whole run's bytes over the link however few commands were written since the last reset.
//! The dispatch reads one flag a command and stores only the flagged records. Both leave every
//! record of the run holding the reset record, all zero (`CommandRecord::default`'s bytes), so
//! long as every writer of the records sets the flag of each record it writes: a clear flag is a
//! record that holds the reset record already.
//!
//! **The module.** One `GLCompute` entry `main` with local size ([`IcbResetModule::LOCAL_SIZE`],
//! 1, 1), [`IcbResetModule::PUSH_BYTES`] of `COMPUTE` push constants at offset 0 (the run's first
//! places, [`IcbResetModule::firsts`]), and in set 0:
//!
//! | Binding | Resource |
//! |---|---|
//! | [`IcbResetModule::RECORDS_BINDING`] | a storage buffer over the run's records, a whole number of 8-byte pairs of words (records lie 8-byte aligned) |
//! | [`IcbResetModule::FLAGS_BINDING`] | a storage buffer over the run's written flags, one 4-byte word a command |
//!
//! Workgroup `c` (along x) is command `c` of the run: it reads flag word `first_flag + c`, and
//! where that is not 0 its invocations store 0 over the record's [`IcbResetModule::RECORD_PAIRS`]
//! pairs from pair `first_record + c · RECORD_PAIRS` (invocation `i` the pairs `i`,
//! `i + LOCAL_SIZE`, and on, so a step's stores are consecutive pairs, 8 bytes a lane), and
//! invocation 0 stores 0 over the flag, after a workgroup barrier that follows every invocation's
//! read of it. A record whose flag is 0 is not written, nor is any byte past the run's records and
//! flags. A dispatch takes one workgroup a command ([`IcbResetModule::groups`]).
//!
//! **Cost.** Where every command of the run was written, the dispatch stores the bytes a fill of
//! the run would, 8 bytes a lane (the widest store every record's alignment takes), and reads a
//! flag a command besides, so it can take longer than a fill: the runtime's measurement is in
//! `blit_indirect.rs`. Where few were written, it stores those alone.
//!
//! **Synchronization** (consumer): the records' and the flags' earlier writes →
//! `SHADER_STORAGE_READ | SHADER_STORAGE_WRITE` in `COMPUTE_SHADER` before the dispatch; its
//! writes → their next use.

use std::sync::Arc;

use crate::budget::{Budget, Meter};
use crate::device::requirements::Requirements;
use crate::device::Device;
use crate::error::Error;
use crate::indirect_commands::COMMAND_BYTES;
use crate::pipeline::sym::{PushField, ResourceKind, ResourceSlot, Stage, Symbol};
use crate::spirv::builder::constructs::{ArmEnd, ArmFn};
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
use crate::spirv::error::BResult;
use crate::spirv::ids::{GlobalId, TypeId, Value};
use crate::spirv::ir::{
    natural_layout, BinOp, Constant, ExecMode, ExecModel, IoDir, LayoutRule, MemAccess,
    MemoryOrder, Scope, Semantics, StorageClass, Type, TypedModule,
};

/// The crate-owned flagged-reset module on one device, validated like any output.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IcbResetModule {
    /// SPIR-V 1.5 words, entry point `main`, bindings numbered as below in set 0.
    pub spirv: Arc<[u32]>,
    /// What the module needs from the device (within every Vulkan 1.2 device).
    pub requirements: Requirements,
}

impl IcbResetModule {
    /// Descriptor set of every binding.
    pub const SET: u32 = 0;
    /// The run's records, a storage buffer the dispatch writes the flagged ones of.
    pub const RECORDS_BINDING: u32 = 0;
    /// The run's written flags, a storage buffer the dispatch reads and clears.
    pub const FLAGS_BINDING: u32 = 1;
    /// Invocations of one workgroup, along x: one command's.
    pub const LOCAL_SIZE: u32 = 64;
    /// 8-byte pairs of words of one record.
    pub const RECORD_PAIRS: u32 = (COMMAND_BYTES / 8) as u32;
    /// Bytes of the `COMPUTE` push-constant range at offset 0: the run's first places.
    pub const PUSH_BYTES: u32 = 8;

    /// The push bytes of a run whose first record starts at pair `first_record` of the records
    /// binding and whose first flag is word `first_flag` of the flags binding (the module docs).
    pub fn firsts(first_record: u32, first_flag: u32) -> [u8; 8] {
        let mut bytes = [0; 8];
        bytes[..4].copy_from_slice(&first_record.to_le_bytes());
        bytes[4..].copy_from_slice(&first_flag.to_le_bytes());
        bytes
    }

    /// The workgroups of a dispatch over a run of `commands` commands: one a command.
    pub fn groups(commands: u32) -> [u32; 3] {
        [commands, 1, 1]
    }
}

/// The binding symbol of the records in the typed module.
const RECORDS: ResourceSlot = ResourceSlot {
    stage: Stage::Kernel,
    kind: ResourceKind::Buffer,
    index: 0,
};
/// The binding symbol of the flags in the typed module.
const FLAGS: ResourceSlot = ResourceSlot {
    stage: Stage::Kernel,
    kind: ResourceKind::Buffer,
    index: 1,
};

/// The push field the first words are declared under: a crate-owned module's own words, which no
/// selection places.
const FIRSTS: PushField = PushField::DispatchRegion;

/// A `Block { elem items[]; }` storage buffer bound at `sym`, `stride` bytes an item.
fn items_buffer(
    b: &mut ModuleBuilder,
    sym: ResourceSlot,
    elem: TypeId,
    stride: u32,
) -> BResult<GlobalId> {
    let rta = b.ty(Type::RuntimeArray {
        elem,
        stride: Some(stride),
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

/// The flagged-reset module as a typed module, before serialization: the module
/// [`icb_reset_module`] serializes, which the CPU interpreter runs. Its bindings are the symbols
/// `RECORDS` and `FLAGS`, and its push field the first places, a `uvec2` (the module docs).
fn typed_module() -> BResult<TypedModule> {
    let mut b = ModuleBuilder::new();
    let u32_ = b.ty(Type::Int {
        width: 32,
        signed: false,
    })?;
    let uvec2 = b.ty(Type::Vector { elem: u32_, len: 2 })?;
    let uvec3 = b.ty(Type::Vector { elem: u32_, len: 3 })?;
    let m_firsts = b.push_member(FIRSTS, uvec2, 0)?;
    let push = b.push_global()?;
    let records = items_buffer(&mut b, RECORDS, uvec2, 8)?;
    let flags = items_buffer(&mut b, FLAGS, u32_, 4)?;
    let null_pair = b.constant(Constant::Null { ty: uvec2 })?;
    let group = b.builtin(crate::spv::BuiltIn::WorkgroupId, IoDir::Input, uvec3)?;
    let local = b.builtin(crate::spv::BuiltIn::LocalInvocationId, IoDir::Input, uvec3)?;
    let void = b.ty(Type::Void)?;
    let mut f = b.function(void)?;
    let zero = u32_value(&mut f, 0)?;
    let zero_pair = f.module().const_value(null_pair)?;
    let group = f.module().global_value(group)?;
    let group = f.load(group, MemAccess::default())?;
    let command = f.composite_extract(group, &[0])?;
    let local = f.module().global_value(local)?;
    let local = f.load(local, MemAccess::default())?;
    let lane = f.composite_extract(local, &[0])?;
    let records = f.module().global_value(records)?;
    let flags = f.module().global_value(flags)?;
    let push = f.module().global_value(push)?;
    let member = u32_value(&mut f, m_firsts.index)?;
    let firsts = f.access_chain(push, &[member])?;
    let firsts = f.load(firsts, MemAccess::default())?;
    let first_record = f.composite_extract(firsts, &[0])?;
    let first_flag = f.composite_extract(firsts, &[1])?;
    let at = f.binary(BinOp::IAdd, first_flag, command)?;
    let flag = f.access_chain(flags, &[zero, at])?;
    let written = f.load(flag, MemAccess::default())?;
    // Every invocation's read of the flag before invocation 0's clear of it.
    let sem = Semantics {
        order: MemoryOrder::AcquireRelease,
        uniform_memory: true,
        ..Semantics::RELAXED
    };
    f.control_barrier(Scope::Workgroup, Scope::Workgroup, sem)?;
    let written = f.binary(BinOp::INotEqual, written, zero)?;
    let reset: ArmFn<'_, ()> = Box::new(move |f, _| {
        let pairs = u32_value(f, IcbResetModule::RECORD_PAIRS)?;
        let base = f.binary(BinOp::IMul, command, pairs)?;
        let base = f.binary(BinOp::IAdd, first_record, base)?;
        for step in 0..IcbResetModule::RECORD_PAIRS.div_ceil(IcbResetModule::LOCAL_SIZE) {
            let offset = step * IcbResetModule::LOCAL_SIZE;
            let pair = match offset {
                0 => lane,
                _ => {
                    let offset = u32_value(f, offset)?;
                    f.binary(BinOp::IAdd, lane, offset)?
                }
            };
            let to = f.binary(BinOp::IAdd, base, pair)?;
            let to = f.access_chain(records, &[zero, to])?;
            if offset + IcbResetModule::LOCAL_SIZE <= IcbResetModule::RECORD_PAIRS {
                f.store(to, zero_pair, MemAccess::default())?;
            } else {
                // The last step's lanes past the record's pairs store nothing.
                let inside = f.binary(BinOp::ULessThan, pair, pairs)?;
                let store: ArmFn<'_, ()> = Box::new(move |f, _| {
                    f.store(to, zero_pair, MemAccess::default())?;
                    Ok(ArmEnd::FallThrough(Vec::new()))
                });
                f.if_(&mut (), inside, store, None)?;
            }
        }
        let first = f.binary(BinOp::IEqual, lane, zero)?;
        let clear: ArmFn<'_, ()> = Box::new(move |f, _| {
            f.store(flag, zero, MemAccess::default())?;
            Ok(ArmEnd::FallThrough(Vec::new()))
        });
        f.if_(&mut (), first, clear, None)?;
        Ok(ArmEnd::FallThrough(Vec::new()))
    });
    f.if_(&mut (), written, reset, None)?;
    f.ret(None)?;
    let id = f.finish()?;
    b.entry(
        ExecModel::GLCompute,
        id,
        vec![ExecMode::LocalSize {
            x: IcbResetModule::LOCAL_SIZE,
            y: 1,
            z: 1,
        }],
    )?;
    b.finalize()
}

/// The fixed binding numbers of the module's symbols.
fn binding_of(sym: &Symbol) -> Option<u32> {
    match sym {
        Symbol::DescriptorSet(_) => Some(IcbResetModule::SET),
        Symbol::Binding(s) if *s == RECORDS => Some(IcbResetModule::RECORDS_BINDING),
        Symbol::Binding(s) if *s == FLAGS => Some(IcbResetModule::FLAGS_BINDING),
        _ => None,
    }
}

/// The flagged-reset module of `device`: built with the typed builder, checked by the owned
/// verifier against the device's profile (requirements included), serialized, and numbered with
/// the fixed bindings of [`IcbResetModule`].
pub fn icb_reset_module(device: &Device) -> Result<IcbResetModule, Error> {
    let profile = device.profile();
    let m = typed_module()?;
    let mut meter = Meter::new(&Budget::default());
    crate::spirv::verify::verify(&m, profile, &mut meter)?;
    let r = crate::spirv::serialize::serialize(m)?;
    profile.check(&r.requirements).map_err(|c| {
        Error::Internal(crate::error::InternalError::new(
            crate::error::InternalKind::RequirementsExceedProfile,
            format!("the flagged reset needs {c:?}"),
        ))
    })?;
    let words = crate::spirv::serialize::relocate(&r, &binding_of)?;
    Ok(IcbResetModule {
        spirv: words.into(),
        requirements: r.requirements,
    })
}

#[cfg(test)]
mod tests {
    use super::{typed_module, IcbResetModule, FLAGS, RECORDS};
    use crate::spirv::interp::{run, BarrierRule, Buffer, Config, Order, Resources};
    use crate::spirv::testing::{assert_valid, committed_profiles};

    /// Words of one record.
    const WORDS: usize = 2 * IcbResetModule::RECORD_PAIRS as usize;

    #[test]
    fn the_flagged_reset_kernel_is_valid_on_every_committed_profile() {
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

    /// The records' and the flags' words after one dispatch over `commands` commands, the run's
    /// first record at pair `firsts[0]` of `records` and its first flag at word `firsts[1]` of
    /// `flags`, in both invocation orders under Vulkan's barrier rule.
    fn reset(
        records: &[u32],
        flags: &[u32],
        firsts: [u32; 2],
        commands: u32,
    ) -> (Vec<u32>, Vec<u32>) {
        let m = typed_module().expect("module");
        let mut out = Vec::new();
        for order in [Order::Forward, Order::Reverse] {
            let res = Resources {
                buffers: vec![
                    Buffer {
                        slot: Some(RECORDS),
                        address: None,
                        bytes: bytes(records),
                    },
                    Buffer {
                        slot: Some(FLAGS),
                        address: None,
                        bytes: bytes(flags),
                    },
                ],
                push: IcbResetModule::firsts(firsts[0], firsts[1]).to_vec(),
                spec: Default::default(),
            };
            let cfg = Config {
                order,
                workgroups: IcbResetModule::groups(commands),
                barriers: BarrierRule::Vulkan,
                ..Config::default()
            };
            let outcome = run(&m, &cfg, res).expect("run");
            out.push((words(&outcome.buffers[0]), words(&outcome.buffers[1])));
        }
        assert_eq!(out[0], out[1], "order independent");
        out.swap_remove(0)
    }

    /// Each flagged record of the run becomes zero and its flag clear; a record whose flag is
    /// clear, the words before the run's first record and past its last, and the flags outside
    /// the run keep their words, the run's first places odd in both bindings.
    #[test]
    fn only_flagged_records_are_reset_and_their_flags_cleared() {
        let (commands, firsts) = (5u32, [3u32, 2u32]);
        let records: Vec<u32> = (0..2 * firsts[0] as usize + commands as usize * WORDS + 7)
            .map(|i| 0x5a00_0000 | i as u32)
            .collect();
        let mut flags = vec![0xeeee_eeeeu32; firsts[1] as usize + commands as usize + 2];
        let set = [1u32, 0, 7, 0, 1];
        for (c, &value) in set.iter().enumerate() {
            flags[firsts[1] as usize + c] = value;
        }
        let (got_records, got_flags) = reset(&records, &flags, firsts, commands);
        let mut want_records = records.clone();
        let mut want_flags = flags.clone();
        for (c, &value) in set.iter().enumerate() {
            if value != 0 {
                let start = 2 * firsts[0] as usize + c * WORDS;
                want_records[start..start + WORDS].fill(0);
                want_flags[firsts[1] as usize + c] = 0;
            }
        }
        assert_eq!(got_records, want_records);
        assert_eq!(got_flags, want_flags);
    }

    /// A run of no flagged command writes nothing.
    #[test]
    fn a_run_of_clear_flags_writes_nothing() {
        let records: Vec<u32> = (0..3 * WORDS as u32).collect();
        let flags = vec![0u32; 3];
        assert_eq!(reset(&records, &flags, [0, 0], 3), (records, flags));
    }
}
