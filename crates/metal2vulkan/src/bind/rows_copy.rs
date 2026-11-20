//! Rows of 4-byte words copied from one buffer into another by a compute dispatch.
//!
//! A copy of many equal rows between two buffers whose rows lie apart (a texture over a buffer's
//! padded rows, a staging buffer's tight ones) is a `vkCmdCopyBuffer` of one region per row. A
//! driver may record each region as a separate operation, at a CPU cost that grows with the
//! regions; one dispatch of [`rows_copy_module`] copies every row at once. Both write the same
//! bytes: each row's words, from the source row's place to the destination row's, and no byte
//! between the rows or past them.
//!
//! **The module.** One `GLCompute` entry `main` with local size ([`RowsCopyModule::LOCAL_SIZE`],
//! 1, 1), [`RowsCopyModule::PUSH_BYTES`] of `COMPUTE` push constants at offset 0 (the rows' shape,
//! [`RowsCopyModule::shape`]), and in set 0:
//!
//! | Binding | Resource |
//! |---|---|
//! | [`RowsCopyModule::SOURCE_BINDING`] | a storage buffer over the source's bytes, a whole number of 4-byte words |
//! | [`RowsCopyModule::DESTINATION_BINDING`] | a storage buffer over the destination's bytes, likewise |
//!
//! The shape's words, in order: `firsts`, the source binding's word of the first row's first word
//! in its low 16 bits and the destination binding's in its high 16; `width`, the words a row;
//! `source_row_words` and `destination_row_words`, the rows apart in each binding in words.
//! Invocation `(x, y)` with `x` below `width` stores source word
//! `first_source + y · source_row_words + x` at destination word
//! `first_destination + y · destination_row_words + x`; one at `x` past `width` does nothing. A
//! dispatch takes `ceil(width / LOCAL_SIZE)` by the rows workgroups ([`RowsCopyModule::groups`]).
//! Nothing writes the source binding, and no destination word between the rows or past them is
//! written. Where the two bindings' words overlap, or two destination rows share words, the
//! result is unordered, as a copy's regions that overlap are.
//!
//! **Synchronization** (consumer): the source's earlier writes → `SHADER_STORAGE_READ` in
//! `COMPUTE_SHADER`, and the destination's earlier accesses → `SHADER_STORAGE_WRITE` there, before
//! the dispatch; its writes → the destination's next use.

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
    natural_layout, BinOp, ExecMode, ExecModel, IoDir, LayoutRule, MemAccess, StorageClass, Type,
    TypedModule,
};

/// The crate-owned rows-copy module on one device, validated like any output.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RowsCopyModule {
    /// SPIR-V 1.5 words, entry point `main`, bindings numbered as below in set 0.
    pub spirv: Arc<[u32]>,
    /// What the module needs from the device (within every Vulkan 1.2 device).
    pub requirements: Requirements,
}

impl RowsCopyModule {
    /// Descriptor set of every binding.
    pub const SET: u32 = 0;
    /// The source's bytes, a storage buffer the dispatch reads.
    pub const SOURCE_BINDING: u32 = 0;
    /// The destination's bytes, a storage buffer the dispatch writes the rows' words into.
    pub const DESTINATION_BINDING: u32 = 1;
    /// Invocations of one workgroup, along x: one word each, of one row.
    pub const LOCAL_SIZE: u32 = 64;
    /// Bytes of the `COMPUTE` push-constant range at offset 0: the rows' shape.
    pub const PUSH_BYTES: u32 = 16;

    /// The push bytes of rows `width` words long, the first row's first word at word
    /// `first_source` of the source binding and `first_destination` of the destination's (each
    /// at most `0xffff`), the rows `source_row_words` and
    /// `destination_row_words` apart (the module docs).
    pub fn shape(
        [first_source, first_destination]: [u32; 2],
        width: u32,
        [source_row_words, destination_row_words]: [u32; 2],
    ) -> [u8; 16] {
        let firsts = (first_source & 0xffff) | (first_destination << 16);
        let words = [firsts, width, source_row_words, destination_row_words];
        let mut bytes = [0; 16];
        for (chunk, word) in bytes.as_chunks_mut::<4>().0.iter_mut().zip(words) {
            chunk.copy_from_slice(&word.to_le_bytes());
        }
        bytes
    }

    /// The workgroups of a dispatch over `rows` rows `width` words long.
    pub fn groups(width: u32, rows: u32) -> [u32; 2] {
        [width.div_ceil(Self::LOCAL_SIZE), rows]
    }
}

/// The binding symbol of the source's bytes in the typed module.
const SOURCE: ResourceSlot = ResourceSlot {
    stage: Stage::Kernel,
    kind: ResourceKind::Buffer,
    index: 0,
};
/// The binding symbol of the destination's bytes in the typed module.
const DESTINATION: ResourceSlot = ResourceSlot {
    stage: Stage::Kernel,
    kind: ResourceKind::Buffer,
    index: 1,
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

/// The rows-copy module as a typed module, before serialization: the module
/// [`rows_copy_module`] serializes, which the CPU interpreter runs. Its bindings are the symbols
/// `SOURCE` and `DESTINATION`, and its push field the shape, a `uvec4` (the module docs).
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
    let source = words_buffer(&mut b, SOURCE, u32_)?;
    let destination = words_buffer(&mut b, DESTINATION, u32_)?;
    let invocation = b.builtin(crate::spv::BuiltIn::GlobalInvocationId, IoDir::Input, uvec3)?;
    let void = b.ty(Type::Void)?;
    let mut f = b.function(void)?;
    let zero = u32_value(&mut f, 0)?;
    let low = u32_value(&mut f, 0xffff)?;
    let sixteen = u32_value(&mut f, 16)?;
    let invocation = f.module().global_value(invocation)?;
    let invocation = f.load(invocation, MemAccess::default())?;
    let x = f.composite_extract(invocation, &[0])?;
    let y = f.composite_extract(invocation, &[1])?;
    let source = f.module().global_value(source)?;
    let destination = f.module().global_value(destination)?;
    let push = f.module().global_value(push)?;
    let member = u32_value(&mut f, m_shape.index)?;
    let shape = f.access_chain(push, &[member])?;
    let shape = f.load(shape, MemAccess::default())?;
    let firsts = f.composite_extract(shape, &[0])?;
    let width = f.composite_extract(shape, &[1])?;
    let source_row_words = f.composite_extract(shape, &[2])?;
    let destination_row_words = f.composite_extract(shape, &[3])?;
    let inside = f.binary(BinOp::ULessThan, x, width)?;
    let copy: ArmFn<'_, ()> = Box::new(move |f, _| {
        let first_source = f.binary(BinOp::BitwiseAnd, firsts, low)?;
        let first_destination = f.binary(BinOp::ShiftRightLogical, firsts, sixteen)?;
        let from = f.binary(BinOp::IMul, y, source_row_words)?;
        let from = f.binary(BinOp::IAdd, first_source, from)?;
        let from = f.binary(BinOp::IAdd, from, x)?;
        let to = f.binary(BinOp::IMul, y, destination_row_words)?;
        let to = f.binary(BinOp::IAdd, first_destination, to)?;
        let to = f.binary(BinOp::IAdd, to, x)?;
        let from = f.access_chain(source, &[zero, from])?;
        let word = f.load(from, MemAccess::default())?;
        let to = f.access_chain(destination, &[zero, to])?;
        f.store(to, word, MemAccess::default())?;
        Ok(ArmEnd::FallThrough(Vec::new()))
    });
    f.if_(&mut (), inside, copy, None)?;
    f.ret(None)?;
    let id = f.finish()?;
    b.entry(
        ExecModel::GLCompute,
        id,
        vec![ExecMode::LocalSize {
            x: RowsCopyModule::LOCAL_SIZE,
            y: 1,
            z: 1,
        }],
    )?;
    b.finalize()
}

/// The fixed binding numbers of the module's symbols.
fn binding_of(sym: &Symbol) -> Option<u32> {
    match sym {
        Symbol::DescriptorSet(_) => Some(RowsCopyModule::SET),
        Symbol::Binding(s) if *s == SOURCE => Some(RowsCopyModule::SOURCE_BINDING),
        Symbol::Binding(s) if *s == DESTINATION => Some(RowsCopyModule::DESTINATION_BINDING),
        _ => None,
    }
}

/// The rows-copy module of `device`: built with the typed builder, checked by the owned verifier
/// against the device's profile (requirements included), serialized, and numbered with the fixed
/// bindings of [`RowsCopyModule`].
pub fn rows_copy_module(device: &Device) -> Result<RowsCopyModule, Error> {
    let profile = device.profile();
    let m = typed_module()?;
    let mut meter = Meter::new(&Budget::default());
    crate::spirv::verify::verify(&m, profile, &mut meter)?;
    let r = crate::spirv::serialize::serialize(m)?;
    profile.check(&r.requirements).map_err(|c| {
        Error::Internal(crate::error::InternalError::new(
            crate::error::InternalKind::RequirementsExceedProfile,
            format!("the rows copy needs {c:?}"),
        ))
    })?;
    let words = crate::spirv::serialize::relocate(&r, &binding_of)?;
    Ok(RowsCopyModule {
        spirv: words.into(),
        requirements: r.requirements,
    })
}

#[cfg(test)]
mod tests {
    use super::{typed_module, RowsCopyModule, DESTINATION, SOURCE};
    use crate::spirv::interp::{run, Buffer, Config, Order, Resources};
    use crate::spirv::testing::{assert_valid, committed_profiles};

    #[test]
    fn the_rows_copy_kernel_is_valid_on_every_committed_profile() {
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

    /// The destination's words after one dispatch over `rows` rows `width` words long, from
    /// `source` (rows `row_words[0]` apart from word `firsts[0]`) into `destination` (rows
    /// `row_words[1]` apart from word `firsts[1]`), in both invocation orders.
    fn copied(
        source: &[u32],
        destination: &[u32],
        firsts: [u32; 2],
        width: u32,
        row_words: [u32; 2],
        rows: u32,
    ) -> Vec<u32> {
        let m = typed_module().expect("module");
        let shape = RowsCopyModule::shape(firsts, width, row_words);
        let mut out = Vec::new();
        for order in [Order::Forward, Order::Reverse] {
            let res = Resources {
                buffers: vec![
                    Buffer {
                        slot: Some(SOURCE),
                        address: None,
                        bytes: bytes(source),
                    },
                    Buffer {
                        slot: Some(DESTINATION),
                        address: None,
                        bytes: bytes(destination),
                    },
                ],
                push: shape.to_vec(),
                spec: Default::default(),
            };
            let [gx, gy] = RowsCopyModule::groups(width, rows);
            let cfg = Config {
                order,
                workgroups: [gx, gy, 1],
                ..Config::default()
            };
            let outcome = run(&m, &cfg, res).expect("run");
            assert_eq!(
                words(&outcome.buffers[0]),
                source,
                "the source is only read"
            );
            out.push(words(&outcome.buffers[1]));
        }
        assert_eq!(out[0], out[1], "order independent");
        out.swap_remove(0)
    }

    /// Every row's words land at the destination row's place, and every word between the rows,
    /// before the first and past the last keeps the destination's own (a padded row's bytes):
    /// tight rows into padded ones and padded into tight, across workgroups.
    #[test]
    fn rows_land_at_their_places_and_no_other_word_changes() {
        let (width, rows) = (70u32, 5u32);
        for (row_words, firsts) in [
            ([70u32, 77], [0u32, 3]),
            ([75, 70], [2, 0]),
            ([90, 81], [5, 9]),
        ] {
            let source: Vec<u32> = (0..firsts[0] + rows * row_words[0] + 4)
                .map(|i| 0x5a00_0000 | i)
                .collect();
            let destination: Vec<u32> = (0..firsts[1] + rows * row_words[1] + 4)
                .map(|i| 0xa5a5_0000 | i)
                .collect();
            let got = copied(&source, &destination, firsts, width, row_words, rows);
            let mut want = destination.clone();
            for y in 0..rows {
                for x in 0..width {
                    let from = firsts[0] + y * row_words[0] + x;
                    let to = firsts[1] + y * row_words[1] + x;
                    want[to as usize] = source[from as usize];
                }
            }
            assert_eq!(got, want, "rows {row_words:?} apart from {firsts:?}");
        }
    }

    /// The first words' two halves: a first word past 255 in either binding is read from its
    /// place.
    #[test]
    fn the_first_words_reach_far_words() {
        let source: Vec<u32> = (0..700).collect();
        let destination = vec![0u32; 1000];
        let got = copied(&source, &destination, [300, 600], 2, [10, 100], 2);
        let mut want = destination.clone();
        want[600] = 300;
        want[601] = 301;
        want[700] = 310;
        want[701] = 311;
        assert_eq!(got, want);
    }
}
