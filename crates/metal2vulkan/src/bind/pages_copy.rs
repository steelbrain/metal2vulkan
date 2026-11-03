//! Whole 4 KiB pages copied by a compute dispatch between places a table scatters over one buffer
//! and consecutive places of another: a gather into the consecutive buffer, or a scatter out of it.
//!
//! A copy of many pages that lie apart in one buffer (a guest's pages in an import of its RAM)
//! into one run of another is a `vkCmdCopyBuffer` of one region per page, or per run of adjacent
//! pages. A driver may record each region as a separate operation, at a CPU cost that grows with
//! them; one dispatch of [`pages_copy_module`] copies every page of a table at once.
//!
//! **The module** (two per device, one per [`PagesCopy`]). One `GLCompute` entry `main` with
//! local size ([`PagesCopyModule::LOCAL_SIZE`], 1, 1), [`PagesCopyModule::PUSH_BYTES`] of
//! `COMPUTE` push constants at offset 0 ([`PagesCopyModule::shape`]), and three storage buffers of
//! 4-byte words in set 0:
//!
//! | Binding | Buffer |
//! |---|---|
//! | [`PagesCopyModule::SCATTERED_BINDING`] | the buffer whose pages the table names one by one |
//! | [`PagesCopyModule::CONSECUTIVE_BINDING`] | the buffer whose pages lie one after another |
//! | [`PagesCopyModule::TABLE_BINDING`] | the table: two words an entry, read only |
//!
//! Entry `e` of the table is the words `2e` and `2e + 1`: the scattered binding's page `s` (its
//! words `1024 s` to `1024 s + 1023`) and the consecutive binding's page `c` (its words from
//! `first_word + 1024 c`). The shape's words, in order: `first_entry`, the entry of the
//! dispatch's first workgroup, and `first_word`, the consecutive binding's word of its page 0;
//! two words of 0. Workgroup `g` takes entry `first_entry + g`, and its invocation `l` the
//! page's words `N k + l` for `k` from 0 to 7 ([`PagesCopyModule::WORDS`], `N`
//! [`PagesCopyModule::LOCAL_SIZE`]), one after the other: a [`PagesCopy::Gather`] stores each
//! word of the scattered page over the consecutive page's, and a [`PagesCopy::Scatter`] the
//! consecutive page's over the scattered one's. No other word of either binding is read or
//! written, and nothing writes the table. Where two entries' destination pages are one, or a
//! destination page is a source page of the dispatch, the result is unordered, as a copy's
//! regions that overlap are. A dispatch takes [`PagesCopyModule::groups`] workgroups.
//!
//! The words a workgroup apart, not eight in a row: each of the eight stores of consecutive
//! invocations falls on consecutive words, so the stores leave no gap between a warp's lanes
//! (`shadow_merge`'s measure on an RTX 5080 Laptop GPU: stores into imported host memory that
//! leave gaps take hundreds of times the time of the same bytes stored without).
//!
//! **Synchronization** (consumer): the table's writes, the source binding's earlier writes and
//! the destination binding's earlier accesses → `SHADER_STORAGE_READ` (the destination:
//! `SHADER_STORAGE_WRITE`) in `COMPUTE_SHADER` before the dispatch; its writes → the
//! destination's next use.

use std::sync::Arc;

use crate::budget::{Budget, Meter};
use crate::device::requirements::Requirements;
use crate::device::Device;
use crate::error::{Error, InternalError, InternalKind};
use crate::pipeline::sym::{PushField, ResourceKind, ResourceSlot, Stage, Symbol};
use crate::spirv::builder::FunctionBuilder;
use crate::spirv::builder::ModuleBuilder;
use crate::spirv::error::BResult;
use crate::spirv::ids::Value;
use crate::spirv::ir::{BinOp, ExecMode, ExecModel, IoDir, MemAccess, Type, TypedModule};

use super::shadow_merge::{u32_value, words_buffer};

/// The crate-owned pages-copy module of one device and direction, validated like any output.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PagesCopyModule {
    /// SPIR-V 1.5 words, entry point `main`, bindings numbered as below in set 0.
    pub spirv: Arc<[u32]>,
    /// What the module needs from the device (within every Vulkan 1.2 device).
    pub requirements: Requirements,
}

impl PagesCopyModule {
    /// Descriptor set of every binding.
    pub const SET: u32 = 0;
    /// The buffer whose pages the table names one by one, a storage buffer.
    pub const SCATTERED_BINDING: u32 = 0;
    /// The buffer whose pages lie one after another, a storage buffer.
    pub const CONSECUTIVE_BINDING: u32 = 1;
    /// The table, two words an entry, a storage buffer the dispatch only reads.
    pub const TABLE_BINDING: u32 = 2;
    /// The bytes of a page, which a workgroup copies.
    pub const PAGE_BYTES: u32 = 4096;
    /// Invocations of one workgroup, along x: [`WORDS`](Self::WORDS) 4-byte words each.
    pub const LOCAL_SIZE: u32 = 128;
    /// The words one invocation copies, one after the other, a workgroup apart.
    pub const WORDS: u32 = 8;
    /// Bytes of the `COMPUTE` push-constant range at offset 0: the dispatch's shape.
    pub const PUSH_BYTES: u32 = 16;

    /// The push bytes of a dispatch whose first workgroup takes entry `first_entry` of the
    /// table, the consecutive binding's page 0 at its word `first_word` (the module docs).
    pub fn shape(first_entry: u32, first_word: u32) -> [u8; 16] {
        let words = [first_entry, first_word, 0, 0];
        let mut bytes = [0; 16];
        for (chunk, word) in bytes.as_chunks_mut::<4>().0.iter_mut().zip(words) {
            chunk.copy_from_slice(&word.to_le_bytes());
        }
        bytes
    }

    /// The workgroups of a dispatch over `entries` entries of the table: one each.
    pub fn groups(entries: u32) -> [u32; 3] {
        [entries, 1, 1]
    }

    /// The table's words of entry `(scattered, consecutive)`, the two pages' indices.
    pub fn entry(scattered: u32, consecutive: u32) -> [u8; 8] {
        let mut bytes = [0; 8];
        bytes[..4].copy_from_slice(&scattered.to_le_bytes());
        bytes[4..].copy_from_slice(&consecutive.to_le_bytes());
        bytes
    }
}

/// Which way a [`pages_copy_module`] moves the pages.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PagesCopy {
    /// The table's scattered pages into the consecutive ones.
    Gather,
    /// The consecutive pages out into the table's scattered ones.
    Scatter,
}

impl PagesCopy {
    /// Both directions.
    pub const ALL: [PagesCopy; 2] = [PagesCopy::Gather, PagesCopy::Scatter];
}

/// The binding symbol of the scattered pages' buffer in the typed module.
const SCATTERED: ResourceSlot = ResourceSlot {
    stage: Stage::Kernel,
    kind: ResourceKind::Buffer,
    index: 0,
};
/// The binding symbol of the consecutive pages' buffer in the typed module.
const CONSECUTIVE: ResourceSlot = ResourceSlot {
    stage: Stage::Kernel,
    kind: ResourceKind::Buffer,
    index: 1,
};
/// The binding symbol of the table in the typed module.
const TABLE: ResourceSlot = ResourceSlot {
    stage: Stage::Kernel,
    kind: ResourceKind::Buffer,
    index: 2,
};

/// The push field the shape is declared under: a crate-owned module's own words, which no
/// selection places.
const SHAPE: PushField = PushField::DispatchRegion;

/// The word `base + step` of a page, `step` a constant (`base` itself at 0).
fn word_at(f: &mut FunctionBuilder<'_>, base: Value, step: u32) -> BResult<Value> {
    match step {
        0 => Ok(base),
        step => {
            let step = u32_value(f, step)?;
            f.binary(BinOp::IAdd, base, step)
        }
    }
}

/// The pages-copy module of `copy` as a typed module, before serialization: the module
/// [`pages_copy_module`] serializes, which the CPU interpreter runs. Its bindings are the symbols
/// `SCATTERED`, `CONSECUTIVE` and `TABLE`, and its push field the shape, a `uvec4` (the module
/// docs).
fn typed_module(copy: PagesCopy) -> BResult<TypedModule> {
    let mut b = ModuleBuilder::new();
    let u32_ = b.ty(Type::Int {
        width: 32,
        signed: false,
    })?;
    let uvec3 = b.ty(Type::Vector { elem: u32_, len: 3 })?;
    let uvec4 = b.ty(Type::Vector { elem: u32_, len: 4 })?;
    let m_shape = b.push_member(SHAPE, uvec4, 0)?;
    let push = b.push_global()?;
    let scattered = words_buffer(&mut b, SCATTERED, u32_)?;
    let consecutive = words_buffer(&mut b, CONSECUTIVE, u32_)?;
    let table = words_buffer(&mut b, TABLE, u32_)?;
    let group = b.builtin(crate::spv::BuiltIn::WorkgroupId, IoDir::Input, uvec3)?;
    let local = b.builtin(crate::spv::BuiltIn::LocalInvocationId, IoDir::Input, uvec3)?;
    let void = b.ty(Type::Void)?;
    let mut f = b.function(void)?;
    let zero = u32_value(&mut f, 0)?;
    let one = u32_value(&mut f, 1)?;
    let group = f.module().global_value(group)?;
    let group = f.load(group, MemAccess::default())?;
    let g = f.composite_extract(group, &[0])?;
    let local = f.module().global_value(local)?;
    let local = f.load(local, MemAccess::default())?;
    let l = f.composite_extract(local, &[0])?;
    let scattered = f.module().global_value(scattered)?;
    let consecutive = f.module().global_value(consecutive)?;
    let table = f.module().global_value(table)?;
    let push = f.module().global_value(push)?;
    let member = u32_value(&mut f, m_shape.index)?;
    let shape = f.access_chain(push, &[member])?;
    let shape = f.load(shape, MemAccess::default())?;
    let first_entry = f.composite_extract(shape, &[0])?;
    let first_word = f.composite_extract(shape, &[1])?;
    // The workgroup's entry, and its two pages.
    let entry = f.binary(BinOp::IAdd, first_entry, g)?;
    let at = f.binary(BinOp::IAdd, entry, entry)?;
    let p = f.access_chain(table, &[zero, at])?;
    let scattered_page = f.load(p, MemAccess::default())?;
    let at = f.binary(BinOp::IAdd, at, one)?;
    let p = f.access_chain(table, &[zero, at])?;
    let consecutive_page = f.load(p, MemAccess::default())?;
    let page_words = u32_value(&mut f, PagesCopyModule::PAGE_BYTES / 4)?;
    // Each page's first word, plus the invocation's place in it: its words lie `LOCAL_SIZE`
    // apart from there (the module docs).
    let s = f.binary(BinOp::IMul, scattered_page, page_words)?;
    let s = f.binary(BinOp::IAdd, s, l)?;
    let c = f.binary(BinOp::IMul, consecutive_page, page_words)?;
    let c = f.binary(BinOp::IAdd, c, first_word)?;
    let c = f.binary(BinOp::IAdd, c, l)?;
    let (from, from_base, to, to_base) = match copy {
        PagesCopy::Gather => (scattered, s, consecutive, c),
        PagesCopy::Scatter => (consecutive, c, scattered, s),
    };
    for k in 0..PagesCopyModule::WORDS {
        let step = k * PagesCopyModule::LOCAL_SIZE;
        let read = word_at(&mut f, from_base, step)?;
        let write = word_at(&mut f, to_base, step)?;
        let p = f.access_chain(from, &[zero, read])?;
        let word = f.load(p, MemAccess::default())?;
        let p = f.access_chain(to, &[zero, write])?;
        f.store(p, word, MemAccess::default())?;
    }
    f.ret(None)?;
    let id = f.finish()?;
    b.entry(
        ExecModel::GLCompute,
        id,
        vec![ExecMode::LocalSize {
            x: PagesCopyModule::LOCAL_SIZE,
            y: 1,
            z: 1,
        }],
    )?;
    b.finalize()
}

/// The fixed binding numbers of the module's symbols.
fn binding_of(sym: &Symbol) -> Option<u32> {
    match sym {
        Symbol::DescriptorSet(_) => Some(PagesCopyModule::SET),
        Symbol::Binding(s) if *s == SCATTERED => Some(PagesCopyModule::SCATTERED_BINDING),
        Symbol::Binding(s) if *s == CONSECUTIVE => Some(PagesCopyModule::CONSECUTIVE_BINDING),
        Symbol::Binding(s) if *s == TABLE => Some(PagesCopyModule::TABLE_BINDING),
        _ => None,
    }
}

/// The pages-copy module of `device` moving pages as `copy` says: built with the typed builder,
/// checked by the owned verifier against the device's profile (requirements included),
/// serialized, and numbered with the fixed bindings of [`PagesCopyModule`].
pub fn pages_copy_module(device: &Device, copy: PagesCopy) -> Result<PagesCopyModule, Error> {
    let profile = device.profile();
    let m = typed_module(copy)?;
    let mut meter = Meter::new(&Budget::default());
    crate::spirv::verify::verify(&m, profile, &mut meter)?;
    let r = crate::spirv::serialize::serialize(m)?;
    profile.check(&r.requirements).map_err(|c| {
        Error::Internal(InternalError::new(
            InternalKind::RequirementsExceedProfile,
            format!("the pages copy needs {c:?}"),
        ))
    })?;
    let words = crate::spirv::serialize::relocate(&r, &binding_of)?;
    Ok(PagesCopyModule {
        spirv: words.into(),
        requirements: r.requirements,
    })
}

#[cfg(test)]
mod tests {
    use super::{typed_module, PagesCopy, PagesCopyModule, CONSECUTIVE, SCATTERED, TABLE};
    use crate::spirv::interp::{run, Buffer, Config, Order, Outcome, Resources};
    use crate::spirv::testing::{assert_valid, committed_profiles};

    const PAGE: usize = PagesCopyModule::PAGE_BYTES as usize;

    #[test]
    fn the_pages_copy_kernels_are_valid_on_every_committed_profile() {
        for (_, p) in committed_profiles().expect("profiles") {
            for copy in PagesCopy::ALL {
                assert_valid(typed_module(copy).expect("module"), &p);
            }
        }
    }

    /// Bytes in which every 4 KiB page differs from every other, from `seed`.
    fn pages(count: usize, seed: u32) -> Vec<u8> {
        (0..count * PAGE / 4)
            .flat_map(|i| {
                (seed.wrapping_mul(0x9e37_79b9) ^ (i as u32).wrapping_mul(0x85eb_ca6b))
                    .to_le_bytes()
            })
            .collect()
    }

    /// The scattered and consecutive buffers after one dispatch of `copy` over `entries` from
    /// entry `first_entry` of `table`, the consecutive pages from word `first_word`, in both
    /// invocation orders; the table unchanged.
    fn copied(
        copy: PagesCopy,
        (scattered, consecutive): (&[u8], &[u8]),
        table: &[(u32, u32)],
        (first_entry, entries, first_word): (u32, u32, u32),
    ) -> (Vec<u8>, Vec<u8>) {
        let m = typed_module(copy).expect("module");
        let table: Vec<u8> = (table.iter())
            .flat_map(|&(s, c)| PagesCopyModule::entry(s, c))
            .collect();
        let mut out = Vec::new();
        for order in [Order::Forward, Order::Reverse] {
            let part = |slot, bytes: &[u8]| Buffer {
                slot: Some(slot),
                address: None,
                bytes: bytes.to_vec(),
            };
            let res = Resources {
                buffers: vec![
                    part(SCATTERED, scattered),
                    part(CONSECUTIVE, consecutive),
                    part(TABLE, &table),
                ],
                push: PagesCopyModule::shape(first_entry, first_word).to_vec(),
                spec: Default::default(),
            };
            let cfg = Config {
                order,
                workgroups: PagesCopyModule::groups(entries),
                ..Config::default()
            };
            let Outcome { buffers, .. } = run(&m, &cfg, res).expect("run");
            assert_eq!(buffers[2], table, "the table is only read");
            out.push((buffers[0].clone(), buffers[1].clone()));
        }
        assert_eq!(out[0], out[1], "order independent");
        out.swap_remove(0)
    }

    /// A gather takes each entry's scattered page into its consecutive page, in the table's order
    /// (descending and repeated pages among them), from the dispatch's first entry and the
    /// consecutive binding's first word: no other byte of either buffer changes.
    #[test]
    fn a_gather_takes_each_entrys_scattered_page_into_its_consecutive_page() {
        let scattered = pages(5, 1);
        // One word of offset, and room for four pages after it.
        let consecutive = pages(5, 2);
        let table = [(9, 9), (4, 0), (3, 1), (0, 2), (3, 3)];
        // Entries 1 to 4: the first is not the dispatch's.
        let (s, c) = copied(
            PagesCopy::Gather,
            (&scattered, &consecutive),
            &table,
            (1, 4, 1),
        );
        assert_eq!(s, scattered, "a gather writes nothing scattered");
        let mut want = consecutive.clone();
        for &(from, to) in &table[1..] {
            let (from, to) = (from as usize * PAGE, 4 + to as usize * PAGE);
            want[to..to + PAGE].copy_from_slice(&scattered[from..from + PAGE]);
        }
        assert_eq!(c, want);
    }

    /// A scatter stores each entry's consecutive page over its scattered page, and writes nothing
    /// else.
    #[test]
    fn a_scatter_stores_each_consecutive_page_over_its_scattered_page() {
        let scattered = pages(6, 3);
        let consecutive = pages(3, 4);
        let table = [(5, 0), (2, 1), (1, 2)];
        let (s, c) = copied(
            PagesCopy::Scatter,
            (&scattered, &consecutive),
            &table,
            (0, 3, 0),
        );
        assert_eq!(c, consecutive, "a scatter writes nothing consecutive");
        let mut want = scattered.clone();
        for &(to, from) in &table {
            let (from, to) = (from as usize * PAGE, to as usize * PAGE);
            want[to..to + PAGE].copy_from_slice(&consecutive[from..from + PAGE]);
        }
        assert_eq!(s, want);
    }
}
