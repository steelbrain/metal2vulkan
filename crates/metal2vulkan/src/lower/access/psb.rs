//! The PSB leaf of a static slot: a slot whose `BufferForm` is `Psb` (a `BufferRangeLimit` variant,
//! or Profile A's per-stage descriptor-limit fallback) is reached through the device address of its
//! allocation.
//!
//! **Root table.** The address comes from the resident root table, at the device address
//! `PushField::RootTableAddress`: 32-byte entries `{ host: u64, guest: u64, size: u64, bind_offset:
//! u64 }` ([`ROOT_ENTRY_BYTES`]), where entry 1 is the write-only sink (`NeutralResource::Sink`, at
//! least 64 bytes) and entry `2 + n` is `[[buffer(n)]]`: `host` the device address of the
//! allocation, `size` its padded size, and `bind_offset` the Metal bind offset into it.
//!
//! **Offsets.** A static-root fat pointer's offset `m` is the Metal displacement from the bind
//! offset modulo 2³². For a slot whose window starts at `start` (0 for `Window::Whole`), the 64-bit
//! displacement is `start + zext((m − start) mod 2³²)`: the unique value in `[start, start + 2³²)`
//! congruent to `m`. It is added to `bind_offset` in `u64`, so the allocation-relative offset is
//! exact for every Metal-defined access of a window of at most 2³² bytes, on either side of the
//! bind offset. Selection refuses a `Bounded` window longer than that (`access::psb_lowering`); a
//! `Whole` window reaches the first 2³² bytes above the bind offset.
//!
//! **Containment** is mandatory for PSB (robust buffer access does not apply to device addresses):
//! the normative check in `u64` against `size`, the access at `select(inb, address, sink)`, and a
//! load's value `select(inb, v, 0)`. A contained store lands in the sink, which no shader reads
//! observably. Only `Containment::UnsafeOff` omits it.
//!
//! The leaf itself (`memory::leaf`) sees a word memory based at the 4-aligned address below the
//! access, and the access's byte offset within that word (0 to 3).
//!
//! **Narrow units.** Where the profile has the 8- or 16-bit storage access ([`narrow_units`]), the
//! memory also loads and stores 1- and 2-byte units: a `u8` or `u16` place at the unit's device
//! address, with the `Aligned` operand of its size, as a descriptor slot's narrow views do. Vulkan's
//! `storageBuffer8BitAccess` and `storageBuffer16BitAccess` cover the `PhysicalStorageBuffer`
//! storage class as they cover `StorageBuffer`. A sub-word store there is one narrow store; without
//! the access it stays the two word atomics ([`PsbMem`]'s `write_bits`).

use crate::device::DeviceProfile;
use crate::memory::leaf::{Units, WordMem, WordOps};
use crate::spirv::builder::FunctionBuilder;
use crate::spirv::error::BResult;
use crate::spirv::ids::{TypeId, Value};
use crate::spirv::ir::{
    AtomicOp, BinOp, ConstBits, Constant, ConvOp, MemAccess, Scope, Semantics, Type, UnOp,
};

/// Bytes of one root-table entry.
pub const ROOT_ENTRY_BYTES: u64 = 32;
/// The sink's root.
pub const SINK_ROOT: u32 = 1;

/// Byte offsets of the entry fields.
const HOST: u64 = 0;
const SIZE: u64 = 16;
const BIND_OFFSET: u64 = 24;

fn u64_ty(fb: &mut FunctionBuilder<'_>) -> BResult<TypeId> {
    fb.module().ty(Type::Int {
        width: 64,
        signed: false,
    })
}

/// A `u64` constant.
pub fn k64(fb: &mut FunctionBuilder<'_>, v: u64) -> BResult<Value> {
    let ty = u64_ty(fb)?;
    let c = fb.module().constant(Constant::Scalar {
        ty,
        bits: ConstBits(v),
    })?;
    fb.module().const_value(c)
}

/// The `u64` field at byte `field` of root `root`'s entry in the table at `table`.
fn entry_field(
    fb: &mut FunctionBuilder<'_>,
    table: Value,
    root: u32,
    field: u64,
) -> BResult<Value> {
    let at = k64(
        fb,
        u64::from(root)
            .saturating_mul(ROOT_ENTRY_BYTES)
            .saturating_add(field),
    )?;
    let addr = fb.binary(BinOp::IAdd, table, at)?;
    let u64_ = u64_ty(fb)?;
    let p = fb.device_place(addr, u64_)?;
    fb.load(
        p,
        MemAccess {
            aligned: Some(8),
            ..MemAccess::default()
        },
    )
}

/// The 64-bit displacement from the bind offset (a `u64`) that the `u32` offset `m` of a
/// static-root fat pointer stands for, in a slot whose window starts at `start` (module doc:
/// `start + zext((m − start) mod 2³²)`, wrapping).
pub fn displacement(fb: &mut FunctionBuilder<'_>, m: Value, start: i64) -> BResult<Value> {
    let u64_ = u64_ty(fb)?;
    if start == 0 {
        return fb.convert(ConvOp::UConvert, m, u64_);
    }
    // The two's-complement bits of `start` are its value modulo 2^32 and 2^64.
    let low = fb.u32_const(start as u32)?;
    let low = fb.module().const_value(low)?;
    let rel = fb.binary(BinOp::ISub, m, low)?;
    let rel = fb.convert(ConvOp::UConvert, rel, u64_)?;
    let start = k64(fb, start as u64)?;
    fb.binary(BinOp::IAdd, start, rel)
}

/// A PSB access prepared for the leaf: the word memory, the byte offset within its first word,
/// and the containment condition (`None` without containment).
#[derive(Clone, Copy, Debug)]
pub struct PsbAccess {
    pub mem: PsbMem,
    /// The byte offset (0 to 3) of the access within the memory's first word, a `u32`.
    pub off: Value,
    pub inb: Option<Value>,
}

/// Prepare an `n`-byte access of root `root` (a static slot whose window starts at `start`) at
/// the `u32` displacement `m` from its bind offset, for the table at `table` (a `u64`).
pub fn prepare(
    fb: &mut FunctionBuilder<'_>,
    table: Value,
    root: u32,
    m: Value,
    start: i64,
    n: u32,
    contained: bool,
) -> BResult<PsbAccess> {
    let host = entry_field(fb, table, root, HOST)?;
    let size = entry_field(fb, table, root, SIZE)?;
    let bind = entry_field(fb, table, root, BIND_OFFSET)?;
    let disp = displacement(fb, m, start)?;
    let off = fb.binary(BinOp::IAdd, bind, disp)?;
    let three = k64(fb, 3)?;
    let low = fb.binary(BinOp::BitwiseAnd, off, three)?;
    let aligned = fb.binary(BinOp::ISub, off, low)?;
    let base = fb.binary(BinOp::IAdd, host, aligned)?;
    let u32_ = fb.module().ty(Type::Int {
        width: 32,
        signed: false,
    })?;
    let low = fb.convert(ConvOp::UConvert, low, u32_)?;
    if !contained {
        return Ok(PsbAccess {
            mem: PsbMem::at(base),
            off: low,
            inb: None,
        });
    }
    let inb = crate::memory::contain::in_bounds(fb, off, size, u64::from(n))?;
    let sink = entry_field(fb, table, SINK_ROOT, HOST)?;
    let base = fb.select(inb, base, sink)?;
    let zero = fb.u32_const(0)?;
    let zero = fb.module().const_value(zero)?;
    let low = fb.select(inb, low, zero)?;
    Ok(PsbAccess {
        mem: PsbMem::at(base),
        off: low,
        inb: Some(inb),
    })
}

/// The narrow units a device-address memory loads and stores directly on profile `p`: the 1- and
/// 2-byte units of [`Units::storage_on`] (the width's storage access and integer type). No `uvec4`
/// unit: a device address below an access is only 4-byte aligned.
pub fn narrow_units(p: &DeviceProfile) -> Units {
    Units {
        vec4: false,
        ..Units::storage_on(p)
    }
}

/// Words at a 4-aligned device address (a `u64`), loaded and stored with memory operands (a
/// coherence and `Volatile`), and the narrow units ([`narrow_units`]) it also loads and stores
/// directly.
#[derive(Clone, Copy, Debug)]
pub struct PsbMem {
    base: Value,
    ops: WordOps,
    units: Units,
}

impl PsbMem {
    /// The words at the 4-aligned device address `base` (a `u64`), with plain accesses and no
    /// narrow unit.
    pub fn at(base: Value) -> PsbMem {
        PsbMem {
            base,
            ops: WordOps::default(),
            units: Units::NONE,
        }
    }

    /// The same words, loaded and stored with `ops`.
    pub fn with(self, ops: WordOps) -> PsbMem {
        PsbMem { ops, ..self }
    }

    /// The same words, also loading and storing the narrow units of `units` (a subset of
    /// [`narrow_units`] of the profile).
    pub fn with_units(self, units: Units) -> PsbMem {
        PsbMem {
            units: Units {
                vec4: false,
                ..units
            },
            ..self
        }
    }

    /// A `PhysicalStorageBuffer` place of the `bytes`-byte unit `index` (a `u32` index of such
    /// units) of the memory, and the unit's type.
    fn unit_ptr(
        &self,
        fb: &mut FunctionBuilder<'_>,
        bytes: u32,
        index: Value,
    ) -> BResult<(Value, TypeId)> {
        let u64_ = u64_ty(fb)?;
        let i = fb.convert(ConvOp::UConvert, index, u64_)?;
        let d = match bytes {
            1 => i,
            _ => {
                let size = k64(fb, u64::from(bytes))?;
                fb.binary(BinOp::IMul, i, size)?
            }
        };
        let addr = fb.binary(BinOp::IAdd, self.base, d)?;
        let ty = fb.module().ty(Type::Int {
            width: if bytes == 1 { 8 } else { 16 },
            signed: false,
        })?;
        Ok((fb.device_place(addr, ty)?, ty))
    }

    /// A `PhysicalStorageBuffer` place of word `word` (a `u32` index) of the memory.
    pub fn word_ptr(&self, fb: &mut FunctionBuilder<'_>, word: Value) -> BResult<Value> {
        let u32_ = fb.module().ty(Type::Int {
            width: 32,
            signed: false,
        })?;
        self.typed_ptr(fb, word, u32_)
    }

    /// A `PhysicalStorageBuffer` place of a `pointee` at word `word` (a `u32` index) of the
    /// memory: a typed alias of the words, which only a device address gives (a descriptor view
    /// is one `u32` array). Atomics take it.
    pub fn typed_ptr(
        &self,
        fb: &mut FunctionBuilder<'_>,
        word: Value,
        pointee: TypeId,
    ) -> BResult<Value> {
        let addr = self.word_address(fb, word)?;
        fb.device_place(addr, pointee)
    }

    /// The device address (a `u64`) of word `word` (a `u32` index) of the memory.
    pub fn word_address(&self, fb: &mut FunctionBuilder<'_>, word: Value) -> BResult<Value> {
        let u64_ = u64_ty(fb)?;
        let w = fb.convert(ConvOp::UConvert, word, u64_)?;
        let four = k64(fb, 4)?;
        let d = fb.binary(BinOp::IMul, w, four)?;
        fb.binary(BinOp::IAdd, self.base, d)
    }
}

const WORD: MemAccess = MemAccess {
    volatile: false,
    aligned: Some(4),
    nontemporal: false,
    make_available: None,
    make_visible: None,
    non_private: false,
};

impl WordMem for PsbMem {
    fn read(&mut self, fb: &mut FunctionBuilder<'_>, word: Value) -> BResult<Value> {
        let p = self.word_ptr(fb, word)?;
        let access = self.ops.access(fb, WORD, false)?;
        fb.load(p, access)
    }

    fn write(&mut self, fb: &mut FunctionBuilder<'_>, word: Value, value: Value) -> BResult<()> {
        let p = self.word_ptr(fb, word)?;
        let access = self.ops.access(fb, WORD, true)?;
        fb.store(p, value, access)
    }

    fn write_bits(
        &mut self,
        fb: &mut FunctionBuilder<'_>,
        word: Value,
        mask: Value,
        bits: Value,
    ) -> BResult<()> {
        let p = self.word_ptr(fb, word)?;
        let keep = fb.unary(UnOp::Not, mask)?;
        let clear = fb.binary(BinOp::BitwiseOr, keep, bits)?;
        fb.atomic(AtomicOp::And, p, Scope::Device, Semantics::RELAXED, clear)?;
        fb.atomic(AtomicOp::Or, p, Scope::Device, Semantics::RELAXED, bits)?;
        Ok(())
    }

    fn units(&self) -> Units {
        self.units
    }

    fn read_unit(
        &mut self,
        fb: &mut FunctionBuilder<'_>,
        bytes: u32,
        index: Value,
    ) -> BResult<Value> {
        let (p, _) = self.unit_ptr(fb, bytes, index)?;
        let access = self.ops.access(fb, unit_access(bytes), false)?;
        let x = fb.load(p, access)?;
        let u32_ = fb.module().ty(Type::Int {
            width: 32,
            signed: false,
        })?;
        let y = fb.convert(ConvOp::UConvert, x, u32_)?;
        if bytes == 2 {
            fb.note_widened16(y, x);
        }
        Ok(y)
    }

    fn write_unit(
        &mut self,
        fb: &mut FunctionBuilder<'_>,
        bytes: u32,
        index: Value,
        value: Value,
    ) -> BResult<()> {
        let (p, ty) = self.unit_ptr(fb, bytes, index)?;
        let x = fb.convert(ConvOp::UConvert, value, ty)?;
        let access = self.ops.access(fb, unit_access(bytes), true)?;
        fb.store(p, x, access)
    }
}

/// The plain memory operands of a `bytes`-byte unit: `Aligned` to its size.
fn unit_access(bytes: u32) -> MemAccess {
    MemAccess {
        aligned: Some(bytes),
        ..WORD
    }
}
