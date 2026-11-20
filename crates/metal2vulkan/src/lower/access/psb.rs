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
//! observably. Only `Containment::UnsafeOff` omits it. Under the identity domains no table is
//! read: an access whose offset is not a translation-time constant is checked against the slot's
//! buffer ([`in_buffer`], `emit::prepare`), a load outside reading the zero range
//! ([`PsbMem::outside`]).
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

use super::SLOT_ADDRESS_BIAS;
use crate::device::DeviceProfile;
use crate::memory::leaf::{k32, u32_op, Units, WordMem, WordOps};
use crate::spirv::builder::FunctionBuilder;
use crate::spirv::error::BResult;
use crate::spirv::ids::{TypeId, Value};
use crate::spirv::ir::{
    AtomicOp, BinOp, ConstBits, Constant, ConvOp, GlslOp, MemAccess, Scope, Semantics, Type, UnOp,
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

/// How a device-address memory reaches its units.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Reach {
    /// From the 4-aligned device address below one access ([`prepare`]): each unit at that
    /// address plus its index times its size, in 64 bits.
    Rounded,
    /// From the 4-aligned device address below one access of a slot under the identity domains
    /// ([`prepare_at`]): each unit at that address plus its index times its size, the product in
    /// 32 bits and zero-extended. A unit lies within 2^31 bytes of that address, the runtime
    /// making no buffer longer than 2 GiB (`metal2vulkan_rt::MAX_BUFFER_LENGTH`), so the product is
    /// exact, and no 64-bit shift or multiply scales an index.
    Below,
    /// From a slot's own device address under the identity domains ([`prepare_direct`]), held
    /// as that address less [`SLOT_ADDRESS_BIAS`] (2^31): each unit at that base plus its 32-bit
    /// byte offset plus 2^31, modulo 2^32, zero-extended, in 64-bit integers, then one
    /// `OpConvertUToPtr` to the unit's block (`NonWritable` for a load of an unwritten slot).
    /// Every byte of a buffer lies within 2^31 bytes of each bind address in it, the runtime
    /// making no buffer longer than 2 GiB (`metal2vulkan_rt::MAX_BUFFER_LENGTH`), so the offset
    /// read as signed is in `[-2^31, 2^31)`, the biased one is that offset plus 2^31 exactly, and
    /// the sum is the address plus the offset. A unit index `x + c` with a constant `c` folds
    /// `c`'s bytes into the bias: `(x + c) * s + 2^31` is `x * s + (c * s + 2^31)` modulo 2^32.
    /// The bias is one 32-bit add, which a device fuses with the index's shift; the offset is
    /// zero-extended, which a device splits into a scalar base and a vector offset; no shift of
    /// the offset in 64 bits and no pointer bitcast.
    Direct,
}

/// Prepare an access of a static slot at the `u32` displacement `m` from its bind offset, read as
/// signed, under the identity domains: `address` (a `u64`) is the slot's own device address, the
/// buffer's plus the bind offset, so no table is read, and the access is at `address` plus `m`
/// sign-extended, below the bind address for a negative `m`, as [`prepare_direct`]'s is. Every
/// byte of a buffer lies within 2^31 bytes of each bind address in it, the runtime making no buffer
/// longer than 2 GiB, so the offset is exact whatever the slot's window. The address need not be
/// 4-aligned: the word below the access and the byte offset in it come from the sum. The check,
/// where the access takes one, is `emit::prepare`'s ([`in_buffer`]).
pub fn prepare_at(fb: &mut FunctionBuilder<'_>, address: Value, m: Value) -> BResult<PsbAccess> {
    let u64_ = u64_ty(fb)?;
    let disp = fb.convert(ConvOp::SConvert, m, u64_)?;
    let at = fb.binary(BinOp::IAdd, address, disp)?;
    let three = k64(fb, 3)?;
    let low = fb.binary(BinOp::BitwiseAnd, at, three)?;
    let base = fb.binary(BinOp::ISub, at, low)?;
    let u32_ = fb.module().ty(Type::Int {
        width: 32,
        signed: false,
    })?;
    let low = fb.convert(ConvOp::UConvert, low, u32_)?;
    Ok(PsbAccess {
        mem: PsbMem {
            reach: Reach::Below,
            ..PsbMem::at(base)
        },
        off: low,
        inb: None,
    })
}

/// Prepare an access of a static slot at the `u32` displacement `m` from its bind offset, under
/// the identity domains, from `base` (a `u64`), the slot's own device address less
/// [`SLOT_ADDRESS_BIAS`], an address the binder holds to a multiple of the argument's alignment
/// (`RuntimeFact::AddressAlignment`): the access is the byte offset `m`, read as signed, from that
/// address, as a descriptor view's is `m` from its base ([`Reach::Direct`]). Every byte of a
/// buffer lies within 2^31 bytes of each bind address in it, the runtime making no buffer longer
/// than 2 GiB (`metal2vulkan_rt::MAX_BUFFER_LENGTH`), so the offset is exact whatever the slot's
/// window. Nothing is rounded. The check, where the access takes one, is `emit::prepare`'s.
pub fn prepare_direct(base: Value, m: Value) -> PsbAccess {
    PsbAccess {
        mem: PsbMem {
            reach: Reach::Direct,
            ..PsbMem::at(base)
        },
        off: m,
        inb: None,
    }
}

/// The limit of an `n`-byte access (`n` at least 1) of a buffer of `length` bytes, an unsigned
/// integer: `length - min(length, n - 1)`, so the access at byte `at` of the buffer lies in it
/// exactly where `at < limit`, the normative check `(length >= n) && (at <= length - n)`
/// (`contain::in_bounds`): where `length >= n` the limit is `length - n + 1`, and below that it is
/// 0, which no offset is under. Nothing wraps. `length` itself for a 1-byte access.
pub fn limit(fb: &mut FunctionBuilder<'_>, length: Value, n: u32) -> BResult<Value> {
    if n <= 1 {
        return Ok(length);
    }
    let k = crate::memory::contain::int_like(fb, length, u64::from(n - 1))?;
    let low = fb.glsl(GlslOp::UMin, &[length, k])?;
    fb.binary(BinOp::ISub, length, low)
}

/// Whether an access at the `u32` byte offset `m` from a slot's bind offset, read as signed, lies
/// in the slot's buffer: `m` plus `before`, the buffer's bytes before the bind offset (from the
/// slot's pushed bounds, `PushField::SlotBounds`), under the slot's [`limit`] for the access's
/// size, both made once per function by `EmitCx::slot_limit`. One add and one compare per access,
/// in `u32`. Exact as the new runtime makes no buffer longer than 2 GiB: an offset that reaches
/// before the buffer's start wraps to at least 2^31, past every limit.
pub fn in_buffer(
    fb: &mut FunctionBuilder<'_>,
    m: Value,
    (before, limit): (Value, Value),
) -> BResult<Value> {
    let at = fb.binary(BinOp::IAdd, m, before)?;
    fb.binary(BinOp::ULessThan, at, limit)
}

/// The narrow units a device-address memory loads and stores directly on profile `p`: the 1- and
/// 2-byte units of [`Units::storage_on`] (the width's storage access and integer type). No `uvec2`
/// or `uvec4` unit: a device address below an access is only 4-byte aligned.
pub fn narrow_units(p: &DeviceProfile) -> Units {
    Units {
        vec2: false,
        vec4: false,
        ..Units::storage_on(p)
    }
}

/// Words at a 4-aligned device address (a `u64`), loaded and stored with memory operands (a
/// coherence and `Volatile`), and the narrow units ([`narrow_units`]) it also loads and stores
/// directly. Memory nothing writes while the module runs is loaded through `NonWritable` places
/// ([`PsbMem::read_only`]).
#[derive(Clone, Copy, Debug)]
pub struct PsbMem {
    base: Value,
    ops: WordOps,
    units: Units,
    read_only: bool,
    reach: Reach,
    /// Where a load's units read when a check fails ([`PsbMem::outside`]): the check and the
    /// zero range's address.
    outside: Option<(Value, Value)>,
}

impl PsbMem {
    /// The words at the 4-aligned device address `base` (a `u64`), with plain accesses and no
    /// narrow unit.
    pub fn at(base: Value) -> PsbMem {
        PsbMem {
            base,
            ops: WordOps::default(),
            units: Units::NONE,
            read_only: false,
            reach: Reach::Rounded,
            outside: None,
        }
    }

    /// The same words, each unit a load reads at `zero` (a `u64`, the identity domains' zero
    /// range: 16 bytes of zeros at a multiple of 16 that nothing writes) where `inb` fails: one
    /// select of the unit's address, so a load outside reads zero, touches nothing of the slot,
    /// and takes no select of its value. Stores and atomics are not redirected: they run only
    /// where `inb` holds.
    pub fn outside(self, inb: Value, zero: Value) -> PsbMem {
        PsbMem {
            outside: Some((inb, zero)),
            ..self
        }
    }

    /// The same words, loaded through `NonWritable` places where `read_only`: memory nothing
    /// writes while the module runs (`AccessPlan`'s unwritten slots), so the device may read it as
    /// read-only memory.
    pub fn read_only(self, read_only: bool) -> PsbMem {
        PsbMem { read_only, ..self }
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
                vec2: false,
                vec4: false,
                ..units
            },
            ..self
        }
    }

    /// The same words, also loading and storing 8- and 16-byte units (`uvec2`, `uvec4`) at
    /// offsets proven multiples of their size, where `vectors`: a memory whose base the offsets
    /// are proven from, under the identity domains, where a slot's device address is its Metal
    /// address.
    pub fn with_vectors(self, vectors: bool) -> PsbMem {
        PsbMem {
            units: Units {
                vec2: vectors,
                vec4: vectors,
                ..self.units
            },
            ..self
        }
    }

    /// A `PhysicalStorageBuffer` place of the `bytes`-byte unit `index` (a `u32` index of such
    /// units, `bytes` 1, 2, 4, 8 or 16) of type `ty`, to load from where `load`.
    fn place_of(
        &self,
        fb: &mut FunctionBuilder<'_>,
        bytes: u32,
        index: Value,
        ty: TypeId,
        load: bool,
    ) -> BResult<Value> {
        match self.reach {
            Reach::Rounded => {
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
                self.place(fb, addr, ty, load)
            }
            Reach::Below => {
                let shift = k32(fb, bytes.trailing_zeros())?;
                let at = u32_op(fb, BinOp::ShiftLeftLogical, index, shift)?;
                let u64_ = u64_ty(fb)?;
                let d = fb.convert(ConvOp::UConvert, at, u64_)?;
                let addr = fb.binary(BinOp::IAdd, self.base, d)?;
                self.place(fb, addr, ty, load)
            }
            Reach::Direct => {
                let shift = bytes.trailing_zeros();
                let (x, c) = fb.sum_of(index).unwrap_or((index, 0));
                let s = k32(fb, shift)?;
                let at = u32_op(fb, BinOp::ShiftLeftLogical, x, s)?;
                let bias = (c << shift).wrapping_add(SLOT_ADDRESS_BIAS as u32);
                let bias = k32(fb, bias)?;
                let at = u32_op(fb, BinOp::IAdd, at, bias)?;
                let u64_ = u64_ty(fb)?;
                let d = fb.convert(ConvOp::UConvert, at, u64_)?;
                let addr = fb.binary(BinOp::IAdd, self.base, d)?;
                self.place(fb, addr, ty, load)
            }
        }
    }

    /// A `PhysicalStorageBuffer` place of the `bytes`-byte unit `index` (a `u32` index of such
    /// units) of the memory, to load from where `load`, and the unit's type.
    fn unit_ptr(
        &self,
        fb: &mut FunctionBuilder<'_>,
        bytes: u32,
        index: Value,
        load: bool,
    ) -> BResult<(Value, TypeId)> {
        let ty = fb.module().ty(Type::Int {
            width: if bytes == 1 { 8 } else { 16 },
            signed: false,
        })?;
        Ok((self.place_of(fb, bytes, index, ty, load)?, ty))
    }

    /// A `PhysicalStorageBuffer` place of word `word` (a `u32` index) of the memory, to load from
    /// where `load`.
    fn word_ptr(&self, fb: &mut FunctionBuilder<'_>, word: Value, load: bool) -> BResult<Value> {
        let u32_ = fb.module().ty(Type::Int {
            width: 32,
            signed: false,
        })?;
        self.place_of(fb, 4, word, u32_, load)
    }

    /// The place of a `pointee` at `addr`: `NonWritable` for a load of read-only memory, and a
    /// load's at the zero range where its check fails ([`PsbMem::outside`]).
    fn place(
        &self,
        fb: &mut FunctionBuilder<'_>,
        addr: Value,
        pointee: TypeId,
        load: bool,
    ) -> BResult<Value> {
        let addr = match (load, self.outside) {
            (true, Some((inb, zero))) => fb.select(inb, addr, zero)?,
            _ => addr,
        };
        match load && self.read_only {
            true => fb.device_place_read_only(addr, pointee),
            false => fb.device_place(addr, pointee),
        }
    }

    /// The `PhysicalStorageBuffer` place of the `uvecN` unit `index` (`N = lanes`, 2 or 4), to
    /// load from where `load`.
    fn vec_ptr(
        &self,
        fb: &mut FunctionBuilder<'_>,
        lanes: u32,
        index: Value,
        load: bool,
    ) -> BResult<Value> {
        let u32_ = fb.module().ty(Type::Int {
            width: 32,
            signed: false,
        })?;
        let len = if lanes == 2 { 2 } else { 4 };
        let ty = fb.module().ty(Type::Vector { elem: u32_, len })?;
        self.place_of(fb, 4 * lanes, index, ty, load)
    }

    /// The `PhysicalStorageBuffer` place of a `pointee` at word `word` (a `u32` index) of the
    /// memory, for an atomic: never `NonWritable`.
    pub fn word_place(
        &self,
        fb: &mut FunctionBuilder<'_>,
        word: Value,
        pointee: TypeId,
    ) -> BResult<Value> {
        self.place_of(fb, 4, word, pointee, false)
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
        let p = self.word_ptr(fb, word, true)?;
        let access = self.ops.access(fb, WORD, false)?;
        fb.load(p, access)
    }

    fn write(&mut self, fb: &mut FunctionBuilder<'_>, word: Value, value: Value) -> BResult<()> {
        let p = self.word_ptr(fb, word, false)?;
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
        let p = self.word_ptr(fb, word, false)?;
        let keep = fb.unary(UnOp::Not, mask)?;
        let clear = fb.binary(BinOp::BitwiseOr, keep, bits)?;
        fb.atomic(AtomicOp::And, p, Scope::Device, Semantics::RELAXED, clear)?;
        fb.atomic(AtomicOp::Or, p, Scope::Device, Semantics::RELAXED, bits)?;
        Ok(())
    }

    fn units(&self) -> Units {
        self.units
    }

    fn read_vec(
        &mut self,
        fb: &mut FunctionBuilder<'_>,
        lanes: u32,
        index: Value,
    ) -> BResult<Value> {
        let p = self.vec_ptr(fb, lanes, index, true)?;
        let access = self.ops.access(fb, unit_access(4 * lanes), false)?;
        fb.load(p, access)
    }

    fn write_vec(
        &mut self,
        fb: &mut FunctionBuilder<'_>,
        lanes: u32,
        index: Value,
        value: Value,
    ) -> BResult<()> {
        let p = self.vec_ptr(fb, lanes, index, false)?;
        let access = self.ops.access(fb, unit_access(4 * lanes), true)?;
        fb.store(p, value, access)
    }

    fn read_unit(
        &mut self,
        fb: &mut FunctionBuilder<'_>,
        bytes: u32,
        index: Value,
    ) -> BResult<Value> {
        let (p, _) = self.unit_ptr(fb, bytes, index, true)?;
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
        let (p, ty) = self.unit_ptr(fb, bytes, index, false)?;
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

#[cfg(test)]
mod tests {
    use crate::spirv::builder::constructs::{ArmEnd, ArmFn};
    use crate::spirv::builder::FunctionBuilder;
    use crate::spirv::error::BResult;
    use crate::spirv::ids::Value;
    use crate::spirv::interp::harness::{address, check_containment, AccessArgs};
    use crate::spirv::ir::{BinOp, Constant, MemAccess};

    use super::limit;

    /// The identity domains' check with no bytes before the bind offset: the access at `offset`
    /// of a buffer of `size` bytes, loaded inside `if_(offset < limit)`.
    fn limited(fb: &mut FunctionBuilder<'_>, a: &AccessArgs) -> BResult<Value> {
        let lim = limit(fb, a.size, a.n)?;
        let inb = fb.binary(BinOp::ULessThan, a.offset, lim)?;
        let null = fb.module().constant(Constant::Null { ty: a.elem })?;
        let var = fb.local_var(a.elem, Some(null))?;
        let result = fb.var_ptr(var)?;
        let args = *a;
        let then: ArmFn<'_, ()> = Box::new(move |fb, _| {
            let p = address(fb, &args)?;
            let v = fb.load(
                p,
                MemAccess {
                    aligned: Some(1),
                    ..MemAccess::default()
                },
            )?;
            fb.store(result, v, MemAccess::default())?;
            Ok(ArmEnd::FallThrough(Vec::new()))
        });
        fb.if_(&mut (), inb, then, None)?;
        fb.load(result, MemAccess::default())
    }

    /// An offset under the limit is exactly one the normative check admits: at offsets around the
    /// buffer's start and end and at `0xFFFF_FFF0..=0xFFFF_FFFF` (and the top of the 64-bit range),
    /// for `n` in {1, 2, 4, 16} in both offset widths, over buffers shorter than one access (whose
    /// limit is 0), as long as one, and longer.
    #[test]
    fn an_offset_under_the_limit_is_one_the_normative_check_admits() {
        for size in [0u64, 1, 3, 4, 15, 16, 22, 64] {
            check_containment(&limited, size).unwrap_or_else(|e| panic!("size {size}: {e}"));
        }
    }
}
