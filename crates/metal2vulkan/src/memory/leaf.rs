//! Leaf accesses and word composition.
//!
//! Every AIR `load`/`store` of a scalar or a vector is one **leaf**: `n` bytes (the type's store
//! size, so a `<3 x T>` never touches its padding) at a byte offset of a word memory
//! ([`WordMem`]: a descriptor view, later the PSB and arena forms). Memory holds the
//! little-endian bytes of the value under the AIR datalayout; the register value is the
//! arithmetic family's representation ([`Form`]): an `iN` zero-extended in a `u32` or `u64`, `i1`
//! as `bool` (one byte, `byte != 0`), floats native.
//!
//! A vector of more than four whole-byte lanes is held **chunked**: its register value is the
//! arithmetic family's composite of 4-lane chunks, the last holding the remaining 1 to 4 lanes.
//! Every chunk but the last spans `4 × lane` bytes, a whole number of words, so chunk `c` is the
//! ordinary leaf of its lanes at word `c × lane` of the vector's words ([`Leaf::chunks`]): the
//! access reads or writes the vector's words as one leaf, and only the conversion between the words
//! and the register value goes chunk by chunk.
//!
//! No narrow storage type is used: a leaf is composed from the 32-bit words covering it. The
//! **path** depends only on the access's alignment `a` ([`Path`]):
//!
//! - [`Path::Vec4`] (`n == 16` and `a >= 16`): the offset is a multiple of 16, so the leaf is
//!   `uvec4` unit `off/16` where the memory has one ([`Units::vec4`]), else [`Path::Aligned`]'s
//!   four words.
//! - [`Path::Aligned`] (`a >= 4`): the offset is a multiple of 4, so the leaf is words
//!   `off/4 .. off/4 + ceil(n/4)`.
//! - [`Path::OneWord`] (`n <= 2` and `a >= n`): the leaf lies inside one word at a dynamic byte
//!   shift.
//! - [`Path::Bytes`] (anything less aligned): the `ceil(n/4)` words at the offset's word and bit
//!   shift, and the word after them only when the leaf reaches into it (decided at translation
//!   time for a constant offset, else by a branch on the shift), each value word joined from two
//!   memory words by a funnel shift. A store writes its first and last words' leaf bytes through
//!   [`WordMem::write_bits`] and whole interior words plainly. No word without a byte of the leaf
//!   is read or written (NVIDIA faults on misaligned wide accesses, and a word beyond the leaf may
//!   be beyond the descriptor).
//!
//! Every word read or written holds at least one byte of the leaf, so an in-bounds leaf never
//! touches a word past the padded range (sizes are multiples of 16).
//!
//! **Size**. Address arithmetic on translation-time constants is folded ([`u32_op`]), so a constant
//! offset costs no instruction; a vector's lanes are converted to or from their words as one vector
//! (one bitcast, one exact narrow-float conversion, one shift placing sub-word lanes), not lane by
//! lane.
//!
//! **Sub-word stores** without a narrow unit replace only the leaf's bits of a word through
//! [`WordMem::write_bits`]: two relaxed word atomics in shared memory, `and(~mask | v)` then
//! `or(v)`, so neighbouring bytes written concurrently by other invocations are never lost and
//! no loop is emitted. Whole words are stored plainly.
//!
//! **Narrow units**. A word memory may also load and store 1- and 2-byte units through typed narrow
//! views ([`WordMem::units`]; a descriptor slot's `u8[]`/`u16[]` views, or a device address's
//! `u8`/`u16` `PhysicalStorageBuffer` places, where the profile has the storage access,
//! `memory::views` and `lower::access::psb`). There a [`Path::OneWord`] leaf of 1 or 2 bytes is its
//! one unit at index `off / bytes` (its alignment makes the offset a multiple of its size), loaded
//! or stored plainly, and the partial last word of an aligned store is stored as its units (a
//! 2-byte part at its 4-aligned start, then a 1-byte part), instead of the word's shift and the
//! sub-word store's two atomics. A unit holds exactly the leaf's bytes, so no byte outside the leaf
//! is read or written, as on the word path; [`leaf_units`] names the units an access uses. Where
//! another store to the same word still takes the sub-word path (a profile with one of the two
//! widths, a [`Path::Bytes`] leaf, a 3-byte tail without both widths), a unit store and that
//! store's atomics overlap at the unit's bytes: the Vulkan memory model calls such a pair from two
//! invocations a data race, as it does a whole-word load beside a sub-word store. The atomics
//! change no byte outside their mask, so either order keeps both stores' bytes where the device
//! performs each atomic as one read-modify-write; the model does not promise that, and no device
//! run has measured it.
//!
//! **`uvec4` units**. A word memory may also load and store 16-byte units through a typed `uvec4`
//! view ([`WordMem::read_vec4`]; a descriptor slot's `{ uvec4[] }` view, or its uniform view's own
//! vectors, `memory::views`). There a [`Path::Vec4`] leaf is one unit at index `off / 16`, its
//! register value one bitcast of the vector (four 32-bit lanes or two 64-bit ones) or the
//! composition of its four words: one load or store where the aligned path moves four words. The
//! unit holds exactly the leaf's bytes, so no byte outside the leaf is read or written, as on the
//! word path.
//!
//! **Volatile**. A volatile access is the same leaf through a word memory whose [`WordOps`] carry
//! `volatile`: every word or unit it loads or stores has the `Volatile` memory operand.
//!
//! The alignment is a promise about the Metal address; it carries to the offset in a word memory
//! because every word memory's base is 4-byte aligned in the Metal address space (the binder's
//! descriptor bases are), and to a `uvec4` unit's index because a descriptor view's base is 16-byte
//! aligned there.

use crate::air::ir::{Type, TypeId as AirTypeId, TypeLayouts};
use crate::air::Entry;
use crate::device::DeviceProfile;
use crate::error::{Error, Feature, LowerFamily, Unsupported};
use crate::lower::arith::legal::{self, wide_form_on};
use crate::lower::arith::narrow;
use crate::lower::arith::ops;
use crate::lower::arith::wide;
use crate::lower::arith::{FloatFmt, Form, Held};
use crate::spirv::builder::constructs::{ArmEnd, ArmFn};
use crate::spirv::builder::FunctionBuilder;
use crate::spirv::error::{BResult, BuildError, Dangling};
use crate::spirv::ids::{TypeId, Value};
use crate::spirv::ir::{BinOp, Constant, ConvOp, MemAccess, Scope, TexelMemory, Type as SType};

/// How a word access of shared memory takes part in the memory model.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Coherence {
    /// The `GLSL450` model: plain memory operands.
    #[default]
    Plain,
    /// The Vulkan model: `NonPrivatePointer`, so barriers and fences order the access with
    /// other invocations' as they do under `GLSL450` (Metal's default for shared memory).
    NonPrivate,
    /// A `*_coherent` access: `NonPrivatePointer` with `MakePointerVisible` (a load) or
    /// `MakePointerAvailable` (a store) at `Device` scope.
    Device,
}

impl Coherence {
    /// `base` with this coherence's memory operands, for a load or (`store`) a store.
    pub fn access(
        self,
        fb: &mut FunctionBuilder<'_>,
        base: MemAccess,
        store: bool,
    ) -> BResult<MemAccess> {
        match self {
            Coherence::Plain => Ok(base),
            Coherence::NonPrivate => Ok(MemAccess {
                non_private: true,
                ..base
            }),
            Coherence::Device => {
                let scope = fb.scope_id(Scope::Device)?;
                Ok(MemAccess {
                    make_available: store.then_some(scope),
                    make_visible: (!store).then_some(scope),
                    non_private: true,
                    ..base
                })
            }
        }
    }

    /// This coherence's texel operands for a storage-image read or write: none for
    /// [`Coherence::Plain`], `NonPrivateTexel` under the Vulkan model, and with it
    /// `MakeTexelVisible` (a read) or `MakeTexelAvailable` (a write) at `Device` scope for
    /// [`Coherence::Device`] (the access's opcode decides which).
    pub fn texel(self, fb: &mut FunctionBuilder<'_>) -> BResult<TexelMemory> {
        Ok(match self {
            Coherence::Plain => TexelMemory::PRIVATE,
            Coherence::NonPrivate => TexelMemory::NonPrivate { volatile: false },
            Coherence::Device => TexelMemory::made(fb.scope_id(Scope::Device)?, false),
        })
    }
}

/// The memory operands of a word memory's loads and stores: its [`Coherence`], and `Volatile` on
/// the words of a volatile access. `Volatile` is the same operand under both memory models: the
/// access is performed, once, and never combined with another; it adds no availability or
/// visibility operation and does not make the access non-private.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct WordOps {
    pub coherence: Coherence,
    pub volatile: bool,
}

impl WordOps {
    /// The operands of a plain access with `coherence`.
    pub fn of(coherence: Coherence) -> WordOps {
        WordOps {
            coherence,
            volatile: false,
        }
    }

    /// `base` with these memory operands, for a load or (`store`) a store.
    pub fn access(
        self,
        fb: &mut FunctionBuilder<'_>,
        base: MemAccess,
        store: bool,
    ) -> BResult<MemAccess> {
        let base = MemAccess {
            volatile: base.volatile || self.volatile,
            ..base
        };
        self.coherence.access(fb, base, store)
    }
}

fn not_lowered() -> Error {
    Unsupported::not_implemented(Feature::Lowering(LowerFamily::Access)).into()
}

/// How a leaf is composed from words.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Path {
    /// A 16-byte leaf at an offset that is a multiple of 16: one `uvec4` unit where the memory
    /// has one ([`Units::vec4`]), else [`Path::Aligned`]'s four words.
    Vec4,
    /// The offset is a multiple of 4.
    Aligned,
    /// A 1- or 2-byte leaf inside one word.
    OneWord,
    /// Joined from the words the leaf's bytes touch at a dynamic shift.
    Bytes,
}

/// One leaf access.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Leaf {
    /// The register form of the value. More than four lanes (only [`leaf_of`] gives them) is a
    /// chunked vector, whose register value is its chunk composite ([`Leaf::chunks`]).
    pub form: Form,
    /// Bytes in memory (the store size).
    pub bytes: u32,
    pub path: Path,
}

impl Leaf {
    /// The words the leaf's bytes occupy when composed from offset 0.
    pub fn words(&self) -> u32 {
        self.bytes.div_ceil(4)
    }

    /// Whether the value is a chunked vector.
    pub fn chunked(&self) -> bool {
        self.form.lanes > 4
    }

    /// The chunks of a chunked leaf, in order: each chunk's own leaf and the index of its first
    /// word among the vector's words. A full chunk holds four lanes of whole bytes, so every
    /// chunk starts on a word, and the chunks' words together are the vector's words.
    pub fn chunks(&self) -> Vec<(Leaf, usize)> {
        let lane = u32::from(self.form.bits() / 8);
        legal::chunk_lanes(u32::from(self.form.lanes))
            .into_iter()
            .enumerate()
            .map(|(c, k)| {
                let leaf = Leaf {
                    form: Form {
                        lanes: k,
                        ..self.form
                    },
                    bytes: u32::from(k) * lane,
                    // A chunk is shorter than the vector: its words start on a word.
                    path: match self.path {
                        Path::Vec4 => Path::Aligned,
                        p => p,
                    },
                };
                (leaf, c * lane as usize)
            })
            .collect()
    }

    /// An upper estimate of the SPIR-V words one access emits, before containment: about five words
    /// per instruction, with a sub-word store's two atomics on every word.
    pub fn cost(&self) -> u32 {
        let per = match self.path {
            Path::Vec4 | Path::Aligned => 48 * self.words(),
            Path::OneWord => 96,
            Path::Bytes => 96 * self.bytes,
        };
        // Conversion between the words and the register form: a few instructions per lane, plus
        // the exact format-bit sequence of a narrow float held widened.
        let lane = 32
            + narrow::bits_words(Form {
                lanes: 1,
                ..self.form
            });
        // A chunked vector also constructs or extracts each chunk.
        let chunks = if self.chunked() {
            8 * u32::from(self.form.lanes).div_ceil(4)
        } else {
            0
        };
        per + lane * u32::from(self.form.lanes) + 32 + chunks
    }
}

/// The leaf of an access of AIR type `ty` with alignment `align` bytes (1 when unstated), or the
/// refusal of a type that is not a scalar or vector the arithmetic family holds on `p`. A
/// vector's lanes must be whole bytes (`<N x i1>` and odd widths are bit-packed in LLVM); one of
/// 5 to 255 lanes is chunked, and a `<1 x T>` (a chunked composite of one scalar) or a vector of
/// more lanes is not implemented.
pub fn leaf_of(
    e: &Entry,
    p: &DeviceProfile,
    layouts: &TypeLayouts,
    ty: AirTypeId,
    align: u64,
) -> Result<Leaf, Error> {
    // A 64-bit integer is two words on a profile without `shaderInt64`, in a chunk too.
    let form = match legal::chunked(e.module.types(), ty) {
        Some((held, lanes)) => Form {
            held: legal::held_on(p, held)?,
            lanes: u8::try_from(lanes)
                .ok()
                .filter(|l| *l > 4)
                .ok_or_else(not_lowered)?,
        },
        None => wide_form_on(e, p, ty)?,
    };
    let bytes = layouts.store_size(ty).ok_or_else(not_lowered)?;
    let bytes = u32::try_from(bytes).map_err(|_| not_lowered())?;
    if form.lanes > 1 {
        let lane_bits = match form.held {
            Held::Int { bits, .. } | Held::Wide { bits } => bits,
            Held::Float(f) => f.bits(),
            Held::Bool => return Err(not_lowered()),
        };
        if !matches!(lane_bits, 8 | 16 | 32 | 64)
            || u32::from(lane_bits / 8) * u32::from(form.lanes) != bytes
        {
            return Err(not_lowered());
        }
    } else {
        let bits = form.bits();
        if bytes == 0
            || bytes > 8
            || (form.held != Held::Bool && u32::from(bits).div_ceil(8) != bytes)
        {
            return Err(not_lowered());
        }
    }
    // Refuse a non-scalar AIR type that `form_on` accepted (it only accepts scalars and vectors).
    if !matches!(
        e.module.types().get(ty),
        Some(
            Type::Int(_)
                | Type::Half
                | Type::BFloat
                | Type::Float
                | Type::Double
                | Type::Vector { .. }
        )
    ) {
        return Err(not_lowered());
    }
    let path = if bytes == 16 && align >= 16 {
        Path::Vec4
    } else if align >= 4 {
        Path::Aligned
    } else if bytes <= 2 && u64::from(bytes) <= align {
        Path::OneWord
    } else {
        Path::Bytes
    };
    Ok(Leaf { form, bytes, path })
}

/// The units a word memory also loads and stores directly: narrow 1- and 2-byte units and 16-byte
/// `uvec4` units.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Units {
    /// 2-byte units (a `u16` view).
    pub u16: bool,
    /// 1-byte units (a `u8` view).
    pub u8: bool,
    /// 16-byte units (a `uvec4` view).
    pub vec4: bool,
}

impl Units {
    /// No unit: every access goes through words.
    pub const NONE: Units = Units {
        u16: false,
        u8: false,
        vec4: false,
    };

    /// Whether `bytes`-byte units (1, 2, or 16) are available.
    pub fn has(self, bytes: u32) -> bool {
        match bytes {
            1 => self.u8,
            2 => self.u16,
            16 => self.vec4,
            _ => false,
        }
    }

    /// Both sets' units.
    pub fn union(self, other: Units) -> Units {
        Units {
            u16: self.u16 || other.u16,
            u8: self.u8 || other.u8,
            vec4: self.vec4 || other.vec4,
        }
    }

    /// The units both sets have.
    pub fn intersect(self, other: Units) -> Units {
        Units {
            u16: self.u16 && other.u16,
            u8: self.u8 && other.u8,
            vec4: self.vec4 && other.vec4,
        }
    }

    /// Whether no unit is set.
    pub fn is_empty(self) -> bool {
        !self.u16 && !self.u8 && !self.vec4
    }

    /// The units profile `p` can declare as storage-buffer views: a narrow width needs its
    /// storage access and, since requirement derivation asks a declared 8- or 16-bit integer type
    /// for its arithmetic capability (`spirv::requirements`), its integer feature; a `uvec4` view
    /// needs nothing.
    pub fn storage_on(p: &DeviceProfile) -> Units {
        let l = &p.fields().leaf;
        Units {
            u16: l.storage_buffer_16bit && l.int16,
            u8: l.storage_buffer_8bit && l.int8,
            vec4: true,
        }
    }
}

/// The units an access of `leaf` (a `store` or a load) uses where its memory has them: a
/// [`Path::Vec4`] leaf's `uvec4` unit, a [`Path::OneWord`] leaf's one narrow unit of its size, and
/// an aligned store's partial last word's 2-byte and 1-byte parts. A load of an aligned leaf reads
/// whole words, and a [`Path::Bytes`] leaf joins words, so neither uses one.
pub fn leaf_units(leaf: &Leaf, store: bool) -> Units {
    match leaf.path {
        Path::Vec4 => Units {
            vec4: true,
            ..Units::NONE
        },
        Path::OneWord => Units {
            u16: leaf.bytes == 2,
            u8: leaf.bytes == 1,
            vec4: false,
        },
        Path::Aligned if store => {
            let tail = leaf.bytes % 4;
            Units {
                u16: tail >= 2,
                u8: tail % 2 == 1,
                vec4: false,
            }
        }
        Path::Aligned | Path::Bytes => Units::NONE,
    }
}

/// A memory of 32-bit words addressed by `u32` word index.
pub trait WordMem {
    /// Load word `word`.
    fn read(&mut self, fb: &mut FunctionBuilder<'_>, word: Value) -> BResult<Value>;
    /// Store the whole word `word`.
    fn write(&mut self, fb: &mut FunctionBuilder<'_>, word: Value, value: Value) -> BResult<()>;
    /// Replace the bits `mask` of word `word` by `bits` (which has no bit outside `mask`),
    /// leaving every other bit as another invocation may concurrently write it.
    fn write_bits(
        &mut self,
        fb: &mut FunctionBuilder<'_>,
        word: Value,
        mask: Value,
        bits: Value,
    ) -> BResult<()>;
    /// The narrow units this memory also loads and stores directly; none unless it has typed narrow
    /// views.
    fn units(&self) -> Units {
        Units::NONE
    }
    /// Load the `bytes`-byte unit `index` (a `u32` index of such units), zero-extended to a `u32`.
    /// Called only for a unit [`WordMem::units`] has.
    fn read_unit(
        &mut self,
        _fb: &mut FunctionBuilder<'_>,
        _bytes: u32,
        _index: Value,
    ) -> BResult<Value> {
        Err(dangling())
    }
    /// Store the low `8 · bytes` bits of `value` (a `u32`) to the `bytes`-byte unit `index`.
    /// Called only for a unit [`WordMem::units`] has.
    fn write_unit(
        &mut self,
        _fb: &mut FunctionBuilder<'_>,
        _bytes: u32,
        _index: Value,
        _value: Value,
    ) -> BResult<()> {
        Err(dangling())
    }
    /// Load the `uvec4` unit `index` (a `u32` index of 16-byte units): the four little-endian
    /// words of its bytes. Called only where [`WordMem::units`] has `vec4`.
    fn read_vec4(&mut self, _fb: &mut FunctionBuilder<'_>, _index: Value) -> BResult<Value> {
        Err(dangling())
    }
    /// Store the `uvec4` `value` to the 16-byte unit `index`. Called only where [`WordMem::units`]
    /// has `vec4`.
    fn write_vec4(
        &mut self,
        _fb: &mut FunctionBuilder<'_>,
        _index: Value,
        _value: Value,
    ) -> BResult<()> {
        Err(dangling())
    }
}

/// A `u32` constant.
pub fn k32(fb: &mut FunctionBuilder<'_>, v: u32) -> BResult<Value> {
    let c = fb.u32_const(v)?;
    fb.module().const_value(c)
}

/// A `uvecN` constant of `lanes` (2 to 4 of them).
fn kvec(fb: &mut FunctionBuilder<'_>, lanes: &[u32]) -> BResult<Value> {
    let u = u32_ty(fb)?;
    let len = u8::try_from(lanes.len()).map_err(|_| dangling())?;
    let ty = fb.module().ty(SType::Vector { elem: u, len })?;
    let mut parts = Vec::with_capacity(lanes.len());
    for l in lanes {
        parts.push(fb.u32_const(*l)?);
    }
    let c = fb.module().constant(Constant::Composite {
        ty,
        parts: parts.into(),
    })?;
    fb.module().const_value(c)
}

/// The value of `v` when it is a plain `u32` constant: address arithmetic on values known at
/// translation time is done then, not emitted.
pub fn known(fb: &mut FunctionBuilder<'_>, v: Value) -> BResult<Option<u32>> {
    let Some(bits) = fb.const_scalar(v)? else {
        return Ok(None);
    };
    let ty = fb.value_ty(v)?;
    let is_u32 = matches!(
        fb.module().types().get(ty)?,
        SType::Int {
            width: 32,
            signed: false
        }
    );
    Ok(if is_u32 {
        u32::try_from(bits).ok()
    } else {
        None
    })
}

/// `a op b` of two known `u32`s, for the operations whose result SPIR-V defines for every
/// operand (a shift by 32 or more is undefined, so it is never folded).
fn fold(op: BinOp, a: u32, b: u32) -> Option<u32> {
    Some(match op {
        BinOp::IAdd => a.wrapping_add(b),
        BinOp::ISub => a.wrapping_sub(b),
        BinOp::IMul => a.wrapping_mul(b),
        BinOp::BitwiseAnd => a & b,
        BinOp::BitwiseOr => a | b,
        BinOp::BitwiseXor => a ^ b,
        BinOp::ShiftLeftLogical if b < 32 => a << b,
        BinOp::ShiftRightLogical if b < 32 => a >> b,
        _ => return None,
    })
}

/// `a op b` on `u32` values: a constant when both are known, an operand when the other is the
/// operation's identity (or its absorbing zero), and the instruction otherwise.
pub fn u32_op(fb: &mut FunctionBuilder<'_>, op: BinOp, a: Value, b: Value) -> BResult<Value> {
    let (x, y) = (known(fb, a)?, known(fb, b)?);
    if let (Some(x), Some(y)) = (x, y) {
        if let Some(r) = fold(op, x, y) {
            return k32(fb, r);
        }
    }
    let shift = matches!(op, BinOp::ShiftLeftLogical | BinOp::ShiftRightLogical);
    match (op, x, y) {
        (BinOp::IAdd | BinOp::ISub | BinOp::BitwiseOr | BinOp::BitwiseXor, _, Some(0)) => Ok(a),
        (BinOp::IAdd | BinOp::BitwiseOr | BinOp::BitwiseXor, Some(0), _) => Ok(b),
        (_, _, Some(0)) if shift => Ok(a),
        (_, Some(0), _) if shift => Ok(a),
        (BinOp::IMul, _, Some(1)) | (BinOp::BitwiseAnd, _, Some(u32::MAX)) => Ok(a),
        (BinOp::IMul, Some(1), _) | (BinOp::BitwiseAnd, Some(u32::MAX), _) => Ok(b),
        (BinOp::IMul | BinOp::BitwiseAnd, Some(0), _)
        | (BinOp::IMul | BinOp::BitwiseAnd, _, Some(0)) => k32(fb, 0),
        _ => fb.binary(op, a, b),
    }
}

fn u32_ty(fb: &mut FunctionBuilder<'_>) -> BResult<TypeId> {
    fb.module().ty(SType::Int {
        width: 32,
        signed: false,
    })
}

fn uvec_ty(fb: &mut FunctionBuilder<'_>, len: u8) -> BResult<TypeId> {
    let u = u32_ty(fb)?;
    fb.module().ty(SType::Vector { elem: u, len })
}

/// `x >> by` (logical), omitted for 0.
fn shr(fb: &mut FunctionBuilder<'_>, x: Value, by: u32) -> BResult<Value> {
    let s = k32(fb, by)?;
    u32_op(fb, BinOp::ShiftRightLogical, x, s)
}

/// `x << by`, omitted for 0.
fn shl(fb: &mut FunctionBuilder<'_>, x: Value, by: u32) -> BResult<Value> {
    let s = k32(fb, by)?;
    u32_op(fb, BinOp::ShiftLeftLogical, x, s)
}

fn and(fb: &mut FunctionBuilder<'_>, x: Value, m: u32) -> BResult<Value> {
    let m = k32(fb, m)?;
    u32_op(fb, BinOp::BitwiseAnd, x, m)
}

fn or(fb: &mut FunctionBuilder<'_>, a: Value, b: Value) -> BResult<Value> {
    u32_op(fb, BinOp::BitwiseOr, a, b)
}

/// The low `bytes` bytes set (`bytes <= 4`).
fn byte_mask(bytes: u32) -> u32 {
    if bytes >= 4 {
        u32::MAX
    } else {
        (1u32 << (8 * bytes)) - 1
    }
}

/// The 64-bit value of words `lo`, `hi`, bitcast to `to` (a 64-bit scalar).
fn join64(fb: &mut FunctionBuilder<'_>, lo: Value, hi: Value, to: TypeId) -> BResult<Value> {
    let v2 = uvec_ty(fb, 2)?;
    let pair = fb.composite_construct(v2, &[lo, hi])?;
    fb.convert(ConvOp::Bitcast, pair, to)
}

/// The two words of a 64-bit scalar.
fn split64(fb: &mut FunctionBuilder<'_>, x: Value) -> BResult<(Value, Value)> {
    let v2 = uvec_ty(fb, 2)?;
    let pair = fb.convert(ConvOp::Bitcast, x, v2)?;
    Ok((
        fb.composite_extract(pair, &[0])?,
        fb.composite_extract(pair, &[1])?,
    ))
}

/// The SPIR-V type of the chunked vector of `form`'s element and lanes.
fn chunked_ty(fb: &mut FunctionBuilder<'_>, form: Form) -> BResult<TypeId> {
    let d = legal::chunked_demote(form.held, u32::from(form.lanes));
    legal::spirv_type(fb.module(), &d)?.ok_or_else(dangling)
}

/// A word index outside the leaf: an internal invariant failure, reported as a dangling value.
fn dangling() -> BuildError {
    BuildError::Dangling(Dangling::Value(Value::new(u32::MAX)))
}

/// Word `i` of `words`.
fn word(words: &[Value], i: usize) -> BResult<Value> {
    words.get(i).copied().ok_or_else(dangling)
}

/// The bytes of one lane of a vector leaf (its lanes are whole bytes, [`leaf_of`]).
fn lane_bytes(form: Form) -> u32 {
    u32::from(form.bits() / 8)
}

/// Lane `i` of a vector leaf sits at byte `at(i)`: the word holding it and the bit shift of its
/// first byte in that word.
fn lane_at(form: Form, i: u32) -> (usize, u32) {
    let at = i * lane_bytes(form);
    ((at / 4) as usize, 8 * (at % 4))
}

/// The register value of `leaf` from its little-endian `words` (`leaf.words()` of them; bytes
/// past `leaf.bytes` may hold anything).
pub fn value_of(fb: &mut FunctionBuilder<'_>, leaf: &Leaf, words: &[Value]) -> BResult<Value> {
    let form = leaf.form;
    if leaf.chunked() {
        let mut parts = Vec::new();
        for (chunk, first) in leaf.chunks() {
            let ws = words
                .get(first..first + chunk.words() as usize)
                .ok_or_else(dangling)?;
            parts.push(value_of(fb, &chunk, ws)?);
        }
        let ty = chunked_ty(fb, form)?;
        return fb.composite_construct(ty, &parts);
    }
    if form.is_wide() {
        // Two words per lane, low first.
        return wide::from_words(fb, form, words);
    }
    if form.lanes > 1 {
        if let Some(v) = vector_of(fb, form, words)? {
            return Ok(v);
        }
    }
    let scalar = Form { lanes: 1, ..form };
    let elem = ops::elem_ty(fb, form.held)?;
    let lane = |fb: &mut FunctionBuilder<'_>, i: u32| -> BResult<Value> {
        match form.held {
            Held::Bool => {
                let b = and(fb, word(words, 0)?, 0xFF)?;
                let zero = k32(fb, 0)?;
                fb.binary(BinOp::INotEqual, b, zero)
            }
            Held::Int {
                bits,
                container: 32,
            } => {
                let lane_bytes = if form.lanes == 1 {
                    leaf.bytes
                } else {
                    u32::from(bits / 8)
                };
                let at = i * lane_bytes;
                let w = word(words, (at / 4) as usize)?;
                let x = shr(fb, w, 8 * (at % 4))?;
                if bits >= 32 {
                    Ok(x)
                } else {
                    and(fb, x, (1u32 << bits) - 1)
                }
            }
            Held::Int { bits, .. } => {
                // A 64-bit container: an `i33..i64` scalar or an `i64` lane.
                let at = (i * 8 / 4) as usize;
                let lo = word(words, at)?;
                let hi = match words.get(at + 1) {
                    Some(h) => *h,
                    None => k32(fb, 0)?,
                };
                let hi_bits = u32::from(bits.saturating_sub(32));
                let hi = if hi_bits >= 32 {
                    hi
                } else {
                    and(fb, hi, (1u32 << hi_bits) - 1)?
                };
                join64(fb, lo, hi, elem)
            }
            Held::Wide { .. } => Err(dangling()),
            Held::Float(FloatFmt::Half | FloatFmt::BFloat) => {
                // A narrow float is held widened: its 16 format bits (`from_bits` reads only the
                // low 16 bits), a `half` widened from its word where the device widens it.
                let at = 2 * i;
                let w = (at / 4) as usize;
                let h = shr(fb, word(words, w)?, 8 * (at % 4))?;
                let tail = words.get(w..).ok_or_else(dangling)?;
                narrow::from_word_halves(fb, scalar, tail, (at % 4) / 2, h)
            }
            Held::Float(FloatFmt::F64) => {
                let lo = word(words, (2 * i) as usize)?;
                let hi = word(words, (2 * i + 1) as usize)?;
                join64(fb, lo, hi, elem)
            }
            Held::Float(FloatFmt::F32) => {
                let w = word(words, i as usize)?;
                fb.convert(ConvOp::Bitcast, w, elem)
            }
        }
    };
    if form.lanes == 1 {
        return lane(fb, 0);
    }
    let mut lanes = Vec::with_capacity(usize::from(form.lanes));
    for i in 0..u32::from(form.lanes) {
        lanes.push(lane(fb, i)?);
    }
    let ty = ops::ty(fb, form)?;
    fb.composite_construct(ty, &lanes)
}

/// A vector leaf's register value, its lanes converted as one vector: each lane's word gathered
/// into a `uvecN`, then one shift and mask for sub-word lanes, one exact narrow-float decode, or
/// one bitcast. `None` for the lane types converted lane by lane (64-bit lanes of a 3- or 4-lane
/// vector, whose words do not fit one SPIR-V vector).
fn vector_of(fb: &mut FunctionBuilder<'_>, form: Form, words: &[Value]) -> BResult<Option<Value>> {
    let n = form.lanes;
    let gathered = |fb: &mut FunctionBuilder<'_>| -> BResult<(Value, Vec<u32>)> {
        let mut src = Vec::with_capacity(usize::from(n));
        let mut shifts = Vec::with_capacity(usize::from(n));
        for i in 0..u32::from(n) {
            let (w, sh) = lane_at(form, i);
            src.push(word(words, w)?);
            shifts.push(sh);
        }
        let ty = uvec_ty(fb, n)?;
        Ok((fb.composite_construct(ty, &src)?, shifts))
    };
    // The low `bits` of each lane of `v`, moved down from their shifts.
    let low_bits = |fb: &mut FunctionBuilder<'_>, v: Value, shifts: &[u32], bits: u8| {
        let v = if shifts.iter().all(|s| *s == 0) {
            v
        } else {
            let s = kvec(fb, shifts)?;
            fb.binary(BinOp::ShiftRightLogical, v, s)?
        };
        if bits >= 32 {
            return Ok(v);
        }
        let u = Form {
            held: Held::Int {
                bits: 32,
                container: 32,
            },
            lanes: n,
        };
        let m = ops::splat(fb, u, (1u64 << bits) - 1)?;
        fb.binary(BinOp::BitwiseAnd, v, m)
    };
    Ok(Some(match form.held {
        Held::Int {
            bits,
            container: 32,
        } => {
            let (v, shifts) = gathered(fb)?;
            low_bits(fb, v, &shifts, bits)?
        }
        Held::Float(FloatFmt::Half | FloatFmt::BFloat) => {
            // A narrow float is held widened: one decode of every lane's bits, which reads only
            // each lane's low 16 bits, or a `half` vector widened from its words at once where the
            // device widens it (its lanes are the words' halves in order).
            let (v, shifts) = gathered(fb)?;
            let h = low_bits(fb, v, &shifts, 32)?;
            narrow::from_word_halves(fb, form, words, 0, h)?
        }
        Held::Float(FloatFmt::F32) => {
            let (v, _) = gathered(fb)?;
            let ty = ops::ty(fb, form)?;
            fb.convert(ConvOp::Bitcast, v, ty)?
        }
        Held::Int { container: 64, .. } | Held::Float(FloatFmt::F64) if n == 2 => {
            let ty = uvec_ty(fb, 4)?;
            let mut src = Vec::with_capacity(4);
            for i in 0..4 {
                src.push(word(words, i)?);
            }
            let v = fb.composite_construct(ty, &src)?;
            let ty = ops::ty(fb, form)?;
            fb.convert(ConvOp::Bitcast, v, ty)?
        }
        _ => return Ok(None),
    }))
}

/// The little-endian words of register value `v` of `leaf` (`leaf.words()` of them; bytes past
/// `leaf.bytes` are zero).
pub fn words_of(fb: &mut FunctionBuilder<'_>, leaf: &Leaf, v: Value) -> BResult<Vec<Value>> {
    let form = leaf.form;
    if leaf.chunked() {
        let mut out = Vec::with_capacity(leaf.words() as usize);
        for (c, (chunk, _)) in leaf.chunks().into_iter().enumerate() {
            let x = fb.composite_extract(v, &[c as u32])?;
            out.extend(words_of(fb, &chunk, x)?);
        }
        return Ok(out);
    }
    if form.is_wide() {
        // Its words are the value's; an `i33..i63` high word covers fewer bytes.
        let mut ws = wide::to_words(fb, form, v)?;
        ws.truncate(leaf.words() as usize);
        return Ok(ws);
    }
    if form.lanes > 1 {
        if let Some(ws) = vector_words(fb, leaf, v)? {
            return Ok(ws);
        }
    }
    let scalar = Form { lanes: 1, ..form };
    let n = leaf.words() as usize;
    let lane_of = |fb: &mut FunctionBuilder<'_>, i: u32| -> BResult<Value> {
        if form.lanes == 1 {
            Ok(v)
        } else {
            fb.composite_extract(v, &[i])
        }
    };
    let mut out: Vec<Option<Value>> = vec![None; n];
    for i in 0..u32::from(form.lanes) {
        let x = lane_of(fb, i)?;
        match form.held {
            Held::Bool => {
                let one = k32(fb, 1)?;
                let zero = k32(fb, 0)?;
                let b = fb.select(x, one, zero)?;
                put(fb, &mut out, 0, b)?;
            }
            Held::Int {
                bits,
                container: 32,
            } => {
                let lane_bytes = if form.lanes == 1 {
                    leaf.bytes
                } else {
                    u32::from(bits / 8)
                };
                put(fb, &mut out, i * lane_bytes, x)?;
            }
            Held::Wide { .. } => return Err(dangling()),
            Held::Int { .. } | Held::Float(FloatFmt::F64) => {
                let (lo, hi) = split64(fb, x)?;
                put(fb, &mut out, 8 * i, lo)?;
                if 8 * i + 4 < leaf.bytes {
                    put(fb, &mut out, 8 * i + 4, hi)?;
                }
            }
            Held::Float(FloatFmt::Half | FloatFmt::BFloat) => {
                // A narrow float is held widened: its 16 format bits, a `half`'s by the device's
                // narrowing (`narrow::to_bits`).
                let w = narrow::to_bits(fb, scalar, x)?;
                put(fb, &mut out, 2 * i, w)?;
            }
            Held::Float(FloatFmt::F32) => {
                let u = u32_ty(fb)?;
                let w = fb.convert(ConvOp::Bitcast, x, u)?;
                put(fb, &mut out, 4 * i, w)?;
            }
        }
    }
    filled(fb, out)
}

/// OR `x` (a `u32` whose bits sit at byte `at % 4` of word `at / 4`) into `out`.
fn put(fb: &mut FunctionBuilder<'_>, out: &mut [Option<Value>], at: u32, x: Value) -> BResult<()> {
    let x = shl(fb, x, 8 * (at % 4))?;
    match out.get_mut((at / 4) as usize) {
        Some(Some(prev)) => {
            let merged = or(fb, *prev, x)?;
            *prev = merged;
        }
        Some(s @ None) => *s = Some(x),
        None => return Err(dangling()),
    }
    Ok(())
}

/// The words of `out`, a word no lane reached being zero.
fn filled(fb: &mut FunctionBuilder<'_>, out: Vec<Option<Value>>) -> BResult<Vec<Value>> {
    let mut words = Vec::with_capacity(out.len());
    for w in out {
        words.push(match w {
            Some(w) => w,
            None => k32(fb, 0)?,
        });
    }
    Ok(words)
}

/// A vector leaf's words, its lanes converted as one vector: one bitcast, one exact narrow-float
/// encode, or one shift placing every sub-word lane at its byte, then the lanes extracted and
/// merged into their words. `None` as for [`vector_of`].
fn vector_words(
    fb: &mut FunctionBuilder<'_>,
    leaf: &Leaf,
    v: Value,
) -> BResult<Option<Vec<Value>>> {
    let form = leaf.form;
    let n = form.lanes;
    // `u` (a `uvecN` of zero-extended lane bits) placed at each lane's byte and merged per word.
    let placed = |fb: &mut FunctionBuilder<'_>, u: Value| -> BResult<Vec<Value>> {
        let shifts: Vec<u32> = (0..u32::from(n)).map(|i| lane_at(form, i).1).collect();
        let u = if shifts.iter().all(|s| *s == 0) {
            u
        } else {
            let s = kvec(fb, &shifts)?;
            fb.binary(BinOp::ShiftLeftLogical, u, s)?
        };
        let mut out: Vec<Option<Value>> = vec![None; leaf.words() as usize];
        for i in 0..u32::from(n) {
            let x = fb.composite_extract(u, &[i])?;
            let (w, _) = lane_at(form, i);
            match out.get_mut(w) {
                Some(Some(prev)) => {
                    let merged = or(fb, *prev, x)?;
                    *prev = merged;
                }
                Some(s @ None) => *s = Some(x),
                None => return Err(dangling()),
            }
        }
        filled(fb, out)
    };
    Ok(Some(match form.held {
        Held::Int { container: 32, .. } => placed(fb, v)?,
        Held::Float(FloatFmt::Half | FloatFmt::BFloat) => {
            // A narrow float is held widened: one encode of every lane, a `half`'s by the
            // device's narrowing (`narrow::to_bits`).
            let bits = narrow::to_bits(fb, form, v)?;
            placed(fb, bits)?
        }
        Held::Float(FloatFmt::F32) => {
            let ty = uvec_ty(fb, n)?;
            let u = fb.convert(ConvOp::Bitcast, v, ty)?;
            placed(fb, u)?
        }
        Held::Int { container: 64, .. } | Held::Float(FloatFmt::F64) if n == 2 => {
            let ty = uvec_ty(fb, 4)?;
            let u = fb.convert(ConvOp::Bitcast, v, ty)?;
            let mut out = Vec::with_capacity(4);
            for i in 0..4 {
                out.push(fb.composite_extract(u, &[i])?);
            }
            out
        }
        _ => return Ok(None),
    }))
}

/// Whether a 16-byte leaf's register value is one `uvec4`'s bits: four 32-bit lanes or two
/// 64-bit ones, held natively (not chunked, nor as word pairs).
fn vec4_bits(leaf: &Leaf) -> bool {
    let form = leaf.form;
    !leaf.chunked()
        && !form.is_wide()
        && matches!(
            (form.held, form.lanes),
            (
                Held::Int {
                    bits: 32,
                    container: 32
                } | Held::Float(FloatFmt::F32),
                4
            ) | (
                Held::Int { container: 64, .. } | Held::Float(FloatFmt::F64),
                2
            )
        )
}

/// `v` as type `to` of the same bits: itself, or one bitcast.
fn as_bits(fb: &mut FunctionBuilder<'_>, v: Value, to: TypeId) -> BResult<Value> {
    if fb.value_ty(v)? == to {
        return Ok(v);
    }
    fb.convert(ConvOp::Bitcast, v, to)
}

/// The register value of 16-byte `leaf` from `v`, the `uvec4` of its little-endian words: one
/// bitcast where its bits are the vector's ([`vec4_bits`]), else [`value_of`] of the four words.
fn value_of_vec4(fb: &mut FunctionBuilder<'_>, leaf: &Leaf, v: Value) -> BResult<Value> {
    if vec4_bits(leaf) {
        let ty = ops::ty(fb, leaf.form)?;
        return as_bits(fb, v, ty);
    }
    let mut words = Vec::with_capacity(4);
    for i in 0..4 {
        words.push(fb.composite_extract(v, &[i])?);
    }
    value_of(fb, leaf, &words)
}

/// The `uvec4` of 16-byte `leaf`'s register value `v`, its little-endian words: one bitcast where
/// its bits are the vector's ([`vec4_bits`]), else [`words_of`] gathered.
fn vec4_of(fb: &mut FunctionBuilder<'_>, leaf: &Leaf, v: Value) -> BResult<Value> {
    let ty = uvec_ty(fb, 4)?;
    if vec4_bits(leaf) {
        return as_bits(fb, v, ty);
    }
    let words = words_of(fb, leaf, v)?;
    fb.composite_construct(ty, &words)
}

/// `off + c` in `u32`, omitted for 0.
fn add(fb: &mut FunctionBuilder<'_>, off: Value, c: u32) -> BResult<Value> {
    let c = k32(fb, c)?;
    u32_op(fb, BinOp::IAdd, off, c)
}

/// The index of the `bytes`-byte unit (1 or 2) at byte offset `off`, a multiple of `bytes`.
fn unit_index(fb: &mut FunctionBuilder<'_>, off: Value, bytes: u32) -> BResult<Value> {
    shr(fb, off, bytes / 2)
}

/// `(word index, bit shift)` of byte offset `off`.
fn locate(fb: &mut FunctionBuilder<'_>, off: Value) -> BResult<(Value, Value)> {
    let two = k32(fb, 2)?;
    let three = k32(fb, 3)?;
    let w = u32_op(fb, BinOp::ShiftRightLogical, off, two)?;
    let b = u32_op(fb, BinOp::BitwiseAnd, off, three)?;
    let s = u32_op(fb, BinOp::ShiftLeftLogical, b, three)?;
    Ok((w, s))
}

/// A word memory as the context of a construct's arms.
struct Mem<'m>(&'m mut dyn WordMem);

/// Whether a leaf at byte shift `sh` (`8 · (off mod 4)`) reaches the word after its
/// `leaf.words()` words.
#[derive(Clone, Copy)]
enum Reach {
    Never,
    Always,
    When(Value),
}

/// Whether a [`Path::Bytes`] leaf at bit shift `sh` reaches into the word after its words: when
/// its first byte's position in its word exceeds the slack `4 · words - bytes`.
fn reaches(fb: &mut FunctionBuilder<'_>, leaf: &Leaf, sh: Value) -> BResult<Reach> {
    let slack = 4 * leaf.words() - leaf.bytes;
    if slack >= 3 {
        return Ok(Reach::Never);
    }
    if let Some(s) = known(fb, sh)? {
        return Ok(if s / 8 > slack {
            Reach::Always
        } else {
            Reach::Never
        });
    }
    let limit = k32(fb, 8 * slack)?;
    Ok(Reach::When(fb.binary(BinOp::UGreaterThan, sh, limit)?))
}

/// `(lo >> sh) | (hi << (32 - sh))` for a bit shift `sh` in `0..32` (only `lo` for 0). SPIR-V
/// leaves a shift by 32 undefined, so the high part shifts by `1` then `31 - sh`.
fn funnel_down(fb: &mut FunctionBuilder<'_>, lo: Value, hi: Value, sh: Value) -> BResult<Value> {
    if let Some(s) = known(fb, sh)? {
        let x = shr(fb, lo, s)?;
        if s == 0 {
            return Ok(x);
        }
        let y = shl(fb, hi, 32 - s)?;
        return or(fb, x, y);
    }
    let x = u32_op(fb, BinOp::ShiftRightLogical, lo, sh)?;
    let y = shl(fb, hi, 1)?;
    let k31 = k32(fb, 31)?;
    let back = fb.binary(BinOp::ISub, k31, sh)?;
    let y = u32_op(fb, BinOp::ShiftLeftLogical, y, back)?;
    or(fb, x, y)
}

/// `(cur << sh) | (prev >> (32 - sh))` for a bit shift `sh` in `0..32` (only `cur` for 0), with
/// the high part shifted by `1` then `31 - sh` as in [`funnel_down`].
fn funnel_up(fb: &mut FunctionBuilder<'_>, prev: Value, cur: Value, sh: Value) -> BResult<Value> {
    if let Some(s) = known(fb, sh)? {
        let x = shl(fb, cur, s)?;
        if s == 0 {
            return Ok(x);
        }
        let y = shr(fb, prev, 32 - s)?;
        return or(fb, x, y);
    }
    let x = u32_op(fb, BinOp::ShiftLeftLogical, cur, sh)?;
    let y = shr(fb, prev, 1)?;
    let k31 = k32(fb, 31)?;
    let back = fb.binary(BinOp::ISub, k31, sh)?;
    let y = u32_op(fb, BinOp::ShiftRightLogical, y, back)?;
    or(fb, x, y)
}

/// Load `leaf` at byte offset `off` (a `u32`) of `mem`; the register value.
pub fn load(
    fb: &mut FunctionBuilder<'_>,
    mem: &mut dyn WordMem,
    leaf: &Leaf,
    off: Value,
) -> BResult<Value> {
    let words = match leaf.path {
        // The leaf's one `uvec4` unit, at `off / 16`.
        Path::Vec4 if mem.units().vec4 => {
            let index = shr(fb, off, 4)?;
            let v = mem.read_vec4(fb, index)?;
            return value_of_vec4(fb, leaf, v);
        }
        Path::Vec4 | Path::Aligned => {
            let two = k32(fb, 2)?;
            let w = u32_op(fb, BinOp::ShiftRightLogical, off, two)?;
            let mut words = Vec::with_capacity(leaf.words() as usize);
            for j in 0..leaf.words() {
                let at = add(fb, w, j)?;
                words.push(mem.read(fb, at)?);
            }
            words
        }
        // The leaf's one narrow unit, at `off / bytes`.
        Path::OneWord if mem.units().has(leaf.bytes) => {
            let index = unit_index(fb, off, leaf.bytes)?;
            vec![mem.read_unit(fb, leaf.bytes, index)?]
        }
        Path::OneWord => {
            let (w, s) = locate(fb, off)?;
            let x = mem.read(fb, w)?;
            vec![u32_op(fb, BinOp::ShiftRightLogical, x, s)?]
        }
        Path::Bytes => {
            // The leaf's words at `off`'s word and bit shift, and the word after them when the
            // leaf reaches into it; each value word joins two memory words.
            let (w0, sh) = locate(fb, off)?;
            let k = leaf.words();
            let mut m = Vec::with_capacity(k as usize + 1);
            for j in 0..k {
                let at = add(fb, w0, j)?;
                m.push(mem.read(fb, at)?);
            }
            let at = add(fb, w0, k)?;
            let last = match reaches(fb, leaf, sh)? {
                Reach::Never => k32(fb, 0)?,
                Reach::Always => mem.read(fb, at)?,
                Reach::When(c) => {
                    let then: ArmFn<'_, Mem<'_>> = Box::new(move |fb, mem| {
                        Ok(ArmEnd::FallThrough(vec![mem.0.read(fb, at)?]))
                    });
                    let otherwise: ArmFn<'_, Mem<'_>> =
                        Box::new(|fb, _| Ok(ArmEnd::FallThrough(vec![k32(fb, 0)?])));
                    let merged = fb.if_(&mut Mem(mem), c, then, Some(otherwise))?;
                    merged.0.first().copied().ok_or_else(dangling)?
                }
            };
            m.push(last);
            let mut out = Vec::with_capacity(k as usize);
            for j in 0..k as usize {
                out.push(funnel_down(fb, word(&m, j)?, word(&m, j + 1)?, sh)?);
            }
            out
        }
    };
    value_of(fb, leaf, &words)
}

/// Store register value `v` as `leaf` at byte offset `off` (a `u32`) of `mem`. Returns whether
/// the store merges a partial word into `mem` through [`WordMem::write_bits`] (the sub-word
/// store). In a buffer slot those are two word atomics, which read the word they modify (the
/// Vulkan memory model's "reads"), so the slot's reported access includes a read.
pub fn store(
    fb: &mut FunctionBuilder<'_>,
    mem: &mut dyn WordMem,
    leaf: &Leaf,
    off: Value,
    v: Value,
) -> BResult<bool> {
    // The leaf's one `uvec4` unit, at `off / 16`.
    if leaf.path == Path::Vec4 && mem.units().vec4 {
        let x = vec4_of(fb, leaf, v)?;
        let index = shr(fb, off, 4)?;
        mem.write_vec4(fb, index, x)?;
        return Ok(false);
    }
    let mut merged = false;
    let words = words_of(fb, leaf, v)?;
    match leaf.path {
        Path::Vec4 | Path::Aligned => {
            let two = k32(fb, 2)?;
            let w = u32_op(fb, BinOp::ShiftRightLogical, off, two)?;
            let units = mem.units().intersect(leaf_units(leaf, true));
            for j in 0..leaf.words() {
                let at = add(fb, w, j)?;
                let x = word(&words, j as usize)?;
                let covered = leaf.bytes - 4 * j;
                if covered >= 4 {
                    mem.write(fb, at, x)?;
                } else if units == leaf_units(leaf, true) {
                    // The partial word's bytes as narrow units: a 2-byte part at the word's start,
                    // then a 1-byte part.
                    let mut done = 0;
                    for size in [2u32, 1] {
                        if covered - done >= size {
                            let byte = add(fb, off, 4 * j + done)?;
                            let index = unit_index(fb, byte, size)?;
                            let part = shr(fb, x, 8 * done)?;
                            mem.write_unit(fb, size, index, part)?;
                            done += size;
                        }
                    }
                } else {
                    let m = k32(fb, byte_mask(covered))?;
                    mem.write_bits(fb, at, m, x)?;
                    merged = true;
                }
            }
        }
        // The leaf's one narrow unit, at `off / bytes`.
        Path::OneWord if mem.units().has(leaf.bytes) => {
            let index = unit_index(fb, off, leaf.bytes)?;
            mem.write_unit(fb, leaf.bytes, index, word(&words, 0)?)?;
        }
        Path::OneWord => {
            let (w, s) = locate(fb, off)?;
            let m = k32(fb, byte_mask(leaf.bytes))?;
            let x = word(&words, 0)?;
            let x = u32_op(fb, BinOp::BitwiseAnd, x, m)?;
            let mask = u32_op(fb, BinOp::ShiftLeftLogical, m, s)?;
            let bits = u32_op(fb, BinOp::ShiftLeftLogical, x, s)?;
            mem.write_bits(fb, w, mask, bits)?;
            merged = true;
        }
        Path::Bytes => {
            // Memory word `j` takes value word `j`'s low bytes and word `j - 1`'s high bytes at
            // `off`'s bit shift; the word after the leaf's words only when the leaf reaches into
            // it.
            let (w0, sh) = locate(fb, off)?;
            let k = leaf.words();
            let reach = reaches(fb, leaf, sh)?;
            let zero = k32(fb, 0)?;
            let part = |fb: &mut FunctionBuilder<'_>, j: u32| -> BResult<(Value, Value)> {
                if j >= k {
                    return Ok((zero, zero));
                }
                let m = k32(fb, byte_mask(leaf.bytes - 4 * j))?;
                Ok((word(&words, j as usize)?, m))
            };
            for j in 0..=k {
                let (cur, cur_mask) = part(fb, j)?;
                let (prev, prev_mask) = match j.checked_sub(1) {
                    Some(i) => part(fb, i)?,
                    None => (zero, zero),
                };
                let bits = funnel_up(fb, prev, cur, sh)?;
                let mask = funnel_up(fb, prev_mask, cur_mask, sh)?;
                let at = add(fb, w0, j)?;
                // A word between the first and the last holds four of the leaf's bytes.
                let whole = j > 0 && leaf.bytes >= 4 * (j + 1);
                match (j < k, reach) {
                    (true, _) if whole => mem.write(fb, at, bits)?,
                    (true, _) | (false, Reach::Always) => {
                        mem.write_bits(fb, at, mask, bits)?;
                        merged = true;
                    }
                    (false, Reach::Never) => {}
                    (false, Reach::When(c)) => {
                        let then: ArmFn<'_, Mem<'_>> = Box::new(move |fb, mem| {
                            mem.0.write_bits(fb, at, mask, bits)?;
                            Ok(ArmEnd::FallThrough(Vec::new()))
                        });
                        fb.if_(&mut Mem(mem), c, then, None)?;
                        merged = true;
                    }
                }
            }
        }
    }
    Ok(merged)
}

/// The zero of `leaf`'s register form (the value of a contained out-of-bounds load).
pub fn zero(fb: &mut FunctionBuilder<'_>, leaf: &Leaf) -> BResult<Value> {
    let ty = if leaf.chunked() {
        chunked_ty(fb, leaf.form)?
    } else {
        ops::ty(fb, leaf.form)?
    };
    let c = fb.module().constant(Constant::Null { ty })?;
    fb.module().const_value(c)
}
