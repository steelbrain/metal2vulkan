//! `lower::texture`: image reads, writes, atomics, queries, fences, and null checks.
//!
//! **Handles** (`memory::handles`). A texture operand is resolved to its static candidates: a
//! parameter bound to one `[[texture(n)]]` slot, an element of a texture array, or a handle
//! selected or merged by a phi from those and the null texture. One candidate is that slot's
//! binding; several are a switch-at-use on the handle index ([`TextureStrategy::Switch`]), one arm
//! per slot and a null arm (a read or query is 0, a write dropped, `is_null` true).
//!
//! **Heap handles** (`memory::handles`). A texture handle loaded from an argument buffer is a set-1
//! heap element. The handles `phi`, `select`, and `freeze` merge, and the loads of one
//! argument-buffer member, form a group ([`HeapPlan`], `memory::handles::HeapGroups`), declared
//! once, from all of its uses, as one runtime array of the heap class its shape, texel class, and
//! member access name, like a slot ([`ImageRef::Heap`]). A group that is only queried has no texel
//! class and is declared `Float`, which its `RuntimeFact::HeapNumericType` states so the binder
//! reports a resident texture of another numeric type. An operation indexes that array with the
//! handle, `NonUniform`, and behaves as the static null arm for the null handle (element 0, also
//! every missed id): a read or query is 0, a write is dropped, `is_null` is `handle == 0`. The
//! arrays need descriptor indexing (`runtimeDescriptorArray`, `descriptorBindingPartiallyBound`,
//! the class's non-uniform indexing and update-after-bind features), and a storage class formatless
//! access; without them the group is `Unsupported{ArgumentBufferTextures}` naming what is missing.
//! A depth texture is read through the float class of its shape, declared without `Depth`, and one
//! compared anywhere through the depth-comparison class of its shape, declared `Depth`. A heap
//! texture is sampled through a separate heap sampler (`lower::sampler`); a sampled cube group is a
//! cube class when it is only sampled. Heap cube reads, writes, and queries, and heap texture
//! buffers, are not implemented yet.
//!
//! **Slots** ([`SlotPlan`]). Every texture slot the function uses is declared once, as one
//! `OpTypeImage` derived from the typed operand positions of the intrinsics that use it (the ABI
//! spells no shape, `air::abi`) and the parameter's declared access: `sample`/`read` textures are
//! sampled images (`Sampled = 1`, format `Unknown`), `write`/`read_write` textures storage images
//! (`Sampled = 2`) whose format is the slot's `StorageImageChoice` (`LowerInputs`). A cube texture
//! that is read or written, and every storage cube, is declared as the 2D array the binder views it
//! as (layer = face + 6 · cube, `RuntimeFact::ImageViewShape`), because Vulkan fetches no cube by
//! face. Texture buffers are texel buffers (`RuntimeFact::TexelBufferRange`).
//!
//! **Operations.** `read` is `OpImageFetch` on a sampled image and `OpImageRead` on a storage
//! image; `write` is `OpImageWrite`; `get_*` are `OpImageQuery*`; `fence` is an image memory
//! barrier; `is_null` reads the slot's `ResourcePresence` specialization constant. Texels move
//! between the image's 32-bit `vec4` and AIR's held forms: a slot's `half` texels convert by the
//! bound format's class, which its `TexelFormat` specialization gives ([`half`]; a texture buffer's
//! included), and a heap element's `half` texels are rounded exactly, half subnormals and the
//! decoded NaN kept (`narrow::round_texel`; the NaN is the device's decode,
//! `PrecisionNote::DeviceNanEncoding`), 16-bit integers are held zero-extended and written
//! sign-extended, and a storage image of class `SwizzleBgra` or `SrgbEncode` converts in the
//! shader, the latter by the device's sRGB transfer (`PrecisionNote::SrgbTexelDecode` for a read,
//! `PrecisionNote::DevicePackConversion` for a write).
//!
//! **Containment**. A write is discarded out of bounds (Metal's semantics) by a builder-closed
//! bounds check unless the profile's robustness discards it already (`robustImageAccess`;
//! `robustBufferAccess2` for texel buffers). A read out of bounds returns Metal's texel, zero with
//! the pixel format's missing alpha read as 1 ([`ReadCheck`]): the device's under
//! `robustImageAccess2` (`robustBufferAccess2` for texel buffers), otherwise, under containment,
//! the shader clamps the level, coordinate, layer, and sample index into the bound image and
//! selects that texel where one was out of range, its alpha from the slot's `OutOfBoundsAlpha`
//! specialization when the function observes it (`alpha`). A heap read's alpha is its handle's: the
//! binder writes the out-of-bounds-alpha flag in the handle-table entry of a texture whose format
//! has no alpha, and a module with such a read looks handles up in
//! `memory::handles::Lookup::Flagged`, which carries the flag in the handle's top bit (masked off
//! before indexing). A cube face or cube index out of range addresses a layer past the 2D-array
//! view, never another cube's face.
//!
//! **Clamp-to-edge reads** (MSL 4.1 §6.13). Metal 4.1's `read(sampler s, ...)` through a constexpr
//! `address::clamp_to_edge` sampler of default state clamps its signed coordinate (plus its offset)
//! into the level it reads before the access, so only its level, layer, face, or sample index can
//! be out of bounds, contained as above ([`Edge`]). The evidence is the specification's text: no OS
//! at hand runs Metal 4.1 to measure it. A read through another constexpr sampler state or an
//! argument sampler is refused.
//!
//! **Storage levels**. A storage access reaches its view's base level only
//! (`OpImageRead`/`OpImageWrite` take no `Lod`). A storage slot that is read or written at a level
//! other than the constant 0 is declared as an array of `N` storage images, `N` the most levels a
//! texture of its type can have on the device (`⌊log2 maxImageDimension⌋ + 1`), which the binder
//! fills with single-level views of the bound texture's levels and neutrals
//! past them (`RuntimeFact::StorageTextureLevelViews`). Every access of such a slot indexes the
//! array: by a constant level below `N` as is (the neutral where the texture lacks that level),
//! else by the level clamped below the bound texture's level count (`PushField::TextureLevels`,
//! `RuntimeFact::StorageTextureLevels`), a `NonUniform` index. A read or write at a level the
//! texture lacks is out of bounds (MSL 4.1 §6.13): Metal ignores the write, and the write is
//! dropped; Metal reads zero components (alpha 1 where the format has none), and the read selects
//! that texel on every profile, whatever the device's robustness, since the element it read is
//! another level or a neutral (the out-of-bounds texel, its alpha from the slot's
//! `OutOfBoundsAlpha` where the function observes it). `get_num_mip_levels` of a storage texture
//! is the pushed count, 0 for a neutral; of a 1D
//! texture or 1D texture array, sampled or storage, always 0 (Metal has no mipmapped 1D
//! textures). Heap storage images take only the constant 0, and a
//! texture atomic on a slot with level views is not implemented.
//!
//! **Size queries at a level** (`emit::level_extent`). A width, height, or depth query at a level
//! that is not the constant 0, of a sampled or storage texture, answers Metal's
//! `max(1, e >> (l & 255))` of the extent `e` of the texture's first level, at every level `l`,
//! past its level count too: it queries no other level, so a storage slot needs no level views
//! for it. A heap storage element's query takes only the constant 0, as its other accesses.
//!
//! **Sampling.** Samples, gathers, and LOD queries take a sampler: they are the sampler family's
//! (`lower::sampler`), selected here once the texture's candidates are known, as
//! [`TextureStrategy::Sampling`] (or a switch of them).
//!
//! **Status words** (`status`). A `{T, i8}` result's status word is Metal's residency status: bit 0
//! set when the access touched a texel that is not resident, the only bit MSL reads
//! (`sparse_color::resident()`). A word nothing observes is built as 0; its texel may be taken
//! through `phi`, `select`, and `freeze` copies of the whole result. A word whose effects depend on
//! bit 0 alone (`status::observed`'s demanded bits) is lowered on a slot: a texture that is not
//! sparse is resident everywhere, so its word is 0 on every device; a sparse one's is its sparse
//! access's (`OpImageSparse*`, `OpImageSparseTexelsResident`), in an arm the slot's `SparseTexture`
//! specialization selects (`RuntimeFact::SparseTexture`), compiled where the profile enables
//! `shaderResourceResidency` and the slot's accesses all have a sparse form. Elsewhere the binder
//! refuses a sparse texture at the slot. The artifact notes `PrecisionNote::SparseResidency`. A
//! word with another bit observed, or a whole `{T, i8}` value exposed, is refused.
//!
//! **Atomics** (`atomic`). `atomic_fetch_max` is `OpAtomic{U,S}Max` on the texel pointer of an
//! `R32ui` / `R32i` storage image; only lane 0 of its result may be read.
//!
//! **The nil arm** (`absent`). The code a shader keeps for an `is_null` of a static slot that
//! says its texture is nil is kept, exact for a nil texture; the default (all-present)
//! specialization never runs it. The census counts its words under [`ABSENT_ARM`], so the
//! output-size gate can except exactly them; nothing emitted changes.
//!
//! Not lowered yet (`Unsupported{Lowering(Texture), NotImplemented}`): an observed status word of
//! a heap element (its residency needs a word per handle-table entry), reads, writes, and level
//! counts of a heap storage image at a level other than the constant 0, and read offsets.

mod absent;
mod alpha;
mod atomic;
mod cooperative;
mod emit;
pub mod half;
pub(crate) mod heap_raw;
mod slice;
mod slot;
mod status;

pub(crate) use absent::absent_arm;
pub use absent::ABSENT_ARM;
pub use atomic::TexAtomic;
pub(crate) use emit::{
    constant_lanes, dispatch, extended_table, half_class, heap_element, heap_half_class,
    heap_raw_form, heap_raw_image, live_or_zero, reinterpret, resident_access, rgb9e5_word,
    slot_image, sparse_spec, srgb_raw_tap, srgb_texel, swizzled_float, swizzled_gather,
    swizzled_held, tile_map, tiled_word, with_word, NullLeaf, TileLevel, RGB9E5_WORDS,
    SRGB_DECODE_WORDS, SRGB_TAP_WORDS, SRGB_TEXEL,
};
pub use emit::{emit, held_texel, to_image};
pub(crate) use slot::depth_comparison;
pub use status::StatusUse;

use crate::lower::access::handle::HandleTarget;
use crate::memory::handles::{Candidate, HeapGroups};
use crate::pipeline::lower_inputs::StorageImageClass;
use crate::pipeline::sym::{ImageAccess, ImageHandleClass};
pub(crate) use slot::image_class;
#[cfg(test)]
pub use slot::max_levels;
pub use slot::{
    finish, heap_layout_features, HeapPlan, ImageDecl, ReadCheck, SlotPlan, TexelClass,
};
use std::borrow::Cow;

use std::collections::{BTreeMap, BTreeSet};

use crate::air::abi::{AbiInt, Access as AbiAccess, ParamBinding, ParamRole};
use crate::air::intrinsics::{
    AirIntrinsic, AtomicOp, Elem, ImageDim, ImageShape, Layout, Query, SamplerOp, Sign, TexOp,
    TexRole,
};
use crate::air::ir::{Callee, Const, Inst, InstId, InstKind, Value, ValueId};
use crate::budget::Meter;
use crate::error::{DriverDeviation, Error, Feature, LowerFamily, Unsupported};
use crate::lower::arith::{FloatFmt, Form, Held};
use crate::lower::emit::EmitCx;
use crate::lower::sampler::Sampling;
use crate::lower::select::{FamilyPlans, NoteClause, NoteOn, NoteTest, Plan, SelectCx};
use crate::lower::strategy::FamilyStrategy;
use crate::pipeline::artifact::PrecisionNote;
use crate::pipeline::sym::{SampledType, SlotClass, SlotRef};
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
use crate::spirv::error::BResult;
use crate::spirv::ids::{ConstId, DemoteTy, FuncId, GlobalId, TypeId, Value as SValue};
use crate::spirv::ir::{BinOp, ConstBits, ExecMode, GlslOp, MemAccess, Type as SType};

fn refuse() -> Error {
    Unsupported::not_implemented(Feature::Lowering(LowerFamily::Texture)).into()
}

/// How a texel moves between the image's 32-bit `vec4` and AIR's held value.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TexelConv {
    /// `float`: as is.
    F32,
    /// `half`, held widened: rounded exactly when read, half subnormals and the decoded NaN kept
    /// (`narrow::round_texel`).
    F16,
    /// A 32-bit integer, held in a `u32`: bitcast.
    I32,
    /// A 16-bit integer, held zero-extended in a `u32`: masked to 16 bits when read, and
    /// sign-extended when written to a signed image (an out-of-range integer texel is undefined
    /// in Vulkan).
    I16 { signed: bool },
    /// Metal's `ulong` texel as the two 32-bit words of an `RG32` image texel, the low word in R
    /// (measured on Apple M3 Max): a read is `(R | G << 32, 0, 0, 1)` (Metal's lanes, the same out
    /// of bounds, where R and G are 0), and a write stores lane 0's low word in R and its high word
    /// in G. Held in `u64`s, or, `wide` on a profile without `shaderInt64`, as the low words' and
    /// the high words' `uvec4`s.
    U64Words { wide: bool },
}

/// The texel an operation reads or writes: its conversion and lane count (4, or 1 for a depth
/// read).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TexelForm {
    pub conv: TexelConv,
    pub lanes: u8,
}

impl TexelForm {
    /// The held form of this texel's lanes as `half`s.
    pub fn half(self) -> Form {
        Form {
            held: Held::Float(FloatFmt::Half),
            lanes: self.lanes,
        }
    }

    /// The estimated words of converting an image texel to this form ([`held_texel`]) beyond a
    /// strategy's base words: a `half`'s exact rounding.
    pub fn read_words(self) -> u32 {
        match self.conv {
            TexelConv::F16 => crate::lower::arith::narrow::round_texel_words(self.half()),
            _ => 0,
        }
    }
}

/// An image coordinate: the plane coordinate (a held `u32` or `uvecN`), and the array layer or
/// cube face.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Coord {
    pub plane: ValueId,
    /// The number of plane components.
    pub lanes: u8,
    pub layer: Option<ValueId>,
    pub face: Option<ValueId>,
}

/// The image an operation reads, writes, or queries.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ImageRef {
    /// The texture slot `[[texture(n)]]`.
    Slot(u32),
    /// Element `handle` (a heap handle's value) of the heap array of heap group `group`.
    Heap { group: u32, handle: ValueId },
}

impl ImageRef {
    pub(crate) fn heap(&self) -> bool {
        matches!(self, ImageRef::Heap { .. })
    }
}

/// A `read`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Read {
    pub image: ImageRef,
    pub at: Coord,
    /// The level of a sampled, single-sampled, non-buffer image; of a storage slot, the level of
    /// its level views when it is not the constant 0.
    pub level: Option<ValueId>,
    /// `level` indexes the slot's level views: past the bound level count the read selects Metal's
    /// out-of-bounds texel on every profile, whatever its [`ReadCheck`].
    pub level_views: bool,
    /// The sample index of a multisampled image.
    pub sample: Option<ValueId>,
    pub texel: TexelForm,
    /// The result is `{T, i8}`, with this lowered type.
    pub result: DemoteTy,
    /// The function observes bit 0 of the result's status word: on a slot with a sparse arm the
    /// word is the sparse read's under the slot's `SparseTexture` specialization; otherwise it is
    /// 0, a texture that is not sparse being resident everywhere.
    pub residency: bool,
    /// How an out-of-bounds read returns Metal's texel.
    pub check: ReadCheck,
    /// Where the shader selects the out-of-bounds texel (under [`ReadCheck::Shader`], or past a
    /// level-view read's level count), its alpha is the bound format's (the function observes
    /// the alpha): a slot's `OutOfBoundsAlpha` specialization, or a heap element's
    /// out-of-bounds-alpha flag, which its handle carries; otherwise it is 0.
    pub alpha: bool,
    /// A Metal 4.1 read through a clamp-to-edge sampler ([`Edge`]).
    pub edge: Option<Edge>,
    /// The read takes an 8-bit sRGB texel's codes from its slot's image, which holds the format's
    /// UNORM twin (`LoweringInputs::srgb_textures`), and decodes them by the device's transfer
    /// (`emit::srgb_texel`, `RuntimeFact::SrgbTexture`).
    pub raw: bool,
    /// The read, of a storage slot whose class is `SrgbEncode`, takes the codes of an 8-bit sRGB
    /// format from its UNORM alias and decodes them by the device's transfer
    /// (`emit::srgb_texel`).
    pub srgb_codes: bool,
    /// The read, of a texture buffer slot, applies the bound texture buffer's swizzle to its
    /// texel, an out-of-bounds texel included, under the slot's `TextureSwizzle` specialization
    /// (`RuntimeFact::TextureSwizzle`).
    pub swizzle: bool,
    /// The read, of a storage slot of `LoweringInputs::swizzled_storage`, applies the bound
    /// texture's swizzle to its held texel, after converting it (an out-of-bounds texel
    /// included), under the slot's `TextureSwizzle` specialization; so does a read of an
    /// extended-range sampled slot of it, after decoding the texel, and of a packed normalized
    /// sampled slot of it, after converting the texel by its class.
    pub storage_swizzle: bool,
    /// The slot's variant binds a texture of this extended-range format: the read decodes the
    /// raw texel ([`extended_range::decode`](crate::lower::arith::extended_range::decode)), and
    /// selects Metal's out-of-bounds texel (zero, alpha the format's) in the shader.
    pub extended: Option<crate::metal::format::ExtendedRange>,
    /// The read, of a sampled float heap group, decodes a member of a form a heap reads where
    /// the handle's class index names one (`heap_raw`): its raw texel fetched from the group's
    /// unsigned integer class (`HeapPlan::raw`), in place of the float array's neutral.
    pub heap_raw: bool,
    /// The read, of a storage heap group, takes Metal's texel `(0, 0, 0, r)` from the image's red
    /// where the handle's class index is `A8Unorm`'s, on a device that holds such a texture's alpha
    /// in red ([`heap_alpha_in_red`]).
    pub heap_alpha_red: bool,
    /// The read, of a slot's color texture on a profile whose sparse textures are sparse images
    /// (it enables `shaderResourceResidency`), sets the lanes Metal reads as constants (a channel
    /// the format lacks, through the swizzle) after it, from the slot's `ConstantLanes`
    /// specialization: a block of a sparse image that holds no mapped tile reads the device's
    /// value (RADV's alpha 0), where Metal reads them as constants.
    pub constant_lanes: bool,
    /// The slot's variant binds a texture of another numeric class
    /// (`LoweringInputs::cross_class`): [`texel`](Self::texel) is that class's own result
    /// ([`native_form`]), whose words the read hands back as this declared conversion
    /// (`emit::reinterpret`).
    pub declared: Option<TexelConv>,
    /// The read, a sibling of a heap read ([`Read::heap_cross`]), reads the member in this other
    /// numeric class: through the group's array of that class (`HeapPlan::cross`), its
    /// [`texel`](Self::texel) that class's own result, handed back as
    /// [`declared`](Self::declared).
    pub heap_class: Option<TexelClass>,
    /// The read, of a sampled color heap group of one sample, reads a member of another numeric
    /// class than its own in that class where the handle's class index names one
    /// (`metal::texel_format::heap_numeric_base`, or a float format's own class): one sibling
    /// read per other class, the read itself for every other member.
    pub heap_cross: Vec<Read>,
}

/// The estimated words of read `r`: coordinate assembly, the access, the texel conversion, and the
/// status composite; the clamps, the in-bounds test, and the out-of-bounds select (with its
/// specialized alpha); the level views' count, index clamp, level test, and select; a heap
/// element's access chain, load, and null select or guard; the sparse arm ([`RESIDENCY_WORDS`]) and
/// the word's select outside the image, with its own in-bounds test where the shader has none; a
/// clamp-to-edge read's size query, offset, and signed clamp; a `half` texel's call of the
/// conversion routine or exact rounding, or an sRGB texel's raw decode; the lanes Metal reads as
/// constants; the reinterpretation of another class's words; and each sibling of a heap read with
/// its class test and branch.
fn read_words(r: &Read) -> u32 {
    let own = 64
        + if r.checked() { 80 } else { 0 }
        + if r.constant_lanes {
            crate::lower::sampler::CONSTANT_LANES_WORDS
        } else {
            0
        }
        + if r.residency {
            RESIDENCY_WORDS + if r.checked() { 8 } else { 48 }
        } else {
            0
        }
        + if r.alpha { 12 } else { 0 }
        + if r.level_views { 48 } else { 0 }
        + if r.image.heap() { 32 } else { 0 }
        + if r.heap_raw { heap_raw::READ_WORDS } else { 0 }
        + if r.heap_alpha_red {
            emit::HEAP_ALPHA_IN_RED_WORDS
        } else {
            0
        }
        + if r.edge.is_some() { 64 } else { 0 }
        + if r.raw || r.srgb_codes {
            SRGB_DECODE_WORDS
        } else if r.extended.is_some() {
            crate::lower::arith::extended_range::DECODE_WORDS + r.texel.read_words()
        } else if r.texel.conv == TexelConv::F16 {
            half::CALL_WORDS.max(r.texel.read_words())
        } else {
            r.texel.read_words()
        };
    let declared = if r.declared.is_some() { 16 } else { 0 };
    let siblings: u32 = r.heap_cross.iter().map(|x| read_words(x) + 24).sum();
    own + declared + siblings
}

/// A Metal 4.1 `read(sampler s, ...)` through a constexpr clamp-to-edge sampler (MSL 4.1
/// §6.13): its signed plane coordinate, plus `offset` where it has one, is clamped into the image
/// at the read's level before the access, so only a level, layer, face, or sample index is out of
/// bounds (each as a plain read's).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Edge {
    /// The per-component offset (2D, 2D array, and 3D; Metal requires `-8..=7`), where not zero.
    pub offset: Option<ValueId>,
}

impl Read {
    /// The shader clamps the read and selects Metal's out-of-bounds texel.
    pub(crate) fn checked(&self) -> bool {
        self.check == ReadCheck::Shader
    }
}

/// A `write`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Write {
    pub image: ImageRef,
    pub at: Coord,
    /// The level of the slot's level views when it is not the constant 0.
    pub level: Option<ValueId>,
    pub value: ValueId,
    pub texel: TexelForm,
    /// Discard an out-of-bounds write in the shader (no robustness discards it).
    pub guard: bool,
    /// The slot's variant binds a texture of this extended-range format: the write encodes the
    /// raw texel ([`extended_range::encode`](crate::lower::arith::extended_range::encode)).
    pub extended: Option<crate::metal::format::ExtendedRange>,
    /// The slot's storage class ([`storage_class`]): a converting one moves or encodes the
    /// written channels (`RotateArgb`'s rotation, `SrgbEncode`'s codes).
    pub class: StorageImageClass,
    /// A `float` or `half` texel written through a heap handle stores its alpha in the image's
    /// red where the handle's class index is `A8Unorm`'s, on a device that holds such a texture's
    /// alpha in red ([`heap_alpha_in_red`]).
    pub heap_alpha_red: bool,
    /// The slot is one of `LoweringInputs::swizzled_storage`: the write undoes the bound
    /// texture's swizzle on its held value, before converting it, under the slot's
    /// `TextureSwizzle` specialization (`emit::unswizzled`).
    pub swizzle: bool,
    /// The slot's variant binds a texture of another numeric class
    /// (`LoweringInputs::cross_class`): `texel` is that class's own form
    /// ([`native_write_form`]), into which the write first turns the declared lanes
    /// (`emit::cross_write`).
    pub cross: Option<CrossWrite>,
    /// How an integer texel's lanes are saturated to the bound format's channels ([`int_range`]).
    pub saturate: IntRange,
}

/// A write's declared texel and the bound format of a texture of another numeric class
/// ([`Write::cross`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CrossWrite {
    /// The conversion of the texel the function writes.
    pub declared: TexelConv,
    /// The format the slot's storage choice declares, the bound texture's.
    pub format: crate::vk::Format,
}

/// How an integer texel's write saturates each lane to its channel's range, as the image's
/// signedness reads the 32-bit word (`emit::saturated`): Metal stores the nearest value the
/// channel holds (`uint` 300 into `R8Uint` stores 255, `int` -200 into `R8Sint` -128), where a
/// Vulkan write of a value the format cannot represent is undefined.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum IntRange {
    /// Not an integer texel's write, or every channel holds the written word: stored as it is.
    Full,
    /// The width of each channel (red to alpha) of the format the slot's storage choice declares.
    Widths([u32; 4]),
    /// A slot whose storage choice declares no format: the widths the binder pushes for the
    /// bound texture (`PushField::TextureIntegerChannels`).
    SlotWord(SlotRef),
}

impl IntRange {
    /// The words of the saturation: a clamp against constants, or the word's load, the bounds'
    /// derivation and the clamp.
    pub(crate) fn words(self) -> u32 {
        match self {
            IntRange::Full => 0,
            IntRange::Widths(_) => 24,
            IntRange::SlotWord(_) => 24 + 48,
        }
    }
}

/// What a `get_*` query reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum QueryKind {
    /// Component `component` of the image size (width 0, height 1, depth 2).
    Size {
        component: u8,
    },
    /// The number of array elements (cubes of a cube array).
    ArraySize,
    Levels,
    /// A 1D texture's or 1D texture array's level count: always 0 (Metal has no mipmapped 1D
    /// textures).
    NoLevels,
    Samples,
}

/// A `get_*` query.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QueryOp {
    pub image: ImageRef,
    pub kind: QueryKind,
    /// The level of a size query of a single-sampled image other than a texel buffer, sampled or
    /// storage, when it is not the constant 0: the query answers from the image's first level
    /// (`emit::level_extent`).
    pub level: Option<ValueId>,
    /// The query, of a sampled heap group, queries a member of a form a heap reads where the
    /// handle's class index names one through the group's unsigned integer class
    /// (`HeapPlan::raw`), which holds it, in place of the float array's neutral.
    pub heap_raw: bool,
    /// The query, of a sampled color heap group of one sample, queries a member of another
    /// numeric class through the group's array of that class (`HeapPlan::cross`) where the
    /// handle's class index names one.
    pub heap_cross: bool,
    /// The query is of a slot whose variant binds a two-plane extended-range 3D texture, whose
    /// depth and level count Metal answers otherwise (`emit::planes_volume`).
    pub volume: bool,
}

/// The texture strategies.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TextureStrategy {
    Read(Read),
    Write(Write),
    /// A texture atomic.
    Atomic(TexAtomic),
    Query(QueryOp),
    /// `fence`: an image memory barrier.
    Fence,
    /// `is_null`: the negated `ResourcePresence` of the slot.
    IsNull {
        index: u32,
    },
    /// `is_null` of a heap handle: the handle is the null (neutral) element.
    HeapIsNull {
        handle: ValueId,
    },
    /// A sample, gather, or LOD query: the sampler family's.
    Sampling(Box<crate::lower::sampler::Sampling>),
    /// `get_null_texture`: the null handle (`memory::handles`).
    NullHandle,
    /// An operation on a multi-candidate texture or sampler: a switch-at-use over the
    /// candidates.
    Switch(Box<Switch>),
    /// A slice write of the kernel's imageblock.
    ImageblockSlice(Box<slice::Slice>),
    /// A sample of a texture slot bound in an extended-range format, emulated by fetches of its
    /// raw texels (`sampler::extended`).
    ExtendedSample(Box<crate::lower::sampler::extended::Sample>),
    /// A sample of a two-plane YCbCr texture slot whose variant reads its planes in place,
    /// computed from their codes (`sampler::in_place`).
    InPlaceSample(Box<crate::lower::sampler::in_place::Sample>),
    /// An operation on the absent handle: a texture or sampler a function constant removed
    /// (`memory::handles::Candidate::Absent`).
    AbsentArgument(AbsentArgument),
}

/// An operation on the absent handle, alone or as a switch-at-use's arm. Metal gives a read through
/// a removed texture or sampler no value, so a read, query, sample, gather, atomic, or `is_null` is
/// the absent read: it marks the command buffer's execution status where it runs
/// (`access::absent::mark`), and yields zero. A write or fence of a removed texture is dropped, as
/// Metal drops one, and marks nothing.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AbsentArgument {
    /// Whether the operation marks the status: every one but a write or a fence.
    pub marks: bool,
    /// What it yields: zero of its result's lowered type, or nothing.
    pub value: NullArm,
}

impl AbsentArgument {
    /// The census name.
    const NAME: &'static str = "texture.absent_argument";

    /// The absent operation of call `at` (`inst`), decoded as `t`: its status write's needs stated
    /// for the stage where it marks (`access::absent::require`).
    pub(crate) fn select(
        cx: &mut SelectCx<'_>,
        at: InstId,
        inst: &Inst,
        t: &crate::air::intrinsics::Texture,
    ) -> Result<TextureStrategy, Error> {
        let marks = !matches!(t.op, TexOp::Write | TexOp::Fence);
        if marks {
            crate::lower::access::absent::require(cx)?;
        }
        let value = match cx.entry.module.types().get(inst.ty) {
            Some(crate::air::ir::Type::Void) | None => NullArm::Nothing,
            _ => NullArm::Zero(cx.lowered_repr(Value::Inst(at), inst.ty)?),
        };
        Ok(TextureStrategy::AbsentArgument(AbsentArgument {
            marks,
            value,
        }))
    }
}

/// What the null arm of a switch-at-use yields: Metal's null texture reads and samples 0, drops
/// writes, and is null (the value is ours, Metal's is open question Q7).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NullArm {
    /// The zero value of the result's lowered type.
    Zero(DemoteTy),
    /// `true` (`is_null`).
    True,
    /// No value (a write).
    Nothing,
}

/// An operation on a texture or sampler with several static candidates: nested switches-at-use on
/// each handle, one arm per candidate combination.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Switch {
    /// The handle indices dispatched on, outermost first (the texture's, then the sampler's).
    pub handles: Vec<ValueId>,
    /// Each arm: its case literal per handle (the candidates' handle indices), and the
    /// operation on that candidate combination, a static-slot strategy.
    pub arms: Vec<(Vec<u32>, TextureStrategy)>,
    pub null: NullArm,
    /// What the switch computes before it dispatches, in the use's own position.
    pub prelude: Prelude,
}

/// What a switch-at-use computes before its dispatch, in the use's own position, so that no
/// tangled instruction lies inside it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Prelude {
    /// Nothing: no arm is tangled.
    Nothing,
    /// The implicit-LOD sample's gradients, which its sampling arms read.
    Gradients(crate::lower::sampler::Hoist),
    /// Every arm's value, a fragment LOD query (which has no gradient form), each candidate's query
    /// at the call's own position with the call's coordinate; the dispatch selects the handles'
    /// one.
    Arms,
}

impl Switch {
    /// The census name of the switch: its arms' operation, with the check they take. The arms are
    /// one operation selected for one image shape on different candidates, so they take one check
    /// (`slot::read_check` and `slot::write_guard` read only the profile and the shape, a query's
    /// clamp its level operand), which the first arm names: a texel buffer's clamp or guard follows
    /// the buffer encoding and an image's the image encoding, so without it a read switch over
    /// texel buffers and one over images were one name under one coverage key.
    fn name(&self) -> Cow<'static, str> {
        let hoisted = matches!(self.prelude, Prelude::Gradients(_));
        Cow::Borrowed(match self.arms.first().map(|(_, s)| s) {
            Some(TextureStrategy::Read(r)) if r.checked() => "texture.read_clamped.switch",
            Some(TextureStrategy::Read(_)) => "texture.read.switch",
            Some(TextureStrategy::Write(w)) if w.guard => "texture.write_guarded.switch",
            Some(TextureStrategy::Write(_)) => "texture.write.switch",
            Some(TextureStrategy::Atomic(a)) => return Cow::Owned(format!("{}.switch", a.name())),
            Some(TextureStrategy::Query(q)) if q.level.is_some() => "texture.query_level.switch",
            Some(TextureStrategy::Query(_)) => "texture.query.switch",
            Some(TextureStrategy::IsNull { .. }) => "texture.is_null.switch",
            Some(TextureStrategy::Sampling(_)) if hoisted => "sampler.sample_hoisted_grad.switch",
            Some(TextureStrategy::Sampling(_)) if self.prelude == Prelude::Arms => {
                "sampler.query_lod.switch"
            }
            Some(TextureStrategy::Sampling(_)) => "sampler.sample.switch",
            Some(TextureStrategy::AbsentArgument(_)) => "texture.absent_argument.switch",
            _ => "texture.null.switch",
        })
    }

    /// The arms' sampling where the switch emulates once (SEMANTICS 1012): two or more arms, each
    /// the same emulated sampling ([`Sampling::shared_shape`]) on its own texture slot, and no LOD
    /// query evaluated before the dispatch. The emulation then runs once, outside the dispatch, and
    /// each arm reads only its own image: its size, its format's bits, its taps, and the device's
    /// access (`sampler::shared_value`).
    pub(crate) fn shared(&self) -> Option<Vec<(&[u32], &Sampling)>> {
        if self.prelude == Prelude::Arms || self.arms.len() < 2 {
            return None;
        }
        let mut out = Vec::with_capacity(self.arms.len());
        let mut shape: Option<Sampling> = None;
        for (lits, s) in &self.arms {
            let TextureStrategy::Sampling(x) = s else {
                return None;
            };
            let k = x.shared_shape()?;
            match &shape {
                None => shape = Some(k),
                Some(first) if *first == k => {}
                Some(_) => return None,
            }
            out.push((lits.as_slice(), x.as_ref()));
        }
        Some(out)
    }

    fn words(&self) -> u32 {
        // The hoisted derivatives, and with a bias their scaling (`bias_gradients`). Arms evaluated
        // first are the arms' words, each selected by a phi operand.
        let hoist = match &self.prelude {
            Prelude::Nothing | Prelude::Arms => 0,
            Prelude::Gradients(h) if h.bias.is_some() => crate::lower::sampler::BIAS_GRADIENT_WORDS,
            Prelude::Gradients(_) => 32,
        };
        // The shared form (SEMANTICS 1012): the emulation once, and per arm its image's reads.
        if let Some(arms) = self.shared() {
            let (once, per_arm) = arms
                .first()
                .map_or((0, 0), |(_, s)| (s.words(), s.shared_arm_words()));
            return once
                .saturating_add(per_arm.saturating_mul(arms.len() as u32))
                .saturating_add(crate::lower::sampler::SHARED_WORDS)
                .saturating_add(hoist);
        }
        let arms = self.arms.iter().fold(0u32, |n, (_, s)| {
            n.saturating_add(s.words()).saturating_add(8)
        });
        arms.saturating_add(24).saturating_add(hoist)
    }
}

impl FamilyStrategy for TextureStrategy {
    /// A read, write, atomic, or query names the in-shader check it takes (`_clamped`,
    /// `_guarded`) on a slot, a heap element (`.heap`), or a switch's arms (`.switch`); a read or
    /// sampling operation whose status word the function observes adds `.resident`.
    fn name(&self, _: &FamilyPlans) -> Cow<'static, str> {
        let name = self.base_name();
        if self.residency() {
            Cow::Owned(format!("{name}.resident"))
        } else {
            name
        }
    }

    fn words(&self) -> u32 {
        self.base_words()
    }
}

impl TextureStrategy {
    /// The sampled images of combined bindings this strategy makes: its own sampling's, or its
    /// switch's arms' (each the static sampling of one binding). Under the separate form each is
    /// made from the separate texture and sampler. A sampling makes one in its open block. A switch
    /// that emulates once ([`Switch::shared`], SEMANTICS 1012) makes none: its bicubic taps read
    /// each arm's image alone.
    pub fn combined_samplings(&self) -> u32 {
        match self {
            TextureStrategy::Sampling(s) => match s.source {
                crate::lower::sampler::Source::Combined(_) => 1,
                _ => 0,
            },
            TextureStrategy::Switch(w) => match w.shared() {
                Some(_) => 0,
                None => w
                    .arms
                    .iter()
                    .fold(0, |n, (_, a)| n.saturating_add(a.combined_samplings())),
            },
            _ => 0,
        }
    }

    /// Whether the function observes the status word of this operation (or of a switch's arm):
    /// its residency.
    fn residency(&self) -> bool {
        match self {
            TextureStrategy::Read(r) => r.residency,
            TextureStrategy::Sampling(s) => s.residency,
            TextureStrategy::Switch(s) => s.arms.iter().any(|(_, a)| a.residency()),
            _ => false,
        }
    }

    /// The census name without the residency suffix.
    fn base_name(&self) -> Cow<'static, str> {
        Cow::Borrowed(match self {
            TextureStrategy::Read(r) if r.image.heap() && r.checked() => {
                "texture.read_clamped.heap"
            }
            TextureStrategy::Read(r) if r.image.heap() => "texture.read.heap",
            TextureStrategy::Read(r) if r.checked() => "texture.read_clamped",
            TextureStrategy::Read(_) => "texture.read",
            TextureStrategy::Write(w) if w.image.heap() && w.guard => "texture.write_guarded.heap",
            TextureStrategy::Write(w) if w.image.heap() => "texture.write.heap",
            TextureStrategy::Write(w) if w.guard => "texture.write_guarded",
            TextureStrategy::Write(_) => "texture.write",
            TextureStrategy::Atomic(a) => return Cow::Owned(a.name()),
            TextureStrategy::Query(q) if q.image.heap() && q.level.is_some() => {
                "texture.query_level.heap"
            }
            TextureStrategy::Query(q) if q.image.heap() => "texture.query.heap",
            TextureStrategy::Query(q) if q.level.is_some() => "texture.query_level",
            TextureStrategy::Query(_) => "texture.query",
            TextureStrategy::Fence => "texture.fence",
            TextureStrategy::IsNull { .. } => "texture.is_null",
            TextureStrategy::HeapIsNull { .. } => "texture.is_null.heap",
            TextureStrategy::Sampling(s) => s.name(),
            TextureStrategy::NullHandle => "texture.null_handle",
            TextureStrategy::Switch(s) => return s.name(),
            TextureStrategy::ImageblockSlice(s) if s.guard => "texture.imageblock_slice_guarded",
            TextureStrategy::ImageblockSlice(_) => "texture.imageblock_slice",
            TextureStrategy::ExtendedSample(e) => e.name(),
            TextureStrategy::InPlaceSample(e) => e.name(),
            TextureStrategy::AbsentArgument(_) => AbsentArgument::NAME,
        })
    }

    /// The estimated words.
    fn base_words(&self) -> u32 {
        match self {
            TextureStrategy::Read(r) => read_words(r),
            // The sRGB encode is the longest conversion.
            TextureStrategy::Write(w) => {
                96 + if w.guard { 48 } else { 0 }
                    + if w.image.heap() { 40 } else { 0 }
                    + if w.extended.is_some() {
                        crate::lower::arith::extended_range::ENCODE_WORDS
                    } else {
                        0
                    }
                    // A two-plane 3D texture's first-slice coordinate and its test.
                    + if matches!(
                        w.extended,
                        Some(crate::metal::format::ExtendedRange::Planes { .. })
                    ) {
                        64
                    } else {
                        0
                    }
                    + emit::class_write_words(w.class)
                    + if w.heap_alpha_red {
                        emit::HEAP_ALPHA_IN_RED_WORDS
                    } else {
                        0
                    }
                    + if w.cross.is_some() { 48 } else { 0 }
                    + w.saturate.words()
            }
            TextureStrategy::Atomic(a) => a.words(),
            TextureStrategy::Query(q) => {
                24 + if q.image.heap() { 32 } else { 0 }
                    + if q.volume { 72 } else { 0 }
                    + if q.level.is_some() { 24 } else { 0 }
                    + if q.heap_raw { heap_raw::QUERY_WORDS } else { 0 }
                    + if q.heap_cross {
                        2 * heap_raw::QUERY_WORDS
                    } else {
                        0
                    }
            }
            TextureStrategy::Fence => 8,
            TextureStrategy::IsNull { .. } => 8,
            TextureStrategy::HeapIsNull { .. } => 12,
            TextureStrategy::Sampling(s) => s.words(),
            TextureStrategy::NullHandle => 4,
            TextureStrategy::Switch(s) => s.words(),
            TextureStrategy::ImageblockSlice(s) => s.words(),
            TextureStrategy::ExtendedSample(e) => e.words(),
            TextureStrategy::InPlaceSample(e) => e.words(),
            // `access::absent::AbsentRead`'s status write, and the zero, a constant.
            TextureStrategy::AbsentArgument(a) => {
                if a.marks {
                    32
                } else {
                    0
                }
            }
        }
    }
}

/// The words of an access's sparse arm: the specialized branch, the sparse access, its two
/// extractions, `OpImageSparseTexelsResident`, the word's select, and the merge's two phis.
pub(crate) const RESIDENCY_WORDS: u32 = 48;

/// The texture family's part of the per-function plan (`select::FamilyPlans`).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TexturePlan {
    /// Every texture slot the function uses, by slot index.
    pub slots: BTreeMap<u32, SlotPlan>,
    /// Texture calls whose `{T, i8}` status word is observed, by how (`status::observed`):
    /// computed on the first query while selecting.
    pub status_read: Option<BTreeMap<InstId, StatusUse>>,
    /// Four-lane reads whose texel's alpha is observed (`alpha::observed`): computed on the first
    /// read the shader checks while selecting.
    pub alpha_read: Option<BTreeSet<InstId>>,
    /// The function's heap texture handles, grouped: computed on the first heap use while
    /// selecting.
    pub heap_groups: Option<HeapGroups>,
    /// Every heap group an operation uses, by group.
    pub heap: BTreeMap<u32, HeapPlan>,
    /// Some heap color texture's sample or gather has a `half` result, which converts by the
    /// class its handle carries (`emit::heap_half_class`): the module defines the `half` routine.
    pub heap_half: bool,
    /// The results of the heap reads that decode a member of a form a heap reads
    /// (`ExtendedRange::HEAP_READ`): `false` a `float` one, `true` a `half` one. The module
    /// defines that decode's routine for each ([`heap_raw::define`]).
    pub heap_raw: BTreeSet<bool>,
    /// The guard conditions of the slice writes every invocation shares (`cooperative`): each
    /// one's branch is taken by every invocation, so its condition is emitted as `true`.
    pub cooperative: BTreeSet<ValueId>,
    /// The heap operations the function selects (texture and sampling): each masks its handle
    /// where the handle-table search carries flags in it (`AccessPlan::lookup`).
    pub heap_uses: u32,
    /// The instructions with a strategy that the default presence specialization never runs (a
    /// nil texture's arm), sorted (`absent_arm`), whose words the census counts under
    /// [`ABSENT_ARM`]: computed after the body's selection.
    pub absent_arm: Vec<ValueId>,
    /// The measured driver deviations, as their properties, the function's storage writes rely
    /// on (`select_write`): a `float` write to a half-float storage image, left to the device's
    /// conversion because it stores Metal's half (`MetalHalfStorageWrites`) and its NaN
    /// (`MetalHalfStorageNans`), a write that may reach a normalized format, because it stores
    /// Metal's codes (`MetalNormStorageWrites`), and a `half` texel's NaN to a 16-bit float format
    /// (`MetalHalfTexelWrites`). The module requires them.
    pub storage_properties: BTreeSet<crate::device::requirements::VkProperty>,
    /// The function's depth comparisons the device makes read depth images that keep values
    /// outside `[0, 1]` without `VK_EXT_depth_range_unrestricted` because the driver's do
    /// (`VkProperty::DepthImageKeepsValuesOutsideUnitRange`, [`depth_comparison`]): the module
    /// requires that property.
    pub kept_depth_range: bool,
}

impl TexturePlan {
    /// Some heap group is declared as a storage-image class, once [`finish`] has run: its reads,
    /// writes, and atomics are storage-image accesses, as a storage slot's are, which the Vulkan
    /// memory model orders only with `NonPrivateTexel` (`sync::memory_model`).
    pub fn heap_storage_images(&self) -> bool {
        self.heap.values().any(|h| {
            h.decl
                .is_some_and(|d| d.usage == crate::spirv::ir::ImageUsage::Storage)
        })
    }

    /// Whether `image` is volatile, once [`finish`] has run: its slot's, or its heap group's
    /// class's.
    fn volatile(&self, image: &ImageRef) -> bool {
        match image {
            ImageRef::Slot(index) => self.slots.get(index).is_some_and(|s| s.volatile),
            ImageRef::Heap { group, .. } => self.heap.get(group).is_some_and(|h| h.volatile),
        }
    }
}

/// Whether `plan`'s accesses of `image` carry the Vulkan memory model's volatile operand
/// (`VolatileTexel`, or an atomic's `Volatile` semantics): a volatile image under that model, which
/// bans the `Volatile` decoration `GLSL450` takes ([`declare`]).
pub(crate) fn volatile_texel(plan: &Plan, image: &ImageRef) -> bool {
    plan.families.memory_model.model == crate::lower::sync::barrier::BarrierModel::Vulkan
        && plan.families.texture.volatile(image)
}

/// Whether `plan` decorates its volatile images `Volatile`: under `GLSL450`.
fn volatile_decorated(plan: &Plan) -> bool {
    plan.families.memory_model.model == crate::lower::sync::barrier::BarrierModel::Glsl450
}

/// The texture family's module-scope objects.
#[derive(Clone, Debug, Default)]
pub struct Declared {
    /// The image variable and image type of each declared slot (the element type of a slot's
    /// level views).
    pub images: BTreeMap<u32, (GlobalId, TypeId)>,
    /// The `ResourcePresence` specialization constant of each slot whose null-ness is observed.
    pub presence: BTreeMap<u32, SValue>,
    /// The `OutOfBoundsAlpha` specialization constant of each slot one of whose reads selects
    /// Metal's out-of-bounds texel in the shader and observes its alpha: a read under
    /// [`ReadCheck::Shader`], or a level-view read under any check.
    pub oob_alpha: BTreeMap<u32, SValue>,
    /// The set-1 heap array of each declared heap class.
    pub heap: BTreeMap<ImageHandleClass, GlobalId>,
    /// The `SparseTexture` specialization constant of each slot with a sparse arm or a tile-map
    /// arm.
    pub sparse: BTreeMap<u32, SValue>,
    /// The tile map of each slot with a tile-map arm: a word view of its `ResourceKind::TileMap`
    /// binding.
    pub tile_maps: BTreeMap<u32, crate::memory::views::View>,
    /// The `TexelFormat` specialization constant of each slot whose texels a read, gather, or
    /// sample converts to a `half` result by the bound format's class.
    pub half_format: BTreeMap<u32, ConstId>,
    /// The module's routine of that conversion ([`half::define`]), defined once where a slot
    /// takes it.
    pub half_routine: Option<FuncId>,
    /// The `PlainHalfTexels` specialization constant of each of those slots: whether the bound
    /// class converts by the plain arm (`half` module docs).
    pub plain_half: BTreeMap<u32, SValue>,
    /// The `TextureSwizzle` specialization constant of each texture buffer slot whose reads
    /// apply the bound texture buffer's swizzle, and of each storage slot of
    /// `LoweringInputs::swizzled_storage` whose reads and writes apply the bound texture's, or
    /// extended-range sampled slot of it whose reads, samples and gathers do.
    pub swizzle: BTreeMap<u32, SValue>,
    /// The module's table of heap texel classes (`metal::texel_format::heap_classes`), a `Private`
    /// `uint[]`, where a heap handle carries its class's index (`AccessPlan::heap_class`):
    /// [`heap_texel_class`] reads it.
    pub heap_classes: Option<GlobalId>,
    /// The module's routine decoding a heap member of a form a heap reads, for a `float` result
    /// (`false`) and a `half` one (`true`), each where a read takes it ([`heap_raw::define`]).
    pub heap_raw: BTreeMap<bool, FuncId>,
    /// The tests of the `ConstantLanes` specialization constant of each slot whose accesses set
    /// the lanes Metal reads as constants.
    pub constant_lanes: BTreeMap<u32, LaneConstants>,
    /// The tests of the `OffsetBaseAxes` specialization constant of each slot whose accesses
    /// place a texel offset by Metal's rule on an axis that may repeat or mirror.
    pub offset_base: BTreeMap<u32, OffsetBase>,
}

/// A slot's `OffsetBaseAxes` specialization as `OpSpecConstantOp` tests, fixed at pipeline
/// creation, so an access spends only its disjunction on it: on plane axis `a`, the base-level
/// size Metal's texel-offset rule reads, a view's parent's, is not a power of two where the bound
/// view's base level's is, where `uneven[a]` holds (`sampler::position::offset_dropped`).
#[derive(Clone, Copy, Debug)]
pub struct OffsetBase {
    pub uneven: [SValue; 3],
}

impl OffsetBase {
    /// Axis `a`'s test, where `base` holds one ([`OffsetBase::uneven`]).
    pub fn axis(base: Option<OffsetBase>, a: u32) -> Option<SValue> {
        base.and_then(|b| b.uneven.get(a as usize).copied())
    }
}

/// A slot's `ConstantLanes` specialization as `OpSpecConstantOp` tests, fixed at pipeline creation,
/// so an access spends only its selections on it: lane `l` is a constant where `constant[l]` holds,
/// and that constant is 1 rather than 0 where `one[l]` does.
#[derive(Clone, Copy, Debug)]
pub struct LaneConstants {
    pub constant: [SValue; 4],
    pub one: [SValue; 4],
}

/// Slot `slot`'s `ConstantLanes` specialization and its tests: bit `l` of the word, and bit `4 + l`
/// (`metal::sampler::CONSTANT_ONES`), each as a `bool` fixed at pipeline creation.
fn lane_constants(mb: &mut ModuleBuilder, slot: SlotRef) -> BResult<LaneConstants> {
    use crate::spirv::ir::SpecOp;
    let u32_ = mb.ty(SType::Int {
        width: 32,
        signed: false,
    })?;
    let bool_ = mb.ty(SType::Bool)?;
    let word = mb.spec(
        crate::pipeline::sym::SpecKey::ConstantLanes(slot),
        u32_,
        ConstBits(0),
    )?;
    let k = |mb: &mut ModuleBuilder, v: u32| {
        mb.constant(crate::spirv::ir::Constant::Scalar {
            ty: u32_,
            bits: ConstBits(u64::from(v)),
        })
    };
    let (zero, one) = (k(mb, 0)?, k(mb, 1)?);
    let bit = |mb: &mut ModuleBuilder, at: u32| -> BResult<SValue> {
        let shift = k(mb, at)?;
        let shifted = mb.spec_op(SpecOp::ShiftRightLogical, u32_, &[word, shift])?;
        let masked = mb.spec_op(SpecOp::BitwiseAnd, u32_, &[shifted, one])?;
        let set = mb.spec_op(SpecOp::ULessThan, bool_, &[zero, masked])?;
        mb.const_value(set)
    };
    let never = mb.const_value(zero)?;
    let mut lanes = LaneConstants {
        constant: [never; 4],
        one: [never; 4],
    };
    let ones = crate::metal::sampler::CONSTANT_ONES;
    for (lane, (constant, one)) in (0u32..).zip(lanes.constant.iter_mut().zip(&mut lanes.one)) {
        *constant = bit(mb, lane)?;
        *one = bit(mb, ones + lane)?;
    }
    Ok(lanes)
}

/// Slot `slot`'s `OffsetBaseAxes` specialization and its tests: bit `a` of the word, each as a
/// `bool` fixed at pipeline creation ([`OffsetBase`]).
fn offset_base(mb: &mut ModuleBuilder, slot: SlotRef) -> BResult<OffsetBase> {
    use crate::spirv::ir::SpecOp;
    let u32_ = mb.ty(SType::Int {
        width: 32,
        signed: false,
    })?;
    let bool_ = mb.ty(SType::Bool)?;
    let word = mb.spec(
        crate::pipeline::sym::SpecKey::OffsetBaseAxes(slot),
        u32_,
        ConstBits(0),
    )?;
    let k = |mb: &mut ModuleBuilder, v: u32| {
        mb.constant(crate::spirv::ir::Constant::Scalar {
            ty: u32_,
            bits: ConstBits(u64::from(v)),
        })
    };
    let (zero, one) = (k(mb, 0)?, k(mb, 1)?);
    let mut bit = |a: u32| -> BResult<SValue> {
        let shift = k(mb, a)?;
        let shifted = mb.spec_op(SpecOp::ShiftRightLogical, u32_, &[word, shift])?;
        let masked = mb.spec_op(SpecOp::BitwiseAnd, u32_, &[shifted, one])?;
        let set = mb.spec_op(SpecOp::ULessThan, bool_, &[zero, masked])?;
        mb.const_value(set)
    };
    Ok(OffsetBase {
        uneven: [bit(0)?, bit(1)?, bit(2)?],
    })
}

/// Declare the image variables and presence constants of the plan's texture slots.
pub fn declare(
    mb: &mut ModuleBuilder,
    plan: &Plan,
    _modes: &mut Vec<ExecMode>,
) -> BResult<Declared> {
    let mut d = Declared::default();
    for (index, s) in &plan.families.texture.slots {
        if let Some(decl) = &s.decl {
            let image = decl.declare(mb)?;
            // A slot's level views are one descriptor array of single-level views.
            let ty = match decl.level_views {
                Some(n) => {
                    let u32_ = mb.ty(SType::Int {
                        width: 32,
                        signed: false,
                    })?;
                    let len = mb.constant(crate::spirv::ir::Constant::Scalar {
                        ty: u32_,
                        bits: ConstBits(u64::from(n)),
                    })?;
                    mb.ty(SType::Array {
                        elem: image,
                        len,
                        stride: None,
                    })?
                }
                None => image,
            };
            // A planes image's slot is the combined image sampler of the device's conversion
            // (`ResourceKind::YCbCrPlanes`).
            let (ty, symbol) = match decl.planes_sampler {
                Some(sampler) => (
                    mb.ty(SType::SampledImage { image: ty })?,
                    crate::pipeline::sym::ResourceSlot {
                        stage: s.slot.stage,
                        kind: crate::pipeline::sym::ResourceKind::YCbCrPlanes(sampler),
                        index: s.slot.index,
                    },
                ),
                None => (ty, crate::pipeline::sym::ResourceSlot::metal(s.slot)),
            };
            let g = mb.resource(symbol, crate::spirv::ir::StorageClass::UniformConstant, ty)?;
            if s.volatile && volatile_decorated(plan) {
                mb.volatile(g)?;
            }
            d.images.insert(*index, (g, image));
        }
        if s.presence {
            let b = mb.ty(SType::Bool)?;
            let c = mb.spec(
                crate::pipeline::sym::SpecKey::Presence(s.slot),
                b,
                ConstBits(1),
            )?;
            d.presence.insert(*index, mb.const_value(c)?);
        }
        if s.oob_alpha {
            let b = mb.ty(SType::Bool)?;
            let c = mb.spec(
                crate::pipeline::sym::SpecKey::OutOfBoundsAlpha(s.slot),
                b,
                ConstBits(0),
            )?;
            d.oob_alpha.insert(*index, mb.const_value(c)?);
        }
        if s.sparse_arm || s.tile_map || s.sparse_writes {
            let b = mb.ty(SType::Bool)?;
            let c = mb.spec(
                crate::pipeline::sym::SpecKey::SparseTexture(s.slot),
                b,
                ConstBits(0),
            )?;
            d.sparse.insert(*index, mb.const_value(c)?);
        }
        if s.tile_map {
            let sym = crate::pipeline::sym::ResourceSlot {
                stage: s.slot.stage,
                kind: crate::pipeline::sym::ResourceKind::TileMap,
                index: s.slot.index,
            };
            let view =
                crate::memory::views::declare_at(mb, sym, crate::memory::views::ViewKind::Storage)?;
            d.tile_maps.insert(*index, view);
        }
        if s.declares_texel_format() {
            let u32_ = mb.ty(SType::Int {
                width: 32,
                signed: false,
            })?;
            let c = mb.spec(
                crate::pipeline::sym::SpecKey::TexelFormat(s.slot),
                u32_,
                ConstBits(u64::from(crate::metal::texel_format::TexelFormat::OTHER.0)),
            )?;
            d.half_format.insert(*index, c);
        }
        if s.unmeasured {
            let b = mb.ty(SType::Bool)?;
            mb.spec(
                crate::pipeline::sym::SpecKey::UnmeasuredTexels(s.slot),
                b,
                ConstBits(0),
            )?;
        }
        if s.half_format && d.half_routine.is_none() {
            d.half_routine = Some(half::define(mb)?);
        }
        if s.half_format {
            let b = mb.ty(SType::Bool)?;
            let c = mb.spec(
                crate::pipeline::sym::SpecKey::PlainHalfTexels(s.slot),
                b,
                ConstBits(1),
            )?;
            d.plain_half.insert(*index, mb.const_value(c)?);
        }
        if s.swizzle {
            let u32_ = mb.ty(SType::Int {
                width: 32,
                signed: false,
            })?;
            let c = mb.spec(
                crate::pipeline::sym::SpecKey::TextureSwizzle(s.slot),
                u32_,
                ConstBits(u64::from(
                    crate::metal::TextureSwizzleChannels::DEFAULT_WORD,
                )),
            )?;
            d.swizzle.insert(*index, mb.const_value(c)?);
        }
        if s.constant_lanes {
            let lanes = lane_constants(mb, s.slot)?;
            d.constant_lanes.insert(*index, lanes);
        }
        if s.offset_base {
            let base = offset_base(mb, s.slot)?;
            d.offset_base.insert(*index, base);
        }
    }
    for h in plan.families.texture.heap.values() {
        let (Some(class), Some(decl)) = (h.class, h.decl) else {
            continue;
        };
        if d.heap.contains_key(&class) {
            continue;
        }
        let g = heap_array(mb, plan.stage, class, decl)?;
        if h.volatile && volatile_decorated(plan) {
            mb.volatile(g)?;
        }
        d.heap.insert(class, g);
    }
    // The float classes the shader's heap comparisons read their members through, each declared
    // once, after the groups' own: a group of that class decorates it.
    for (class, decl) in plan.families.texture.heap.values().filter_map(|h| h.plain) {
        if d.heap.contains_key(&class) {
            continue;
        }
        let g = heap_array(mb, plan.stage, class, decl)?;
        d.heap.insert(class, g);
    }
    // The unsigned integer classes the reads of raw heap members fetch them from, each declared
    // once, and the routines decoding them.
    for (class, decl) in plan.families.texture.heap.values().filter_map(|h| h.raw) {
        if d.heap.contains_key(&class) {
            continue;
        }
        let g = heap_array(mb, plan.stage, class, decl)?;
        d.heap.insert(class, g);
    }
    // The classes of other numeric types the reads and queries of a member of another class
    // read it through, each declared once.
    for (class, decl) in plan
        .families
        .texture
        .heap
        .values()
        .flat_map(|h| h.cross.values().copied())
    {
        if d.heap.contains_key(&class) {
            continue;
        }
        let g = heap_array(mb, plan.stage, class, decl)?;
        d.heap.insert(class, g);
    }
    for &half in &plan.families.texture.heap_raw {
        d.heap_raw.insert(half, heap_raw::define(mb, half)?);
    }
    if plan.families.access.heap_class {
        d.heap_classes = Some(declare_heap_classes(mb)?);
    }
    if plan.families.texture.heap_half && d.half_routine.is_none() {
        d.half_routine = Some(half::define(mb)?);
    }
    Ok(d)
}

/// Declare the module's table of heap texel classes: a `Private` `uint[N]` initialized with
/// `metal::texel_format::heap_classes`, whose element `i` is the class index `i` names.
fn declare_heap_classes(mb: &mut ModuleBuilder) -> BResult<GlobalId> {
    let classes = crate::metal::texel_format::heap_classes();
    let u32_ = mb.ty(SType::Int {
        width: 32,
        signed: false,
    })?;
    let len = mb.constant(crate::spirv::ir::Constant::Scalar {
        ty: u32_,
        bits: ConstBits(classes.len() as u64),
    })?;
    let arr = mb.ty(SType::Array {
        elem: u32_,
        len,
        stride: None,
    })?;
    let mut parts = Vec::with_capacity(classes.len());
    for &class in classes {
        parts.push(mb.constant(crate::spirv::ir::Constant::Scalar {
            ty: u32_,
            bits: ConstBits(u64::from(class)),
        })?);
    }
    let init = mb.constant(crate::spirv::ir::Constant::Composite {
        ty: arr,
        parts: parts.into(),
    })?;
    mb.plain_global(crate::spirv::ir::StorageClass::Private, arr, Some(init))
}

/// The words [`heap_texel_class`] emits, at most: the field's shift and mask, the clamp into the
/// table, the access chain and the load, with their constants.
pub(crate) const HEAP_CLASS_WORDS: u32 = 48;

/// The words of the module's table of heap texel classes, counted once: a constant per class, the
/// composite, the array type, and the variable.
pub(crate) fn heap_classes_words() -> u64 {
    16 + 4 * crate::metal::texel_format::heap_classes().len() as u64
}

/// The texel class (`metal::texel_format::TexelFormat`'s bits, a `u32`) of the heap texture
/// whose handle is `h`, a handle of a module whose search carries the class's index
/// ([`HANDLE_TEXEL_CLASS`](crate::memory::handles::HANDLE_TEXEL_CLASS)): the module's table's
/// element at that index, clamped into the table. A null handle's, or a missed id's, is index 0,
/// `TexelFormat::OTHER`.
pub(crate) fn heap_texel_class(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    h: SValue,
) -> BResult<SValue> {
    use crate::memory::handles::HANDLE_TEXEL_CLASS;
    let Some(table) = cx.decl.families.texture.heap_classes else {
        return Err(cx.fail("a heap texel class without the module's table"));
    };
    let u32_ = fb.module().ty(SType::Int {
        width: 32,
        signed: false,
    })?;
    let k = |fb: &mut FunctionBuilder<'_>, v: u32| -> BResult<SValue> {
        let c = fb.module().constant(crate::spirv::ir::Constant::Scalar {
            ty: u32_,
            bits: ConstBits(u64::from(v)),
        })?;
        fb.module().const_value(c)
    };
    let shift = k(fb, HANDLE_TEXEL_CLASS.trailing_zeros())?;
    let mask = k(
        fb,
        HANDLE_TEXEL_CLASS >> HANDLE_TEXEL_CLASS.trailing_zeros(),
    )?;
    let last = crate::metal::texel_format::heap_classes()
        .len()
        .saturating_sub(1);
    let last = k(fb, u32::try_from(last).unwrap_or(0))?;
    let index = fb.binary(BinOp::ShiftRightLogical, h, shift)?;
    let index = fb.binary(BinOp::BitwiseAnd, index, mask)?;
    let index = fb.glsl(GlslOp::UMin, &[index, last])?;
    let base = fb.module().global_value(table)?;
    let at = fb.access_chain(base, &[index])?;
    fb.load(at, MemAccess::default())
}

/// The set-1 runtime array of heap class `class`, of `decl`'s image type, for stage `stage`.
fn heap_array(
    mb: &mut ModuleBuilder,
    stage: crate::pipeline::sym::Stage,
    class: ImageHandleClass,
    decl: ImageDecl,
) -> BResult<GlobalId> {
    let ty = decl.declare(mb)?;
    let array = mb.ty(SType::RuntimeArray {
        elem: ty,
        stride: None,
    })?;
    mb.resource(
        crate::pipeline::sym::ResourceSlot {
            stage,
            kind: crate::pipeline::sym::ResourceKind::Heap(
                crate::pipeline::sym::HandleClass::Image(class),
            ),
            index: 0,
        },
        crate::spirv::ir::StorageClass::UniformConstant,
        array,
    )
}

/// Plan a used texture parameter: a static handle, or an array of `N` handles in slots
/// `n .. n + N`. Its uses declare each slot's shape.
pub fn select_param(cx: &mut SelectCx<'_>, b: &ParamBinding) -> Result<(), Error> {
    let ParamRole::Texture(t) = &b.role else {
        return Err(refuse());
    };
    let (AbiInt::Const(first), AbiInt::Const(count)) = (t.location.index, t.location.count) else {
        return Err(refuse());
    };
    for k in 0..count {
        let index = first.checked_add(k).ok_or_else(refuse)?;
        let slot = SlotRef {
            stage: cx.stage,
            class: SlotClass::Texture,
            index,
        };
        cx.families
            .texture
            .slots
            .entry(index)
            .or_insert_with(|| SlotPlan::new(slot, t.access));
    }
    Ok(())
}

/// The value of a texture parameter: its handle index, or offset 0 into a handle array
/// (`memory::handles`). Texture operations name slots; this value feeds only handle selects, phis,
/// and array element loads.
pub fn param_value(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    b: &ParamBinding,
) -> BResult<SValue> {
    let v = match crate::memory::handles::binding_slots(b) {
        Some((_, first, 1))
            if crate::memory::provenance::handle_array_of(cx.entry, b).is_none() =>
        {
            Candidate::Texture(first).handle()
        }
        Some(_) => 0,
        None => return Err(cx.fail(format!("parameter {} is no texture slot", b.param))),
    };
    let c = fb.u32_const(v)?;
    fb.module().const_value(c)
}

/// How the `{T, i8}` status word of texture call `at` is observed ([`status::observed`]); `None`
/// where it is not.
fn status_use(cx: &mut SelectCx<'_>, at: InstId) -> Option<StatusUse> {
    if cx.families.texture.status_read.is_none() {
        let observed = status::observed(cx, cx.body);
        cx.families.texture.status_read = Some(observed);
    }
    cx.families
        .texture
        .status_read
        .as_ref()
        .and_then(|s| s.get(&at).copied())
}

/// Whether the function observes the residency of texture call `at` on `image`: bit 0 of its status
/// word alone, on a slot. A heap element's word (its residency needs a word per handle-table
/// entry), and a word with another bit observed, are refused. A read's and a fetch's residency and
/// value are exact on both arms, as are a nearest sample's and a gather's of Metal's footprint; a
/// filtered sample's, and an access's outside the footprint, records
/// `PrecisionNote::SparseResidency` where it is not (`lower::sampler`).
pub(crate) fn residency(cx: &mut SelectCx<'_>, at: InstId, image: ImageRef) -> Result<bool, Error> {
    match status_use(cx, at) {
        None => Ok(false),
        Some(StatusUse::Residency) if !image.heap() => {
            mark(cx, image, |u| u.residency = true);
            Ok(true)
        }
        Some(_) => Err(refuse()),
    }
}

/// Whether an access with a `float` or `half` result reads one texel of one level, its value a
/// texel's channels the device converted from their codes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum OneTexel {
    /// Always: a read, a 2D gather, a pixel fetch, or a sample through a constexpr sampler that
    /// reads one texel of one level (`lower::sampler::linear::unfiltered`).
    Always,
    /// Never: a sample through a constexpr sampler that filters, a bicubic sample, or a cube
    /// gather (whose corner texels are made up).
    Never,
    /// Where runtime sampler slot `n`'s word states a sampler that reads one texel of one level
    /// (its `SamplerState` specialization's filter fields, `SamplerWord::LINEAR_FILTER`).
    Word(u32),
    /// Where an argument buffer's sampler's word, its table entry's flags for handle `h`, states
    /// one; no specialization decides it.
    Heap(ValueId),
}

/// A read, gather, or sample of `image`, of shape `shape`, with a `half` result, a sample where
/// `sample` (a sample's and a pixel fetch's form: not a read's or a gather's). Its `half` is the
/// device's `f32` texel rounded to nearest even (`narrow::round_texel`) on every device, recorded
/// as `PrecisionNote::HalfTexelConversion` where Metal's `half` result may differ from it.
///
/// Where the device's texel is its own (`device`, [`device_texels`]: off Metal's own texture unit)
/// the access records the note outright, a heap element's and a depth texture's too. On a device
/// that samples through Metal's own texture unit (MoltenVK on Apple's GPU, whose `f32` texel is
/// Metal's `float` result), a slot's color texture converts through the module's routine (`half`),
/// which keeps a YCbCr field's ties away from zero on the grid `8192`, by the bound format's class
/// (its `TexelFormat` specialization, `RuntimeFact::TexelFormat`); a heap color texture's sample or
/// gather by the class its handle carries (its caller asks the routine). The note holds there
/// where the bound class's `half` is not that `float` rounded to nearest even: a class measured in
/// none of the families (the slot's `UnmeasuredTexels` specialization), a 16-bit normalized grid
/// (`BicubicTexels::WIDE16`: three steps of each round twice), and a sample of a 32-bit float
/// texel (`BicubicTexels::LINEAR`: Metal rounds toward zero, saturating), tests of the bound
/// format the runtime announces at a draw or dispatch binding such a texture. A heap color
/// texture's and a depth texture's access records it outright.
pub(crate) fn half_texels(
    cx: &mut SelectCx<'_>,
    image: ImageRef,
    shape: ImageShape,
    sample: bool,
    device: bool,
    m: &mut Meter,
) -> Result<(), Error> {
    use crate::metal::bicubic::BicubicTexels as B;
    let index = match image {
        ImageRef::Slot(index) if !shape.depth && !device => index,
        _ => {
            cx.precision.insert(PrecisionNote::HalfTexelConversion);
            return Ok(());
        }
    };
    // The module requires the device's sampler to be Metal's.
    cx.families.sampler.metal_sampler = true;
    if let Some(s) = cx.families.texture.slots.get_mut(&index) {
        s.unmeasured = true;
    }
    let mut when = vec![
        [NoteTest::equal(NoteOn::Unmeasured(index), 1, 1)].into(),
        [NoteTest::differ(NoteOn::Bound(index), B::WIDE16, 0)].into(),
    ];
    if sample {
        when.push([NoteTest::differ(NoteOn::Bound(index), B::LINEAR, 0)].into());
    }
    cx.note_when(PrecisionNote::HalfTexelConversion, when, m)?;
    let slot = slot_of(cx, index)?.slot;
    if let Some(s) = cx.families.texture.slots.get_mut(&index) {
        s.half_format = true;
    }
    cx.facts
        .insert(crate::pipeline::facts::RuntimeFact::TexelFormat { slot });
    Ok(())
}

/// Record that a `half` read, sample or gather of `image`, of shape `shape`, hands back the
/// device's decode of a NaN texel: `PrecisionNote::DeviceNanEncoding`. Of a color texture slot,
/// only where the class of the format bound there may hold a NaN (none of `TexelFormat::NAN_FREE`'s
/// families): a test of `SpecKey::BoundTexelFormat`, which the runtime reads from the bound texture
/// and no pipeline's key carries. A normalized, sRGB or extended-range texel decodes to no NaN, so
/// the device's encoding of one cannot show. Of a heap's element or a depth texture, wherever it
/// reads.
pub(crate) fn half_texel_nans(
    cx: &mut SelectCx<'_>,
    image: ImageRef,
    shape: ImageShape,
    m: &mut Meter,
) -> Result<(), Error> {
    match image {
        ImageRef::Slot(index) if !shape.depth => {
            let free = crate::metal::texel_format::TexelFormat::NAN_FREE;
            let when = vec![[NoteTest::equal(NoteOn::Bound(index), free, 0)].into()];
            cx.note_when(PrecisionNote::DeviceNanEncoding, when, m)
        }
        _ => {
            cx.precision.insert(PrecisionNote::DeviceNanEncoding);
            Ok(())
        }
    }
}

/// Record what a read, gather, or sample of `image`, of shape `shape`, with a `float` result that
/// may read one texel (`one` not [`OneTexel::Never`]) answers on a profile without Metal's own
/// sampler (`VkProperty::MetalSampler`, whose device value is Metal's): the device's conversion of
/// the texel's codes, `PrecisionNote::DeviceTexelConversion`; of a heap's texture buffer, which no
/// class reaches, a half-float texel's NaN the device's decode, `PrecisionNote::DeviceNanEncoding`.
pub(crate) fn float_texel_notes(
    cx: &mut SelectCx<'_>,
    image: ImageRef,
    shape: ImageShape,
    one: OneTexel,
) {
    if one == OneTexel::Never || !device_texels(cx) {
        return;
    }
    let note = match image.heap() && shape.dim == ImageDim::Buffer {
        true => PrecisionNote::DeviceNanEncoding,
        false => PrecisionNote::DeviceTexelConversion,
    };
    cx.precision.insert(note);
}

/// Whether an access's `float` or `half` texel is the device's conversion: on a profile whose
/// device does not sample through Metal's own texture unit (`VkProperty::MetalSampler`, whose value
/// is Metal's and keeps the conversion). Every access of a kind agrees: the selection states a
/// sample's `half` conversion on the access (`Sampling::half_texel`), which the emission follows.
pub(crate) fn device_texels(cx: &SelectCx<'_>) -> bool {
    !cx.profile
        .fields()
        .enabled
        .properties
        .contains(&crate::device::requirements::VkProperty::MetalSampler)
}

/// The address-mode map constexpr sampler `words` samples texture slot `index` under: the map
/// of the texture the slot's variant binds there (`LoweringInputs::address_maps`), `None` for
/// the identity. The slot records `RuntimeFact::ConstexprSampled` with the sampler's modes,
/// which puts it in the variant where the binder binds a texture whose map changes one of them.
pub(crate) fn constexpr_address_map(
    cx: &mut SelectCx<'_>,
    index: u32,
    words: [u64; 2],
) -> Result<Option<crate::metal::sampler::AddressModeMap>, Error> {
    let slot = slot_of(cx, index)?.slot;
    let modes = crate::lower::sampler::address_modes(words)
        .ok_or_else(|| invariant("a constexpr sampler symbol with undecodable words"))?;
    cx.facts
        .insert(crate::pipeline::facts::RuntimeFact::ConstexprSampled { slot, modes });
    Ok(cx.inputs.address_maps.get(&slot).copied())
}

/// A read, sample, or gather of `image` (shape `shape`) with a `half` or `float` result, which an
/// 8-bit sRGB format bound to it would decode (`metal::texel_format`). On a sampled color texture
/// slot (not a texture buffer, a depth texture, or a storage slot) the module records
/// `RuntimeFact::SrgbTexture`.
///
/// Where the slot is one of `LoweringInputs::srgb_textures` (its variant: bound to a format the
/// device cannot sample, held as its UNORM twin, whose view reads the codes) the access decodes
/// what the view reads (one texel's codes, or the device's filter of them) by the MSL transfer
/// through the device's `Pow` (`emit::srgb_texel`), returning `true` and recording
/// `PrecisionNote::SrgbTexelDecode`.
///
/// Elsewhere the access reads the device's own decode through the slot's view, an sRGB format's
/// sRGB view: the module records `PrecisionNote::SrgbTexelDecode` where the bound format's class
/// is an sRGB one (`BicubicTexels::SRGB`, a test of the bound format that the runtime announces at
/// a draw or dispatch binding such a texture), and outright for a heap element, whose format is
/// never known. On a device that decodes through Metal's own texture unit
/// (`VkProperty::MetalSampler`: MoltenVK on Apple's GPU) the device's decode is Metal's, so the
/// module requires the property there instead of the note.
pub(crate) fn srgb_texels(
    cx: &mut SelectCx<'_>,
    image: ImageRef,
    shape: ImageShape,
    m: &mut Meter,
) -> Result<bool, Error> {
    if shape.dim == ImageDim::Buffer || shape.depth {
        return Ok(false);
    }
    let ImageRef::Slot(index) = image else {
        device_decode(cx, None, m)?;
        return Ok(false);
    };
    let s = slot_of(cx, index)?;
    if s.storage() {
        return Ok(false);
    }
    let slot = s.slot;
    cx.facts
        .insert(crate::pipeline::facts::RuntimeFact::SrgbTexture { slot });
    if !cx.inputs.srgb_textures.contains(&slot) {
        device_decode(cx, Some(index), m)?;
        return Ok(false);
    }
    cx.precision.insert(PrecisionNote::SrgbTexelDecode);
    Ok(true)
}

/// An access that reads the device's decode of a texture that may be bound in an 8-bit sRGB
/// format ([`srgb_texels`]): `PrecisionNote::SrgbTexelDecode` where texture slot `index` is bound
/// in such a format, or outright for a heap element (`None`); on a device that decodes through
/// Metal's own texture unit, the requirement of `VkProperty::MetalSampler` instead.
fn device_decode(cx: &mut SelectCx<'_>, index: Option<u32>, m: &mut Meter) -> Result<(), Error> {
    use crate::metal::bicubic::BicubicTexels as B;
    let metal = cx
        .profile
        .fields()
        .enabled
        .properties
        .contains(&crate::device::requirements::VkProperty::MetalSampler);
    match (metal, index) {
        (true, _) => cx.families.sampler.metal_sampler = true,
        (false, Some(index)) => {
            let srgb = NoteTest::equal(NoteOn::Bound(index), B::SRGB, B::SRGB);
            cx.note_when(PrecisionNote::SrgbTexelDecode, vec![[srgb].into()], m)?;
        }
        (false, None) => {
            cx.precision.insert(PrecisionNote::SrgbTexelDecode);
        }
    }
    Ok(())
}

/// The value id of an operand (constants included: emission materializes them).
fn id(cx: &SelectCx<'_>, v: Value) -> Result<ValueId, Error> {
    cx.values
        .id(v)
        .ok_or_else(|| cx.invariant("a texture operand out of the value range"))
}

/// `v` as an integer constant, if it is one.
fn const_int(cx: &SelectCx<'_>, v: Value) -> Option<u64> {
    let Value::Const(c) = v else { return None };
    match &cx.entry.module.consts().get(c)?.value {
        Const::Int(u) => Some(*u),
        Const::Null => Some(0),
        _ => None,
    }
}

/// Whether `v` is the zero constant (scalar or vector).
fn is_zero(cx: &SelectCx<'_>, v: Value) -> bool {
    let Value::Const(c) = v else { return false };
    match cx.entry.module.consts().get(c).map(|k| &k.value) {
        Some(Const::Null) | Some(Const::Int(0)) => true,
        Some(Const::Aggregate(parts)) => parts.iter().all(|p| {
            cx.entry
                .module
                .consts()
                .get(*p)
                .is_some_and(|k| matches!(k.value, Const::Null | Const::Int(0)))
        }),
        _ => false,
    }
}

/// The texel conversion of texel element `elem` with sign `sign`, and its numeric class.
pub(crate) fn texel_conv(elem: Elem, sign: Option<Sign>) -> Result<(TexelConv, TexelClass), Error> {
    Ok(match (elem, sign) {
        (Elem::F32, None) => (TexelConv::F32, TexelClass::Float),
        (Elem::F16, None) => (TexelConv::F16, TexelClass::Float),
        (Elem::I32, Some(Sign::Signed)) => (TexelConv::I32, TexelClass::Sint),
        (Elem::I32, Some(Sign::Unsigned)) => (TexelConv::I32, TexelClass::Uint),
        (Elem::I16, Some(Sign::Signed)) => (TexelConv::I16 { signed: true }, TexelClass::Sint),
        (Elem::I16, Some(Sign::Unsigned)) => (TexelConv::I16 { signed: false }, TexelClass::Uint),
        _ => return Err(refuse()),
    })
}

/// [`texel_conv`] of a read, write, or atomic on profile `p`, which also takes Metal's `ulong`
/// texel (no sampling takes it), held as `p` holds 64-bit integers.
fn image_texel_conv(
    elem: Elem,
    sign: Option<Sign>,
    p: &crate::device::profile::DeviceProfile,
) -> Result<(TexelConv, TexelClass), Error> {
    if (elem, sign) == (Elem::I64, Some(Sign::Unsigned)) {
        let wide = crate::lower::arith::legal::holds_wide(p);
        return Ok((TexelConv::U64Words { wide }, TexelClass::Uint64));
    }
    texel_conv(elem, sign)
}

/// The number of plane coordinates of `dim` for reads and writes (a cube addresses a face by
/// `(x, y)` and a face index).
fn plane_lanes(dim: ImageDim) -> u8 {
    match dim {
        ImageDim::D1 | ImageDim::Buffer => 1,
        ImageDim::D2 | ImageDim::Cube => 2,
        ImageDim::D3 => 3,
    }
}

/// The operands of a texture call, by role.
struct Operands<'a> {
    layout: Layout<TexRole>,
    args: &'a [Value],
}

impl Operands<'_> {
    fn get(&self, role: TexRole) -> Option<Value> {
        let i = self.layout.index(role)?;
        self.args.get(i).copied()
    }
}

/// Select the strategy of texture instruction `inst`.
pub fn select(
    cx: &mut SelectCx<'_>,
    at: InstId,
    inst: &Inst,
    m: &mut Meter,
) -> Result<TextureStrategy, Error> {
    let InstKind::Call(call) = &inst.kind else {
        return Err(cx.invariant("a texture instruction that is no call"));
    };
    let Callee::Air {
        op: AirIntrinsic::Texture(t),
        ..
    } = &call.callee
    else {
        return Err(cx.invariant("a texture call without a texture intrinsic"));
    };
    let ops = Operands {
        layout: t.layout(),
        args: &call.args,
    };
    let sampling = matches!(
        t.op,
        TexOp::Sample { .. }
            | TexOp::SampleGrad { .. }
            | TexOp::Gather { .. }
            | TexOp::CalculateLod { .. }
    );
    match t.op {
        // Sampling, gathers, and LOD queries take a sampler: the sampler family's, below.
        TexOp::Sample { .. }
        | TexOp::SampleGrad { .. }
        | TexOp::Gather { .. }
        | TexOp::CalculateLod { .. } => {}
        // The null handle (`memory::handles`).
        TexOp::GetNull => return Ok(TextureStrategy::NullHandle),
        // An image memory barrier: it names no image.
        TexOp::Fence => return Ok(TextureStrategy::Fence),
        TexOp::WriteImageblockSlice
        | TexOp::Read
        | TexOp::Write
        | TexOp::Atomic(_)
        | TexOp::Query(_)
        | TexOp::IsNull => {}
    }
    let tex = ops
        .get(TexRole::Texture)
        .ok_or_else(|| cx.invariant("a texture operation without a texture"))?;
    // A status word with a bit other than residency observed, or a whole `{T, i8}` value
    // exposed, is not established, whatever the texture. The null texture's word is 0, as its texel
    // is (the null arm, Q7).
    if status_use(cx, at) == Some(StatusUse::Other) {
        return Err(refuse());
    }
    let slice = t.op == TexOp::WriteImageblockSlice;
    if let Some(handle) = crate::lower::access::handle::heap_target(cx, tex)? {
        let group = heap_group(cx, tex, m)?;
        if slice {
            // A slice write to a heap element writes it as a heap write does.
            return slice::select(cx, (at, t), &ops, ImageRef::Heap { group, handle }, m);
        }
        if sampling {
            // The heap element through a separate heap sampler; its residency is refused (an LOD
            // query returns a plain float).
            if !matches!(t.op, TexOp::CalculateLod { .. }) {
                residency(cx, at, ImageRef::Heap { group, handle })?;
            }
            return crate::lower::sampler::select_heap_sampling(
                cx, at, inst, t, &call.args, group, handle, m,
            );
        }
        return select_image(cx, (at, inst), t, &ops, ImageRef::Heap { group, handle }, m);
    }
    let target = crate::lower::access::handle::texture_target(cx, tex, m)?;
    // A slot bound in an extended-range format is read, written, and sampled (the sampler
    // family's selection); its imageblock slices are not built yet.
    if slice {
        let slots: Vec<u32> = match &target {
            HandleTarget::One(Candidate::Texture(index)) => vec![*index],
            HandleTarget::Switch { cands, .. } => cands
                .iter()
                .filter_map(|c| match c {
                    Candidate::Texture(index) => Some(*index),
                    _ => None,
                })
                .collect(),
            _ => Vec::new(),
        };
        for index in slots {
            not_in_place(cx, index);
            if extended_range(cx, ImageRef::Slot(index))?.is_some() {
                return Err(Unsupported::not_implemented(Feature::Format).into());
            }
        }
    }
    if slice {
        return match target {
            HandleTarget::One(Candidate::Texture(index)) => {
                slice::select(cx, (at, t), &ops, ImageRef::Slot(index), m)
            }
            // A slice write to a switched or null texture is not modelled yet.
            _ => Err(Unsupported::not_implemented(Feature::Imageblocks).into()),
        };
    }
    if sampling {
        return crate::lower::sampler::select_sampling(cx, at, inst, t, &call.args, target, m);
    }
    let null = |cx: &SelectCx<'_>| -> Result<NullArm, Error> {
        Ok(match t.op {
            TexOp::IsNull => NullArm::True,
            TexOp::Write | TexOp::Atomic(AtomicOp::Store) => NullArm::Nothing,
            _ => NullArm::Zero(cx.lowered_repr(Value::Inst(at), inst.ty)?),
        })
    };
    match target {
        HandleTarget::One(Candidate::Texture(index)) => {
            select_image(cx, (at, inst), t, &ops, ImageRef::Slot(index), m)
        }
        HandleTarget::One(Candidate::Absent) => AbsentArgument::select(cx, at, inst, t),
        HandleTarget::One(_) => Err(cx.invariant("a texture operand with a sampler candidate")),
        // The constant null handle: the null arm alone, with nothing to dispatch on.
        HandleTarget::Null => Ok(TextureStrategy::Switch(Box::new(Switch {
            handles: Vec::new(),
            arms: Vec::new(),
            null: null(cx)?,
            prelude: Prelude::Nothing,
        }))),
        HandleTarget::Switch { handle, cands } => {
            let mut arms = Vec::with_capacity(cands.len());
            for c in cands {
                let arm = match c {
                    // A switch-at-use's arm reads its slot's image.
                    Candidate::Texture(index) => {
                        not_in_place(cx, index);
                        select_image(cx, (at, inst), t, &ops, ImageRef::Slot(index), m)?
                    }
                    Candidate::Absent => AbsentArgument::select(cx, at, inst, t)?,
                    _ => return Err(cx.invariant("a texture operand with a sampler candidate")),
                };
                arms.push((vec![c.handle()], arm));
            }
            Ok(TextureStrategy::Switch(Box::new(Switch {
                handles: vec![handle],
                arms,
                null: null(cx)?,
                prelude: Prelude::Nothing,
            })))
        }
    }
}

/// The heap group of heap texture handle `tex`, planned for its first use: the function's heap
/// handles are grouped once (`memory::handles::HeapGroups`), metered by `m`.
fn heap_group(cx: &mut SelectCx<'_>, tex: Value, m: &mut Meter) -> Result<u32, Error> {
    if cx.families.texture.heap_groups.is_none() {
        cx.families.texture.heap_groups =
            Some(HeapGroups::analyze(cx.entry, cx.body, cx.memory, m)?);
    }
    let groups = cx
        .families
        .texture
        .heap_groups
        .as_ref()
        .ok_or_else(|| cx.invariant("heap handle groups"))?;
    let Value::Inst(i) = tex else {
        return Err(cx.invariant("a heap handle that is no instruction"));
    };
    let group = groups
        .group(i)
        .ok_or_else(|| cx.invariant("a heap handle outside every group"))?;
    let access = groups
        .access(group)
        .ok_or_else(|| cx.invariant("a heap group without an access"))?;
    cx.families
        .texture
        .heap
        .entry(group)
        .or_insert_with(|| HeapPlan::new(access));
    cx.families.texture.heap_uses = cx.families.texture.heap_uses.saturating_add(1);
    Ok(group)
}

/// The words of the flags a heap handle carries: where the handle-table search carries any, every
/// heap operation masks its handle to its element (an `OpBitwiseAnd`), and a read that selects the
/// out-of-bounds alpha also tests that flag (an `OpBitwiseAnd` and an `OpINotEqual`), with their
/// constants once; the census's for the strategy that asks for them, `texture.read_clamped.heap`
/// where the alpha is carried, else [`HEAP_CLASS_CENSUS`]; with the module's table of heap texel classes
/// ([`heap_classes_words`]) under [`HEAP_CLASS_CENSUS`] where a sample reads it. And the words of
/// the `half` conversion routine (`half`), the census's for [`half::CENSUS_NAME`], where a slot
/// takes it.
pub fn routine_words(
    texture: &TexturePlan,
    access: &crate::lower::access::AccessPlan,
) -> impl Iterator<Item = (&'static str, u64)> {
    let flags = (access.heap_alpha || access.carries_class()).then(|| {
        let words = 16 + 16 * u64::from(texture.heap_uses);
        let name = match access.heap_alpha {
            true => "texture.read_clamped.heap",
            false => HEAP_CLASS_CENSUS,
        };
        (name, words)
    });
    let classes = access
        .heap_class
        .then(|| (HEAP_CLASS_CENSUS, heap_classes_words()));
    // The `half` conversion routine, once per module that defines it.
    let half = texture
        .slots
        .values()
        .any(|s| s.half_format)
        .then_some((half::CENSUS_NAME, half::ROUTINE_WORDS));
    // The heap's raw decode, once per result type.
    let raw = (!texture.heap_raw.is_empty()).then(|| {
        (
            heap_raw::CENSUS_NAME,
            heap_raw::ROUTINE_WORDS * texture.heap_raw.len() as u64,
        )
    });
    flags.into_iter().chain(classes).chain(half).chain(raw)
}

/// The census name of the words a heap texel class costs: the module's table of classes and, where
/// no other flag is carried, the masks of the heap handles.
pub const HEAP_CLASS_CENSUS: &str = "sampler.heap.texel_class";

/// What a `float` write to texture `image` (of storage class `class`) relies on or records where it
/// may reach a half-float image ([`may_store_half`]), which the device converts whole. Its rounding
/// of each channel to `half` (Metal rounds toward zero, a finite value past 65504 to 65504) is
/// Metal's where the profile takes `DriverDeviation::MetalHalfStorageWrites` (the module requires
/// its property), else the device's (`PrecisionNote::DeviceColorConversion`). A write of an
/// integer texel into a texture of the float class (`cross`, `emit::cross_write`), which Metal
/// stores rounded to nearest even onto a half-float or small-float channel, records the note
/// wherever it may reach one, the device's conversion rounding as it rounds a `float` write
/// (toward zero on every measured driver: within one unit in the last place of the channel's
/// Metal value, a finite value past 65504 stored as 65504 where Metal's `half` is the infinity).
/// Its NaN is Metal's
/// `0x7e00` where the profile takes `DriverDeviation::MetalHalfStorageNans` (the module requires its
/// property), else the device's encoding of it (`PrecisionNote::DeviceNanEncoding`). A write that
/// may reach a packed small float image ([`small_float_storage`]) stores its NaN as the device
/// converts it too: Metal stores every NaN there as its canonical one, and RADV and NVIDIA keep
/// another NaN's payload bits. A slot declared without a format records each note under the format
/// bound there (`m` charged for the clauses), which the runtime announces at a draw or dispatch
/// that binds such a texture.
pub(super) fn half_storage_facts(
    cx: &mut SelectCx<'_>,
    image: ImageRef,
    (class, cross): (StorageImageClass, bool),
    m: &mut Meter,
) -> Result<(), Error> {
    use crate::device::requirements::VkProperty;
    let half = match class {
        StorageImageClass::Identity => may_store_half(cx, image)?,
        _ => Vec::new(),
    };
    if !half.is_empty() {
        if !cross
            && cx
                .profile
                .takes_driver_deviation(DriverDeviation::MetalHalfStorageWrites)
        {
            cx.families
                .texture
                .storage_properties
                .insert(VkProperty::MetalHalfStorageWrites);
        } else {
            cx.note_when(PrecisionNote::DeviceColorConversion, half.clone(), m)?;
        }
        if cx
            .profile
            .takes_driver_deviation(DriverDeviation::MetalHalfStorageNans)
        {
            cx.families
                .texture
                .storage_properties
                .insert(VkProperty::MetalHalfStorageNans);
        } else {
            cx.note_when(PrecisionNote::DeviceNanEncoding, half, m)?;
        }
    }
    let small = small_float_storage(cx, image)?;
    if cross {
        cx.note_when(PrecisionNote::DeviceColorConversion, small.clone(), m)?;
    }
    cx.note_when(PrecisionNote::DeviceNanEncoding, small, m)
}

/// Where a write to texture `image` may store into a packed small float format (`RG11B10Float`;
/// `RGB9E5Float` is written through its word alias, whose word is Metal's), as the clauses of
/// [`stored_where`]: a heap element's, whose handle may name one, a slot's whose storage choice
/// declares one, and a slot's declared without a format where the texture bound there is of the
/// small floats' class (`metal::texel_format::PackedTexels::SmallFloat`).
fn small_float_storage(cx: &SelectCx<'_>, image: ImageRef) -> Result<Vec<NoteClause>, Error> {
    use crate::metal::texel_format::{PackedTexels, TexelFormat};
    let small = PackedTexels::SmallFloat.class().0;
    stored_where(
        cx,
        image,
        |f| f == crate::vk::Format::B10G11R11_UFLOAT_PACK32,
        |index| NoteTest::equal(NoteOn::Bound(index), TexelFormat::PACKED_MASK, small),
    )
}

/// Where a write to texture `image` left to the device's conversion stores into a format
/// `declared` accepts, as clauses for [`SelectCx::note_when`]: always (one clause of no test) for
/// a heap element, whose handle may name any format; for a slot whose storage choice declares a
/// format, always where `declared` accepts it, else never (no clause); and for a slot declared
/// without a format (a formatless choice, or a variant that states no choice, which
/// `slot::storage_format` declares formatless), where the format bound there passes `bound`'s
/// test of its class (`NoteOn::Bound`), which the runtime reads at each draw or dispatch.
fn stored_where(
    cx: &SelectCx<'_>,
    image: ImageRef,
    declared: impl Fn(crate::vk::Format) -> bool,
    bound: impl Fn(u32) -> NoteTest,
) -> Result<Vec<NoteClause>, Error> {
    let ImageRef::Slot(index) = image else {
        return Ok(vec![NoteClause::new()]);
    };
    let slot = slot_of(cx, index)?.slot;
    Ok(
        match cx.inputs.storage_images.get(&slot).map(|c| c.vk_format) {
            None | Some(crate::vk::Format::UNDEFINED) => vec![[bound(index)].into()],
            Some(f) if declared(f) => vec![NoteClause::new()],
            Some(_) => Vec::new(),
        },
    )
}

/// What a `half` texel's write to texture `image` records of its NaN. Metal stores a written
/// half's NaN as the half's own bits in a 16-bit float channel and as its conversion of the
/// canonical `0x7fc00000` in any other (the M3 Max under macOS 26.7, measured by the consumer; the
/// M4 under macOS 27.0.1: every NaN class, both signs, `write` and `read_write`, `Shared` and
/// `Private`). The write stores the half widened, its NaN's payload in the significand's top bits,
/// which the device converts: a 16-bit float channel's NaN is Metal's on a device of
/// `DriverDeviation::MetalHalfTexelWrites` (the module requires its property), and every other
/// float channel's, a 32-bit one's among them, is the device's.
///
/// The write records `PrecisionNote::DeviceNanEncoding` where the stored NaN stays the device's:
/// a 16-bit float channel's on a profile without `DriverDeviation::MetalHalfTexelWrites`, a 32-bit
/// or packed small float channel's, a slot's whose choice declares no format, a texel buffer's
/// declaring none, and a heap element's, which may be any of them. A normalized or integer channel
/// stores no NaN.
pub(super) fn half_write_nans(cx: &mut SelectCx<'_>, image: ImageRef) -> Result<(), Error> {
    use crate::device::requirements::VkProperty;
    use crate::vk::NumericFormat as N;
    // Whether the stored NaN is exact (`Exact`), a 16-bit float channel's (`Sixteen`, Metal's on
    // the measured deviation), or the device's (`Device`).
    #[derive(PartialEq)]
    enum Stored {
        Exact,
        Sixteen,
        Device,
    }
    let declared = |f: crate::vk::Format| -> Stored {
        let Some(b) = f.block() else {
            return Stored::Device;
        };
        match (b.color, b.component_bits, b.packed_bits) {
            (Some(N::Sfloat), Some(16), None) => Stored::Sixteen,
            (Some(N::Sfloat | N::Ufloat), _, _) => Stored::Device,
            _ => Stored::Exact,
        }
    };
    let stored = match image {
        ImageRef::Heap { .. } => Stored::Device,
        ImageRef::Slot(index) => {
            let slot = slot_of(cx, index)?.slot;
            match cx.inputs.storage_images.get(&slot) {
                Some(c) => match c.class {
                    StorageImageClass::Identity => declared(c.vk_format),
                    _ => Stored::Exact,
                },
                None => Stored::Device,
            }
        }
    };
    let measured = cx
        .profile
        .takes_driver_deviation(DriverDeviation::MetalHalfTexelWrites);
    match stored {
        Stored::Sixteen if measured => {
            cx.families
                .texture
                .storage_properties
                .insert(VkProperty::MetalHalfTexelWrites);
        }
        Stored::Sixteen | Stored::Device => {
            cx.precision.insert(PrecisionNote::DeviceNanEncoding);
        }
        Stored::Exact => {}
    }
    Ok(())
}

/// Whether a storage read or write of a `float` or `half` texel (`texel`) through heap handle
/// `image` of shape `shape` converts an `A8Unorm` member's texels: a heap access, not to a texture
/// buffer (no heap class holds one), on a device that holds such a texture's alpha in
/// its `R8_UNORM` image's red (`metal::format::alpha_in_red`), which the storage array views without
/// a component mapping (`bind::heap`'s storage elements). The handle carries its class index, and
/// where it is `A8Unorm`'s (`metal::texel_format::heap_alpha_in_red_index`) a read is Metal's
/// `(0, 0, 0, r)` and a write stores the alpha in red (`StorageImageClass::AlphaInRed`'s
/// conversion), as Metal reads and writes such a member (probe `a8`'s argument buffers).
pub(super) fn heap_alpha_in_red(
    cx: &mut SelectCx<'_>,
    image: ImageRef,
    shape: ImageShape,
    texel: TexelConv,
) -> bool {
    let applies = image.heap()
        && matches!(texel, TexelConv::F32 | TexelConv::F16)
        && shape.dim != ImageDim::Buffer
        && crate::metal::format::alpha_in_red(crate::metal::PixelFormat::A8Unorm, cx.profile);
    if applies {
        cx.families.access.heap_class_index = true;
    }
    applies
}

/// What a `float` (or `half`, held widened) texel's write in form `texel` to texture `image` of
/// shape `shape` relies on or records where the texture's format may be a normalized one, whose
/// conversion the device makes: Metal stores each channel's value clamped to its range (a NaN 0)
/// times its maximum code exactly, rounded to nearest with a tie away from zero, a 1-bit channel 1
/// only above one half. A device measured to store those codes
/// (`DriverDeviation::MetalNormStorageWrites`, whose property the module then requires) stores
/// them, but for a 1-bit channel, which it rounds otherwise (RADV rounds its half up): a write of
/// four lanes that may reach a texture of a 1-bit alpha (a heap element, or a slot whose storage
/// choice declares no format, where the device stores such a texture) records
/// `PrecisionNote::DeviceColorConversion`. Elsewhere every such write records it (ANV and
/// lavapipe round the `f32` product with ties to even). A slot whose storage choice declares a
/// format that is not normalized, or encodes it by its class (sRGB, a YCbCr write, the half and
/// small floats), records nothing.
pub(super) fn norm_storage_facts(
    cx: &mut SelectCx<'_>,
    image: ImageRef,
    shape: ImageShape,
    texel: TexelForm,
) -> Result<(), Error> {
    use crate::device::requirements::VkProperty;
    use crate::metal::format::{one_bit_storage, StorageUse};
    use crate::vk::NumericFormat as N;
    // Whether the write may reach a normalized format, and whether the bound texture decides it.
    let (reached, bound) = match image {
        ImageRef::Heap { .. } => (true, true),
        ImageRef::Slot(index) => {
            let slot = slot_of(cx, index)?.slot;
            match cx.inputs.storage_images.get(&slot) {
                None => (true, true),
                Some(c) => {
                    let formatless = c.vk_format == crate::vk::Format::UNDEFINED;
                    let normalized = formatless
                        || c.vk_format
                            .block()
                            .is_some_and(|b| matches!(b.color, Some(N::Unorm | N::Snorm)));
                    let converted = matches!(
                        c.class,
                        StorageImageClass::Identity
                            | StorageImageClass::SwizzleBgra
                            | StorageImageClass::AlphaInRed
                            | StorageImageClass::RotateArgb
                    );
                    (normalized && converted, formatless)
                }
            }
        }
    };
    if !reached {
        return Ok(());
    }
    if !cx
        .profile
        .takes_driver_deviation(DriverDeviation::MetalNormStorageWrites)
    {
        cx.precision.insert(PrecisionNote::DeviceColorConversion);
        return Ok(());
    }
    cx.families
        .texture
        .storage_properties
        .insert(VkProperty::MetalNormStorageWrites);
    let use_ = match shape.dim == ImageDim::Buffer {
        true => StorageUse::texel_buffer(false, true),
        false => StorageUse::texture(false, true),
    };
    if bound && texel.lanes == 4 && one_bit_storage(use_, cx.profile) {
        cx.precision.insert(PrecisionNote::DeviceColorConversion);
    }
    Ok(())
}

/// A fragment function's write of a texel of form `texel` to texture `image` (`extended`: an
/// extended-range texture's raw texels, never such) that Apple's `textureWriteRoundingMode` 2
/// stores otherwise than the default, under the pipeline's mode (`LoweringInputs::
/// texture_write_rounding`; on an Apple M4 under macOS 27.0.1 and the M3 Max's
/// package m3pkg-twr under macOS 26.7, alike): a `float` texel into a half-float texture (rounded
/// to nearest even at mode 2) or a packed small-float one (`RG11B10Float`: rounded to nearest even,
/// held at the largest finite value; `RGB9E5Float`: unmeasured), and a `half` texel into a packed
/// small-float one (unmeasured). Kernel, vertex, object, mesh and tile functions write alike at
/// every mode. The write is built under the default's rule alone, so: a write that may be such
/// records the mode in `StageIo::texture_write_rounding`, for link to match against the pipeline's;
/// under `NearestEven` one whose slot's storage choice declares such a format is refused
/// `NotImplemented(PrivateRenderState(TextureWriteRounding))`, and one whose bound texture
/// decides (a slot or texture buffer whose choice declares no format; a heap handle) states
/// `RuntimeFact::StorageTextureWriteRounding` or `RuntimeFact::HeapWriteRounding`, which the
/// binder refuses where such a texture is bound or reached. A write to any other texture draws
/// as Metal's at either mode.
fn mode_rounded_write(
    cx: &mut SelectCx<'_>,
    image: ImageRef,
    texel: TexelForm,
    extended: bool,
) -> Result<(), Error> {
    use crate::metal::TextureWriteRoundingMode as Mode;
    use crate::pipeline::facts::RuntimeFact;
    use crate::vk::Format as F;
    // Where the write may go: a format the slot declares, or one the bound texture decides.
    enum Target {
        Declared { half: bool, small: bool },
        Slot(SlotRef),
        Heap,
    }
    let halves = match texel.conv {
        TexelConv::F32 => true,
        TexelConv::F16 => false,
        _ => return Ok(()),
    };
    if cx.stage != crate::pipeline::sym::Stage::Fragment || extended {
        return Ok(());
    }
    let target = match image {
        ImageRef::Heap { .. } => Target::Heap,
        ImageRef::Slot(index) => {
            let slot = slot_of(cx, index)?.slot;
            match cx.inputs.storage_images.get(&slot) {
                None => Target::Slot(slot),
                Some(c) => match c.class {
                    StorageImageClass::Identity if c.vk_format == F::UNDEFINED => {
                        Target::Slot(slot)
                    }
                    StorageImageClass::Identity => Target::Declared {
                        half: matches!(
                            c.vk_format,
                            F::R16_SFLOAT | F::R16G16_SFLOAT | F::R16G16B16A16_SFLOAT
                        ),
                        small: matches!(
                            c.vk_format,
                            F::B10G11R11_UFLOAT_PACK32 | F::E5B9G9R9_UFLOAT_PACK32
                        ),
                    },
                    _ => Target::Declared {
                        half: false,
                        small: false,
                    },
                },
            }
        }
    };
    if let Target::Declared { half, small } = target {
        if !(small || (half && halves)) {
            return Ok(());
        }
    }
    let mode = cx.inputs.texture_write_rounding;
    cx.families.stage_io.interface.texture_write_rounding = Some(mode);
    if mode == Mode::TowardZero {
        return Ok(());
    }
    match target {
        Target::Declared { .. } => {
            return Err(Unsupported::not_implemented(Feature::PrivateRenderState(
                crate::error::PrivateRenderField::TextureWriteRounding,
            ))
            .into());
        }
        Target::Slot(slot) => {
            cx.facts
                .insert(RuntimeFact::StorageTextureWriteRounding { slot, halves });
        }
        Target::Heap => {
            cx.facts.insert(RuntimeFact::HeapWriteRounding {
                stage: cx.stage,
                halves,
            });
        }
    }
    Ok(())
}

/// Where a write to `image` left to the device's conversion may store a half-float texel, as the
/// clauses of [`stored_where`]: a heap element's may, a slot's whose storage choice declares a
/// half-float format does, and a slot's declared without a format does where the texture bound
/// there is of a half-float class (`BicubicTexels::HALF` without a packed field, which the small
/// floats' class carries).
fn may_store_half(cx: &SelectCx<'_>, image: ImageRef) -> Result<Vec<NoteClause>, Error> {
    use crate::metal::bicubic::BicubicTexels;
    use crate::metal::texel_format::TexelFormat;
    use crate::vk::Format as F;
    let mask = BicubicTexels::HALF | TexelFormat::PACKED_MASK;
    stored_where(
        cx,
        image,
        |f| matches!(f, F::R16_SFLOAT | F::R16G16_SFLOAT | F::R16G16B16A16_SFLOAT),
        |index| NoteTest::equal(NoteOn::Bound(index), mask, BicubicTexels::HALF),
    )
}

/// The class of texture slot `image`'s storage choice, where its variant states one, else
/// `Identity`: a write through a converting class moves or encodes the written channels
/// (`RotateArgb`'s rotation, `SrgbEncode`'s codes). A heap element's is `Identity`.
pub(crate) fn storage_class(
    cx: &SelectCx<'_>,
    image: ImageRef,
) -> Result<StorageImageClass, Error> {
    let ImageRef::Slot(index) = image else {
        return Ok(StorageImageClass::Identity);
    };
    let slot = slot_of(cx, index)?.slot;
    Ok(cx
        .inputs
        .storage_images
        .get(&slot)
        .map_or(StorageImageClass::Identity, |c| c.class))
}

/// The extended-range format of texture slot `image`'s variant (`LoweringInputs::extended_range`),
/// if it binds one: the function then reads and writes the bound texture's raw texels. A heap
/// element is never one (the binder refuses such a member).
pub(crate) fn extended_range(
    cx: &SelectCx<'_>,
    image: ImageRef,
) -> Result<Option<crate::metal::format::ExtendedRange>, Error> {
    let ImageRef::Slot(index) = image else {
        return Ok(None);
    };
    let slot = slot_of(cx, index)?.slot;
    Ok(cx.inputs.extended_range.get(&slot).copied())
}

/// Whether texture slot `index`'s variant reads a two-plane YCbCr texture's planes in place
/// (`LoweringInputs::ycbcr_in_place`, `sampler::in_place`).
pub(crate) fn ycbcr_in_place(cx: &SelectCx<'_>, index: u32) -> Result<bool, Error> {
    let slot = slot_of(cx, index)?.slot;
    Ok(cx.inputs.ycbcr_in_place.contains(&slot))
}

/// Record an access of texture slot `index` that its in-place form does not read
/// (`slot::InPlaceReads::blocked`): no texture is read in place there.
pub(crate) fn not_in_place(cx: &mut SelectCx<'_>, index: u32) {
    if let Some(s) = cx.families.texture.slots.get_mut(&index) {
        s.in_place.blocked = true;
    }
}

/// Whether the function observes any of texture call `at`'s status word.
pub(crate) fn observes_status(cx: &mut SelectCx<'_>, at: InstId) -> bool {
    status_use(cx, at).is_some()
}

/// The numeric class of the texture texture slot `image`'s variant binds where it is not the
/// class the function declares there (`LoweringInputs::cross_class`), else `None`: the slot's
/// image is declared of that class (`slot::image_class`), and its reads, samples and gathers take
/// that class's own result ([`native_form`]). A heap element never has one.
pub(crate) fn cross_class(cx: &SelectCx<'_>, image: ImageRef) -> Result<Option<TexelClass>, Error> {
    let ImageRef::Slot(index) = image else {
        return Ok(None);
    };
    let slot = slot_of(cx, index)?.slot;
    Ok(match cx.inputs.cross_class.get(&slot) {
        None => None,
        Some(SampledType::Float) => Some(TexelClass::Float),
        Some(SampledType::Sint) => Some(TexelClass::Sint),
        Some(SampledType::Uint) => Some(TexelClass::Uint),
        Some(_) => return Err(Unsupported::not_implemented(Feature::Format).into()),
    })
}

/// The form a read, sample or gather declared `declared` takes on a slot whose variant binds a
/// texture of class `bound` ([`cross_class`]): `bound`'s own 32-bit result (`float` for a
/// float class, a 32-bit integer for either integer class) of the same lanes, whose words plain
/// Metal hands back as the declared result (`emit::reinterpret`; measured on Apple M3 Max and M4).
/// A `short` or `ushort` result takes a 32-bit integer of an integer class, whose low 16 bits Metal
/// hands back, and the `half` of the float class, whose bits it hands back (on an Apple M3 Max).
/// `None` for a declared `ulong` result, whose answer is not measured, which the slot's compile
/// states no `RuntimeFact::CrossClassReads` for.
pub(crate) fn native_form(declared: TexelForm, bound: TexelClass) -> Option<TexelForm> {
    let conv = match (declared.conv, bound) {
        (TexelConv::F32 | TexelConv::F16 | TexelConv::I32, TexelClass::Float) => TexelConv::F32,
        (TexelConv::I16 { .. }, TexelClass::Float) => TexelConv::F16,
        (
            TexelConv::F32 | TexelConv::F16 | TexelConv::I32 | TexelConv::I16 { .. },
            TexelClass::Sint | TexelClass::Uint,
        ) => TexelConv::I32,
        _ => return None,
    };
    Some(TexelForm { conv, ..declared })
}

/// The form a write of declared texel `declared` takes on a storage slot whose variant binds a
/// texture of class `bound` in format `format` (the slot's storage choice declares it): `bound`'s
/// own form, into which `emit::cross_write` turns the written lanes as plain Metal stores them
/// (on an Apple M3 Max): a 32-bit integer of an integer class, a `float` of the float class from a
/// 32-bit integer, and from a `short` or `ushort` the `half` its bits hold, or the `float` holding
/// it where the format's channels are small floats (which Metal rounds to nearest even there).
/// `None` for a `ulong` texel, or an integer texel into an `RGB9E5Float` texture, whose stores are
/// not measured.
pub(crate) fn native_write_form(
    declared: TexelForm,
    bound: TexelClass,
    format: crate::vk::Format,
) -> Option<TexelForm> {
    use crate::vk::Format as F;
    let conv = match (declared.conv, bound) {
        (TexelConv::F32 | TexelConv::F16 | TexelConv::I32, TexelClass::Sint | TexelClass::Uint)
        | (TexelConv::I16 { .. }, TexelClass::Sint | TexelClass::Uint) => TexelConv::I32,
        (TexelConv::I32 | TexelConv::I16 { .. }, TexelClass::Float)
            if format == F::E5B9G9R9_UFLOAT_PACK32 =>
        {
            return None
        }
        (TexelConv::I32, TexelClass::Float) => TexelConv::F32,
        (TexelConv::I16 { .. }, TexelClass::Float) if format == F::B10G11R11_UFLOAT_PACK32 => {
            TexelConv::F32
        }
        (TexelConv::I16 { .. }, TexelClass::Float) => TexelConv::F16,
        _ => return None,
    };
    Some(TexelForm { conv, ..declared })
}

/// Mark slot `image`'s use of a result `form` whose answer from a texture of another class is not
/// measured (`slot::Uses::narrow_result`): a `ulong` one.
pub(crate) fn mark_result(cx: &mut SelectCx<'_>, image: ImageRef, form: TexelForm) {
    if matches!(form.conv, TexelConv::U64Words { .. }) {
        mark(cx, image, |u| u.narrow_result = true);
    }
}

/// Whether texture slot `image` is one of `LoweringInputs::swizzled_storage`, a storage slot
/// bound to a swizzled texture: the function applies the swizzle to the texels it reads there
/// and undoes it on those it writes; or an extended-range sampled slot bound to one, whose decoded
/// texels the function's reads, samples and gathers swizzle; or a packed normalized sampled slot
/// bound to one on a device without Metal's own sampler, whose converted texels they swizzle. A
/// heap element never is.
pub(crate) fn swizzled_storage(cx: &SelectCx<'_>, image: ImageRef) -> Result<bool, Error> {
    let ImageRef::Slot(index) = image else {
        return Ok(false);
    };
    let slot = slot_of(cx, index)?.slot;
    Ok(cx.inputs.swizzled_storage.contains(&slot))
}

/// Record that texture slot `index`'s function applies its bound texture's swizzle: the slot's
/// `TextureSwizzle` specialization (`RuntimeFact::TextureSwizzle`).
pub(crate) fn swizzles(cx: &mut SelectCx<'_>, index: u32) -> Result<(), Error> {
    let slot = slot_of(cx, index)?.slot;
    if let Some(s) = cx.families.texture.slots.get_mut(&index) {
        s.swizzle = true;
    }
    cx.facts
        .insert(crate::pipeline::facts::RuntimeFact::TextureSwizzle { slot });
    Ok(())
}

/// Record an operation of shape `shape` and texel class `texel` on `image`'s declaration.
fn observe(
    cx: &mut SelectCx<'_>,
    image: ImageRef,
    shape: ImageShape,
    texel: Option<TexelClass>,
) -> Result<(), Error> {
    let t = &mut cx.families.texture;
    match image {
        ImageRef::Slot(index) => match t.slots.get_mut(&index) {
            Some(s) => s.observe(shape, texel),
            None => Err(invariant("an unplanned texture slot")),
        },
        ImageRef::Heap { group, .. } => match t.heap.get_mut(&group) {
            Some(h) => h.observe(shape, texel),
            None => Err(invariant("an unplanned heap group")),
        },
    }
}

fn invariant(detail: &str) -> Error {
    Error::Internal(crate::error::InternalError::new(
        crate::error::InternalKind::Invariant,
        detail,
    ))
}

/// Whether `image` is a storage image, and whether it is write-only (a heap group's members
/// say only sampled or storage).
fn image_access(cx: &SelectCx<'_>, image: ImageRef) -> Result<(bool, bool), Error> {
    match image {
        ImageRef::Slot(index) => {
            let s = slot_of(cx, index)?;
            Ok((s.storage(), s.access == AbiAccess::Write))
        }
        ImageRef::Heap { group, .. } => match cx.families.texture.heap.get(&group) {
            Some(h) => Ok((h.access == ImageAccess::Storage, false)),
            None => Err(cx.invariant("an unplanned heap group")),
        },
    }
}

/// Mark a use of `image`.
fn mark(cx: &mut SelectCx<'_>, image: ImageRef, f: impl FnOnce(&mut slot::Uses)) {
    let t = &mut cx.families.texture;
    let uses = match image {
        ImageRef::Slot(index) => t.slots.get_mut(&index).map(|s| &mut s.uses),
        ImageRef::Heap { group, .. } => t.heap.get_mut(&group).map(|h| &mut h.uses),
    };
    if let Some(u) = uses {
        f(u);
    }
}

/// Select texture operation `inst` (not a sampling one) on `image`: a static texture slot, or a
/// heap element.
fn select_image(
    cx: &mut SelectCx<'_>,
    (at, inst): (InstId, &Inst),
    t: &crate::air::intrinsics::Texture,
    ops: &Operands<'_>,
    image: ImageRef,
    m: &mut Meter,
) -> Result<TextureStrategy, Error> {
    let shape = t.image;
    // The in-place form reads a slot's width, height and level count, and nothing else here.
    if let ImageRef::Slot(index) = image {
        if !matches!(
            t.op,
            TexOp::Query(Query::Width | Query::Height | Query::NumMipLevels)
        ) {
            not_in_place(cx, index);
        }
    }
    let texel = match t.texel {
        Some(x) if matches!(t.op, TexOp::Read | TexOp::Write | TexOp::Atomic(_)) => Some(
            image_texel_conv(x.ty.elem, x.sign, cx.profile).map(|(c, class)| {
                let lanes = u8::try_from(x.ty.lanes).unwrap_or(0);
                (TexelForm { conv: c, lanes }, class)
            })?,
        ),
        _ => None,
    };
    observe(cx, image, shape, texel.map(|(_, c)| c))?;
    if let Some(form) = extended_range(cx, image)? {
        // Metal binds an extended-range texture only to a `float` or `half` parameter; a read or
        // write of a whole texel of a texture (not a texel buffer) is built.
        let whole = texel
            .is_none_or(|(f, _)| matches!(f.conv, TexelConv::F32 | TexelConv::F16) && f.lanes == 4);
        if !whole || shape.dim == ImageDim::Buffer || matches!(t.op, TexOp::Atomic(_)) {
            return Err(Unsupported::not_implemented(Feature::Format).into());
        }
        if matches!(t.op, TexOp::Read | TexOp::Write) {
            cx.families.arith.converts(form);
        }
    }
    match t.op {
        TexOp::IsNull => {
            let index = match image {
                ImageRef::Slot(index) => index,
                ImageRef::Heap { handle, .. } => {
                    return Ok(TextureStrategy::HeapIsNull { handle });
                }
            };
            if let Some(s) = cx.families.texture.slots.get_mut(&index) {
                s.presence = true;
            }
            cx.facts
                .insert(crate::pipeline::facts::RuntimeFact::ResourcePresence {
                    slot: SlotRef {
                        stage: cx.stage,
                        class: SlotClass::Texture,
                        index,
                    },
                });
            Ok(TextureStrategy::IsNull { index })
        }
        TexOp::Query(q) => select_query(cx, ops, image, shape, q),
        TexOp::Read => {
            let (form, _) = texel.ok_or_else(refuse)?;
            let class = texel.map(|(_, c)| c);
            select_read(cx, (at, inst), ops, (image, shape, form), class, m)
        }
        TexOp::Write => {
            let (form, _) = texel.ok_or_else(refuse)?;
            select_write(cx, ops, (image, shape, form), m)
        }
        TexOp::Atomic(op) => {
            let texel = texel.ok_or_else(refuse)?;
            atomic::select(cx, (at, inst), op, ops, (image, shape), texel)
        }
        _ => Err(refuse()),
    }
}

fn coord(cx: &SelectCx<'_>, ops: &Operands<'_>, shape: ImageShape) -> Result<Coord, Error> {
    let plane = ops
        .get(TexRole::Coord)
        .ok_or_else(|| cx.invariant("a texture access without a coordinate"))?;
    let layer = match ops.get(TexRole::Layer) {
        Some(v) => Some(id(cx, v)?),
        None => None,
    };
    let face = match ops.get(TexRole::Face) {
        Some(v) => Some(id(cx, v)?),
        None => None,
    };
    Ok(Coord {
        plane: id(cx, plane)?,
        lanes: plane_lanes(shape.dim),
        layer,
        face,
    })
}

/// A level operand that must be the constant 0: a multisampled image and a texture buffer have
/// one level, and a heap storage element is a view of its texture's base level.
fn level_zero(cx: &SelectCx<'_>, ops: &Operands<'_>) -> Result<(), Error> {
    match ops.get(TexRole::Lod) {
        None => Ok(()),
        Some(v) if const_int(cx, v) == Some(0) => Ok(()),
        Some(_) => Err(refuse()),
    }
}

/// The level operand of a storage access on `image`: `None` for the constant 0 (or none), which
/// every view's base level is; any other level indexes the slot's level views, which it marks. A
/// heap storage element takes only the constant 0.
fn storage_level(
    cx: &mut SelectCx<'_>,
    ops: &Operands<'_>,
    image: ImageRef,
) -> Result<Option<ValueId>, Error> {
    let Some(v) = ops.get(TexRole::Lod) else {
        return Ok(None);
    };
    let constant = const_int(cx, v);
    if constant == Some(0) {
        return Ok(None);
    }
    if image.heap() {
        return Err(refuse());
    }
    let level = id(cx, v)?;
    mark(cx, image, |u| {
        u.level_views = true;
        match constant {
            Some(k) => u.top_level = u.top_level.max(k),
            None => u.dynamic_level = true,
        }
    });
    Ok(Some(level))
}

fn select_read(
    cx: &mut SelectCx<'_>,
    (at, inst): (InstId, &Inst),
    ops: &Operands<'_>,
    (image, shape, texel): (ImageRef, ImageShape, TexelForm),
    class: Option<TexelClass>,
    m: &mut Meter,
) -> Result<TextureStrategy, Error> {
    let mut r = select_read_as(cx, (at, inst), ops, (image, shape, texel), None, m)?;
    // A sampled color heap group's read of one sample reads a member of another numeric class in
    // that class, its words handed back as declared, as Metal reads it (the probe's argument
    // buffer half, the slot's rule): one sibling read per other class.
    let (storage, _) = image_access(cx, image)?;
    let crosses = matches!(image, ImageRef::Heap { .. })
        && !storage
        && !shape.ms
        && !shape.depth
        && shape.dim != ImageDim::Buffer
        && !r.residency
        && matches!(texel.conv, TexelConv::F32 | TexelConv::F16 | TexelConv::I32);
    if let (false, ImageRef::Heap { group, .. }) = (crosses, image) {
        if let Some(h) = cx.families.texture.heap.get_mut(&group) {
            h.uncrossed = true;
        }
    }
    if let (true, ImageRef::Heap { group, .. }, Some(own)) = (crosses, image, class) {
        for k in [TexelClass::Float, TexelClass::Sint, TexelClass::Uint] {
            if k == own {
                continue;
            }
            let sibling = select_read_as(cx, (at, inst), ops, (image, shape, texel), Some(k), m)?;
            r.heap_cross.push(sibling);
            if let Some(h) = cx.families.texture.heap.get_mut(&group) {
                h.cross_reads.insert(k);
            }
        }
        cx.families.access.heap_class_index = true;
    }
    Ok(TextureStrategy::Read(r))
}

/// The read of `image` (`select_read`), or, for `as_class`, its sibling reading a heap member in
/// that other numeric class (`Read::heap_class`).
fn select_read_as(
    cx: &mut SelectCx<'_>,
    (at, inst): (InstId, &Inst),
    ops: &Operands<'_>,
    (image, shape, texel): (ImageRef, ImageShape, TexelForm),
    as_class: Option<TexelClass>,
    m: &mut Meter,
) -> Result<Read, Error> {
    // The depth word's only observed value is 1; any other is not modelled.
    if let Some(w) = ops.get(TexRole::DepthWord) {
        if const_int(cx, w) != Some(1) {
            return Err(refuse());
        }
    }
    // A texture of another class, read in its own class, its words handed back as declared.
    mark_result(cx, image, texel);
    let bound = match as_class {
        Some(k) => Some(k),
        None => cross_class(cx, image)?,
    };
    let (texel, declared) = match bound {
        Some(bound) => match native_form(texel, bound) {
            Some(native) => (native, Some(texel.conv)),
            None => return Err(Unsupported::not_implemented(Feature::Format).into()),
        },
        None => (texel, None),
    };
    let edge = read_edge(cx, ops, m)?;
    // Metal 4.1's `sparse_read(sampler s, ...)` is an edge read whose status word the function
    // observes. What clamp-to-edge addressing does to its residency the specification does not
    // say, and no OS at hand runs Metal 4.1 to measure it, so it is refused.
    if edge.is_some() && status_use(cx, at).is_some() {
        return Err(refuse());
    }
    let residency = residency(cx, at, image)?;
    let (storage, write_only) = image_access(cx, image)?;
    if write_only {
        return Err(refuse());
    }
    let level = if shape.ms || shape.dim == ImageDim::Buffer {
        level_zero(cx, ops)?;
        None
    } else if storage {
        storage_level(cx, ops, image)?
    } else {
        match ops.get(TexRole::Lod) {
            Some(v) => Some(id(cx, v)?),
            None => return Err(cx.invariant("a sampled read without a level")),
        }
    };
    let sample = match ops.get(TexRole::SampleIndex) {
        Some(v) => Some(id(cx, v)?),
        None => None,
    };
    let at_ = coord(cx, ops, shape)?;
    let result = cx.lowered_repr(Value::Inst(at), inst.ty)?;
    // An extended-range texel is decoded from its raw code, so the device's out-of-bounds texel
    // (raw zero) is not Metal's (zero, measured): the shader selects it.
    let extended = extended_range(cx, image)?;
    if extended.is_some() && residency {
        return Err(Unsupported::not_implemented(Feature::Format).into());
    }
    let check = match slot::read_check(cx, shape.dim == ImageDim::Buffer) {
        ReadCheck::Device if extended.is_some() => ReadCheck::Shader,
        c => c,
    };
    // A storage read at a level reads the slot's level views, and past the bound level count
    // selects Metal's out-of-bounds texel on every profile.
    let level_views = storage && level.is_some();
    // Metal's out-of-bounds alpha is the bound format's (1 where it has none): a read the shader
    // checks, or tests against its level views, whose alpha the function observes takes it from the
    // slot's specialization, or from a heap element's handle-table entry. So does one whose
    // residency it observes, under either of its slot's arms: a texel that is not resident reads
    // Metal's unmapped value, zero memory as the bound format reads it, which is the out-of-bounds
    // texel, where a device's block that holds no mapped tile reads the device's value. A texture
    // buffer's swizzle applies to its reads: Metal returns each channel from the source the bound
    // texture buffer's swizzle names, out of bounds too. A `ulong` read's swizzle is not measured;
    // the binder refuses a swizzled texture buffer there.
    let swizzle = matches!(image, ImageRef::Slot(_))
        && shape.dim == ImageDim::Buffer
        && !matches!(texel.conv, TexelConv::U64Words { .. });
    // A storage slot bound to a swizzled texture applies the swizzle under its variant, to the
    // converted texel, out of bounds too, as plain Metal reads it. A `ulong` texel's swizzle, and a
    // read of fewer lanes, are not measured. So does an extended-range sampled slot of the variant,
    // to the decoded texel. So does a packed normalized sampled slot of the
    // variant, to the texel its class converts: the variant holds no other sampled slot.
    let storage_swizzle = shape.dim != ImageDim::Buffer && swizzled_storage(cx, image)?;
    if storage_swizzle && (texel.lanes != 4 || matches!(texel.conv, TexelConv::U64Words { .. })) {
        return Err(Unsupported::not_implemented(Feature::TextureSwizzle).into());
    }
    // A `ulong` read's lanes past the first are Metal's constants `(0, 0, 1)` in and out of
    // bounds, never the image's. A swizzled read may return the alpha in any lane, so it observes
    // the alpha wherever the function reads a lane.
    let observed = (check == ReadCheck::Shader || level_views || residency)
        && texel.lanes == 4
        && !matches!(texel.conv, TexelConv::U64Words { .. })
        && (swizzle || storage_swizzle || alpha_read(cx, at));
    let alpha = match image {
        ImageRef::Slot(index) if observed => {
            let slot = slot_of(cx, index)?.slot;
            if let Some(s) = cx.families.texture.slots.get_mut(&index) {
                s.oob_alpha = true;
            }
            cx.facts
                .insert(crate::pipeline::facts::RuntimeFact::OutOfBoundsAlpha { slot });
            true
        }
        // A heap element's format is the resident texture's: its handle-table entry's flags say
        // whether it has alpha, which the search then carries in the handle.
        ImageRef::Heap { .. } if observed => {
            cx.families.access.heap_alpha = true;
            true
        }
        _ => false,
    };
    // An 8-bit sRGB texel's codes, under the slot's twin variant, decoded by the device's
    // transfer; another `half` texel converts by the bound format's class, and its NaN is the
    // device's decode.
    let raw =
        matches!(texel.conv, TexelConv::F16 | TexelConv::F32) && srgb_texels(cx, image, shape, m)?;
    // A storage slot's 8-bit sRGB codes, through its UNORM alias, decoded by the device's
    // transfer, as a UNORM twin's are.
    let srgb_codes = storage
        && matches!(texel.conv, TexelConv::F16 | TexelConv::F32)
        && matches!(
            storage_class(cx, image)?,
            StorageImageClass::SrgbEncode | StorageImageClass::SrgbEncodeBgra
        );
    if srgb_codes {
        cx.precision.insert(PrecisionNote::SrgbTexelDecode);
    }
    // An `RGB9E5Float` slot's word through its `R32_UINT` alias, decoded exactly (the slot's
    // planning declares it).
    let words = storage
        && matches!(texel.conv, TexelConv::F16 | TexelConv::F32)
        && storage_class(cx, image)? == StorageImageClass::SharedExponentBits;
    // The shader decodes these texels itself, recording its own notes. An extended-range texel is
    // decoded exactly to `f32`, which a `half` result rounds to nearest even, as Metal answers. A
    // read is one texel.
    let decoded = raw || srgb_codes || words;
    if texel.conv == TexelConv::F16 && !decoded && extended.is_none() {
        half_texels(cx, image, shape, false, device_texels(cx), m)?;
        half_texel_nans(cx, image, shape, m)?;
    }
    if texel.conv == TexelConv::F32 && !decoded && extended.is_none() {
        float_texel_notes(cx, image, shape, OneTexel::Always);
    }
    // A float read of a sampled heap group of one sample decodes a member of a form a heap reads
    // where its handle names one, from the group's unsigned integer class.
    let heap_raw = match image {
        ImageRef::Heap { group, .. }
            if !storage
                && as_class.is_none()
                && !shape.ms
                && shape.dim != ImageDim::Buffer
                && texel.lanes == 4
                && matches!(texel.conv, TexelConv::F32 | TexelConv::F16) =>
        {
            if let Some(h) = cx.families.texture.heap.get_mut(&group) {
                h.reads_raw = true;
            }
            cx.families
                .texture
                .heap_raw
                .insert(texel.conv == TexelConv::F16);
            cx.families.access.heap_class_index = true;
            true
        }
        _ => false,
    };
    if let (true, ImageRef::Slot(index)) = (swizzle || storage_swizzle, image) {
        swizzles(cx, index)?;
    }
    let heap_alpha_red = storage && heap_alpha_in_red(cx, image, shape, texel.conv);
    // Where the profile's sparse textures are sparse images, a block of one that holds no mapped
    // tile reads the device's value (RADV's alpha 0), where Metal reads a channel the format
    // lacks as its constant: a color read of four 32-bit held lanes sets them after it, whether
    // the function reads its residency or not.
    let sparse_images = cx
        .profile
        .fields()
        .enabled
        .features
        .contains(&crate::device::requirements::VkFeature::ShaderResourceResidency);
    let constant_lanes = match image {
        ImageRef::Slot(index)
            if sparse_images
                && shape.dim != ImageDim::Buffer
                && texel.lanes == 4
                && !matches!(texel.conv, TexelConv::U64Words { .. }) =>
        {
            if let Some(s) = cx.families.texture.slots.get_mut(&index) {
                s.constant_lanes = true;
                let slot = s.slot;
                cx.facts
                    .insert(crate::pipeline::facts::RuntimeFact::ConstantLanes { slot });
            }
            true
        }
        _ => false,
    };
    mark(cx, image, |u| u.read = true);
    Ok(Read {
        image,
        at: at_,
        level,
        level_views,
        sample,
        texel,
        result,
        residency,
        check,
        alpha,
        edge,
        raw,
        srgb_codes,
        swizzle,
        storage_swizzle,
        extended,
        heap_raw,
        heap_alpha_red,
        constant_lanes,
        declared,
        heap_class: as_class,
        heap_cross: Vec::new(),
    })
}

/// The sampler state whose read the MSL 4.1 specification states (§6.13): `constexpr
/// sampler(address::clamp_to_edge)`'s, as Apple's `metal -std=metal4.1` encodes it (clamp-to-edge
/// on every axis, and the default for every other field).
const EDGE_SAMPLER_WORD: u64 = 0x007B_FF00_0008_0049;

/// The addressing of read `ops`. Metal's sampler-less `read` passes the front end's read sampler
/// (`air.get_read_sampler`), which addresses nothing, and a zero offset. Metal 4.1's `read(sampler
/// s, ...)` passes `s` and its offset: "Clamp-to-edge applies only when the texture coordinate
/// (after adding the offset) falls out of bounds. It doesn't apply to other parameters such as
/// face, array, or lod" (MSL 4.1 §6.13). That is exact for a constexpr sampler whose state is
/// [`EDGE_SAMPLER_WORD`]'s, an [`Edge`]. What another address mode or another field does to a read
/// the specification does not say, and no OS at hand runs Metal 4.1 to measure it; an argument
/// sampler's state is the binder's; so either is refused. The specification's overloads take `int`
/// coordinates and offsets; a narrower lane, which lowering holds zero-extended, has no overload
/// whose meaning it could take, so it is refused too.
fn read_edge(cx: &SelectCx<'_>, ops: &Operands<'_>, m: &mut Meter) -> Result<Option<Edge>, Error> {
    let offset = ops.get(TexRole::Offset).filter(|o| !is_zero(cx, *o));
    let edge = match ops.get(TexRole::Sampler) {
        None => false,
        Some(s) if read_sampler(cx, s) => false,
        Some(s) => {
            let Some(want) = crate::air::abi::SamplerState::from_words([EDGE_SAMPLER_WORD, 0])
            else {
                return Err(cx.invariant("the clamp-to-edge sampler state does not decode"));
            };
            let state = match crate::lower::access::handle::sampler_target(cx, s, m)? {
                HandleTarget::One(Candidate::Constexpr { global, .. }) => cx
                    .entry
                    .static_samplers
                    .iter()
                    .find(|(g, _)| *g == global)
                    .map(|(_, state)| *state),
                _ => None,
            };
            if state != Some(want) {
                return Err(refuse());
            }
            let int = |v: Value| {
                cx.value_ty(v)
                    .and_then(|t| cx.entry.module.types().int_bits(t))
                    == Some(32)
            };
            let coordinate = ops.get(TexRole::Coord).is_some_and(int);
            if !coordinate || ops.get(TexRole::Offset).is_some_and(|o| !int(o)) {
                return Err(refuse());
            }
            true
        }
    };
    match (edge, offset) {
        (true, o) => Ok(Some(Edge {
            offset: o.map(|o| id(cx, o)).transpose()?,
        })),
        (false, None) => Ok(None),
        // A sampler-less read has no offset.
        (false, Some(_)) => Err(refuse()),
    }
}

/// Whether `v` is the front end's read sampler (`air.get_read_sampler`).
fn read_sampler(cx: &SelectCx<'_>, v: Value) -> bool {
    let Value::Inst(i) = v else {
        return false;
    };
    cx.body.insts.get(i).is_some_and(|inst| {
        matches!(
            &inst.kind,
            InstKind::Call(c) if matches!(
                c.callee,
                Callee::Air { op: AirIntrinsic::Sampler(SamplerOp::GetReadSampler), .. }
            )
        )
    })
}

/// Whether the alpha of four-lane read `at` is observed ([`alpha::observed`]).
fn alpha_read(cx: &mut SelectCx<'_>, at: InstId) -> bool {
    if cx.families.texture.alpha_read.is_none() {
        let observed = alpha::observed(cx, cx.body);
        cx.families.texture.alpha_read = Some(observed);
    }
    cx.families
        .texture
        .alpha_read
        .as_ref()
        .is_some_and(|s| s.contains(&at))
}

fn slot_of<'a>(cx: &'a SelectCx<'_>, index: u32) -> Result<&'a SlotPlan, Error> {
    cx.families
        .texture
        .slots
        .get(&index)
        .ok_or_else(|| cx.invariant("an unplanned texture slot"))
}

fn select_write(
    cx: &mut SelectCx<'_>,
    ops: &Operands<'_>,
    (image, shape, texel): (ImageRef, ImageShape, TexelForm),
    m: &mut Meter,
) -> Result<TextureStrategy, Error> {
    if !image_access(cx, image)?.0 {
        return Err(refuse());
    }
    let level = if shape.ms || shape.dim == ImageDim::Buffer {
        level_zero(cx, ops)?;
        None
    } else {
        storage_level(cx, ops, image)?
    };
    slot::stage_stores(cx)?;
    let value = ops
        .get(TexRole::Value)
        .ok_or_else(|| cx.invariant("a texture write without a value"))?;
    if matches!(value, Value::Const(_)) && cx.constant(value).is_none() {
        return Err(Unsupported::not_implemented(Feature::Lowering(LowerFamily::Arith)).into());
    }
    let at = coord(cx, ops, shape)?;
    // A texture of another class: the declared lanes turned into its own class's texel, as plain
    // Metal stores them, then written as that class writes.
    let (texel, cross) = match cross_class(cx, image)? {
        Some(bound) => {
            let format = match image {
                ImageRef::Slot(index) => {
                    let slot = slot_of(cx, index)?.slot;
                    // The bound image's own format: an alias view's image (an `RGB9E5Float`
                    // texture written through its `R32_UINT` view), else the declared one.
                    cx.inputs
                        .storage_images
                        .get(&slot)
                        .map(|c| c.alias.map_or(c.vk_format, |a| a.image_format))
                        .filter(|f| *f != crate::vk::Format::UNDEFINED)
                }
                ImageRef::Heap { .. } => None,
            };
            let Some(format) = format else {
                return Err(cx.invariant("a cross-class write without its declared format"));
            };
            match native_write_form(texel, bound, format) {
                Some(native) => (
                    native,
                    Some(CrossWrite {
                        declared: texel.conv,
                        format,
                    }),
                ),
                None => return Err(Unsupported::not_implemented(Feature::Format).into()),
            }
        }
        None => (texel, None),
    };
    let guard = slot::write_guard(cx, shape.dim == ImageDim::Buffer);
    let extended = extended_range(cx, image)?;
    let class = storage_class(cx, image)?;
    mode_rounded_write(cx, image, texel, extended.is_some())?;
    // Its alpha in red where the class is `A8Unorm`'s and the device holds such a texture so.
    let heap_alpha_red = heap_alpha_in_red(cx, image, shape, texel.conv);
    if texel.conv == TexelConv::F32 {
        half_storage_facts(cx, image, (class, cross.is_some()), m)?;
    }
    if texel.conv == TexelConv::F16 && extended.is_none() {
        half_write_nans(cx, image)?;
    }
    // A `half` texel's too, held widened: Metal stores its value by the same rule.
    if matches!(texel.conv, TexelConv::F32 | TexelConv::F16) && extended.is_none() {
        norm_storage_facts(cx, image, shape, texel)?;
    }
    // A storage slot bound to a swizzled texture stores each channel from the lane the swizzle
    // takes it to, under its variant, as plain Metal writes it. A `ulong` texel's is not measured.
    let swizzle = shape.dim != ImageDim::Buffer && swizzled_storage(cx, image)?;
    if swizzle && (texel.lanes != 4 || matches!(texel.conv, TexelConv::U64Words { .. })) {
        return Err(Unsupported::not_implemented(Feature::TextureSwizzle).into());
    }
    if let (true, ImageRef::Slot(index)) = (swizzle, image) {
        swizzles(cx, index)?;
    }
    let saturate = match extended {
        Some(_) => IntRange::Full,
        None => int_range(cx, image, texel)?,
    };
    mark(cx, image, |u| u.write = true);
    Ok(TextureStrategy::Write(Write {
        image,
        at,
        level,
        value: id(cx, value)?,
        texel,
        guard,
        extended,
        class,
        heap_alpha_red,
        swizzle,
        cross,
        saturate,
    }))
}

/// How a write of `texel` (the image's own class's form) to `image` saturates its lanes
/// ([`IntRange`]): an integer texel's to the widths of the format the slot's storage choice
/// declares, or, where it declares none, to those the binder pushes for the bound texture. A
/// 16-bit lane, held extended to the word by the image's signedness, needs no saturation into a
/// channel of 16 bits or more. A heap element's write is not saturated: its handle names no
/// integer format's widths (the classes), so the device's conversion stores its out-of-range lanes.
fn int_range(cx: &mut SelectCx<'_>, image: ImageRef, texel: TexelForm) -> Result<IntRange, Error> {
    let narrowest = match texel.conv {
        TexelConv::I32 => 32,
        TexelConv::I16 { .. } => 16,
        _ => return Ok(IntRange::Full),
    };
    let ImageRef::Slot(index) = image else {
        return Ok(IntRange::Full);
    };
    let slot = slot_of(cx, index)?.slot;
    let declared = cx
        .inputs
        .storage_images
        .get(&slot)
        .map(|c| c.alias.map_or(c.vk_format, |a| a.image_format))
        .filter(|f| *f != crate::vk::Format::UNDEFINED);
    let Some(format) = declared else {
        mark(cx, image, |u| u.int_channels = true);
        return Ok(IntRange::SlotWord(slot));
    };
    let lanes = usize::from(texel.lanes).min(4);
    Ok(match crate::metal::format::integer_channel_bits(format) {
        Some(bits) if bits.iter().take(lanes).any(|b| *b < narrowest) => IntRange::Widths(bits),
        _ => IntRange::Full,
    })
}

fn select_query(
    cx: &mut SelectCx<'_>,
    ops: &Operands<'_>,
    image: ImageRef,
    shape: ImageShape,
    q: Query,
) -> Result<TextureStrategy, Error> {
    let storage = image_access(cx, image)?.0;
    let kind = match q {
        Query::Width => QueryKind::Size { component: 0 },
        Query::Height => QueryKind::Size { component: 1 },
        Query::Depth => QueryKind::Size { component: 2 },
        Query::ArraySize => QueryKind::ArraySize,
        // Metal has no mipmapped 1D textures: `get_num_mip_levels` of a 1D texture or 1D texture
        // array always returns 0 (MSL 4.1, their member functions), sampled or not.
        Query::NumMipLevels if shape.dim == ImageDim::D1 => QueryKind::NoLevels,
        // A storage image's level count is not queryable (VUID-StandaloneSpirv-
        // OpImageQuerySizeLod-04659): the binder pushes the bound texture's. A heap element has no
        // slot to push it for.
        Query::NumMipLevels if storage && image.heap() => return Err(refuse()),
        Query::NumMipLevels if storage => {
            mark(cx, image, |u| u.levels = true);
            QueryKind::Levels
        }
        Query::NumMipLevels => QueryKind::Levels,
        Query::NumSamples => QueryKind::Samples,
    };
    // A size query at a level that is not the constant 0 answers from the image's first level
    // (`emit::level_extent`), as Metal does at every level: it queries no level the image may
    // lack, and a storage slot no level view. A heap storage element takes only the constant 0.
    let level = match kind {
        QueryKind::Size { .. } if !shape.ms && shape.dim != ImageDim::Buffer => {
            match ops.get(TexRole::Lod) {
                Some(v) if const_int(cx, v) == Some(0) => None,
                Some(_) if storage && image.heap() => return Err(refuse()),
                Some(v) => Some(id(cx, v)?),
                None if storage => None,
                None => return Err(cx.invariant("a size query without a level")),
            }
        }
        QueryKind::Size { .. } => {
            level_zero(cx, ops)?;
            None
        }
        _ => None,
    };
    // A sampled heap group's query of one sample reads a member of a form a heap reads through
    // the group's unsigned integer class, where its handle names one.
    let heap_raw = match image {
        ImageRef::Heap { group, .. } if !storage && !shape.ms && shape.dim != ImageDim::Buffer => {
            if let Some(h) = cx.families.texture.heap.get_mut(&group) {
                h.reads_raw = true;
            }
            cx.families.access.heap_class_index = true;
            true
        }
        _ => false,
    };
    // And a member of another numeric class, through the group's array of that class.
    let heap_cross = match image {
        ImageRef::Heap { group, .. } if heap_raw && !shape.depth => {
            if let Some(h) = cx.families.texture.heap.get_mut(&group) {
                h.cross_queries = true;
            }
            true
        }
        ImageRef::Heap { group, .. } => {
            if let Some(h) = cx.families.texture.heap.get_mut(&group) {
                h.uncrossed = true;
            }
            false
        }
        _ => false,
    };
    let volume = shape.dim == ImageDim::D3
        && !shape.array
        && matches!(
            extended_range(cx, image)?,
            Some(crate::metal::format::ExtendedRange::Planes { .. })
        );
    mark(cx, image, |u| u.query = true);
    Ok(TextureStrategy::Query(QueryOp {
        image,
        kind,
        level,
        heap_raw,
        heap_cross,
        volume,
    }))
}
