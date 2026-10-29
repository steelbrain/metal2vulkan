//! Tile functions: a kernel compiled for a tile render pipeline
//! (`LoweringInputs::tile_attachments`, `Device::compile_tile_pipeline`).
//!
//! **What Metal gives** (measured on Apple silicon). `dispatchThreadsPerTile:` runs one threadgroup
//! of the given size per tile of the render pass's render area, the tile's index its
//! `threadgroup_position_in_grid`, so the dispatch's builtins are a compute dispatch's over a grid
//! of `ceil(area / tile)` threadgroups; `get_imageblock_width`/`_height` are the tile's size, and
//! `[[render_target_array_index]]` is 0 in a pass that is not layered. The implicit imageblock is
//! the tile's pixels of every color attachment: cell `(x, y)` of tile `t` is pixel `t · tile + (x,
//! y)`, loaded before the threadgroup runs and stored once it ends; a cell past the render area
//! holds no pixel (undefined until the function writes it, its writes kept until the threadgroup
//! ends) and is never stored. `air.load.implicit_imageblock.<T>(color, coord, mode, sample)` reads
//! attachment `color`'s texel of cell `coord` as `T`, and `air.store.implicit_imageblock.<T>(v,
//! color, coord, mode, sample)` writes it. An explicit layout aliased onto the implicit imageblock
//! (`air.alias_implicit_imageblock`) is the same cells: its member `i` holds attachment `i`'s texel
//! bits at the member's offset, reached through `air.imageblock_data(coord, 0, 0)` as a kernel's
//! explicit imageblock is. The threads of a tile's threadgroup are laid out as a fragment pass's:
//! `thread_index_in_threadgroup` runs over 2x2 quads, and a simdgroup is a `b x 32/b` block of
//! positions, `b = min(width, 8)`, in columns of the threadgroup at most 16 wide
//! (`lower::dispatch`, measured for the threadgroups the binder alone runs, up to 32x32).
//!
//! **Tile staging.** The tile's cells are the threadgroup arena's imageblock region: its extent is
//! the tile size (`SpecKey::ImageblockWidth`/`Height`, which the binder specializes from
//! `BindState::imageblock_size`), and a cell holds each attachment's texel at its layout's offset
//! (`memory::plan`: the aliased layout's, or the implicit layout's). The entry is a new function
//! ([`wrap_entry`]) that loads the tile's pixels of every attachment the function reaches into the
//! cells (a cell past the render area, `SpecKey::RenderAreaWidth`/`Height`, is zeroed: Metal leaves
//! it undefined), runs a workgroup barrier, calls the function, runs a second barrier, and stores
//! the cells of every attachment the function may write back to the pixels inside the render area.
//! The threads share the cells round robin by thread index, each loading and storing the same
//! pixels, so no pixel is accessed by two invocations; tiles are disjoint. An implicit access is
//! the arena leaf of the attachment's texel at its cell's offset, contained as every threadgroup
//! arena access. The attachments are storage images (`ResourceKind::TileAttachment`), read and
//! written in their staging format ([`staging_format`], below).
//!
//! **Exact formats.** An attachment is staged only in an unnormalized 16- or 32-bit float or
//! integer format of 1, 2, or 4 channels, whose texel a cell holds unchanged: the staging moves
//! the texel's bits, never its value. An integer attachment's image is its own format, whose read
//! widens each channel exactly (a 16-bit one masked) and whose write narrows the same value back.
//! A float attachment's image is the unsigned-integer format of the same channels and widths
//! ([`staging_format`]: `R16Float` through `R16Uint`), a view of the same compatibility class
//! (`WriteSource::TileAttachment`'s format), so its read and write are those of an integer
//! attachment. A float view would not be exact. Vulkan decodes a float texel to its value
//! ("Texel Decode"), and a conversion keeps only that a NaN is a NaN, not its payload
//! ("Floating-Point Format Conversions"). A shader may flush a 32-bit denormal input
//! ("Precision and Operation of SPIR-V Instructions": any denormalized value input into a shader
//! may be flushed to zero), and `OpQuantizeToF16` flushes every half denormal.
//!
//! **Normalized formats**. Metal converts a normalized attachment's texels between the pixel and an
//! implicit access as its pack functions convert them (Metal probes, on an M3 Max and an M4, plain
//! and under the API validation layer: every code's read, every boundary between two codes a store
//! makes, NaN and the other specials, the channel order, and a read after the function's own
//! write): `RGBA8Unorm` and `BGRA8Unorm` as `unorm4x8`, their sRGB forms as `unorm4x8_srgb`
//! (Metal's twelve-bit tables), `RGBA8Snorm` as `snorm4x8`, `RGB10A2Unorm` and `BGR10A2Unorm` as
//! `unorm.rgb10a2`, the 16-bit formats of two and four channels as `unorm2x16` and `snorm2x16` a
//! word at a time, and `RG8Unorm`, `RG8Snorm`, `R16Unorm` and `R16Snorm` as the low lanes of those
//! of their channels' width; the BGR formats hold blue in a word's lowest field ([`normalized`]).
//! The cell holds the texel's bits, which the staging moves through the unsigned-integer view of
//! its bytes in 32-bit words (`R32Uint`, `RG32Uint`, or `R16Uint` for a texel of two bytes), so a
//! read after the function's own write gives the value of the code it stored, as Metal's does. An
//! implicit access converts them ([`Norm`]) by the device's conversion (`arith::pack`, the
//! instruction or its definition on the format's fields; `PrecisionNote::DevicePackConversion`):
//! a load unpacks each word (`pack::unpack`), a store packs (`pack::pack`). The access's member is
//! `float` lanes of the format's channels, or four, whose lanes past the channels a load reads as
//! green and blue 0 and alpha 1 and a store drops (probes 9d and 9e), or `half` lanes of the same:
//! a load rounds each channel's unpacked `float` to `half` by the device's rounding (Metal rounds
//! the exact value `c / D` once, probe 12), and a store converts a `half` as the `float` of the
//! same value ([`NormAccess`]). Every other format
//! converts between the pixel and the cell in a way not read yet:
//! `Unsupported{TileShading, NotImplemented}`.
//!
//! **Tile threadgroup memory**. A tile function's `[[threadgroup(n)]]` arguments are its tile's
//! threadgroup memory, which Metal keeps for the tile across the render pass's tile dispatches (MSL
//! §4.4; measured on an M4 and an M3 Max: a dispatch reads what an earlier one over the same tile
//! left, the memory starts at zero with each pass, and arguments the encoder places at overlapping
//! offsets alias). The threadgroup arena begins with that memory (`memory::arena`, `TgShape::tile`:
//! `SpecKey::TileThreadgroupLength` bytes, each argument at its `SpecKey::TgOffset`), and the pass
//! keeps every tile's in one storage buffer (`SyntheticBinding::TileMemory`, tile `t` at `t ·
//! TileThreadgroupLength`, `t = x + y · SpecKey::TileGridWidth`). The entry copies its tile's words
//! from the buffer into the arena before the function runs and back after it ([`wrap_entry`], the
//! threads sharing the words round robin, between the same barriers as the cells), so every access
//! in the function is a plain threadgroup one. Tiles are disjoint, one threadgroup each, so no word
//! is copied by two threadgroups. A tile function that declares threadgroup variables is
//! `InvalidDescriptor(TileThreadgroupVariable)`, as Metal refuses its pipeline. A fragment
//! function's arguments are the same memory, which it reaches in the buffer itself
//! (`access::tile_memory`).
//!
//! **An explicit imageblock that stands alone**: one no implicit imageblock stands beside and that
//! is not aliased onto one. Metal keeps it in the tile memory a fragment function's
//! `[[imageblock_data]]` is in, each pixel's same bytes, for the whole pass (measured on an M4,
//! `runtime::fragment_imageblock`). Its cells are the pass's imageblock storage
//! (`SyntheticBinding::Imageblock`, a fragment imageblock's `R32Uint` 2D-array storage image,
//! layer `w` word `w` of pixel `(x, y)`'s cell at texel `(x, y)`, which covers every pixel of
//! every tile): the entry stages each word of every cell of its tile from the storage into the
//! threadgroup cells before the function runs, and back after it where the function may write
//! them, as it stages attachments, a cell past the render area too. The function records
//! `RuntimeFact::TileImageblock`, its size and alignment, which the consumer matches against the
//! pass's storage and Metal's `imageblockSampleLength` reads. A cell of part of a word is not
//! implemented.
//!
//! **Refused.** A kernel that reaches the implicit imageblock without tile inputs is
//! `NeedsInput(TileAttachments)`. Not implemented (`Unsupported{TileShading, NotImplemented}`):
//! an explicit imageblock beside the implicit one that is not aliased onto it, an explicit one
//! alone whose cell is not a whole number of words, two implicit imageblocks, layouts that place
//! an attachment at different offsets or sizes, a member that is an array or a struct, a member
//! whose size is not its attachment's texel's (a normalized attachment's: four or two bytes a
//! lane of its channels or of four), an implicit access of another type than its attachment's
//! texel (a normalized attachment's: `float` or `half` lanes of its channels or four that fill its
//! member), of an attachment no layout holds, or
//! whose attachment index, data rate, or sample is not a constant of a single-sample pass (mode 0
//! or 1, sample 0), and a normalized attachment an aliased layout holds, a cell pointer, or a
//! slice write reaches. An attachment a layout holds without a pixel format in the pipeline
//! descriptor is `InvalidDescriptor(ColorAttachment)`.

use std::collections::btree_map::Entry as MapEntry;
use std::collections::{BTreeMap, BTreeSet};

use crate::air::abi::{ImageblockArg, ImageblockKind, ParamBinding, ParamRole};
use crate::air::intrinsics::{Imageblock, PackDir, PackFormat};
use crate::air::ir::{Call, Const, Value};
use crate::device::profile::FloatFunctions;
use crate::device::requirements::{FormatUsage, VkFeature};
use crate::error::{
    Attachment, Capability, DescriptorError, Error, Feature, RuntimeInput, Unsupported,
};
use crate::lower::arith::nan::NanTest;
use crate::lower::arith::{math, ops, pack, FloatFmt, Form, Held};
use crate::lower::dispatch::KernelBuiltin;
use crate::lower::emit::EmitCx;
use crate::lower::select::SelectCx;
use crate::lower::sync::barrier::{memory_model, semantics};
use crate::lower::texture::{held_texel, to_image, ImageDecl, TexelClass, TexelConv, TexelForm};
use crate::memory::arena::ArenaMem;
use crate::memory::leaf::{self, Leaf, Path, WordMem, WordOps};
use crate::metal::format::FormatEmulation;
use crate::metal::PixelFormat;
use crate::pipeline::artifact::PrecisionNote;
use crate::pipeline::facts::RuntimeFact;
use crate::pipeline::sym::{
    PushField, ResourceKind, ResourceSlot, SpecKey, Stage, SyntheticBinding,
};
use crate::spirv::builder::constructs::{ArmEnd, ArmFn, LoopFn, LoopSpec};
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
use crate::spirv::error::BResult;
use crate::spirv::ids::{ConstId, FuncId, GlobalId, Value as SValue};
use crate::spirv::ir::{
    BinOp, ConstBits, Dim, ImageOperands, ImageType, ImageUsage, MemAccess, MemoryOrder, Scope,
    StorageClass, Type as SType,
};

use super::imageblock::{self, ImageblockStrategy};

fn not_implemented() -> Error {
    Unsupported::not_implemented(Feature::TileShading).into()
}

/// The refusal of a kernel that reaches the implicit imageblock without tile inputs.
pub(crate) fn needs_tile() -> Error {
    Error::NeedsInput(RuntimeInput::TileAttachments)
}

/// Whether the function compiles as a tile function: a kernel with tile inputs.
pub(crate) fn is_tile(cx: &SelectCx<'_>) -> bool {
    cx.stage == Stage::Kernel && cx.inputs.tile_attachments.is_some()
}

/// The refusal of a tile function that declares threadgroup variables (`local::use_arena`,
/// [`finish`]), as Metal refuses its pipeline: its threadgroup memory is its tile's, its
/// `[[threadgroup(n)]]` arguments alone. A fragment function's is too (`access::tile_memory`), and
/// Metal's compiler refuses its threadgroup variables.
pub(crate) fn threadgroup_variable() -> Error {
    Error::InvalidDescriptor(crate::error::DescriptorError::TileThreadgroupVariable)
}

/// Whether the tile function's threadgroup arena holds its tile's threadgroup memory: its
/// `[[threadgroup(n)]]` arguments (`TgShape::tile`), which the entry stages ([`wrap_entry`]). A
/// fragment function's (`TgShape::fragment`) is the pass's buffer itself, which nothing stages.
pub fn has_memory(access: &super::AccessPlan) -> bool {
    access
        .local
        .tg
        .as_ref()
        .is_some_and(|s| s.tile && s.fragment.is_none())
}

/// One color attachment a layout of the tile function holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TileSlot {
    /// The color index.
    pub color: u32,
    /// The texel's byte offset in a cell.
    pub offset: u32,
    /// The texel as a cell holds it, which an implicit access reads and writes (through
    /// [`TileSlot::norm`]'s conversion where it is set).
    pub leaf: Leaf,
    /// The texel's bits as the staging moves them: [`TileSlot::leaf`]'s bytes as unsigned
    /// integer lanes of its lanes' width.
    pub bits: Leaf,
    /// The attachment's pixel format.
    pub pixel_format: PixelFormat,
    /// The texel class and conversion of the image the staging views the attachment as
    /// ([`staging_format`]).
    pub class: TexelClass,
    pub conv: TexelConv,
    /// The image's declared format (set once the slot is staged, [`finish`]).
    pub format: crate::spv::ImageFormat,
    /// The function reaches the attachment: the staging loads it.
    pub staged: bool,
    /// The function may write the attachment: the staging stores it.
    pub stored: bool,
    /// Where the attachment is normalized: the cell holds its texel's bits
    /// ([`TileSlot::bits`]), which an implicit access converts to and from its lanes.
    pub norm: Option<Norm>,
    /// The bytes of the layout member that holds the attachment: its texel's
    /// ([`TileSlot::leaf`]'s), or a normalized attachment's lanes (four bytes a `float` lane, two
    /// a `half` one, of its channels or four), which an implicit access's type fills.
    pub member: u32,
}

/// The conversion between a normalized attachment's texel bits, which its cell holds, and the
/// lanes an implicit access reads and writes ([`NormAccess`]): the device's conversion of the pack
/// function of the format's channels (`lower::arith::pack`), one 32-bit word of the bits at a time
/// (see the module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Norm {
    /// The pack function of each word of the bits.
    pub format: PackFormat,
    /// The format's channels.
    pub channels: u8,
    /// The format holds blue before red (`BGRA8Unorm`, `BGR10A2Unorm`): a word's lanes 0 and 2
    /// are the access's lanes 2 and 0.
    pub bgra: bool,
}

impl Norm {
    /// The lanes of one word's pack function (4 for `unorm4x8`, `snorm4x8` and `unorm.rgb10a2`,
    /// 2 for `unorm2x16` and `snorm2x16`).
    pub fn word_lanes(&self) -> u8 {
        math::pack_layout(self.format).map_or(0, |(widths, _)| widths.len() as u8)
    }
}

/// An implicit access of a normalized attachment: the attachment's conversion and the access's
/// lanes, `float` or `half` ones of the format's channels or four, whose lanes past the channels
/// a load reads as Metal does (green and blue 0, alpha 1) and a store drops (probes).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NormAccess {
    pub norm: Norm,
    /// The access's lanes: the channels, or four.
    pub lanes: u8,
    /// The lanes are `half`, held as their `f32`: a load rounds each channel's unpacked `float` to
    /// `half` (`narrow::round`), and a store converts the held value as a `float` one (probe 12).
    pub half: bool,
}

impl NormAccess {
    /// An upper estimate of the words the access's conversion of `words` words of bits emits:
    /// per word, the pack or unpack sequence as `air.pack`'s strategy charges it (`arith::pack`; a
    /// store packs, `store`), and per lane an extract and a construct; a `half` load's rounding per
    /// lane (`narrow::round`).
    pub fn words(&self, words: u8, store: bool) -> u32 {
        let dir = match store {
            true => PackDir::Pack,
            false => PackDir::Unpack,
        };
        let floats = Form {
            held: Held::Float(FloatFmt::F32),
            lanes: self.norm.word_lanes(),
        };
        let per =
            pack::words(self.norm.format, dir, floats) + 24 * u32::from(self.norm.word_lanes());
        let round = if self.half && !store { 64 } else { 0 };
        u32::from(words) * per + (16 + round) * u32::from(self.lanes)
    }
}

/// The conversion and the staging format of an attachment of normalized pixel format `pf`, or
/// `None` for another format. Each is a format whose implicit imageblock Metal converts as its pack
/// functions do (Metal probes, on the M3 Max and an M4); its staging format is the unsigned-integer
/// format of its texel's bytes in 32-bit words (`R16Uint` for a texel of two bytes), in the same
/// compatibility class.
pub fn normalized(pf: PixelFormat) -> Option<(Norm, PixelFormat)> {
    use PackFormat as F;
    use PixelFormat as P;
    let (format, channels, bgra, staging) = match pf {
        P::RGBA8Unorm => (F::Unorm4x8, 4, false, P::R32Uint),
        P::BGRA8Unorm => (F::Unorm4x8, 4, true, P::R32Uint),
        P::RGBA8UnormSRGB => (F::Unorm4x8Srgb, 4, false, P::R32Uint),
        P::BGRA8UnormSRGB => (F::Unorm4x8Srgb, 4, true, P::R32Uint),
        P::RGBA8Snorm => (F::Snorm4x8, 4, false, P::R32Uint),
        P::RGB10A2Unorm => (F::UnormRgb10a2, 4, false, P::R32Uint),
        P::BGR10A2Unorm => (F::UnormRgb10a2, 4, true, P::R32Uint),
        P::RG16Unorm => (F::Unorm2x16, 2, false, P::R32Uint),
        P::RG16Snorm => (F::Snorm2x16, 2, false, P::R32Uint),
        P::RGBA16Unorm => (F::Unorm2x16, 4, false, P::RG32Uint),
        P::RGBA16Snorm => (F::Snorm2x16, 4, false, P::RG32Uint),
        P::RG8Unorm => (F::Unorm4x8, 2, false, P::R16Uint),
        P::RG8Snorm => (F::Snorm4x8, 2, false, P::R16Uint),
        P::R16Unorm => (F::Unorm2x16, 1, false, P::R16Uint),
        P::R16Snorm => (F::Snorm2x16, 1, false, P::R16Uint),
        _ => return None,
    };
    Some((
        Norm {
            format,
            channels,
            bgra,
        },
        staging,
    ))
}

/// The tile function's attachments (`AccessPlan::tile`), by color index.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TilePlan {
    /// The imageblock region's cell stride (`memory::plan`), a whole number of words.
    pub cell_bytes: u32,
    pub slots: Vec<TileSlot>,
    /// The words of a cell that no staged texel covers whole, which the staging zeroes before it
    /// loads the texels ([`finish`]): no word of a cell is read before it is written.
    pub pad_words: Vec<u32>,
    /// Where the function's one imageblock is an explicit layout that no implicit one stands
    /// beside: its cells are the pass's imageblock storage (the module docs).
    pub pass_cells: Option<PassCells>,
    /// The bytes of each of those cells the dispatch's render pass keeps, where they are fewer
    /// than a cell holds (`LoweringInputs::kept_imageblock_bytes`, a whole number of words):
    /// every access reads the words past them as zero and drops their writes ([`KeptMem`]).
    pub kept_bytes: Option<u32>,
}

/// A tile function's explicit imageblock that stands alone (`TilePlan::pass_cells`): the pass's
/// imageblock storage, `SyntheticBinding::Imageblock`, which the pass's fragment functions'
/// imageblocks are in, each cell's words staged into the tile's cells and back.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PassCells {
    /// The function reaches its cells: the staging loads them.
    pub staged: bool,
    /// The function may write them: the staging stores them.
    pub stored: bool,
}

impl TilePlan {
    /// The staged slots, by color index.
    pub fn staged(&self) -> impl Iterator<Item = &TileSlot> {
        self.slots.iter().filter(|s| s.staged)
    }

    /// The pass's imageblock storage the staging loads, where the function reaches its cells.
    pub fn pass_cells(&self) -> Option<PassCells> {
        self.pass_cells.filter(|c| c.staged)
    }

    /// An upper estimate of the words the staging adds: two loops, two barriers, the call, and
    /// per staged slot its image, a read, a guarded leaf store, and a leaf load and write; per
    /// word of the pass's imageblock storage, a read, a word store, a word load and a write.
    pub fn words_estimate(&self) -> u64 {
        let pass = match self.pass_cells() {
            Some(_) => 96 + 64 * u64::from(self.cell_bytes / 4),
            None => 0,
        };
        let per: u64 = pass
            + self
                .staged()
                .map(|s| 96 + 4 * u64::from(s.leaf.cost()))
                .sum::<u64>();
        if per == 0 {
            0
        } else {
            256 + per
        }
    }
}

/// The alignment a constant byte offset guarantees (at most 16).
fn offset_align(offset: u32) -> u32 {
    1u32 << offset.trailing_zeros().min(4)
}

/// The texel a cell holds unchanged of an attachment of pixel format `pf`: its held element, its
/// channels, and the image's class (see the module docs), or `None` for a format that converts.
fn texel_of(pf: PixelFormat) -> Option<(Held, u8, TexelClass)> {
    use PixelFormat as P;
    use TexelClass::{Float, Sint, Uint};
    let half = Held::Float(FloatFmt::Half);
    let f32_ = Held::Float(FloatFmt::F32);
    let i16_ = Held::Int {
        bits: 16,
        container: 32,
    };
    let i32_ = Held::Int {
        bits: 32,
        container: 32,
    };
    Some(match pf {
        P::R16Float => (half, 1, Float),
        P::RG16Float => (half, 2, Float),
        P::RGBA16Float => (half, 4, Float),
        P::R32Float => (f32_, 1, Float),
        P::RG32Float => (f32_, 2, Float),
        P::RGBA32Float => (f32_, 4, Float),
        P::R16Uint => (i16_, 1, Uint),
        P::RG16Uint => (i16_, 2, Uint),
        P::RGBA16Uint => (i16_, 4, Uint),
        P::R16Sint => (i16_, 1, Sint),
        P::RG16Sint => (i16_, 2, Sint),
        P::RGBA16Sint => (i16_, 4, Sint),
        P::R32Uint => (i32_, 1, Uint),
        P::RG32Uint => (i32_, 2, Uint),
        P::RGBA32Uint => (i32_, 4, Uint),
        P::R32Sint => (i32_, 1, Sint),
        P::RG32Sint => (i32_, 2, Sint),
        P::RGBA32Sint => (i32_, 4, Sint),
        _ => return None,
    })
}

/// The format the staging views an attachment of pixel format `pf` through: its own for an integer
/// format, and for a float one the unsigned-integer format of the same channels and widths, which
/// is in the same compatibility class, so the staging moves the texel's bits. A float view would
/// convert them to values (see the module docs).
pub fn staging_format(pf: PixelFormat) -> PixelFormat {
    use PixelFormat as P;
    if let Some((_, staging)) = normalized(pf) {
        return staging;
    }
    match pf {
        P::R16Float => P::R16Uint,
        P::RG16Float => P::RG16Uint,
        P::RGBA16Float => P::RGBA16Uint,
        P::R32Float => P::R32Uint,
        P::RG32Float => P::RG32Uint,
        P::RGBA32Float => P::RGBA32Uint,
        other => other,
    }
}

/// The image read's conversion of the integer element `held` of an image of class `class`.
fn conv_of(held: Held, class: TexelClass) -> TexelConv {
    match held {
        Held::Int { bits: 16, .. } => TexelConv::I16 {
            signed: class == TexelClass::Sint,
        },
        _ => TexelConv::I32,
    }
}

/// The tile function's explicit imageblock where it stands alone, with its parameter: no other
/// imageblock beside it, and not aliased onto an implicit one. Its cells are the pass's imageblock
/// storage (the module docs).
fn standing_alone<'e>(cx: &'e SelectCx<'_>) -> Option<(&'e ParamBinding, &'e ImageblockArg)> {
    let mut imageblocks = cx.entry.ep.params.iter().filter_map(|b| match &b.role {
        ParamRole::Imageblock(arg) => Some((b, arg)),
        _ => None,
    });
    match (imageblocks.next(), imageblocks.next()) {
        (Some((b, arg)), None) if arg.kind == ImageblockKind::Explicit && !arg.alias_implicit => {
            Some((b, arg))
        }
        _ => None,
    }
}

/// The tile function's plan, built on first use: one slot per attachment its imageblock layouts
/// hold (see the module docs for what is refused).
fn plan<'p>(cx: &'p mut SelectCx<'_>) -> Result<&'p mut TilePlan, Error> {
    if cx.families.access.tile.is_none() {
        let built = build(cx)?;
        return Ok(cx.families.access.tile.insert(built));
    }
    cx.families.access.tile.as_mut().ok_or_else(|| {
        Error::Internal(crate::error::InternalError::new(
            crate::error::InternalKind::Invariant,
            "a tile plan without its slots",
        ))
    })
}

fn build(cx: &SelectCx<'_>) -> Result<TilePlan, Error> {
    let Some(formats) = cx.inputs.tile_attachments.as_ref() else {
        return Err(needs_tile());
    };
    let cell_bytes = cx
        .memory
        .tg_arena
        .imageblock
        .map(|r| r.cell_bytes)
        .ok_or_else(not_implemented)?;
    // An explicit layout alone is the pass's imageblock storage, a whole number of words a cell,
    // of which the pass may keep fewer bytes.
    let kept_bytes = cx.inputs.kept_imageblock_bytes;
    if standing_alone(cx).is_some() {
        if cell_bytes % 4 != 0 {
            return Err(not_implemented());
        }
        if kept_bytes.is_some_and(|kept| kept % 4 != 0 || kept >= cell_bytes) {
            return Err(cx.invariant("kept imageblock bytes that are not a cell's first words"));
        }
        return Ok(TilePlan {
            cell_bytes,
            slots: Vec::new(),
            pad_words: Vec::new(),
            pass_cells: Some(PassCells::default()),
            kept_bytes,
        });
    }
    if kept_bytes.is_some() {
        return Err(kept_without_cells(cx));
    }
    // Each attachment's offset and size, which every layout holding it must agree on, and the
    // attachments an aliased layout holds.
    let mut members: BTreeMap<u32, (u32, u32)> = BTreeMap::new();
    let mut aliased_colors = BTreeSet::new();
    for b in &cx.entry.ep.params {
        let ParamRole::Imageblock(arg) = &b.role else {
            continue;
        };
        let aliased = match arg.kind {
            ImageblockKind::Implicit => false,
            ImageblockKind::Explicit if arg.alias_implicit => true,
            // An explicit layout beside the implicit imageblock, which is not aliased onto it.
            _ => return Err(not_implemented()),
        };
        let layout = arg
            .layout
            .and_then(|r| cx.entry.structs.get(usize::try_from(r.0).ok()?))
            .ok_or_else(not_implemented)?;
        for (i, m) in layout.members.iter().enumerate() {
            let color = if aliased {
                u32::try_from(i).map_err(|_| not_implemented())?
            } else {
                m.render_target.ok_or_else(not_implemented)?
            };
            if m.array_len != 0 || m.layout.is_some() {
                return Err(not_implemented());
            }
            if aliased {
                aliased_colors.insert(color);
            }
            match members.entry(color) {
                MapEntry::Occupied(e) if *e.get() != (m.offset, m.size) => {
                    return Err(not_implemented())
                }
                MapEntry::Occupied(_) => {}
                MapEntry::Vacant(e) => {
                    e.insert((m.offset, m.size));
                }
            }
        }
    }
    let mut slots = Vec::with_capacity(members.len());
    for (color, (offset, size)) in members {
        let Some(&pixel_format) = formats.get(&color) else {
            return Err(Error::InvalidDescriptor(DescriptorError::ColorAttachment {
                index: color,
            }));
        };
        let aliased = aliased_colors.contains(&color);
        let at = Member {
            color,
            offset,
            size,
            aliased,
        };
        slots.push(
            slot(cx, at, pixel_format, cell_bytes)
                .map_err(|e| e.at_attachment(Attachment::Color(color)))?,
        );
    }
    if cell_bytes % 4 != 0 {
        return Err(not_implemented());
    }
    Ok(TilePlan {
        cell_bytes,
        slots,
        pad_words: Vec::new(),
        pass_cells: None,
        kept_bytes: None,
    })
}

/// An attachment's member of the function's imageblock layouts: its color index, byte offset
/// and size in a cell, and whether an aliased explicit layout holds it.
struct Member {
    color: u32,
    offset: u32,
    size: u32,
    aliased: bool,
}

/// The slot of the attachment of `pixel_format` the function's layouts hold as `at`, in cells of
/// `cell_bytes`.
fn slot(
    cx: &SelectCx<'_>,
    at: Member,
    pixel_format: PixelFormat,
    cell_bytes: u32,
) -> Result<TileSlot, Error> {
    let Member {
        color,
        offset,
        size,
        aliased,
    } = at;
    let (raw, raw_lanes, class) =
        texel_of(staging_format(pixel_format)).ok_or_else(not_implemented)?;
    // The cell holds a normalized texel's bits, whose member is `float` or `half` lanes of
    // the format's channels or four, as an access's type tells; an aliased layout's member of it is
    // not read yet.
    let (norm, form, bytes, member) = match normalized(pixel_format) {
        Some((n, _)) => {
            if aliased {
                return Err(not_implemented());
            }
            let lanes = |width: u32| {
                u8::try_from(size / width)
                    .ok()
                    .filter(|&l| size % width == 0 && (l == n.channels || l == 4))
            };
            if lanes(4).is_none() && lanes(2).is_none() {
                return Err(not_implemented());
            }
            let form = Form {
                held: raw,
                lanes: raw_lanes,
            };
            let bytes = u32::from(form.bits() / 8) * u32::from(raw_lanes);
            (Some(n), form, bytes, size)
        }
        None => {
            let (held, lanes, _) = texel_of(pixel_format).ok_or_else(not_implemented)?;
            let held = crate::lower::arith::legal::held_on(cx.profile, held)?;
            let form = Form { held, lanes };
            let bytes = u32::from(form.bits() / 8) * u32::from(lanes);
            (None, form, bytes, bytes)
        }
    };
    if member != size || offset.checked_add(size).is_none_or(|e| e > cell_bytes) {
        return Err(not_implemented());
    }
    let align = offset_align(offset).min(offset_align(cell_bytes));
    let path = if align >= 4 {
        Path::Aligned
    } else if bytes <= 2 && bytes <= align {
        Path::OneWord
    } else {
        Path::Bytes
    };
    let bits = Leaf {
        form: Form {
            held: raw,
            lanes: raw_lanes,
        },
        bytes,
        path,
    };
    Ok(TileSlot {
        color,
        offset,
        leaf: Leaf { form, bytes, path },
        bits,
        pixel_format,
        class,
        conv: conv_of(raw, class),
        format: crate::spv::ImageFormat::Unknown,
        staged: false,
        stored: false,
        norm,
        member,
    })
}

/// The refusal of kept imageblock bytes for a function whose cells are not the pass's imageblock
/// storage: only a tile function's explicit imageblock standing alone has them (the binder asks
/// for them by `RuntimeFact::TileImageblock`).
fn kept_without_cells(cx: &SelectCx<'_>) -> Error {
    cx.invariant("kept imageblock bytes for a function without a standalone tile imageblock")
}

/// The words of the tile's cells the pass keeps (`TilePlan::kept_bytes`), as the open function
/// reaches its threadgroup arena: word `w` of the arena is kept below the imageblock region
/// (`region_word`, the tile's threadgroup memory), and in it where `(w - region_word) %
/// cell_words < kept_words`. Plain Metal keeps the words of a standalone explicit imageblock
/// below the pass's imageblock sample length rounded up to 8 bytes and reads every other as zero,
/// dropping its writes, a dispatch's own included (Metal probes, on an M4).
#[derive(Clone, Copy, Debug)]
pub(crate) struct KeptCells {
    region_word: SValue,
    cell_words: u32,
    kept_words: u32,
}

impl KeptCells {
    /// The kept cells of the open function's plan, `None` where it keeps every word: a function
    /// whose staged cells are the pass's imageblock storage under `kept_bytes`.
    pub(crate) fn of(cx: &mut EmitCx<'_>, fb: &mut FunctionBuilder<'_>) -> BResult<Option<Self>> {
        let Some(p) = cx.plan.families.access.tile.as_ref() else {
            return Ok(None);
        };
        let Some(kept) = p.kept_bytes.filter(|_| p.pass_cells().is_some()) else {
            return Ok(None);
        };
        let (cell_words, kept_words) = (p.cell_bytes / 4, kept / 4);
        let region = imageblock::region(cx, fb)?;
        let two = leaf::k32(fb, 2)?;
        let region_word = fb.binary(BinOp::ShiftRightLogical, region, two)?;
        Ok(Some(Self {
            region_word,
            cell_words,
            kept_words,
        }))
    }

    /// Whether arena word `word` (a `u32`) is kept.
    pub(crate) fn kept(&self, fb: &mut FunctionBuilder<'_>, word: SValue) -> BResult<SValue> {
        let below = fb.binary(BinOp::ULessThan, word, self.region_word)?;
        let rel = fb.binary(BinOp::ISub, word, self.region_word)?;
        let cell_words = leaf::k32(fb, self.cell_words)?;
        let at = fb.binary(BinOp::UMod, rel, cell_words)?;
        let kept_words = leaf::k32(fb, self.kept_words)?;
        let inside = fb.binary(BinOp::ULessThan, at, kept_words)?;
        fb.binary(BinOp::LogicalOr, below, inside)
    }
}

/// A threadgroup arena's word memory that reads the words of the tile's cells the pass does not
/// keep ([`KeptCells`]) as zero and drops their writes; every word with `kept` `None`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct KeptMem {
    pub(crate) mem: ArenaMem,
    pub(crate) kept: Option<KeptCells>,
}

impl WordMem for KeptMem {
    fn read(&mut self, fb: &mut FunctionBuilder<'_>, word: SValue) -> BResult<SValue> {
        let v = self.mem.read(fb, word)?;
        let Some(cells) = self.kept else {
            return Ok(v);
        };
        let kept = cells.kept(fb, word)?;
        let zero = leaf::k32(fb, 0)?;
        fb.select(kept, v, zero)
    }

    fn write(&mut self, fb: &mut FunctionBuilder<'_>, word: SValue, value: SValue) -> BResult<()> {
        let Some(cells) = self.kept else {
            return self.mem.write(fb, word, value);
        };
        let kept = cells.kept(fb, word)?;
        let mut mem = self.mem;
        let then: ArmFn<'_, ()> = Box::new(move |fb, _| {
            mem.write(fb, word, value)?;
            Ok(ArmEnd::FallThrough(Vec::new()))
        });
        fb.if_(&mut (), kept, then, None)?;
        Ok(())
    }

    fn write_bits(
        &mut self,
        fb: &mut FunctionBuilder<'_>,
        word: SValue,
        mask: SValue,
        bits: SValue,
    ) -> BResult<()> {
        let Some(cells) = self.kept else {
            return self.mem.write_bits(fb, word, mask, bits);
        };
        let kept = cells.kept(fb, word)?;
        let mut mem = self.mem;
        let then: ArmFn<'_, ()> = Box::new(move |fb, _| {
            mem.write_bits(fb, word, mask, bits)?;
            Ok(ArmEnd::FallThrough(Vec::new()))
        });
        fb.if_(&mut (), kept, then, None)?;
        Ok(())
    }
}

/// Use every one of the tile's cells (an aliased layout's cell pointer, or a slice write), whose
/// cell bytes this returns: the function reads every attachment a layout holds, and may write
/// each when `stored`.
pub(super) fn use_cells(cx: &mut SelectCx<'_>, stored: bool) -> Result<u32, Error> {
    let p = plan(cx)?;
    // A normalized attachment's cell holds its bits, which only an implicit access converts.
    if p.slots.iter().any(|s| s.norm.is_some()) {
        return Err(not_implemented());
    }
    for s in &mut p.slots {
        s.staged = true;
        s.stored |= stored;
    }
    if let Some(cells) = &mut p.pass_cells {
        cells.staged = true;
        cells.stored |= stored;
    }
    Ok(p.cell_bytes)
}

/// The tile plan's cell bytes, for an extent query.
pub(super) fn cell_bytes(cx: &mut SelectCx<'_>) -> Result<u32, Error> {
    Ok(plan(cx)?.cell_bytes)
}

/// The integer constant `v`, if it is one.
fn const_u64(cx: &SelectCx<'_>, v: Value) -> Option<u64> {
    let Value::Const(c) = v else { return None };
    match cx.entry.module.consts().get(c).map(|k| &k.value) {
        Some(Const::Int(n)) => Some(*n),
        Some(Const::Null) => Some(0),
        _ => None,
    }
}

/// Select an implicit imageblock access (`op` a `Load` or `Store`) of a tile function.
pub(super) fn select_texel(
    cx: &mut SelectCx<'_>,
    call: &Call,
    ty: crate::air::ir::TypeId,
    op: &Imageblock,
) -> Result<ImageblockStrategy, Error> {
    if !is_tile(cx) {
        return Err(needs_tile());
    }
    let store = matches!(op, Imageblock::Store(_));
    let (value, rest) = match (store, &call.args[..]) {
        (true, [v, rest @ ..]) => (Some(*v), rest),
        (false, rest) => (None, rest),
        (true, []) => return Err(cx.invariant("an implicit imageblock store without operands")),
    };
    let [color, coord, mode, sample] = rest[..] else {
        return Err(cx.invariant("an implicit imageblock access of another arity"));
    };
    let color = const_u64(cx, color)
        .and_then(|c| u32::try_from(c).ok())
        .ok_or_else(not_implemented)?;
    // A single-sample pass: the pixel's own data (`read(coord)`), or its color-rate data at
    // sample 0, which is the pixel's.
    if !matches!(const_u64(cx, mode), Some(0 | 1)) || const_u64(cx, sample) != Some(0) {
        return Err(not_implemented());
    }
    let access_ty = match value {
        Some(v) => cx
            .value_ty(v)
            .ok_or_else(|| cx.invariant("an implicit imageblock value without a type"))?,
        None => ty,
    };
    let leaf = leaf::leaf_of(cx.entry, cx.profile, &cx.memory.layouts, access_ty, 1)?;
    let contained = imageblock::contained(cx);
    let p = plan(cx)?;
    let cell_bytes = p.cell_bytes;
    let slot = p
        .slots
        .iter_mut()
        .find(|s| s.color == color)
        .ok_or_else(not_implemented)?;
    // The access's type is the attachment's texel's (signedness is not part of an AIR type), or
    // a normalized attachment's channels or four as `float` or `half` lanes that fill its member.
    let norm = match slot.norm {
        Some(n) => {
            let half = match leaf.form.held {
                Held::Float(FloatFmt::F32) => false,
                Held::Float(FloatFmt::Half) => true,
                _ => return Err(not_implemented()),
            };
            let lanes = leaf.form.lanes;
            if (lanes != n.channels && lanes != 4) || leaf.bytes != slot.member {
                return Err(not_implemented());
            }
            Some(NormAccess {
                norm: n,
                lanes,
                half,
            })
        }
        None if leaf.form == slot.leaf.form && leaf.bytes == slot.leaf.bytes => None,
        None => return Err(not_implemented()),
    };
    slot.staged = true;
    slot.stored |= store;
    let (offset, leaf) = (slot.offset, slot.leaf);
    // The device's conversion, but a `float` one of a format `GLSL.std.450` names on a device
    // whose `float` functions are Metal's (Metal's own `pack_float_to_*` and `unpack_*_to_float`,
    // as `air.pack`'s strategy notes it); a `half` load rounds the unpacked `float` once more.
    if let Some(n) = norm {
        let dir = match store {
            true => PackDir::Pack,
            false => PackDir::Unpack,
        };
        let own = cx.profile.fields().float_functions == FloatFunctions::Metal
            && (store || !n.half)
            && pack::instruction(n.norm.format, dir).is_some();
        if !own {
            cx.precision.insert(PrecisionNote::DevicePackConversion);
        }
    }
    imageblock::place_region(cx, cell_bytes)?;
    let coord = imageblock::operand(cx, coord)?;
    let value = match value {
        Some(v) => Some(imageblock::operand(cx, v)?),
        None => None,
    };
    Ok(ImageblockStrategy::Texel {
        coord,
        cell_bytes,
        offset,
        leaf,
        value,
        contained,
        norm,
    })
}

/// Complete a tile function's plan once its body is selected: refuse the threadgroup variables
/// it reaches, as Metal refuses its pipeline (the specialized entry holds only what it reaches),
/// then its fact; for the staging of its tile's threadgroup memory, the buffer's descriptor; and
/// for the staging of each attachment it reaches, its image's format and descriptor; and the
/// dispatch builtins and push field either staging reads ([`crate::lower::dispatch::plan_kernel`]
/// declares the builtins).
pub fn finish(cx: &mut SelectCx<'_>) -> Result<(), Error> {
    // Kept imageblock bytes are a standalone tile imageblock's alone.
    if cx.inputs.kept_imageblock_bytes.is_some() && (!is_tile(cx) || standing_alone(cx).is_none()) {
        return Err(kept_without_cells(cx));
    }
    if !is_tile(cx) {
        return Ok(());
    }
    if cx.memory.has_static_threadgroup_objects() {
        return Err(threadgroup_variable());
    }
    cx.facts.insert(RuntimeFact::TileFunction);
    if has_memory(&cx.families.access) {
        cx.descriptors.storage_buffers = cx.descriptors.storage_buffers.saturating_add(1);
        // The tile's index is a grid builtin.
        cx.push.insert(PushField::DispatchRegion);
    }
    let aliased = cx.entry.ep.params.iter().find_map(|b| match &b.role {
        ParamRole::Imageblock(arg)
            if arg.kind == ImageblockKind::Explicit && arg.alias_implicit =>
        {
            arg.data_size
        }
        _ => None,
    });
    if let Some(data_size) = aliased {
        cx.facts
            .insert(RuntimeFact::AliasedImageblock { data_size });
    }
    let alone = standing_alone(cx).and_then(|(b, arg)| Some((arg.data_size?, b.ty.align?)));
    if let Some((data_size, alignment)) = alone {
        cx.facts.insert(RuntimeFact::TileImageblock {
            data_size,
            alignment,
        });
    }
    let profile = cx.profile;
    let Some(p) = cx.families.access.tile.as_mut() else {
        return Ok(());
    };
    // The pass's imageblock storage, and the tile's index, a grid builtin.
    if p.pass_cells().is_some() {
        cx.descriptors.storage_images = cx.descriptors.storage_images.saturating_add(1);
        cx.push.insert(PushField::DispatchRegion);
    }
    let mut staged = 0u32;
    for s in p.slots.iter_mut().filter(|s| s.staged) {
        s.format = storage_format(profile, s.pixel_format)
            .map_err(|e| e.at_attachment(Attachment::Color(s.color)))?;
        cx.facts.insert(RuntimeFact::TileAttachmentFormat {
            color: s.color,
            format: s.pixel_format,
        });
        staged += 1;
    }
    if staged == 0 {
        return Ok(());
    }
    let covered = |at: u32| {
        p.slots
            .iter()
            .any(|s| s.staged && at >= s.offset && at < s.offset + s.leaf.bytes)
    };
    p.pad_words = (0..p.cell_bytes / 4)
        .filter(|w| !(4 * w..4 * w + 4).all(covered))
        .collect();
    cx.descriptors.storage_images = cx.descriptors.storage_images.saturating_add(staged);
    // The tile's index is a grid builtin.
    cx.push.insert(PushField::DispatchRegion);
    Ok(())
}

/// Whether the plan stages an attachment, so the kernel reads the builtins the staging needs.
pub fn stages(plan: Option<&TilePlan>) -> bool {
    plan.is_some_and(|p| p.staged().next().is_some() || p.pass_cells().is_some())
}

/// Whether the entry stages anything: an attachment, or its tile's threadgroup memory.
pub fn stages_any(access: &super::AccessPlan) -> bool {
    stages(access.tile.as_ref()) || has_memory(access)
}

/// The builtins the staging reads, beside the threads its threadgroup runs
/// ([`running_threads`]).
pub const BUILTINS: [KernelBuiltin; 2] = [
    KernelBuiltin::ThreadgroupPositionInGrid,
    KernelBuiltin::ThreadIndexInThreadgroup,
];

/// The declared format of the image of an attachment of pixel format `pf`: the Vulkan format of
/// its staging format ([`staging_format`]), storage-capable on the profile, refusing what the
/// profile lacks.
fn storage_format(
    profile: &crate::device::DeviceProfile,
    pf: PixelFormat,
) -> Result<crate::spv::ImageFormat, Error> {
    let usages = BTreeSet::from([FormatUsage::StorageImage]);
    let mapping = crate::metal::format::vk_format(staging_format(pf), &usages, profile)?;
    if mapping.emulation != FormatEmulation::Direct {
        return Err(not_implemented());
    }
    let format =
        crate::metal::format::spirv_image_format(mapping.format).ok_or_else(not_implemented)?;
    if crate::spirv::requirements::extended_format(format)
        && !profile.fields().images.storage_extended_formats
    {
        return Err(Unsupported::needs(
            Feature::Format,
            Capability::Feature(VkFeature::ShaderStorageImageExtendedFormats),
        )
        .into());
    }
    Ok(format)
}

/// The image declaration of slot `s`.
fn image_decl(s: &TileSlot) -> ImageDecl {
    ImageDecl {
        dim: Dim::D2,
        arrayed: false,
        multisampled: false,
        depth: false,
        usage: ImageUsage::Storage,
        format: s.format,
        texel: s.class,
        cube_as_array: false,
        class: crate::pipeline::lower_inputs::StorageImageClass::Identity,
        level_views: None,
        extended: None,
        x_channel: false,
        in_place: false,
        planes_sampler: None,
    }
}

/// The tile function's module-scope objects: each staged attachment's storage image, by color
/// index, and the render area's specialization constants; and where it stages its tile's
/// threadgroup memory, the pass's buffer of it and the tile grid's width.
#[derive(Clone, Debug, Default)]
pub struct TileDecl {
    pub images: Vec<(u32, GlobalId)>,
    pub area: Option<(ConstId, ConstId)>,
    pub memory: Option<(crate::memory::views::View, ConstId)>,
    /// The pass's imageblock storage, where the function's cells are its.
    pub pass_cells: Option<GlobalId>,
}

/// Declare the staged attachments' images and the render area's constants, and the tile
/// threadgroup memory's buffer and grid width.
pub fn declare(mb: &mut ModuleBuilder, access: &super::AccessPlan) -> BResult<TileDecl> {
    let mut d = TileDecl::default();
    if has_memory(access) {
        let slot = ResourceSlot {
            stage: Stage::Kernel,
            kind: ResourceKind::Synthetic(SyntheticBinding::TileMemory),
            index: 0,
        };
        let view =
            crate::memory::views::declare_at(mb, slot, crate::memory::views::ViewKind::Storage)?;
        let u32_ = mb.ty(SType::Int {
            width: 32,
            signed: false,
        })?;
        let grid = mb.spec(SpecKey::TileGridWidth, u32_, ConstBits(0))?;
        d.memory = Some((view, grid));
    }
    let Some(plan) = access.tile.as_ref().filter(|p| stages(Some(p))) else {
        return Ok(d);
    };
    for s in plan.staged() {
        let ty = image_decl(s).declare(mb)?;
        let slot = ResourceSlot {
            stage: Stage::Kernel,
            kind: ResourceKind::TileAttachment(s.color),
            index: s.color,
        };
        d.images.push((
            s.color,
            mb.resource(slot, StorageClass::UniformConstant, ty)?,
        ));
    }
    let u32_ = mb.ty(SType::Int {
        width: 32,
        signed: false,
    })?;
    if plan.pass_cells().is_some() {
        // The fragment functions' `R32Uint` 2D-array storage image (`lower::fragment::imageblock`):
        // layer `w` is word `w` of every pixel's cell.
        let ty = mb.ty(SType::Image(ImageType {
            sampled_type: u32_,
            dim: Dim::D2,
            depth: false,
            arrayed: true,
            multisampled: false,
            usage: ImageUsage::Storage,
            format: crate::spv::ImageFormat::R32ui,
        }))?;
        let slot = ResourceSlot {
            stage: Stage::Kernel,
            kind: ResourceKind::Synthetic(SyntheticBinding::Imageblock),
            index: 0,
        };
        d.pass_cells = Some(mb.resource(slot, StorageClass::UniformConstant, ty)?);
    }
    let w = mb.spec(SpecKey::RenderAreaWidth, u32_, ConstBits(0))?;
    let h = mb.spec(SpecKey::RenderAreaHeight, u32_, ConstBits(0))?;
    d.area = Some((w, h));
    Ok(d)
}

/// The value of kernel builtin `b` in the open block.
fn builtin(cx: &mut EmitCx<'_>, fb: &mut FunctionBuilder<'_>, b: KernelBuiltin) -> BResult<SValue> {
    let decl: &crate::lower::emit::Declared = cx.decl;
    let Some(k) = decl.families.dispatch.kernel.as_ref() else {
        return Err(cx.fail("a tile function's staging outside a kernel"));
    };
    k.value(fb, b, &mut |fb| {
        cx.push_field(fb, PushField::DispatchRegion)
    })
}

/// The threads the tile's threadgroup runs, which the staging's passes share out: Metal's
/// `threads_per_threadgroup` is the tile where the pipeline's threadgroup is the tile, whatever
/// threads run.
fn running_threads(cx: &mut EmitCx<'_>, fb: &mut FunctionBuilder<'_>) -> BResult<SValue> {
    match cx.decl.families.dispatch.kernel.as_ref() {
        Some(k) => k.running_threads(fb),
        None => Err(cx.fail("a tile function's staging outside a kernel")),
    }
}

/// The open block's workgroup barrier over the arena (`mem_threadgroup_imageblock`).
fn barrier(cx: &EmitCx<'_>, fb: &mut FunctionBuilder<'_>) -> BResult<()> {
    let flags = crate::air::intrinsics::MemFlags {
        threadgroup_imageblock: true,
        ..Default::default()
    };
    let sem = semantics(flags, MemoryOrder::AcquireRelease, memory_model(cx.plan));
    fb.control_barrier(Scope::Workgroup, Scope::Workgroup, sem)
}

/// What one pass over the tile's cells does with each staged slot, and one over its threadgroup
/// memory with each word.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Pass {
    /// Load each pixel inside the render area into its cell, and zero the others; load each
    /// word of the tile's memory from the pass's buffer.
    Load,
    /// Store each stored slot's cell to its pixel inside the render area; store each word of the
    /// tile's memory back.
    Store,
}

/// The staged slots with their images.
fn slots_with_images(cx: &mut EmitCx<'_>) -> BResult<Vec<(TileSlot, GlobalId)>> {
    let Some(plan) = cx.plan.families.access.tile.as_ref() else {
        return Ok(Vec::new());
    };
    let images = &cx.decl.families.access.tile.images;
    let mut out = Vec::new();
    for s in plan.staged() {
        match images.iter().find(|(c, _)| *c == s.color) {
            Some((_, g)) => out.push((*s, *g)),
            None => return Err(cx.fail("a staged attachment without its image")),
        }
    }
    Ok(out)
}

/// One pass over the tile's cells: cell `k = lin, lin + n, ...` below `width · height`, `n` the
/// threadgroup's threads, is `(k mod width, k / width)`, pixel `tile · (width, height) + cell`.
fn cells_pass<'c>(
    cx: &mut EmitCx<'c>,
    fb: &mut FunctionBuilder<'_>,
    pass: Pass,
    slots: &[(TileSlot, GlobalId)],
) -> BResult<()> {
    let u = fb.module().ty(SType::Int {
        width: 32,
        signed: false,
    })?;
    let uvec2 = fb.module().ty(SType::Vector { elem: u, len: 2 })?;
    let lin = builtin(cx, fb, KernelBuiltin::ThreadIndexInThreadgroup)?;
    let threads = running_threads(cx, fb)?;
    let tile = builtin(cx, fb, KernelBuiltin::ThreadgroupPositionInGrid)?;
    let tx = fb.composite_extract(threads, &[0])?;
    let ty_ = fb.composite_extract(threads, &[1])?;
    let tz = fb.composite_extract(threads, &[2])?;
    let txy = fb.binary(BinOp::IMul, tx, ty_)?;
    let n = fb.binary(BinOp::IMul, txy, tz)?;
    let (w, h) = imageblock::extent(fb)?;
    let cells = fb.binary(BinOp::IMul, w, h)?;
    let Some((aw, ah)) = cx.decl.families.access.tile.area else {
        return Err(cx.fail("a tile staging without its render area"));
    };
    let aw = fb.module().const_value(aw)?;
    let ah = fb.module().const_value(ah)?;
    let gx = fb.composite_extract(tile, &[0])?;
    let gy = fb.composite_extract(tile, &[1])?;
    let ox = fb.binary(BinOp::IMul, gx, w)?;
    let oy = fb.binary(BinOp::IMul, gy, h)?;
    let var = fb.local_var(u, None)?;
    let counter = fb.var_ptr(var)?;
    fb.store(counter, lin, MemAccess::default())?;
    let (cell_bytes, pads, pass_cells) = match &cx.plan.families.access.tile {
        Some(p) => (p.cell_bytes, p.pad_words.clone(), p.pass_cells()),
        None => return Err(cx.fail("a tile staging without its plan")),
    };
    // The pass's imageblock storage and whether the staging stores it.
    let storage = match (pass_cells, cx.decl.families.access.tile.pass_cells) {
        (Some(cells), Some(g)) => Some((g, cells.stored)),
        (None, _) => None,
        (Some(_), None) => return Err(cx.fail("a pass imageblock staging without its image")),
    };
    let words = cell_bytes / 4;
    let slots = slots.to_vec();
    let body: LoopFn<'_, EmitCx<'c>> = Box::new(move |fb, cx, lc| {
        let k = fb.load(counter, MemAccess::default())?;
        let done = fb.binary(BinOp::UGreaterThanEqual, k, cells)?;
        let exit: ArmFn<'_, EmitCx<'c>> = Box::new(move |fb, _| {
            lc.break_(fb)?;
            Ok(ArmEnd::Exits)
        });
        fb.if_(cx, done, exit, None)?;
        let next = fb.binary(BinOp::IAdd, k, n)?;
        fb.store(counter, next, MemAccess::default())?;
        let x = fb.binary(BinOp::UMod, k, w)?;
        let y = fb.binary(BinOp::UDiv, k, w)?;
        let cell = imageblock::cell_offset(cx, fb, x, y, cell_bytes)?;
        if pass == Pass::Load {
            for w in &pads {
                let off = field(fb, cell, 4 * w)?;
                let z = leaf::k32(fb, 0)?;
                cell_access(cx, fb, WORD, off, Some(z))?;
            }
        }
        let px = fb.binary(BinOp::IAdd, ox, x)?;
        let py = fb.binary(BinOp::IAdd, oy, y)?;
        // The pass's imageblock storage holds every cell of every tile the pass covers, a cell
        // past the render area too, so each cell is staged.
        if let Some((g, stored)) = storage {
            if pass == Pass::Load || stored {
                storage_words(cx, fb, pass, g, (px, py), cell, words)?;
            }
        }
        if slots.is_empty() {
            return Ok(());
        }
        let in_x = fb.binary(BinOp::ULessThan, px, aw)?;
        let in_y = fb.binary(BinOp::ULessThan, py, ah)?;
        let inside = fb.binary(BinOp::LogicalAnd, in_x, in_y)?;
        let pixel = fb.composite_construct(uvec2, &[px, py])?;
        let staged = slots.clone();
        let then: ArmFn<'_, EmitCx<'c>> = Box::new(move |fb, cx| {
            for (s, g) in &staged {
                if pass == Pass::Store && !s.stored {
                    continue;
                }
                let p = fb.module().global_value(*g)?;
                let img = fb.load(p, MemAccess::default())?;
                let off = field(fb, cell, s.offset)?;
                let form = TexelForm {
                    conv: s.conv,
                    lanes: s.bits.form.lanes,
                };
                match pass {
                    Pass::Load => {
                        let mem = crate::lower::sync::memory_model::coherence(cx.plan).texel(fb)?;
                        let texel = fb.read(img, pixel, &ImageOperands::default(), mem)?;
                        let texel = lanes(fb, texel, s.bits.form.lanes)?;
                        let held = held_texel(fb, s.class, form, texel)?;
                        cell_access(cx, fb, s.bits, off, Some(held))?;
                    }
                    Pass::Store => {
                        let Some(held) = cell_access(cx, fb, s.bits, off, None)? else {
                            return Err(cx.fail("a tile cell load without a value"));
                        };
                        let held = widen(fb, held, s)?;
                        let texel = to_image(
                            fb,
                            &image_decl(s),
                            TexelForm {
                                conv: s.conv,
                                lanes: 4,
                            },
                            held,
                        )?;
                        let mem = crate::lower::sync::memory_model::coherence(cx.plan).texel(fb)?;
                        fb.write(img, pixel, texel, &ImageOperands::default(), mem)?;
                    }
                }
            }
            Ok(ArmEnd::FallThrough(Vec::new()))
        });
        let zeroed = slots.clone();
        let otherwise: Option<ArmFn<'_, EmitCx<'c>>> = match pass {
            Pass::Store => None,
            Pass::Load => Some(Box::new(move |fb, cx| {
                for (s, _) in &zeroed {
                    let off = field(fb, cell, s.offset)?;
                    let z = leaf::zero(fb, &s.bits)?;
                    cell_access(cx, fb, s.bits, off, Some(z))?;
                }
                Ok(ArmEnd::FallThrough(Vec::new()))
            })),
        };
        fb.if_(cx, inside, then, otherwise)?;
        Ok(())
    });
    fb.loop_(cx, LoopSpec::default(), body)
}

/// A word of a cell, as the staging moves it.
const WORD: Leaf = Leaf {
    form: Form {
        held: Held::Int {
            bits: 32,
            container: 32,
        },
        lanes: 1,
    },
    bytes: 4,
    path: Path::Aligned,
};

/// Copy the `words` words of pixel `(px, py)`'s cell between the pass's imageblock storage `g`
/// (layer `w` holds word `w`) and the tile's cell at arena offset `cell`: into the cell
/// (`Pass::Load`), or back (`Pass::Store`), each texel with the function's coherence.
fn storage_words(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    pass: Pass,
    g: GlobalId,
    (px, py): (SValue, SValue),
    cell: SValue,
    words: u32,
) -> BResult<()> {
    let u = fb.module().ty(SType::Int {
        width: 32,
        signed: false,
    })?;
    let uvec3 = fb.module().ty(SType::Vector { elem: u, len: 3 })?;
    let uvec4 = fb.module().ty(SType::Vector { elem: u, len: 4 })?;
    let image = fb.module().global_value(g)?;
    let image = fb.load(image, MemAccess::default())?;
    for w in 0..words {
        let layer = leaf::k32(fb, w)?;
        let at = fb.composite_construct(uvec3, &[px, py, layer])?;
        let off = field(fb, cell, 4 * w)?;
        let mem = crate::lower::sync::memory_model::coherence(cx.plan).texel(fb)?;
        match pass {
            Pass::Load => {
                let texel = fb.read(image, at, &ImageOperands::default(), mem)?;
                let word = fb.composite_extract(texel, &[0])?;
                cell_access(cx, fb, WORD, off, Some(word))?;
            }
            Pass::Store => {
                let Some(word) = cell_access(cx, fb, WORD, off, None)? else {
                    return Err(cx.fail("a tile cell load without a value"));
                };
                let zero = leaf::k32(fb, 0)?;
                let texel = fb.composite_construct(uvec4, &[word, zero, zero, zero])?;
                fb.write(image, at, texel, &ImageOperands::default(), mem)?;
            }
        }
    }
    Ok(())
}

/// `cell + offset`.
fn field(fb: &mut FunctionBuilder<'_>, cell: SValue, offset: u32) -> BResult<SValue> {
    if offset == 0 {
        return Ok(cell);
    }
    let o = leaf::k32(fb, offset)?;
    fb.binary(BinOp::IAdd, cell, o)
}

/// The first `n` lanes of four-lane texel `t` (all of them for 4).
fn lanes(fb: &mut FunctionBuilder<'_>, t: SValue, n: u8) -> BResult<SValue> {
    match n {
        1 => fb.composite_extract(t, &[0]),
        4 => Ok(t),
        _ => {
            let comps: Vec<u32> = (0..u32::from(n)).collect();
            fb.vector_shuffle(t, t, &comps)
        }
    }
}

/// The four-lane held texel of the bits `v` of slot `s`'s lanes ([`TileSlot::bits`], `u32`
/// lanes), the lanes past them 0.
fn widen(fb: &mut FunctionBuilder<'_>, v: SValue, s: &TileSlot) -> BResult<SValue> {
    let n = s.bits.form.lanes;
    if n == 4 {
        return Ok(v);
    }
    let elem = fb.module().ty(SType::Int {
        width: 32,
        signed: false,
    })?;
    let ty = fb.module().ty(SType::Vector { elem, len: 4 })?;
    let z = fb
        .module()
        .constant(crate::spirv::ir::Constant::Null { ty: elem })?;
    let z = fb.module().const_value(z)?;
    let mut parts = Vec::with_capacity(4);
    for i in 0..u32::from(n) {
        parts.push(if n == 1 {
            v
        } else {
            fb.composite_extract(v, &[i])?
        });
    }
    while parts.len() < 4 {
        parts.push(z);
    }
    fb.composite_construct(ty, &parts)
}

/// Load (`stored` `None`) or store `leaf` at arena offset `off` of the imageblock region, which
/// the staging always reaches inside a cell.
fn cell_access(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    leaf: Leaf,
    off: SValue,
    stored: Option<SValue>,
) -> BResult<Option<SValue>> {
    let a = super::local::LocalAccess {
        arena: crate::memory::arena::Arena::Threadgroup,
        ptr: super::local::LocalPtr::Offset(0),
        value: None,
        leaf,
        contained: false,
        volatile: false,
    };
    super::local::emit_access_at(cx, fb, &a, off, stored)
}

/// The entry-wrapper hook of `lower::emit` for a tile function: the entry function of finished
/// function `body`, which is `body` itself unless the function stages an attachment or its tile's
/// threadgroup memory; then a new function `load; barrier; call body; barrier; store; return` (see
/// the module docs).
pub fn wrap_entry(cx: &mut EmitCx<'_>, mb: &mut ModuleBuilder, body: FuncId) -> BResult<FuncId> {
    let cells = stages(cx.plan.families.access.tile.as_ref());
    let memory = has_memory(&cx.plan.families.access);
    if !cells && !memory {
        return Ok(body);
    }
    let slots = slots_with_images(cx)?;
    let void = mb.ty(SType::Void)?;
    let mut fb = mb.function(void)?;
    if cells {
        cells_pass(cx, &mut fb, Pass::Load, &slots)?;
    }
    if memory {
        memory_pass(cx, &mut fb, Pass::Load)?;
    }
    barrier(cx, &mut fb)?;
    fb.call(body, &[])?;
    barrier(cx, &mut fb)?;
    if cells {
        cells_pass(cx, &mut fb, Pass::Store, &slots)?;
    }
    if memory {
        memory_pass(cx, &mut fb, Pass::Store)?;
    }
    fb.ret(None)?;
    fb.finish()
}

/// One pass over the tile's threadgroup memory: word `k = lin, lin + n, ...` below
/// `TileThreadgroupLength / 4`, `n` the threadgroup's threads, is copied from the pass's buffer
/// at word `t · TileThreadgroupLength / 4 + k` into arena word `k` (`Pass::Load`), or back
/// (`Pass::Store`), `t = x + y · TileGridWidth` the tile.
fn memory_pass<'c>(cx: &mut EmitCx<'c>, fb: &mut FunctionBuilder<'_>, pass: Pass) -> BResult<()> {
    let u = fb.module().ty(SType::Int {
        width: 32,
        signed: false,
    })?;
    let Some((view, grid)) = cx.decl.families.access.tile.memory else {
        return Err(cx.fail("a tile memory staging without its buffer"));
    };
    let arena = match cx.decl.families.access.local.tg.as_ref() {
        Some(t) => t.decl,
        None => return Err(cx.fail("a tile memory staging without its arena")),
    };
    let lin = builtin(cx, fb, KernelBuiltin::ThreadIndexInThreadgroup)?;
    let threads = running_threads(cx, fb)?;
    let tile = builtin(cx, fb, KernelBuiltin::ThreadgroupPositionInGrid)?;
    let tx = fb.composite_extract(threads, &[0])?;
    let ty_ = fb.composite_extract(threads, &[1])?;
    let tz = fb.composite_extract(threads, &[2])?;
    let txy = fb.binary(BinOp::IMul, tx, ty_)?;
    let n = fb.binary(BinOp::IMul, txy, tz)?;
    let length = crate::memory::arena::tile_length(fb.module())?;
    let length = fb.module().const_value(length)?;
    let two = leaf::k32(fb, 2)?;
    let words = fb.binary(BinOp::ShiftRightLogical, length, two)?;
    let grid = fb.module().const_value(grid)?;
    let gx = fb.composite_extract(tile, &[0])?;
    let gy = fb.composite_extract(tile, &[1])?;
    let row = fb.binary(BinOp::IMul, gy, grid)?;
    let index = fb.binary(BinOp::IAdd, row, gx)?;
    let base = fb.binary(BinOp::IMul, index, words)?;
    let coherence = crate::lower::sync::memory_model::coherence(cx.plan);
    let var = fb.local_var(u, None)?;
    let counter = fb.var_ptr(var)?;
    fb.store(counter, lin, MemAccess::default())?;
    let body: LoopFn<'_, EmitCx<'c>> = Box::new(move |fb, cx, lc| {
        let k = fb.load(counter, MemAccess::default())?;
        let done = fb.binary(BinOp::UGreaterThanEqual, k, words)?;
        let exit: ArmFn<'_, EmitCx<'c>> = Box::new(move |fb, _| {
            lc.break_(fb)?;
            Ok(ArmEnd::Exits)
        });
        fb.if_(cx, done, exit, None)?;
        let next = fb.binary(BinOp::IAdd, k, n)?;
        fb.store(counter, next, MemAccess::default())?;
        let at = fb.binary(BinOp::IAdd, base, k)?;
        let Some(stored) = crate::memory::views::word_ptr(fb, view, at)? else {
            return Err(cx.fail("a tile memory buffer that is not a storage view"));
        };
        let mut mem = ArenaMem(arena, WordOps::of(coherence));
        match pass {
            Pass::Load => {
                let access = coherence.access(fb, MemAccess::default(), false)?;
                let word = fb.load(stored, access)?;
                mem.write(fb, k, word)?;
            }
            Pass::Store => {
                let word = mem.read(fb, k)?;
                let access = coherence.access(fb, MemAccess::default(), true)?;
                fb.store(stored, word, access)?;
            }
        }
        Ok(())
    });
    fb.loop_(cx, LoopSpec::default(), body)
}

/// Emit an implicit imageblock access (`ImageblockStrategy::Texel`), defining `vid` for a load.
pub(super) fn emit_texel(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    s: &ImageblockStrategy,
) -> BResult<Option<SValue>> {
    let ImageblockStrategy::Texel {
        coord,
        cell_bytes,
        offset,
        leaf,
        value,
        contained,
        norm,
    } = *s
    else {
        return Err(cx.fail("a texel emit of another imageblock strategy"));
    };
    // The coordinate's lanes are held zero-extended (`ushort2`).
    let c = cx.operand(fb, coord)?;
    let x = fb.composite_extract(c, &[0])?;
    let y = fb.composite_extract(c, &[1])?;
    let cell = imageblock::cell_offset(cx, fb, x, y, cell_bytes)?;
    let off = field(fb, cell, offset)?;
    let stored = match value {
        Some(v) => Some(cx.operand(fb, v)?),
        None => None,
    };
    let a = super::local::LocalAccess {
        arena: crate::memory::arena::Arena::Threadgroup,
        ptr: super::local::LocalPtr::Offset(0),
        value: None,
        leaf,
        contained,
        volatile: false,
    };
    // A normalized attachment's cell holds its bits: a load unpacks them, a store packs.
    let Some(n) = norm else {
        return super::local::emit_access_at(cx, fb, &a, off, stored);
    };
    match stored {
        Some(v) => {
            let bits = pack_texel(fb, n, leaf.form, v)?;
            super::local::emit_access_at(cx, fb, &a, off, Some(bits))
        }
        None => {
            let Some(bits) = super::local::emit_access_at(cx, fb, &a, off, None)? else {
                return Err(cx.fail("an implicit imageblock load without a value"));
            };
            unpack_texel(fb, n, leaf.form, bits).map(Some)
        }
    }
}

/// The lanes access `a` reads of a normalized texel whose bits `bits` are of form `words` (one or
/// two `u32` words, or a 16-bit one zero-extended): each word unpacked by the device's conversion
/// of the format's pack function (`pack::unpack`), each lane rounded to `half` where the access's
/// lanes are (`narrow::round`), its channels in Metal's order, then green and blue 0 and alpha 1 up
/// to the access's lanes.
fn unpack_texel(
    fb: &mut FunctionBuilder<'_>,
    a: NormAccess,
    words: Form,
    bits: SValue,
) -> BResult<SValue> {
    let n = a.norm;
    let f32_ = Held::Float(FloatFmt::F32);
    let per = n.word_lanes();
    let word = Form {
        held: match a.half {
            true => Held::Float(FloatFmt::Half),
            false => f32_,
        },
        lanes: per,
    };
    let mut lanes = Vec::with_capacity(usize::from(words.lanes) * usize::from(per));
    for i in 0..words.lanes {
        let w = if words.lanes == 1 {
            bits
        } else {
            fb.composite_extract(bits, &[u32::from(i)])?
        };
        let v = pack::unpack(fb, n.format, word, w)?;
        for k in 0..per {
            lanes.push(fb.composite_extract(v, &[u32::from(k)])?);
        }
    }
    lanes.truncate(usize::from(n.channels));
    if n.bgra {
        lanes.swap(0, 2);
    }
    // The lanes past the channels: green and blue 0, alpha 1, each a `half` as it is a `float`.
    for i in n.channels..a.lanes {
        let x = if i == 3 { 1.0 } else { 0.0 };
        lanes.push(ops::fsplat(
            fb,
            Form {
                held: f32_,
                lanes: 1,
            },
            x,
        )?);
    }
    match lanes[..] {
        [one] => Ok(one),
        _ => {
            let t = ops::ty(
                fb,
                Form {
                    held: f32_,
                    lanes: a.lanes,
                },
            )?;
            fb.composite_construct(t, &lanes)
        }
    }
}

/// The bits, of form `words`, of the lanes `v` access `a` writes (`a.lanes` of them, in Metal's
/// channel order; `half` ones held as their `f32`, which Metal converts as that `float`, a probe):
/// its channels, then 0 in a word's lanes past them, each word packed by the device's conversion of
/// the format's pack function (`pack::pack`), whose lanes past a 16-bit word's are 0, so it is
/// zero-extended.
fn pack_texel(
    fb: &mut FunctionBuilder<'_>,
    a: NormAccess,
    words: Form,
    v: SValue,
) -> BResult<SValue> {
    let n = a.norm;
    let f32_ = Held::Float(FloatFmt::F32);
    let per = usize::from(n.word_lanes());
    let mut lanes = Vec::with_capacity(usize::from(words.lanes) * per);
    if a.lanes == 1 {
        lanes.push(v);
    } else {
        for k in 0..n.channels {
            lanes.push(fb.composite_extract(v, &[u32::from(k)])?);
        }
    }
    if n.bgra {
        lanes.swap(0, 2);
    }
    let zero = ops::fsplat(
        fb,
        Form {
            held: f32_,
            lanes: 1,
        },
        0.0,
    )?;
    lanes.resize(usize::from(words.lanes) * per, zero);
    let word = Form {
        held: f32_,
        lanes: n.word_lanes(),
    };
    let t = ops::ty(fb, word)?;
    let mut packed = Vec::with_capacity(usize::from(words.lanes));
    for chunk in lanes.chunks(per.max(1)) {
        let x = fb.composite_construct(t, chunk)?;
        packed.push(pack::pack(fb, n.format, word, x)?);
    }
    match packed[..] {
        [one] => Ok(one),
        _ => {
            let t = ops::ty(fb, words)?;
            fb.composite_construct(t, &packed)
        }
    }
}
