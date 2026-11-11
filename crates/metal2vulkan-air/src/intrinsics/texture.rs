//! Textures and samplers: `air.<op>_{texture,depth}_<dim>[.i16][.<s|u>].<type>` and
//! `air.get_read_sampler`.
//!
//! The operand layout of each operation is derived from the variant ([`Texture::layout`]); every
//! position carries a [`TexRole`]. The layouts are the ones Apple's compiler emits, read from the
//! declarations of the local corpus census and the public fixtures:
//!
//! - **sample** `{T, i8} (tex, smp, [depth: i32], coord, [array: layer], [compare: ref],
//!   [not cube: i1 offset?, offset], i1 level?, float bias-or-level, float min-lod-clamp,
//!   i32 access)`; the **grad** form replaces the three LOD positions by `dPdx, dPdy,
//!   float min-lod-clamp` after the coordinate, layer, and comparison reference, before the
//!   offset pair (which a cube has not).
//! - **gather** `{T, i8} (tex, smp, [depth: i32], coord, [array: layer], [compare: ref],
//!   [not cube: i1 offset?, offset], [texture: i32 component], i32 access)`; a comparison
//!   gather (`gather_compare`) names its depth texel `f32` and returns four of them.
//! - **read**, current form `{T, i8} (tex, smp, [depth: i32], coord, [cube: face], [array: layer],
//!   [2d/3d, not ms: offset], ms ? sample : lod, i32 access)`; the **legacy** form omits the
//!   sampler and offset positions. A texture buffer has no level: `(tex, [smp], coord,
//!   i32 access)`.
//! - a sample, gather, or read **states a residency status** ([`Texture::status`]) from AIR 2.2
//!   on: its result is `{T, i8}`, the texel and Metal's residency byte, and a sample carries the
//!   min-LOD-clamp position above. Every AIR up to 2.1 states neither: such a declaration returns
//!   the texel alone and its sample has no clamp position. The declaration picks the form, as it
//!   does for a read's or an atomic's operand positions, so the whole range below 2.2 needs no
//!   case of its own. The two changed in one emitter step, and no module has been seen with one
//!   without the other, so the mixtures stay unknown: a current-form read (with the sampler and
//!   offset) without the status is not a form of it.
//! - **write** `void (tex, coord, [cube: face], [array: layer], texel, [not buffer: lod],
//!   i32 access)`.
//! - **block_read** (Metal 4.1's `block_read<W, H>` and `sparse_block_read<W, H>` of a 2D, 2D
//!   array, or 2D depth texture) `void (tex, smp, <2 x i32> coord, [array: i32 layer],
//!   <2 x i32> offset, i32 lod, i32 access, ptr out, ptr status, <2 x i32> size)`: the `W x H`
//!   texels at `coord + offset` written to `out`, and their residency bytes to `status` where it
//!   is not null. Canonicalization expands each into reads (`canon::block_read`).
//! - a read's or write's texel is `f32` or `f16`, `s`/`u` `i32` or `i16`, or, for Metal's
//!   `ulong` textures (1D, 1D array, 2D, 2D array, 3D, and buffer), `u` `i64`.
//! - a `ulong` texture's atomics are `atomic_{min,max}_explicit` (no `fetch_`) of a `u` `v4i64`
//!   texel, which return nothing: `void (tex, coord, [array: layer], [2d/3d: offset], T value,
//!   i32 order, i32 access)`.
//! - **atomics** (`atomic_<op>_explicit`, on 1D, 1D array, 2D, 2D array, 3D, and buffer
//!   textures of `s`/`u` `v4i32` texels): `<op>` is `fetch_{add,sub,and,or,xor,min,max}` or
//!   `exchange`, `T (tex, coord, [array: layer], [2d/3d: offset], T value, i32 order,
//!   i32 access)`; `load` `T (tex, coord, [layer], [offset], i32 order, i32 access)`; `store`
//!   `void (tex, T value, coord, [layer], [offset], i32 order, i32 access)`; and
//!   `compare_exchange_weak` `i1 (tex, coord, [layer], [offset], ptr expected, T desired,
//!   i32 order, i32 failure order, i32 access)`, whose `expected` is a thread pointer to a `T`.
//!   The **legacy** form of a 2D or 3D one (a `ulong` one's too) omits the offset position: the
//!   AIR macOS 26.6's Metal on x86-64 hands its device for a library whose own declarations
//!   pass the offset.
//! - queries, null checks, fences, LOD queries, and the imageblock slice write as documented on
//!   [`TexOp`].
//!
//! `.i16` makes every integer coordinate-like position (coordinate, offset, layer, face, lod,
//! sample) 16-bit. Each operation admits only the dimensions whose declarations have been
//! observed; any other combination stays unknown rather than a guessed layout, because a known name
//! whose declaration disagrees is malformed input.
//!
//! Immediates read into the variant: the access qualifier (sample 0, read 1, write 2,
//! read_write 3), the sample LOD mode, the offset-present flag, and a gather's component. A
//! value outside those enumerations leaves the call unknown. Positions whose meaning is not
//! established (the depth word, the imageblock slice flags) are ordinary typed operands that
//! lowering must interpret or refuse.

use super::atomic::AtomicOp;
use super::layout::Layout;
use super::sig::{HandleKind, ImageDim, ImageShape, ImmKind, Immediates, Operand, Signature};
use super::ty::{Elem, NumTy, Sign};
use crate::error::IntrinsicError;
use crate::ir::AddrSpace;

/// A texture's access qualifier, as the trailing `i32` of most texture operations spells it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Access {
    /// `access::sample` (0).
    Sample,
    /// `access::read` (1).
    Read,
    /// `access::write` (2).
    Write,
    /// `access::read_write` (3).
    ReadWrite,
}

impl Access {
    fn from_raw(v: u64) -> Option<Access> {
        Some(match v {
            0 => Access::Sample,
            1 => Access::Read,
            2 => Access::Write,
            3 => Access::ReadWrite,
            _ => return None,
        })
    }
}

/// How a sample chooses its level: the `i1` before the LOD value.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum LodMode {
    /// `false`: implicit LOD with the value as a bias (a zero bias is Metal's default sample).
    Bias,
    /// `true`: the value is an explicit level (`level(n)`).
    Level,
}

/// A gather's component (`component::x` .. `w`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Component {
    /// `component::x` (0).
    X,
    /// `component::y` (1).
    Y,
    /// `component::z` (2).
    Z,
    /// `component::w` (3).
    W,
}

impl Component {
    fn from_raw(v: u64) -> Option<Component> {
        Some(match v {
            0 => Component::X,
            1 => Component::Y,
            2 => Component::Z,
            3 => Component::W,
            _ => return None,
        })
    }
}

/// A texture size query.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Query {
    /// `get_width`: the width in texels, of a level where the texture has levels.
    Width,
    /// `get_height`: the height in texels, of a level where the texture has levels.
    Height,
    /// `get_depth`: a 3D texture's depth in texels, of a level.
    Depth,
    /// `get_array_size`: the number of array layers.
    ArraySize,
    /// `get_num_mip_levels`: the number of mip levels.
    NumMipLevels,
    /// `get_num_samples`: a multisampled texture's samples per texel.
    NumSamples,
    /// `get_channel_order`: OpenCL C's `get_image_channel_order`, as Apple's OpenCL compiler
    /// writes it (`i32 (tex)`, macOS 27.0.1 and 26.7.1). A pipeline of OpenCL mode reads it from
    /// its third hidden buffer (`canon::opencl`); lowering refuses one any other pipeline keeps.
    /// Metal makes no pipeline of OpenCL mode that queries a texture buffer or a depth texture
    /// (measured on an M3 Max and a Radeon Pro 5300M), so those kinds stay undecoded.
    ChannelOrder,
    /// `get_channel_data_type`: OpenCL C's `get_image_channel_data_type`, as
    /// [`Query::ChannelOrder`].
    ChannelDataType,
}

/// A texture operation. The bound immediates live in the variant.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TexOp {
    /// `sample` (and, for a depth texture, `sample_compare` with `compare`).
    Sample {
        /// `sample_compare`: compares with the reference value (`TexRole::CompareRef`).
        compare: bool,
        /// How the LOD value is read: the call's LOD-mode immediate.
        lod: LodMode,
        /// The call's offset-present immediate: whether the offset applies (`false` for a cube).
        offset: bool,
    },
    /// `sample_{texture,depth}_<dim>_grad`: explicit gradients (and, for a depth texture,
    /// `sample_compare` with `compare`).
    SampleGrad {
        /// `sample_compare`: compares with the reference value (`TexRole::CompareRef`).
        compare: bool,
        /// The call's offset-present immediate: whether the offset applies (`false` for a cube).
        offset: bool,
    },
    /// `gather`; a depth gather has no component (and `gather_compare` compares).
    Gather {
        /// The gathered component, the call's immediate; `None` for a depth gather.
        component: Option<Component>,
        /// `gather_compare`: compares with the reference value (`TexRole::CompareRef`).
        compare: bool,
        /// The call's offset-present immediate: whether the offset applies (`false` for a cube).
        offset: bool,
    },
    /// `read`: one texel at integer coordinates, in the current or the legacy form.
    Read,
    /// `write`: stores one texel at integer coordinates.
    Write,
    /// `atomic_<op>_explicit`: a texture atomic, with the layouts of the module documentation.
    /// The operations are a memory atomic's ([`AtomicOp`]): `fetch_<op>` the read-modify-writes,
    /// `exchange` `Xchg`, and `compare_exchange_weak` `CmpXchgWeak`. A `ulong` texture's
    /// `atomic_{min,max}_explicit` is `Min` or `Max` of a 64-bit texel and returns nothing
    /// ([`Texture::ulong`]).
    Atomic(AtomicOp),
    /// `fence_texture`: `void (tex)`.
    Fence,
    /// `get_<query>`: `i32 (tex[, lod])`.
    Query(Query),
    /// `is_null`: `i1 (tex)`.
    IsNull,
    /// `get_null`: `tex ()`.
    GetNull,
    /// `calculate_{clamped,unclamped}_lod`: `float (tex, smp, coord, i32 access)`.
    CalculateLod {
        /// `calculate_clamped_lod` rather than `calculate_unclamped_lod`.
        clamped: bool,
    },
    /// `write_imageblock_slice_to_texture`: `void (tex, imageblock ptr, i1, <2 x i16>,
    /// <2 x i16>, coord, [array: layer], lod, i1, i32 access)`.
    WriteImageblockSlice,
    /// `block_read`: `void (tex, smp, coord, [array: layer], offset, lod, i32 access, ptr out,
    /// ptr status, <2 x i32> size)`, a block of texels written through `out` (module doc).
    BlockRead,
}

/// The texel type a texture operation's name spells.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Texel {
    pub ty: NumTy,
    /// The sign token (`s`/`u`) of an integer texel.
    pub sign: Option<Sign>,
}

/// The two operand forms of a texture read or a 2D or 3D texture atomic.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Form {
    /// The current form (a read's with a sampler and, where the shape has one, an offset
    /// position; an atomic's with its offset position).
    Current,
    /// The older form: a read's without the sampler and offset positions, an atomic's without
    /// the offset position.
    Legacy,
}

/// A decoded texture operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Texture {
    /// The operation, with the call's immediates bound.
    pub op: TexOp,
    /// The texture shape the name spells: its object kind and dimension token.
    pub image: ImageShape,
    /// The texel the name spells; `None` for the operations whose name spells none.
    pub texel: Option<Texel>,
    /// `.i16`: 16-bit integer coordinates, offsets, layers, faces, levels, and sample indices.
    pub coord16: bool,
    /// The operand form: a read's or a 2D or 3D atomic's current or legacy form; `Current` for
    /// every other operation.
    pub form: Form,
    /// Whether a sample, gather, or read states a residency status: its result is `{T, i8}` (the
    /// texel and Metal's residency byte) and a sample carries a min-LOD-clamp operand. AIR 2.2
    /// added both in one step; an AIR 2.1 declaration states neither, so its result is the texel
    /// alone and its sample has no clamp position. `true` for every operation whose result
    /// carries no status in either form (a write, a query, an atomic).
    pub status: bool,
    /// The access qualifier, for operations that carry one.
    pub access: Option<Access>,
    /// Whether the declaration states the trailing access-qualifier immediate. AIR 2.0 added it
    /// to every texture operation that has one; every emitter before it declares the same
    /// operation without it, and [`Texture::access`] is then `None`. `true` for an operation
    /// whose layout states none in either form.
    pub states_access: bool,
}

/// What one texture operand position is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TexRole {
    /// The texture handle.
    Texture,
    /// The sampler handle.
    Sampler,
    /// The `i32` a depth operation passes before its coordinate.
    DepthWord,
    /// The coordinate: `float` lanes for a sample, gather, or LOD query, integer lanes otherwise.
    Coord,
    /// The array layer.
    Layer,
    /// A cube texture's face, in a read or write.
    Face,
    /// A comparison's reference value (`float`).
    CompareRef,
    /// The `i1` immediate saying whether the offset applies.
    OffsetFlag,
    /// The texel offset.
    Offset,
    /// A sample's `i1` LOD-mode immediate (`LodMode`).
    LodMode,
    /// A sample's bias or level (float).
    LodValue,
    /// The minimum LOD clamp (`float`).
    MinLodClamp,
    /// A gradient sample's derivative of the coordinate along x.
    Dpdx,
    /// A gradient sample's derivative of the coordinate along y.
    Dpdy,
    /// A gather's component immediate (`i32`).
    Component,
    /// A read's or write's integer level.
    Lod,
    /// A multisampled read's sample index.
    SampleIndex,
    /// The value a write or atomic stores (a compare-exchange's desired value).
    Value,
    /// An atomic's memory order (`i32`), and a compare-exchange's failure order.
    Order,
    /// A compare-exchange's memory order on failure (`i32`).
    FailureOrder,
    /// A compare-exchange's `expected`: a thread pointer to the texel the comparison reads, which
    /// the old texel is written back through.
    Expected,
    /// The imageblock pointer a slice write copies from.
    Imageblock,
    /// The thread pointer a block read writes its texels through, in row order.
    BlockTexels,
    /// The thread pointer a block read writes its texels' residency bytes through, or null.
    BlockStatus,
    /// A block read's `<2 x i32>` width and height in texels.
    BlockSize,
    /// A position whose meaning is not modeled (the operand index within the operation).
    Opaque(u8),
    /// The access-qualifier immediate (`i32`).
    Access,
}

/// The operation part of a name, and the dimension tokens it admits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum OpName {
    Sample,
    SampleCompare,
    Gather,
    GatherCompare,
    Read,
    Write,
    Atomic(AtomicOp),
    /// A `ulong` texture's `atomic_{min,max}_explicit`, which returns nothing.
    UlongAtomic(AtomicOp),
    Fence,
    Query(Query),
    IsNull,
    GetNull,
    CalculateLod {
        clamped: bool,
    },
    WriteImageblockSlice,
    BlockRead,
}

/// Every operation stem, the object kinds (`texture`, `depth`) it takes, and the dimension
/// tokens observed for it. `None` admits every dimension the object kind has.
pub(super) const OPS: &[(&str, OpName, &[&str], &[&str])] = {
    use OpName::*;
    const ALL_T: &[&str] = &[
        "1d",
        "1d_array",
        "2d",
        "2d_array",
        "2d_ms",
        "2d_ms_array",
        "3d",
        "cube",
        "cube_array",
        "buffer_1d",
    ];
    const ALL_D: &[&str] = &[
        "2d",
        "2d_array",
        "2d_ms",
        "2d_ms_array",
        "cube",
        "cube_array",
    ];
    // A depth texture's sample and sample_compare, at a level or with gradients.
    const SAMPLED_D: &[&str] = &[
        "2d",
        "2d_array",
        "cube",
        "cube_array",
        "2d_grad",
        "2d_array_grad",
        "cube_grad",
        "cube_array_grad",
    ];
    const GATHERED_T: &[&str] = &["2d", "2d_array", "cube", "cube_array"];
    const GATHERED_D: &[&str] = &["2d", "2d_array", "cube", "cube_array"];
    // The LOD queries of the textures with levels and implicit derivatives.
    const LOD_T: &[&str] = &["2d", "2d_array", "3d", "cube", "cube_array"];
    const LOD_D: &[&str] = &["2d", "2d_array", "cube", "cube_array"];
    // The textures Metal's atomics take (Apple's metal 32023.921 declares no cube one).
    const ATOMIC_T: &[&str] = &["1d", "1d_array", "2d", "2d_array", "3d", "buffer_1d"];
    // The images whose format a pipeline of OpenCL mode answers (OpenCL C's image types but the
    // buffer and depth ones, whose queries Metal refuses).
    const CL_T: &[&str] = &["1d", "1d_array", "2d", "2d_array", "3d"];
    &[
        (
            "sample",
            Sample,
            &[
                "1d",
                "1d_array",
                "2d",
                "2d_array",
                "3d",
                "cube",
                "cube_array",
                "2d_grad",
                "2d_array_grad",
                "3d_grad",
                "cube_grad",
                "cube_array_grad",
            ],
            SAMPLED_D,
        ),
        ("sample_compare", SampleCompare, &[], SAMPLED_D),
        ("gather", Gather, GATHERED_T, GATHERED_D),
        ("gather_compare", GatherCompare, &[], GATHERED_D),
        (
            "read",
            Read,
            &[
                "1d",
                "1d_array",
                "2d",
                "2d_array",
                "2d_ms",
                "2d_ms_array",
                "3d",
                "cube",
                "cube_array",
                "buffer_1d",
            ],
            &[
                "2d",
                "2d_array",
                "2d_ms",
                "2d_ms_array",
                "cube",
                "cube_array",
            ],
        ),
        (
            "write",
            Write,
            &[
                "1d",
                "1d_array",
                "2d",
                "2d_array",
                "3d",
                "cube",
                "cube_array",
                "buffer_1d",
            ],
            &[],
        ),
        (
            "atomic_fetch_add_explicit",
            Atomic(AtomicOp::Add),
            ATOMIC_T,
            &[],
        ),
        (
            "atomic_fetch_sub_explicit",
            Atomic(AtomicOp::Sub),
            ATOMIC_T,
            &[],
        ),
        (
            "atomic_fetch_and_explicit",
            Atomic(AtomicOp::And),
            ATOMIC_T,
            &[],
        ),
        (
            "atomic_fetch_or_explicit",
            Atomic(AtomicOp::Or),
            ATOMIC_T,
            &[],
        ),
        (
            "atomic_fetch_xor_explicit",
            Atomic(AtomicOp::Xor),
            ATOMIC_T,
            &[],
        ),
        (
            "atomic_fetch_min_explicit",
            Atomic(AtomicOp::Min),
            ATOMIC_T,
            &[],
        ),
        (
            "atomic_fetch_max_explicit",
            Atomic(AtomicOp::Max),
            ATOMIC_T,
            &[],
        ),
        // A `ulong` texture's; Apple's compiler takes no other atomic of one.
        (
            "atomic_min_explicit",
            UlongAtomic(AtomicOp::Min),
            ATOMIC_T,
            &[],
        ),
        (
            "atomic_max_explicit",
            UlongAtomic(AtomicOp::Max),
            ATOMIC_T,
            &[],
        ),
        (
            "atomic_exchange_explicit",
            Atomic(AtomicOp::Xchg),
            ATOMIC_T,
            &[],
        ),
        (
            "atomic_load_explicit",
            Atomic(AtomicOp::Load),
            ATOMIC_T,
            &[],
        ),
        (
            "atomic_store_explicit",
            Atomic(AtomicOp::Store),
            ATOMIC_T,
            &[],
        ),
        (
            "atomic_compare_exchange_weak_explicit",
            Atomic(AtomicOp::CmpXchgWeak),
            ATOMIC_T,
            &[],
        ),
        ("fence", Fence, ALL_T, &[]),
        (
            "get_width",
            Query(super::texture::Query::Width),
            &[
                "1d",
                "1d_array",
                "2d",
                "2d_array",
                "2d_ms",
                "2d_ms_array",
                "3d",
                "cube",
                "cube_array",
                "buffer_1d",
            ],
            ALL_D,
        ),
        (
            "get_height",
            Query(super::texture::Query::Height),
            &[
                "2d",
                "2d_array",
                "2d_ms",
                "2d_ms_array",
                "3d",
                "cube",
                "cube_array",
            ],
            ALL_D,
        ),
        (
            "get_depth",
            Query(super::texture::Query::Depth),
            &["3d"],
            &[],
        ),
        (
            "get_array_size",
            Query(super::texture::Query::ArraySize),
            &["1d_array", "2d_array", "2d_ms_array", "cube_array"],
            &["2d_array", "2d_ms_array", "cube_array"],
        ),
        (
            "get_num_mip_levels",
            Query(super::texture::Query::NumMipLevels),
            &[
                "1d",
                "1d_array",
                "2d",
                "2d_array",
                "3d",
                "cube",
                "cube_array",
            ],
            &["2d", "2d_array", "cube", "cube_array"],
        ),
        (
            "get_num_samples",
            Query(super::texture::Query::NumSamples),
            &["2d_ms", "2d_ms_array"],
            &["2d_ms", "2d_ms_array"],
        ),
        // OpenCL C's image format queries (Apple's OpenCL compiler, macOS 27.0.1 and 26.7.1).
        (
            "get_channel_order",
            Query(super::texture::Query::ChannelOrder),
            CL_T,
            &[],
        ),
        (
            "get_channel_data_type",
            Query(super::texture::Query::ChannelDataType),
            CL_T,
            &[],
        ),
        ("is_null", IsNull, ALL_T, ALL_D),
        ("get_null", GetNull, ALL_T, ALL_D),
        (
            "calculate_clamped_lod",
            CalculateLod { clamped: true },
            LOD_T,
            LOD_D,
        ),
        (
            "calculate_unclamped_lod",
            CalculateLod { clamped: false },
            LOD_T,
            LOD_D,
        ),
        (
            "write_imageblock_slice_to",
            WriteImageblockSlice,
            &["2d", "2d_array"],
            &[],
        ),
        // Metal 4.1's (Apple's metal 32023.922 declares no 3D, cube, or depth array one).
        ("block_read", BlockRead, &["2d", "2d_array"], &["2d"]),
    ]
};

/// The shape a dimension token names, and whether it is the grad form.
fn dim(token: &str, depth: bool) -> Option<(ImageShape, bool)> {
    let (token, grad) = match token.strip_suffix("_grad") {
        Some(t) => (t, true),
        None => (token, false),
    };
    let (dim, array, ms) = match token {
        "1d" => (ImageDim::D1, false, false),
        "1d_array" => (ImageDim::D1, true, false),
        "2d" => (ImageDim::D2, false, false),
        "2d_array" => (ImageDim::D2, true, false),
        "2d_ms" => (ImageDim::D2, false, true),
        "2d_ms_array" => (ImageDim::D2, true, true),
        "3d" => (ImageDim::D3, false, false),
        "cube" => (ImageDim::Cube, false, false),
        "cube_array" => (ImageDim::Cube, true, false),
        "buffer_1d" => (ImageDim::Buffer, false, false),
        _ => return None,
    };
    Some((
        ImageShape {
            dim,
            array,
            ms,
            depth,
        },
        grad,
    ))
}

/// The number of coordinate components of `dim`, not counting array layers or cube faces:
/// `(sample coordinate lanes, integer coordinate lanes)`.
fn coord_lanes(dim: ImageDim) -> (u32, u32) {
    match dim {
        ImageDim::D1 | ImageDim::Buffer => (1, 1),
        ImageDim::D2 => (2, 2),
        ImageDim::D3 => (3, 3),
        // Sampling uses a direction; reads and writes address a face by (x, y) and a face index.
        ImageDim::Cube => (3, 2),
    }
}

fn num(elem: Elem, lanes: u32) -> Operand {
    Operand::Num(NumTy { elem, lanes })
}

impl Texture {
    /// Parse the part of a name after `air.`. Returns the current-form template; a read and a 2D
    /// or 3D atomic also have a [`Form::Legacy`] alternative ([`Texture::alternative`]).
    pub(super) fn parse(rest: &str) -> Option<Texture> {
        let (head, tail) = match rest.split_once('.') {
            Some((h, t)) => (h, Some(t)),
            None => (rest, None),
        };
        let (op_name, depth, dim_token) =
            OPS.iter().find_map(|&(stem, op, textures, depths)| {
                let after = head.strip_prefix(stem)?;
                if let Some(d) = after.strip_prefix("_texture_") {
                    textures.contains(&d).then_some((op, false, d))
                } else if let Some(d) = after.strip_prefix("_depth_") {
                    depths.contains(&d).then_some((op, true, d))
                } else {
                    None
                }
            })?;
        let (image, grad) = dim(dim_token, depth)?;
        let mut tokens = tail
            .map(|t| t.split('.').collect::<Vec<_>>())
            .unwrap_or_default();
        let coord16 = tokens.first() == Some(&"i16") && tokens.len() > 1;
        if coord16 {
            tokens.remove(0);
        }
        let texel = match tokens[..] {
            [] => None,
            [ty] => Some(Texel {
                ty: NumTy::parse(ty)?,
                sign: None,
            }),
            [sign, ty] => Some(Texel {
                ty: NumTy::parse(ty)?,
                sign: Some(Sign::parse(sign)?),
            }),
            _ => return None,
        };
        let op = match op_name {
            OpName::Sample | OpName::SampleCompare if grad => TexOp::SampleGrad {
                compare: op_name == OpName::SampleCompare,
                offset: false,
            },
            OpName::Sample | OpName::SampleCompare => TexOp::Sample {
                compare: op_name == OpName::SampleCompare,
                lod: LodMode::Bias,
                offset: false,
            },
            OpName::Gather | OpName::GatherCompare => TexOp::Gather {
                component: (!depth).then_some(Component::X),
                compare: op_name == OpName::GatherCompare,
                offset: false,
            },
            OpName::Read => TexOp::Read,
            OpName::Write => TexOp::Write,
            OpName::Atomic(a) | OpName::UlongAtomic(a) => TexOp::Atomic(a),
            OpName::Fence => TexOp::Fence,
            OpName::Query(q) => TexOp::Query(q),
            OpName::IsNull => TexOp::IsNull,
            OpName::GetNull => TexOp::GetNull,
            OpName::CalculateLod { clamped } => TexOp::CalculateLod { clamped },
            OpName::WriteImageblockSlice => TexOp::WriteImageblockSlice,
            OpName::BlockRead => TexOp::BlockRead,
        };
        let t = Texture {
            op,
            image,
            texel,
            coord16,
            form: Form::Current,
            status: true,
            access: None,
            states_access: true,
        };
        // The `ulong` stems take a `ulong` texel, and the others none.
        let ulong_stem = matches!(op_name, OpName::UlongAtomic(_));
        let ulong_atomic = matches!(t.op, TexOp::Atomic(_)) && t.ulong();
        (t.valid() && ulong_stem == ulong_atomic).then_some(t)
    }

    /// The next declaration of the same name a module may have, searched in order: a read's, or
    /// a 2D or 3D atomic's, legacy operand form; then, for an operation whose result carries a
    /// residency status, the AIR 2.1 form that states none; then the pre-AIR-2.0 form, which
    /// states no trailing access qualifier. A read has all four (AIR 2.8's sampler and offset,
    /// AIR 2.2's status, AIR 2.1's bare texel, and AIR 1.x's absent qualifier); a sample or
    /// gather has the status and qualifier axes, and an atomic the operand and qualifier axes.
    ///
    /// The chain is a list and not the product of the axes because each position implies the ones
    /// the emitter added before it: a read's sampler and offset appear only with the status, and
    /// the status only with the qualifier. So walking the axes newest first reaches every shape
    /// an emitter wrote and no other, and a new axis belongs at the head of the chain, not in the
    /// middle. The forms of one name must also stay distinguishable by function type alone --
    /// otherwise the first match would answer for both and the choice would be a guess -- which
    /// `families_tests::every_texture_form_is_recovered_from_its_own_declaration` holds over the
    /// grammar.
    pub(super) fn alternative(self) -> Option<Texture> {
        let legacy = match self.op {
            TexOp::Read => true,
            TexOp::Atomic(_) => matches!(self.image.dim, ImageDim::D2 | ImageDim::D3),
            _ => false,
        };
        if legacy && self.form == Form::Current {
            return Some(Texture {
                form: Form::Legacy,
                ..self
            });
        }
        if self.status && self.states_status() {
            return Some(Texture {
                status: false,
                ..self
            });
        }
        // The pre-AIR-2.0 form of the same declaration: no trailing access-qualifier immediate.
        // Reached only where the layout states one, so an operation that never carried it has no
        // such form; the earlier axes are exhausted first, so a read reaches its legacy operand
        // form and its bare-texel form before this one.
        (self.states_access && self.layout().index(TexRole::Access).is_some()).then_some(Texture {
            states_access: false,
            ..self
        })
    }

    /// Whether this operation's result carries a residency status in the current form: a sample,
    /// a gather, or a read ([`Texture::status`]).
    fn states_status(&self) -> bool {
        matches!(
            self.op,
            TexOp::Sample { .. } | TexOp::SampleGrad { .. } | TexOp::Gather { .. } | TexOp::Read
        )
    }

    /// Whether the texel is Metal's `ulong` (a `u` `i64`): a `ulong` atomic returns nothing.
    pub fn ulong(&self) -> bool {
        self.texel.is_some_and(|t| t.ty.elem == Elem::I64)
    }

    /// Whether the texel, `.i16`, and operation agree.
    fn valid(&self) -> bool {
        // Metal's `ulong` textures: unsigned 64-bit texels of 1D, 1D array, 2D, 2D array, 3D, and
        // buffer textures, which Apple's compiler reads, writes, and updates with
        // `atomic_{min,max}` only (it compiles no cube, multisampled, or depth `ulong` texture, and
        // no `long` one).
        let ulong_shape = !self.image.depth && !self.image.ms && self.image.dim != ImageDim::Cube;
        let ulong_op = matches!(
            self.op,
            TexOp::Read | TexOp::Write | TexOp::Atomic(AtomicOp::Min | AtomicOp::Max)
        );
        let texel_ok = |t: Texel, depth_ok: bool| {
            let elem_ok = match t.ty.elem {
                Elem::F16 | Elem::F32 => t.sign.is_none(),
                Elem::I16 | Elem::I32 => t.sign.is_some(),
                Elem::I64 => t.sign == Some(Sign::Unsigned) && ulong_shape && ulong_op,
                _ => false,
            };
            let shape_ok = if self.image.depth {
                depth_ok && t.ty.elem.is_float()
            } else {
                t.ty.lanes == 4
            };
            elem_ok && shape_ok
        };
        let lanes = self.texel.map_or(0, |t| t.ty.lanes);
        let spells_texel = match self.op {
            // A depth sample, read, and block read produce scalars, a depth gather four samples.
            TexOp::Sample { .. } | TexOp::SampleGrad { .. } | TexOp::Read | TexOp::BlockRead => {
                self.texel.is_some_and(|t| texel_ok(t, lanes == 1))
            }
            // A comparison gather names its scalar depth texel; it returns four.
            TexOp::Gather { compare, .. } => self
                .texel
                .is_some_and(|t| texel_ok(t, lanes == if compare { 1 } else { 4 })),
            TexOp::Write => self.texel.is_some_and(|t| texel_ok(t, false)),
            // Metal's texture atomics take 32-bit integer texels, and `ulong` ones.
            TexOp::Atomic(_) => self
                .texel
                .is_some_and(|t| texel_ok(t, false) && matches!(t.ty.elem, Elem::I32 | Elem::I64)),
            // The slice write names the imageblock's element type, not a texel.
            TexOp::WriteImageblockSlice => self.texel.is_some_and(|t| {
                matches!(t.ty.elem, Elem::F16 | Elem::F32 | Elem::I16 | Elem::I32)
                    && matches!(t.ty.lanes, 1 | 2 | 4)
                    && t.sign.is_none()
            }),
            _ => self.texel.is_none(),
        };
        let coord16_ok = !self.coord16
            || matches!(
                self.op,
                TexOp::Read | TexOp::Write | TexOp::Atomic(_) | TexOp::WriteImageblockSlice
            );
        spells_texel && coord16_ok
    }

    /// The operand layout.
    pub fn layout(&self) -> Layout<TexRole> {
        use TexRole as R;
        let shape = self.image;
        let tex = Operand::Handle(HandleKind::Texture(shape));
        let smp = Operand::Handle(HandleKind::Sampler);
        let i32s = num(Elem::I32, 1);
        let f32s = num(Elem::F32, 1);
        let flag = Operand::Imm(ImmKind::Bool);
        let access = Operand::Imm(ImmKind::I32);
        let ci = if self.coord16 { Elem::I16 } else { Elem::I32 };
        let (sample_lanes, int_lanes) = coord_lanes(shape.dim);
        let texel = self.texel.map_or(Operand::Void, |t| Operand::Num(t.ty));
        // The result of a sample, gather, or read: `{T, i8}` where the operation states a
        // residency status, the texel alone in the AIR 2.1 form that states none.
        let status = self.texel.map_or(Operand::Void, |t| {
            if self.status {
                Operand::WithStatus(t.ty)
            } else {
                Operand::Num(t.ty)
            }
        });
        let cube = shape.dim == ImageDim::Cube;
        let mut l = match self.op {
            TexOp::Gather { compare: true, .. } => {
                Layout::new(self.texel.map_or(Operand::Void, |t| {
                    let four = NumTy {
                        elem: t.ty.elem,
                        lanes: 4,
                    };
                    if self.status {
                        Operand::WithStatus(four)
                    } else {
                        Operand::Num(four)
                    }
                }))
            }
            TexOp::Sample { .. } | TexOp::SampleGrad { .. } | TexOp::Gather { .. } => {
                Layout::new(status)
            }
            TexOp::Read => Layout::new(status),
            TexOp::Write | TexOp::Fence | TexOp::WriteImageblockSlice | TexOp::BlockRead => {
                Layout::new(Operand::Void)
            }
            TexOp::Atomic(AtomicOp::Store) => Layout::new(Operand::Void),
            TexOp::Atomic(_) if self.ulong() => Layout::new(Operand::Void),
            TexOp::Atomic(AtomicOp::CmpXchgWeak) => Layout::new(num(Elem::I1, 1)),
            TexOp::Atomic(_) => Layout::new(texel),
            TexOp::Query(_) => Layout::new(i32s),
            TexOp::IsNull => Layout::new(num(Elem::I1, 1)),
            TexOp::GetNull => Layout::new(tex),
            TexOp::CalculateLod { .. } => Layout::new(f32s),
        };
        match self.op {
            TexOp::Sample { compare, .. } => {
                l.push(R::Texture, tex);
                l.push(R::Sampler, smp);
                l.push_if(shape.depth, R::DepthWord, i32s);
                l.push(R::Coord, num(Elem::F32, sample_lanes));
                l.push_if(shape.array, R::Layer, i32s);
                l.push_if(compare, R::CompareRef, f32s);
                if !cube {
                    l.push(R::OffsetFlag, flag);
                    l.push(R::Offset, num(Elem::I32, sample_lanes));
                }
                l.push(R::LodMode, flag);
                l.push(R::LodValue, f32s);
                l.push_if(self.status, R::MinLodClamp, f32s);
                l.push_if(self.states_access, R::Access, access);
            }
            TexOp::SampleGrad { compare, .. } => {
                let grad = num(Elem::F32, sample_lanes);
                l.push(R::Texture, tex);
                l.push(R::Sampler, smp);
                l.push_if(shape.depth, R::DepthWord, i32s);
                l.push(R::Coord, grad);
                l.push_if(shape.array, R::Layer, i32s);
                l.push_if(compare, R::CompareRef, f32s);
                l.push(R::Dpdx, grad);
                l.push(R::Dpdy, grad);
                l.push_if(self.status, R::MinLodClamp, f32s);
                if !cube {
                    l.push(R::OffsetFlag, flag);
                    l.push(R::Offset, num(Elem::I32, sample_lanes));
                }
                l.push_if(self.states_access, R::Access, access);
            }
            TexOp::Gather {
                component, compare, ..
            } => {
                l.push(R::Texture, tex);
                l.push(R::Sampler, smp);
                l.push_if(shape.depth, R::DepthWord, i32s);
                l.push(R::Coord, num(Elem::F32, sample_lanes));
                l.push_if(shape.array, R::Layer, i32s);
                l.push_if(compare, R::CompareRef, f32s);
                if !cube {
                    l.push(R::OffsetFlag, flag);
                    l.push(R::Offset, num(Elem::I32, sample_lanes));
                }
                l.push_if(component.is_some(), R::Component, access);
                l.push_if(self.states_access, R::Access, access);
            }
            TexOp::Read => {
                let current = self.form == Form::Current;
                l.push(R::Texture, tex);
                l.push_if(current, R::Sampler, smp);
                l.push_if(shape.depth, R::DepthWord, i32s);
                l.push(R::Coord, num(ci, int_lanes));
                // A cube array's face precedes its array index (MSL `read(coord, face, array)`).
                l.push_if(cube, R::Face, num(ci, 1));
                l.push_if(shape.array, R::Layer, num(ci, 1));
                let has_offset = matches!(shape.dim, ImageDim::D2 | ImageDim::D3) && !shape.ms;
                l.push_if(current && has_offset, R::Offset, num(ci, int_lanes));
                if shape.ms {
                    l.push(R::SampleIndex, num(ci, 1));
                } else if shape.dim != ImageDim::Buffer {
                    l.push(R::Lod, num(ci, 1));
                }
                l.push_if(self.states_access, R::Access, access);
            }
            TexOp::Write => {
                l.push(R::Texture, tex);
                l.push(R::Coord, num(ci, int_lanes));
                // As in a read, a cube array's face precedes its array index (MSL `write(color,
                // coord, face, array, lod)`; declaration observed from Apple's compiler).
                l.push_if(cube, R::Face, num(ci, 1));
                l.push_if(shape.array, R::Layer, num(ci, 1));
                l.push(R::Value, texel);
                l.push_if(shape.dim != ImageDim::Buffer, R::Lod, num(ci, 1));
                l.push_if(self.states_access, R::Access, access);
            }
            TexOp::Atomic(op) => {
                let has_offset =
                    matches!(shape.dim, ImageDim::D2 | ImageDim::D3) && self.form == Form::Current;
                l.push(R::Texture, tex);
                // A store passes its value first (MSL `atomic_store(value, coord)`).
                l.push_if(op == AtomicOp::Store, R::Value, texel);
                l.push(R::Coord, num(ci, int_lanes));
                l.push_if(shape.array, R::Layer, num(ci, 1));
                l.push_if(has_offset, R::Offset, num(ci, int_lanes));
                l.push_if(
                    op == AtomicOp::CmpXchgWeak,
                    R::Expected,
                    Operand::Ptr(AddrSpace::Thread),
                );
                l.push_if(
                    !matches!(op, AtomicOp::Load | AtomicOp::Store),
                    R::Value,
                    texel,
                );
                l.push(R::Order, i32s);
                l.push_if(op == AtomicOp::CmpXchgWeak, R::FailureOrder, i32s);
                l.push_if(self.states_access, R::Access, access);
            }
            TexOp::Fence | TexOp::IsNull => l.push(R::Texture, tex),
            TexOp::Query(q) => {
                l.push(R::Texture, tex);
                // A multisampled texture and a texture buffer have one level: their size queries
                // take none (`texture_buffer::get_width()`).
                let per_level = matches!(q, Query::Width | Query::Height | Query::Depth);
                let levels = !shape.ms && shape.dim != ImageDim::Buffer;
                l.push_if(per_level && levels, R::Lod, i32s);
            }
            TexOp::GetNull => {}
            TexOp::CalculateLod { .. } => {
                l.push(R::Texture, tex);
                l.push(R::Sampler, smp);
                l.push(R::Coord, num(Elem::F32, sample_lanes));
                l.push_if(self.states_access, R::Access, access);
            }
            TexOp::WriteImageblockSlice => {
                let i16v2 = num(Elem::I16, 2);
                l.push(R::Texture, tex);
                l.push(
                    R::Imageblock,
                    Operand::Ptr(AddrSpace::ThreadgroupImageblock),
                );
                l.push(R::Opaque(2), num(Elem::I1, 1));
                l.push(R::Opaque(3), i16v2);
                l.push(R::Opaque(4), i16v2);
                l.push(R::Coord, num(ci, int_lanes));
                l.push_if(shape.array, R::Layer, num(ci, 1));
                l.push(R::Lod, num(ci, 1));
                l.push(R::Opaque(8), num(Elem::I1, 1));
                l.push_if(self.states_access, R::Access, access);
            }
            TexOp::BlockRead => {
                let i32v2 = num(Elem::I32, 2);
                l.push(R::Texture, tex);
                l.push(R::Sampler, smp);
                l.push(R::Coord, i32v2);
                l.push_if(shape.array, R::Layer, i32s);
                l.push(R::Offset, i32v2);
                l.push(R::Lod, i32s);
                l.push_if(self.states_access, R::Access, access);
                l.push(R::BlockTexels, Operand::Ptr(AddrSpace::Thread));
                l.push(R::BlockStatus, Operand::Ptr(AddrSpace::Thread));
                l.push(R::BlockSize, i32v2);
            }
        }
        l
    }

    /// The typed signature.
    pub fn signature(&self) -> Signature {
        self.layout().signature()
    }

    /// Read the call's immediates. `Ok(None)`: a value outside the modeled enumerations.
    pub(super) fn bind(self, imm: &Immediates<'_>) -> Result<Option<Texture>, IntrinsicError> {
        let layout = self.layout();
        let at = |role| layout.index(role);
        let flag = |role| at(role).map(|i| imm.flag(i)).transpose();
        let access = match at(TexRole::Access) {
            Some(i) => match Access::from_raw(imm.get(i)?) {
                Some(a) => Some(a),
                None => return Ok(None),
            },
            None => None,
        };
        let offset = flag(TexRole::OffsetFlag)?.unwrap_or(false);
        let op = match self.op {
            TexOp::Sample { compare, .. } => TexOp::Sample {
                compare,
                lod: if flag(TexRole::LodMode)?.unwrap_or(false) {
                    LodMode::Level
                } else {
                    LodMode::Bias
                },
                offset,
            },
            TexOp::SampleGrad { compare, .. } => TexOp::SampleGrad { compare, offset },
            TexOp::Gather {
                component, compare, ..
            } => {
                let component = match (component, at(TexRole::Component)) {
                    (Some(_), Some(i)) => match Component::from_raw(imm.get(i)?) {
                        Some(c) => Some(c),
                        None => return Ok(None),
                    },
                    _ => None,
                };
                TexOp::Gather {
                    component,
                    compare,
                    offset,
                }
            }
            other => other,
        };
        Ok(Some(Texture { op, access, ..self }))
    }
}

/// A sampler operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SamplerOp {
    /// `air.get_read_sampler`: `smp ()`, the sampler a sampled texture's `read` uses.
    GetReadSampler,
}

impl SamplerOp {
    pub(super) fn parse(rest: &str) -> Option<SamplerOp> {
        (rest == "get_read_sampler").then_some(SamplerOp::GetReadSampler)
    }

    /// The typed signature.
    pub fn signature(&self) -> Signature {
        match self {
            SamplerOp::GetReadSampler => Signature::new(Operand::Handle(HandleKind::Sampler), &[]),
        }
    }
}
