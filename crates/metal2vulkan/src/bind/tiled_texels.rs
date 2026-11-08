//! A heap texture's texels moved between the image that holds them and Metal's tiled layout of
//! them in the heap's bytes.
//!
//! On Apple's GPUs a texture placed in a heap keeps its elements (texels, or blocks) in tiles: a
//! level's elements lie in tiles of `tw x th` elements (powers of two), the tiles row after row,
//! `across` of them a row, and the elements of a tile in Morton order, x's bit lowest, the longer
//! side's extra bits on top. Each slice of the texture (each depth slice of a 3D texture) holds
//! its own chain of levels, the chains a stride apart. A consumer keeps the texture's image a
//! shadow of those bytes: it copies the texture's bytes into a buffer (the tiled buffer), one
//! dispatch of [`tiled_texels_module`] moves them into a tight buffer of the image's elements,
//! which it copies into the image; and back: the heap's bytes into the tiled buffer again (so
//! the bytes between the elements are the heap's), the image into the tight buffer, a dispatch
//! into the tiled buffer, and each level's bytes on into the heap.
//!
//! **The module** (one per device). One `GLCompute` entry `main` with local size
//! ([`TiledTexelsModule::LOCAL_SIZE`], 1, 1), no push constants, and three storage buffers of
//! 4-byte words in set 0:
//!
//! | Binding | Buffer |
//! |---|---|
//! | [`TiledTexelsModule::TIGHT_BINDING`] | the image's elements, each level's from its `tight_first` byte, row after row, image after image |
//! | [`TiledTexelsModule::TILED_BINDING`] | the elements in Metal's tiled layout |
//! | [`TiledTexelsModule::SHAPE_BINDING`] | the move's shape, [`TiledTexelsModule::SHAPE_WORDS`] words |
//!
//! The shape ([`TiledTexelsModule::shape`]): the grid's row width, the direction, and up to
//! [`TiledTexelsModule::MAX_LEVELS`] levels ([`TiledLevelMove`]), each with the first invocation
//! that moves it and how many do. Invocation `(gx, gy)` takes `i = gy · grid_width + gx`, the
//! level whose invocations hold `i`, and `j`, `i` less that level's first:
//!
//! - **Into the image** (tight), word `j` of the level's tight elements: each of its bytes the
//!   byte of the tiled buffer its element and byte place lie at; a byte past the level's
//!   elements 0, which no consumer copies on. A level whose elements are 32-bit float depths
//!   clamped into the image ([`TiledLevelMove::clamps_depth`]) takes each in `[0, 1]`: a
//!   negative one 0.0, one above 1.0 1.0, a NaN 1.0, and the others (`-0.0` among them) bit for
//!   bit, compared as the word's bits, so no denormal is flushed.
//! - **Back** (tiled), word `j` of the level's regions in the tiled buffer (one region a chain,
//!   `region_bytes` long, from `tiled_first` past the chain's first byte): each byte an element's
//!   lies in, that element's byte from the tight buffer; every other byte as it was.
//!
//! One past every level moves nothing. The consumer keeps every byte the shape names inside the
//! bindings, each binding within `maxStorageBufferRange`, and the grid within
//! `maxComputeWorkGroupCount`.
//!
//! **Synchronization** (consumer): the writes of both buffers and the shape (`TRANSFER_WRITE`)
//! → `SHADER_STORAGE_READ | SHADER_STORAGE_WRITE` in `COMPUTE_SHADER` before the dispatch; its
//! writes → `TRANSFER_READ` before the tight or tiled buffer is copied on.

use std::sync::Arc;

use crate::budget::{Budget, Meter};
use crate::device::requirements::Requirements;
use crate::device::Device;
use crate::error::{Error, InternalError, InternalKind};
use crate::pipeline::sym::{ResourceKind, ResourceSlot, Stage, Symbol};
use crate::spirv::builder::constructs::{ArmEnd, ArmFn};
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
use crate::spirv::error::BResult;
use crate::spirv::ids::{GlobalId, TypeId, Value};
use crate::spirv::ir::{
    natural_layout, BinOp, ExecMode, ExecModel, IoDir, LayoutRule, MemAccess, StorageClass, Type,
    TypedModule,
};

/// The crate-owned tiled texel module of one device, validated like any output.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TiledTexelsModule {
    /// SPIR-V 1.5 words, entry point `main`, bindings numbered as below in set 0.
    pub spirv: Arc<[u32]>,
    /// What the module needs from the device (within every Vulkan 1.2 device).
    pub requirements: Requirements,
}

/// One level of a move between the tight and the tiled buffer ([`TiledTexelsModule::shape`]).
/// Every byte offset and length is a multiple of 4.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TiledLevelMove {
    /// The bytes of an element: 1, 2, 4, 8 or 16.
    pub unit: u32,
    /// Its elements across and down.
    pub extent: [u32; 2],
    /// Its images: the chains it lies in (a texture's slices, or a 3D level's depth slices).
    pub images: u32,
    /// Its tile, elements across and down: powers of two.
    pub tile: [u32; 2],
    /// Its tiles a row.
    pub across: u32,
    /// Its first byte in the tiled buffer, in the first chain.
    pub tiled_first: u32,
    /// The bytes from one chain to the next in the tiled buffer.
    pub chain_stride: u32,
    /// Its first byte in the tight buffer, where its images lie one after another, each its rows
    /// of elements.
    pub tight_first: u32,
    /// The bytes of its region in one chain of the tiled buffer, which the move back writes.
    pub region_bytes: u32,
    /// Whether its elements are 32-bit float depths, of 4 bytes, that the move into the image
    /// clamps into `[0, 1]` (the module docs): for a depth image of a device without
    /// `VK_EXT_depth_range_unrestricted`, whose copy from a buffer takes no value outside
    /// (`VUID-vkCmdCopyBufferToImage-pRegions-07931`). The move back takes them as they are.
    pub clamps_depth: bool,
}

impl TiledLevelMove {
    /// A level of `extent` elements of `unit` bytes in `images` images, tiled in tiles of `tile`
    /// elements `across` a row, from `tiled_first` of the tiled buffer's first chain in a region
    /// of `region_bytes`, chains `chain_stride` apart, and from `tight_first` of the tight
    /// buffer.
    pub const fn new(
        unit: u32,
        extent: [u32; 2],
        images: u32,
        tile: [u32; 2],
        across: u32,
        tiled_first: u32,
        chain_stride: u32,
        tight_first: u32,
        region_bytes: u32,
    ) -> TiledLevelMove {
        TiledLevelMove {
            unit,
            extent,
            images,
            tile,
            across,
            tiled_first,
            chain_stride,
            tight_first,
            region_bytes,
            clamps_depth: false,
        }
    }

    /// The level, its elements 32-bit float depths that the move into the image clamps into
    /// `[0, 1]` ([`clamps_depth`](Self::clamps_depth)).
    pub const fn clamping_depth(self) -> TiledLevelMove {
        TiledLevelMove {
            clamps_depth: true,
            ..self
        }
    }

    /// The invocations that move the level: a word of its tight elements each, into the image;
    /// a word of its tiled regions each, back.
    pub const fn invocations(&self, into_tight: bool) -> u32 {
        if into_tight {
            let bytes = self.extent[0] * self.extent[1] * self.images * self.unit;
            bytes.div_ceil(4)
        } else {
            self.images * (self.region_bytes / 4)
        }
    }
}

impl TiledTexelsModule {
    /// Descriptor set of every binding.
    pub const SET: u32 = 0;
    /// The image's elements: a storage buffer.
    pub const TIGHT_BINDING: u32 = 0;
    /// The elements in Metal's tiled layout: a storage buffer.
    pub const TILED_BINDING: u32 = 1;
    /// The move's shape: a storage buffer of [`SHAPE_WORDS`](Self::SHAPE_WORDS) words.
    pub const SHAPE_BINDING: u32 = 2;
    /// Invocations of one workgroup, along x.
    pub const LOCAL_SIZE: u32 = 64;
    /// The most levels one move takes.
    pub const MAX_LEVELS: u32 = 16;
    /// The words of a level's record in the shape.
    const LEVEL_WORDS: u32 = 14;
    /// The words of the shape's header, before the levels' records.
    const HEADER_WORDS: u32 = 4;
    /// The words of the shape.
    pub const SHAPE_WORDS: u32 = Self::HEADER_WORDS + Self::LEVEL_WORDS * Self::MAX_LEVELS;

    /// The invocations of a move of `levels`, into the image where `into_tight`.
    pub fn invocations(levels: &[TiledLevelMove], into_tight: bool) -> u32 {
        levels
            .iter()
            .map(|level| level.invocations(into_tight))
            .sum()
    }

    /// The shape's bytes of a move of `levels` into the image (the tight buffer) where
    /// `into_tight`, else back, in a grid whose rows are `grid_width` invocations wide; `None`
    /// past [`MAX_LEVELS`](Self::MAX_LEVELS) levels.
    pub fn shape(levels: &[TiledLevelMove], into_tight: bool, grid_width: u32) -> Option<Vec<u8>> {
        if levels.len() > Self::MAX_LEVELS as usize {
            return None;
        }
        let mut words = vec![0u32; Self::SHAPE_WORDS as usize];
        words[0] = grid_width;
        words[1] = u32::from(into_tight);
        words[2] = levels.len() as u32;
        let mut start = 0u32;
        for (k, level) in levels.iter().enumerate() {
            let at = (Self::HEADER_WORDS + Self::LEVEL_WORDS * k as u32) as usize;
            let count = level.invocations(into_tight);
            let record = [
                start,
                count,
                level.unit.trailing_zeros(),
                level.extent[0],
                level.extent[1],
                level.tile[0].trailing_zeros(),
                level.tile[1].trailing_zeros(),
                level.across,
                level.tiled_first,
                level.chain_stride,
                level.tight_first,
                level.region_bytes / 4,
                level.images,
                u32::from(level.clamps_depth),
            ];
            words[at..at + record.len()].copy_from_slice(&record);
            start += count;
        }
        Some(words.iter().flat_map(|word| word.to_le_bytes()).collect())
    }
}

/// The binding symbol of the tight buffer in the typed module.
const TIGHT: ResourceSlot = ResourceSlot {
    stage: Stage::Kernel,
    kind: ResourceKind::Buffer,
    index: 0,
};
/// The binding symbol of the tiled buffer in the typed module.
const TILED: ResourceSlot = ResourceSlot {
    stage: Stage::Kernel,
    kind: ResourceKind::Buffer,
    index: 1,
};
/// The binding symbol of the shape in the typed module.
const SHAPE: ResourceSlot = ResourceSlot {
    stage: Stage::Kernel,
    kind: ResourceKind::Buffer,
    index: 2,
};

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

/// The fields of the moving level's record, loaded once.
#[derive(Clone, Copy)]
struct Level {
    unit_log2: Value,
    width: Value,
    height: Value,
    tile_log2: [Value; 2],
    across: Value,
    tiled_first: Value,
    chain_stride: Value,
    tight_first: Value,
    region_words: Value,
    images: Value,
    clamps_depth: Value,
}

/// `v`'s low 16 bits spread to the even bits.
fn spread(f: &mut FunctionBuilder<'_>, v: Value) -> BResult<Value> {
    let mut v = v;
    for (shift, mask) in [
        (8u32, 0x00ff_00ffu32),
        (4, 0x0f0f_0f0f),
        (2, 0x3333_3333),
        (1, 0x5555_5555),
    ] {
        let s = u32_value(f, shift)?;
        let m = u32_value(f, mask)?;
        let shifted = f.binary(BinOp::ShiftLeftLogical, v, s)?;
        let both = f.binary(BinOp::BitwiseOr, v, shifted)?;
        v = f.binary(BinOp::BitwiseAnd, both, m)?;
    }
    Ok(v)
}

/// `v`'s even bits compacted to its low 16.
fn compact(f: &mut FunctionBuilder<'_>, v: Value) -> BResult<Value> {
    let even = u32_value(f, 0x5555_5555)?;
    let mut v = f.binary(BinOp::BitwiseAnd, v, even)?;
    for (shift, mask) in [
        (1u32, 0x3333_3333u32),
        (2, 0x0f0f_0f0f),
        (4, 0x00ff_00ff),
        (8, 0x0000_ffff),
    ] {
        let s = u32_value(f, shift)?;
        let m = u32_value(f, mask)?;
        let shifted = f.binary(BinOp::ShiftRightLogical, v, s)?;
        let both = f.binary(BinOp::BitwiseOr, v, shifted)?;
        v = f.binary(BinOp::BitwiseAnd, both, m)?;
    }
    Ok(v)
}

/// The tile's shorter side's bits, and whether x's side is the longer.
fn sides(f: &mut FunctionBuilder<'_>, l: Level) -> BResult<(Value, Value)> {
    let [tx, ty] = l.tile_log2;
    let x_longer = f.binary(BinOp::UGreaterThan, tx, ty)?;
    let short = f.select(x_longer, ty, tx)?;
    Ok((short, x_longer))
}

/// `(1 << bits) - 1`.
fn low_mask(f: &mut FunctionBuilder<'_>, bits: Value) -> BResult<Value> {
    let one = u32_value(f, 1)?;
    let bit = f.binary(BinOp::ShiftLeftLogical, one, bits)?;
    f.binary(BinOp::ISub, bit, one)
}

/// The Morton index of the element at `(x, y)` of its tile.
fn morton(f: &mut FunctionBuilder<'_>, l: Level, x: Value, y: Value) -> BResult<Value> {
    let (short, x_longer) = sides(f, l)?;
    let mask = low_mask(f, short)?;
    let xl = f.binary(BinOp::BitwiseAnd, x, mask)?;
    let yl = f.binary(BinOp::BitwiseAnd, y, mask)?;
    let mx = spread(f, xl)?;
    let my = spread(f, yl)?;
    let one = u32_value(f, 1)?;
    let my = f.binary(BinOp::ShiftLeftLogical, my, one)?;
    let low = f.binary(BinOp::BitwiseOr, mx, my)?;
    let long = f.select(x_longer, x, y)?;
    let high = f.binary(BinOp::ShiftRightLogical, long, short)?;
    let both = f.binary(BinOp::IAdd, short, short)?;
    let high = f.binary(BinOp::ShiftLeftLogical, high, both)?;
    f.binary(BinOp::BitwiseOr, low, high)
}

/// The element `(x, y)` of its tile at Morton index `m`.
fn demorton(f: &mut FunctionBuilder<'_>, l: Level, m: Value) -> BResult<(Value, Value)> {
    let (short, x_longer) = sides(f, l)?;
    let both = f.binary(BinOp::IAdd, short, short)?;
    let mask = low_mask(f, both)?;
    let low = f.binary(BinOp::BitwiseAnd, m, mask)?;
    let one = u32_value(f, 1)?;
    let x = compact(f, low)?;
    let odd = f.binary(BinOp::ShiftRightLogical, low, one)?;
    let y = compact(f, odd)?;
    let high = f.binary(BinOp::ShiftRightLogical, m, both)?;
    let high = f.binary(BinOp::ShiftLeftLogical, high, short)?;
    let zero = u32_value(f, 0)?;
    let hx = f.select(x_longer, high, zero)?;
    let hy = f.select(x_longer, zero, high)?;
    let x = f.binary(BinOp::BitwiseOr, x, hx)?;
    let y = f.binary(BinOp::BitwiseOr, y, hy)?;
    Ok((x, y))
}

/// The byte of the tiled buffer at which the element `(x, y)` of image `image` starts.
fn tiled_byte(
    f: &mut FunctionBuilder<'_>,
    l: Level,
    image: Value,
    x: Value,
    y: Value,
) -> BResult<Value> {
    let [tx, ty] = l.tile_log2;
    let row = f.binary(BinOp::ShiftRightLogical, y, ty)?;
    let column = f.binary(BinOp::ShiftRightLogical, x, tx)?;
    let tile = f.binary(BinOp::IMul, row, l.across)?;
    let tile = f.binary(BinOp::IAdd, tile, column)?;
    let tile_bits = f.binary(BinOp::IAdd, tx, ty)?;
    let tile = f.binary(BinOp::ShiftLeftLogical, tile, tile_bits)?;
    let xm = low_mask(f, tx)?;
    let ym = low_mask(f, ty)?;
    let xi = f.binary(BinOp::BitwiseAnd, x, xm)?;
    let yi = f.binary(BinOp::BitwiseAnd, y, ym)?;
    let m = morton(f, l, xi, yi)?;
    let element = f.binary(BinOp::IAdd, tile, m)?;
    let element = f.binary(BinOp::ShiftLeftLogical, element, l.unit_log2)?;
    let chain = f.binary(BinOp::IMul, image, l.chain_stride)?;
    let first = f.binary(BinOp::IAdd, chain, l.tiled_first)?;
    f.binary(BinOp::IAdd, first, element)
}

/// The image, x and y of tight element `e` of the level (counted row after row, image after
/// image).
fn tight_place(f: &mut FunctionBuilder<'_>, l: Level, e: Value) -> BResult<[Value; 3]> {
    let plane = f.binary(BinOp::IMul, l.width, l.height)?;
    let image = f.binary(BinOp::UDiv, e, plane)?;
    let r = f.binary(BinOp::UMod, e, plane)?;
    let y = f.binary(BinOp::UDiv, r, l.width)?;
    let x = f.binary(BinOp::UMod, r, l.width)?;
    Ok([image, x, y])
}

/// The element of the tiled region at element `e` of it (in tile order): its x and y, and
/// whether the level has it.
fn region_place(f: &mut FunctionBuilder<'_>, l: Level, e: Value) -> BResult<[Value; 3]> {
    let [tx, ty] = l.tile_log2;
    let tile_bits = f.binary(BinOp::IAdd, tx, ty)?;
    let tile = f.binary(BinOp::ShiftRightLogical, e, tile_bits)?;
    let mask = low_mask(f, tile_bits)?;
    let m = f.binary(BinOp::BitwiseAnd, e, mask)?;
    let (xi, yi) = demorton(f, l, m)?;
    let column = f.binary(BinOp::UMod, tile, l.across)?;
    let row = f.binary(BinOp::UDiv, tile, l.across)?;
    let x = f.binary(BinOp::ShiftLeftLogical, column, tx)?;
    let x = f.binary(BinOp::BitwiseOr, x, xi)?;
    let y = f.binary(BinOp::ShiftLeftLogical, row, ty)?;
    let y = f.binary(BinOp::BitwiseOr, y, yi)?;
    let in_x = f.binary(BinOp::ULessThan, x, l.width)?;
    let in_y = f.binary(BinOp::ULessThan, y, l.height)?;
    let has = f.binary(BinOp::LogicalAnd, in_x, in_y)?;
    Ok([x, y, has])
}

/// The tight buffer's byte at which element `(x, y)` of image `image` starts.
fn tight_byte(
    f: &mut FunctionBuilder<'_>,
    l: Level,
    image: Value,
    x: Value,
    y: Value,
) -> BResult<Value> {
    let plane = f.binary(BinOp::IMul, l.width, l.height)?;
    let at = f.binary(BinOp::IMul, image, plane)?;
    let row = f.binary(BinOp::IMul, y, l.width)?;
    let at = f.binary(BinOp::IAdd, at, row)?;
    let at = f.binary(BinOp::IAdd, at, x)?;
    let at = f.binary(BinOp::ShiftLeftLogical, at, l.unit_log2)?;
    f.binary(BinOp::IAdd, l.tight_first, at)
}

/// The word `v` of a level's element into the image: where the level clamps its 32-bit float
/// depths ([`TiledLevelMove::clamps_depth`]), in `[0, 1]` by its bits (the module docs), else as
/// it is.
fn clamped_depth(f: &mut FunctionBuilder<'_>, l: Level, v: Value) -> BResult<Value> {
    let zero = u32_value(f, 0)?;
    let one = u32_value(f, 0x3f80_0000)?;
    let sign = u32_value(f, 0x8000_0000)?;
    let infinity = u32_value(f, 0x7f80_0000)?;
    let magnitude = u32_value(f, 0x7fff_ffff)?;
    let abs = f.binary(BinOp::BitwiseAnd, v, magnitude)?;
    let nan = f.binary(BinOp::UGreaterThan, abs, infinity)?;
    // Above 1.0's bits: a positive above 1.0, and every value with its sign bit, of which -0.0
    // alone stays.
    let above = f.binary(BinOp::UGreaterThan, v, one)?;
    let signed = f.binary(BinOp::UGreaterThanEqual, v, sign)?;
    let minus_zero = f.binary(BinOp::IEqual, v, sign)?;
    let r = f.select(above, one, v)?;
    let r = f.select(signed, zero, r)?;
    let r = f.select(minus_zero, v, r)?;
    let r = f.select(nan, one, r)?;
    let clamps = f.binary(BinOp::INotEqual, l.clamps_depth, zero)?;
    f.select(clamps, r, v)
}

/// The byte at byte `at` of the words `words`.
fn byte_at(f: &mut FunctionBuilder<'_>, words: Value, at: Value) -> BResult<Value> {
    let zero = u32_value(f, 0)?;
    let two = u32_value(f, 2)?;
    let three = u32_value(f, 3)?;
    let mask = u32_value(f, 0xff)?;
    let word = f.binary(BinOp::ShiftRightLogical, at, two)?;
    let p = f.access_chain(words, &[zero, word])?;
    let v = f.load(p, MemAccess::default())?;
    let place = f.binary(BinOp::BitwiseAnd, at, three)?;
    let shift = f.binary(BinOp::ShiftLeftLogical, place, three)?;
    let v = f.binary(BinOp::ShiftRightLogical, v, shift)?;
    f.binary(BinOp::BitwiseAnd, v, mask)
}

/// The module as a typed module, before serialization: the module [`tiled_texels_module`]
/// serializes, which the CPU interpreter runs. Its bindings are the symbols `TIGHT`, `TILED` and
/// `SHAPE` (the module docs).
fn typed_module() -> BResult<TypedModule> {
    let mut b = ModuleBuilder::new();
    let u32_ = b.ty(Type::Int {
        width: 32,
        signed: false,
    })?;
    let uvec3 = b.ty(Type::Vector { elem: u32_, len: 3 })?;
    let tight = words_buffer(&mut b, TIGHT, u32_)?;
    let tiled = words_buffer(&mut b, TILED, u32_)?;
    let shape = words_buffer(&mut b, SHAPE, u32_)?;
    let invocation = b.builtin(crate::spv::BuiltIn::GlobalInvocationId, IoDir::Input, uvec3)?;
    let void = b.ty(Type::Void)?;
    let mut f = b.function(void)?;
    let zero = u32_value(&mut f, 0)?;
    let invocation = f.module().global_value(invocation)?;
    let invocation = f.load(invocation, MemAccess::default())?;
    let gx = f.composite_extract(invocation, &[0])?;
    let gy = f.composite_extract(invocation, &[1])?;
    let tight = f.module().global_value(tight)?;
    let tiled = f.module().global_value(tiled)?;
    let shape = f.module().global_value(shape)?;
    let word = |f: &mut FunctionBuilder<'_>, index: Value| -> BResult<Value> {
        let p = f.access_chain(shape, &[zero, index])?;
        f.load(p, MemAccess::default())
    };
    let field = |f: &mut FunctionBuilder<'_>, index: u32| -> BResult<Value> {
        let index = u32_value(f, index)?;
        word(f, index)
    };
    let grid_width = field(&mut f, 0)?;
    let into = field(&mut f, 1)?;
    let into = f.binary(BinOp::INotEqual, into, zero)?;
    let levels = field(&mut f, 2)?;
    let grid_row = f.binary(BinOp::IMul, gy, grid_width)?;
    let i = f.binary(BinOp::IAdd, grid_row, gx)?;
    // The record of the last level whose first invocation is at most `i`.
    let header = TiledTexelsModule::HEADER_WORDS;
    let record_words = TiledTexelsModule::LEVEL_WORDS;
    let mut record = u32_value(&mut f, header)?;
    for k in 1..TiledTexelsModule::MAX_LEVELS {
        let at = header + record_words * k;
        let start = field(&mut f, at)?;
        let kth = u32_value(&mut f, k)?;
        let listed = f.binary(BinOp::ULessThan, kth, levels)?;
        let reached = f.binary(BinOp::UGreaterThanEqual, i, start)?;
        let taken = f.binary(BinOp::LogicalAnd, listed, reached)?;
        let this = u32_value(&mut f, at)?;
        record = f.select(taken, this, record)?;
    }
    let at = |f: &mut FunctionBuilder<'_>, offset: u32| -> BResult<Value> {
        let offset = u32_value(f, offset)?;
        let index = f.binary(BinOp::IAdd, record, offset)?;
        word(f, index)
    };
    let start = at(&mut f, 0)?;
    let count = at(&mut f, 1)?;
    let l = Level {
        unit_log2: at(&mut f, 2)?,
        width: at(&mut f, 3)?,
        height: at(&mut f, 4)?,
        tile_log2: [at(&mut f, 5)?, at(&mut f, 6)?],
        across: at(&mut f, 7)?,
        tiled_first: at(&mut f, 8)?,
        chain_stride: at(&mut f, 9)?,
        tight_first: at(&mut f, 10)?,
        region_words: at(&mut f, 11)?,
        images: at(&mut f, 12)?,
        clamps_depth: at(&mut f, 13)?,
    };
    let has_levels = f.binary(BinOp::ULessThan, zero, levels)?;
    let past_start = f.binary(BinOp::UGreaterThanEqual, i, start)?;
    let j = f.binary(BinOp::ISub, i, start)?;
    let in_count = f.binary(BinOp::ULessThan, j, count)?;
    let moves = f.binary(BinOp::LogicalAnd, past_start, in_count)?;
    let moves = f.binary(BinOp::LogicalAnd, has_levels, moves)?;
    let two = u32_value(&mut f, 2)?;
    let wide = f.binary(BinOp::UGreaterThanEqual, l.unit_log2, two)?;
    let body: ArmFn<'_, ()> = Box::new(move |f, _| {
        // Into the image: word `j` of the level's tight elements.
        let inward: ArmFn<'_, ()> = Box::new(move |f, _| {
            let two = u32_value(f, 2)?;
            let first_word = f.binary(BinOp::ShiftRightLogical, l.tight_first, two)?;
            let to = f.binary(BinOp::IAdd, first_word, j)?;
            let at_to = f.access_chain(tight, &[zero, to])?;
            let byte0 = f.binary(BinOp::ShiftLeftLogical, j, two)?;
            // An element of 4 bytes or more: the word is one word of one element.
            let whole: ArmFn<'_, ()> = Box::new(move |f, _| {
                let e = f.binary(BinOp::ShiftRightLogical, byte0, l.unit_log2)?;
                let unit_mask = low_mask(f, l.unit_log2)?;
                let b = f.binary(BinOp::BitwiseAnd, byte0, unit_mask)?;
                let [image, x, y] = tight_place(f, l, e)?;
                let from = tiled_byte(f, l, image, x, y)?;
                let from = f.binary(BinOp::IAdd, from, b)?;
                let two = u32_value(f, 2)?;
                let from = f.binary(BinOp::ShiftRightLogical, from, two)?;
                let p = f.access_chain(tiled, &[zero, from])?;
                let v = f.load(p, MemAccess::default())?;
                let v = clamped_depth(f, l, v)?;
                f.store(at_to, v, MemAccess::default())?;
                Ok(ArmEnd::FallThrough(Vec::new()))
            });
            // Narrower elements: byte by byte, 0 past the level's elements.
            let bytes: ArmFn<'_, ()> = Box::new(move |f, _| {
                let plane = f.binary(BinOp::IMul, l.width, l.height)?;
                let all = f.binary(BinOp::IMul, plane, l.images)?;
                let all = f.binary(BinOp::ShiftLeftLogical, all, l.unit_log2)?;
                let unit_mask = low_mask(f, l.unit_log2)?;
                let mut assembled = u32_value(f, 0)?;
                for k in 0..4u32 {
                    let kv = u32_value(f, k)?;
                    let t = f.binary(BinOp::IAdd, byte0, kv)?;
                    let inside = f.binary(BinOp::ULessThan, t, all)?;
                    let zero = u32_value(f, 0)?;
                    let t = f.select(inside, t, zero)?;
                    let e = f.binary(BinOp::ShiftRightLogical, t, l.unit_log2)?;
                    let b = f.binary(BinOp::BitwiseAnd, t, unit_mask)?;
                    let [image, x, y] = tight_place(f, l, e)?;
                    let from = tiled_byte(f, l, image, x, y)?;
                    let from = f.binary(BinOp::IAdd, from, b)?;
                    let v = byte_at(f, tiled, from)?;
                    let v = f.select(inside, v, zero)?;
                    let place = u32_value(f, 8 * k)?;
                    let v = f.binary(BinOp::ShiftLeftLogical, v, place)?;
                    assembled = f.binary(BinOp::BitwiseOr, assembled, v)?;
                }
                f.store(at_to, assembled, MemAccess::default())?;
                Ok(ArmEnd::FallThrough(Vec::new()))
            });
            f.if_(&mut (), wide, whole, Some(bytes))?;
            Ok(ArmEnd::FallThrough(Vec::new()))
        });
        // Back: word `j` of the level's tiled regions.
        let outward: ArmFn<'_, ()> = Box::new(move |f, _| {
            let two = u32_value(f, 2)?;
            let image = f.binary(BinOp::UDiv, j, l.region_words)?;
            let w = f.binary(BinOp::UMod, j, l.region_words)?;
            let chain = f.binary(BinOp::IMul, image, l.chain_stride)?;
            let first = f.binary(BinOp::IAdd, chain, l.tiled_first)?;
            let first = f.binary(BinOp::ShiftRightLogical, first, two)?;
            let to = f.binary(BinOp::IAdd, first, w)?;
            let at_to = f.access_chain(tiled, &[zero, to])?;
            let byte0 = f.binary(BinOp::ShiftLeftLogical, w, two)?;
            // An element of 4 bytes or more: the word is one word of one element, or of none.
            let whole: ArmFn<'_, ()> = Box::new(move |f, _| {
                let e = f.binary(BinOp::ShiftRightLogical, byte0, l.unit_log2)?;
                let unit_mask = low_mask(f, l.unit_log2)?;
                let b = f.binary(BinOp::BitwiseAnd, byte0, unit_mask)?;
                let [x, y, has] = region_place(f, l, e)?;
                let element: ArmFn<'_, ()> = Box::new(move |f, _| {
                    let from = tight_byte(f, l, image, x, y)?;
                    let from = f.binary(BinOp::IAdd, from, b)?;
                    let two = u32_value(f, 2)?;
                    let from = f.binary(BinOp::ShiftRightLogical, from, two)?;
                    let p = f.access_chain(tight, &[zero, from])?;
                    let v = f.load(p, MemAccess::default())?;
                    f.store(at_to, v, MemAccess::default())?;
                    Ok(ArmEnd::FallThrough(Vec::new()))
                });
                f.if_(&mut (), has, element, None)?;
                Ok(ArmEnd::FallThrough(Vec::new()))
            });
            // Narrower elements: byte by byte, a byte of no element as it was.
            let bytes: ArmFn<'_, ()> = Box::new(move |f, _| {
                let unit_mask = low_mask(f, l.unit_log2)?;
                let was = f.load(at_to, MemAccess::default())?;
                let mask = u32_value(f, 0xff)?;
                let mut assembled = u32_value(f, 0)?;
                for k in 0..4u32 {
                    let kv = u32_value(f, k)?;
                    let o = f.binary(BinOp::IAdd, byte0, kv)?;
                    let e = f.binary(BinOp::ShiftRightLogical, o, l.unit_log2)?;
                    let b = f.binary(BinOp::BitwiseAnd, o, unit_mask)?;
                    let [x, y, has] = region_place(f, l, e)?;
                    let zero = u32_value(f, 0)?;
                    let x = f.select(has, x, zero)?;
                    let y = f.select(has, y, zero)?;
                    let from = tight_byte(f, l, image, x, y)?;
                    let from = f.binary(BinOp::IAdd, from, b)?;
                    let v = byte_at(f, tight, from)?;
                    let place = u32_value(f, 8 * k)?;
                    let kept = f.binary(BinOp::ShiftRightLogical, was, place)?;
                    let kept = f.binary(BinOp::BitwiseAnd, kept, mask)?;
                    let v = f.select(has, v, kept)?;
                    let v = f.binary(BinOp::ShiftLeftLogical, v, place)?;
                    assembled = f.binary(BinOp::BitwiseOr, assembled, v)?;
                }
                f.store(at_to, assembled, MemAccess::default())?;
                Ok(ArmEnd::FallThrough(Vec::new()))
            });
            f.if_(&mut (), wide, whole, Some(bytes))?;
            Ok(ArmEnd::FallThrough(Vec::new()))
        });
        f.if_(&mut (), into, inward, Some(outward))?;
        Ok(ArmEnd::FallThrough(Vec::new()))
    });
    f.if_(&mut (), moves, body, None)?;
    f.ret(None)?;
    let id = f.finish()?;
    b.entry(
        ExecModel::GLCompute,
        id,
        vec![ExecMode::LocalSize {
            x: TiledTexelsModule::LOCAL_SIZE,
            y: 1,
            z: 1,
        }],
    )?;
    b.finalize()
}

/// The fixed binding numbers of the module's symbols.
fn binding_of(sym: &Symbol) -> Option<u32> {
    match sym {
        Symbol::DescriptorSet(_) => Some(TiledTexelsModule::SET),
        Symbol::Binding(s) if *s == TIGHT => Some(TiledTexelsModule::TIGHT_BINDING),
        Symbol::Binding(s) if *s == TILED => Some(TiledTexelsModule::TILED_BINDING),
        Symbol::Binding(s) if *s == SHAPE => Some(TiledTexelsModule::SHAPE_BINDING),
        _ => None,
    }
}

/// The tiled texel module of `device`: built with the typed builder, checked by the owned
/// verifier against the device's profile (requirements included), serialized, and numbered with
/// the fixed bindings of [`TiledTexelsModule`].
pub fn tiled_texels_module(device: &Device) -> Result<TiledTexelsModule, Error> {
    let profile = device.profile();
    let m = typed_module()?;
    let mut meter = Meter::new(&Budget::default());
    crate::spirv::verify::verify(&m, profile, &mut meter)?;
    let r = crate::spirv::serialize::serialize(m)?;
    profile.check(&r.requirements).map_err(|c| {
        Error::Internal(InternalError::new(
            InternalKind::RequirementsExceedProfile,
            format!("the tiled texel module needs {c:?}"),
        ))
    })?;
    let words = crate::spirv::serialize::relocate(&r, &binding_of)?;
    Ok(TiledTexelsModule {
        spirv: words.into(),
        requirements: r.requirements,
    })
}

#[cfg(test)]
mod tests;
