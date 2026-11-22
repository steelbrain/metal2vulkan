//! Threadgroup and private arenas with exact sizes.
//!
//! An arena is one `u32` array, and a pointer into it is a `u32` byte offset, so `phi`, `select`,
//! and GEP arithmetic are plain integer operations. Leaves are word-composed (`memory::leaf`) on
//! the floor representation : no explicit-layout or narrow storage feature is needed.
//!
//! - **Threadgroup arena** (`Workgroup`): the static threadgroup globals at the offsets the memory
//!   plan gives them (but those it holds as typed `Workgroup` objects of their own,
//!   `memory::plan::SharedPlan`), then the `[[threadgroup(n)]]` regions, each 16-byte aligned. A
//!   region's length is the specialization constant `SpecKey::TgLength(n)`
//!   (`RuntimeFact::ThreadgroupMemoryLength`) declared through `lower::dispatch`, so the region
//!   offsets and the arena's length are `OpSpecConstantOp` sums ([`declare_threadgroup`]). Sub-word
//!   stores are two relaxed `Workgroup`-scope word atomics: neighbouring bytes other
//!   invocations write concurrently are never lost.
//! - **Private arena** (`Private`, one per invocation): the allocas that are not typed, at their
//!   plan offsets. Sub-word stores are a plain read-modify-write (the memory is the invocation's
//!   own).
//!
//! **Sizes are exact**: an arena's containment domain is the byte end of its last object (for the
//! threadgroup arena with regions, of its last specialized region), with no sink word. The `u32`
//! array rounds that up to whole words and to at least [`MIN_ARENA_BYTES`], the widest leaf, so a
//! contained load, which reads at offset 0, stays inside the array; a 32768-byte threadgroup
//! allocation is a 32768-byte array and fits `maxComputeSharedMemorySize = 32768`.
//!
//! **A kernel's arena of `float` words** ([`TgWords::Floats`], on a profile whose
//! `ThreadgroupMemory` is `Native`, `Device::from_queries`'): where every access of a kernel's
//! arena is a whole, aligned `float` word, a scalar or a vector of them (`lower::access::local`),
//! the array is of `float`, `N` long, `N` the arena's whole words (at least [`FLOAT_LEAST`]), and
//! an access of `lanes` lanes reads or writes them at the words from `off / 4`. Nothing contains
//! it: an access past the arena is undefined, as in Metal. The array is the arena's own words, so
//! its threadgroup memory is Metal's static length and lengths, as for any arena.
//!
//! A kernel's explicit imageblock is the arena's last region: `width · height` cells of
//! `cell_bytes`, its extent the specialization constants `SpecKey::ImageblockWidth` and
//! `SpecKey::ImageblockHeight` (`RuntimeFact::ImageblockSize`, declared through
//! `lower::access::imageblock`), its length rounded up to 16 like a `[[threadgroup(n)]]` region's.
//!
//! **A tile function's arena** (`TgShape::tile`) starts with its tile's threadgroup memory,
//! `SpecKey::TileThreadgroupLength` bytes (the render pass's, rounded up to 16), which
//! `lower::access::tile` stages from and back to the pass's `SyntheticBinding::TileMemory`.
//! Region `n` starts where the encoder placed it in that memory, at `SpecKey::TgOffset(n)`, so
//! two regions placed at overlapping offsets alias as Metal's do; a region's length takes no part
//! in the layout. A tile function has no static threadgroup objects (Metal refuses its pipeline).
//! The imageblock follows the tile's memory.
//!
//! **A fragment function's arena** (`TgShape::fragment`) is its tile's threadgroup memory itself,
//! with no array of its own: a fragment has no workgroup, and the tile's fragments, its tile
//! dispatches and the pass's other draws all share that memory. Its words are the fragment's tile's
//! in the pass's `SyntheticBinding::TileMemory` storage buffer ([`ArenaDecl::tile_base`]: from word
//! `t · TileThreadgroupLength / 4`, `t` the tile of the fragment's pixel,
//! `lower::access::tile_memory`), its size `SpecKey::TileThreadgroupLength`, and region `n` at
//! `SpecKey::TgOffset(n)` as a tile function's. Its sub-word stores are `Device`-scope word
//! atomics: the words are a storage buffer's.
//!
//! **`maxComputeSharedMemorySize`** ([`TgShape::check`]): the static part (the objects, and the
//! aligned start of the regions when there are any) is checked here against the profile; the
//! specialized total, [`TgShape::static_bytes`] plus each region's length rounded up to 16, is
//! the binder's check when it specializes. An object or mesh function's is
//! `maxTaskSharedMemorySize` or `maxMeshSharedMemorySize`, which its entry wrapper's own words
//! share ([`workgroup_limit`]).

use crate::device::profile::Limits;
use crate::device::requirements::VkLimit;
use crate::error::{Capability, Error, Feature, Unsupported};
use crate::memory::leaf::{Coherence, WordMem, WordOps};
use crate::memory::plan::ArenaLayout;
use crate::pipeline::sym::{ResourceKind, ResourceSlot, SpecKey, Stage, SyntheticBinding};
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
use crate::spirv::error::{BResult, BuildError, Dangling};
use crate::spirv::ids::{ConstId, GlobalId, TypeId, Value};
use crate::spirv::ir::{
    AtomicOp, BinOp, ConstBits, Constant, MemAccess, Scope, Semantics, SpecOp, StorageClass, Type,
    UnOp,
};

/// The smallest arena array in bytes: the widest leaf (4 lanes of 8 bytes). A contained
/// out-of-bounds load reads at offset 0, which must stay inside the array.
pub const MIN_ARENA_BYTES: u64 = 32;

/// Metal aligns every `[[threadgroup(n)]]` region, and its length, to 16 bytes.
pub const REGION_ALIGN: u64 = 16;

/// The least length of an arena of `float` words ([`TgWords::Floats`]): Metal's 16 bytes
/// ([`REGION_ALIGN`]), the grain of its threadgroup memory, so an arena whose lengths are 0 is
/// still an array.
pub const FLOAT_LEAST: u64 = REGION_ALIGN / 4;

/// What the threadgroup accesses selection has seen say of the arena's words.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum TgWords {
    /// No access yet, in a kernel on a profile that holds such an arena in `float`s.
    #[default]
    Unseen,
    /// Every access so far a whole, aligned `float` word access; the arena is of `float`s once
    /// selection keeps it so (`lower::access::local::choose_words`).
    Floats,
    /// The `u32` array of composed words: an access of another shape, an atomic, a memory
    /// intrinsic, a pointer cell, or a stage, tier or profile that keeps the composed form.
    Composed,
}

/// Which arena a pointer points into.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Arena {
    /// `threadgroup` memory: one `Workgroup` arena per threadgroup.
    Threadgroup,
    /// The private arena: one `Private` array per invocation.
    Private,
    /// An object or mesh function's payload (`object_data`): one `TaskPayloadWorkgroupEXT` arena
    /// per object threadgroup, which the mesh threadgroups it launches read.
    Payload,
}

/// The shape of the threadgroup arena that selection fixes: the static objects' size and the
/// parameter regions in order.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct TgShape {
    /// The end of the last static threadgroup object (the plan's `static_size`).
    pub static_size: u64,
    /// The `[[threadgroup(n)]]` regions in arena order: `(parameter, n)`.
    pub regions: Vec<(u32, u32)>,
    /// The cell bytes of the kernel's explicit imageblock, after the regions, once an imageblock
    /// intrinsic uses it (`lower::access::imageblock`).
    pub imageblock: Option<u32>,
    /// A tile function's arena: the regions lie in the tile's threadgroup memory at the offsets
    /// the encoder sets (the module docs).
    pub tile: bool,
    /// A fragment function's arena: the tile's threadgroup memory in the pass's buffer, the tile
    /// that of the pixel of the `FragCoord` input, stage-IO object `fragment` (the module docs).
    /// `tile` holds too.
    pub fragment: Option<usize>,
    /// Whether the arena's words are `float`s (the module docs).
    pub words: TgWords,
}

impl TgShape {
    /// The shape of the plan's threadgroup arena, before any imageblock use.
    pub fn of(layout: &ArenaLayout) -> TgShape {
        TgShape {
            static_size: layout.static_size,
            regions: layout.dynamic.iter().map(|r| (r.param, r.index)).collect(),
            imageblock: None,
            tile: false,
            fragment: None,
            words: TgWords::Unseen,
        }
    }

    /// The bytes the arena needs before specialization: the static objects, and when there are
    /// regions (or an imageblock), the 16-byte-aligned offset at which the first one starts.
    /// `None` on overflow.
    pub fn static_bytes(&self) -> Option<u64> {
        if self.regions.is_empty() && self.imageblock.is_none() {
            Some(self.static_size)
        } else {
            self.static_size.checked_next_multiple_of(REGION_ALIGN)
        }
    }

    /// The check of the static part of an entry's arena against its workgroup memory limit `w`
    /// ([`workgroup_limit`]). The specialized regions add their lengths, which the binder checks
    /// when it specializes.
    pub fn check(&self, w: WorkgroupLimit) -> Result<(), Error> {
        w.check(self.static_bytes().unwrap_or(u64::MAX))
    }
}

/// The workgroup memory limit of an entry: the device's limit for its stage, and the bytes its
/// entry wrapper declares beside the function's own.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WorkgroupLimit {
    pub limit: VkLimit,
    pub available: u64,
    /// The wrapper's own workgroup words: an object function's mesh grid (a `uvec3`, counted
    /// whether or not the function sets it) and a mesh function's primitive count (a `uint`), and
    /// its staged indices where it has them (`lower::mesh::workgroup_limit`).
    pub wrapper: u64,
    /// An object or mesh entry's limit on its workgroup memory and its payload together
    /// (`maxTaskPayloadAndSharedMemorySize`, `maxMeshPayloadAndSharedMemorySize`), with the
    /// payload's bytes ([`WorkgroupLimit::with_payload`]).
    pub shared: Option<SharedLimit>,
}

/// The limit an object or mesh entry's workgroup memory shares with its payload.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SharedLimit {
    pub limit: VkLimit,
    pub available: u64,
    /// The entry's payload arena's bytes ([`payload_bytes`]), 0 for none.
    pub payload: u64,
}

impl WorkgroupLimit {
    /// Whether `bytes` of the function's workgroup memory fit beside the wrapper's, and with the
    /// payload's where they share a limit: else `Unsupported{ThreadgroupMemorySize}` naming the
    /// limit and the total.
    pub fn check(self, bytes: u64) -> Result<(), Error> {
        let required = bytes.saturating_add(self.wrapper);
        let refused = |limit, required, available| {
            Err(Unsupported::needs(
                Feature::ThreadgroupMemorySize,
                Capability::Limit {
                    limit,
                    required,
                    available,
                },
            )
            .into())
        };
        if required > self.available {
            return refused(self.limit, required, self.available);
        }
        if let Some(s) = self.shared {
            let together = required.saturating_add(s.payload);
            if together > s.available {
                return refused(s.limit, together, s.available);
            }
        }
        Ok(())
    }

    /// This limit for an entry whose payload arena has `payload` bytes.
    pub fn with_payload(self, payload: u64) -> WorkgroupLimit {
        WorkgroupLimit {
            shared: self.shared.map(|s| SharedLimit { payload, ..s }),
            ..self
        }
    }
}

/// The workgroup memory limit of a `stage` entry: `maxComputeSharedMemorySize` (a kernel's),
/// `maxTaskSharedMemorySize` (an object function's) or `maxMeshSharedMemorySize` (a mesh
/// function's), with its entry wrapper's own words, and an object or mesh function's limit on its
/// workgroup memory and payload together, for no payload ([`WorkgroupLimit::with_payload`] adds
/// it).
pub fn workgroup_limit(stage: Stage, limits: &Limits) -> WorkgroupLimit {
    let m = &limits.mesh;
    let shared = |limit, available: u32| {
        Some(SharedLimit {
            limit,
            available: u64::from(available),
            payload: 0,
        })
    };
    let (limit, available, wrapper, shared) = match stage {
        Stage::Object => (
            VkLimit::MaxTaskSharedMemorySize,
            m.max_task_shared_memory_size,
            12,
            shared(
                VkLimit::MaxTaskPayloadAndSharedMemorySize,
                m.max_task_payload_and_shared_memory_size,
            ),
        ),
        Stage::Mesh => (
            VkLimit::MaxMeshSharedMemorySize,
            m.max_mesh_shared_memory_size,
            4,
            shared(
                VkLimit::MaxMeshPayloadAndSharedMemorySize,
                m.max_mesh_payload_and_shared_memory_size,
            ),
        ),
        Stage::Kernel | Stage::Vertex | Stage::Fragment => (
            VkLimit::MaxComputeSharedMemorySize,
            limits.max_compute_shared_memory_size,
            0,
            None,
        ),
    };
    WorkgroupLimit {
        limit,
        available: u64::from(available),
        wrapper,
        shared,
    }
}

/// A declared arena: its `u32` array variable and its byte size, the containment domain (a
/// plain `u32` constant, or for a threadgroup arena with regions an `OpSpecConstantOp`), or a
/// threadgroup arena's `float` array. A private arena in device memory has no variable: its words
/// are the invocation's slot of the runtime's memory, through `slot`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ArenaDecl {
    pub arena: Arena,
    pub global: Option<GlobalId>,
    pub size: ConstId,
    /// A fragment function's threadgroup arena (the module docs): `global` is the pass's tile
    /// memory buffer, a storage view, whose word `tile_base` (a `u32` of the function that emits
    /// the access, `lower::access::tile_memory::base`) is the arena's word 0. `None` for an
    /// arena that is its own array, and in the declaration, before a function computes it.
    pub tile_base: Option<Value>,
    /// An arena of `float` words (the module docs): `global` is a `float` array.
    pub floats: bool,
    /// A private arena in device memory (`memory::plan::ArenaLayout::device`): the invocation's
    /// slot placed as a `PhysicalStorageBuffer` pointer to its words, a runtime array of `u32`
    /// whose element 0 is the arena's word 0 (a value of the function that emits the access,
    /// `lower::emit::EmitCx::arena_slot`). `None` for any other arena, and in the declaration.
    pub slot: Option<Value>,
}

/// The declared threadgroup arena and the byte offset of each region: `(parameter, offset)` in
/// [`TgShape::regions`] order, and the imageblock's.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TgDecl {
    pub decl: ArenaDecl,
    pub regions: Vec<(u32, ConstId)>,
    pub imageblock: Option<ConstId>,
    /// A fragment function's tile memory (`TgShape::fragment`): what places its tile in the
    /// pass's buffer.
    pub fragment: Option<FragmentTile>,
}

/// What places a fragment's tile in the pass's tile memory buffer (the module docs): the
/// `FragCoord` input's stage-IO object, and the specialization constants `SpecKey::TileGridWidth`
/// (the tiles across the render area) and `SpecKey::ImageblockWidth` and
/// `SpecKey::ImageblockHeight` (the pass's tile size).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FragmentTile {
    pub coord: usize,
    pub grid: ConstId,
    pub width: ConstId,
    pub height: ConstId,
}

fn u32_ty(mb: &mut ModuleBuilder) -> BResult<TypeId> {
    mb.ty(Type::Int {
        width: 32,
        signed: false,
    })
}

/// A plain `u32` constant (the arena's sizes fit 32 bits: the limit check comes first).
fn konst(mb: &mut ModuleBuilder, v: u64) -> BResult<ConstId> {
    let ty = u32_ty(mb)?;
    mb.constant(Constant::Scalar {
        ty,
        bits: ConstBits(v.min(u64::from(u32::MAX))),
    })
}

/// The `class` variable of type `ty`, with a null initializer where it is `zeroed`: a
/// threadgroup arena, whose memory reads zero until the program writes it.
fn variable(
    mb: &mut ModuleBuilder,
    class: StorageClass,
    ty: TypeId,
    zeroed: bool,
) -> BResult<GlobalId> {
    let init = match zeroed {
        true => Some(mb.constant(Constant::Null { ty })?),
        false => None,
    };
    mb.plain_global(class, ty, init)
}

/// The `u32` array of `words` (a constant) in storage class `class`, null-initialized where it is
/// `zeroed`.
fn array(
    mb: &mut ModuleBuilder,
    class: StorageClass,
    words: ConstId,
    zeroed: bool,
) -> BResult<GlobalId> {
    let elem = u32_ty(mb)?;
    let ty = mb.ty(Type::Array {
        elem,
        len: words,
        stride: None,
    })?;
    variable(mb, class, ty, zeroed)
}

/// The length of an arena of `float` words of `words` whole words: its words, and never under
/// [`FLOAT_LEAST`].
pub fn float_len(words: u64) -> u64 {
    words.max(FLOAT_LEAST)
}

/// The `Workgroup` `float` array of `len` (a constant) words: an arena of `float` words,
/// null-initialized where it is `zeroed`.
fn float_array(mb: &mut ModuleBuilder, len: ConstId, zeroed: bool) -> BResult<GlobalId> {
    let elem = mb.ty(Type::Float { width: 32 })?;
    let ty = mb.ty(Type::Array {
        elem,
        len,
        stride: None,
    })?;
    variable(mb, StorageClass::Workgroup, ty, zeroed)
}

/// Words of a static arena of `bytes`: whole words, at least [`MIN_ARENA_BYTES`].
fn static_words(bytes: u64) -> u64 {
    bytes.max(MIN_ARENA_BYTES).div_ceil(4)
}

/// Declare the threadgroup arena of `shape`. Without regions its length is a plain constant. With
/// regions, region `i` starts at `off_i`: `off_0` is the static size rounded up to 16, and
/// `off_{i+1} = (off_i + len_i + 15) & ~15` with `len_i` the specialization constant
/// `SpecKey::TgLength(n_i)` (default 0: the binder always specializes it). The imageblock is one
/// more region after them, of length `width · height · cell_bytes` from its specialization
/// constants (default 0 by 0). The size is the last region's end, and the array's length `max(size
/// / 4, MIN_ARENA_BYTES / 4)`, all `OpSpecConstantOp`s. An arena of `float` words
/// ([`TgWords::Floats`]) is a `float` array of its whole words, at least [`FLOAT_LEAST`]
/// ([`float_len`]). A `zeroed` arena's array has a null initializer: the program's threadgroup
/// memory, its tile's memory and its imageblock among it, reads zero until it is written.
pub fn declare_threadgroup(
    mb: &mut ModuleBuilder,
    shape: &TgShape,
    zeroed: bool,
) -> BResult<TgDecl> {
    let u32_ = u32_ty(mb)?;
    if shape.regions.is_empty() && shape.imageblock.is_none() {
        let size = konst(mb, shape.static_size)?;
        let (global, floats) = if shape.words == TgWords::Floats {
            let len = konst(mb, float_len(shape.static_size.div_ceil(4)))?;
            (float_array(mb, len, zeroed)?, true)
        } else {
            let words = konst(mb, static_words(shape.static_size))?;
            (array(mb, StorageClass::Workgroup, words, zeroed)?, false)
        };
        return Ok(TgDecl {
            decl: ArenaDecl {
                arena: Arena::Threadgroup,
                global: Some(global),
                size,
                tile_base: None,
                floats,
                slot: None,
            },
            regions: Vec::new(),
            imageblock: None,
            fragment: None,
        });
    }
    // Its tile's memory in the pass's buffer: no `Workgroup` variable.
    if let Some(coord) = shape.fragment {
        return declare_fragment(mb, shape, coord);
    }
    let base = shape
        .static_size
        .checked_next_multiple_of(REGION_ALIGN)
        .unwrap_or(u64::MAX);
    let mut off = konst(mb, base)?;
    let fifteen = konst(mb, REGION_ALIGN - 1)?;
    let mask = konst(mb, u64::from(!(REGION_ALIGN as u32 - 1)))?;
    let mut regions = Vec::with_capacity(shape.regions.len());
    if shape.tile {
        // The tile's threadgroup memory, each region where the encoder placed it.
        for &(param, index) in &shape.regions {
            regions.push((param, tile_offset(mb, index)?));
        }
        let tile = tile_length(mb)?;
        off = mb.spec_op(SpecOp::IAdd, u32_, &[off, tile])?;
    } else {
        for &(param, index) in &shape.regions {
            regions.push((param, off));
            let len = crate::lower::dispatch::threadgroup_length(mb, index, 0)?;
            let end = mb.spec_op(SpecOp::IAdd, u32_, &[off, len])?;
            let end = mb.spec_op(SpecOp::IAdd, u32_, &[end, fifteen])?;
            off = mb.spec_op(SpecOp::BitwiseAnd, u32_, &[end, mask])?;
        }
    }
    let imageblock = match shape.imageblock {
        Some(cell_bytes) => {
            let at = off;
            let len = crate::lower::access::imageblock::region_length(mb, cell_bytes)?;
            let end = mb.spec_op(SpecOp::IAdd, u32_, &[off, len])?;
            let end = mb.spec_op(SpecOp::IAdd, u32_, &[end, fifteen])?;
            off = mb.spec_op(SpecOp::BitwiseAnd, u32_, &[end, mask])?;
            Some(at)
        }
        None => None,
    };
    let size = off;
    let two = konst(mb, 2)?;
    let (global, floats) = if shape.words == TgWords::Floats {
        // The regions end on 16 bytes, so `words` is the arena's whole words: the array is
        // `max(words, FLOAT_LEAST)` long ([`float_len`]).
        let words = mb.spec_op(SpecOp::ShiftRightLogical, u32_, &[size, two])?;
        let least = konst(mb, FLOAT_LEAST)?;
        let bool_ = mb.ty(Type::Bool)?;
        let small = mb.spec_op(SpecOp::ULessThan, bool_, &[words, least])?;
        let len = mb.spec_op(SpecOp::Select, u32_, &[small, least, words])?;
        (float_array(mb, len, zeroed)?, true)
    } else {
        // The composed form's words, made in this order so its module is the same word for word.
        let min = konst(mb, MIN_ARENA_BYTES / 4)?;
        let words = mb.spec_op(SpecOp::ShiftRightLogical, u32_, &[size, two])?;
        let bool_ = mb.ty(Type::Bool)?;
        let small = mb.spec_op(SpecOp::ULessThan, bool_, &[words, min])?;
        let words = mb.spec_op(SpecOp::Select, u32_, &[small, min, words])?;
        (array(mb, StorageClass::Workgroup, words, zeroed)?, false)
    };
    Ok(TgDecl {
        decl: ArenaDecl {
            arena: Arena::Threadgroup,
            global: Some(global),
            size,
            tile_base: None,
            floats,
            slot: None,
        },
        regions,
        imageblock,
        fragment: None,
    })
}

/// Declare a fragment function's threadgroup arena, its tile's memory (the module docs): the
/// pass's tile memory buffer, a storage view of the fragment stage, each region at its
/// `SpecKey::TgOffset`, and the size `SpecKey::TileThreadgroupLength`. A fragment has no
/// imageblock region: the product refuses a fragment function's imageblock
/// (`Feature::Imageblocks`).
fn declare_fragment(mb: &mut ModuleBuilder, shape: &TgShape, coord: usize) -> BResult<TgDecl> {
    let u32_ = u32_ty(mb)?;
    let slot = ResourceSlot {
        stage: Stage::Fragment,
        kind: ResourceKind::Synthetic(SyntheticBinding::TileMemory),
        index: 0,
    };
    let view = crate::memory::views::declare_at(mb, slot)?;
    let mut regions = Vec::with_capacity(shape.regions.len());
    for &(param, index) in &shape.regions {
        regions.push((param, tile_offset(mb, index)?));
    }
    Ok(TgDecl {
        decl: ArenaDecl {
            arena: Arena::Threadgroup,
            global: Some(view.global),
            size: tile_length(mb)?,
            tile_base: None,
            floats: false,
            slot: None,
        },
        regions,
        imageblock: None,
        fragment: Some(FragmentTile {
            coord,
            grid: mb.spec(SpecKey::TileGridWidth, u32_, ConstBits(0))?,
            width: mb.spec(SpecKey::ImageblockWidth, u32_, ConstBits(0))?,
            height: mb.spec(SpecKey::ImageblockHeight, u32_, ConstBits(0))?,
        }),
    })
}

/// The specialization constant `SpecKey::TgOffset(index)`: where a tile function's threadgroup
/// argument `index` lies in its tile's threadgroup memory (default 0).
fn tile_offset(mb: &mut ModuleBuilder, index: u32) -> BResult<ConstId> {
    let u32_ = u32_ty(mb)?;
    mb.spec(SpecKey::TgOffset(index), u32_, ConstBits(0))
}

/// The specialization constant `SpecKey::TileThreadgroupLength`: the bytes of each tile's
/// threadgroup memory, a multiple of 16 (default 0).
pub fn tile_length(mb: &mut ModuleBuilder) -> BResult<ConstId> {
    let u32_ = u32_ty(mb)?;
    mb.spec(SpecKey::TileThreadgroupLength, u32_, ConstBits(0))
}

/// The bytes of the payload arena of a payload of `size` bytes: whole words, at least
/// [`MIN_ARENA_BYTES`]. An object entry's and a mesh entry's payloads of one size are of one type.
pub fn payload_bytes(size: u64) -> u64 {
    static_words(size).saturating_mul(4)
}

/// Declare the payload arena of a payload of `size` bytes: a `TaskPayloadWorkgroupEXT` array of
/// [`payload_bytes`], with no initializer: an object function's payload "is assumed uninitialized
/// at the entry of the object function" (the Metal Shading Language Specification's `payload`
/// attribute).
pub fn declare_payload(mb: &mut ModuleBuilder, size: u64) -> BResult<ArenaDecl> {
    let bytes = konst(mb, size)?;
    let words = konst(mb, static_words(size))?;
    let global = array(mb, StorageClass::TaskPayloadWorkgroupEXT, words, false)?;
    Ok(ArenaDecl {
        arena: Arena::Payload,
        global: Some(global),
        size: bytes,
        tile_base: None,
        floats: false,
        slot: None,
    })
}

/// Declare the private arena of `size` bytes: a `Private` array, or where it is in `device`
/// memory, no variable (its words are the invocation's slot of the runtime's memory).
pub fn declare_private(mb: &mut ModuleBuilder, size: u64, device: bool) -> BResult<ArenaDecl> {
    let bytes = konst(mb, size)?;
    let global = match device {
        true => None,
        false => {
            let words = konst(mb, static_words(size))?;
            Some(array(mb, StorageClass::Private, words, false)?)
        }
    };
    Ok(ArenaDecl {
        arena: Arena::Private,
        global,
        size: bytes,
        tile_base: None,
        floats: false,
        slot: None,
    })
}

/// The word memory of an arena, with the memory operands of its loads and stores: the
/// threadgroup arena's carry its coherence; the private arena is never shared, so its accesses
/// carry only `Volatile`. The payload arena's are plain under `GLSL450`; under the Vulkan model,
/// which orders only non-private accesses between invocations and lets no `TaskPayloadWorkgroupEXT`
/// access carry `NonPrivatePointer`, they are relaxed `Workgroup` word atomics (an atomic is
/// non-private), so barriers order them as they order the threadgroup arena's.
#[derive(Clone, Copy, Debug)]
pub struct ArenaMem(pub ArenaDecl, pub WordOps);

impl ArenaMem {
    /// A pointer to word `word` (a `u32` index) of the arena: of its array, for a fragment
    /// function's tile memory word `tile_base + word` of the pass's buffer, or for a private arena
    /// in device memory element `word` of its slot's words, the `PhysicalStorageBuffer` word
    /// `4 · word` bytes past the slot: one access chain from the pointer the function placed once.
    pub fn ptr(&self, fb: &mut FunctionBuilder<'_>, word: Value) -> BResult<Value> {
        let global = match (self.0.slot, self.0.global) {
            (Some(words), _) => return fb.access_chain(words, &[word]),
            (None, Some(global)) => global,
            // A private arena in device memory reached before its function placed its slot
            // (`lower::access::local::decl_of` places it).
            (None, None) => {
                return Err(BuildError::Dangling(Dangling::Value(Value::new(u32::MAX))))
            }
        };
        let base = fb.module().global_value(global)?;
        match self.0.tile_base {
            None => fb.access_chain(base, &[word]),
            Some(first) => {
                let at = fb.binary(BinOp::IAdd, first, word)?;
                let u32_ = u32_ty(fb.module())?;
                let zero = fb.module().constant(Constant::Scalar {
                    ty: u32_,
                    bits: ConstBits(0),
                })?;
                let zero = fb.module().const_value(zero)?;
                fb.access_chain(base, &[zero, at])
            }
        }
    }

    /// The scope of the arena's word atomics: its workgroup's, or for a fragment function's tile
    /// memory, a storage buffer's, the device's.
    pub fn scope(&self) -> Scope {
        match self.0.tile_base {
            None => Scope::Workgroup,
            Some(_) => Scope::Device,
        }
    }
}

impl ArenaMem {
    fn access(&self, fb: &mut FunctionBuilder<'_>, store: bool) -> BResult<MemAccess> {
        match self.0.arena {
            Arena::Threadgroup => self.1.access(fb, MemAccess::default(), store),
            // A word in device memory is aligned to 4, as a `PhysicalStorageBuffer` access
            // states.
            Arena::Private | Arena::Payload => Ok(MemAccess {
                volatile: self.1.volatile,
                aligned: self.0.slot.map(|_| 4),
                ..MemAccess::default()
            }),
        }
    }

    /// Whether the arena's words are accessed as atomics: the payload arena's under the Vulkan
    /// model.
    fn atomic_words(&self) -> bool {
        self.0.arena == Arena::Payload && self.1.coherence != Coherence::Plain
    }
}

impl WordMem for ArenaMem {
    fn read(&mut self, fb: &mut FunctionBuilder<'_>, word: Value) -> BResult<Value> {
        let p = self.ptr(fb, word)?;
        if self.atomic_words() {
            return fb.atomic_load(p, Scope::Workgroup, Semantics::RELAXED);
        }
        let access = self.access(fb, false)?;
        fb.load(p, access)
    }

    fn write(&mut self, fb: &mut FunctionBuilder<'_>, word: Value, value: Value) -> BResult<()> {
        let p = self.ptr(fb, word)?;
        if self.atomic_words() {
            return fb.atomic_store(p, Scope::Workgroup, Semantics::RELAXED, value);
        }
        let access = self.access(fb, true)?;
        fb.store(p, value, access)
    }

    fn write_bits(
        &mut self,
        fb: &mut FunctionBuilder<'_>,
        word: Value,
        mask: Value,
        bits: Value,
    ) -> BResult<()> {
        let p = self.ptr(fb, word)?;
        let keep = fb.unary(UnOp::Not, mask)?;
        match self.0.arena {
            Arena::Threadgroup | Arena::Payload => {
                // Two relaxed word atomics: neighbouring bytes are shared with other invocations.
                let clear = fb.binary(BinOp::BitwiseOr, keep, bits)?;
                let scope = self.scope();
                fb.atomic(AtomicOp::And, p, scope, Semantics::RELAXED, clear)?;
                fb.atomic(AtomicOp::Or, p, scope, Semantics::RELAXED, bits)?;
                Ok(())
            }
            Arena::Private => {
                let load = self.access(fb, false)?;
                let old = fb.load(p, load)?;
                let kept = fb.binary(BinOp::BitwiseAnd, old, keep)?;
                let new = fb.binary(BinOp::BitwiseOr, kept, bits)?;
                let store = self.access(fb, true)?;
                fb.store(p, new, store)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The static part is checked exactly against `maxComputeSharedMemorySize`; regions start
    /// 16-byte aligned after it; the array is whole words and at least the widest leaf.
    #[test]
    fn the_static_part_is_checked_exactly() {
        let limits = Limits {
            max_compute_shared_memory_size: 32768,
            ..Limits::default()
        };
        let shape = |static_size, regions: &[(u32, u32)]| TgShape {
            static_size,
            regions: regions.to_vec(),
            imageblock: None,
            tile: false,
            fragment: None,
            words: TgWords::Unseen,
        };
        let check = |s: TgShape| s.check(workgroup_limit(Stage::Kernel, &limits));
        assert_eq!(check(shape(32768, &[])), Ok(()));
        assert_eq!(
            check(shape(32769, &[])),
            Err(Unsupported::needs(
                Feature::ThreadgroupMemorySize,
                Capability::Limit {
                    limit: VkLimit::MaxComputeSharedMemorySize,
                    required: 32769,
                    available: 32768,
                },
            )
            .into())
        );
        assert_eq!(shape(32753, &[(1, 0)]).static_bytes(), Some(32768));
        assert_eq!(check(shape(32753, &[(1, 0)])), Ok(()));
        assert!(check(shape(32769, &[(1, 0)])).is_err());
        assert_eq!(shape(u64::MAX, &[(1, 0)]).static_bytes(), None);
        assert_eq!(static_words(0), 8);
        assert_eq!(static_words(6), 8);
        assert_eq!(static_words(33), 9);
        assert_eq!(static_words(32768), 8192);
    }

    /// An object function's arena shares `maxTaskSharedMemorySize` with its wrapper's mesh grid (12
    /// bytes), and a mesh function's shares `maxMeshSharedMemorySize` with its primitive count (4
    /// bytes); a kernel's has `maxComputeSharedMemorySize` alone.
    #[test]
    fn an_object_or_mesh_arena_shares_its_stage_limit_with_the_wrapper() {
        let mut limits = Limits {
            max_compute_shared_memory_size: 32768,
            ..Limits::default()
        };
        limits.mesh.max_task_shared_memory_size = 1024;
        limits.mesh.max_mesh_shared_memory_size = 2048;
        limits.mesh.max_task_payload_and_shared_memory_size = 1536;
        limits.mesh.max_mesh_payload_and_shared_memory_size = 2560;
        let refused = |limit: VkLimit, required: u64, available: u64| -> Result<(), Error> {
            Err(Unsupported::needs(
                Feature::ThreadgroupMemorySize,
                Capability::Limit {
                    limit,
                    required,
                    available,
                },
            )
            .into())
        };
        let object = workgroup_limit(Stage::Object, &limits);
        assert_eq!(object.check(1012), Ok(()));
        assert_eq!(
            object.check(1013),
            refused(VkLimit::MaxTaskSharedMemorySize, 1025, 1024)
        );
        let mesh = workgroup_limit(Stage::Mesh, &limits);
        assert_eq!(mesh.check(2044), Ok(()));
        assert_eq!(
            mesh.check(2045),
            refused(VkLimit::MaxMeshSharedMemorySize, 2049, 2048)
        );
        let kernel = workgroup_limit(Stage::Kernel, &limits);
        assert_eq!(kernel.check(32768), Ok(()));
        assert_eq!(
            kernel.check(32769),
            refused(VkLimit::MaxComputeSharedMemorySize, 32769, 32768)
        );
        // A payload shares the stage's limit on workgroup memory and payload together; a
        // kernel has none.
        let object = object.with_payload(payload_bytes(600));
        assert_eq!(object.check(924), Ok(()));
        assert_eq!(
            object.check(925),
            refused(VkLimit::MaxTaskPayloadAndSharedMemorySize, 1537, 1536)
        );
        let mesh = mesh.with_payload(payload_bytes(1));
        assert_eq!(mesh.check(2044), Ok(()));
        assert_eq!(
            workgroup_limit(Stage::Mesh, &limits)
                .with_payload(payload_bytes(600))
                .check(1957),
            refused(VkLimit::MaxMeshPayloadAndSharedMemorySize, 2561, 2560)
        );
        assert_eq!(kernel.with_payload(64).check(32768), Ok(()));
        assert_eq!(payload_bytes(1), 32);
        assert_eq!(payload_bytes(33), 36);
    }
}
