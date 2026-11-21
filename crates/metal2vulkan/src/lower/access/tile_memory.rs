//! A fragment function's threadgroup memory: its tile's, which Metal keeps for the tile across the
//! render pass, as it keeps a tile function's.
//!
//! **What Metal gives** (measured on an M4, and on an M4 and an M3 Max):
//!
//! - A fragment function's `[[threadgroup(n)]]` argument is its tile's threadgroup memory: the
//!   allocation the pass's tile functions' arguments reach, placed where
//!   `setThreadgroupMemoryLength:offset:atIndex:` placed argument `n`. A tile dispatch reads what
//!   the draws before it wrote there, and the draws after it read what it wrote.
//! - Every fragment of the tile shares it: a relaxed atomic add from each fragment of a draw over
//!   a 16x16 tile counts 256.
//! - It starts at zero with each pass.
//! - Draws see each other's writes in encode order: two draws whose fragments each store their
//!   own value to one word leave the later draw's.
//! - In a multisampled pass the function runs once per covered sample: at 2 samples, a tile of
//!   256 covered pixels counts 512, as at 4 samples it counts 1024.
//! - Metal's compiler refuses a `threadgroup` variable in a fragment function ("variables in the
//!   threadgroup address space cannot be declared in a fragment function"), so the function's
//!   threadgroup memory is its arguments alone; one that declares any is
//!   `InvalidDescriptor(TileThreadgroupVariable)`, as a tile function that does.
//!
//! **What runs.** The function's threadgroup arena is its tile's words in the pass's buffer of
//! every tile's memory (`SyntheticBinding::TileMemory`, the buffer a tile function's entry stages
//! from and back to; `memory::arena`): tile `t = (y / h) · TileGridWidth + x / w`, `(x, y)` the
//! integer part of the fragment's `FragCoord` (its pixel) and `w x h` the pass's tile
//! (`SpecKey::ImageblockWidth`, `SpecKey::ImageblockHeight`), whose word 0 is the buffer's word
//! `t · TileThreadgroupLength / 4` ([`base`], computed once in the function's first block). Each
//! argument is a region of the arena at its `SpecKey::TgOffset`, as a tile function's, and every
//! access is the arena's: contained against `TileThreadgroupLength`, its sub-word stores and its
//! atomics `Device`-scope word atomics of the buffer, ordering it as device memory. A write of it
//! (a store, or any atomic) needs `fragmentStoresAndAtomics`. A function that writes it writes
//! memory (`StageInterface::writes_memory`), so in a multisampled pipeline it runs as many times
//! per pixel as the device runs it (`PrecisionNote::DeviceFragmentRuns`), where Apple's GPUs run
//! it once per covered sample; one that only reads it reads what each of the pixel's samples
//! would.
//!
//! **Order.** The fragments of one draw reach the memory as Metal's do, unordered but for their
//! atomics. The render encoder orders draws that reach it in encode order across render pass
//! instances (`runtime::render`), as Vulkan orders no storage write of one draw before another's
//! read inside an instance but pixel by pixel.

use crate::device::requirements::VkFeature;
use crate::error::{Capability, Error, Feature, Unsupported};
use crate::lower::emit::EmitCx;
use crate::lower::select::SelectCx;
use crate::lower::stage_io;
use crate::lower::sync::memory_model;
use crate::memory::arena::{self, FragmentTile};
use crate::pipeline::sym::Stage;
use crate::spirv::builder::FunctionBuilder;
use crate::spirv::error::BResult;
use crate::spirv::ids::Value as SValue;
use crate::spirv::ir::{BinOp, ConvOp, MemAccess, Type as SType};

/// Plan a fragment function's tile memory as its arena is first used (the module docs): its
/// `FragCoord` input, whose stage-IO object this returns, and the buffer's descriptor.
pub(crate) fn plan(cx: &mut SelectCx<'_>) -> usize {
    cx.descriptors.storage_buffers = cx.descriptors.storage_buffers.saturating_add(1);
    stage_io::frag_coord(cx)
}

/// A write of the threadgroup arena (a store, or any atomic): in a fragment function, of its
/// tile's memory in the pass's buffer, a storage buffer written by the fragment stage, which
/// needs `fragmentStoresAndAtomics`; its sub-word stores and atomics are `Device`-scope (under
/// the Vulkan memory model, `vulkanMemoryModelDeviceScope`).
pub(crate) fn written(cx: &mut SelectCx<'_>) -> Result<(), Error> {
    if cx.stage != Stage::Fragment {
        return Ok(());
    }
    if !cx.profile.fields().leaf.fragment_stores_and_atomics {
        return Err(Unsupported::needs(
            Feature::StageStoresAndAtomics,
            Capability::Feature(VkFeature::FragmentStoresAndAtomics),
        )
        .into());
    }
    memory_model::note_device_scope(cx);
    Ok(())
}

/// The first word of the fragment's tile's memory in the pass's buffer, in the open block: `t ·
/// TileThreadgroupLength / 4` for the tile `t` of the fragment's pixel (the module docs).
pub(crate) fn base(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    tile: FragmentTile,
) -> BResult<SValue> {
    let frag = stage_io::object_global(cx, tile.coord)?;
    let frag = fb.module().global_value(frag)?;
    let frag = fb.load(frag, MemAccess::default())?;
    let u32_ = fb.module().ty(SType::Int {
        width: 32,
        signed: false,
    })?;
    // `FragCoord` is the pixel's center, or a sample's position inside the pixel: its integer
    // part is the pixel.
    let fx = fb.composite_extract(frag, &[0])?;
    let fy = fb.composite_extract(frag, &[1])?;
    let x = fb.convert(ConvOp::ConvertFToU, fx, u32_)?;
    let y = fb.convert(ConvOp::ConvertFToU, fy, u32_)?;
    let width = fb.module().const_value(tile.width)?;
    let height = fb.module().const_value(tile.height)?;
    let grid = fb.module().const_value(tile.grid)?;
    let column = fb.binary(BinOp::UDiv, x, width)?;
    let row = fb.binary(BinOp::UDiv, y, height)?;
    let row = fb.binary(BinOp::IMul, row, grid)?;
    let index = fb.binary(BinOp::IAdd, row, column)?;
    let length = arena::tile_length(fb.module())?;
    let length = fb.module().const_value(length)?;
    let two = crate::memory::leaf::k32(fb, 2)?;
    let words = fb.binary(BinOp::ShiftRightLogical, length, two)?;
    fb.binary(BinOp::IMul, index, words)
}

/// The work before a fragment function's body, in its first block: its tile's first word in the
/// pass's buffer, which every access of its tile memory reads (`EmitCx::tile_base`).
pub(crate) fn emit_entry(cx: &mut EmitCx<'_>, fb: &mut FunctionBuilder<'_>) -> BResult<()> {
    let tile = cx
        .decl
        .families
        .access
        .local
        .tg
        .as_ref()
        .and_then(|t| t.fragment);
    if let Some(tile) = tile {
        cx.tile_base(fb, tile)?;
    }
    Ok(())
}
