//! A device replay of GPU-encoded draws merged into one draw ([`DrawReplay::merged`]): an
//! execution whose commands all draw one list type or points runs as one indirect draw of every
//! command's vertices in the commands' order, whatever the GPU wrote, with no draw per command.
//!
//! - **What a command draws.** A command's draw of `n` vertices and `k` instances draws `k` times
//!   its whole primitives' vertices: `n - n mod c`, `c` the vertices per primitive (3 for
//!   triangles, 2 for lines, 1 for points), which Metal draws of it (a list's trailing vertices
//!   that make no primitive draw nothing). Its **corners** are those vertices times `k`.
//! - **The prefix.** After the draw replay's preprocess wrote each command's
//!   `VkDrawIndirectCommand` (zero for a command that draws nothing), the two merge passes
//!   ([`draw_merge_module`]) write each command's first corner in the merged draw: the corners of
//!   every command before it. The commands are split into [`MergedLayout::blocks`] blocks of about
//!   `√count` commands; [`MergePhase::Blocks`] runs one invocation per block over its commands,
//!   writing each command's offset within its block and the block's corners;
//!   [`MergePhase::Starts`] runs one invocation per block again, which adds the corners of every
//!   block before its own to its commands' offsets. The last block's invocation writes the total
//!   and the merged draw's two `VkDrawIndirectCommand`s.
//! - **The merged draw** is two draws, recorded in this order: the first draws
//!   [`MERGED_CHUNK`] vertices in each of `⌊total / MERGED_CHUNK⌋` instances (none where the total
//!   is below it), the second the remaining `total mod MERGED_CHUNK` vertices from first vertex
//!   [`MERGED_REST_FIRST`] in one instance. An invocation's corner is `InstanceIndex ·
//!   MERGED_CHUNK + VertexIndex` in the first and `⌊total / MERGED_CHUNK⌋ · MERGED_CHUNK +
//!   (VertexIndex - MERGED_REST_FIRST)` in the second (a `VertexIndex` at or past
//!   `MERGED_REST_FIRST` marks it, which the first never reaches), counting from 0 in the
//!   commands' order with no gap. A chunk holds whole primitives of every class (a multiple of 6),
//!   so each primitive of the merged draw is one primitive of one command.
//! - **The vertex variant** (`LoweringInputs::merged_draws`) finds its corner's command as the
//!   last one whose first corner is at or below it (a binary search over the prefix: a command
//!   that draws nothing has no corner, so none lands in it), and from the corner's place in it the
//!   instance and the vertex: Metal's `[[vertex_id]]` is the command's vertex start plus the
//!   vertex, `[[instance_id]]` its base instance plus the instance, `[[base_instance]]` and
//!   `[[base_vertex]]` its base instance and vertex start, as the per-command draws of a draw
//!   replay give them; its buffers are in the command's vertex slot table. It writes the
//!   command's index to a flat `uint` varying (`IoRole::MergedCommand`), the same at every vertex
//!   of a primitive.
//! - **Indexed commands** (`DrawReplay::indexed`). A command's index entry
//!   ([`DRAW_INDEX_ENTRY_BYTES`], which the preprocess writes) holds the device address of its
//!   first index, the whole indices from there to its index buffer's end and the index size (0 for
//!   a draw without indices). An indexed command's vertex is the index at the vertex's position
//!   plus its base vertex, wrapping as `u32`s (`[[base_vertex]]` is the base vertex); a position
//!   past the buffer's whole indices reads index 0, as the encoder's indexed draw does. Its
//!   corners are its index count's whole primitives in each instance. A list does not restart at
//!   the all-ones index (a strip, which Metal restarts, is not merged), so nothing else reads
//!   the indices.
//! - **The fragment variant** of a fragment function that reads a command's buffers reads its
//!   slot table from the header: the command's tables at its flat `IoRole::MergedCommand`
//!   varying, its fragment table past the vertex table (`LoweringInputs::merged_draws`).
//! - **Order.** Vulkan rasterizes one draw's primitives in instance order, then in the order
//!   within an instance ("Primitive Order", the Vulkan specification's rasterization chapter), so
//!   the merged draw's order is the commands' order, then each command's instances, then its
//!   primitives: the order of Metal's draws of the commands.
//! - **What it cannot run** is a total of `2^32 · MERGED_CHUNK` corners or more (above `2^62`),
//!   which no two draws hold: the draws are zero and the merge sets
//!   [`StatusBits::REPLAY_FAULT`](crate::bind::StatusBits) with [`ReplayFaults::VERTICES`] in
//!   the command buffer's `ExecutionStatus`.
//! - **The scratch** is the draw replay's ([`DrawReplayLayout::merged`]): the header
//!   ([`MERGED_HEADER_BYTES`], uploaded with the parameters,
//!   [`DrawReplayPreprocess::header`](super::DrawReplayPreprocess::header)),
//!   then the index entries, which the preprocess writes, and the prefix (a `u64` per command),
//!   the block sums (a `u64` per block) and the two draws, which the passes write.
//! - **Recording** (consumer): the preprocess, then each pass in order, each after the previous
//!   one's `SHADER_WRITE` (`COMPUTE_SHADER`, `SHADER_READ | SHADER_WRITE`), the draws after the
//!   last pass's (`DRAW_INDIRECT`, `INDIRECT_COMMAND_READ`; `VERTEX_SHADER`, `SHADER_READ`). Each
//!   pass is dispatched [`MergedLayout::group_count`] with the header's address and the
//!   `ExecutionStatus` buffer's in its push constants ([`DrawMergeModule`]); the variant's
//!   `MergedDrawAddress` hole (the vertex function's, and the fragment function's where it reads
//!   a command's buffers) takes the header's address.
//! - **Where it runs.** Profile A, as the draw replay. Elsewhere [`draw_merge_module`] names what
//!   is missing (`GpuEncodedIndirectCommands`).

use std::sync::Arc;

use super::draws::{DrawReplay, DrawReplayLayout, DRAW_ARGUMENT_BYTES};
use super::{
    at, count_loop, counter, k32, k64, load32, load64, report, store32, store64, wide, Types,
};
use crate::bind::status::ReplayFaults;
use crate::budget::{Budget, Meter};
use crate::device::profile::BufferAccess;
use crate::device::Device;
use crate::error::{Error, Feature, InternalError, InternalKind, Unsupported};
use crate::pipeline::layout::{PipelineLayout, PushConstantRange};
use crate::pipeline::sym::{PushField, Symbol};
use crate::spirv::builder::constructs::{ArmEnd, ArmFn, LoopFn, LoopSpec};
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
use crate::spirv::error::BResult;
use crate::spirv::ids::Value;
use crate::spirv::ir::{
    BinOp, ConvOp, ExecMode, ExecModel, IoDir, MemAccess, Type, TypedModule, UnOp,
};
use crate::vk;

/// Vertices per instance of a merged draw's first draw: below `2^31`, and a multiple of 6, so it
/// holds whole primitives of every class.
pub const MERGED_CHUNK: u32 = (1 << 31) - 2;
/// The first vertex of a merged draw's second draw, which draws what the first's whole chunks
/// leave: a `VertexIndex` at or past it is the second draw's.
pub const MERGED_REST_FIRST: u32 = 1 << 31;
/// Bytes of a merged draw's header.
pub const MERGED_HEADER_BYTES: u64 = 56;
/// Bytes of one command's index entry (the module docs): the device address of its first index
/// (`u64`), the whole indices from there to its index buffer's end (`u32`) and the index size
/// (`u32`: 2, 4, or 0 for a draw without indices).
pub(super) const DRAW_INDEX_ENTRY_BYTES: u64 = 16;

// The header's words, in bytes from its start. The consumer writes all but `H_CHUNKS`, which the
// last pass writes; every `*_AT` is an offset from the header.
/// The commands.
const H_COUNT: u64 = 0;
/// The commands of a block.
const H_BLOCK: u64 = 4;
/// The blocks.
const H_BLOCKS: u64 = 8;
/// The vertices per primitive of the type the commands draw.
const H_CORNERS: u64 = 12;
/// The vertices of a chunk: [`MERGED_CHUNK`] (below `2^31`, as the last pass's division needs).
const H_CHUNK: u64 = 16;
/// The whole chunks the first draw draws, which the last pass writes.
const H_CHUNKS: u64 = 20;
/// The commands' `VkDrawIndirectCommand`s.
const H_ARGUMENTS_AT: u64 = 24;
/// The prefix.
const H_PREFIX_AT: u64 = 28;
/// The block sums.
const H_SUMS_AT: u64 = 32;
/// The commands' slot tables, and the bytes from one command's to the next.
const H_TABLES_AT: u64 = 36;
const H_TABLE_STRIDE: u64 = 40;
/// The merged draw's two `VkDrawIndirectCommand`s.
const H_DRAWS_AT: u64 = 44;
/// The commands' index entries.
const H_INDICES_AT: u64 = 48;
/// The bytes from a command's vertex slot table to its fragment slot table.
const H_FRAGMENT_TABLE: u64 = 52;

/// Where a merged draw replay's sections lie in the scratch, in bytes from its start.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MergedLayout {
    /// The header ([`DrawReplayPreprocess::header`](super::DrawReplayPreprocess::header)): the
    /// vertex variant's `MergedDrawAddress` and the passes' push constants take the scratch's
    /// address plus this.
    pub header_offset: u64,
    /// Each command's index entry (16 bytes), which the preprocess writes.
    pub indices_offset: u64,
    /// Each command's first corner (a `u64`).
    pub prefix_offset: u64,
    /// Each block's corners (a `u64`).
    pub sums_offset: u64,
    /// The merged draw's two `VkDrawIndirectCommand`s, drawn in order.
    pub draws_offset: u64,
    /// The commands of a block, and the blocks.
    pub block_commands: u32,
    pub blocks: u32,
}

impl MergedLayout {
    /// The group count of each pass: one invocation per block.
    pub fn group_count(&self) -> [u32; 3] {
        [self.blocks.div_ceil(DrawMergeModule::LOCAL_SIZE), 1, 1]
    }

    /// The `index`th of the merged draw's two `VkDrawIndirectCommand`s.
    pub fn draw_offset(&self, index: u32) -> u64 {
        self.draws_offset + u64::from(index) * DRAW_ARGUMENT_BYTES
    }
}

/// The commands of a block of a merged replay of `count` commands: `⌈√count⌉`, so both passes
/// loop about `√count` times per invocation.
pub(super) fn block_commands(count: u32) -> u32 {
    let mut b = (f64::from(count)).sqrt().ceil() as u32;
    while u64::from(b) * u64::from(b) < u64::from(count) {
        b += 1;
    }
    b.max(1)
}

/// The header of a merged replay of `replay` laid out as `layout` (whose `merged` is `Some`),
/// drawing `corners` vertices per primitive, with chunks of `chunk` vertices ([`MERGED_CHUNK`]).
pub(super) fn header(
    replay: &DrawReplay,
    layout: &DrawReplayLayout,
    m: &MergedLayout,
    (corners, chunk): (u32, u32),
) -> Vec<u8> {
    let rel = |offset: u64| (offset - m.header_offset) as u32;
    let words = [
        replay.count,
        m.block_commands,
        m.blocks,
        corners,
        chunk,
        0,
        rel(layout.arguments_offset),
        rel(m.prefix_offset),
        rel(m.sums_offset),
        rel(layout.tables_offset),
        layout
            .vertex_table_offset(1)
            .wrapping_sub(layout.vertex_table_offset(0)) as u32,
        rel(m.draws_offset),
        rel(m.indices_offset),
        layout
            .fragment_table_offset(0)
            .wrapping_sub(layout.vertex_table_offset(0)) as u32,
    ];
    let mut out = vec![0u8; MERGED_HEADER_BYTES as usize];
    for (w, o) in words.iter().zip(out.as_chunks_mut::<4>().0.iter_mut()) {
        o.copy_from_slice(&w.to_le_bytes());
    }
    out
}

/// The vertices per primitive of `primitive_type` (an `MTLPrimitiveType` raw value) where a
/// merged replay draws it: a list type or points; `None` for a strip.
pub(super) fn corners_per_primitive(primitive_type: u32) -> Option<u32> {
    match primitive_type {
        0 => Some(1),
        1 => Some(2),
        3 => Some(3),
        _ => None,
    }
}

/// Which of a merged replay's two passes a [`DrawMergeModule`] runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MergePhase {
    /// Each block's commands' offsets within the block, and the block's corners.
    Blocks,
    /// Each command's first corner, and the merged draw.
    Starts,
}

/// A crate-owned merge pass module of one device, validated like any output.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DrawMergeModule {
    /// SPIR-V words, entry point `main`, no descriptor.
    pub spirv: Arc<[u32]>,
    /// What the module needs from the device (device addresses and `shaderInt64`).
    pub requirements: crate::device::requirements::Requirements,
    /// The pipeline layout the consumer creates the pass's pipeline with: no descriptor set, and
    /// one `COMPUTE` push-constant range of [`DrawMergeModule::PUSH_BYTES`] at offset 0.
    pub layout: PipelineLayout,
}

impl DrawMergeModule {
    /// Bytes of the `COMPUTE` push-constant range at offset 0.
    pub const PUSH_BYTES: u32 = 16;
    /// The header's device address (`u64`).
    pub const HEADER_PUSH: u32 = 0;
    /// The `ExecutionStatus` buffer's device address (`u64`).
    pub const STATUS_PUSH: u32 = 8;
    /// The invocations of one workgroup: each one block.
    pub const LOCAL_SIZE: u32 = 64;

    /// The push bytes for the header at `header` and the `ExecutionStatus` buffer at `status`.
    pub fn push_constants(header: u64, status: u64) -> Vec<u8> {
        let mut out = header.to_le_bytes().to_vec();
        out.extend_from_slice(&status.to_le_bytes());
        out
    }
}

/// The merge pass `phase` module of `device`: built with the typed builder, checked by the owned
/// verifier against the device's profile (requirements included), and serialized. A device
/// without device addresses is `Unsupported{GpuEncodedIndirectCommands}` naming what it lacks.
pub fn draw_merge_module(device: &Device, phase: MergePhase) -> Result<DrawMergeModule, Error> {
    let profile = device.profile();
    if profile.fields().buffer_access != BufferAccess::Addressed {
        return Err(Unsupported::needs_any_of(
            Feature::GpuEncodedIndirectCommands,
            vec![profile.addressed_needs()],
        )
        .into());
    }
    let m = typed_module(phase)?;
    let mut meter = Meter::new(&Budget::default());
    crate::spirv::verify::verify(&m, profile, &mut meter)?;
    let r = crate::spirv::serialize::serialize(m)?;
    profile.check(&r.requirements).map_err(|c| {
        Error::Internal(InternalError::new(
            InternalKind::RequirementsExceedProfile,
            format!("the draw merge pass needs {c:?}"),
        ))
    })?;
    let words = crate::spirv::serialize::relocate(&r, &no_binding)?;
    Ok(DrawMergeModule {
        spirv: words.into(),
        requirements: r.requirements,
        layout: PipelineLayout {
            sets: Vec::new(),
            push_constants: vec![PushConstantRange {
                stages: vk::ShaderStageFlags::COMPUTE,
                offset: 0,
                size: DrawMergeModule::PUSH_BYTES,
            }],
            push_descriptor_eligible: false,
        },
    })
}

/// The module declares no binding.
fn no_binding(_: &Symbol) -> Option<u32> {
    None
}

/// `a + b` (`u64`s), held at `u64::MAX` where it would wrap.
fn saturating_add(fb: &mut FunctionBuilder<'_>, a: Value, b: Value) -> BResult<Value> {
    let s = fb.binary(BinOp::IAdd, a, b)?;
    let wrapped = fb.binary(BinOp::ULessThan, s, a)?;
    let most = k64(fb, u64::MAX)?;
    fb.select(wrapped, most, s)
}

/// The corners of the command whose `VkDrawIndirectCommand` is at `args`, drawing `corners`
/// vertices per primitive: its whole primitives' vertices times its instances, a `u64`.
fn command_corners(
    fb: &mut FunctionBuilder<'_>,
    t: Types,
    args: Value,
    corners: Value,
) -> BResult<Value> {
    let vertices = load32(fb, t, args)?;
    let instances_at = at(fb, args, 4)?;
    let instances = load32(fb, t, instances_at)?;
    let whole = whole_vertices(fb, vertices, corners)?;
    let whole = wide(fb, t, whole)?;
    let instances = wide(fb, t, instances)?;
    fb.binary(BinOp::IMul, whole, instances)
}

/// `⌊n / d⌋` and `n mod d` (`u32`s) of the `u64` `n` whose words are `(high, low)`, for a `d`
/// of 1 to `2^31 - 1` above `high` (so the quotient is below `2^32`): restoring division, a bit
/// of `low` at a time, in `u32`s alone. The last pass divides so, not by `u64` `OpUDiv` and
/// `OpUMod`: with those, an RTX 5080 laptop GPU under NVIDIA's 615.71.09 ran the pass with the
/// chunk as the total's high word (a total of 31 corners drew nothing and reported
/// `ReplayFaults::VERTICES`), and in `u32` words it runs exact.
fn divide_words(
    fb: &mut FunctionBuilder<'_>,
    t: Types,
    (high, low): (Value, Value),
    d: Value,
) -> BResult<(Value, Value)> {
    let remainder = fb.local_var(t.u32_, None)?;
    let remainder = fb.var_ptr(remainder)?;
    fb.store(remainder, high, MemAccess::default())?;
    let quotient = fb.local_var(t.u32_, None)?;
    let quotient = fb.var_ptr(quotient)?;
    let zero = k32(fb, 0)?;
    fb.store(quotient, zero, MemAccess::default())?;
    let bits = k32(fb, 32)?;
    count_loop(fb, t, bits, move |fb, i| {
        // Bit `31 - i` of `low`, from the top.
        let top = k32(fb, 31)?;
        let at = fb.binary(BinOp::ISub, top, i)?;
        let bit = fb.binary(BinOp::ShiftRightLogical, low, at)?;
        let one = k32(fb, 1)?;
        let bit = fb.binary(BinOp::BitwiseAnd, bit, one)?;
        // The remainder stays below `d`, below `2^31`: shifted, it fits a `u32`.
        let r = fb.load(remainder, MemAccess::default())?;
        let r = fb.binary(BinOp::ShiftLeftLogical, r, one)?;
        let r = fb.binary(BinOp::BitwiseOr, r, bit)?;
        let taken = fb.binary(BinOp::UGreaterThanEqual, r, d)?;
        let less = fb.binary(BinOp::ISub, r, d)?;
        let r = fb.select(taken, less, r)?;
        fb.store(remainder, r, MemAccess::default())?;
        let q = fb.load(quotient, MemAccess::default())?;
        let q = fb.binary(BinOp::ShiftLeftLogical, q, one)?;
        let zero = k32(fb, 0)?;
        let digit = fb.select(taken, one, zero)?;
        let q = fb.binary(BinOp::BitwiseOr, q, digit)?;
        fb.store(quotient, q, MemAccess::default())
    })?;
    let q = fb.load(quotient, MemAccess::default())?;
    let r = fb.load(remainder, MemAccess::default())?;
    Ok((q, r))
}

/// `vertices - vertices mod corners` (`u32`s): the vertices of a draw's whole primitives.
fn whole_vertices(fb: &mut FunctionBuilder<'_>, vertices: Value, corners: Value) -> BResult<Value> {
    let left = fb.binary(BinOp::UMod, vertices, corners)?;
    fb.binary(BinOp::ISub, vertices, left)
}

/// What a merged draw's vertex variant runs for one invocation ([`merged_vertex`]): Metal's
/// `[[vertex_id]]`, `[[instance_id]]`, `[[base_instance]]` and `[[base_vertex]]` (`u32`s), its
/// command's vertex slot table's device address (`u64`), and the command's index (`u32`), which
/// it writes to its `IoRole::MergedCommand` varying.
#[derive(Clone, Copy, Debug)]
pub(crate) struct MergedVertex {
    pub(crate) vertex: Value,
    pub(crate) instance: Value,
    pub(crate) base_instance: Value,
    pub(crate) base_vertex: Value,
    pub(crate) table: Value,
    pub(crate) command: Value,
}

/// The vertex an invocation of a merged draw's vertex variant runs (the module docs), from the
/// header at `header` (`u64`) and its `VertexIndex` and `InstanceIndex` (`u32`s), in the open
/// block: its corner, the command whose corners hold it (a binary search over the prefix, about
/// `log2(count)` iterations), and the instance and vertex of the corner in that command. Its loop
/// leaves the open block at its merge.
pub(crate) fn merged_vertex(
    fb: &mut FunctionBuilder<'_>,
    header: Value,
    (vertex_index, instance_index): (Value, Value),
) -> BResult<MergedVertex> {
    let u32_ = fb.module().ty(Type::Int {
        width: 32,
        signed: false,
    })?;
    let u64_ = fb.module().ty(Type::Int {
        width: 64,
        signed: false,
    })?;
    let t = Types { u32_, u64_ };
    let word = |fb: &mut FunctionBuilder<'_>, offset: u64| -> BResult<Value> {
        let a = at(fb, header, offset)?;
        load32(fb, t, a)
    };
    let section = |fb: &mut FunctionBuilder<'_>, offset: u64| -> BResult<Value> {
        let rel = word(fb, offset)?;
        let rel = wide(fb, t, rel)?;
        fb.binary(BinOp::IAdd, header, rel)
    };
    let chunk = word(fb, H_CHUNK)?;
    let chunk = wide(fb, t, chunk)?;
    let chunks = word(fb, H_CHUNKS)?;
    let rest_first = k32(fb, MERGED_REST_FIRST)?;
    let second = fb.binary(BinOp::UGreaterThanEqual, vertex_index, rest_first)?;
    let rest_vertex = fb.binary(BinOp::ISub, vertex_index, rest_first)?;
    let instance = fb.select(second, chunks, instance_index)?;
    let vertex = fb.select(second, rest_vertex, vertex_index)?;
    let instance = wide(fb, t, instance)?;
    let vertex = wide(fb, t, vertex)?;
    let corner = fb.binary(BinOp::IMul, instance, chunk)?;
    let corner = fb.binary(BinOp::IAdd, corner, vertex)?;

    // The last command whose first corner is at or below the corner: `lo` over `len` commands,
    // halved until one is left.
    let count = word(fb, H_COUNT)?;
    let prefix = section(fb, H_PREFIX_AT)?;
    let lo = counter(fb, t, 0)?;
    let len = counter(fb, t, 0)?;
    let zero = k32(fb, 0)?;
    fb.store(lo, zero, MemAccess::default())?;
    fb.store(len, count, MemAccess::default())?;
    let first_of = move |fb: &mut FunctionBuilder<'_>, c: Value| -> BResult<Value> {
        let c64 = wide(fb, t, c)?;
        let eight = k64(fb, 8)?;
        let rel = fb.binary(BinOp::IMul, c64, eight)?;
        let a = fb.binary(BinOp::IAdd, prefix, rel)?;
        load64(fb, t, a)
    };
    let searched: LoopFn<'_, ()> = Box::new(move |fb, ctx, lc| {
        let l = fb.load(len, MemAccess::default())?;
        let one = k32(fb, 1)?;
        let done = fb.binary(BinOp::ULessThanEqual, l, one)?;
        let exit: ArmFn<'_, ()> = Box::new(move |fb, _| {
            lc.break_(fb)?;
            Ok(ArmEnd::Exits)
        });
        fb.if_(ctx, done, exit, None)?;
        let shift = k32(fb, 1)?;
        let half = fb.binary(BinOp::ShiftRightLogical, l, shift)?;
        let low = fb.load(lo, MemAccess::default())?;
        let mid = fb.binary(BinOp::IAdd, low, half)?;
        let at_mid = first_of(fb, mid)?;
        let inside = fb.binary(BinOp::ULessThanEqual, at_mid, corner)?;
        let low = fb.select(inside, mid, low)?;
        fb.store(lo, low, MemAccess::default())?;
        let l = fb.binary(BinOp::ISub, l, half)?;
        fb.store(len, l, MemAccess::default())
    });
    fb.loop_(&mut (), LoopSpec::default(), searched)?;
    let command = fb.load(lo, MemAccess::default())?;
    let first = first_of(fb, command)?;
    let within = fb.binary(BinOp::ISub, corner, first)?;

    // The command's draw: its whole primitives' vertices per instance.
    let arguments = section(fb, H_ARGUMENTS_AT)?;
    let command64 = wide(fb, t, command)?;
    let bytes = k64(fb, DRAW_ARGUMENT_BYTES)?;
    let rel = fb.binary(BinOp::IMul, command64, bytes)?;
    let args = fb.binary(BinOp::IAdd, arguments, rel)?;
    let vertices = load32(fb, t, args)?;
    let start_at = at(fb, args, 8)?;
    let vertex_start = load32(fb, t, start_at)?;
    let base_at = at(fb, args, 12)?;
    let base_instance = load32(fb, t, base_at)?;
    let corners = word(fb, H_CORNERS)?;
    let per_instance = whole_vertices(fb, vertices, corners)?;
    let per_instance = wide(fb, t, per_instance)?;
    let instance = fb.binary(BinOp::UDiv, within, per_instance)?;
    let vertex = fb.binary(BinOp::UMod, within, per_instance)?;
    let instance = fb.convert(ConvOp::UConvert, instance, u32_)?;
    let vertex = fb.convert(ConvOp::UConvert, vertex, u32_)?;
    let tables = section(fb, H_TABLES_AT)?;
    let stride = word(fb, H_TABLE_STRIDE)?;
    let stride = wide(fb, t, stride)?;
    let rel = fb.binary(BinOp::IMul, command64, stride)?;
    let table = fb.binary(BinOp::IAdd, tables, rel)?;

    // The command's indices: an indexed command's vertex is the index at the vertex's position
    // plus its base vertex, the first vertex of its arguments; a position past its whole indices
    // reads 0. A `u16` index is read from the aligned word that holds it.
    let indices = section(fb, H_INDICES_AT)?;
    let entry_bytes = k64(fb, DRAW_INDEX_ENTRY_BYTES)?;
    let rel = fb.binary(BinOp::IMul, command64, entry_bytes)?;
    let entry = fb.binary(BinOp::IAdd, indices, rel)?;
    let address = load64(fb, t, entry)?;
    let limit_at = at(fb, entry, 8)?;
    let limit = load32(fb, t, limit_at)?;
    let size_at = at(fb, entry, 12)?;
    let size = load32(fb, t, size_at)?;
    let zero = k32(fb, 0)?;
    let indexed = fb.binary(BinOp::INotEqual, size, zero)?;
    let inside = fb.binary(BinOp::ULessThan, vertex, limit)?;
    let reads = fb.binary(BinOp::LogicalAnd, indexed, inside)?;
    let raw = counter(fb, t, 0)?;
    let read: ArmFn<'_, ()> = Box::new(move |fb, _| {
        let vertex64 = wide(fb, t, vertex)?;
        let size64 = wide(fb, t, size)?;
        let rel = fb.binary(BinOp::IMul, vertex64, size64)?;
        let byte = fb.binary(BinOp::IAdd, address, rel)?;
        let low = k64(fb, !3)?;
        let word_at = fb.binary(BinOp::BitwiseAnd, byte, low)?;
        let w = load32(fb, t, word_at)?;
        let three = k64(fb, 3)?;
        let within = fb.binary(BinOp::BitwiseAnd, byte, three)?;
        let within = fb.convert(ConvOp::UConvert, within, u32_)?;
        let eight = k32(fb, 8)?;
        let shift = fb.binary(BinOp::IMul, within, eight)?;
        let w = fb.binary(BinOp::ShiftRightLogical, w, shift)?;
        let four = k32(fb, 4)?;
        let whole = fb.binary(BinOp::IEqual, size, four)?;
        let (all, half) = (k32(fb, u32::MAX)?, k32(fb, 0xFFFF)?);
        let mask = fb.select(whole, all, half)?;
        let index = fb.binary(BinOp::BitwiseAnd, w, mask)?;
        fb.store(raw, index, MemAccess::default())?;
        Ok(ArmEnd::FallThrough(Vec::new()))
    });
    fb.if_(&mut (), reads, read, None)?;
    let raw = fb.load(raw, MemAccess::default())?;
    let position = fb.select(indexed, raw, vertex)?;
    Ok(MergedVertex {
        vertex: fb.binary(BinOp::IAdd, vertex_start, position)?,
        instance: fb.binary(BinOp::IAdd, base_instance, instance)?,
        base_instance,
        base_vertex: vertex_start,
        table,
        command,
    })
}

/// The device address of the fragment slot table of the command whose index is `command` (a
/// `u32`, the fragment variant's flat `IoRole::MergedCommand` input), from the header at `header`
/// (`u64`), in the open block (the module docs).
pub(crate) fn merged_fragment_table(
    fb: &mut FunctionBuilder<'_>,
    header: Value,
    command: Value,
) -> BResult<Value> {
    let u32_ = fb.module().ty(Type::Int {
        width: 32,
        signed: false,
    })?;
    let u64_ = fb.module().ty(Type::Int {
        width: 64,
        signed: false,
    })?;
    let t = Types { u32_, u64_ };
    let word = |fb: &mut FunctionBuilder<'_>, offset: u64| -> BResult<Value> {
        let a = at(fb, header, offset)?;
        load32(fb, t, a)
    };
    let tables = word(fb, H_TABLES_AT)?;
    let tables = wide(fb, t, tables)?;
    let tables = fb.binary(BinOp::IAdd, header, tables)?;
    let stride = word(fb, H_TABLE_STRIDE)?;
    let stride = wide(fb, t, stride)?;
    let command = wide(fb, t, command)?;
    let rel = fb.binary(BinOp::IMul, command, stride)?;
    let table = fb.binary(BinOp::IAdd, tables, rel)?;
    let fragment = word(fb, H_FRAGMENT_TABLE)?;
    let fragment = wide(fb, t, fragment)?;
    fb.binary(BinOp::IAdd, table, fragment)
}

/// The pass `phase` ([`draw_merge_module`]; the module docs).
pub fn typed_module(phase: MergePhase) -> BResult<TypedModule> {
    let mut b = ModuleBuilder::new();
    let u32_ = b.ty(Type::Int {
        width: 32,
        signed: false,
    })?;
    let u64_ = b.ty(Type::Int {
        width: 64,
        signed: false,
    })?;
    let t = Types { u32_, u64_ };
    let uvec3 = b.ty(Type::Vector { elem: u32_, len: 3 })?;
    let m_header = b.push_member(
        PushField::MergedDrawAddress,
        u64_,
        DrawMergeModule::HEADER_PUSH,
    )?;
    let m_status = b.push_member(
        PushField::ExecutionStatusAddress,
        u64_,
        DrawMergeModule::STATUS_PUSH,
    )?;
    let push = b.push_global()?;
    let invocation = b.builtin(crate::spv::BuiltIn::GlobalInvocationId, IoDir::Input, uvec3)?;
    let void = b.ty(Type::Void)?;
    let mut f = b.function(void)?;
    let fb = &mut f;

    let push_value = fb.module().global_value(push)?;
    let field = |fb: &mut FunctionBuilder<'_>, index: u32| -> BResult<Value> {
        let i = k32(fb, index)?;
        let p = fb.access_chain(push_value, &[i])?;
        fb.load(p, MemAccess::default())
    };
    let header = field(fb, m_header.index)?;
    let status = field(fb, m_status.index)?;
    let id = fb.module().global_value(invocation)?;
    let id = fb.load(id, MemAccess::default())?;
    let block = fb.composite_extract(id, &[0])?;
    let word = |fb: &mut FunctionBuilder<'_>, offset: u64| -> BResult<Value> {
        let a = at(fb, header, offset)?;
        load32(fb, t, a)
    };
    let section = |fb: &mut FunctionBuilder<'_>, offset: u64| -> BResult<Value> {
        let rel = word(fb, offset)?;
        let rel = wide(fb, t, rel)?;
        fb.binary(BinOp::IAdd, header, rel)
    };
    let count = word(fb, H_COUNT)?;
    let block_commands = word(fb, H_BLOCK)?;
    let blocks = word(fb, H_BLOCKS)?;
    let corners = word(fb, H_CORNERS)?;
    let arguments = section(fb, H_ARGUMENTS_AT)?;
    let prefix = section(fb, H_PREFIX_AT)?;
    let sums = section(fb, H_SUMS_AT)?;

    // An invocation past the blocks does nothing.
    let past = fb.binary(BinOp::UGreaterThanEqual, block, blocks)?;
    let leave: ArmFn<'_, ()> = Box::new(|fb, _| {
        fb.ret(None)?;
        Ok(ArmEnd::Exits)
    });
    fb.if_(&mut (), past, leave, None)?;

    // The block's commands: `first .. first + n`, `n` up to `block_commands`.
    let first = fb.binary(BinOp::IMul, block, block_commands)?;
    let left = fb.binary(BinOp::ISub, count, first)?;
    let short = fb.binary(BinOp::ULessThan, left, block_commands)?;
    let n = fb.select(short, left, block_commands)?;
    let eight = k64(fb, 8)?;
    let block64 = wide(fb, t, block)?;
    let sum_rel = fb.binary(BinOp::IMul, block64, eight)?;
    let sum_at = fb.binary(BinOp::IAdd, sums, sum_rel)?;
    let running = fb.local_var(u64_, None)?;
    let running = fb.var_ptr(running)?;
    let command_at = move |fb: &mut FunctionBuilder<'_>, i: Value| -> BResult<(Value, Value)> {
        let c = fb.binary(BinOp::IAdd, first, i)?;
        let c64 = wide(fb, t, c)?;
        let eight = k64(fb, 8)?;
        let rel = fb.binary(BinOp::IMul, c64, eight)?;
        let offset_at = fb.binary(BinOp::IAdd, prefix, rel)?;
        let bytes = k64(fb, DRAW_ARGUMENT_BYTES)?;
        let rel = fb.binary(BinOp::IMul, c64, bytes)?;
        let args = fb.binary(BinOp::IAdd, arguments, rel)?;
        Ok((offset_at, args))
    };
    match phase {
        MergePhase::Blocks => {
            let zero = k64(fb, 0)?;
            fb.store(running, zero, MemAccess::default())?;
            count_loop(fb, t, n, move |fb, i| {
                let (offset_at, args) = command_at(fb, i)?;
                let r = fb.load(running, MemAccess::default())?;
                store64(fb, t, offset_at, r)?;
                let mine = command_corners(fb, t, args, corners)?;
                let r = saturating_add(fb, r, mine)?;
                fb.store(running, r, MemAccess::default())
            })?;
            let r = fb.load(running, MemAccess::default())?;
            store64(fb, t, sum_at, r)?;
        }
        MergePhase::Starts => {
            // The block's first corner: the corners of every block before it.
            let zero = k64(fb, 0)?;
            fb.store(running, zero, MemAccess::default())?;
            count_loop(fb, t, block, move |fb, i| {
                let i64_ = wide(fb, t, i)?;
                let eight = k64(fb, 8)?;
                let rel = fb.binary(BinOp::IMul, i64_, eight)?;
                let a = fb.binary(BinOp::IAdd, sums, rel)?;
                let s = load64(fb, t, a)?;
                let r = fb.load(running, MemAccess::default())?;
                let r = saturating_add(fb, r, s)?;
                fb.store(running, r, MemAccess::default())
            })?;
            let start = fb.load(running, MemAccess::default())?;
            count_loop(fb, t, n, move |fb, i| {
                let (offset_at, _) = command_at(fb, i)?;
                let local = load64(fb, t, offset_at)?;
                let global = saturating_add(fb, local, start)?;
                store64(fb, t, offset_at, global)
            })?;
            // The last block's invocation writes the total's draws.
            let one = k32(fb, 1)?;
            let last = fb.binary(BinOp::ISub, blocks, one)?;
            let is_last = fb.binary(BinOp::IEqual, block, last)?;
            let draws: ArmFn<'_, ()> = Box::new(move |fb, _| {
                let own = load64(fb, t, sum_at)?;
                let total = saturating_add(fb, start, own)?;
                let chunk = word(fb, H_CHUNK)?;
                // The whole chunks fit in a `u32` where the total's high word is below a chunk.
                let low = fb.convert(ConvOp::UConvert, total, u32_)?;
                let shift = k64(fb, 32)?;
                let high = fb.binary(BinOp::ShiftRightLogical, total, shift)?;
                let high = fb.convert(ConvOp::UConvert, high, u32_)?;
                let fits = fb.binary(BinOp::ULessThan, high, chunk)?;
                let (chunks32, rest32) = divide_words(fb, t, (high, low), chunk)?;
                let zero = k32(fb, 0)?;
                let one = k32(fb, 1)?;
                let rest_first = k32(fb, MERGED_REST_FIRST)?;
                let draws = section(fb, H_DRAWS_AT)?;
                let chunks_at = at(fb, header, H_CHUNKS)?;
                let kept_chunks = fb.select(fits, chunks32, zero)?;
                store32(fb, t, chunks_at, kept_chunks)?;
                let words = [
                    [chunk, chunks32, zero, zero],
                    [rest32, one, rest_first, zero],
                ];
                for (d, w) in words.into_iter().enumerate() {
                    for (i, v) in w.into_iter().enumerate() {
                        let v = fb.select(fits, v, zero)?;
                        let a = at(fb, draws, d as u64 * DRAW_ARGUMENT_BYTES + 4 * i as u64)?;
                        store32(fb, t, a, v)?;
                    }
                }
                let refused = fb.unary(UnOp::LogicalNot, fits)?;
                let reported: ArmFn<'_, ()> = Box::new(move |fb, _| {
                    let vertices = k32(fb, ReplayFaults::VERTICES.0)?;
                    report(fb, t, status, vertices)?;
                    Ok(ArmEnd::FallThrough(Vec::new()))
                });
                fb.if_(&mut (), refused, reported, None)?;
                Ok(ArmEnd::FallThrough(Vec::new()))
            });
            fb.if_(&mut (), is_last, draws, None)?;
        }
    }
    fb.ret(None)?;
    let id = f.finish()?;
    b.entry(
        ExecModel::GLCompute,
        id,
        vec![ExecMode::LocalSize {
            x: DrawMergeModule::LOCAL_SIZE,
            y: 1,
            z: 1,
        }],
    )?;
    b.finalize()
}
