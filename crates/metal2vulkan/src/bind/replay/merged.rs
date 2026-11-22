//! What the variants of a merged draw read: GPU-encoded draws of one list type or points run as one
//! indirect draw of every command's vertices in the commands' order, with no draw per command. The
//! consumer writes the header and what it points to; this module builds the variants' reads.
//!
//! - **What a command draws.** A command's draw of `n` vertices and `k` instances draws `k` times
//!   its whole primitives' vertices: `n - n mod c`, `c` the vertices per primitive (3 for
//!   triangles, 2 for lines, 1 for points), which Metal draws of it (a list's trailing vertices
//!   that make no primitive draw nothing). Its **corners** are those vertices times `k`, and the
//!   prefix holds each command's first corner in the merged draw (a `u64` per command): the corners
//!   of every command before it.
//! - **The merged draw** is two draws: the first draws the header's chunk of vertices in each of
//!   its whole chunks' instances, the second the remaining vertices from first vertex
//!   [`MERGED_REST_FIRST`] in one instance. An invocation's corner is `InstanceIndex · chunk +
//!   VertexIndex` in the first and `chunks · chunk + (VertexIndex - MERGED_REST_FIRST)` in the
//!   second (a `VertexIndex` at or past `MERGED_REST_FIRST` marks it, which the first never
//!   reaches), counting from 0 in the commands' order with no gap. A chunk holds whole primitives
//!   of every class (a multiple of 6), so each primitive of the merged draw is one primitive of one
//!   command.
//! - **The vertex variant** (`LoweringInputs::merged_draws`, [`merged_vertex`]) finds its corner's
//!   command as the last one whose first corner is at or below it (a binary search over the
//!   prefix: a command that draws nothing has no corner, so none lands in it), and from the
//!   corner's place in it the instance and the vertex: Metal's `[[vertex_id]]` is the command's
//!   vertex start plus the vertex, `[[instance_id]]` its base instance plus the instance,
//!   `[[base_instance]]` and `[[base_vertex]]` its base instance and vertex start, read from the
//!   command's `VkDrawIndirectCommand` ([`DRAW_ARGUMENT_BYTES`]); its buffers are in the command's
//!   vertex slot table. It writes the command's index to a flat `uint` varying
//!   (`IoRole::MergedCommand`), the same at every vertex of a primitive.
//! - **Indexed commands.** A command's index entry ([`DRAW_INDEX_ENTRY_BYTES`]) holds the device
//!   address of its first index, the whole indices from there to its index buffer's end and the
//!   index size (0 for a draw without indices). An indexed command's vertex is the index at the
//!   vertex's position plus its base vertex, wrapping as `u32`s (`[[base_vertex]]` is the base
//!   vertex); a position past the buffer's whole indices reads index 0, as the encoder's indexed
//!   draw does. Its corners are its index count's whole primitives in each instance.
//! - **The fragment variant** of a fragment function that reads a command's buffers reads its
//!   slot table from the header ([`merged_fragment_table`]): the command's tables at its flat
//!   `IoRole::MergedCommand` varying, its fragment table past the vertex table.
//! - **Order.** Vulkan rasterizes one draw's primitives in instance order, then in the order
//!   within an instance ("Primitive Order", the Vulkan specification's rasterization chapter), so
//!   the merged draw's order is the commands' order, then each command's instances, then its
//!   primitives: the order of Metal's draws of the commands.
//! - **The header** (its words below) is at the variants' `MergedDrawAddress` push field (the
//!   vertex function's, and the fragment function's where it reads a command's buffers).

use super::{at, counter, k32, k64, load32, load64, wide, Types};
use crate::spirv::builder::constructs::{ArmEnd, ArmFn, LoopFn, LoopSpec};
use crate::spirv::builder::FunctionBuilder;
use crate::spirv::error::BResult;
use crate::spirv::ids::Value;
use crate::spirv::ir::{BinOp, ConvOp, MemAccess, Type};

/// The first vertex of a merged draw's second draw, which draws what the first's whole chunks
/// leave: a `VertexIndex` at or past it is the second draw's.
pub const MERGED_REST_FIRST: u32 = 1 << 31;
/// Bytes of one command's index entry (the module docs), from the merged draw header's index entries
/// in command order: the device address of its first index (`u64`), the whole indices from there
/// to its index buffer's end (`u32`) and the index size (`u32`: 2, 4, or 0 for a draw without
/// indices).
pub const DRAW_INDEX_ENTRY_BYTES: u64 = 16;
/// Bytes of one `VkDrawIndirectCommand`.
pub const DRAW_ARGUMENT_BYTES: u64 = 16;

// The header's words the variants read, in bytes from its start; every `*_AT` is an offset from
// the header.
/// The commands.
const H_COUNT: u64 = 0;
/// The vertices per primitive of the type the commands draw.
const H_CORNERS: u64 = 12;
/// The vertices of a chunk (below `2^31`).
const H_CHUNK: u64 = 16;
/// The whole chunks the first draw draws.
const H_CHUNKS: u64 = 20;
/// The commands' `VkDrawIndirectCommand`s.
const H_ARGUMENTS_AT: u64 = 24;
/// The prefix.
const H_PREFIX_AT: u64 = 28;
/// The commands' slot tables, and the bytes from one command's to the next.
const H_TABLES_AT: u64 = 36;
const H_TABLE_STRIDE: u64 = 40;
/// The commands' index entries.
const H_INDICES_AT: u64 = 48;
/// The bytes from a command's vertex slot table to its fragment slot table.
const H_FRAGMENT_TABLE: u64 = 52;

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
