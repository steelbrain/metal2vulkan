//! The index rewrite pass of a draw whose primitives Vulkan's input assembly does not form as
//! Metal does: one dispatch before the render pass instance reads the draw's Metal indices (or,
//! for an indirect draw without indices, its count) and writes the primitives Metal draws as a
//! list of 32-bit indices, which the draw then draws as a list over its arguments, with primitive
//! restart off.
//!
//! The programs ([`RewriteProgram`], a closed enum: another way of forming primitives from indices
//! is another arm of the same pass, not another pass):
//!
//! - an indexed strip after Apple's `setPrimitiveRestartEnabled:index:`, which restarts at the
//!   call's index `k'` alone, where Vulkan restarts at the index type's all-ones index alone: the
//!   strip's pieces between restarts, each triangle in Vulkan's strip order with its parity
//!   counted from its piece's start (a line strip's segments as pairs), in order, so every
//!   primitive keeps its winding, its provoking vertex and its place in `primitive_id`'s count,
//!   which Metal counts on across a restart (probe 5's `prim_*`). The all-ones index, where it is
//!   not `k'`, is drawn as a vertex;
//! - Apple's private primitive types 5 to 8, which Vulkan has no topology for: a line loop, a
//!   triangle fan, quads and a quad strip, each the list [`RewriteProgram`]'s arms state, in the
//!   order Metal draws it (Metal probes on the M3 Max, macOS 27.0.1): each list primitive's first
//!   vertex is the one Metal's provoking vertex mode 0 takes its flat inputs from, its last the
//!   one mode 1 takes them from, and its order the barycentric order Metal gives at either mode.
//!   The line loop, fan and quads never restart, drawing every index as a vertex; the quad strip
//!   restarts as the strips do, at `k'` (the consumer's restart index: the index type's all-ones
//!   index where the encoder made no call), each piece pairing from its own start. Metal counts
//!   one `primitive_id` per quad, where the list has two triangles: that count is the consumer's.
//! - a line or triangle list ([`RewriteProgram::LineList`], [`RewriteProgram::TriangleList`]),
//!   which never restarts: its primitives in their own order, which a draw under Apple's
//!   `setProvokingVertexMode:` with the last vertex writes in [`RewriteOrder::LastFirst`].
//!
//! **The order** ([`RewriteOrder`], parameter word 14): each primitive in Metal's order at
//! provoking vertex mode 0, the first vertex (the default); in Metal's order at mode 1, the last
//! vertex, which differs only for a triangle strip's odd triangle, `(i + 1, i, i + 2)` where mode 0
//! gives `(i, i + 2, i + 1)` (Metal probes on the M3 Max, macOS 27.0.1: the Vulkan
//! specification's last-vertex strip order); or in Metal's order at mode 1 with the last vertex
//! moved first, `(c, a, b)` of a triangle `(a, b, c)` and `(b, a)` of a line `(a, b)`, so a
//! first-vertex pipeline takes its flat inputs from Metal's provoking vertex.
//!
//! - **Three dispatches** ([`RewritePhase`]) of workgroups of [`IndexRewriteModule::LOCAL_SIZE`]
//!   invocations. The draw's positions are split into one run per invocation of
//!   [`IndexRewritePass::groups`] workgroups. A position completes a step where neither it nor
//!   the [`RewriteProgram::span`] less one positions before it restarts (quads: at every fourth
//!   position; a quad strip: at every second position of its piece), which a run decides from its
//!   own positions and the three before it, so the runs are independent but for two sums over
//!   the runs before each: its primitives (its first primitive's place in the output) and the
//!   start of the piece it begins in (the position after the last restart before it, from which a
//!   triangle's parity and a quad strip's pairs count). A quad strip's run counts its primitives
//!   for both parities of its first piece's positions before it, and the sums take the one the
//!   piece's start gives. [`RewritePhase::Count`] sums each group's runs;
//!   [`RewritePhase::Place`], one workgroup, places each group, reserves the output and writes
//!   the draw; [`RewritePhase::Write`] places each run within its group and writes its
//!   primitives. Each run walks a few dozen positions of a million-position draw, so the walks
//!   spread over the GPU: one workgroup walking every position (the first pass) took 56 ms for a
//!   million positions on an M4.
//! - **Positions** are Metal's: from the arguments' first index, each position inside the
//!   buffer's whole indices from the offset at its index, each past them at index 0, as the fetch
//!   reads it, restart included. Past the buffer every position reads 0, so the walk stops
//!   [`RewriteProgram::span`] positions past the last whole index inside; the rest of the draw's
//!   positions either all restart (`k'` = 0 of a restarting program: no primitive) or complete
//!   primitives of index 0 (a fan's of its first index and two 0s), written without a walk; a line
//!   loop's last segment, from its last position to its first, follows them. A draw without
//!   indices ([`RewriteSource::IndirectVertices`]) reads each position as its own index, walks
//!   them all, and restarts nowhere.
//! - **The output** is either the draw's own region ([`RewriteOutput::Owned`], a direct draw:
//!   its capacity [`RewriteProgram::capacity`] of its count, from index 0), or a reservation in
//!   the command buffer's index pool ([`RewriteOutput::Pooled`], an indirect draw, whose count
//!   only the GPU knows), shared with the index bounds pass (`index_bounds.rs`): the pool's
//!   counter in its first word, bytes reserved from its index region, which starts
//!   [`IndexRewritePass::POOL_HEADER_BYTES`] in. A draw whose primitives its output cannot hold
//!   draws nothing and sets [`StatusBits::INDEX_POOL_OVERFLOW`], which the consumer reports as the
//!   command buffer's error. Nothing is drawn short silently. Of a pooled output, the count
//!   dispatch reads nothing, the place dispatch reads and writes only the counter, and the write
//!   dispatch writes only index bytes, so a consumer may bind the counter for the first two and
//!   another buffer of the pool's bytes for the write, as the index bounds pass's plan and fill
//!   may.
//!
//! **The rewrite words** (one `u32` array of [`IndexRewritePass::BYTES`], a storage buffer, and
//! the indirect source of the draw):
//!
//! | Words | Written by | Contents |
//! |---|---|---|
//! | 0 to 7 | the consumer | [`IndexRewritePass::parameters`]: the arguments' first word in the bound arguments range plus 1 (0 for a direct draw), the index offset's byte in the bound indices range, the whole indices from the offset (saturated to `u32`), 1 for 32-bit indices, `k'`, the output's capacity in indices (owned) or the pool's index bytes (pooled), 1 where pooled, the groups |
//! | 8 to 12 | the consumer | a direct draw's Metal arguments (`MTLDrawIndexedPrimitivesIndirectArguments`: count, instances, first index 0, base vertex, base instance); 0 for an indirect draw, whose arguments are read from the Metal buffer |
//! | 13 | the consumer | 1 for a draw without indices, whose arguments are `MTLDrawPrimitivesIndirectArguments` (count, instances, vertex start, base instance), else 0 |
//! | 14 | the consumer | the order ([`RewriteOrder::word`]): 0 first, 1 last, 2 last first |
//! | 15 | the consumer | 1 for a list drawn in the private list form ([`IndexRewritePass::listed`]): Metal's base vertex in the word before the list, the draw without indices, else 0 |
//! | 16 to 20 | the place dispatch | the draw's `VkDrawIndexedIndirectCommand` ([`IndexRewritePass::DRAW_OFFSET`]), or in the private list form its `VkDrawIndirectCommand` (the list's indices, instances, its first index's word as first vertex, base instance) and a 0 |
//! | 21 | the place dispatch | the primitives written |
//! | 22 to 25 | the place dispatch | the output's first word, 1 where the draw is written, the primitives before those past the walk, and those past it |
//! | 32 on | the count and place dispatches | three words per group: its primitives (where its first piece's positions before it are even), then those before it; its primitives where they are odd, then the start of the piece it begins in (0 for none); the start of its last piece (0 for none) |
//!
//! **Bindings** (set 0, storage buffers): [`IndexRewriteModule::WORDS_BINDING`],
//! `ARGUMENTS_BINDING` ([`IndexRewritePass::arguments_range`], or any 4 bytes for a direct draw),
//! `INDICES_BINDING` ([`IndexRewritePass::indices_range`], or any 4 bytes where it is `None`),
//! `OUTPUT_BINDING` (the draw's region, or the whole pool from its counter) and `STATUS_BINDING`
//! (the `ExecutionStatus` buffer). Every read of the arguments and indices is contained (an index
//! past the bound range reads 0), and every write lies inside the output.
//!
//! **Synchronization** (consumer): the arguments' and indices' writes and the upload of the
//! words → `SHADER_READ` in `COMPUTE_SHADER` before the count dispatch; each dispatch's
//! `SHADER_WRITE` → `SHADER_READ` and `SHADER_WRITE` in `COMPUTE_SHADER` before the next;
//! passes and index bounds plans one after another (each reads and writes the pool's counter);
//! the write dispatch's writes (and the place dispatch's) → `INDEX_READ` in `INDEX_INPUT` and
//! `INDIRECT_COMMAND_READ` in `DRAW_INDIRECT` before the draw.
//!
//! **A run** ([`IndexRewriteRun`], [`index_rewrite_run_module`]): direct indexed draws of one
//! program, order and `k'` over one index buffer, each of one group, take one dispatch for all
//! of them, each draw's three phases in turn in a workgroup of its own, its words a slot of the
//! run's and its list a region of the run's output.

use std::sync::Arc;

use crate::bind::index_bounds::{finished, word_range, IndexBoundsPass};
use crate::bind::instantiate::{BufferRange, IndirectRef};
use crate::bind::replay::checked_mul;
use crate::bind::status::{StatusBits, StatusWord};
use crate::device::requirements::Requirements;
use crate::device::Device;
use crate::error::Error;
use crate::pipeline::sym::{ResourceKind, ResourceSlot, Stage, Symbol, SyntheticBinding};
use crate::spirv::builder::constructs::{ArmEnd, ArmFn, LoopFn, LoopSpec};
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
use crate::spirv::error::BResult;
use crate::spirv::ids::{GlobalId, TypeId, Value};
use crate::spirv::ir::{
    natural_layout, AtomicOp, BinOp, ConstBits, Constant, ExecMode, ExecModel, GlslOp, IoDir,
    LayoutRule, MemAccess, MemoryOrder, Scope, Semantics, StorageClass, Type, TypedModule, UnOp,
};

/// How the pass forms a draw's primitives from its indices: one module each.
///
/// Apple's private primitive types 5 to 8 are programs too, each a list in the order Metal draws
/// it (Metal probes on the M3 Max, macOS 27.0.1, every count from 0 to 12, 16- and 32-bit indices,
/// both provoking vertex modes): barycentric order the list's, the flat vertex its first vertex
/// at mode 0 and its last at mode 1.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum RewriteProgram {
    /// A line strip restarting at `k'` alone: each segment of a piece as a pair.
    LineStrip,
    /// A triangle strip restarting at `k'` alone: each triangle of a piece in Vulkan's strip
    /// order, its parity counted from the piece's start.
    TriangleStrip,
    /// Apple's line loop (primitive type 5): the segments `(i, i + 1)`, then `(last, first)`
    /// where there are two positions or more. It never restarts: every index is a vertex.
    LineLoop,
    /// Apple's triangle fan (6): the triangles `(first, i + 1, i + 2)`. It never restarts.
    TriangleFan,
    /// Apple's quads (7): each four positions `a, b, c, d` the triangles `(a, b, d)` and
    /// `(b, c, d)`, the positions past the last whole four none. It never restarts.
    Quads,
    /// Apple's quad strip (8): each pair `c, d` after a pair `a, b` of a piece the triangles
    /// `(a, b, d)` and `(c, a, d)`, an odd last position none. It restarts at `k'` alone, as the
    /// strips do, each piece pairing from its own start.
    QuadStrip,
    /// A line list: each two positions a segment, an odd last position none. It never restarts.
    LineList,
    /// A triangle list: each three positions a triangle, the positions past the last whole three
    /// none. It never restarts.
    TriangleList,
}

impl RewriteProgram {
    /// Every program.
    pub const ALL: [RewriteProgram; 8] = [
        RewriteProgram::LineStrip,
        RewriteProgram::TriangleStrip,
        RewriteProgram::LineLoop,
        RewriteProgram::TriangleFan,
        RewriteProgram::Quads,
        RewriteProgram::QuadStrip,
        RewriteProgram::LineList,
        RewriteProgram::TriangleList,
    ];

    /// The indices of one primitive of the output list: 2 for lines, 3 for triangles.
    pub fn vertices(self) -> u32 {
        match self {
            RewriteProgram::LineStrip | RewriteProgram::LineLoop | RewriteProgram::LineList => 2,
            _ => 3,
        }
    }

    /// The positions a step of the walk forms primitives over: the last `span` positions of a
    /// piece, through the one that completes them.
    pub fn span(self) -> u32 {
        match self {
            RewriteProgram::LineStrip | RewriteProgram::LineLoop | RewriteProgram::LineList => 2,
            RewriteProgram::TriangleStrip
            | RewriteProgram::TriangleFan
            | RewriteProgram::TriangleList => 3,
            RewriteProgram::Quads | RewriteProgram::QuadStrip => 4,
        }
    }

    /// The list primitives a completing position writes: a quad's two triangles, else one.
    pub fn per_step(self) -> u32 {
        match self {
            RewriteProgram::Quads | RewriteProgram::QuadStrip => 2,
            _ => 1,
        }
    }

    /// Whether the program's pieces end at `k'`: the strips', as Metal restarts them. A line
    /// loop, a fan and quads draw every index as a vertex.
    pub fn restarts(self) -> bool {
        matches!(
            self,
            RewriteProgram::LineStrip | RewriteProgram::TriangleStrip | RewriteProgram::QuadStrip
        )
    }

    /// The most indices the program writes for a draw of `count` positions: those of `count`
    /// positions that no index restarts.
    pub fn capacity(self, count: u64) -> u64 {
        let primitives = match self {
            RewriteProgram::LineStrip => count.saturating_sub(1),
            RewriteProgram::TriangleStrip | RewriteProgram::TriangleFan => count.saturating_sub(2),
            RewriteProgram::LineLoop if count >= 2 => count,
            RewriteProgram::Quads => count / 4 * 2,
            RewriteProgram::QuadStrip if count >= 4 => (count - 2) / 2 * 2,
            RewriteProgram::LineList => count / 2,
            RewriteProgram::TriangleList => count / 3,
            RewriteProgram::LineLoop | RewriteProgram::QuadStrip => 0,
        };
        primitives * u64::from(self.vertices())
    }

    /// The list Metal draws of `values` (each position's index, as the fetch reads it) under
    /// this program, restarting at `restart` where the program restarts, each primitive in
    /// `order`: what the pass writes, on the CPU (a direct draw without indices, whose positions
    /// are its vertices from 0).
    pub fn list(self, values: &[u32], restart: u32, order: RewriteOrder) -> Vec<u32> {
        let mut out = Vec::with_capacity(self.capacity(values.len() as u64) as usize);
        let mut put = |first: &[u32], last: &[u32]| out.extend(order.arrange(first, last));
        if self == RewriteProgram::LineLoop {
            if let [first, .., last] = values {
                for pair in values.windows(2) {
                    put(pair, pair);
                }
                put(&[*last, *first], &[*last, *first]);
            }
            return out;
        }
        let mut piece: Vec<u32> = Vec::new();
        for &x in values {
            if self.restarts() && x == restart {
                piece.clear();
                continue;
            }
            piece.push(x);
            let n = piece.len();
            match self {
                RewriteProgram::LineStrip if n >= 2 => {
                    let line = [piece[n - 2], piece[n - 1]];
                    put(&line, &line);
                }
                RewriteProgram::TriangleStrip if n >= 3 => {
                    let (a, b, c) = (piece[n - 3], piece[n - 2], piece[n - 1]);
                    match (n - 3).is_multiple_of(2) {
                        true => put(&[a, b, c], &[a, b, c]),
                        false => put(&[a, c, b], &[b, a, c]),
                    }
                }
                RewriteProgram::TriangleFan if n >= 3 => {
                    let triangle = [piece[0], piece[n - 2], piece[n - 1]];
                    put(&triangle, &triangle);
                }
                RewriteProgram::Quads if n.is_multiple_of(4) => {
                    let (a, b, c, d) = (piece[n - 4], piece[n - 3], piece[n - 2], piece[n - 1]);
                    put(&[a, b, d], &[a, b, d]);
                    put(&[b, c, d], &[b, c, d]);
                }
                RewriteProgram::QuadStrip if n >= 4 && n.is_multiple_of(2) => {
                    let (a, b, c, d) = (piece[n - 4], piece[n - 3], piece[n - 2], piece[n - 1]);
                    put(&[a, b, d], &[a, b, d]);
                    put(&[c, a, d], &[c, a, d]);
                }
                RewriteProgram::LineList if n.is_multiple_of(2) => {
                    let line = [piece[n - 2], piece[n - 1]];
                    put(&line, &line);
                }
                RewriteProgram::TriangleList if n.is_multiple_of(3) => {
                    let triangle = [piece[n - 3], piece[n - 2], piece[n - 1]];
                    put(&triangle, &triangle);
                }
                _ => {}
            }
        }
        out
    }

    /// [`list`](Self::list) of a draw without indices of `count` positions: of the positions
    /// themselves, `0..count`, none of which restarts, each primitive in `order`.
    pub fn sequential_list(self, count: u32, order: RewriteOrder) -> Vec<u32> {
        let mut out = Vec::with_capacity(self.capacity(u64::from(count)) as usize);
        let mut put = |first: &[u32], last: &[u32]| out.extend(order.arrange(first, last));
        let n = count;
        match self {
            RewriteProgram::LineStrip => (1..n).for_each(|q| put(&[q - 1, q], &[q - 1, q])),
            RewriteProgram::TriangleStrip => (2..n).for_each(|q| match q % 2 {
                0 => put(&[q - 2, q - 1, q], &[q - 2, q - 1, q]),
                _ => put(&[q - 2, q, q - 1], &[q - 1, q - 2, q]),
            }),
            RewriteProgram::LineLoop if n >= 2 => {
                (1..n).for_each(|q| put(&[q - 1, q], &[q - 1, q]));
                put(&[n - 1, 0], &[n - 1, 0]);
            }
            RewriteProgram::TriangleFan => (2..n).for_each(|q| put(&[0, q - 1, q], &[0, q - 1, q])),
            RewriteProgram::Quads => (0..n / 4).map(|k| 4 * k).for_each(|a| {
                put(&[a, a + 1, a + 3], &[a, a + 1, a + 3]);
                put(&[a + 1, a + 2, a + 3], &[a + 1, a + 2, a + 3]);
            }),
            RewriteProgram::QuadStrip if n >= 4 => (0..(n - 2) / 2).map(|k| 2 * k).for_each(|a| {
                put(&[a, a + 1, a + 3], &[a, a + 1, a + 3]);
                put(&[a + 2, a, a + 3], &[a + 2, a, a + 3]);
            }),
            RewriteProgram::LineList => (0..n / 2).map(|k| 2 * k).for_each(|a| {
                put(&[a, a + 1], &[a, a + 1]);
            }),
            RewriteProgram::TriangleList => (0..n / 3).map(|k| 3 * k).for_each(|a| {
                put(&[a, a + 1, a + 2], &[a, a + 1, a + 2]);
            }),
            RewriteProgram::LineLoop | RewriteProgram::QuadStrip => {}
        }
        out
    }
}

/// The vertex order the pass writes each primitive in (the module docs say why): Metal's at
/// provoking vertex mode 0 ([`First`](Self::First), the default), Metal's at mode 1
/// ([`Last`](Self::Last)), or Metal's at mode 1 with the last vertex first
/// ([`LastFirst`](Self::LastFirst)). A draw of a pipeline whose provoking vertex is the first
/// takes its flat inputs from each primitive's first vertex, and one of a pipeline whose provoking
/// vertex is the last (`VK_EXT_provoking_vertex`) from its last; each takes its barycentric order
/// from the order its vertices are listed in (the Vulkan specification's primitive topologies and
/// barycentric interpolation).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum RewriteOrder {
    /// Metal's order at provoking vertex mode 0: its flat inputs from the first vertex.
    #[default]
    First,
    /// Metal's order at provoking vertex mode 1: its flat inputs from the last vertex. A
    /// triangle strip's odd triangle is `(i + 1, i, i + 2)`, every other primitive as in
    /// [`First`](Self::First).
    Last,
    /// [`Last`](Self::Last) with each primitive's last vertex moved first: a triangle `(a, b, c)`
    /// as `(c, a, b)`, which keeps its winding, and a line `(a, b)` as `(b, a)`, which runs the
    /// other way.
    LastFirst,
}

impl RewriteOrder {
    /// The order's parameter word (14).
    pub const fn word(self) -> u32 {
        match self {
            RewriteOrder::First => 0,
            RewriteOrder::Last => 1,
            RewriteOrder::LastFirst => 2,
        }
    }

    /// A primitive whose vertices are `first` in Metal's order at mode 0 and `last` at mode 1,
    /// in this order.
    pub fn arrange<T: Copy>(self, first: &[T], last: &[T]) -> Vec<T> {
        match self {
            RewriteOrder::First => first.to_vec(),
            RewriteOrder::Last => last.to_vec(),
            RewriteOrder::LastFirst => match last.split_last() {
                Some((lead, rest)) => std::iter::once(*lead).chain(rest.iter().copied()).collect(),
                None => Vec::new(),
            },
        }
    }
}

/// One of a pass's three dispatches, in the order the consumer records them (the module docs
/// say what each does): one module each, of each program.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum RewritePhase {
    /// Over [`IndexRewritePass::groups`] workgroups: each sums its runs' primitives and finds
    /// the start of its last piece.
    Count,
    /// One workgroup: places each group, reserves the output, and writes the draw.
    Place,
    /// Over the groups again: each places its runs and writes their primitives.
    Write,
}

impl RewritePhase {
    /// The phases, in the order the consumer dispatches them.
    pub const ALL: [RewritePhase; 3] = [
        RewritePhase::Count,
        RewritePhase::Place,
        RewritePhase::Write,
    ];

    /// The workgroups of the phase's dispatch for `pass`.
    pub fn workgroups(self, pass: &IndexRewritePass) -> u32 {
        match self {
            RewritePhase::Place => 1,
            RewritePhase::Count | RewritePhase::Write => pass.groups,
        }
    }
}

/// Where a draw's Metal arguments are.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RewriteSource {
    /// A direct draw: its index count, instances, base vertex and base instance.
    Direct {
        /// Metal's index count.
        count: u32,
        /// Metal's instance count.
        instances: u32,
        /// Metal's base vertex.
        base_vertex: i32,
        /// Metal's base instance.
        base_instance: u32,
    },
    /// An indirect draw: Metal's `MTLDrawIndexedPrimitivesIndirectArguments` at `arguments`.
    Indirect {
        /// The arguments' place in the Metal indirect buffer.
        arguments: IndirectRef,
    },
    /// An indirect draw without indices: Metal's `MTLDrawPrimitivesIndirectArguments` at
    /// `arguments`. Each position is its own index from 0, the draw's vertex start its base
    /// vertex, and no position restarts.
    IndirectVertices {
        /// The arguments' place in the Metal indirect buffer.
        arguments: IndirectRef,
    },
}

/// Where the pass writes a draw's indices.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RewriteOutput {
    /// A region of the draw's own, of `capacity` indices, written from index 0.
    Owned {
        /// The region's indices.
        capacity: u32,
    },
    /// A reservation in the command buffer's index pool of `pool_bytes` index bytes (a multiple
    /// of 4) behind its header.
    Pooled {
        /// The pool's index bytes.
        pool_bytes: u32,
    },
}

/// One draw's index rewrite pass: what the consumer binds and uploads (the module docs say how).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct IndexRewritePass {
    /// The range of the Metal indirect buffer to bind at
    /// [`IndexRewriteModule::ARGUMENTS_BINDING`]: from the arguments' offset aligned down to
    /// `minStorageBufferOffsetAlignment` (and to 4) through their 20 bytes (16 without indices);
    /// `None` for a direct draw.
    pub arguments_range: Option<BufferRange>,
    /// The range of the Metal index buffer to bind at [`IndexRewriteModule::INDICES_BINDING`]:
    /// from the index offset aligned down likewise through the whole indices inside the buffer,
    /// to a whole word (`index_bounds::word_range`); `None` when none lies inside it from the
    /// offset, and for a draw without indices.
    pub indices_range: Option<BufferRange>,
    /// The words' first [`IndexRewritePass::PARAMETER_WORDS`] words, which the consumer uploads
    /// before the pass.
    pub parameters: [u32; IndexRewritePass::PARAMETER_WORDS],
    /// The workgroups of the count and write dispatches ([`RewritePhase::workgroups`]): one per
    /// [`IndexRewritePass::GROUP_POSITIONS`] positions the walk may take, from 1 to
    /// [`IndexRewritePass::MAX_GROUPS`] (all of them for an indirect draw without indices, whose
    /// count only the GPU knows).
    pub groups: u32,
}

impl IndexRewritePass {
    /// Bytes of the rewrite words: through the groups' words of [`MAX_GROUPS`](Self::MAX_GROUPS).
    pub const BYTES: u64 =
        4 * (Output::Groups as u64 + GROUP_WORDS as u64 * Self::MAX_GROUPS as u64);
    /// The most workgroups of the count and write dispatches.
    pub const MAX_GROUPS: u32 = 256;
    /// The positions of a walk per workgroup of the count and write dispatches, below which a
    /// draw takes fewer: 32 per invocation.
    pub const GROUP_POSITIONS: u64 = 32 * IndexRewriteModule::LOCAL_SIZE as u64;
    /// The most positions a walk takes past the buffer's whole indices: the most positions of a
    /// program's step ([`RewriteProgram::span`]).
    const WALK_PAST: u64 = 4;
    /// The consumer's words at the rewrite words' start: the parameters and a direct draw's
    /// arguments.
    pub const PARAMETER_WORDS: usize = 16;
    /// The draw's `VkDrawIndexedIndirectCommand` in the rewrite words.
    pub const DRAW_OFFSET: u64 = 64;
    /// The index pool's header, as the index bounds pass's: its counter, then reserved bytes.
    pub const POOL_HEADER_BYTES: u64 = IndexBoundsPass::POOL_HEADER_BYTES;
    /// Bytes of Metal's `MTLDrawIndexedPrimitivesIndirectArguments`.
    const ARGUMENT_BYTES: u64 = 20;
    /// Bytes of Metal's `MTLDrawPrimitivesIndirectArguments`.
    const VERTEX_ARGUMENT_BYTES: u64 = 16;

    /// The pass of a draw from `source` restarting at `restart` (`k'`), drawing indices of
    /// `index_size` bytes (2 or 4) from `index_offset` of an index buffer whose bytes end at
    /// `index_end` (its length, in the same terms as the offset; its `VkBuffer` holds the bytes
    /// to `index_end` rounded up to 4 there, as the padding does), into `output`, with ranges bound
    /// at offsets aligned to `alignment` (`minStorageBufferOffsetAlignment`) over at most
    /// `max_range` bytes (`maxStorageBufferRange`). A draw without indices
    /// ([`RewriteSource::IndirectVertices`]) binds none: its index terms are 4, 0 and 0. `None`
    /// when the arguments' offset is not a multiple of 4, the index size is neither 2 nor 4, the
    /// index offset is not a multiple of it, a pool's bytes are not a multiple of 4, or a range
    /// overflows or exceeds `max_range`.
    pub fn new(
        source: RewriteSource,
        restart: u32,
        (index_size, index_offset, index_end): (u64, u64, u64),
        output: RewriteOutput,
        alignment: u64,
        max_range: u32,
    ) -> Option<IndexRewritePass> {
        if !matches!(index_size, 2 | 4) || !index_offset.is_multiple_of(index_size) {
            return None;
        }
        let aligned = |offset: u64, end: u64| word_range(offset, end, alignment, max_range);
        let mut parameters = [0; Self::PARAMETER_WORDS];
        let sequential = matches!(source, RewriteSource::IndirectVertices { .. });
        let arguments_range = match source {
            RewriteSource::Direct {
                count,
                instances,
                base_vertex,
                base_instance,
            } => {
                parameters[Parameter::Arguments as usize] = count;
                parameters[Parameter::Arguments as usize + 1] = instances;
                parameters[Parameter::Arguments as usize + 3] = base_vertex as u32;
                parameters[Parameter::Arguments as usize + 4] = base_instance;
                None
            }
            RewriteSource::Indirect { arguments }
            | RewriteSource::IndirectVertices { arguments } => {
                if !arguments.offset.is_multiple_of(4) {
                    return None;
                }
                let bytes = match sequential {
                    true => Self::VERTEX_ARGUMENT_BYTES,
                    false => Self::ARGUMENT_BYTES,
                };
                let end = arguments.offset.checked_add(bytes)?;
                let (range, residual) = aligned(arguments.offset, end)?;
                parameters[Parameter::ArgumentsWord as usize] =
                    u32::try_from(residual / 4 + 1).ok()?;
                Some(range)
            }
        };
        let inside = match sequential {
            true => 0,
            false => index_end.saturating_sub(index_offset) / index_size,
        };
        // The positions the walk may take: an indirect draw's count is the GPU's.
        let walk = inside.saturating_add(Self::WALK_PAST);
        let walk = match source {
            RewriteSource::Direct { count, .. } => walk.min(u64::from(count)),
            RewriteSource::Indirect { .. } => walk,
            RewriteSource::IndirectVertices { .. } => u64::MAX,
        };
        let groups = walk
            .div_ceil(Self::GROUP_POSITIONS)
            .clamp(1, u64::from(Self::MAX_GROUPS)) as u32;
        let (indices_range, indices_residual) = if inside == 0 {
            (None, 0)
        } else {
            let end = index_offset.checked_add(inside.checked_mul(index_size)?)?;
            let (range, residual) = aligned(index_offset, end)?;
            (Some(range), residual)
        };
        let (capacity, pooled) = match output {
            RewriteOutput::Owned { capacity } => (capacity, 0),
            RewriteOutput::Pooled { pool_bytes } => {
                if !pool_bytes.is_multiple_of(4) {
                    return None;
                }
                (pool_bytes, 1)
            }
        };
        parameters[Parameter::IndicesByte as usize] = u32::try_from(indices_residual).ok()?;
        parameters[Parameter::Inside as usize] = u32::try_from(inside).unwrap_or(u32::MAX);
        parameters[Parameter::Wide as usize] = u32::from(index_size == 4);
        parameters[Parameter::Restart as usize] = restart;
        parameters[Parameter::Capacity as usize] = capacity;
        parameters[Parameter::Pooled as usize] = pooled;
        parameters[Parameter::Groups as usize] = groups;
        parameters[Parameter::Sequential as usize] = u32::from(sequential);
        Some(IndexRewritePass {
            arguments_range,
            indices_range,
            parameters,
            groups,
        })
    }

    /// The pass writing each primitive in `order` (the pass [`new`](Self::new) makes writes
    /// [`RewriteOrder::First`]).
    pub fn with_order(mut self, order: RewriteOrder) -> IndexRewritePass {
        self.parameters[Parameter::Order as usize] = order.word();
        self
    }

    /// The pass of a list drawn in the private list form (`VariantKey::private_list`): the
    /// output holds Metal's base vertex (the draw's, its vertex start without indices) in the
    /// word before the list's first index, which an owned output's capacity counts too, and the
    /// draw at [`DRAW_OFFSET`](Self::DRAW_OFFSET) is a `VkDrawIndirectCommand` of the list's
    /// positions, its first vertex the list's first index in the output (from the pool's index
    /// region where pooled) and its first instance Metal's base instance.
    pub fn listed(mut self) -> IndexRewritePass {
        self.parameters[Parameter::Listed as usize] = 1;
        self
    }
}

/// The consumer's words, by index into the rewrite words.
#[derive(Clone, Copy)]
enum Parameter {
    ArgumentsWord = 0,
    IndicesByte = 1,
    Inside = 2,
    Wide = 3,
    Restart = 4,
    Capacity = 5,
    Pooled = 6,
    Groups = 7,
    Arguments = 8,
    /// 1 for a draw without indices ([`RewriteSource::IndirectVertices`]).
    Sequential = 13,
    /// The order ([`RewriteOrder::word`]).
    Order = 14,
    /// 1 for a list drawn in the private list form ([`IndexRewritePass::listed`]).
    Listed = 15,
    /// A run's draw's: the first word of its region of the run's output
    /// ([`IndexRewriteRun`]); a pass's draw writes from word 0 and reads no such word.
    OwnedFirst = 26,
}

/// The pass's words in the rewrite words, by index.
#[derive(Clone, Copy)]
enum Output {
    Draw = 16,
    Primitives = 21,
    /// The output's first word.
    Base = 22,
    /// 1 where the draw is written.
    Written = 23,
    /// The primitives before those past the walk.
    TailAt = 24,
    /// The primitives past the walk.
    Tail = 25,
    /// The groups' words, [`GROUP_WORDS`] per group.
    Groups = 32,
}

/// The words of each group: its primitives where its first piece's positions before it are
/// even, and where they are odd (a quad strip's pairs count from its piece's start); and the
/// start of its last piece. The place dispatch turns the first two into the primitives before
/// the group and the start of the piece it begins in.
const GROUP_WORDS: u32 = 3;

/// The crate-owned module of one program on one device, validated like any output.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IndexRewriteModule {
    /// SPIR-V 1.5 words, entry point `main`, bindings numbered as below in set 0.
    pub spirv: Arc<[u32]>,
    /// What the module needs from the device (within every Vulkan 1.2 device).
    pub requirements: Requirements,
}

impl IndexRewriteModule {
    /// Descriptor set of every binding.
    pub const SET: u32 = 0;
    /// The draw's rewrite words, a storage buffer.
    pub const WORDS_BINDING: u32 = 0;
    /// The Metal indirect buffer, a storage buffer.
    pub const ARGUMENTS_BINDING: u32 = 1;
    /// The Metal index buffer, a storage buffer.
    pub const INDICES_BINDING: u32 = 2;
    /// The draw's output: its region, or the command buffer's index pool, a storage buffer.
    pub const OUTPUT_BINDING: u32 = 3;
    /// The command buffer's `ExecutionStatus` buffer, a storage buffer.
    pub const STATUS_BINDING: u32 = 4;
    /// Invocations of each workgroup, along x: Vulkan's least `maxComputeWorkGroupSize[0]`
    /// and `maxComputeWorkGroupInvocations`, so every device takes it.
    pub const LOCAL_SIZE: u32 = 128;
}

/// The words' binding symbol in the typed module.
pub const WORDS: ResourceSlot = ResourceSlot {
    stage: Stage::Kernel,
    kind: ResourceKind::Buffer,
    index: 0,
};
/// The arguments' binding symbol ([`IndexRewriteModule::ARGUMENTS_BINDING`]).
pub const ARGUMENTS: ResourceSlot = ResourceSlot {
    stage: Stage::Kernel,
    kind: ResourceKind::Buffer,
    index: 1,
};
/// The indices' binding symbol ([`IndexRewriteModule::INDICES_BINDING`]).
pub const INDICES: ResourceSlot = ResourceSlot {
    stage: Stage::Kernel,
    kind: ResourceKind::Buffer,
    index: 2,
};
/// The output's binding symbol ([`IndexRewriteModule::OUTPUT_BINDING`]).
pub const OUTPUT: ResourceSlot = ResourceSlot {
    stage: Stage::Kernel,
    kind: ResourceKind::Buffer,
    index: 3,
};
/// The status buffer's binding symbol ([`IndexRewriteModule::STATUS_BINDING`]).
pub const STATUS: ResourceSlot = ResourceSlot {
    stage: Stage::Kernel,
    kind: ResourceKind::Synthetic(SyntheticBinding::ExecutionStatus),
    index: 0,
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

fn k(f: &mut FunctionBuilder<'_>, v: u32) -> BResult<Value> {
    let c = f.u32_const(v)?;
    f.module().const_value(c)
}

/// The `bool` constant `v`.
fn boolean(f: &mut FunctionBuilder<'_>, v: bool) -> BResult<Value> {
    let ty = f.bool_ty(1)?;
    let c = f.module().constant(Constant::Scalar {
        ty,
        bits: ConstBits(u64::from(v)),
    })?;
    f.module().const_value(c)
}

/// The pointer to word `index` of `buffer`'s runtime array.
fn word_ptr(f: &mut FunctionBuilder<'_>, buffer: Value, index: Value) -> BResult<Value> {
    let zero = k(f, 0)?;
    f.access_chain(buffer, &[zero, index])
}

fn load(f: &mut FunctionBuilder<'_>, buffer: Value, index: Value) -> BResult<Value> {
    let p = word_ptr(f, buffer, index)?;
    f.load(p, MemAccess::default())
}

fn load_at(f: &mut FunctionBuilder<'_>, buffer: Value, index: u32) -> BResult<Value> {
    let i = k(f, index)?;
    load(f, buffer, i)
}

fn store(f: &mut FunctionBuilder<'_>, buffer: Value, index: Value, v: Value) -> BResult<()> {
    let p = word_ptr(f, buffer, index)?;
    f.store(p, v, MemAccess::default())
}

/// Word `index` of `buffer` where it lies inside the buffer's bound range, else 0 (contained).
fn contained(f: &mut FunctionBuilder<'_>, buffer: Value, index: Value) -> BResult<Value> {
    let zero = k(f, 0)?;
    let len = f.array_length(buffer, 0)?;
    let inside = f.binary(BinOp::ULessThan, index, len)?;
    let safe = f.select(inside, index, zero)?;
    let loaded = load(f, buffer, safe)?;
    f.select(inside, loaded, zero)
}

/// Store `v` at word `index` of `buffer` where it lies inside the buffer's bound range.
fn contained_store(
    f: &mut FunctionBuilder<'_>,
    buffer: Value,
    index: Value,
    v: Value,
) -> BResult<()> {
    let len = f.array_length(buffer, 0)?;
    let inside = f.binary(BinOp::ULessThan, index, len)?;
    when(f, inside, move |f| store(f, buffer, index, v))
}

/// `a + b`, or `u32::MAX` where it wraps.
fn saturating_add(f: &mut FunctionBuilder<'_>, a: Value, b: Value) -> BResult<Value> {
    let sum = f.binary(BinOp::IAdd, a, b)?;
    let wrapped = f.binary(BinOp::ULessThan, sum, a)?;
    let max = k(f, u32::MAX)?;
    f.select(wrapped, max, sum)
}

/// `a - b`, or 0 where `b` is larger.
fn saturating_sub(f: &mut FunctionBuilder<'_>, a: Value, b: Value) -> BResult<Value> {
    let diff = f.binary(BinOp::ISub, a, b)?;
    let over = f.binary(BinOp::UGreaterThan, b, a)?;
    let zero = k(f, 0)?;
    f.select(over, zero, diff)
}

/// `a / b` rounded up (`b` not 0), without wrapping.
fn div_ceil(f: &mut FunctionBuilder<'_>, a: Value, b: Value) -> BResult<Value> {
    let q = f.binary(BinOp::UDiv, a, b)?;
    let r = f.binary(BinOp::UMod, a, b)?;
    let zero = k(f, 0)?;
    let one = k(f, 1)?;
    let partial = f.binary(BinOp::INotEqual, r, zero)?;
    let extra = f.select(partial, one, zero)?;
    f.binary(BinOp::IAdd, q, extra)
}

/// Run `each(f, i)` for `i` from `lo` below `hi`, `step` apart (`step` not 0).
pub(crate) fn for_range<'a>(
    f: &mut FunctionBuilder<'_>,
    u32_: TypeId,
    (lo, hi, step): (Value, Value, Value),
    each: impl FnOnce(&mut FunctionBuilder<'_>, Value) -> BResult<()> + 'a,
) -> BResult<()> {
    let var = f.local_var(u32_, None)?;
    let counter = f.var_ptr(var)?;
    f.store(counter, lo, MemAccess::default())?;
    let body: LoopFn<'a, ()> = Box::new(move |f, _, lc| {
        let i = f.load(counter, MemAccess::default())?;
        let done = f.binary(BinOp::UGreaterThanEqual, i, hi)?;
        let exit: ArmFn<'_, ()> = Box::new(move |f, _| {
            lc.break_(f)?;
            Ok(ArmEnd::Exits)
        });
        f.if_(&mut (), done, exit, None)?;
        // The next `i`, or past `hi` where it would wrap.
        let next = saturating_add(f, i, step)?;
        f.store(counter, next, MemAccess::default())?;
        each(f, i)
    });
    f.loop_(&mut (), LoopSpec::default(), body)
}

/// Run `then(f)` where `c` holds.
pub(crate) fn when<'a>(
    f: &mut FunctionBuilder<'_>,
    c: Value,
    then: impl FnOnce(&mut FunctionBuilder<'_>) -> BResult<()> + 'a,
) -> BResult<()> {
    let arm: ArmFn<'a, ()> = Box::new(move |f, _| {
        then(f)?;
        Ok(ArmEnd::FallThrough(Vec::new()))
    });
    f.if_(&mut (), c, arm, None)?;
    Ok(())
}

/// The workgroup barrier over the runs' words.
fn barrier(f: &mut FunctionBuilder<'_>) -> BResult<()> {
    let sem = Semantics {
        order: MemoryOrder::AcquireRelease,
        workgroup_memory: true,
        ..Semantics::RELAXED
    };
    f.control_barrier(Scope::Workgroup, Scope::Workgroup, sem)
}

/// The workgroup barrier over the runs' words and the draw's words, between a run's phases.
fn phase_barrier(f: &mut FunctionBuilder<'_>) -> BResult<()> {
    let sem = Semantics {
        order: MemoryOrder::AcquireRelease,
        workgroup_memory: true,
        uniform_memory: true,
        ..Semantics::RELAXED
    };
    f.control_barrier(Scope::Workgroup, Scope::Workgroup, sem)
}

/// What reads the draw's Metal index at a position.
#[derive(Clone, Copy)]
struct Fetch {
    indices: Value,
    first_byte: Value,
    inside: Value,
    start: Value,
    wide: Value,
    /// A draw without indices: each position is its own index.
    sequential: Value,
}

impl Fetch {
    /// The index at the draw's position `q` (Metal position `start + q`): where it lies inside
    /// the whole indices (and did not wrap past `start`), its index, else 0; 4 bytes each where
    /// `wide`, else 2. A draw without indices reads `q`.
    fn at(self, f: &mut FunctionBuilder<'_>, q: Value) -> BResult<Value> {
        let zero = k(f, 0)?;
        let one = k(f, 1)?;
        let two = k(f, 2)?;
        let three = k(f, 3)?;
        let p = f.binary(BinOp::IAdd, self.start, q)?;
        let unwrapped = f.binary(BinOp::UGreaterThanEqual, p, self.start)?;
        let below = f.binary(BinOp::ULessThan, p, self.inside)?;
        let valid = f.binary(BinOp::LogicalAnd, unwrapped, below)?;
        let shift = f.select(self.wide, two, one)?;
        let offset = f.binary(BinOp::ShiftLeftLogical, p, shift)?;
        let byte = f.binary(BinOp::IAdd, self.first_byte, offset)?;
        let word = f.binary(BinOp::ShiftRightLogical, byte, two)?;
        let indexed = f.unary(UnOp::LogicalNot, self.sequential)?;
        let read = f.binary(BinOp::LogicalAnd, valid, indexed)?;
        let safe = f.select(read, word, zero)?;
        let loaded = contained(f, self.indices, safe)?;
        let half = f.binary(BinOp::BitwiseAnd, byte, two)?;
        let bits = f.binary(BinOp::ShiftLeftLogical, half, three)?;
        let shifted = f.binary(BinOp::ShiftRightLogical, loaded, bits)?;
        let mask = k(f, 0xFFFF)?;
        let narrow = f.binary(BinOp::BitwiseAnd, shifted, mask)?;
        let value = f.select(self.wide, loaded, narrow)?;
        let value = f.select(valid, value, zero)?;
        f.select(self.sequential, q, value)
    }
}

/// The pass's dispatch of `phase` for `program` as a typed module, before serialization: the
/// module [`index_rewrite_module`] serializes, which the CPU interpreter runs. Its bindings are the
/// symbols [`WORDS`], [`ARGUMENTS`], [`INDICES`], [`OUTPUT`] and [`STATUS`].
pub fn rewrite_typed_module(program: RewriteProgram, phase: RewritePhase) -> BResult<TypedModule> {
    typed_module_of(program, Form::Phase(phase))
}

/// The dispatch of a run of `program`'s draws ([`IndexRewriteRun`]) as a typed module, before
/// serialization: the module [`index_rewrite_run_module`] serializes. Its bindings are
/// [`index_rewrite_module`]'s, the words those of the run's slots.
pub fn rewrite_run_typed_module(program: RewriteProgram) -> BResult<TypedModule> {
    typed_module_of(program, Form::Run)
}

/// What one dispatch of the pass does.
#[derive(Clone, Copy)]
enum Form {
    /// One phase of one draw.
    Phase(RewritePhase),
    /// Every phase of one draw of one group per workgroup, the draw's words its slot's.
    Run,
}

/// The module of `program`'s dispatch of `form`.
fn typed_module_of(program: RewriteProgram, form: Form) -> BResult<TypedModule> {
    let mut b = ModuleBuilder::new();
    let u32_ = b.ty(Type::Int {
        width: 32,
        signed: false,
    })?;
    let uvec3 = b.ty(Type::Vector { elem: u32_, len: 3 })?;
    let most = b.constant(Constant::Scalar {
        ty: u32_,
        bits: ConstBits(u64::from(IndexRewritePass::MAX_GROUPS)),
    })?;
    let list_ty = b.ty(Type::Array {
        elem: u32_,
        len: most,
        stride: None,
    })?;
    // Per run (count and write) or per group (place): its primitives where its first piece's
    // positions before it are even, then those before it; the same where they are odd; the start
    // of its last piece, then of the piece it begins in.
    let counts = b.plain_global(StorageClass::Workgroup, list_ty, None)?;
    let odds = b.plain_global(StorageClass::Workgroup, list_ty, None)?;
    let starts = b.plain_global(StorageClass::Workgroup, list_ty, None)?;
    let words = words_buffer(&mut b, WORDS, u32_)?;
    let arguments = words_buffer(&mut b, ARGUMENTS, u32_)?;
    let indices = words_buffer(&mut b, INDICES, u32_)?;
    let output = words_buffer(&mut b, OUTPUT, u32_)?;
    let status = words_buffer(&mut b, STATUS, u32_)?;
    let invocation = b.builtin(
        crate::spv::BuiltIn::LocalInvocationIndex,
        IoDir::Input,
        u32_,
    )?;
    let workgroup = b.builtin(crate::spv::BuiltIn::WorkgroupId, IoDir::Input, uvec3)?;
    let void = b.ty(Type::Void)?;
    let mut f = b.function(void)?;
    let g = Globals {
        words: f.module().global_value(words)?,
        arguments: f.module().global_value(arguments)?,
        indices: f.module().global_value(indices)?,
        output: f.module().global_value(output)?,
        status: f.module().global_value(status)?,
        counts: f.module().global_value(counts)?,
        odds: f.module().global_value(odds)?,
        starts: f.module().global_value(starts)?,
        slot: None,
        u32_,
    };
    let invocation = f.module().global_value(invocation)?;
    let lin = f.load(invocation, MemAccess::default())?;
    let workgroup = f.module().global_value(workgroup)?;
    let workgroup = f.load(workgroup, MemAccess::default())?;
    let group = f.composite_extract(workgroup, &[0])?;
    // A run's workgroup is its draw's slot, and its draw's only group.
    let (g, at) = match form {
        Form::Phase(_) => (g, Invocation { lin, group }),
        Form::Run => {
            let per = k(&mut f, IndexRewriteRun::SLOT_WORDS)?;
            let first = f.binary(BinOp::IMul, group, per)?;
            let group = k(&mut f, 0)?;
            let g = Globals {
                slot: Some(first),
                ..g
            };
            (g, Invocation { lin, group })
        }
    };
    let d = DrawWalk::read(&mut f, g, program)?;
    match form {
        Form::Phase(RewritePhase::Count) => count_phase(&mut f, g, d, at)?,
        Form::Phase(RewritePhase::Place) => place_phase(&mut f, g, d, lin)?,
        Form::Phase(RewritePhase::Write) => write_phase(&mut f, g, d, at)?,
        Form::Run => {
            count_phase(&mut f, g, d, at)?;
            phase_barrier(&mut f)?;
            place_phase(&mut f, g, d, lin)?;
            phase_barrier(&mut f)?;
            write_phase(&mut f, g, d, at)?;
        }
    }
    f.ret(None)?;
    let id = f.finish()?;
    b.entry(
        ExecModel::GLCompute,
        id,
        vec![ExecMode::LocalSize {
            x: IndexRewriteModule::LOCAL_SIZE,
            y: 1,
            z: 1,
        }],
    )?;
    b.finalize()
}

/// The module's buffers and workgroup lists, and its `u32` type.
#[derive(Clone, Copy)]
struct Globals {
    words: Value,
    /// A run's draw: the first of its slot's words ([`IndexRewriteRun::SLOT_WORDS`] per
    /// workgroup), from which its words lie as a pass's do; `None` for a pass's dispatch.
    slot: Option<Value>,
    arguments: Value,
    indices: Value,
    output: Value,
    status: Value,
    /// Per run or group: its primitives where its first piece's positions before it are even,
    /// then those before it.
    counts: Value,
    /// Per run or group: its primitives where its first piece's positions before it are odd.
    odds: Value,
    /// Per run or group: the start of its last piece, then of the piece it begins in.
    starts: Value,
    u32_: TypeId,
}

/// One entry of the workgroup lists: its primitives where its first piece's positions before it
/// are even and where they are odd, and the start of its last piece (0 for none).
#[derive(Clone, Copy)]
struct Entry {
    even: Value,
    odd: Value,
    start: Value,
}

impl Globals {
    /// The index into the words buffer of the draw's word `index`: past a run's slot's start.
    fn word(self, f: &mut FunctionBuilder<'_>, index: Value) -> BResult<Value> {
        match self.slot {
            None => Ok(index),
            Some(first) => f.binary(BinOp::IAdd, first, index),
        }
    }

    /// The draw's word `index`.
    fn word_load(self, f: &mut FunctionBuilder<'_>, index: Value) -> BResult<Value> {
        let i = self.word(f, index)?;
        load(f, self.words, i)
    }

    /// The draw's word `index`, a constant.
    fn word_load_at(self, f: &mut FunctionBuilder<'_>, index: u32) -> BResult<Value> {
        let i = k(f, index)?;
        self.word_load(f, i)
    }

    /// Store `v` as the draw's word `index`.
    fn word_store(self, f: &mut FunctionBuilder<'_>, index: Value, v: Value) -> BResult<()> {
        let i = self.word(f, index)?;
        store(f, self.words, i, v)
    }

    /// Store `v` as the draw's word `index`, a constant.
    fn word_store_at(self, f: &mut FunctionBuilder<'_>, index: u32, v: Value) -> BResult<()> {
        let i = k(f, index)?;
        self.word_store(f, i, v)
    }

    /// A `u32` function variable holding `init`, by its pointer.
    fn local(self, f: &mut FunctionBuilder<'_>, init: Value) -> BResult<Value> {
        let var = f.local_var(self.u32_, None)?;
        let ptr = f.var_ptr(var)?;
        f.store(ptr, init, MemAccess::default())?;
        Ok(ptr)
    }

    /// Store `entry` as entry `i` of the workgroup lists.
    fn list(self, f: &mut FunctionBuilder<'_>, i: Value, entry: Entry) -> BResult<()> {
        for (list, v) in [
            (self.counts, entry.even),
            (self.odds, entry.odd),
            (self.starts, entry.start),
        ] {
            let p = f.access_chain(list, &[i])?;
            f.store(p, v, MemAccess::default())?;
        }
        Ok(())
    }

    /// Entry `i` of the workgroup lists.
    fn entry(self, f: &mut FunctionBuilder<'_>, i: Value) -> BResult<Entry> {
        let mut read = |list| -> BResult<Value> {
            let p = f.access_chain(list, &[i])?;
            f.load(p, MemAccess::default())
        };
        Ok(Entry {
            even: read(self.counts)?,
            odd: read(self.odds)?,
            start: read(self.starts)?,
        })
    }

    /// The first of group `group`'s [`GROUP_WORDS`] words.
    fn group_word(self, f: &mut FunctionBuilder<'_>, group: Value) -> BResult<Value> {
        let per = k(f, GROUP_WORDS)?;
        let first = k(f, Output::Groups as u32)?;
        let w = f.binary(BinOp::IMul, group, per)?;
        f.binary(BinOp::IAdd, first, w)
    }
}

/// An invocation: its index in its workgroup, and its workgroup's along x.
#[derive(Clone, Copy)]
struct Invocation {
    lin: Value,
    group: Value,
}

/// An invocation's run of positions, `[lo, hi)`; its place among the runs, and the runs.
#[derive(Clone, Copy)]
struct Run {
    lo: Value,
    hi: Value,
    t: Value,
    runs: Value,
}

/// What every dispatch of a pass reads of its draw: how the walk fetches, `k'`, how far it walks,
/// Metal's other arguments and the output's terms.
#[derive(Clone, Copy)]
struct DrawWalk {
    fetch: Fetch,
    restart: Value,
    /// The draw's positions.
    count: Value,
    /// The positions the runs walk.
    walked: Value,
    /// The draw's positions past the walk.
    tail: Value,
    /// The walk's positions per run.
    chunk: Value,
    instances: Value,
    base: Value,
    base_instance: Value,
    capacity: Value,
    /// The first word of an owned output: a run's draw's region's, else 0.
    owned: Value,
    /// Whether the output is the index pool.
    pooled: Value,
    /// The groups, from 1 to `MAX_GROUPS`.
    groups: Value,
    /// The order's word ([`RewriteOrder::word`]).
    order: Value,
    /// 1 where the list is drawn in the private list form, else 0.
    listed: Value,
    program: RewriteProgram,
}

impl DrawWalk {
    fn read(f: &mut FunctionBuilder<'_>, g: Globals, program: RewriteProgram) -> BResult<DrawWalk> {
        let zero = k(f, 0)?;
        let one = k(f, 1)?;
        let argument_word = g.word_load_at(f, Parameter::ArgumentsWord as u32)?;
        let first_byte = g.word_load_at(f, Parameter::IndicesByte as u32)?;
        let inside = g.word_load_at(f, Parameter::Inside as u32)?;
        let wide = g.word_load_at(f, Parameter::Wide as u32)?;
        let restart = g.word_load_at(f, Parameter::Restart as u32)?;
        let capacity = g.word_load_at(f, Parameter::Capacity as u32)?;
        let pooled = g.word_load_at(f, Parameter::Pooled as u32)?;
        let groups = g.word_load_at(f, Parameter::Groups as u32)?;
        let sequential = g.word_load_at(f, Parameter::Sequential as u32)?;
        let order = g.word_load_at(f, Parameter::Order as u32)?;
        let listed = g.word_load_at(f, Parameter::Listed as u32)?;
        let owned = match g.slot {
            Some(_) => g.word_load_at(f, Parameter::OwnedFirst as u32)?,
            None => zero,
        };
        let listed = f.glsl(GlslOp::UMin, &[listed, one])?;
        // A run's draw is its workgroup's alone.
        let groups = match g.slot {
            Some(_) => one,
            None => {
                let most = k(f, IndexRewritePass::MAX_GROUPS)?;
                let groups = f.glsl(GlslOp::UMin, &[groups, most])?;
                f.glsl(GlslOp::UMax, &[groups, one])?
            }
        };
        let wide = f.binary(BinOp::INotEqual, wide, zero)?;
        let pooled = f.binary(BinOp::INotEqual, pooled, zero)?;
        let sequential = f.binary(BinOp::INotEqual, sequential, zero)?;
        // Metal's arguments: count, instances, first index, base vertex, base instance; an
        // indirect draw's from its buffer, a direct draw's from the words. A draw without indices
        // has count, instances, vertex start and base instance: its vertex start is its base
        // vertex, and it starts at position 0.
        let indirect = f.binary(BinOp::INotEqual, argument_word, zero)?;
        let at = f.binary(BinOp::ISub, argument_word, one)?;
        let mut argument = Vec::with_capacity(5);
        for i in 0..5 {
            let i = k(f, i)?;
            let w = f.binary(BinOp::IAdd, at, i)?;
            let w = f.select(indirect, w, zero)?;
            let read = contained(f, g.arguments, w)?;
            let given = k(f, Parameter::Arguments as u32)?;
            let given = f.binary(BinOp::IAdd, given, i)?;
            let given = g.word_load(f, given)?;
            argument.push(f.select(indirect, read, given)?);
        }
        let (count, instances) = (argument[0], argument[1]);
        let start = f.select(sequential, zero, argument[2])?;
        let base = f.select(sequential, argument[2], argument[3])?;
        let base_instance = f.select(sequential, argument[3], argument[4])?;
        // A draw without indices holds every position.
        let inside = f.select(sequential, count, inside)?;
        let fetch = Fetch {
            indices: g.indices,
            first_byte,
            inside,
            start,
            wide,
            sequential,
        };
        // The walk: the positions inside the buffer from the first index, and the program's span
        // past them (which read 0); past it every position reads 0.
        let span = k(f, program.span())?;
        let held = saturating_sub(f, inside, start)?;
        let reach = saturating_add(f, held, span)?;
        let walked = f.glsl(GlslOp::UMin, &[count, reach])?;
        let tail = f.binary(BinOp::ISub, count, walked)?;
        let lanes = k(f, IndexRewriteModule::LOCAL_SIZE)?;
        let runs = f.binary(BinOp::IMul, groups, lanes)?;
        let chunk = div_ceil(f, walked, runs)?;
        Ok(DrawWalk {
            fetch,
            restart,
            count,
            walked,
            tail,
            chunk,
            instances,
            base,
            base_instance,
            capacity,
            owned,
            pooled,
            groups,
            order,
            listed,
            program,
        })
    }

    /// A primitive whose vertices are `first` in Metal's order at mode 0 and `last` at mode 1, in
    /// the draw's order ([`RewriteOrder::arrange`]).
    fn ordered(
        self,
        f: &mut FunctionBuilder<'_>,
        first: &[Value],
        last: &[Value],
    ) -> BResult<Vec<Value>> {
        let n = last.len();
        let first_word = k(f, RewriteOrder::First.word())?;
        let last_word = k(f, RewriteOrder::Last.word())?;
        let is_first = f.binary(BinOp::IEqual, self.order, first_word)?;
        let is_last = f.binary(BinOp::IEqual, self.order, last_word)?;
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let led = f.select(is_last, last[i], last[(i + n - 1) % n])?;
            out.push(f.select(is_first, first[i], led)?);
        }
        Ok(out)
    }

    /// Whether a position of index `value` restarts: where the program restarts, the draw has
    /// indices, and `value` is `k'`.
    fn restarts(self, f: &mut FunctionBuilder<'_>, value: Value) -> BResult<Value> {
        if !self.program.restarts() {
            return boolean(f, false);
        }
        let at = f.binary(BinOp::IEqual, value, self.restart)?;
        let indexed = f.unary(UnOp::LogicalNot, self.fetch.sequential)?;
        f.binary(BinOp::LogicalAnd, at, indexed)
    }

    /// The first position of run `t` of `groups · LOCAL_SIZE` runs (`t` among them, or the
    /// first past them): `t · chunk`, at most the walk. Below 2^32: `t · chunk` is at most the
    /// walk where the chunk is at least the runs, and below 2^31 where it is fewer.
    fn lo_of(self, f: &mut FunctionBuilder<'_>, t: Value) -> BResult<Value> {
        let from = f.binary(BinOp::IMul, t, self.chunk)?;
        f.glsl(GlslOp::UMin, &[from, self.walked])
    }

    /// `at`'s run: one of `groups · LOCAL_SIZE` runs of the walk's positions, in order, each of
    /// as many as the walk over the runs, rounded up; empty where its workgroup is past the
    /// groups.
    fn run(self, f: &mut FunctionBuilder<'_>, at: Invocation) -> BResult<Run> {
        let lanes = k(f, IndexRewriteModule::LOCAL_SIZE)?;
        let runs = f.binary(BinOp::IMul, self.groups, lanes)?;
        let before = f.binary(BinOp::IMul, at.group, lanes)?;
        let t = f.binary(BinOp::IAdd, before, at.lin)?;
        let among = f.binary(BinOp::ULessThan, at.group, self.groups)?;
        let from = self.lo_of(f, t)?;
        let lo = f.select(among, from, self.walked)?;
        let end = saturating_add(f, lo, self.chunk)?;
        let hi = f.glsl(GlslOp::UMin, &[end, self.walked])?;
        Ok(Run { lo, hi, t, runs })
    }

    /// Before position `lo`: the positions of its piece there, at most `span - 1` (those of
    /// `lo - 1`, `lo - 2`, ... that exist, back to the first that restarts), and the indices at
    /// `lo - 1`, `lo - 2` and `lo - 3` (0 where there is none, or the span does not reach).
    fn behind(self, f: &mut FunctionBuilder<'_>, lo: Value) -> BResult<(Value, [Value; 3])> {
        let zero = k(f, 0)?;
        let mut values = [zero; 3];
        let mut streak = zero;
        let mut all = boolean(f, true)?;
        for n in 1..self.program.span() {
            let back = k(f, n)?;
            let has = f.binary(BinOp::UGreaterThanEqual, lo, back)?;
            let q = f.binary(BinOp::ISub, lo, back)?;
            let fetched = self.fetch.at(f, q)?;
            let value = f.select(has, fetched, zero)?;
            let restarts = self.restarts(f, value)?;
            let other = f.unary(UnOp::LogicalNot, restarts)?;
            let live = f.binary(BinOp::LogicalAnd, has, other)?;
            all = f.binary(BinOp::LogicalAnd, all, live)?;
            let n_ = k(f, n)?;
            streak = f.select(all, n_, streak)?;
            values[n as usize - 1] = value;
        }
        Ok((streak, values))
    }

    /// Position `q`, of a piece that starts at `piece`, reads `value`: whether it restarts, and
    /// whether it completes a step, its piece's positions through it (at most the span) at
    /// `streak`. A strip, line loop and fan complete at every position the span reaches back
    /// from; quads at every fourth position of the draw; a quad strip at every second position of
    /// its piece the span reaches back from.
    fn step(
        self,
        f: &mut FunctionBuilder<'_>,
        (streak, q, piece): (Value, Value, Value),
        value: Value,
    ) -> BResult<(Value, Value)> {
        let zero = k(f, 0)?;
        let one = k(f, 1)?;
        let span = k(f, self.program.span())?;
        let restarts = self.restarts(f, value)?;
        let s = f.load(streak, MemAccess::default())?;
        let grown = f.binary(BinOp::IAdd, s, one)?;
        let grown = f.glsl(GlslOp::UMin, &[grown, span])?;
        let s = f.select(restarts, zero, grown)?;
        f.store(streak, s, MemAccess::default())?;
        let spanned = f.binary(BinOp::UGreaterThanEqual, s, span)?;
        let completes = match self.program {
            RewriteProgram::Quads | RewriteProgram::LineList | RewriteProgram::TriangleList => {
                // Every `span`th position of the draw, which never restarts.
                let last = k(f, self.program.span() - 1)?;
                let phase = f.binary(BinOp::UMod, q, span)?;
                f.binary(BinOp::IEqual, phase, last)?
            }
            RewriteProgram::QuadStrip => {
                let into = f.binary(BinOp::ISub, q, piece)?;
                let parity = f.binary(BinOp::BitwiseAnd, into, one)?;
                let odd = f.binary(BinOp::INotEqual, parity, zero)?;
                f.binary(BinOp::LogicalAnd, spanned, odd)?
            }
            _ => spanned,
        };
        Ok((restarts, completes))
    }

    /// `run`'s primitives where the piece it begins in starts at `piece` (only its parity
    /// matters, so any position of the same parity stands for it), and the start of its last
    /// piece (0 where nothing in it restarts), its piece's positions before it `streak`.
    fn count(
        self,
        f: &mut FunctionBuilder<'_>,
        g: Globals,
        run: Run,
        (streak, piece): (Value, Value),
    ) -> BResult<(Value, Value)> {
        let zero = k(f, 0)?;
        let one = k(f, 1)?;
        let streak = g.local(f, streak)?;
        let piece = g.local(f, piece)?;
        let count = g.local(f, zero)?;
        let start = g.local(f, zero)?;
        let per_step = self.program.per_step();
        for_range(f, g.u32_, (run.lo, run.hi, one), move |f, q| {
            let value = self.fetch.at(f, q)?;
            let p = f.load(piece, MemAccess::default())?;
            let (restarts, completes) = self.step(f, (streak, q, p), value)?;
            let zero = k(f, 0)?;
            let one = k(f, 1)?;
            let per = k(f, per_step)?;
            let add = f.select(completes, per, zero)?;
            let c = f.load(count, MemAccess::default())?;
            let c = f.binary(BinOp::IAdd, c, add)?;
            f.store(count, c, MemAccess::default())?;
            let next = f.binary(BinOp::IAdd, q, one)?;
            let s = f.load(start, MemAccess::default())?;
            let s = f.select(restarts, next, s)?;
            f.store(start, s, MemAccess::default())?;
            let p = f.select(restarts, next, p)?;
            f.store(piece, p, MemAccess::default())
        })?;
        let count = f.load(count, MemAccess::default())?;
        let start = f.load(start, MemAccess::default())?;
        Ok((count, start))
    }

    /// `run`'s entry: its primitives under both parities of its first piece's positions before
    /// it (alike but for a quad strip), and the start of its last piece, its piece's positions
    /// before it `streak`.
    fn entry(
        self,
        f: &mut FunctionBuilder<'_>,
        g: Globals,
        run: Run,
        streak: Value,
    ) -> BResult<Entry> {
        let (even, start) = self.count(f, g, run, (streak, run.lo))?;
        let odd = match self.program {
            RewriteProgram::QuadStrip => {
                let one = k(f, 1)?;
                let before = f.binary(BinOp::ISub, run.lo, one)?;
                self.count(f, g, run, (streak, before))?.0
            }
            _ => even,
        };
        Ok(Entry { even, odd, start })
    }

    /// An entry's primitives where the piece it begins in starts at `piece`, its first position
    /// `lo`.
    fn placed(
        self,
        f: &mut FunctionBuilder<'_>,
        entry: Entry,
        (lo, piece): (Value, Value),
    ) -> BResult<Value> {
        if self.program != RewriteProgram::QuadStrip {
            return Ok(entry.even);
        }
        let zero = k(f, 0)?;
        let one = k(f, 1)?;
        let into = f.binary(BinOp::ISub, lo, piece)?;
        let parity = f.binary(BinOp::BitwiseAnd, into, one)?;
        let odd = f.binary(BinOp::INotEqual, parity, zero)?;
        f.select(odd, entry.odd, entry.even)
    }
}

/// The entries of a scan: `n` of the workgroup lists' first, entry `i` starting at run
/// `first + i · stride` (its first position, which a quad strip's parity reads).
#[derive(Clone, Copy)]
struct Entries {
    n: Value,
    first: Value,
    stride: Value,
}

/// The first invocation's sums over `entries` from the piece start `carry`, without changing the
/// lists: their primitives, and the start of the last piece among them (0 for none).
fn total(
    f: &mut FunctionBuilder<'_>,
    g: Globals,
    d: DrawWalk,
    entries: Entries,
    carry: Value,
) -> BResult<(Value, Value)> {
    let zero = k(f, 0)?;
    let one = k(f, 1)?;
    let place = g.local(f, zero)?;
    let carry = g.local(f, carry)?;
    let last = g.local(f, zero)?;
    for_range(f, g.u32_, (zero, entries.n, one), move |f, i| {
        let entry = g.entry(f, i)?;
        let t = f.binary(BinOp::IMul, i, entries.stride)?;
        let t = f.binary(BinOp::IAdd, entries.first, t)?;
        let lo = d.lo_of(f, t)?;
        let begun = f.load(carry, MemAccess::default())?;
        let count = d.placed(f, entry, (lo, begun))?;
        let before = f.load(place, MemAccess::default())?;
        let next = saturating_add(f, before, count)?;
        f.store(place, next, MemAccess::default())?;
        let zero = k(f, 0)?;
        let restarted = f.binary(BinOp::INotEqual, entry.start, zero)?;
        let next = f.select(restarted, entry.start, begun)?;
        f.store(carry, next, MemAccess::default())?;
        let l = f.load(last, MemAccess::default())?;
        let l = f.glsl(GlslOp::UMax, &[l, entry.start])?;
        f.store(last, l, MemAccess::default())
    })?;
    let place = f.load(place, MemAccess::default())?;
    let last = f.load(last, MemAccess::default())?;
    Ok((place, last))
}

/// The first invocation's scan of `entries`, in order, from `place` and `carry`: each even count
/// becomes the primitives before its entry (`place` and the counts before it, saturated, each
/// taken at its entry's parity), each start the start of the piece its entry begins in (`carry`,
/// or the last start before it). The two totals' pointers.
fn scan(
    f: &mut FunctionBuilder<'_>,
    g: Globals,
    d: DrawWalk,
    entries: Entries,
    (place, carry): (Value, Value),
) -> BResult<(Value, Value)> {
    let zero = k(f, 0)?;
    let one = k(f, 1)?;
    let place = g.local(f, place)?;
    let carry = g.local(f, carry)?;
    for_range(f, g.u32_, (zero, entries.n, one), move |f, i| {
        let entry = g.entry(f, i)?;
        let t = f.binary(BinOp::IMul, i, entries.stride)?;
        let t = f.binary(BinOp::IAdd, entries.first, t)?;
        let lo = d.lo_of(f, t)?;
        let begun = f.load(carry, MemAccess::default())?;
        let count = d.placed(f, entry, (lo, begun))?;
        let before = f.load(place, MemAccess::default())?;
        let p = f.access_chain(g.counts, &[i])?;
        f.store(p, before, MemAccess::default())?;
        let next = saturating_add(f, before, count)?;
        f.store(place, next, MemAccess::default())?;
        let p = f.access_chain(g.starts, &[i])?;
        f.store(p, begun, MemAccess::default())?;
        let zero = k(f, 0)?;
        let restarted = f.binary(BinOp::INotEqual, entry.start, zero)?;
        let next = f.select(restarted, entry.start, begun)?;
        f.store(carry, next, MemAccess::default())
    })?;
    Ok((place, carry))
}

/// [`RewritePhase::Count`]: each run's primitives and last piece's start, summed over the
/// workgroup into its group's words, under both parities of the group's first piece's positions
/// before it.
fn count_phase(
    f: &mut FunctionBuilder<'_>,
    g: Globals,
    d: DrawWalk,
    at: Invocation,
) -> BResult<()> {
    let run = d.run(f, at)?;
    let (streak, _) = d.behind(f, run.lo)?;
    let entry = d.entry(f, g, run, streak)?;
    g.list(f, at.lin, entry)?;
    barrier(f)?;
    let zero = k(f, 0)?;
    let first = f.binary(BinOp::IEqual, at.lin, zero)?;
    let among = f.binary(BinOp::ULessThan, at.group, d.groups)?;
    let first = f.binary(BinOp::LogicalAnd, first, among)?;
    when(f, first, move |f| {
        let one = k(f, 1)?;
        let two = k(f, 2)?;
        let lanes = k(f, IndexRewriteModule::LOCAL_SIZE)?;
        let first_run = f.binary(BinOp::IMul, at.group, lanes)?;
        let entries = Entries {
            n: lanes,
            first: first_run,
            stride: one,
        };
        let lo = d.lo_of(f, first_run)?;
        let (even, last) = total(f, g, d, entries, lo)?;
        let odd = match d.program {
            RewriteProgram::QuadStrip => {
                let before = f.binary(BinOp::ISub, lo, one)?;
                total(f, g, d, entries, before)?.0
            }
            _ => even,
        };
        let w = g.group_word(f, at.group)?;
        g.word_store(f, w, even)?;
        let w1 = f.binary(BinOp::IAdd, w, one)?;
        g.word_store(f, w1, odd)?;
        let w2 = f.binary(BinOp::IAdd, w, two)?;
        g.word_store(f, w2, last)
    })
}

/// The primitives past the walk, every position there reading 0, after the walk's last piece,
/// which starts at `last`: none where `k'` is 0 and the program restarts (each restarts); else a
/// strip's, line loop's and fan's one at each position, quads' two at every fourth position of
/// the draw, a list's one at every second or third, and a quad strip's two at every second
/// position of its piece (the walk ends the span past the buffer, so the span reaches back from
/// each).
fn tail_primitives(f: &mut FunctionBuilder<'_>, d: DrawWalk, last: Value) -> BResult<Value> {
    let zero = k(f, 0)?;
    let two = k(f, 2)?;
    let some = f.binary(BinOp::INotEqual, d.restart, zero)?;
    match d.program {
        RewriteProgram::LineStrip | RewriteProgram::TriangleStrip => f.select(some, d.tail, zero),
        RewriteProgram::LineLoop | RewriteProgram::TriangleFan => Ok(d.tail),
        RewriteProgram::Quads | RewriteProgram::LineList | RewriteProgram::TriangleList => {
            // The positions `q` in `[walked, count)` with `q mod span = span - 1`, each of
            // `per_step` primitives.
            let span = k(f, d.program.span())?;
            let to = f.binary(BinOp::UDiv, d.count, span)?;
            let from = f.binary(BinOp::UDiv, d.walked, span)?;
            let steps = f.binary(BinOp::ISub, to, from)?;
            let per = k(f, d.program.per_step())?;
            f.binary(BinOp::IMul, steps, per)
        }
        RewriteProgram::QuadStrip => {
            // The positions `q` in `[walked, count)` with `q - last` odd.
            let pairs = |f: &mut FunctionBuilder<'_>, n: Value| -> BResult<Value> {
                let one = k(f, 1)?;
                let into = f.binary(BinOp::ISub, n, last)?;
                f.binary(BinOp::ShiftRightLogical, into, one)
            };
            let to = pairs(f, d.count)?;
            let from = pairs(f, d.walked)?;
            let quads = f.binary(BinOp::ISub, to, from)?;
            let quads = f.binary(BinOp::IMul, quads, two)?;
            f.select(some, quads, zero)
        }
    }
}

/// [`RewritePhase::Place`]: each group's words become the primitives before it and the start of
/// the piece it begins in; the output is reserved, the draw written, and what the write
/// dispatch needs stated.
fn place_phase(f: &mut FunctionBuilder<'_>, g: Globals, d: DrawWalk, lin: Value) -> BResult<()> {
    let lanes = k(f, IndexRewriteModule::LOCAL_SIZE)?;
    // The groups' words into the workgroup lists, the invocations taking turns.
    for_range(f, g.u32_, (lin, d.groups, lanes), move |f, i| {
        let one = k(f, 1)?;
        let two = k(f, 2)?;
        let w = g.group_word(f, i)?;
        let even = g.word_load(f, w)?;
        let w1 = f.binary(BinOp::IAdd, w, one)?;
        let odd = g.word_load(f, w1)?;
        let w2 = f.binary(BinOp::IAdd, w, two)?;
        let start = g.word_load(f, w2)?;
        g.list(f, i, Entry { even, odd, start })
    })?;
    barrier(f)?;
    let zero = k(f, 0)?;
    let first = f.binary(BinOp::IEqual, lin, zero)?;
    when(f, first, move |f| {
        let zero = k(f, 0)?;
        let one = k(f, 1)?;
        let vk_ = k(f, d.program.vertices())?;
        let lanes = k(f, IndexRewriteModule::LOCAL_SIZE)?;
        let entries = Entries {
            n: d.groups,
            first: zero,
            stride: lanes,
        };
        let (total, last) = scan(f, g, d, entries, (zero, zero))?;
        for_range(f, g.u32_, (zero, d.groups, one), move |f, i| {
            let one = k(f, 1)?;
            let w = g.group_word(f, i)?;
            let p = f.access_chain(g.counts, &[i])?;
            let before = f.load(p, MemAccess::default())?;
            g.word_store(f, w, before)?;
            let w = f.binary(BinOp::IAdd, w, one)?;
            let p = f.access_chain(g.starts, &[i])?;
            let begun = f.load(p, MemAccess::default())?;
            g.word_store(f, w, begun)
        })?;
        let walked_total = f.load(total, MemAccess::default())?;
        let last = f.load(last, MemAccess::default())?;
        let tail_drawn = tail_primitives(f, d, last)?;
        let primitives = saturating_add(f, walked_total, tail_drawn)?;
        // A line loop's last segment, from its last position back to its first.
        let primitives = match d.program {
            RewriteProgram::LineLoop => {
                let two = k(f, 2)?;
                let closes = f.binary(BinOp::UGreaterThanEqual, d.count, two)?;
                let closing = f.select(closes, one, zero)?;
                saturating_add(f, primitives, closing)?
            }
            _ => primitives,
        };
        // The output's indices, where they fit a `u32`: where the product's high word is 0
        // (`replay::checked_mul`, whose docs say why). In the private list form Metal's base
        // vertex takes a word before them.
        let (written, past) = checked_mul(f, primitives, vk_, zero)?;
        let words_needed = f.binary(BinOp::IAdd, written, d.listed)?;
        let wrapped = f.binary(BinOp::ULessThan, words_needed, written)?;
        let past = f.binary(BinOp::LogicalOr, past, wrapped)?;
        let countable = f.unary(UnOp::LogicalNot, past)?;
        // Placed in the pool's room, or in the draw's own region (a run's draws' always: the run
        // reads no counter).
        let counter = match g.slot {
            Some(_) => None,
            None => {
                let counter_ptr = word_ptr(f, g.output, zero)?;
                let counter = f.load(counter_ptr, MemAccess::default())?;
                Some((counter_ptr, counter))
            }
        };
        let used = match counter {
            Some((_, counter)) => f.glsl(GlslOp::UMin, &[counter, d.capacity])?,
            None => zero,
        };
        let room = f.binary(BinOp::ISub, d.capacity, used)?;
        let four = k(f, 4)?;
        let room = f.binary(BinOp::UDiv, room, four)?;
        let room = f.select(d.pooled, room, d.capacity)?;
        let fits = f.binary(BinOp::ULessThanEqual, words_needed, room)?;
        let ok = f.binary(BinOp::LogicalAnd, countable, fits)?;
        let reserve = f.binary(BinOp::LogicalAnd, ok, d.pooled)?;
        if let Some((counter_ptr, _)) = counter {
            when(f, reserve, move |f| {
                let four = k(f, 4)?;
                let bytes = f.binary(BinOp::IMul, words_needed, four)?;
                let next = f.binary(BinOp::IAdd, used, bytes)?;
                f.store(counter_ptr, next, MemAccess::default())
            })?;
        }
        // The list's first index, past the base vertex's word in the private list form.
        let first_index = f.binary(BinOp::UDiv, used, four)?;
        let first_index = f.select(d.pooled, first_index, d.owned)?;
        let first_index = f.binary(BinOp::IAdd, first_index, d.listed)?;
        let header = k(f, (IndexRewritePass::POOL_HEADER_BYTES / 4) as u32)?;
        let pool_base = f.binary(BinOp::IAdd, header, first_index)?;
        let base_word = f.select(d.pooled, pool_base, first_index)?;
        let ok_word = f.select(ok, one, zero)?;
        for (word, value) in [
            (Output::Base, base_word),
            (Output::Written, ok_word),
            (Output::TailAt, walked_total),
            (Output::Tail, tail_drawn),
        ] {
            g.word_store_at(f, word as u32, value)?;
        }
        let drawn = f.select(ok, written, zero)?;
        let primitives_drawn = f.select(ok, primitives, zero)?;
        // An indexed draw's `VkDrawIndexedIndirectCommand`, or in the private list form a
        // `VkDrawIndirectCommand` of the list's positions from its first index as first vertex.
        let listed = f.binary(BinOp::INotEqual, d.listed, zero)?;
        let fourth = f.select(listed, d.base_instance, d.base)?;
        let fifth = f.select(listed, zero, d.base_instance)?;
        for (i, value) in [drawn, d.instances, first_index, fourth, fifth]
            .into_iter()
            .enumerate()
        {
            g.word_store_at(f, Output::Draw as u32 + i as u32, value)?;
        }
        g.word_store_at(f, Output::Primitives as u32, primitives_drawn)?;
        // A draw whose primitives the output cannot hold draws nothing and says so.
        let over = f.unary(UnOp::LogicalNot, ok)?;
        when(f, over, move |f| {
            let word = k(f, StatusWord::Status.index())?;
            let bit = k(f, StatusBits::INDEX_POOL_OVERFLOW.0)?;
            let p = word_ptr(f, g.status, word)?;
            f.atomic(AtomicOp::Or, p, Scope::Device, Semantics::RELAXED, bit)?;
            Ok(())
        })
    })
}

/// The list primitives a completing position `q` writes, from the indices at `q - 3` to `q`
/// (`a`, `b`, `c`, `value`), a fan's first index `hub`, and the start of its piece `piece`, each
/// in the draw's order ([`DrawWalk::ordered`]).
fn primitives(
    f: &mut FunctionBuilder<'_>,
    d: DrawWalk,
    (a, b, c, value): (Value, Value, Value, Value),
    (hub, q, piece): (Value, Value, Value),
) -> BResult<Vec<Vec<Value>>> {
    let same = |f: &mut FunctionBuilder<'_>, p: &[Value]| d.ordered(f, p, p);
    Ok(match d.program {
        RewriteProgram::LineStrip | RewriteProgram::LineLoop | RewriteProgram::LineList => {
            vec![same(f, &[c, value])?]
        }
        RewriteProgram::TriangleStrip => {
            // Vulkan's strip orders: the piece's `i`th triangle (`q - piece - 2`) of positions
            // `(i, i + 1, i + 2)` takes them so where `i` is even; where it is odd, `(i, i + 2,
            // i + 1)` at mode 0 and `(i + 1, i, i + 2)` at mode 1.
            let zero = k(f, 0)?;
            let one = k(f, 1)?;
            let i = f.binary(BinOp::ISub, q, piece)?;
            let parity = f.binary(BinOp::BitwiseAnd, i, one)?;
            let odd = f.binary(BinOp::INotEqual, parity, zero)?;
            let second = f.select(odd, value, c)?;
            let third = f.select(odd, c, value)?;
            let led = f.select(odd, c, b)?;
            let then = f.select(odd, b, c)?;
            vec![d.ordered(f, &[b, second, third], &[led, then, value])?]
        }
        RewriteProgram::TriangleFan => vec![same(f, &[hub, c, value])?],
        RewriteProgram::Quads => vec![same(f, &[a, b, value])?, same(f, &[b, c, value])?],
        RewriteProgram::QuadStrip => vec![same(f, &[a, b, value])?, same(f, &[c, a, value])?],
        RewriteProgram::TriangleList => vec![same(f, &[b, c, value])?],
    })
}

/// [`RewritePhase::Write`]: each run, placed after the runs before it from its group's words,
/// writes its primitives where the draw is written; the runs then write the primitives past the
/// walk, and the first invocation a line loop's last segment.
fn write_phase(
    f: &mut FunctionBuilder<'_>,
    g: Globals,
    d: DrawWalk,
    at: Invocation,
) -> BResult<()> {
    let program = d.program;
    let run = d.run(f, at)?;
    let (streak, behind) = d.behind(f, run.lo)?;
    let entry = d.entry(f, g, run, streak)?;
    g.list(f, at.lin, entry)?;
    barrier(f)?;
    let zero = k(f, 0)?;
    let first = f.binary(BinOp::IEqual, at.lin, zero)?;
    let among = f.binary(BinOp::ULessThan, at.group, d.groups)?;
    let first = f.binary(BinOp::LogicalAnd, first, among)?;
    when(f, first, move |f| {
        let one = k(f, 1)?;
        let lanes = k(f, IndexRewriteModule::LOCAL_SIZE)?;
        let w = g.group_word(f, at.group)?;
        let place = g.word_load(f, w)?;
        let w = f.binary(BinOp::IAdd, w, one)?;
        let carry = g.word_load(f, w)?;
        let first_run = f.binary(BinOp::IMul, at.group, lanes)?;
        let entries = Entries {
            n: lanes,
            first: first_run,
            stride: one,
        };
        scan(f, g, d, entries, (place, carry))?;
        Ok(())
    })?;
    barrier(f)?;
    let written = g.word_load_at(f, Output::Written as u32)?;
    let ok = f.binary(BinOp::INotEqual, written, zero)?;
    let base_word = g.word_load_at(f, Output::Base as u32)?;
    // In the private list form, Metal's base vertex in the word before the list, written once.
    let listed = f.binary(BinOp::INotEqual, d.listed, zero)?;
    let base_too = f.binary(BinOp::LogicalAnd, ok, listed)?;
    let group0 = f.binary(BinOp::IEqual, at.group, zero)?;
    let base_too = f.binary(BinOp::LogicalAnd, base_too, first)?;
    let base_too = f.binary(BinOp::LogicalAnd, base_too, group0)?;
    when(f, base_too, move |f| {
        let one = k(f, 1)?;
        let before = f.binary(BinOp::ISub, base_word, one)?;
        contained_store(f, g.output, before, d.base)
    })?;
    let v = program.vertices();
    // A fan's first index, every triangle's first vertex.
    let hub = match program {
        RewriteProgram::TriangleFan => d.fetch.at(f, zero)?,
        _ => zero,
    };
    when(f, ok, move |f| {
        let one = k(f, 1)?;
        let vk_ = k(f, v)?;
        let p = f.access_chain(g.counts, &[at.lin])?;
        let place = f.load(p, MemAccess::default())?;
        let p = f.access_chain(g.starts, &[at.lin])?;
        let begun = f.load(p, MemAccess::default())?;
        let into = f.binary(BinOp::IMul, place, vk_)?;
        let at0 = f.binary(BinOp::IAdd, base_word, into)?;
        let next_word = g.local(f, at0)?;
        let streak = g.local(f, streak)?;
        let piece = g.local(f, begun)?;
        // The indices one, two and three positions back.
        let old = g.local(f, behind[0])?;
        let older = g.local(f, behind[1])?;
        let oldest = g.local(f, behind[2])?;
        for_range(f, g.u32_, (run.lo, run.hi, one), move |f, q| {
            let value = d.fetch.at(f, q)?;
            let p = f.load(piece, MemAccess::default())?;
            let (restarts, completes) = d.step(f, (streak, q, p), value)?;
            let one = k(f, 1)?;
            let next = f.binary(BinOp::IAdd, q, one)?;
            let p_next = f.select(restarts, next, p)?;
            f.store(piece, p_next, MemAccess::default())?;
            let a = f.load(oldest, MemAccess::default())?;
            let b = f.load(older, MemAccess::default())?;
            let c = f.load(old, MemAccess::default())?;
            when(f, completes, move |f| {
                let w = f.load(next_word, MemAccess::default())?;
                let written = primitives(f, d, (a, b, c, value), (hub, q, p))?;
                let mut slot = 0;
                for primitive in written {
                    for index in primitive {
                        let i = k(f, slot)?;
                        let at = f.binary(BinOp::IAdd, w, i)?;
                        contained_store(f, g.output, at, index)?;
                        slot += 1;
                    }
                }
                let n = k(f, slot)?;
                let next = f.binary(BinOp::IAdd, w, n)?;
                f.store(next_word, next, MemAccess::default())
            })?;
            f.store(oldest, b, MemAccess::default())?;
            f.store(older, c, MemAccess::default())?;
            f.store(old, value, MemAccess::default())
        })?;
        // The primitives past the walk, the runs taking turns: every index 0, but a fan's first.
        let tail_at = g.word_load_at(f, Output::TailAt as u32)?;
        let tail = g.word_load_at(f, Output::Tail as u32)?;
        for_range(f, g.u32_, (run.t, tail, run.runs), move |f, j| {
            let vk_ = k(f, v)?;
            let zero = k(f, 0)?;
            let primitive = f.binary(BinOp::IAdd, tail_at, j)?;
            let into = f.binary(BinOp::IMul, primitive, vk_)?;
            let w = f.binary(BinOp::IAdd, base_word, into)?;
            let indices: Vec<Value> = (0..v)
                .map(|i| match (program, i) {
                    (RewriteProgram::TriangleFan, 0) => hub,
                    _ => zero,
                })
                .collect();
            let indices = d.ordered(f, &indices, &indices)?;
            for (i, index) in indices.into_iter().enumerate() {
                let i = k(f, i as u32)?;
                let slot = f.binary(BinOp::IAdd, w, i)?;
                contained_store(f, g.output, slot, index)?;
            }
            Ok(())
        })?;
        // A line loop's last segment, after every other: from its last position to its first.
        if program == RewriteProgram::LineLoop {
            let zero = k(f, 0)?;
            let two = k(f, 2)?;
            let closes = f.binary(BinOp::UGreaterThanEqual, d.count, two)?;
            let t0 = f.binary(BinOp::IEqual, run.t, zero)?;
            let writes = f.binary(BinOp::LogicalAnd, closes, t0)?;
            when(f, writes, move |f| {
                let one = k(f, 1)?;
                let two = k(f, 2)?;
                let zero = k(f, 0)?;
                let last = f.binary(BinOp::ISub, d.count, one)?;
                let from = d.fetch.at(f, last)?;
                let to = d.fetch.at(f, zero)?;
                let segment = d.ordered(f, &[from, to], &[from, to])?;
                let primitive = f.binary(BinOp::IAdd, tail_at, tail)?;
                let into = f.binary(BinOp::IMul, primitive, two)?;
                let w = f.binary(BinOp::IAdd, base_word, into)?;
                contained_store(f, g.output, w, segment[0])?;
                let w = f.binary(BinOp::IAdd, w, one)?;
                contained_store(f, g.output, w, segment[1])
            })?;
        }
        Ok(())
    })
}

/// The fixed binding numbers of the module's symbols.
fn binding_of(sym: &Symbol) -> Option<u32> {
    match sym {
        Symbol::DescriptorSet(_) => Some(IndexRewriteModule::SET),
        Symbol::Binding(s) if *s == WORDS => Some(IndexRewriteModule::WORDS_BINDING),
        Symbol::Binding(s) if *s == ARGUMENTS => Some(IndexRewriteModule::ARGUMENTS_BINDING),
        Symbol::Binding(s) if *s == INDICES => Some(IndexRewriteModule::INDICES_BINDING),
        Symbol::Binding(s) if *s == OUTPUT => Some(IndexRewriteModule::OUTPUT_BINDING),
        Symbol::Binding(s) if *s == STATUS => Some(IndexRewriteModule::STATUS_BINDING),
        _ => None,
    }
}

/// The index rewrite module of `program`'s dispatch of `phase` for `device`: built with the typed
/// builder, verified, serialized, and numbered with the fixed bindings of [`IndexRewriteModule`].
pub fn index_rewrite_module(
    device: &Device,
    program: RewriteProgram,
    phase: RewritePhase,
) -> Result<IndexRewriteModule, Error> {
    let (spirv, requirements) = finished(
        device,
        rewrite_typed_module(program, phase)?,
        binding_of,
        "the index rewrite",
    )?;
    Ok(IndexRewriteModule {
        spirv,
        requirements,
    })
}

/// The index rewrite module of a run of `program`'s draws ([`IndexRewriteRun`]) for `device`,
/// as [`index_rewrite_module`]'s phases are made: one workgroup per draw, each draw's three
/// phases in turn.
pub fn index_rewrite_run_module(
    device: &Device,
    program: RewriteProgram,
) -> Result<IndexRewriteModule, Error> {
    let (spirv, requirements) = finished(
        device,
        rewrite_run_typed_module(program)?,
        binding_of,
        "the index rewrite run",
    )?;
    Ok(IndexRewriteModule {
        spirv,
        requirements,
    })
}

mod run;
pub use run::{IndexRewriteRun, RewriteRunDraw};

#[cfg(test)]
mod tests;
