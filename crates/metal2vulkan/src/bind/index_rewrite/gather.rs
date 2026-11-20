//! The index gather pass: the lists of a run of draws whose primitives follow from their
//! positions alone, written by one dispatch for every draw of the run, where the index rewrite
//! pass takes three dispatches per draw (the parent module). A run holds direct indexed draws of a
//! program that never restarts ([`RewriteProgram::gathers`]: line and triangle lists, Apple's
//! triangle fan and quads) over one index buffer, or indirect draws over one arguments buffer:
//! indexed ones of such a program over one index buffer, or ones without indices of any program
//! but a line loop ([`RewriteProgram::gathers_without_indices`]), whose positions never restart.
//! Such a draw's list is its positions' list ([`RewriteProgram::sequential_list`] in the draw's
//! order, whose start is the list of a draw of fewer positions) with each position read as the
//! fetch reads it: its index where it lies inside the buffer's whole indices from the draw's
//! offset, else 0; a draw without indices, the position itself. That is the list the rewrite pass
//! writes for the draw, index for index.
//!
//! - **The pattern** ([`IndexGatherModule::PATTERN_BINDING`], [`IndexGather::pattern`]): the
//!   positions' list in closed form, which every draw of the run reads from its start: its period
//!   `P` (word 0), then `P` multipliers and `P` addends, the list's index `o` being the position
//!   `mult[o % P] * (o / P) + add[o % P]` ([`RewriteProgram::gather_pattern`]).
//! - **The table** ([`IndexGather::table`], a storage buffer at
//!   [`IndexGatherModule::TABLE_BINDING`]): [`IndexGatherModule::HEADER_WORDS`] words of header,
//!   then one record of [`IndexGatherModule::RECORD_WORDS`] words per draw.
//!
//!   | Header word | Written by | Contents |
//!   |---|---|---|
//!   | 0 | the consumer | the run's draws |
//!   | 1 | the consumer | 1 where its draws have no indices, else 0 |
//!   | 2 | the consumer (direct), the plan (indirect) | the run's workgroups |
//!   | 3 | the consumer | an indirect run's: the index pool's index bytes |
//!   | 4 to 6 | the plan | an indirect run's: the dispatch's `VkDispatchIndirectCommand` ([`IndexGather::DISPATCH_OFFSET`]) |
//!   | 7 to 9 | the consumer | an indirect run's: its program's terms of a list's indices from Metal's count `n`, `per * ((n - lead) / step)` ([`RewriteProgram::capacity_terms`]): `lead`, `step`, `per` |
//!
//!   | Record word | Written by | Contents |
//!   |---|---|---|
//!   | 0 | the consumer (direct), the plan (indirect) | the draw's first workgroup |
//!   | 1 | the consumer, the plan | its list's indices |
//!   | 2 | the consumer, the plan | its list's first word in the output |
//!   | 3 | the consumer | its index offset's byte in the bound indices range |
//!   | 4 | the consumer | its whole indices from the offset (saturated to `u32`) |
//!   | 5 | the consumer | 1 for 32-bit indices |
//!   | 6 | the plan | an indirect draw's first index (Metal's `indexStart`), 0 without indices |
//!   | 7 | the consumer | an indirect draw's arguments' first word in the bound arguments range |
//!
//! - **The dispatch**: workgroups of [`IndexGatherModule::LOCAL_SIZE`] invocations over the run's
//!   workgroups (header word 2), each draw's list over workgroups of its own from its first; a
//!   workgroup finds its draw by its first workgroup (a binary search of the records), and each
//!   invocation writes one index of its draw's list. A dispatch of fewer workgroups takes the
//!   run's workgroups in turn: its own index's, then each the dispatch's workgroups past it.
//! - **An indirect run** is planned first: the plan ([`plan::index_gather_plan_module`], one
//!   workgroup) reads each draw's Metal arguments, reserves its list in the command buffer's index
//!   pool (the index bounds pass's, `index_bounds.rs`), and writes its record, the run's
//!   workgroups and dispatch, and each draw's `VkDrawIndexedIndirectCommand`
//!   ([`plan::IndexGatherPlanModule`]); the gather then dispatches indirectly from the table's
//!   [`IndexGather::DISPATCH_OFFSET`], its output the whole pool.
//! - **Bindings** (set 0, storage buffers): the table, the pattern, the indices
//!   ([`IndexGather::indices_range`]: the index buffer's bytes the consumer names, so one binding
//!   serves every draw of the run; any 4 bytes where it is `None`) and the output (a direct run's
//!   own, or the whole pool). Every read is contained (a word past the bound range reads 0), and
//!   every write lies inside the output.
//! - **Synchronization** (consumer): the indices' writes and the table's upload → `SHADER_READ`
//!   in `COMPUTE_SHADER` before the dispatch (an indirect run's plan before it, its writes →
//!   `SHADER_READ` in `COMPUTE_SHADER` and `INDIRECT_COMMAND_READ` in `DRAW_INDIRECT`); its
//!   `SHADER_WRITE` → `INDEX_READ` in `INDEX_INPUT` before the draws (and the plan's →
//!   `INDIRECT_COMMAND_READ` in `DRAW_INDIRECT`).

use super::{
    contained, contained_store, finished, for_range, k, when, words_buffer, Fetch, RewriteOrder,
    RewriteProgram, Symbol, WORDS,
};
use crate::bind::index_bounds::word_range;
use crate::bind::instantiate::BufferRange;
use crate::device::requirements::Requirements;
use crate::device::Device;
use crate::error::Error;
use crate::pipeline::sym::{ResourceKind, ResourceSlot, Stage};
use crate::spirv::builder::FunctionBuilder;
use crate::spirv::builder::ModuleBuilder;
use crate::spirv::error::BResult;
use crate::spirv::ids::{TypeId, Value};
use crate::spirv::ir::{BinOp, ExecMode, ExecModel, GlslOp, IoDir, MemAccess, Type, TypedModule};
use std::sync::Arc;

pub mod plan;

impl RewriteProgram {
    /// Whether the index gather pass writes an indexed draw's list (the module docs): the
    /// programs that never restart and whose positions' list for fewer positions is the start
    /// of the list for more, a line loop's not (its last segment closes on the draw's last
    /// position).
    pub fn gathers(self) -> bool {
        matches!(
            self,
            RewriteProgram::LineList
                | RewriteProgram::TriangleList
                | RewriteProgram::TriangleFan
                | RewriteProgram::Quads
        )
    }

    /// Whether the index gather pass writes the list of a draw without indices, whose positions
    /// never restart: every program but a line loop.
    pub fn gathers_without_indices(self) -> bool {
        self != RewriteProgram::LineLoop
    }

    /// The positions' list in `order` in closed form ([`IndexGather::pattern`]): its period `P`
    /// and each of its `P` places' multiplier and addend, the list's index `o` being the position
    /// `mult[o % P] * (o / P) + add[o % P]`; `None` for a line loop.
    pub fn gather_pattern(self, order: RewriteOrder) -> Option<(u32, Vec<u32>, Vec<u32>)> {
        let period: usize = match self {
            RewriteProgram::LineLoop => return None,
            RewriteProgram::LineStrip | RewriteProgram::LineList => 2,
            RewriteProgram::TriangleFan | RewriteProgram::TriangleList => 3,
            RewriteProgram::TriangleStrip | RewriteProgram::Quads | RewriteProgram::QuadStrip => 6,
        };
        // Two periods of the list: the first's places, and the step to the second's. Twelve
        // positions hold them for every program.
        let list = self.sequential_list(12, order);
        let add: Vec<u32> = list.get(..period)?.to_vec();
        let mult = (0..period)
            .map(|r| Some(list.get(period + r)? - list[r]))
            .collect::<Option<Vec<u32>>>()?;
        Some((period as u32, mult, add))
    }

    /// The terms of [`capacity`](Self::capacity) of a draw of `n` positions: `(lead, step, per)`,
    /// its indices `per * ((n - lead) / step)` (0 where `n` is below `lead`); `None` for a line
    /// loop, whose last segment closes on its first position.
    pub fn capacity_terms(self) -> Option<(u32, u32, u32)> {
        Some(match self {
            RewriteProgram::LineLoop => return None,
            RewriteProgram::LineStrip => (1, 1, 2),
            RewriteProgram::TriangleStrip | RewriteProgram::TriangleFan => (2, 1, 3),
            RewriteProgram::Quads => (0, 4, 6),
            RewriteProgram::QuadStrip => (2, 2, 6),
            RewriteProgram::LineList => (0, 2, 2),
            RewriteProgram::TriangleList => (0, 3, 3),
        })
    }
}

/// The crate-owned gather module on one device, validated like any output.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IndexGatherModule {
    /// SPIR-V 1.5 words, entry point `main`, bindings numbered as below in set 0.
    pub spirv: Arc<[u32]>,
    /// What the module needs from the device (within every Vulkan 1.2 device).
    pub requirements: Requirements,
}

impl IndexGatherModule {
    /// Descriptor set of every binding.
    pub const SET: u32 = 0;
    /// The run's table ([`IndexGather::table`]), a storage buffer.
    pub const TABLE_BINDING: u32 = 0;
    /// The run's pattern ([`IndexGather::pattern`]), a storage buffer.
    pub const PATTERN_BINDING: u32 = 1;
    /// The Metal index buffer, a storage buffer.
    pub const INDICES_BINDING: u32 = 2;
    /// The run's lists, a storage buffer.
    pub const OUTPUT_BINDING: u32 = 3;
    /// Invocations of each workgroup, along x.
    pub const LOCAL_SIZE: u32 = 128;
    /// The table's words before its records.
    pub const HEADER_WORDS: u32 = 12;
    /// The words of one draw's record.
    pub const RECORD_WORDS: u32 = 8;
    /// The most workgroups of a dispatch: Vulkan's least `maxComputeWorkGroupCount[0]`. A direct
    /// run's workgroups are at most these.
    pub const MAX_GROUPS: u32 = 65_535;
}

/// The table's binding symbol ([`IndexGatherModule::TABLE_BINDING`]).
const TABLE: ResourceSlot = WORDS;
/// The pattern's binding symbol ([`IndexGatherModule::PATTERN_BINDING`]).
const PATTERN: ResourceSlot = ResourceSlot {
    stage: Stage::Kernel,
    kind: ResourceKind::Buffer,
    index: 1,
};
/// The indices' binding symbol ([`IndexGatherModule::INDICES_BINDING`]).
const INDICES: ResourceSlot = ResourceSlot {
    stage: Stage::Kernel,
    kind: ResourceKind::Buffer,
    index: 2,
};
/// The output's binding symbol ([`IndexGatherModule::OUTPUT_BINDING`]).
const OUTPUT: ResourceSlot = ResourceSlot {
    stage: Stage::Kernel,
    kind: ResourceKind::Buffer,
    index: 3,
};

/// One direct draw of a run: Metal's index count, its index offset and the end of the bytes its
/// fetch reads (in the bound buffer's offsets), and whether its indices are 32-bit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GatherDraw {
    /// Metal's index count: the draw's positions.
    pub count: u32,
    /// The draw's index offset: a multiple of its index size.
    pub index_offset: u64,
    /// Where the draw's fetch stops reading bytes: a position from there on reads index 0.
    pub index_end: u64,
    /// 4-byte indices, else 2-byte ones.
    pub wide: bool,
}

/// The whole indices of `wide` (4 bytes, else 2) from `offset` before `end`, saturated to `u32`.
fn whole(offset: u64, end: u64, wide: bool) -> u32 {
    let size = if wide { 4 } else { 2 };
    u32::try_from(end.saturating_sub(offset) / size).unwrap_or(u32::MAX)
}

/// One indirect draw of a run: where its Metal arguments lie (in the bound arguments buffer's
/// offsets), and for an indexed draw its index offset, the end of the bytes its fetch reads (in
/// the bound index buffer's offsets) and whether its indices are 32-bit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IndirectGatherDraw {
    /// The first byte of its `MTLDrawIndexedPrimitivesIndirectArguments` (of an indexed draw) or
    /// `MTLDrawPrimitivesIndirectArguments`: a multiple of 4.
    pub arguments_offset: u64,
    /// An indexed draw's index offset: a multiple of its index size. 0 without indices.
    pub index_offset: u64,
    /// Where an indexed draw's fetch stops reading bytes. 0 without indices.
    pub index_end: u64,
    /// 4-byte indices, else 2-byte ones. `false` without indices.
    pub wide: bool,
}

/// What a run's draws draw from: direct draws into an output of its own, or indirect draws over
/// one arguments buffer into the index pool.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Source {
    /// Direct draws into an output of `output_words` words.
    Direct { output_words: u32 },
    /// Indirect draws over the bound arguments range, without indices where `sequential`, into a
    /// pool of `pool_bytes` index bytes.
    Indirect {
        arguments: BufferRange,
        sequential: bool,
        pool_bytes: u32,
    },
}

/// A run of draws of one program in one order (over one Metal index buffer), into one output:
/// what the consumer uploads, binds and dispatches (the module docs).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IndexGather {
    program: RewriteProgram,
    order: RewriteOrder,
    source: Source,
    /// The bound indices range, and the end of the bytes it binds (a whole word).
    indices: Option<(BufferRange, u64)>,
    /// Each draw's record (the module docs), the plan's words of an indirect draw 0.
    draws: Vec<[u32; 8]>,
    groups: u32,
    written: u32,
}

impl IndexGather {
    /// The byte of an indirect run's `VkDispatchIndirectCommand` in its table (header words 4
    /// to 6).
    pub const DISPATCH_OFFSET: u64 = 16;

    /// An empty run of direct draws of `program` in `order` over an index buffer whose bytes from
    /// `index_start` to `index_end` it binds (its `VkBuffer` holds the bytes to `index_end`
    /// rounded up to 4), at an offset aligned to `alignment` (`minStorageBufferOffsetAlignment`)
    /// over at most `max_range` bytes (`maxStorageBufferRange`), into an output of
    /// `output_words` words. `None` for a program that does not gather
    /// ([`RewriteProgram::gathers`]), and where the bytes are past `max_range`.
    pub fn new(
        (program, order): (RewriteProgram, RewriteOrder),
        (index_start, index_end): (u64, u64),
        (alignment, max_range): (u64, u32),
        output_words: u32,
    ) -> Option<IndexGather> {
        if !program.gathers() {
            return None;
        }
        Some(IndexGather {
            program,
            order,
            source: Source::Direct { output_words },
            indices: indices_bound((index_start, index_end), (alignment, max_range))?,
            draws: Vec::new(),
            groups: 0,
            written: 0,
        })
    }

    /// An empty run of indirect draws of `program` in `order` whose arguments lie in the bytes
    /// from `arguments_start` to `arguments_end` of a buffer, and with `indices` the bytes of an
    /// index buffer it binds (`None` for draws without indices), each bound at an offset aligned
    /// to `alignment` over at most `max_range` bytes, into an index pool of `pool_bytes` index
    /// bytes. `None` for a program that does not gather draws of its kind
    /// ([`RewriteProgram::gathers`], [`RewriteProgram::gathers_without_indices`]), and where the
    /// bytes are none or past `max_range`.
    pub fn indirect(
        (program, order): (RewriteProgram, RewriteOrder),
        (arguments_start, arguments_end): (u64, u64),
        indices: Option<(u64, u64)>,
        (alignment, max_range): (u64, u32),
        pool_bytes: u32,
    ) -> Option<IndexGather> {
        let gathers = match indices {
            Some(_) => program.gathers(),
            None => program.gathers_without_indices(),
        };
        if !gathers || arguments_end <= arguments_start {
            return None;
        }
        let (arguments, _) = word_range(arguments_start, arguments_end, alignment, max_range)?;
        let bound = match indices {
            Some(bytes) => indices_bound(bytes, (alignment, max_range))?,
            None => None,
        };
        Some(IndexGather {
            program,
            order,
            source: Source::Indirect {
                arguments,
                sequential: indices.is_none(),
                pool_bytes,
            },
            indices: bound,
            draws: Vec::new(),
            groups: 0,
            written: 0,
        })
    }

    /// The indices of a direct `draw`'s list ([`RewriteProgram::capacity`] of its count), where
    /// they fit a `u32`.
    pub fn capacity(&self, draw: GatherDraw) -> Option<u32> {
        u32::try_from(self.program.capacity(u64::from(draw.count))).ok()
    }

    /// The record words of an index offset `offset` whose fetch stops at `end`, of `wide`
    /// indices, `inside` of which it reads: its byte in the bound range and `inside` (0 and 0
    /// where it reads none, whose byte is never read); `None` where it reads indices outside the
    /// bound bytes or its offset is not a multiple of its size.
    fn index_words(&self, offset: u64, end: u64, wide: bool, inside: u32) -> Option<(u32, u32)> {
        let size = if wide { 4 } else { 2 };
        if !offset.is_multiple_of(size) {
            return None;
        }
        if inside == 0 {
            return Some((0, 0));
        }
        let (range, bound_end) = self.indices?;
        if offset < range.offset || end > bound_end {
            return None;
        }
        Some((u32::try_from(offset - range.offset).ok()?, inside))
    }

    /// Add a direct `draw` to the run: its list's first word in the output. `None` where the run
    /// cannot take it, and is as before: an indirect run, a draw of no primitive, an offset not a
    /// multiple of the index size, indices read outside the bound bytes, a list past the output,
    /// or workgroups past [`IndexGatherModule::MAX_GROUPS`].
    pub fn push(&mut self, draw: GatherDraw) -> Option<u32> {
        let Source::Direct { output_words } = self.source else {
            return None;
        };
        let capacity = self.capacity(draw)?;
        if capacity == 0 {
            return None;
        }
        let inside = whole(draw.index_offset, draw.index_end, draw.wide).min(draw.count);
        let (byte, inside) =
            self.index_words(draw.index_offset, draw.index_end, draw.wide, inside)?;
        let first = self.written;
        let written = first.checked_add(capacity)?;
        let groups = self
            .groups
            .checked_add(capacity.div_ceil(IndexGatherModule::LOCAL_SIZE))?;
        if written > output_words || groups > IndexGatherModule::MAX_GROUPS {
            return None;
        }
        self.draws.push([
            self.groups,
            capacity,
            first,
            byte,
            inside,
            u32::from(draw.wide),
            0,
            0,
        ]);
        self.written = written;
        self.groups = groups;
        Some(first)
    }

    /// The run's draws.
    pub fn len(&self) -> usize {
        self.draws.len()
    }

    /// Whether the run has no draw.
    pub fn is_empty(&self) -> bool {
        self.draws.is_empty()
    }

    /// Whether the run's draws are indirect, planned on the GPU.
    pub fn is_indirect(&self) -> bool {
        matches!(self.source, Source::Indirect { .. })
    }

    /// The workgroups of a direct run's dispatch (an indirect run dispatches from its table's
    /// [`DISPATCH_OFFSET`](Self::DISPATCH_OFFSET)).
    pub fn groups(&self) -> u32 {
        self.groups
    }

    /// The output's words a direct run's lists take.
    pub fn written(&self) -> u32 {
        self.written
    }

    /// The range of the Metal index buffer to bind at [`IndexGatherModule::INDICES_BINDING`]:
    /// the bytes [`new`](Self::new) or [`indirect`](Self::indirect) names, from their start
    /// aligned down, to a whole word; `None` where they are none.
    pub fn indices_range(&self) -> Option<BufferRange> {
        self.indices.map(|(range, _)| range)
    }

    /// The range of an indirect run's arguments buffer to bind at the plan's
    /// [`ARGUMENTS_BINDING`](plan::IndexGatherPlanModule::ARGUMENTS_BINDING): the bytes
    /// [`indirect`](Self::indirect) names, from their start aligned down, to a whole word.
    /// `None` for a direct run.
    pub fn arguments_range(&self) -> Option<BufferRange> {
        match self.source {
            Source::Indirect { arguments, .. } => Some(arguments),
            Source::Direct { .. } => None,
        }
    }

    /// The pattern to bind at [`IndexGatherModule::PATTERN_BINDING`] (the module docs): the
    /// program's in the run's order ([`RewriteProgram::gather_pattern`]).
    pub fn pattern(&self) -> Vec<u32> {
        // Every program of a run has a pattern (`new`, `indirect`).
        let (period, mult, add) =
            self.program
                .gather_pattern(self.order)
                .unwrap_or((1, vec![0], vec![0]));
        let mut words = vec![period];
        words.extend(mult);
        words.extend(add);
        words
    }

    /// The table to upload: the header, then each draw's record (the module docs).
    pub fn table(&self) -> Vec<u32> {
        let mut table = vec![0; IndexGatherModule::HEADER_WORDS as usize];
        table[0] = self.draws.len() as u32;
        match self.source {
            Source::Direct { .. } => table[2] = self.groups,
            Source::Indirect {
                sequential,
                pool_bytes,
                ..
            } => {
                table[1] = u32::from(sequential);
                table[3] = pool_bytes;
                // Every program of an indirect run has terms (`indirect`).
                let (lead, step, per) = self.program.capacity_terms().unwrap_or((0, 1, 0));
                table[7..10].copy_from_slice(&[lead, step, per]);
            }
        }
        for record in &self.draws {
            table.extend(record);
        }
        table
    }

    /// The table's bytes ([`table`](Self::table)).
    pub fn table_bytes(&self) -> Vec<u8> {
        self.table().iter().flat_map(|w| w.to_le_bytes()).collect()
    }
}

/// The bound range of an index buffer's bytes from `start` to `end` (to a whole word past `end`:
/// a partial last index's fetch reads the bytes there, `IndexBufferRef::fetch_end`), and the end
/// of the bytes it binds; `Some(None)` where they are none, `None` where they are past
/// `max_range`.
fn indices_bound(
    (start, end): (u64, u64),
    (alignment, max_range): (u64, u32),
) -> Option<Option<(BufferRange, u64)>> {
    match end > start {
        true => {
            let (range, _) = word_range(start, end, alignment, max_range)?;
            Some(Some((range, range.offset + range.range)))
        }
        false => Some(None),
    }
}

/// The gather dispatch as a typed module, before serialization: the module
/// [`index_gather_module`] serializes, which the CPU interpreter runs.
pub(crate) fn gather_typed_module() -> BResult<TypedModule> {
    let mut b = ModuleBuilder::new();
    let u32_ = b.ty(Type::Int {
        width: 32,
        signed: false,
    })?;
    let uvec3 = b.ty(Type::Vector { elem: u32_, len: 3 })?;
    let table = words_buffer(&mut b, TABLE, u32_)?;
    let pattern = words_buffer(&mut b, PATTERN, u32_)?;
    let indices = words_buffer(&mut b, INDICES, u32_)?;
    let output = words_buffer(&mut b, OUTPUT, u32_)?;
    let invocation = b.builtin(
        crate::spv::BuiltIn::LocalInvocationIndex,
        IoDir::Input,
        u32_,
    )?;
    let workgroup = b.builtin(crate::spv::BuiltIn::WorkgroupId, IoDir::Input, uvec3)?;
    let dispatched = b.builtin(crate::spv::BuiltIn::NumWorkgroups, IoDir::Input, uvec3)?;
    let void = b.ty(Type::Void)?;
    let mut f = b.function(void)?;
    let table = f.module().global_value(table)?;
    let pattern = f.module().global_value(pattern)?;
    let indices = f.module().global_value(indices)?;
    let output = f.module().global_value(output)?;
    let invocation = f.module().global_value(invocation)?;
    let lin = f.load(invocation, MemAccess::default())?;
    let workgroup = f.module().global_value(workgroup)?;
    let workgroup = f.load(workgroup, MemAccess::default())?;
    let own = f.composite_extract(workgroup, &[0])?;
    let dispatched = f.module().global_value(dispatched)?;
    let dispatched = f.load(dispatched, MemAccess::default())?;
    let dispatched = f.composite_extract(dispatched, &[0])?;
    let zero = k(&mut f, 0)?;
    let one = k(&mut f, 1)?;
    let two = k(&mut f, 2)?;
    let draws = contained(&mut f, table, zero)?;
    let sequential = contained(&mut f, table, one)?;
    let sequential = f.binary(BinOp::INotEqual, sequential, zero)?;
    let groups = contained(&mut f, table, two)?;
    // The pattern's period (at least 1).
    let period = contained(&mut f, pattern, zero)?;
    let period = f.glsl(GlslOp::UMax, &[period, one])?;
    // A dispatch of no workgroup along x runs none of this.
    let stride = f.glsl(GlslOp::UMax, &[dispatched, one])?;
    let terms = Terms {
        table,
        pattern,
        indices,
        output,
        draws,
        sequential,
        period,
        lin,
    };
    // The run's workgroups: this one's own, then each the dispatch's workgroups past it.
    for_range(&mut f, u32_, (own, groups, stride), move |f, group| {
        gather_group(f, u32_, terms, group)
    })?;
    f.ret(None)?;
    let id = f.finish()?;
    b.entry(
        ExecModel::GLCompute,
        id,
        vec![ExecMode::LocalSize {
            x: IndexGatherModule::LOCAL_SIZE,
            y: 1,
            z: 1,
        }],
    )?;
    b.finalize()
}

/// What every workgroup of the gather reads: its buffers, the run's draws and whether they have
/// indices, the pattern's period, and the invocation's index in its workgroup.
#[derive(Clone, Copy)]
struct Terms {
    table: Value,
    pattern: Value,
    indices: Value,
    output: Value,
    draws: Value,
    sequential: Value,
    period: Value,
    lin: Value,
}

/// The run's workgroup `group`: the indices of its draw's list it writes, one per invocation.
fn gather_group(f: &mut FunctionBuilder<'_>, u32_: TypeId, t: Terms, group: Value) -> BResult<()> {
    let zero = k(f, 0)?;
    let one = k(f, 1)?;
    let header = k(f, IndexGatherModule::HEADER_WORDS)?;
    let per = k(f, IndexGatherModule::RECORD_WORDS)?;
    let table = t.table;
    // The draw: the last record whose first workgroup is at or before this one's, the first's
    // being 0. Each step halves `[lo, hi)`; 32 steps end every search a `u32` counts.
    let lo_var = f.local_var(u32_, None)?;
    let lo = f.var_ptr(lo_var)?;
    f.store(lo, zero, MemAccess::default())?;
    let hi_var = f.local_var(u32_, None)?;
    let hi = f.var_ptr(hi_var)?;
    f.store(hi, t.draws, MemAccess::default())?;
    let steps = k(f, 32)?;
    for_range(f, u32_, (zero, steps, one), move |f, _| {
        let l = f.load(lo, MemAccess::default())?;
        let h = f.load(hi, MemAccess::default())?;
        let span = f.binary(BinOp::ISub, h, l)?;
        let one = k(f, 1)?;
        let wide = f.binary(BinOp::UGreaterThan, span, one)?;
        when(f, wide, move |f| {
            let sum = f.binary(BinOp::IAdd, l, h)?;
            let one = k(f, 1)?;
            let mid = f.binary(BinOp::ShiftRightLogical, sum, one)?;
            let header = k(f, IndexGatherModule::HEADER_WORDS)?;
            let per = k(f, IndexGatherModule::RECORD_WORDS)?;
            let at = f.binary(BinOp::IMul, mid, per)?;
            let at = f.binary(BinOp::IAdd, header, at)?;
            let first = contained(f, table, at)?;
            let before = f.binary(BinOp::ULessThanEqual, first, group)?;
            let l2 = f.select(before, mid, l)?;
            let h2 = f.select(before, h, mid)?;
            f.store(lo, l2, MemAccess::default())?;
            f.store(hi, h2, MemAccess::default())
        })
    })?;
    let found = f.load(lo, MemAccess::default())?;
    let at = f.binary(BinOp::IMul, found, per)?;
    let record = f.binary(BinOp::IAdd, header, at)?;
    let word = |f: &mut FunctionBuilder<'_>, i: u32| -> BResult<_> {
        let i = k(f, i)?;
        let at = f.binary(BinOp::IAdd, record, i)?;
        contained(f, table, at)
    };
    let first_group = word(f, 0)?;
    let capacity = word(f, 1)?;
    let first_word = word(f, 2)?;
    let first_byte = word(f, 3)?;
    let inside = word(f, 4)?;
    let wide = word(f, 5)?;
    let start = word(f, 6)?;
    let wide = f.binary(BinOp::INotEqual, wide, zero)?;
    let local = k(f, IndexGatherModule::LOCAL_SIZE)?;
    let groups_in = f.binary(BinOp::ISub, group, first_group)?;
    let o = f.binary(BinOp::IMul, groups_in, local)?;
    let o = f.binary(BinOp::IAdd, o, t.lin)?;
    let has = f.binary(BinOp::ULessThan, zero, t.draws)?;
    let within = f.binary(BinOp::ULessThan, o, capacity)?;
    let writes = f.binary(BinOp::LogicalAnd, has, within)?;
    let fetch = Fetch {
        indices: t.indices,
        first_byte,
        inside,
        start,
        wide,
        sequential: t.sequential,
    };
    let (pattern, period, output) = (t.pattern, t.period, t.output);
    when(f, writes, move |f| {
        // The position: the pattern's place `o % P` of period `o / P`.
        let unit = f.binary(BinOp::UDiv, o, period)?;
        let place = f.binary(BinOp::UMod, o, period)?;
        let one = k(f, 1)?;
        let m = f.binary(BinOp::IAdd, one, place)?;
        let mult = contained(f, pattern, m)?;
        let a = f.binary(BinOp::IAdd, m, period)?;
        let add = contained(f, pattern, a)?;
        let position = f.binary(BinOp::IMul, mult, unit)?;
        let position = f.binary(BinOp::IAdd, position, add)?;
        let value = fetch.at(f, position)?;
        let at = f.binary(BinOp::IAdd, first_word, o)?;
        contained_store(f, output, at, value)
    })
}

/// The fixed binding numbers of the module's symbols.
fn binding_of(sym: &Symbol) -> Option<u32> {
    match sym {
        Symbol::DescriptorSet(_) => Some(IndexGatherModule::SET),
        Symbol::Binding(s) if *s == TABLE => Some(IndexGatherModule::TABLE_BINDING),
        Symbol::Binding(s) if *s == PATTERN => Some(IndexGatherModule::PATTERN_BINDING),
        Symbol::Binding(s) if *s == INDICES => Some(IndexGatherModule::INDICES_BINDING),
        Symbol::Binding(s) if *s == OUTPUT => Some(IndexGatherModule::OUTPUT_BINDING),
        _ => None,
    }
}

/// The index gather module for `device`: built with the typed builder, verified, serialized, and
/// numbered with the fixed bindings of [`IndexGatherModule`].
pub fn index_gather_module(device: &Device) -> Result<IndexGatherModule, Error> {
    let (spirv, requirements) = finished(
        device,
        gather_typed_module()?,
        binding_of,
        "the index gather",
    )?;
    Ok(IndexGatherModule {
        spirv,
        requirements,
    })
}

#[cfg(test)]
mod tests;
