//! The device replay's generated-commands path: where the consumer enabled
//! `VK_EXT_device_generated_commands` with compute pipeline binding
//! ([`Device::generated_commands`]), the same preprocess writes a token stream in place of the
//! per-variant arguments, and the consumer records one `vkCmdExecuteGeneratedCommandsEXT` per
//! possible segment in place of a dispatch per command, region, and variant.
//!
//! **The sequences.** One per (command, region), in command order then region order
//! ([`GeneratedLayout::sequence_offset`]), each [`GeneratedLayout::stride`] bytes of three tokens:
//!
//! 1. at [`GeneratedLayout::INDEX_OFFSET`], an `EXECUTION_SET` token's `u32`: the index of the
//!    region's variant in the indirect execution set, which holds the registered variants in
//!    their order (0 for a region that runs nothing);
//! 2. at [`GeneratedLayout::PUSH_OFFSET`], a `PUSH_CONSTANT` token's
//!    [`GeneratedLayout::push_bytes`]: the variant's push bytes (its template, [`GeneratedReplay`])
//!    with its `RootTableAddress` the command's slot table, its `ReplayRegionAddress` the region's
//!    record, and its `ExecutionStatusAddress` the status, as K14c's path pushes them;
//! 3. at [`GeneratedLayout::dispatch_offset`], a `DISPATCH` token's `VkDispatchIndirectCommand`:
//!    the region's group count, as K14c's arguments give the selected variant, or zero.
//!
//! **The segments.** A command whose record sets the barrier waits for every earlier command; the
//! others of a segment may run concurrently, as Metal's do. Commands `s` with `s` a multiple of
//! [`GeneratedLayout::segment_commands`] start a segment too, so one execution runs at most that
//! many commands (an added barrier is one more valid schedule) and the preprocess memory the
//! executions reserve stays linear in the commands. Per command `s` the preprocess writes the
//! sequence count of the execution that starts there ([`GeneratedLayout::count_offset`]): its
//! segment's commands times the regions where `s` starts one, 0 elsewhere. It reads at most
//! `segment_commands - 1` later records' barrier words to find the segment's end, so its work per
//! command stays bounded.
//!
//! **Recording** (consumer): per command `s` in range order, a barrier from the previous
//! execution (as K14c's between commands), then `vkCmdExecuteGeneratedCommandsEXT` with
//! `indirectAddress` the scratch plus `sequence_offset(s, 0)`, `sequenceCountAddress` the scratch
//! plus `count_offset(s)`, `maxSequenceCount` [`GeneratedLayout::max_sequences`]`(s)`, and
//! preprocess memory of its own (`vkGetGeneratedCommandsMemoryRequirementsEXT` for that count).
//! The indirect commands layout is the three tokens at those offsets with that stride, its
//! pipeline layout [`GeneratedReplay::pipeline_layout`]: every variant's pipeline is created with
//! it (`VK_PIPELINE_CREATE_2_INDIRECT_BINDABLE_BIT_EXT`), the first as the execution set's initial
//! pipeline (bound before each execution) and the others written at their indices. The
//! variants' descriptor sets are bound once: they are one set ([`GeneratedRefusal::Descriptors`]
//! otherwise). A consumer may preprocess the executions apart instead: the indirect commands
//! layout made with `VK_INDIRECT_COMMANDS_LAYOUT_USAGE_EXPLICIT_PREPROCESS_BIT_EXT`, after the
//! preprocess's barrier every execution's `vkCmdPreprocessGeneratedCommandsEXT` with the state it
//! runs with, a barrier, then each execution as above, preprocessed.
//!
//! **Its output is K14c's.** Each command's regions select the same variant with the same push
//! values and group count as K14c's arguments; only the schedule between barriers differs, and a
//! racy command set compares only under a class that admits any order.

use crate::bind::dispatch::{DispatchCommands, IndirectArgs};
use crate::bind::instantiate::PushHole;
use crate::bind::plan::{BindingPlan, WriteSource};
use crate::bind::replay::ReplayVariant;
use crate::device::{Device, GeneratedCommandsProperties};
use crate::error::BindError;
use crate::pipeline::layout::{PipelineLayout, PushConstantRange};
use crate::pipeline::ComputePipeline;
use crate::vk;

use super::{refused, ReplayRefusal};

/// The most commands one execution runs: a segment of more barrier-free commands is split at each
/// multiple of this many commands from the range's start.
pub const MAX_SEGMENT_COMMANDS: u32 = 8;
/// Bytes of a variant template's head: the offsets in its push bytes of its `RootTableAddress`,
/// `ReplayRegionAddress`, and `ExecutionStatusAddress` holes ([`NO_HOLE`] for one it does not
/// have), then a pad word. Its push bytes follow.
pub const TEMPLATE_HEAD_BYTES: u64 = 16;
/// A template head's word for a hole the variant does not have.
pub const NO_HOLE: u32 = u32::MAX;
/// Bytes of a `VkDispatchIndirectCommand` token.
const DISPATCH_BYTES: u32 = 12;
/// The tokens of a sequence: execution set, push constants, dispatch.
const TOKENS: u32 = 3;

/// One registered variant as the generated-commands path binds it: its pipeline (its layout and
/// constant blob), its plan (its descriptor writes), and its
/// [`instantiate_replay`](crate::bind::BindingPlan::instantiate_replay) commands (its push bytes
/// and holes, and its descriptor contents).
#[non_exhaustive]
#[derive(Clone, Copy, Debug)]
pub struct GeneratedVariant<'a> {
    pub pipeline: &'a ComputePipeline,
    pub plan: &'a BindingPlan,
    pub commands: &'a DispatchCommands,
}

impl<'a> GeneratedVariant<'a> {
    /// The variant of `pipeline`, planned as `plan`, instantiated as `commands`.
    pub fn new(
        pipeline: &'a ComputePipeline,
        plan: &'a BindingPlan,
        commands: &'a DispatchCommands,
    ) -> GeneratedVariant<'a> {
        GeneratedVariant {
            pipeline,
            plan,
            commands,
        }
    }
}

/// Why the generated-commands path sets up no replay; K14c's path remains.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GeneratedRefusal {
    /// The device's description states no `VK_EXT_device_generated_commands` enabled
    /// ([`Device::generated_commands`]).
    NotEnabled,
    /// The device binds no compute pipeline through an execution set: compute is not among its
    /// `supportedIndirectCommandsShaderStagesPipelineBinding` (or its
    /// `supportedIndirectCommandsShaderStages`).
    NoComputePipelineBinding,
    /// Not one generated variant per registered variant.
    Variants { registered: usize, generated: usize },
    /// More variants than the device's `maxIndirectPipelineCount`.
    PipelineCount { variants: u32, limit: u32 },
    /// The three tokens exceed the device's `maxIndirectCommandsTokenCount`.
    TokenCount { limit: u32 },
    /// The dispatch token's offset exceeds the device's `maxIndirectCommandsTokenOffset`.
    TokenOffset { offset: u32, limit: u32 },
    /// The sequence stride exceeds the device's `maxIndirectCommandsIndirectStride`.
    Stride { stride: u32, limit: u32 },
    /// One command's sequences (its regions) exceed the device's `maxIndirectSequenceCount`.
    SequenceCount { sequences: u32, limit: u32 },
    /// Variant `index` binds other descriptors than the first: its pipeline layout's sets, its
    /// plan's writes, its descriptor contents, or its constant blob differ, and one execution
    /// binds one descriptor state for every sequence.
    Descriptors { index: usize },
    /// Generated variant `index` was not instantiated for the registered variant `index`'s
    /// threadgroup size: its region's specialization is not the one
    /// [`instantiate_replay`](crate::bind::BindingPlan::instantiate_replay) gives that size. (The
    /// template and the threadgroup lengths stay the consumer's to pair, as on K14c's path.)
    Unmatched { index: usize },
}

/// Where the generated-commands path's sections lie in the replay's scratch, in bytes from its
/// start, and the token stream's shape.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GeneratedLayout {
    /// Commands replayed.
    pub slots: u32,
    /// Regions per command: the sequences of each command.
    pub regions: u32,
    /// Bytes of one sequence.
    pub stride: u32,
    /// Bytes of the `PUSH_CONSTANT` token: its update range is `[0, push_bytes)` of `COMPUTE`.
    pub push_bytes: u32,
    /// The most commands of one execution: 8 (`MAX_SEGMENT_COMMANDS`), or fewer where the device's
    /// `maxIndirectSequenceCount` holds fewer commands' regions.
    pub segment_commands: u32,
    /// The variants' templates ([`GeneratedReplay::templates`]).
    pub templates_offset: u64,
    /// One `u32` sequence count per command, which the preprocess writes.
    pub counts_offset: u64,
    /// The sequences, which the preprocess writes.
    pub sequences_offset: u64,
}

impl GeneratedLayout {
    /// The `EXECUTION_SET` token's offset in a sequence.
    pub const INDEX_OFFSET: u32 = 0;
    /// The `PUSH_CONSTANT` token's offset in a sequence.
    pub const PUSH_OFFSET: u32 = 4;

    /// The `DISPATCH` token's offset in a sequence.
    pub fn dispatch_offset(&self) -> u32 {
        Self::PUSH_OFFSET + self.push_bytes
    }

    /// The sequence of region `region` of the `slot`th command.
    pub fn sequence_offset(&self, slot: u32, region: u32) -> u64 {
        self.sequences_offset
            + (u64::from(slot) * u64::from(self.regions) + u64::from(region))
                * u64::from(self.stride)
    }

    /// The sequence count of the execution recorded at the `slot`th command.
    pub fn count_offset(&self, slot: u32) -> u64 {
        self.counts_offset + 4 * u64::from(slot)
    }

    /// The most sequences the execution recorded at the `slot`th command runs: the regions of the
    /// commands from `slot` to the next multiple of `segment_commands` or the range's end.
    pub fn max_sequences(&self, slot: u32) -> u32 {
        let l = self.segment_commands.max(1);
        let window = l - slot % l;
        self.regions * window.min(self.slots.saturating_sub(slot))
    }
}

/// What the consumer uploads and creates for the generated-commands path.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GeneratedReplay {
    /// Each variant's template, in variant order, at `layout.generated.templates_offset`:
    /// 16 bytes (`TEMPLATE_HEAD_BYTES`): the offsets of its `RootTableAddress`,
    /// `ReplayRegionAddress`, and `ExecutionStatusAddress` holes in its push bytes (all-ones for
    /// one it does not have) and a pad word; then its push bytes (`push_bytes`, zero where it
    /// writes none).
    pub templates: Vec<u8>,
    /// The pipeline layout of every variant's pipeline and of the indirect commands layout: the
    /// variants' one descriptor set layout, and one `COMPUTE` push-constant range `[0,
    /// push_bytes)`.
    pub pipeline_layout: PipelineLayout,
}

/// What [`plan`] decides.
#[derive(Clone, Debug)]
pub(super) struct Planned {
    pub(super) push_bytes: u32,
    pub(super) stride: u32,
    pub(super) segment_commands: u32,
    pub(super) replay: GeneratedReplay,
}

fn refuse(reason: GeneratedRefusal) -> BindError {
    refused(ReplayRefusal::GeneratedCommands { reason })
}

/// Whether `a` and `b` bind the same descriptors: the same set layouts, writes, descriptor
/// contents and ranges, and (where a write reads it) constant blob.
fn binds_alike(a: &GeneratedVariant<'_>, b: &GeneratedVariant<'_>) -> bool {
    let blob = |v: &GeneratedVariant<'_>| {
        v.plan
            .descriptors
            .iter()
            .any(|w| matches!(w.source, WriteSource::ConstantBlob { .. }))
            .then(|| v.pipeline.stage.constant_blob.clone())
    };
    a.pipeline.layout.sets == b.pipeline.layout.sets
        && a.plan.descriptors == b.plan.descriptors
        && a.commands.descriptor_bytes == b.commands.descriptor_bytes
        && a.commands.descriptor_ranges == b.commands.descriptor_ranges
        && blob(a) == blob(b)
}

/// The template of `v` over `push_bytes`: its hole offsets and its push bytes. A region that is
/// not one replayed region, a write or hole outside `COMPUTE` or the push bytes, or another hole
/// is not a replay variant's.
fn template(v: &GeneratedVariant<'_>, push_bytes: u32) -> Result<Vec<u8>, BindError> {
    let not_replay = || refused(ReplayRefusal::NotReplayVariant);
    let [region] = v.commands.regions.as_slice() else {
        return Err(not_replay());
    };
    if region.indirect != Some(IndirectArgs::Replay) {
        return Err(not_replay());
    }
    let head = TEMPLATE_HEAD_BYTES as usize;
    let mut out = vec![0u8; head + push_bytes as usize];
    for w in &region.push_constants {
        let start = w.offset as usize;
        let end = start + w.bytes.len();
        if w.stages != vk::ShaderStageFlags::COMPUTE || end > push_bytes as usize {
            return Err(not_replay());
        }
        out[head + start..head + end].copy_from_slice(&w.bytes);
    }
    let mut holes = [NO_HOLE; 3];
    for hole in &v.commands.push_holes {
        let (k, stages, offset) = match *hole {
            PushHole::RootTableAddress { stages, offset } => (0, stages, offset),
            PushHole::ReplayRegionAddress { stages, offset } => (1, stages, offset),
            PushHole::ExecutionStatusAddress { stages, offset } => (2, stages, offset),
            _ => return Err(not_replay()),
        };
        let inside = u64::from(offset) + 8 <= u64::from(push_bytes);
        if stages != vk::ShaderStageFlags::COMPUTE || !inside || !offset.is_multiple_of(4) {
            return Err(not_replay());
        }
        holes[k] = offset;
    }
    for (k, h) in holes.iter().enumerate() {
        out[4 * k..4 * k + 4].copy_from_slice(&h.to_le_bytes());
    }
    Ok(out)
}

/// The generated-commands path of `variants` for commands of `regions` regions on `device`, held
/// to the limits its description states: its token stream's shape, the variants' templates, and
/// the pipeline layout they share. Host work is per variant, never per command.
pub(super) fn plan(
    device: &Device,
    regions: u32,
    registered: &[ReplayVariant],
    variants: &[GeneratedVariant<'_>],
) -> Result<Planned, BindError> {
    let facts: &GeneratedCommandsProperties = device
        .generated_commands()
        .ok_or_else(|| refuse(GeneratedRefusal::NotEnabled))?;
    let compute = vk::ShaderStageFlags::COMPUTE;
    if !facts.pipeline_binding_stages.contains(compute) || !facts.supported_stages.contains(compute)
    {
        return Err(refuse(GeneratedRefusal::NoComputePipelineBinding));
    }
    if variants.len() != registered.len() || variants.is_empty() {
        return Err(refuse(GeneratedRefusal::Variants {
            registered: registered.len(),
            generated: variants.len(),
        }));
    }
    // Each generated variant is the registered one's threadgroup size: the execution set's index
    // `i` must run what K14c's arguments of variant `i` run.
    for (index, (v, r)) in variants.iter().zip(registered).enumerate() {
        let expected = v.plan.replay_specialization(r.threads_per_threadgroup)?;
        if !matches!(v.commands.regions.as_slice(), [region] if region.specialization == expected) {
            return Err(refuse(GeneratedRefusal::Unmatched { index }));
        }
    }
    let count = variants.len() as u32;
    if count > facts.max_indirect_pipeline_count {
        return Err(refuse(GeneratedRefusal::PipelineCount {
            variants: count,
            limit: facts.max_indirect_pipeline_count,
        }));
    }
    if TOKENS > facts.max_indirect_commands_token_count {
        return Err(refuse(GeneratedRefusal::TokenCount {
            limit: facts.max_indirect_commands_token_count,
        }));
    }
    let first = &variants[0];
    if let Some(index) = variants.iter().position(|v| !binds_alike(first, v)) {
        return Err(refuse(GeneratedRefusal::Descriptors { index }));
    }
    // The push bytes cover every variant's compute range and writes.
    let mut push_bytes = 0u32;
    for v in variants {
        for range in &v.pipeline.layout.push_constants {
            if range.stages.contains(compute) {
                push_bytes = push_bytes.max(range.offset + range.size);
            }
        }
        for region in &v.commands.regions {
            for w in &region.push_constants {
                push_bytes = push_bytes.max(w.offset + w.bytes.len() as u32);
            }
        }
    }
    // At least one word: a push-constant token's range is never empty, even for variants that
    // push nothing.
    let push_bytes = push_bytes.max(4).next_multiple_of(4);
    let dispatch = GeneratedLayout::PUSH_OFFSET + push_bytes;
    if dispatch > facts.max_indirect_commands_token_offset {
        return Err(refuse(GeneratedRefusal::TokenOffset {
            offset: dispatch,
            limit: facts.max_indirect_commands_token_offset,
        }));
    }
    let stride = dispatch + DISPATCH_BYTES;
    if stride > facts.max_indirect_commands_indirect_stride {
        return Err(refuse(GeneratedRefusal::Stride {
            stride,
            limit: facts.max_indirect_commands_indirect_stride,
        }));
    }
    if regions > facts.max_indirect_sequence_count {
        return Err(refuse(GeneratedRefusal::SequenceCount {
            sequences: regions,
            limit: facts.max_indirect_sequence_count,
        }));
    }
    let segment_commands = MAX_SEGMENT_COMMANDS.min(facts.max_indirect_sequence_count / regions);
    let mut templates = Vec::new();
    for v in variants {
        templates.extend(template(v, push_bytes)?);
    }
    let pipeline_layout = PipelineLayout {
        sets: first.pipeline.layout.sets.clone(),
        push_constants: vec![PushConstantRange {
            stages: compute,
            offset: 0,
            size: push_bytes,
        }],
    };
    Ok(Planned {
        push_bytes,
        stride,
        segment_commands,
        replay: GeneratedReplay {
            templates,
            pipeline_layout,
        },
    })
}
