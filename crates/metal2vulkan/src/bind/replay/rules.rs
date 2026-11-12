//! Which executions of render commands the GPU may write run as a draw replay on the device
//! ([`draw_replay_preprocess`](super::draw_replay_preprocess)), and in which form: the rules of the
//! buffer, of what its records may hold, of the inherited state and of its instantiated draw that
//! hold of every consumer of the replay. The runtime's GPU form applies them beside its own (the
//! call's check, the pass's, `render/executions/gpu.rs` there), and so does the validation
//! executor's device replay of an authored case's executions; a broken rule is a
//! [`DrawReplayRule`], which a host-step notice names.

use std::fmt;

use super::super::{BindingPlan, DrawCommands, PushHole, ViewSetup};
use crate::cache::VariantKey;
use crate::metal::{IndirectCommandType, PrimitiveType};
use crate::pipeline::{CommandWrites, GraphicsState, PipelineInterface, SlotClass, Stage};
use crate::vk;

/// What the records of a draw replay may hold beside draws: buffers, resets and cleared barriers
/// (the `CommandWrites` operation bits).
pub const DRAW_REPLAY_WRITES: u32 = CommandWrites::DRAW
    | CommandWrites::BUFFER
    | CommandWrites::RESET
    | CommandWrites::CLEAR_BARRIER;

/// A rule that keeps an execution of render commands from a draw replay: a host step records it.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DrawReplayRule {
    /// The buffer's command types are not draws of primitives and indexed draws alone.
    CommandTypes,
    /// The buffer's commands set their own pipeline states.
    OwnPipelineStates,
    /// The buffer's commands inherit the encoder's buffers.
    InheritedBuffers,
    /// The records may hold operations past [`DRAW_REPLAY_WRITES`]: those bits.
    Operations(u32),
    /// The records hold no draws.
    NoDraws,
    /// The records' draws may pass more than one primitive type (one the writes do not state
    /// counts as all five): the `CommandWrites` primitive-type bits.
    PrimitiveTypes(u8),
    /// The state's vertex function writes memory and it rasterizes, or the draws merge.
    VertexWritesMemory,
    /// The state's fragment function writes memory.
    FragmentWritesMemory,
    /// The state has a fragment imageblock.
    FragmentImageblock,
    /// The state tests stencil in its shaders.
    ShaderStencil,
    /// The state tests depth or stencil in its shaders.
    ShaderTests,
    /// The state reads its attachments.
    InputAttachments,
    /// The state's indirect draw needs its arguments on the CPU.
    ArgumentsOnCpu,
    /// The state's draw bounds its indices.
    IndexBounds,
    /// The state's draw dispatches before it draws (a mesh or tessellation dispatch).
    DispatchesFirst,
    /// The state's draw is amplified or draws views.
    Amplified,
    /// The state's vertex function fetches attributes.
    VertexAttributes,
    /// The state's fragment function reads the primitive ID of an indexed draw.
    IndexedPrimitiveIds,
    /// The state's functions share a slot table.
    SharedSlotTable,
    /// The state's variant does not read the merged draw's header, or reads it unmerged.
    MergedHeader,
}

impl fmt::Display for DrawReplayRule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let rule = match *self {
            DrawReplayRule::Operations(bits) => {
                f.write_str("the buffer's records may hold ")?;
                let names = OPERATIONS
                    .iter()
                    .filter(|(bit, _)| bits & bit != 0)
                    .map(|(_, name)| *name);
                for (at, name) in names.enumerate() {
                    if at > 0 {
                        f.write_str(", ")?;
                    }
                    f.write_str(name)?;
                }
                return Ok(());
            }
            DrawReplayRule::PrimitiveTypes(bits) => {
                return write!(
                    f,
                    "the buffer's draws may pass more than one primitive type (MTLPrimitiveType \
                     bits {bits:#07b})"
                )
            }
            DrawReplayRule::CommandTypes => {
                "the buffer's command types are not draws of primitives and indexed draws alone"
            }
            DrawReplayRule::OwnPipelineStates => {
                "the buffer's commands set their own pipeline states"
            }
            DrawReplayRule::InheritedBuffers => {
                "the buffer's commands inherit the encoder's buffers"
            }
            DrawReplayRule::NoDraws => "the buffer's records hold no draws",
            DrawReplayRule::VertexWritesMemory => {
                "the inherited state's vertex function writes memory"
            }
            DrawReplayRule::FragmentWritesMemory => {
                "the inherited state's fragment function writes memory"
            }
            DrawReplayRule::FragmentImageblock => "the inherited state has a fragment imageblock",
            DrawReplayRule::ShaderStencil => "the inherited state tests stencil in its shaders",
            DrawReplayRule::ShaderTests => {
                "the inherited state tests depth or stencil in its shaders"
            }
            DrawReplayRule::InputAttachments => "the inherited state reads its attachments",
            DrawReplayRule::ArgumentsOnCpu => {
                "the inherited state's indirect draw needs its arguments on the CPU"
            }
            DrawReplayRule::IndexBounds => "the inherited state's draw bounds its indices",
            DrawReplayRule::DispatchesFirst => {
                "the inherited state's draw dispatches before it draws"
            }
            DrawReplayRule::Amplified => "the inherited state's draw is amplified or draws views",
            DrawReplayRule::VertexAttributes => {
                "the inherited state's vertex function fetches attributes"
            }
            DrawReplayRule::IndexedPrimitiveIds => {
                "the inherited state's fragment function reads the primitive ID of an indexed draw"
            }
            DrawReplayRule::SharedSlotTable => "the inherited state's functions share a slot table",
            DrawReplayRule::MergedHeader => {
                "the inherited state's variant does not read the merged draw's header"
            }
        };
        f.write_str(rule)
    }
}

/// The operations past a draw replay's, by their `CommandWrites` bit.
const OPERATIONS: [(u32, &str); 11] = [
    (CommandWrites::DRAW_INDEXED, "indexed draws"),
    (CommandWrites::DRAW_PATCHES, "draws of patches"),
    (CommandWrites::DRAW_MESH, "mesh draws"),
    (CommandWrites::DISPATCH, "dispatches"),
    (CommandWrites::PIPELINE, "pipeline states"),
    (CommandWrites::BARRIER, "barriers"),
    (CommandWrites::RENDER_STATE, "render state setters"),
    (
        CommandWrites::THREADGROUP_MEMORY,
        "threadgroup memory lengths",
    ),
    (CommandWrites::STAGE_IN_REGION, "stage-in regions"),
    (CommandWrites::IMAGEBLOCK_SIZE, "imageblock sizes"),
    (CommandWrites::COPY, "copied commands"),
];

/// The primitive type every command draws of an execution of a buffer of `command_types` that
/// inherits the encoder's pipeline state where `inherit_pipeline_state` and its buffers where
/// `inherit_buffers`, whose records may hold `written` (what every route that writes them may
/// write, its CPU's setters among them): a draw replay runs commands of draws of one primitive
/// type alone, under the encoder's state, with their own buffers, whose records hold draws,
/// buffers, resets and cleared barriers alone; else the first rule broken.
pub fn draw_replay_primitive_type(
    command_types: IndirectCommandType,
    (inherit_pipeline_state, inherit_buffers): (bool, bool),
    written: CommandWrites,
) -> Result<PrimitiveType, DrawReplayRule> {
    let draws = IndirectCommandType::DRAW | IndirectCommandType::DRAW_INDEXED;
    if command_types.is_empty() || !draws.contains(command_types) {
        return Err(DrawReplayRule::CommandTypes);
    }
    if !inherit_pipeline_state {
        return Err(DrawReplayRule::OwnPipelineStates);
    }
    if inherit_buffers {
        return Err(DrawReplayRule::InheritedBuffers);
    }
    let past = written.operations & !(DRAW_REPLAY_WRITES | CommandWrites::DRAW_INDEXED);
    if past != 0 {
        return Err(DrawReplayRule::Operations(past));
    }
    if !written.may(CommandWrites::DRAW) && !written.may(CommandWrites::DRAW_INDEXED) {
        return Err(DrawReplayRule::NoDraws);
    }
    let one = match written.primitive_types.count_ones() {
        1 => PrimitiveType::try_from(written.primitive_types.trailing_zeros()).ok(),
        _ => None,
    };
    one.ok_or(DrawReplayRule::PrimitiveTypes(written.primitive_types))
}

/// Whether records that may hold `written` may hold indexed draws (`DrawReplay::indexed`), which
/// a merged draw replay alone runs.
pub fn draw_replay_indexed(written: CommandWrites) -> bool {
    written.may(CommandWrites::DRAW_INDEXED)
}

/// Whether draws of `primitive_type` merge into one draw (`DrawReplay::merged`): a list type or
/// points, whose primitives a merged draw keeps whole.
pub fn draw_replay_merges(primitive_type: PrimitiveType) -> bool {
    matches!(
        primitive_type,
        PrimitiveType::Point | PrimitiveType::Line | PrimitiveType::Triangle
    )
}

/// The inherited state's rules (its `graphics` facts) for a draw replay, `merged` into one draw or
/// not: its functions write no memory (but a vertex function where nothing rasterizes, unless the
/// draws merge: a merged draw runs a vertex for each corner of a whole primitive, which only a
/// vertex function that writes no memory cannot tell from Metal's), and it has no fragment
/// imageblock, shader-side stencil or depth test or attachment read.
pub fn draw_replay_state(graphics: &GraphicsState, merged: bool) -> Result<(), DrawReplayRule> {
    let rules = [
        (
            graphics.vertex_writes_memory && (merged || !graphics.rasterizer_discard),
            DrawReplayRule::VertexWritesMemory,
        ),
        (
            graphics.fragment_writes_memory,
            DrawReplayRule::FragmentWritesMemory,
        ),
        (
            graphics.fragment_imageblock.is_some(),
            DrawReplayRule::FragmentImageblock,
        ),
        (
            graphics.shadow_stencil.is_some(),
            DrawReplayRule::ShaderStencil,
        ),
        (graphics.shader_tests.is_some(), DrawReplayRule::ShaderTests),
        (
            !graphics.input_attachments.is_empty(),
            DrawReplayRule::InputAttachments,
        ),
    ];
    match rules.into_iter().find(|(breaks, _)| *breaks) {
        Some((_, rule)) => Err(rule),
        None => Ok(()),
    }
}

/// The instantiated draw's rules for a draw replay, `merged` into one draw or not: `commands`,
/// instantiated for one indirect draw by `plan`, the plan of the replay variant `key`, record as
/// one indirect draw whose arguments the device reads, with no index bound, dispatch, amplification,
/// view or fetched attribute, each slot table one function's (a merged draw's in its header alone,
/// the vertex function's).
pub fn draw_replay_draw(
    commands: &DrawCommands,
    (plan, key): (&BindingPlan, &VariantKey),
    merged: bool,
) -> Result<(), DrawReplayRule> {
    let (vertex, fragment) = (vk::ShaderStageFlags::VERTEX, vk::ShaderStageFlags::FRAGMENT);
    let one_function = commands.push_holes.iter().all(|hole| match *hole {
        PushHole::RootTableAddress { stages, .. } => {
            !merged && (stages == vertex || stages == fragment)
        }
        PushHole::MergedDrawAddress { stages, .. } => {
            merged && !stages.is_empty() && (vertex | fragment).contains(stages)
        }
        _ => true,
    });
    let header =
        (commands.push_holes.iter()).any(|hole| matches!(hole, PushHole::MergedDrawAddress { .. }));
    let rules = [
        (
            commands.indirect_needs_arguments,
            DrawReplayRule::ArgumentsOnCpu,
        ),
        (commands.index_bounds_pass, DrawReplayRule::IndexBounds),
        (
            commands.mesh_dispatch.is_some() || commands.tessellation_dispatch.is_some(),
            DrawReplayRule::DispatchesFirst,
        ),
        (
            !commands.amplifications.is_empty() || plan.views != ViewSetup::default(),
            DrawReplayRule::Amplified,
        ),
        (
            !plan.vertex_buffers.is_empty(),
            DrawReplayRule::VertexAttributes,
        ),
        (
            key.primitive_id_indices.is_some(),
            DrawReplayRule::IndexedPrimitiveIds,
        ),
        (!one_function, DrawReplayRule::SharedSlotTable),
        (merged != header, DrawReplayRule::MergedHeader),
    ];
    match rules.into_iter().find(|(breaks, _)| *breaks) {
        Some((_, rule)) => Err(rule),
        None => Ok(()),
    }
}

/// The buffer slots of the vertex function and of the fragment function a replay variant of
/// `interface` reads, each as one past its highest buffer index (`DrawReplay::vertex_slots`,
/// `DrawReplay::fragment_slots`).
pub fn draw_replay_slots(interface: &PipelineInterface) -> (u32, u32) {
    let slots = |stage: Stage| {
        (interface.slots.keys())
            .filter(|slot| slot.stage == stage && slot.class == SlotClass::Buffer)
            .map(|slot| slot.index + 1)
            .max()
            .unwrap_or(0)
    };
    (slots(Stage::Vertex), slots(Stage::Fragment))
}

#[cfg(test)]
mod tests;
