use super::*;
use crate::indirect_commands::{Action, CommandRecord};

fn writes(operations: u32, primitive_types: u8) -> CommandWrites {
    CommandWrites {
        operations,
        primitive_types,
        kinds: CommandWrites::RENDER_COMMANDS,
    }
}

/// A draw replay takes draws and indexed draws of one primitive type under the inherited state,
/// whose records hold draws, buffers, resets and cleared barriers alone, whether the buffer
/// inherits the encoder's buffers or not (its preprocess reads the parent's then); each other
/// buffer or record breaks the first rule it meets, which names it. Indexed draws among the writes
/// are a replay's indexed draws.
#[test]
fn a_draw_replay_takes_draws_of_one_primitive_type_under_the_inherited_state() {
    let draws = IndirectCommandType::DRAW;
    let triangles = writes(CommandWrites::DRAW | CommandWrites::BUFFER, 1 << 3);
    assert_eq!(
        draw_replay_primitive_type(draws, true, triangles),
        Ok(PrimitiveType::Triangle)
    );
    let all = writes(DRAW_REPLAY_WRITES, 1 << 1);
    assert_eq!(
        draw_replay_primitive_type(draws, true, all),
        Ok(PrimitiveType::Line)
    );
    let indexed = IndirectCommandType::DRAW | IndirectCommandType::DRAW_INDEXED;
    let indexed_triangles = writes(CommandWrites::DRAW_INDEXED | CommandWrites::BUFFER, 1 << 3);
    for types in [indexed, IndirectCommandType::DRAW_INDEXED] {
        assert_eq!(
            draw_replay_primitive_type(types, true, indexed_triangles),
            Ok(PrimitiveType::Triangle)
        );
    }
    assert!(draw_replay_indexed(indexed_triangles) && !draw_replay_indexed(triangles));
    let patches = IndirectCommandType::DRAW | IndirectCommandType::DRAW_PATCHES;
    let cases = [
        (patches, true, triangles, DrawReplayRule::CommandTypes),
        (
            IndirectCommandType::empty(),
            true,
            triangles,
            DrawReplayRule::CommandTypes,
        ),
        (draws, false, triangles, DrawReplayRule::OwnPipelineStates),
        (
            draws,
            true,
            writes(CommandWrites::DRAW | CommandWrites::BARRIER, 1 << 3),
            DrawReplayRule::Operations(CommandWrites::BARRIER),
        ),
        (
            draws,
            true,
            writes(CommandWrites::BUFFER | CommandWrites::RESET, 0),
            DrawReplayRule::NoDraws,
        ),
        (
            draws,
            true,
            writes(CommandWrites::DRAW, 0b1001),
            DrawReplayRule::PrimitiveTypes(0b1001),
        ),
        (
            draws,
            true,
            writes(CommandWrites::DRAW, CommandWrites::ALL_PRIMITIVE_TYPES),
            DrawReplayRule::PrimitiveTypes(CommandWrites::ALL_PRIMITIVE_TYPES),
        ),
    ];
    for (types, inherits, written, rule) in cases {
        assert_eq!(
            draw_replay_primitive_type(types, inherits, written),
            Err(rule),
            "{rule}"
        );
    }
    assert_eq!(
        DrawReplayRule::Operations(CommandWrites::BARRIER | CommandWrites::PIPELINE).to_string(),
        "the buffer's records may hold pipeline states, barriers"
    );
}

/// Points, lines and triangles merge into one draw; strips do not.
#[test]
fn lists_and_points_merge() {
    let merges = [
        PrimitiveType::Point,
        PrimitiveType::Line,
        PrimitiveType::LineStrip,
        PrimitiveType::Triangle,
        PrimitiveType::TriangleStrip,
    ]
    .map(draw_replay_merges);
    assert_eq!(merges, [true, true, false, true, false]);
}

/// A record states its action's operation and a draw's primitive type (a value past Metal's five
/// as all five), and its barrier; a reset record nothing.
#[test]
fn a_record_states_its_action_and_barrier() {
    let draw = |primitive_type| Action::DrawPrimitives {
        primitive_type,
        vertex_start: 0,
        vertex_count: 3,
        instance_count: 1,
        base_instance: 0,
    };
    let mut record = CommandRecord {
        action: draw(3),
        ..CommandRecord::default()
    };
    assert_eq!(record.writes(), writes(CommandWrites::DRAW, 1 << 3));
    record.barrier = true;
    record.action = draw(9);
    assert_eq!(
        record.writes(),
        writes(
            CommandWrites::DRAW | CommandWrites::BARRIER,
            CommandWrites::ALL_PRIMITIVE_TYPES
        )
    );
    assert_eq!(CommandRecord::default().writes(), CommandWrites::NONE);
}

/// A draw replay pulls every fixed-function layout of its state's vertex input, static strides
/// and pulled attributes as they are; a layout whose stride is set per bind (a fixed-function
/// binding's, or a pulled attribute's that steps) keeps it from the replay, and a constant step's,
/// which reads no stride, does not.
#[test]
fn a_draw_replay_pulls_every_fixed_function_layout_of_a_static_stride() {
    use crate::metal::{VertexFormat, VertexStepFunction};
    use crate::pipeline::{PulledAttribute, VertexBinding, VertexInputState, VertexStride};
    let binding = |buffer_index, stride| VertexBinding {
        binding: buffer_index,
        buffer_index,
        input_rate: vk::VertexInputRate::VERTEX,
        divisor: 1,
        stride,
    };
    let pulled = |buffer_index, step_function, stride| PulledAttribute {
        attribute: 7,
        buffer_index,
        format: VertexFormat::Float2,
        offset: 1,
        step_function,
        step_rate: 1,
        stride,
    };
    let mut input = VertexInputState::default();
    assert_eq!(draw_replay_vertex_layouts(&input), Ok(BTreeSet::new()));
    input.bindings = vec![
        binding(0, VertexStride::Static(8)),
        binding(3, VertexStride::Static(0)),
    ];
    input.pulled = vec![
        pulled(1, VertexStepFunction::PerVertex, VertexStride::Static(10)),
        pulled(2, VertexStepFunction::Constant, VertexStride::PerBind),
    ];
    assert_eq!(
        draw_replay_vertex_layouts(&input),
        Ok(BTreeSet::from([0, 3]))
    );
    let mut per_bind = input.clone();
    per_bind.bindings[1].stride = VertexStride::PerBind;
    assert_eq!(
        draw_replay_vertex_layouts(&per_bind),
        Err(DrawReplayRule::VertexAttributes)
    );
    let mut stepped = input;
    stepped.pulled[1].step_function = VertexStepFunction::PerInstance;
    assert_eq!(
        draw_replay_vertex_layouts(&stepped),
        Err(DrawReplayRule::VertexAttributes)
    );
    assert_eq!(
        DrawReplayRule::VertexAttributes.to_string(),
        "the inherited state's vertex function fetches attributes at a stride set per bind"
    );
}
