use super::*;
use crate::indirect_commands::{Action, CommandRecord};

fn writes(operations: u32, primitive_types: u8) -> CommandWrites {
    CommandWrites {
        operations,
        primitive_types,
    }
}

/// A draw replay takes draws and indexed draws of one primitive type under the inherited state
/// with the commands' own buffers, whose records hold draws, buffers, resets and cleared barriers
/// alone; each other buffer or record breaks the first rule it meets, which names it. Indexed
/// draws among the writes are a replay's indexed draws.
#[test]
fn a_draw_replay_takes_draws_of_one_primitive_type_with_their_own_buffers() {
    let draws = IndirectCommandType::DRAW;
    let triangles = writes(CommandWrites::DRAW | CommandWrites::BUFFER, 1 << 3);
    assert_eq!(
        draw_replay_primitive_type(draws, (true, false), triangles),
        Ok(PrimitiveType::Triangle)
    );
    let all = writes(DRAW_REPLAY_WRITES, 1 << 1);
    assert_eq!(
        draw_replay_primitive_type(draws, (true, false), all),
        Ok(PrimitiveType::Line)
    );
    let indexed = IndirectCommandType::DRAW | IndirectCommandType::DRAW_INDEXED;
    let indexed_triangles = writes(CommandWrites::DRAW_INDEXED | CommandWrites::BUFFER, 1 << 3);
    for types in [indexed, IndirectCommandType::DRAW_INDEXED] {
        assert_eq!(
            draw_replay_primitive_type(types, (true, false), indexed_triangles),
            Ok(PrimitiveType::Triangle)
        );
    }
    assert!(draw_replay_indexed(indexed_triangles) && !draw_replay_indexed(triangles));
    let patches = IndirectCommandType::DRAW | IndirectCommandType::DRAW_PATCHES;
    let cases = [
        (
            patches,
            (true, false),
            triangles,
            DrawReplayRule::CommandTypes,
        ),
        (
            IndirectCommandType::empty(),
            (true, false),
            triangles,
            DrawReplayRule::CommandTypes,
        ),
        (
            draws,
            (false, false),
            triangles,
            DrawReplayRule::OwnPipelineStates,
        ),
        (
            draws,
            (true, true),
            triangles,
            DrawReplayRule::InheritedBuffers,
        ),
        (
            draws,
            (true, false),
            writes(CommandWrites::DRAW | CommandWrites::BARRIER, 1 << 3),
            DrawReplayRule::Operations(CommandWrites::BARRIER),
        ),
        (
            draws,
            (true, false),
            writes(CommandWrites::BUFFER | CommandWrites::RESET, 0),
            DrawReplayRule::NoDraws,
        ),
        (
            draws,
            (true, false),
            writes(CommandWrites::DRAW, 0b1001),
            DrawReplayRule::PrimitiveTypes(0b1001),
        ),
        (
            draws,
            (true, false),
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
