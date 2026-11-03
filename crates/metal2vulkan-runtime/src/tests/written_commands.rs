//! What an indirect command buffer's records may hold (`written_commands.rs`): each route adds,
//! a blit copy's destination gains what its source gains later, and nothing is taken away.

use std::sync::Arc;

use metal2vulkan::bind::indirect_commands::{Action, CommandRecord};
use metal2vulkan::pipeline::CommandWrites as W;

use crate::write_routes::Unmarked;
use crate::written_commands::{Named, Writers, WrittenCommands};

fn draws(primitive_types: u8) -> W {
    W {
        operations: W::DRAW,
        primitive_types,
    }
}

/// A record the CPU left adds its draw's operation and primitive type, its barrier, and the
/// state it names; a type past Metal's five adds every type; an empty record adds nothing.
#[test]
fn a_record_adds_what_it_holds() {
    let written = Arc::new(WrittenCommands::default());
    written
        .add_record(CommandRecord::default().writes(), None)
        .mark();
    assert!(written.writes().is_empty());
    let mut record = CommandRecord {
        action: Action::DrawPrimitives {
            primitive_type: 3,
            vertex_start: 0,
            vertex_count: 3,
            instance_count: 1,
            base_instance: 0,
        },
        ..CommandRecord::default()
    };
    written.add_record(record.writes(), None).mark();
    assert_eq!(written.writes(), draws(0b1000));
    record.barrier = true;
    let named = Named {
        encodes: true,
        writes: W::NONE,
    };
    written.add_record(record.writes(), Some(named)).mark();
    assert_eq!(
        written.writes().operations,
        W::DRAW | W::BARRIER | W::PIPELINE
    );
    assert!(written.named().encodes);
    record.action = Action::DrawIndexedPrimitives {
        primitive_type: 9,
        index_count: 3,
        index_type: 0,
        index_buffer: 0,
        instance_count: 1,
        base_vertex: 0,
        base_instance: 0,
    };
    written.add_record(record.writes(), None).mark();
    assert_eq!(written.writes().primitive_types, W::ALL_PRIMITIVE_TYPES);
}

/// A writer that may set a pipeline or copy a command may name any state; one that only draws
/// names none. One that may copy a command may write anything: the copied command holds what
/// its source held (a dispatch, a barrier), which the copier's own operations do not state.
#[test]
fn a_writer_that_sets_pipelines_names_any_state() {
    let drawing = Arc::new(WrittenCommands::default());
    drawing.add_writer(draws(0b1000)).mark();
    assert_eq!(drawing.named(), Named::default());
    for bit in [W::PIPELINE, W::COPY] {
        let setting = Arc::new(WrittenCommands::default());
        let writes = W {
            operations: bit,
            primitive_types: 0,
        };
        setting.add_writer(writes).mark();
        assert!(setting.named().encodes);
        assert_eq!(setting.named().writes, W::ANY);
        let held = match bit {
            W::COPY => W::ANY,
            _ => writes,
        };
        assert_eq!(setting.writes(), held, "{bit:#x}");
    }
}

/// A copy's destination takes its source's writes at the copy and what the source gains after,
/// through a chain of copies and around a cycle; the source gains nothing from its destination.
#[test]
fn a_copy_gains_what_its_source_gains() {
    let [a, b, c] = [(); 3].map(|_| Arc::new(WrittenCommands::default()));
    a.add_writer(draws(0b1)).mark();
    b.copied_from(&a).mark();
    assert_eq!(b.writes(), draws(0b1));
    c.copied_from(&b).mark();
    a.add_writer(draws(0b1000)).mark();
    assert_eq!(c.writes(), draws(0b1001));
    b.add_writer(W {
        operations: W::DRAW_INDEXED,
        primitive_types: 0,
    })
    .mark();
    assert_eq!(a.writes(), draws(0b1001));
    assert_eq!(c.writes().operations, W::DRAW | W::DRAW_INDEXED);
    // A cycle ends.
    a.copied_from(&c).mark();
    c.add_writer(draws(0b10)).mark();
    for buffer in [&a, &b, &c] {
        assert_eq!(buffer.writes().primitive_types, 0b1011);
    }
}

/// An encoder's writers: a buffer made resident takes the state already set and each set after,
/// and an execution's named states, however many times it is made resident; a buffer it did not
/// make resident takes none.
#[test]
fn an_encoder_adds_its_states() {
    let buffer = Arc::new(WrittenCommands::default());
    let other = Arc::new(WrittenCommands::default());
    let mut writers = Writers::default();
    writers.state_set(draws(0b1));
    assert!(buffer.writes().is_empty());
    writers.made_writable(&buffer);
    writers.made_writable(&buffer);
    assert_eq!(buffer.writes(), draws(0b1));
    writers.state_set(draws(0b100));
    writers.runs(W {
        operations: W::DISPATCH,
        primitive_types: 0,
    });
    assert_eq!(buffer.writes().operations, W::DRAW | W::DISPATCH);
    assert_eq!(buffer.writes().primitive_types, 0b101);
    assert!(other.writes().is_empty());
}

/// A commit's unmarked buffers (`write_routes.rs`) are held by an executed buffer's summary until
/// it first names a state whose functions may write records, which sets them off once, through a
/// blit copy's destination too; a summary that names one already holds none.
#[test]
fn a_state_that_may_write_records_sets_off_the_followers() {
    let [source, copy] = [(); 2].map(|_| Arc::new(WrittenCommands::default()));
    copy.copied_from(&source).mark();
    let unmarked = Unmarked::of(Vec::new());
    assert!(source.follow(&unmarked));
    assert!(copy.follow(&unmarked));
    let drawn = CommandRecord::default().writes();
    // A state that may encode, but whose functions call no operation, writes no record.
    let taking = Named {
        encodes: true,
        writes: W::NONE,
    };
    assert!(source.add_record(drawn, Some(taking)).0.is_empty());
    let resetting = Named {
        encodes: true,
        writes: W {
            operations: W::RESET,
            primitive_types: 0,
        },
    };
    let followed = source.add_record(drawn, Some(resetting));
    assert_eq!(followed.0.len(), 2);
    followed.mark();
    assert!(source.add_record(drawn, Some(resetting)).0.is_empty());
    assert!(!source.follow(&unmarked));
    assert!(!copy.follow(&unmarked));
}
