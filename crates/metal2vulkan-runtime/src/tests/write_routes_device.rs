//! A command buffer's write routes (`write_routes.rs`): what each route may write joins what the
//! records may hold (`written_commands.rs`) of every buffer the command buffer made resident for
//! writing, whichever of its encoders met which, and marks the buffer.
//!
//! Device tests: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects, which the buffers need; nothing is
//! encoded or submitted.

use super::blit_holds::{device, Scratch};
use crate::write_routes::{AtCommit, Routes};
use crate::{IndirectCommandBufferDescriptor, ResourceOptions};
use metal2vulkan::metal::IndirectCommandType;
use metal2vulkan::pipeline::CommandWrites as W;

fn draws(primitive_types: u8) -> W {
    W {
        operations: W::DRAW,
        primitive_types,
    }
}

/// A buffer made resident for writing takes what every state set in its command buffer may
/// write: one set in a later encoder, one set before it was made resident, as each is set, and
/// one set in an encoder open beside its own (a parallel render encoder's) at the commit; a buffer
/// the command buffer did not make resident takes nothing, and the commit lets the routes go.
#[test]
fn a_route_reaches_every_buffer_its_command_buffer_made_writable() {
    let scratch = Scratch::new("write-routes");
    let device = device(&scratch.0);
    let mut descriptor = IndirectCommandBufferDescriptor::new();
    descriptor.command_types = IndirectCommandType::DRAW;
    descriptor.inherit_pipeline_state = true;
    let shared = ResourceOptions::shared();
    let icb = || (device.new_indirect_command_buffer(&descriptor, 1, shared)).unwrap();
    let (before, after, beside, next) = (icb(), icb(), icb(), icb());
    let functions = || vec!["writer".to_owned()];
    let mut command_buffer = Routes::default();
    let mut first = command_buffer.carried();
    first.writable(&before);
    command_buffer.join(first);
    assert!(before.command_writes().is_empty());
    assert!(!before.gpu_written());
    let mut second = command_buffer.carried();
    second.state_set(draws(0b1), functions);
    assert_eq!(before.command_writes(), draws(0b1));
    assert!(before.gpu_written());
    second.writable(&after);
    assert_eq!(after.command_writes(), draws(0b1));
    command_buffer.join(second);
    let (mut one, mut other) = (command_buffer.carried(), command_buffer.carried());
    one.writable(&beside);
    other.state_set(draws(0b10), functions);
    assert_eq!(beside.command_writes(), draws(0b1));
    assert_eq!(before.command_writes(), draws(0b11));
    command_buffer.join(one);
    command_buffer.join(other);
    assert_eq!(beside.command_writes(), draws(0b1));
    let (decided, followed) = command_buffer.at_commit();
    assert!(matches!(decided, AtCommit::Unmarked(0, None)));
    followed.mark();
    for icb in [&before, &after, &beside] {
        assert_eq!(icb.command_writes(), draws(0b11));
        assert!(icb.gpu_written());
    }
    assert!(next.command_writes().is_empty());
    command_buffer.writable(&next);
    assert!(next.command_writes().is_empty());
    assert!(!next.gpu_written());
}
