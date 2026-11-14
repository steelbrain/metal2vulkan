//! A command buffer's write routes (`write_routes.rs`): what each route may write joins what the
//! records may hold (`written_commands.rs`) of every buffer the command buffer made resident for
//! writing, whichever of its encoders met which, and marks the buffer.
//!
//! Device tests: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects, which the buffers need; nothing is
//! encoded or submitted.

use super::blit_holds::{device, Scratch};
use crate::write_routes::Routes;
use crate::{IndirectCommandBufferDescriptor, ResourceOptions};
use metal2vulkan::metal::IndirectCommandType;
use metal2vulkan::pipeline::CommandWrites as W;

fn draws(primitive_types: u8) -> W {
    W {
        operations: W::DRAW,
        primitive_types,
        kinds: W::RENDER_COMMANDS,
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
    assert!(decided.marked.is_empty());
    assert!(matches!(decided.unmarked, (0, None)));
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

/// A route reaches the buffers of its command kind alone: a kernel that writes compute commands
/// (a dispatch and a reset) neither adds to nor marks a buffer of draws its command buffer made
/// resident for writing, before or after it is set, while a buffer of dispatches beside it takes
/// its writes; a state that writes draws reaches the buffer of draws and not the buffer of
/// dispatches; and the commit marks each with the first state of its kind.
#[test]
fn a_route_reaches_the_buffers_of_its_command_kind_alone() {
    let scratch = Scratch::new("write-route-kinds");
    let device = device(&scratch.0);
    let shared = ResourceOptions::shared();
    let buffer = |command_types| {
        let mut descriptor = IndirectCommandBufferDescriptor::new();
        descriptor.command_types = command_types;
        descriptor.inherit_pipeline_state = true;
        (device.new_indirect_command_buffer(&descriptor, 1, shared)).unwrap()
    };
    let (render, compute) = (
        buffer(IndirectCommandType::DRAW_INDEXED),
        buffer(IndirectCommandType::CONCURRENT_DISPATCH),
    );
    let dispatches = W {
        operations: W::DISPATCH | W::RESET,
        primitive_types: 0,
        kinds: W::COMPUTE_COMMANDS,
    };
    let mut routes = Routes::default();
    routes.writable(&render);
    routes.state_set(dispatches, || vec!["dispatcher".to_owned()]);
    routes.writable(&compute);
    assert!(render.command_writes().is_empty());
    assert!(!render.gpu_written());
    assert_eq!(compute.command_writes(), dispatches);
    assert!(compute.gpu_written());
    routes.state_set(draws(0b1000), || vec!["drawer".to_owned()]);
    assert_eq!(render.command_writes(), draws(0b1000));
    assert!(render.gpu_written());
    assert_eq!(compute.command_writes(), dispatches);
    let (decided, followed) = routes.at_commit();
    followed.mark();
    assert!(decided.marked.is_empty());
    assert_eq!(render.command_writes(), draws(0b1000));
    assert_eq!(compute.command_writes(), dispatches);
    let names = |icb: &crate::IndirectCommandBuffer| icb.gpu_writer().map(|w| w.to_string());
    assert!(
        names(&render).is_some_and(|w| w.contains("drawer")),
        "{:?}",
        names(&render)
    );
    assert!(
        names(&compute).is_some_and(|w| w.contains("dispatcher")),
        "{:?}",
        names(&compute)
    );
}
