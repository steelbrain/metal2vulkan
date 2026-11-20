//! The guards a candidate keeps around a case's out-of-bounds targets: where each guarded target
//! ([`out_of_bounds::guarded`]) lies, how far its guard reaches, what a run changed there, and how
//! those counts join the run's out-of-bounds facts, for the executor's own context (`platform`,
//! each target in an allocation of its own).

use super::ComputeCase;
use crate::case::out_of_bounds::{ContainmentForm, OutOfBoundsResource, OutOfBoundsTarget};
use crate::executor_contract::out_of_bounds::{self, OutOfBoundsRun};
use std::collections::BTreeMap;

/// What a run changed around a guarded target.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GuardCount {
    /// Guard bytes that no longer hold `GUARD_FILL`.
    pub escaped: u64,
    /// Bytes past the target's end that the memory bound to it holds (filled with `PAD_FILL`)
    /// and that no longer hold it.
    pub past_end: u64,
}

/// A guard's reach on each side of a guarded target, in bytes: the target's declared reach
/// (`case::out_of_bounds::OutOfBoundsTarget`), at least [`MIN_GUARD`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GuardReach {
    pub before: u32,
    pub after: u32,
}

/// The least guard on either side of a guarded target, so a stray access just before or past
/// the target is caught even where the case declares no reach on that side.
pub const MIN_GUARD: u64 = 256;

/// The guard reach of `resource` when `targets` declares it an out-of-bounds target.
pub fn declared_reach(
    targets: &[OutOfBoundsTarget],
    resource: OutOfBoundsResource,
) -> Option<GuardReach> {
    targets.iter().find_map(|target| {
        (target.resource == resource).then_some(GuardReach {
            before: target.reach_before,
            after: target.reach_after,
        })
    })
}

/// The out-of-bounds target the buffer at Metal index `index` holds in `case`: the buffer
/// itself, or the stage input whose data the buffer binds.
pub fn buffer_target(case: &ComputeCase, index: u32) -> OutOfBoundsResource {
    let stage_input = case.stage_input.as_ref().and_then(|descriptor| {
        descriptor
            .attributes
            .iter()
            .find(|(_, attribute)| attribute.buffer_index == index)
            .map(|(&location, _)| location)
    });
    match stage_input {
        Some(location) => OutOfBoundsResource::AttributeBuffer { location },
        None => OutOfBoundsResource::Buffer { binding: index },
    }
}

/// The guard reach of the buffer at Metal index `index` when `case` declares it, or the stage
/// input it binds, an out-of-bounds target.
pub fn guard_reach(case: &ComputeCase, index: u32) -> Option<GuardReach> {
    declared_reach(&case.out_of_bounds, buffer_target(case, index))
}

/// The guard reach of the argument-buffer buffer at `(owner, field offset)` when `case` declares
/// it an out-of-bounds target.
pub fn embedded_guard_reach(
    case: &ComputeCase,
    (buffer_binding, field_offset): (u32, u32),
) -> Option<GuardReach> {
    declared_reach(
        &case.out_of_bounds,
        OutOfBoundsResource::ArgumentBufferBuffer {
            buffer_binding,
            field_offset,
        },
    )
}

/// The guard reach of the texture at Metal slot `binding` when `case` declares it an
/// out-of-bounds target of the texel-buffer form: a texture buffer, whose texels are buffer
/// memory. An image's memory is opaque and has no guard.
pub fn texel_guard_reach(case: &ComputeCase, binding: u32) -> Option<GuardReach> {
    case.out_of_bounds.iter().find_map(|target| {
        (target.resource == OutOfBoundsResource::Texture { binding }
            && target.form == ContainmentForm::TexelBuffer)
            .then_some(GuardReach {
                before: target.reach_before,
                after: target.reach_after,
            })
    })
}

/// `run` with the guard and padding bytes the run changed around each target it guarded (a
/// buffer, an argument buffer's included, an attribute buffer, an index buffer, or a texture
/// buffer), as it counted them by target, each target in an allocation of its own.
pub fn with_escaped(
    run: OutOfBoundsRun,
    escaped: &BTreeMap<OutOfBoundsResource, GuardCount>,
) -> OutOfBoundsRun {
    OutOfBoundsRun {
        targets: run
            .targets
            .into_iter()
            .map(|target| {
                let count = escaped.get(&target.resource);
                out_of_bounds::TargetRun {
                    escaped: count.map(|count| count.escaped),
                    past_end: count.map(|count| count.past_end),
                    ..target
                }
            })
            .collect(),
    }
}
