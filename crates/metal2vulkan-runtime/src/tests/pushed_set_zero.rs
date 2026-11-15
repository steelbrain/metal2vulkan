//! What a compute encoder's dispatches hold at the compute bind point of the `VkCommandBuffer`
//! they record into (`binding::PushedSetZero`): the writes a push carries, when a pipeline is
//! bound again, and what forgets them. Device runs: `runtime_lane::dispatch_work`.

use crate::binding::{Info, PushedSetZero, SetWrite};
use ash::vk::{self, Handle};

/// A storage buffer write at binding `binding` of the buffer whose handle is `buffer`.
fn storage(binding: u32, buffer: u64) -> SetWrite {
    SetWrite {
        binding,
        element: 0,
        ty: vk::DescriptorType::STORAGE_BUFFER,
        info: Info::Buffer(vk::DescriptorBufferInfo {
            buffer: vk::Buffer::from_raw(buffer),
            offset: 0,
            range: 64,
        }),
    }
}

/// The `VkCommandBuffer`, pipeline layout and pipelines the dispatches record with.
const RAW: u64 = 1;
const LAYOUT: u64 = 2;
const PIPELINE: u64 = 3;
const OTHER_PIPELINE: u64 = 4;

/// A dispatch of `writes` recorded into `raw` at the encoder's count of recordings `mark`, with
/// `layout` and `pipeline`, as `Recording::record` records one region: whether it bound the
/// pipeline, and the bindings of the writes its push carried.
fn dispatch(
    held: &mut PushedSetZero,
    (raw, mark): (u64, u64),
    (layout, pipeline): (u64, u64),
    writes: &[SetWrite],
    stale: bool,
) -> (bool, Vec<u32>) {
    let raw = vk::CommandBuffer::from_raw(raw);
    held.hold(raw, mark);
    let layout = vk::PipelineLayout::from_raw(layout);
    held.records_with(layout);
    let bound = held.binds(vk::Pipeline::from_raw(pipeline));
    let carried = held.carry(layout, writes, stale);
    let carried = carried.iter().map(|write| write.binding).collect();
    held.recorded(raw, mark);
    (bound, carried)
}

/// Dispatches one after another into one `VkCommandBuffer`, nothing else recorded between: the
/// first binds its pipeline and pushes set 0 whole, one of the same writes binds and pushes
/// nothing, and one of other writes pushes those alone.
#[test]
fn a_dispatch_pushes_only_the_writes_that_changed() {
    let held = &mut PushedSetZero::default();
    let (a, b) = (
        [storage(0, 10), storage(1, 11)],
        [storage(0, 20), storage(1, 21)],
    );
    let at = (LAYOUT, PIPELINE);
    assert_eq!(dispatch(held, (RAW, 1), at, &a, false), (true, vec![0, 1]));
    assert_eq!(dispatch(held, (RAW, 2), at, &a, false), (false, vec![]));
    assert_eq!(dispatch(held, (RAW, 3), at, &b, false), (false, vec![0, 1]));
    let half = [storage(0, 20), storage(1, 31)];
    assert_eq!(dispatch(held, (RAW, 4), at, &half, false), (false, vec![1]));
}

/// What forgets what was pushed and bound: another `VkCommandBuffer`, anything the encoder
/// recorded between (its count of recordings two past the last dispatch's), set 0 bound from a
/// set, a dispatch of another layout, other bindings, and a recording that forgets it (a
/// command of the runtime's own, a preprocess).
#[test]
fn anything_else_recorded_between_pushes_set_zero_whole() {
    let a = [storage(0, 10), storage(1, 11)];
    let at = (LAYOUT, PIPELINE);
    let whole = (true, vec![0, 1]);
    let held = &mut PushedSetZero::default();
    assert_eq!(dispatch(held, (RAW, 1), at, &a, false), whole);
    assert_eq!(
        dispatch(held, (RAW + 10, 2), at, &a, false),
        whole,
        "another"
    );
    assert_eq!(
        dispatch(held, (RAW + 10, 4), at, &a, false),
        whole,
        "a recording between"
    );
    held.forget();
    assert_eq!(
        dispatch(held, (RAW + 10, 5), at, &a, false),
        whole,
        "forgotten"
    );
    held.set_bound();
    let carried = dispatch(held, (RAW + 10, 6), at, &a, false);
    assert_eq!(carried, (false, vec![0, 1]), "set 0 bound from a set");
    let other = (LAYOUT + 10, PIPELINE);
    let carried = dispatch(held, (RAW + 10, 7), other, &a, false);
    assert_eq!(carried, (false, vec![0, 1]), "another layout");
    let carried = dispatch(held, (RAW + 10, 8), at, &a, false);
    assert_eq!(carried, (false, vec![0, 1]), "the first layout again");
    let fewer = [storage(0, 10)];
    let carried = dispatch(held, (RAW + 10, 9), at, &fewer, false);
    assert_eq!(carried, (false, vec![0]), "other bindings");
    assert_eq!(
        dispatch(held, (RAW + 10, 10), at, &fewer, false),
        (false, vec![])
    );
}

/// A dispatch that binds another pipeline of the layout pushes what changed; where the driver's
/// pushed descriptors go stale across a pipeline bind, at least one write, set 0's first where
/// none changed, and nothing again while the pipeline stays bound.
#[test]
fn a_pipeline_bind_pushes_one_write_where_pushes_go_stale_across_it() {
    let a = [storage(0, 10), storage(1, 11)];
    for stale in [false, true] {
        let held = &mut PushedSetZero::default();
        let (first, other) = ((LAYOUT, PIPELINE), (LAYOUT, OTHER_PIPELINE));
        assert_eq!(
            dispatch(held, (RAW, 1), first, &a, stale),
            (true, vec![0, 1])
        );
        let renewed = match stale {
            true => vec![0],
            false => vec![],
        };
        let carried = dispatch(held, (RAW, 2), other, &a, stale);
        assert_eq!(carried, (true, renewed), "stale {stale}");
        let carried = dispatch(held, (RAW, 3), other, &a, stale);
        assert_eq!(carried, (false, vec![]), "stale {stale}");
        let changed = [storage(0, 10), storage(1, 12)];
        let carried = dispatch(held, (RAW, 4), first, &changed, stale);
        assert_eq!(carried, (true, vec![1]), "stale {stale}");
    }
}
