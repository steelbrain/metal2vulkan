//! The hazard tracker (`tracker.rs`): every read and write order, tracked and untracked
//! resources, byte ranges and image subresources, fences, retirement, the order across
//! `VkQueue`s, and the legacy masks.

use crate::binding::ResidentPrefix;
use crate::tracker::{
    join_looked_at, merge, ordered, Access, Barrier, EncoderUses, Footprint, GeometryWrites,
    IndirectReads, LegacyBarrier, Point, Reach, Summary, Tracker, UnorderedWrites, Use, UseRange,
    Wait, ENTRIES_PER_KEY,
};
use ash::vk;
use std::ops::Range;

type Key = u64;

const COMPUTE: vk::PipelineStageFlags2 = vk::PipelineStageFlags2::COMPUTE_SHADER;
const TRANSFER: vk::PipelineStageFlags2 = vk::PipelineStageFlags2::TRANSFER;
const FRAGMENT: vk::PipelineStageFlags2 = vk::PipelineStageFlags2::FRAGMENT_SHADER;
const NO_ACCESS: vk::AccessFlags2 = vk::AccessFlags2::NONE;

fn shader_read() -> Access {
    Access::new(COMPUTE, vk::AccessFlags2::SHADER_STORAGE_READ)
}

fn shader_write() -> Access {
    Access::new(COMPUTE, vk::AccessFlags2::SHADER_STORAGE_WRITE)
}

fn copy_read() -> Access {
    Access::new(TRANSFER, vk::AccessFlags2::TRANSFER_READ)
}

fn copy_write() -> Access {
    Access::new(TRANSFER, vk::AccessFlags2::TRANSFER_WRITE)
}

fn bytes(range: Range<u64>) -> UseRange {
    UseRange::Bytes(range)
}

fn image(aspects: vk::ImageAspectFlags, levels: Range<u32>, layers: Range<u32>) -> UseRange {
    UseRange::Image {
        aspects,
        levels,
        layers,
    }
}

fn reads(key: Key, range: UseRange, access: Access) -> Use<Key> {
    Use::reads(key, true, range, access)
}

fn writes(key: Key, range: UseRange, access: Access) -> Use<Key> {
    Use::writes(key, true, range, access)
}

fn barrier(src: Access, dst: Access) -> Option<Barrier> {
    Some(Barrier { src, dst })
}

/// An ordered encoder with one command per entry of `commands`.
fn encoder(commands: &[&[Use<Key>]]) -> Summary<Key> {
    let mut uses = EncoderUses::new(true);
    for command in commands {
        uses.order(vk::PipelineStageFlags2::NONE, command, Reach::Unnamed);
    }
    uses.finish()
}

/// An encoder of one command.
fn one(command: &[Use<Key>]) -> Summary<Key> {
    encoder(&[command])
}

/// The batch of `VkQueue` 0 at timeline value `value`.
fn at(value: u64) -> Point {
    Point { queue: 0, value }
}

/// The batch of `VkQueue` `queue` at timeline value `value`.
fn on(queue: usize, value: u64) -> Point {
    Point { queue, value }
}

/// Submit a batch of `encoders` at `point`: the barriers before each, and the batch's waits.
fn submit(
    queue: &mut Tracker<Key>,
    encoders: &[Summary<Key>],
    point: Point,
) -> (Vec<Option<Barrier>>, Vec<Wait>) {
    let plan = queue.plan(encoders, point);
    let barriers = plan.barriers().to_vec();
    let waits = plan.waits().to_vec();
    queue.apply(plan);
    (barriers, waits)
}

/// Submit a batch of `encoders` on `VkQueue` 0 at `value`: the barriers before each. A batch on
/// one queue never waits.
fn commit(queue: &mut Tracker<Key>, encoders: &[Summary<Key>], value: u64) -> Vec<Option<Barrier>> {
    let (barriers, waits) = submit(queue, encoders, at(value));
    assert_eq!(waits, []);
    barriers
}

/// A wait for `point` before `stages`.
fn wait(point: Point, stages: vk::PipelineStageFlags2) -> Wait {
    Wait { at: point, stages }
}

/// Commit each encoder in its own command buffer: the barrier before each.
fn commit_each(queue: &mut Tracker<Key>, encoders: &[Summary<Key>]) -> Vec<Option<Barrier>> {
    (0..)
        .zip(encoders)
        .map(|(serial, summary)| commit(queue, std::slice::from_ref(summary), serial + 1)[0])
        .collect()
}

#[test]
fn a_read_after_a_write_waits_for_the_write() {
    let mut queue = Tracker::default();
    let got = commit_each(
        &mut queue,
        &[
            one(&[writes(1, bytes(0..256), shader_write())]),
            one(&[reads(1, bytes(0..256), shader_read())]),
        ],
    );
    assert_eq!(got, [None, barrier(shader_write(), shader_read())]);
}

#[test]
fn reads_after_reads_need_nothing() {
    let mut queue = Tracker::default();
    let got = commit_each(
        &mut queue,
        &[
            one(&[reads(1, bytes(0..256), shader_read())]),
            one(&[reads(1, bytes(0..256), copy_read())]),
            one(&[reads(1, bytes(0..256), shader_read())]),
        ],
    );
    assert_eq!(got, [None, None, None]);
}

#[test]
fn a_write_after_reads_waits_for_their_execution() {
    let mut queue = Tracker::default();
    let got = commit_each(
        &mut queue,
        &[
            one(&[reads(1, bytes(0..256), copy_read())]),
            one(&[reads(
                1,
                bytes(0..256),
                Access::new(FRAGMENT, vk::AccessFlags2::SHADER_SAMPLED_READ),
            )]),
            one(&[writes(1, bytes(0..256), shader_write())]),
        ],
    );
    // An execution dependency only: no access types on either side.
    assert_eq!(
        got,
        [
            None,
            None,
            barrier(
                Access::new(TRANSFER | FRAGMENT, NO_ACCESS),
                Access::new(COMPUTE, NO_ACCESS)
            )
        ]
    );
}

#[test]
fn a_write_after_a_write_is_a_memory_dependency() {
    let mut queue = Tracker::default();
    let got = commit_each(
        &mut queue,
        &[
            one(&[writes(1, bytes(0..256), copy_write())]),
            one(&[writes(1, bytes(0..256), shader_write())]),
        ],
    );
    assert_eq!(got, [None, barrier(copy_write(), shader_write())]);
}

#[test]
fn a_write_after_a_read_after_a_write_orders_after_both() {
    let mut queue = Tracker::default();
    let got = commit_each(
        &mut queue,
        &[
            one(&[writes(1, bytes(0..256), copy_write())]),
            one(&[reads(1, bytes(0..256), shader_read())]),
            one(&[writes(1, bytes(0..256), shader_write())]),
        ],
    );
    assert_eq!(got[1], barrier(copy_write(), shader_read()));
    // The read's execution, and the copy's write: it was made visible to shader reads, not to
    // shader writes.
    assert_eq!(
        got[2],
        barrier(
            Access::new(TRANSFER | COMPUTE, vk::AccessFlags2::TRANSFER_WRITE),
            shader_write()
        )
    );
}

#[test]
fn a_write_visible_to_a_read_needs_no_second_barrier() {
    let mut queue = Tracker::default();
    let got = commit_each(
        &mut queue,
        &[
            one(&[writes(1, bytes(0..256), shader_write())]),
            one(&[reads(1, bytes(0..256), shader_read())]),
            one(&[reads(1, bytes(0..256), shader_read())]),
            one(&[reads(1, bytes(0..64), shader_read())]),
            // Another stage and access type: not yet visible there.
            one(&[reads(1, bytes(0..256), copy_read())]),
            one(&[reads(1, bytes(0..256), copy_read())]),
        ],
    );
    assert_eq!(
        got,
        [
            None,
            barrier(shader_write(), shader_read()),
            None,
            None,
            barrier(shader_write(), copy_read()),
            None
        ]
    );
}

#[test]
fn byte_ranges_order_only_where_they_meet() {
    let mut queue = Tracker::default();
    let got = commit_each(
        &mut queue,
        &[
            one(&[writes(1, bytes(0..64), shader_write())]),
            one(&[reads(1, bytes(64..128), shader_read())]),
            one(&[writes(1, bytes(128..256), copy_write())]),
            one(&[reads(1, bytes(32..96), shader_read())]),
            one(&[writes(1, bytes(96..160), shader_write())]),
        ],
    );
    assert_eq!(
        got,
        [
            None,
            None,
            None,
            barrier(shader_write(), shader_read()),
            // After the reads of 64..128 and 32..96, and the copy's write of 128..256.
            barrier(
                Access::new(COMPUTE | TRANSFER, vk::AccessFlags2::TRANSFER_WRITE),
                shader_write()
            ),
        ]
    );
}

#[test]
fn resources_are_keyed_by_identity() {
    let mut queue = Tracker::default();
    let got = commit_each(
        &mut queue,
        &[
            one(&[writes(1, bytes(0..256), shader_write())]),
            one(&[reads(2, bytes(0..256), shader_read())]),
            one(&[writes(3, bytes(0..256), shader_write())]),
            one(&[reads(1, bytes(0..256), shader_read())]),
        ],
    );
    assert_eq!(
        got,
        [None, None, None, barrier(shader_write(), shader_read())]
    );
}

#[test]
fn a_covering_write_replaces_the_older_writes() {
    let mut queue = Tracker::default();
    let got = commit_each(
        &mut queue,
        &[
            one(&[writes(1, bytes(0..16), copy_write())]),
            one(&[writes(1, bytes(16..32), copy_write())]),
            one(&[writes(1, bytes(0..64), shader_write())]),
            one(&[reads(
                1,
                bytes(0..8),
                Access::new(FRAGMENT, vk::AccessFlags2::SHADER_STORAGE_READ),
            )]),
        ],
    );
    assert_eq!(got[2], barrier(copy_write(), shader_write()));
    // Only the covering write is left to order after.
    assert_eq!(
        got[3],
        barrier(
            shader_write(),
            Access::new(FRAGMENT, vk::AccessFlags2::SHADER_STORAGE_READ)
        )
    );
}

#[test]
fn image_subresources_order_only_where_they_meet() {
    let color = vk::ImageAspectFlags::COLOR;
    let depth = vk::ImageAspectFlags::DEPTH;
    let stencil = vk::ImageAspectFlags::STENCIL;
    let sampled = Access::new(COMPUTE, vk::AccessFlags2::SHADER_SAMPLED_READ);
    let mut queue = Tracker::default();
    let got = commit_each(
        &mut queue,
        &[
            one(&[writes(7, image(color, 0..1, 2..3), copy_write())]),
            // Another level, then another layer of the same level.
            one(&[reads(7, image(color, 1..4, 0..6), sampled)]),
            one(&[reads(7, image(color, 0..1, 3..6), sampled)]),
            // The written subresource among others.
            one(&[reads(7, image(color, 0..4, 0..6), sampled)]),
        ],
    );
    assert_eq!(got, [None, None, None, barrier(copy_write(), sampled)]);

    let mut queue = Tracker::default();
    let got = commit_each(
        &mut queue,
        &[
            one(&[writes(8, image(depth, 0..1, 0..1), copy_write())]),
            one(&[reads(8, image(stencil, 0..1, 0..1), copy_read())]),
            one(&[reads(8, image(depth | stencil, 0..1, 0..1), copy_read())]),
        ],
    );
    assert_eq!(got, [None, None, barrier(copy_write(), copy_read())]);
}

#[test]
fn untracked_resources_are_left_to_the_program_between_encoders() {
    let mut queue = Tracker::default();
    let untracked = |access| Use::writes(5, false, bytes(0..256), access);
    let read_untracked = Use::reads(5, false, bytes(0..256), shader_read());
    let got = commit_each(
        &mut queue,
        &[
            one(&[untracked(shader_write())]),
            one(std::slice::from_ref(&read_untracked)),
            one(&[untracked(copy_write())]),
        ],
    );
    assert_eq!(got, [None, None, None]);
    // The summary holds no untracked use, but its stages: a fence wait orders all the work.
    let summary = one(&[untracked(shader_write())]);
    assert!(summary.uses().is_empty());
    assert!(summary.stages() == COMPUTE);
}

#[test]
fn one_barrier_holds_every_hazard_of_an_encoder() {
    let mut queue = Tracker::default();
    let got = commit_each(
        &mut queue,
        &[
            encoder(&[
                &[writes(1, bytes(0..256), shader_write())],
                &[reads(2, bytes(0..256), copy_read())],
            ]),
            encoder(&[
                &[reads(1, bytes(0..256), shader_read())],
                &[writes(2, bytes(0..256), shader_write())],
            ]),
        ],
    );
    assert_eq!(
        got[1],
        barrier(
            Access::new(COMPUTE | TRANSFER, vk::AccessFlags2::SHADER_STORAGE_WRITE),
            Access::new(COMPUTE, vk::AccessFlags2::SHADER_STORAGE_READ)
        )
    );
}

#[test]
fn encoders_of_one_command_buffer_order_among_themselves() {
    let mut queue = Tracker::default();
    let got = commit(
        &mut queue,
        &[
            one(&[writes(1, bytes(0..256), copy_write())]),
            one(&[reads(1, bytes(0..256), shader_read())]),
            one(&[reads(1, bytes(0..256), shader_read())]),
            one(&[writes(1, bytes(0..256), copy_write())]),
        ],
        1,
    );
    assert_eq!(
        got,
        [
            None,
            barrier(copy_write(), shader_read()),
            None,
            barrier(
                Access::new(TRANSFER | COMPUTE, vk::AccessFlags2::TRANSFER_WRITE),
                copy_write()
            ),
        ]
    );
}

#[test]
fn the_order_is_the_commit_order_and_an_unapplied_plan_changes_nothing() {
    let mut queue = Tracker::default();
    let write = [one(&[writes(1, bytes(0..256), shader_write())])];
    let read = [one(&[reads(1, bytes(0..256), shader_read())])];
    // A plan that is never submitted leaves the state as it was.
    let dropped = queue.plan(&write, at(1));
    assert_eq!(dropped.barriers(), [None]);
    drop(dropped);
    assert_eq!(commit(&mut queue, &read, 2), [None]);
    // The command buffer committed later orders after the earlier one, whichever was made first.
    assert_eq!(
        commit(&mut queue, &write, 3),
        [barrier(
            Access::new(COMPUTE, NO_ACCESS),
            Access::new(COMPUTE, NO_ACCESS)
        )]
    );
    assert_eq!(
        commit(&mut queue, &read, 4),
        [barrier(shader_write(), shader_read())]
    );
}

#[test]
fn a_fence_wait_orders_after_the_last_update_in_commit_order() {
    let mut queue = Tracker::default();
    let fence: Key = 100;
    // Untracked work: only the fence orders it.
    let mut first = EncoderUses::new(true);
    first.record(
        TRANSFER,
        &[Use::writes(5, false, bytes(0..64), copy_write())],
        None,
    );
    first.update_fence(fence, None);
    let mut second = EncoderUses::new(false);
    second.wait_for_fence(fence, None);
    second.record(
        COMPUTE,
        &[Use::reads(5, false, bytes(0..64), shader_read())],
        None,
    );
    let got = commit_each(&mut queue, &[first.finish(), second.finish()]);
    assert_eq!(
        got,
        [
            None,
            barrier(
                Access::new(TRANSFER, vk::AccessFlags2::MEMORY_WRITE),
                Access::new(
                    COMPUTE,
                    vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE
                )
            )
        ]
    );
}

#[test]
fn a_fence_never_updated_orders_nothing() {
    let mut queue = Tracker::default();
    let mut waiting = EncoderUses::<Key>::new(false);
    waiting.wait_for_fence(100, None);
    waiting.record(COMPUTE, &[], None);
    assert_eq!(commit(&mut queue, &[waiting.finish()], 1), [None]);
}

#[test]
fn a_fence_wait_sees_only_the_latest_update() {
    let fence: Key = 100;
    let updating = |stages| {
        let mut uses = EncoderUses::<Key>::new(false);
        uses.record(stages, &[], None);
        uses.update_fence(fence, None);
        uses.finish()
    };
    let mut waiting = EncoderUses::<Key>::new(false);
    waiting.wait_for_fence(fence, None);
    waiting.record(FRAGMENT, &[], None);
    let mut queue = Tracker::default();
    let got = commit(
        &mut queue,
        &[updating(TRANSFER), updating(COMPUTE), waiting.finish()],
        1,
    );
    assert_eq!(
        got,
        [
            None,
            None,
            barrier(
                Access::new(COMPUTE, vk::AccessFlags2::MEMORY_WRITE),
                Access::new(
                    FRAGMENT,
                    vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE
                )
            )
        ]
    );
}

#[test]
fn a_fence_passes_through_an_encoder_that_waits_and_updates() {
    let (first, second): (Key, Key) = (100, 101);
    let mut a = EncoderUses::<Key>::new(true);
    a.record(TRANSFER, &[], None);
    a.update_fence(first, None);
    // No work of its own: it only waits for one fence and updates another.
    let mut b = EncoderUses::<Key>::new(false);
    b.wait_for_fence(first, None);
    b.update_fence(second, None);
    let mut c = EncoderUses::<Key>::new(false);
    c.wait_for_fence(second, None);
    c.record(COMPUTE, &[], None);
    let mut queue = Tracker::default();
    let got = commit_each(&mut queue, &[a.finish(), b.finish(), c.finish()]);
    assert_eq!(got[1], None);
    assert_eq!(
        got[2].map(|barrier| barrier.src),
        Some(Access::new(TRANSFER, vk::AccessFlags2::MEMORY_WRITE))
    );
}

/// A wait after the same encoder updated the fence waits for the update before the encoder, as
/// plain Metal's does: the second encoder orders after the first's update, and the third, waiting
/// for the fence, after the second's.
#[test]
fn a_wait_after_an_update_in_one_encoder_waits_for_the_update_before_it() {
    let fence: Key = 100;
    let mut first = EncoderUses::<Key>::new(false);
    first.record(TRANSFER, &[], None);
    first.update_fence(fence, None);
    let mut second = EncoderUses::<Key>::new(false);
    second.update_fence(fence, None);
    second.wait_for_fence(fence, None);
    second.record(COMPUTE, &[], None);
    let mut third = EncoderUses::<Key>::new(false);
    third.wait_for_fence(fence, None);
    third.record(FRAGMENT, &[], None);
    let mut queue = Tracker::default();
    let got = commit_each(
        &mut queue,
        &[first.finish(), second.finish(), third.finish()],
    );
    assert_eq!(
        got[1].map(|barrier| barrier.src),
        Some(Access::new(TRANSFER, vk::AccessFlags2::MEMORY_WRITE))
    );
    assert_eq!(
        got[2].map(|barrier| barrier.src),
        Some(Access::new(
            TRANSFER | COMPUTE,
            vk::AccessFlags2::MEMORY_WRITE
        ))
    );
}

/// A render encoder's wait before the fragment stage alone (`fence.rs`) orders the draws'
/// fragment stages and its work outside its draws (a copy before its pass) after the update, and
/// leaves its draws' vertex stages out of the barrier, as an M3 Max's vertex function ran before
/// the fence's update; a wait before every stage orders them all.
#[test]
fn a_fence_wait_before_the_fragment_stage_leaves_the_vertex_stages_out() {
    use crate::render::{FRAGMENT_DRAW_STAGES, GEOMETRY_DRAW_STAGES};
    let fence: Key = 100;
    let pass = GEOMETRY_DRAW_STAGES | FRAGMENT_DRAW_STAGES;
    let waits = |draws: Option<vk::PipelineStageFlags2>| {
        let mut update = EncoderUses::<Key>::new(false);
        update.record(COMPUTE, &[], None);
        update.update_fence(fence, None);
        let mut render = EncoderUses::<Key>::new(false);
        render.wait_for_fence(fence, draws);
        render.record(TRANSFER, &[], None);
        render.record(pass, &[], None);
        let mut queue = Tracker::default();
        commit_each(&mut queue, &[update.finish(), render.finish()])[1]
    };
    let written = Access::new(COMPUTE, vk::AccessFlags2::MEMORY_WRITE);
    let read_write = vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE;
    assert_eq!(
        waits(Some(FRAGMENT_DRAW_STAGES)),
        barrier(
            written,
            Access::new(TRANSFER | FRAGMENT_DRAW_STAGES, read_write)
        )
    );
    assert_eq!(
        waits(None),
        barrier(written, Access::new(TRANSFER | pass, read_write))
    );
}

/// A render encoder's update after the vertex stage alone (`fence.rs`) covers its draws' vertex
/// stages and its work outside its draws, not their fragment stages: a later wait's barrier does
/// not wait for the fragments, as an M3 Max's later kernel ran before the pass's fragment function
/// ended. Its source covers the stages of a fence the encoder waited for too.
#[test]
fn a_fence_update_after_the_vertex_stage_leaves_the_fragment_stages_out() {
    use crate::render::{FRAGMENT_DRAW_STAGES, GEOMETRY_DRAW_STAGES};
    let (earlier, fence): (Key, Key) = (99, 100);
    let vertex = vk::PipelineStageFlags2::VERTEX_SHADER;
    let pass = GEOMETRY_DRAW_STAGES | FRAGMENT_DRAW_STAGES;
    let updates = |draws: Option<vk::PipelineStageFlags2>| {
        let mut first = EncoderUses::<Key>::new(false);
        first.record(TRANSFER, &[], None);
        first.update_fence(earlier, None);
        let mut render = EncoderUses::<Key>::new(false);
        render.wait_for_fence(earlier, None);
        render.record(pass, &[], None);
        render.record(COMPUTE, &[], None);
        render.update_fence(fence, draws);
        let mut waiting = EncoderUses::<Key>::new(false);
        waiting.wait_for_fence(fence, None);
        waiting.record(COMPUTE, &[], None);
        let mut queue = Tracker::default();
        let got = commit_each(
            &mut queue,
            &[first.finish(), render.finish(), waiting.finish()],
        );
        got[2].map(|barrier| barrier.src)
    };
    let written = |stages| Some(Access::new(stages, vk::AccessFlags2::MEMORY_WRITE));
    assert_eq!(updates(Some(vertex)), written(vertex | COMPUTE | TRANSFER));
    assert_eq!(updates(None), written(pass | COMPUTE | TRANSFER));
}

#[test]
fn ordered_commands_wait_for_earlier_commands_of_the_encoder() {
    let mut uses = EncoderUses::<Key>::new(true);
    let copy_in = [writes(1, bytes(0..256), copy_write())];
    let copy_out = [
        reads(1, bytes(0..256), copy_read()),
        writes(2, bytes(0..256), copy_write()),
    ];
    let again = [reads(1, bytes(0..64), copy_read())];
    let overwrite = [writes(1, bytes(0..256), copy_write())];

    assert_eq!(uses.barrier(&copy_in, Reach::Unnamed), None);
    uses.record(TRANSFER, &copy_in, None);
    let before_out = uses.barrier(&copy_out, Reach::Unnamed);
    assert_eq!(before_out, barrier(copy_write(), copy_read()));
    uses.record(TRANSFER, &copy_out, before_out);
    // Visible already.
    assert_eq!(uses.barrier(&again, Reach::Unnamed), None);
    uses.record(TRANSFER, &again, None);
    // After the reads, and the write already visible to transfer reads only.
    assert_eq!(
        uses.barrier(&overwrite, Reach::Unnamed),
        barrier(
            Access::new(TRANSFER, vk::AccessFlags2::TRANSFER_WRITE),
            copy_write()
        )
    );
}

#[test]
fn unordered_encoders_order_nothing_inside() {
    let mut uses = EncoderUses::<Key>::new(false);
    let write = [writes(1, bytes(0..256), shader_write())];
    uses.record(COMPUTE, &write, None);
    assert_eq!(
        uses.barrier(&[reads(1, bytes(0..256), shader_read())], Reach::Unnamed),
        None
    );
    assert_eq!(uses.barrier(&write, Reach::Unnamed), None);
}

#[test]
fn untracked_memory_is_one_class_inside_an_encoder() {
    let mut uses = EncoderUses::<Key>::new(true);
    // Two untracked resources may alias (heap resources): they order.
    uses.record(
        TRANSFER,
        &[Use::writes(5, false, bytes(0..16), copy_write())],
        None,
    );
    assert_eq!(
        uses.barrier(
            &[Use::reads(6, false, bytes(1024..2048), copy_read())],
            Reach::Unnamed
        ),
        barrier(copy_write(), copy_read())
    );
    // Tracked resources order per identity.
    let mut uses = EncoderUses::<Key>::new(true);
    uses.record(TRANSFER, &[writes(5, bytes(0..16), copy_write())], None);
    assert_eq!(
        uses.barrier(&[reads(6, bytes(0..16), copy_read())], Reach::Unnamed),
        None
    );
    // A tracked and an untracked resource are different memory.
    assert_eq!(
        uses.barrier(
            &[Use::reads(7, false, bytes(0..16), copy_read())],
            Reach::Unnamed
        ),
        None
    );
}

#[test]
fn named_untracked_memory_orders_per_key_and_range_inside_an_encoder() {
    let named = |uses: &mut EncoderUses<Key>, command: &[Use<Key>]| {
        uses.order(TRANSFER, command, Reach::Named)
    };
    let write = |key, range| [Use::writes(key, false, bytes(range), copy_write())];
    let read = |key, range| [Use::reads(key, false, bytes(range), copy_read())];
    // Two blits into disjoint bytes of one heap, and into another's: no order.
    let mut uses = EncoderUses::<Key>::new(true);
    assert_eq!(named(&mut uses, &write(5, 0..16)), None);
    assert_eq!(named(&mut uses, &write(5, 16..32)), None);
    assert_eq!(named(&mut uses, &write(6, 0..16)), None);
    // Overlapping bytes of the same key order.
    assert_eq!(
        named(&mut uses, &read(5, 8..24)),
        barrier(copy_write(), copy_read())
    );
    // A command whose uses do not name what it touches orders after every untracked access,
    // and every later command after it.
    let mut uses = EncoderUses::<Key>::new(true);
    assert_eq!(named(&mut uses, &write(5, 0..16)), None);
    let shader = [Use::reads(9, false, bytes(0..4), shader_read())];
    assert_eq!(
        uses.barrier(&shader, Reach::Unnamed),
        barrier(copy_write(), shader_read())
    );
    uses.record(
        COMPUTE,
        &[Use::writes(9, false, bytes(0..4), shader_write())],
        None,
    );
    assert_eq!(
        uses.barrier(&write(6, 64..128), Reach::Named),
        barrier(shader_write(), copy_write())
    );
    // A tracked use of a key and a named untracked use of it are the same bytes (an untracked
    // heap's use carries the key of a tracked buffer over its host pages).
    let mut uses = EncoderUses::<Key>::new(true);
    uses.record(
        TRANSFER,
        &[Use::writes(5, true, bytes(0..16), copy_write())],
        None,
    );
    assert_eq!(
        named(&mut uses, &read(5, 8..24)),
        barrier(copy_write(), copy_read())
    );
    assert_eq!(named(&mut uses, &read(5, 16..32)), None);
}

#[test]
fn a_command_does_not_order_against_itself() {
    let mut uses = EncoderUses::<Key>::new(true);
    // A copy between two ranges of one buffer.
    let copy = [
        reads(1, bytes(0..64), copy_read()),
        writes(1, bytes(64..128), copy_write()),
    ];
    assert_eq!(uses.barrier(&copy, Reach::Unnamed), None);
    uses.record(TRANSFER, &copy, None);
    // The same copy again: after its write (write after write), and after its read (write after
    // read on 0..64 is not touched; read after write on 64..128 is not read).
    assert_eq!(
        uses.barrier(&copy, Reach::Unnamed),
        barrier(
            Access::new(TRANSFER, vk::AccessFlags2::TRANSFER_WRITE),
            copy_write()
        )
    );
}

#[test]
fn uses_that_touch_nothing_are_ignored() {
    let mut uses = EncoderUses::<Key>::new(true);
    uses.record(TRANSFER, &[writes(1, bytes(0..256), copy_write())], None);
    assert_eq!(
        uses.barrier(&[reads(1, bytes(16..16), copy_read())], Reach::Unnamed),
        None
    );
    assert_eq!(
        uses.barrier(&[reads(1, bytes(0..16), Access::NONE)], Reach::Unnamed),
        None
    );
    let summary = one(&[writes(1, bytes(8..8), copy_write())]);
    assert!(summary.uses().is_empty());
}

#[test]
fn completed_submissions_retire() {
    let mut queue = Tracker::default();
    let fence: Key = 100;
    let mut update = EncoderUses::<Key>::new(true);
    update.record(TRANSFER, &[writes(1, bytes(0..256), copy_write())], None);
    update.update_fence(fence, None);
    assert_eq!(commit(&mut queue, &[update.finish()], 1), [None]);
    // Not completed yet: still ordered.
    let read = || one(&[reads(1, bytes(0..256), shader_read())]);
    assert_eq!(queue.entries(), 2);
    assert_eq!(
        queue.plan(&[read()], at(2)).barriers(),
        [barrier(copy_write(), shader_read())]
    );
    // Completed: the host saw it before the next submission, so nothing is left to order.
    queue.completed(0, 1);
    assert_eq!(queue.entries(), 0);
    let mut wait = EncoderUses::<Key>::new(false);
    wait.wait_for_fence(fence, None);
    wait.record(COMPUTE, &[reads(1, bytes(0..256), shader_read())], None);
    assert_eq!(commit(&mut queue, &[wait.finish()], 2), [None]);
}

#[test]
fn the_state_retires_only_through_the_completed_value() {
    let mut queue = Tracker::default();
    // Two reads in two batches; the first completed, the second not.
    commit(
        &mut queue,
        &[one(&[reads(1, bytes(0..256), copy_read())])],
        1,
    );
    commit(
        &mut queue,
        &[one(&[reads(1, bytes(0..256), shader_read())])],
        2,
    );
    queue.completed(0, 1);
    // The second read may still run: a write orders after it. One queue's reads of a range are
    // kept together, at the latest batch, so they retire together (ordering more, never less).
    assert_eq!(
        queue
            .plan(&[one(&[writes(1, bytes(0..256), shader_write())])], at(3))
            .barriers(),
        [barrier(
            Access::new(TRANSFER | COMPUTE, NO_ACCESS),
            Access::new(COMPUTE, NO_ACCESS)
        )]
    );
    // An older value, or the same one again, changes nothing.
    queue.completed(0, 1);
    queue.completed(0, 0);
    assert_eq!(queue.entries(), 1);
    queue.completed(0, 2);
    assert_eq!(queue.entries(), 0);
}

/// A batch's count of the earlier batches of its `VkQueue` the host has not seen complete
/// ([`Tracker::pending_before`], which `WorkCounts::batches_with_none_pending` and kin count) is
/// the tracker's own, from the completed value its plan retires by: a read after the write of the
/// batch before it has a barrier exactly where that batch is counted pending. A count from a
/// later read of the queue's value could find that batch complete where the plan ordered after it.
#[test]
fn a_batch_is_counted_pending_behind_exactly_the_work_its_plan_orders_after() {
    let write = || one(&[writes(1, bytes(0..256), copy_write())]);
    let read = || one(&[reads(1, bytes(0..256), copy_read())]);
    for seen in [0, 1] {
        let mut queue = Tracker::default();
        assert_eq!(queue.pending_before(at(1)), 0);
        assert_eq!(commit(&mut queue, &[write()], 1), [None]);
        queue.completed(0, seen);
        let pending = queue.pending_before(at(2));
        let barriers = commit(&mut queue, &[read()], 2);
        let ordered = (seen == 0)
            .then(|| barrier(copy_write(), copy_read()))
            .flatten();
        assert_eq!(
            (pending, barriers),
            (1 - seen, vec![ordered]),
            "seen {seen}"
        );
    }
    // Another queue's completion, and an older value, leave a queue's count as it was.
    let mut queue = Tracker::default();
    for value in 1..=3 {
        assert_eq!(commit(&mut queue, &[read()], value), [None]);
    }
    queue.completed(0, 2);
    queue.completed(1, 3);
    queue.completed(0, 1);
    assert_eq!(queue.pending_before(at(4)), 1);
    assert_eq!(queue.pending_before(on(1, 4)), 0);
}

/// The chain of `runtime_lane::commit_work`'s chained batches, on one `VkQueue`: a fill of link
/// 0, then a batch for each later link copying the link before into it. Planned while the batch
/// before it is in flight (the tracker has not seen it complete), each copy begins with a barrier
/// after that batch's write, and counts it pending; planned after it completed, neither. The
/// device lane cannot hold a batch in flight for the host (no batch is submitted waiting for a
/// value the host sets, `event_timeline.rs`), so this sets the tracker's state.
#[test]
fn a_chain_of_copies_orders_each_after_the_one_before_only_while_it_is_in_flight() {
    const CHAIN: u64 = 8;
    let link = || bytes(0..4096);
    for in_flight in [true, false] {
        let mut queue = Tracker::default();
        let fill = one(&[writes(0, link(), copy_write())]);
        assert_eq!(commit(&mut queue, &[fill], 1), [None]);
        for to in 1..=CHAIN {
            let value = to + 1;
            if !in_flight {
                queue.completed(0, value - 1);
            }
            let copy = one(&[
                reads(to - 1, link(), copy_read()),
                writes(to, link(), copy_write()),
            ]);
            let pending = queue.pending_before(at(value));
            let barriers = commit(&mut queue, &[copy], value);
            let (want_pending, ordered) = match in_flight {
                true => (value - 1, barrier(copy_write(), copy_read())),
                false => (0, None),
            };
            assert_eq!(
                (pending, barriers),
                (want_pending, vec![ordered]),
                "link {to}, in flight {in_flight}"
            );
        }
    }
}

/// A completion looks only at the resources its batches touched: a thousand batches in flight,
/// each reading a resource of its own, and the first one's completion retires its resource alone.
#[test]
fn a_completion_looks_only_at_what_its_batches_touched() {
    let mut queue = Tracker::default();
    for serial in 1..=1000u64 {
        let read = one(&[reads(serial, bytes(0..64), copy_read())]);
        assert_eq!(commit(&mut queue, &[read], serial), [None]);
    }
    assert_eq!((queue.entries(), queue.to_retire()), (1000, 1000));
    queue.completed(0, 1);
    assert_eq!((queue.entries(), queue.to_retire()), (999, 999));
    queue.completed(0, 1000);
    assert_eq!((queue.entries(), queue.to_retire()), (0, 0));
}

/// Batches of reads and writes of a few resources' ranges and fences, over three queues, each
/// plan applied or dropped, and each queue's batches completing in runs: after every completion
/// the state holds no access or fence update of a completed batch, as a look at every resource
/// and fence would leave it, though each completion looks only at what its batches touched.
#[test]
fn completions_leave_no_access_of_a_completed_batch() {
    let mut next = numbers();
    let mut queue = Tracker::default();
    let (mut submitted, mut completed) = ([0u64; 3], [0u64; 3]);
    let mut retired = 0;
    for _ in 0..2000 {
        let encoders: Vec<Summary<Key>> = (0..1 + next(3))
            .map(|_| random_encoder(&mut next))
            .collect();
        let on_queue = next(3) as usize;
        let plan = queue.plan(&encoders, on(on_queue, submitted[on_queue] + 1));
        if next(8) == 0 {
            drop(plan);
        } else {
            queue.apply(plan);
            submitted[on_queue] += 1;
        }
        let done = next(3) as usize;
        if completed[done] < submitted[done] {
            completed[done] += 1 + next(submitted[done] - completed[done]);
            let before = queue.entries();
            queue.completed(done, completed[done]);
            retired += before - queue.entries();
            assert!(!queue.holds_completed());
        }
    }
    // The run retired entries, and kept a bounded state.
    assert!(retired > 0);
    assert!(queue.entries() <= 6 * ENTRIES_PER_KEY + 2);
}

/// A batch of one encoder is planned against the state itself and admitted into it as its plan
/// is applied; the same encoder followed by an empty one is planned through copies of the
/// entries. Over random batches on three queues, plans applied or dropped and completions in
/// runs, both plan the same barriers and waits and leave the same state.
#[test]
fn a_batch_of_one_encoder_plans_as_one_of_several_does() {
    let mut next = numbers();
    let empty = EncoderUses::<Key>::new(true).finish();
    let (mut alone, mut copied) = (Tracker::default(), Tracker::default());
    let (mut submitted, mut completed) = ([0u64; 3], [0u64; 3]);
    let mut applied = 0;
    for _ in 0..3000 {
        let encoder = random_encoder(&mut next);
        let on_queue = next(3) as usize;
        let point = on(on_queue, submitted[on_queue] + 1);
        let one = alone.plan([&encoder], point);
        let two = copied.plan([&encoder, &empty], point);
        assert_eq!(one.barriers(), &two.barriers()[..1]);
        assert_eq!(two.barriers()[1], None);
        assert_eq!(one.waits(), two.waits());
        if next(8) == 0 {
            drop((one, two));
        } else {
            alone.apply(one);
            copied.apply(two);
            submitted[on_queue] += 1;
            applied += 1;
        }
        let done = next(3) as usize;
        if completed[done] < submitted[done] {
            completed[done] += 1 + next(submitted[done] - completed[done]);
            alone.completed(done, completed[done]);
            copied.completed(done, completed[done]);
        }
        assert!(alone.same_state(&copied));
        assert_eq!(alone.to_retire(), copied.to_retire());
    }
    assert!(applied > 2000);
}

/// A batch of one encoder copies no entry of the state: it is planned against the state and
/// admitted into it as its plan is applied. A batch of several copies those of the resources it
/// uses.
#[test]
fn a_batch_of_one_encoder_copies_no_entries() {
    let mut queue = Tracker::default();
    let write = one(&[writes(1, bytes(0..256), shader_write())]);
    let read = one(&[
        reads(1, bytes(0..256), shader_read()),
        reads(2, bytes(0..64), shader_read()),
    ]);
    assert_eq!(commit(&mut queue, std::slice::from_ref(&write), 1), [None]);
    let plan = queue.plan([&read], at(2));
    assert_eq!(plan.copied(), 0);
    assert_eq!(plan.barriers(), [barrier(shader_write(), shader_read())]);
    queue.apply(plan);
    let plan = queue.plan([&write, &read], at(3));
    assert_eq!(plan.copied(), 2);
    assert_eq!(
        plan.barriers(),
        [
            barrier(shader_write(), shader_write()),
            barrier(shader_write(), shader_read())
        ]
    );
}

/// A run's pseudo-random numbers below a bound: xorshift, from a fixed seed.
fn numbers() -> impl FnMut(u64) -> u64 {
    let mut seed = 0x9e37_79b9_7f4a_7c15u64;
    move |bound: u64| {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed % bound
    }
}

/// An encoder, ordered or not, of one to four commands, each reading or writing one of six
/// resources' byte ranges, which may wait for and update one of two fences.
fn random_encoder(next: &mut impl FnMut(u64) -> u64) -> Summary<Key> {
    let mut uses = EncoderUses::<Key>::new(next(2) == 0);
    for _ in 0..1 + next(4) {
        let start = next(8) * 32;
        let range = bytes(start..start + 32 * (1 + next(4)));
        let key = next(6);
        let used = match next(4) {
            0 => writes(key, range, copy_write()),
            1 => writes(key, range, shader_write()),
            2 => reads(key, range, copy_read()),
            _ => reads(key, range, shader_read()),
        };
        let command = [used];
        uses.order(vk::PipelineStageFlags2::NONE, &command, Reach::Unnamed);
    }
    if next(4) == 0 {
        uses.wait_for_fence(100 + next(2), None);
    }
    if next(4) == 0 {
        uses.update_fence(100 + next(2), None);
    }
    uses.finish()
}

#[test]
fn each_queue_retires_by_its_own_timeline() {
    let mut queue = Tracker::default();
    // A write on queue 0 and a read of it on queue 1.
    submit(
        &mut queue,
        &[one(&[writes(1, bytes(0..256), copy_write())])],
        on(0, 1),
    );
    let (_, waits) = submit(
        &mut queue,
        &[one(&[reads(1, bytes(0..256), shader_read())])],
        on(1, 1),
    );
    assert_eq!(waits, [wait(on(0, 1), COMPUTE)]);
    // Queue 1's value 1 completing retires its read, not queue 0's write.
    queue.completed(1, 1);
    let (_, waits) = submit(
        &mut queue,
        &[one(&[reads(1, bytes(0..256), copy_read())])],
        on(2, 1),
    );
    assert_eq!(waits, [wait(on(0, 1), TRANSFER)]);
    queue.completed(0, 1);
    // The write retired; the read on queue 2 has not.
    let (barriers, waits) = submit(
        &mut queue,
        &[one(&[writes(1, bytes(0..256), shader_write())])],
        on(1, 2),
    );
    assert_eq!(barriers, [None]);
    assert_eq!(waits, [wait(on(2, 1), COMPUTE)]);
}

#[test]
fn a_read_on_another_queue_waits_for_the_write() {
    let mut queue = Tracker::default();
    submit(
        &mut queue,
        &[one(&[writes(1, bytes(0..256), shader_write())])],
        on(0, 1),
    );
    let (barriers, waits) = submit(
        &mut queue,
        &[one(&[reads(1, bytes(0..256), copy_read())])],
        on(1, 4),
    );
    // No barrier on queue 1: the wait for queue 0's batch makes the write visible.
    assert_eq!(barriers, [None]);
    assert_eq!(waits, [wait(on(0, 1), TRANSFER)]);
}

#[test]
fn a_wait_covers_the_later_batches_of_its_queue() {
    let mut queue = Tracker::default();
    submit(
        &mut queue,
        &[one(&[writes(1, bytes(0..256), shader_write())])],
        on(0, 1),
    );
    submit(
        &mut queue,
        &[one(&[reads(1, bytes(0..256), copy_read())])],
        on(1, 1),
    );
    // The same stages again on queue 1: the earlier wait covers them.
    let (_, waits) = submit(
        &mut queue,
        &[one(&[reads(1, bytes(0..256), copy_read())])],
        on(1, 2),
    );
    assert_eq!(waits, []);
    // Other stages need a wait of their own.
    let (_, waits) = submit(
        &mut queue,
        &[one(&[reads(1, bytes(0..256), shader_read())])],
        on(1, 3),
    );
    assert_eq!(waits, [wait(on(0, 1), COMPUTE)]);
}

#[test]
fn a_write_waits_for_every_queue_that_wrote_or_read() {
    let mut queue = Tracker::default();
    submit(
        &mut queue,
        &[one(&[writes(1, bytes(0..256), shader_write())])],
        on(0, 1),
    );
    submit(
        &mut queue,
        &[one(&[reads(1, bytes(0..128), copy_read())])],
        on(1, 1),
    );
    submit(
        &mut queue,
        &[one(&[reads(1, bytes(128..256), copy_read())])],
        on(1, 2),
    );
    let (barriers, waits) = submit(
        &mut queue,
        &[one(&[writes(1, bytes(0..256), copy_write())])],
        on(2, 1),
    );
    assert_eq!(barriers, [None]);
    // The last write's queue and the latest read of the other queue, once each.
    assert_eq!(waits, [wait(on(0, 1), TRANSFER), wait(on(1, 2), TRANSFER)]);
    // The write replaced them: a read on queue 0 now follows queue 2 alone.
    let (barriers, waits) = submit(
        &mut queue,
        &[one(&[reads(1, bytes(0..256), shader_read())])],
        on(0, 2),
    );
    assert_eq!(barriers, [None]);
    assert_eq!(waits, [wait(on(2, 1), COMPUTE)]);
}

#[test]
fn logical_queues_on_one_vk_queue_order_by_barriers() {
    // Two logical queues share VkQueue 0: their batches are one submission order there.
    let mut queue = Tracker::default();
    let (_, waits) = submit(
        &mut queue,
        &[one(&[writes(1, bytes(0..256), shader_write())])],
        on(0, 1),
    );
    assert_eq!(waits, []);
    let (barriers, waits) = submit(
        &mut queue,
        &[one(&[reads(1, bytes(0..256), shader_read())])],
        on(0, 2),
    );
    assert_eq!(barriers, [barrier(shader_write(), shader_read())]);
    assert_eq!(waits, []);
}

/// Where every batch ends with [`Barrier::to_all`] (`batch_order.rs`), every write before every
/// access of every later command of its `VkQueue`, whichever logical queue committed it (logical
/// queues sharing a `VkQueue` share its submission order), a later batch there needs no barrier
/// from an earlier one's writes, reads or fence updates. Inside a batch the encoders still order
/// among themselves, and a batch of another `VkQueue` still waits.
#[test]
fn batches_of_one_vk_queue_ordered_by_their_ends_need_no_barrier_between_them() {
    let fence: Key = 100;
    let mut update = EncoderUses::<Key>::new(true);
    update.record(TRANSFER, &[writes(1, bytes(0..256), copy_write())], None);
    update.update_fence(fence, None);
    let mut waiting = EncoderUses::<Key>::new(false);
    waiting.wait_for_fence(fence, None);
    waiting.record(COMPUTE, &[], None);
    let mut queue = Tracker::by_ends();
    assert_eq!(commit(&mut queue, &[update.finish()], 1), [None]);
    // A read after the write, a write after the read, a wait for the fence.
    let read = one(&[reads(1, bytes(0..256), shader_read())]);
    assert_eq!(commit(&mut queue, std::slice::from_ref(&read), 2), [None]);
    let write = one(&[writes(1, bytes(0..256), shader_write())]);
    assert_eq!(commit(&mut queue, std::slice::from_ref(&write), 3), [None]);
    assert_eq!(commit(&mut queue, &[waiting.finish()], 4), [None]);
    // A read and a write after it in one batch: an execution dependency between them alone.
    let copy = one(&[writes(1, bytes(0..256), copy_write())]);
    assert_eq!(
        commit(&mut queue, &[read, copy], 5),
        [
            None,
            barrier(
                Access::new(COMPUTE, NO_ACCESS),
                Access::new(TRANSFER, NO_ACCESS)
            )
        ]
    );
    // Another `VkQueue` waits for the last write.
    let (barriers, waits) = submit(
        &mut queue,
        &[one(&[reads(1, bytes(0..256), copy_read())])],
        on(1, 1),
    );
    assert_eq!(barriers, [None]);
    assert_eq!(waits, [wait(at(5), TRANSFER)]);
}

/// The same batches in the exact form plan a barrier for each dependency on an earlier batch of
/// the `VkQueue`, as within one.
#[test]
fn batches_of_one_vk_queue_in_the_exact_form_order_by_barriers() {
    let mut queue = Tracker::default();
    assert_eq!(
        commit(
            &mut queue,
            &[one(&[writes(1, bytes(0..256), copy_write())])],
            1
        ),
        [None]
    );
    let read = one(&[reads(1, bytes(0..256), shader_read())]);
    assert_eq!(
        commit(&mut queue, std::slice::from_ref(&read), 2),
        [barrier(copy_write(), shader_read())]
    );
    assert_eq!(
        commit(
            &mut queue,
            &[one(&[writes(1, bytes(0..256), shader_write())])],
            3
        ),
        // After the read, and after the copy's write, which the read's barrier made visible to
        // reads alone.
        [barrier(
            Access::new(TRANSFER | COMPUTE, vk::AccessFlags2::TRANSFER_WRITE),
            shader_write()
        )]
    );
}

/// Merged entries ordered by their batches' ends: a write over the merged reads of earlier
/// batches needs nothing on their `VkQueue`, and one on another waits for them at their latest.
#[test]
fn merged_entries_ordered_by_their_ends_still_order_another_queue() {
    let mut queue = Tracker::by_ends();
    for serial in 0..1000u64 {
        let start = serial % 120;
        let got = commit(
            &mut queue,
            &[one(&[reads(1, bytes(start..start + 8), copy_read())])],
            serial + 1,
        );
        assert_eq!(got, [None]);
        assert!(queue.entries() <= ENTRIES_PER_KEY);
    }
    let write = one(&[writes(1, bytes(32..48), shader_write())]);
    assert_eq!(
        commit(&mut queue, std::slice::from_ref(&write), 1001),
        [None]
    );
    let (barriers, waits) = submit(&mut queue, &[write], on(1, 1));
    assert_eq!(barriers, [None]);
    assert_eq!(waits, [wait(at(1001), COMPUTE)]);
}

#[test]
fn a_fence_updated_on_another_queue_is_a_wait() {
    let fence: Key = 100;
    let mut update = EncoderUses::<Key>::new(true);
    update.record(
        TRANSFER,
        &[Use::writes(5, false, bytes(0..64), copy_write())],
        None,
    );
    update.update_fence(fence, None);
    let mut waiting = EncoderUses::<Key>::new(false);
    waiting.wait_for_fence(fence, None);
    waiting.record(
        COMPUTE,
        &[Use::reads(5, false, bytes(0..64), shader_read())],
        None,
    );
    let mut queue = Tracker::default();
    submit(&mut queue, &[update.finish()], on(0, 3));
    let (barriers, waits) = submit(&mut queue, &[waiting.finish()], on(1, 1));
    assert_eq!(barriers, [None]);
    assert_eq!(waits, [wait(on(0, 3), COMPUTE)]);
    // Retired once queue 0's value 3 completed.
    queue.completed(0, 3);
    assert_eq!(queue.entries(), 0);
}

#[test]
fn a_waits_stages_come_from_the_whole_batch() {
    // Two encoders of one batch on queue 1 read what queue 0 wrote: one wait, for both stages.
    let mut queue = Tracker::default();
    submit(
        &mut queue,
        &[one(&[
            writes(1, bytes(0..64), shader_write()),
            writes(2, bytes(0..64), shader_write()),
        ])],
        on(0, 1),
    );
    submit(
        &mut queue,
        &[one(&[writes(3, bytes(0..64), shader_write())])],
        on(0, 2),
    );
    let (barriers, waits) = submit(
        &mut queue,
        &[
            one(&[reads(1, bytes(0..64), copy_read())]),
            one(&[reads(2, bytes(0..64), shader_read())]),
            one(&[reads(3, bytes(0..64), shader_read())]),
        ],
        on(1, 1),
    );
    assert_eq!(barriers, [None, None, None]);
    assert_eq!(waits, [wait(on(0, 2), TRANSFER | COMPUTE)]);
}

#[test]
fn merged_entries_of_several_queues_still_order_every_access() {
    let mut queue = Tracker::default();
    // Writes on queues 0 and 1, alternating, past the entry bound: they merge.
    for index in 0..(ENTRIES_PER_KEY as u64 * 2) {
        let start = index * 64;
        submit(
            &mut queue,
            &[one(&[writes(1, bytes(start..start + 64), copy_write())])],
            on((index % 2) as usize, index + 1),
        );
    }
    assert!(queue.entries() <= ENTRIES_PER_KEY);
    // A read on queue 2 of the first bytes waits for queue 0, which wrote them, and for queue 1,
    // whose writes merged with them (more than Metal needs, never less).
    let (barriers, waits) = submit(
        &mut queue,
        &[one(&[reads(1, bytes(0..8), shader_read())])],
        on(2, 1),
    );
    assert_eq!(barriers, [None]);
    let waits_for = |waits: &[Wait], queue: usize, stages| {
        waits
            .iter()
            .any(|wait| wait.at.queue == queue && wait.at.value >= 1 && wait.stages == stages)
    };
    assert_eq!(waits.len(), 2);
    assert!(waits_for(&waits, 0, COMPUTE) && waits_for(&waits, 1, COMPUTE));
    // A read on queue 0 needs a barrier from its own queue's writes and a wait for queue 1's.
    let (barriers, waits) = submit(
        &mut queue,
        &[one(&[reads(1, bytes(0..8), copy_read())])],
        on(0, 100),
    );
    assert_eq!(barriers, [barrier(copy_write(), copy_read())]);
    assert_eq!(waits.len(), 1);
    assert!(waits_for(&waits, 1, TRANSFER));
}

#[test]
fn the_state_stays_bounded_and_still_orders() {
    let mut queue = Tracker::default();
    for serial in 1..=1000u64 {
        let at = serial * 16;
        let got = commit(
            &mut queue,
            &[one(&[reads(1, bytes(at..at + 16), copy_read())])],
            serial,
        );
        assert_eq!(got, [None]);
        assert!(queue.entries() <= ENTRIES_PER_KEY);
    }
    // The merged reads still order a write over any of them.
    assert_eq!(
        commit(
            &mut queue,
            &[one(&[writes(1, bytes(32..48), shader_write())])],
            1001
        ),
        [barrier(
            Access::new(TRANSFER, NO_ACCESS),
            Access::new(COMPUTE, NO_ACCESS)
        )]
    );
}

#[test]
fn merged_writes_still_order_every_read() {
    let mut queue = Tracker::default();
    for serial in 1..=(ENTRIES_PER_KEY as u64 * 4) {
        let at = serial * 64;
        commit(
            &mut queue,
            &[one(&[writes(1, bytes(at..at + 64), copy_write())])],
            serial,
        );
    }
    assert!(queue.entries() <= ENTRIES_PER_KEY);
    // A read of the first write's bytes, long merged away.
    assert_eq!(
        commit(
            &mut queue,
            &[one(&[reads(1, bytes(64..72), shader_read())])],
            1000
        ),
        [barrier(copy_write(), shader_read())]
    );
}

#[test]
fn an_encoder_summary_stays_bounded() {
    let mut uses = EncoderUses::<Key>::new(true);
    for at in 0..1000u64 {
        let command = [reads(1, bytes(at * 4..at * 4 + 4), copy_read())];
        uses.order(TRANSFER, &command, Reach::Unnamed);
    }
    let summary = uses.finish();
    assert!(summary.uses().len() <= ENTRIES_PER_KEY);
    // What it claims holds every read.
    for at in 0..1000u64 {
        let read = bytes(at * 4..at * 4 + 4);
        assert!(summary.uses().iter().any(|used| used.range.contains(&read)));
    }
}

#[test]
fn ranges_meet_and_hold() {
    assert!(bytes(0..16).overlaps(&bytes(15..17)));
    assert!(!bytes(0..16).overlaps(&bytes(16..32)));
    assert!(bytes(0..16).contains(&bytes(0..16)));
    assert!(!bytes(0..16).contains(&bytes(8..17)));
    let color = vk::ImageAspectFlags::COLOR;
    assert!(image(color, 0..2, 0..2).overlaps(&image(color, 1..3, 1..3)));
    assert!(!image(color, 0..2, 0..2).overlaps(&image(color, 2..3, 0..2)));
    assert!(image(color, 0..4, 0..4).contains(&image(color, 1..2, 3..4)));
    assert!(!image(color, 0..4, 0..4).contains(&image(
        color | vk::ImageAspectFlags::DEPTH,
        1..2,
        3..4
    )));
    // Different kinds of range on one key: taken to meet, never to hold one another.
    assert!(bytes(0..16).overlaps(&image(color, 0..1, 0..1)));
    assert!(!bytes(0..u64::MAX).contains(&image(color, 0..1, 0..1)));
    assert!(bytes(4..4).is_empty());
    assert!(image(vk::ImageAspectFlags::empty(), 0..1, 0..1).is_empty());
}

#[test]
fn legacy_masks_keep_the_shared_bits() {
    let got = Barrier {
        src: Access::new(
            COMPUTE | TRANSFER,
            vk::AccessFlags2::SHADER_WRITE | vk::AccessFlags2::TRANSFER_WRITE,
        ),
        dst: Access::new(FRAGMENT, vk::AccessFlags2::UNIFORM_READ),
    }
    .legacy();
    assert_eq!(
        got,
        LegacyBarrier {
            src_stages: vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::TRANSFER,
            src_access: vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::TRANSFER_WRITE,
            dst_stages: vk::PipelineStageFlags::FRAGMENT_SHADER,
            dst_access: vk::AccessFlags::UNIFORM_READ,
        }
    );
}

#[test]
fn the_command_buffer_barriers_keep_their_masks() {
    let start = Barrier::after_completed().legacy();
    assert_eq!(
        start,
        LegacyBarrier {
            src_stages: vk::PipelineStageFlags::TOP_OF_PIPE,
            src_access: vk::AccessFlags::empty(),
            dst_stages: vk::PipelineStageFlags::ALL_COMMANDS,
            dst_access: vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE,
        }
    );
    let end = Barrier::to_host().legacy();
    assert_eq!(
        end,
        LegacyBarrier {
            src_stages: vk::PipelineStageFlags::ALL_COMMANDS,
            src_access: vk::AccessFlags::MEMORY_WRITE,
            dst_stages: vk::PipelineStageFlags::HOST,
            dst_access: vk::AccessFlags::HOST_READ | vk::AccessFlags::HOST_WRITE,
        }
    );
    let all = Barrier::to_all().legacy();
    assert_eq!(
        all,
        LegacyBarrier {
            src_stages: vk::PipelineStageFlags::ALL_COMMANDS,
            src_access: vk::AccessFlags::MEMORY_WRITE,
            dst_stages: vk::PipelineStageFlags::HOST | vk::PipelineStageFlags::ALL_COMMANDS,
            dst_access: vk::AccessFlags::HOST_READ
                | vk::AccessFlags::HOST_WRITE
                | vk::AccessFlags::MEMORY_READ
                | vk::AccessFlags::MEMORY_WRITE,
        }
    );
}

#[test]
fn legacy_masks_map_the_newer_bits() {
    use vk::AccessFlags2 as A2;
    use vk::PipelineStageFlags2 as S2;
    let got = Barrier {
        src: Access::new(S2::COPY | S2::CLEAR, A2::SHADER_STORAGE_WRITE),
        dst: Access::new(
            S2::INDEX_INPUT | S2::PRE_RASTERIZATION_SHADERS,
            A2::SHADER_STORAGE_READ | A2::SHADER_SAMPLED_READ,
        ),
    }
    .legacy();
    assert_eq!(
        got,
        LegacyBarrier {
            src_stages: vk::PipelineStageFlags::TRANSFER,
            src_access: vk::AccessFlags::SHADER_WRITE,
            dst_stages: vk::PipelineStageFlags::VERTEX_INPUT | vk::PipelineStageFlags::ALL_GRAPHICS,
            dst_access: vk::AccessFlags::SHADER_READ,
        }
    );
    // A stage or access type the older family lacks orders everything.
    let got = Barrier {
        src: Access::new(S2::VIDEO_DECODE_KHR, A2::SHADER_BINDING_TABLE_READ_KHR),
        dst: Access::new(S2::COMPUTE_SHADER, A2::NONE),
    }
    .legacy();
    assert_eq!(
        got,
        LegacyBarrier {
            src_stages: vk::PipelineStageFlags::ALL_COMMANDS,
            src_access: vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE,
            dst_stages: vk::PipelineStageFlags::COMPUTE_SHADER,
            dst_access: vk::AccessFlags::empty(),
        }
    );
    // No stage is the top of the pipe as a source, the bottom as a destination.
    let got = Barrier {
        src: Access::NONE,
        dst: Access::NONE,
    }
    .legacy();
    assert_eq!(
        got,
        LegacyBarrier {
            src_stages: vk::PipelineStageFlags::TOP_OF_PIPE,
            src_access: vk::AccessFlags::empty(),
            dst_stages: vk::PipelineStageFlags::BOTTOM_OF_PIPE,
            dst_access: vk::AccessFlags::empty(),
        }
    );
}

#[test]
fn layout_moves_outside_encoders_order_as_writes_of_the_whole_image() {
    // A batch moves a caller's image into `GENERAL` before its encoders and home after them: each
    // move is a write of the whole image in every stage, ordered against the image's other uses on
    // any queue.
    let mut queue = Tracker::default();
    let whole = image(vk::ImageAspectFlags::COLOR, 0..1, 0..1);
    let moved = Access::new(
        vk::PipelineStageFlags2::ALL_COMMANDS,
        vk::AccessFlags2::MEMORY_WRITE,
    );
    let entry = Summary::outside_encoders(vec![writes(1, whole.clone(), moved)]);
    let exit = Summary::outside_encoders(vec![writes(1, whole.clone(), moved)]);
    assert_eq!(entry.keys().collect::<Vec<_>>(), [1]);
    submit(
        &mut queue,
        &[one(&[reads(1, whole.clone(), copy_read())])],
        on(1, 1),
    );
    let (barriers, waits) = submit(
        &mut queue,
        &[
            entry,
            one(&[writes(1, whole.clone(), shader_write())]),
            exit,
        ],
        on(0, 1),
    );
    assert_eq!(
        waits,
        [wait(on(1, 1), vk::PipelineStageFlags2::ALL_COMMANDS)]
    );
    assert_eq!(
        barriers,
        [
            None,
            barrier(moved, shader_write()),
            barrier(shader_write(), moved)
        ]
    );
    // The next batch's read of the image orders after the move home.
    let got = commit(&mut queue, &[one(&[reads(1, whole, copy_read())])], 2);
    assert_eq!(got, [barrier(moved, copy_read())]);
}

/// A segment's summary joined to its encoder's (an indirect command buffer's commands, recorded
/// at commit) orders later encoders as one encoder of every command would: its stages and
/// tracked uses join, a use of a range the encoder used widens it, and its fences join.
#[test]
fn a_joined_summary_orders_as_one_encoder() {
    let commands: [&[Use<Key>]; 3] = [
        &[writes(1, bytes(0..256), copy_write())],
        &[reads(1, bytes(0..256), shader_read())],
        &[writes(2, bytes(0..64), shader_write())],
    ];
    let mut joined = encoder(&commands[..1]);
    joined.join(encoder(&commands[1..]));
    let whole = encoder(&commands);
    let mut fenced = EncoderUses::<Key>::new(true);
    fenced.update_fence(9, None);
    joined.join(fenced.finish());
    assert_eq!(joined.uses(), whole.uses());
    let later = [one(&[
        writes(1, bytes(0..256), copy_write()),
        reads(2, bytes(0..64), copy_read()),
    ])];
    let (mut a, mut b) = (Tracker::default(), Tracker::default());
    assert_eq!(
        commit(&mut a, &[joined, later[0].clone()], 1),
        commit(&mut b, &[whole, later[0].clone()], 1)
    );
    // The joined fence update orders a later wait after every stage the joined work ran in.
    let mut waits = EncoderUses::<Key>::new(false);
    waits.wait_for_fence(9, None);
    waits.record(FRAGMENT, &[], None);
    assert_eq!(
        commit(&mut a, &[waits.finish()], 2),
        [barrier(
            Access::new(TRANSFER | COMPUTE, vk::AccessFlags2::MEMORY_WRITE),
            Access::new(
                FRAGMENT,
                vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE
            )
        )]
    );
}

/// A joined summary keeps at most [`ENTRIES_PER_KEY`] ranges of a resource, and what it claims
/// holds every use joined.
#[test]
fn a_joined_summary_stays_bounded() {
    let mut joined = one(&[reads(1, bytes(0..4), copy_read())]);
    for at in 1..100u64 {
        joined.join(one(&[reads(1, bytes(at * 8..at * 8 + 4), copy_read())]));
    }
    assert!(joined.uses().len() <= ENTRIES_PER_KEY);
    for at in 0..100u64 {
        let read = bytes(at * 8..at * 8 + 4);
        assert!(joined.uses().iter().any(|used| used.range.contains(&read)));
    }
}

/// `uses` with `other`'s joined as [`Summary::join`] joins them, each against every known use in
/// turn: the answer each join must give.
fn joined_use_by_use(mut uses: Vec<Use<Key>>, other: &[Use<Key>]) -> Vec<Use<Key>> {
    for used in other.iter().cloned() {
        let same = |known: &&mut Use<Key>| known.key == used.key && known.range == used.range;
        if let Some(known) = uses.iter_mut().find(same) {
            known.read = known.read.union(used.read);
            known.write = known.write.union(used.write);
            continue;
        }
        let ranges = uses.iter().filter(|known| known.key == used.key).count();
        let hull = (ranges >= ENTRIES_PER_KEY)
            .then(|| {
                uses.iter_mut().find_map(|known| {
                    let range =
                        (known.key == used.key).then(|| known.range.hull(&used.range))??;
                    Some((known, range))
                })
            })
            .flatten();
        match hull {
            Some((known, range)) => {
                known.range = range;
                known.read = known.read.union(used.read);
                known.write = known.write.union(used.write);
            }
            None => uses.push(used),
        }
    }
    uses.sort_by_key(|used| used.key);
    uses
}

/// `count` uses of resources below `keys`, from `seed`: reads and writes of byte ranges and of
/// image subresources, several of a resource, some of the same range, and some resources past
/// [`ENTRIES_PER_KEY`] ranges.
fn mixed_uses(seed: u64, count: usize, keys: u64) -> Vec<Use<Key>> {
    let mut at = seed;
    let mut next = move || {
        at = at
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        at >> 33
    };
    (0..count)
        .map(|_| {
            let key = next() % keys;
            let start = next() % 24;
            let range = match next() % 3 {
                0 => image(vk::ImageAspectFlags::COLOR, 0..1 + (start % 3) as u32, 0..1),
                _ => bytes(start * 16..start * 16 + 16),
            };
            match next() % 2 {
                0 => reads(key, range, shader_read()),
                _ => writes(key, range, copy_write()),
            }
        })
        .collect()
}

/// A summary's join gives what joining each use against every known use in turn gives, over
/// mixed uses (ranges met again, resources past [`ENTRIES_PER_KEY`] ranges, ranges of two
/// kinds), and looks at each resource's own uses alone: joining two summaries of 4096 resources
/// each, one range a resource and half of them in both, looks at fewer known uses than twice the
/// uses joined, where a scan of every known use for each would look at millions.
#[test]
fn a_join_looks_at_each_resource_s_own_uses_alone() {
    for seed in 0..64 {
        let known = Summary::outside_encoders(mixed_uses(seed, 200, 24));
        let other = Summary::outside_encoders(mixed_uses(seed + 1000, 200, 24));
        let want = joined_use_by_use(known.uses().to_vec(), other.uses());
        let mut joined = known.clone();
        joined.join(other);
        assert_eq!(joined.uses(), want, "seed {seed}");
    }
    let one = |keys: Range<u64>| {
        let uses = keys.map(|key| reads(key, bytes(0..16), shader_read()));
        Summary::outside_encoders(uses.collect())
    };
    let (mut joined, other) = (one(0..4096), one(2048..6144));
    let want = joined_use_by_use(joined.uses().to_vec(), other.uses());
    let before = join_looked_at();
    joined.join(other);
    let looked = join_looked_at() - before;
    assert_eq!(joined.uses(), want);
    assert!(looked < 2 * 4096, "{looked} known uses looked at");
}

/// A split encoder's commands after the split need no barrier after those before (they run once
/// everything before completed), and its parts' summaries take its fence waits first and its
/// updates last.
#[test]
fn a_split_encoder_waits_in_its_first_part_and_updates_in_its_last() {
    let write = [writes(1, bytes(0..256), shader_write())];
    let read = [reads(1, bytes(0..256), shader_read())];
    let mut whole = EncoderUses::<Key>::new(true);
    whole.record(COMPUTE, &write, None);
    assert!(whole.barrier(&read, Reach::Unnamed).is_some());
    let mut split = EncoderUses::<Key>::new(true);
    split.record(COMPUTE, &write, None);
    split.split();
    assert_eq!(split.barrier(&read, Reach::Unnamed), None);

    let (first, second): (Key, Key) = (100, 101);
    let mut a = EncoderUses::<Key>::new(false);
    a.record(FRAGMENT, &[], None);
    a.update_fence(first, None);
    let mut parted = EncoderUses::<Key>::new(false);
    parted.wait_for_fence(first, None);
    parted.record(TRANSFER, &[], None);
    parted.split();
    parted.record(COMPUTE, &[], None);
    parted.update_fence(second, None);
    let mut c = EncoderUses::<Key>::new(false);
    c.wait_for_fence(second, None);
    c.record(COMPUTE, &[], None);
    let parts = parted.finish_parts();
    assert_eq!(parts.len(), 2);
    let mut summaries = vec![a.finish()];
    summaries.extend(parts);
    summaries.push(c.finish());
    let mut queue = Tracker::default();
    let got = commit_each(&mut queue, &summaries);
    let written = |stages| Some(Access::new(stages, vk::AccessFlags2::MEMORY_WRITE));
    assert_eq!(got[1].map(|barrier| barrier.src), written(FRAGMENT));
    assert_eq!(got[2], None);
    assert_eq!(got[3].map(|barrier| barrier.src), written(COMPUTE));
}

/// An unordered encoder's commands so far are seen to write what a later command reads only where
/// a tracked range overlaps (the index bounds pass's hazard check, `index_bounds.rs`): not a read,
/// not another range or resource, and not an untracked write, which the encoder does not record.
#[test]
fn an_earlier_write_is_seen_only_where_a_tracked_read_overlaps_it() {
    let mut uses = EncoderUses::new(false);
    uses.record(
        FRAGMENT,
        &[
            writes(1, bytes(64..128), shader_write()),
            reads(2, bytes(0..64), shader_read()),
            Use::writes(3, false, bytes(0..64), shader_write()),
        ],
        None,
    );
    let read = |key, range, tracked| [Use::reads(key, tracked, bytes(range), shader_read())];
    assert!(uses.written(&read(1, 120..140, true)));
    assert!(!uses.written(&read(1, 0..64, true)));
    assert!(!uses.written(&read(2, 0..64, true)));
    assert!(!uses.written(&read(4, 0..64, true)));
    assert!(!uses.written(&read(3, 0..64, false)));
    // A write-only use reads nothing.
    assert!(!uses.written(&[writes(1, bytes(64..128), shader_write())]));
}

/// The footprint of a batch of `summaries`.
fn footprint(summaries: &[Summary<Key>]) -> Footprint<Key> {
    Footprint::of(summaries)
}

/// Two batches across queues run in commit order where a tracked use of each shares a range
/// that one of them writes, across the batch's encoders; reads of one range, disjoint ranges and
/// other resources do not.
#[test]
fn a_batch_conflicts_with_an_earlier_one_where_a_shared_range_is_written() {
    let write = footprint(&[
        one(&[reads(9, bytes(0..64), copy_read())]),
        one(&[writes(1, bytes(0..64), copy_write())]),
    ]);
    let read = footprint(&[one(&[reads(1, bytes(32..96), shader_read())])]);
    assert!(ordered(Some(&write), Some(&read)));
    assert!(ordered(Some(&read), Some(&write)));
    let elsewhere = footprint(&[one(&[reads(1, bytes(64..128), shader_read())])]);
    assert!(!ordered(Some(&write), Some(&elsewhere)));
    let other = footprint(&[one(&[writes(2, bytes(0..64), shader_write())])]);
    assert!(!ordered(Some(&write), Some(&other)));
    assert!(!ordered(Some(&read), Some(&elsewhere)));
    let levels = |levels: Range<u32>| image(vk::ImageAspectFlags::COLOR, levels, 0..1);
    let level0 = footprint(&[one(&[writes(3, levels(0..1), copy_write())])]);
    let level1 = footprint(&[one(&[reads(3, levels(1..2), copy_read())])]);
    assert!(!ordered(Some(&level0), Some(&level1)));
    let both = footprint(&[one(&[reads(3, levels(0..2), copy_read())])]);
    assert!(ordered(Some(&level0), Some(&both)));
}

/// A fence one batch updates orders it with a batch that waits for or updates it, whatever the
/// resources; a batch whose uses are not known yet conflicts with any that uses a tracked
/// resource or names a fence, and with none that does neither.
#[test]
fn fences_and_unknown_uses_order_batches_across_queues() {
    let fence: Key = 100;
    let mut updating = EncoderUses::new(true);
    updating.record(
        TRANSFER,
        &[Use::writes(5, false, bytes(0..64), copy_write())],
        None,
    );
    updating.update_fence(fence, None);
    let updating = footprint(&[updating.finish()]);
    let mut waiting = EncoderUses::new(true);
    waiting.wait_for_fence(fence, None);
    let waiting = footprint(&[waiting.finish()]);
    assert!(ordered(Some(&updating), Some(&waiting)));
    assert!(ordered(Some(&waiting), Some(&updating)));
    assert!(!ordered(Some(&waiting), Some(&waiting)));
    let untracked = footprint(&[one(&[Use::writes(5, false, bytes(0..64), copy_write())])]);
    let tracked = footprint(&[one(&[reads(1, bytes(0..4), copy_read())])]);
    assert!(ordered(None, Some(&tracked)));
    assert!(ordered(Some(&tracked), None));
    assert!(ordered(None, Some(&waiting)));
    assert!(!ordered(None, Some(&untracked)));
    assert!(!ordered(Some(&untracked), None));
    assert!(ordered::<Key>(None, None));
}

/// A draw's indirect argument reads meet a later draw's writes of the same bytes, of a tracked
/// resource or an untracked one alike, in the later draw's write stages alone; its other reads,
/// and writes elsewhere, meet nothing.
#[test]
fn a_draw_after_one_that_read_its_arguments_writes_them_in_its_write_stages() {
    use vk::AccessFlags2 as A2;
    use vk::PipelineStageFlags2 as S2;
    let arguments = Access::new(S2::DRAW_INDIRECT, A2::INDIRECT_COMMAND_READ);
    let vertex_write = Access::new(S2::VERTEX_SHADER, A2::SHADER_STORAGE_WRITE);
    let fragment_write = Access::new(FRAGMENT, A2::SHADER_STORAGE_WRITE);
    let mut known = IndirectReads::new();
    known.add(&[
        reads(1, bytes(0..20), arguments),
        Use::reads(2, false, bytes(64..80), arguments),
        reads(
            3,
            bytes(0..64),
            Access::new(S2::VERTEX_SHADER, A2::SHADER_STORAGE_READ),
        ),
    ]);
    let one = known.written_in(&[writes(1, bytes(0..8), vertex_write)]);
    assert!(one == S2::VERTEX_SHADER);
    let both = known.written_in(&[
        Use::writes(2, false, bytes(72..76), fragment_write),
        writes(1, bytes(16..20), vertex_write),
    ]);
    assert!(both == S2::VERTEX_SHADER | FRAGMENT);
    assert!(known
        .written_in(&[writes(1, bytes(20..64), vertex_write)])
        .is_empty());
    assert!(known
        .written_in(&[writes(3, bytes(0..64), vertex_write)])
        .is_empty());
    assert!(known
        .written_in(&[reads(1, bytes(0..20), arguments)])
        .is_empty());
    // Merged past the entries a resource keeps, the reads still meet every byte they read.
    let mut many = IndirectReads::new();
    for at in 0..=ENTRIES_PER_KEY as u64 {
        many.add(&[reads(1, bytes(at * 64..at * 64 + 20), arguments)]);
    }
    for at in 0..=ENTRIES_PER_KEY as u64 {
        let write = writes(1, bytes(at * 64 + 16..at * 64 + 20), vertex_write);
        assert!(many.written_in(&[write]) == S2::VERTEX_SHADER);
    }
}

/// A draw's writes before its fragments meet a later draw's fragment-shader reads and writes of
/// the same bytes, of a tracked resource or an untracked one alike: the barrier runs from the
/// writes' stages and access types to the fragment shader and the later accesses there. Reads
/// and writes in other stages, other bytes, and the earlier draw's own fragment writes meet
/// nothing.
#[test]
fn a_draw_s_fragments_after_one_that_wrote_before_its_fragments_wait_for_the_write() {
    use vk::AccessFlags2 as A2;
    use vk::PipelineStageFlags2 as S2;
    let vertex_write = Access::new(S2::VERTEX_SHADER, A2::SHADER_STORAGE_WRITE);
    let evaluation_write =
        Access::new(S2::TESSELLATION_EVALUATION_SHADER, A2::SHADER_STORAGE_WRITE);
    let fragment_read = Access::new(FRAGMENT, A2::SHADER_STORAGE_READ);
    let fragment_write = Access::new(FRAGMENT, A2::SHADER_STORAGE_WRITE);
    let vertex_read = Access::new(S2::VERTEX_SHADER, A2::SHADER_STORAGE_READ);
    let mut known = GeometryWrites::new();
    known.add(&[
        writes(1, bytes(0..16), vertex_write),
        Use::writes(2, false, bytes(64..80), evaluation_write),
        writes(3, bytes(0..64), fragment_write),
    ]);
    let read = known.barrier_before(&[reads(1, bytes(8..12), fragment_read)]);
    assert_eq!(read, barrier(vertex_write, fragment_read));
    let both = known.barrier_before(&[
        writes(1, bytes(0..4), fragment_write),
        Use::reads(2, false, bytes(72..76), fragment_read),
    ]);
    assert_eq!(
        both,
        barrier(
            vertex_write.union(evaluation_write),
            Access::new(FRAGMENT, A2::SHADER_STORAGE_READ | A2::SHADER_STORAGE_WRITE),
        )
    );
    assert_eq!(
        known.barrier_before(&[reads(1, bytes(16..64), fragment_read)]),
        None
    );
    assert_eq!(
        known.barrier_before(&[reads(1, bytes(0..16), vertex_read)]),
        None
    );
    assert_eq!(
        known.barrier_before(&[reads(3, bytes(0..64), fragment_read)]),
        None
    );
}

/// A resource the encoder's commands use at range after range (a buffer whose draws move along
/// it) makes room for every range it keeps before they merge, and the merge's own, at its second:
/// its list allocates once, through the merges, and the uses it hands on are those it kept.
#[test]
fn a_resource_used_at_moving_ranges_allocates_its_list_once() {
    let mut uses = EncoderUses::new(false);
    for at in 0..3 * ENTRIES_PER_KEY as u64 {
        let used = [reads(1, bytes(256 * at..256 * at + 16), shader_read())];
        uses.order(COMPUTE, &used, Reach::Named);
        if at > 0 {
            assert_eq!(uses.range_room(1), Some(ENTRIES_PER_KEY + 1), "range {at}");
        }
    }
    let summaries = uses.finish_parts();
    let kept = summaries[0].uses();
    assert!(!kept.is_empty() && kept.len() <= ENTRIES_PER_KEY);
    assert!(kept.iter().all(|used| used.key == 1));
}

/// An encoder's summary lists its tracked uses by resource, in key order, and each resource's
/// ranges in the order its commands first used them, however the resources were first used: a
/// render encoder's draws over a buffer at offsets that move back and forth, over several
/// buffers named in no order. So it does for few resources, whose keys are ordered on the stack,
/// and for more than that holds.
#[test]
fn a_summary_lists_its_uses_by_resource_each_in_recorded_order() {
    let few: Vec<Key> = vec![41, 7, 19, 3, 88, 23];
    // 40 distinct keys in no order: past what the stack orders.
    let many: Vec<Key> = (0..40).map(|i| (i * 37 + 11) % 101).collect();
    for keys in [few, many] {
        let mut uses = EncoderUses::new(false);
        // Each resource's ranges, 100 bytes apart so none meets another, used in this order.
        let starts: [u64; 4] = [300, 100, 500, 200];
        for &start in &starts {
            for &key in &keys {
                let used = reads(key, bytes(start..start + 10), shader_read());
                uses.record(FRAGMENT, &[used], None);
            }
        }
        let summary = uses.finish();
        let mut sorted = keys.clone();
        sorted.sort_unstable();
        let expected: Vec<(Key, Range<u64>)> = (sorted.iter())
            .flat_map(|&key| starts.iter().map(move |&start| (key, start..start + 10)))
            .collect();
        let got: Vec<(Key, Range<u64>)> = (summary.uses().iter())
            .map(|used| match &used.range {
                UseRange::Bytes(range) => (used.key, range.clone()),
                other => panic!("a use of bytes, not {other:?}"),
            })
            .collect();
        assert_eq!(got, expected, "{} resources", keys.len());
    }
}

/// An ordered encoder's commands write the uses it orders them by over the last command's storage
/// ([`EncoderUses::ordered_storage`]): commands of as many uses as the first make no list again,
/// and a command whose every use is tracked keeps the list it compared, making none apart; a
/// command with an untracked use makes the one it keeps once, and the commands like it after it
/// none.
#[test]
fn an_encoder_orders_its_commands_in_storage_it_keeps() {
    let mut uses = EncoderUses::<Key>::new(true);
    let tracked = [
        reads(1, bytes(0..64), copy_read()),
        writes(2, bytes(0..64), copy_write()),
        reads(1, bytes(0..64), shader_read()),
    ];
    uses.order(TRANSFER, &tracked, Reach::Unnamed);
    let first = uses.ordered_storage();
    assert_eq!(first[1].1, 0, "a list kept apart for tracked uses");
    for _ in 0..64 {
        uses.order(TRANSFER, &tracked, Reach::Unnamed);
        assert_eq!(uses.ordered_storage(), first);
    }
    let untracked = [
        reads(1, bytes(0..64), copy_read()),
        Use::writes(3, false, bytes(0..64), copy_write()),
    ];
    uses.order(TRANSFER, &untracked, Reach::Named);
    let second = uses.ordered_storage();
    assert_eq!(second[0], first[0]);
    assert!(second[1].1 > 0, "no list kept apart for an untracked use");
    for _ in 0..64 {
        uses.order(TRANSFER, &untracked, Reach::Named);
        uses.order(TRANSFER, &tracked, Reach::Unnamed);
        assert_eq!(uses.ordered_storage(), second);
    }
}

/// The uses an encoder orders a command by are those a merge through a map of each key and range
/// makes, sorted by key ([`merge`]'s reference, the form it took before): for commands of random
/// uses of few keys and ranges, repeated and touching nothing among them, the same uses in the
/// same order, so the encoder's barriers are the same.
#[test]
fn merging_in_place_keeps_each_key_and_range_once_in_order() {
    fn reference(uses: &[Use<Key>]) -> Vec<Use<Key>> {
        let mut merged: Vec<Use<Key>> = Vec::new();
        let mut index = std::collections::HashMap::new();
        for used in uses.iter().filter(|used| !used.is_nothing()) {
            match index.get(&(used.key, used.range.clone())) {
                Some(&at) => {
                    let known: &mut Use<Key> = &mut merged[at];
                    known.read = known.read.union(used.read);
                    known.write = known.write.union(used.write);
                }
                None => {
                    index.insert((used.key, used.range.clone()), merged.len());
                    merged.push(used.clone());
                }
            }
        }
        merged.sort_by_key(|used| used.key);
        merged
    }
    let mut seed = 0x2545_f491_4f6c_dd1du64;
    let mut next = |n: u64| {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed % n
    };
    for _ in 0..2000 {
        let uses: Vec<Use<Key>> = (0..next(12))
            .map(|_| {
                let start = 16 * next(3);
                let range = bytes(start..start + 16 * next(3));
                let access =
                    [copy_read(), copy_write(), shader_read(), Access::NONE][next(4) as usize];
                match next(2) {
                    0 => reads(next(4), range, access),
                    _ => writes(next(4), range, access),
                }
            })
            .collect();
        let mut merged: Vec<Use<Key>> = uses.iter().filter(|u| !u.is_nothing()).cloned().collect();
        merge(&mut merged);
        assert_eq!(merged, reference(&uses), "{uses:?}");
    }
}

/// A draw's writes of what its encoder made resident, added by their identity
/// (`UnorderedWrites::add_draw`), are walked once until a memory barrier clears them: the draws
/// after the first of that identity walk none of them, past the entries a resource keeps, and a
/// later read of what they wrote meets them all along; after the clear, the next draw of that
/// identity walks them again, and the read meets them again. A draw of another identity walks its
/// own, and the writes after them are added at every draw.
#[test]
fn draws_add_the_writes_of_what_their_encoder_made_resident_once_until_a_barrier() {
    let prefix = ResidentPrefix { id: 7, len: 2 };
    let uses = [
        writes(1, bytes(0..16), shader_write()),
        reads(2, bytes(0..16), shader_read()),
        writes(3, bytes(0..8), shader_write()),
    ];
    let read = |key| [reads(key, bytes(4..8), shader_read())];
    let mut unordered = UnorderedWrites::new();
    assert!(unordered.add_draw((&uses, prefix)));
    for _ in 0..=ENTRIES_PER_KEY {
        assert!(!unordered.add_draw((&uses, prefix)));
    }
    assert!(unordered.written(&read(1)) && unordered.written(&read(3)));
    unordered.clear();
    assert!(!unordered.written(&read(1)) && !unordered.written(&read(3)));
    assert!(
        unordered.add_draw((&uses, prefix)),
        "the draw after the barrier walks them again"
    );
    assert!(unordered.written(&read(1)) && unordered.written(&read(3)));
    let other = ResidentPrefix { id: 8, len: 2 };
    assert!(
        unordered.add_draw((&uses, other)),
        "another identity walks its own"
    );
    assert!(
        unordered.add_draw((&uses, prefix)),
        "and the first's again after it"
    );
    unordered.clear();
    unordered.add(&uses[2..]);
    assert!(unordered.add_draw((&uses, other)));
    assert!(unordered.written(&read(1)));
}

/// A draw's indirect argument reads of what its encoder made resident, added by their identity
/// (`IndirectReads::add_draw`), are walked once while the identity holds: the draws after the
/// first of that identity walk none of them, past the entries a resource keeps, and a later write
/// of what they read meets them all along; a draw of another identity walks its own, and the
/// first's walks them again after it. A new instance's reads (`IndirectReads::new`) hold none.
#[test]
fn draws_add_the_indirect_reads_of_what_their_encoder_made_resident_once_by_identity() {
    use vk::AccessFlags2 as A2;
    use vk::PipelineStageFlags2 as S2;
    let arguments = Access::new(S2::DRAW_INDIRECT, A2::INDIRECT_COMMAND_READ);
    let vertex_write = Access::new(S2::VERTEX_SHADER, A2::SHADER_STORAGE_WRITE);
    let prefix = ResidentPrefix { id: 7, len: 2 };
    let other = ResidentPrefix { id: 8, len: 2 };
    let uses = [
        reads(1, bytes(0..16), arguments),
        reads(2, bytes(0..16), shader_read()),
        reads(3, bytes(0..16), arguments),
    ];
    let write = |key| [writes(key, bytes(4..8), vertex_write)];
    let mut known = IndirectReads::new();
    assert!(known.add_draw((&uses, prefix)));
    for _ in 0..=ENTRIES_PER_KEY {
        assert!(!known.add_draw((&uses, prefix)));
    }
    assert!(known.written_in(&write(1)) == S2::VERTEX_SHADER);
    assert!(known.written_in(&write(3)) == S2::VERTEX_SHADER);
    assert!(
        known.add_draw((&uses, other)),
        "another identity walks its own"
    );
    assert!(
        known.add_draw((&uses, prefix)),
        "and the first's again after it"
    );
    let mut next = IndirectReads::new();
    assert!(next.written_in(&write(1)).is_empty());
    assert!(next.add_draw((&uses, prefix)));
    assert!(next.written_in(&write(1)) == S2::VERTEX_SHADER);
}

/// A draw's writes before its fragments of what its encoder made resident, added by their identity
/// (`GeometryWrites::add_draw`), are walked once while the identity holds: the draws after the
/// first of that identity walk none of them, past the entries a resource keeps, and a later
/// draw's fragment read of what they wrote waits for them all along; a draw of another identity
/// walks its own, and the first's walks them again after it. A new instance's writes
/// (`GeometryWrites::new`) hold none.
#[test]
fn draws_add_the_geometry_writes_of_what_their_encoder_made_resident_once_by_identity() {
    use vk::AccessFlags2 as A2;
    use vk::PipelineStageFlags2 as S2;
    let vertex_write = Access::new(S2::VERTEX_SHADER, A2::SHADER_STORAGE_WRITE);
    let fragment_read = Access::new(FRAGMENT, A2::SHADER_STORAGE_READ);
    let prefix = ResidentPrefix { id: 7, len: 2 };
    let other = ResidentPrefix { id: 8, len: 2 };
    let uses = [
        writes(1, bytes(0..16), vertex_write),
        reads(2, bytes(0..16), shader_read()),
        writes(3, bytes(0..16), vertex_write),
    ];
    let read = |key| [reads(key, bytes(4..8), fragment_read)];
    let mut known = GeometryWrites::new();
    assert!(known.add_draw((&uses, prefix)));
    for _ in 0..=ENTRIES_PER_KEY {
        assert!(!known.add_draw((&uses, prefix)));
    }
    assert_eq!(
        known.barrier_before(&read(1)),
        barrier(vertex_write, fragment_read)
    );
    assert_eq!(
        known.barrier_before(&read(3)),
        barrier(vertex_write, fragment_read)
    );
    assert!(
        known.add_draw((&uses, other)),
        "another identity walks its own"
    );
    assert!(
        known.add_draw((&uses, prefix)),
        "and the first's again after it"
    );
    let mut next = GeometryWrites::new();
    assert_eq!(next.barrier_before(&read(1)), None);
    assert!(next.add_draw((&uses, prefix)));
    assert_eq!(
        next.barrier_before(&read(1)),
        barrier(vertex_write, fragment_read)
    );
}

/// A render pass instance whose draws wrote nothing before their fragments answers no barrier
/// before a later draw's fragments, walking none of its uses (`GeometryWrites::barrier_before`,
/// counted only where it walks: `WorkCounts::resident_uses_geometry_checked`); once a draw wrote,
/// the later draw's fragment read of it waits for the write.
#[test]
fn an_instance_without_geometry_writes_answers_no_barrier_before_fragments() {
    use vk::AccessFlags2 as A2;
    use vk::PipelineStageFlags2 as S2;
    let vertex_write = Access::new(S2::VERTEX_SHADER, A2::SHADER_STORAGE_WRITE);
    let fragment_read = Access::new(FRAGMENT, A2::SHADER_STORAGE_READ);
    let mut known = GeometryWrites::new();
    let read = [reads(1, bytes(0..16), fragment_read)];
    assert!(known.is_empty());
    assert_eq!(known.barrier_before(&read), None);
    known.add(&[reads(1, bytes(0..16), shader_read())]);
    assert!(known.is_empty(), "a read before the fragments is no write");
    known.add(&[writes(1, bytes(0..16), vertex_write)]);
    assert_eq!(
        known.barrier_before(&read),
        barrier(vertex_write, fragment_read)
    );
}

/// A draw's dependency before its fragments on what earlier draws of its render pass instance
/// wrote before theirs, by the identity of the uses of what its encoder made resident
/// (`GeometryWrites::barrier_before_draw`): the first draw of an identity walks all of those,
/// a later one none, but those of a resource a draw wrote since that its fragments touch (a
/// write of one read only before the fragments walks none), and one of another identity walks
/// its own; each answers as a walk of all its uses does (`barrier_before`), merged writes
/// included.
#[test]
fn draws_check_what_their_encoder_made_resident_against_geometry_writes_once_by_identity() {
    use vk::AccessFlags2 as A2;
    use vk::PipelineStageFlags2 as S2;
    let vertex_write = Access::new(S2::VERTEX_SHADER, A2::SHADER_STORAGE_WRITE);
    let fragment_read = Access::new(FRAGMENT, A2::SHADER_STORAGE_READ);
    let vertex_read = Access::new(S2::VERTEX_SHADER, A2::SHADER_STORAGE_READ);
    let prefix = ResidentPrefix { id: 7, len: 3 };
    let other = ResidentPrefix { id: 8, len: 3 };
    let resident = [
        reads(1, bytes(8..12), fragment_read),
        reads(2, bytes(0..16), vertex_read),
        reads(3, bytes(0..16), fragment_read),
    ];
    let draw = |own: Key| -> Vec<Use<Key>> {
        let own = reads(own, bytes(0..16), fragment_read);
        resident.iter().cloned().chain([own]).collect()
    };
    let written = barrier(vertex_write, fragment_read);
    let mut known = GeometryWrites::new();
    let check = |known: &mut GeometryWrites<Key>, uses: &[Use<Key>], prefix| {
        let checked = known.barrier_before_draw((uses, prefix));
        assert_eq!(
            checked.0,
            known.barrier_before(uses),
            "the walk of all its uses"
        );
        checked
    };
    assert_eq!(
        check(&mut known, &draw(4), prefix),
        (None, 0),
        "no write yet"
    );
    known.add(&[writes(4, bytes(0..16), vertex_write)]);
    assert_eq!(
        check(&mut known, &draw(4), prefix),
        (written, 3),
        "the first walks all"
    );
    assert_eq!(
        check(&mut known, &draw(4), prefix),
        (written, 0),
        "a later one none"
    );
    assert_eq!(
        check(&mut known, &draw(5), prefix),
        (None, 0),
        "its own uses meet none"
    );
    known.add(&[writes(3, bytes(4..8), vertex_write)]);
    assert_eq!(
        check(&mut known, &draw(5), prefix),
        (written, 1),
        "the one of key 3"
    );
    assert_eq!(
        check(&mut known, &draw(5), prefix),
        (written, 0),
        "then none"
    );
    known.add(&[writes(2, bytes(0..16), vertex_write)]);
    assert_eq!(
        check(&mut known, &draw(5), prefix),
        (written, 0),
        "key 2 read before"
    );
    assert_eq!(
        check(&mut known, &draw(5), other),
        (written, 3),
        "another identity"
    );
    // Writes of key 1 apart from its resident read, merged past the entries a resource keeps
    // into one that meets it.
    let mut merged = GeometryWrites::new();
    merged.add(&[writes(4, bytes(0..16), vertex_write)]);
    assert_eq!(check(&mut merged, &draw(5), prefix), (None, 3));
    for at in 0..=ENTRIES_PER_KEY as u64 {
        let start = [0, 16][(at % 2) as usize] + 32 * at;
        merged.add(&[writes(1, bytes(start..start + 4), vertex_write)]);
        let (found, _) = check(&mut merged, &draw(5), prefix);
        let met = merged.barrier_before(&resident[..1]);
        assert_eq!(found, met, "write {at}");
    }
    assert_eq!(
        merged.barrier_before(&resident[..1]),
        written,
        "merged into one that meets"
    );
}
