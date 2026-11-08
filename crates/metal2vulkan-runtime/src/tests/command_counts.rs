//! The counts a command adds as its encoder records it (`work_counts::CommandCounters`), from
//! threads that encode at once: each encoder adds into a tally of its own while it is open
//! (`shards::Tally`), and a read of the device's counts sums every tally, so the counts of two
//! encoders recorded at once on two threads are each encoder's counts, whole, whether read while
//! the encoders are open or once they ended.
//!
//! Device tests: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. The functions are the owned fixture
//! `validation/fixtures/public/engine/tob_renders.ll` (`llvm-as`).

use super::blit_holds::{Scratch, WAIT};
use super::draw_again::{adding_state, device, rectangle, rectangles_from_columns, warm, Heard};
use super::encoder_holds::{bytes_of, pass};
use crate::{
    Buffer, CommandBufferStatus, Device, RenderCommandEncoder, RenderPipelineState,
    ResourceOptions, Texture, TextureDescriptor, WorkCounts,
};
use metal2vulkan::metal::{PixelFormat, TextureUsage};
use std::sync::Barrier;

/// The draws each encoder records: enough that two threads' draws interleave many times.
const DRAWS: u64 = 4000;

/// The command counts the encoders' draws add, by name: those that follow from an encoder's own
/// calls alone, so that each encoder of the same calls adds the same.
const COUNTS: [(&str, fn(&WorkCounts) -> u64); 12] = [
    ("draws_rebound", |counts| counts.draws_rebound),
    ("draws_moved", |counts| counts.draws_moved),
    ("draws_as_before", |counts| counts.draws_as_before),
    ("stage_tables_read", |counts| counts.stage_tables_read),
    ("stage_tables_read_in_part", |counts| {
        counts.stage_tables_read_in_part
    }),
    ("draw_states_recorded", |counts| counts.draw_states_recorded),
    ("set_zero_writes_pushed", |counts| {
        counts.set_zero_writes_pushed
    }),
    ("set_zero_sets_reused", |counts| counts.set_zero_sets_reused),
    ("argument_tables_held_at_bind", |counts| {
        counts.argument_tables_held_at_bind
    }),
    ("bound_offsets_moved", |counts| counts.bound_offsets_moved),
    ("commands_noted_as_before", |counts| {
        counts.commands_noted_as_before
    }),
    ("uses_noted_covered", |counts| counts.uses_noted_covered),
];

/// The command counts `after` holds over `before`, by name: below zero where a count read later
/// is below one read earlier, which a whole count never is.
fn moved(before: &WorkCounts, after: &WorkCounts) -> Vec<(&'static str, i128)> {
    (COUNTS.iter())
        .map(|&(name, count)| (name, i128::from(count(after)) - i128::from(count(before))))
        .collect()
}

/// A cleared `RGBA8Unorm` target of 16 by 16 texels.
fn target(device: &Device) -> Texture {
    let mut descriptor = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, 16, 16, false);
    descriptor.usage = TextureUsage::RENDER_TARGET;
    descriptor.resource_options = ResourceOptions::private();
    device.new_texture(&descriptor).unwrap()
}

/// [`DRAWS`] draws of `state`'s rectangle: the `at`th at `16 * (at % 9)` bytes of `columns`
/// (vertex buffer 0's offset set before each draw but the first), its color its fragment bytes,
/// set before each: binds that change the stage tables' slots alone, so that many of the draws
/// are drawn again rebound (`draws_rebound`).
fn draws(encoder: &mut RenderCommandEncoder, (state, columns): (&RenderPipelineState, &Buffer)) {
    encoder.set_render_pipeline_state(state).unwrap();
    encoder.set_vertex_buffer(Some(columns), 0, 0).unwrap();
    for at in 0..DRAWS {
        if at > 0 {
            encoder.set_vertex_buffer_offset(16 * (at % 9), 0).unwrap();
        }
        let color = bytes_of(&[(at % 4) as f32 / 64.0; 4], f32::to_le_bytes);
        encoder.set_fragment_bytes(&color, 0).unwrap();
        rectangle(encoder);
    }
}

/// Two threads encoding at once, each its own queue's command buffer and one pass of
/// [`draws`], read once both drew: the command counts the device holds, read while both encoders
/// are open, and once both ended, are those one thread's same pass adds, twice.
#[test]
fn counts_of_encoders_recorded_at_once_on_two_threads_are_whole() {
    let scratch = Scratch::new("command-counts");
    let heard = Heard::default();
    let device = device(&scratch.0, &[], &heard);
    let state = adding_state(&device);
    let columns = rectangles_from_columns(&device);
    let drawn = (&state, &columns);
    warm(&device, |encoder| draws(encoder, drawn));
    // One thread's pass.
    let one = {
        let queue = device.new_command_queue().unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        let target = target(&device);
        let before = device.work_counts();
        let mut encoder = command_buffer
            .render_command_encoder(&pass(&target))
            .unwrap();
        draws(&mut encoder, drawn);
        encoder.end_encoding().unwrap();
        let after = device.work_counts();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        moved(&before, &after)
    };
    assert!(
        one.iter()
            .any(|&(name, count)| name == "draws_rebound" && count >= i128::from(DRAWS / 4)),
        "the pass rebinds many of its draws: {one:?}"
    );
    let twice: Vec<(&str, i128)> = one.iter().map(|&(name, count)| (name, 2 * count)).collect();
    // Two threads' at once: each opens its encoder, then both draw, after a barrier.
    let before = device.work_counts();
    let (opened, drew, read) = (Barrier::new(3), Barrier::new(3), Barrier::new(3));
    let open_read = std::thread::scope(|scope| {
        for _ in 0..2 {
            let (device, opened, drew, read) = (&device, &opened, &drew, &read);
            scope.spawn(move || {
                let queue = device.new_command_queue().unwrap();
                let command_buffer = queue.command_buffer().unwrap();
                let target = target(device);
                let mut encoder = command_buffer
                    .render_command_encoder(&pass(&target))
                    .unwrap();
                opened.wait();
                draws(&mut encoder, drawn);
                drew.wait();
                // The main thread reads while both encoders are open.
                read.wait();
                encoder.end_encoding().unwrap();
                command_buffer.commit().unwrap();
                command_buffer.wait_until_completed_for(WAIT).unwrap();
                assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
            });
        }
        opened.wait();
        drew.wait();
        let open = device.work_counts();
        read.wait();
        open
    });
    let after = device.work_counts();
    assert_eq!(
        moved(&before, &open_read),
        twice,
        "read while both encoders were open"
    );
    assert_eq!(moved(&before, &after), twice, "read once both ended");
}
