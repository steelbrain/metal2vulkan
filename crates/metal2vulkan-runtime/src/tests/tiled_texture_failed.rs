//! A tiled texture's copy taken at use (`shadow.rs`'s **Copies taken at use**) taken by an encoder
//! that runs nothing: its command buffer fails before the encoder runs, so its take is given back
//! (`State::fail_part`, `shadow::Takes::give_back`) and the next command buffer's first use takes
//! the copy, as though the failed one had never used the texture.
//!
//! - **Committed as failed** (`CommandBuffer::commit_failed`): every encoder runs nothing, the
//!   texture's first use among them, and the next use reads the bytes in every level.
//! - **One encoder failed**: the encoder failed is the texture's first use, and an encoder before
//!   it runs; the next use takes the copy. Where the encoder that takes it runs and a later one
//!   that uses the texture fails, nothing is given back: the next use takes no copy and reads the
//!   levels as taken.
//! - **Failed at its submission**: the encoder that takes the copy wrote a no-copy buffer
//!   detached before its command buffer was first submitted (behind an event wait), so the
//!   submission runs nothing (`DetachedBacking`); the next use takes the copy.
//! - **Never committed**: the command buffer whose encoder takes the copy is dropped without a
//!   commit, so nothing of it runs (`CommandBufferShared`'s drop); the next use takes the copy.
//! - **Its `vkQueueSubmit` failed** (`Device::fail_next_submissions`): the batch of the encoder
//!   that takes the copy, behind an event wait, is refused by the driver, so it runs nothing and
//!   the command buffer fails out of memory (`CommandBufferShared::failed`); the next use takes
//!   the copy. A failed host step's batch is `tiled_texture_step.rs`'s.
//!
//! Device tests: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. Printed (`TILED-FAILED`): each leg's
//! status and the next use's copies.

use super::blit_holds::{over_host, Host, Scratch, WAIT};
use super::tiled_texture_device::{
    descriptor, near_each, one_take, read, rig, write_rows, zeroed, Rig, LENGTH, NONE, PITCH,
};
use super::tiled_texture_reencode::{behind_a_wait, blit_levels, GREY, SHAPE};
use crate::{
    Buffer, CommandBuffer, CommandBufferError, CommandBufferStatus, CommitFailure, FailedEncoders,
    ResourceOptions, Texture,
};
use std::collections::BTreeSet;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// A tiled texture of the guest's first shape over a new zeroed buffer, the grey written into
/// its bytes after the making, before any use.
pub(super) fn grey_texture(rig: &Rig) -> (Buffer, Texture) {
    let ((width, height), _) = SHAPE;
    let target = zeroed(&rig.device, LENGTH);
    let texture = target
        .new_tiled_texture(&descriptor(SHAPE), 0, PITCH)
        .unwrap();
    write_rows(&target, ((width, height), PITCH), |_, _| GREY);
    (target, texture)
}

/// A `Shared` buffer that holds every level of [`SHAPE`] packed.
fn levels_out(rig: &Rig) -> Buffer {
    (rig.device.new_buffer(16384, ResourceOptions::shared())).unwrap()
}

/// `command_buffer` committed as failed with `failure`, waited for: it completes with status
/// `Error`.
fn fail(command_buffer: &CommandBuffer, failure: CommitFailure) {
    command_buffer.commit_failed(failure).unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Error);
}

/// The next command buffer's use of `texture`, its first since the failed one: it takes the
/// copy, and every level holds the grey.
pub(super) fn next_use_takes(rig: &Rig, texture: &Texture, leg: &str) {
    let ((width, height), _) = SHAPE;
    let (levels, copies) = rig.levels(texture, SHAPE);
    eprintln!("TILED-FAILED {leg}: the next use {copies:?}");
    assert_eq!(copies, one_take(width * height), "{leg}: the next use");
    for (level, bytes) in levels.iter().enumerate() {
        let first = bytes.get(..4);
        assert!(near_each(bytes, GREY), "{leg}: level {level} {first:?}");
    }
}

#[test]
fn a_take_in_a_command_buffer_committed_as_failed_is_taken_at_the_next_use() {
    let scratch = Scratch::new("tiled-failed-commit");
    let rig = rig(&scratch);
    let (_target, texture) = grey_texture(&rig);
    let out = levels_out(&rig);
    let command_buffer = rig.queue.command_buffer().unwrap();
    blit_levels(&command_buffer, &texture, &out, SHAPE);
    fail(&command_buffer, CommitFailure::new(5));
    next_use_takes(&rig, &texture, "committed as failed");
}

#[test]
fn a_take_in_a_command_buffer_dropped_uncommitted_is_taken_at_the_next_use() {
    let scratch = Scratch::new("tiled-failed-dropped");
    let rig = rig(&scratch);
    let (_target, texture) = grey_texture(&rig);
    let out = levels_out(&rig);
    let command_buffer = rig.queue.command_buffer().unwrap();
    blit_levels(&command_buffer, &texture, &out, SHAPE);
    drop(command_buffer);
    next_use_takes(&rig, &texture, "dropped uncommitted");
}

#[test]
fn a_take_in_the_one_encoder_failed_is_taken_at_the_next_use() {
    let scratch = Scratch::new("tiled-failed-encoder");
    let rig = rig(&scratch);
    let (_target, texture) = grey_texture(&rig);
    let (filled, out) = (levels_out(&rig), levels_out(&rig));
    let command_buffer = rig.queue.command_buffer().unwrap();
    // Encoder 0 runs and uses no tiled texture; encoder 1, the texture's first use, fails.
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.fill_buffer(&filled, 0..16384, 7).unwrap();
    blit.end_encoding().unwrap();
    blit_levels(&command_buffer, &texture, &out, SHAPE);
    let failure = CommitFailure {
        code: 5,
        encoders: FailedEncoders::Set(BTreeSet::from([1])),
        signals: false,
    };
    fail(&command_buffer, failure);
    let ran = read(&filled);
    assert!(ran.iter().all(|&byte| byte == 7), "encoder 0 runs");
    next_use_takes(&rig, &texture, "one encoder failed");
}

#[test]
fn a_take_that_ran_is_kept_when_a_later_encoder_fails() {
    let scratch = Scratch::new("tiled-failed-later");
    let rig = rig(&scratch);
    let (target, texture) = grey_texture(&rig);
    let (taken, out) = (levels_out(&rig), levels_out(&rig));
    let command_buffer = rig.queue.command_buffer().unwrap();
    // Encoder 0 takes the copy and runs; encoder 1 uses the texture again and fails.
    blit_levels(&command_buffer, &texture, &taken, SHAPE);
    blit_levels(&command_buffer, &texture, &out, SHAPE);
    let failure = CommitFailure {
        code: 5,
        encoders: FailedEncoders::Set(BTreeSet::from([1])),
        signals: false,
    };
    fail(&command_buffer, failure);
    // Written after the take with no `did_modify_data`: no level sees it.
    let ((width, height), _) = SHAPE;
    write_rows(&target, ((width, height), PITCH), |_, _| [0x40; 4]);
    let (levels, copies) = rig.levels(&texture, SHAPE);
    eprintln!("TILED-FAILED a later encoder failed: the next use {copies:?}");
    assert_eq!(copies, NONE, "the next use after a take that ran");
    for (level, bytes) in levels.iter().enumerate() {
        assert!(near_each(bytes, GREY), "level {level} as taken");
    }
}

#[test]
fn a_take_in_a_submission_that_runs_nothing_is_taken_at_the_next_use() {
    // The pages, declared before the device that uses them, so dropped after it.
    let host = Host::new();
    let scratch = Scratch::new("tiled-failed-submission");
    let rig = rig(&scratch);
    let runs = Arc::new(AtomicUsize::new(0));
    let Some(out) = over_host(&rig.device, &host, &runs) else {
        return;
    };
    let (_target, texture) = grey_texture(&rig);
    let (command_buffer, gate) = behind_a_wait(&rig, |cb| blit_levels(cb, &texture, &out, SHAPE));
    // Detached before the first submission: the submission of the encoder that names it runs
    // nothing.
    assert_eq!(out.detach_backing(), Ok(true), "the detach");
    gate.set_signaled_value(1);
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    let (status, error) = (command_buffer.status(), command_buffer.error());
    eprintln!("TILED-FAILED submission: {status:?} {error:?}");
    assert_eq!(status, CommandBufferStatus::Error, "{error:?}");
    next_use_takes(&rig, &texture, "failed at its submission");
    drop((texture, out));
    drop(rig);
}

#[test]
fn a_take_in_a_batch_whose_queue_submit_fails_is_taken_at_the_next_use() {
    let scratch = Scratch::new("tiled-failed-queue-submit");
    let rig = rig(&scratch);
    let (_target, texture) = grey_texture(&rig);
    let out = levels_out(&rig);
    let (command_buffer, gate) = behind_a_wait(&rig, |cb| blit_levels(cb, &texture, &out, SHAPE));
    // The batch's `vkQueueSubmit`, made once the host signals, fails.
    rig.device.fail_next_submissions(1);
    gate.set_signaled_value(1);
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    let (status, error) = (command_buffer.status(), command_buffer.error());
    let failing = (rig.device.shared().failing_submissions).load(Ordering::Relaxed);
    eprintln!("TILED-FAILED queue submit: {status:?} {error:?}, {failing} failures left");
    assert_eq!(failing, 0, "the batch's submission failed");
    assert_eq!(status, CommandBufferStatus::Error, "{error:?}");
    assert_eq!(error, Some(CommandBufferError::OutOfMemory));
    next_use_takes(&rig, &texture, "its vkQueueSubmit failed");
}
