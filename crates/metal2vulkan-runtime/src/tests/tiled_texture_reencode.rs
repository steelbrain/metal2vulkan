//! A tiled texture's copy taken at use (`shadow.rs`'s **Copies taken at use**) in an encoder that a
//! rebind encodes again (`command_buffer/reencode.rs`). The rebind puts the encoding again in the
//! encoder's place and its first recording is never submitted, so the encoding again takes the copy
//! where its first encoding took it, and leaves the texture's mark as it finds it.
//!
//! Each leg commits a command buffer behind a shared-event wait, which holds its first submission
//! until the host signals, as `rebind_behind_a_wait.rs` does. The command buffer's one encoder is the
//! texture's first use and writes a no-copy buffer over host pages. The leg rebinds that buffer
//! before the signal, so the rebind encodes the encoder again (`WorkCounts::encoders_reencoded`),
//! then reads what the encoder wrote into the new pages:
//!
//! - **A blit** of every level reads, in each, the bytes written after the making, and a later
//!   command buffer takes no copy.
//! - **A kernel's sample at LOD 1** reads level 1 made from the bytes.
//! - **Two rebinds** encode the encoder again twice, and each encoding takes the copy.
//! - **A `did_modify_data` after the commit** is left to the next use: the encoding again does not
//!   take that call's mark, so the next command buffer takes the copy again.
//!
//! Device tests: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. Printed (`TILED-REENCODE`): each leg's
//! counts.

use super::blit_holds::{over_host, Host, Scratch, WAIT};
use super::tiled_texture_device::{
    checker, copies_of, descriptor, level_extent, near_each, one_take, read, rig, texel_of, write,
    write_rows, zeroed, Copies, Rig, Shape, LENGTH, NONE, PITCH, SHAPES,
};
use crate::{
    BlitOption, Buffer, CommandBuffer, CommandBufferStatus, EventRef, Origin, SharedEvent, Size,
    Texture,
};
use std::sync::atomic::AtomicUsize;
use std::sync::Arc;

/// The grey written into the bytes after the making.
pub(super) const GREY: [u8; 4] = [128, 128, 128, 255];

/// The guest's compositor's first shape: every level fits in one host page.
pub(super) const SHAPE: Shape = SHAPES[0];

/// A command buffer of `rig`'s queue whose work waits for the host's signal of 1 (returned with
/// it), `encode`'s encoders after the wait, committed.
pub(super) fn behind_a_wait(
    rig: &Rig,
    encode: impl FnOnce(&CommandBuffer),
) -> (CommandBuffer, SharedEvent) {
    let gate = rig.device.new_shared_event().unwrap();
    let command_buffer = rig.queue.command_buffer().unwrap();
    (command_buffer.encode_wait_for_event(EventRef::Shared(&gate), 1)).unwrap();
    encode(&command_buffer);
    command_buffer.commit().unwrap();
    (command_buffer, gate)
}

/// Rebind `buffer` over each of `pages` in turn, then signal `gate` and wait for `command_buffer`,
/// which must complete with no error. The encoders the rebinds encoded again, none refused.
fn rebind_then_run(
    rig: &Rig,
    buffer: &Buffer,
    pages: &[&Host],
    (command_buffer, gate): (CommandBuffer, SharedEvent),
) -> u64 {
    let before = rig.device.work_counts();
    for page in pages {
        // SAFETY: each page outlives the buffer (declared before the device, dropped after it),
        // and only the buffer's commands use it while it lives.
        unsafe { buffer.rebind_bytes_no_copy(page.memory(), None, None) }.unwrap();
    }
    let after = rig.device.work_counts();
    gate.set_signaled_value(1);
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    assert_eq!(command_buffer.error(), None);
    let refused = after.reencodes_refused - before.reencodes_refused;
    assert_eq!(refused, 0, "a rebind refused to encode the encoder again");
    after.encoders_reencoded - before.encoders_reencoded
}

/// A blit of every level of `texture`, of `shape`, into `out`, each level's packed texels after
/// the one before's.
pub(super) fn blit_levels(
    command_buffer: &CommandBuffer,
    texture: &Texture,
    out: &Buffer,
    shape: Shape,
) {
    let ((width, height), levels) = shape;
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    let mut at = 0;
    for level in 0..levels {
        let (w, h) = level_extent((width, height), level);
        let size = Size::new(w, h, 1);
        let origin = Origin::new(0, 0, 0);
        (blit.copy_from_texture_to_buffer(
            texture,
            0,
            level,
            origin,
            size,
            out,
            at,
            w * 4,
            0,
            BlitOption::None,
        ))
        .unwrap();
        at += w * h * 4;
    }
    blit.end_encoding().unwrap();
}

/// Each level of `shape` in `bytes`, as [`blit_levels`] lays them out.
pub(super) fn levels_in(bytes: &[u8], shape: Shape) -> Vec<Vec<u8>> {
    let ((width, height), levels) = shape;
    let mut at = 0;
    (0..levels)
        .map(|level| {
            let (w, h) = level_extent((width, height), level);
            let length = (w * h * 4) as usize;
            at += length;
            bytes[at - length..at].to_vec()
        })
        .collect()
}

/// The leg of a blit of every level as the texture's first use, its buffer rebound over each of
/// `rebinds` before the signal: every level of the new pages holds the grey written after the
/// making, and a later command buffer takes no copy and reads it so.
fn blit_leg(name: &str, rebinds: usize) {
    // The pages, declared before the device that uses them, so dropped after it.
    let hosts: Vec<Host> = (0..=rebinds).map(|_| Host::new()).collect();
    let scratch = Scratch::new(name);
    let rig = rig(&scratch);
    let runs = Arc::new(AtomicUsize::new(0));
    let Some(out) = over_host(&rig.device, &hosts[0], &runs) else {
        return;
    };
    let ((width, height), _) = SHAPE;
    let target = zeroed(&rig.device, LENGTH);
    let texture = target
        .new_tiled_texture(&descriptor(SHAPE), 0, PITCH)
        .unwrap();
    write_rows(&target, ((width, height), PITCH), |_, _| GREY);
    let pages: Vec<&Host> = hosts[1..].iter().collect();
    let mut reencoded = 0;
    let copies = copies_of(&rig.device, || {
        let waiting = behind_a_wait(&rig, |cb| blit_levels(cb, &texture, &out, SHAPE));
        reencoded = rebind_then_run(&rig, &out, &pages, waiting);
    });
    eprintln!("TILED-REENCODE {name}: {reencoded} encoded again, {copies:?}");
    assert_eq!(reencoded, rebinds as u64, "the blit encoder encoded again");
    let got = levels_in(&read(&out), SHAPE);
    for (level, bytes) in got.iter().enumerate() {
        let first = bytes.get(..4);
        assert!(
            near_each(bytes, GREY),
            "level {level} through the encoding again: {first:?}"
        );
    }
    // Counted as each encoding records them: the first encoding's take, never submitted, and each
    // encoding again's.
    let recorded = rebinds as u64 + 1;
    let first = width * height;
    let taken = Copies {
        copies: recorded,
        texels_in: recorded * first,
        texels_back: 0,
    };
    assert_eq!(copies, taken, "a take in each encoding");
    let (later, copies) = rig.levels(&texture, SHAPE);
    assert_eq!(copies, NONE, "a use after the take");
    for (level, bytes) in later.iter().enumerate() {
        assert!(near_each(bytes, GREY), "level {level} at the next use");
    }
    drop((texture, out));
    drop(rig);
}

#[test]
fn a_blit_encoded_again_at_a_rebind_takes_the_copy_its_first_encoding_took() {
    blit_leg("tiled-reencode-blit", 1);
}

#[test]
fn a_blit_encoded_again_at_two_rebinds_takes_the_copy_each_time() {
    blit_leg("tiled-reencode-twice", 2);
}

#[test]
fn a_sample_encoded_again_at_a_rebind_reads_level_one_made_from_the_bytes() {
    let (first, second) = (Host::new(), Host::new());
    let scratch = Scratch::new("tiled-reencode-sample");
    let rig = rig(&scratch);
    let runs = Arc::new(AtomicUsize::new(0));
    let Some(out) = over_host(&rig.device, &first, &runs) else {
        return;
    };
    let ((width, height), _) = SHAPE;
    let target = zeroed(&rig.device, LENGTH);
    let texture = target
        .new_tiled_texture(&descriptor(SHAPE), 0, PITCH)
        .unwrap();
    write_rows(&target, ((width, height), PITCH), checker);
    let waiting = behind_a_wait(&rig, |cb| rig.sample(cb, &rig.level_one, (&texture, &out)));
    let reencoded = rebind_then_run(&rig, &out, &[&second], waiting);
    // A sample at LOD 1 reads level 1's averaged blue, which level 0 holds nowhere.
    let samples = Rig::samples(&out);
    eprintln!("TILED-REENCODE sample: {reencoded} encoded again, level 1 sampled {samples:?}");
    assert_eq!(reencoded, 1, "the compute encoder encoded again");
    for sample in samples {
        let [blue, _, red, alpha] = texel_of(sample);
        assert!((127..=128).contains(&blue), "blue {blue}");
        assert_eq!([red, alpha], [0x40, 0xff]);
    }
    drop((texture, out));
    drop(rig);
}

#[test]
fn did_modify_data_after_the_commit_is_left_to_the_next_use() {
    let (first, second) = (Host::new(), Host::new());
    let scratch = Scratch::new("tiled-reencode-did-modify");
    let rig = rig(&scratch);
    let runs = Arc::new(AtomicUsize::new(0));
    let Some(out) = over_host(&rig.device, &first, &runs) else {
        return;
    };
    let ((width, height), _) = SHAPE;
    let rows = ((width, height), PITCH);
    let target = zeroed(&rig.device, LENGTH);
    let texture = target
        .new_tiled_texture(&descriptor(SHAPE), 0, PITCH)
        .unwrap();
    write_rows(&target, rows, |_, _| GREY);
    let waiting = behind_a_wait(&rig, |cb| blit_levels(cb, &texture, &out, SHAPE));
    // The call comes after the encoder that takes the copy was encoded: the copy is taken again at
    // the first use encoded after it.
    write_rows(&target, rows, |_, _| [0x40; 4]);
    texture.did_modify_data();
    let reencoded = rebind_then_run(&rig, &out, &[&second], waiting);
    assert_eq!(reencoded, 1, "the blit encoder encoded again");
    // Taken as the command buffer ran, after the CPU's write.
    let got = levels_in(&read(&out), SHAPE);
    for (level, bytes) in got.iter().enumerate() {
        assert!(near_each(bytes, [0x40; 4]), "level {level} as it ran");
    }
    write(&target, |bytes| bytes.fill(0x20));
    let (next, copies) = rig.levels(&texture, SHAPE);
    eprintln!("TILED-REENCODE did_modify_data: the next use {copies:?}");
    assert_eq!(
        copies,
        one_take(width * height),
        "the next use takes the copy"
    );
    for (level, bytes) in next.iter().enumerate() {
        assert!(near_each(bytes, [0x20; 4]), "level {level} at the next use");
    }
    drop((texture, out));
    drop(rig);
}
