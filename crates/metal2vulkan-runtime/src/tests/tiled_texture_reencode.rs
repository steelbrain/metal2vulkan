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
//! - **A pass recorded on the GPU alone** (it executes indirect commands the GPU writes,
//!   `render/executions/gpu.rs`), whose indexed draw reads the texture and takes its indices from
//!   the buffer: the commit records the pass in a segment, and the encoding again records it again
//!   in a segment of its own, which takes the copy where the commit's took it
//!   (`EncoderCore::finish_replay`), so the draw reads the bytes.
//!
//! Device tests: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. Printed (`TILED-REENCODE`): each leg's
//! counts.

use super::blit_holds::{over_host, Host, Scratch, WAIT};
use super::encoder_holds::{library, pass, render_state};
use super::tiled_texture_device::{
    checker, copies_of, descriptor, level_extent, near_each, one_take, read, rig, texel_of, write,
    write_rows, zeroed, Copies, Rig, Shape, LENGTH, NONE, PITCH, SHAPES,
};
use super::tiled_texture_failed::grey_texture;
use crate::{
    BlitOption, Buffer, CommandBuffer, CommandBufferStatus, Device, EventRef,
    IndirectCommandBufferDescriptor, Origin, PipelineOptions, RenderPipelineState, ResourceOptions,
    SharedEvent, Size, Texture, TextureDescriptor,
};
use metal2vulkan::metal::{
    IndexType, IndirectCommandType, PixelFormat, PrimitiveType, TextureUsage,
};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
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

/// The fixture of [`a_pass_on_the_gpu_alone_encoded_again_takes_the_copy_where_its_commit_took_it`]'s
/// functions, under `validation/fixtures/public/`.
const DIRECT_ARGUMENTS: &str = "examples/icb_direct_arguments.ll";

/// The side of its target, in pixels.
const PASS_SIDE: u64 = 4;

/// A state of the fixture's `plain_vertex` and `plain_fragment` into `RGBA8Unorm` that indirect
/// command buffers' commands may inherit.
fn inherited_state(device: &Device) -> RenderPipelineState {
    let library = library(device, DIRECT_ARGUMENTS);
    let function = |name: &str| {
        let function = library.new_function_with_name(name).unwrap();
        StageSource::Function(function.function_descriptor())
    };
    let mut descriptor = RenderPipelineDescriptor::new(function("plain_vertex"));
    descriptor.fragment = Some(function("plain_fragment"));
    descriptor.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
    descriptor.support_indirect_command_buffers = true;
    (device.new_render_pipeline_state(&descriptor, &PipelineOptions::default())).unwrap()
}

#[test]
fn a_pass_on_the_gpu_alone_encoded_again_takes_the_copy_where_its_commit_took_it() {
    let (first, second) = (Host::new(), Host::new());
    let scratch = Scratch::new("tiled-reencode-gpu-pass");
    let rig = rig(&scratch);
    let runs = Arc::new(AtomicUsize::new(0));
    let Some(indices) = over_host(&rig.device, &first, &runs) else {
        return;
    };
    // The draw's indices, in the pages the buffer lies in at each encoding.
    for host in [&first, &second] {
        host.set(0, &[0, 1, 2]);
    }
    let (_bytes, texture) = grey_texture(&rig);
    let sampled = render_state(
        &rig.device,
        DIRECT_ARGUMENTS,
        "plain_vertex",
        "texture_fragment",
    );
    let inherited = inherited_state(&rig.device);
    let mut commands = IndirectCommandBufferDescriptor::new();
    commands.command_types = IndirectCommandType::DRAW;
    commands.inherit_pipeline_state = true;
    let icb = (rig.device)
        .new_indirect_command_buffer(&commands, 1, ResourceOptions::shared())
        .unwrap();
    let mut target =
        TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, PASS_SIDE, PASS_SIDE, false);
    target.usage = TextureUsage::RENDER_TARGET;
    target.resource_options = ResourceOptions::private();
    let target = rig.device.new_texture(&target).unwrap();
    let row = PASS_SIDE * 4;
    let back = (rig.device)
        .new_buffer(PASS_SIDE * row, ResourceOptions::shared())
        .unwrap();
    let forms = rig.device.work_counts().gpu_form_commands;
    let waiting = behind_a_wait(&rig, |command_buffer| {
        // The GPU writes the buffer's records: a pass that executes them is recorded on the GPU
        // alone at the commit.
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.reset_commands_in_buffer(&icb, 0..1).unwrap();
        blit.end_encoding().unwrap();
        let mut encoder = (command_buffer.render_command_encoder(&pass(&target))).unwrap();
        encoder.set_render_pipeline_state(&sampled).unwrap();
        // The texture's first use, by the draw whose indices the buffer the rebind moves holds.
        (encoder.set_fragment_texture(Some(&texture), 0)).unwrap();
        (encoder.draw_indexed_primitives(
            PrimitiveType::Triangle,
            3,
            IndexType::UInt32,
            &indices,
            0,
            1,
            0,
            0,
        ))
        .unwrap();
        encoder.set_render_pipeline_state(&inherited).unwrap();
        encoder.execute_commands_in_buffer(&icb, 0..1).unwrap();
        encoder.end_encoding().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        (blit.copy_from_texture_to_buffer(
            &target,
            0,
            0,
            Origin::new(0, 0, 0),
            Size::new(PASS_SIDE, PASS_SIDE, 1),
            &back,
            0,
            row,
            PASS_SIDE * row,
            BlitOption::None,
        ))
        .unwrap();
        blit.end_encoding().unwrap();
    });
    let reencoded = rebind_then_run(&rig, &indices, &[&second], waiting);
    let forms = rig.device.work_counts().gpu_form_commands - forms;
    let texels = read(&back);
    let first_texel = texels.get(..4);
    eprintln!(
        "TILED-REENCODE gpu pass: {reencoded} encoded again, {forms} GPU-form commands, \
         drew {first_texel:?}"
    );
    assert_eq!(reencoded, 1, "the render encoder encoded again");
    // The commit's recording and the encoding again's, each of the one command.
    assert_eq!(forms, 2, "the pass on the GPU alone, recorded twice");
    assert!(near_each(&texels, GREY), "the draw read {first_texel:?}");
    drop((texture, indices, icb));
    drop(rig);
}
