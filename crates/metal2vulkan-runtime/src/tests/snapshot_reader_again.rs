//! A reader through snapshots kept (`render/snapshot.rs`, `render/draw_again.rs`): drawn again,
//! rebound after a bind, after another state's draw and after another reader's, and carried into
//! the next render encoder of its command buffer (`render/carried.rs`), each recording its
//! snapshot draw and its drawing draw as every step records them, so that it reads what the draws
//! before it wrote.
//!
//! The readers (`tob_read`: each fragment writes the texel of its slot-0 texture, the target's,
//! under it) blend `One` and `One`, so each doubles the channels it writes of the pixels it
//! covers: one that read a snapshot older than the draws before it would double an older texel.
//! [`Of::All`] writes every channel, [`Of::Red`] red and alpha alone. The fill (`tob_solid`)
//! writes [`FILLED`].
//!
//! Each test draws one frame of kept draws and one that takes every step (a call that keeps no
//! draw before each draw: `setTessellationFactorScale:`, which no draw reads), each into a queue
//! of its own over targets of its own, private `BGRA8Unorm` ones of `snapshot_reader_keeps`'s
//! side, loaded and stored, filled before with small texels that vary by column and target. The
//! kept frame must leave the texels the other leaves, byte for byte. On a device that takes the
//! snapshot path, each reader reads through snapshots, both frames must leave the texels worked
//! out on the CPU, and the counts moved are the ones the tests name; elsewhere each reader reads
//! its pass's copy of the color (`texture/attached_reads.rs`).
//!
//! - [`ALTERNATING`]: two reader states alternating over one rectangle of one color, the fill
//!   over part of it between them. Each reader's kept draw is set aside at the other's draw and
//!   at the fill's, kept still, and rebound at its next draw, which must snapshot what the other
//!   reader and the fill wrote.
//! - [`AGAIN`], in each of three render encoders over a target of its own: the reader drawn,
//!   drawn again twice, rebound after a bind, the fill, the reader rebound after it, the other
//!   reader, and the reader rebound after it. Each encoder ends with the reader kept, the fill and
//!   the other reader set aside, and the next encoder opens from them: its first draw of each is
//!   rebound, carried, binding the snapshots of its own target.
//!
//! In builds with debug assertions the first and every second draw again or rebound since a draw
//! was kept, and the first and every second draw carried in since the command buffer's first
//! render encoder, take every step instead and check that the shortcut worked out what every step
//! works out; the tests count them so.
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. The functions are `tob_rect`,
//! `tob_solid` and `tob_read` of the owned fixture
//! `validation/fixtures/public/engine/tob_renders.ll` (`llvm-as`).

use super::blit_holds::{device, Scratch, WAIT};
use super::encoder_holds::bytes_of;
use super::snapshot_reader_keeps::{
    corners, download, settled, state_into, target, upload, FORMAT, SIDE,
};
use crate::framebuffer_fetch::through_storage;
use crate::{
    Device, LoadAction, RenderPassDescriptor, RenderPipelineState, StoreAction, WorkCounts,
};
use metal2vulkan::metal::{BlendFactor, ColorWriteMask, PrimitiveType};
use metal2vulkan::pipeline::ColorAttachmentDescriptor;

/// The rectangles the draws cover, `[x0, y0, x1, y1]`: the readers' two, overlapping, and the
/// fill's, over part of each.
const FIRST: [u64; 4] = [100, 100, 200, 200];
const SECOND: [u64; 4] = [150, 150, 250, 250];
const FILL: [u64; 4] = [180, 120, 300, 220];

/// The fill's texel, in `BGRA8Unorm`'s byte order.
const FILLED: [u8; 4] = [7, 3, 5, 255];

/// A pipeline state of the tests.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Of {
    /// The reader of every channel.
    All,
    /// The reader of red and alpha.
    Red,
    /// The fill.
    Fill,
}

/// A draw: the pipeline state it sets, where it sets one, and the rectangle whose corners it binds
/// as its vertex bytes, where it binds them; else it draws through those the draw before it drew.
#[derive(Clone, Copy)]
struct Draw(Option<Of>, Option<[u64; 4]>);

/// The alternating test's one encoder (the module docs).
const ALTERNATING: [Draw; 8] = [
    Draw(Some(Of::All), Some(FIRST)),
    Draw(Some(Of::Red), Some(FIRST)),
    Draw(Some(Of::Fill), Some(FILL)),
    Draw(Some(Of::All), Some(FIRST)),
    Draw(Some(Of::Red), Some(FIRST)),
    Draw(Some(Of::Fill), Some(FILL)),
    Draw(Some(Of::All), Some(FIRST)),
    Draw(Some(Of::Red), Some(FIRST)),
];

/// The draw-again test's encoder, each of three (the module docs).
const AGAIN: [Draw; 8] = [
    Draw(Some(Of::All), Some(FIRST)),
    Draw(None, None),
    Draw(None, None),
    Draw(None, Some(SECOND)),
    Draw(Some(Of::Fill), Some(FILL)),
    Draw(Some(Of::All), Some(SECOND)),
    Draw(Some(Of::Red), Some(FIRST)),
    Draw(Some(Of::All), Some(FIRST)),
];

/// The tests' pipeline states.
struct States {
    all: RenderPipelineState,
    red: RenderPipelineState,
    fill: RenderPipelineState,
}

impl States {
    fn new(device: &Device) -> States {
        let reader = |mask: ColorWriteMask| {
            let mut color = ColorAttachmentDescriptor::new(FORMAT);
            color.blending_enabled = true;
            color.destination_rgb_blend_factor = BlendFactor::One;
            color.destination_alpha_blend_factor = BlendFactor::One;
            color.write_mask = mask;
            state_into(device, "tob_read", color)
        };
        States {
            all: reader(ColorWriteMask::ALL),
            red: reader(ColorWriteMask::RED | ColorWriteMask::ALPHA),
            fill: state_into(device, "tob_solid", ColorAttachmentDescriptor::new(FORMAT)),
        }
    }

    fn of(&self, of: Of) -> &RenderPipelineState {
        match of {
            Of::All => &self.all,
            Of::Red => &self.red,
            Of::Fill => &self.fill,
        }
    }
}

/// The texels of the `encoder`th target before its pass: blue 1 or 2 by column, green one more
/// than `encoder`, red 1, alpha 255.
fn before(encoder: usize) -> Vec<u8> {
    let green = 1 + encoder as u8;
    (0..SIDE * SIDE)
        .flat_map(|at| [1 + (at % SIDE % 2) as u8, green, 1, 255])
        .collect()
}

/// `texels` after `draws` (the module docs).
fn after(mut texels: Vec<u8>, draws: &[Draw]) -> Vec<u8> {
    let (mut of, mut rect) = (None, None);
    for &Draw(set, bound) in draws {
        of = set.or(of);
        rect = bound.or(rect);
        let (Some(of), Some(rect)) = (of, rect) else {
            unreachable!("a draw before a state and corners")
        };
        let [x0, y0, x1, y1] = rect;
        for at in (y0..y1).flat_map(|y| (x0..x1).map(move |x| (y * SIDE + x) as usize * 4)) {
            let texel = &mut texels[at..at + 4];
            match of {
                Of::Fill => texel.copy_from_slice(&FILLED),
                Of::All => texel.iter_mut().for_each(|c| *c = c.saturating_mul(2)),
                // `BGRA8Unorm`'s red is its third byte, alpha its fourth.
                Of::Red => texel[2..].iter_mut().for_each(|c| *c = c.saturating_mul(2)),
            }
        }
    }
    texels
}

/// The frame of `encoders`, a render encoder over a new target each, of one command buffer, each
/// draw after a call that keeps no draw where `apart`: the device's counts before and after it,
/// and each target's texels after it. Into a queue of its own, which no earlier frame's kept draw
/// is carried along (`render/carried.rs`).
fn frame(
    device: &Device,
    states: &States,
    encoders: &[&[Draw]],
    apart: bool,
) -> ((WorkCounts, WorkCounts), Vec<Vec<u8>>) {
    let queue = &device.new_command_queue().unwrap();
    let targets: Vec<_> = (0..encoders.len())
        .map(|encoder| {
            let target = target(device);
            upload(queue, &target, &before(encoder));
            target
        })
        .collect();
    let filled = FILLED.map(|c| f32::from(c) / 255.0);
    // The fill's color, `RGBA`.
    let rgba = bytes_of(
        &[filled[2], filled[1], filled[0], filled[3]],
        f32::to_le_bytes,
    );
    let before = device.work_counts();
    let command_buffer = queue.command_buffer().unwrap();
    for (target, draws) in targets.iter().zip(encoders) {
        let mut pass = RenderPassDescriptor::new();
        let color = &mut pass.color_attachments[0];
        color.texture = Some(target.clone());
        color.load_action = LoadAction::Load;
        color.store_action = StoreAction::Store;
        let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
        encoder.set_fragment_texture(Some(target), 0).unwrap();
        encoder.set_fragment_bytes(&rgba, 0).unwrap();
        for &Draw(of, rect) in draws.iter() {
            if let Some(of) = of {
                encoder.set_render_pipeline_state(states.of(of)).unwrap();
            }
            if let Some(rect) = rect {
                encoder.set_vertex_bytes(&corners(rect), 0).unwrap();
            }
            if apart {
                encoder.set_tessellation_factor_scale(1.0);
            }
            (encoder.draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)).unwrap();
        }
        encoder.end_encoding().unwrap();
    }
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
    let after = device.work_counts();
    let texels = targets.iter().map(|target| download(queue, target));
    ((before, after), texels.collect())
}

/// What a frame moved of the counts between `before` and `after`.
#[derive(Debug, PartialEq, Eq, Default)]
struct Keeps {
    /// `WorkCounts::draws_none_kept`, `draws_as_before`, `draws_rebound`,
    /// `draws_rebound_carried`, `kept_draws_carried`, `kept_draws_carried_aside`,
    /// `reading_draws_snapshotted`, `reading_draws_snapshotted_again`,
    /// `reading_draws_snapshotted_stopped`, `draws_rebind_stopped`, `draws_carry_stopped` and
    /// `draws_unkept`.
    none_kept: u64,
    as_before: u64,
    rebound: u64,
    rebound_carried: u64,
    carried: u64,
    carried_aside: u64,
    snapshotted: u64,
    again: u64,
    stopped: u64,
    rebind_stopped: u64,
    carry_stopped: u64,
    unkept: u64,
}

fn moved((before, after): &(WorkCounts, WorkCounts)) -> Keeps {
    let delta = |count: fn(&WorkCounts) -> u64| count(after) - count(before);
    Keeps {
        none_kept: delta(|c| c.draws_none_kept),
        as_before: delta(|c| c.draws_as_before),
        rebound: delta(|c| c.draws_rebound),
        rebound_carried: delta(|c| c.draws_rebound_carried),
        carried: delta(|c| c.kept_draws_carried),
        carried_aside: delta(|c| c.kept_draws_carried_aside),
        snapshotted: delta(|c| c.reading_draws_snapshotted),
        again: delta(|c| c.reading_draws_snapshotted_again),
        stopped: delta(|c| c.reading_draws_snapshotted_stopped),
        rebind_stopped: delta(|c| c.draws_rebind_stopped),
        carry_stopped: delta(|c| c.draws_carry_stopped),
        unkept: delta(|c| c.draws_unkept),
    }
}

/// Draw `encoders` kept and apart (the module docs), after a frame that makes every pipeline
/// they take, and check the kept frame's texels against the apart one's; where the device takes
/// the snapshot path, both against the CPU's, the kept frame's counts against `kept` and the
/// apart one's against `apart`.
fn check(test: &str, encoders: &[&[Draw]], (kept, apart): (Keeps, Keeps)) {
    let scratch = Scratch::new(test);
    let device = device(&scratch.0);
    let states = States::new(&device);
    frame(&device, &states, encoders, false);
    settled(&device);
    let (kept_counts, kept_texels) = frame(&device, &states, encoders, false);
    let (apart_counts, apart_texels) = frame(&device, &states, encoders, true);
    let snapshots = device.shared().color_write_masks.is_some()
        && through_storage(device.compile_device().profile());
    for (encoder, draws) in encoders.iter().enumerate() {
        let (kept, apart) = (&kept_texels[encoder], &apart_texels[encoder]);
        let want = after(before(encoder), draws);
        let mut checks = vec![("kept", kept, ("apart frame's", apart))];
        // Elsewhere each reader reads its pass's copy of the color (`texture/attached_reads.rs`),
        // which the CPU does not work out: the frames must match each other alone.
        if snapshots {
            checks.push(("apart", apart, ("CPU's", &want)));
        }
        for (frame, texels, (of, want)) in checks {
            if let Some(at) = (0..texels.len()).find(|&at| texels[at] != want[at]) {
                let (x, y) = ((at / 4) as u64 % SIDE, (at / 4) as u64 / SIDE);
                panic!(
                    "{test}, {frame} frame, encoder {encoder}: pixel ({x}, {y}) holds {:?}, the \
                     {of} {:?} (BGRA)",
                    &texels[at / 4 * 4..at / 4 * 4 + 4],
                    &want[at / 4 * 4..at / 4 * 4 + 4],
                );
            }
        }
    }
    let (kept_moved, apart_moved) = (moved(&kept_counts), moved(&apart_counts));
    eprintln!(
        "SNAPSHOT-READER-AGAIN {test} {}: snapshot path {snapshots}; kept {kept_moved:?}; apart \
         {apart_moved:?}",
        device.name()
    );
    if !snapshots {
        assert_eq!(kept_moved.snapshotted, 0, "the readers through snapshots");
        return;
    }
    assert_eq!(
        kept_moved,
        kept,
        "{test}: the kept frame's counts on {}",
        device.name()
    );
    assert_eq!(
        apart_moved,
        apart,
        "{test}: the apart frame's counts on {}",
        device.name()
    );
}

#[test]
fn two_readers_alternating_with_a_fill_between_read_what_the_draws_before_them_wrote() {
    // The first draw of each state finds nothing kept; each later one is rebound, but in builds
    // with debug assertions the first and every second one since a draw was kept, which take
    // every step: the first reader's second draw, the fill's second and the second reader's
    // third.
    let debug = cfg!(debug_assertions);
    let kept = Keeps {
        none_kept: 3,
        rebound: if debug { 2 } else { 5 },
        snapshotted: 6,
        again: if debug { 2 } else { 4 },
        ..Keeps::default()
    };
    let apart = Keeps {
        none_kept: 8,
        snapshotted: 6,
        ..Keeps::default()
    };
    check(
        "snapshot-reader-alternating",
        &[&ALTERNATING],
        (kept, apart),
    );
}

#[test]
fn a_reader_drawn_again_rebound_and_carried_reads_what_the_draws_before_it_wrote() {
    // The first encoder's first draw of each state finds nothing kept. Each later encoder opens
    // from the kept reader, with the fill and the other reader set aside; its first draw of each
    // is rebound, carried. In builds with debug assertions, of the draws again and rebound of
    // each encoder, the first and every second one since a draw was kept take every step, and of
    // the draws carried, the first and every second one since the command buffer's first
    // encoder: drawn again 1 of 2 in each encoder; rebound 0 of 3 in the first, 2 of 6 in the
    // second (the fill carried and the reader after it), 3 of 6 in the third (the reader carried,
    // the reader after the fill and the other reader carried).
    let debug = cfg!(debug_assertions);
    let kept = Keeps {
        none_kept: 3,
        as_before: if debug { 3 } else { 6 },
        rebound: if debug { 5 } else { 15 },
        rebound_carried: if debug { 3 } else { 6 },
        carried: 2,
        carried_aside: 4,
        snapshotted: 21,
        again: if debug { 7 } else { 19 },
        ..Keeps::default()
    };
    // Each encoder opens from the reader kept at its last draw, which the call before its first
    // draw lets go, with the others kept under an older epoch.
    let apart = Keeps {
        none_kept: 24,
        carried: 2,
        snapshotted: 21,
        ..Keeps::default()
    };
    check(
        "snapshot-reader-again",
        &[&AGAIN, &AGAIN, &AGAIN],
        (kept, apart),
    );
}
