//! A texture's faces (`texture/faces.rs`) turned by passes whose render area is short of the
//! texture (`render/face_swap.rs`, **A render area short of the texture**): each pass renders
//! [`AREA`] of the `64 x 16` texture from the origin (`renderTargetWidth`, `renderTargetHeight`),
//! clears it and reads, through `attached_read_faces`' [`RECT`] inside it, the texel under each
//! pixel from before the pass, writing it back. Metal's clear reaches the area rounded up to whole
//! 32-pixel tiles ([`REACH`], `clear_reach.rs`). So after each pass the texels outside [`REACH`]
//! hold what the last clear of the whole texture wrote, those inside [`RECT`] what the solid draw
//! after it wrote, and the rest of the reach the pass's clear: Metal's answer, and the copy's (the
//! copy in ahead of the pass, as a pass beside an open command buffer takes it).
//!
//! The first pass that reads makes the faces and copies in. A clear of the whole texture then
//! turns them, so the face presented holds its value outside the reach and the other the value
//! from before. The next pass turns them back, and the face it turns to takes the texels outside
//! the reach from the face presented before (`WorkCounts::attached_read_swap_area_texels`): every
//! one of them, since nothing stood between the faces; without them it would read the value from
//! before the clear there. Each later turn takes none: the turned pass and its clear change the
//! reach alone, so the faces share every texel outside it as the command buffer ends
//! (`shadow_standing.rs`). The same passes beside an open command buffer turn nothing and copy in
//! (`attached_read_unswapped_busy`), and read the same bytes.
//!
//! Device tests: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects.

use super::attached_read_faces::{texture, Draw, Moved, Tools, HEIGHT, RECT, WIDTH};
use super::blit_holds::{device, Scratch};
use crate::{Texture, WorkCounts};

/// The passes' render area, from the origin: `[0, 32) x [0, 12)`, which holds [`RECT`].
const AREA: [u64; 2] = [32, 12];

/// What a pass's clear over [`AREA`] reaches, from the origin: the area rounded up to whole tiles
/// of 32 pixels a side, cut to the texture.
const REACH: [u64; 2] = [32, HEIGHT];

/// The texels outside [`REACH`]: the `32 x 16` half of the texture beside it.
const OUTSIDE: u64 = WIDTH * HEIGHT - REACH[0] * REACH[1];

/// The texture's bytes where every texel outside [`REACH`] holds `outside`, every one inside
/// [`RECT`] `inside`, and the rest `area`, in each channel.
fn image(outside: u8, area: u8, inside: u8) -> Vec<u8> {
    let mut bytes = Vec::with_capacity((WIDTH * HEIGHT * 4) as usize);
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let rect = (RECT.y..RECT.y + RECT.height).contains(&y)
                && (RECT.x..RECT.x + RECT.width).contains(&x);
            let value = match (rect, x < REACH[0] && y < REACH[1]) {
                (true, _) => inside,
                (false, true) => area,
                (false, false) => outside,
            };
            bytes.extend([value; 4]);
        }
    }
    bytes
}

/// What a pass moved: the faces' counts, and the turns for a short area with the texels their
/// copies moved (`attached_read_swaps_short_area`, `attached_read_swap_area_texels`).
#[derive(Debug, PartialEq, Eq)]
struct Passed {
    moved: Moved,
    short: (u64, u64),
}

/// A pass over `texture` through `tools`, cleared to `clear` and reading, beside an open command
/// buffer where `busy`: what it moved.
fn pass(tools: &Tools, texture: &Texture, clear: u8, busy: bool) -> Passed {
    let device = tools.queue.device();
    let short = |c: &WorkCounts| {
        (
            c.attached_read_swaps_short_area,
            c.attached_read_swap_area_texels,
        )
    };
    let before = short(&device.work_counts());
    let open = busy.then(|| tools.queue.command_buffer().unwrap());
    let moved = tools.pass(texture, Some(clear), Draw::Read);
    drop(open);
    let after = short(&device.work_counts());
    Passed {
        moved,
        short: (after.0 - before.0, after.1 - before.1),
    }
}

/// `texture`'s bytes through `tools`, checked against `want`: how many differ.
fn differ(tools: &Tools, texture: &Texture, want: &[u8]) -> usize {
    let (got, _) = tools.bytes(texture);
    (0..got.len()).filter(|&at| got[at] != want[at]).count()
}

/// A pass over [`AREA`] of `texture` that reads (beside an open command buffer where `busy`),
/// cleared to `clear`, after which the texels outside its reach hold `outside`: what it moved, and
/// how many bytes after it are off Metal's answer.
fn read(
    tools: &Tools,
    texture: &Texture,
    (clear, outside): (u8, u8),
    busy: bool,
) -> (Passed, usize) {
    let passed = pass(tools, texture, clear, busy);
    (passed, differ(tools, texture, &image(outside, clear, 0x40)))
}

/// The whole texture cleared to `clear`, and `0x40` drawn inside [`RECT`].
fn whole(tools: &mut Tools, texture: &Texture, clear: u8) {
    tools.area = None;
    tools.pass(texture, Some(clear), Draw::Nothing);
    tools.pass(texture, None, Draw::Solid(0x40));
    tools.area = Some(AREA);
}

/// The passes of the module docs over `texture`, the reads beside an open command buffer where
/// `busy`: what each read moved, and the bytes after it off Metal's answer.
fn lane(tools: &mut Tools, texture: &Texture, busy: bool) -> Vec<(Passed, usize)> {
    whole(tools, texture, 0x10);
    let mut reads = vec![read(tools, texture, (0x20, 0x10), busy)];
    whole(tools, texture, 0x70);
    for clear in [0x30, 0x50, 0x60] {
        reads.push(read(tools, texture, (clear, 0x70), busy));
    }
    reads
}

#[test]
fn passes_short_of_their_texture_turn_its_faces_and_copy_only_what_the_faces_do_not_share() {
    let scratch = Scratch::new("attached-read-area-faces");
    let device = device(&scratch.0);
    let mut tools = Tools::new(&device);
    let turned = texture(&device);
    let turns = lane(&mut tools, &turned, false);
    let copied = texture(&device);
    let copies = lane(&mut tools, &copied, true);
    for (at, ((turn, turn_differ), (copy, copy_differ))) in turns.iter().zip(&copies).enumerate() {
        eprintln!("ATTACHED-READ-AREA-FACES pass {at}: turned {turn:?}; copied {copy:?}");
        assert_eq!(
            (*turn_differ, *copy_differ),
            (0, 0),
            "pass {at}: bytes off Metal's answer, turned and copied"
        );
    }
    let [(first, _), (turn, _), again @ ..] = &turns[..] else {
        unreachable!("four passes");
    };
    assert_eq!(
        (first.moved.made, first.moved.first, first.short),
        (1, 1, (0, 0)),
        "the first read makes the faces"
    );
    assert_eq!(
        (turn.moved.swaps, turn.moved.reads, turn.moved.copied_in),
        (1, 1, 0),
        "the first turn reads the face from before it, with no copy in"
    );
    assert_eq!(
        turn.short,
        (1, OUTSIDE),
        "the first turn takes every texel outside the reach"
    );
    for (passed, _) in again {
        assert_eq!(
            (passed.moved.swaps, passed.moved.copied_in, passed.short),
            (1, 0, (1, 0)),
            "a later turn takes nothing: the faces share every texel outside the reach"
        );
    }
    for (passed, _) in &copies[1..] {
        assert_eq!(
            (passed.moved.swaps, passed.moved.busy, passed.short),
            (0, 1, (0, 0)),
            "beside an open command buffer, no turn: the copy in"
        );
    }
}
