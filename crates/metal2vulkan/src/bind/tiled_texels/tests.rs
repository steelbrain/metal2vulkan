use super::{typed_module, TiledLevelMove, TiledTexelsModule, SHAPE, TIGHT, TILED};
use crate::spirv::interp::{run, Buffer, Config, Order, Resources};
use crate::spirv::testing::{assert_valid, committed_profiles};

#[test]
fn the_tiled_texel_kernel_is_valid_on_every_committed_profile() {
    for (_, p) in committed_profiles().expect("profiles") {
        assert_valid(typed_module().expect("module"), &p);
    }
}

/// The Morton index of `(x, y)` in a tile of `2^tx x 2^ty` elements, bit by bit: x's bit lowest,
/// the longer side's extra bits on top.
fn morton(x: u32, y: u32, tx: u32, ty: u32) -> u32 {
    let (mut m, mut k) = (0, 0);
    for i in 0..tx.max(ty) {
        if i < tx {
            m |= ((x >> i) & 1) << k;
            k += 1;
        }
        if i < ty {
            m |= ((y >> i) & 1) << k;
            k += 1;
        }
    }
    m
}

/// The tiled buffer's byte at which element `(x, y)` of image `image` of `level` starts.
fn tiled_at(level: &TiledLevelMove, image: u32, x: u32, y: u32) -> usize {
    let [tw, th] = level.tile;
    let tile = (y / th) * level.across + x / tw;
    let inside = morton(x % tw, y % th, tw.trailing_zeros(), th.trailing_zeros());
    let element = tile * tw * th + inside;
    (image * level.chain_stride + level.tiled_first + element * level.unit) as usize
}

/// The tight buffer's byte at which element `(x, y)` of image `image` of `level` starts.
fn tight_at(level: &TiledLevelMove, image: u32, x: u32, y: u32) -> usize {
    let [w, h] = level.extent;
    (level.tight_first + ((image * h + y) * w + x) * level.unit) as usize
}

/// Every element of every image of `levels`: the level, image, x and y.
fn elements(levels: &[TiledLevelMove]) -> Vec<(usize, u32, u32, u32)> {
    let mut out = Vec::new();
    for (k, level) in levels.iter().enumerate() {
        for image in 0..level.images {
            for y in 0..level.extent[1] {
                for x in 0..level.extent[0] {
                    out.push((k, image, x, y));
                }
            }
        }
    }
    out
}

/// One move of `levels` in a grid two workgroups wide, both invocation orders alike: the tight
/// and the tiled bytes after it.
fn moved(
    levels: &[TiledLevelMove],
    into_tight: bool,
    tight: &[u8],
    tiled: &[u8],
) -> (Vec<u8>, Vec<u8>) {
    let m = typed_module().expect("module");
    let across = 2;
    let grid = across * TiledTexelsModule::LOCAL_SIZE;
    let rows = TiledTexelsModule::invocations(levels, into_tight)
        .div_ceil(grid)
        .max(1);
    let shape = TiledTexelsModule::shape(levels, into_tight, grid).expect("a shape");
    let mut out = Vec::new();
    for order in [Order::Forward, Order::Reverse] {
        let part = |slot, bytes: Vec<u8>| Buffer {
            slot: Some(slot),
            address: None,
            bytes,
        };
        let res = Resources {
            buffers: vec![
                part(TIGHT, tight.to_vec()),
                part(TILED, tiled.to_vec()),
                part(SHAPE, shape.to_vec()),
            ],
            push: Vec::new(),
            spec: Default::default(),
        };
        let cfg = Config {
            order,
            workgroups: [across, rows, 1],
            ..Config::default()
        };
        let buffers = run(&m, &cfg, res).expect("run").buffers;
        assert_eq!(buffers[2], shape, "the shape is only read");
        out.push((buffers[0].clone(), buffers[1].clone()));
    }
    assert_eq!(out[0], out[1], "order independent");
    out.swap_remove(0)
}

/// Both moves of `levels` between a tight buffer of `tight_len` bytes and a tiled one of
/// `tiled_len`, each filled with bytes that name their place: into the image, each element's
/// bytes are its tiled bytes, a byte past a level's elements 0; back, each element's tiled bytes
/// are its tight bytes, and every other tiled byte is as it was.
fn moves_match_the_layout(levels: &[TiledLevelMove], tight_len: usize, tiled_len: usize) {
    let tight: Vec<u8> = (0..tight_len).map(|k| (k * 7 + 3) as u8).collect();
    let tiled: Vec<u8> = (0..tiled_len).map(|k| (k * 13 + 5) as u8).collect();
    // Into the image.
    let (t, p) = moved(levels, true, &tight, &tiled);
    assert_eq!(p, tiled, "the tiled bytes are only read");
    let mut want = tight.clone();
    for level in levels {
        let bytes = (level.extent[0] * level.extent[1] * level.images * level.unit) as usize;
        let end = level.tight_first as usize + bytes;
        for byte in want.iter_mut().take(end.next_multiple_of(4)).skip(end) {
            *byte = 0;
        }
    }
    for (k, image, x, y) in elements(levels) {
        let level = &levels[k];
        let (from, to) = (tiled_at(level, image, x, y), tight_at(level, image, x, y));
        let n = level.unit as usize;
        want[to..to + n].copy_from_slice(&tiled[from..from + n]);
    }
    assert_eq!(t, want, "into the image");
    // And back.
    let (t, p) = moved(levels, false, &tight, &tiled);
    assert_eq!(t, tight, "the tight bytes are only read");
    let mut want = tiled.clone();
    for (k, image, x, y) in elements(levels) {
        let level = &levels[k];
        let (to, from) = (tiled_at(level, image, x, y), tight_at(level, image, x, y));
        let n = level.unit as usize;
        want[to..to + n].copy_from_slice(&tight[from..from + n]);
    }
    assert_eq!(p, want, "back");
}

/// One-byte elements: a level of 13 x 7 in tiles of 16 x 8, one of 6 x 3 in tiles of 8 x 4
/// (its tight bytes ending mid-word), and one of 3 x 1 in a tile of 4 x 1, three images each,
/// chains 512 bytes apart.
#[test]
fn one_byte_elements_move_between_the_image_and_metals_tiles() {
    let levels = [
        TiledLevelMove::new(1, [13, 7], 3, [16, 8], 1, 0, 512, 0, 128),
        TiledLevelMove::new(1, [6, 3], 3, [8, 4], 1, 128, 512, 288, 128),
        TiledLevelMove::new(1, [3, 1], 3, [4, 1], 1, 256, 512, 352, 128),
    ];
    moves_match_the_layout(&levels, 368, 1536);
}

/// Two-byte elements of a level taller than its tile is wide (x the shorter side): 5 x 19 in
/// tiles of 8 x 32, two images.
#[test]
fn two_byte_elements_of_a_tall_tile_move() {
    let levels = [TiledLevelMove::new(
        2,
        [5, 19],
        2,
        [8, 32],
        1,
        0,
        1024,
        0,
        512,
    )];
    moves_match_the_layout(&levels, 384, 2048);
}

/// Four-byte elements: a level of 70 x 3 two tiles of 64 x 4 across, and one of 35 x 1 in a
/// tile of 64 x 1, two images, chains 4096 bytes apart.
#[test]
fn four_byte_elements_across_tiles_move() {
    let levels = [
        TiledLevelMove::new(4, [70, 3], 2, [64, 4], 2, 0, 4096, 0, 2048),
        TiledLevelMove::new(4, [35, 1], 2, [64, 1], 1, 2048, 4096, 1680, 256),
    ];
    moves_match_the_layout(&levels, 1968, 8192);
}

/// Eight- and sixteen-byte elements: 9 x 6 in tiles of 16 x 8, and 3 x 3 of sixteen bytes in a
/// tile of 4 x 4, one image.
#[test]
fn wide_elements_move() {
    let eight = [TiledLevelMove::new(
        8,
        [9, 6],
        1,
        [16, 8],
        1,
        0,
        1024,
        0,
        1024,
    )];
    moves_match_the_layout(&eight, 432, 1024);
    let sixteen = [TiledLevelMove::new(
        16,
        [3, 3],
        1,
        [4, 4],
        1,
        0,
        256,
        0,
        256,
    )];
    moves_match_the_layout(&sixteen, 144, 256);
}

/// The shape takes at most [`TiledTexelsModule::MAX_LEVELS`] levels.
#[test]
fn a_shape_takes_at_most_sixteen_levels() {
    let level = TiledLevelMove::new(4, [1, 1], 1, [1, 1], 1, 0, 128, 0, 128);
    let limit = TiledTexelsModule::MAX_LEVELS as usize;
    assert!(TiledTexelsModule::shape(&vec![level; limit], true, 64).is_some());
    assert!(TiledTexelsModule::shape(&vec![level; limit + 1], true, 64).is_none());
}
