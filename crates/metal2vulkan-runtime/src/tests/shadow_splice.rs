//! A copy back's cut for its splice (`shadow_splice.rs`): run as the GPU and the CPU would, over a
//! map of host bytes, it leaves every texel's bytes in its place and no other byte written, and no
//! copy the GPU makes into the buffer writes a byte whose line holds a byte outside its row's
//! texels.

use crate::shadow_splice::{image, staged, Cut};
use ash::vk;
use std::collections::BTreeMap;

const LINE: u64 = 64;

/// Byte `k` of the texel at `(x, y)` of layer `layer`.
fn value(x: i64, y: i64, layer: u32, k: u64) -> u8 {
    (x * 31 + y * 17 + i64::from(layer) * 7 + k as i64) as u8
}

/// A region of `width` x `height` texels at `(x, y)` of `layer`, from byte `offset` of the
/// buffer, rows `row` texels apart.
fn region(
    offset: u64,
    row: u32,
    (x, y): (i32, i32),
    (width, height): (u32, u32),
    layer: u32,
) -> vk::BufferImageCopy {
    vk::BufferImageCopy {
        buffer_offset: offset,
        buffer_row_length: row,
        buffer_image_height: 0,
        image_subresource: vk::ImageSubresourceLayers {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            mip_level: 0,
            base_array_layer: layer,
            layer_count: 1,
        },
        image_offset: vk::Offset3D { x, y, z: 0 },
        image_extent: vk::Extent3D {
            width,
            height,
            depth: 1,
        },
    }
}

/// The host bytes a copy of `regions` writes, by host address: each texel's at its place.
fn texels(regions: &[vk::BufferImageCopy], texel: u64, base: u64) -> BTreeMap<u64, u8> {
    let mut bytes = BTreeMap::new();
    for r in regions {
        let row = u64::from(r.buffer_row_length.max(r.image_extent.width)) * texel;
        for y in 0..u64::from(r.image_extent.height) {
            for x in 0..u64::from(r.image_extent.width) {
                for k in 0..texel {
                    let at = base + r.buffer_offset + y * row + x * texel + k;
                    let (ix, iy) = (
                        i64::from(r.image_offset.x) + x as i64,
                        i64::from(r.image_offset.y) + y as i64,
                    );
                    let layer = r.image_subresource.base_array_layer;
                    bytes.insert(at, value(ix, iy, layer, k));
                }
            }
        }
    }
    bytes
}

/// Run `cut` as the GPU and the CPU would: the whole copies into the host bytes, each checked to
/// write only lines its row's texels fill (`rows`, each row's host bytes), the edges into an
/// arena, and the pieces from it.
fn run(
    cut: &Cut<vk::BufferImageCopy>,
    texel: u64,
    base: u64,
    rows: &[(u64, u64)],
) -> BTreeMap<u64, u8> {
    let mut host = texels(&cut.whole, texel, base);
    for &at in host.keys() {
        let line = at - at % LINE;
        assert!(
            rows.iter()
                .any(|&(start, end)| start <= line && line + LINE <= end),
            "the GPU writes byte {at:#x}, whose line holds a byte outside its row's texels"
        );
    }
    let mut arena = vec![None; cut.length as usize];
    for (&at, &byte) in &texels(&cut.edges, texel, 0) {
        assert!(
            arena[at as usize].is_none(),
            "two edges write arena byte {at}"
        );
        arena[at as usize] = Some(byte);
    }
    for &(at, length, from) in &cut.pieces {
        for i in 0..length {
            let byte =
                arena[(from + i) as usize].expect("a piece reads an arena byte no edge wrote");
            assert!(
                host.insert(at + i, byte).is_none(),
                "byte {:#x} written twice",
                at + i
            );
        }
    }
    host
}

/// Each row's host bytes of `regions`.
fn rows_of(regions: &[vk::BufferImageCopy], texel: u64, base: u64) -> Vec<(u64, u64)> {
    let mut rows = Vec::new();
    for r in regions {
        let row = u64::from(r.buffer_row_length.max(r.image_extent.width)) * texel;
        for y in 0..u64::from(r.image_extent.height) {
            let start = base + r.buffer_offset + y * row;
            rows.push((start, start + u64::from(r.image_extent.width) * texel));
        }
    }
    rows
}

#[test]
fn a_guest_render_target_s_rows_splice_at_their_last_line() {
    // 34x34 `RGBA16Float` at rows 512 bytes apart from a line: 32 texels (four lines) a row
    // straight in, the last two (16 bytes of the fifth line) through the arena.
    let regions = [region(0, 64, (0, 0), (34, 34), 0)];
    let cut = image(&regions, 8, 0x10_0000, LINE);
    assert_eq!(cut.whole.len(), 1);
    assert_eq!(cut.whole[0].image_extent.width, 32);
    assert_eq!(cut.whole[0].image_extent.height, 34);
    assert_eq!(cut.edges.len(), 1);
    assert_eq!(cut.edges[0].image_offset.x, 32);
    assert_eq!(cut.pieces.len(), 34);
    assert!(cut.pieces.iter().all(|&(_, length, _)| length == 16));
    // A 1920x1080 one whose rows end on lines: nothing through the arena.
    let wide = [region(0, 1920, (0, 0), (1920, 1080), 0)];
    let cut = image(&wide, 8, 0x10_0000, LINE);
    assert!(cut.pieces.is_empty() && cut.edges.is_empty());
}

#[test]
fn every_cut_writes_each_texel_once_and_no_shared_line_from_the_gpu() {
    for texel in [1, 2, 4, 8, 16] {
        for width in [1, 2, 3, 5, 7, 8, 13, 31, 33, 64, 65, 100] {
            for extra in [0, 3, 8, 64] {
                for phase in [0, 4, 16, 48, 60] {
                    for (x, y) in [(0, 0), (3, 1)] {
                        let row = width + extra + x as u32;
                        let regions = [
                            region(16 * texel, row, (x, y), (width, 3), 0),
                            region(
                                16 * texel + u64::from(row) * 4 * texel,
                                row,
                                (x, y),
                                (width, 2),
                                1,
                            ),
                        ];
                        let base = 0x10_0000 + phase;
                        let cut = image(&regions, texel, base, LINE);
                        let rows = rows_of(&regions, texel, base);
                        let ran = run(&cut, texel, base, &rows);
                        assert_eq!(
                            ran,
                            texels(&regions, texel, base),
                            "texel {texel}, width {width}, extra {extra}, phase {phase}"
                        );
                        assert!(cut
                            .edges
                            .iter()
                            .all(|e| e.buffer_offset.is_multiple_of(texel)));
                    }
                }
            }
        }
    }
}

#[test]
fn a_staged_cut_writes_each_byte_once_and_no_shared_line_from_the_gpu() {
    for size in [1, 15, 16, 63, 64, 65, 200, 1000] {
        for phase in [0, 1, 32, 63] {
            let back = [
                vk::BufferCopy {
                    src_offset: 0,
                    dst_offset: 100,
                    size,
                },
                vk::BufferCopy {
                    src_offset: size,
                    dst_offset: 100 + size + 37,
                    size,
                },
            ];
            let base = 0x10_0000 + phase;
            let cut = staged(&back, base, LINE);
            let source = |at: u64| (at * 13 + 5) as u8;
            let mut host = BTreeMap::new();
            for copy in &cut.whole {
                for i in 0..copy.size {
                    let at = base + copy.dst_offset + i;
                    let line = at - at % LINE;
                    assert!(
                        back.iter().any(|b| {
                            let (start, end) = (base + b.dst_offset, base + b.dst_offset + b.size);
                            start <= line && line + LINE <= end
                        }),
                        "size {size}, phase {phase}: the GPU writes a shared line"
                    );
                    host.insert(at, source(copy.src_offset + i));
                }
            }
            let mut arena = vec![None; cut.length as usize];
            for copy in &cut.edges {
                for i in 0..copy.size {
                    arena[(copy.dst_offset + i) as usize] = Some(source(copy.src_offset + i));
                }
            }
            for &(at, length, from) in &cut.pieces {
                for i in 0..length {
                    let byte = arena[(from + i) as usize].unwrap();
                    assert!(host.insert(at + i, byte).is_none());
                }
            }
            let mut want = BTreeMap::new();
            for b in &back {
                for i in 0..b.size {
                    want.insert(base + b.dst_offset + i, source(b.src_offset + i));
                }
            }
            assert_eq!(host, want, "size {size}, phase {phase}");
        }
    }
}
