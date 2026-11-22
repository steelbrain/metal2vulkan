//! Range arithmetic at extremes, the residual, the resident set, and push fields' bytes.

use super::*;

const MAX: u64 = u64::MAX;

/// The contract every range meets: aligned start, at least 16 bytes, inside the padded
/// allocation, and every Metal-defined byte of the window that lies in the allocation is at
/// `residual + m` inside the range.
fn check(bind_offset: u64, length: u64, window: Window, a: u64) {
    let (r, residual) = descriptor_range(bind_offset, length, window, a, MAX)
        .unwrap_or_else(|e| panic!("{bind_offset} {length} {window:?} {a}: {e:?}"));
    let end = padded(length);
    let a4 = a.max(4);
    assert_eq!(r.offset % a4, 0, "aligned start");
    assert!(r.range >= 16, "at least 16 bytes");
    assert!(r.offset + r.range <= end, "inside the padded allocation");
    assert_eq!(
        u64::from(residual),
        bind_offset.wrapping_sub(r.offset) & 0xFFFF_FFFF,
        "residual is bind offset minus base, mod 2^32"
    );
    let (start, len) = match window {
        Window::Bounded { start, len } => (i128::from(start), i128::from(len)),
        Window::Whole => (0, i128::from(end) - i128::from(bind_offset)),
    };
    // Probe the window's first and last byte and its middle.
    for m in [start, start + len / 2, start + len - 1] {
        if len <= 0 {
            break;
        }
        let abs = i128::from(bind_offset) + m;
        if abs < 0 || abs >= i128::from(end) {
            continue;
        }
        let rel = (i128::from(residual) + m).rem_euclid(1 << 32);
        let expect = abs - i128::from(r.offset);
        assert_eq!(rel, expect.rem_euclid(1 << 32), "byte {m} of {window:?}");
        assert!(
            expect >= 0 && expect < i128::from(r.range),
            "byte {m} of {window:?} at {bind_offset} inside [{}, +{})",
            r.offset,
            r.range
        );
    }
}

#[test]
fn bounded_windows_bind_the_aligned_padded_window() {
    // A 6-byte allocation: the last byte is covered by the 16-byte padded range.
    let (r, res) = descriptor_range(0, 6, Window::Bounded { start: 5, len: 1 }, 16, MAX).unwrap();
    assert_eq!((r.offset, r.range, res), (0, 16, 0));
    // A window starting beyond the alignment block: the base moves up, the residual wraps.
    let (r, res) = descriptor_range(
        8,
        4096,
        Window::Bounded {
            start: 256,
            len: 16,
        },
        64,
        MAX,
    )
    .unwrap();
    assert_eq!(r.offset, 256);
    assert_eq!(r.range, 32, "bytes 264..280 round up to 288");
    assert_eq!(res, 8u32.wrapping_sub(256));
    // A negative start clamps to 0: the residual is the bind offset.
    let (r, res) = descriptor_range(
        100,
        4096,
        Window::Bounded {
            start: -200,
            len: 400,
        },
        64,
        MAX,
    )
    .unwrap();
    assert_eq!((r.offset, res), (0, 100));
    assert_eq!(r.range, 304);
    // Whole: from the aligned bind offset to the padded end.
    let (r, res) = descriptor_range(100, 1000, Window::Whole, 64, MAX).unwrap();
    assert_eq!((r.offset, r.range, res), (64, 1008 - 64, 36));
}

#[test]
fn windows_past_the_allocation_still_bind_sixteen_bytes() {
    // Bound past the end, or a window entirely outside: 16 bytes inside the allocation.
    for (b, len, w) in [
        (4096, 4096, Window::Whole),
        (1 << 40, 64, Window::Whole),
        (
            0,
            64,
            Window::Bounded {
                start: 1 << 20,
                len: 4,
            },
        ),
        (0, 64, Window::Bounded { start: -64, len: 8 }),
        (60, 64, Window::Bounded { start: 0, len: 4 }),
    ] {
        let (r, _) = descriptor_range(b, len, w, 16, MAX).unwrap();
        assert!(
            r.range >= 16 && r.offset + r.range <= padded(len),
            "{b} {w:?}"
        );
        check(b, len, w, 16);
    }
    assert_eq!(
        descriptor_range(0, 0, Window::Whole, 16, MAX),
        Err(RangeError::Empty)
    );
}

#[test]
fn ranges_hold_at_extremes_of_every_operand() {
    let offsets = [
        0,
        1,
        3,
        4,
        15,
        16,
        255,
        256,
        (1 << 32) - 4,
        (1 << 32) + 4,
        (1 << 40) + 12,
        MAX - 4095,
    ];
    let lengths = [1, 6, 10, 16, 17, 4096, 1 << 33, MAX - 64];
    let windows = [
        Window::Whole,
        Window::Bounded { start: 0, len: 1 },
        Window::Bounded { start: 5, len: 1 },
        Window::Bounded { start: -1, len: 2 },
        Window::Bounded {
            start: -(1 << 31),
            len: 1 << 32,
        },
        Window::Bounded {
            start: i64::MIN,
            len: MAX,
        },
        Window::Bounded {
            start: i64::MAX,
            len: MAX,
        },
        Window::Bounded {
            start: 1 << 32,
            len: 64,
        },
    ];
    for &b in &offsets {
        for &len in &lengths {
            for &w in &windows {
                for a in [1, 4, 16, 64, 256] {
                    check(b, len, w, a);
                }
            }
        }
    }
}

#[test]
fn ranges_above_the_descriptor_limit_are_refused() {
    let w = Window::Bounded {
        start: 0,
        len: 1 << 20,
    };
    assert_eq!(
        descriptor_range(0, 1 << 30, w, 16, 65536),
        Err(RangeError::TooLarge { needed: 1 << 20 })
    );
    assert!(descriptor_range(0, 1 << 30, w, 16, 1 << 20).is_ok());
    assert_eq!(
        descriptor_range(0, 1 << 30, Window::Whole, 16, u64::from(u32::MAX)),
        Ok((
            BufferRange {
                offset: 0,
                range: 1 << 30
            },
            0
        ))
    );
}

/// A planned buffer slot of `window`, under a view base alignment of `alignment` and a range limit
/// of `max_range`.
fn planned(window: Window, alignment: u64, max_range: u64) -> PlannedBuffer {
    use crate::pipeline::sym::{SlotClass, Stage};
    PlannedBuffer {
        slot: SlotRef {
            stage: Stage::Vertex,
            class: SlotClass::Buffer,
            index: 0,
        },
        window,
        form: BufferForm::Descriptor,
        access: crate::pipeline::interface::AccessSet::READ_WRITE,
        binding: Some((0, 0)),
        alignment,
        max_range,
        bound: BoundKind::Buffer {
            device_address: false,
            length_class: crate::bind::state::length_class(1 << 12),
        },
    }
}

/// A storage view of a whole buffer binds its space from 0 to the padded end at every bind offset
/// before the end, its residual the bind offset, so a draw that moves only the offset binds the
/// same descriptor; every byte from the bind offset to the end is at `residual + m`, inside the
/// range, as `descriptor_range` places it.
#[test]
fn a_storage_view_of_a_whole_buffer_binds_its_space_from_its_start() {
    const LENGTH: u64 = 4096;
    for alignment in [16, 64, 256] {
        let pb = planned(Window::Whole, alignment, MAX);
        for bind_offset in (0..LENGTH).step_by(4).chain([1, 15, LENGTH - 1]) {
            let what = format!("{alignment} {bind_offset}");
            let (range, residual) =
                planned_range(&pb, bind_offset, LENGTH).unwrap_or_else(|e| panic!("{what}: {e:?}"));
            let whole = BufferRange {
                offset: 0,
                range: LENGTH,
            };
            assert_eq!((range, u64::from(residual)), (whole, bind_offset), "{what}");
            // The byte `descriptor_range`'s range holds at `residual + m` for the Metal offset
            // `m` is the same byte of the space here.
            let (from_bind, at) =
                descriptor_range(bind_offset, LENGTH, Window::Whole, alignment, MAX).unwrap();
            for m in [0, (LENGTH - bind_offset) / 2, LENGTH - bind_offset - 1] {
                let byte = range.offset + u64::from(residual) + m;
                assert!(
                    byte < range.offset + range.range,
                    "{what} {m}: inside the range"
                );
                assert_eq!(byte, from_bind.offset + u64::from(at) + m, "{what} {m}");
            }
        }
    }
    // A buffer of 2^32 bytes, bound at its last word: the residual still fits.
    let pb = planned(Window::Whole, 16, MAX);
    let last = (1 << 32) - 4;
    let (range, residual) = planned_range(&pb, last, 1 << 32).unwrap();
    assert_eq!(
        (range.offset, range.range, residual),
        (0, 1 << 32, u32::MAX - 3)
    );
}

/// Every other slot takes `descriptor_range`'s range: a storage view's range from 0 past the
/// view's limit or past `2^32`, a bind offset at or past the padded end, a bounded window no
/// shared range holds. A limit the range from the bind offset meets binds; one it does not is
/// refused as before.
#[test]
fn other_slots_bind_the_range_from_their_bind_offset() {
    let same = |pb: &PlannedBuffer, bind_offset: u64, length: u64| {
        assert_eq!(
            planned_range(pb, bind_offset, length),
            descriptor_range(bind_offset, length, pb.window, pb.alignment, pb.max_range),
            "{:?} {:?} {bind_offset} {length}",
            pb.window,
            pb.form
        );
    };
    let storage = |max_range| planned(Window::Whole, 64, max_range);
    // Past the view's limit: the tail from the bind offset fits; the whole buffer from 0 is
    // refused as before, at the range from the bind offset.
    let limited = storage(1 << 16);
    let length = 1 << 20;
    same(&limited, length - 4096, length);
    assert!(planned_range(&limited, length - 4096, length).is_ok());
    same(&limited, 64, length);
    assert_eq!(
        planned_range(&limited, 64, length),
        Err(RangeError::TooLarge {
            needed: length - 64
        })
    );
    // At the limit exactly, from 0.
    let (range, _) = planned_range(&storage(length), 64, length).unwrap();
    assert_eq!(range.offset, 0);
    // Past 2^32 bytes: a residual and an offset of 32 bits could not state the bind offset.
    for length in [(1 << 32) + 16, 1 << 33] {
        for bind_offset in [0, 64, 1 << 31, (1 << 32) - 64, 1 << 32] {
            same(&storage(MAX), bind_offset, length);
        }
    }
    // At or past the padded end, where the range is widened down into the allocation, a whole
    // window's or a bounded one's.
    for bind_offset in [4096, 4100, 8192, MAX] {
        same(&storage(MAX), bind_offset, 4096);
        same(&storage(MAX), bind_offset, 4090);
        let bounded = Window::Bounded { start: 0, len: 16 };
        same(&planned(bounded, 64, MAX), bind_offset, 4096);
    }
}

/// The contract every shared range meets where it is given, as `descriptor_range`'s does: an
/// aligned start, at least 16 bytes, inside the padded allocation and the view's limit, and every
/// byte of the window inside the allocation at `residual + m` inside the range.
fn check_shared(bind_offset: u64, length: u64, window: Window, a: u64, max_range: u64) {
    let Some((r, residual)) = shared_range(bind_offset, length, window, a, max_range) else {
        return;
    };
    let what = format!("{bind_offset} {length} {window:?} {a} {max_range}");
    let end = padded(length);
    assert_eq!(r.offset % a.max(4), 0, "{what}: aligned start");
    assert!(r.range >= 16, "{what}: at least 16 bytes");
    assert!(
        r.offset + r.range <= end,
        "{what}: inside the padded allocation"
    );
    assert!(
        r.range <= max_range.min(1 << 32),
        "{what}: inside the view's limit and 4 GiB"
    );
    let Window::Bounded { start, len } = window else {
        panic!("{what}: a shared range of a whole window");
    };
    let (start, len) = (i128::from(start), i128::from(len));
    for m in [start, start + len / 2, start + len - 1] {
        if len <= 0 {
            break;
        }
        let abs = i128::from(bind_offset) + m;
        if abs < 0 || abs >= i128::from(end) {
            continue;
        }
        let rel = (i128::from(residual) + m).rem_euclid(1 << 32);
        let expect = abs - i128::from(r.offset);
        assert_eq!(rel, expect.rem_euclid(1 << 32), "{what}: byte {m}");
        assert!(
            expect >= 0 && expect < i128::from(r.range),
            "{what}: byte {m} inside [{}, +{})",
            r.offset,
            r.range
        );
    }
}

#[test]
fn shared_ranges_hold_every_window_byte_at_the_residual() {
    let offsets = [
        0,
        4,
        16,
        255,
        256,
        32 * 1024 - 16,
        32 * 1024,
        65520,
        65536,
        (1 << 20) - 16,
        (1 << 32) - 4,
        (1 << 32) + 4,
        MAX - 4095,
    ];
    let lengths = [1, 16, 4096, 65536, 65537, 1 << 20, 1 << 33, MAX - 64];
    let windows = [
        Window::Bounded { start: 0, len: 16 },
        Window::Bounded { start: 0, len: 1 },
        Window::Bounded { start: -8, len: 40 },
        Window::Bounded {
            start: 256,
            len: 16,
        },
        Window::Bounded {
            start: 0,
            len: 40000,
        },
        Window::Bounded {
            start: i64::MIN,
            len: MAX,
        },
        Window::Bounded {
            start: 1 << 32,
            len: 64,
        },
    ];
    for &b in &offsets {
        for &len in &lengths {
            for &w in &windows {
                for a in [4, 16, 64, 256] {
                    for max_range in [64, 65536, 1 << 27, MAX] {
                        check_shared(b, len, w, a, max_range);
                    }
                }
            }
        }
    }
}

/// A slot of a bounded window binds one range at every bind offset of a buffer that fits the
/// view's limit, from 0, its residual the bind offset. Past the limit, the bind offsets of one
/// block of half the limit share one; the next block binds the next.
#[test]
fn bounded_slots_read_at_their_residual_share_one_range_along_a_buffer() {
    const CHUNK: u64 = 64 * 1024;
    let bounded = Window::Bounded { start: 0, len: 16 };
    let limited = planned(bounded, 64, CHUNK);
    let storage = planned(bounded, 64, 1 << 27);
    let whole = BufferRange {
        offset: 0,
        range: CHUNK,
    };
    for bind_offset in (0..CHUNK).step_by(256) {
        let shared = (whole, bind_offset as u32);
        assert_eq!(
            planned_range(&limited, bind_offset, CHUNK),
            Ok(shared),
            "limited {bind_offset}"
        );
        assert_eq!(
            planned_range(&storage, bind_offset, CHUNK),
            Ok(shared),
            "storage {bind_offset}"
        );
    }
    // A buffer of 1 MiB, past the view's limit: blocks of 32 KiB.
    let length = 1 << 20;
    let block = CHUNK / 2;
    for bind_offset in (0..length).step_by(256) {
        let base = bind_offset - bind_offset % block;
        let (range, residual) = planned_range(&limited, bind_offset, length).unwrap();
        assert_eq!(
            (range.offset, range.range, u64::from(residual)),
            (base, CHUNK.min(length - base), bind_offset - base),
            "{bind_offset}"
        );
    }
    // A window never straddles its block's range: one of half the limit, starting in a block's
    // last 16 bytes, ends inside the block's range; a longer one takes no block, and binds from
    // its bind offset.
    for start in [0, block - 16, 3 * block - 16] {
        let half = Window::Bounded {
            start: 0,
            len: block,
        };
        let (range, residual) = shared_range(start, length, half, 64, CHUNK).unwrap();
        assert_eq!(range.offset, start - start % block, "{start}");
        assert!(
            u64::from(residual) + block <= range.range,
            "{start}: the window inside its block's range"
        );
        let longer = Window::Bounded {
            start: 0,
            len: block + 16,
        };
        assert_eq!(
            shared_range(start, length, longer, 64, CHUNK),
            None,
            "{start}"
        );
        let pb = planned(longer, 64, CHUNK);
        assert_eq!(
            planned_range(&pb, start, length),
            descriptor_range(start, length, longer, 64, CHUNK),
            "{start}"
        );
    }
}

fn entry(host: u64, guest: u64, size: u64) -> RootEntry {
    RootEntry {
        host,
        guest,
        size,
        bind_offset: 0,
    }
}

#[test]
fn the_root_table_follows_d76_layout() {
    let statics = [(0, entry(0x10, 0x20, 0x30)), (3, entry(0x40, 0x50, 0x60))];
    let resident = [entry(0x100, 0x200, 0x300)];
    let mut e = statics[1].1;
    e.bind_offset = 8;
    let statics = [statics[0], (3, e)];
    let t = RootTable::build(&statics, 6, &resident);
    let word = |root: u64, field: u64| {
        let at = (root * ENTRY_BYTES + field) as usize;
        u64::from_le_bytes(t.bytes[at..at + 8].try_into().unwrap())
    };
    assert_eq!(t.bytes.len(), 7 * 32);
    assert!(t.bytes[..32].iter().all(|&b| b == 0), "entry 0 is null");
    assert_eq!(word(1, HOST), 0, "the consumer fills the sink address");
    assert_eq!(RootTable::SINK_HOST_OFFSET, 32);
    assert_eq!((word(1, GUEST), word(1, SIZE)), (6, 1), "R0 and N");
    assert_eq!(
        (word(2, HOST), word(2, GUEST), word(2, SIZE)),
        (0x10, 0x20, 0x30)
    );
    assert_eq!((word(5, HOST), word(5, BIND_OFFSET)), (0x40, 8));
    assert!(t.bytes[3 * 32..5 * 32].iter().all(|&b| b == 0), "unbound");
    assert_eq!(
        (word(6, HOST), word(6, GUEST), word(6, SIZE)),
        (0x100, 0x200, 0x300)
    );
}

/// A texel-buffer view spans the whole blocks of the bound length and its element count
/// (`⌊length / block size⌋ × texels per block`) is checked exactly against
/// `maxTexelBufferElements`; an unaligned offset or a view without a whole block is not
/// representable.
#[test]
fn texel_views_count_whole_blocks_against_the_element_limit() {
    use crate::bind::plan::PlannedTexel;
    use crate::bind::state::StageTable;
    use crate::pipeline::sym::{SlotClass, Stage};
    let slot = SlotRef {
        stage: Stage::Kernel,
        class: SlotClass::Texture,
        index: 3,
    };
    let block = vk::Format::R32G32B32_SFLOAT.block().expect("block");
    let template = ResourceTemplate {
        texel_buffers: vec![PlannedTexel {
            set: 0,
            binding: 5,
            slot,
            alignment: 16,
            max_elements: 100,
            block_size: u64::from(block.size),
            texels_per_block: u64::from(block.texels),
            parts: 1,
            part_texels: 96,
        }],
        ..ResourceTemplate::default()
    };
    let state = |offset: u64, length: u64| {
        let mut t = StageTable::default();
        t.textures.insert(
            3,
            BoundTexture::Buffer {
                pixel_format: crate::metal::PixelFormat::RGBA32Float,
                offset,
                length,
                bytes_per_row: length,
                resource_id: None,
                swizzle: crate::metal::TextureSwizzleChannels::DEFAULT,
                parts: 1,
            },
        );
        let mut s = BindState::default();
        s.stages.insert(Stage::Kernel, t);
        s
    };
    let range = |offset, length| texel_ranges(&template, &state(offset, length));
    // 12-byte texels: 100 whole texels fit, the ragged tail is not part of the view.
    assert_eq!(range(32, 1200).expect("at the limit")[0].range.range, 1200);
    assert_eq!(range(32, 1210).expect("ragged")[0].range.range, 1200);
    assert_eq!(range(0, 1211).expect("ragged")[0].range.range, 1200);
    assert_eq!(
        range(0, 1212),
        Err(BindError::TexelBufferTooLarge {
            slot,
            elements: 101,
            limit: 100
        })
    );
    let not_representable = Err(BindError::Unsupported(Unsupported::not_representable(
        Feature::TexelBuffer,
    )));
    assert_eq!(range(0, 11), not_representable, "no whole texel");
    assert_eq!(range(8, 120), not_representable, "unaligned offset");
    // A compressed block counts all of its texels.
    let bc1 = vk::Format::BC1_RGB_UNORM_BLOCK.block().expect("block");
    let mut compressed = template.clone();
    compressed.texel_buffers[0].block_size = u64::from(bc1.size);
    compressed.texel_buffers[0].texels_per_block = u64::from(bc1.texels);
    assert!(texel_ranges(&compressed, &state(0, 48)).is_ok());
    assert_eq!(
        texel_ranges(&compressed, &state(0, 56)),
        Err(BindError::TexelBufferTooLarge {
            slot,
            elements: 112,
            limit: 100
        })
    );
}

/// A texture buffer bound through its parts (`bind::texel_parts`) takes one range per part at its
/// array element: part `k` from the offset plus `k · P` texels, `P` long but the last, which
/// holds the rest; past one view's limit, the same buffer bound as one view is
/// `TexelBufferTooLarge`. A format of blocks of several texels takes no parts, and texels the
/// parts do not hold are the bind state's disagreement with the plan's shape.
#[test]
fn texel_parts_view_the_limit_s_texels_each() {
    use crate::bind::plan::PlannedTexel;
    use crate::bind::state::StageTable;
    use crate::pipeline::sym::{SlotClass, Stage};
    let slot = SlotRef {
        stage: Stage::Kernel,
        class: SlotClass::Texture,
        index: 2,
    };
    // `R32_UINT` texels, 100 to a view, parts of 96 (100 rounded down to the 16-byte alignment's
    // 16 texels).
    let planned = |parts| ResourceTemplate {
        texel_buffers: vec![PlannedTexel {
            set: 0,
            binding: 4,
            slot,
            alignment: 16,
            max_elements: 100,
            block_size: 4,
            texels_per_block: 1,
            parts,
            part_texels: 96,
        }],
        ..ResourceTemplate::default()
    };
    let state = |length: u64| {
        let mut t = StageTable::default();
        t.textures.insert(
            2,
            BoundTexture::Buffer {
                pixel_format: crate::metal::PixelFormat::R32Uint,
                offset: 64,
                length,
                bytes_per_row: length,
                resource_id: None,
                swizzle: crate::metal::TextureSwizzleChannels::DEFAULT,
                parts: 2,
            },
        );
        let mut s = BindState::default();
        s.stages.insert(Stage::Kernel, t);
        s
    };
    let at = |element, offset, range| DescriptorRange {
        set: 0,
        binding: 4,
        element,
        range: BufferRange { offset, range },
    };
    // 150 texels: 96 in part 0, 54 in part 1 from texel 96.
    assert_eq!(
        texel_ranges(&planned(2), &state(600)),
        Ok(vec![at(0, 64, 384), at(1, 64 + 384, 216)])
    );
    // 192 texels: two full parts; a ragged tail past the last whole texel is no texel.
    assert_eq!(
        texel_ranges(&planned(2), &state(770)),
        Ok(vec![at(0, 64, 384), at(1, 64 + 384, 384)])
    );
    // 97 texels: one past the first part, the last holding one.
    assert_eq!(
        texel_ranges(&planned(2), &state(388)),
        Ok(vec![at(0, 64, 384), at(1, 64 + 384, 4)])
    );
    // The same buffer as one view: past the limit.
    assert_eq!(
        texel_ranges(&planned(1), &state(600)),
        Err(BindError::TexelBufferTooLarge {
            slot,
            elements: 150,
            limit: 100
        })
    );
    // Blocks of two texels take no parts.
    let mut blocks = planned(2);
    blocks.texel_buffers[0].texels_per_block = 2;
    assert_eq!(
        texel_ranges(&blocks, &state(600)),
        Err(BindError::TexelBufferTooLarge {
            slot,
            elements: 300,
            limit: 100
        })
    );
    // Texels two parts do not hold, too many or too few.
    for length in [772, 384] {
        assert!(
            matches!(
                texel_ranges(&planned(2), &state(length)),
                Err(BindError::Internal(_))
            ),
            "{length} bytes"
        );
    }
}

/// A push field's bytes are the bytes it was made of, however they were made: held in place up
/// to `FieldBytes::INLINE` bytes, past it on the heap, a collect that crosses the bound included.
#[test]
fn field_bytes_hold_what_they_were_made_of() {
    for len in [0, 1, 4, 31, 32, 33, 64, 200] {
        let bytes: Vec<u8> = (0..len).map(|i| (i * 7 + 3) as u8).collect();
        let of = FieldBytes::of(&bytes);
        let collected: FieldBytes = bytes.iter().copied().collect();
        for field in [&of, &collected] {
            assert_eq!(&field[..], &bytes[..], "{len} bytes");
            let inline = matches!(field, FieldBytes::Inline { .. });
            assert_eq!(inline, len <= FieldBytes::INLINE, "{len} bytes");
        }
    }
    let words = FieldBytes::words([1, 0x0403_0201, u32::MAX]);
    assert_eq!(&words[..], &[1, 0, 0, 0, 1, 2, 3, 4, 255, 255, 255, 255]);
}
