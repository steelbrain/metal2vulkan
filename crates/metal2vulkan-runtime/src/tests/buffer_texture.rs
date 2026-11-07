//! Texture buffers' texel buffer views (`buffer_texture`): the usage a view takes beside its
//! buffer's, by the kinds of texel buffer its format serves on the device; a staged shadow's
//! regions back into its buffer; and Metal's `bytesPerRow` rule for a texture over an IOSurface
//! plane, which no test may build the texture for.

use super::super::buffer_texture::{
    inside_rows, rotated_rows, staged_rows_back, turned_plane, view_usage, ViewUsage,
};
use super::super::error::{Error, Invalid, Problem, RuntimeGap};
use super::super::linear_plan::{
    plan_buffer_texture, plane_rows_share_bytes, BufferFacts, BufferLayout, BufferTexture,
    BufferTextureCall, BufferTextureProblem,
};
use super::super::texture_descriptor::TextureLayout;
use super::super::values::StorageMode;
use ash::vk;
use metal2vulkan::device::FormatUsage;
use metal2vulkan::metal::{PixelFormat, TextureType, TextureUsage};

/// Whether `result` is the typed gap of a view the runtime does not make.
fn not_made(result: Result<ViewUsage, Error>) -> bool {
    matches!(result, Err(Error::NotImplemented(RuntimeGap::Textures)))
}

/// A view takes its buffer's usage where its format serves both kinds of texel buffer. Where the
/// format serves only the view's own kind (NVIDIA takes `R64_UINT` for storage texel buffers
/// only), a device with `maintenance5` makes the view naming that kind alone; one without it does
/// not make the view, nor does any device make one in a format that does not serve its kind.
#[test]
fn a_view_names_the_one_kind_its_format_serves_where_maintenance5_allows() {
    let both = |_: FormatUsage| true;
    let storage_only = |usage: FormatUsage| usage == FormatUsage::StorageTexelBuffer;
    let uniform_only = |usage: FormatUsage| usage == FormatUsage::UniformTexelBuffer;
    for storage in [false, true] {
        for maintenance5 in [false, true] {
            assert_eq!(
                view_usage(both, storage, maintenance5).unwrap(),
                ViewUsage::Buffer
            );
        }
    }
    assert_eq!(
        view_usage(storage_only, true, true).unwrap(),
        ViewUsage::Only { storage: true }
    );
    assert_eq!(
        view_usage(uniform_only, false, true).unwrap(),
        ViewUsage::Only { storage: false }
    );
    assert!(not_made(view_usage(storage_only, true, false)));
    assert!(not_made(view_usage(uniform_only, false, false)));
    assert!(not_made(view_usage(storage_only, false, true)));
    assert!(not_made(view_usage(uniform_only, true, true)));
    assert!(not_made(view_usage(|_| false, true, true)));
}

/// A staged shadow's texels go back into the buffer as one region where its rows lie one after
/// another (a 1080p surface's packed rows: one region, not 1080), and a region a row where bytes
/// lie between them; each slice of packed rows joins the next where the slices lie one after
/// another too. Rows that share bytes (an IOSurface plane's on Metal's GPU) join as well: a
/// region of each would write the shared bytes twice in one `vkCmdCopyBuffer`
/// (`VUID-vkCmdCopyBuffer-pRegions-12482`).
#[test]
fn packed_rows_go_back_as_one_region() {
    let layout = |row_pitch: u64, rows: u64, slices: u64| BufferLayout {
        offset: 0,
        row_pitch,
        slice_pitch: row_pitch * rows,
        row_bytes: 7680,
        rows,
        slices,
    };
    let regions = |layout: BufferLayout, start: u64, at: u64| -> Vec<(u64, u64, u64)> {
        (staged_rows_back(&layout, start, at).iter())
            .map(|copy: &vk::BufferCopy| (copy.src_offset, copy.dst_offset, copy.size))
            .collect()
    };
    assert_eq!(
        regions(layout(7680, 1080, 1), 64, 3),
        [(64, 3, 7680 * 1080)]
    );
    assert_eq!(regions(layout(7680, 4, 2), 0, 5), [(0, 5, 7680 * 8)]);
    assert_eq!(
        regions(layout(7744, 3, 1), 0, 5),
        [
            (0, 5, 7680),
            (7744, 7749, 7680),
            (2 * 7744, 2 * 7744 + 5, 7680)
        ]
    );
    // Rows 7664 apart, 7680 bytes of texels each: consecutive rows share 16 bytes, and the three
    // go back as one region of their whole span.
    assert_eq!(regions(layout(7664, 3, 1), 0, 5), [(0, 5, 2 * 7664 + 7680)]);
}

/// No rotated texture over an IOSurface plane has GPU rows that share bytes, which no shadow of
/// a turned frame holds (`linear_plan.rs`'s **Rows that overlap**): `turned_plane` takes only a
/// `bytesPerRow` that is a multiple of 16, so the plane's GPU pitch is its own pitch and covers
/// its row. The shapes below are exactly the ones a plain plane holds as a shadow of two copy
/// regions each way, and every rotation refuses them for their pitch alone.
#[test]
fn no_rotated_plane_s_gpu_rows_share_bytes() {
    let refusal = |rotation: u8, width: u32, height: u32, bytes_per_row: u64| {
        let texture = over_a_plane(width, height, bytes_per_row);
        let length = bytes_per_row * u64::from(width.max(height)) + 1024;
        match turned_plane("t", &texture, rotation, &BufferFacts { length }) {
            // A plane whose pitch is a multiple of 16 lies at it, a row apart or further.
            Ok((_, placed)) => {
                assert_eq!(placed.gpu.shared_bytes(), 0, "rotation {rotation}");
                None
            }
            Err(error) => Some(format!("{error:?}")),
        }
    };
    let pitch = Some(format!(
        "{:?}",
        Error::NotImplemented(RuntimeGap::RotatedTexture)
    ));
    for rotation in 1..=4u8 {
        // 20 bytes of texels a row at 20 and at 28: a plain plane's rows would lie 16 apart and
        // share 4 bytes. Here the pitch itself is refused.
        assert_eq!(refusal(rotation, 5, 4, 20), pitch, "rotation {rotation}");
        assert_eq!(refusal(rotation, 5, 4, 28), pitch, "rotation {rotation}");
        assert_eq!(refusal(rotation, 4, 5, 20), pitch, "rotation {rotation}");
        // A pitch that is a multiple of 16 and covers both rows is taken, its rows a row apart.
        assert_eq!(refusal(rotation, 5, 4, 32), None, "rotation {rotation}");
    }
}

/// The only rows of a texture over a buffer that share bytes are an IOSurface plane's on Metal's
/// GPU, whose own rows lie a row apart or further
/// (`Texels::plane_rows_overlap`, `linear_plan::plane_rows_share_bytes`): no plan makes a texture
/// whose CPU rows share bytes as well, so a copy that addresses a texture's rows as bytes meets
/// that one sharing and no other.
///
/// The sweep holds `plan_buffer_texture` to it over every call, a plain format and a two-plane
/// YCbCr one (whose rows of codes are the only ones Metal lets fall below the row), pitches from
/// well below the row to well past it, and offsets on and off 16; and `turned_plane` with
/// `rotated_rows` over every rotation, whose one layout serves the GPU and the CPU alike.
#[test]
fn no_texture_over_a_buffer_has_cpu_rows_that_share_bytes() {
    let calls = [
        BufferTextureCall::Texture,
        BufferTextureCall::Linear,
        BufferTextureCall::IoSurfacePlane { planar: false },
        BufferTextureCall::IoSurfacePlane { planar: true },
    ];
    let (mut made, mut sharing) = (0, 0);
    for call in calls {
        for (pf, texel_bytes) in [(PixelFormat::BGRA8Unorm, 4), (PixelFormat::YCBCR84221P, 1)] {
            for width in [1u32, 5, 8, 17] {
                for height in [1u32, 2, 4, 9] {
                    for bytes_per_row in (0..=u64::from(width) * texel_bytes + 40).step_by(3) {
                        for offset in [0u64, 4, 32] {
                            let texture = BufferTexture {
                                layout: TextureLayout {
                                    texture_type: TextureType::Type2D,
                                    pixel_format: pf,
                                    extent: [width, height, 1],
                                    levels: 1,
                                    layers: 1,
                                    samples: 1,
                                },
                                texel_bytes,
                                storage: StorageMode::Shared,
                                offset,
                                bytes_per_row,
                                call,
                                usage: TextureUsage::SHADER_READ,
                            };
                            let facts = BufferFacts { length: 1 << 20 };
                            let Ok(placed) = plan_buffer_texture(&texture, &facts) else {
                                continue;
                            };
                            made += 1;
                            assert_eq!(
                                placed.cpu.shared_bytes(),
                                0,
                                "{call:?} {pf:?} {width}x{height} at {bytes_per_row}, offset                                  {offset}"
                            );
                            assert_eq!(
                                plane_rows_share_bytes(&placed.gpu, &placed.cpu),
                                placed.gpu.shared_bytes() > 0,
                                "{call:?} {pf:?} {width}x{height} at {bytes_per_row}, offset                                  {offset}"
                            );
                            sharing += u64::from(placed.gpu.shared_bytes() > 0);
                        }
                    }
                }
            }
        }
    }
    // The sweep reached both answers: planes whose GPU rows share bytes, and textures with none.
    assert!(
        sharing > 0 && made > sharing,
        "{made} made, {sharing} sharing"
    );
    // A rotated plane: one layout for the GPU and the CPU, whose rows share no byte.
    let (mut turned_ok, mut turned_refused) = (0, 0);
    for rotation in 1..=4u8 {
        for width in [4u32, 5, 16, 64] {
            for height in [4u32, 5, 16, 64] {
                for bytes_per_row in [16u64, 20, 32, 48, 64, 256, 272] {
                    let texture = over_a_plane(width, height, bytes_per_row);
                    let length = bytes_per_row * u64::from(width.max(height)) + 1024;
                    match turned_plane("t", &texture, rotation, &BufferFacts { length }) {
                        Ok((_, placed)) => {
                            turned_ok += 1;
                            let rows = rotated_rows(&texture);
                            assert_eq!(
                                (rows.shared_bytes(), placed.gpu.shared_bytes()),
                                (0, 0),
                                "rotation {rotation}, {width}x{height} at {bytes_per_row}"
                            );
                            assert!(
                                !plane_rows_share_bytes(&rows, &rows),
                                "rotation {rotation}, {width}x{height} at {bytes_per_row}"
                            );
                        }
                        Err(_) => turned_refused += 1,
                    }
                }
            }
        }
    }
    assert!(
        turned_ok > 0 && turned_refused > 0,
        "{turned_ok} made, {turned_refused} refused"
    );
}

/// A `width x height` texture of 4-byte texels over an IOSurface plane whose rows are
/// `bytes_per_row` bytes apart.
fn over_a_plane(width: u32, height: u32, bytes_per_row: u64) -> BufferTexture {
    BufferTexture {
        layout: TextureLayout {
            texture_type: TextureType::Type2D,
            pixel_format: PixelFormat::BGRA8Unorm,
            extent: [width, height, 1],
            levels: 1,
            layers: 1,
            samples: 1,
        },
        texel_bytes: 4,
        storage: StorageMode::Shared,
        offset: 0,
        bytes_per_row,
        call: BufferTextureCall::IoSurfacePlane { planar: false },
        usage: TextureUsage::SHADER_READ,
    }
}

/// `turned_plane`'s answer for a `width x height` texture of `rotation` over a plane whose rows
/// are `bytes_per_row` apart, in a buffer long enough for either way round.
fn turned(
    rotation: u8,
    width: u32,
    height: u32,
    bytes_per_row: u64,
) -> Result<(), BufferTextureProblem> {
    let texture = over_a_plane(width, height, bytes_per_row);
    let length = bytes_per_row * u64::from(width.max(height)) + 1024;
    match turned_plane("t", &texture, rotation, &BufferFacts { length }) {
        Ok(_) => Ok(()),
        Err(Error::Invalid(Invalid {
            problem: Problem::BufferTexture(problem),
            ..
        })) => Err(problem),
        Err(other) => panic!("rotation {rotation}, {width}x{height} at {bytes_per_row}: {other}"),
    }
}

/// Metal holds an IOSurface texture's `bytesPerRow` to the **texture's** own row, whatever the
/// plane holds and whatever the rotation: measured on the M3 Max under plain Metal with no
/// validation layer, a 4x8 plane at its tight stride of 16 behind an 8x4 texture ends the
/// process, `_mtlValidateStrideTextureParameters:1938: failed assertion 'Texture Descriptor
/// Validation IOSurface texture: bytesPerRow (16) must be greater or equal to (32) bytes'`.
///
/// A quarter turn (1, 3 and 4) stores the frame turned, so its plane's own tight row is the
/// shorter of the two and the stored extent's plan takes a pitch Metal asserts on; a half turn
/// (2) and no turn store it as it is, and were always held to the texture's row. The refusal is
/// reached through the plan from the input's own shape with no device, because the texture it
/// stands for ends Metal's process and no test may build it.
#[test]
fn an_iosurface_pitch_below_the_textures_own_row_is_refused_at_every_rotation() {
    let below = |bytes_per_row, row_bytes| {
        Err(BufferTextureProblem::RowPitch {
            bytes_per_row,
            row_bytes,
        })
    };
    for rotation in 1..=4u8 {
        // The measured shape: an 8x4 texture whose plane is 4x8 at the plane's tight 16.
        assert_eq!(
            turned(rotation, 8, 4, 16),
            below(16, 32),
            "rotation {rotation}"
        );
        // The texture's own row, and a padded one, are taken.
        assert_eq!(turned(rotation, 8, 4, 32), Ok(()), "rotation {rotation}");
        assert_eq!(turned(rotation, 8, 4, 48), Ok(()), "rotation {rotation}");
        // The rotated lane's own shape: a 64x16 texture whose quarter-turned plane is 16 blocks
        // wide, so the plane's tight row is 64 while the texture's own is 256.
        assert_eq!(
            turned(rotation, 64, 16, 64),
            below(64, 256),
            "rotation {rotation}"
        );
        assert_eq!(turned(rotation, 64, 16, 256), Ok(()), "rotation {rotation}");
        // A texture taller than wide, at a pitch covering both rows.
        assert_eq!(turned(rotation, 16, 64, 256), Ok(()), "rotation {rotation}");
    }
    // A half turn stores the frame as it is, so the plane's row and the texture's are the same
    // one and this rule adds nothing: the stored extent's plan already held the pitch to it.
    // That is why the bug was a quarter turn's alone.
    assert_eq!(turned(2, 16, 64, 64), Ok(()));
    assert_eq!(turned(2, 16, 64, 48), below(48, 64));
    // A quarter turn of the same texture stores it 64 texels wide, so its plane needs 256 bytes
    // a row for the stored frame -- which the stored extent's plan has always required. The rule
    // added here is the other direction, the one no plan held: the texture's own row where the
    // plane's is shorter, which is the 64x16 pair above.
    assert_eq!(turned(1, 16, 64, 64), below(64, 256));
}

/// A tail's copies back into its buffer move the texels' bytes inside the buffer's length alone
/// (`buffer_texture.rs`'s **Tails**): a row past them is dropped, one across them cut where they
/// end, the rows before as they were.
#[test]
fn a_tail_s_rows_back_stop_at_the_buffer_s_length() {
    let regions = |back: Vec<vk::BufferCopy>| -> Vec<(u64, u64, u64)> {
        back.iter()
            .map(|c| (c.src_offset, c.dst_offset, c.size))
            .collect()
    };
    // 64x17 RGBA8 at rows of 256 over 4096 bytes: one region of the 17 rows, cut to 16 of them.
    let whole = BufferLayout {
        offset: 0,
        row_pitch: 256,
        slice_pitch: 256 * 17,
        row_bytes: 256,
        rows: 17,
        slices: 1,
    };
    let back = staged_rows_back(&whole, 0, 0);
    assert_eq!(regions(inside_rows(back, 0, 4096)), [(0, 0, 4096)]);
    // Rows of 128 at a pitch of 256 from byte 32 of the buffer's VkBuffer, 2900 bytes inside:
    // rows 0 to 10 whole, row 11 (from 2816) cut to its first 84 bytes, the rest dropped.
    let apart = BufferLayout {
        row_bytes: 128,
        ..whole
    };
    let back = inside_rows(staged_rows_back(&apart, 0, 32), 32, 2900);
    assert_eq!(back.len(), 12);
    assert_eq!(regions(back[11..].to_vec()), [(2816, 2848, 84)]);
    assert!(back[..11].iter().all(|c| c.size == 128));
    // Every texel past the length: no row back.
    assert!(inside_rows(staged_rows_back(&whole, 0, 0), 0, 0).is_empty());
}
