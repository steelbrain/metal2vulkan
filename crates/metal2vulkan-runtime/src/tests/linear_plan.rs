//! Textures over buffers (`linear_plan`): which pixel formats a buffer can back and their
//! alignment, where Metal lays a texture out in its buffer and which textures it refuses, as
//! probes 9 to 9f measured them on an Apple M3 Max (macOS 26.7) and an M4 (27.0.1).

use super::super::buffer_texture::{plane_copy_regions, staged_rows_back, Blocks};
use super::super::image_plan::plan_image;
use super::super::linear_plan::{
    aliases, check_linear_limits, holding, is_linear, linear_alignment, linear_lack,
    plan_buffer_texture, BufferFacts, BufferLayout, BufferRefusal, BufferTexture,
    BufferTextureCall, BufferTextureProblem, LinearImage, LinearLack, Placed, LINEAR_ALIGNMENT,
};
use super::super::texture_descriptor::{effective_usage, no_texture_alone, TextureDescriptor};
use super::super::values::StorageMode;
use super::image_plan::{committed, facts};
use super::texture_descriptor::APPLE;
use ash::vk;
use metal2vulkan::metal::format::{byte_format, direct_format};
use metal2vulkan::metal::PixelFormat as P;
use metal2vulkan::metal::TextureType;
use metal2vulkan::metal::TextureUsage;
use metal2vulkan::vk::{Format, FormatFeatureFlags2, ImageCreateFlags, ImageUsageFlags};

/// The formats for which Metal answered 16 (the linear and the texture-buffer alignment alike);
/// `ARGB8Unorm` and the two-plane extended-range formats, which are not in Metal's public header,
/// were asked on the M3 Max.
const ANSWERED: [P; 59] = [
    P::A8Unorm,
    P::R8Unorm,
    P::R8UnormSRGB,
    P::R8Snorm,
    P::R8Uint,
    P::R8Sint,
    P::R16Unorm,
    P::R16Snorm,
    P::R16Uint,
    P::R16Sint,
    P::R16Float,
    P::RG8Unorm,
    P::RG8UnormSRGB,
    P::RG8Snorm,
    P::RG8Uint,
    P::RG8Sint,
    P::B5G6R5Unorm,
    P::A1BGR5Unorm,
    P::ABGR4Unorm,
    P::BGR5A1Unorm,
    P::R32Uint,
    P::R32Sint,
    P::R32Float,
    P::RG16Unorm,
    P::RG16Snorm,
    P::RG16Uint,
    P::RG16Sint,
    P::RG16Float,
    P::RGBA8Unorm,
    P::RGBA8UnormSRGB,
    P::RGBA8Snorm,
    P::RGBA8Uint,
    P::RGBA8Sint,
    P::BGRA8Unorm,
    P::BGRA8UnormSRGB,
    P::RGB10A2Unorm,
    P::RGB10A2Uint,
    P::RG11B10Float,
    P::RGB9E5Float,
    P::BGR10A2Unorm,
    P::BGRA10XR,
    P::BGRA10XRSRGB,
    P::BGR10XR,
    P::BGR10XRSRGB,
    P::ARGB8Unorm,
    P::RGB10A82PXR10,
    P::RGB10A82PXR10SRGB,
    P::RG32Uint,
    P::RG32Sint,
    P::RG32Float,
    P::RGBA16Unorm,
    P::RGBA16Snorm,
    P::RGBA16Uint,
    P::RGBA16Sint,
    P::RGBA16Float,
    P::RGBA32Uint,
    P::RGBA32Sint,
    P::RGBA32Float,
    P::Unspecialized,
];

/// The added linear formats for which an Apple M4 under macOS 27.0.1 answered 16, the linear and
/// the texture-buffer alignment alike; it asserted on the other 18, the formats
/// macOS 27.0.1 describes none of (`pixel_format_info::describes`).
const ANSWERED_M4: [P; 62] = [
    P::R16Bfloat,
    P::BGRA4Unorm,
    P::RG16Bfloat,
    P::ARGB8UnormSRGB,
    P::RGBA16Bfloat,
    P::RGBX8Unorm,
    P::RGBX8UnormSRGB,
    P::RGBX8Snorm,
    P::RGBX8Uint,
    P::RGBX8Sint,
    P::RGBX16Uint,
    P::RGBX16Sint,
    P::RGBX16Float,
    P::RGBX32Uint,
    P::RGBX32Sint,
    P::RGBX32Float,
    P::YCBCR84221P,
    P::YCBCR104441P,
    P::YCBCR84221PSRGB,
    P::YCBCR104441PSRGB,
    P::YCBCRA84441P,
    P::YCBCR84221PREV,
    P::YCBCR104441PPQ,
    P::R10UnormX6,
    P::R10UnormX6SRGB,
    P::RG10UnormX12,
    P::RG10UnormX12SRGB,
    P::RGB10A2UnormSRGB,
    P::RGB10A2UnormPQ,
    P::R10UnormPACKED,
    P::RG10UnormPACKED,
    P::YCBCR104441PXR,
    P::R12UnormX4,
    P::R12UnormX4PQ,
    P::RG12UnormX8,
    P::R10UnormX6PQ,
    P::YCBCR104441PHLG,
    P::R10UnormX6HLG,
    P::R12UnormX4HLG,
    P::RGB10A2UnormHLG,
    P::RGB10A82P,
    P::RGB10A82PSRGB,
    P::RGB10A82PPQ,
    P::RGB10A82PHLG,
    P::BGR10A2UnormSRGB,
    P::BGR10A2UnormPQ,
    P::BGRA10UintPACKED,
    P::BGRA10XRPACKED,
    P::BGRA10XRPACKEDSRGB,
    P::BGRA10UnormPACKED,
    P::BGRA10UnormPACKEDSRGB,
    P::BGRA10UnormPACKEDHLG,
    P::BGRA10UnormPACKEDPQ,
    P::BGRA10Uint,
    P::BGRA10Unorm,
    P::BGRA10UnormSRGB,
    P::BGRA10UnormHLG,
    P::BGRA10UnormPQ,
    P::R10UintX6,
    P::R10UintPACKED,
    P::RG10UintX12,
    P::RG10UintPACKED,
];

/// The formats of [`ANSWERED_M4`] an Apple M3 Max asserted on, under macOS 26.7 and 27.0.1 alike:
/// the 16 from 630 to 645, which it describes none of, nor does either Metal level.
const ASSERTED_M3: [P; 16] = [
    P::BGRA10UintPACKED,
    P::BGRA10XRPACKED,
    P::BGRA10XRPACKEDSRGB,
    P::BGRA10UnormPACKED,
    P::BGRA10UnormPACKEDSRGB,
    P::BGRA10UnormPACKEDHLG,
    P::BGRA10UnormPACKEDPQ,
    P::BGRA10Uint,
    P::BGRA10Unorm,
    P::BGRA10UnormSRGB,
    P::BGRA10UnormHLG,
    P::BGRA10UnormPQ,
    P::R10UintX6,
    P::R10UintPACKED,
    P::RG10UintX12,
    P::RG10UintPACKED,
];

/// Metal answers 16 for exactly the formats it answered for; it asserts on every other one
/// (compressed formats, the 4:2:2 ones among them, and depth and stencil formats; on the M3 and
/// M4, `Depth24Unorm_Stencil8` and `X24_Stencil8` as unsupported formats), which the runtime
/// refuses, at each Metal level the M3 Max's (at macOS 27.0.1: of the raw values
/// from 0 to 700, 175 answered and every other one asserted, each as the runtime answers).
/// `Invalid` was not asked.
#[test]
fn buffers_back_exactly_the_formats_metal_answers_for() {
    // Probe 9 took none of the formats Metal makes no texture of alone (the 69 two-plane YCbCr
    // formats and three more): a buffer's texture of one is refused before this rule
    // (`texture_over`). It did take `R10Unorm_PACKED` and `RG10Unorm_PACKED`, which answer 16 for
    // the alignment and whose texture over a buffer Metal does make
    // (`metal::format::packed_codes`), so they stay in the sweep.
    let skipped =
        |pf: P| no_texture_alone(pf) && !matches!(pf, P::R10UnormPACKED | P::RG10UnormPACKED);
    let unprobed = P::ALL.iter().copied().filter(|&pf| skipped(pf)).count();
    assert_eq!(unprobed, 72);
    for &pf in P::ALL {
        if pf == P::Invalid || skipped(pf) {
            continue;
        }
        let asserted = ASSERTED_M3.contains(&pf);
        let want = if ANSWERED.contains(&pf) || (ANSWERED_M4.contains(&pf) && !asserted) {
            Ok(LINEAR_ALIGNMENT)
        } else {
            Err(BufferTextureProblem::NotLinear { pixel_format: pf })
        };
        assert_eq!(linear_alignment(pf), want, "{pf:?}");
    }
    // Nor ASTC's 4x2 and 8x4 formats, which the runtime refuses as the compressed formats they
    // are (their type rules are measured as `ASTC_4x4_LDR`'s).
    let astc = [
        P::ASTC4x2LDR,
        P::ASTC4x2SRGB,
        P::ASTC4x2HDR,
        P::ASTC8x4LDR,
        P::ASTC8x4SRGB,
        P::ASTC8x4HDR,
    ];
    // + the 18 added linear formats Metal asserted on.
    assert_eq!(
        P::ALL.len() - 1 - unprobed - astc.len() - ANSWERED.len() - ANSWERED_M4.len(),
        83 + 18,
        "one probe saw 83 formats refused, another 18 more"
    );
    assert_eq!(
        linear_alignment(P::Invalid),
        Err(BufferTextureProblem::NotLinear {
            pixel_format: P::Invalid
        })
    );
    // Nor `AnyDepth` and `AnyStencil`, of which Metal makes no texture alone: it asserts as on
    // every depth and stencil format ("Linear textures do not support
    // depth/stencil pixel formats").
    for pf in [P::AnyDepth, P::AnyStencil] {
        assert_eq!(
            linear_alignment(pf),
            Err(BufferTextureProblem::NotLinear { pixel_format: pf }),
            "{pf:?}"
        );
    }
}

/// One texture over a buffer: the call, the descriptor's type, format, width, height, slices,
/// levels and storage mode (`Shared` by default), the offset, `bytesPerRow`, and the buffer's
/// length.
#[derive(Clone, Copy, Debug)]
struct Case {
    call: BufferTextureCall,
    texture_type: TextureType,
    pixel_format: P,
    width: u64,
    height: u64,
    slices: u64,
    levels: u64,
    offset: u64,
    bytes_per_row: u64,
    length: u64,
    storage: StorageMode,
    /// The descriptor's usage, which only an IOSurface plane whose GPU rows overlap is held to.
    usage: TextureUsage,
}

/// A 2D RGBA8Unorm texture through the public call.
fn texture(width: u64, height: u64, offset: u64, bytes_per_row: u64, length: u64) -> Case {
    Case {
        call: BufferTextureCall::Texture,
        texture_type: TextureType::Type2D,
        pixel_format: P::RGBA8Unorm,
        width,
        height,
        slices: 1,
        levels: 1,
        offset,
        bytes_per_row,
        length,
        storage: StorageMode::Shared,
        usage: TextureUsage::SHADER_READ,
    }
}

/// A 2D array of RGBA8Unorm through the SPI.
fn array(
    (width, height, slices): (u64, u64, u64),
    offset: u64,
    bytes_per_row: u64,
    length: u64,
) -> Case {
    Case {
        call: BufferTextureCall::Linear,
        texture_type: TextureType::Type2DArray,
        slices,
        ..texture(width, height, offset, bytes_per_row, length)
    }
}

/// A texture buffer of RGBA8Unorm through the public call.
fn texture_buffer(width: u64, offset: u64, bytes_per_row: u64, length: u64) -> Case {
    Case {
        texture_type: TextureType::TextureBuffer,
        ..texture(width, 1, offset, bytes_per_row, length)
    }
}

fn plan(case: Case) -> Result<BufferLayout, BufferRefusal> {
    placed(case).map(|placed| {
        // Only an IOSurface plane's CPU layout differs from its GPU one.
        assert_eq!(placed.gpu, placed.cpu, "{case:?}");
        placed.gpu
    })
}

/// Where `case` lies for the GPU and the CPU.
fn placed(case: Case) -> Result<Placed, BufferRefusal> {
    let layout = TextureDescriptor {
        texture_type: case.texture_type,
        pixel_format: case.pixel_format,
        width: case.width,
        height: case.height,
        depth: if case.texture_type == TextureType::Type3D {
            2
        } else {
            1
        },
        array_length: case.slices,
        mipmap_level_count: case.levels,
        sample_count: if case.texture_type == TextureType::Type2DMultisample {
            4
        } else {
            1
        },
        ..TextureDescriptor::default()
    }
    .check(APPLE)
    .unwrap_or_else(|problem| panic!("{case:?}: {problem:?}"));
    let texel_bytes = direct_format(case.pixel_format)
        .and_then(|f| f.block())
        .map_or(0, |block| u64::from(block.size));
    plan_buffer_texture(
        &BufferTexture {
            layout,
            texel_bytes,
            storage: case.storage,
            offset: case.offset,
            bytes_per_row: case.bytes_per_row,
            call: case.call,
            usage: case.usage,
        },
        &BufferFacts {
            length: case.length,
        },
    )
}

fn refused(problem: BufferTextureProblem) -> Result<BufferLayout, BufferRefusal> {
    Err(BufferRefusal::Metal(problem))
}

/// The probes' 2D textures and texture buffers through the public call: what Metal made, and
/// what it asserted on or stopped under validation.
#[test]
fn two_dimensional_textures_and_texture_buffers_follow_metal() {
    use BufferTextureProblem as Pr;
    let made = |offset, row_pitch, rows, row_bytes| {
        Ok(BufferLayout {
            offset,
            row_pitch,
            slice_pitch: row_pitch * rows,
            row_bytes,
            rows,
            slices: 1,
        })
    };
    let cases = [
        // Probe 9's base, and bytesPerRow below the packed row of 8 RGBA8 texels.
        (texture(8, 4, 0, 32, 4096), made(0, 32, 4, 32)),
        (
            texture(8, 4, 0, 16, 4096),
            refused(Pr::RowPitch {
                bytes_per_row: 16,
                row_bytes: 32,
            }),
        ),
        (
            texture(8, 4, 0, 0, 4096),
            refused(Pr::RowPitch {
                bytes_per_row: 0,
                row_bytes: 32,
            }),
        ),
        // The length: the last row's packed bytes may end on the buffer's end.
        (texture(8, 4, 3872, 64, 4096), made(3872, 64, 4, 32)),
        // Plain Metal makes a texture whose rows end past the length of a buffer under 16 KiB,
        // whose `allocatedSize` is its length; the runtime has no bytes there (`past.m`).
        (
            texture(8, 4, 3888, 64, 4096),
            Err(BufferRefusal::TexelsPastLength),
        ),
        (texture(8, 4, 3856, 64, 4096), made(3856, 64, 4, 32)),
        (texture(8, 4, 3920, 48, 4096), made(3920, 48, 4, 32)),
        (
            texture(4, 1, 4096, 16, 4096),
            Err(BufferRefusal::TexelsPastLength),
        ),
        // Offsets and pitches are multiples of 16.
        (
            texture(4, 4, 8, 16, 4096),
            refused(Pr::OffsetAlignment {
                offset: 8,
                alignment: 16,
            }),
        ),
        (
            texture(5, 4, 0, 24, 4096),
            refused(Pr::RowPitchAlignment {
                bytes_per_row: 24,
                alignment: 16,
            }),
        ),
        // One level, and no other type.
        (
            Case {
                levels: 2,
                ..texture(8, 4, 0, 32, 4096)
            },
            refused(Pr::MipmapLevelCount { count: 2 }),
        ),
        (
            array((4, 4, 1), 0, 16, 4096),
            Ok(BufferLayout {
                slice_pitch: 128,
                ..made(0, 16, 4, 16).unwrap()
            }),
        ),
        (
            Case {
                call: BufferTextureCall::Texture,
                ..array((4, 4, 1), 0, 16, 4096)
            },
            refused(Pr::TextureType {
                texture_type: TextureType::Type2DArray,
            }),
        ),
        (
            Case {
                texture_type: TextureType::Type2DMultisample,
                ..texture(4, 4, 0, 16, 4096)
            },
            refused(Pr::TextureType {
                texture_type: TextureType::Type2DMultisample,
            }),
        ),
        (
            Case {
                texture_type: TextureType::Cube,
                call: BufferTextureCall::Linear,
                ..texture(4, 4, 0, 16, 4096)
            },
            refused(Pr::TextureType {
                texture_type: TextureType::Cube,
            }),
        ),
        (
            Case {
                texture_type: TextureType::Type3D,
                call: BufferTextureCall::Linear,
                ..texture(4, 4, 0, 16, 4096)
            },
            refused(Pr::TextureType {
                texture_type: TextureType::Type3D,
            }),
        ),
        // A compressed format.
        (
            Case {
                pixel_format: P::BC1RGBA,
                ..texture(4, 4, 0, 16, 4096)
            },
            refused(Pr::NotLinear {
                pixel_format: P::BC1RGBA,
            }),
        ),
        // Texture buffers: any offset (one off 16, which Metal's validation alone stops, its
        // functions read from the offset rounded down), any bytesPerRow that covers the row, and a
        // length bound of the packed row. The SPI's keeps an offset of 16.
        (texture_buffer(64, 16, 256, 4096), made(16, 256, 1, 256)),
        (texture_buffer(64, 16, 260, 4096), made(16, 260, 1, 256)),
        (texture_buffer(64, 3840, 512, 4096), made(3840, 512, 1, 256)),
        (texture_buffer(64, 4, 256, 4096), made(4, 256, 1, 256)),
        (texture_buffer(64, 1, 256, 4096), made(1, 256, 1, 256)),
        (texture_buffer(64, 20, 256, 4096), made(20, 256, 1, 256)),
        (
            Case {
                call: BufferTextureCall::Linear,
                ..texture_buffer(64, 4, 256, 4096)
            },
            refused(Pr::OffsetAlignment {
                offset: 4,
                alignment: 16,
            }),
        ),
        (
            texture_buffer(64, 16, 128, 4096),
            refused(Pr::RowPitch {
                bytes_per_row: 128,
                row_bytes: 256,
            }),
        ),
        (
            texture_buffer(64, 3856, 256, 4096),
            Err(BufferRefusal::TexelsPastLength),
        ),
        (
            texture_buffer(1024, 16, 4096, 4096),
            Err(BufferRefusal::TexelsPastLength),
        ),
        // Through the SPI, a 2D texture keeps the 2D rules.
        (
            Case {
                call: BufferTextureCall::Linear,
                ..texture(4, 4, 16, 16, 16384)
            },
            made(16, 16, 4, 16),
        ),
        (
            Case {
                call: BufferTextureCall::Linear,
                ..texture(4, 4, 0, 32, 111)
            },
            Err(BufferRefusal::TexelsPastLength),
        ),
        (
            Case {
                call: BufferTextureCall::Linear,
                ..texture(4, 4, 0, 32, 112)
            },
            made(0, 32, 4, 16),
        ),
    ];
    for (case, want) in cases {
        assert_eq!(plan(case), want, "{case:?}");
    }
}

/// Metal's resource-option rules: a texture over a buffer takes the buffer's options whatever
/// its descriptor names, through the public call and the SPI (probe 9d; `Buffer::texture_over`),
/// and an IOSurface plane its descriptor's, so the plan refuses no storage mode but
/// a plane's `Memoryless`, whose use is unmeasured.
#[test]
fn only_a_memoryless_plane_is_refused_its_storage_mode() {
    use StorageMode::{Managed, Memoryless, Private, Shared};
    let calls = [
        BufferTextureCall::Texture,
        BufferTextureCall::Linear,
        BufferTextureCall::IoSurfacePlane { planar: true },
        BufferTextureCall::IoSurfacePlane { planar: false },
    ];
    for call in calls {
        for storage in [Shared, Managed, Private, Memoryless] {
            let case = Case {
                call,
                storage,
                ..texture(8, 4, 0, 32, 4096)
            };
            let surface = matches!(call, BufferTextureCall::IoSurfacePlane { .. });
            let want = if surface && storage == Memoryless {
                Err(BufferRefusal::MemorylessPlane)
            } else {
                Ok(())
            };
            assert_eq!(placed(case).map(|_| ()), want, "{call:?}: {storage:?}");
        }
    }
}

/// A 2D array's layout through the SPI (probes 9b, 9c and 9e): offsets at multiples of 128,
/// `bytesPerRow` at multiples of 16, and each slice `bytesPerRow × height` rounded up to 128
/// after the one before, whatever `bytesPerImage` is (the plan takes none).
#[test]
fn an_arrays_slices_start_at_metals_own_stride() {
    use BufferTextureProblem as Pr;
    // (width, height, bytesPerRow) and the slice stride Metal used.
    let strides = [
        ((4, 4, 16), 128),
        ((4, 3, 16), 128),
        ((4, 5, 16), 128),
        ((4, 4, 32), 128),
        ((4, 4, 48), 256),
        ((4, 4, 64), 256),
        ((4, 3, 64), 256),
        ((4, 4, 128), 512),
        ((4, 4, 256), 1024),
        ((5, 4, 32), 128),
        ((64, 2, 256), 512),
        ((64, 2, 272), 640),
    ];
    for ((w, h, bpr), stride) in strides {
        for offset in [0, 128, 256] {
            let case = array((w, h, 2), offset, bpr, 16384);
            let layout = plan(case).unwrap_or_else(|e| panic!("{case:?}: {e:?}"));
            assert_eq!(layout.slice_pitch, stride, "{case:?}");
            assert_eq!(layout.offset, offset);
        }
    }
    // RGBA32Float's 16-byte texels at bytesPerRow 64 and height 5: 320 rounds up to 384.
    let wide = Case {
        pixel_format: P::RGBA32Float,
        ..array((4, 5, 3), 128, 64, 16384)
    };
    assert_eq!(plan(wide).unwrap().slice_pitch, 384);
    for offset in [16, 64, 4] {
        assert_eq!(
            plan(array((4, 4, 2), offset, 16, 16384)),
            refused(Pr::OffsetAlignment {
                offset,
                alignment: 128,
            })
        );
    }
    for bytes_per_row in [20, 24, 28] {
        assert_eq!(
            plan(array((5, 4, 2), 0, bytes_per_row, 16384)),
            refused(Pr::RowPitchAlignment {
                bytes_per_row,
                alignment: 16,
            })
        );
    }
    assert_eq!(
        plan(Case {
            levels: 2,
            ..array((4, 4, 2), 0, 16, 16384)
        }),
        refused(Pr::MipmapLevelCount { count: 2 })
    );
}

/// Where an array may end (probes 9b, 9c, 9e and 9f): Metal answers nil where a buffer of at
/// least 16 KiB has the rounded slices end past its `allocatedSize`; elsewhere it makes an array
/// whose last slice runs past the length, which the runtime refuses as not implemented. Its
/// validation's own count, the rows as if the slices were not rounded, is no refusal: every case
/// marked so below is one the count stops and plain Metal makes.
#[test]
fn an_array_ends_where_metal_lets_it() {
    use BufferTextureProblem as Pr;
    let nil = |end, allocation| {
        Err::<(), _>(BufferRefusal::Metal(Pr::PastAllocation { end, allocation }))
    };
    let gap = Err::<(), _>(BufferRefusal::TexelsPastLength);
    let ok = |case: Case| plan(case).map(|_| ());
    let cases = [
        // Two 4x4 slices at bytesPerRow 16 from 16128 (a 64-byte image, a 128-byte stride).
        (array((4, 4, 2), 16128, 16, 16255), gap), // counted past
        (array((4, 4, 2), 16128, 16, 16256), gap),
        (array((4, 4, 2), 16128, 16, 16319), gap),
        (array((4, 4, 2), 16128, 16, 16320), Ok(())),
        (array((4, 4, 2), 16128, 16, 16384), Ok(())),
        // From 16256 the slices end at 16512: nil in 16384 bytes, made in 16385 or more.
        (array((4, 4, 2), 16256, 16, 16384), nil(16512, 16384)),
        (array((4, 4, 2), 16256, 16, 16385), gap),
        (array((4, 4, 2), 16256, 16, 16512), Ok(())),
        (array((4, 4, 2), 16384, 16, 16385), gap), // counted past
        (array((4, 4, 2), 32640, 16, 32768), nil(32896, 32768)),
        // Small buffers: made past the length whatever the rounded slices' end.
        (array((4, 4, 2), 0, 16, 64), gap),  // counted past
        (array((4, 4, 2), 0, 16, 127), gap), // counted past
        (array((4, 4, 2), 0, 16, 128), gap),
        (array((4, 4, 2), 0, 16, 191), gap),
        (array((4, 4, 2), 0, 16, 192), Ok(())),
        (array((4, 4, 3), 0, 16, 300), gap),
        (array((4, 4, 64), 0, 16, 4096), gap),
        (array((4, 4, 128), 0, 16, 4096), gap), // counted past
        (array((4, 4, 129), 0, 16, 4096), gap), // counted past
        // bytesPerRow above the packed row: validation counts the last row's packed bytes.
        (array((4, 4, 2), 0, 32, 239), gap), // counted past
        (array((4, 4, 2), 0, 32, 240), Ok(())),
        (array((4, 3, 3), 0, 48, 399), gap), // counted past
        (array((4, 3, 3), 0, 48, 400), gap),
        (array((4, 3, 3), 0, 48, 432), gap),
    ];
    for (case, want) in cases {
        assert_eq!(ok(case), want, "{case:?}");
    }
}

/// A 2D texture's and a texture buffer's own end, measured on an Apple M3 Max under macOS 27.0.1
/// with the API validation layer off (`past.m`, one case per process): plain Metal makes the
/// texture whenever its rows end inside the buffer's `allocatedSize`, however far past the
/// `length` they run, and answers nil only past the allocation. Metal's validation counts those
/// rows against the `length` and stops every case here but the two made ones; that count is the
/// layer's alone, so the runtime does not refuse by it.
#[test]
fn a_texture_over_a_buffer_ends_where_metals_allocation_does() {
    use BufferTextureProblem as Pr;
    let gap = Err::<(), _>(BufferRefusal::TexelsPastLength);
    let nil = |end, allocation| {
        Err::<(), _>(BufferRefusal::Metal(Pr::PastAllocation { end, allocation }))
    };
    let ok = |case: Case| plan(case).map(|_| ());
    // Each row is `texture(width, height, offset, bytes_per_row, length)` as the probe ran it, in
    // RGBA8Unorm: 64 texels are a 256-byte row.
    let cases = [
        // A 4096-byte buffer, whose `allocatedSize` is its length: made 128 and 12288 bytes past.
        (texture_buffer(64, 3968, 256, 4096), gap),
        (texture(64, 1, 3968, 256, 4096), gap),
        (texture(64, 64, 0, 256, 4096), gap),
        // A 20000-byte buffer, `allocatedSize` 32768: made inside the allocation, nil past it.
        (texture(64, 4, 19200, 256, 20000), gap),
        (texture_buffer(64, 20000, 256, 20000), gap),
        (texture(64, 4, 32000, 256, 20000), nil(33024, 32768)),
        (texture_buffer(64, 32700, 256, 20000), nil(32956, 32768)),
        // A 16384-byte buffer, whose `allocatedSize` is its length: the last row may end on it.
        (texture(64, 4, 15616, 256, 16384), nil(16640, 16384)),
        (texture(64, 4, 15360, 256, 16384), Ok(())),
    ];
    for (case, want) in cases {
        assert_eq!(ok(case), want, "{case:?}");
    }
}

/// A linear image aliases a texture's place in its buffer only where every one of its reports
/// is Metal's layout's, and its subresources hold no byte but the texture's texels.
#[test]
fn a_linear_image_aliases_only_metals_layout() {
    let single = BufferLayout {
        offset: 256,
        row_pitch: 80,
        slice_pitch: 320,
        row_bytes: 80,
        rows: 4,
        slices: 1,
    };
    let image = LinearImage {
        alignment: 16,
        size: 320,
        offset: 0,
        subresource: 320,
        row_pitch: 80,
        array_pitch: 320,
    };
    assert!(aliases(&single, 4096 + 256, 320, &image));
    // The binding off the image's alignment (RADV's 256 at a 16-byte offset).
    assert!(!aliases(
        &single,
        4096 + 272,
        320,
        &LinearImage {
            alignment: 256,
            ..image
        }
    ));
    // The driver's own pitch where another was asked for (NVIDIA's explicit layouts).
    assert!(!aliases(
        &single,
        4096,
        320,
        &LinearImage {
            row_pitch: 128,
            ..image
        }
    ));
    // Texels not at the binding, or memory past the buffer's.
    assert!(!aliases(
        &single,
        4096,
        320,
        &LinearImage {
            offset: 64,
            ..image
        }
    ));
    assert!(!aliases(&single, 4096, 319, &image));
    // Memory past the subresource, within the buffer's: no write through the image reaches it.
    assert!(aliases(
        &single,
        4096,
        4096,
        &LinearImage {
            size: 4096,
            ..image
        }
    ));
    // A subresource past the texels (lavapipe's rows rounded up to 4, of a texture of 3).
    let three = BufferLayout { rows: 3, ..single };
    assert!(!aliases(&three, 4096, 320, &image));
    assert!(aliases(
        &three,
        4096,
        320,
        &LinearImage {
            subresource: 240,
            ..image
        }
    ));
    // Rows with the buffer's bytes between them, at Metal's pitch and the image's alike.
    let padded = BufferLayout {
        row_bytes: 68,
        ..single
    };
    assert!(!aliases(&padded, 4096, 320, &image));
    // Whatever its usage: a texture that only reads takes a first move over them too.
    for usage in [
        TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ,
        TextureUsage::SHADER_READ,
    ] {
        assert!(holding(&padded, usage).is_err(), "{usage:?}");
        assert_eq!(holding(&single, usage), Ok(()), "{usage:?}");
    }
    // One row has nothing between rows, but its pitch's padding past the texels.
    let row = BufferLayout {
        rows: 1,
        slice_pitch: 80,
        ..padded
    };
    assert_eq!(holding(&row, TextureUsage::SHADER_READ), Ok(()));
    assert!(!aliases(
        &row,
        4096,
        80,
        &LinearImage {
            size: 80,
            subresource: 80,
            ..image
        }
    ));
    assert!(aliases(
        &row,
        4096,
        80,
        &LinearImage {
            size: 80,
            subresource: 68,
            ..image
        }
    ));
    // An array's slices at Metal's stride, and at no other.
    let slices = BufferLayout {
        slices: 3,
        ..single
    };
    let stacked = LinearImage { size: 960, ..image };
    assert!(aliases(&slices, 4096, 960, &stacked));
    assert!(!aliases(
        &slices,
        4096,
        960,
        &LinearImage {
            array_pitch: 384,
            ..stacked
        }
    ));
    // Slices with the buffer's bytes between them (an array's slices rounded up to 128).
    let apart = BufferLayout {
        slice_pitch: 384,
        ..slices
    };
    for usage in [
        TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ,
        TextureUsage::SHADER_READ,
    ] {
        assert!(holding(&apart, usage).is_err(), "{usage:?}");
    }
    assert!(!aliases(
        &apart,
        4096,
        1088,
        &LinearImage {
            size: 1088,
            array_pitch: 384,
            ..image
        }
    ));
    // A driver that reports the array's whole bytes as its first subresource's.
    assert!(!aliases(
        &slices,
        4096,
        960,
        &LinearImage {
            subresource: 960,
            ..stacked
        }
    ));
    // A single slice ignores the array pitch a driver reports.
    assert!(aliases(
        &single,
        4096,
        320,
        &LinearImage {
            array_pitch: 0,
            ..image
        }
    ));
}

/// A texture over padded rows is a shadow, whatever its usage, naming the operations Vulkan lets
/// write the padding: a render target its image's layout transitions (its first move among them)
/// and a render pass's load and store operations; any other (one that only reads, or that a
/// shader writes) its layout transitions alone. Its usage as Metal reports it decides, `Unknown`
/// a render target's.
#[test]
fn a_padded_texture_s_shadow_names_what_may_write_its_padding() {
    let padded = BufferLayout {
        offset: 0,
        row_pitch: 80,
        slice_pitch: 320,
        row_bytes: 68,
        rows: 4,
        slices: 1,
    };
    let transitions = "its rows lie 80 bytes apart, 68 bytes of texels each: an image over them \
                       holds the bytes between them, which Vulkan lets its layout transitions \
                       read and write, its first move from the layout it is made in among them";
    let passes = "its rows lie 80 bytes apart, 68 bytes of texels each: an image over them holds \
                  the bytes between them, which Vulkan lets its layout transitions read and write, \
                  and a render pass's load and store operations write back";
    for usage in [
        TextureUsage::SHADER_READ,
        TextureUsage::SHADER_READ | TextureUsage::PIXEL_FORMAT_VIEW,
        TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE,
        TextureUsage::SHADER_WRITE,
        TextureUsage::SHADER_ATOMIC,
        TextureUsage::BLOCK_WRITES_ONLY,
    ] {
        assert_eq!(
            holding(&padded, usage),
            Err(transitions.to_owned()),
            "{usage:?}"
        );
    }
    for usage in [
        TextureUsage::RENDER_TARGET,
        TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ,
        effective_usage(TextureUsage::UNKNOWN),
    ] {
        assert_eq!(holding(&padded, usage), Err(passes.to_owned()), "{usage:?}");
    }
}

/// A linear image serves a texture where its tiling has every feature the optimal one has for
/// the texture's uses, in every format it is viewed as; the formats a device's linear tiling
/// lacks send the texture to a shadow image.
#[test]
fn a_linear_image_serves_only_what_its_tiling_holds() {
    let mut served = 0;
    for (name, desc, device) in committed() {
        let facts = facts(&desc, &device);
        for usage in [
            TextureUsage::SHADER_READ,
            TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE,
        ] {
            let layout = TextureDescriptor {
                pixel_format: P::RGBA8Unorm,
                width: 8,
                height: 4,
                usage,
                ..TextureDescriptor::default()
            }
            .check(APPLE)
            .unwrap();
            let Ok(plan) = plan_image(&layout, usage, false, &facts) else {
                continue;
            };
            let serves = linear_lack(&plan, &facts).is_none();
            served += usize::from(serves);
            // With the linear tiling of the image's format stripped, nothing is served.
            let mut bare = desc.formats.clone();
            if let Some(p) = bare.formats.get_mut(&plan.format) {
                p.linear_tiling = FormatFeatureFlags2::empty();
            }
            let stripped = super::super::image_plan::ImageFacts {
                formats: &bare,
                ..facts
            };
            assert!(linear_lack(&plan, &stripped).is_some(), "{name} {usage:?}");
            // A plan of extended usage is checked as any plan is: served exactly where the plan
            // without it is, and not with the image's format's tiling stripped.
            let mutable = super::super::image_plan::ImagePlan {
                flags: plan.flags | ImageCreateFlags::MUTABLE_FORMAT,
                ..plan.clone()
            };
            let extended = super::super::image_plan::ImagePlan {
                flags: mutable.flags | ImageCreateFlags::EXTENDED_USAGE,
                ..plan.clone()
            };
            assert_eq!(
                linear_lack(&extended, &facts).is_none(),
                linear_lack(&mutable, &facts).is_none(),
                "{name} {usage:?}"
            );
            assert!(
                !linear_lack(&extended, &stripped).is_none(),
                "{name} {usage:?}"
            );
            // The devices measured sample and store RGBA8Unorm through linear images.
            if MEASURED.contains(&name.as_str()) {
                assert!(serves, "{name}: {:?}", facts.linear(Format::R8G8B8A8_UNORM));
            }
        }
    }
    assert!(served > 0);
}

/// What a linear image lacks is named exactly (a heap texture's linear image): with a format's
/// linear tiling stripped, the features its optimal tiling has for the texture's usage; a texture
/// of several levels also relies on the blit and linear-filter features its mipmaps need, and one
/// level does not; views of any format are checked in every format a view may take; and extended
/// usage format by format, each format asked only what its optimal tiling takes.
#[test]
fn a_linear_image_names_what_its_tiling_lacks() {
    use FormatFeatureFlags2 as F;
    let (mut named, mut any_checked, mut extended_checked) = (0, 0, 0);
    for (name, desc, device) in committed() {
        let facts = facts(&desc, &device);
        let plan_of = |levels: u64| {
            let layout = TextureDescriptor {
                pixel_format: P::RGBA8Unorm,
                width: 64,
                height: 64,
                mipmap_level_count: levels,
                usage: TextureUsage::SHADER_READ,
                ..TextureDescriptor::default()
            }
            .check(APPLE)
            .unwrap();
            plan_image(&layout, TextureUsage::SHADER_READ, false, &facts).ok()
        };
        let (Some(one), Some(seven)) = (plan_of(1), plan_of(7)) else {
            continue;
        };
        let optimal = facts.optimal(one.format);
        let sampled = F::SAMPLED_IMAGE
            | F::SAMPLED_IMAGE_FILTER_LINEAR
            | F::SAMPLED_IMAGE_FILTER_MINMAX
            | F::TRANSFER_SRC
            | F::TRANSFER_DST;
        // Stripped: what the optimal tiling has of the usage's features.
        let mut bare = desc.formats.clone();
        let mut blitless = desc.formats.clone();
        if let Some(p) = bare.formats.get_mut(&one.format) {
            p.linear_tiling = F::empty();
        }
        if let Some(p) = blitless.formats.get_mut(&one.format) {
            p.linear_tiling = p.optimal_tiling.difference(F::BLIT_DST);
        }
        let stripped = super::super::image_plan::ImageFacts {
            formats: &bare,
            ..facts
        };
        let expected = optimal.intersection(sampled);
        if !expected.is_empty() {
            named += 1;
            assert_eq!(
                linear_lack(&one, &stripped),
                Some(LinearLack {
                    format: one.format,
                    features: expected
                }),
                "{name}"
            );
        }
        // Without `BLIT_DST`, where the device's linear images serve both: only the texture of
        // several levels lacks it.
        let blitless = super::super::image_plan::ImageFacts {
            formats: &blitless,
            ..facts
        };
        let served = linear_lack(&one, &facts).is_none() && linear_lack(&seven, &facts).is_none();
        if served {
            assert_eq!(linear_lack(&one, &blitless), None, "{name}");
        }
        if served && optimal.contains(F::BLIT_DST) {
            assert_eq!(
                linear_lack(&seven, &blitless),
                Some(LinearLack {
                    format: one.format,
                    features: F::BLIT_DST
                }),
                "{name}"
            );
        }
        // Views of any format (`pixelFormatView`) are checked in each format a view may take:
        // `R32Float`'s among `RGBA8Unorm`'s, which the plan's own list does not hold.
        let any = super::super::image_plan::ImagePlan {
            flags: one.flags | ImageCreateFlags::MUTABLE_FORMAT,
            view_formats: Vec::new(),
            ..one.clone()
        };
        let mut r32 = desc.formats.clone();
        if let Some(p) = r32.formats.get_mut(&Format::R32_SFLOAT) {
            p.linear_tiling = F::empty();
        }
        let r32 = super::super::image_plan::ImageFacts {
            formats: &r32,
            ..facts
        };
        let r32_needed = facts.optimal(Format::R32_SFLOAT).intersection(sampled);
        if linear_lack(&any, &facts).is_none() && !r32_needed.is_empty() {
            any_checked += 1;
            assert_eq!(
                linear_lack(&any, &r32),
                Some(LinearLack {
                    format: Format::R32_SFLOAT,
                    features: r32_needed
                }),
                "{name}"
            );
            assert_eq!(linear_lack(&one, &r32), None, "{name}");
        }
        // Extended usage: storage where the image's own format takes none in either tiling, through
        // views of any format. Nothing is asked of the image's format for it, and each view format
        // is asked what its optimal tiling takes: `R32Float`'s storage, named once its linear
        // tiling lacks it.
        let stored = super::super::image_plan::ImagePlan {
            flags: any.flags | ImageCreateFlags::EXTENDED_USAGE,
            usage: any.usage | ImageUsageFlags::STORAGE,
            ..any.clone()
        };
        let mut storeless = desc.formats.clone();
        if let Some(p) = storeless.formats.get_mut(&one.format) {
            p.optimal_tiling = p.optimal_tiling.difference(F::STORAGE_IMAGE);
            p.linear_tiling = p.linear_tiling.difference(F::STORAGE_IMAGE);
        }
        let mut r32_storeless = storeless.clone();
        if let Some(p) = r32_storeless.formats.get_mut(&Format::R32_SFLOAT) {
            p.linear_tiling = p.linear_tiling.difference(F::STORAGE_IMAGE);
        }
        let storeless = super::super::image_plan::ImageFacts {
            formats: &storeless,
            ..facts
        };
        let r32_storeless = super::super::image_plan::ImageFacts {
            formats: &r32_storeless,
            ..facts
        };
        let r32_stores = facts.optimal(Format::R32_SFLOAT).contains(F::STORAGE_IMAGE);
        if linear_lack(&stored, &storeless).is_none() && r32_stores {
            extended_checked += 1;
            assert_eq!(
                linear_lack(&stored, &r32_storeless),
                Some(LinearLack {
                    format: Format::R32_SFLOAT,
                    features: F::STORAGE_IMAGE
                }),
                "{name}"
            );
        }
    }
    assert!(named > 0 && any_checked > 0 && extended_checked > 0);
}

/// A linear image's limits name what it lacks: no image at all where the device reports none (every
/// limit 0, as the RTX 3090 reports for a linear cube, 3D or depth image), else the first extent,
/// the levels, the layers or the sample count past the device's.
#[test]
fn a_linear_image_s_limits_name_what_it_lacks() {
    use ash::vk;
    use metal2vulkan::error::LinearNeed;
    let (_, desc, device) = committed().into_iter().next().unwrap();
    let facts = facts(&desc, &device);
    let plan = |width: u64, levels: u64, slices: u64, samples: u64| {
        let layout = TextureDescriptor {
            pixel_format: P::RGBA8Unorm,
            texture_type: if slices > 1 {
                TextureType::Type2DArray
            } else if samples > 1 {
                TextureType::Type2DMultisample
            } else {
                TextureType::Type2D
            },
            width,
            height: 64,
            mipmap_level_count: levels,
            array_length: slices,
            sample_count: samples,
            usage: TextureUsage::SHADER_READ,
            ..TextureDescriptor::default()
        }
        .check(APPLE)
        .unwrap();
        plan_image(&layout, TextureUsage::SHADER_READ, false, &facts).unwrap()
    };
    // The RTX 3090's limits for a linear 2D RGBA8Unorm image (NVIDIA 615.71.09).
    let nvidia = vk::ImageFormatProperties {
        max_extent: vk::Extent3D {
            width: 32768,
            height: 32768,
            depth: 1,
        },
        max_mip_levels: 1,
        max_array_layers: 1,
        sample_counts: vk::SampleCountFlags::TYPE_1,
        max_resource_size: 1 << 32,
    };
    assert_eq!(check_linear_limits(&plan(64, 1, 1, 1), &nvidia), Ok(()));
    assert_eq!(
        check_linear_limits(&plan(64, 1, 1, 1), &vk::ImageFormatProperties::default()),
        Err(LinearNeed::Image)
    );
    assert_eq!(
        check_linear_limits(&plan(64, 7, 1, 1), &nvidia),
        Err(LinearNeed::MipLevels {
            required: 7,
            available: 1
        })
    );
    assert_eq!(
        check_linear_limits(&plan(64, 1, 2, 1), &nvidia),
        Err(LinearNeed::ArrayLayers {
            required: 2,
            available: 1
        })
    );
    assert_eq!(
        check_linear_limits(&plan(64, 1, 1, 4), &nvidia),
        Err(LinearNeed::SampleCount(4))
    );
    let narrow = vk::ImageFormatProperties {
        max_extent: vk::Extent3D {
            width: 32,
            height: 32768,
            depth: 1,
        },
        ..nvidia
    };
    assert_eq!(
        check_linear_limits(&plan(64, 1, 1, 1), &narrow),
        Err(LinearNeed::Extent {
            required: 64,
            available: 32
        })
    );
}

/// A texture over a buffer that takes no linear image says which need failed: the features the
/// format's linear tiling lacks, the format query's refusal with the format, usage and flags,
/// a limit with what it needs and what the device holds, an import that is not importable.
#[test]
fn a_refused_linear_image_names_the_need_that_failed() {
    use super::super::texture::LinearRefusal;
    use ash::vk;
    use metal2vulkan::error::LinearNeed;
    let (_, desc, device) = committed().into_iter().next().unwrap();
    let facts = facts(&desc, &device);
    let layout = TextureDescriptor {
        pixel_format: P::BGRA8Unorm,
        texture_type: TextureType::Type2DArray,
        width: 64,
        height: 64,
        array_length: 2,
        usage: TextureUsage::SHADER_READ | TextureUsage::RENDER_TARGET,
        ..TextureDescriptor::default()
    }
    .check(APPLE)
    .unwrap();
    let usage = TextureUsage::SHADER_READ | TextureUsage::RENDER_TARGET;
    let plan = plan_image(&layout, usage, false, &facts).unwrap();
    let describe = |refusal: LinearRefusal| refusal.describe(&plan);
    let lacks = describe(LinearRefusal::Need(LinearNeed::Features {
        format: Format::B8G8R8A8_UNORM,
        features: FormatFeatureFlags2::COLOR_ATTACHMENT,
    }));
    assert!(
        lacks.contains("B8G8R8A8_UNORM") && lacks.contains("COLOR_ATTACHMENT"),
        "{lacks}"
    );
    let refused = describe(LinearRefusal::Need(LinearNeed::Image));
    assert!(
        refused.contains("B8G8R8A8_UNORM") && refused.contains("COLOR_ATTACHMENT"),
        "{refused}"
    );
    let layers = describe(LinearRefusal::Need(LinearNeed::ArrayLayers {
        required: 2,
        available: 1,
    }));
    assert!(
        layers.starts_with("2 array layers, where its linear images of")
            && layers.ends_with("hold 1"),
        "{layers}"
    );
    let import = describe(LinearRefusal::NotImportable(
        vk::ExternalMemoryFeatureFlags::DEDICATED_ONLY,
    ));
    assert!(
        import.contains("importable false, dedicated only true"),
        "{import}"
    );
}

/// The committed profiles of the devices linear images were measured on.
const MEASURED: [&str; 5] = [
    "moltenvk-m3",
    "amd-780m-radv",
    "intel-irisxe-rplp-anv",
    "nvidia-rtx5080-laptop",
    "gate-lavapipe-x86_64",
];

/// The IOSurface stand-in's planes (`BufferTextureCall::IoSurfacePlane`) where plain Metal put
/// them on the M3 Max and an M4: the CPU's layout is the plane's; the GPU's starts at the
/// plane's offset rounded down to 16, rows the pitch rounded down to 16 apart, or 16 apart under
/// 16 bytes a row (`narrow`, where Metal's reads past the first row are memory-dependent); rows
/// that overlap lie at that pitch, which a plain plane's shadow holds and a render target's is
/// refused for; a surface without planes takes only a pitch and an offset of 16 bytes.
#[test]
fn iosurface_planes_lie_where_metal_puts_them() {
    use BufferTextureProblem as Pr;
    let plane = |pf: P, (width, height), (offset, bpr), planar, length| Case {
        call: BufferTextureCall::IoSurfacePlane { planar },
        pixel_format: pf,
        ..texture(width, height, offset, bpr, length)
    };
    let at = |offset, pitch, row_bytes, rows| BufferLayout {
        offset,
        row_pitch: pitch,
        slice_pitch: pitch * rows,
        row_bytes,
        rows,
        slices: 1,
    };
    // (format, texel bytes, extent, (offset, bytesPerRow), the GPU's (offset, pitch)), each a
    // case Metal made.
    let made = [
        // The consumer's 4x4 planes: R8Unorm at 4 and RG8Unorm at 8 bytes a row (narrow), and
        // the ones Buffer::new_texture already took.
        (P::R8Unorm, 1, (4, 4), (0, 4), (0, 16)),
        (P::RG8Unorm, 2, (4, 4), (0, 8), (0, 16)),
        (P::RGBA8Unorm, 4, (4, 4), (0, 16), (0, 16)),
        (P::RGBA16Uint, 8, (4, 4), (0, 32), (0, 32)),
        (P::RGBA32Uint, 16, (4, 4), (0, 64), (0, 64)),
        // Other pitches under 16: rows 16 apart.
        (P::R8Unorm, 1, (4, 4), (0, 5), (0, 16)),
        (P::R8Unorm, 1, (4, 4), (0, 15), (0, 16)),
        (P::R8Unorm, 1, (4, 8), (0, 4), (0, 16)),
        (P::R8Unorm, 1, (1, 4), (0, 1), (0, 16)),
        (P::R8Unorm, 1, (12, 4), (0, 13), (0, 16)),
        (P::RG8Unorm, 2, (4, 4), (0, 9), (0, 16)),
        (P::RG8Unorm, 2, (6, 4), (0, 14), (0, 16)),
        // From 16 up: the pitch rounded down to 16, at least the row.
        (P::R8Unorm, 1, (4, 4), (0, 16), (0, 16)),
        (P::R8Unorm, 1, (4, 4), (0, 40), (0, 32)),
        (P::R8Unorm, 1, (16, 4), (0, 17), (0, 16)),
        (P::RG8Unorm, 2, (4, 4), (0, 40), (0, 32)),
        (P::RGBA8Unorm, 4, (4, 4), (0, 17), (0, 16)),
        (P::RGBA8Unorm, 4, (4, 4), (0, 20), (0, 16)),
        (P::RGBA8Unorm, 4, (4, 4), (0, 40), (0, 32)),
        (P::RGBA8Unorm, 4, (5, 4), (0, 32), (0, 32)),
        (P::RGBA16Uint, 8, (4, 4), (0, 33), (0, 32)),
        (P::RGBA16Uint, 8, (4, 4), (0, 40), (0, 32)),
        (P::RGBA16Uint, 8, (4, 4), (0, 56), (0, 48)),
        (P::RGBA32Uint, 16, (4, 4), (0, 65), (0, 64)),
        (P::RGBA32Uint, 16, (4, 4), (0, 88), (0, 80)),
        // Plane offsets: 4 from 0, 16 from 16, NV12 chroma planes' 36 from 32, 48 from 48 and 96
        // from 96, the consumer's '420v' planes at 256 and 512.
        (P::R8Unorm, 1, (4, 4), (4, 4), (0, 16)),
        (P::R8Unorm, 1, (4, 4), (16, 4), (16, 16)),
        (P::RG8Unorm, 2, (4, 2), (36, 10), (32, 16)),
        (P::RG8Unorm, 2, (4, 2), (48, 12), (48, 16)),
        (P::RG8Unorm, 2, (4, 2), (96, 24), (96, 16)),
        (P::RG8Unorm, 2, (2, 2), (256, 16), (256, 16)),
        (P::RG8Unorm, 2, (2, 2), (512, 16), (512, 16)),
    ];
    for (pf, texel, (w, h), (offset, bpr), (gpu_offset, gpu_pitch)) in made {
        let case = plane(pf, (w, h), (offset, bpr), true, 16384);
        assert_eq!(
            placed(case),
            Ok(Placed {
                gpu: at(gpu_offset, gpu_pitch, w * texel, h),
                cpu: at(offset, bpr, w * texel, h),
                narrow: bpr < 16,
            }),
            "{case:?}"
        );
    }
    let refused = |problem| Err(BufferRefusal::Metal(problem));
    let r8 = P::R8Unorm;
    let cases = [
        // A surface without planes: a pitch and an offset of 16 bytes only.
        (
            plane(r8, (4, 4), (0, 4), false, 16384),
            refused(Pr::RowPitchAlignment {
                bytes_per_row: 4,
                alignment: LINEAR_ALIGNMENT,
            }),
        ),
        (
            plane(r8, (4, 4), (8, 16), false, 16384),
            refused(Pr::OffsetAlignment {
                offset: 8,
                alignment: LINEAR_ALIGNMENT,
            }),
        ),
        (
            plane(r8, (4, 4), (0, 16), false, 16384),
            Ok(Placed {
                gpu: at(0, 16, 4, 4),
                cpu: at(0, 16, 4, 4),
                narrow: false,
            }),
        ),
        // Rows that overlap on the GPU lie at the pitch itself, which the texture's shadow holds
        // in two copy regions each way: RGBA8Unorm 5 wide at 20 and 28 bytes a row (20 bytes of
        // texels 16 apart, sharing 4), R8Unorm 17 wide at 31 (sharing 1) and RGBA16Uint 3 wide
        // at 24 (sharing 8).
        (
            plane(P::RGBA8Unorm, (5, 4), (0, 20), true, 16384),
            Ok(Placed {
                gpu: at(0, 16, 20, 4),
                cpu: at(0, 20, 20, 4),
                narrow: false,
            }),
        ),
        (
            plane(P::RGBA8Unorm, (5, 4), (0, 28), true, 16384),
            Ok(Placed {
                gpu: at(0, 16, 20, 4),
                cpu: at(0, 28, 20, 4),
                narrow: false,
            }),
        ),
        (
            plane(r8, (17, 4), (0, 31), true, 16384),
            Ok(Placed {
                gpu: at(0, 16, 17, 4),
                cpu: at(0, 31, 17, 4),
                narrow: false,
            }),
        ),
        (
            plane(P::RGBA16Uint, (3, 4), (0, 24), true, 16384),
            Ok(Placed {
                gpu: at(0, 16, 24, 4),
                cpu: at(0, 24, 24, 4),
                narrow: false,
            }),
        ),
        // A plane a render pass stores into is refused there: what Metal leaves of the bytes two
        // rows share is not measured. Its rows a row apart or further are made as any plane's.
        (
            Case {
                usage: TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ,
                ..plane(P::RGBA8Unorm, (5, 4), (0, 20), true, 16384)
            },
            Err(BufferRefusal::OverlappingRows),
        ),
        (
            Case {
                usage: TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ,
                ..plane(P::RGBA8Unorm, (4, 4), (0, 20), true, 16384)
            },
            Ok(Placed {
                gpu: at(0, 16, 16, 4),
                cpu: at(0, 20, 16, 4),
                narrow: false,
            }),
        ),
        // One such row overlaps nothing; nor does one row under 16 bytes read past itself.
        (
            plane(r8, (20, 1), (0, 20), true, 16384),
            Ok(Placed {
                gpu: at(0, 32, 20, 1),
                cpu: at(0, 20, 20, 1),
                narrow: false,
            }),
        ),
        (
            plane(r8, (4, 1), (4, 4), true, 16384),
            Ok(Placed {
                gpu: at(0, 16, 4, 1),
                cpu: at(4, 4, 4, 1),
                narrow: false,
            }),
        ),
        // No plane below its row; a plane past the buffer; GPU rows past it.
        (
            plane(r8, (4, 4), (0, 3), true, 16384),
            refused(Pr::RowPitch {
                bytes_per_row: 3,
                row_bytes: 4,
            }),
        ),
        // A plane of a surface with planes whose rows end past the buffer: plain Metal makes such
        // a texture and reads zeros past the plane, which the runtime has no bytes for. Over a
        // surface without planes plain Metal asserts on it ("Descriptor height*arrayLength >
        // IOSurface height"), so the runtime refuses the same shape `Invalid` there (`past.m`).
        (
            plane(r8, (4, 4), (0, 4), true, 15),
            Err(BufferRefusal::TexelsPastLength),
        ),
        (
            plane(r8, (4, 4), (0, 16), false, 48),
            refused(Pr::PastBuffer {
                counted: 52,
                length: 48,
            }),
        ),
        (
            plane(r8, (4, 4), (0, 4), true, 16),
            Err(BufferRefusal::TexelsPastLength),
        ),
        // A 2D texture alone, of one level.
        (
            Case {
                texture_type: TextureType::Type2DArray,
                slices: 2,
                ..plane(r8, (4, 4), (0, 16), true, 16384)
            },
            refused(Pr::TextureType {
                texture_type: TextureType::Type2DArray,
            }),
        ),
        (
            Case {
                levels: 2,
                ..plane(r8, (4, 4), (0, 16), true, 16384)
            },
            refused(Pr::MipmapLevelCount { count: 2 }),
        ),
    ];
    for (case, want) in cases {
        assert_eq!(placed(case), want, "{case:?}");
    }
}

/// The depth and stencil formats Metal makes over a plane of a surface with planes (an M4 under
/// macOS 27.0.1, TTG2's format scout), sampled or read and drawn.
const PLANAR_DEPTH: [P; 5] = [
    P::Depth16Unorm,
    P::Depth32Float,
    P::Depth32FloatStencil8,
    P::Stencil8,
    P::X32Stencil8,
];

/// A depth or stencil texture over a plane of a surface with planes, which Metal makes, is laid
/// out as a colour texture over the plane is, once Metal's own stops passed; over a surface
/// without planes, over a buffer and through the linear SPI Metal stops it (`NotLinear`), as it
/// stops every buffer's texture of those formats.
#[test]
fn a_planar_surface_s_depth_or_stencil_texture_is_laid_out_as_a_colour_one() {
    use BufferTextureCall::{IoSurfacePlane, Linear, Texture};
    for pf in PLANAR_DEPTH {
        let case = |call| Case {
            call,
            pixel_format: pf,
            ..texture(4, 4, 0, 64, 4096)
        };
        let planar = IoSurfacePlane { planar: true };
        let made = placed(case(planar)).unwrap_or_else(|refusal| panic!("{pf:?}: {refusal:?}"));
        assert_eq!(
            (made.gpu.offset, made.gpu.row_pitch, made.gpu.rows),
            (0, 64, 4),
            "{pf:?}"
        );
        for call in [Texture, Linear, IoSurfacePlane { planar: false }] {
            assert_eq!(
                placed(case(call)).map(|_| ()),
                Err(BufferRefusal::Metal(BufferTextureProblem::NotLinear {
                    pixel_format: pf
                })),
                "{pf:?} {call:?}"
            );
        }
        // Metal makes a mipmapped texture over a plane and lays out no one place for its later
        // levels (`surface_texture_envelope`): the runtime's gap.
        let levels = Case {
            levels: 2,
            ..case(planar)
        };
        assert_eq!(
            placed(levels).map(|_| ()),
            Err(BufferRefusal::PlaneShape),
            "{pf:?}"
        );
        let memoryless = Case {
            storage: StorageMode::Memoryless,
            ..case(planar)
        };
        assert_eq!(
            placed(memoryless).map(|_| ()),
            Err(BufferRefusal::MemorylessPlane),
            "{pf:?}"
        );
    }
}

/// The textures over an IOSurface plane beyond a 2D one of its rows that Metal makes, as
/// `oracle_surftex` lists them (its case numbers below) and `surface_texture_envelope` measured
/// their bytes: a texture buffer, a 1D texture whose row fits and a 2D array of one slice lie as
/// the 2D texture's rows do; every other shape over a planar surface's plane is the runtime's gap
/// (`PlaneShape`), where over a surface without planes Metal itself stops it.
#[test]
fn a_surface_plane_s_other_shapes_follow_the_probe() {
    use BufferTextureProblem as Pr;
    let plane = |texture_type, pf, (width, height), (offset, bpr), planar, length| Case {
        call: BufferTextureCall::IoSurfacePlane { planar },
        texture_type,
        pixel_format: pf,
        ..texture(width, height, offset, bpr, length)
    };
    let one_row = |offset, row_bytes| BufferLayout {
        offset,
        row_pitch: row_bytes,
        slice_pitch: row_bytes,
        row_bytes,
        rows: 1,
        slices: 1,
    };
    let rows = |offset, pitch, row_bytes, rows| BufferLayout {
        offset,
        row_pitch: pitch,
        slice_pitch: pitch * rows,
        row_bytes,
        rows,
        slices: 1,
    };
    let (bgra, r8) = (P::BGRA8Unorm, P::R8Unorm);
    let (tb, d1, d2) = (
        TextureType::TextureBuffer,
        TextureType::Type1D,
        TextureType::Type2D,
    );
    let a = TextureType::Type2DArray;
    // Made, each laid out as the 2D texture's rows from the plane's offset.
    let made = [
        // 35 bgra.typebuffer, over a surface without planes.
        (
            plane(tb, bgra, (64, 1), (0, 256), false, 16384),
            one_row(0, 256),
        ),
        // 121 bgra1.typebuffer, and one past its row over a planar surface's plane.
        (
            plane(tb, bgra, (64, 1), (0, 256), true, 16384),
            one_row(0, 256),
        ),
        (
            plane(tb, bgra, (128, 1), (0, 256), true, 32768),
            one_row(0, 512),
        ),
        // 129 420v.p0.typebuffer, at the x86 Macs' plane offset of 32.
        (
            plane(tb, r8, (64, 1), (32, 64), true, 6432),
            one_row(32, 64),
        ),
        // 116 bgra1.type1d.
        (
            plane(d1, bgra, (64, 1), (0, 256), true, 16384),
            one_row(0, 256),
        ),
        // 117 bgra1.type2darray and 127 420v.p0.type2darray, of one slice.
        (
            plane(a, bgra, (64, 64), (0, 256), true, 16384),
            rows(0, 256, 256, 64),
        ),
        (
            plane(a, r8, (64, 64), (32, 64), true, 6432),
            rows(32, 64, 64, 64),
        ),
    ];
    for (case, at) in made {
        let placed = placed(case).unwrap_or_else(|refusal| panic!("{case:?}: {refusal:?}"));
        // One row's pitch is no part of where its texels lie.
        let pitch = |layout: BufferLayout| (layout.rows > 1).then_some(layout.row_pitch);
        let lies = |layout: BufferLayout| {
            let BufferLayout {
                offset,
                row_bytes,
                rows,
                slices,
                ..
            } = layout;
            (offset, row_bytes, rows, slices, pitch(layout))
        };
        assert_eq!(lies(placed.cpu), lies(at), "{case:?}");
        assert_eq!(lies(placed.gpu), lies(at), "{case:?}");
    }
    let gap = Err(BufferRefusal::PlaneShape);
    let stopped = |problem| Err(BufferRefusal::Metal(problem));
    let refused = [
        // 17 bgra1.65x64 and 21 420v.p1.33x32: rows wider than the plane's.
        (plane(d2, bgra, (65, 64), (0, 256), true, 32768), gap),
        (
            plane(d2, P::RG8Unorm, (33, 32), (4128, 64), true, 6432),
            gap,
        ),
        // 123 bgra1.mips2 and 42 420v.p0.mips2.
        (
            Case {
                levels: 2,
                ..plane(d2, bgra, (64, 64), (0, 256), true, 32768)
            },
            gap,
        ),
        (
            Case {
                levels: 2,
                ..plane(d2, r8, (64, 64), (32, 64), true, 6432)
            },
            gap,
        ),
        // 118 bgra1.type2darray.a2, 120 bgra1.type3d.d2, 119 bgra1.typecube and 122
        // bgra1.type2dms.private.
        (
            Case {
                slices: 2,
                ..plane(a, bgra, (64, 32), (0, 256), true, 32768)
            },
            gap,
        ),
        (
            plane(TextureType::Type3D, bgra, (64, 32), (0, 256), true, 32768),
            gap,
        ),
        (
            plane(TextureType::Cube, bgra, (64, 64), (0, 256), true, 32768),
            gap,
        ),
        (
            plane(
                TextureType::Type2DMultisample,
                bgra,
                (64, 64),
                (0, 256),
                true,
                32768,
            ),
            gap,
        ),
        // A 1D texture wider than the plane's row.
        (plane(d1, bgra, (128, 1), (0, 256), true, 32768), gap),
        // A texture buffer over a plane off 16 bytes, unmeasured.
        (plane(tb, r8, (64, 1), (36, 64), true, 6432), gap),
        // Over a surface without planes Metal stops them: 136 bgra.typebuffer.65, 12 bgra.65x64,
        // 25 bgra.type1d, 27 bgra.type2darray and 38 bgra.mips2.
        (
            plane(tb, bgra, (65, 1), (0, 256), false, 16384),
            stopped(Pr::RowPitch {
                bytes_per_row: 256,
                row_bytes: 260,
            }),
        ),
        (
            plane(d2, bgra, (65, 64), (0, 256), false, 16384),
            stopped(Pr::RowPitch {
                bytes_per_row: 256,
                row_bytes: 260,
            }),
        ),
        (
            plane(d1, bgra, (64, 1), (0, 256), false, 16384),
            stopped(Pr::TextureType { texture_type: d1 }),
        ),
        (
            plane(a, bgra, (64, 64), (0, 256), false, 16384),
            stopped(Pr::TextureType { texture_type: a }),
        ),
        (
            Case {
                levels: 2,
                ..plane(d2, bgra, (64, 64), (0, 256), false, 16384)
            },
            stopped(Pr::MipmapLevelCount { count: 2 }),
        ),
    ];
    for (case, expected) in refused {
        assert_eq!(placed(case).map(|_| ()), expected, "{case:?}");
    }
}

/// A BC texture over a surface's rows, which the x86 Macs make (`oracle_surftex` 66, 67, 125,
/// 131, 147, 148) and lay out a row of blocks to the surface's `bytesPerRow`
/// (`surface_texture_envelope`): its rows are rows of blocks, each block its format's bytes, the
/// last of a row the texture's last texels alone; over a surface without planes Metal stops rows
/// of blocks wider than the surface's and rows of texels past them (132 to 135). Every other
/// compressed format over a surface stays no buffer's.
#[test]
fn a_bc_texture_over_a_surface_lies_a_row_of_blocks_a_row() {
    use BufferTextureProblem as Pr;
    let bc = |pf, (width, height), (offset, bpr), planar, length| Case {
        call: BufferTextureCall::IoSurfacePlane { planar },
        pixel_format: pf,
        ..texture(width, height, offset, bpr, length)
    };
    let blocks = |offset, pitch, row_bytes, rows| BufferLayout {
        offset,
        row_pitch: pitch,
        slice_pitch: pitch * rows,
        row_bytes,
        rows,
        slices: 1,
    };
    let made = [
        // 66 bgra.fmt.bc1 and 148 bgra.fmt.bc1.private (storage is no part of the plan).
        (
            bc(P::BC1RGBA, (64, 64), (0, 256), false, 16384),
            blocks(0, 256, 128, 16),
        ),
        // 67 bgra.fmt.bc7.
        (
            bc(P::BC7RGBAUnorm, (64, 64), (0, 256), false, 16384),
            blocks(0, 256, 256, 16),
        ),
        // 131 bgra.fmt.bc1.128x64: a row of blocks as long as the surface's.
        (
            bc(P::BC1RGBA, (128, 64), (0, 256), false, 16384),
            blocks(0, 256, 256, 16),
        ),
        // 147 bgra.fmt.bc1.62x64: its last block of a row partly the texture's.
        (
            bc(P::BC1RGBA, (62, 64), (0, 256), false, 16384),
            blocks(0, 256, 128, 16),
        ),
        // The padded surface's 512-byte rows (surface_texture_envelope bgrapad.bc1).
        (
            bc(P::BC1RGBA, (64, 64), (0, 512), false, 32768),
            blocks(0, 512, 128, 16),
        ),
        // 125 bgra1.fmt.bc1, over a planar surface's plane, and over the x86 Macs' 420v
        // planes at 32 and 4128.
        (
            bc(P::BC1RGBA, (64, 64), (0, 256), true, 16384),
            blocks(0, 256, 128, 16),
        ),
        (
            bc(P::BC1RGBA, (16, 64), (32, 64), true, 6432),
            blocks(32, 64, 32, 16),
        ),
        (
            bc(P::BC1RGBA, (16, 32), (4128, 64), true, 6432),
            blocks(4128, 64, 32, 8),
        ),
        // The rest of the family, of 8- and 16-byte blocks.
        (
            bc(P::BC4RSnorm, (64, 64), (0, 256), false, 16384),
            blocks(0, 256, 128, 16),
        ),
        (
            bc(P::BC6HRGBFloat, (64, 64), (0, 256), false, 16384),
            blocks(0, 256, 256, 16),
        ),
    ];
    for (case, at) in made {
        let placed = placed(case).unwrap_or_else(|refusal| panic!("{case:?}: {refusal:?}"));
        assert_eq!(placed.cpu, at, "{case:?}");
        assert_eq!(placed.gpu, at, "{case:?}");
    }
    let stopped = |problem| Err(BufferRefusal::Metal(problem));
    let refused = [
        // 132 bgra.fmt.bc1.132x64 and 133 bgra.fmt.bc7.68x64: "Descriptor rowBytes (264) >
        // IOSurface rowBytes (256)".
        (
            bc(P::BC1RGBA, (132, 64), (0, 256), false, 16384),
            stopped(Pr::RowPitch {
                bytes_per_row: 256,
                row_bytes: 264,
            }),
        ),
        (
            bc(P::BC7RGBAUnorm, (68, 64), (0, 256), false, 16384),
            stopped(Pr::RowPitch {
                bytes_per_row: 256,
                row_bytes: 272,
            }),
        ),
        // 135 bgra.fmt.bc1.64x68 and 134 bgra.fmt.bc1.64x256: "Descriptor height*arrayLength >
        // IOSurface height", counted in rows of texels.
        (
            bc(P::BC1RGBA, (64, 68), (0, 256), false, 16384),
            stopped(Pr::PastBuffer {
                counted: 67 * 256 + 128,
                length: 16384,
            }),
        ),
        (
            bc(P::BC1RGBA, (64, 256), (0, 256), false, 16384),
            stopped(Pr::PastBuffer {
                counted: 255 * 256 + 128,
                length: 16384,
            }),
        ),
        // Over a plane: rows of blocks wider than its rows, another type, an offset off 16.
        (
            bc(P::BC1RGBA, (132, 64), (0, 256), true, 32768),
            Err(BufferRefusal::PlaneShape),
        ),
        (
            Case {
                texture_type: TextureType::Type2DArray,
                ..bc(P::BC1RGBA, (64, 64), (0, 256), true, 16384)
            },
            Err(BufferRefusal::PlaneShape),
        ),
        (
            bc(P::BC1RGBA, (16, 16), (40, 64), true, 6432),
            Err(BufferRefusal::PlaneShape),
        ),
        // Past the buffer standing for a planar surface.
        (
            bc(P::BC1RGBA, (16, 256), (4128, 64), true, 6432),
            Err(BufferRefusal::TexelsPastLength),
        ),
        // ASTC and ETC2 over a surface: no buffer's, measured or not.
        (
            bc(P::ASTC4x4LDR, (64, 64), (0, 256), false, 16384),
            stopped(Pr::NotLinear {
                pixel_format: P::ASTC4x4LDR,
            }),
        ),
        (
            bc(P::ETC2RGB8, (64, 64), (0, 256), true, 16384),
            stopped(Pr::NotLinear {
                pixel_format: P::ETC2RGB8,
            }),
        ),
        // Over a buffer, no BC texture at all.
        (
            Case {
                call: BufferTextureCall::Texture,
                ..bc(P::BC1RGBA, (64, 64), (0, 256), false, 16384)
            },
            stopped(Pr::NotLinear {
                pixel_format: P::BC1RGBA,
            }),
        ),
    ];
    for (case, expected) in refused {
        assert_eq!(placed(case).map(|_| ()), expected, "{case:?}");
    }
}

/// A BC texture's shadow copies its rows of blocks as the texels they hold: one region a slice,
/// its row length the blocks of the surface's pitch in texels, its extent the texture's own
/// texels, a last block of a row partly past them (`VUID-vkCmdCopyBufferToImage-imageExtent-00207`
/// lets that region end at the image's edge).
#[test]
fn a_bc_shadow_copies_its_rows_of_blocks_as_texels() {
    let layout = |row_bytes, rows| BufferLayout {
        offset: 0,
        row_pitch: 256,
        slice_pitch: 256 * rows,
        row_bytes,
        rows,
        slices: 1,
    };
    // (block bytes, the texture's texels, its layout, the row length and extent copied)
    let cases = [
        (8, [64, 64], layout(128, 16), 128, [64, 64]),
        (8, [62, 64], layout(128, 16), 128, [62, 64]),
        (8, [64, 62], layout(128, 16), 128, [64, 62]),
        (16, [64, 64], layout(256, 16), 64, [64, 64]),
        (8, [128, 64], layout(256, 16), 128, [128, 64]),
    ];
    for (bytes, texels, layout, row_length, [width, height]) in cases {
        let blocks = Blocks {
            extent: [4, 4],
            texels,
        };
        let (into, back) = plane_copy_regions(
            &layout,
            (bytes, Some(blocks)),
            64,
            vk::ImageAspectFlags::COLOR,
        );
        let regions = |r: &[vk::BufferImageCopy]| {
            (r.iter())
                .map(|r| (r.buffer_offset, r.buffer_row_length, r.image_extent.width))
                .collect::<Vec<_>>()
        };
        assert_eq!(regions(&into), regions(&back), "{texels:?}");
        let [region] = into.as_slice() else {
            panic!("{texels:?}: {into:?}");
        };
        assert_eq!(region.buffer_offset, 64, "{texels:?}");
        assert_eq!(region.buffer_row_length, row_length, "{texels:?}");
        assert_eq!(
            (region.image_extent.width, region.image_extent.height),
            (width, height),
            "{texels:?}"
        );
        assert_eq!(
            (region.image_offset.x, region.image_offset.y),
            (0, 0),
            "{texels:?}"
        );
    }
    // An uncompressed texture's region is its rows of texels.
    let (into, _) = plane_copy_regions(&layout(256, 64), (4, None), 0, vk::ImageAspectFlags::COLOR);
    assert_eq!(
        (
            into[0].buffer_row_length,
            into[0].image_extent.width,
            into[0].image_extent.height
        ),
        (64, 64, 64)
    );
}

/// A render target's linear image serves it where the device enabled `linearColorAttachment` and
/// its formats' linear tiling has `LINEAR_COLOR_ATTACHMENT_NV` (the NVIDIA profiles), which the
/// facts read as the color attachment feature under the extension's rule; without the feature
/// enabled it does not. A tiling with the color attachment feature itself renders beside any
/// attachment, so no format is under the rule there.
#[test]
fn a_render_target_s_linear_image_serves_it_by_the_linear_attachment_rule() {
    let mut ruled = 0;
    for (name, desc, device) in committed() {
        let off = facts(&desc, &device);
        let on = super::super::image_plan::ImageFacts {
            linear_attachments: true,
            ..off
        };
        let reported = desc
            .formats
            .formats
            .get(&Format::B8G8R8A8_UNORM)
            .map(|p| p.linear_tiling)
            .unwrap_or_default();
        let bit = reported.contains(FormatFeatureFlags2::LINEAR_COLOR_ATTACHMENT_NV);
        let core = reported.contains(FormatFeatureFlags2::COLOR_ATTACHMENT);
        assert!(
            !off.linear_attachment_rule(Format::B8G8R8A8_UNORM),
            "{name}"
        );
        assert_eq!(
            on.linear_attachment_rule(Format::B8G8R8A8_UNORM),
            bit && !core,
            "{name}"
        );
        let usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
        let layout = TextureDescriptor {
            pixel_format: P::BGRA8Unorm,
            width: 1920,
            height: 1080,
            usage,
            ..TextureDescriptor::default()
        }
        .check(APPLE)
        .unwrap();
        let Ok(plan) = plan_image(&layout, usage, false, &on) else {
            continue;
        };
        if !(bit && !core) {
            assert_eq!(
                linear_lack(&plan, &on).is_none(),
                linear_lack(&plan, &off).is_none(),
                "{name}"
            );
            continue;
        }
        ruled += 1;
        assert!(
            plan.usage.contains(ImageUsageFlags::COLOR_ATTACHMENT),
            "{name}"
        );
        assert!(
            linear_lack(&plan, &off).is_some(),
            "{name}: served without the feature"
        );
        assert!(
            linear_lack(&plan, &on).is_none(),
            "{name}: {:?}",
            linear_lack(&plan, &on)
        );
    }
    assert!(
        ruled >= 2,
        "the NVIDIA profiles render linear images by the rule"
    );
}

/// Every format a buffer can back has a texel size that divides 16, so Metal's GPU pitch for an
/// IOSurface plane (its `bytesPerRow` rounded down to 16) is a whole number of texels.
#[test]
fn every_backed_format_s_texel_divides_the_linear_alignment() {
    let mut backed = 0;
    for &pf in P::ALL {
        if !is_linear(pf) {
            continue;
        }
        let Some(texel) = byte_format(pf)
            .and_then(|format| format.block())
            .map(|block| u64::from(block.size))
        else {
            continue;
        };
        backed += 1;
        assert!(
            texel != 0 && LINEAR_ALIGNMENT.is_multiple_of(texel),
            "{pf:?}: a texel of {texel} bytes does not divide {LINEAR_ALIGNMENT}"
        );
    }
    assert!(
        backed > 50,
        "the sweep took {backed} formats a buffer backs"
    );
}

/// How far the rows of an IOSurface plane overlap, and that the copies that hold such a plane
/// tile its bytes exactly once. A plane's `bytesPerRow` covers its own row
/// (`BufferTextureProblem::RowPitch`), so rounding it down to 16 takes under 16 bytes off: the
/// rows overlap by 1 to 15 bytes, a whole number of texels (above), and between consecutive rows
/// alone, since twice the GPU pitch is always past one row. A plane whose rows overlap further is
/// a two-plane YCbCr texture's rows of codes, which no row rule holds (`ycbcr.rs`).
///
/// For every such plane the sweep takes the shadow's copy regions (`plane_copy_regions`) and the
/// staged copy back (`staged_rows_back`) and holds them to what Vulkan allows and to Metal's
/// order:
///
/// - a copy in writes each of the image's texels once, each from the byte its own row holds it
///   at, so no two destination regions write one texel
///   (`VUID-vkCmdCopyBufferToImage-pRegions-12483`), and every region's `bufferRowLength` covers
///   its width (`-bufferRowLength-00195`);
/// - a copy back writes each of the plane's bytes once, so no two destination regions write one
///   byte (`VUID-vkCmdCopyImageToBuffer-pRegions-12485`), and a shared byte comes from the
///   earlier of the two rows that hold it, the order Metal's blit writes them in;
/// - a staged copy back's destination ranges never overlap
///   (`VUID-vkCmdCopyBuffer-pRegions-12482`) and keep each row's distance from the first byte.
///
/// A plane a render pass stores into is refused instead, at exactly the shapes whose rows
/// overlap.
#[test]
fn a_plane_whose_rows_overlap_is_copied_through_two_regions_that_tile_it_once() {
    let formats = [
        (P::R8Unorm, 1),
        (P::RG8Unorm, 2),
        (P::RGBA8Unorm, 4),
        (P::RGBA16Uint, 8),
        (P::RGBA32Uint, 16),
    ];
    // The staging buffer's first byte and the texels' first byte in their own `VkBuffer`, which
    // the staged copy back moves between.
    let (start, at) = (64, 48);
    let (mut overlapping, mut apart, mut refused) = (0, 0, 0);
    for (pf, texel) in formats {
        for width in 1..=40 {
            let row_bytes = width * texel;
            for bytes_per_row in row_bytes..row_bytes + 48 {
                for height in 2..=3 {
                    let case = Case {
                        call: BufferTextureCall::IoSurfacePlane { planar: true },
                        pixel_format: pf,
                        width,
                        height,
                        bytes_per_row,
                        ..texture(width, height, 0, bytes_per_row, 1 << 20)
                    };
                    let pitch = bytes_per_row - bytes_per_row % LINEAR_ALIGNMENT;
                    let made = placed(case).unwrap_or_else(|e| panic!("{case:?}: {e:?}"));
                    let gpu = made.gpu;
                    let shared = gpu.shared_bytes();
                    let target = Case {
                        usage: TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ,
                        ..case
                    };
                    let (into, back) =
                        plane_copy_regions(&gpu, (texel, None), start, vk::ImageAspectFlags::COLOR);
                    if shared == 0 {
                        apart += 1;
                        assert!(made.cpu.row_pitch >= made.cpu.row_bytes, "{case:?}");
                        let same = |r: &[vk::BufferImageCopy]| {
                            (r.iter())
                                .map(|r| {
                                    let e = r.image_extent;
                                    (
                                        r.buffer_offset,
                                        r.buffer_row_length,
                                        r.image_offset.x,
                                        (e.width, e.height, e.depth),
                                    )
                                })
                                .collect::<Vec<_>>()
                        };
                        assert_eq!(same(&into), same(&back), "{case:?}");
                        assert_eq!(into.len(), 1, "{case:?}");
                        assert!(placed(target).is_ok(), "{case:?}: a render target refused");
                        continue;
                    }
                    overlapping += 1;
                    refused += 1;
                    assert_eq!(
                        placed(target).err(),
                        Some(BufferRefusal::OverlappingRows),
                        "{case:?}: a render target over rows that overlap"
                    );
                    assert!(
                        (1..LINEAR_ALIGNMENT).contains(&shared)
                            && shared.is_multiple_of(texel)
                            && pitch >= LINEAR_ALIGNMENT
                            && gpu.row_pitch == pitch
                            && 2 * pitch > row_bytes,
                        "{pf:?} {width}x{height} at {bytes_per_row} bytes a row: \
                         {shared} bytes of overlap at a GPU pitch of {pitch}"
                    );
                    hold_the_copies(
                        &gpu,
                        texel,
                        (start, at),
                        (&into, &back),
                        &format!("{case:?}"),
                    );
                }
            }
        }
    }
    assert_eq!((overlapping, apart, refused), (1608, 17592, 1608));
}

/// A shadow's copy regions for a plane laid out as `gpu`, `texel` bytes a texel, its texels at
/// `start` of the buffer the copies address and at `at` of the texture's own: each of the image's
/// texels copied in once from the byte its own row holds it at, each of the plane's bytes copied
/// back once and a shared byte from the earlier of its two rows, and a staged copy back whose
/// destination ranges never overlap (the sweep's rules). `case` names the plane in a failure.
fn hold_the_copies(
    gpu: &BufferLayout,
    texel: u64,
    (start, at): (u64, u64),
    (into, back): (&[vk::BufferImageCopy], &[vk::BufferImageCopy]),
    case: &str,
) {
    use std::collections::BTreeMap;
    let span = gpu.end() - gpu.offset;
    let width = gpu.row_bytes / texel;
    assert_eq!((into.len(), back.len()), (2, 2), "{case}");
    for region in into.iter().chain(back) {
        assert!(
            u64::from(region.buffer_row_length) >= u64::from(region.image_extent.width),
            "{case}: a row length of {} under a width of {}",
            region.buffer_row_length,
            region.image_extent.width
        );
    }
    // Going in: every texel once, from its own row's byte.
    let mut read: BTreeMap<(u32, u32), u64> = BTreeMap::new();
    for region in into {
        for (byte, texel_at) in copied_bytes(region, texel) {
            if byte % texel == 0 {
                assert!(
                    read.insert(texel_at, byte).is_none(),
                    "{case}: texel {texel_at:?} copied in twice"
                );
            }
        }
    }
    assert_eq!(read.len() as u64, width * gpu.rows, "{case}");
    for ((x, y), byte) in &read {
        let want = start + u64::from(*y) * gpu.row_pitch + u64::from(*x) * texel;
        assert_eq!(*byte, want, "{case}: texel {:?} read at {byte}", (x, y));
    }
    // Going back: every byte of the plane once, a shared byte from the earlier of its two rows.
    let mut written: BTreeMap<u64, (u32, u32)> = BTreeMap::new();
    for region in back {
        for (byte, texel_at) in copied_bytes(region, texel) {
            assert!(
                written.insert(byte, texel_at).is_none(),
                "{case}: byte {byte} written back twice"
            );
        }
    }
    assert_eq!(written.len() as u64, span, "{case}");
    for (byte, from) in &written {
        assert_eq!(*from, earliest_row(gpu, texel, byte - start), "{case}");
    }
    // The staged copy back, whose ranges lie in the buffer as the plane's bytes do.
    let mut staged: BTreeMap<u64, u64> = BTreeMap::new();
    for copy in staged_rows_back(gpu, start, at) {
        assert_eq!(copy.dst_offset - at, copy.src_offset - start, "{case}");
        for byte in 0..copy.size {
            assert!(
                staged
                    .insert(copy.dst_offset + byte, copy.src_offset)
                    .is_none(),
                "{case}: byte {} written twice by the staged copy back",
                copy.dst_offset + byte
            );
        }
    }
    assert_eq!(staged.len() as u64, span, "{case}");
}

/// Each byte `region` moves between a buffer and an image, paired with the image texel it belongs
/// to: its region-relative texel `(i, j)` lies at `bufferOffset + (j * bufferRowLength + i) *
/// texel` and is image texel `(imageOffset.x + i, j)` (Vulkan's Buffer and Image Addressing).
fn copied_bytes(region: &vk::BufferImageCopy, texel: u64) -> Vec<(u64, (u32, u32))> {
    let extent = region.image_extent;
    let row = u64::from(region.buffer_row_length);
    let x0 = region.image_offset.x as u32;
    let mut bytes = Vec::new();
    for j in 0..extent.height {
        for i in 0..extent.width {
            let first = region.buffer_offset + (u64::from(j) * row + u64::from(i)) * texel;
            for byte in 0..texel {
                bytes.push((first + byte, (x0 + i, j)));
            }
        }
    }
    bytes
}

/// The texel of the earliest row of a plane laid out as `gpu` that holds the byte `offset` bytes
/// into it, `texel` bytes a texel: the row whose writes Metal's blit leaves where two meet.
fn earliest_row(gpu: &BufferLayout, texel: u64, offset: u64) -> (u32, u32) {
    for row in 0..gpu.rows {
        let first = row * gpu.row_pitch;
        if (first..first + gpu.row_bytes).contains(&offset) {
            return (((offset - first) / texel) as u32, row as u32);
        }
    }
    panic!("byte {offset} lies in no row of {gpu:?}");
}
