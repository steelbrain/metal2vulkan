//! Texture views: Metal's rules as measured on an Apple M3 Max and M4, the swizzle composition
//! against Metal's measured texels, and the Vulkan views of bindings.

use super::super::image_plan::srgb_twin;
use super::super::texture_descriptor::{TextureDescriptor, TextureSwizzleChannels};
use super::super::view_plan::{
    check_view, keeps_components, reinterpretation_refused, ViewProblem, ViewRecord, ViewRefusal,
    ViewRequest,
};
use super::image_plan::committed;
use super::texture_descriptor::APPLE;
use metal2vulkan::device::{Device, DeviceDescription, FormatUsage};
use metal2vulkan::error::{Feature, Needs};
use metal2vulkan::metal::format::vk_format;
use metal2vulkan::metal::{PixelFormat, TextureSwizzle, TextureType, TextureUsage};
use metal2vulkan::vk::{ComponentMapping, ComponentSwizzle};

fn lavapipe() -> (DeviceDescription, Device) {
    let (_, desc, device) = committed()
        .into_iter()
        .find(|(n, ..)| n == "lavapipe")
        .unwrap();
    (desc, device)
}

fn record(desc: TextureDescriptor) -> ViewRecord {
    let layout = desc.check(APPLE).expect("a valid descriptor");
    ViewRecord::of_texture(&layout, desc.usage, desc.swizzle)
}

fn texture(pf: PixelFormat, usage: TextureUsage) -> ViewRecord {
    record(TextureDescriptor {
        pixel_format: pf,
        width: 8,
        height: 8,
        usage,
        ..TextureDescriptor::default()
    })
}

/// The image's levels and layers `record` presents.
fn ranges(record: &ViewRecord) -> (std::ops::Range<u32>, std::ops::Range<u32>) {
    (
        record.base_level..record.base_level + record.level_count,
        record.base_layer..record.base_layer + record.layer_count,
    )
}

fn swizzle(s: &str) -> TextureSwizzleChannels {
    let c = |ch: char| match ch {
        'R' => TextureSwizzle::Red,
        'G' => TextureSwizzle::Green,
        'B' => TextureSwizzle::Blue,
        'A' => TextureSwizzle::Alpha,
        '0' => TextureSwizzle::Zero,
        _ => TextureSwizzle::One,
    };
    let v: Vec<TextureSwizzle> = s.chars().map(c).collect();
    TextureSwizzleChannels::new(v[0], v[1], v[2], v[3])
}

/// The texel a view with component mapping `m` returns for the channels `texel` its format
/// gives, in bytes (`ONE` as 255).
fn apply(m: ComponentMapping, texel: [u8; 4]) -> [u8; 4] {
    use ComponentSwizzle as S;
    let pick = |c: ComponentSwizzle, own: usize| match c {
        S::IDENTITY => texel[own],
        S::ZERO => 0,
        S::ONE => 255,
        S::R => texel[0],
        S::G => texel[1],
        S::B => texel[2],
        _ => texel[3],
    };
    [pick(m.r, 0), pick(m.g, 1), pick(m.b, 2), pick(m.a, 3)]
}

/// The bytes of an `RGBA8Unorm` texel as `BGRA8Unorm` reads them.
fn as_bgra(bytes: [u8; 4]) -> [u8; 4] {
    [bytes[2], bytes[1], bytes[0], bytes[3]]
}

#[test]
fn a_texture_record_presents_its_whole_image() {
    let r = record(TextureDescriptor {
        texture_type: TextureType::CubeArray,
        width: 16,
        height: 16,
        array_length: 2,
        mipmap_level_count: 3,
        usage: TextureUsage::SHADER_READ | TextureUsage::PIXEL_FORMAT_VIEW,
        ..TextureDescriptor::default()
    });
    assert_eq!(ranges(&r), (0..3, 0..12));
    assert_eq!((r.array_length(), r.extent()), (2, [16, 16, 1]));
}

/// Every view-format pair probe 2 measured, with plain Metal's verdict: Metal's API validation also
/// stops the pairs `plain` marks on a texture without `PixelFormatView` usage, which plain Metal
/// makes (Metal probes on the M3 Max and the M4).
#[test]
fn view_formats_follow_metals_measured_verdicts() {
    use PixelFormat as P;
    let (_, device) = lavapipe();
    let ok = Ok(());
    let plain = Ok(());
    let no = Err(ViewProblem::FormatIncompatible);
    let rgba8_same_size = [
        P::R32Uint,
        P::R32Float,
        P::RG16Float,
        P::RGB10A2Unorm,
        P::RG11B10Float,
        P::RGB9E5Float,
        P::RGBA8Uint,
        P::RGBA8Snorm,
        P::RG16Uint,
        P::BGRA8Unorm,
        P::BGRA8UnormSRGB,
    ];
    let mut cases: Vec<(P, bool, P, Result<(), ViewProblem>)> = Vec::new();
    for &v in &rgba8_same_size {
        cases.push((P::RGBA8Unorm, true, v, ok));
        cases.push((P::RGBA8Unorm, false, v, plain));
    }
    for flag in [false, true] {
        cases.extend([
            (P::RGBA8Unorm, flag, P::RGBA8UnormSRGB, ok),
            (P::RGBA8Unorm, flag, P::R16Uint, no),
            (P::RGBA8Unorm, flag, P::RGBA16Float, no),
            (P::BC1RGBA, flag, P::BC1RGBASRGB, ok),
            (P::BC1RGBA, flag, P::BC4RUnorm, no),
            (P::BC1RGBA, flag, P::RG32Uint, no),
            (P::Depth32FloatStencil8, flag, P::Depth32Float, no),
            (P::Depth32Float, flag, P::R32Float, no),
            (P::R32Float, flag, P::Depth32Float, no),
        ]);
    }
    // Probe 3.
    for (src, dst) in [
        (P::BC4RUnorm, P::BC4RSnorm),
        (P::BC5RGUnorm, P::BC5RGSnorm),
        (P::BC6HRGBFloat, P::BC6HRGBUfloat),
        (P::B5G6R5Unorm, P::R16Uint),
        (P::R16Uint, P::B5G6R5Unorm),
        (P::R8Unorm, P::A8Unorm),
        (P::RGBA8Unorm, P::RG16Snorm),
    ] {
        cases.push((src, true, dst, ok));
        cases.push((src, false, dst, plain));
    }
    for flag in [false, true] {
        cases.push((P::Depth16Unorm, flag, P::R16Unorm, no));
        cases.push((P::R16Unorm, flag, P::Depth16Unorm, no));
    }
    cases.extend([
        (P::RGBA8UnormSRGB, true, P::RGBA8Unorm, ok),
        (P::RGBA8UnormSRGB, true, P::R32Uint, ok),
        (P::RG11B10Float, true, P::R32Uint, ok),
        (P::RG11B10Float, true, P::RGBA8Unorm, ok),
        (P::RG11B10Float, false, P::R32Uint, plain),
        (P::RG11B10Float, false, P::RGBA8Unorm, plain),
        (P::RGB9E5Float, true, P::R32Uint, ok),
        (P::RGB9E5Float, false, P::R32Uint, plain),
        (P::Depth32FloatStencil8, true, P::X32Stencil8, ok),
        (P::Depth32FloatStencil8, false, P::X32Stencil8, plain),
    ]);
    for (src, flag, dst, want) in cases {
        let usage = if flag {
            TextureUsage::SHADER_READ | TextureUsage::PIXEL_FORMAT_VIEW
        } else {
            TextureUsage::SHADER_READ
        };
        let source = texture(src, usage);
        let got = check_view(
            &source,
            &ViewRequest::of_format(&source, dst),
            device.profile(),
        )
        .map(|v| assert_eq!(v.pixel_format, dst))
        .map_err(|e| match e {
            ViewRefusal::Invalid(p) => p,
            other => panic!("{src:?} as {dst:?}: {other:?}"),
        });
        assert_eq!(got, want, "{src:?} ({flag}) as {dst:?}");
    }
    let source = texture(P::RGBA8Unorm, TextureUsage::SHADER_READ);
    assert_eq!(
        check_view(
            &source,
            &ViewRequest::of_format(&source, P::Invalid),
            device.profile()
        ),
        Err(ViewRefusal::Nil(ViewProblem::NotAFormat))
    );
    // A format with no Vulkan representation is refused as the format table refuses it.
    let source = texture(
        P::RGBA16Float,
        TextureUsage::SHADER_READ | TextureUsage::PIXEL_FORMAT_VIEW,
    );
    match check_view(
        &source,
        &ViewRequest::of_format(&source, P::RGB8Unorm),
        device.profile(),
    ) {
        Err(ViewRefusal::Unsupported(u)) => assert_eq!(u.feature, Feature::Format),
        other => panic!("{other:?}"),
    }
    // Metal views an extended-range format as its sRGB twin alone, and no other format as one,
    // though their bytes per pixel match (the consumer's report, an M3 Max: 116 pairs abort;
    // the two-plane ones' 72 pairs abort on an M4).
    let xr = [
        P::BGRA10XR,
        P::BGRA10XRSRGB,
        P::BGR10XR,
        P::BGR10XRSRGB,
        P::RGB10A82PXR10,
        P::RGB10A82PXR10SRGB,
    ];
    let others = [
        P::RGBA16Float,
        P::RG32Float,
        P::RGBA16Uint,
        P::RGBA8Unorm,
        P::R32Uint,
        P::RG11B10Float,
        P::BGR10A2Unorm,
    ];
    let incompatible = Err(ViewRefusal::Invalid(ViewProblem::FormatIncompatible));
    for format in xr {
        let usage = TextureUsage::SHADER_READ | TextureUsage::PIXEL_FORMAT_VIEW;
        let source = texture(format, usage);
        for other in others {
            let view = ViewRequest::of_format(&source, other);
            assert_eq!(
                check_view(&source, &view, device.profile()).map(|_| ()),
                incompatible,
                "{format:?} as {other:?}"
            );
            let from = texture(other, usage);
            let view = ViewRequest::of_format(&from, format);
            assert_eq!(
                check_view(&from, &view, device.profile()).map(|_| ()),
                incompatible,
                "{other:?} as {format:?}"
            );
        }
        for view in xr {
            let made = check_view(
                &source,
                &ViewRequest::of_format(&source, view),
                device.profile(),
            );
            let twins = view == format || srgb_twin(format) == Some(view);
            // A two-plane texture's view as its twin, which Metal makes, is the runtime's gap.
            let two_plane_twins = format != view
                && [format, view]
                    .iter()
                    .all(|&pf| matches!(pf, P::RGB10A82PXR10 | P::RGB10A82PXR10SRGB));
            match made {
                Err(ViewRefusal::Unsupported(u)) if two_plane_twins => {
                    assert_eq!(u.feature, Feature::Format, "{format:?} as {view:?}")
                }
                made => assert_eq!(made.is_ok(), twins, "{format:?} as {view:?}"),
            }
        }
    }
}

/// The one depth or stencil view of another format than its texture's that a texture of a format
/// Metal makes takes is the stencil view of a combined texture (`X32Stencil8` of
/// `Depth32FloatStencil8`), and its reverse, which `Texture::new_texture_view` stops: so the
/// blits' and the CPU copies' refusal of another (`NotImplemented(Textures)`) meets no view.
#[test]
fn the_only_depth_view_of_another_format_is_the_stencil_view_of_a_combined_texture() {
    use metal2vulkan::metal::format::is_depth_stencil;
    use PixelFormat as P;
    let (_, device) = lavapipe();
    let depth: Vec<P> = P::ALL
        .iter()
        .copied()
        .filter(|&pf| is_depth_stencil(pf))
        .collect();
    let mut made = Vec::new();
    for &format in &depth {
        let usage = TextureUsage::SHADER_READ | TextureUsage::PIXEL_FORMAT_VIEW;
        let descriptor = TextureDescriptor {
            pixel_format: format,
            width: 8,
            height: 8,
            usage,
            ..TextureDescriptor::default()
        };
        let Ok(layout) = descriptor.check(APPLE) else {
            continue;
        };
        let source = ViewRecord::of_texture(&layout, usage, descriptor.swizzle);
        for view in P::ALL.iter().copied().filter(|&pf| pf != format) {
            let request = ViewRequest::of_format(&source, view);
            if check_view(&source, &request, device.profile()).is_ok() {
                made.push((format, view));
            }
        }
    }
    assert_eq!(
        made,
        [
            (P::Depth32FloatStencil8, P::X32Stencil8),
            (P::X32Stencil8, P::Depth32FloatStencil8),
        ]
    );
}

/// The texture-type pairs probe 2 measured.
#[test]
fn view_types_follow_metals_measured_verdicts() {
    use TextureType as T;
    let (_, device) = lavapipe();
    let source = |ty: T, slices: u64| {
        let (h, d) = match ty {
            T::Type1D | T::Type1DArray | T::TextureBuffer => (1, 1),
            T::Type3D => (4, 4),
            _ => (4, 1),
        };
        record(TextureDescriptor {
            texture_type: ty,
            width: 4,
            height: h,
            depth: d,
            array_length: slices,
            sample_count: if matches!(ty, T::Type2DMultisample) {
                4
            } else {
                1
            },
            usage: TextureUsage::SHADER_READ,
            ..TextureDescriptor::default()
        })
    };
    let cases = [
        (T::Type2DArray, 6, T::Cube, 0..6, Ok(())),
        (T::Type2DArray, 12, T::CubeArray, 0..12, Ok(())),
        (T::Type2D, 1, T::Type2DArray, 0..1, Ok(())),
        (T::Cube, 1, T::Type2D, 0..1, Ok(())),
        (T::Cube, 1, T::Type2DArray, 0..6, Ok(())),
        (
            T::Type3D,
            1,
            T::Type2D,
            0..1,
            Err(ViewProblem::TypeIncompatible),
        ),
        (
            T::Type3D,
            1,
            T::Type2DArray,
            0..1,
            Err(ViewProblem::TypeIncompatible),
        ),
        (T::Type2D, 1, T::Cube, 0..1, Err(ViewProblem::SliceCount)),
        (T::Type1D, 1, T::Type1DArray, 0..1, Ok(())),
        (
            T::Type2DMultisample,
            1,
            T::Type2DMultisampleArray,
            0..1,
            Ok(()),
        ),
        // A texture buffer views as a texture buffer only; Metal asserts on the others, plain and
        // validated (Metal probes on the M3 Max and an M4).
        (T::TextureBuffer, 1, T::TextureBuffer, 0..1, Ok(())),
        (
            T::TextureBuffer,
            1,
            T::Type2D,
            0..1,
            Err(ViewProblem::TypeIncompatible),
        ),
        (
            T::Type2D,
            1,
            T::TextureBuffer,
            0..1,
            Err(ViewProblem::TypeIncompatible),
        ),
    ];
    for (ty, slices, view, range, want) in cases {
        let s = source(ty, slices);
        let request = ViewRequest {
            texture_type: view,
            levels: 0..1,
            slices: range,
            ..ViewRequest::of_format(&s, PixelFormat::RGBA8Unorm)
        };
        let got = check_view(&s, &request, device.profile())
            .map(|_| ())
            .map_err(|e| match e {
                ViewRefusal::Invalid(p) => p,
                other => panic!("{other:?}"),
            });
        assert_eq!(got, want, "{ty:?} as {view:?}");
    }
}

#[test]
fn view_ranges_count_from_the_viewed_texture() {
    let (_, device) = lavapipe();
    let t = record(TextureDescriptor {
        texture_type: TextureType::Type2DArray,
        width: 8,
        height: 8,
        array_length: 4,
        mipmap_level_count: 3,
        usage: TextureUsage::SHADER_READ,
        ..TextureDescriptor::default()
    });
    let request = |ty, levels, slices| ViewRequest {
        texture_type: ty,
        levels,
        slices,
        ..ViewRequest::of_format(&t, PixelFormat::RGBA8Unorm)
    };
    let a = check_view(
        &t,
        &request(TextureType::Type2DArray, 1..3, 1..4),
        device.profile(),
    )
    .unwrap();
    assert_eq!((ranges(&a), a.extent()), ((1..3, 1..4), [4, 4, 1]));
    let b = check_view(
        &a,
        &ViewRequest {
            texture_type: TextureType::Type2D,
            levels: 1..2,
            slices: 1..2,
            ..ViewRequest::of_format(&a, PixelFormat::RGBA8Unorm)
        },
        device.profile(),
    )
    .unwrap();
    assert_eq!((ranges(&b), b.extent()), ((2..3, 2..3), [2, 2, 1]));
    for (levels, slices, want) in [
        (1..1, 0..1, ViewProblem::LevelRange),
        (2..4, 0..1, ViewProblem::LevelRange),
        (0..1, 0..0, ViewProblem::SliceRange),
        (0..1, 3..5, ViewProblem::SliceRange),
        (0..1, 0..2, ViewProblem::SliceCount),
    ] {
        assert_eq!(
            check_view(
                &t,
                &request(TextureType::Type2D, levels, slices),
                device.profile()
            ),
            Err(ViewRefusal::Invalid(want))
        );
    }
    // Metal makes a cube view of a 2D array that is not square (probe 3); Vulkan cannot.
    let wide = record(TextureDescriptor {
        texture_type: TextureType::Type2DArray,
        width: 8,
        height: 4,
        array_length: 6,
        ..TextureDescriptor::default()
    });
    let cube = ViewRequest {
        texture_type: TextureType::Cube,
        ..ViewRequest::of_format(&wide, PixelFormat::RGBA8Unorm)
    };
    match check_view(&wide, &cube, device.profile()) {
        Err(ViewRefusal::Unsupported(u)) => {
            assert_eq!(u.feature, Feature::TextureView);
            assert_eq!(u.needs, Needs::NotRepresentable);
        }
        other => panic!("{other:?}"),
    }
    // A cube of the last six of seven slices (probe 3).
    let seven = record(TextureDescriptor {
        texture_type: TextureType::Type2DArray,
        width: 4,
        height: 4,
        array_length: 7,
        ..TextureDescriptor::default()
    });
    let cube = ViewRequest {
        texture_type: TextureType::Cube,
        slices: 1..7,
        ..ViewRequest::of_format(&seven, PixelFormat::RGBA8Unorm)
    };
    assert_eq!(
        ranges(&check_view(&seven, &cube, device.profile()).unwrap()).1,
        1..7
    );
}

/// Probe 2's `usage.views`: a texture of every usage keeps it in format views, subviews and
/// identity-swizzled views, and a swizzled view keeps `ShaderRead | PixelFormatView` (0x11).
#[test]
fn swizzled_views_keep_only_read_and_format_view_usage() {
    let (_, device) = lavapipe();
    let all = TextureUsage::SHADER_READ
        | TextureUsage::SHADER_WRITE
        | TextureUsage::RENDER_TARGET
        | TextureUsage::PIXEL_FORMAT_VIEW;
    let t = texture(PixelFormat::RGBA8Unorm, all);
    let p = device.profile();
    let fmt = check_view(&t, &ViewRequest::of_format(&t, PixelFormat::BGRA8Unorm), p).unwrap();
    assert_eq!(fmt.usage, all);
    let sub = ViewRequest {
        texture_type: TextureType::Type2D,
        levels: 0..1,
        slices: 0..1,
        ..ViewRequest::of_format(&t, PixelFormat::RGBA8Unorm)
    };
    assert_eq!(check_view(&t, &sub, p).unwrap().usage, all);
    let swizzled = ViewRequest {
        swizzle: Some(swizzle("GBAR")),
        ..sub.clone()
    };
    let v = check_view(&t, &swizzled, p).unwrap();
    assert_eq!(v.usage.as_raw(), 0x11);
    assert_eq!(v.swizzle, swizzle("GBAR"));
    let identity = ViewRequest {
        swizzle: Some(swizzle("RGBA")),
        ..sub
    };
    assert_eq!(check_view(&t, &identity, p).unwrap().usage, all);
}

/// Apple's private usage bits in a swizzled view, as the consumer measured on the M3 Max: the
/// view keeps `NoConcurrentAccess` (0x4000) and `BlockWritesOnly` (0x10000) beside `ShaderRead`
/// and `PixelFormatView`, and drops `Video` (0x8000) with the write and render target bits; a
/// view without a swizzle keeps all three.
#[test]
fn swizzled_views_keep_the_private_usage_bits_metal_keeps() {
    let (_, device) = lavapipe();
    let p = device.profile();
    for (made, view) in [(0x1_c01d, 0x1_4011), (0x1_0005, 0x1_0001)] {
        let usage = TextureUsage::try_from(made).unwrap();
        let t = texture(PixelFormat::RGBA8Unorm, usage);
        let sub = ViewRequest {
            texture_type: TextureType::Type2D,
            levels: 0..1,
            slices: 0..1,
            ..ViewRequest::of_format(&t, PixelFormat::RGBA8Unorm)
        };
        assert_eq!(check_view(&t, &sub, p).unwrap().usage, usage, "{made:#x}");
        let swizzled = ViewRequest {
            swizzle: Some(swizzle("GBAR")),
            ..sub
        };
        let v = check_view(&t, &swizzled, p).unwrap();
        assert_eq!(v.usage.without_dropped().as_raw(), view, "{made:#x}");
    }
}

/// Bit 0x8, which Metal's descriptor keeps and its texture drops (an Apple M4):
/// a texture of 0x8 alone answers no usage, as its views without a swizzle do, and Metal asserts
/// on its swizzled view; one of 0x9 answers `ShaderRead`, its swizzled view too. A swizzled view
/// of a texture of `ShaderWrite`, `RenderTarget` or `Video` alone answers no usage, made.
#[test]
fn views_answer_their_usage_without_the_bit_metal_drops() {
    let (_, device) = lavapipe();
    let p = device.profile();
    for (made, swizzled) in [
        (0x8, None),
        (0x9, Some(0x1)),
        (0x19, Some(0x11)),
        (0x2, Some(0x0)),
        (0x4, Some(0x0)),
        (0xc, Some(0x0)),
        (0x8004, Some(0x0)),
    ] {
        let usage = TextureUsage::try_from(made).unwrap();
        let t = texture(PixelFormat::RGBA8Unorm, usage);
        assert_eq!(
            t.usage.without_dropped(),
            usage.without_dropped(),
            "{made:#x}"
        );
        let plain = ViewRequest::of_format(&t, PixelFormat::RGBA8Unorm);
        let view = check_view(&t, &plain, p).unwrap();
        assert_eq!(
            view.usage.without_dropped(),
            usage.without_dropped(),
            "{made:#x}"
        );
        let gbar = ViewRequest {
            swizzle: Some(swizzle("GBAR")),
            ..plain
        };
        let answer = check_view(&t, &gbar, p).map(|v| v.usage.without_dropped().as_raw());
        match swizzled {
            Some(kept) => assert_eq!(answer, Ok(kept), "{made:#x}"),
            None => assert_eq!(
                answer,
                Err(ViewRefusal::Invalid(ViewProblem::SwizzledViewOfNoUsage)),
                "{made:#x}"
            ),
        }
    }
}

/// Probe 3: a swizzled view needs no `PixelFormatView` and keeps `ShaderRead` of a texture
/// without it; `Unknown` usage is Metal's 0x17, of which a swizzled view keeps 0x11 and a format
/// view all; a swizzled view in another format still needs `PixelFormatView`.
#[test]
fn swizzled_views_of_textures_without_format_view_usage() {
    let (_, device) = lavapipe();
    let p = device.profile();
    let sub = |s: &ViewRecord, pf, sw: &str| ViewRequest {
        texture_type: TextureType::Type2D,
        levels: 0..1,
        slices: 0..1,
        swizzle: Some(swizzle(sw)),
        ..ViewRequest::of_format(s, pf)
    };
    let read = texture(PixelFormat::RGBA8Unorm, TextureUsage::SHADER_READ);
    let v = check_view(&read, &sub(&read, PixelFormat::RGBA8Unorm, "GBAR"), p).unwrap();
    assert_eq!(v.usage.as_raw(), 0x1);
    let unknown = texture(PixelFormat::RGBA8Unorm, TextureUsage::empty());
    assert_eq!(unknown.usage.as_raw(), 0x17);
    let v = check_view(&unknown, &sub(&unknown, PixelFormat::RGBA8Unorm, "GBAR"), p).unwrap();
    assert_eq!(v.usage.as_raw(), 0x11);
    let v = check_view(
        &unknown,
        &ViewRequest::of_format(&unknown, PixelFormat::R32Uint),
        p,
    )
    .unwrap();
    assert_eq!(v.usage.as_raw(), 0x17);
    let rw = texture(
        PixelFormat::RGBA8Unorm,
        TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE,
    );
    let v = check_view(&rw, &sub(&rw, PixelFormat::RGBA8Unorm, "GBAR"), p).unwrap();
    assert_eq!(v.usage.as_raw(), 0x1);
    // Another format of the class without `PixelFormatView` usage: plain Metal's view.
    let v = check_view(&read, &sub(&read, PixelFormat::BGRA8Unorm, "GBAR"), p).unwrap();
    assert_eq!(v.pixel_format, PixelFormat::BGRA8Unorm);
}

/// Probe 2's `sw.*` cases: the texel `(10, 20, 30, 40)` read through each texture and view, with
/// the swizzle each ends up with (inherited without a swizzle argument, replaced with one).
#[test]
fn swizzles_inherit_replace_and_compose_as_metal_measured() {
    let (_, device) = lavapipe();
    let p = device.profile();
    let texel = [10, 20, 30, 40];
    let usage = TextureUsage::SHADER_READ | TextureUsage::PIXEL_FORMAT_VIEW;
    let plain = texture(PixelFormat::RGBA8Unorm, usage);
    let swizzled = record(TextureDescriptor {
        pixel_format: PixelFormat::RGBA8Unorm,
        usage,
        swizzle: swizzle("GBAR"),
        ..TextureDescriptor::default()
    });
    let sub = |s: &ViewRecord, sw: Option<&str>| ViewRequest {
        texture_type: TextureType::Type2D,
        levels: 0..1,
        slices: 0..1,
        swizzle: sw.map(swizzle),
        ..ViewRequest::of_format(s, PixelFormat::RGBA8Unorm)
    };
    let view = |s: &ViewRecord, r: &ViewRequest| check_view(s, r, p).unwrap();
    let read = |r: &ViewRecord| {
        let m = vk_format(r.pixel_format, &[FormatUsage::SampledImage].into(), p).unwrap();
        let bytes = if r.pixel_format == PixelFormat::BGRA8Unorm {
            as_bgra(texel)
        } else {
            texel
        };
        // The view format's channels, as its mapping presents them, then the composition.
        let channels = apply(m.components, bytes);
        assert_eq!(
            apply(r.swizzle.compose(m.components), bytes),
            apply(r.swizzle.compose(ComponentMapping::IDENTITY), channels)
        );
        apply(r.swizzle.compose(m.components), bytes)
    };
    let b0r1 = view(&plain, &sub(&plain, Some("B0R1")));
    let cases: Vec<(&str, ViewRecord, [u8; 4])> = vec![
        ("sw.base", plain, [10, 20, 30, 40]),
        ("sw.desc", swizzled, [20, 30, 40, 10]),
        (
            "sw.desc.view",
            view(
                &swizzled,
                &ViewRequest::of_format(&swizzled, PixelFormat::RGBA8Unorm),
            ),
            [20, 30, 40, 10],
        ),
        (
            "sw.desc.subview",
            view(&swizzled, &sub(&swizzled, None)),
            [20, 30, 40, 10],
        ),
        (
            "sw.desc.swview",
            view(&swizzled, &sub(&swizzled, Some("B0R1"))),
            [30, 0, 10, 255],
        ),
        (
            "sw.desc.fmtview",
            view(
                &swizzled,
                &ViewRequest::of_format(&swizzled, PixelFormat::BGRA8Unorm),
            ),
            [20, 10, 40, 30],
        ),
        ("sw.view", b0r1, [30, 0, 10, 255]),
        (
            "sw.view.view",
            view(
                &b0r1,
                &ViewRequest::of_format(&b0r1, PixelFormat::RGBA8Unorm),
            ),
            [30, 0, 10, 255],
        ),
        (
            "sw.view.subview",
            view(&b0r1, &sub(&b0r1, None)),
            [30, 0, 10, 255],
        ),
        (
            "sw.view.swview",
            view(&b0r1, &sub(&b0r1, Some("GBAR"))),
            [20, 30, 40, 10],
        ),
    ];
    for (name, r, want) in cases {
        assert_eq!(read(&r), want, "{name}");
    }
}

/// A Vulkan view keeps its image's components where it has as many of the same width, which is
/// all a device whose portability subset withholds `imageViewFormatReinterpretation` views
/// (`VUID-VkImageViewCreateInfo-imageViewFormatReinterpretation-04466`); every other view of
/// such a device is refused, typed. Formats of mixed widths and block-compressed formats keep only
/// their own, as the format table records no more of them.
#[test]
fn a_view_keeps_its_images_components_of_one_count_and_width() {
    use metal2vulkan::vk::Format as F;
    let keeps = [
        (F::R8G8B8A8_UNORM, F::R8G8B8A8_UNORM),
        (F::R8G8B8A8_UNORM, F::R8G8B8A8_UINT),
        (F::R8G8B8A8_UNORM, F::R8G8B8A8_SRGB),
        (F::R8G8B8A8_SINT, F::R8G8B8A8_UINT),
        (F::R16G16_SFLOAT, F::R16G16_UINT),
        (F::R16G16B16A16_SFLOAT, F::R16G16B16A16_UINT),
        (F::R32_SFLOAT, F::R32_UINT),
        (F::R32G32B32A32_SFLOAT, F::R32G32B32A32_UINT),
        (F::D32_SFLOAT, F::D32_SFLOAT),
        (F::A2B10G10R10_UNORM_PACK32, F::A2B10G10R10_UNORM_PACK32),
        (F::BC1_RGBA_UNORM_BLOCK, F::BC1_RGBA_UNORM_BLOCK),
    ];
    let reinterprets = [
        (F::R8G8B8A8_UNORM, F::R32_UINT),
        (F::R8G8_UNORM, F::R16_UINT),
        (F::R16G16_SFLOAT, F::R32_UINT),
        (F::R16G16B16A16_SFLOAT, F::R32G32_UINT),
        (F::R32G32_SFLOAT, F::R16G16B16A16_UINT),
        (F::R5G6B5_UNORM_PACK16, F::R16_UINT),
        (F::A2B10G10R10_UNORM_PACK32, F::A2B10G10R10_UINT_PACK32),
        (F::BC4_UNORM_BLOCK, F::BC4_SNORM_BLOCK),
    ];
    for (image, view) in keeps {
        assert!(keeps_components(image, view), "{image:?} as {view:?}");
        assert!(keeps_components(view, image), "{view:?} as {image:?}");
    }
    for (image, view) in reinterprets {
        assert!(!keeps_components(image, view), "{image:?} as {view:?}");
        assert!(!keeps_components(view, image), "{view:?} as {image:?}");
    }
    let refused = reinterpretation_refused();
    assert_eq!(refused.feature, Feature::ViewFormatReinterpretation);
    assert_eq!(refused.needs, Needs::NotRepresentable);
}

/// A two-plane YCbCr texture viewed in its own format retyped as a 2D array passes the plan (its
/// view reads the luma codes, `Texture::luma_view`); one in its `_sRGB` twin's is refused, typed
/// (`NotImplemented(TextureView)`, naming the view's format), unmeasured. A view of its own type
/// is made.
#[test]
fn a_ycbcr_texture_retyped_in_its_twin_is_refused_typed() {
    let (_, device) = lavapipe();
    for (pf, view) in [
        (PixelFormat::YCBCR84202P, PixelFormat::YCBCR84202P),
        (PixelFormat::YCBCR84202P, PixelFormat::YCBCR84202PSRGB),
        (PixelFormat::RGB84202P, PixelFormat::RGB84202P),
        (PixelFormat::YCBCR104442P, PixelFormat::YCBCR104442P),
        (PixelFormat::YCBCR104442P, PixelFormat::YCBCR104442PSRGB),
        (PixelFormat::RGB104222P, PixelFormat::RGB104222P),
    ] {
        let mut source = texture(PixelFormat::RGBA8Unorm, TextureUsage::SHADER_READ);
        source.pixel_format = pf;
        let same = ViewRequest::of_format(&source, view);
        assert!(
            check_view(&source, &same, device.profile()).is_ok(),
            "{pf:?} as {view:?}"
        );
        let retyped = ViewRequest {
            texture_type: TextureType::Type2DArray,
            ..ViewRequest::of_format(&source, view)
        };
        match check_view(&source, &retyped, device.profile()) {
            Ok(_) if pf == view => {}
            Err(ViewRefusal::Unsupported(u)) if pf != view => {
                assert_eq!(u.feature, Feature::TextureView, "{pf:?} as {view:?}");
                assert_eq!(u.needs, Needs::NotImplemented, "{pf:?} as {view:?}");
            }
            other => panic!("{pf:?} as {view:?}: {:?}", other.map(|_| ())),
        }
    }
}

/// Plain Metal views an `RGB10A2Unorm_sRGB` texture as `RGB10A2Unorm` and the other order (an
/// Apple M3 Max under macOS 27.0.1); their reads are not measured, so each is refused typed
/// (`NotImplemented(Format)`, naming the view's format), not as a pair Metal stops. A view in its
/// own format is made.
#[test]
fn an_rgb10a2_srgb_view_in_another_format_is_the_runtimes_gap() {
    let (_, device) = lavapipe();
    let usage = TextureUsage::SHADER_READ | TextureUsage::PIXEL_FORMAT_VIEW;
    for (pf, view) in [
        (PixelFormat::RGB10A2UnormSRGB, PixelFormat::RGB10A2Unorm),
        (PixelFormat::RGB10A2Unorm, PixelFormat::RGB10A2UnormSRGB),
    ] {
        let source = texture(pf, usage);
        let own = ViewRequest::of_format(&source, pf);
        assert!(
            check_view(&source, &own, device.profile()).is_ok(),
            "{pf:?}"
        );
        match check_view(
            &source,
            &ViewRequest::of_format(&source, view),
            device.profile(),
        ) {
            Err(ViewRefusal::Unsupported(u)) => {
                assert_eq!(u.feature, Feature::Format, "{pf:?} as {view:?}");
                assert_eq!(u.needs, Needs::NotImplemented, "{pf:?} as {view:?}");
            }
            other => panic!("{pf:?} as {view:?}: {:?}", other.map(|_| ())),
        }
    }
}
