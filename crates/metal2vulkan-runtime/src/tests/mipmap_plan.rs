//! `generateMipmapsForTexture:`: Metal's rules as probes 2 and 3 measured them, the blit chain, and
//! the compute filter where the device's blits cannot filter a format.

use super::super::driver_defects;
use super::super::image_plan::plan_image;
use super::super::mipmap_plan::{
    format_problem, not_implemented, picks, plan_mipmaps, plan_picks, MipBlit, MipmapPlan,
    MipmapProblem, MipmapRefusal, PickPlan,
};
use super::super::texture_descriptor::TextureDescriptor;
use super::super::view_plan::{check_view, ViewRecord, ViewRequest};
use super::image_plan::{committed, facts};
use super::texture_descriptor::APPLE;
use metal2vulkan::bind::MipmapFilterChain;
use metal2vulkan::error::{Feature, Needs};
use metal2vulkan::metal::{PixelFormat, TextureType, TextureUsage};
use metal2vulkan::vk::FormatFeatureFlags2;
use std::collections::HashSet;

fn planned(
    desc: TextureDescriptor,
    profile: &str,
) -> (ViewRecord, Result<MipmapPlan, MipmapRefusal>) {
    let (_, d, device) = committed()
        .into_iter()
        .find(|(n, ..)| n == profile)
        .unwrap();
    let facts = facts(&d, &device);
    let layout = desc.check(APPLE).unwrap();
    let image = plan_image(&layout, desc.usage, false, &facts).unwrap();
    let record = ViewRecord::of_texture(&layout, desc.usage, desc.swizzle);
    let result = plan_mipmaps(&record, &image, &facts, &driver_defects::of(&d));
    (record, result)
}

fn blits(plan: MipmapPlan) -> Vec<MipBlit> {
    match plan {
        MipmapPlan::Blits(blits) => blits,
        MipmapPlan::Picks(picks) => panic!("picks where blits were planned: {picks:?}"),
        MipmapPlan::Filter => panic!("the compute filter where blits were planned"),
    }
}

fn plan(
    desc: TextureDescriptor,
    profile: &str,
) -> (ViewRecord, Result<Vec<MipBlit>, MipmapRefusal>) {
    let (record, result) = planned(desc, profile);
    (record, result.map(blits))
}

fn mipped(pf: PixelFormat, ty: TextureType, extent: [u64; 3], layers: u64) -> TextureDescriptor {
    let mut d = TextureDescriptor {
        texture_type: ty,
        pixel_format: pf,
        width: extent[0],
        height: extent[1],
        depth: extent[2],
        array_length: layers,
        ..TextureDescriptor::default()
    };
    d.mipmap_level_count = super::super::texture_descriptor::full_chain(match ty {
        TextureType::Type3D => &extent[..],
        _ => &extent[..2],
    });
    d
}

#[test]
fn metals_rules_stop_compressed_and_unfilterable_formats() {
    use PixelFormat as P;
    let invalid = |p| Err(MipmapRefusal::Invalid(p));
    // One level of any format, and a depth or stencil texture of any levels, are left as they
    // are, as plain Metal leaves them.
    for pf in [P::RGBA8Unorm, P::RGBA8Uint, P::BC1RGBA, P::Depth32Float] {
        let one = TextureDescriptor {
            pixel_format: pf,
            width: 8,
            height: 8,
            ..TextureDescriptor::default()
        };
        assert_eq!(plan(one, "lavapipe").1, Ok(Vec::new()), "{pf:?}");
    }
    for pf in [
        P::Depth16Unorm,
        P::Depth32Float,
        P::Stencil8,
        P::Depth32FloatStencil8,
    ] {
        let d = mipped(pf, TextureType::Type2D, [8, 8, 1], 1);
        assert_eq!(plan(d, "lavapipe").1, Ok(Vec::new()), "{pf:?}");
    }
    // `mipx.bc1` (plain Metal zeroes part of the levels: undefined).
    let d = mipped(P::BC1RGBA, TextureType::Type2D, [8, 8, 1], 1);
    assert_eq!(
        plan(d, "lavapipe").1,
        invalid(MipmapProblem::NotColorRenderable)
    );
    // Probe 2's integer formats, which Metal's validation stops, are picked as plain Metal picks
    // them.
    for pf in [
        P::RGBA8Uint,
        P::R32Uint,
        P::RGBA16Sint,
        P::R8Sint,
        P::RGBA32Uint,
    ] {
        let d = mipped(pf, TextureType::Type2D, [8, 8, 1], 1);
        assert!(
            matches!(planned(d, "lavapipe").1, Ok(MipmapPlan::Picks(_))),
            "{pf:?}"
        );
    }
    // `mipx.nort` and `mipx.rgba8snorm.nort`: no render-target usage needed.
    for pf in [P::RGBA8Unorm, P::RGBA8Snorm] {
        let d = TextureDescriptor {
            usage: TextureUsage::SHADER_READ,
            ..mipped(pf, TextureType::Type2D, [4, 4, 1], 1)
        };
        assert_eq!(plan(d, "lavapipe").1.unwrap().len(), 2, "{pf:?}");
    }
}

/// No blit gives an extended-range texture's mipmaps (its image holds raw codes): they take the
/// compute filter, not a refusal and not as a format Metal stops, and a texture of one level is
/// left as it is, as for every format.
#[test]
fn extended_range_mipmaps_take_the_compute_filter() {
    use PixelFormat as P;
    for pf in [P::BGR10XR, P::BGR10XRSRGB, P::BGRA10XR, P::BGRA10XRSRGB] {
        // Floats held raw in an integer image: not an integer format Metal picks.
        assert!(!picks(pf), "{pf:?}");
        for (ty, extent) in [
            (TextureType::Type2D, [8, 8, 1]),
            (TextureType::Cube, [8, 8, 1]),
            (TextureType::Type3D, [8, 8, 2]),
        ] {
            let d = mipped(pf, ty, extent, 1);
            assert_eq!(
                planned(d, "lavapipe").1,
                Ok(MipmapPlan::Filter),
                "{pf:?} {ty:?}"
            );
        }
        let mut one = mipped(pf, TextureType::Type2D, [8, 8, 1], 1);
        one.mipmap_level_count = 1;
        let none = plan(one, "lavapipe").1.unwrap();
        assert!(none.is_empty(), "{pf:?}: one level");
    }
}

/// Every format Metal filters, on every committed profile, takes the device's blit chain where
/// its blits filter the image's format and the compute filter where they do not: no refusal for
/// how the answer would differ from Metal's. Some profile's blits cannot filter a format the
/// filter computes (lavapipe's `RGB9E5Float`, which takes no blits into it, and `vk12-floor`'s
/// 32-bit floats, which it does not filter linearly), and those take the filter.
#[test]
fn every_filtered_format_takes_the_blit_chain_or_the_compute_filter() {
    let blit_filters = |features: FormatFeatureFlags2| {
        features.contains(
            FormatFeatureFlags2::SAMPLED_IMAGE_FILTER_LINEAR
                | FormatFeatureFlags2::BLIT_SRC
                | FormatFeatureFlags2::BLIT_DST,
        )
    };
    let mut filtered = HashSet::new();
    let mut blitted = 0;
    for (name, d, device) in committed() {
        let facts = facts(&d, &device);
        for &pf in PixelFormat::ALL {
            if picks(pf) || not_implemented(pf) || format_problem(pf).is_some() {
                continue;
            }
            let Some(texel) = MipmapFilterChain::texel_bytes(pf) else {
                continue;
            };
            let desc = mipped(pf, TextureType::Type2D, [8, 8, 1], 1);
            let Ok(layout) = desc.check(APPLE) else {
                continue;
            };
            let Ok(image) = plan_image(&layout, desc.usage, false, &facts) else {
                continue;
            };
            let record = ViewRecord::of_texture(&layout, desc.usage, desc.swizzle);
            let raw = metal2vulkan::metal::format::ExtendedRange::of(pf).is_some()
                || super::super::image_plan::holds_srgb_twin(&image);
            let blits = !raw && blit_filters(facts.optimal(image.format));
            assert_eq!(
                image.format.block().map(|b| b.size),
                Some(texel),
                "{name} {pf:?}: an image of the texel's bytes"
            );
            match plan_mipmaps(&record, &image, &facts, &driver_defects::of(&d)) {
                Ok(MipmapPlan::Blits(chain)) if blits => {
                    assert_eq!(chain.len(), 3, "{name} {pf:?}");
                    blitted += 1;
                }
                Ok(MipmapPlan::Filter) if !blits => {
                    filtered.insert((name.clone(), pf));
                }
                other => panic!("{name} {pf:?}: {other:?} (blits filter it: {blits})"),
            }
        }
    }
    assert!(blitted > 0);
    for expected in [
        ("lavapipe", PixelFormat::RGB9E5Float),
        ("vk12-floor", PixelFormat::RGBA32Float),
        ("lavapipe", PixelFormat::BGRA10XR),
    ] {
        assert!(
            filtered.contains(&(expected.0.to_owned(), expected.1)),
            "{expected:?} takes the compute filter: {filtered:?}"
        );
    }
}

/// A 3D texture of a format the compute filter computes takes it on a driver whose blits do not
/// filter a volume (`RuntimeDriverDefect::VolumeBlitsUnfiltered`, MoltenVK's), and the blit chain
/// elsewhere; its 2D and cube textures take the blit chain there too.
#[test]
fn a_volume_takes_the_compute_filter_where_the_driver_s_blits_do_not_filter_it() {
    for (profile, ty, filter) in [
        ("moltenvk-m3", TextureType::Type3D, true),
        ("moltenvk-m3", TextureType::Type2D, false),
        ("moltenvk-m3", TextureType::Cube, false),
        ("lavapipe", TextureType::Type3D, false),
    ] {
        let extent = match ty {
            TextureType::Type3D => [8, 8, 4],
            _ => [8, 8, 1],
        };
        for pf in [PixelFormat::RGBA8Unorm, PixelFormat::RGBA16Float] {
            let what = format!("{pf:?} {ty:?} on {profile}");
            match planned(mipped(pf, ty, extent, 1), profile).1 {
                Ok(MipmapPlan::Filter) => assert!(filter, "{what}"),
                Ok(MipmapPlan::Blits(chain)) => assert!(!filter && chain.len() == 3, "{what}"),
                other => panic!("{what}: {other:?}"),
            }
        }
    }
}

#[test]
fn the_chain_blits_each_level_from_the_one_above() {
    let (_, blits) = plan(
        mipped(PixelFormat::RGBA8Unorm, TextureType::Type3D, [5, 3, 7], 1),
        "lavapipe",
    );
    let extents: Vec<([u32; 3], [u32; 3])> = blits
        .unwrap()
        .iter()
        .map(|b| (b.src_extent, b.dst_extent))
        .collect();
    assert_eq!(extents, [([5, 3, 7], [2, 1, 3]), ([2, 1, 3], [1, 1, 1])]);
    let (_, blits) = plan(
        mipped(PixelFormat::RGBA16Float, TextureType::Cube, [5, 5, 1], 1),
        "lavapipe",
    );
    let blits = blits.unwrap();
    assert_eq!(blits.len(), 2);
    assert!(blits.iter().all(|b| b.base_layer == 0 && b.layers == 6));
    assert_eq!((blits[1].src_level, blits[1].dst_level), (1, 2));
}

/// `mipx.view`: on a view of levels 1 and 2, level 2 comes from level 1, and the others stay.
#[test]
fn a_views_chain_is_its_own_levels_and_slices() {
    let (_, d, device) = committed()
        .into_iter()
        .find(|(n, ..)| n == "lavapipe")
        .unwrap();
    let facts = facts(&d, &device);
    let desc = TextureDescriptor {
        texture_type: TextureType::Type2DArray,
        array_length: 3,
        usage: TextureUsage::SHADER_READ | TextureUsage::PIXEL_FORMAT_VIEW,
        ..mipped(
            PixelFormat::RGBA8Unorm,
            TextureType::Type2DArray,
            [8, 8, 1],
            3,
        )
    };
    let layout = desc.check(APPLE).unwrap();
    let image = plan_image(&layout, desc.usage, false, &facts).unwrap();
    let whole = ViewRecord::of_texture(&layout, desc.usage, desc.swizzle);
    let view = check_view(
        &whole,
        &ViewRequest {
            texture_type: TextureType::Type2DArray,
            levels: 1..3,
            slices: 1..3,
            ..ViewRequest::of_format(&whole, PixelFormat::RGBA8Unorm)
        },
        device.profile(),
    )
    .unwrap();
    let blits = blits(plan_mipmaps(&view, &image, &facts, &driver_defects::of(&d)).unwrap());
    assert_eq!(
        blits,
        [MipBlit {
            src_level: 1,
            dst_level: 2,
            base_layer: 1,
            layers: 2,
            src_extent: [4, 4, 1],
            dst_extent: [2, 2, 1],
        }]
    );
}

/// An 8-bit sRGB texture a device holds as its UNORM twin (`image_plan::holds_srgb_twin`) takes
/// the compute filter, which decodes its codes, not the twin's blit, which would average them. At
/// least one committed profile holds one (lavapipe's `R8Unorm_sRGB`).
#[test]
fn an_srgb_texture_held_as_its_unorm_twin_takes_the_compute_filter() {
    use super::super::image_plan::holds_srgb_twin;
    let mut held = 0;
    for (name, d, device) in committed() {
        let facts = facts(&d, &device);
        for pf in [
            PixelFormat::R8UnormSRGB,
            PixelFormat::RG8UnormSRGB,
            PixelFormat::RGBA8UnormSRGB,
            PixelFormat::BGRA8UnormSRGB,
        ] {
            let mut desc = mipped(pf, TextureType::Type2D, [4, 4, 1], 1);
            desc.usage = TextureUsage::SHADER_READ;
            let layout = desc.check(APPLE).unwrap();
            let Ok(image) = plan_image(&layout, desc.usage, false, &facts) else {
                continue;
            };
            let record = ViewRecord::of_texture(&layout, desc.usage, desc.swizzle);
            if !holds_srgb_twin(&image) {
                continue;
            }
            held += 1;
            assert_eq!(
                plan_mipmaps(&record, &image, &facts, &driver_defects::of(&d)),
                Ok(MipmapPlan::Filter),
                "{name} {pf:?}"
            );
        }
    }
    assert!(
        held > 0,
        "no committed profile holds an sRGB texture as its twin"
    );
}

/// A view of another format than its texture's takes neither the image's blits nor its bytes
/// filtered as the image's format: refused, `NotImplemented(MipmapGeneration)`.
#[test]
fn a_view_of_another_format_is_not_implemented() {
    let (_, d, device) = committed()
        .into_iter()
        .find(|(n, ..)| n == "lavapipe")
        .unwrap();
    let facts = facts(&d, &device);
    let desc = TextureDescriptor {
        usage: TextureUsage::SHADER_READ | TextureUsage::PIXEL_FORMAT_VIEW,
        ..mipped(PixelFormat::RGBA8Unorm, TextureType::Type2D, [8, 8, 1], 1)
    };
    let layout = desc.check(APPLE).unwrap();
    let image = plan_image(&layout, desc.usage, false, &facts).unwrap();
    let whole = ViewRecord::of_texture(&layout, desc.usage, desc.swizzle);
    let view = check_view(
        &whole,
        &ViewRequest::of_format(&whole, PixelFormat::RGBA8UnormSRGB),
        device.profile(),
    )
    .unwrap();
    match plan_mipmaps(&view, &image, &facts, &driver_defects::of(&d)) {
        Err(MipmapRefusal::Unsupported(u)) => {
            assert_eq!(u.feature, Feature::MipmapGeneration);
            assert_eq!(u.needs, Needs::NotImplemented);
        }
        other => panic!("{other:?}"),
    }
}

/// Level `level`'s texel `at` (of the view's levels) as the first level's texel Metal's rule
/// takes it from: level by level, texel `floor((2i + 1) W / 2w)` of the level above on each axis.
fn metal_source(extent: impl Fn(u32) -> [u32; 3], level: u32, at: [u32; 3]) -> [u32; 3] {
    let mut c = at;
    for n in (1..=level).rev() {
        let (above, here) = (extent(n - 1), extent(n));
        for axis in 0..3 {
            let (big, small) = (u64::from(above[axis]), u64::from(here[axis]));
            c[axis] = ((2 * u64::from(c[axis]) + 1) * big / (2 * small)) as u32;
        }
    }
    c
}

/// Runs `plan`'s copies as Vulkan addresses them ("Copying Data Between Buffers and Images"):
/// every texel of the levels below the first that has at most `budget` texels is written exactly
/// once, from the scratch's copy of the first level's texel Metal's rule picks, of its own layer;
/// and every region keeps Vulkan's rules. Returns the texels checked.
fn run_picks(record: &ViewRecord, texel: u32, plan: &PickPlan, budget: u64) -> u64 {
    let volume = record.texture_type == TextureType::Type3D;
    let extent = |n: u32| {
        let mut e = record.extent().map(|axis| (axis >> n).max(1));
        if !volume {
            e[2] = 1;
        }
        e
    };
    let [width, height, depth] = extent(0).map(u64::from);
    let layers = u64::from(record.layer_count);
    let rows = if plan.image_height == 0 {
        height
    } else {
        u64::from(plan.image_height)
    };
    assert!(rows >= height);
    assert_eq!(plan.bytes, width * rows * depth * layers * u64::from(texel));
    let small = |n: u32| {
        let e = extent(n).map(u64::from);
        e[0] * e[1] * e[2] * layers <= budget
    };
    let mut written = HashSet::new();
    for pick in &plan.picks {
        let n = pick.level - record.base_level;
        let level = extent(n);
        assert!((1..record.level_count).contains(&n), "{pick:?}");
        assert_eq!(pick.extent[0], 1, "{pick:?}");
        for ((origin, extent), level) in pick.origin.iter().zip(&pick.extent).zip(&level) {
            assert!(origin + extent <= *level, "{pick:?}");
        }
        assert!(pick.row_length == 0 || pick.row_length >= pick.extent[0]);
        assert!(pick.image_height == 0 || pick.image_height >= pick.extent[1]);
        assert!(
            u64::from(pick.row_length) * u64::from(texel) < 1 << 31,
            "{pick:?}"
        );
        assert_eq!(pick.buffer_offset % u64::from(texel), 0, "{pick:?}");
        if !small(n) {
            continue;
        }
        let row_length = u64::from(if pick.row_length == 0 {
            pick.extent[0]
        } else {
            pick.row_length
        });
        let image_height = u64::from(if pick.image_height == 0 {
            pick.extent[1]
        } else {
            pick.image_height
        });
        let base = pick.buffer_offset / u64::from(texel);
        for layer in 0..layers {
            for z in 0..pick.extent[2] {
                for y in 0..pick.extent[1] {
                    let slice = if volume { u64::from(z) } else { layer };
                    let address = base + (slice * image_height + u64::from(y)) * row_length;
                    let x0 = address % width;
                    let y0 = address / width % rows;
                    let slice0 = address / width / rows;
                    let (z0, layer0) = if volume { (slice0, 0) } else { (0, slice0) };
                    let at = [pick.origin[0], pick.origin[1] + y, pick.origin[2] + z];
                    let source = metal_source(extent, n, at).map(u64::from);
                    assert_eq!(
                        [x0, y0, z0, layer0],
                        [source[0], source[1], source[2], layer],
                        "level {n} texel {at:?} of layer {layer}"
                    );
                    assert!(written.insert((n, at, layer)), "{n} {at:?} {layer} twice");
                }
            }
        }
    }
    let mut checked = 0;
    for n in (1..record.level_count).filter(|&n| small(n)) {
        let e = extent(n).map(u64::from);
        checked += e[0] * e[1] * e[2] * layers;
    }
    assert_eq!(written.len() as u64, checked);
    checked
}

fn record_of(desc: &TextureDescriptor) -> ViewRecord {
    let layout = desc.check(APPLE).unwrap();
    ViewRecord::of_texture(&layout, desc.usage, desc.swizzle)
}

/// A probe's texture: its type, format and extent, and its levels below the first.
type ProbeCase = (TextureType, PixelFormat, [u32; 3], &'static [&'static [u8]]);

/// Our probe's integer mipmaps (an M4, macOS 27.0.1, twice each): each texel holds the
/// low byte of its index in the level, and the levels below hold these.
#[test]
fn integer_mipmaps_pick_the_texels_plain_metal_picks() {
    use PixelFormat as P;
    use TextureType as T;
    let cases: [ProbeCase; 7] = [
        (
            T::Type2D,
            P::R8Uint,
            [6, 6, 1],
            &[
                &[0x07, 0x09, 0x0b, 0x13, 0x15, 0x17, 0x1f, 0x21, 0x23],
                &[0x15],
            ],
        ),
        (
            T::Type2D,
            P::R8Uint,
            [7, 5, 1],
            &[&[0x08, 0x0a, 0x0c, 0x16, 0x18, 0x1a], &[0x18]],
        ),
        (
            T::Type2D,
            P::R8Uint,
            [12, 10, 1],
            &[
                &[
                    0x0d, 0x0f, 0x11, 0x13, 0x15, 0x17, 0x25, 0x27, 0x29, 0x2b, 0x2d, 0x2f, 0x3d,
                    0x3f, 0x41, 0x43, 0x45, 0x47, 0x55, 0x57, 0x59, 0x5b, 0x5d, 0x5f, 0x6d, 0x6f,
                    0x71, 0x73, 0x75, 0x77,
                ],
                &[0x27, 0x2b, 0x2f, 0x57, 0x5b, 0x5f],
                &[0x5b],
            ],
        ),
        (
            T::Type2D,
            P::R8Uint,
            [15, 9, 1],
            &[
                &[
                    0x10, 0x12, 0x14, 0x16, 0x18, 0x1a, 0x1c, 0x2e, 0x30, 0x32, 0x34, 0x36, 0x38,
                    0x3a, 0x4c, 0x4e, 0x50, 0x52, 0x54, 0x56, 0x58, 0x6a, 0x6c, 0x6e, 0x70, 0x72,
                    0x74, 0x76,
                ],
                &[0x30, 0x34, 0x38, 0x6c, 0x70, 0x74],
                &[0x70],
            ],
        ),
        (
            T::Type3D,
            P::R8Uint,
            [4, 4, 4],
            &[&[0x15, 0x17, 0x1d, 0x1f, 0x35, 0x37, 0x3d, 0x3f], &[0x3f]],
        ),
        (
            T::Type3D,
            P::R8Uint,
            [6, 4, 6],
            &[
                &[
                    0x1f, 0x21, 0x23, 0x2b, 0x2d, 0x2f, 0x4f, 0x51, 0x53, 0x5b, 0x5d, 0x5f, 0x7f,
                    0x81, 0x83, 0x8b, 0x8d, 0x8f,
                ],
                &[0x5d],
            ],
        ),
        (
            T::Type2D,
            P::RGBA32Sint,
            [5, 5, 1],
            &[&[0x06, 0x08, 0x10, 0x12], &[0x12]],
        ),
    ];
    for (texture_type, pf, extent, levels) in cases {
        let desc = mipped(pf, texture_type, extent.map(u64::from), 1);
        let (record, result) = planned(desc, "lavapipe");
        let Ok(MipmapPlan::Picks(plan)) = result else {
            panic!("{extent:?}: {result:?}");
        };
        assert_eq!(record.level_count as usize, levels.len() + 1, "{extent:?}");
        let texel = if pf == P::R8Uint { 1 } else { 16 };
        run_picks(&record, texel, &plan, u64::MAX);
        let [w, h, _] = extent;
        let shape = |n: u32| {
            let mut e = extent.map(|axis| (axis >> n).max(1));
            if texture_type != T::Type3D {
                e[2] = 1;
            }
            e
        };
        for (n, want) in (1u32..).zip(levels) {
            let [lw, lh, ld] = shape(n);
            let mut got = Vec::new();
            for z in 0..ld {
                for y in 0..lh {
                    for x in 0..lw {
                        let [x0, y0, z0] = metal_source(shape, n, [x, y, z]);
                        got.push(((z0 * h + y0) * w + x0) as u8);
                    }
                }
            }
            assert_eq!(&got, want, "{extent:?} level {n}");
        }
    }
}

/// The picks of every shape: 2D, 3D, arrays and cubes of odd and even sizes, views of later
/// levels and slices, and the deep levels of a large texture, whose rows lie too far apart for
/// one region a column.
#[test]
fn picks_copy_metal_s_texel_for_every_shape() {
    use TextureType as T;
    let mut checked = 0;
    for w in 1..=13u32 {
        for h in 1..=11u32 {
            if w == 1 && h == 1 {
                continue;
            }
            let desc = mipped(
                PixelFormat::RG8Uint,
                T::Type2D,
                [u64::from(w), u64::from(h), 1],
                1,
            );
            let record = record_of(&desc);
            checked += run_picks(&record, 2, &plan_picks(&record, 2).unwrap(), u64::MAX);
            let arrays = mipped(
                PixelFormat::R16Sint,
                T::Type2DArray,
                [u64::from(w), u64::from(h), 1],
                3,
            );
            let record = record_of(&arrays);
            checked += run_picks(&record, 2, &plan_picks(&record, 2).unwrap(), u64::MAX);
        }
    }
    for extent in [
        [5, 3, 7],
        [9, 9, 2],
        [2, 8, 3],
        [1, 6, 5],
        [7, 1, 1],
        [16, 16, 16],
    ] {
        let desc = mipped(PixelFormat::RGBA8Uint, T::Type3D, extent, 1);
        let record = record_of(&desc);
        checked += run_picks(&record, 4, &plan_picks(&record, 4).unwrap(), u64::MAX);
    }
    for size in [5, 12] {
        let record = record_of(&mipped(
            PixelFormat::R32Uint,
            T::CubeArray,
            [size, size, 1],
            2,
        ));
        assert_eq!(record.layer_count, 12);
        checked += run_picks(&record, 4, &plan_picks(&record, 4).unwrap(), u64::MAX);
    }
    // A view of levels 1 to 3 and slices 1 and 2 of a 2D array of 4 levels: its first level
    // is the source, its slices the ones it presents.
    let whole = record_of(&mipped(PixelFormat::R8Uint, T::Type2DArray, [11, 7, 1], 4));
    let view = ViewRecord {
        base_level: 1,
        level_count: 3,
        base_layer: 1,
        layer_count: 2,
        ..whole
    };
    let plan = plan_picks(&view, 1).unwrap();
    assert!(plan.picks.iter().all(|p| (2..4).contains(&p.level)));
    checked += run_picks(&view, 1, &plan, u64::MAX);
    // A 16384 x 16384 `RGBA32Uint` texture: its level 14 rows lie 2^14 rows of 2^18 bytes
    // apart, past `bufferRowLength`'s bound, and take a region a row; the levels of up to 4096
    // texels are run.
    let large = record_of(&mipped(
        PixelFormat::RGBA32Uint,
        T::Type2D,
        [16384, 16384, 1],
        1,
    ));
    let plan = plan_picks(&large, 16).unwrap();
    assert!(plan
        .picks
        .iter()
        .any(|p| p.extent[1] == 1 && p.origin[1] > 0));
    checked += run_picks(&large, 16, &plan, 4096);
    assert!(checked > 10_000, "{checked}");
}
