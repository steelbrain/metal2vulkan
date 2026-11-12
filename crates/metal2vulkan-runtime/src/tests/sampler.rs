//! Metal's sampler-descriptor rules (`sampler::taken_descriptor`), case for case as probe 7
//! measured them on an Apple M3 Max (macOS 26.7) and an M4 (27.0.1), and as our probes measured
//! plain Metal's samples; and the bound on the device's live samplers
//! (`SamplerCount`).

use super::super::sampler::{
    metal_anisotropy, metal_lod_clamps, sampler_border_color, taken_descriptor, SamplerCount,
};
use super::super::{Error, SamplerLimit};
use metal2vulkan::bind::SamplerDescriptor;
use metal2vulkan::metal::{
    CompareFunction, SamplerAddressMode as A, SamplerBorderColor, SamplerMinMagFilter as F,
    SamplerMipFilter as M,
};

/// `edit` applied to Metal's default descriptor.
fn descriptor(edit: impl FnOnce(&mut SamplerDescriptor)) -> SamplerDescriptor {
    let mut d = SamplerDescriptor::new();
    edit(&mut d);
    d
}

/// A pixel-coordinate descriptor, `edit` applied.
fn pixel(edit: impl FnOnce(&mut SamplerDescriptor)) -> SamplerDescriptor {
    descriptor(|d| {
        d.normalized_coordinates = false;
        edit(d);
    })
}

#[test]
fn metal_takes_what_probe_7_saw_it_take() {
    let taken = [
        ("default", descriptor(|_| {})),
        ("aniso 1", descriptor(|d| d.max_anisotropy = 1)),
        ("aniso 16", descriptor(|d| d.max_anisotropy = 16)),
        (
            "aniso 4, nearest filters",
            descriptor(|d| {
                d.max_anisotropy = 4;
                d.min_filter = F::Nearest;
                d.mag_filter = F::Nearest;
            }),
        ),
        (
            "comparison with anisotropy",
            descriptor(|d| {
                d.compare_function = CompareFunction::Less;
                d.max_anisotropy = 4;
            }),
        ),
        (
            "comparison always, linear",
            descriptor(|d| {
                d.compare_function = CompareFunction::Always;
                d.min_filter = F::Linear;
                d.mag_filter = F::Linear;
            }),
        ),
        ("lod average", descriptor(|d| d.lod_average = true)),
        (
            "mirror clamp to edge",
            descriptor(|d| d.s_address_mode = A::MirrorClampToEdge),
        ),
        (
            "opaque white border",
            descriptor(|d| {
                d.s_address_mode = A::ClampToBorderColor;
                d.border_color = SamplerBorderColor::OpaqueWhite;
            }),
        ),
        ("pixel default", pixel(|_| {})),
        (
            "pixel, filters differ",
            pixel(|d| {
                d.min_filter = F::Linear;
                d.mag_filter = F::Nearest;
            }),
        ),
        (
            "pixel, comparison",
            pixel(|d| d.compare_function = CompareFunction::Less),
        ),
        ("pixel, lod average", pixel(|d| d.lod_average = true)),
        ("pixel, lod max 0", pixel(|d| d.lod_max_clamp = 0.0)),
        ("pixel, lod min 1", pixel(|d| d.lod_min_clamp = 1.0)),
        (
            "pixel, clamp to zero",
            pixel(|d| {
                d.s_address_mode = A::ClampToZero;
                d.t_address_mode = A::ClampToZero;
                d.r_address_mode = A::ClampToZero;
            }),
        ),
        (
            "pixel, clamp to border",
            pixel(|d| {
                d.s_address_mode = A::ClampToBorderColor;
                d.t_address_mode = A::ClampToBorderColor;
            }),
        ),
    ];
    // Each is taken as it is: Metal samples it as its descriptor states.
    for (name, d) in taken {
        assert_eq!(taken_descriptor(&d), d, "{name}");
    }
    // Both border modes, which plain Metal makes and reads the border color on every border axis
    // of, whatever the color.
    for color in [
        SamplerBorderColor::TransparentBlack,
        SamplerBorderColor::OpaqueBlack,
        SamplerBorderColor::OpaqueWhite,
    ] {
        let mixed = descriptor(|d| {
            d.s_address_mode = A::ClampToZero;
            d.r_address_mode = A::ClampToBorderColor;
            d.border_color = color;
        });
        assert_eq!(taken_descriptor(&mixed), mixed, "{color:?}");
    }
    // Any LOD clamps, at creation (inverted ones as the upper clamp alone,
    // `inverted_lod_clamps_read_the_upper_clamps_level`).
    for (min, max) in [
        (1.0, 0.0),
        (2.0, 2.0),
        (1000.0, f32::MAX),
        (-1.0, f32::MAX),
        (0.0, -1.0),
        (f32::NAN, 1.0),
        (0.0, f32::NAN),
        (f32::NEG_INFINITY, f32::INFINITY),
        (f32::INFINITY, f32::INFINITY),
    ] {
        let d = descriptor(|d| {
            d.lod_min_clamp = min;
            d.lod_max_clamp = max;
        });
        let taken = taken_descriptor(&d);
        let [a, b] = metal_lod_clamps([min, max]);
        assert_eq!(
            [taken.lod_min_clamp.to_bits(), taken.lod_max_clamp.to_bits()],
            [a.to_bits(), b.to_bits()],
            "clamps {min}..{max}"
        );
    }
    // Any LOD bias, taken as none: Metal samples every level as with no bias (an M3 Max on macOS
    // 27.0.1, `runtime_lane::lod_bias` in the validation crate).
    for bias in [
        1.0,
        -1.0,
        -0.0,
        15.999,
        -16.0,
        100.0,
        f32::MAX,
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
    ] {
        for d in [descriptor(|_| {}), pixel(|_| {})] {
            let mut biased = d;
            biased.lod_bias = bias;
            let taken = taken_descriptor(&biased);
            assert_eq!(taken.lod_bias.to_bits(), 0, "bias {bias}");
            assert_eq!(taken, taken_descriptor(&d), "bias {bias}");
        }
    }
}

/// A `maxAnisotropy` outside 1 to 16, which Metal's validation stops, samples as plain Metal
/// samples it (an M4 on macOS 27.0.1): the low five bits of the value taken through a 32-bit float,
/// saturated, 0 as 1 and 17 to 31 as 32. One of 1 to 16 samples as the power of two at or above it.
#[test]
fn an_anisotropy_metal_stops_samples_as_plain_metal_samples_it() {
    for (value, sampled) in [
        (1, 1),
        (2, 2),
        (3, 4),
        (4, 4),
        (5, 8),
        (6, 8),
        (7, 8),
        (8, 8),
        (9, 16),
        (12, 16),
        (15, 16),
        (16, 16),
    ] {
        assert_eq!(metal_anisotropy(value), sampled, "{value}");
    }
    for value in 17..=31 {
        assert_eq!(metal_anisotropy(value), 32, "{value}");
    }
    for (value, sampled) in [
        // samp2: multiples of 32 as 1, one past them as 1; samp5: 17 to 31 as 32, one level
        // finer than 16 at footprints of 32 to 1 and more.
        (0, 1),
        (17, 32),
        (24, 32),
        (31, 32),
        (32, 1),
        (33, 1),
        (48, 16),
        (63, 32),
        (65, 1),
        (1000, 8),
        (65536, 1),
        (1 << 31, 1),
        // samp3: remainders of 2, 3, 4 and 8 (3 samples as 4 does, as an in-range 3 does).
        (34, 2),
        (35, 4),
        (36, 4),
        (40, 8),
        (98, 2),
        (104, 8),
        // samp3 and samp4: the float rounds to a multiple of 32, or saturates.
        ((1 << 31) + 2, 1),
        ((1 << 25) + 2, 1),
        ((1 << 26) + 2, 1),
        ((1 << 28) + 18, 1),
        ((1 << 30) + 66, 1),
        ((1 << 31) + 34, 1),
        (u32::MAX - 27, 32),
        (u32::MAX, 32),
    ] {
        assert_eq!(metal_anisotropy(value), sampled, "{value}");
        let d = descriptor(|d| d.max_anisotropy = value);
        let taken = taken_descriptor(&d);
        assert_eq!(taken.max_anisotropy, sampled, "{value}");
        let mut want = d;
        want.max_anisotropy = sampled;
        assert_eq!(taken, want, "{value}");
    }
}

/// Inverted LOD clamps, a descriptor's or a bind's, read the upper clamp's level as plain
/// Metal's do, filtering as the upper clamp alone does: the clamps `[max, max]`. Ordered clamps and
/// NaN ones are kept as they are.
#[test]
fn inverted_lod_clamps_read_the_upper_clamps_level() {
    for ([min, max], taken) in [
        ([1.0, 0.0], [0.0, 0.0]),
        ([2.0, 1.0], [1.0, 1.0]),
        ([3.0, 1.0], [1.0, 1.0]),
        ([2.5, 0.5], [0.5, 0.5]),
        ([1000.0, 0.0], [0.0, 0.0]),
        ([0.0, -1.0], [-1.0, -1.0]),
        ([f32::INFINITY, 2.0], [2.0, 2.0]),
        ([0.0, 1.0], [0.0, 1.0]),
        ([1.0, 1.0], [1.0, 1.0]),
        ([-4.0, f32::MAX], [-4.0, f32::MAX]),
    ] {
        assert_eq!(metal_lod_clamps([min, max]), taken, "{min}..{max}");
        let d = descriptor(|d| {
            d.lod_min_clamp = min;
            d.lod_max_clamp = max;
        });
        let got = taken_descriptor(&d);
        assert_eq!(
            [got.lod_min_clamp, got.lod_max_clamp],
            taken,
            "{min}..{max}"
        );
    }
    for [min, max] in [[f32::NAN, 1.0], [0.0, f32::NAN], [f32::NAN, f32::NAN]] {
        let [a, b] = metal_lod_clamps([min, max]);
        assert_eq!([a.to_bits(), b.to_bits()], [min.to_bits(), max.to_bits()]);
    }
}

/// Metal keeps a raw border color's low two bits and samples 3 as transparent black.
#[test]
fn a_raw_border_color_samples_as_plain_metal_samples_it() {
    use SamplerBorderColor::{OpaqueBlack, OpaqueWhite, TransparentBlack};
    for (raw, color) in [
        (0, TransparentBlack),
        (1, OpaqueBlack),
        (2, OpaqueWhite),
        (3, TransparentBlack),
        (4, TransparentBlack),
        (5, OpaqueBlack),
        (7, TransparentBlack),
        (8, TransparentBlack),
        (255, TransparentBlack),
        (256, TransparentBlack),
        (65535, TransparentBlack),
        (1 << 31, TransparentBlack),
        (1 << 32, TransparentBlack),
        ((1 << 32) + 2, OpaqueWhite),
        (u64::MAX, TransparentBlack),
    ] {
        assert_eq!(sampler_border_color(raw), color, "{raw}");
    }
    for &color in SamplerBorderColor::ALL {
        assert_eq!(sampler_border_color(u64::from(color.as_raw())), color);
    }
}

/// A pixel-coordinate sampler Metal's validation stops samples as plain Metal samples it: repeat
/// and mirrored-repeat axes as `ClampToEdge`, level 0, no anisotropy; a mirror-clamp `s` or `t`
/// axis keeps its mode, which Metal mirrors once by texel index and then clamps.
#[test]
fn a_pixel_sampler_samples_as_plain_metal_samples_it() {
    let clamped = pixel(|d| {
        d.min_filter = F::Linear;
        d.mag_filter = F::Linear;
    });
    let taken = [
        ("aniso 2", pixel(|d| d.max_anisotropy = 2)),
        ("aniso 16", pixel(|d| d.max_anisotropy = 16)),
        ("mip nearest", pixel(|d| d.mip_filter = M::Nearest)),
        ("mip linear", pixel(|d| d.mip_filter = M::Linear)),
        ("s repeat", pixel(|d| d.s_address_mode = A::Repeat)),
        ("t repeat", pixel(|d| d.t_address_mode = A::Repeat)),
        ("r repeat", pixel(|d| d.r_address_mode = A::Repeat)),
        (
            "s mirrored repeat",
            pixel(|d| d.s_address_mode = A::MirrorRepeat),
        ),
        (
            "t mirrored repeat",
            pixel(|d| d.t_address_mode = A::MirrorRepeat),
        ),
        (
            "r mirror clamp",
            pixel(|d| d.r_address_mode = A::MirrorClampToEdge),
        ),
        (
            "all of them",
            pixel(|d| {
                d.max_anisotropy = 8;
                d.mip_filter = M::Linear;
                d.s_address_mode = A::Repeat;
                d.t_address_mode = A::MirrorRepeat;
                d.r_address_mode = A::Repeat;
            }),
        ),
    ];
    for (name, mut d) in taken {
        let pixel_default = pixel(|_| {});
        assert_eq!(taken_descriptor(&d), pixel_default, "{name}");
        d.min_filter = F::Linear;
        d.mag_filter = F::Linear;
        assert_eq!(taken_descriptor(&d), clamped, "{name}, linear");
    }
    // Clamping axes, and a normalized descriptor, are taken as they are.
    let border = pixel(|d| {
        d.s_address_mode = A::ClampToZero;
        d.t_address_mode = A::ClampToBorderColor;
        d.border_color = SamplerBorderColor::OpaqueWhite;
    });
    assert_eq!(taken_descriptor(&border), border);
    let normalized = descriptor(|d| {
        d.max_anisotropy = 8;
        d.mip_filter = M::Linear;
        d.s_address_mode = A::Repeat;
    });
    assert_eq!(taken_descriptor(&normalized), normalized);
    for d in [
        pixel(|d| d.s_address_mode = A::MirrorClampToEdge),
        pixel(|d| d.t_address_mode = A::MirrorClampToEdge),
        pixel(|d| {
            d.s_address_mode = A::MirrorClampToEdge;
            d.t_address_mode = A::MirrorClampToEdge;
            d.min_filter = F::Linear;
            d.mag_filter = F::Linear;
        }),
    ] {
        assert_eq!(taken_descriptor(&d), d, "{d:?}");
    }
    // Its `r` axis's, which addresses nothing it samples, clamps as a repeating one's.
    let r = pixel(|d| {
        d.s_address_mode = A::MirrorClampToEdge;
        d.r_address_mode = A::MirrorClampToEdge;
    });
    let r_taken = pixel(|d| d.s_address_mode = A::MirrorClampToEdge);
    assert_eq!(taken_descriptor(&r), r_taken);
}

#[test]
fn live_samplers_stop_at_the_device_s_limit() {
    let count = SamplerCount::new(3);
    for _ in 0..3 {
        assert_eq!(count.take(), Ok(()));
    }
    let refused = Err(Error::SamplerLimit(SamplerLimit { limit: 3 }));
    assert_eq!(
        count.take(),
        refused,
        "a fourth sampler is refused, not made"
    );
    assert_eq!(count.take(), refused, "and stays refused while three live");
    count.give();
    assert_eq!(count.take(), Ok(()), "one destroyed makes room for one");
    assert_eq!(count.take(), refused);
}

#[test]
fn live_samplers_never_pass_the_limit_from_many_threads() {
    let count = std::sync::Arc::new(SamplerCount::new(64));
    let taken: usize = (0..8)
        .map(|_| {
            let count = std::sync::Arc::clone(&count);
            std::thread::spawn(move || (0..32).filter(|_| count.take().is_ok()).count())
        })
        .collect::<Vec<_>>()
        .into_iter()
        .map(|thread| thread.join().unwrap())
        .sum();
    assert_eq!(taken, 64, "256 asked of 64: exactly 64 made");
}
