use std::collections::HashMap;

use super::cpu::{small_float_bits, small_float_value};
use super::fetch::{components, fits, Recover};
use super::kernel::typed_module;
use super::layout::{axis_tap, layout, Rule, Texel};
use super::{
    FetchView, MipmapFilterChain, MipmapFilterKernel, MipmapFilterModule, DESTINATION, PARAMS,
    SOURCE, SOURCE_IMAGE,
};
use crate::metal::format::candidate_formats;
use crate::metal::PixelFormat;
use crate::spirv::interp::{run, run_with_images, Buffer, Config, Image, Order, Resources};
use crate::spirv::ir::TypedModule;
use crate::spirv::testing::{check_valid, committed_profiles};
use crate::vk::Format;

/// The sampled views through which a fetch reads the first level of `pf`: one per candidate
/// Vulkan format of its image whose fetch recovers the texel (`MipmapFilterChain::fetch_format`).
fn fetch_views(pf: PixelFormat) -> Vec<Format> {
    let Some(l) = layout(pf) else {
        return Vec::new();
    };
    let mut views: Vec<Format> = candidate_formats(pf)
        .into_iter()
        .filter_map(MipmapFilterChain::fetch_format)
        .filter(|&view| components(view).is_some_and(|c| fits(&l, &c)))
        .collect();
    views.dedup();
    views
}

/// Every kernel a chain may dispatch: each format's levels, with the depth taps or not; and, for
/// each view a fetch reads its first level through, the first later level fetching from it (from
/// a 2D array, a 3D texture one texel deep, or a deeper one).
pub(crate) fn every_kernel() -> Vec<MipmapFilterKernel> {
    let mut out = Vec::new();
    for &pixel_format in PixelFormat::ALL {
        if layout(pixel_format).is_none() {
            continue;
        }
        for depth in [false, true] {
            out.push(MipmapFilterKernel {
                pixel_format,
                depth,
                fetch: None,
            });
        }
        for format in fetch_views(pixel_format) {
            for (depth, volume) in [(false, false), (false, true), (true, true)] {
                out.push(MipmapFilterKernel {
                    pixel_format,
                    depth,
                    fetch: Some(FetchView { format, volume }),
                });
            }
        }
    }
    out
}

/// The formats the filter computes.
fn formats() -> Vec<PixelFormat> {
    PixelFormat::ALL
        .iter()
        .copied()
        .filter(|&pf| layout(pf).is_some())
        .collect()
}

/// Every kernel is valid on every committed profile (the owned verifier, the profile's
/// requirements and `spirv-val`), checked on as many threads as the host has: the filter needs
/// no feature past Vulkan's floor (`shaderInt64` among those it does not need).
#[test]
fn every_filter_kernel_is_valid_on_every_committed_profile() {
    let profiles = committed_profiles().expect("profiles");
    assert!(
        profiles.iter().any(|(_, p)| !p.fields().leaf.int64),
        "a committed profile without shaderInt64"
    );
    let kernels = every_kernel();
    let fetching = kernels.iter().filter(|k| k.fetches()).count();
    assert_eq!(formats().len(), 38, "the formats the filter computes");
    assert_eq!(
        kernels.len() - fetching,
        2 * 38,
        "the kernels of the 38 formats"
    );
    assert_eq!(
        fetching,
        3 * 39,
        "the fetching kernels of the 39 views of 30 formats"
    );
    let modules: Vec<(MipmapFilterKernel, TypedModule)> = kernels
        .iter()
        .map(|k| {
            (
                *k,
                typed_module(&k.spec().expect("a spec")).expect("module"),
            )
        })
        .collect();
    let jobs: Vec<(usize, usize)> = (0..modules.len())
        .flat_map(|m| (0..profiles.len()).map(move |p| (m, p)))
        .collect();
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let failures: Vec<String> = std::thread::scope(|scope| {
        let handles: Vec<_> = jobs
            .chunks(jobs.len().div_ceil(threads))
            .map(|chunk| {
                let (modules, profiles) = (&modules, &profiles);
                scope.spawn(move || {
                    let mut failed = Vec::new();
                    for &(m, p) in chunk {
                        let (kernel, module) = &modules[m];
                        let (name, profile) = &profiles[p];
                        if let Err(e) = check_valid(module.clone(), profile) {
                            failed.push(format!("{kernel:?} on {name}: {e}"));
                        }
                    }
                    failed
                })
            })
            .collect();
        handles
            .into_iter()
            .flat_map(|h| h.join().expect("a checker thread"))
            .collect()
    });
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// One dispatch's buffers: the level's words and workgroups, the level above's bytes, and the
/// level's bytes.
struct Dispatch<'a> {
    params: &'a [u32],
    groups: [u32; 2],
    above: &'a [u8],
    above_bytes: u64,
    bytes: u64,
    /// Whether the destination is zeroed first ([`MipmapFilterChain::zeroed`]); else it holds
    /// the sentinel 0xa5, which the dispatch must overwrite.
    zeroed: bool,
    /// The first level's sampled view, where the kernel fetches it (`above` is then unread).
    first: Option<&'a Image>,
}

/// A dispatch run in the interpreter, both invocation orders, over a destination zeroed or not
/// (`zeroed`) and 8 sentinel bytes past its words, which it must leave: the level's bytes.
fn dispatch(m: &TypedModule, d: &Dispatch<'_>) -> Vec<u8> {
    let mut source = d.above.to_vec();
    source.resize(d.above_bytes.next_multiple_of(4).max(4) as usize, 0);
    let words = d.bytes.next_multiple_of(4) as usize;
    let mut destination = vec![if d.zeroed { 0 } else { 0xa5 }; words];
    destination.extend([0xee; 8]);
    let mut results = Vec::new();
    for order in [Order::Forward, Order::Reverse] {
        let part = |slot, bytes: Vec<u8>| Buffer {
            slot: Some(slot),
            address: None,
            bytes,
        };
        let res = Resources {
            buffers: vec![
                part(SOURCE, source.clone()),
                part(DESTINATION, destination.clone()),
                part(
                    PARAMS,
                    d.params.iter().flat_map(|w| w.to_le_bytes()).collect(),
                ),
            ],
            push: Vec::new(),
            spec: Default::default(),
        };
        let cfg = Config {
            order,
            workgroups: [d.groups[0], d.groups[1], 1],
            step_limit: 1 << 30,
            ..Config::default()
        };
        let buffers = match d.first {
            Some(image) => {
                let (outcome, images) =
                    run_with_images(m, &cfg, res, vec![image.clone()]).expect("run");
                assert_eq!(
                    images[0].texels, image.texels,
                    "the first level is only read"
                );
                outcome.buffers
            }
            None => run(m, &cfg, res).expect("run").buffers,
        };
        assert_eq!(buffers[0], source, "the level above is only read");
        let written = &buffers[1];
        assert_eq!(
            written[words..],
            [0xee; 8],
            "nothing past the level's words"
        );
        assert!(
            written[d.bytes as usize..words].iter().all(|&b| b == 0),
            "the last word's unused bytes stay 0"
        );
        results.push(written[..d.bytes as usize].to_vec());
    }
    assert_eq!(results[0], results[1], "order independent");
    results.swap_remove(0)
}

/// The modules of the kernels the tests have run, each built once.
#[derive(Default)]
struct Modules(HashMap<MipmapFilterKernel, TypedModule>);

impl Modules {
    fn of(&mut self, kernel: MipmapFilterKernel) -> &TypedModule {
        self.0
            .entry(kernel)
            .or_insert_with(|| typed_module(&kernel.spec().expect("a spec")).expect("module"))
    }
}

/// Every level of `chain` (of a 3D texture where `volume`) run in the interpreter from `first`,
/// each by its kernel from the level above: the levels, the first included. Where the chain
/// fetches its first level ([`MipmapFilterChain::fetch`]), the first later level reads it from its
/// view, each texel what [`fetched_texel`] says a device returns.
fn run_chain(
    modules: &mut Modules,
    chain: &MipmapFilterChain,
    volume: bool,
    first: &[u8],
) -> Vec<Vec<u8>> {
    let mut levels = vec![first.to_vec()];
    let view = chain
        .fetch
        .map(|format| first_view(chain, volume, format, first));
    for level in 1..chain.levels.len() {
        let (a, l) = (&chain.levels[level - 1], &chain.levels[level]);
        let (_, words) = l.params.as_ref().expect("params");
        let m = modules.of(l.kernel.expect("a kernel"));
        let bytes = dispatch(
            m,
            &Dispatch {
                params: words,
                groups: l.groups,
                above: &levels[level - 1],
                above_bytes: a.bytes,
                bytes: l.bytes,
                zeroed: chain.zeroed,
                first: view.as_ref().filter(|_| level == 1),
            },
        );
        levels.push(bytes);
    }
    levels
}

/// What a device's fetch of a sampled view of `view` returns for a texel of `bytes`, as the
/// interpreter holds it (four 32-bit components): a unorm code `c` of `b` bits as `c / (2^b -
/// 1)` and a snorm one as `max(c / (2^(b-1) - 1), -1)`, each the `f32` nearest it; a half as its
/// own value; an integer component's bits.
fn fetched_texel(view: Format, bytes: &[u8]) -> [u32; 4] {
    let mut word = [0u8; 8];
    word[..bytes.len()].copy_from_slice(bytes);
    let all = u64::from_le_bytes(word);
    let mut out = [0, 0, 0, 1f32.to_bits()];
    for (lane, c) in components(view).expect("a fetched view").iter().enumerate() {
        let Some(c) = c else {
            continue;
        };
        let code = (all >> c.offset) & ((1u64 << c.bits) - 1);
        out[lane] = match c.recover {
            Recover::Unorm => (code as f32 / ((1u64 << c.bits) - 1) as f32).to_bits(),
            Recover::Snorm => {
                let signed = ((code << (64 - c.bits)) as i64 >> (64 - c.bits)) as f32;
                (signed / ((1u64 << (c.bits - 1)) - 1) as f32)
                    .max(-1.0)
                    .to_bits()
            }
            Recover::Half => small_float_value(code as u32, 10, true).to_bits(),
            Recover::Bits => code as u32,
        };
    }
    out
}

/// The first level of `chain` (of a 3D texture where `volume`), `first` (every layer's texels),
/// as its sampled view of `view` holds it: a 2D array of every layer, or a 3D image.
fn first_view(chain: &MipmapFilterChain, volume: bool, view: Format, first: &[u8]) -> Image {
    let l = &chain.levels[0];
    let (extent, layers) = if volume {
        (l.extent, 1)
    } else {
        ([l.extent[0], l.extent[1], 1], chain.layers)
    };
    let mut image = Image::new(SOURCE_IMAGE, extent, layers, 1, 1);
    let bytes = layout(chain.pixel_format).expect("a layout").bytes as usize;
    for (texel, b) in image.texels.iter_mut().zip(first.chunks(bytes)) {
        *texel = fetched_texel(view, b);
    }
    image
}

/// The chain of `pf` of `levels` levels of `layers` layers from `extent`, on generous limits,
/// copied in or fetched through `fetch`.
fn plan_of(
    pf: PixelFormat,
    extent: [u32; 3],
    layers: u32,
    levels: u32,
    volume: bool,
    fetch: Option<Format>,
) -> MipmapFilterChain {
    MipmapFilterChain::plan(
        pf,
        extent,
        layers,
        levels,
        volume,
        64,
        u32::MAX,
        [65535, 65535],
        fetch,
    )
    .unwrap_or_else(|| panic!("a chain of {pf:?} {extent:?}"))
}

/// `n` bytes from a seed (xorshift).
fn seeded(n: usize, mut seed: u64) -> Vec<u8> {
    (0..n)
        .map(|_| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 24) as u8
        })
        .collect()
}

/// The shapes the chains run: `(extent, layers, levels, volume)`. Halving and not, a strip one
/// texel high, an array, a cube's six faces, 3D textures that halve and do not, one whose depth
/// reaches one texel before its width does, and a 3D texture one texel deep.
const SHAPES: [([u32; 3], u32, u32, bool); 9] = [
    ([7, 5, 1], 1, 3, false),
    ([8, 8, 1], 2, 4, false),
    ([4, 3, 1], 6, 3, false),
    ([17, 1, 1], 1, 5, false),
    ([66, 2, 1], 1, 3, false),
    ([5, 4, 3], 1, 3, true),
    ([4, 4, 4], 1, 3, true),
    ([8, 4, 2], 1, 4, true),
    ([5, 4, 1], 1, 2, true),
];

/// Each later level of `got` (the interpreter's chain, the first level included) within the
/// filter's bound of the CPU's filter of the level above as the interpreter wrote it, and, where
/// the format has no float field, byte for byte that filter: the interpreter rounds each
/// operation as IEEE does and takes `Pow` as the model does, so only a NaN's bits may differ.
fn assert_within(chain: &MipmapFilterChain, got: &[Vec<u8>], what: &str) {
    let floats = !matches!(
        layout(chain.pixel_format).expect("a layout").texel,
        Texel::Fields(fields) if fields.iter().all(|f| !matches!(f.rule, Rule::Float { .. }))
    );
    for level in 1..got.len() {
        let outside = chain
            .outside_bound(level, &got[level - 1], &got[level])
            .expect("a level");
        assert!(
            outside.is_empty(),
            "{what} level {level}: texels {outside:?}"
        );
        if !floats {
            let one = MipmapFilterChain {
                levels: chain.levels[level - 1..=level].to_vec(),
                fetch: None,
                ..chain.clone()
            };
            let want = one.expected_levels(&got[level - 1]).expect("levels");
            assert_eq!(got[level], want[1], "{what} level {level}");
        }
    }
}

/// Every format the filter computes, over every shape of [`SHAPES`], from random bytes (every bit
/// pattern: NaNs, infinities, subnormals, out-of-range codes): each level of the kernel's chain
/// is within the bound of the CPU's filter of the level above (`assert_within`).
#[test]
fn the_kernel_is_the_chain_on_the_cpu() {
    let mut modules = Modules::default();
    for (seed, pf) in formats().into_iter().enumerate() {
        let l = layout(pf).expect("a layout");
        for (s, &(extent, layers, levels, volume)) in SHAPES.iter().enumerate() {
            let chain = plan_of(pf, extent, layers, levels, volume, None);
            let texels = extent.iter().product::<u32>() * layers;
            let first = seeded(
                (texels * l.bytes) as usize,
                0x9e37_79b9 ^ (seed * 31 + s) as u64,
            );
            let got = run_chain(&mut modules, &chain, volume, &first);
            assert_within(&chain, &got, &format!("{pf:?} {extent:?} x{layers}"));
        }
    }
}

/// Every view a fetch reads a first level through, over the shapes of [`SHAPES`]: the chain reads
/// the level in place (no copy of it in the scratch buffer), each level within the bound of the
/// CPU's filter of the level above (`assert_within`).
#[test]
fn a_first_level_fetched_from_its_view_is_the_chain_on_the_cpu() {
    let mut modules = Modules::default();
    let mut views = 0;
    for (seed, pf) in formats().into_iter().enumerate() {
        let l = layout(pf).expect("a layout");
        for view in fetch_views(pf) {
            views += 1;
            for (s, &(extent, layers, levels, volume)) in SHAPES.iter().enumerate() {
                let copied = plan_of(pf, extent, layers, levels, volume, None);
                let chain = plan_of(pf, extent, layers, levels, volume, Some(view));
                assert_eq!(chain.fetch, Some(view), "{pf:?} through {view:?}");
                let first = chain.levels[0].bytes.next_multiple_of(64);
                assert_eq!(
                    chain.bytes + first,
                    copied.bytes,
                    "no first level in the scratch"
                );
                let texels = extent.iter().product::<u32>() * layers;
                let first = seeded(
                    (texels * l.bytes) as usize,
                    0x51f1_5eed ^ (seed * 31 + s) as u64,
                );
                let got = run_chain(&mut modules, &chain, volume, &first);
                assert_within(
                    &chain,
                    &got,
                    &format!("{pf:?} through {view:?} {extent:?} x{layers}"),
                );
            }
        }
    }
    assert_eq!(views, 39, "the views of the formats a fetch reads");
}

/// The taps are a linear sampler's at each destination texel's centre: a halving axis averages
/// texels `2i` and `2i + 1`, an axis of one texel reads it alone, and an odd axis takes the
/// centre's exact fraction.
#[test]
fn the_taps_are_a_sampler_s_at_the_texel_centres() {
    for big in [2, 8, 64, 16384] {
        for i in 0..big / 2 {
            assert_eq!(axis_tap(i, big, big / 2), (2 * i, 0.5), "{big} {i}");
        }
    }
    assert_eq!(axis_tap(0, 1, 1), (0, 0.0));
    // 7 to 3: centres at 7/6, 7/2 and 35/6 of the level above, less a half.
    assert_eq!(axis_tap(0, 7, 3), (0, (2.0f64 / 3.0) as f32));
    assert_eq!(axis_tap(1, 7, 3), (3, 0.0));
    assert_eq!(axis_tap(2, 7, 3), (5, (1.0f64 / 3.0) as f32));
    // 5 to 2: centres at 5/4 and 15/4, less a half.
    assert_eq!(axis_tap(0, 5, 2), (0, 0.75));
    assert_eq!(axis_tap(1, 5, 2), (3, 0.25));
    // 3 to 1: the middle texel alone, the next weighing 0.
    assert_eq!(axis_tap(0, 3, 1), (1, 0.0));
}

/// The levels of known texels: halving boxes average their four (eight in 3D) texels, each
/// format's code rounded to nearest even from the `float` mean.
#[test]
fn a_halving_box_is_the_mean_of_its_texels() {
    let level = |pf, extent: [u32; 3], volume, first: &[u8]| {
        let chain = plan_of(pf, extent, 1, 2, volume, None);
        chain.expected_levels(first).expect("levels").swap_remove(1)
    };
    // 8-bit unorm: (0 + 255 + 255 + 0) / 4 = 127.5 rounds to 128; (1 + 2 + 2 + 2) / 4 = 1.75 to 2.
    assert_eq!(
        level(
            PixelFormat::RG8Unorm,
            [2, 2, 1],
            false,
            &[0, 1, 255, 2, 255, 2, 0, 2]
        ),
        [128, 2]
    );
    // 8-bit snorm: -128 reads as -1, as -127 does; (-1 - 1 + 1 + 1) / 4 = 0.
    assert_eq!(
        level(
            PixelFormat::R8Snorm,
            [2, 2, 1],
            false,
            &[0x80, 0x81, 127, 127]
        ),
        [0]
    );
    // 32-bit float, eight texels of 3D: their mean.
    let floats: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    assert_eq!(
        level(PixelFormat::R32Float, [2, 2, 2], true, &floats),
        4.5f32.to_le_bytes()
    );
    // A NaN tap makes a NaN (the CPU's here, the device's on a device); an infinity stays one.
    let special: Vec<u8> = [
        f32::from_bits(0x7f80_0001),
        1.0,
        f32::INFINITY,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
    ]
    .iter()
    .flat_map(|v| v.to_le_bytes())
    .collect();
    let got = level(PixelFormat::R32Float, [4, 2, 1], false, &special);
    assert!(f32::from_le_bytes(got[..4].try_into().unwrap()).is_nan());
    assert_eq!(got[4..], f32::INFINITY.to_le_bytes());
    // A half, rounded to nearest even: (1 + 1 + 1 + 1 + 2^-10) / 4 rounds back to 1.
    let one = 0x3c00u16.to_le_bytes();
    let next = 0x3c01u16.to_le_bytes();
    assert_eq!(
        level(
            PixelFormat::R16Float,
            [2, 2, 1],
            false,
            &[one, one, one, next].concat()
        ),
        one
    );
    // `RGB9E5Float`: the mean of 1, 2, 3 and 4 in red, 0 elsewhere, is 2.5 = 320 x 2^-7.
    let rgb9e5 = |r: u32, e: u32| (r | (e << 27)).to_le_bytes();
    let first = [
        rgb9e5(256, 16),
        rgb9e5(256, 17),
        rgb9e5(384, 17),
        rgb9e5(256, 18),
    ]
    .concat();
    assert_eq!(
        level(PixelFormat::RGB9E5Float, [2, 2, 1], false, &first),
        rgb9e5(320, 17)
    );
    // `BGR10_XR`: codes of 0.0 and 1.0 (384 and 894) average to 0.5 (639), of -0.5 and 0.5 (129
    // and 639) to 0 (384); the top bits set.
    let xr = |b: u32, g: u32, r: u32| (b | (g << 10) | (r << 20) | (3 << 30)).to_le_bytes();
    let first = [
        xr(384, 894, 129),
        xr(894, 894, 129),
        xr(384, 384, 639),
        xr(894, 384, 639),
    ]
    .concat();
    assert_eq!(
        level(PixelFormat::BGR10XR, [2, 2, 1], false, &first),
        xr(639, 639, 384)
    );
}

/// The bound a device's level keeps to the CPU's filter: off a rounding boundary the CPU's code
/// alone, at one either neighbour and no further; a float within `2^-17` of its terms'
/// magnitudes; an 8-bit sRGB code of a box of one code that code alone; a NaN any NaN.
#[test]
fn the_bound_takes_a_boundary_s_neighbours_and_nothing_further() {
    let outside = |pf, above: &[u8], got: &[u8]| {
        let chain = plan_of(pf, [2, 2, 1], 1, 2, false, None);
        !chain
            .outside_bound(1, above, got)
            .expect("a level")
            .is_empty()
    };
    let r8 = PixelFormat::R8Unorm;
    // (0 + 0 + 1 + 1) / 4 = 0.5 of a code: 0 or 1, never 2.
    assert!(!outside(r8, &[0, 0, 1, 1], &[0]));
    assert!(!outside(r8, &[0, 0, 1, 1], &[1]));
    assert!(outside(r8, &[0, 0, 1, 1], &[2]));
    // (0 + 0 + 0 + 1) / 4 = 0.25 of a code: 0 alone.
    assert!(!outside(r8, &[0, 0, 0, 1], &[0]));
    assert!(outside(r8, &[0, 0, 0, 1], &[1]));
    // A box of one 8-bit sRGB code is that code alone, through the transfer both ways.
    for c in 0..=255u8 {
        let srgb = PixelFormat::R8UnormSRGB;
        assert!(!outside(srgb, &[c; 4], &[c]), "{c}");
        assert!(outside(srgb, &[c; 4], &[c.wrapping_add(1)]), "{c}");
    }
    let floats = |v: [f32; 4]| -> Vec<u8> { v.iter().flat_map(|v| v.to_le_bytes()).collect() };
    let r32 = PixelFormat::R32Float;
    let above = floats([1.0, 2.0, 3.0, 4.0]);
    let next = f32::from_bits(2.5f32.to_bits() + 1);
    assert!(!outside(r32, &above, &2.5f32.to_le_bytes()));
    assert!(!outside(r32, &above, &next.to_le_bytes()));
    assert!(outside(
        r32,
        &above,
        &(2.5f32 * (1.0 + 2f32.powi(-14))).to_le_bytes()
    ));
    let nan = floats([f32::NAN, 1.0, 1.0, 1.0]);
    assert!(!outside(
        r32,
        &nan,
        &f32::from_bits(0xffc0_0001).to_le_bytes()
    ));
    assert!(outside(r32, &nan, &0.0f32.to_le_bytes()));
}

/// The packed float conversions are exact both ways where the value is a float of the format,
/// round to nearest even, overflow to infinity, keep a NaN quiet, and take a negative value of an
/// unsigned format as 0.
#[test]
fn the_packed_floats_convert_exactly() {
    for mantissa in [10, 6, 5] {
        let signed = mantissa == 10;
        let codes = if signed {
            1u32 << 16
        } else {
            1 << (mantissa + 5)
        };
        for b in 0..codes {
            let v = small_float_value(b, mantissa, signed);
            let back = small_float_bits(v, mantissa, signed);
            if v.is_nan() {
                assert_eq!(back, (31 << mantissa) | (1 << (mantissa - 1)), "{b:#x}");
            } else {
                assert_eq!(back, b, "{b:#x} of {mantissa}");
            }
        }
    }
    // Halves: a tie between 1 and its successor rounds to even, past 65504 to infinity, the
    // smallest subnormal and below.
    assert_eq!(small_float_bits(1.0 + 2f32.powi(-11), 10, true), 0x3c00);
    assert_eq!(
        small_float_bits(1.0 + 3.0 * 2f32.powi(-11), 10, true),
        0x3c02
    );
    assert_eq!(small_float_bits(65520.0, 10, true), 0x7c00);
    assert_eq!(small_float_bits(65519.0, 10, true), 0x7bff);
    assert_eq!(small_float_bits(2f32.powi(-24), 10, true), 0x0001);
    assert_eq!(small_float_bits(2f32.powi(-25), 10, true), 0x0000);
    assert_eq!(small_float_bits(1.5 * 2f32.powi(-25), 10, true), 0x0001);
    assert_eq!(small_float_bits(-2.0, 10, true), 0xc000);
    assert_eq!(small_float_bits(-2.0, 6, false), 0);
    assert_eq!(small_float_bits(f32::NAN, 6, false), 0x7e0);
}

/// The plan lays out each level, its words and its dispatch, and is `None` where the device's
/// limits do not fit or the filter does not compute the format.
#[test]
fn the_plan_lays_the_levels_and_refuses_what_it_cannot_run() {
    let pf = PixelFormat::RGBA8Unorm;
    let chain = MipmapFilterChain::plan(
        pf,
        [8, 4, 1],
        2,
        3,
        false,
        256,
        u32::MAX,
        [65535, 65535],
        None,
    )
    .expect("a chain");
    let offsets: Vec<u64> = chain.levels.iter().map(|l| l.offset).collect();
    // 8x4, 4x2 and 2x1 texels of 4 bytes over two layers: 256, 64 and 16 bytes; level 1's words
    // between levels 1 and 2.
    let (at1, w1) = chain.levels[1].params.clone().expect("params");
    let (at2, w2) = chain.levels[2].params.clone().expect("params");
    assert_eq!(offsets[..2], [0, 256]);
    assert_eq!(at1, 512);
    assert_eq!(
        offsets[2],
        (at1 + 4 * w1.len() as u64).next_multiple_of(256)
    );
    let bytes: Vec<u64> = chain.levels.iter().map(|l| l.bytes).collect();
    assert_eq!(bytes, [256, 64, 16]);
    assert!(chain.levels[0].params.is_none());
    assert_eq!(at2, (offsets[2] + 16).next_multiple_of(256));
    assert_eq!(
        chain.bytes,
        (at2 + 4 * w2.len() as u64).next_multiple_of(256)
    );
    // Level 2's header: 4 texels over two layers, one row of 64, its extent and the one above,
    // then its tap tables: x (2 entries), y (1), z (1).
    assert_eq!(w2[..8], [4, 64, 2, 1, 1, 4, 2, 1]);
    assert_eq!(w2[8..11], [11, 15, 17]);
    assert_eq!(
        w2[11..],
        [
            0,
            0.5f32.to_bits(),
            2,
            0.5f32.to_bits(),
            0,
            0.5f32.to_bits(),
            0,
            0
        ]
    );
    let kernel = MipmapFilterKernel {
        pixel_format: pf,
        depth: false,
        fetch: None,
    };
    let kernels: Vec<_> = chain.levels.iter().map(|l| l.kernel).collect();
    assert_eq!(kernels, [None, Some(kernel), Some(kernel)]);
    assert_eq!(chain.levels[2].groups, [1, 1]);
    assert_eq!(MipmapFilterModule::LOCAL_SIZE, 64);
    // A 4-byte texel's levels are written whole; a narrower texel's are zeroed and ORed into.
    assert!(!chain.zeroed);
    assert!(plan_of(PixelFormat::RG8Unorm, [8, 4, 1], 1, 2, false, None).zeroed);
    // A 3D level whose level above is more than one texel deep takes the depth taps.
    let deep = plan_of(pf, [8, 8, 2], 1, 3, true, None);
    let depths: Vec<_> = deep.levels[1..]
        .iter()
        .map(|l| l.kernel.expect("a kernel").depth)
        .collect();
    assert_eq!(depths, [true, false]);
    let none = |pf, extent, layers, levels, max_range, max_groups| {
        MipmapFilterChain::plan(
            pf, extent, layers, levels, false, 64, max_range, max_groups, None,
        )
        .is_none()
    };
    let (big, groups) = (u32::MAX, [65535, 65535]);
    // An integer format, a compressed one, a depth one, an extended-range form Metal's mipmaps
    // of are not measured.
    for other in [
        PixelFormat::RGBA8Uint,
        PixelFormat::BC1RGBA,
        PixelFormat::Depth32Float,
        PixelFormat::RGB10A82PXR10,
    ] {
        assert!(none(other, [8, 8, 1], 1, 2, big, groups), "{other:?}");
        assert_eq!(MipmapFilterChain::texel_bytes(other), None, "{other:?}");
    }
    assert_eq!(
        MipmapFilterChain::texel_bytes(PixelFormat::BGRA10XR),
        Some(8)
    );
    assert_eq!(
        MipmapFilterChain::texel_bytes(PixelFormat::B5G6R5Unorm),
        Some(2)
    );
    // One level, no layers, an empty extent.
    assert!(none(pf, [8, 8, 1], 1, 1, big, groups));
    assert!(none(pf, [8, 8, 1], 0, 2, big, groups));
    assert!(none(pf, [0, 8, 1], 1, 2, big, groups));
    // A level past the storage range, a dispatch past the workgroups.
    // (64x64 texels of 4 bytes: 16384; each level's params fit in far less.)
    assert!(none(pf, [64, 64, 1], 1, 2, 16383, groups));
    assert!(!none(pf, [64, 64, 1], 1, 2, 16384, groups));
    // Level 1: 64x64 texels, 64 workgroups.
    assert!(none(pf, [128, 128, 1], 1, 2, big, [1, 63]));
    assert!(!none(pf, [128, 128, 1], 1, 2, big, [1, 64]));
}

/// A fetch the chain cannot take, a format whose fetch converts or reads bits, and the view an
/// sRGB image is read through.
#[test]
fn the_plan_fetches_only_what_a_fetch_recovers() {
    let fetched = |pf, view| plan_of(pf, [8, 8, 1], 1, 3, false, view).fetch;
    // A view whose components are not the format's fields: none.
    assert_eq!(
        fetched(PixelFormat::RGBA8Unorm, Some(Format::R8G8B8A8_SNORM)),
        None
    );
    assert_eq!(
        fetched(PixelFormat::RGBA16Float, Some(Format::R16G16B16A16_UNORM)),
        None
    );
    // A format no fetch recovers: copied in.
    assert_eq!(
        fetched(PixelFormat::RGBA32Float, Some(Format::R32G32B32A32_SFLOAT)),
        None
    );
    assert_eq!(
        fetched(PixelFormat::RGB9E5Float, Some(Format::R32_UINT)),
        None
    );
    assert_eq!(fetched(PixelFormat::RGBA8Unorm, None), None);
    assert_eq!(
        fetched(PixelFormat::RGBA8Unorm, Some(Format::B8G8R8A8_UNORM)),
        Some(Format::B8G8R8A8_UNORM)
    );
    assert_eq!(
        fetched(PixelFormat::BGRA10XRSRGB, Some(Format::R16G16B16A16_UINT)),
        Some(Format::R16G16B16A16_UINT)
    );
    assert_eq!(
        MipmapFilterChain::fetch_format(Format::B8G8R8A8_SRGB),
        Some(Format::B8G8R8A8_UNORM)
    );
    assert_eq!(MipmapFilterChain::fetch_format(Format::R32_SFLOAT), None);
    assert!(MipmapFilterChain::fetch_converts(
        Format::R16G16B16A16_SFLOAT
    ));
    assert!(MipmapFilterChain::fetch_converts(
        Format::A2B10G10R10_UNORM_PACK32
    ));
    assert!(!MipmapFilterChain::fetch_converts(Format::R32_UINT));
    assert!(!MipmapFilterChain::fetch_converts(
        Format::R16G16B16A16_UINT
    ));
}
