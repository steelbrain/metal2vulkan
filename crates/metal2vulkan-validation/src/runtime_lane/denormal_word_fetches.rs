//! A device test of denormal 32-bit float texels read through a runtime sampler, an argument
//! buffer's sampler, and a heap texture on the device the cfg selects, under the validation layer:
//! a gather and a nearest sample of an `R32Float` texture whose texels are mostly denormals of both
//! signs and signed zeros answer each texel's bits, as Metal's do (Metal keeps a denormal texel;
//! its runtime and argument buffer's samplers filter as its constexpr samplers of the same state).
//!
//! RADV's gathers and nearest samples, NVIDIA's gathers and ANV's nearest samples read such a
//! texel as the zero of its sign. Those are the device's own samples, which the stages record
//! (`PrecisionNote::DeviceSampling`): off MoltenVK the lane prints the lanes that differ
//! (`DENORMAL-TEXEL-DIFFERS` lines) without failing. The texture is read at texture slot 0
//! through runtime sampler slot 0, and as the texture an argument buffer holds through runtime
//! sampler slot 0, through the sampler the argument buffer holds, and through a constexpr sampler:
//! a gather at each interior texel corner reads the four texels around it, in gather order, and a
//! nearest sample at each texel centre its texel. The samplers are nearest, without a mip filter,
//! clamped to the edge, normalized. On MoltenVK (`VkProperty::MetalSampler`) Metal's own sampler
//! reads them, which must keep every texel's bits.

use super::argument_linear_samples::kernel;
use super::degrade_notices::counts;
use super::depth_compares::{pipeline_of, run_held, Held, Through};
use super::{
    at_plan, layer_reported_nothing_at_all, validated_device_with, Messages, ScratchCache,
};
use crate::gpu::TEST_DEVICE;
use metal2vulkan::bind::SamplerDescriptor;
use metal2vulkan::device::VkProperty;
use metal2vulkan::metal::PixelFormat;
use metal2vulkan_devtools::llvm_assemble_text;
use metal2vulkan_runtime::{Degradation, Region, TextureDescriptor};

/// A constexpr sampler's words: nearest filters, no mip filter, normalized coordinates clamped to
/// the edge.
const NEAREST_EDGE: u64 = 0x007b_ff00_0008_0000 | 1 | 1 << 3 | 1 << 6;

/// The texels of the 4x4 `R32Float` texture, row by row: denormals of both signs, signed zeros,
/// the smallest normal, and ordinary values.
const TEXELS: [u32; 16] = [
    0x0000_0001,
    0x8000_0002,
    0x007f_ffff,
    0x807f_ffff,
    0x0040_0000,
    0x8000_0000,
    0x0000_0000,
    0x0080_0000,
    0x3f80_0000,
    0xbf00_0000,
    0x0012_3456,
    0x8065_4321,
    0x4120_0000,
    0x0000_0100,
    0x8000_0001,
    0x3e80_0000,
];

/// [`kernel`]'s 2D `float` kernel of `heap` and `through`, its sample a gather of component 0
/// where `gather`.
fn plain_kernel(heap: bool, through: Through, gather: bool) -> String {
    let text = kernel(heap, through, false, false);
    if !gather {
        return text;
    }
    let mut out = text;
    for (from, to) in [
        (
            "@air.sample_texture_2d.v4f32(",
            "@air.gather_texture_2d.v4f32(",
        ),
        (
            "<2 x i32> zeroinitializer, i1 true, float 0.0, float 0.0, i32 0)",
            "<2 x i32> zeroinitializer, i32 0, i32 0)",
        ),
        (
            "<2 x float>, i1, <2 x i32>, i1, float, float, i32)",
            "<2 x float>, i1, <2 x i32>, i32, i32)",
        ),
    ] {
        assert!(out.contains(from), "no {from}");
        out = out.replace(from, to);
    }
    out
}

/// Every gather at an interior texel corner and every nearest sample at a texel centre of the
/// denormal texture, through a runtime sampler slot (of a slot texture and of a heap texture),
/// an argument buffer's sampler and a constexpr sampler (of a heap texture), answers the bits of
/// Metal's texels (module docs).
#[test]
fn denormal_texels_read_through_words_and_heaps_keep_their_bits() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), at_plan).unwrap();
        let name = device.name().to_owned();
        let own = !device
            .compile_device()
            .profile()
            .enabled()
            .properties
            .contains(&VkProperty::MetalSampler);
        let texture = device
            .new_texture(&TextureDescriptor::texture_2d(
                PixelFormat::R32Float,
                4,
                4,
                false,
            ))
            .unwrap();
        let bytes: Vec<u8> = TEXELS.iter().flat_map(|t| t.to_le_bytes()).collect();
        texture
            .replace_region(Region::new_2d(0, 0, 4, 4), 0, 0, &bytes, 16, 0)
            .unwrap();
        let sampler = device.new_sampler_state(&SamplerDescriptor::new()).unwrap();
        let texel = |x: u32, y: u32| TEXELS[(4 * y + x) as usize];
        let corners: Vec<(u32, u32)> = (0..3).flat_map(|j| (0..3).map(move |i| (i, j))).collect();
        let centres: Vec<(u32, u32)> = (0..4).flat_map(|y| (0..4).map(move |x| (x, y))).collect();
        let mut compared = 0usize;
        let mut wrong = Vec::new();
        for gather in [true, false] {
            let (jobs, want): (Vec<[f32; 4]>, Vec<[u32; 4]>) = match gather {
                true => corners
                    .iter()
                    .map(|&(i, j)| {
                        (
                            [(i + 1) as f32 / 4.0, (j + 1) as f32 / 4.0, 0.0, 0.0],
                            [
                                texel(i, j + 1),
                                texel(i + 1, j + 1),
                                texel(i + 1, j),
                                texel(i, j),
                            ],
                        )
                    })
                    .unzip(),
                false => centres
                    .iter()
                    .map(|&(x, y)| {
                        (
                            [(x as f32 + 0.5) / 4.0, (y as f32 + 0.5) / 4.0, 0.0, 0.0],
                            [texel(x, y), 0, 0, 1f32.to_bits()],
                        )
                    })
                    .unzip(),
            };
            let read = if gather { "gather" } else { "nearest sample" };
            for (what, heap, through, held, runtime) in [
                (
                    "slot texture, runtime sampler",
                    false,
                    Through::Runtime,
                    Held::Slot,
                    true,
                ),
                (
                    "heap texture, runtime sampler",
                    true,
                    Through::Runtime,
                    Held::Argument(None),
                    true,
                ),
                (
                    "heap texture, argument buffer's sampler",
                    true,
                    Through::Argument,
                    Held::Argument(Some(&sampler)),
                    false,
                ),
                (
                    "heap texture, constexpr sampler",
                    true,
                    Through::Constexpr(NEAREST_EDGE),
                    Held::Argument(None),
                    false,
                ),
            ] {
                let text = plain_kernel(heap, through, gather);
                let state = pipeline_of(&device, &llvm_assemble_text(&text).unwrap(), "k");
                let s = runtime.then_some(&sampler);
                let out = run_held(&device, (&texture, held, s), &state, &jobs, 1);
                for (j, want) in want.iter().enumerate() {
                    for (lane, &w) in want.iter().enumerate() {
                        let got = out[4 * j + lane].to_bits();
                        compared += 1;
                        if got != w {
                            wrong.push(format!(
                                "{what} {read} at {:?} lane {lane}: {got:#010x}, Metal's {w:#010x}",
                                &jobs[j][..2]
                            ));
                        }
                    }
                }
            }
        }
        eprintln!("{name}: {} of {compared} lanes differ", wrong.len());
        let announced = counts(&messages.take_notices());
        assert_eq!(
            announced.contains_key(&Degradation::DeviceSampling),
            own,
            "{name}: device-sampling exactly where the device's sampler is not Metal's"
        );
        if own {
            for w in &wrong {
                eprintln!("DENORMAL-TEXEL-DIFFERS {name}: {w}");
            }
        } else {
            assert!(
                wrong.is_empty(),
                "{name}: {} of {compared} lanes differ:\n{}",
                wrong.len(),
                wrong.join("\n")
            );
        }
    }
    layer_reported_nothing_at_all(&messages);
}
