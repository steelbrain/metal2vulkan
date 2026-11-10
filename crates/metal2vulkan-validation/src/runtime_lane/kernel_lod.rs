//! A kernel's LOD queries against Metal's measured answer on the device the cfg selects, under the
//! validation layer.
//!
//! Outside a fragment function Metal answers `calculate_unclamped_lod` with 0 and
//! `calculate_clamped_lod` with the sampler's minimum level bounded by the texture's last level,
//! whatever the coordinate (the M3 Max and an M4): the minimum level is 0 without a mip filter;
//! under a nearest mip filter `L + 1` where the fraction of `lodMinClamp` above
//! `L = floor(lodMinClamp)`, truncated onto sixty-fourths, exceeds one half,
//! else `L`; under a linear one `lodMinClamp` truncated onto sixty-fourths. A runtime sampler's
//! clamp is the descriptor's `f32`, a constexpr sampler's its `half`. The rule is written here
//! from the measurements, apart from the product's `metal::sampler::query_min_level`.
//!
//! The kernels are `rfs_kernel_lod`'s (`validation/fixtures/public/engine`): `lod_2d`,
//! `lod_2d_array`, `lod_3d` and `lod_cube` query through the runtime sampler at slot 0, on an
//! `r32Float` texture of five levels (and `lod_2d` on one of one level too), through each of the
//! clamps [`clamps`] under each mip filter and both filters; the constexpr kernels query through
//! their own samplers. Each thread of an 8x8 grid writes its clamped and its unclamped level, so
//! 64 coordinates per query.

use super::mip_levels::{half, texture, LEVELS, SIDE};
use super::{
    at_plan, buffer_with, layer_reported_nothing_at_all, read_buffer, validated_device_with,
    Messages, ScratchCache, WAIT,
};
use crate::air_library::AirLibrary;
use crate::gpu::TEST_DEVICE;
use metal2vulkan::bind::SamplerDescriptor;
use metal2vulkan::metal::{PixelFormat, SamplerMinMagFilter, SamplerMipFilter, TextureType};
use metal2vulkan::pipeline::{ComputePipelineDescriptor, StageSource};
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use metal2vulkan_runtime::{
    ComputePipelineState, Device, PipelineOptions, Region, ResourceOptions, SamplerState, Size,
    Texture, TextureDescriptor,
};

/// The threads of the 8x8 grid, each one query.
const THREADS: u64 = 64;

/// The level clamps the runtime samplers query through, as `f32`s: none, minimums on and off the
/// sixty-fourths and at and just past a nearest mip filter's half-way points, a maximum below the
/// last level, an empty clamp, and minimums at, past and far past the last level.
fn clamps() -> Vec<(f32, f32)> {
    let below = |x: f32| x.next_down();
    vec![
        (0.0, f32::MAX),
        (0.0001, f32::MAX),
        (0.3, 4.0),
        (0.5, f32::MAX),
        (below(33.0 / 64.0), f32::MAX),
        (33.0 / 64.0, f32::MAX),
        (0.7, 1.0),
        (1.1, 1.9),
        (1.5, f32::MAX),
        (2.0 + 1.0 / 256.0, 4.0),
        (2.2, 2.2),
        (2.7, 3.2),
        (3.99, f32::MAX),
        (4.0, f32::MAX),
        (4.9, 9.0),
        (10.0, 20.0),
        (f32::MAX, f32::MAX),
    ]
}

/// Metal's minimum level of a sampler of mip filter `mip` and minimum clamp `lo`, before the
/// texture's last level bounds it.
fn minimum(mip: SamplerMipFilter, lo: f32) -> f64 {
    let lo = f64::from(lo);
    let on_grid = |x: f64| (x * 64.0).floor() / 64.0;
    match mip {
        SamplerMipFilter::NotMipmapped => 0.0,
        SamplerMipFilter::Nearest => {
            let l = lo.floor();
            if on_grid(lo - l) > 0.5 {
                l + 1.0
            } else {
                l
            }
        }
        SamplerMipFilter::Linear => on_grid(lo),
        other => panic!("no measured rule for {other:?}"),
    }
}

/// Metal's clamped level for a texture of `levels` levels.
fn clamped(mip: SamplerMipFilter, lo: f32, levels: u64) -> f32 {
    minimum(mip, lo).min((levels - 1) as f64) as f32
}

/// The kernel `name` of the fixture's library on `device`.
fn pipeline(device: &Device, name: &str) -> ComputePipelineState {
    let bitcode = llvm_assemble(&engine_dir().join("rfs_kernel_lod.ll")).unwrap();
    let library = device.new_library_with_air(&bitcode).unwrap();
    let function = library.new_function_with_name(name).unwrap();
    device
        .new_compute_pipeline_state(
            &ComputePipelineDescriptor::new(StageSource::Function(function.function_descriptor())),
            &PipelineOptions::default(),
        )
        .unwrap()
}

/// A sampler state of `filter` both ways, `mip`, and `clamp`.
fn sampler(
    device: &Device,
    filter: SamplerMinMagFilter,
    mip: SamplerMipFilter,
    (lo, hi): (f32, f32),
) -> SamplerState {
    let mut descriptor = SamplerDescriptor::new();
    descriptor.min_filter = filter;
    descriptor.mag_filter = filter;
    descriptor.mip_filter = mip;
    descriptor.lod_min_clamp = lo;
    descriptor.lod_max_clamp = hi;
    device.new_sampler_state(&descriptor).unwrap()
}

/// A 2D `r32Float` texture of one level.
fn one_level(device: &Device) -> Texture {
    let descriptor = TextureDescriptor::texture_2d(PixelFormat::R32Float, SIDE, SIDE, false);
    let texture = device.new_texture(&descriptor).unwrap();
    let bytes = vec![0u8; (4 * SIDE * SIDE) as usize];
    texture
        .replace_region(
            Region::new_2d(0, 0, SIDE, SIDE),
            0,
            0,
            &bytes,
            4 * SIDE,
            4 * SIDE * SIDE,
        )
        .unwrap();
    texture
}

/// The (clamped, unclamped) levels `state` writes, `texture` and `sampler` at slot 0, one pair
/// per thread of the 8x8 grid.
fn run(
    device: &Device,
    state: &ComputePipelineState,
    texture: &Texture,
    sampler: Option<&SamplerState>,
) -> Vec<(f32, f32)> {
    let p: Vec<u8> = [1.0f32, 1.0, 0.5, 0.5]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let p = buffer_with(device, &p).unwrap();
    let length = THREADS * 8;
    let output = device
        .new_buffer(length, ResourceOptions::shared())
        .unwrap();
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(state).unwrap();
    encoder.set_texture(Some(texture), 0).unwrap();
    if let Some(sampler) = sampler {
        encoder.set_sampler_state(Some(sampler), 0).unwrap();
    }
    encoder.set_buffer(Some(&output), 0, 0).unwrap();
    encoder.set_buffer(Some(&p), 0, 1).unwrap();
    encoder
        .dispatch_threads(Size::new(8, 8, 1), Size::new(8, 8, 1))
        .unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
    read_buffer(&output, 0, length)
        .unwrap()
        .as_chunks::<8>()
        .0
        .iter()
        .map(|c| {
            let word = |at: usize| f32::from_le_bytes(c[at..at + 4].try_into().unwrap());
            (word(0), word(4))
        })
        .collect()
}

/// The mismatches of `got` against Metal's `want` clamped level and 0 unclamped, bit for bit.
fn check(what: &str, got: &[(f32, f32)], want: f32, wrong: &mut Vec<String>) {
    for (thread, &(c, u)) in got.iter().enumerate() {
        if c.to_bits() != want.to_bits() || u.to_bits() != 0 {
            wrong.push(format!(
                "{what} thread {thread}: clamped {c:e}, unclamped {u:e}; want {want:e} and 0"
            ));
        }
    }
}

/// Every LOD query of a kernel answers Metal's level: through a runtime sampler of each clamp
/// under each mip filter and both filters, on 2D (of five levels and of one), 2D-array, 3D and
/// cube textures, and through the fixture's constexpr samplers. Prints the queries checked.
#[test]
fn kernel_lod_queries_answer_metals_level() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), at_plan).unwrap();
        let name = device.name().to_owned();
        let level = |l: u64, _: u64, _: u64, _: u64| l as f32;
        let textures = [
            (
                "lod_2d",
                texture(&device, TextureType::Type2D, level),
                LEVELS,
            ),
            ("lod_2d", one_level(&device), 1),
            (
                "lod_2d_array",
                texture(&device, TextureType::Type2DArray, level),
                LEVELS,
            ),
            (
                "lod_3d",
                texture(&device, TextureType::Type3D, level),
                LEVELS,
            ),
            (
                "lod_cube",
                texture(&device, TextureType::Cube, level),
                LEVELS,
            ),
        ];
        let mut wrong = Vec::new();
        let mut checked = 0usize;
        for (kernel, texture, levels) in &textures {
            let state = pipeline(&device, kernel);
            for clamp in clamps() {
                for mip in [
                    SamplerMipFilter::NotMipmapped,
                    SamplerMipFilter::Nearest,
                    SamplerMipFilter::Linear,
                ] {
                    for filter in [SamplerMinMagFilter::Nearest, SamplerMinMagFilter::Linear] {
                        let s = sampler(&device, filter, mip, clamp);
                        let got = run(&device, &state, texture, Some(&s));
                        let what = format!(
                            "{kernel} ({levels} levels) {filter:?} {mip:?} [{:e}, {:e}]",
                            clamp.0, clamp.1
                        );
                        check(&what, &got, clamped(mip, clamp.0, *levels), &mut wrong);
                        checked += got.len();
                    }
                }
            }
        }
        // The constexpr samplers: (kernel, texture, mip filter, minimum clamp as written).
        let constexpr = [
            ("lod_2d_linear_clamp", 0, SamplerMipFilter::Linear, 0.3),
            ("lod_2d_nearest_clamp", 0, SamplerMipFilter::Nearest, 2.7),
            (
                "lod_2d_unmipped_clamp",
                0,
                SamplerMipFilter::NotMipmapped,
                2.5,
            ),
            ("lod_2d_linear_open", 0, SamplerMipFilter::Linear, 0.0),
            ("lod_2d_linear_past", 0, SamplerMipFilter::Linear, 10.0),
            ("lod_2d_linear_clamp", 1, SamplerMipFilter::Linear, 0.3),
            ("lod_2d_nearest_clamp", 1, SamplerMipFilter::Nearest, 2.7),
            ("lod_3d_nearest_tie", 3, SamplerMipFilter::Nearest, 1.5),
            ("lod_cube_linear_clamp", 4, SamplerMipFilter::Linear, 1.1),
        ];
        for (kernel, at, mip, lo) in constexpr {
            let (_, texture, levels) = &textures[at];
            let state = pipeline(&device, kernel);
            let got = run(&device, &state, texture, None);
            let what = format!("{kernel} ({levels} levels)");
            check(&what, &got, clamped(mip, half(lo), *levels), &mut wrong);
            checked += got.len();
        }
        println!(
            "KERNEL-LOD {name}: {checked} queries checked, {} differ from Metal",
            wrong.len()
        );
        for line in wrong.iter().take(60) {
            println!("KERNEL-LOD-DIFFERS {name}: {line}");
        }
        assert!(wrong.is_empty(), "{name}: {} queries differ", wrong.len());
    }
    layer_reported_nothing_at_all(&messages);
}

/// The rule's own arithmetic at the measured points: a nearest mip filter's half-way point, the
/// grid's truncation, and the bound at the last level.
#[test]
fn the_rule_reads_the_measured_levels() {
    use SamplerMipFilter::{Linear, Nearest, NotMipmapped};
    for (mip, lo, levels, want) in [
        (Linear, 0.3, 5, 0.296_875),
        (Linear, half(0.3), 5, 0.296_875),
        (Linear, 13.99, 15, 13.984_375),
        (Linear, 14.9, 15, 14.0),
        (Linear, f32::MAX, 15, 14.0),
        (Nearest, 13.6, 15, 14.0),
        (Nearest, 0.507_812_5, 5, 0.0),
        (Nearest, 0.515_625, 5, 1.0),
        (Nearest, 1.5, 5, 1.0),
        (NotMipmapped, 2.5, 5, 0.0),
        (Linear, 2.5, 1, 0.0),
    ] {
        assert_eq!(
            clamped(mip, lo, levels),
            want,
            "{mip:?} {lo:e} over {levels}"
        );
    }
}
