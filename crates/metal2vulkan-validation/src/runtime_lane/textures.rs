//! Device tests of the runtime's textures on the device the cfg selects: textures of every pixel
//! format made or refused typed, CPU copies through `replace_region` and `get_bytes`, views of
//! levels, slices and formats, a view of another component layout read in a kernel, and Metal's
//! refusals, under the validation layer.

use super::{
    at_plan, buffer_with, layer_reported_nothing_at_all, read_buffer, validated_device,
    validated_device_with, Messages, ScratchCache, WAIT,
};
use crate::candidate::runtime_samples::encode_through;
use crate::candidate::runtime_samples::SampleFill;
use crate::case::{TextureFormat as CaseFormat, TextureType as CaseType};
use crate::executor_contract::multisample::{ReadbackView, SampleRead};
use crate::gpu::TEST_DEVICE;
use crate::literal::texture_layout;
use metal2vulkan::error::{Feature, Needs};
use metal2vulkan::metal::{PixelFormat, TextureType, TextureUsage};
use metal2vulkan_runtime::{
    BlitOption, Buffer, CommandBufferStatus, CopyProblem, CpuCacheMode, Device, Error,
    HazardTrackingMode, HeapDescriptor, HeapType, Invalid, Origin, Problem, Region,
    ResourceOptions, Size, StorageMode, Texture, TextureDescriptor, TextureProblem, ViewProblem,
};

fn invalid(call: &'static str, problem: Problem) -> Error {
    Error::Invalid(Invalid::new(call, problem))
}

/// `n` bytes of a pattern seeded by `seed`.
pub(super) fn pattern(n: usize, seed: u8) -> Vec<u8> {
    (0..n)
        .map(|i| (i as u8).wrapping_mul(31).wrapping_add(seed))
        .collect()
}

/// The bytes of one texel of `pf`, for the uncompressed color formats this test round-trips:
/// none for a format whose image does not hold Metal's bytes (`byte_format`), whose CPU copies
/// the runtime refuses.
pub(super) fn texel_bytes(pf: PixelFormat) -> Option<usize> {
    let block = metal2vulkan::metal::format::byte_format(pf)?.block()?;
    let compressed =
        block.texels > 1 || (block.packed_bits.is_none() && block.component_bits.is_none());
    (block.color.is_some() && !compressed).then_some(block.size as usize)
}

/// Whether Metal makes no texture of `pf` from a descriptor alone, answering nil to the device's
/// and a heap's calls (`TextureProblem::NoTextureOfFormat`), in the storage modes those callers
/// ask: `Shared` for the device's textures here, `Private` and the heap's own for
/// `heap_textures`. Measured on an Apple M4 under macOS 27.0.1: the two-plane YCbCr formats
/// (each of the 69), made only over an IOSurface's planes, and `RGBA16Float_XR`, `AnyDepth` and
/// `AnyStencil`.
///
/// `R10Unorm_PACKED` and `RG10Unorm_PACKED` are here on a measurement of their own, re-read on an
/// Apple M3 Max under macOS 27.0.1 (26A434) with one cell per fresh process, four processes a
/// cell, and the storage mode varied rather than left to a default, because a storage mode's nil
/// and a format's nil look identical at the call: `newTextureWithDescriptor:` for a 2D 8x8
/// texture answered nil for both formats in `Private`, `Shared` and `Managed` storage, 4 of 4 in
/// each of `ShaderRead`, `ShaderRead | ShaderWrite`, `ShaderRead | RenderTarget` and
/// `RenderTarget` usage, while `RGBA8Unorm` and `R10Unorm_X6` beside them in the same mode and
/// usage were made 4 of 4 — so neither the mode nor the usage refused the cell, and neither does
/// "padded 10-bit" stand in for these two. Their heap size query answers 0 bytes
/// (`metal::format::Footprint::NoTexture`), which is a separate path and not what settles this.
///
/// `Memoryless` storage is the one mode where that same measurement made a texture of both, 4 of
/// 4 at all four usages. No caller of this helper asks it (a heap holds no memoryless texture,
/// and the device's textures here are `Shared`), so the helper does not answer for it: the
/// runtime's memoryless answer is its own refusal, which
/// [`memoryless_packed_ten_bit_is_our_gap_and_not_metals_nil`] holds.
pub(super) fn no_texture_alone(pf: PixelFormat) -> bool {
    metal2vulkan::metal::format::ycbcr_format(pf).is_some()
        || matches!(
            pf,
            PixelFormat::RGBA16FloatXR
                | PixelFormat::AnyDepth
                | PixelFormat::AnyStencil
                | PixelFormat::R10UnormPACKED
                | PixelFormat::RG10UnormPACKED
        )
}

/// What `Device::new_texture` answers for a descriptor, as the three answers this rule has to
/// keep apart: a texture, Metal's nil, and the runtime's own typed refusal. A test that only
/// asked "not a texture" would pass for either refusal, and the whole point of
/// [`memoryless_packed_ten_bit_is_our_gap_and_not_metals_nil`] is which of the two it is.
#[derive(Debug, PartialEq, Eq)]
enum Answer {
    /// A texture of the format asked for.
    Texture(PixelFormat),
    /// `Error::Nil`: plain Metal makes no texture of this descriptor.
    Nil(Problem),
    /// `Error::Unsupported`: Metal makes the texture and this crate does not build it.
    Unsupported(Feature, Needs, Option<String>),
    /// Any other refusal, rendered.
    Other(String),
}

/// [`Answer`] for `device.new_texture(descriptor)`.
fn answer(device: &Device, descriptor: &TextureDescriptor) -> Answer {
    match device.new_texture(descriptor) {
        Ok(texture) => Answer::Texture(texture.pixel_format()),
        Err(Error::Nil(nil)) => Answer::Nil(nil.problem),
        Err(Error::Unsupported(u)) => {
            Answer::Unsupported(u.feature, u.needs, u.symbol.as_ref().map(|s| s.to_string()))
        }
        Err(other) => Answer::Other(format!("{other:?}")),
    }
}

/// `R10Unorm_PACKED` and `RG10Unorm_PACKED` are the one family plain Metal makes a texture of in
/// `Memoryless` storage and in no mode that holds memory, so the runtime's answer there is its
/// own gap and not Metal's nil: `Unsupported(Format, NotImplemented)` naming the format, as every
/// format this crate holds no image of answers, where `Private` and `Shared` keep
/// `Nil(NoTextureOfFormat)`.
///
/// The distinction is the point. `Error::Nil` says *Metal* rejects the descriptor, which the
/// measurement shows is false for `Memoryless`, and a consumer branching on it would conclude its
/// own descriptor is wrong. `Error::Unsupported` says this crate does not implement the format,
/// which is true: both are in the translator's `formats_named_not_held!` family, so no image of
/// either exists to back a texture. So the three answers are told apart here ([`Answer`]) rather
/// than collapsed into "not a texture".
///
/// Measured on an Apple M3 Max under macOS 27.0.1 (26A434) with plain Metal, one cell per fresh
/// process and four processes a cell, each cell printing its own format, storage mode and usage
/// (`metal2vulkan-conformance/probes/packed10_texture_alone.m`):
/// `newTextureWithDescriptor:` for a 2D 8x8 texture of either format answered **nil** 4 of 4 in
/// `Private`, `Shared` and `Managed` storage and a **texture** 4 of 4 in `Memoryless`, under each
/// of `ShaderRead`, `ShaderRead | ShaderWrite`, `ShaderRead | RenderTarget` and `RenderTarget`
/// usage. `RGBA8Unorm` and `R10Unorm_X6` stood beside every cell in the same mode and usage and
/// were made in all of them, so no cell's answer is its mode's or its usage's, and "padded
/// 10-bit" does not stand in for these two. `Managed` is left out below: this test asks the
/// runtime, whose managed textures are a question of their own.
#[test]
fn memoryless_packed_ten_bit_is_our_gap_and_not_metals_nil() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let usages = [
            TextureUsage::SHADER_READ,
            TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE,
            TextureUsage::SHADER_READ | TextureUsage::RENDER_TARGET,
            TextureUsage::RENDER_TARGET,
        ];
        let mut cells = 0;
        let mut controls = 0;
        for mode in [
            StorageMode::Private,
            StorageMode::Shared,
            StorageMode::Memoryless,
        ] {
            let options = ResourceOptions::new(
                mode,
                CpuCacheMode::DefaultCache,
                HazardTrackingMode::Default,
            );
            for usage in usages {
                let of = |pf| {
                    let mut d = TextureDescriptor::texture_2d(pf, 8, 8, false);
                    d.usage = usage;
                    d.resource_options = options;
                    d
                };
                // The controls, in this very mode and usage: a mode's or a usage's refusal would
                // show here first and the cell below would say nothing.
                for control in [PixelFormat::RGBA8Unorm, PixelFormat::R10UnormX6] {
                    assert_eq!(
                        answer(&device, &of(control)),
                        Answer::Texture(control),
                        "{control:?} {mode:?} {usage:?}"
                    );
                    controls += 1;
                }
                for pf in [PixelFormat::R10UnormPACKED, PixelFormat::RG10UnormPACKED] {
                    let want = match mode {
                        StorageMode::Memoryless => Answer::Unsupported(
                            Feature::Format,
                            Needs::NotImplemented,
                            Some(pf.name().to_owned()),
                        ),
                        _ => Answer::Nil(Problem::Texture(TextureProblem::NoTextureOfFormat)),
                    };
                    assert_eq!(answer(&device, &of(pf)), want, "{pf:?} {mode:?} {usage:?}");
                    cells += 1;
                }
            }
        }
        assert_eq!((cells, controls), (24, 24));
        // The carve is keyed on the format's footprint and not on a shape, so every type it lets
        // through must answer the same refusal and none may answer a texture: this crate holds no
        // image and no texel-buffer format for either, so a texture of any shape would be a claim
        // it cannot back. A memoryless texture buffer is the one that could slip -- the runtime
        // makes one of `RGBA8Unorm`, announcing it -- so it is asked here beside a 2D array.
        let memoryless = ResourceOptions::new(
            StorageMode::Memoryless,
            CpuCacheMode::DefaultCache,
            HazardTrackingMode::Default,
        );
        for pf in [PixelFormat::R10UnormPACKED, PixelFormat::RG10UnormPACKED] {
            let want = Answer::Unsupported(
                Feature::Format,
                Needs::NotImplemented,
                Some(pf.name().to_owned()),
            );
            let mut array = TextureDescriptor::texture_2d(pf, 8, 8, false);
            array.texture_type = TextureType::Type2DArray;
            array.array_length = 3;
            array.usage = TextureUsage::SHADER_READ | TextureUsage::RENDER_TARGET;
            array.resource_options = memoryless;
            assert_eq!(answer(&device, &array), want, "{pf:?} memoryless 2D array");
            let texels =
                TextureDescriptor::texture_buffer(pf, 64, memoryless, TextureUsage::SHADER_READ);
            assert_eq!(answer(&device, &texels), want, "{pf:?} memoryless buffer");
        }
        eprintln!(
            "packed 10-bit textures on {}: {cells} cells beside {controls} controls",
            device.name()
        );
    }
    layer_reported_nothing_at_all(&messages);
}

/// `R10Unorm_X6` and `RG10Unorm_X12`, and their `_sRGB` twins, are made in memory of their own as
/// readily as over memory a texture is given, as Metal's are: the device's `new_texture`, an
/// automatic heap's, a placement heap's `new_texture_at_offset`, the heap size query, a caller's
/// buffer's and `new_shared_texture` all answer a texture, for a 2D texture and a texture buffer
/// alike. A multisample one stays refused, Metal's validation stopping it ("does not support
/// multisample textures").
///
/// Measured on an Apple M3 Max under macOS 27.0.1 (26A434) for all four formats: every one of
/// those creators answered a texture in `Shared`, `Private` and `Memoryless` storage, under
/// `ShaderRead`, `ShaderRead | ShaderWrite` and `ShaderRead | RenderTarget` usage, with four
/// levels, as an array of three slices and as a texture buffer of 64 texels; a sparse heap placed
/// one too; and the heap size query answered the tight linear size at an alignment of 128. The
/// multisample descriptor ended the process in Metal's validation.
///
/// `newSharedTextureWithDescriptor:` answers a shareable texture of each of the four as well,
/// re-read on that M3 Max with one cell per exec'd process and eight processes a cell: 2D 8x8,
/// 2D 16x16, a texture buffer of 8 texels and one of 16, `Private` storage under `ShaderRead`,
/// 8 of 8 each, the handle made. Its nil is the storage mode's and no format's: `Shared`,
/// `Managed` and `Memoryless` answered nil 8 of 8 for these four and for `RGBA8Unorm` and
/// `R16Uint` beside them.
#[test]
fn padded_formats_are_made_in_memory_of_their_own() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let private = ResourceOptions::private();
        let heap = device
            .new_heap(&HeapDescriptor::new(HeapType::Automatic, 1 << 20, private))
            .unwrap();
        for pf in [
            PixelFormat::R10UnormX6,
            PixelFormat::RG10UnormX12,
            PixelFormat::R10UnormX6SRGB,
            PixelFormat::RG10UnormX12SRGB,
        ] {
            let usage = TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE;
            let mut plain = TextureDescriptor::texture_2d(pf, 16, 16, false);
            plain.usage = usage;
            plain.resource_options = private;
            let texels = TextureDescriptor::texture_buffer(pf, 64, private, usage);
            for (what, d) in [("2D", &plain), ("texture buffer", &texels)] {
                let what = format!("{pf:?} {what}");
                let made = |r: Result<Texture, Error>, who: &str| {
                    let texture = r.unwrap_or_else(|e| panic!("{what}: {who}: {e}"));
                    assert_eq!(texture.pixel_format(), pf, "{what}: {who}");
                };
                made(device.new_texture(d), "the device's");
                made(heap.new_texture(d), "an automatic heap's");
                let answer = device
                    .heap_texture_size_and_align(d)
                    .unwrap_or_else(|e| panic!("{what}: the heap size query: {e}"));
                assert!(answer.size > 0, "{what}: the heap size query: {answer:?}");
                let placing =
                    HeapDescriptor::new(HeapType::Placement, answer.size.max(1 << 16), private);
                let placed = device.new_heap(&placing).unwrap();
                made(
                    placed.new_texture_at_offset(d, 0),
                    "a placement heap's at 0",
                );
                // The shareable creator makes one too, as plain Metal's does.
                made(device.new_shared_texture(d), "the shareable creator's");
            }
            // A multisample one stays refused: Metal's validation stops it.
            let mut multisample = plain;
            multisample.texture_type = TextureType::Type2DMultisample;
            multisample.sample_count = 4;
            multisample.usage = TextureUsage::RENDER_TARGET;
            assert_eq!(
                device.new_texture(&multisample).err(),
                Some(invalid(
                    "new_texture",
                    Problem::Texture(TextureProblem::TypeForFormat)
                )),
                "{pf:?} multisample"
            );
            let bpr = 256;
            let buffer = device
                .new_buffer(bpr * 16, ResourceOptions::shared())
                .unwrap();
            let mut over = plain;
            over.resource_options = ResourceOptions::shared();
            let texture = buffer
                .new_texture(&over, 0, bpr)
                .unwrap_or_else(|e| panic!("{pf:?} over a buffer: {e}"));
            assert_eq!(texture.pixel_format(), pf);
        }
    }
    layer_reported_nothing_at_all(&messages);
}

/// A padded 10-bit texture in memory of its own reads zero when new and round-trips its words
/// through `replace_region` and `get_bytes` at two extents, 8x8 and 64x64, with the bytes between
/// its rows untouched both ways.
///
/// This is where the runtime is more exact than Metal, and it is an accepted difference
/// (docs/API.md). Metal's own host copies of such a texture are exact only while it keeps the
/// texture linear: measured on an Apple M3 Max under macOS 27.0.1, `replaceRegion:` then
/// `getBytes:` returned the written bytes at 8x8 and at every height to 8 at widths to 1024, and
/// at 16x16 once the usage included `ShaderWrite`; at 16x16 and 64x64 under `ShaderRead` alone,
/// where `allocatedSize` carries 128 bytes or more beyond the tight linear size, both calls were
/// silent no-ops — `getBytes:` left its destination untouched and a shader read 0 at every texel
/// after a `replaceRegion:` — while a blit either way still moved the texels. Predicting which
/// extents Metal compresses is predicting an undocumented, per-chip choice, so the runtime moves
/// the caller's bytes at every extent instead.
#[test]
fn padded_textures_of_their_own_memory_round_trip_their_words() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        for pf in [
            PixelFormat::R10UnormX6,
            PixelFormat::RG10UnormX12,
            PixelFormat::R10UnormX6SRGB,
            PixelFormat::RG10UnormX12SRGB,
        ] {
            let bytes =
                texel_bytes(pf).unwrap_or_else(|| panic!("{pf:?}: a padded word has a texel size"));
            for side in [8u64, 64] {
                let mut d = TextureDescriptor::texture_2d(pf, side, side, false);
                d.usage = TextureUsage::SHADER_READ;
                let texture = device
                    .new_texture(&d)
                    .unwrap_or_else(|e| panic!("{pf:?} {side}x{side}: {e}"));
                // Rows of `side` texels, one texel of padding after each.
                let row = side as usize * bytes;
                let bpr = row + bytes;
                let n = bpr * side as usize;
                let region = Region::new_2d(0, 0, side, side);
                let mut fresh = vec![0xeeu8; n];
                texture
                    .get_bytes(&mut fresh, bpr as u64, 0, region, 0, 0)
                    .unwrap_or_else(|e| panic!("{pf:?} {side}: {e}"));
                for y in 0..side as usize {
                    assert!(
                        fresh[y * bpr..y * bpr + row].iter().all(|&b| b == 0),
                        "{pf:?} {side}x{side}: a new Shared texture reads zero"
                    );
                    assert!(
                        fresh[y * bpr + row..(y + 1) * bpr]
                            .iter()
                            .all(|&b| b == 0xee),
                        "{pf:?} {side}x{side}: the bytes between the rows were written"
                    );
                }
                let src = pattern(n, pf.as_raw() as u8);
                texture
                    .replace_region(region, 0, 0, &src, bpr as u64, 0)
                    .unwrap_or_else(|e| panic!("{pf:?} {side}: {e}"));
                let mut back = vec![0xeeu8; n];
                texture
                    .get_bytes(&mut back, bpr as u64, 0, region, 0, 0)
                    .unwrap_or_else(|e| panic!("{pf:?} {side}: {e}"));
                for y in 0..side as usize {
                    let r = y * bpr..y * bpr + row;
                    assert_eq!(back[r.clone()], src[r], "{pf:?} {side}x{side} row {y}");
                    assert!(
                        back[y * bpr + row..(y + 1) * bpr]
                            .iter()
                            .all(|&b| b == 0xee),
                        "{pf:?} {side}x{side}: the bytes between the rows were written"
                    );
                }
            }
        }
    }
    layer_reported_nothing_at_all(&messages);
}

/// A 2D texture of `pf` and `extent`, read by shaders, with `levels` levels.
fn texture_2d(
    device: &Device,
    pf: PixelFormat,
    extent: u64,
    levels: u64,
) -> Result<Texture, Error> {
    let mut d = TextureDescriptor::texture_2d(pf, extent, extent, false);
    d.mipmap_level_count = levels;
    device.new_texture(&d)
}

/// Every pixel format makes a texture or is refused typed, and a format Metal describes none of is
/// stopped (`NotAFormat`), the 24-bit depth formats on every venue; every
/// uncompressed color format's new `Shared` texture reads zero, as Metal's do, takes bytes through
/// `replace_region` and gives the same bytes back through `get_bytes`, with padded rows left alone.
#[test]
fn every_format_makes_a_texture_or_is_refused_typed() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    let (mut made, mut refused, mut round_trips) = (0, 0, 0);
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        // The formats Metal describes (the 24-bit depth formats not among them).
        let describes = |pf: PixelFormat| {
            device
                .pixel_format_info(u64::from(pf.as_raw()))
                .describes_a_format()
        };
        for &pf in PixelFormat::ALL {
            match texture_2d(&device, pf, 8, 1) {
                Ok(texture) => {
                    made += 1;
                    assert!(describes(pf), "{pf:?} made");
                    assert_eq!(texture.pixel_format(), pf);
                    assert_eq!(
                        (texture.width(), texture.height(), texture.depth()),
                        (8, 8, 1)
                    );
                    let Some(bytes) = texel_bytes(pf) else {
                        continue;
                    };
                    // Rows of 8 texels, a texel of padding after each.
                    let row = 8 * bytes;
                    let bpr = row + bytes;
                    let region = Region::new_2d(0, 0, 8, 8);
                    let mut fresh = vec![0xeeu8; bpr * 8];
                    texture
                        .get_bytes(&mut fresh, bpr as u64, 0, region, 0, 0)
                        .unwrap_or_else(|e| panic!("{pf:?}: {e}"));
                    for y in 0..8 {
                        assert!(
                            fresh[y * bpr..y * bpr + row].iter().all(|&b| b == 0),
                            "{pf:?}: a new Shared texture reads zero"
                        );
                    }
                    let src = pattern(bpr * 8, pf.as_raw() as u8);
                    texture
                        .replace_region(region, 0, 0, &src, bpr as u64, 0)
                        .unwrap_or_else(|e| panic!("{pf:?}: {e}"));
                    let mut back = vec![0xeeu8; bpr * 8];
                    texture
                        .get_bytes(&mut back, bpr as u64, 0, region, 0, 0)
                        .unwrap_or_else(|e| panic!("{pf:?}: {e}"));
                    for y in 0..8 {
                        let r = y * bpr..y * bpr + row;
                        assert_eq!(back[r.clone()], src[r], "{pf:?} row {y}");
                        assert!(back[y * bpr + row..(y + 1) * bpr]
                            .iter()
                            .all(|&b| b == 0xee));
                    }
                    round_trips += 1;
                }
                // `Invalid`, and every format the device's Metal describes none of, which
                // Metal's validation stops as it stops `Invalid` ("has invalid pixelFormat").
                Err(Error::Invalid(Invalid {
                    problem: Problem::Texture(TextureProblem::NotAFormat),
                    ..
                })) => {
                    assert!(!describes(pf), "{pf:?} stopped");
                    refused += 1;
                }
                // Metal makes no such texture from a descriptor alone: it answers nil (a
                // two-plane YCbCr texture is made over a surface only).
                Err(Error::Nil(nil)) => {
                    let want = match pf {
                        PixelFormat::Unspecialized => TextureProblem::Unspecialized,
                        _ => {
                            assert!(no_texture_alone(pf), "{pf:?}");
                            TextureProblem::NoTextureOfFormat
                        }
                    };
                    assert_eq!(nil.problem, Problem::Texture(want), "{pf:?}");
                    refused += 1;
                }
                Err(Error::Unsupported(u)) => {
                    assert!(
                        matches!(u.feature, Feature::Format | Feature::TextureSize),
                        "{pf:?}: {u:?}"
                    );
                    refused += 1;
                }
                Err(other) => panic!("{pf:?}: {other}"),
            }
        }
        eprintln!(
            "textures on {}: {made} formats made, {refused} refused, {round_trips} round trips",
            device.name()
        );
        assert!(made > 50 && round_trips > 40, "{made} {round_trips}");
    }
    layer_reported_nothing_at_all(&messages);
}

/// Levels, slices and 3D images copy where Metal puts them; views present their texture's levels
/// and slices and another format of the same bytes; Metal's refusals are typed.
#[test]
fn textures_copy_levels_slices_and_views() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), at_plan).unwrap();
        let rgba = PixelFormat::RGBA8Unorm;

        // A 2D array of 3 slices and 3 levels: each level and slice its own bytes.
        let mut d = TextureDescriptor::texture_2d(rgba, 8, 8, true);
        d.texture_type = TextureType::Type2DArray;
        d.array_length = 3;
        d.mipmap_level_count = 3;
        d.usage = TextureUsage::SHADER_READ | TextureUsage::PIXEL_FORMAT_VIEW;
        let array = device.new_texture(&d).unwrap();
        assert_eq!((array.array_length(), array.mipmap_level_count()), (3, 3));
        assert_eq!(array.storage_mode(), StorageMode::Shared);
        for level in 0..3u64 {
            let side = 8 >> level;
            for slice in 0..3u64 {
                let src = pattern((side * side * 4) as usize, (level * 16 + slice) as u8);
                array
                    .replace_region(
                        Region::new_2d(0, 0, side, side),
                        level,
                        slice,
                        &src,
                        side * 4,
                        0,
                    )
                    .unwrap();
            }
        }
        let read = |t: &Texture, level: u64, slice: u64, side: u64| {
            let mut b = vec![0u8; (side * side * 4) as usize];
            t.get_bytes(
                &mut b,
                side * 4,
                0,
                Region::new_2d(0, 0, side, side),
                level,
                slice,
            )
            .unwrap();
            b
        };
        for level in 0..3u64 {
            for slice in 0..3u64 {
                let side = 8 >> level;
                let want = pattern((side * side * 4) as usize, (level * 16 + slice) as u8);
                assert_eq!(read(&array, level, slice, side), want, "{level}/{slice}");
            }
        }

        // A view of levels 1..3 and slices 1..3 as 2D array, then a 2D view of its level 1 and
        // slice 1 (the texture's level 2, slice 2), in another format of the same bytes.
        let view = array
            .new_texture_view_with_levels_slices(rgba, TextureType::Type2DArray, 1..3, 1..3)
            .unwrap();
        assert_eq!(
            (view.width(), view.mipmap_level_count(), view.array_length()),
            (4, 2, 2)
        );
        assert_eq!(
            (view.parent_relative_level(), view.parent_relative_slice()),
            (1, 1)
        );
        let inner = view
            .new_texture_view_with_levels_slices(
                PixelFormat::R32Uint,
                TextureType::Type2D,
                1..2,
                1..2,
            )
            .unwrap();
        assert_eq!(inner.parent_texture(), Some(&view));
        assert_eq!(
            (inner.width(), inner.pixel_format()),
            (2, PixelFormat::R32Uint)
        );
        assert_eq!(
            read(&inner, 0, 0, 2),
            pattern(16, 2 * 16 + 2),
            "the view's level 0 and slice 0 are the texture's level 2 and slice 2"
        );
        let srgb = array.new_texture_view(PixelFormat::RGBA8UnormSRGB).unwrap();
        assert_eq!(read(&srgb, 1, 2, 4), pattern(64, 16 + 2));

        // Writing through a view writes the texture.
        let src = pattern(16, 0x55);
        inner
            .replace_region(Region::new_2d(0, 0, 2, 2), 0, 0, &src, 8, 0)
            .unwrap();
        assert_eq!(read(&array, 2, 2, 2), src);

        // A 3D texture, with images apart by more than their rows.
        let mut d = TextureDescriptor::texture_2d(rgba, 4, 4, false);
        d.texture_type = TextureType::Type3D;
        d.depth = 3;
        let volume = device.new_texture(&d).unwrap();
        let (bpr, bpi) = (16u64, 80u64);
        let src = pattern((bpi * 3) as usize, 9);
        let region = Region::new_3d(0, 0, 0, 4, 4, 3);
        volume.replace_region(region, 0, 0, &src, bpr, bpi).unwrap();
        let mut back = vec![0u8; (bpi * 3) as usize];
        volume.get_bytes(&mut back, bpr, bpi, region, 0, 0).unwrap();
        for z in 0..3 {
            let image = (z * bpi) as usize..(z * bpi + 64) as usize;
            assert_eq!(back[image.clone()], src[image], "image {z}");
        }
        // A `bytesPerImage` off Metal's rules, as plain Metal takes it:
        // `get_bytes` writes images exactly that far apart, in turn, a later image's bytes over an
        // earlier one's; a blit reads and writes each row at its offset rounded down to 4 bytes,
        // out of the texture in turn too.
        let images: Vec<&[u8]> = (0..3).map(|z| &src[z * 80..z * 80 + 64]).collect();
        let in_turn = |bpi: usize, len: usize| {
            let mut want = vec![0xabu8; len];
            for (z, image) in images.iter().enumerate() {
                let at = z * bpi - z * bpi % 4;
                want[at..at + 64].copy_from_slice(image);
            }
            want
        };
        let mut close = vec![0xabu8; 136];
        volume.get_bytes(&mut close, bpr, 36, region, 0, 0).unwrap();
        assert_eq!(close, in_turn(36, 136), "get_bytes at 36");
        let queue = device.new_command_queue().unwrap();
        let size = Size::new(4, 4, 3);
        for bpi in [0u64, 36, 70] {
            let len = (2 * bpi + 64) as usize;
            let out = buffer_with(&device, &vec![0xab; len]).unwrap();
            let command_buffer = queue.command_buffer().unwrap();
            let mut e = command_buffer.blit_command_encoder().unwrap();
            e.copy_from_texture_to_buffer(
                &volume,
                0,
                0,
                Origin::new(0, 0, 0),
                size,
                &out,
                0,
                bpr,
                bpi,
                BlitOption::None,
            )
            .unwrap();
            e.end_encoding().unwrap();
            command_buffer.commit().unwrap();
            command_buffer.wait_until_completed_for(WAIT).unwrap();
            assert_eq!(command_buffer.error(), None);
            let written = read_buffer(&out, 0, len as u64).unwrap();
            assert_eq!(written, in_turn(bpi as usize, len), "a blit out at {bpi}");
        }

        // A texture of more than half a memory block, placed in a block of its own.
        let big = texture_2d(&device, rgba, 4096, 1).unwrap();
        let src = pattern(4 * 4096, 3);
        let last_row = Region::new_2d(0, 4095, 4096, 1);
        big.replace_region(last_row, 0, 0, &src, 4 * 4096, 0)
            .unwrap();
        let mut back = vec![0u8; 4 * 4096];
        big.get_bytes(&mut back, 4 * 4096, 0, last_row, 0, 0)
            .unwrap();
        assert_eq!(back, src);
        drop(big);

        // Depth-only and stencil-only formats copy their texels (depth in [0, 1], which every
        // device keeps); a combined depth/stencil format is refused CPU access, as Metal does.
        for (pf, bytes) in [
            (PixelFormat::Depth16Unorm, pattern(8, 1)),
            (
                PixelFormat::Depth32Float,
                [0.0f32, 0.25, 0.5, 1.0]
                    .iter()
                    .flat_map(|v| v.to_ne_bytes())
                    .collect(),
            ),
            (PixelFormat::Stencil8, pattern(4, 2)),
        ] {
            let texture = match texture_2d(&device, pf, 2, 1) {
                Ok(texture) => texture,
                Err(Error::Unsupported(u)) => {
                    eprintln!("{pf:?} refused: {u:?}");
                    continue;
                }
                Err(other) => panic!("{pf:?}: {other}"),
            };
            let bpr = bytes.len() as u64 / 2;
            let region = Region::new_2d(0, 0, 2, 2);
            let mut fresh = vec![0xeeu8; bytes.len()];
            texture.get_bytes(&mut fresh, bpr, 0, region, 0, 0).unwrap();
            assert!(fresh.iter().all(|&b| b == 0), "{pf:?} reads zero when new");
            texture
                .replace_region(region, 0, 0, &bytes, bpr, 0)
                .unwrap();
            let mut back = vec![0u8; bytes.len()];
            texture.get_bytes(&mut back, bpr, 0, region, 0, 0).unwrap();
            assert_eq!(back, bytes, "{pf:?}");
        }
        // A combined depth/stencil format copies its depth, 4 bytes a texel, at any pitch, the
        // stencil left as it is, as plain Metal does.
        if let Ok(combined) = texture_2d(&device, PixelFormat::Depth32FloatStencil8, 2, 1) {
            let queue = device.new_command_queue().unwrap();
            let blit = |buffer: &Buffer, into: bool, option: BlitOption, bpr: u64| {
                let command_buffer = queue.command_buffer().unwrap();
                let mut e = command_buffer.blit_command_encoder().unwrap();
                let (origin, size) = (Origin::new(0, 0, 0), Size::new(2, 2, 1));
                if into {
                    e.copy_from_buffer_to_texture(
                        buffer, 0, bpr, 0, size, &combined, 0, 0, origin, option,
                    )
                } else {
                    e.copy_from_texture_to_buffer(
                        &combined, 0, 0, origin, size, buffer, 0, bpr, 0, option,
                    )
                }
                .unwrap();
                e.end_encoding().unwrap();
                command_buffer.commit().unwrap();
                command_buffer.wait_until_completed_for(WAIT).unwrap();
                assert_eq!(command_buffer.error(), None);
            };
            let stencil = buffer_with(&device, &[1, 2, 3, 4]).unwrap();
            blit(&stencil, true, BlitOption::StencilFromDepthStencil, 2);
            let floats = |v: [f32; 2]| v.iter().flat_map(|f| f.to_ne_bytes()).collect::<Vec<u8>>();
            let depth = [
                floats([0.125, 0.375]),
                vec![0x99; 2],
                floats([0.625, 0.875]),
            ]
            .concat();
            let region = Region::new_2d(0, 0, 2, 2);
            combined
                .replace_region(region, 0, 0, &depth, 10, 0)
                .unwrap();
            let mut back = vec![0xabu8; 20];
            combined.get_bytes(&mut back, 10, 0, region, 0, 0).unwrap();
            let want = [&depth[..8], &[0xab; 2], &depth[10..], &[0xab; 2]].concat();
            assert_eq!(back, want, "depth through the CPU");
            let read = buffer_with(&device, &[0; 4]).unwrap();
            blit(&read, false, BlitOption::StencilFromDepthStencil, 2);
            assert_eq!(
                read_buffer(&read, 0, 4).unwrap(),
                [1, 2, 3, 4],
                "stencil kept"
            );
        }

        // Metal's refusals.
        let mut d = TextureDescriptor::texture_2d(rgba, 4, 4, false);
        d.resource_options = ResourceOptions::private();
        let private = device.new_texture(&d).unwrap();
        assert_eq!(private.storage_mode(), StorageMode::Private);
        assert_eq!(
            private.replace_region(Region::new_2d(0, 0, 1, 1), 0, 0, &[0; 4], 4, 0),
            Err(invalid(
                "replace_region",
                Problem::Copy(CopyProblem::CpuAccessPrivate)
            ))
        );
        let small = texture_2d(&device, rgba, 4, 1).unwrap();
        assert_eq!(
            small.replace_region(Region::new_2d(2, 0, 4, 1), 0, 0, &[0; 16], 16, 0),
            Err(invalid(
                "replace_region",
                Problem::Copy(CopyProblem::Region)
            ))
        );
        // A pitch between 0 and a row copies nothing, as plain Metal copies nothing: the new
        // texture still reads zero.
        assert_eq!(
            small.replace_region(Region::new_2d(0, 0, 4, 2), 0, 0, &[0xab; 32], 8, 0),
            Ok(())
        );
        let mut kept = vec![0xeeu8; 64];
        small
            .get_bytes(&mut kept, 16, 0, Region::new_2d(0, 0, 4, 4), 0, 0)
            .unwrap();
        assert!(
            kept.iter().all(|&b| b == 0),
            "a short pitch wrote: {kept:?}"
        );
        // An empty region writes nothing, as plain Metal does.
        assert_eq!(
            small.replace_region(Region::new_2d(0, 0, 0, 2), 0, 0, &[0; 32], 16, 0),
            Ok(())
        );
        assert_eq!(
            small.replace_region(Region::new_2d(0, 0, 4, 2), 0, 0, &[0; 20], 16, 0),
            Err(invalid(
                "replace_region",
                Problem::Copy(CopyProblem::BufferRange)
            ))
        );
        assert_eq!(
            device
                .new_texture(&TextureDescriptor::texture_2d(rgba, 0, 4, false))
                .err(),
            Some(invalid(
                "new_texture",
                Problem::Texture(TextureProblem::ZeroExtent)
            ))
        );
        // A view of another format of the class without `PixelFormatView` usage, which plain
        // Metal makes (its validation stops it): the CPU reads the texture's bytes through it.
        let words = pattern(16, 9);
        small
            .replace_region(Region::new_2d(0, 0, 4, 1), 0, 0, &words, 16, 0)
            .unwrap();
        let view = small.new_texture_view(PixelFormat::R32Uint).unwrap();
        assert_eq!(view.pixel_format(), PixelFormat::R32Uint);
        let mut through = vec![0u8; 16];
        view.get_bytes(&mut through, 16, 0, Region::new_2d(0, 0, 4, 1), 0, 0)
            .unwrap();
        assert_eq!(through, words);
        assert_eq!(
            array
                .new_texture_view_with_levels_slices(rgba, TextureType::Type2D, 0..1, 0..2)
                .err(),
            Some(invalid(
                "new_texture_view_with_levels_slices",
                Problem::TextureView(ViewProblem::SliceCount)
            ))
        );
    }
    layer_reported_nothing_at_all(&messages);
}

/// A view of another component layout presents its texture's bits, as Metal's does: a 4x
/// `RGBA8Unorm` texture filled sample by sample, viewed as `R32Uint` and read sample by sample in
/// a kernel, gives each texel's four bytes as one word. A device whose Vulkan views reinterpret no
/// image's components (a portability subset without `imageViewFormatReinterpretation`) refuses
/// the view, typed; MoltenVK's subset has it, and the runtime enables it.
#[test]
fn a_view_of_another_component_layout_reads_the_textures_bits() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    let (extent, samples) = ([3u32, 2u32], 4u32);
    // Each pixel's samples in order, rows in order: four bytes a texel.
    let bytes = pattern((extent[0] * extent[1] * samples * 4) as usize, 7);
    let answer;
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let mut d = TextureDescriptor::texture_2d(
            PixelFormat::RGBA8Unorm,
            extent[0].into(),
            extent[1].into(),
            false,
        );
        d.texture_type = TextureType::Type2DMultisample;
        d.sample_count = samples.into();
        d.usage = TextureUsage::RENDER_TARGET
            | TextureUsage::SHADER_READ
            | TextureUsage::PIXEL_FORMAT_VIEW;
        d.resource_options = ResourceOptions::private();
        let texture = device.new_texture(&d).unwrap();
        answer = match texture.new_texture_view(PixelFormat::R32Uint) {
            Err(Error::Unsupported(u)) if u.feature == Feature::ViewFormatReinterpretation => {
                "refused, typed".to_owned()
            }
            Err(error) => panic!("new_texture_view: {error}"),
            Ok(view) => {
                let layout =
                    texture_layout(CaseType::D2Multisample, [extent[0], extent[1], 1], samples)
                        .unwrap();
                let fill =
                    SampleFill::new(&device, &texture, CaseFormat::Rgba8Unorm, layout, &bytes)
                        .unwrap();
                let queue = device.new_command_queue().unwrap();
                let command_buffer = queue.command_buffer().unwrap();
                fill.encode(&command_buffer).unwrap();
                let readback = encode_through(
                    &device,
                    &command_buffer,
                    &view,
                    "rt6_read_words1",
                    SampleRead::Color(ReadbackView::R32Uint),
                    extent,
                    samples,
                )
                .unwrap();
                command_buffer.commit().unwrap();
                command_buffer.wait_until_completed_for(WAIT).unwrap();
                assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
                assert_eq!(
                    readback.bytes().unwrap(),
                    bytes,
                    "the R32Uint view's words are not the texture's bytes"
                );
                "the texture's bits".to_owned()
            }
        };
        eprintln!(
            "view reinterpretation on {}: RGBA8Unorm as R32Uint, {answer}",
            device.name()
        );
    }
    layer_reported_nothing_at_all(&messages);
}

/// Labels and purgeable states of textures as Metal answered them (an Apple M3 Max and an M4): each
/// texture object's label its own, an opened shared texture's starting as its texture's; the state
/// kept by a texture of 16 KiB or more and by a shareable texture (shared with the texture opened
/// from its handle), `Empty` keeping the texels; `NonVolatile` to every request from a smaller
/// texture, a view, and a texture over a buffer, whose buffer keeps its own state, and from a
/// heap's texture, whose heap keeps its own.
#[test]
fn texture_labels_and_purgeable_states_answer_as_metal() {
    use metal2vulkan_runtime::PurgeableState::{Empty, KeepCurrent, NonVolatile, Volatile};
    use metal2vulkan_runtime::{HeapDescriptor, HeapType, IdPlacement, PurgeableState};
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let rgba = PixelFormat::RGBA8Unorm;
        let sequence = |texture: &Texture| -> Vec<PurgeableState> {
            [
                KeepCurrent,
                Volatile,
                KeepCurrent,
                Empty,
                KeepCurrent,
                NonVolatile,
                KeepCurrent,
            ]
            .into_iter()
            .map(|state| texture.set_purgeable_state(state).unwrap())
            .collect()
        };
        let kept = [
            NonVolatile,
            NonVolatile,
            Volatile,
            Volatile,
            Empty,
            Empty,
            NonVolatile,
        ];
        let never = [NonVolatile; 7];

        // A 64x64 texture (16 KiB) keeps its state, and `Empty` keeps its texels.
        let big = device
            .new_texture(&TextureDescriptor::texture_2d(rgba, 64, 64, false))
            .unwrap();
        assert_eq!(big.label(), None);
        big.set_label("big");
        assert_eq!(big.label().as_deref(), Some("big"));
        let texels = pattern(64 * 64 * 4, 7);
        let whole = Region::new_2d(0, 0, 64, 64);
        big.replace_region(whole, 0, 0, &texels, 256, 0).unwrap();
        assert_eq!(sequence(&big), kept);
        let mut read = vec![0; texels.len()];
        big.get_bytes(&mut read, 256, 0, whole, 0, 0).unwrap();
        assert_eq!(read, texels);

        // A 32x32 texture (4 KiB) answers `NonVolatile` to every request.
        let small = device
            .new_texture(&TextureDescriptor::texture_2d(rgba, 32, 32, false))
            .unwrap();
        assert_eq!(sequence(&small), never);

        // A view's label and state are its own: `NonVolatile` while its texture is volatile.
        let view = big.new_texture_view(rgba).unwrap();
        assert_eq!(view.label(), None);
        view.set_label("view");
        assert_eq!(big.label().as_deref(), Some("big"));
        assert_eq!(big.set_purgeable_state(Volatile), Ok(NonVolatile));
        assert_eq!(sequence(&view), never);
        assert_eq!(big.set_purgeable_state(NonVolatile), Ok(Volatile));

        // A texture over a 64-KiB buffer answers `NonVolatile`; the buffer keeps its own state.
        let buffer = device.new_buffer(65536, ResourceOptions::shared()).unwrap();
        let mut over = TextureDescriptor::texture_2d(rgba, 128, 128, false);
        over.usage = TextureUsage::SHADER_READ;
        let over = buffer.new_texture(&over, 0, 512).unwrap();
        assert_eq!(buffer.set_purgeable_state(Volatile), Ok(NonVolatile));
        assert_eq!(sequence(&over), never);
        assert_eq!(buffer.set_purgeable_state(NonVolatile), Ok(Volatile));

        // A 16x16 shareable texture keeps its state, one with the texture opened from its
        // handle, which starts with its label; their views answer `NonVolatile`.
        let mut private = TextureDescriptor::texture_2d(rgba, 16, 16, false);
        private.resource_options = ResourceOptions::private();
        let shared = device.new_shared_texture(&private).unwrap();
        shared.set_label("original");
        let handle = shared.new_shared_texture_handle().unwrap();
        let opened = device
            .new_shared_texture_with_handle(&handle, IdPlacement::Any)
            .unwrap();
        assert_eq!(opened.label().as_deref(), Some("original"));
        shared.set_label("relabelled");
        assert_eq!(opened.label().as_deref(), Some("original"));
        assert_eq!(opened.set_purgeable_state(Volatile), Ok(NonVolatile));
        assert_eq!(shared.set_purgeable_state(KeepCurrent), Ok(Volatile));
        let shared_view = shared.new_texture_view(rgba).unwrap();
        assert_eq!(
            shared_view.set_purgeable_state(KeepCurrent),
            Ok(NonVolatile)
        );
        assert_eq!(shared.set_purgeable_state(NonVolatile), Ok(Volatile));
        assert_eq!(opened.set_purgeable_state(KeepCurrent), Ok(NonVolatile));

        // A heap's texture answers `NonVolatile` to every request and changes nothing, its
        // heap's state included, as plain Metal answers where its validation stops it; its label is
        // its own.
        let heap = device
            .new_heap(&HeapDescriptor::new(
                HeapType::Automatic,
                1 << 20,
                ResourceOptions::private(),
            ))
            .unwrap();
        let in_heap = heap.new_texture(&private).unwrap();
        for state in [KeepCurrent, Volatile, Empty, NonVolatile] {
            assert_eq!(in_heap.set_purgeable_state(state), Ok(NonVolatile));
            assert_eq!(heap.set_purgeable_state(KeepCurrent), Ok(NonVolatile));
        }
        in_heap.set_label("in a heap");
        assert_eq!(in_heap.label().as_deref(), Some("in a heap"));
    }
    layer_reported_nothing_at_all(&messages);
}
