//! What a new resource reads before anything writes it, on the device the cfg selects, under the
//! validation layer.
//!
//! **Metal's answer** (an Apple M4 under macOS 27.0.1, plain, our own probe): each
//! case made first, six times right after a resource of its shape was written and released, and
//! six times right after a buffer of its allocated size was, in three fresh processes and one
//! after 768 MiB of written and released memory, 52 reads a case:
//!
//! - **Zero, every read:** buffers of every storage mode (16 B to 128 MiB, also where the new
//!   buffer took the released one's GPU address); `Shared` and `Managed` textures of every type
//!   and format tried; `Private` textures Metal keeps uncompressed (`ShaderWrite` usage, GPU
//!   optimization off, a block-compressed format, `RGB10A8_2P_XR10` that shaders write, both
//!   planes); a texture over a new buffer; a new heap's bytes.
//! - **A fixed pattern, every read:** `Private` textures Metal compresses (`RGBA8Unorm` reads
//!   `ff00ffff` in every texel; depth reads all ones), in each level whose shorter side is 16 or
//!   more. The runtime gives that pattern, and zero where the device cannot write it exactly,
//!   announcing `fresh-pattern-zeroed` where a command first reads that
//!   content unwritten, never as the texture is made: each case expects the pattern, or zero where
//!   the device announced the kind at its read.
//!
//! Each case here is made right after one of its shape was written and released, so a runtime
//! that handed the old memory out as it lies would read the old bytes:
//!
//! - **Buffers** `Shared`, `Private` and `Managed` of 16 B to 16 MiB, read through a blit.
//! - **Textures** `Shared`, `Managed` and `Private`: a 2D, a mipmapped, a 3D, a 2D-array and a
//!   cube `RGBA8Unorm` render target, an `R32Float` that shaders write, an `RGBA16Float`, a
//!   `BC1_RGBA` where the device has it, a `Private` `RGBA8Unorm` without GPU optimization and a
//!   `Private` `Depth32Float`; every level and slice read through a blit. The `Private` ones
//!   Metal compresses read its pattern (the 2D, mipmapped, 3D, array and cube `RGBA8Unorm`, the
//!   `RGBA16Float`, the `Depth32Float`).
//! - **Each format's pattern**: a new `Private` 64² texture of each format whose pattern Metal's
//!   probe read, read by a blit; and mipmapped chains, level by level.
//! - **Multisampled textures**: `Private` 4x `RGBA8Unorm` targets of 4² (Metal keeps it
//!   uncompressed: zero) and 8² (its sample grid 16²: the pattern), and a 4x `RGBA16Float` of 8²
//!   (a NaN pattern no clear writes exactly: copied from an unsigned-integer scratch, or zero,
//!   announced where loaded, where the device makes no such scratch), loaded and resolved;
//!   a `Private` 4x `Depth32Float` (a NaN depth no clear writes: zero) cleared then read by a
//!   shader (its clear, nothing announced), loaded never written (announced as the pass begins),
//!   and read by a shader never written (announced at the read).
//! - **The two-plane `RGB10A8_2P_XR10`**, `Private`, that shaders write: both planes read by the
//!   owned fixture `xr_two_planes`'s kernels, the same as a new `Shared` one reads (zero bytes).
//! - **Textures over new buffers**, `Shared` and `Private`.
//!
//! A memoryless attachment loaded `DontCare` reads zero: that case is
//! `framebuffer_fetch::a_dont_care_memoryless_attachment_reads_zero`, where the fetches are.
//!
//! **On a device without Metal's compression** (`DeviceProperties::lossless_compression` false: a
//! device that presents a Mac whose GPU keeps no texture compressed, as an Intel or AMD GPU's
//! does), a new `Private` texture holds what its memory held: zero, or a released texture's texels
//! (the consumer's probe of new `Private` textures on an AMD Radeon Pro 5300M and an Intel UHD 630:
//! zero, and a released colour texture's texels exactly where one of the same descriptor was
//! released first). The runtime writes nothing into it (`WorkCounts::private_textures_unfilled`).
//! Each case there holds its `Private` textures to that: the count moved by one a texture, nothing
//! announced `fresh-pattern-zeroed`, and the texels, which nothing wrote, are not Metal's whole
//! pattern; it prints what they held (`FRESH-UNFILLED`). Its `Shared` and `Managed` textures and
//! its buffers read zero, as on every device. On a device with Metal's compression the count stays
//! zero.

use super::degrade_notices::counts;
use super::extended_range_two_planes::{reads, write, Dispatch, Kernels};
use super::read_buffer;
use super::resolves::{lane, lane_with, quiet, Lane, Shape as LaneShape, ONE};
use super::Messages;
use crate::candidate::runtime_samples::encode;
use crate::executor_contract::multisample::{ReadbackView, SampleRead};
use metal2vulkan::metal::{PixelFormat, TextureType, TextureUsage};
use metal2vulkan_runtime::{
    BlitCommandEncoder, BlitOption, Buffer, ClearColor, CpuCacheMode, Degradation, Device, Error,
    HazardTrackingMode, LoadAction, MacClass, Origin, RenderPassDescriptor, ResourceOptions, Size,
    StorageMode, StoreAction, Texture, TextureDescriptor,
};

/// The byte the released resources are written with.
const OLD: u8 = 0xa5;

/// The kind a new texture announces where the device cannot write Metal's pattern exactly: the
/// lane site `degrade_notices` lists.
pub(super) const FRESH_PATTERNS: [Degradation; 1] = [Degradation::FreshPatternZeroed];

/// Metal's pattern of `RGBA8Unorm`: opaque magenta.
const MAGENTA: &[u8] = &[0xff, 0x00, 0xff, 0xff];

/// A lane device announcing `fresh-pattern-zeroed` at each texture it is for.
fn pattern_lane() -> (Messages, super::ScratchCache, Lane) {
    lane_with(|options| options.announce_each = FRESH_PATTERNS.into_iter().collect())
}

/// Whether the device announced `fresh-pattern-zeroed` since the notices were last taken.
pub(super) fn zeroed(messages: &Messages) -> bool {
    counts(&messages.take_notices()).contains_key(&Degradation::FreshPatternZeroed)
}

/// Whether `device` is without Metal's compression, where a new `Private` texture holds what
/// its memory held, written by nothing (the module docs). Every lane device presents its GPU's
/// Mac class (`DeviceOptions::mac_class` unset), which keeps no texture compressed on a GPU that
/// is not Apple's alone, so the device writes nothing there; an Apple GPU presenting a Mac with an
/// Intel or AMD GPU instead fills it with zeros, which the runtime's own `mac_class` tests hold.
pub(super) fn unfilled(device: &Device) -> bool {
    !device.properties().lossless_compression
}

/// The kinds a read of a new `Private` texture's unwritten pattern announces on `device` where
/// the device cannot write it: [`FRESH_PATTERNS`], or none on a device without Metal's
/// compression, whose new textures hold what their memory held, which nothing announces.
pub(super) fn patterns_of(device: &Device) -> &'static [Degradation] {
    match unfilled(device) {
        true => &[],
        false => &FRESH_PATTERNS,
    }
}

/// The new `Private` textures `device` made holding what their memory held, so far.
pub(super) fn unfilled_count(device: &Device) -> u64 {
    device.work_counts().private_textures_unfilled
}

/// What `read`, a new `Private` texture's texels that nothing wrote, held: zero, the released
/// texture's `old` texel in every texel, or other bytes.
fn held(read: &[u8], old: &[u8]) -> &'static str {
    if read.iter().all(|&byte| byte == 0) {
        "zero"
    } else if !old.is_empty() && read.chunks(old.len()).all(|texel| texel == old) {
        "the released texture's texels"
    } else {
        "other bytes"
    }
}

/// The three storage modes a resource of its own memory takes.
fn storages() -> [(&'static str, ResourceOptions); 3] {
    let managed = ResourceOptions::new(
        StorageMode::Managed,
        CpuCacheMode::DefaultCache,
        HazardTrackingMode::Default,
    );
    [
        ("Shared", ResourceOptions::shared()),
        ("Managed", managed),
        ("Private", ResourceOptions::private()),
    ]
}

/// Commit `encode`'s blits on the lane's queue and wait.
fn blits(lane: &Lane, encode: impl FnOnce(&mut BlitCommandEncoder)) {
    let command_buffer = lane.queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    encode(&mut blit);
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer
        .wait_until_completed_for(super::WAIT)
        .unwrap();
    assert_eq!(command_buffer.error(), None);
}

/// `buffer`'s bytes, through a blit into a new `Shared` buffer.
fn bytes_of(lane: &Lane, buffer: &Buffer) -> Vec<u8> {
    let length = buffer.length();
    let out = lane
        .device
        .new_buffer(length, ResourceOptions::shared())
        .unwrap();
    blits(lane, |blit| {
        blit.copy_from_buffer_to_buffer(buffer, 0, &out, 0, length)
            .unwrap()
    });
    read_buffer(&out, 0, length).unwrap()
}

/// Where `bytes` holds a byte other than zero: its offset and the 16 bytes from there.
fn non_zero(bytes: &[u8]) -> Option<(usize, Vec<u8>)> {
    let at = bytes.iter().position(|&b| b != 0)?;
    Some((at, bytes[at..bytes.len().min(at + 16)].to_vec()))
}

/// New buffers of every storage mode read zero, each made right after one of its length was
/// filled with [`OLD`] and released (the module docs).
#[test]
fn new_buffers_read_zero_after_written_ones_are_released() {
    let (messages, cache, lane) = lane();
    let name = lane.device.name();
    for (storage, options) in storages() {
        for length in [16u64, 4096, 65536, 1 << 20, 16 << 20] {
            let old = lane.device.new_buffer(length, options).unwrap();
            blits(&lane, |blit| {
                blit.fill_buffer(&old, 0..length, OLD).unwrap()
            });
            drop(old);
            let new = lane.device.new_buffer(length, options).unwrap();
            if new.contents().is_some() {
                let host = read_buffer(&new, 0, length).unwrap();
                assert_eq!(
                    non_zero(&host),
                    None,
                    "a new {storage} buffer of {length} bytes read by the host on {name}"
                );
            }
            assert_eq!(
                non_zero(&bytes_of(&lane, &new)),
                None,
                "a new {storage} buffer of {length} bytes on {name}"
            );
        }
    }
    drop(cache);
    quiet(lane, &messages);
}

/// A texture case: its name, descriptor, and bytes per block of `block`² texels; and where Metal
/// reads its pattern, the texel and the levels it reads it in.
struct Shape {
    name: String,
    descriptor: TextureDescriptor,
    bytes: u64,
    block: u64,
    pattern: Option<(&'static [u8], u64)>,
}

impl Shape {
    /// A 2D texture of `pf` of `side`², `usage`, `options`, `bytes` per texel.
    fn new(
        name: &str,
        pf: PixelFormat,
        (side, bytes): (u64, u64),
        usage: TextureUsage,
        (storage, options): (&str, ResourceOptions),
    ) -> Shape {
        let mut descriptor = TextureDescriptor::texture_2d(pf, side, side, false);
        descriptor.usage = usage;
        descriptor.resource_options = options;
        Shape {
            name: format!("a new {storage} {name}"),
            descriptor,
            bytes,
            block: 1,
            pattern: None,
        }
    }

    /// The case, which Metal reads `texel` in, in its first `levels` levels where it is
    /// `Private`.
    fn patterned(mut self, texel: &'static [u8], levels: u64) -> Shape {
        if matches!(
            self.descriptor.resource_options.storage_mode(),
            Ok(StorageMode::Private)
        ) {
            self.pattern = Some((texel, levels));
        }
        self
    }

    /// What every level and slice reads, as [`texels`] reads them: the pattern in its levels,
    /// zero elsewhere, or zero throughout where `zeroed`.
    fn expected(&self, zeroed: bool) -> Vec<u8> {
        let mut all = Vec::new();
        for (level, _) in self.subresources() {
            let (size, _, image) = self.level(level);
            let length = (image * size.depth) as usize;
            match self.pattern {
                Some((texel, levels)) if level < levels && !zeroed => {
                    all.extend(texel.iter().copied().cycle().take(length))
                }
                _ => all.extend(std::iter::repeat_n(0, length)),
            }
        }
        all
    }

    /// Whether it is `Private`.
    fn private(&self) -> bool {
        matches!(
            self.descriptor.resource_options.storage_mode(),
            Ok(StorageMode::Private)
        )
    }

    /// Its slices: six faces a cube, else its array length.
    fn slices(&self) -> u64 {
        match self.descriptor.texture_type {
            TextureType::Cube => 6,
            _ => self.descriptor.array_length,
        }
    }

    /// Level `level`'s extent in texels, and its rows and images in bytes.
    fn level(&self, level: u64) -> (Size, u64, u64) {
        let d = &self.descriptor;
        let [w, h] = [d.width, d.height].map(|side| (side >> level).max(1));
        let depth = match d.texture_type {
            TextureType::Type3D => (d.depth >> level).max(1),
            _ => 1,
        };
        let row = w.div_ceil(self.block) * self.bytes;
        let image = row * h.div_ceil(self.block);
        (Size::new(w, h, depth), row, image)
    }

    /// Every level's and slice's `(level, slice)`.
    fn subresources(&self) -> impl Iterator<Item = (u64, u64)> + '_ {
        (0..self.descriptor.mipmap_level_count)
            .flat_map(move |level| (0..self.slices()).map(move |slice| (level, slice)))
    }
}

/// Every texel's bytes of `texture` of `shape`, level by level and slice by slice, through blits.
fn texels(lane: &Lane, shape: &Shape, texture: &Texture) -> Vec<u8> {
    let mut all = Vec::new();
    for (level, slice) in shape.subresources() {
        let (size, row, image) = shape.level(level);
        let length = image * size.depth;
        let out = lane
            .device
            .new_buffer(length, ResourceOptions::shared())
            .unwrap();
        blits(lane, |blit| {
            blit.copy_from_texture_to_buffer(
                texture,
                slice,
                level,
                Origin::new(0, 0, 0),
                size,
                &out,
                0,
                row,
                image,
                BlitOption::None,
            )
            .unwrap()
        });
        all.extend(read_buffer(&out, 0, length).unwrap());
    }
    all
}

/// Write `texel` (one block's bytes) into every texel of `texture` of `shape`, through blits.
fn write_all(lane: &Lane, shape: &Shape, texture: &Texture, texel: &[u8]) {
    for (level, slice) in shape.subresources() {
        let (size, row, image) = shape.level(level);
        let bytes = texel.repeat((image * size.depth / shape.bytes) as usize);
        let source = super::buffer_with(&lane.device, &bytes).unwrap();
        blits(lane, |blit| {
            blit.copy_from_buffer_to_texture(
                &source,
                0,
                row,
                image,
                size,
                texture,
                slice,
                level,
                Origin::new(0, 0, 0),
                BlitOption::None,
            )
            .unwrap()
        });
    }
}

/// The lane's texture cases (the module docs): those a device may lack are made by the test,
/// which skips them where it refuses them.
fn shapes() -> Vec<(Shape, Vec<u8>)> {
    let rt = TextureUsage::SHADER_READ | TextureUsage::RENDER_TARGET;
    let rgba8 = (64, 4);
    let mut shapes = Vec::new();
    for storage in storages() {
        let old = vec![OLD; 4];
        shapes.push((
            Shape::new("2D RGBA8Unorm", PixelFormat::RGBA8Unorm, rgba8, rt, storage)
                .patterned(MAGENTA, 1),
            old.clone(),
        ));
        let mut mips = Shape::new(
            "mipmapped RGBA8Unorm",
            PixelFormat::RGBA8Unorm,
            rgba8,
            rt,
            storage,
        );
        mips.descriptor.mipmap_level_count = 7;
        // Levels 64, 32 and 16.
        shapes.push((mips.patterned(MAGENTA, 3), old.clone()));
        let mut volume = Shape::new(
            "3D RGBA8Unorm",
            PixelFormat::RGBA8Unorm,
            (16, 4),
            rt,
            storage,
        );
        volume.descriptor.texture_type = TextureType::Type3D;
        volume.descriptor.depth = 16;
        shapes.push((volume.patterned(MAGENTA, 1), old.clone()));
        let mut array = Shape::new(
            "2D-array RGBA8Unorm",
            PixelFormat::RGBA8Unorm,
            (32, 4),
            rt,
            storage,
        );
        array.descriptor.texture_type = TextureType::Type2DArray;
        array.descriptor.array_length = 3;
        shapes.push((array.patterned(MAGENTA, 1), old.clone()));
        let mut cube = Shape::new(
            "cube RGBA8Unorm",
            PixelFormat::RGBA8Unorm,
            (32, 4),
            rt,
            storage,
        );
        cube.descriptor.texture_type = TextureType::Cube;
        shapes.push((cube.patterned(MAGENTA, 1), old.clone()));
        let written = TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE;
        shapes.push((
            Shape::new(
                "written R32Float",
                PixelFormat::R32Float,
                (64, 4),
                written,
                storage,
            ),
            old.clone(),
        ));
        shapes.push((
            Shape::new(
                "RGBA16Float",
                PixelFormat::RGBA16Float,
                (64, 8),
                rt,
                storage,
            )
            .patterned(&[0xff, 0xff, 0, 0, 0xff, 0xff, 0xff, 0xff], 1),
            vec![OLD; 8],
        ));
        let mut bc1 = Shape::new(
            "BC1_RGBA",
            PixelFormat::BC1RGBA,
            (64, 8),
            TextureUsage::SHADER_READ,
            storage,
        );
        bc1.block = 4;
        shapes.push((bc1, vec![OLD; 8]));
    }
    let private = ("Private", ResourceOptions::private());
    let mut plain = Shape::new(
        "RGBA8Unorm without GPU optimization",
        PixelFormat::RGBA8Unorm,
        (64, 4),
        rt,
        private,
    );
    plain.descriptor.allow_gpu_optimized_contents = false;
    shapes.push((plain, vec![OLD; 4]));
    let depth = TextureUsage::SHADER_READ | TextureUsage::RENDER_TARGET;
    // A depth of 0.5: a value every depth image holds as it is.
    let half = 0.5f32.to_le_bytes().to_vec();
    shapes.push((
        Shape::new(
            "Depth32Float",
            PixelFormat::Depth32Float,
            (64, 4),
            depth,
            private,
        )
        .patterned(&[0xff; 4], 1),
        half,
    ));
    shapes
}

/// New textures of every storage mode read Metal's answer in every texel: zero, or the pattern of a
/// `Private` one Metal compresses, in its levels, zero where the device announced it cannot write
/// it; each made right after one of its shape was written with [`OLD`] bytes and released (the
/// module docs). A `Private` one on a device without Metal's compression holds what its
/// memory held, counted.
#[test]
fn new_textures_read_metal_s_answer_after_written_ones_are_released() {
    let (messages, cache, lane) = pattern_lane();
    let name = lane.device.name();
    let (mut made, mut patterned, mut announced, mut left) = (0, 0, 0, 0);
    for (shape, old) in shapes() {
        let first = match lane.device.new_texture(&shape.descriptor) {
            Ok(texture) => texture,
            Err(Error::Unsupported(why)) => {
                println!("FRESH-SKIP {} on {name}: {why:?}", shape.name);
                continue;
            }
            Err(error) => panic!("{} on {name}: {error:?}", shape.name),
        };
        write_all(&lane, &shape, &first, &old);
        assert!(
            texels(&lane, &shape, &first)
                .chunks(old.len())
                .all(|t| t == old),
            "{} on {name}: the written texture reads what was written",
            shape.name
        );
        drop(first);
        zeroed(&messages);
        let before = unfilled_count(&lane.device);
        let new = lane.device.new_texture(&shape.descriptor).unwrap();
        assert!(
            !zeroed(&messages),
            "{} on {name}: announced as made",
            shape.name
        );
        let unfilled = shape.private() && unfilled(&lane.device);
        assert_eq!(
            unfilled_count(&lane.device) - before,
            u64::from(unfilled),
            "{} on {name}: counted as made unfilled",
            shape.name
        );
        let read = texels(&lane, &shape, &new);
        let zeroed = zeroed(&messages);
        if unfilled {
            let what = held(&read, &old);
            let pattern = shape.pattern.is_some() && read == shape.expected(false);
            assert!(
                !zeroed && !pattern,
                "{} on {name}, without Metal's compression: announced {zeroed}, reads Metal's \
                 pattern {pattern}, which nothing wrote",
                shape.name
            );
            println!("FRESH-UNFILLED {} on {name}: {what}", shape.name);
            (made, left) = (made + 1, left + 1);
            continue;
        }
        assert!(
            !zeroed || shape.pattern.is_some(),
            "{} on {name}: announced fresh-pattern-zeroed, where Metal reads zero",
            shape.name
        );
        let expected = shape.expected(zeroed);
        assert!(
            read == expected,
            "{} on {name}, made after a written one was released: first difference at {:?}",
            shape.name,
            read.iter().zip(&expected).position(|(a, b)| a != b)
        );
        if zeroed {
            println!("FRESH-ZEROED {} on {name}", shape.name);
            announced += 1;
        } else if shape.pattern.is_some() {
            patterned += 1;
        }
        made += 1;
    }
    println!(
        "FRESH-TEXTURES {made} cases read Metal's answer on {name}: {patterned} its pattern, \
         {announced} zero announced, {left} Private ones holding what their memory held"
    );
    drop(cache);
    quiet(lane, &messages);
}

/// Metal's pattern of each format its probes read (a new `Private` 64² texture, every texel), and
/// the bytes of a texel.
const PATTERNS: &[(PixelFormat, &[u8])] = &[
    (PixelFormat::A8Unorm, &[0xff]),
    (PixelFormat::R8Unorm, &[0xff]),
    (PixelFormat::R8UnormSRGB, &[0xff]),
    (PixelFormat::R8Snorm, &[0xff]),
    (PixelFormat::R8Uint, &[0xff]),
    (PixelFormat::R8Sint, &[0xff]),
    (PixelFormat::RG8Unorm, &[0xff, 0x00]),
    (PixelFormat::RG8UnormSRGB, &[0xff, 0x00]),
    (PixelFormat::RG8Snorm, &[0xff, 0x00]),
    (PixelFormat::RG8Uint, &[0xff, 0x00]),
    (PixelFormat::RG8Sint, &[0xff, 0x00]),
    (PixelFormat::R16Unorm, &[0xff, 0xff]),
    (PixelFormat::R16Snorm, &[0xff, 0xff]),
    (PixelFormat::R16Uint, &[0xff, 0xff]),
    (PixelFormat::R16Sint, &[0xff, 0xff]),
    (PixelFormat::R16Float, &[0xff, 0xff]),
    (PixelFormat::B5G6R5Unorm, &[0x1f, 0xf8]),
    (PixelFormat::A1BGR5Unorm, &[0xc1, 0xff]),
    (PixelFormat::ABGR4Unorm, &[0x0f, 0xff]),
    (PixelFormat::BGR5A1Unorm, &[0x1f, 0xfc]),
    (PixelFormat::RGBA8Unorm, MAGENTA),
    (PixelFormat::RGBA8UnormSRGB, MAGENTA),
    (PixelFormat::RGBA8Snorm, MAGENTA),
    (PixelFormat::RGBA8Uint, MAGENTA),
    (PixelFormat::RGBA8Sint, MAGENTA),
    (PixelFormat::BGRA8Unorm, MAGENTA),
    (PixelFormat::BGRA8UnormSRGB, MAGENTA),
    (PixelFormat::RGB10A2Unorm, &[0xff, 0x03, 0xf0, 0xff]),
    (PixelFormat::RGB10A2Uint, &[0xff, 0x03, 0xf0, 0xff]),
    (PixelFormat::BGR10A2Unorm, &[0xff, 0x03, 0xf0, 0xff]),
    (PixelFormat::RG11B10Float, &[0xe0, 0x07, 0x00, 0xfc]),
    (PixelFormat::RGB9E5Float, &[0xff, 0x01, 0xfc, 0xff]),
    (PixelFormat::RG16Unorm, &[0xff, 0xff, 0x00, 0x00]),
    (PixelFormat::RG16Snorm, &[0xff, 0xff, 0x00, 0x00]),
    (PixelFormat::RG16Uint, &[0xff, 0xff, 0x00, 0x00]),
    (PixelFormat::RG16Sint, &[0xff, 0xff, 0x00, 0x00]),
    (PixelFormat::RG16Float, &[0xff, 0xff, 0x00, 0x00]),
    (PixelFormat::R32Float, &[0xff; 4]),
    (PixelFormat::R32Uint, &[0xff; 4]),
    (PixelFormat::R32Sint, &[0xff; 4]),
    (PixelFormat::BGR10XR, &[0xff, 0x03, 0xf0, 0xff]),
    (PixelFormat::BGR10XRSRGB, &[0xff, 0x03, 0xf0, 0xff]),
    (
        PixelFormat::RGBA16Float,
        &[0xff, 0xff, 0, 0, 0xff, 0xff, 0xff, 0xff],
    ),
    (
        PixelFormat::RGBA16Unorm,
        &[0xff, 0xff, 0, 0, 0xff, 0xff, 0xff, 0xff],
    ),
    (
        PixelFormat::RGBA16Snorm,
        &[0xff, 0xff, 0, 0, 0xff, 0xff, 0xff, 0xff],
    ),
    (
        PixelFormat::RGBA16Uint,
        &[0xff, 0xff, 0, 0, 0xff, 0xff, 0xff, 0xff],
    ),
    (
        PixelFormat::RGBA16Sint,
        &[0xff, 0xff, 0, 0, 0xff, 0xff, 0xff, 0xff],
    ),
    (
        PixelFormat::RG32Float,
        &[0xff, 0xff, 0xff, 0xff, 0, 0, 0, 0],
    ),
    (PixelFormat::RG32Uint, &[0xff, 0xff, 0xff, 0xff, 0, 0, 0, 0]),
    (PixelFormat::RG32Sint, &[0xff, 0xff, 0xff, 0xff, 0, 0, 0, 0]),
    (
        PixelFormat::BGRA10XR,
        &[0xc0, 0xff, 0, 0, 0xc0, 0xff, 0x80, 0xdf],
    ),
    (
        PixelFormat::BGRA10XRSRGB,
        &[0xc0, 0xff, 0, 0, 0xc0, 0xff, 0x80, 0xdf],
    ),
    (
        PixelFormat::RGBA32Float,
        &[
            0xff, 0xff, 0xff, 0xff, 0, 0, 0, 0, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
        ],
    ),
    (
        PixelFormat::RGBA32Uint,
        &[
            0xff, 0xff, 0xff, 0xff, 0, 0, 0, 0, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
        ],
    ),
    (
        PixelFormat::RGBA32Sint,
        &[
            0xff, 0xff, 0xff, 0xff, 0, 0, 0, 0, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
        ],
    ),
    (PixelFormat::Depth16Unorm, &[0xff, 0xff]),
    (PixelFormat::Depth32Float, &[0xff; 4]),
    (PixelFormat::Stencil8, &[0xff]),
];

/// A new `Private` 64² texture of each format of [`PATTERNS`], `ShaderRead` (a render target too
/// where the format is a depth or stencil one), reads Metal's pattern in every texel, or zero
/// where the device announced it cannot write it; formats the device refuses are skipped. On a
/// device without Metal's compression each holds what its memory held, counted, and not
/// the pattern, which nothing wrote.
#[test]
fn new_private_textures_read_metal_s_pattern_by_format() {
    let (messages, cache, lane) = pattern_lane();
    let name = lane.device.name();
    let (mut patterned, mut announced, mut left) = (Vec::new(), Vec::new(), Vec::new());
    for &(pf, texel) in PATTERNS {
        let bytes = texel.len() as u64;
        let usage = match pf {
            PixelFormat::Depth16Unorm | PixelFormat::Depth32Float | PixelFormat::Stencil8 => {
                TextureUsage::SHADER_READ | TextureUsage::RENDER_TARGET
            }
            _ => TextureUsage::SHADER_READ,
        };
        let private = ("Private", ResourceOptions::private());
        let shape =
            Shape::new(&format!("{pf:?}"), pf, (64, bytes), usage, private).patterned(texel, 1);
        zeroed(&messages);
        let before = unfilled_count(&lane.device);
        let texture = match lane.device.new_texture(&shape.descriptor) {
            Ok(texture) => texture,
            Err(Error::Unsupported(why)) => {
                println!("FRESH-SKIP {pf:?} on {name}: {why:?}");
                continue;
            }
            Err(error) => panic!("{pf:?} on {name}: {error:?}"),
        };
        assert!(!zeroed(&messages), "{pf:?} on {name}: announced as made");
        let unfilled = unfilled(&lane.device);
        assert_eq!(
            unfilled_count(&lane.device) - before,
            u64::from(unfilled),
            "{pf:?} on {name}: counted as made unfilled"
        );
        let read = texels(&lane, &shape, &texture);
        let zeroed = zeroed(&messages);
        if unfilled {
            let what = held(&read, &[]);
            assert!(
                !zeroed && read != shape.expected(false),
                "{pf:?} on {name}, without Metal's compression: announced {zeroed}, or reads Metal's \
                 pattern, which nothing wrote"
            );
            left.push(format!("{pf:?} {what}"));
            continue;
        }
        assert!(
            read == shape.expected(zeroed),
            "{} on {name} (announced: {zeroed}): reads {:02x?} first",
            shape.name,
            &read[..bytes as usize]
        );
        match zeroed {
            true => announced.push(pf),
            false => patterned.push(pf),
        }
    }
    println!(
        "FRESH-PATTERNS on {name}: Metal's pattern {patterned:?}; zero announced {announced:?}; \
         holding what their memory held {left:?}"
    );
    drop(cache);
    quiet(lane, &messages);
}

/// New `Private` mipmapped `RGBA8Unorm` chains read Metal's pattern in each level whose shorter
/// side is 16 or more, zero in the rest (a 64² chain's levels 64, 32 and 16; 100x40's 100x40 and
/// 50x20; 16x256's and 17²'s first), level by level. On a device without Metal's compression each
/// holds what its memory held, counted, and not the pattern.
#[test]
fn new_private_chains_read_the_pattern_in_their_wide_levels() {
    let (messages, cache, lane) = pattern_lane();
    let name = lane.device.name();
    let private = ("Private", ResourceOptions::private());
    for (width, height, levels, wide) in [
        (64, 64, 7, 3),
        (32, 32, 6, 2),
        (100, 40, 7, 2),
        (16, 256, 9, 1),
        (17, 17, 5, 1),
    ] {
        let mut shape = Shape::new(
            "mipmapped RGBA8Unorm",
            PixelFormat::RGBA8Unorm,
            (width, 4),
            TextureUsage::SHADER_READ,
            private,
        );
        shape.descriptor.height = height;
        shape.descriptor.mipmap_level_count = levels;
        let shape = shape.patterned(MAGENTA, wide);
        let before = unfilled_count(&lane.device);
        let texture = lane.device.new_texture(&shape.descriptor).unwrap();
        let unfilled = unfilled(&lane.device);
        assert_eq!(
            unfilled_count(&lane.device) - before,
            u64::from(unfilled),
            "a new Private {width}x{height} chain on {name}: counted as made unfilled"
        );
        let read = texels(&lane, &shape, &texture);
        match unfilled {
            true => {
                let what = held(&read, &[]);
                assert!(
                    read != shape.expected(false),
                    "a new Private {width}x{height} chain on {name}, without Metal's compression: \
                     reads Metal's pattern, which nothing wrote"
                );
                println!("FRESH-UNFILLED a {width}x{height} chain on {name}: {what}");
            }
            false => assert!(
                read == shape.expected(false),
                "a new Private {width}x{height} chain of {levels} levels on {name}"
            ),
        }
        assert!(!zeroed(&messages), "{width}x{height} on {name}: announced");
    }
    drop(cache);
    quiet(lane, &messages);
}

/// The side of [`msaa_depth`]'s textures: a 4x grid of 32², which Metal compresses.
const DEPTH_SIDE: u32 = 16;

/// A new `Private` 4x `Depth32Float` render target of [`DEPTH_SIDE`]², sampled too: Metal reads
/// its pattern, a NaN depth (all ones), which no clear of a multisampled image writes, so it
/// reads zero here and announces `fresh-pattern-zeroed` where a command first reads it unwritten.
/// With it, whether it holds what its memory held, written by nothing (on a device presented
/// without Metal's compression, counted).
fn msaa_depth(lane: &Lane) -> (Texture, bool) {
    let side = u64::from(DEPTH_SIDE);
    let pf = PixelFormat::Depth32Float;
    let mut descriptor = TextureDescriptor::texture_2d(pf, side, side, false);
    descriptor.texture_type = TextureType::Type2DMultisample;
    descriptor.sample_count = 4;
    descriptor.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
    descriptor.resource_options = ResourceOptions::private();
    let before = unfilled_count(&lane.device);
    let texture = lane.device.new_texture(&descriptor).unwrap();
    let unfilled = unfilled(&lane.device);
    assert_eq!(
        unfilled_count(&lane.device) - before,
        u64::from(unfilled),
        "a new 4x Depth32Float: counted as made unfilled"
    );
    (texture, unfilled)
}

/// A pass of `depth` alone, loaded by `load` (cleared to `clear`) and stored, with no draw.
fn depth_pass(lane: &Lane, depth: &Texture, load: LoadAction, clear: f64) {
    let mut descriptor = RenderPassDescriptor::new();
    let attachment = &mut descriptor.depth_attachment;
    attachment.texture = Some(depth.clone());
    attachment.load_action = load;
    attachment.clear_depth = clear;
    attachment.store_action = StoreAction::Store;
    let command_buffer = lane.queue.command_buffer().unwrap();
    let encoder = command_buffer.render_command_encoder(&descriptor).unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer
        .wait_until_completed_for(super::WAIT)
        .unwrap();
    assert_eq!(command_buffer.error(), None);
}

/// Each sample's depth bits of [`msaa_depth`]'s `texture`, read by a fragment shader.
fn sample_depths(lane: &Lane, texture: &Texture) -> Vec<u32> {
    let command_buffer = lane.queue.command_buffer().unwrap();
    let extent = [DEPTH_SIDE, DEPTH_SIDE];
    let read = encode(
        &lane.device,
        &command_buffer,
        texture,
        SampleRead::Depth,
        extent,
        4,
    );
    let read = read.unwrap();
    command_buffer.commit().unwrap();
    command_buffer
        .wait_until_completed_for(super::WAIT)
        .unwrap();
    assert_eq!(command_buffer.error(), None);
    let bytes = read.bytes().unwrap();
    let words = bytes
        .chunks(4)
        .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]));
    words.collect()
}

/// Whether `depths` are Metal's answer for a never-written texture, by whether the device
/// announced it: zero where it did, else Metal's NaN (all ones).
fn reads_fresh(depths: &[u32], zeroed: bool) -> bool {
    let fresh = if zeroed { 0 } else { u32::MAX };
    !depths.is_empty() && depths.iter().all(|&d| d == fresh)
}

/// A new multisampled `Depth32Float` target cleared whole before anything reads it reads its
/// clear, and the device announces nothing: the difference from Metal's pattern is never seen.
#[test]
fn a_cleared_multisampled_depth_target_reads_its_clear_silently() {
    let (messages, cache, lane) = pattern_lane();
    let name = lane.device.name();
    let (depth, _) = msaa_depth(&lane);
    depth_pass(&lane, &depth, LoadAction::Clear, 0.25);
    let depths = sample_depths(&lane, &depth);
    let quarter = 0.25f32.to_bits();
    assert!(
        depths.iter().all(|&d| d == quarter),
        "a cleared 4x Depth32Float on {name}: {:08x?} first",
        &depths[..4.min(depths.len())]
    );
    assert!(
        !zeroed(&messages),
        "a cleared 4x Depth32Float on {name}: announced"
    );
    drop(cache);
    quiet(lane, &messages);
}

/// A new multisampled `Depth32Float` target that a pass without a draw call clears over a render
/// area of a quarter of it reads its clear in every sample, and the device announces nothing:
/// Metal's clear of a depth it keeps compressed reaches the whole level (`clear_reach.rs`), and
/// the runtime counts the texels that clear wrote, past the render area too
/// (`fresh_unwritten.rs`). On a device presenting an Apple silicon Mac, whose such texture reads
/// zero, noted, where Metal reads its pattern; a runtime counting the render area alone announces
/// `fresh-pattern-zeroed` at the read.
#[test]
fn a_multisampled_depth_cleared_past_its_render_area_reads_its_clear_silently() {
    let (messages, cache, lane) = lane_with(|options| {
        options.announce_each = FRESH_PATTERNS.into_iter().collect();
        options.mac_class = Some(MacClass::AppleSilicon);
    });
    let name = lane.device.name();
    let (depth, unfilled) = msaa_depth(&lane);
    assert!(!unfilled, "a new 4x Depth32Float on {name}: unfilled");
    let mut descriptor = RenderPassDescriptor::new();
    let attachment = &mut descriptor.depth_attachment;
    attachment.texture = Some(depth.clone());
    attachment.load_action = LoadAction::Clear;
    attachment.clear_depth = 0.25;
    attachment.store_action = StoreAction::Store;
    descriptor.render_target_width = u64::from(DEPTH_SIDE / 2);
    descriptor.render_target_height = u64::from(DEPTH_SIDE / 2);
    let command_buffer = lane.queue.command_buffer().unwrap();
    let encoder = command_buffer.render_command_encoder(&descriptor).unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer
        .wait_until_completed_for(super::WAIT)
        .unwrap();
    assert_eq!(command_buffer.error(), None);
    let depths = sample_depths(&lane, &depth);
    let quarter = 0.25f32.to_bits();
    let held = depths.iter().filter(|&&d| d == quarter).count();
    assert_eq!(
        held,
        depths.len(),
        "a 4x Depth32Float on {name} cleared over 8x8 of 16x16: {held} of {} samples hold the clear",
        depths.len()
    );
    assert!(
        !zeroed(&messages),
        "a 4x Depth32Float on {name} cleared over 8x8 of 16x16: announced at its read"
    );
    drop(cache);
    quiet(lane, &messages);
}

/// A new multisampled `Depth32Float` target a pass loads before anything wrote it announces
/// `fresh-pattern-zeroed` as the pass's encoder is made, where the device writes zero for Metal's
/// NaN, and reads that. On a device without Metal's compression it holds what its memory
/// held, and announces nothing.
#[test]
fn a_loaded_never_written_multisampled_depth_announces() {
    let (messages, cache, lane) = pattern_lane();
    let name = lane.device.name();
    let (depth, unfilled) = msaa_depth(&lane);
    assert!(
        !zeroed(&messages),
        "a new 4x Depth32Float on {name}: announced as made"
    );
    depth_pass(&lane, &depth, LoadAction::Load, 0.0);
    let zeroed = zeroed(&messages);
    let depths = sample_depths(&lane, &depth);
    assert!(
        (unfilled && !zeroed) || reads_fresh(&depths, zeroed),
        "a loaded 4x Depth32Float on {name} (announced: {zeroed}): {:08x?} first",
        &depths[..4.min(depths.len())]
    );
    println!("FRESH-OBSERVED loaded 4x Depth32Float on {name}: announced {zeroed}");
    drop(cache);
    quiet(lane, &messages);
}

/// A new multisampled `Depth32Float` target a shader reads before anything wrote it announces
/// `fresh-pattern-zeroed` at that read, where the device writes zero for Metal's NaN. On a device
/// without Metal's compression it holds what its memory held, and announces nothing.
#[test]
fn a_never_written_multisampled_depth_read_by_a_shader_announces() {
    let (messages, cache, lane) = pattern_lane();
    let name = lane.device.name();
    let (depth, unfilled) = msaa_depth(&lane);
    assert!(
        !zeroed(&messages),
        "a new 4x Depth32Float on {name}: announced as made"
    );
    let depths = sample_depths(&lane, &depth);
    let zeroed = zeroed(&messages);
    assert!(
        (unfilled && !zeroed) || reads_fresh(&depths, zeroed),
        "a read 4x Depth32Float on {name} (announced: {zeroed}): {:08x?} first",
        &depths[..4.min(depths.len())]
    );
    println!("FRESH-OBSERVED read 4x Depth32Float on {name}: announced {zeroed}");
    drop(cache);
    quiet(lane, &messages);
}

/// A pass of `target` loaded by `load` (cleared to `clear`) and stored by `store`, resolving
/// into `resolve`.
fn msaa_pass(
    target: &Texture,
    (load, clear): (LoadAction, ClearColor),
    store: StoreAction,
    resolve: Option<&Texture>,
) -> RenderPassDescriptor {
    let mut descriptor = RenderPassDescriptor::new();
    let color = &mut descriptor.color_attachments[0];
    color.texture = Some(target.clone());
    color.load_action = load;
    color.clear_color = clear;
    color.store_action = store;
    color.resolve_texture = resolve.cloned();
    descriptor
}

/// A new `Private` 4x multisampled `RGBA8Unorm` target, made right after one cleared to a color
/// and stored was released, resolves to zero when loaded (the module docs). On a device without
/// Metal's compression it holds what its memory held, counted, its resolve unchecked.
#[test]
fn a_new_multisampled_texture_reads_zero() {
    let (messages, cache, lane) = lane();
    let name = lane.device.name();
    let pf = PixelFormat::RGBA8Unorm;
    let old = lane.msaa(pf);
    let color = ClearColor::new(0.6, 0.3, 0.9, 0.7);
    let cleared = msaa_pass(&old, (LoadAction::Clear, color), StoreAction::Store, None);
    lane.run(&cleared, |_, _, _| {}).unwrap();
    drop(old);
    let before = unfilled_count(&lane.device);
    let new = lane.msaa(pf);
    let unfilled = unfilled(&lane.device);
    assert_eq!(
        unfilled_count(&lane.device) - before,
        u64::from(unfilled),
        "a new Private 4x RGBA8Unorm target on {name}: counted as made unfilled"
    );
    let mut shape = ONE;
    shape.shared = true;
    let resolved = lane.texture(pf, shape, TextureUsage::RENDER_TARGET);
    let none = ClearColor::new(0.0, 0.0, 0.0, 0.0);
    let loaded = msaa_pass(
        &new,
        (LoadAction::Load, none),
        StoreAction::StoreAndMultisampleResolve,
        Some(&resolved),
    );
    lane.run(&loaded, |_, _, _| {}).unwrap();
    let texels = lane.read(&resolved, (0, 0), 4, BlitOption::None);
    match unfilled {
        true => println!(
            "FRESH-UNFILLED a 4x RGBA8Unorm target's resolve on {name}: {}",
            held(&texels, &[])
        ),
        false => assert_eq!(
            non_zero(&texels),
            None,
            "a new Private 4x RGBA8Unorm target's resolve on {name}"
        ),
    }
    drop(cache);
    quiet(lane, &messages);
}

/// New `Private` 4x 8² targets, whose sample grid (16²) Metal compresses, read its pattern when
/// loaded and resolved: `RGBA8Unorm` opaque magenta, written by a clear; `RGBA16Float`'s pattern
/// holds NaNs (halves `ffff 0000 ffff ffff`), whose bits no clear of a multisampled float image
/// controls, so it is copied from an unsigned-integer scratch cleared to its bits. Metal's resolve
/// of those samples stores zero (an M4 under macOS 27.0.1; the consumer's M3 Max alike), and so
/// does MoltenVK's, which is Metal's; a device whose resolve keeps the NaNs (NaN in red, blue and
/// alpha, zero in green) is printed, the resolve's rule and not this pattern's. Where the device
/// makes no such scratch the texture reads zero and the device announces `fresh-pattern-zeroed`
/// where the pass loads it, not as it is made (the module docs).
#[test]
fn new_multisampled_textures_read_the_pattern_or_announce_zero() {
    let (messages, cache, lane) = pattern_lane();
    let name = lane.device.name();
    let msaa = LaneShape {
        size: 8,
        samples: 4,
        slices: 1,
        levels: 1,
        shared: false,
    };
    let resolve = LaneShape {
        samples: 1,
        shared: true,
        ..msaa
    };
    let mut copied = Vec::new();
    for (pf, bytes, nan) in [
        (PixelFormat::RGBA8Unorm, 4, false),
        (PixelFormat::RGBA16Float, 8, true),
    ] {
        zeroed(&messages);
        let before = unfilled_count(&lane.device);
        let new = lane.texture(pf, msaa, TextureUsage::RENDER_TARGET);
        assert!(
            !zeroed(&messages),
            "a new Private 4x 8x8 {pf:?} on {name}: as made"
        );
        let unfilled = unfilled(&lane.device);
        assert_eq!(
            unfilled_count(&lane.device) - before,
            u64::from(unfilled),
            "a new Private 4x 8x8 {pf:?} on {name}: counted as made unfilled"
        );
        let resolved = lane.texture(pf, resolve, TextureUsage::RENDER_TARGET);
        let none = ClearColor::new(0.0, 0.0, 0.0, 0.0);
        let loaded = msaa_pass(
            &new,
            (LoadAction::Load, none),
            StoreAction::StoreAndMultisampleResolve,
            Some(&resolved),
        );
        lane.run(&loaded, |_, _, _| {}).unwrap();
        let announced = zeroed(&messages);
        assert!(
            (nan && !unfilled) || !announced,
            "a new Private 4x 8x8 {pf:?} on {name}, loaded: announced"
        );
        let texels = lane.read(&resolved, (0, 0), bytes, BlitOption::None);
        if unfilled {
            let what = held(&texels, &[]);
            println!("FRESH-UNFILLED a 4x 8x8 {pf:?} target's resolve on {name}: {what}");
            continue;
        }
        let first = &texels[..bytes as usize];
        let zero = texels.iter().all(|&byte| byte == 0);
        // The pattern's NaNs kept by a resolve that is not Metal's.
        let kept = texels.chunks(8).all(|texel| {
            let half = |at: usize| u16::from_le_bytes([texel[at], texel[at + 1]]);
            let nan = |bits: u16| bits & 0x7c00 == 0x7c00 && bits & 0x03ff != 0;
            nan(half(0)) && half(2) == 0 && nan(half(4)) && nan(half(6))
        });
        let fresh = match (nan, announced) {
            (false, _) => texels.chunks(4).all(|texel| texel == MAGENTA),
            (true, true) => zero,
            (true, false) => zero || kept,
        };
        assert!(
            fresh,
            "a new Private 4x 8x8 {pf:?} target's resolve on {name} (announced {announced}): \
             {first:02x?} first"
        );
        if nan {
            copied.push(format!(
                "{pf:?} {}",
                match (announced, zero) {
                    (true, _) => "zero, announced",
                    (false, true) => "copied, resolved to zero as Metal's resolve stores it",
                    (false, false) =>
                        "copied, its NaNs kept by the device's resolve (Metal's stores zero)",
                }
            ));
        }
    }
    println!("FRESH-MSAA-NAN on {name}: {}", copied.join(", "));
    drop(cache);
    quiet(lane, &messages);
}

/// A new `Private` 4x 8² `RGBA16Float` target's samples hold Metal's pattern (halves `ffff 0000
/// ffff ffff`), read through a blit copy into a `PixelFormatView` texture and its
/// unsigned view; or, on a device whose multisampled copies convert float NaNs (the pattern comes
/// from a copy), each NaN half another NaN and `device-nan-encoding` announced at that first read;
/// or zero where the device announced `fresh-pattern-zeroed`. Prints `FRESH-MSAA-SAMPLES`.
#[test]
fn a_new_multisampled_float_texture_s_samples_hold_metal_s_pattern_or_announce() {
    let (messages, cache, lane) = pattern_lane();
    let name = lane.device.name().to_owned();
    let pf = PixelFormat::RGBA16Float;
    let msaa = LaneShape {
        size: 8,
        samples: 4,
        slices: 1,
        levels: 1,
        shared: false,
    };
    zeroed(&messages);
    let before = unfilled_count(&lane.device);
    let new = lane.texture(pf, msaa, TextureUsage::RENDER_TARGET);
    let unfilled = unfilled(&lane.device);
    assert_eq!(
        unfilled_count(&lane.device) - before,
        u64::from(unfilled),
        "a new Private 4x 8x8 {pf:?} on {name}: counted as made unfilled"
    );
    let viewed =
        TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ | TextureUsage::PIXEL_FORMAT_VIEW;
    let copy = lane.texture(pf, msaa, viewed);
    let command_buffer = lane.queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.copy_texture(&new, &copy).unwrap();
    blit.end_encoding().unwrap();
    let read = encode(
        &lane.device,
        &command_buffer,
        &copy,
        SampleRead::Color(ReadbackView::Rg32Uint),
        [8, 8],
        4,
    )
    .unwrap();
    command_buffer.commit().unwrap();
    command_buffer
        .wait_until_completed_for(super::WAIT)
        .unwrap();
    assert_eq!(command_buffer.error(), None);
    let samples = read.bytes().unwrap();
    let notices = counts(&messages.take_notices());
    let converted = notices.contains_key(&Degradation::DeviceNanEncoding);
    let zero = notices.contains_key(&Degradation::FreshPatternZeroed);
    let pattern: [u8; 8] = [0xff, 0xff, 0, 0, 0xff, 0xff, 0xff, 0xff];
    let half = |s: &[u8], at: usize| u16::from_le_bytes([s[at], s[at + 1]]);
    let nan = |bits: u16| bits & 0x7c00 == 0x7c00 && bits & 0x03ff != 0;
    let exact = samples.chunks(8).all(|s| s == pattern);
    let nans = samples
        .chunks(8)
        .all(|s| nan(half(s, 0)) && half(s, 2) == 0 && nan(half(s, 4)) && nan(half(s, 6)));
    let zeros = samples.iter().all(|&b| b == 0);
    println!(
        "FRESH-MSAA-SAMPLES on {name}: {} (device-nan-encoding {converted}, \
         fresh-pattern-zeroed {zero}); first {:02x?}",
        match (exact, nans, zeros) {
            (true, ..) => "Metal's pattern",
            (_, true, _) => "the pattern's NaNs converted",
            (.., true) => "zero",
            _ => "other",
        },
        &samples[..8]
    );
    // On a device without Metal's compression nothing wrote the samples, and nothing
    // announces their pattern zeroed.
    assert!(
        (unfilled && !zero) || exact || (nans && converted) || (zeros && zero),
        "a new Private 4x 8x8 {pf:?} target's samples on {name}: {:02x?} first",
        &samples[..8]
    );
    drop(cache);
    quiet(lane, &messages);
}

/// A new `Private` `RGB10A8_2P_XR10` texture that shaders write reads, in both planes, what a new
/// `Shared` one reads (zero bytes), made right after one written by a kernel was released (the
/// module docs). On a device without Metal's compression it holds what its memory held,
/// counted, its planes unchecked.
#[test]
fn a_new_two_plane_texture_reads_zero_in_both_planes() {
    let (messages, cache, lane) = lane();
    let name = lane.device.name();
    let pf = PixelFormat::RGB10A82PXR10;
    let side = 64;
    let kernels = Kernels::new(&lane.device);
    let dispatch = Dispatch::Texels(side, side);
    let made = |options: ResourceOptions| {
        let mut descriptor = TextureDescriptor::texture_2d(pf, side, side, false);
        descriptor.usage = TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE;
        descriptor.resource_options = options;
        lane.device.new_texture(&descriptor).unwrap()
    };
    let zero = reads(
        &lane.device,
        &lane.queue,
        &kernels,
        &made(ResourceOptions::shared()),
        dispatch,
    );
    let old = made(ResourceOptions::private());
    // Every texel (0.5, 0.25, 0.75, 0.75) as `half4`s: a color word and an alpha byte not zero.
    let texel: Vec<u8> = [0x3800u16, 0x3400, 0x3a00, 0x3a00]
        .iter()
        .flat_map(|h| h.to_le_bytes())
        .collect();
    let values = texel.repeat((side * side) as usize);
    write(
        &lane.device,
        &lane.queue,
        &kernels,
        &old,
        (values.as_slice(), true),
        dispatch,
    );
    let written = reads(&lane.device, &lane.queue, &kernels, &old, dispatch);
    assert_ne!(written, zero, "the written texture on {name}");
    drop(old);
    let before = unfilled_count(&lane.device);
    let new = made(ResourceOptions::private());
    let unfilled = unfilled(&lane.device);
    assert_eq!(
        unfilled_count(&lane.device) > before,
        unfilled,
        "a new Private {pf:?} texture on {name}: counted as made unfilled"
    );
    let read = reads(&lane.device, &lane.queue, &kernels, &new, dispatch);
    let same = read == zero;
    match unfilled {
        true => println!(
            "FRESH-UNFILLED a {pf:?} texture on {name}: {}",
            match same {
                true => "a new Shared one's planes",
                false => "other planes",
            }
        ),
        false => assert!(
            same,
            "a new Private {pf:?} texture on {name} reads other planes than a new Shared one's"
        ),
    }
    drop(cache);
    quiet(lane, &messages);
}

/// Textures over new `Shared` and `Private` buffers read zero (the module docs).
#[test]
fn textures_over_new_buffers_read_zero() {
    let (messages, cache, lane) = lane();
    let name = lane.device.name();
    let (side, row) = (64u64, 256u64);
    for (storage, options) in [
        ("Shared", ResourceOptions::shared()),
        ("Private", ResourceOptions::private()),
    ] {
        let length = row * side;
        let old = lane.device.new_buffer(length, options).unwrap();
        blits(&lane, |blit| {
            blit.fill_buffer(&old, 0..length, OLD).unwrap()
        });
        drop(old);
        let buffer = lane.device.new_buffer(length, options).unwrap();
        let shape = Shape::new(
            "RGBA8Unorm over a new buffer",
            PixelFormat::RGBA8Unorm,
            (side, 4),
            TextureUsage::SHADER_READ,
            (storage, options),
        );
        let texture = buffer.new_texture(&shape.descriptor, 0, row).unwrap();
        assert_eq!(
            non_zero(&texels(&lane, &shape, &texture)),
            None,
            "{} on {name}",
            shape.name
        );
    }
    drop(cache);
    quiet(lane, &messages);
}
