//! Blit copies between a buffer and a multisampled depth or stencil texture
//! (`runtime::sample_interleave`, whose scratch the multisampled blit copies' move,
//! `runtime::sample_copy`, carries into and out of the texture's aspect), on the device the cfg
//! selects, under the validation layer, against Metal's answer.
//!
//! **Metal's answer** (our own probe, an Apple M4 under macOS 27.0.1, every case
//! below): each copy unit of the buffer lands in its sample by the layout measured for colour,
//! texel `(x, y)` sample `s` at `offset + y * bytesPerRow + (x * n + s) * bpp`, the samples outside
//! a region keep the clear, and the copy back out returns the buffer bit for bit, every edge value
//! below included (none flushed, quieted or clamped).
//!
//! Each case clears its texture (depth 0.25, stencil `0x5a`), copies a buffer in, reads every
//! sample by a kernel (`candidate::runtime_samples`), and copies the region back out into a
//! second buffer of the same layout. The first case is the consumer's screensaver's copy: a
//! `Depth32Float` texture of 54x54 at 4 samples, whole, `bytesPerRow` 864 and `bytesPerImage`
//! 46656; the second its login window's at HiDPI, 80x80, `bytesPerRow` 1280. A `Depth32Float` buffer's first 16 words are edge values (signed zeros, denormals, 1
//! and the float after it, 2, -1, infinities, NaN payloads, the largest float); every other word
//! is a distinct normal value in `(0, 0.18]`, which every path copies exactly.
//!
//! - **Where each sample lands**: the kernel's reads of the normal values and of the stencil and
//!   16-bit codes are the buffer's, by Metal's layout, and the samples outside the region keep
//!   the clear.
//! - **The copy back**: the buffer, bit for bit, but for an edge value the device is known to
//!   change, each change announced or a recorded driver fact: by the `FragDepth` draw where the
//!   driver's fragment depth is not measured to keep every value (`drawn-depth-copy`, its driver's
//!   measured changes: NVIDIA's denormals as a zero of their sign and NaN as -inf; MoltenVK's
//!   positive NaN writing nothing, a value with the sign bit set as +0, one past 1 as 1; on a
//!   driver not measured, what Vulkan leaves to it: any value outside `[0, 1]`, a denormal as a
//!   zero, -0 as +0), by a depth image without
//!   `VK_EXT_depth_range_unrestricted` whose driver may clamp (`restricted-depth-upload`; ANV's
//!   stores a value past 1 as 1, and NaN and a value with the sign bit set as +0), and by a depth
//!   image whose driver is not measured to keep a denormal (no
//!   `DriverFact::DepthImageKeepsDenormals`: RADV's and ANV's hold one as a zero).
//! - **Refused** only where the move refuses: a 16-bit depth on a device whose maintenance8
//!   copies are missing or wrong (`RuntimeDriverDefect::MultisampledDepthStencilColourCopies`),
//!   whose draws would convert it (unmeasured), as `NotImplemented(MultisampledBufferCopy)`. A
//!   32-bit float depth is copied in on every such device, by the draw, announced where inexact.
//!
//! Each case prints a `DEPTH-BUFFER` line, and each edge value a `DEPTH-EDGE` line (what the
//! kernel read and what came back), so a lane's log shows every device's answer.

use super::degrade_notices::{assert_made_once, forced, loses_depth_denormals, signs_depth_tests};
use super::degrade_notices::{made_compute_kinds, shared_in_system_memory};
use super::{
    buffer_with, layer_reported_nothing_at_all, read_buffer, validated_device, Messages,
    ScratchCache, WAIT,
};
use crate::candidate::runtime_samples::{encode, read_function};
use crate::executor_contract::multisample::SampleRead;
use crate::gpu::TEST_DEVICE;
use metal2vulkan::device::{DriverFact, VkExtension, VkFeature};
use metal2vulkan::error::RuntimeDriverDefect;
use metal2vulkan::metal::{PixelFormat, TextureType, TextureUsage};
use metal2vulkan::vk::DriverId;
use metal2vulkan_runtime::{
    BlitOption, CommandBufferStatus, Degradation, Device, Error, LoadAction, Origin,
    RenderPassDescriptor, ResourceOptions, RuntimeGap, Size, StoreAction, TextureDescriptor,
};
use std::collections::BTreeSet;

/// The depth and stencil the texture is cleared to before the copy.
const CLEAR_DEPTH: f32 = 0.25;
const CLEAR_STENCIL: u32 = 0x5a;

/// A `Depth32Float` buffer's first words: +0, -0, the smallest and largest denormals, a negative
/// denormal, the smallest normal, 1, the float after 1, 2, -1, +inf, -inf, a quiet NaN, a
/// signalling NaN's payload, a negative NaN's payload, the largest float.
const EDGES: [u32; 16] = [
    0x0000_0000,
    0x8000_0000,
    0x0000_0001,
    0x007f_ffff,
    0x8000_0001,
    0x0080_0000,
    0x3f80_0000,
    0x3f80_0001,
    0x4000_0000,
    0xbf80_0000,
    0x7f80_0000,
    0xff80_0000,
    0x7fc0_0000,
    0x7f80_0001,
    0xffc0_0001,
    0x7f7f_ffff,
];

/// One copy: the texture's format, samples and extent, and the region's origin and size.
#[derive(Clone, Copy, Debug)]
struct Case {
    pf: PixelFormat,
    samples: u32,
    extent: [u32; 2],
    origin: [u32; 2],
    size: [u32; 2],
}

const fn whole(pf: PixelFormat, samples: u32) -> Case {
    square(pf, samples, 54)
}

const fn square(pf: PixelFormat, samples: u32, side: u32) -> Case {
    Case {
        pf,
        samples,
        extent: [side, side],
        origin: [0, 0],
        size: [side, side],
    }
}

const fn part(pf: PixelFormat) -> Case {
    Case {
        pf,
        samples: 4,
        extent: [8, 8],
        origin: [2, 1],
        size: [4, 3],
    }
}

/// The cases, the screensaver's first, then the login window's at HiDPI (80x80 at 4 samples,
/// `bytesPerRow` 1280).
const CASES: [Case; 9] = [
    whole(PixelFormat::Depth32Float, 4),
    square(PixelFormat::Depth32Float, 4, 80),
    whole(PixelFormat::Depth32Float, 2),
    part(PixelFormat::Depth32Float),
    whole(PixelFormat::Stencil8, 4),
    whole(PixelFormat::Stencil8, 2),
    part(PixelFormat::Stencil8),
    whole(PixelFormat::Depth16Unorm, 4),
    whole(PixelFormat::Depth16Unorm, 2),
];

impl Case {
    /// The bytes of a copy unit in the buffer.
    fn unit_bytes(self) -> usize {
        match self.pf {
            PixelFormat::Stencil8 => 1,
            PixelFormat::Depth16Unorm => 2,
            _ => 4,
        }
    }

    /// The copy units of the region, in the buffer's order.
    fn units(self) -> usize {
        (self.size[0] * self.size[1] * self.samples) as usize
    }

    /// The bytes of a row of the region in the buffer, tight.
    fn bytes_per_row(self) -> u64 {
        u64::from(self.size[0] * self.samples) * self.unit_bytes() as u64
    }

    /// The buffer's copy units (the module docs).
    fn input(self) -> Vec<u32> {
        (0..self.units())
            .map(|i| match self.pf {
                PixelFormat::Stencil8 => ((i * 37 + 11) & 0xff) as u32,
                PixelFormat::Depth16Unorm => ((i * 7919 + 13) & 0xffff) as u32,
                _ => match EDGES.get(i) {
                    Some(&edge) => edge,
                    None => ((i + 1) as f32 / 65536.0).to_bits(),
                },
            })
            .collect()
    }

    /// How the kernel reads the texture.
    fn read(self) -> SampleRead {
        match self.pf {
            PixelFormat::Stencil8 => SampleRead::Stencil,
            _ => SampleRead::Depth,
        }
    }

    /// A sample's copy unit as the kernel's read gives it: a stencil's value, a 16-bit depth's
    /// code (its normalized value scaled back), a 32-bit float depth's bits.
    fn read_unit(self, word: u32) -> u32 {
        match self.pf {
            PixelFormat::Depth16Unorm => (f32::from_bits(word) * 65535.0).round() as u32,
            _ => word,
        }
    }

    /// Whether the case's copy units are 32-bit float depths.
    fn float_depth(self) -> bool {
        self.pf == PixelFormat::Depth32Float
    }
}

/// What a device does to a 32-bit float depth on the way: its driver, and whether the copy in
/// ran by an inexact `FragDepth` draw, may clamp outside `[0, 1]`, or may lose denormals.
#[derive(Clone, Copy)]
struct Changes {
    driver: Option<DriverId>,
    drawn: bool,
    clamped: bool,
    flushed: bool,
}

fn is_nan(word: u32) -> bool {
    f32::from_bits(word).is_nan()
}

fn is_denormal(word: u32) -> bool {
    word & 0x7f80_0000 == 0 && word & 0x007f_ffff != 0
}

impl Changes {
    /// Whether `word`, copied over a sample holding `before`, may come back as `back` (the module
    /// docs).
    fn allows(self, word: u32, back: u32, before: u32) -> bool {
        let value = f32::from_bits(word);
        let one = 1.0f32.to_bits();
        let sign = word & 0x8000_0000;
        if back == word {
            return true;
        }
        // A depth image that may clamp (ANV's): a value past 1 lands as 1, and
        // a NaN or a value with the sign bit set (-0 and negative denormals included) as +0.
        if self.clamped {
            let clamped = match is_nan(word) || sign != 0 {
                true => 0,
                false if value > 1.0 => one,
                false => word,
            };
            if back == clamped {
                return true;
            }
        }
        if self.flushed && is_denormal(word) && back == sign {
            return true;
        }
        if !self.drawn {
            return false;
        }
        match self.driver {
            // `depth_copy`'s measured changes of NVIDIA's fragment depth: denormals flushed to a
            // zero of their sign, NaN stored as -inf.
            Some(DriverId::NVIDIA_PROPRIETARY) => {
                (is_denormal(word) && back == sign) || (is_nan(word) && back == 0xff80_0000)
            }
            // MoltenVK's: a positive NaN writes nothing, the sample keeping
            // what it held; a value with the sign bit set (-0, -inf, a negative NaN) lands as +0;
            // a value past 1 and +inf as 1.
            Some(DriverId::MOLTENVK) => match (is_nan(word), sign != 0) {
                (true, false) => back == before,
                (_, true) => back == 0,
                (false, false) => value > 1.0 && back == one,
            },
            // A driver whose fragment depth is not measured (`depth_copy::UNMEASURED_FLOAT_DEPTH`):
            // Vulkan leaves a value outside `[0, 1]` undefined, and the driver may flush a
            // denormal to zero or store -0 as +0.
            _ => {
                let outside = is_nan(word) || !(0.0..=1.0).contains(&value);
                outside
                    || (is_denormal(word) && back & 0x7fff_ffff == 0)
                    || (word == sign && back == 0)
            }
        }
    }
}

/// What one case did.
enum Outcome {
    Ran,
    Refused,
    NoTexture,
}

/// Run `case` on `device` (the module docs), `changes` what the device may do to an edge value.
fn run(device: &Device, case: Case, changes: Changes) -> Outcome {
    let name = device.name().to_owned();
    let [width, height] = case.extent;
    let mut d = TextureDescriptor::texture_2d(case.pf, width.into(), height.into(), false);
    d.texture_type = TextureType::Type2DMultisample;
    d.sample_count = case.samples.into();
    d.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
    d.resource_options = ResourceOptions::private();
    let Ok(texture) = device.new_texture(&d) else {
        return Outcome::NoTexture;
    };
    let input = case.input();
    let unit = case.unit_bytes();
    let bytes: Vec<u8> = input
        .iter()
        .flat_map(|word| word.to_le_bytes()[..unit].to_vec())
        .collect();
    let bpr = case.bytes_per_row();
    let bpi = bytes.len() as u64;
    let source = buffer_with(device, &bytes).unwrap();
    let out = buffer_with(device, &vec![0xee; bytes.len()]).unwrap();
    let [ox, oy] = case.origin;
    let origin = Origin::new(ox.into(), oy.into(), 0);
    let size = Size::new(case.size[0].into(), case.size[1].into(), 1);
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut pass = RenderPassDescriptor::new();
    if case.pf == PixelFormat::Stencil8 {
        pass.stencil_attachment.texture = Some(texture.clone());
        pass.stencil_attachment.load_action = LoadAction::Clear;
        pass.stencil_attachment.store_action = StoreAction::Store;
        pass.stencil_attachment.clear_stencil = CLEAR_STENCIL;
    } else {
        pass.depth_attachment.texture = Some(texture.clone());
        pass.depth_attachment.load_action = LoadAction::Clear;
        pass.depth_attachment.store_action = StoreAction::Store;
        pass.depth_attachment.clear_depth = CLEAR_DEPTH.into();
    }
    command_buffer
        .render_command_encoder(&pass)
        .unwrap()
        .end_encoding()
        .unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    let into = blit.copy_from_buffer_to_texture(
        &source,
        0,
        bpr,
        bpi,
        size,
        &texture,
        0,
        0,
        origin,
        BlitOption::None,
    );
    match into {
        Ok(()) => {}
        Err(Error::NotImplemented(RuntimeGap::MultisampledBufferCopy)) => {
            blit.end_encoding().unwrap();
            return Outcome::Refused;
        }
        Err(other) => panic!("{name} {case:?}: the copy in: {other}"),
    }
    blit.end_encoding().unwrap();
    let read = encode(
        device,
        &command_buffer,
        &texture,
        case.read(),
        case.extent,
        case.samples,
    )
    .unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.copy_from_texture_to_buffer(
        &texture,
        0,
        0,
        origin,
        size,
        &out,
        0,
        bpr,
        bpi,
        BlitOption::None,
    )
    .unwrap_or_else(|e| panic!("{name} {case:?}: the copy out, once the copy in ran: {e}"));
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);

    // Where each sample landed: the kernel's read of every sample, texel by texel.
    let texel = case.read().texel_bytes();
    let samples = case.samples as usize;
    let read: Vec<u32> = read
        .bytes()
        .unwrap()
        .chunks_exact(texel)
        .map(|b| {
            let mut word = [0u8; 4];
            word[..texel].copy_from_slice(b);
            u32::from_le_bytes(word)
        })
        .collect();
    let edges = if case.float_depth() { EDGES.len() } else { 0 };
    let mut outside = Vec::new();
    for y in 0..height as usize {
        for x in 0..width as usize {
            for s in 0..samples {
                let got = read[(y * width as usize + x) * samples + s];
                let (rx, ry) = (x.wrapping_sub(ox as usize), y.wrapping_sub(oy as usize));
                if rx >= case.size[0] as usize || ry >= case.size[1] as usize {
                    outside.push(got);
                    continue;
                }
                let i = (ry * case.size[0] as usize + rx) * samples + s;
                if i >= edges {
                    assert_eq!(
                        case.read_unit(got),
                        input[i],
                        "{name} {case:?}: sample {s} of ({x}, {y}) holds unit {i}"
                    );
                }
            }
        }
    }
    outside.sort_unstable();
    outside.dedup();
    let clear = match case.pf {
        PixelFormat::Depth32Float => vec![CLEAR_DEPTH.to_bits()],
        PixelFormat::Stencil8 => vec![CLEAR_STENCIL],
        _ => outside.first().copied().into_iter().collect(),
    };
    assert!(
        outside.is_empty() || outside == clear,
        "{name} {case:?}: the samples outside the region hold {outside:x?}"
    );

    // The copy back: the buffer's units, bit for bit, but for an edge value's known change.
    let back: Vec<u32> = read_buffer(&out, 0, bpi)
        .unwrap()
        .chunks_exact(unit)
        .map(|b| {
            let mut word = [0u8; 4];
            word[..unit].copy_from_slice(b);
            u32::from_le_bytes(word)
        })
        .collect();
    // Every edge value's answer first, so a lane's log shows each one whatever fails.
    for i in 0..edges {
        let (x, s) = (ox as usize + i / samples, i % samples);
        let at = (oy as usize * width as usize + x) * samples + s;
        println!(
            "DEPTH-EDGE {name} n={} in={:08x} read={:08x} back={:08x}",
            case.samples, input[i], read[at], back[i]
        );
    }
    let changed: Vec<String> = input
        .iter()
        .zip(&back)
        .enumerate()
        .filter(|&(i, (&word, &got))| {
            word != got && !(i < edges && changes.allows(word, got, CLEAR_DEPTH.to_bits()))
        })
        .map(|(i, (word, got))| format!("unit {i} {word:08x} came back {got:08x}"))
        .collect();
    assert!(changed.is_empty(), "{name} {case:?}: {changed:?}");
    Outcome::Ran
}

/// Every case in turn on one device, both ways (the module docs).
#[test]
fn a_buffer_copies_every_sample_of_a_multisampled_depth_or_stencil_both_ways() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let name = device.name().to_owned();
        let properties = device.properties();
        let defect = RuntimeDriverDefect::MultisampledDepthStencilColourCopies;
        let copies = device.enabled().features.contains(&VkFeature::Maintenance8)
            && !properties.driver_defects.contains(&defect);
        let facts = &properties.driver_facts;
        let exact_draw = facts.contains(&DriverFact::FragmentDepthStoresUnchanged);
        let unrestricted = device
            .enabled()
            .extensions
            .contains(&VkExtension::ExtDepthRangeUnrestricted);
        let changes = Changes {
            driver: properties.description.driver_id,
            drawn: !copies && !exact_draw,
            clamped: !unrestricted,
            flushed: !facts.contains(&DriverFact::DepthImageKeepsDenormals),
        };
        messages.take_notices();
        let mut drawn_depth = false;
        // Whether a copy into a 32-bit float depth ran, and the kinds the readbacks' state makings
        // announce.
        let mut float_depth = false;
        // Whether a 16-bit depth texture was cleared: 0.25 is 16383.75 codes, which Metal's
        // clear truncates and the device converts its own way (`unorm-depth-rounding`).
        let mut cleared16 = false;
        let mut made = BTreeSet::new();
        for case in CASES {
            let supported = properties
                .supports_texture_sample_count(case.samples.into())
                .unwrap_or(false);
            let outcome = run(&device, case, changes);
            cleared16 |=
                case.pf == PixelFormat::Depth16Unorm && !matches!(outcome, Outcome::NoTexture);
            let word = match outcome {
                Outcome::Ran => "RAN",
                Outcome::Refused => "REFUSED",
                Outcome::NoTexture => "NO-TEXTURE",
            };
            println!(
                "DEPTH-BUFFER {name} {:?} n={} {}x{} at ({}, {}) of {}x{}: {word}",
                case.pf,
                case.samples,
                case.size[0],
                case.size[1],
                case.origin[0],
                case.origin[1],
                case.extent[0],
                case.extent[1]
            );
            match outcome {
                Outcome::Ran => {
                    drawn_depth |= case.float_depth() && changes.drawn;
                    float_depth |= case.float_depth();
                    let read = read_function(&device, case.pf, case.read()).unwrap();
                    made.extend(made_compute_kinds(&device, read));
                }
                // The draws convert a 16-bit depth (unmeasured), so it moves only by
                // maintenance8's copies.
                Outcome::Refused => assert!(
                    case.pf == PixelFormat::Depth16Unorm && !copies,
                    "{name} refused {case:?}"
                ),
                Outcome::NoTexture => assert!(!supported, "{name} made no {case:?}"),
            }
        }
        // The copies in announce what a float depth's may store, and the buffers are `Shared`.
        let mut first = forced(&[
            (Degradation::DrawnDepthCopy, drawn_depth),
            (
                Degradation::RestrictedDepthUpload,
                float_depth && changes.clamped,
            ),
            (
                Degradation::DepthDenormals,
                float_depth && loses_depth_denormals(&device, false),
            ),
            (
                Degradation::SignedDepthTest,
                float_depth && signs_depth_tests(&device),
            ),
            (
                Degradation::SharedInSystemMemory,
                shared_in_system_memory(&device),
            ),
            (Degradation::UnormDepthRounding, cleared16),
        ]);
        first.extend(made);
        let announced = assert_made_once(&messages, &[], &first);
        assert_eq!(
            announced.contains_key(&Degradation::DrawnDepthCopy),
            drawn_depth,
            "drawn-depth-copy on {name} exactly where a depth copy in ran by an inexact draw: \
             {announced:?}"
        );
        assert_eq!(
            announced.contains_key(&Degradation::RestrictedDepthUpload),
            float_depth && changes.clamped,
            "restricted-depth-upload on {name} exactly where its depth images may clamp: \
             {announced:?}"
        );
    }
    layer_reported_nothing_at_all(&messages);
}
