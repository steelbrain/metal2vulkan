//! Device tests of framebuffer fetch through sample planes (its fourth part) on the device the cfg
//! selects, under the validation layer.
//!
//! In the pixel interlock on a device without multisampled storage images (ANV, MoltenVK), the
//! compile layer reads a multisampled pass's fetched attachment as its sample planes, which the
//! runtime splits from the attachment before the draws that fetch it and merges back after them
//! (`runtime::sample_planes`). Each case draws 4x 4² passes whose samples differ and reads back
//! the resolve, which averages a float format's samples exactly:
//!
//! - **Split around a plain draw, stored and resolved**: a 4x target cleared to 0, drawn by
//!   `rt6_resolve_sample_values` (s, 0.1 s, -s, 0.25 s on sample s) and then fetched by
//!   `msf_fetch_sample` (x + 1, s, 2z, w), stored and resolved: per sample (s + 1, s, -2s,
//!   0.25 s), whose average is (2.5, 1.5, -3, 0.375). The fetch reads what the plain draw wrote
//!   only where the pass moves the samples into the planes between the two draws, and the pass
//!   resolves them only where it moves them back before its resolve.
//! - **Loaded**: the stored target loaded and fetched once more, resolved: (s + 2, s, -4s,
//!   0.25 s), averaging (3.5, 1.5, -6, 0.375). The planes hold the stored samples only where the
//!   load moves them in.
//!
//! One format per copy format of its texel size (`bind::PlanesCopy`): `R8Unorm` (8 bits),
//! `RG8Unorm` (16), `R32Float`, `RG16Float` and `RGBA8Unorm` (32), `RGBA16Float` (64) and
//! `RGBA32Float` (128). A unorm format clamps: after either fetch each sample holds 1 in x and
//! min(s, 1) in y, 0 in z, so its resolve holds 255, 191 and 0. Its alpha, 0.25 s rounded
//! from the plain draw, can average to a rounding tie, where a device's resolve may differ from
//! Metal's (`DeviceRoundedResolve`), so it is not compared.
//!
//! Through input attachments (lavapipe) the runtime draws the same passes, Metal's answer too,
//! and so it does in the interlock on a device with multisampled storage images (RADV, NVIDIA),
//! which reads and writes the attachment's own multisampled image. A format the device does not
//! store, which the interlock form needs, is refused at the pipeline (`Unsupported`), printed.
//!
//! A pass whose last draws fetch an attachment that resolves resolves it by the device's average,
//! after its sample planes merge back (`runtime::sample_planes`), never by the crate-owned
//! resolve (`runtime::color_resolve`), a memoryless attachment the pass only fetches and resolves
//! included.

use super::{library, loading, multisampled, pass, pipeline, steps};
use super::{pipeline_descriptor, scratched};
use crate::runtime_lane::degrade_notices::{assert_made_once, counts, forced, resolve_kind};
use crate::runtime_lane::memoryless::memoryless_msaa;
use crate::runtime_lane::resolves::{lane, quiet, Lane, MSAA, ONE};
use metal2vulkan::device::DriverFact;
use metal2vulkan::metal::{PixelFormat, TextureUsage};
use metal2vulkan_runtime::{BlitOption, ClearColor, Degradation, Error, StoreAction, Texture};

/// The kinds [`sample_planes_carry_every_sample`] takes where the device reads sample planes.
pub(in crate::runtime_lane) const PLANES: [Degradation; 1] = [Degradation::SamplePlanes];

/// The per-sample values of `rt6_resolves`.
const SAMPLE_VALUES: &str = "rt6_resolve_sample_values";

/// The texels of a 4² target.
const TEXELS: usize = 16;

/// The bits of `value`, a normal number of 11 significant bits or fewer, as a half.
fn half(value: f32) -> [u8; 2] {
    let bits = value.to_bits();
    let sign = (bits >> 16) & 0x8000;
    let exponent = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mantissa = (bits >> 13) & 0x3ff;
    let half = if value == 0.0 {
        sign
    } else {
        sign | ((exponent as u32) << 10) | mantissa
    };
    (half as u16).to_le_bytes()
}

/// The bytes of one resolved texel of `pf` whose four float components are `value`, and which of
/// them the check compares (the module docs): a float format's every byte, a unorm one's red,
/// green and blue.
fn texel(pf: PixelFormat, value: [f32; 4]) -> (Vec<u8>, Vec<bool>) {
    let floats = |n: usize| value[..n].iter().flat_map(|c| c.to_le_bytes()).collect();
    let halves = |n: usize| value[..n].iter().flat_map(|&c| half(c)).collect();
    // The unorm passes' resolve: 1, min(s, 1) and 0 per sample (the module docs).
    let unorm = |n: usize| [255u8, 191, 0, 0][..n].to_vec();
    let bytes: Vec<u8> = match pf {
        PixelFormat::R32Float => floats(1),
        PixelFormat::RGBA32Float => floats(4),
        PixelFormat::RG16Float => halves(2),
        PixelFormat::RGBA16Float => halves(4),
        PixelFormat::R8Unorm => unorm(1),
        PixelFormat::RG8Unorm => unorm(2),
        PixelFormat::RGBA8Unorm => unorm(4),
        other => panic!("no texel of {other:?}"),
    };
    let compared = (0..bytes.len())
        .map(|i| !(pf == PixelFormat::RGBA8Unorm && i == 3))
        .collect();
    (bytes, compared)
}

/// `bytes`, texels of `compared`'s length, with every byte `compared` leaves out set to 0.
fn masked(bytes: &[u8], compared: &[bool]) -> Vec<u8> {
    bytes
        .chunks(compared.len())
        .flat_map(|texel| {
            texel
                .iter()
                .zip(compared)
                .map(|(&b, &c)| if c { b } else { 0 })
        })
        .collect()
}

/// The 4x targets' formats, one per copy format of their texel size (the module docs).
const FORMATS: [PixelFormat; 7] = [
    PixelFormat::R8Unorm,
    PixelFormat::RG8Unorm,
    PixelFormat::R32Float,
    PixelFormat::RG16Float,
    PixelFormat::RGBA8Unorm,
    PixelFormat::RGBA16Float,
    PixelFormat::RGBA32Float,
];

/// A 4x 4² render target of `pf`, stored by the passes.
fn target(lane: &Lane, pf: PixelFormat) -> Texture {
    lane.texture(pf, MSAA, TextureUsage::RENDER_TARGET)
}

/// Each sample's value reaches the planes from a plain draw and from a load, and leaves them for
/// the store and the resolve, for each copy format (the module docs). Where the device reads
/// sample planes, they are announced once.
#[test]
fn sample_planes_carry_every_sample() {
    let (messages, cache, lane) = lane();
    if !fetches(&lane.device) {
        return;
    }
    let device = lane.device.name().to_owned();
    let fetch = library(&lane, "msf_fetch_ms");
    let vertices = &library(&lane, "rt6_resolves");
    // The pipelines' makings and the resolves of the formats drawn.
    let mut first = std::collections::BTreeSet::new();
    for pf in FORMATS {
        let made =
            pipeline(&lane, vertices, (vertices, SAMPLE_VALUES), &[pf], 4).and_then(|values| {
                let fetch = pipeline(&lane, vertices, (&fetch, "msf_fetch_sample"), &[pf], 4)?;
                Ok((values, fetch))
            });
        let (values, fetching) = match made {
            Ok(made) => made,
            Err(error @ Error::Unsupported(_)) => {
                eprintln!("refused: sample planes of {pf:?} on {device}: {error}");
                continue;
            }
            Err(error) => panic!("sample planes of {pf:?} on {device}: {error}"),
        };
        first.extend(super::made(
            &lane,
            &[
                pipeline_descriptor(vertices, (vertices, SAMPLE_VALUES), &[pf], 4),
                pipeline_descriptor(vertices, (&fetch, "msf_fetch_sample"), &[pf], 4),
            ],
        ));
        first.extend(resolve_kind(&lane.device, 4));
        let msaa = target(&lane, pf);
        let zero = ClearColor::new(0.0, 0.0, 0.0, 0.0);
        let stored = lane.texture(pf, ONE, TextureUsage::RENDER_TARGET);
        let targets = [(&msaa, zero, StoreAction::StoreAndMultisampleResolve)];
        let read = |texture: &Texture, bytes: usize| {
            lane.read(texture, (0, 0), bytes as u64, BlitOption::None)
        };
        let (first, compared) = texel(pf, [2.5, 1.5, -3.0, 0.375]);
        let bytes = compared.len();
        let ran = steps(&lane, &pass(&targets, Some(&stored)), &[&values, &fetching])
            .map(|()| masked(&read(&stored, bytes), &compared));
        let name = format!("of {pf:?} through a split, stored and resolved, 4x");
        multisampled(&lane, &name, ran, &masked(&first, &compared).repeat(TEXELS));
        let (second, _) = texel(pf, [3.5, 1.5, -6.0, 0.375]);
        let resolved = lane.texture(pf, ONE, TextureUsage::RENDER_TARGET);
        let mut descriptor = loading(&msaa, StoreAction::MultisampleResolve);
        descriptor.color_attachments[0].resolve_texture = Some(resolved.clone());
        let ran = steps(&lane, &descriptor, &[&fetching])
            .map(|()| masked(&read(&resolved, bytes), &compared));
        let name = format!("of {pf:?} loaded and resolved, 4x");
        multisampled(
            &lane,
            &name,
            ran,
            &masked(&second, &compared).repeat(TEXELS),
        );
    }
    let planes = super::reads_planes(&lane);
    let expected: &[Degradation] = if planes { &PLANES } else { &[] };
    // The loaded pass stores its 4x target `MultisampleResolve`, which no driver is measured to
    // store as Metal's where the planes split the pass, nor any not measured to store `DONT_CARE`
    // as Metal's; a texture fetched through a scratch takes its storage form.
    let metals = lane
        .device
        .properties()
        .driver_facts
        .contains(&DriverFact::DontCareStoreIsMetals);
    let scratch = scratched(&messages);
    // In the pixel interlock at more than one sample, through the sample planes or the attachment's
    // own multisampled storage image, the fetching draw reads the attachment out of its subpass and
    // the plain draw before it in theirs, and only a single-sampled pass keeps such draws in one
    // instance: the pass splits there (on the RTX 3080 too, through its multisampled storage
    // images). The interlock alone makes that split, so on a driver measured to
    // run such a split no slower than whole (`DriverFact::InterlockedPassSplitNoSlower`, ANV on the
    // Iris Xe of RPL-P and the RPL-S, MoltenVK on the M3 Max, NVIDIA on the RTX 3090 and the RTX
    // 5080 Laptop GPU) it is the device's faster form and is not announced.
    first.extend(forced(&[
        (
            Degradation::RenderPassSplit,
            super::announces_interlock_splits(&lane),
        ),
        (Degradation::DontCareStore, !metals || planes),
        (Degradation::FetchThroughScratch, scratch),
        (Degradation::FetchedTargetStorage, scratch),
    ]));
    let notices = messages.take_notices();
    messages.restore_notices(notices.clone());
    let counts = assert_made_once(&messages, expected, &first);
    assert_eq!(
        counts.contains_key(&Degradation::SamplePlanes),
        planes,
        "sample planes announced on {device}: {counts:?}"
    );
    for notice in notices
        .iter()
        .filter_map(|message| message.degradation.as_ref())
    {
        if [Degradation::SamplePlanes, Degradation::FetchThroughScratch]
            .contains(&notice.degradation)
        {
            eprintln!(
                "sample planes' notice on {device}: {:?}: {}: {}",
                notice.degradation, notice.object, notice.reason
            );
        }
    }
    drop(cache);
    quiet(lane, &messages);
}

/// A memoryless 4x target the pass only fetches and resolves: cleared to 0 and fetched once by
/// `msf_fetch_sample`, so each sample holds (1, s, 0, 0), resolved Metal's way on every device
/// in the channels compared. The device's average
/// resolves it, after its sample planes merge back where the device reads them: no device draws
/// the crate-owned resolve (`shader-resolve`).
#[test]
fn a_memoryless_attachment_only_fetched_resolves_by_the_device_s_average() {
    let (messages, cache, lane) = lane();
    let device = lane.device.name().to_owned();
    let fetch = library(&lane, "msf_fetch_ms");
    let vertices = &library(&lane, "rt6_resolves");
    for pf in [PixelFormat::R32Float, PixelFormat::RGBA8Unorm] {
        let fetching = match pipeline(&lane, vertices, (&fetch, "msf_fetch_sample"), &[pf], 4) {
            Ok(fetching) => fetching,
            Err(error @ Error::Unsupported(_)) => {
                eprintln!("refused: a memoryless fetch of {pf:?} on {device}: {error}");
                continue;
            }
            Err(error) => panic!("a memoryless fetch of {pf:?} on {device}: {error}"),
        };
        let msaa = memoryless_msaa(&lane, pf);
        let zero = ClearColor::new(0.0, 0.0, 0.0, 0.0);
        let resolved = lane.texture(pf, ONE, TextureUsage::RENDER_TARGET);
        let targets = [(&msaa, zero, StoreAction::MultisampleResolve)];
        let (texel, compared) = texel(pf, [1.0, 0.0, 0.0, 0.0]);
        let bytes = compared.len() as u64;
        let ran = steps(&lane, &pass(&targets, Some(&resolved)), &[&fetching]).map(|()| {
            let read = lane.read(&resolved, (0, 0), bytes, BlitOption::None);
            masked(&read, &compared)
        });
        let name = format!("of a memoryless {pf:?} fetched and resolved, 4x");
        multisampled(&lane, &name, ran, &masked(&texel, &compared).repeat(TEXELS));
    }
    let notices = messages.take_notices();
    let counts = counts(&notices);
    eprintln!("a memoryless fetch and resolve's notices on {device}: {counts:?}");
    assert!(
        !counts.contains_key(&Degradation::ShaderResolve),
        "shader-resolve on {device}"
    );
    drop(cache);
    quiet(lane, &messages);
}

/// A 4x `RGBA8Unorm` target the pass clears to 0, fetches once by `msf_fetch_sample` (each sample
/// (1, s, 0, 0), clamped) and resolves into its sRGB twin: `MultisampleResolve` resolves the
/// samples as `StoreAndMultisampleResolve` does, Metal's answer either way (the store does not
/// change what is resolved). The twin's resolve is the crate-owned one, which reads the
/// attachment itself (`color_resolve.rs`), so where the sample planes hold the samples as the pass
/// ends they merge back before it under either action.
#[test]
fn a_twin_resolve_of_a_fetched_attachment_reads_its_samples() {
    let (messages, cache, lane) = lane();
    let device = lane.device.name().to_owned();
    let fetch = library(&lane, "msf_fetch_ms");
    let vertices = &library(&lane, "rt6_resolves");
    let pf = PixelFormat::RGBA8Unorm;
    match pipeline(&lane, vertices, (&fetch, "msf_fetch_sample"), &[pf], 4) {
        Ok(fetching) => {
            let zero = ClearColor::new(0.0, 0.0, 0.0, 0.0);
            let resolve = |store| {
                let msaa = target(&lane, pf);
                let usage = TextureUsage::RENDER_TARGET;
                let resolved = lane.texture(PixelFormat::RGBA8UnormSRGB, ONE, usage);
                let targets = [(&msaa, zero, store)];
                steps(&lane, &pass(&targets, Some(&resolved)), &[&fetching])
                    .map(|()| lane.read(&resolved, (0, 0), 4, BlitOption::None))
            };
            let stored = resolve(StoreAction::StoreAndMultisampleResolve)
                .unwrap_or_else(|error| panic!("a stored twin resolve on {device}: {error}"));
            // Each pixel's red is 255 in every sample and its blue 0.
            assert!(
                stored.chunks(4).all(|t| t[0] == 0xff && t[2] == 0),
                "a stored twin resolve on {device}: {stored:02x?}"
            );
            let name = "of a fetched RGBA8Unorm resolved into its sRGB twin alone, 4x";
            multisampled(
                &lane,
                name,
                resolve(StoreAction::MultisampleResolve),
                &stored,
            );
        }
        Err(error @ Error::Unsupported(_)) => {
            eprintln!("refused: a fetched twin resolve on {device}: {error}");
        }
        Err(error) => panic!("a fetched twin resolve on {device}: {error}"),
    }
    let _ = messages.take_notices();
    drop(cache);
    quiet(lane, &messages);
}

/// The memory requirements of a 4x color render target's image without and with the usage the
/// sample planes' copies need (sampled, mutable, its texel size's unsigned-integer format in its
/// format list), which the runtime gives the image only where the two are the same
/// (`framebuffer_fetch::with_free_usage`); printed for each format and size, beside those of
/// the image the runtime made for such a texture. The create infos follow `image_plan.rs` for a
/// texture of `RenderTarget` usage alone: color attachment and transfer usage, input attachment
/// usage where the device's fetch reads input attachments, and the sRGB twin in the format list
/// of `RGBA8Unorm`.
#[test]
fn planes_usage_memory_requirements() {
    use ash::vk;
    let (messages, cache, lane) = lane();
    let device = lane.device.name().to_owned();
    // SAFETY: used only while the device lives; every image made here is destroyed at once.
    let raw = unsafe { lane.device.raw() };
    let inputs = super::through_inputs(&lane);
    // The memory requirements of a 4x color image of `format`, `side`², `usage` and `flags`,
    // listing `formats` (none: no list).
    let requirements = |format, side, usage, flags, formats: &[vk::Format]| {
        let mut list = vk::ImageFormatListCreateInfo::default().view_formats(formats);
        let mut info = vk::ImageCreateInfo::default()
            .flags(flags)
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(vk::Extent3D {
                width: side,
                height: side,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_4)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        if !formats.is_empty() {
            info = info.push_next(&mut list);
        }
        // SAFETY: the device is live, and `info` a valid 2D color image's; the image is
        // destroyed at once, with no memory bound.
        unsafe {
            let image = raw.device.create_image(&info, None).unwrap();
            let needs = raw.device.get_image_memory_requirements(image);
            raw.device.destroy_image(image, None);
            (needs.size, needs.alignment)
        }
    };
    let cases = [
        (
            PixelFormat::RGBA8Unorm,
            vk::Format::R8G8B8A8_UNORM,
            Some(vk::Format::R8G8B8A8_SRGB),
            vk::Format::R32_UINT,
        ),
        (
            PixelFormat::RGBA16Float,
            vk::Format::R16G16B16A16_SFLOAT,
            None,
            vk::Format::R32G32_UINT,
        ),
        (
            PixelFormat::R32Float,
            vk::Format::R32_SFLOAT,
            None,
            vk::Format::R32_UINT,
        ),
    ];
    for (pf, format, twin, copy) in cases {
        for side in [16u32, 256, 2048] {
            let mut usage = vk::ImageUsageFlags::COLOR_ATTACHMENT
                | vk::ImageUsageFlags::TRANSFER_SRC
                | vk::ImageUsageFlags::TRANSFER_DST;
            if inputs {
                usage |= vk::ImageUsageFlags::INPUT_ATTACHMENT;
            }
            let mut own: Vec<vk::Format> = vec![format];
            own.extend(twin);
            let (without_flags, without_formats) = if twin.is_some() {
                (vk::ImageCreateFlags::MUTABLE_FORMAT, own.clone())
            } else {
                (vk::ImageCreateFlags::empty(), Vec::new())
            };
            let without = requirements(format, side, usage, without_flags, &without_formats);
            let mut with_formats = own;
            with_formats.push(copy);
            let with = requirements(
                format,
                side,
                usage | vk::ImageUsageFlags::SAMPLED,
                vk::ImageCreateFlags::MUTABLE_FORMAT,
                &with_formats,
            );
            let texture = lane.texture(
                pf,
                crate::runtime_lane::resolves::Shape {
                    size: side.into(),
                    ..MSAA
                },
                TextureUsage::RENDER_TARGET,
            );
            // SAFETY: read only while the texture lives.
            let image = unsafe { texture.vk_image() }.expect("a texture with an image");
            // SAFETY: a live image of the device.
            let made = unsafe { raw.device.get_image_memory_requirements(image) };
            eprintln!(
                "planes usage of a 4x {pf:?} {side}x{side} on {device}: without {without:?}, \
                 with {with:?} ({}), the runtime's image {:?}",
                if with == without { "free" } else { "differs" },
                (made.size, made.alignment)
            );
        }
    }
    drop(cache);
    quiet(lane, &messages);
}
