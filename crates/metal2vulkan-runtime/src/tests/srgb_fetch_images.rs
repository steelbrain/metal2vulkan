//! The images a fetching draw reads and writes an 8-bit sRGB color attachment through in the
//! pixel interlock (the sRGB fetch form, with the compile form): a storage view in the
//! unsigned-integer format of its texel size (`framebuffer_fetch::storage_view_format`), beside the
//! view in its own format that the draws that do not fetch it render through.
//!
//! - **A scratch or backing** (`attachment_stand_in::stand_in`): a single-sampled one takes
//!   `STORAGE`, made mutable to the view with extended usage (its own format and the view's
//!   listed); a multisampled one too where the fetch reads the attachment's own multisampled
//!   image and the device stores images of its samples; both views of it are made.
//! - **A texture of `PixelFormatView` and `ShaderWrite` usage** (mutable to any format of its
//!   texel size, with `STORAGE`): the storage view is made of its own image.
//! - **Sample planes** (`sample_planes.rs`), where the fetch reads them: made mutable to the view,
//!   which the fetching draws' storage view of them takes.
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects, asking for the validation layer: a view
//! the image does not take, or a view whose usage its format lacks, is a failing message. What the
//! claim of the layer's silence holds is what the device's instance enabled of the layer it asked
//! for (`tests/layer_claim.rs`), so a host the loader can give it no layer says so rather than
//! passing on an empty message list.

use super::layer_claim::Watched;
use crate::attachment_stand_in::stand_in;
use crate::framebuffer_fetch::{
    storage_view_format, stores_samples, through_multisampled_storage, through_planes, views_in,
};
use crate::image_plan::ImagePlan;
use crate::render_pass::Target;
use crate::render_target::{color_view, storage_view_in};
use crate::sample_planes::Planes;
use crate::{
    Device, DeviceKind, DeviceOptions, ResourceOptions, ShaderCacheOptions, Texture,
    TextureDescriptor,
};
use metal2vulkan::device::FormatUsage;
use metal2vulkan::metal::{PixelFormat, TextureType, TextureUsage};
use metal2vulkan::vk::{Format, ImageCreateFlags, ImageUsageFlags};
use std::path::{Path, PathBuf};

/// The attachments' side, and the multisampled one's samples.
const SIDE: u32 = 16;
const SAMPLES: u32 = 4;

/// The 8-bit sRGB formats the compile layer reads through the storage form, and the view.
const FORMATS: [(PixelFormat, Format); 2] = [
    (PixelFormat::BGRA8UnormSRGB, Format::B8G8R8A8_SRGB),
    (PixelFormat::RGBA8UnormSRGB, Format::R8G8B8A8_SRGB),
];

/// A scratch directory for the device's shader cache, removed when dropped.
struct Scratch(PathBuf);

impl Scratch {
    fn new() -> Scratch {
        let path =
            std::env::temp_dir().join(format!("m2v-srgb-fetch-images-{}", std::process::id()));
        std::fs::create_dir_all(&path).unwrap();
        Scratch(path)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// The device the cfg selects, its shader cache in `cache`, asking for the validation layer, and
/// what its instance had the layer report ([`Watched::layer_reported_nothing`] says what a claim
/// of its silence holds; the instance's own answer, never the loader's listing).
fn watched(cache: &Path) -> (Device, Watched) {
    let watched = Watched::default();
    let sink = watched.sink(|_| {});
    let options = DeviceOptions {
        kinds: match cfg!(m2v_cpu_icd) {
            true => vec![DeviceKind::Cpu],
            false => vec![
                DeviceKind::Discrete,
                DeviceKind::Integrated,
                DeviceKind::Virtual,
            ],
        },
        shader_cache: ShaderCacheOptions {
            directory: Some(cache.to_owned()),
            ..ShaderCacheOptions::default()
        },
        layers: watched.layers(),
        messages: Some(sink),
        ..DeviceOptions::default()
    };
    let device = Device::system_default_with(&options).unwrap();
    (device, watched)
}

/// A private 2D texture of `pf`, `SIDE` square, of `samples` and `usage`.
fn texture(device: &Device, pf: PixelFormat, samples: u32, usage: TextureUsage) -> Texture {
    device
        .new_texture(&TextureDescriptor {
            texture_type: match samples {
                1 => TextureType::Type2D,
                _ => TextureType::Type2DMultisample,
            },
            pixel_format: pf,
            width: SIDE.into(),
            height: SIDE.into(),
            sample_count: samples.into(),
            usage,
            resource_options: ResourceOptions::private(),
            ..TextureDescriptor::default()
        })
        .unwrap()
}

/// The attachment of `texture`'s level and layer, of `format` and `samples`.
fn target(texture: Texture, format: Format, samples: u32) -> Target {
    Target {
        texture,
        level: 0,
        layer: 0,
        layers: 1,
        format,
        samples,
        extent: [SIDE, SIDE],
    }
}

/// Whether `plan` is made mutable to `view`, with extended usage and `STORAGE`, its own format
/// and the view's listed first. A multisampled stand-in a resolve into its format's twin reads
/// through the twin's view lists that view after them (`color_resolve::twin_read_view`: the NVIDIA
/// RTX 5080 Laptop's `BGRA8Unorm_sRGB` at 4 samples, `B8G8R8A8_UNORM`).
fn mutable_to(plan: &ImagePlan, format: Format, view: Format) -> bool {
    plan.usage.contains(ImageUsageFlags::STORAGE)
        && plan
            .flags
            .contains(ImageCreateFlags::MUTABLE_FORMAT | ImageCreateFlags::EXTENDED_USAGE)
        && plan.view_formats.starts_with(&[format, view])
}

#[test]
fn an_srgb_attachment_is_fetched_through_its_unsigned_integer_view() {
    let cache = Scratch::new();
    let (device, watched) = watched(&cache.0);
    // The command buffer whose encoder would make the scratch images.
    let queue = device.new_command_queue().unwrap();
    let maker = queue.command_buffer().unwrap();
    let profile = device.compile_device().profile();
    let (mut stand_ins, mut multisampled, mut planes) = (0, 0, 0);
    for (pf, format) in FORMATS {
        let view = storage_view_format(profile, format);
        assert_eq!(view, Format::R32_UINT, "{pf:?}");
        if !profile.supports_format(view, FormatUsage::StorageImage) {
            eprintln!("{pf:?}: the device stores no {view:?} image: refused, typed");
            continue;
        }
        // A render target alone: its scratch takes the view; the pass renders the scratch in
        // its own format and fetches it in the view's.
        let own = target(
            texture(&device, pf, 1, TextureUsage::RENDER_TARGET),
            format,
            1,
        );
        let scratch = stand_in(&maker, &own, true).unwrap();
        let plan = scratch.texture.image_plan().unwrap();
        assert!(mutable_to(plan, format, view), "{pf:?}: {plan:?}");
        storage_view_in(&scratch, view).unwrap();
        color_view(&scratch, false).unwrap();
        stand_ins += 1;
        // A texture mutable to any format of its texel size, with `STORAGE`: fetched in place.
        let usage = TextureUsage::RENDER_TARGET
            | TextureUsage::SHADER_WRITE
            | TextureUsage::PIXEL_FORMAT_VIEW;
        let writable = target(texture(&device, pf, 1, usage), format, 1);
        let plan = writable.texture.image_plan().unwrap();
        if plan.usage.contains(ImageUsageFlags::STORAGE) && views_in(plan, view) {
            storage_view_in(&writable, view).unwrap();
            color_view(&writable, false).unwrap();
        }
        // Multisampled: the attachment's own image where the device stores its samples, or its
        // sample planes.
        let own = target(
            texture(&device, pf, SAMPLES, TextureUsage::RENDER_TARGET),
            format,
            SAMPLES,
        );
        let scratch = match stand_in(&maker, &own, true) {
            Ok(scratch) => scratch,
            Err(error) => {
                eprintln!("{pf:?} at {SAMPLES}x: no scratch: {error:?}");
                continue;
            }
        };
        let plan = scratch.texture.image_plan().unwrap();
        if through_multisampled_storage(profile, SAMPLES) && stores_samples(profile, SAMPLES) {
            assert!(
                mutable_to(plan, format, view),
                "{pf:?} at {SAMPLES}x: {plan:?}"
            );
            storage_view_in(&scratch, view).unwrap();
            color_view(&scratch, false).unwrap();
            multisampled += 1;
        } else if through_planes(profile, SAMPLES) {
            assert!(
                !plan.usage.contains(ImageUsageFlags::STORAGE),
                "{pf:?}: {plan:?}"
            );
            match Planes::new(&maker, &scratch, SAMPLES, [SIDE, SIDE]) {
                Ok(made) => {
                    assert_eq!(made.view, view, "{pf:?}");
                    let plan = made.texture.image_plan().unwrap();
                    assert!(views_in(plan, view), "{pf:?}: {plan:?}");
                    made.draw_view().unwrap();
                    planes += 1;
                }
                Err(error) => eprintln!("{pf:?} at {SAMPLES}x: no sample planes: {error:?}"),
            }
        }
    }
    eprintln!(
        "srgb fetch images: {stand_ins} scratch images, {multisampled} multisampled, \
         {planes} sample planes"
    );
    drop((maker, queue, device));
    watched.layer_reported_nothing("every fetch image's commands were valid and nothing leaked");
}
