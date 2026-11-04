//! Where a NaN clear keeps a `half` NaN's sign (`nan_clear::keeps_nan_sign`): only in a pass
//! without a draw call, at a level of 16 texels or more a side of a texture Metal keeps
//! compressed (an M4 under macOS 27.0.1), on a device presented with unified memory; nowhere on
//! one presented without, which keeps no texture compressed.
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. It makes textures and records nothing.
//! So does the check that a multisampled NaN-clear target takes its unsigned-integer view on a
//! device whose copies convert float NaNs.

use crate::framebuffer_fetch::views_in;
use crate::nan_clear::{keeps_nan_sign, view_candidate};
use crate::render_pass::Target;
use crate::{Device, DeviceKind, DeviceOptions, ResourceOptions, ShaderCacheOptions};
use crate::{Texture, TextureDescriptor};
use metal2vulkan::bind::PlanesCopy;
use metal2vulkan::error::RuntimeDriverDefect;
use metal2vulkan::metal::{PixelFormat, TextureType, TextureUsage};
use metal2vulkan::vk::Format;
use std::path::{Path, PathBuf};

/// A scratch directory for the device's shader cache, removed when dropped.
struct Scratch(PathBuf);

impl Scratch {
    fn new() -> Scratch {
        let path = std::env::temp_dir().join(format!("m2v-nan-signs-{}", std::process::id()));
        std::fs::create_dir_all(&path).unwrap();
        Scratch(path)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// The device the cfg selects, its shader cache in `cache`.
fn device(cache: &Path) -> Device {
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
        ..DeviceOptions::default()
    };
    Device::system_default_with(&options).unwrap()
}

/// The attachment of level 0 of a new private `RGBA16Float` texture, `side` square, of `samples`
/// and `usage`.
fn target(device: &Device, side: u32, samples: u32, usage: TextureUsage) -> Target {
    let texture: Texture = device
        .new_texture(&TextureDescriptor {
            texture_type: match samples {
                1 => TextureType::Type2D,
                _ => TextureType::Type2DMultisample,
            },
            pixel_format: PixelFormat::RGBA16Float,
            width: side.into(),
            height: side.into(),
            sample_count: samples.into(),
            usage,
            resource_options: ResourceOptions::private(),
            ..TextureDescriptor::default()
        })
        .unwrap();
    Target {
        texture,
        level: 0,
        layer: 0,
        layers: 1,
        format: Format::R16G16B16A16_SFLOAT,
        samples,
        extent: [side, side],
    }
}

/// A pass with a draw call drops the sign at every level; one without keeps it at a plain level
/// of 16 or more, and drops it at a smaller one and at one of `PixelFormatView` usage (which Metal
/// keeps uncompressed). On a device presented without unified memory every pass drops it.
#[test]
fn a_draw_call_drops_a_half_nan_s_sign() {
    let scratch = Scratch::new();
    let device = device(&scratch.0);
    let compresses = device.properties().has_unified_memory;
    let plain = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
    let viewed = plain | TextureUsage::PIXEL_FORMAT_VIEW;
    for samples in [1, 4] {
        for (side, usage, kept) in [
            (16, plain, true),
            (64, plain, true),
            (8, plain, false),
            (15, plain, false),
            (64, viewed, false),
        ] {
            let target = target(&device, side, samples, usage);
            let case = format!("{samples}x {side}x{side} {usage:?}");
            assert_eq!(
                keeps_nan_sign(&target, false),
                kept && compresses,
                "{case} without a draw call"
            );
            assert!(!keeps_nan_sign(&target, true), "{case} with a draw call");
        }
    }
}

/// On a device whose multisampled copies convert float NaNs (RADV), a multisampled render target
/// the runtime makes, of a format of floats narrower than 32 bits, takes a view in the
/// unsigned-integer format of its texel size, through which a copy keeps its samples' bits
/// (`nan_clear.rs`), wherever it costs the image nothing; where the view would change the image's
/// memory requirements, its copies are announced instead. Elsewhere nothing asks for it. Each
/// prints `NAN-CLEAR-VIEW`.
#[test]
fn a_nan_clear_target_takes_an_unsigned_view_where_copies_convert_and_it_is_free() {
    let scratch = Scratch::new();
    let device = device(&scratch.0);
    let converts = device
        .properties()
        .driver_defects
        .contains(&RuntimeDriverDefect::MultisampledCopiesConvertFloatNans);
    for pf in [
        PixelFormat::R16Float,
        PixelFormat::RG16Float,
        PixelFormat::RGBA16Float,
        PixelFormat::RG11B10Float,
    ] {
        let texture = device
            .new_texture(&TextureDescriptor {
                texture_type: TextureType::Type2DMultisample,
                pixel_format: pf,
                width: 16,
                height: 16,
                sample_count: 4,
                usage: TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ,
                resource_options: ResourceOptions::private(),
                ..TextureDescriptor::default()
            })
            .unwrap();
        let plan = texture.image_plan().unwrap();
        let view = PlanesCopy::for_format(plan.format).unwrap().format();
        let candidate = view_candidate(plan, &device).unwrap();
        let requirements = |plan| {
            device
                .image_requirements(plan, TextureType::Type2DMultisample)
                .unwrap()
                .map(|r| (r.size, r.alignment))
        };
        // The view the plan was not given costs the image (its requirements differ).
        let costs = candidate
            .as_ref()
            .map(|candidate| requirements(candidate) != requirements(plan));
        println!(
            "NAN-CLEAR-VIEW {pf:?} 4x on {}: copies convert NaNs {converts}; {view:?} view {}; \
             not taken, costing {costs:?}",
            device.name(),
            views_in(plan, view)
        );
        assert!(converts || candidate.is_none(), "{pf:?}");
        assert!(costs.is_none_or(|costs| costs), "{pf:?}: {plan:?}");
    }
}
