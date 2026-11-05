//! The class of Mac a device presents (`MacClass`, `DeviceOptions::mac_class`), on the device the
//! GPU cfg selects, under both settings: the default, its GPU's class, and a declared class other
//! than its GPU's.
//!
//! - **It is presented.** `DeviceProperties::lossless_compression` is true exactly where the
//!   class is `MacClass::AppleSilicon`.
//! - **A new `Private` texture starts as that Mac's.** A 64 x 64 `RGBA8Unorm` render target,
//!   read by a blit before anything writes it:
//!   - an Apple silicon Mac's reads Metal's fixed pattern, opaque magenta, in every texel
//!     (`texture/heap_tiled.rs`'s `fresh_pattern`), which the runtime writes whatever the GPU;
//!   - a Mac with an Intel or AMD GPU's holds what its memory held: on a GPU that is not Apple's
//!     the runtime writes nothing (`WorkCounts::private_textures_unfilled` moves by one), and the
//!     test prints what it held (`MAC-CLASS-UNFILLED`); on an Apple GPU, whose own compressed
//!     image would read Apple's pattern, it reads zero
//!     (`WorkCounts::private_textures_zero_filled` moves by one).

use super::blit_holds::{Scratch, WAIT};
use crate::texture::heap_tiled::{fresh_pattern, FreshPattern};
use crate::{
    BlitOption, Device, DeviceKind, DeviceOptions, MacClass, Origin, ResourceOptions,
    ShaderCacheOptions, Size, TextureDescriptor,
};
use metal2vulkan::metal::{PixelFormat, TextureUsage};
use std::path::Path;

const SIDE: u64 = 64;

/// The device the cfg selects, presenting `class` (`None`: its GPU's), its shader cache in
/// `cache`.
fn device(cache: &Path, class: Option<MacClass>) -> Device {
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
        mac_class: class,
        ..DeviceOptions::default()
    };
    Device::system_default_with(&options).unwrap()
}

/// What a new `Private` render target of `device` reads, blitted out before anything writes it,
/// and how far it moved the unfilled and zero-filled counts.
fn fresh(device: &Device) -> (Vec<u8>, u64, u64) {
    let mut descriptor = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, SIDE, SIDE, false);
    descriptor.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
    descriptor.resource_options = ResourceOptions::private();
    let before = device.work_counts();
    let texture = device.new_texture(&descriptor).unwrap();
    let after = device.work_counts();
    let length = SIDE * SIDE * 4;
    let out = device
        .new_buffer(length, ResourceOptions::shared())
        .unwrap();
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.copy_from_texture_to_buffer(
        &texture,
        0,
        0,
        Origin::new(0, 0, 0),
        Size::new(SIDE, SIDE, 1),
        &out,
        0,
        SIDE * 4,
        length,
        BlitOption::None,
    )
    .unwrap();
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    assert!(command_buffer.wait_until_completed_for(WAIT).is_ok());
    // SAFETY: the shared buffer's bytes, which no command uses now.
    let read =
        unsafe { std::slice::from_raw_parts(out.contents().unwrap().as_ptr(), length as usize) }
            .to_vec();
    (
        read,
        after.private_textures_unfilled - before.private_textures_unfilled,
        after.private_textures_zero_filled - before.private_textures_zero_filled,
    )
}

/// `device`, presenting `class`, presents it and starts a new `Private` texture as that Mac's
/// (the module docs).
fn starts_as(device: &Device, class: MacClass) {
    let name = device.name();
    let compresses = class == MacClass::AppleSilicon;
    assert_eq!(
        device.properties().lossless_compression,
        compresses,
        "{name}: {class:?}"
    );
    let apple_gpu = MacClass::of_vendor(device.properties().vendor_id) == MacClass::AppleSilicon;
    let (read, unfilled, zero_filled) = fresh(device);
    match (class, apple_gpu) {
        (MacClass::AppleSilicon, _) => {
            let Some(FreshPattern::Color { texel, .. }) = fresh_pattern(PixelFormat::RGBA8Unorm)
            else {
                panic!("RGBA8Unorm has a colour pattern");
            };
            assert_eq!((unfilled, zero_filled), (0, 0), "{name}: {class:?}");
            assert!(
                read.chunks(texel.len()).all(|t| t == texel),
                "{name}: {class:?}: a new texture reads Metal's pattern"
            );
        }
        (MacClass::IntelOrAmd, true) => {
            assert_eq!((unfilled, zero_filled), (0, 1), "{name}: {class:?}");
            assert!(
                read.iter().all(|&byte| byte == 0),
                "{name}: {class:?}: a new texture on an Apple GPU reads zero"
            );
        }
        (MacClass::IntelOrAmd, false) => {
            assert_eq!((unfilled, zero_filled), (1, 0), "{name}: {class:?}");
            let held = match read.iter().all(|&byte| byte == 0) {
                true => "zero",
                false => "other bytes",
            };
            println!("MAC-CLASS-UNFILLED on {name}: {held}");
        }
    }
    println!("MAC-CLASS on {name}: {class:?}, Apple GPU {apple_gpu}");
}

/// The default is the class of the device's GPU: an Apple silicon Mac's on an Apple GPU, a Mac
/// with an Intel or AMD GPU's on every other.
#[test]
fn the_default_class_is_the_gpu_s() {
    let scratch = Scratch::new("mac-class-default");
    let device = device(&scratch.0, None);
    let own = MacClass::of_vendor(device.properties().vendor_id);
    starts_as(&device, own);
}

/// A declared class other than the GPU's is the one the device presents.
#[test]
fn a_declared_class_other_than_the_gpu_s_is_presented() {
    let scratch = Scratch::new("mac-class-declared");
    let own = MacClass::of_vendor(device(&scratch.0, None).properties().vendor_id);
    let other = match own {
        MacClass::AppleSilicon => MacClass::IntelOrAmd,
        MacClass::IntelOrAmd => MacClass::AppleSilicon,
    };
    let device = device(&scratch.0, Some(other));
    starts_as(&device, other);
}
