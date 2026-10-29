//! The shadow load: valid on every committed device that has what it needs and refused where it
//! does not, and each texel of the shadow its codes' tile value, run through the interpreter
//! against the tile model of `lower::arith::extended_range`'s tests.

use super::*;
use crate::lower::arith::extended_range::tests::{raw, tile_model, EVERY_FORM};
use crate::metal::format::tests::committed_devices;
use crate::spirv::interp::{run_with_images, Config, Image, Order, Resources};
use crate::spirv::testing::assert_valid;

/// The forms whose tile memory is unmeasured.
const UNMEASURED: [ExtendedRange; 7] = [
    ExtendedRange::R10X6,
    ExtendedRange::Rg10X12,
    ExtendedRange::R10X6Srgb,
    ExtendedRange::Rg10X12Srgb,
    ExtendedRange::R16Bfloat,
    ExtendedRange::Rg16Bfloat,
    ExtendedRange::Rgba16Bfloat,
];

/// On every committed device the load of each measured form is valid where the device has what
/// it needs, and refused (`Unsupported`, under `FramebufferFetch`) where it does not; an
/// unmeasured form's is `NotImplemented(Format)` everywhere.
#[test]
fn the_shadow_load_is_valid_where_the_device_has_what_it_needs() {
    let mut built = 0;
    for (name, device) in committed_devices() {
        let profile = device.profile();
        for form in EVERY_FORM {
            let m = || {
                extended_shadow_typed_module(form)
                    .expect("a measured form")
                    .expect("built")
            };
            let fit = crate::spirv::serialize::serialize(m())
                .map(|r| profile.check(&r.requirements).is_ok())
                .expect("serialize");
            match extended_shadow_module(&device, form) {
                Ok(module) => {
                    assert!(fit, "{name} {form:?}");
                    assert_valid(m(), profile);
                    assert!(!module.spirv.is_empty(), "{name} {form:?}");
                    built += 1;
                }
                Err(Error::Unsupported(u)) => {
                    assert!(!fit, "{name} {form:?}");
                    assert_eq!(u.feature, Feature::FramebufferFetch, "{name} {form:?}");
                }
                Err(other) => panic!("{name} {form:?}: {other}"),
            }
        }
        for form in UNMEASURED {
            assert!(extended_shadow_typed_module(form).is_none(), "{form:?}");
            match extended_shadow_module(&device, form) {
                Err(Error::Unsupported(u)) => assert_eq!(
                    u,
                    Unsupported::not_implemented(Feature::Format),
                    "{name} {form:?}"
                ),
                other => panic!("{name} {form:?}: {other:?}"),
            }
        }
    }
    assert!(built > 0, "no committed device builds the load");
}

/// A shadow of 37 by 29 texels over codes of 40 by 30, in 5 by 4 workgroups: every texel of the
/// shadow takes the tile value of the codes' texel there (each of the 1024 codes of every
/// channel, the bits Metal does not read set), the codes stay, and the invocations past the
/// shadow touch nothing (the codes end at row 30, before the last workgroup row's 32), in either
/// invocation order.
#[test]
fn each_texel_of_the_shadow_takes_its_codes_tile_value() {
    let (width, height) = (37u32, 29u32);
    let groups = [
        width.div_ceil(ExtendedShadowModule::LOCAL_SIZE),
        height.div_ceil(ExtendedShadowModule::LOCAL_SIZE),
        1,
    ];
    // The `_sRGB` forms' decode takes the transfer by the device's `Pow`, which the interpreter
    // does not model: their tile values are the device's.
    for form in EVERY_FORM.into_iter().filter(|f| !f.srgb()) {
        let m = extended_shadow_typed_module(form)
            .expect("a measured form")
            .expect("built");
        let alpha = |c: u32| match form {
            ExtendedRange::Planes { .. } => c % 256,
            _ => c,
        };
        let codes_of = |c: u32| [c, 1023 - c, (c * 7) % 1024, alpha(c)];
        let mut codes = Image::new(CODES, [40, 30, 1], 1, 1, 1);
        for y in 0..30 {
            for x in 0..40 {
                let n = y * width + x;
                let at = codes.index(0, 0, 0, [x, y, 0]).expect("inside");
                codes.texels[at] = raw(form, codes_of(n % 1024), n * 13);
            }
        }
        let mut shadow = Image::new(SHADOW, [width, height, 1], 1, 1, 1);
        shadow.texels.fill([0xdead_beef; 4]);
        for order in [Order::Forward, Order::Reverse] {
            let cfg = Config {
                order,
                workgroups: groups,
                ..Config::default()
            };
            let images = vec![codes.clone(), shadow.clone()];
            let (_, images) = run_with_images(&m, &cfg, Resources::default(), images).expect("run");
            assert_eq!(images[0], codes, "{form:?}: the codes stay");
            for y in 0..height {
                for x in 0..width {
                    let n = y * width + x;
                    let at = images[1].index(0, 0, 0, [x, y, 0]).expect("inside");
                    let want = tile_model(form, codes_of(n % 1024)).map(f32::to_bits);
                    assert_eq!(images[1].texels[at], want, "{form:?} ({x}, {y}) {order:?}");
                }
            }
        }
    }
}
