//! The formats the function's logic operation serves, and the outputs it takes.

use super::*;

/// Every channel an 8-, 10- or 16-bit unorm one (a 10-bit format's codes operated on at 16 bits),
/// an 8- or 16-bit snorm one, or an 8-, 16- or 32-bit integer one, in Metal's channel order; a
/// packed 16-bit, packed integer, sRGB or float format is not served.
#[test]
fn the_operation_serves_the_per_channel_formats() {
    use LogicNumeric::{Sint, Snorm, Uint, Unorm};
    let served = [
        (PixelFormat::RGBA8Unorm, Unorm, [Some(8); 4]),
        (PixelFormat::BGRA8Unorm, Unorm, [Some(8); 4]),
        (PixelFormat::R8Unorm, Unorm, [Some(8), None, None, None]),
        (PixelFormat::RGBA16Unorm, Unorm, [Some(16); 4]),
        (PixelFormat::RGBA8Snorm, Snorm, [Some(8); 4]),
        (
            PixelFormat::RG16Snorm,
            Snorm,
            [Some(16), Some(16), None, None],
        ),
        (PixelFormat::RGBA8Uint, Uint, [Some(8); 4]),
        (PixelFormat::R16Uint, Uint, [Some(16), None, None, None]),
        (PixelFormat::R32Uint, Uint, [Some(32), None, None, None]),
        (PixelFormat::RGBA8Sint, Sint, [Some(8); 4]),
        (
            PixelFormat::RG16Sint,
            Sint,
            [Some(16), Some(16), None, None],
        ),
        (PixelFormat::RGBA32Sint, Sint, [Some(32); 4]),
        (PixelFormat::RGB10A2Unorm, Unorm, [Some(16); 4]),
        (PixelFormat::BGR10A2Unorm, Unorm, [Some(16); 4]),
    ];
    for (pf, numeric, bits) in served {
        assert_eq!(
            channels(pf),
            Some(LogicChannels { numeric, bits }),
            "{pf:?}"
        );
    }
    for pf in [
        PixelFormat::RGB10A2Uint,
        PixelFormat::B5G6R5Unorm,
        PixelFormat::A1BGR5Unorm,
        PixelFormat::ABGR4Unorm,
        PixelFormat::BGR5A1Unorm,
        PixelFormat::RGBA8UnormSRGB,
        PixelFormat::RGBA16Float,
        PixelFormat::RG11B10Float,
    ] {
        assert_eq!(channels(pf), None, "{pf:?}");
    }
}

/// Vulkan's `logicOp`, which operates on a channel's stored codes and stores its result as it is,
/// may store another code than Metal's operation into the 10-bit unorm formats, whose codes Metal
/// widens, and the signed normalized ones, whose lowest code Metal reads and stores as the code
/// above it; into every other format the two take the same codes.
#[test]
fn vulkan_s_operation_differs_on_10_bit_and_signed_codes_alone() {
    for pf in [
        PixelFormat::RGB10A2Unorm,
        PixelFormat::BGR10A2Unorm,
        PixelFormat::R8Snorm,
        PixelFormat::RG8Snorm,
        PixelFormat::RGBA8Snorm,
        PixelFormat::R16Snorm,
        PixelFormat::RG16Snorm,
        PixelFormat::RGBA16Snorm,
    ] {
        assert!(vulkan_differs(pf), "{pf:?}");
    }
    for pf in [
        PixelFormat::RGBA8Unorm,
        PixelFormat::R8Unorm,
        PixelFormat::RGBA16Unorm,
        PixelFormat::RGBA8Uint,
        PixelFormat::RGBA8Sint,
        PixelFormat::R32Sint,
        PixelFormat::RGB10A2Uint,
        PixelFormat::B5G6R5Unorm,
        PixelFormat::RGBA8UnormSRGB,
        PixelFormat::RGBA16Float,
    ] {
        assert!(!vulkan_differs(pf), "{pf:?}");
    }
}

/// A normalized format takes a `float` or `half` output, an integer one a 32-bit integer output.
#[test]
fn the_operation_takes_its_class_s_outputs() {
    let unorm = channels(PixelFormat::RGBA8Unorm).expect("served");
    let uint = channels(PixelFormat::RGBA8Uint).expect("served");
    let form = |held| Form { held, lanes: 4 };
    let float = form(Held::Float(FloatFmt::F32));
    let half = form(Held::Float(FloatFmt::Half));
    let int32 = form(Held::Int {
        bits: 32,
        container: 32,
    });
    let int16 = form(Held::Int {
        bits: 16,
        container: 32,
    });
    assert!(takes(unorm, float) && takes(unorm, half));
    assert!(!takes(unorm, int32));
    assert!(takes(uint, int32));
    assert!(!takes(uint, int16) && !takes(uint, float));
}
