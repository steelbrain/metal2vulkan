//! The formats the function's logic operation serves, and the outputs it takes.

use super::*;

/// Every channel an 8- or 16-bit unorm or snorm one, or an 8-, 16- or 32-bit integer one, in
/// Metal's channel order; a 10-bit, packed 16-bit, packed integer, sRGB or float format is not
/// served.
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
    ];
    for (pf, numeric, bits) in served {
        assert_eq!(
            channels(pf),
            Some(LogicChannels { numeric, bits }),
            "{pf:?}"
        );
    }
    for pf in [
        PixelFormat::RGB10A2Unorm,
        PixelFormat::BGR10A2Unorm,
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
