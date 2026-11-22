use super::*;
use crate::metal::{PixelFormat, TextureUsage};
use crate::pipeline::sym::{SampledType, StorageFormatClass};
use crate::spirv::testing::committed_profiles;

fn class(
    dim: ImageDim,
    arrayed: bool,
    ms: bool,
    ty: SampledType,
    access: ImageAccess,
) -> ImageHandleClass {
    ImageHandleClass {
        dim,
        arrayed,
        multisampled: ms,
        sampled_type: ty,
        depth_compare: false,
        access,
        format: if access == ImageAccess::Storage {
            StorageFormatClass::Unknown
        } else {
            StorageFormatClass::NotStorage
        },
    }
}

fn shape(ty: TextureType, pf: PixelFormat, samples: u32, usage: TextureUsage) -> TextureShape {
    TextureShape::Texture {
        address_modes: crate::metal::sampler::AddressModeMap::IDENTITY,
        offset_base_axes: 0,
        texture_type: ty,
        pixel_format: pf,
        sample_count: samples,
        usage,
        mipmap_level_count: 1,
        sparse: false,
        swizzle: crate::metal::TextureSwizzleChannels::DEFAULT,
    }
}

const READ: TextureUsage = TextureUsage::SHADER_READ;
const WRITE: TextureUsage = TextureUsage::SHADER_WRITE;

/// A member fits a class by its texture type's dimension, arrayedness, and multisampling, its
/// sample count, its numeric type, and the usage the class's views need; a storage class never
/// holds a depth or stencil format, and no class holds a texture buffer.
#[test]
fn fit_is_shape_samples_numeric_type_usage_and_aspect() {
    use ImageAccess::{Sampled, Storage};
    use SampledType::{Float, Uint};
    use TextureType as T;
    let f2d = class(ImageDim::D2, false, false, Float, Sampled);
    let rgba = PixelFormat::RGBA8Unorm;
    assert_eq!(fit(&shape(T::Type2D, rgba, 1, READ), &f2d), Fit::Holds);
    assert_eq!(
        fit(&shape(T::Type2D, PixelFormat::Depth32Float, 1, READ), &f2d),
        Fit::Holds
    );
    assert_eq!(
        fit(&shape(T::Type2D, PixelFormat::R32Uint, 1, READ), &f2d),
        Fit::NumericType
    );
    assert_eq!(
        fit(
            &shape(T::Type2D, rgba, 1, TextureUsage::RENDER_TARGET),
            &f2d
        ),
        Fit::No
    );
    assert_eq!(fit(&shape(T::Type2DArray, rgba, 1, READ), &f2d), Fit::No);
    // A format the table holds no representation of fits no class: the heap does not know which
    // of its textures the function reads, so a resident one it cannot view is the neutral, and
    // nothing is refused.
    let unheld = PixelFormat::RGB8Unorm;
    assert_eq!(fit(&shape(T::Type2D, unheld, 1, READ), &f2d), Fit::No);
    assert_eq!(
        fit(
            &shape(T::Type2D, unheld, 1, READ),
            &class(ImageDim::D2, false, false, Uint, Sampled)
        ),
        Fit::No
    );
    assert_eq!(fit(&shape(T::Type2D, rgba, 4, READ), &f2d), Fit::No);
    assert_eq!(fit(&shape(T::Cube, rgba, 1, READ), &f2d), Fit::No);
    let ms = class(ImageDim::D2, false, true, Float, Sampled);
    assert_eq!(
        fit(&shape(T::Type2DMultisample, rgba, 4, READ), &ms),
        Fit::Holds
    );
    assert_eq!(
        fit(&shape(T::Type2DMultisample, rgba, 1, READ), &ms),
        Fit::No
    );
    let u2d = class(ImageDim::D2, false, false, Uint, Sampled);
    assert_eq!(
        fit(&shape(T::Type2D, PixelFormat::R32Uint, 1, READ), &u2d),
        Fit::Holds
    );
    let s2d = class(ImageDim::D2, false, false, Float, Storage);
    assert_eq!(fit(&shape(T::Type2D, rgba, 1, WRITE), &s2d), Fit::Holds);
    assert_eq!(fit(&shape(T::Type2D, rgba, 1, READ), &s2d), Fit::No);
    // Apple's private `BlockWritesOnly` is a usage Metal's validation takes a write through, as
    // `ShaderWrite` (CoreAnimation's blur textures are `0x10005`).
    let blocked = TextureUsage::BLOCK_WRITES_ONLY;
    for usage in [blocked, READ | TextureUsage::RENDER_TARGET | blocked] {
        assert_eq!(fit(&shape(T::Type2D, rgba, 1, usage), &s2d), Fit::Holds);
        assert_eq!(fit(&shape(T::Type2D, rgba, 1, usage), &f2d), {
            if usage.contains(READ) {
                Fit::Holds
            } else {
                Fit::No
            }
        });
    }
    assert_eq!(
        fit(&shape(T::Type2D, PixelFormat::Depth32Float, 1, WRITE), &s2d),
        Fit::No
    );
    assert_eq!(
        fit(&shape(T::Type2D, PixelFormat::R32Uint, 1, WRITE), &s2d),
        Fit::NumericType
    );
    // A depth-comparison class holds exactly the depth formats of its shape, as a compared slot
    // takes; the float class of the shape holds them too (a read without comparison).
    let d2d = ImageHandleClass {
        depth_compare: true,
        ..f2d
    };
    for pf in [
        PixelFormat::Depth32Float,
        PixelFormat::Depth16Unorm,
        PixelFormat::Depth32FloatStencil8,
    ] {
        assert_eq!(
            fit(&shape(T::Type2D, pf, 1, READ), &d2d),
            Fit::Holds,
            "{pf:?}"
        );
    }
    assert_eq!(fit(&shape(T::Type2D, rgba, 1, READ), &d2d), Fit::No);
    assert_eq!(
        fit(
            &shape(T::Type2DArray, PixelFormat::Depth32Float, 1, READ),
            &d2d
        ),
        Fit::No
    );
    let dcube = ImageHandleClass {
        depth_compare: true,
        ..class(ImageDim::Cube, false, false, Float, Sampled)
    };
    assert_eq!(
        fit(&shape(T::Cube, PixelFormat::Depth32Float, 1, READ), &dcube),
        Fit::Holds
    );
}

/// On lavapipe a heap's storage conversions are the alias a slot takes for each of the three
/// formats `read-write-tier-held` sweeps beyond every tier, keyed apart from their red-first and
/// UNORM kin by class and red-blue order; on the M3 Max's MoltenVK, which stores each format
/// itself, there are none, so no module there converts or notes a conversion.
#[test]
fn a_heap_converts_by_class_and_red_blue_order() {
    let key = |pf| {
        (
            crate::metal::texel_format::heap_class_index(
                pf,
                crate::metal::TextureSwizzleChannels::DEFAULT,
            ),
            crate::metal::format::red_blue_exchanged(pf),
        )
    };
    for (stem, p) in committed_profiles().expect("profiles") {
        let got: BTreeMap<(u32, bool), StorageImageClass> = storage_conversions(&p)
            .into_iter()
            .map(|c| ((c.index, c.red_blue), c.class))
            .collect();
        match stem.as_str() {
            "lavapipe" => {
                assert_eq!(
                    got.get(&key(PixelFormat::RGBA8UnormSRGB)),
                    Some(&StorageImageClass::SrgbEncode)
                );
                assert_eq!(
                    got.get(&key(PixelFormat::BGRA8UnormSRGB)),
                    Some(&StorageImageClass::SrgbEncode)
                );
                assert_eq!(
                    got.get(&key(PixelFormat::BGR10A2Unorm)),
                    Some(&StorageImageClass::SwizzleBgra)
                );
                for pf in [
                    PixelFormat::RGBA8Unorm,
                    PixelFormat::BGRA8Unorm,
                    PixelFormat::RGB10A2Unorm,
                ] {
                    assert_eq!(got.get(&key(pf)), None, "{pf:?} is stored as it is");
                }
                assert_ne!(
                    key(PixelFormat::BGR10A2Unorm),
                    key(PixelFormat::RGB10A2Unorm)
                );
            }
            "moltenvk-m3" => assert!(got.is_empty(), "{got:?}"),
            _ => {}
        }
    }
}

/// A texture buffer fits only a texel-buffer class: the one of its numeric type holds it, where its
/// usage has the class's access, and one of another numeric type fits it in everything but that;
/// no image class fits it.
#[test]
fn a_texture_buffer_is_held_by_the_texel_buffer_class_of_its_numeric_type() {
    use ImageAccess::{Sampled, Storage};
    use SampledType::{Float, Uint};
    let fb = class(ImageDim::Buffer, false, false, Float, Sampled);
    let ub = class(ImageDim::Buffer, false, false, Uint, Sampled);
    let fsb = class(ImageDim::Buffer, false, false, Float, Storage);
    let f2d = class(ImageDim::D2, false, false, Float, Sampled);
    let rgba = PixelFormat::RGBA32Float;
    let read = shape(TextureType::TextureBuffer, rgba, 1, READ);
    let write = shape(TextureType::TextureBuffer, rgba, 1, WRITE);
    assert_eq!(fit(&read, &fb), Fit::Holds);
    assert_eq!(fit(&read, &ub), Fit::NumericType);
    assert_eq!(fit(&read, &f2d), Fit::No);
    assert_eq!(fit(&read, &fsb), Fit::No);
    assert_eq!(fit(&write, &fsb), Fit::Holds);
    assert_eq!(fit(&write, &fb), Fit::No);
    assert_eq!(
        fit(&shape(TextureType::Type2D, rgba, 1, READ), &fb),
        Fit::No
    );
}
