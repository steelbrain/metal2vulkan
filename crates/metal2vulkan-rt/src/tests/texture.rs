//! Textures: their shapes, their indices, their views' component mappings, and on devices their
//! bytes.
//!
//! - A descriptor Metal stops is refused, each rule by its own refusal: a zero extent, a height
//!   or depth the type does not take, a cube that is not square, an array length or sample count
//!   the type does not take, and levels past the full chain (one for 1D and multisample types).
//! - An index has one owner, the object placed there last, in each class: a placement where
//!   another object holds the index displaces it, and only the owner's vacating frees the index;
//!   placing at one not granted is `Refusal::IndicesNotGranted`, and a grant is not released while
//!   an object holds one of its indices (`Refusal::IndicesInUse`); vacated, it is placed at and
//!   released again.
//! - A view's components are Metal's swizzle over the format's own mapping: zero and one stay,
//!   and a channel source reads through the format's mapping. A view's swizzle combines with its
//!   parent's as Metal's does: each channel it names is the parent's as the parent's swizzle
//!   reads it.
//! - The linear row rounding starts at the texel's bytes: rows laid at the row itself answer the
//!   texel's bytes, rows padded to a power of two answer it, and rows padded otherwise answer none.
//! - Over every venue's linear layout, a render target over a buffer stays in place only where the
//!   subresource the device advertises ends within its own rows and the buffer; a sampled texture
//!   keeps the bound of its texels.
//! - Under a device cfg, on every device of the cfg's kind: bytes written with `replace_region`
//!   read back with `get_bytes`, the same, for a level of a 2D texture, rows apart at a pitch with
//!   padding, a 3D region two images deep, a slice of an array, a cube's face, a depth texture,
//!   and a block-compressed texture where the device makes one; padding in the caller's bytes is
//!   left alone. A Private texture's bytes are refused. A texture's ID is its index and its
//!   format's word. An index is taken while its texture lives and free once it is gone, and its
//!   grant is released only then. The gauges of textures, descriptor slots and blocks come back to
//!   zero with the device, and the copies are counted in `bytes_copied`.
//! - Under a device cfg, `replace_region` reads its bytes in place through `HostBytes::from_raw`,
//!   from memory no slice covers, and copies the region's bytes once: `bytes_copied` grows by the
//!   region's rows alone, not the padding between them, and the texture reads them back.

use crate::arrays::Grants;
use crate::error::{Error, Refusal};
use crate::texture::{combined, components, linear_rounding, shape};
use crate::{IndexClass, PixelFormat, TextureDescriptor, TextureType};
use ash::vk;
use metal2vulkan::metal::{TextureSwizzle, TextureSwizzleChannels};

/// The refusal of `result`.
fn refused<T>(result: Result<T, Error>) -> Refusal {
    match result {
        Err(Error::Refused(refusal)) => refusal,
        Err(other) => panic!("not a refusal: {other}"),
        Ok(_) => panic!("not a refusal: it was made"),
    }
}

#[test]
fn each_shape_metal_stops_is_refused() {
    let base = TextureDescriptor::new(TextureType::Type2D, PixelFormat::RGBA8Unorm, 8, 8);
    let with = |f: fn(&mut TextureDescriptor)| {
        let mut d = base;
        f(&mut d);
        refused(shape(&d))
    };
    assert_eq!(with(|d| d.width = 0), Refusal::TextureZeroExtent);
    assert_eq!(with(|d| d.depth = 0), Refusal::TextureZeroExtent);
    assert_eq!(with(|d| d.depth = 2), Refusal::TextureExtentForType);
    assert_eq!(
        with(|d| {
            d.texture_type = TextureType::Type1D;
            d.height = 2;
        }),
        Refusal::TextureExtentForType
    );
    assert_eq!(
        with(|d| {
            d.texture_type = TextureType::Cube;
            d.height = 4;
        }),
        Refusal::TextureCubeNotSquare
    );
    assert_eq!(
        with(|d| d.array_length = 2),
        Refusal::TextureArrayLength { array_length: 2 }
    );
    assert_eq!(
        with(|d| {
            d.texture_type = TextureType::Type2DArray;
            d.array_length = 0;
        }),
        Refusal::TextureArrayLength { array_length: 0 }
    );
    assert_eq!(
        with(|d| d.sample_count = 4),
        Refusal::TextureSampleCount { sample_count: 4 }
    );
    assert_eq!(
        with(|d| d.mipmap_level_count = 5),
        Refusal::TextureLevelCount { levels: 5, max: 4 }
    );
    assert_eq!(
        with(|d| d.mipmap_level_count = 0),
        Refusal::TextureLevelCount { levels: 0, max: 4 }
    );
    assert_eq!(
        with(|d| {
            d.texture_type = TextureType::Type2DMultisample;
            d.sample_count = 4;
            d.mipmap_level_count = 2;
        }),
        Refusal::TextureLevelCount { levels: 2, max: 1 }
    );
    // What Metal makes passes: a full chain, a cube array's six layers a cube, a 3D extent.
    let mut d = base;
    d.mipmap_level_count = 4;
    assert!(shape(&d).is_ok());
    d = base;
    d.texture_type = TextureType::CubeArray;
    d.array_length = 2;
    assert_eq!(shape(&d).map(|s| s.layers).ok(), Some(12));
    d = base;
    d.texture_type = TextureType::Type3D;
    d.depth = 16;
    d.mipmap_level_count = 5;
    assert!(shape(&d).is_ok());
}

#[test]
fn an_index_has_one_owner_the_object_placed_there_last() {
    for (class, first) in [(IndexClass::Texture, 4096), (IndexClass::Sampler, 1)] {
        let mut grants = Grants::new();
        let grant = grants.grant(class, first, 4).expect("the grant");
        let at = first + 1;
        let place = |grants: &mut Grants| grants.place(class, at).expect("a placement");
        let earlier = place(&mut grants);
        assert!(!earlier.displaced, "{class:?}");
        let later = place(&mut grants);
        assert!(later.displaced, "{class:?}");
        assert_ne!(earlier.owner, later.owner, "{class:?}");
        assert!(grants.owns(class, at, later.owner), "{class:?}");
        // The displaced object's vacating leaves the index the later one's.
        assert!(!grants.vacate(class, at, earlier.owner), "{class:?}");
        assert!(grants.owns(class, at, later.owner), "{class:?}");
        assert!(matches!(
            grants.place(class, first + 4),
            Err(Error::Refused(Refusal::IndicesNotGranted { index, .. })) if index == first + 4
        ));
        assert_eq!(
            refused(grants.release(grant)),
            Refusal::IndicesInUse { class, index: at }
        );
        assert!(grants.vacate(class, at, later.owner), "{class:?}");
        assert!(!grants.owns(class, at, later.owner), "{class:?}");
        let again = place(&mut grants);
        assert!(!again.displaced, "{class:?}");
        assert!(grants.vacate(class, at, again.owner), "{class:?}");
        assert!(grants.release(grant).is_ok(), "{class:?}");
    }
    // The classes are apart: a texture grant grants no sampler index.
    let mut grants = Grants::new();
    grants.grant(IndexClass::Texture, 1, 4).expect("the grant");
    assert_eq!(
        refused(grants.place(IndexClass::Sampler, 2)),
        Refusal::IndicesNotGranted {
            class: IndexClass::Sampler,
            index: 2
        }
    );
}

#[test]
fn a_views_components_are_the_swizzle_over_the_formats_mapping() {
    use vk::ComponentSwizzle as C;
    let mapping = |r, g, b, a| vk::ComponentMapping { r, g, b, a };
    // `ash`'s mapping has no `PartialEq`: compare its four parts.
    let parts =
        |m: Option<vk::ComponentMapping>| m.map(|m| [m.r, m.g, m.b, m.a].map(|c| c.as_raw()));
    let components = |m, swizzle| parts(components(m, swizzle));
    let identity = vk::ComponentMapping::default();
    assert_eq!(
        components(identity, TextureSwizzleChannels::DEFAULT),
        parts(Some(mapping(C::R, C::G, C::B, C::A)))
    );
    let swizzle = TextureSwizzleChannels::new(
        TextureSwizzle::Blue,
        TextureSwizzle::Zero,
        TextureSwizzle::One,
        TextureSwizzle::Red,
    );
    assert_eq!(
        components(identity, swizzle),
        parts(Some(mapping(C::B, C::ZERO, C::ONE, C::R)))
    );
    // `A8Unorm` held in red: Metal's alpha is the image's red.
    let alpha_in_red = mapping(C::ZERO, C::ZERO, C::ZERO, C::R);
    let alpha_everywhere = TextureSwizzleChannels::new(
        TextureSwizzle::Alpha,
        TextureSwizzle::Alpha,
        TextureSwizzle::Red,
        TextureSwizzle::One,
    );
    assert_eq!(
        components(alpha_in_red, alpha_everywhere),
        parts(Some(mapping(C::R, C::R, C::ZERO, C::ONE)))
    );
}

#[test]
fn a_views_swizzle_reads_its_parents_swizzled_channels() {
    use TextureSwizzle::{Alpha, Blue, Green, One, Red, Zero};
    let channels = |[red, green, blue, alpha]: [TextureSwizzle; 4]| TextureSwizzleChannels {
        red,
        green,
        blue,
        alpha,
    };
    let identity = TextureSwizzleChannels::DEFAULT;
    // Apple's example: `[R, G, A, B]` over `[R, 1, 1, G]` is `[R, 1, G, 1]`.
    let parent = channels([Red, One, One, Green]);
    let view = channels([Red, Green, Alpha, Blue]);
    assert_eq!(combined(parent, view), channels([Red, One, Green, One]));
    // The identity on either side leaves the other; zero and one stay whatever the parent's.
    assert_eq!(combined(parent, identity), parent);
    assert_eq!(combined(identity, view), view);
    let constants = channels([Zero, One, Zero, One]);
    assert_eq!(combined(parent, constants), constants);
    // Taken twice, a rotation's channels move twice.
    let rotation = channels([Green, Blue, Alpha, Red]);
    assert_eq!(
        combined(rotation, rotation),
        channels([Blue, Alpha, Red, Green])
    );
}

#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod on_device {
    use super::refused;
    use crate::tests::devices_of_the_cfgs_kind;
    use crate::{
        Device, DeviceOptions, Error, Gap, IndexClass, PixelFormat, Refusal, Region, StorageMode,
        Texture, TextureDescriptor, TextureType, TextureUsage,
    };
    use metal2vulkan::metal::resource_word::TextureWord;
    use metal2vulkan::metal::TextureSwizzleChannels;

    /// A byte pattern of `len` bytes, different at every offset of one call and between calls.
    fn pattern(len: usize, seed: u8) -> Vec<u8> {
        (0..len)
            .map(|i| (i as u8).wrapping_mul(31).wrapping_add(seed))
            .collect()
    }

    /// One round trip: `bytes_per_row` and `bytes_per_image` apart, the region's rows of `row`
    /// bytes, `rows` of them in each of `images`.
    struct Trip {
        region: Region,
        level: u32,
        slice: u32,
        row: usize,
        rows: usize,
        images: usize,
        bytes_per_row: usize,
        bytes_per_image: usize,
    }

    impl Trip {
        fn len(&self) -> usize {
            (self.images - 1) * self.bytes_per_image
                + (self.rows - 1) * self.bytes_per_row
                + self.row
        }

        /// Whether `offset` is a byte of the region's rows.
        fn holds(&self, offset: usize) -> bool {
            let image = offset / self.bytes_per_image.max(1);
            let within = offset - image * self.bytes_per_image;
            let row = within / self.bytes_per_row.max(1);
            image < self.images && row < self.rows && within - row * self.bytes_per_row < self.row
        }

        /// Writes the pattern, reads it back into bytes of another pattern, and checks the
        /// region's bytes came back and the padding is untouched.
        fn run(&self, name: &str, texture: &Texture, seed: u8) {
            let len = self.len();
            let written = pattern(len, seed);
            texture
                .replace_region(
                    self.region,
                    self.level,
                    self.slice,
                    crate::HostBytes::from(&written[..]),
                    self.bytes_per_row as u64,
                    self.bytes_per_image as u64,
                )
                .unwrap_or_else(|error| panic!("{name}: replace_region: {error}"));
            let mut read = pattern(len, seed.wrapping_add(7));
            let before = read.clone();
            texture
                .get_bytes(
                    &mut read,
                    self.bytes_per_row as u64,
                    self.bytes_per_image as u64,
                    self.region,
                    self.level,
                    self.slice,
                )
                .unwrap_or_else(|error| panic!("{name}: get_bytes: {error}"));
            for offset in 0..len {
                let want = match self.holds(offset) {
                    true => written[offset],
                    false => before[offset],
                };
                assert_eq!(
                    read[offset],
                    want,
                    "{name}: {:?} byte {offset}",
                    texture.descriptor().pixel_format
                );
            }
        }
    }

    fn descriptor(
        texture_type: TextureType,
        pixel_format: PixelFormat,
        extent: [u32; 3],
    ) -> TextureDescriptor {
        let mut d = TextureDescriptor::new(texture_type, pixel_format, extent[0], extent[1]);
        d.depth = extent[2];
        d
    }

    /// Bytes go in and come back on every shape, the IDs and indices follow their textures, and
    /// every gauge comes back to zero.
    #[test]
    fn every_device_round_trips_texture_bytes() {
        let instance = crate::tests::validation::instance(&[]);
        let devices = devices_of_the_cfgs_kind(&instance);
        assert!(
            !devices.is_empty(),
            "the loader lists no device of this cfg's kind"
        );
        for (name, physical) in devices {
            let device = Device::new(&physical, DeviceOptions::default())
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let counters = device.counters();
            let sampled = device
                .grant_indices(IndexClass::Texture, 4096, 16)
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let storage = device
                .grant_indices(IndexClass::Texture, 1, 16)
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let make = |d: &TextureDescriptor, index| {
                device
                    .new_texture(d, index)
                    .unwrap_or_else(|error| panic!("{name}: {d:?}: {error}"))
            };

            // A level of a 2D texture, rows at a pitch with padding.
            let mut d = descriptor(TextureType::Type2D, PixelFormat::RGBA8Unorm, [13, 7, 1]);
            d.mipmap_level_count = 3;
            let rgba = make(&d, Some(4096));
            assert_eq!(rgba.gpu_resource_id().index, 4096);
            assert_eq!(
                Some(TextureWord(rgba.gpu_resource_id().word)),
                TextureWord::of(PixelFormat::RGBA8Unorm, TextureSwizzleChannels::DEFAULT)
            );
            Trip {
                region: Region {
                    origin: [1, 0, 0],
                    size: [5, 3, 1],
                },
                level: 1,
                slice: 0,
                row: 20,
                rows: 3,
                images: 1,
                bytes_per_row: 28,
                bytes_per_image: 84,
            }
            .run(&name, &rgba, 1);

            // A storage-capable half-float texture, Managed.
            let mut d = descriptor(TextureType::Type2D, PixelFormat::RGBA16Float, [9, 5, 1]);
            d.usage = TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE;
            d.storage_mode = StorageMode::Managed;
            let half = make(&d, Some(1));
            Trip {
                region: Region {
                    origin: [0, 0, 0],
                    size: [9, 5, 1],
                },
                level: 0,
                slice: 0,
                row: 72,
                rows: 5,
                images: 1,
                bytes_per_row: 72,
                bytes_per_image: 360,
            }
            .run(&name, &half, 2);

            // A 3D region two images deep.
            let d = descriptor(TextureType::Type3D, PixelFormat::R32Float, [4, 4, 3]);
            let volume = make(&d, Some(4097));
            Trip {
                region: Region {
                    origin: [1, 1, 1],
                    size: [3, 2, 2],
                },
                level: 0,
                slice: 0,
                row: 12,
                rows: 2,
                images: 2,
                bytes_per_row: 16,
                bytes_per_image: 40,
            }
            .run(&name, &volume, 3);

            // A slice of an array, and a cube's last face.
            let mut d = descriptor(TextureType::Type2DArray, PixelFormat::R8Unorm, [4, 4, 1]);
            d.array_length = 3;
            let array = make(&d, Some(4098));
            let face = Region {
                origin: [0, 0, 0],
                size: [4, 4, 1],
            };
            let trip = |slice| Trip {
                region: face,
                level: 0,
                slice,
                row: 4,
                rows: 4,
                images: 1,
                bytes_per_row: 4,
                bytes_per_image: 16,
            };
            trip(2).run(&name, &array, 4);
            let d = descriptor(TextureType::Cube, PixelFormat::R8Unorm, [4, 4, 1]);
            let cube = make(&d, Some(4099));
            trip(5).run(&name, &cube, 5);

            // A 32-bit float depth texture: its bytes come out, and none go in.
            let d = descriptor(TextureType::Type2D, PixelFormat::Depth32Float, [8, 8, 1]);
            let depth = make(&d, Some(4100));
            let whole = Region {
                origin: [0; 3],
                size: [8, 8, 1],
            };
            let mut read = vec![0u8; 256];
            depth
                .get_bytes(&mut read, 32, 256, whole, 0, 0)
                .unwrap_or_else(|error| panic!("{name}: depth get_bytes: {error}"));
            assert!(
                matches!(
                    depth.replace_region(whole, 0, 0, crate::HostBytes::from(&read[..]), 32, 256),
                    Err(Error::Unsupported(Gap::FloatDepthUpload {
                        pixel_format: PixelFormat::Depth32Float
                    }))
                ),
                "{name}"
            );

            // A block-compressed texture, where the device makes one: rows of 4 x 4 blocks.
            let d = descriptor(TextureType::Type2D, PixelFormat::BC1RGBA, [8, 8, 1]);
            match device.new_texture(&d, Some(4101)) {
                Ok(bc1) => Trip {
                    region: Region {
                        origin: [4, 0, 0],
                        size: [4, 8, 1],
                    },
                    level: 0,
                    slice: 0,
                    row: 8,
                    rows: 2,
                    images: 1,
                    bytes_per_row: 8,
                    bytes_per_image: 16,
                }
                .run(&name, &bc1, 7),
                Err(Error::Unsupported(Gap::TextureFormat { .. })) => {
                    println!("{name}: no BC1 texture on this device")
                }
                Err(error) => panic!("{name}: BC1: {error}"),
            }

            // A cube array texture is not built: its view would need `imageCubeArray`.
            let mut cubes = descriptor(TextureType::CubeArray, PixelFormat::RGBA8Unorm, [4, 4, 1]);
            cubes.array_length = 2;
            assert!(
                matches!(
                    device.new_texture(&cubes, Some(4103)),
                    Err(Error::Unsupported(Gap::CubeArrayTexture))
                ),
                "{name}: a cube array texture"
            );

            // A Private texture's bytes are the GPU's alone.
            let mut d = descriptor(TextureType::Type2D, PixelFormat::RGBA8Unorm, [4, 4, 1]);
            d.storage_mode = StorageMode::Private;
            let private = make(&d, Some(4102));
            assert_eq!(
                refused(private.replace_region(
                    face,
                    0,
                    0,
                    crate::HostBytes::from(&[0; 64][..]),
                    16,
                    64
                )),
                Refusal::TextureBytesPrivate,
                "{name}"
            );

            // An index holds its texture until it is gone, and its grant waits for it.
            assert_eq!(
                refused(device.new_texture(&d, None)),
                Refusal::TextureIndex { index: None },
                "{name}"
            );
            assert_eq!(
                refused(device.release_indices(sampled)),
                Refusal::IndicesInUse {
                    class: IndexClass::Texture,
                    index: 4096
                },
                "{name}"
            );
            // Seven live at their indices; a dropped texture's image may wait for its upload.
            let gauges = counters.gauges();
            assert_eq!(gauges.descriptor_slots, 7, "{name}");
            assert!(gauges.textures >= 7, "{name}: {gauges:?}");
            drop(private);
            make(&d, Some(4102));
            drop((rgba, half, volume, array, cube, depth));
            device
                .release_indices(sampled)
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            device
                .release_indices(storage)
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            assert_eq!(counters.gauges().descriptor_slots, 0, "{name}");
            assert!(counters.metrics().bytes_copied > 0, "{name}");
            drop(device);
            let gauges = counters.gauges();
            assert_eq!(
                (gauges.textures, gauges.descriptor_slots, gauges.blocks),
                (0, 0, 0),
                "{name}"
            );
        }
    }

    /// `replace_region` reads bytes given by a pointer and a length, which the test writes and
    /// never covers with a slice, and copies the region's bytes once, at the call.
    #[test]
    fn every_device_reads_replace_region_bytes_in_place_once() {
        use crate::tests::command::on_device::each_device;
        use crate::HostBytes;
        use std::alloc::Layout;
        // Five texels of four bytes in each of three rows, the rows 32 bytes apart.
        const ROW: usize = 32;
        const USED: usize = 20;
        const ROWS: usize = 3;
        const LEN: usize = (ROWS - 1) * ROW + USED;
        each_device("host_bytes", |name, _, device| {
            let mut d = descriptor(TextureType::Type2D, PixelFormat::RGBA8Unorm, [8, 4, 1]);
            d.storage_mode = StorageMode::Shared;
            // No shader usage, so no index: a shader-read texture takes one.
            d.usage = TextureUsage::empty();
            let texture =
                (device.new_texture(&d, None)).unwrap_or_else(|error| panic!("{name}: {error}"));
            let region = Region {
                origin: [1, 1, 0],
                size: [5, 3, 1],
            };
            let layout = Layout::from_size_align(LEN, 1).expect("a layout");
            // SAFETY: the layout's size is not zero.
            let start = unsafe { std::alloc::alloc(layout) };
            assert!(!start.is_null(), "{name}: an allocation");
            let byte = |at: usize| (at as u8).wrapping_mul(13).wrapping_add(5);
            for at in 0..LEN {
                // SAFETY: `at` is inside the allocation.
                unsafe { start.add(at).write(byte(at)) };
            }
            let counters = device.counters();
            let before = counters.metrics().bytes_copied;
            // SAFETY: the test's allocation, readable for `LEN` bytes until it is freed below,
            // after the call.
            let bytes = unsafe { HostBytes::from_raw(start, LEN) };
            (texture.replace_region(region, 0, 0, bytes, ROW as u64, 0))
                .unwrap_or_else(|error| panic!("{name}: replace_region: {error}"));
            let copied = counters.metrics().bytes_copied - before;
            // SAFETY: allocated above with `layout`; nothing reads it any more.
            unsafe { std::alloc::dealloc(start, layout) };
            assert_eq!(
                copied,
                (ROWS * USED) as u64,
                "{name}: the region's bytes, once"
            );
            let mut read = vec![0u8; LEN];
            (texture.get_bytes(&mut read, ROW as u64, 0, region, 0, 0))
                .unwrap_or_else(|error| panic!("{name}: get_bytes: {error}"));
            for row in 0..ROWS {
                let start = row * ROW;
                for (at, &got) in (start..).zip(&read[start..start + USED]) {
                    assert_eq!(got, byte(at), "{name}: byte {at}");
                }
            }
            println!("host_bytes: {name}: {copied} bytes copied of the caller's {LEN}");
        });
    }
}

/// The linear row rounding of a format of `texel` bytes over the minimum linear texture
/// alignment's widths, where the device lays a row of `row` bytes at `pitch(row)`.
fn rounding(texel: u64, pitch: impl Fn(u64) -> u64) -> Option<u64> {
    let rows: Vec<(u64, u64)> = [1, 3, 17, 64, 65, 100, 129, 1000, 1920]
        .map(|width| {
            let row = width * texel;
            (row, pitch(row))
        })
        .to_vec();
    linear_rounding(&rows, texel)
}

/// A device that lays a linear row at the row itself (ANV, Mesa 26.2.6, on the Iris Xe of RPL-P:
/// every format and width the over test asks) answers each format's texel bytes, below 16 for all
/// but the 16-byte texels; one that pads rows to a power of two (RADV on Polaris: 256 bytes, 512
/// for 8-byte texels) answers that power; one whose rows have a floor besides their rounding
/// answers none, which the device then reports as 16.
#[test]
fn the_linear_row_rounding_starts_at_the_texel() {
    for texel in [1, 2, 4, 8, 16] {
        assert_eq!(rounding(texel, |row| row), Some(texel), "{texel}");
    }
    assert_eq!(rounding(4, |row| row.next_multiple_of(256)), Some(256));
    assert_eq!(rounding(8, |row| row.next_multiple_of(512)), Some(512));
    assert_eq!(rounding(4, |row| row.max(64)), None);
}

/// A device's linear image of a 2D color format: the pitch it lays a row of `width` texels of
/// `texel` bytes at, and the bytes from the first texel its subresource advertises
/// (`VkSubresourceLayout::size`) for `height` rows.
struct Venue {
    name: &'static str,
    pitch: fn(width: u64, texel: u64) -> u64,
    size: fn(height: u64, pitch: u64, row: u64) -> u64,
}

/// The venues' layouts, as each answered `vkGetImageSubresourceLayout` for linear images over the
/// formats and extents the over tests make: rows to the row rounded up to the device's answer,
/// and a subresource that ends at the rows' end, but on ANV, which lays a row at the row itself
/// and ends at the last texel, and llvmpipe, which lays a row of a width rounded up to 4 texels
/// to a multiple of 64 bytes and advertises a height rounded up to 4 rows.
const VENUES: [Venue; 7] = [
    Venue {
        name: "RADV (780M, XTX)",
        pitch: |width, texel| (width * texel).next_multiple_of(256),
        size: |height, pitch, _| height * pitch,
    },
    Venue {
        name: "RADV (RX 550)",
        pitch: |width, texel| (width * texel).next_multiple_of(64 * texel.max(4)),
        size: |height, pitch, _| height * pitch,
    },
    Venue {
        name: "NVIDIA",
        pitch: |width, texel| (width * texel).next_multiple_of(128),
        size: |height, pitch, _| height * pitch,
    },
    Venue {
        name: "ANV (UHD 630)",
        pitch: |width, texel| (width * texel).next_multiple_of(if texel == 8 { 128 } else { 64 }),
        size: |height, pitch, _| height * pitch,
    },
    Venue {
        name: "ANV (Iris Xe)",
        pitch: |width, texel| width * texel,
        size: |height, pitch, row| (height - 1) * pitch + row,
    },
    Venue {
        name: "KosmicKrisp (M3)",
        pitch: |width, texel| (width * texel).next_multiple_of(16),
        size: |height, pitch, _| height * pitch,
    },
    Venue {
        name: "llvmpipe",
        pitch: |width, texel| (width.next_multiple_of(4) * texel).next_multiple_of(64),
        size: |height, pitch, _| height.next_multiple_of(4) * pitch,
    },
];

/// A render target over a buffer is in place only where the subresource the device advertises
/// ends within its own rows and the buffer (an attachment's store, clear or resolve may write
/// back any byte of it); a texture of any other usage keeps the bound of its texels. Over every
/// venue's layout, at the device's own pitch, for the render targets over buffers the recorded
/// streams make (texel bytes, width, height) and heights of 3 and 4:
/// - in a buffer that ends at the rows' end, as every one of those streams' does, a render target
///   stays in place on every venue but llvmpipe, which copies those whose height is not a
///   multiple of 4, as it does in a buffer longer than its rounded rows;
/// - in a buffer that ends at the last texel, one whose pitch is wider than its row is copied, and
///   one whose rows have no padding stays in place;
/// - a sampled texture stays in place on every venue in each.
#[test]
fn a_render_target_in_place_ends_within_its_rows_and_its_buffer() {
    use crate::texture::{in_reach, render_reach};
    const SHAPES: [(u64, u64, u64); 12] = [
        (4, 130, 130),
        (4, 128, 128),
        (1, 128, 128),
        (1, 48, 48),
        (4, 50, 50),
        (4, 34, 34),
        (1, 32, 32),
        (4, 48, 48),
        (4, 258, 258),
        (1, 256, 256),
        (4, 16, 3),
        (4, 16, 4),
    ];
    for venue in &VENUES {
        for (texel, width, height) in SHAPES {
            let pitch = (venue.pitch)(width, texel);
            let row = width * texel;
            let size = (venue.size)(height, pitch, row);
            let rows = height * pitch;
            let llvmpipe = venue.name == "llvmpipe";
            let shape = format!("{}: {texel}-byte texels, {width} x {height}", venue.name);
            // A buffer that ends at the rows' end, and one longer than the rounded rows.
            for rest in [rows, height.next_multiple_of(4) * pitch + 4096] {
                let render = in_reach(size, Some(render_reach(rows, rest)));
                assert_eq!(
                    render,
                    !llvmpipe || height.is_multiple_of(4),
                    "{shape}, {rest} bytes"
                );
                assert!(in_reach(size, None), "{shape}, {rest} bytes");
            }
            // A buffer that ends at the last texel.
            let rest = (height - 1) * pitch + row;
            let render = in_reach(size, Some(render_reach(rows, rest)));
            assert_eq!(
                render,
                pitch == row && (!llvmpipe || height.is_multiple_of(4)),
                "{shape}"
            );
            assert!(in_reach(size, None), "{shape}");
        }
    }
}
