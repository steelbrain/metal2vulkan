//! The Direct3D 11 surface import, a Windows host's, under a device cfg. On every device of the
//! cfg's kind, its typed refusals: no plane, five planes, and a plane whose image format's texel
//! is not its descriptor's. Under `m2v_gpu`, on every device: a picture's two planes made as shared
//! Direct3D 11 textures on the DXGI adapter of the device's LUID (luma `R8_UNORM` at 64x32, chroma
//! `R16_UINT` at 32x16, which rt's texture views as `RG8Unorm`), imported by their NT handles
//! with no submission. Direct3D 11 draws picture 1 (one triangle a plane, its pixel shader
//! computing each texel's bytes, then an event query waited for), rt acquires the planes
//! (`external_writes_done`) and blits them into a Shared buffer, whose bytes are picture 1's
//! exactly. Direct3D 11 draws picture 2, each of whose bytes differs from picture 1's, and rt's
//! next read is picture 2's exactly and matches picture 1 at no byte: the negative control, which
//! a copy taken at the import, or a read of bytes the second draw did not reach, fails. Then rt
//! writes picture 3 into the planes by a blit and releases them, and Direct3D 11 reads picture 3
//! exactly through staging copies (`CopyResource`, `Map`). Each of rt's three submissions keeps
//! the planes, and each counts once in `Metrics::d3d11_plane_submissions`.
//! While the surface lives, the surface imports gauge holds its two planes; it has no buffer and
//! no export handle. The same planes made with `PixelFormatView` usage take a swizzled view in
//! another format of their class (`R8Uint` of the luma, `RG8Uint` of the chroma) exactly where
//! the import query takes the list of every format the plane's texture views as, else the view
//! is `Gap::TextureView` of `SurfacePixelFormat`; the query's answers are printed.

#[cfg(all(windows, any(m2v_gpu, m2v_cpu_icd)))]
mod on_device {
    use crate::tests::command::on_device::each_device;
    use crate::{
        D3d11Plane, Error, ImportHandle, Limit, PixelFormat, Refusal, TextureDescriptor,
        TextureType,
    };
    use std::os::windows::io::AsHandle;

    #[test]
    fn every_device_refuses_a_d3d11_surface_typed() {
        each_device("import_d3d11_surface_refusals", |name, _, device| {
            // A handle the calls borrow and never read: the test's own executable's.
            let file = std::env::current_exe()
                .and_then(std::fs::File::open)
                .expect("a file");
            let luma = D3d11Plane {
                handle: file.as_handle(),
                image_format: PixelFormat::R8Unorm,
                descriptor: TextureDescriptor::new(
                    TextureType::Type2D,
                    PixelFormat::R8Unorm,
                    64,
                    32,
                ),
                index: None,
            };
            for count in [0, 5] {
                let planes = vec![luma; count];
                let error = device.import_d3d11_surface(&planes).err();
                assert!(
                    matches!(
                        error,
                        Some(Error::Refused(Refusal::SurfacePlanes {
                            planes,
                            pitch_multiple: 1,
                            start_multiple: 1,
                        })) if planes == count
                    ),
                    "{name}: {count} planes: {error:?}"
                );
            }
            let mut wide = luma;
            wide.descriptor.pixel_format = PixelFormat::RG8Unorm;
            let error = device.import_d3d11_surface(&[wide]).err();
            let takes = device.core().exports.win32().is_some();
            if takes {
                assert!(
                    matches!(
                        error,
                        Some(Error::Refused(Refusal::TextureViewFormat {
                            texture: PixelFormat::R8Unorm,
                            view: PixelFormat::RG8Unorm,
                        }))
                    ),
                    "{name}: a two-byte view of a one-byte texture: {error:?}"
                );
            } else {
                assert!(
                    matches!(
                        error,
                        Some(Error::Limit(Limit::NoImport {
                            handle: ImportHandle::D3d11Texture,
                        }))
                    ),
                    "{name}: no VK_KHR_external_memory_win32: {error:?}"
                );
            }
            println!(
                "d3d11: {name}: the import's refusals typed; the device takes Direct3D 11 handles: \
                 {takes}"
            );
        });
    }

    /// A picture's two planes as Direct3D 11 textures, written and read by Direct3D 11 and by rt.
    #[cfg(m2v_gpu)]
    mod picture {
        use super::d3d::{self, D3d, SharedTexture};
        use crate::d3d11::{import_query, plane_list};
        use crate::tests::command::on_device::{each_device, finished, within};
        use crate::texture::TextureObject;
        use crate::{
            BufferLayout, D3d11Plane, Device, Error, Gap, IndexClass, MemoryClass, PixelFormat,
            Surface, TextureDescriptor, TexturePlace, TextureSwizzle, TextureSwizzleChannels,
            TextureType, TextureUsage, TextureViewKind,
        };
        use ash::vk;
        use std::os::windows::io::AsHandle;

        /// A plane: its width and height, its texel's bytes, rt's format and the Direct3D 11
        /// texture's, in rt's terms and in DXGI's.
        struct Plane {
            width: u32,
            height: u32,
            texel: u32,
            view: PixelFormat,
            image: PixelFormat,
            dxgi: u32,
        }

        const PLANES: [Plane; 2] = [
            Plane {
                width: 64,
                height: 32,
                texel: 1,
                view: PixelFormat::R8Unorm,
                image: PixelFormat::R8Unorm,
                dxgi: d3d::DXGI_FORMAT_R8_UNORM,
            },
            Plane {
                width: 32,
                height: 16,
                texel: 2,
                view: PixelFormat::RG8Unorm,
                image: PixelFormat::R16Uint,
                dxgi: d3d::DXGI_FORMAT_R16_UINT,
            },
        ];

        impl Plane {
            fn row(&self) -> u32 {
                self.width * self.texel
            }

            fn bytes(&self) -> usize {
                (self.row() * self.height) as usize
            }
        }

        /// Each plane's offset in a picture, the planes packed at their own rows.
        fn offsets() -> [usize; 2] {
            [0, PLANES[0].bytes()]
        }

        /// What picture `n`'s bytes are XORed with: each picture's every byte unlike the others'.
        fn mask(n: u32) -> u8 {
            [0x00, 0xff, 0x5a][n as usize - 1]
        }

        /// The two planes' bytes, packed: picture `n`'s pattern, byte `b` of row `y` of plane `i`
        /// being `(b * 7 + y * 13 + i * 101 + 1) mod 256`, XORed with [`mask`].
        fn picture(n: u32) -> Vec<u8> {
            let mut out = Vec::with_capacity(PLANES.iter().map(Plane::bytes).sum());
            for (i, plane) in PLANES.iter().enumerate() {
                for y in 0..plane.height {
                    for b in 0..plane.row() {
                        out.push((b * 7 + y * 13 + i as u32 * 101 + 1) as u8 ^ mask(n));
                    }
                }
            }
            out
        }

        /// The pixel shader that draws plane `i` of picture `n`, [`picture`]'s bytes: the luma's
        /// one byte a texel as that byte over 255, which its `UNORM` format stores as the byte,
        /// and the chroma's two bytes a texel packed into its `UINT` value, the first low.
        fn pixel_shader(n: u32, i: usize) -> String {
            let byte = format!(
                "uint byte_at(uint b, uint y) {{ return ((b * 7 + y * 13 + {}) & 255) ^ {}; }}\n",
                i * 101 + 1,
                mask(n)
            );
            let main = match PLANES[i].texel {
                1 => {
                    "float main(float4 p : SV_Position) : SV_Target {
    uint2 q = uint2(p.xy);
    return byte_at(q.x, q.y) / 255.0;
}"
                }
                _ => {
                    "uint main(float4 p : SV_Position) : SV_Target {
    uint2 q = uint2(p.xy);
    return byte_at(2 * q.x, q.y) | (byte_at(2 * q.x + 1, q.y) << 8);
}"
                }
            };
            byte + main
        }

        /// The first byte at which `got` differs from `want`, and how many bytes agree.
        fn compare(want: &[u8], got: &[u8]) -> (Option<usize>, usize) {
            assert_eq!(want.len(), got.len());
            let first = (want.iter().zip(got)).position(|(a, b)| a != b);
            let same = (want.iter().zip(got)).filter(|(a, b)| a == b).count();
            (first, same)
        }

        /// The device's LUID (`VkPhysicalDeviceIDProperties`), where it is valid.
        fn luid(device: &Device) -> Option<[u8; 8]> {
            let raw = device.raw();
            let mut id = vk::PhysicalDeviceIDProperties::default();
            let mut properties = vk::PhysicalDeviceProperties2::default().push_next(&mut id);
            // SAFETY: the physical device is the instance's.
            unsafe {
                (raw.instance).get_physical_device_properties2(raw.physical_device, &mut properties)
            };
            (id.device_luid_valid == vk::TRUE).then_some(id.device_luid)
        }

        /// rt reads both planes into one Shared buffer, packed; their bytes.
        fn rt_read(name: &str, device: &Device, surface: &Surface) -> Vec<u8> {
            let len = PLANES.iter().map(Plane::bytes).sum::<usize>() as u64;
            let block = (device.take_block(MemoryClass::Shared, len)).expect("a block");
            let out = (device.new_buffer(block.id, 0, len, None)).expect("a buffer");
            let queue = device.new_command_queue(1).expect("a queue");
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let mut encoder = command_buffer.blit_command_encoder().expect("an encoder");
            for ((texture, plane), offset) in surface.planes().iter().zip(&PLANES).zip(offsets()) {
                let layout = BufferLayout {
                    offset: offset as u64,
                    bytes_per_row: u64::from(plane.row()),
                    bytes_per_image: 0,
                };
                let size = [plane.width, plane.height, 1];
                (encoder.copy_from_texture_to_buffer(
                    texture,
                    TexturePlace::default(),
                    size,
                    &out,
                    layout,
                ))
                .unwrap_or_else(|error| panic!("{name}: the read: {error}"));
            }
            encoder.end_encoding();
            command_buffer.commit().expect("a commit");
            finished(name, &command_buffer);
            assert_eq!(command_buffer.error(), None, "{name}");
            let contents = out.contents().expect("Shared");
            // SAFETY: the buffer's bytes, which no pending command reaches.
            unsafe { std::slice::from_raw_parts(contents.as_ptr(), len as usize) }.to_vec()
        }

        /// rt writes `bytes` (packed) into both planes and releases them to Direct3D 11, and
        /// waits for the command buffer.
        fn rt_write(name: &str, device: &Device, surface: &Surface, bytes: &[u8]) {
            let len = bytes.len() as u64;
            let block = (device.take_block(MemoryClass::Shared, len)).expect("a block");
            let from = (device.new_buffer(block.id, 0, len, None)).expect("a buffer");
            let contents = from.contents().expect("Shared");
            // SAFETY: the buffer's bytes, which no command reaches yet and nothing else borrows.
            unsafe { std::slice::from_raw_parts_mut(contents.as_ptr(), bytes.len()) }
                .copy_from_slice(bytes);
            let queue = device.new_command_queue(1).expect("a queue");
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let mut encoder = command_buffer.blit_command_encoder().expect("an encoder");
            for ((texture, plane), offset) in surface.planes().iter().zip(&PLANES).zip(offsets()) {
                let layout = BufferLayout {
                    offset: offset as u64,
                    bytes_per_row: u64::from(plane.row()),
                    bytes_per_image: 0,
                };
                let size = [plane.width, plane.height, 1];
                (encoder.copy_from_buffer(&from, layout, size, texture, TexturePlace::default()))
                    .unwrap_or_else(|error| panic!("{name}: the write: {error}"));
            }
            (encoder.release_surface_to_external(surface))
                .unwrap_or_else(|error| panic!("{name}: the release: {error}"));
            encoder.end_encoding();
            command_buffer.commit().expect("a commit");
            finished(name, &command_buffer);
            assert_eq!(command_buffer.error(), None, "{name}");
        }

        /// Direct3D 11 draws picture `n` into the textures, one triangle over each, and waits for
        /// its work.
        fn d3d_draw(d3d: &D3d, textures: &[SharedTexture; 2], n: u32) {
            for (i, (texture, plane)) in textures.iter().zip(&PLANES).enumerate() {
                d3d.draw(texture, (plane.width, plane.height), &pixel_shader(n, i));
            }
            d3d.finish();
        }

        /// Direct3D 11 reads both textures, packed, through staging copies.
        fn d3d_read(d3d: &D3d, textures: &[SharedTexture; 2]) -> Vec<u8> {
            let mut out = Vec::new();
            for (texture, plane) in textures.iter().zip(&PLANES) {
                out.extend(d3d.read(texture, plane.width, plane.height, plane.dxgi, plane.row()));
            }
            out
        }

        #[test]
        fn every_device_reads_and_writes_direct3d_11_planes_in_place() {
            each_device("import_d3d11_surface", |name, _, device| {
                let luid = luid(&device).unwrap_or_else(|| panic!("{name}: no valid LUID"));
                let d3d = D3d::on_luid(luid)
                    .unwrap_or_else(|| panic!("{name}: no DXGI adapter of LUID {luid:02x?}"));
                let textures = PLANES
                    .each_ref()
                    .map(|p| d3d.shared_texture(p.width, p.height, p.dxgi));
                let planes: Vec<D3d11Plane<'_>> = (textures.iter().zip(&PLANES))
                    .map(|(texture, plane)| D3d11Plane {
                        handle: texture.handle.as_handle(),
                        image_format: plane.image,
                        descriptor: TextureDescriptor::new(
                            TextureType::Type2D,
                            plane.view,
                            plane.width,
                            plane.height,
                        ),
                        index: None,
                    })
                    .collect();
                let counters = device.counters();
                let gauges = counters.gauges();
                let before = counters.metrics();
                let surface = (device.import_d3d11_surface(&planes))
                    .unwrap_or_else(|error| panic!("{name}: the import: {error}"));
                let after = counters.metrics();
                assert_eq!(
                    after.submissions, before.submissions,
                    "{name}: the import submits nothing"
                );
                assert_eq!(
                    counters.gauges().surface_imports,
                    gauges.surface_imports + 2,
                    "{name}: each plane's import counted"
                );
                assert_eq!(surface.handle(), None, "{name}: no export handle");
                assert!(
                    matches!(
                        surface.buffer(),
                        Err(Error::Unsupported(Gap::D3d11SurfaceBuffer))
                    ),
                    "{name}: no buffer"
                );
                let formats: Vec<_> = (surface.planes().iter())
                    .map(|texture| texture.descriptor().pixel_format)
                    .collect();
                assert_eq!(
                    formats,
                    [PixelFormat::R8Unorm, PixelFormat::RG8Unorm],
                    "{name}: the planes' textures are in their descriptors' formats"
                );

                // Direct3D 11 draws, rt reads.
                let (one, two, three) = (picture(1), picture(2), picture(3));
                d3d_draw(&d3d, &textures, 1);
                surface.external_writes_done().expect("the acquire");
                let read = rt_read(name, &device, &surface);
                assert_eq!(compare(&one, &read).0, None, "{name}: picture 1 read by rt");
                d3d_draw(&d3d, &textures, 2);
                surface.external_writes_done().expect("the acquire");
                let read = rt_read(name, &device, &surface);
                assert_eq!(compare(&two, &read).0, None, "{name}: picture 2 read by rt");
                // The negative control: the second read is not the first picture's at any byte.
                assert_eq!(
                    compare(&one, &read),
                    (Some(0), 0),
                    "{name}: the second read against picture 1"
                );

                // rt writes, Direct3D 11 reads.
                rt_write(name, &device, &surface, &three);
                let read = d3d_read(&d3d, &textures);
                assert_eq!(
                    compare(&three, &read).0,
                    None,
                    "{name}: picture 3 read by D3D11"
                );
                assert_eq!(
                    compare(&two, &read),
                    (Some(0), 0),
                    "{name}: D3D11's read against picture 2"
                );
                let metrics = counters.metrics();
                let kept = metrics.d3d11_plane_submissions - before.d3d11_plane_submissions;
                println!(
                    "d3d11: {name}: imported two planes (R8_UNORM 64x32, R16_UINT 32x16 viewed as \
                     RG8Unorm) with no submission; pictures 1 and 2 drawn by D3D11 read exact by \
                     rt, the second unlike the first at every byte; picture 3 written by rt read \
                     exact by D3D11; {} acquires, {} releases; {kept} of rt's 3 submissions \
                     that keep the planes counted",
                    metrics.external_acquires - before.external_acquires,
                    metrics.external_releases - before.external_releases,
                );
                assert_eq!(
                    kept, 3,
                    "{name}: each of rt's submissions that keep the planes counted"
                );
                drop(surface);
            });
        }

        /// Each plane made with `ShaderRead` and `PixelFormatView` usage, reached through its views
        /// alone, then a swizzled view of it in another format of its class (`R8Uint` of the luma,
        /// `RG8Uint` of the chroma): made, in that format, exactly where the import query takes,
        /// with the image's usage, the list of every format the plane's texture views as; else
        /// `Gap::TextureView` of `SurfacePixelFormat`. Printed for each plane: the query's answer
        /// with that list, and with each of its other formats alone beside the image's own.
        #[test]
        fn every_device_views_a_direct3d_11_plane_in_another_format_where_its_import_lists_it() {
            each_device("import_d3d11_surface_views", |name, _, device| {
                let luid = luid(&device).unwrap_or_else(|| panic!("{name}: no valid LUID"));
                let d3d = D3d::on_luid(luid)
                    .unwrap_or_else(|| panic!("{name}: no DXGI adapter of LUID {luid:02x?}"));
                let textures = PLANES
                    .each_ref()
                    .map(|p| d3d.shared_texture(p.width, p.height, p.dxgi));
                let descriptors = PLANES.each_ref().map(|p| {
                    let mut d =
                        TextureDescriptor::new(TextureType::Type2D, p.view, p.width, p.height);
                    d.usage = TextureUsage::SHADER_READ | TextureUsage::PIXEL_FORMAT_VIEW;
                    d
                });
                let planes: Vec<D3d11Plane<'_>> = (textures.iter().zip(&PLANES))
                    .zip(&descriptors)
                    .map(|((texture, plane), descriptor)| D3d11Plane {
                        handle: texture.handle.as_handle(),
                        image_format: plane.image,
                        descriptor: *descriptor,
                        index: None,
                    })
                    .collect();
                let surface = (device.import_d3d11_surface(&planes))
                    .unwrap_or_else(|error| panic!("{name}: the import: {error}"));
                let first = IndexClass::Texture.range().start;
                let _grant =
                    (device.grant_indices(IndexClass::Texture, first, 2)).expect("a grant");
                let swizzle = TextureSwizzleChannels {
                    red: TextureSwizzle::Red,
                    green: TextureSwizzle::Red,
                    blue: TextureSwizzle::Red,
                    alpha: TextureSwizzle::One,
                };
                // Each view's format, and its index.
                let views = [
                    (PixelFormat::R8Uint, first),
                    (PixelFormat::RG8Uint, first + 1),
                ];
                for ((texture, plane), (d, (view, at))) in
                    (surface.planes().iter().zip(&PLANES)).zip(descriptors.iter().zip(views))
                {
                    let plan = device.plane_plan(d).expect("the plane's plan");
                    let mut own = *d;
                    own.pixel_format = plane.image;
                    let format = device.plane_plan(&own).expect("the image's plan").format;
                    let usage = texture.object().image_usage();
                    let full = plane_list(format, plan.format, &plan.view_formats);
                    let answer =
                        |list: &[vk::Format]| import_query(device.core(), format, usage, list);
                    let takes = (answer(&full)).is_ok_and(|features| {
                        features.contains(vk::ExternalMemoryFeatureFlags::IMPORTABLE)
                    });
                    // The answer as the query gives it: the external memory features' bits, or
                    // the `VkResult`.
                    let said = |list: &[vk::Format]| match answer(list) {
                        Ok(features) => format!("features {:#x}", features.as_raw()),
                        Err(result) => format!("VkResult {}", result.as_raw()),
                    };
                    let raw = |list: &[vk::Format]| -> Vec<i32> {
                        list.iter().map(|listed| listed.as_raw()).collect()
                    };
                    let alone: Vec<String> = (full.iter().filter(|&&listed| listed != format))
                        .map(|&listed| format!("{} {}", listed.as_raw(), said(&[format, listed])))
                        .collect();
                    let made = texture.new_texture_view_with_swizzle(
                        view,
                        TextureType::Type2D,
                        0..1,
                        0..1,
                        swizzle,
                        Some(at),
                    );
                    let label = format!("{name}: {:?} as {:?}", plane.image, plane.view);
                    match takes {
                        true => {
                            let made = made.unwrap_or_else(|error| {
                                panic!("{label}: the {view:?} view, the list imported: {error}")
                            });
                            assert_eq!(made.descriptor().pixel_format, view, "{label}");
                        }
                        false => assert!(
                            matches!(
                                made.as_ref().err(),
                                Some(Error::Unsupported(Gap::TextureView {
                                    kind: TextureViewKind::SurfacePixelFormat,
                                }))
                            ),
                            "{label}: the {view:?} view, the list refused: {:?}",
                            made.err()
                        ),
                    }
                    println!(
                        "d3d11-views: {label}: image VkFormat {}, usage {:#x}; the import query \
                         with the list {:?}: {}; with each other format alone beside the image's \
                         own: {}; the swizzled {view:?} view made: {takes}",
                        format.as_raw(),
                        usage.as_raw(),
                        raw(&full),
                        said(&full),
                        alone.join(", "),
                    );
                }
            });
        }

        /// The gauges a Direct3D 11 surface counts in, which its free brings back: surface
        /// imports, their bytes, textures.
        fn import_gauges(device: &Device) -> [u64; 3] {
            let gauges = device.counters().gauges();
            [
                gauges.surface_imports,
                gauges.surface_import_bytes,
                gauges.textures,
            ]
        }

        /// A surface of Direct3D 11 planes let go of while its acquire is not yet submitted is
        /// freed with no later call, as a picture imported and acquired but never used is: 32
        /// pictures' planes, each pair imported and acquired (`external_writes_done`), dropped
        /// together, their gauges back with no other call, through one submission for the whole
        /// burst; a picture whose acquire the submission of a command buffer that read it
        /// carried, dropped once that command buffer completed, freed with no submission added;
        /// one acquired again after such a read and dropped, freed through one submission; then
        /// 200 rounds of import, acquire, a read on every other round and drop, each freed before
        /// the next, the gauges the same at rounds 100 and 200. Without the drop's send the
        /// acquires' upload holds the planes' imports, and with them the Direct3D 11 textures'
        /// memory, until a later call, and the waits for the frees time out.
        #[test]
        fn every_device_frees_direct3d_11_planes_at_the_surface_drop() {
            each_device("import_d3d11_surface_drop", |name, _, device| {
                let luid = luid(&device).unwrap_or_else(|| panic!("{name}: no valid LUID"));
                let d3d = D3d::on_luid(luid)
                    .unwrap_or_else(|| panic!("{name}: no DXGI adapter of LUID {luid:02x?}"));
                let pair = || {
                    PLANES
                        .each_ref()
                        .map(|p| d3d.shared_texture(p.width, p.height, p.dxgi))
                };
                let acquired = |textures: &[SharedTexture; 2]| -> Surface {
                    let planes: Vec<D3d11Plane<'_>> = (textures.iter().zip(&PLANES))
                        .map(|(texture, plane)| D3d11Plane {
                            handle: texture.handle.as_handle(),
                            image_format: plane.image,
                            descriptor: TextureDescriptor::new(
                                TextureType::Type2D,
                                plane.view,
                                plane.width,
                                plane.height,
                            ),
                            index: None,
                        })
                        .collect();
                    let surface = (device.import_d3d11_surface(&planes))
                        .unwrap_or_else(|error| panic!("{name}: the import: {error}"));
                    (surface.external_writes_done())
                        .unwrap_or_else(|error| panic!("{name}: the acquire: {error}"));
                    surface
                };
                let counters = device.counters();
                let base = import_gauges(&device);
                let freed = |what: &str| {
                    within(&format!("{name}: the free of {what}"), || {
                        import_gauges(&device) == base
                    })
                };
                let sends = || {
                    let metrics = counters.metrics();
                    [metrics.submissions, metrics.held_release_starts]
                };
                let since = |before: [u64; 2]| {
                    let now = sends();
                    [now[0] - before[0], now[1] - before[1]]
                };

                // 32 pictures acquired and never used, dropped together.
                let pictures: Vec<[SharedTexture; 2]> = (0..32).map(|_| pair()).collect();
                let surfaces: Vec<Surface> = pictures.iter().map(acquired).collect();
                assert_eq!(
                    import_gauges(&device)[0],
                    base[0] + 64,
                    "{name}: each plane's import counted"
                );
                let before = sends();
                drop(surfaces);
                freed("32 pictures never used");
                let burst = since(before);
                assert_eq!(
                    burst,
                    [1, 1],
                    "{name}: one submission, one held release starting it, for the 32 drops"
                );
                drop(pictures);

                // The acquire carried by a read's submission: the drop sends nothing.
                let textures = pair();
                let surface = acquired(&textures);
                rt_read(name, &device, &surface);
                let before = sends();
                drop(surface);
                freed("a picture read");
                let used = since(before);
                assert_eq!(
                    used,
                    [0, 0],
                    "{name}: the drop of a picture read sends nothing"
                );

                // Acquired again after a read: the drop sends that acquire.
                let surface = acquired(&textures);
                rt_read(name, &device, &surface);
                surface.external_writes_done().expect("the second acquire");
                let before = sends();
                drop(surface);
                freed("a picture acquired again after its read");
                let again = since(before);
                assert_eq!(again, [1, 1], "{name}: the drop sends the second acquire");

                // 200 rounds, each freed before the next.
                let mut at = Vec::new();
                for round in 1..=200 {
                    let surface = acquired(&textures);
                    if round % 2 == 1 {
                        rt_read(name, &device, &surface);
                    }
                    drop(surface);
                    freed(&format!("round {round}"));
                    if round == 100 || round == 200 {
                        at.push((import_gauges(&device), counters.gauges().blocks));
                    }
                }
                assert_eq!(at[0].0, at[1].0, "{name}: the gauges at rounds 100 and 200");
                let metrics = counters.metrics();
                println!(
                    "d3d11-drop: {name}: submissions and held releases that started one: 32 drops \
                     of pictures never used {burst:?}, a read picture's drop {used:?}, a drop \
                     after a second acquire {again:?}; 200 rounds: surface imports, their bytes, \
                     textures {:?} at round 100 and {:?} at 200, blocks {} and {}; the device's \
                     held releases {} of {} bytes, {} starts",
                    at[0].0,
                    at[1].0,
                    at[0].1,
                    at[1].1,
                    metrics.held_releases,
                    metrics.held_release_bytes,
                    metrics.held_release_starts,
                );
            });
        }
    }

    /// Direct3D 11 and DXGI, bound by hand: the calls the test makes, at their vtable slots in
    /// the Windows SDK's `d3d11.h`, `dxgi.h`, `dxgi1_2.h` and `d3dcommon.h` (`ID3DBlob`), and
    /// `D3DCompile` (`d3dcompiler.h`), loaded from `d3dcompiler_47.dll`.
    #[cfg(m2v_gpu)]
    mod d3d {
        use std::ffi::{c_char, c_void, CStr};
        use std::os::windows::io::{FromRawHandle, OwnedHandle, RawHandle};
        use std::time::{Duration, Instant};

        pub(super) const DXGI_FORMAT_R16_UINT: u32 = 57;
        pub(super) const DXGI_FORMAT_R8_UNORM: u32 = 61;
        const D3D_DRIVER_TYPE_UNKNOWN: i32 = 0;
        const D3D11_SDK_VERSION: u32 = 7;
        const D3D11_USAGE_DEFAULT: u32 = 0;
        const D3D11_USAGE_STAGING: u32 = 3;
        const D3D11_BIND_SHADER_RESOURCE: u32 = 0x8;
        const D3D11_BIND_RENDER_TARGET: u32 = 0x20;
        const D3D11_RESOURCE_MISC_SHARED: u32 = 0x2;
        const D3D11_RESOURCE_MISC_SHARED_NTHANDLE: u32 = 0x800;
        const D3D11_CPU_ACCESS_READ: u32 = 0x20000;
        const D3D11_MAP_READ: u32 = 1;
        const D3D11_QUERY_EVENT: u32 = 0;
        const D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST: u32 = 4;
        const DXGI_SHARED_RESOURCE_READ: u32 = 0x8000_0000;
        const DXGI_SHARED_RESOURCE_WRITE: u32 = 0x1;
        const DXGI_ERROR_NOT_FOUND: i32 = 0x887a_0002_u32 as i32;
        const S_OK: i32 = 0;

        /// A `GUID` in memory (`Data1`, `Data2`, `Data3`, then `Data4`'s eight bytes), as words.
        type Guid = [u32; 4];

        const fn guid(data1: u32, data2: u16, data3: u16, data4: [u8; 8]) -> Guid {
            let [a, b, c, d, e, f, g, h] = data4;
            [
                data1,
                data2 as u32 | (data3 as u32) << 16,
                u32::from_le_bytes([a, b, c, d]),
                u32::from_le_bytes([e, f, g, h]),
            ]
        }

        const IID_IDXGI_FACTORY1: Guid = guid(
            0x770a_ae78,
            0xf26f,
            0x4dba,
            [0xa8, 0x29, 0x25, 0x3c, 0x83, 0xd1, 0xb3, 0x87],
        );
        const IID_IDXGI_RESOURCE1: Guid = guid(
            0x3096_1379,
            0x4609,
            0x4a41,
            [0x99, 0x8e, 0x54, 0xfe, 0x56, 0x7e, 0xe0, 0xc1],
        );

        /// `DXGI_ADAPTER_DESC1`'s bytes: `Description` (128 `WCHAR`s), `VendorId` at 256,
        /// `DeviceId` at 260, `SubSysId`, `Revision`, three `SIZE_T` memory sizes, `AdapterLuid`
        /// (its `LowPart`, then its `HighPart`) at 296, and `Flags`, aligned as its `SIZE_T`s.
        #[repr(C, align(8))]
        struct AdapterDesc1([u8; 312]);

        impl AdapterDesc1 {
            fn word(&self, at: usize) -> u32 {
                u32::from_le_bytes(self.0[at..at + 4].try_into().expect("four bytes"))
            }

            fn luid(&self) -> [u8; 8] {
                self.0[296..304].try_into().expect("eight bytes")
            }
        }

        /// `D3D11_TEXTURE2D_DESC` of one level, one slice and one sample: `Width`, `Height`,
        /// `MipLevels`, `ArraySize`, `Format`, `SampleDesc` (`Count`, `Quality`), `Usage`,
        /// `BindFlags`, `CPUAccessFlags` and `MiscFlags`.
        fn texture_desc(
            (width, height, format): (u32, u32, u32),
            usage: u32,
            (bind, cpu, misc): (u32, u32, u32),
        ) -> [u32; 11] {
            [width, height, 1, 1, format, 1, 0, usage, bind, cpu, misc]
        }

        /// `D3D11_MAPPED_SUBRESOURCE`: `pData`, then `RowPitch` and `DepthPitch`.
        #[repr(C)]
        struct Mapped {
            data: *mut c_void,
            pitches: [u32; 2],
        }

        #[link(name = "d3d11")]
        extern "system" {
            fn D3D11CreateDevice(
                adapter: *mut c_void,
                driver_type: i32,
                software: *mut c_void,
                flags: u32,
                feature_levels: *const u32,
                feature_level_count: u32,
                sdk_version: u32,
                device: *mut *mut c_void,
                feature_level: *mut u32,
                context: *mut *mut c_void,
            ) -> i32;
        }

        #[link(name = "dxgi")]
        extern "system" {
            fn CreateDXGIFactory1(riid: *const Guid, factory: *mut *mut c_void) -> i32;
        }

        #[link(name = "kernel32")]
        extern "system" {
            fn LoadLibraryW(name: *const u16) -> *mut c_void;
            fn GetProcAddress(module: *mut c_void, name: *const c_char) -> *mut c_void;
        }

        /// The vertex shader: one triangle over the viewport, from the vertex's index alone.
        const VERTEX_SHADER: &str = "float4 main(uint id : SV_VertexID) : SV_Position {
    float2 t = float2((id << 1) & 2, id & 2);
    return float4(t * float2(2, -2) + float2(-1, 1), 0, 1);
}";

        /// `D3DCompile` (`d3dcompiler.h`): the source and its length, its name, defines,
        /// includes, entry point, target, two flag words, and the code and errors written.
        type D3dCompile = unsafe extern "system" fn(
            *const c_void,
            usize,
            *const c_char,
            *const c_void,
            *mut c_void,
            *const c_char,
            *const c_char,
            u32,
            u32,
            *mut *mut c_void,
            *mut *mut c_void,
        ) -> i32;

        /// `source`'s `main` compiled for `target` by `D3DCompile`, which `d3dcompiler_47.dll`
        /// exports.
        fn compile(source: &str, target: &CStr) -> Vec<u8> {
            let library: Vec<u16> = "d3dcompiler_47.dll".encode_utf16().chain([0]).collect();
            // SAFETY: `library` is NUL-terminated UTF-16 naming a system library.
            let module = unsafe { LoadLibraryW(library.as_ptr()) };
            assert!(!module.is_null(), "LoadLibraryW(d3dcompiler_47.dll)");
            // SAFETY: `module` is loaded and never freed; the name is NUL-terminated.
            let function = unsafe { GetProcAddress(module, c"D3DCompile".as_ptr()) };
            assert!(!function.is_null(), "GetProcAddress(D3DCompile)");
            // SAFETY: the function is `D3DCompile`, of that type.
            let compile = unsafe { std::mem::transmute::<*mut c_void, D3dCompile>(function) };
            let (mut code, mut errors) = (std::ptr::null_mut(), std::ptr::null_mut());
            // SAFETY: the source is `source.len()` bytes; no name, defines or includes; the entry
            // point and target are NUL-terminated; the call writes the two blobs.
            let hr = unsafe {
                compile(
                    source.as_ptr().cast(),
                    source.len(),
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null_mut(),
                    c"main".as_ptr(),
                    target.as_ptr(),
                    0,
                    0,
                    &mut code,
                    &mut errors,
                )
            };
            let said = (!errors.is_null()).then(|| blob(&Com(errors)));
            assert!(
                hr >= 0,
                "D3DCompile for {target:?}: HRESULT {hr:#x}: {}",
                String::from_utf8_lossy(&said.unwrap_or_default())
            );
            blob(&Com::made("D3DCompile", hr, code))
        }

        /// A blob's bytes (`ID3DBlob`).
        fn blob(blob: &Com) -> Vec<u8> {
            // SAFETY: ID3DBlob's `GetBufferPointer` and `GetBufferSize`, whose bytes live as long
            // as the blob.
            unsafe {
                let pointer = method::<BlobPointer>(blob.0, BLOB_GET_BUFFER_POINTER)(blob.0);
                let size = method::<BlobSize>(blob.0, BLOB_GET_BUFFER_SIZE)(blob.0);
                std::slice::from_raw_parts(pointer, size).to_vec()
            }
        }

        type QueryInterface =
            unsafe extern "system" fn(*mut c_void, *const Guid, *mut *mut c_void) -> i32;
        type Release = unsafe extern "system" fn(*mut c_void) -> u32;
        type EnumAdapters1 = unsafe extern "system" fn(*mut c_void, u32, *mut *mut c_void) -> i32;
        type GetDesc1 = unsafe extern "system" fn(*mut c_void, *mut AdapterDesc1) -> i32;
        type CreateTexture2D = unsafe extern "system" fn(
            *mut c_void,
            *const [u32; 11],
            *const c_void,
            *mut *mut c_void,
        ) -> i32;
        type CreateQuery =
            unsafe extern "system" fn(*mut c_void, *const [u32; 2], *mut *mut c_void) -> i32;
        type End = unsafe extern "system" fn(*mut c_void, *mut c_void);
        type GetData =
            unsafe extern "system" fn(*mut c_void, *mut c_void, *mut c_void, u32, u32) -> i32;
        type CreateView = unsafe extern "system" fn(
            *mut c_void,
            *mut c_void,
            *const c_void,
            *mut *mut c_void,
        ) -> i32;
        type CreateShader = unsafe extern "system" fn(
            *mut c_void,
            *const c_void,
            usize,
            *mut c_void,
            *mut *mut c_void,
        ) -> i32;
        type SetTopology = unsafe extern "system" fn(*mut c_void, u32);
        type SetShader =
            unsafe extern "system" fn(*mut c_void, *mut c_void, *const *mut c_void, u32);
        type SetTargets =
            unsafe extern "system" fn(*mut c_void, u32, *const *mut c_void, *mut c_void);
        type SetViewports = unsafe extern "system" fn(*mut c_void, u32, *const [f32; 6]);
        type Draw = unsafe extern "system" fn(*mut c_void, u32, u32);
        type BlobPointer = unsafe extern "system" fn(*mut c_void) -> *const u8;
        type BlobSize = unsafe extern "system" fn(*mut c_void) -> usize;
        type CopyResource = unsafe extern "system" fn(*mut c_void, *mut c_void, *mut c_void);
        type Map =
            unsafe extern "system" fn(*mut c_void, *mut c_void, u32, u32, u32, *mut Mapped) -> i32;
        type Unmap = unsafe extern "system" fn(*mut c_void, *mut c_void, u32);
        type CreateSharedHandle = unsafe extern "system" fn(
            *mut c_void,
            *const c_void,
            u32,
            *const u16,
            *mut *mut c_void,
        ) -> i32;

        // The vtable slots: IUnknown's three, then each interface's own in its header's order.
        const QUERY_INTERFACE: usize = 0;
        const RELEASE: usize = 2;
        const FACTORY1_ENUM_ADAPTERS1: usize = 12;
        const ADAPTER1_GET_DESC1: usize = 10;
        const DEVICE_CREATE_TEXTURE_2D: usize = 5;
        const DEVICE_CREATE_RENDER_TARGET_VIEW: usize = 9;
        const DEVICE_CREATE_VERTEX_SHADER: usize = 12;
        const DEVICE_CREATE_PIXEL_SHADER: usize = 15;
        const DEVICE_CREATE_QUERY: usize = 24;
        const CONTEXT_PS_SET_SHADER: usize = 9;
        const CONTEXT_VS_SET_SHADER: usize = 11;
        const CONTEXT_DRAW: usize = 13;
        const CONTEXT_MAP: usize = 14;
        const CONTEXT_UNMAP: usize = 15;
        const CONTEXT_IA_SET_PRIMITIVE_TOPOLOGY: usize = 24;
        const CONTEXT_END: usize = 28;
        const CONTEXT_GET_DATA: usize = 29;
        const CONTEXT_OM_SET_RENDER_TARGETS: usize = 33;
        const CONTEXT_RS_SET_VIEWPORTS: usize = 44;
        const CONTEXT_COPY_RESOURCE: usize = 47;
        const RESOURCE1_CREATE_SHARED_HANDLE: usize = 13;
        const BLOB_GET_BUFFER_POINTER: usize = 3;
        const BLOB_GET_BUFFER_SIZE: usize = 4;

        /// Method `index` of COM object `this`'s vtable, as `F`.
        ///
        /// # Safety
        ///
        /// `this` is a live COM object whose vtable's method `index` is a function of type `F`.
        unsafe fn method<F: Copy>(this: *mut c_void, index: usize) -> F {
            assert_eq!(size_of::<F>(), size_of::<*const c_void>());
            // SAFETY: the caller's: a COM object's first word points at its vtable, an array of
            // function pointers.
            unsafe {
                let vtable = *this.cast::<*const *const c_void>();
                std::mem::transmute_copy::<*const c_void, F>(&*vtable.add(index))
            }
        }

        /// A COM object the test holds a reference to, released at its drop.
        struct Com(*mut c_void);

        impl Com {
            /// `made`'s object, where `hr` succeeded.
            fn made(what: &str, hr: i32, made: *mut c_void) -> Com {
                assert!(hr >= 0 && !made.is_null(), "{what}: HRESULT {hr:#x}");
                Com(made)
            }

            /// Its interface `iid`.
            fn query(&self, what: &str, iid: &Guid) -> Com {
                let mut out = std::ptr::null_mut();
                // SAFETY: a live COM object's QueryInterface, writing one pointer.
                let hr = unsafe {
                    method::<QueryInterface>(self.0, QUERY_INTERFACE)(self.0, iid, &mut out)
                };
                Com::made(what, hr, out)
            }
        }

        impl Drop for Com {
            fn drop(&mut self) {
                // SAFETY: the test's reference to a live COM object, released once.
                unsafe { method::<Release>(self.0, RELEASE)(self.0) };
            }
        }

        /// A shared texture and its NT handle, which the test owns and closes.
        pub(super) struct SharedTexture {
            texture: Com,
            pub(super) handle: OwnedHandle,
        }

        /// A Direct3D 11 device on one adapter, and its immediate context.
        pub(super) struct D3d {
            device: Com,
            context: Com,
        }

        impl D3d {
            /// The device on the DXGI adapter whose LUID is `luid`'s bytes (a `LUID`'s low
            /// word, then its high, as Vulkan's `deviceLUID` holds it); `None` where no adapter has it.
            pub(super) fn on_luid(luid: [u8; 8]) -> Option<D3d> {
                let mut factory = std::ptr::null_mut();
                // SAFETY: the call writes one pointer.
                let hr = unsafe { CreateDXGIFactory1(&IID_IDXGI_FACTORY1, &mut factory) };
                let factory = Com::made("CreateDXGIFactory1", hr, factory);
                for index in 0.. {
                    let mut adapter = std::ptr::null_mut();
                    // SAFETY: IDXGIFactory1::EnumAdapters1, writing one pointer.
                    let hr = unsafe {
                        method::<EnumAdapters1>(factory.0, FACTORY1_ENUM_ADAPTERS1)(
                            factory.0,
                            index,
                            &mut adapter,
                        )
                    };
                    if hr == DXGI_ERROR_NOT_FOUND {
                        return None;
                    }
                    let adapter = Com::made("EnumAdapters1", hr, adapter);
                    let mut desc = AdapterDesc1([0; 312]);
                    // SAFETY: IDXGIAdapter1::GetDesc1, writing the description.
                    let hr = unsafe {
                        method::<GetDesc1>(adapter.0, ADAPTER1_GET_DESC1)(adapter.0, &mut desc)
                    };
                    assert!(hr >= 0, "GetDesc1: HRESULT {hr:#x}");
                    if desc.luid() != luid {
                        continue;
                    }
                    println!(
                        "d3d11: the DXGI adapter of the device's LUID: {:04x}:{:04x}",
                        desc.word(256),
                        desc.word(260)
                    );
                    let (mut device, mut context, mut level) =
                        (std::ptr::null_mut(), std::ptr::null_mut(), 0);
                    // SAFETY: the adapter is live; the call writes the device, its level and its
                    // immediate context.
                    let hr = unsafe {
                        D3D11CreateDevice(
                            adapter.0,
                            D3D_DRIVER_TYPE_UNKNOWN,
                            std::ptr::null_mut(),
                            0,
                            std::ptr::null(),
                            0,
                            D3D11_SDK_VERSION,
                            &mut device,
                            &mut level,
                            &mut context,
                        )
                    };
                    let device = Com::made("D3D11CreateDevice", hr, device);
                    let context = Com::made("D3D11CreateDevice's context", hr, context);
                    return Some(D3d { device, context });
                }
                None
            }

            /// A texture of `format` at `width` by `height`, default usage, shared by an NT
            /// handle, and the handle.
            pub(super) fn shared_texture(
                &self,
                width: u32,
                height: u32,
                format: u32,
            ) -> SharedTexture {
                let desc = texture_desc(
                    (width, height, format),
                    D3D11_USAGE_DEFAULT,
                    (
                        D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET,
                        0,
                        D3D11_RESOURCE_MISC_SHARED | D3D11_RESOURCE_MISC_SHARED_NTHANDLE,
                    ),
                );
                let texture = self.texture(&desc);
                let resource = texture.query("IDXGIResource1", &IID_IDXGI_RESOURCE1);
                let mut handle = std::ptr::null_mut();
                // SAFETY: IDXGIResource1::CreateSharedHandle with no attributes and no name,
                // writing one handle.
                let hr = unsafe {
                    method::<CreateSharedHandle>(resource.0, RESOURCE1_CREATE_SHARED_HANDLE)(
                        resource.0,
                        std::ptr::null(),
                        DXGI_SHARED_RESOURCE_READ | DXGI_SHARED_RESOURCE_WRITE,
                        std::ptr::null(),
                        &mut handle,
                    )
                };
                assert!(
                    hr >= 0 && !handle.is_null(),
                    "CreateSharedHandle: HRESULT {hr:#x}"
                );
                // SAFETY: the call made the handle, which nothing else owns.
                let handle = unsafe { OwnedHandle::from_raw_handle(handle as RawHandle) };
                SharedTexture { texture, handle }
            }

            fn texture(&self, desc: &[u32; 11]) -> Com {
                let mut texture = std::ptr::null_mut();
                // SAFETY: ID3D11Device::CreateTexture2D with no initial data, writing one
                // pointer.
                let hr = unsafe {
                    method::<CreateTexture2D>(self.device.0, DEVICE_CREATE_TEXTURE_2D)(
                        self.device.0,
                        desc,
                        std::ptr::null(),
                        &mut texture,
                    )
                };
                Com::made("CreateTexture2D", hr, texture)
            }

            /// Records one triangle over the whole of `texture`, `width` by `height`, as a render
            /// target, each pixel written by the pixel shader compiled from `pixel` (HLSL, its
            /// entry `main`); then unbinds the target.
            pub(super) fn draw(
                &self,
                texture: &SharedTexture,
                (width, height): (u32, u32),
                pixel: &str,
            ) {
                let vertex = self.shader(
                    "CreateVertexShader",
                    DEVICE_CREATE_VERTEX_SHADER,
                    &compile(VERTEX_SHADER, c"vs_5_0"),
                );
                let pixel = self.shader(
                    "CreatePixelShader",
                    DEVICE_CREATE_PIXEL_SHADER,
                    &compile(pixel, c"ps_5_0"),
                );
                let mut view = std::ptr::null_mut();
                // SAFETY: ID3D11Device::CreateRenderTargetView of a render target texture of the
                // device's, in its own format, writing one pointer.
                let hr = unsafe {
                    method::<CreateView>(self.device.0, DEVICE_CREATE_RENDER_TARGET_VIEW)(
                        self.device.0,
                        texture.texture.0,
                        std::ptr::null(),
                        &mut view,
                    )
                };
                let view = Com::made("CreateRenderTargetView", hr, view);
                // `D3D11_VIEWPORT`: `TopLeftX`, `TopLeftY`, `Width`, `Height`, `MinDepth`,
                // `MaxDepth`. A side of at most 2^24 is exact in an f32.
                let viewport = [0.0, 0.0, width as f32, height as f32, 0.0, 1.0];
                let targets = [view.0];
                let context = self.context.0;
                // SAFETY: each is ID3D11DeviceContext's named method, with objects of the
                // device's; the vertex shader reads no input, so no input layout is bound.
                unsafe {
                    method::<SetTopology>(context, CONTEXT_IA_SET_PRIMITIVE_TOPOLOGY)(
                        context,
                        D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST,
                    );
                    method::<SetShader>(context, CONTEXT_VS_SET_SHADER)(
                        context,
                        vertex.0,
                        std::ptr::null(),
                        0,
                    );
                    method::<SetShader>(context, CONTEXT_PS_SET_SHADER)(
                        context,
                        pixel.0,
                        std::ptr::null(),
                        0,
                    );
                    method::<SetTargets>(context, CONTEXT_OM_SET_RENDER_TARGETS)(
                        context,
                        1,
                        targets.as_ptr(),
                        std::ptr::null_mut(),
                    );
                    method::<SetViewports>(context, CONTEXT_RS_SET_VIEWPORTS)(
                        context, 1, &viewport,
                    );
                    method::<Draw>(context, CONTEXT_DRAW)(context, 3, 0);
                    method::<SetTargets>(context, CONTEXT_OM_SET_RENDER_TARGETS)(
                        context,
                        0,
                        std::ptr::null(),
                        std::ptr::null_mut(),
                    );
                }
            }

            /// A shader of the device's from `code`, by its method `index`
            /// (`CreateVertexShader` or `CreatePixelShader`, which take the same arguments).
            fn shader(&self, what: &str, index: usize, code: &[u8]) -> Com {
                let mut shader = std::ptr::null_mut();
                // SAFETY: the method is ID3D11Device's `index`, of compiled code of its stage,
                // with no class linkage, writing one pointer.
                let hr = unsafe {
                    method::<CreateShader>(self.device.0, index)(
                        self.device.0,
                        code.as_ptr().cast(),
                        code.len(),
                        std::ptr::null_mut(),
                        &mut shader,
                    )
                };
                Com::made(what, hr, shader)
            }

            /// Waits until the work recorded so far has completed: an event query's end, then
            /// its data.
            pub(super) fn finish(&self) {
                let mut query = std::ptr::null_mut();
                // `D3D11_QUERY_DESC`: `Query`, then `MiscFlags`.
                let desc = [D3D11_QUERY_EVENT, 0];
                // SAFETY: ID3D11Device::CreateQuery, writing one pointer.
                let hr = unsafe {
                    method::<CreateQuery>(self.device.0, DEVICE_CREATE_QUERY)(
                        self.device.0,
                        &desc,
                        &mut query,
                    )
                };
                let query = Com::made("CreateQuery", hr, query);
                let context = self.context.0;
                // SAFETY: ID3D11DeviceContext::End of the event query.
                unsafe { method::<End>(context, CONTEXT_END)(context, query.0) };
                let deadline = Instant::now() + Duration::from_secs(10);
                loop {
                    let mut done = 0_i32;
                    // SAFETY: ID3D11DeviceContext::GetData of an event query, a BOOL; with no
                    // flag it flushes.
                    let hr = unsafe {
                        method::<GetData>(context, CONTEXT_GET_DATA)(
                            context,
                            query.0,
                            std::ptr::addr_of_mut!(done).cast(),
                            4,
                            0,
                        )
                    };
                    assert!(hr >= 0, "GetData: HRESULT {hr:#x}");
                    if hr == S_OK && done != 0 {
                        return;
                    }
                    assert!(
                        Instant::now() < deadline,
                        "Direct3D 11's work did not complete"
                    );
                    std::thread::sleep(Duration::from_millis(1));
                }
            }

            /// `texture`'s bytes, rows of `row` bytes, read through a staging copy.
            pub(super) fn read(
                &self,
                texture: &SharedTexture,
                width: u32,
                height: u32,
                format: u32,
                row: u32,
            ) -> Vec<u8> {
                let desc = texture_desc(
                    (width, height, format),
                    D3D11_USAGE_STAGING,
                    (0, D3D11_CPU_ACCESS_READ, 0),
                );
                let staging = self.texture(&desc);
                let context = self.context.0;
                // SAFETY: ID3D11DeviceContext::CopyResource between two textures of one format
                // and size.
                unsafe {
                    method::<CopyResource>(context, CONTEXT_COPY_RESOURCE)(
                        context,
                        staging.0,
                        texture.texture.0,
                    )
                };
                let mut mapped = Mapped {
                    data: std::ptr::null_mut(),
                    pitches: [0; 2],
                };
                // SAFETY: ID3D11DeviceContext::Map of the staging texture for reading, which
                // waits for the copy.
                let hr = unsafe {
                    method::<Map>(context, CONTEXT_MAP)(
                        context,
                        staging.0,
                        0,
                        D3D11_MAP_READ,
                        0,
                        &mut mapped,
                    )
                };
                assert!(hr >= 0 && !mapped.data.is_null(), "Map: HRESULT {hr:#x}");
                let mut out = Vec::with_capacity((row * height) as usize);
                for y in 0..height as usize {
                    // SAFETY: the mapping holds `height` rows `RowPitch` apart, each of at least
                    // `row` bytes.
                    let line = unsafe {
                        std::slice::from_raw_parts(
                            mapped.data.cast::<u8>().add(y * mapped.pitches[0] as usize),
                            row as usize,
                        )
                    };
                    out.extend_from_slice(line);
                }
                // SAFETY: ID3D11DeviceContext::Unmap of the mapping, which nothing reads after.
                unsafe { method::<Unmap>(context, CONTEXT_UNMAP)(context, staging.0, 0) };
                out
            }
        }
    }
}
