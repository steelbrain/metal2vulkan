//! The Direct3D 11 surface import, a Windows host's, under a device cfg. On every device of the
//! cfg's kind, its typed refusals: no plane, five planes, a plane whose image format's texel is
//! not its descriptor's, and a plane wider than the import query's `maxExtent` for its image. Under `m2v_gpu`, on every device: a picture's two planes made as shared
//! Direct3D 11 textures on the DXGI adapter of the device's LUID (luma `R8_UNORM` at 64x32, chroma
//! `R16_UINT` at 32x16, which rt's texture views as `RG8Unorm`), imported by their NT handles
//! with no submission. Direct3D 11 draws picture 1 (one triangle a plane, its pixel shader
//! computing each texel's bytes, then an event query waited for), and rt blits the planes into
//! a Shared buffer, whose bytes are picture 1's exactly. Direct3D 11 draws picture 2, each of whose bytes differs from picture 1's, and rt's
//! next read is picture 2's exactly and matches picture 1 at no byte: the negative control, which
//! a copy taken at the import, or a read of bytes the second draw did not reach, fails. Then rt
//! writes picture 3 into the planes by a blit, and Direct3D 11 reads picture 3 exactly through
//! staging copies (`CopyResource`, `Map`). Each of rt's three submissions keeps the planes,
//! acquires them from `VK_QUEUE_FAMILY_EXTERNAL` at its start and releases them at its end: three
//! in `Metrics::d3d11_plane_submissions`, three acquires and three releases, the consumer's
//! `external_writes_done` and `release_surface_to_external` calls, still made, recording none.
//! Two reads committed back to back on queues of their own, the first not waited for before the
//! second commits, read picture 1 exactly and count two plane-keeping submissions, two acquires
//! and two releases; on an instance without the validation layer, 64 reads committed back to back
//! on one queue are timed, per command buffer, and printed.
//! While the surface lives, the surface imports gauge holds its two planes; it has no buffer and
//! no export handle. Each plane's image takes the usage its texture's bind flags give and no
//! other: made rendered and sampled, no storage; made sampled alone, not rendered; and a texture
//! over a plane whose uses ask storage is `Gap::SurfaceUsage`, where the device stores its
//! format. Planes made with `PixelFormatView` usage, an 8-bit luma, a chroma and a 10-bit luma,
//! list no format past their own, their texture's and that one's sRGB sibling (the 10-bit luma's
//! image not mutable), and a swizzled view in a third format of their class (`R8Uint`, `RG8Uint`,
//! `R16Uint`) is `Gap::TextureView` of `SurfacePixelFormat`.
//!
//! Under `m2v_gpu`, on every device, the real-size pictures' planes (`real_size.rs`): luma
//! `R16_UNORM` at 3840x2160 with chroma `R32_UINT` at 1920x1080 (which rt views as `RG16Unorm`),
//! and luma `R8_UNORM` with chroma `R16_UINT` (`RG8Unorm`), made as above, each imported with an
//! index under a texture of the consumer's planes' usage (`ShaderRead | RenderTarget |
//! PixelFormatView`). Direct3D 11 draws each plane's complement and then its words; then rt
//! samples every texel, reads it through `read()`, copies it out and samples it again, Direct3D
//! 11 reads it back through a staging copy, and rt samples it a third time, each read printed
//! with its count of wrong texels; then Direct3D 11 draws the complement, and rt's next sample
//! reads it. Two images over the same textures in the form the import made before its images
//! took their textures' bind flags (one with storage usage, one with the long format list) are
//! each sampled after a fresh draw and printed, not asserted.

#[cfg(all(windows, any(m2v_gpu, m2v_cpu_icd)))]
mod on_device {
    use crate::d3d11::import_query;
    use crate::tests::command::on_device::each_device;
    use crate::{
        D3d11Plane, Error, ImportHandle, Limit, PixelFormat, Refusal, TextureDescriptor,
        TextureType,
    };
    use std::os::windows::io::AsHandle;

    /// `D3D11_BIND_SHADER_RESOURCE` (`d3d11.h`).
    const SHADER_RESOURCE: u32 = 0x8;
    /// `D3D11_BIND_RENDER_TARGET` (`d3d11.h`).
    const RENDER_TARGET: u32 = 0x20;

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
                bind_flags: SHADER_RESOURCE | RENDER_TARGET,
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
                // A plane one texel wider than the import query's `maxExtent` for its image with
                // the handle: `Limit::TextureSize`, at that answer or the plain query's below it.
                let core = device.core();
                let (usage, list) =
                    (device.d3d11_plane_form(core, &luma)).expect("the luma's form");
                let format = (device.plane_plan(&luma.descriptor))
                    .expect("a plan")
                    .format;
                let (_, made) =
                    (import_query(core, format, usage, &list)).expect("the import query");
                let most = made.max_extent.width;
                let mut past = luma;
                past.descriptor.width = most + 1;
                let error = device.import_d3d11_surface(&[past]).err();
                println!(
                    "d3d11: {name}: the import query's maxExtent for the luma {:?}; a plane {} wide: \
                     {error:?}",
                    made.max_extent,
                    most + 1
                );
                assert!(
                    matches!(
                        error,
                        Some(Error::Limit(Limit::TextureSize {
                            extent: [width, 32, 1],
                            layers: 1,
                            max_extent: [max, ..],
                            ..
                        })) if width == most + 1 && max <= most
                    ),
                    "{name}: a plane past the import's maxExtent: {error:?}"
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
        use super::{RENDER_TARGET, SHADER_RESOURCE};
        use crate::device::lock;
        use crate::submit::IN_FLIGHT;
        use crate::tests::command::on_device::{
            behind_busy, each_device, finished, on_each_device, within,
        };
        use crate::texture::TextureObject;
        use crate::{
            Buffer, BufferLayout, CommandBuffer, CommandBufferStatus, D3d11Plane, Device, Error,
            Gap, IndexClass, MemoryClass, PixelFormat, Surface, TextureDescriptor, TexturePlace,
            TextureSwizzle, TextureSwizzleChannels, TextureType, TextureUsage, TextureViewKind,
        };
        use ash::vk;
        use metal2vulkan::metal::format::srgb_sibling;
        use std::os::windows::io::AsHandle;
        use std::sync::Arc;
        use std::time::Instant;

        /// The bind flags each plane's texture is made with, as a video's are: rendered and
        /// sampled.
        const BOUND: u32 = SHADER_RESOURCE | RENDER_TARGET;

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
        pub(super) fn luid(device: &Device) -> Option<[u8; 8]> {
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
            read_bytes(name, rt_read_committed(name, device, surface))
        }

        /// The bytes of a picture's two planes, packed.
        fn picture_len() -> u64 {
            PLANES.iter().map(Plane::bytes).sum::<usize>() as u64
        }

        /// A new Shared buffer of a picture's bytes.
        fn picture_buffer(device: &Device) -> Buffer {
            let len = picture_len();
            let block = (device.take_block(MemoryClass::Shared, len)).expect("a block");
            (device.new_buffer(block.id, 0, len, None)).expect("a buffer")
        }

        /// rt's read of both planes into a new Shared buffer, packed, committed in a command
        /// buffer of a queue of its own and not waited for: the command buffer and the buffer.
        fn rt_read_committed(
            name: &str,
            device: &Device,
            surface: &Surface,
        ) -> (CommandBuffer, Buffer) {
            let out = picture_buffer(device);
            let queue = device.new_command_queue(1).expect("a queue");
            let command_buffer = queue.command_buffer().expect("a command buffer");
            encode_read(name, &command_buffer, surface, &out);
            command_buffer.commit().expect("a commit");
            (command_buffer, out)
        }

        /// The bytes of a read ([`rt_read_committed`]) once its command buffer completed.
        fn read_bytes(name: &str, (command_buffer, out): (CommandBuffer, Buffer)) -> Vec<u8> {
            finished(name, &command_buffer);
            assert_eq!(command_buffer.error(), None, "{name}");
            let contents = out.contents().expect("Shared");
            // SAFETY: the buffer's bytes, which no pending command reaches.
            unsafe { std::slice::from_raw_parts(contents.as_ptr(), picture_len() as usize) }
                .to_vec()
        }

        /// Encodes the blits of both planes into `out`, packed, in `command_buffer`.
        fn encode_read(
            name: &str,
            command_buffer: &CommandBuffer,
            surface: &Surface,
            out: &Buffer,
        ) {
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
                    out,
                    layout,
                ))
                .unwrap_or_else(|error| panic!("{name}: the read: {error}"));
            }
            encoder.end_encoding();
        }

        /// The surface of `textures`, imported by their handles, its planes reached through their
        /// views alone.
        fn import(name: &str, device: &Device, textures: &[SharedTexture; 2]) -> Surface {
            let planes: Vec<D3d11Plane<'_>> = (textures.iter().zip(&PLANES))
                .map(|(texture, plane)| D3d11Plane {
                    handle: texture.handle.as_handle(),
                    image_format: plane.image,
                    bind_flags: BOUND,
                    descriptor: TextureDescriptor::new(
                        TextureType::Type2D,
                        plane.view,
                        plane.width,
                        plane.height,
                    ),
                    index: None,
                })
                .collect();
            (device.import_d3d11_surface(&planes))
                .unwrap_or_else(|error| panic!("{name}: the import: {error}"))
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
                    .map(|p| d3d.shared_texture(p.width, p.height, p.dxgi, BOUND));
                let planes: Vec<D3d11Plane<'_>> = (textures.iter().zip(&PLANES))
                    .map(|(texture, plane)| D3d11Plane {
                        handle: texture.handle.as_handle(),
                        image_format: plane.image,
                        bind_flags: BOUND,
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
                let handed = [
                    metrics.d3d11_plane_submissions - before.d3d11_plane_submissions,
                    metrics.external_acquires - before.external_acquires,
                    metrics.external_releases - before.external_releases,
                ];
                println!(
                    "d3d11: {name}: imported two planes (R8_UNORM 64x32, R16_UINT 32x16 viewed as \
                     RG8Unorm) with no submission; pictures 1 and 2 drawn by D3D11 read exact by \
                     rt, the second unlike the first at every byte; picture 3 written by rt read \
                     exact by D3D11; of rt's 3 submissions that keep the planes: counted, \
                     acquires, releases {handed:?}"
                );
                assert_eq!(
                    handed,
                    [3, 3, 3],
                    "{name}: each of rt's submissions that keep the planes counted, acquiring and \
                     releasing them once"
                );
                drop(surface);
            });
        }

        /// Two reads of the planes Direct3D 11 drew, each a command buffer of a queue of its own,
        /// so each goes in a submission of its own, the second committed before the first is
        /// waited for: both read picture 1 exactly, and the two submissions count two in
        /// `Metrics::d3d11_plane_submissions`, with two acquires and two releases. Each acquire
        /// follows a global barrier, which orders the first submission's release before the
        /// second's acquire on rt's one queue; the barriers recorded are printed. A job may run it
        /// under the validation layer's synchronization checks
        /// (`VK_KHRONOS_VALIDATION_VALIDATE_SYNC`), whose messages fail it as any other does.
        #[test]
        fn every_device_reads_direct3d_11_planes_twice_back_to_back() {
            each_device("import_d3d11_surface_back_to_back", |name, _, device| {
                let luid = luid(&device).unwrap_or_else(|| panic!("{name}: no valid LUID"));
                let d3d = D3d::on_luid(luid)
                    .unwrap_or_else(|| panic!("{name}: no DXGI adapter of LUID {luid:02x?}"));
                let textures = PLANES
                    .each_ref()
                    .map(|p| d3d.shared_texture(p.width, p.height, p.dxgi, BOUND));
                let surface = import(name, &device, &textures);
                d3d_draw(&d3d, &textures, 1);
                surface.external_writes_done().expect("the acquire");
                let counters = device.counters();
                let before = counters.metrics();
                let first = rt_read_committed(name, &device, &surface);
                let second = rt_read_committed(name, &device, &surface);
                let reads = [read_bytes(name, first), read_bytes(name, second)];
                let one = picture(1);
                for (at, read) in reads.iter().enumerate() {
                    let n = at + 1;
                    assert_eq!(compare(&one, read).0, None, "{name}: read {n} of picture 1");
                }
                let metrics = counters.metrics();
                let handed = [
                    metrics.d3d11_plane_submissions - before.d3d11_plane_submissions,
                    metrics.external_acquires - before.external_acquires,
                    metrics.external_releases - before.external_releases,
                ];
                println!(
                    "d3d11-back-to-back: {name}: two reads committed back to back read picture 1 \
                     exact; plane-keeping submissions, acquires, releases {handed:?}; barriers {}",
                    metrics.barriers - before.barriers
                );
                assert_eq!(
                    handed,
                    [2, 2, 2],
                    "{name}: each read's submission hands the planes over once"
                );
            });
        }

        /// The planes' hand-over in each part of a group sent member by member. On one queue,
        /// behind as many busy submissions as a queue keeps in flight, a read of the planes
        /// Direct3D 11 drew (A) and a second command buffer join one group: a second read (B), or
        /// a copy between two buffers that uses no plane (F). Sent at once at the queue's run
        /// (`CommandQueue::send_now`), A and B go as one part: one `vkQueueSubmit2`, one acquire
        /// and one release. Sent once a run of 1 s was seen (`QueueInner::ran`, the state the
        /// completion thread sets as a command buffer retires), the group goes member by member,
        /// one `vkQueueSubmit2` a member and none more: A and B each acquire the planes at their
        /// part's head and release them at its end, two of each; A and F acquire and release
        /// once, in A's part, so A, seen complete at its own part's value, is seen complete after
        /// the planes' release. Each such call of a group sent apart counts in
        /// `Metrics::d3d11_plane_parts`, two and one, and the group sent as one part counts none
        /// there. Every read is picture 1's exactly. The order within each part is the part
        /// builder's (`plane_parts`).
        #[test]
        fn every_device_hands_direct3d_11_planes_over_in_each_part_of_a_group_sent_apart() {
            each_device("import_d3d11_surface_apart", |name, _, device| {
                let luid = luid(&device).unwrap_or_else(|| panic!("{name}: no valid LUID"));
                let d3d = D3d::on_luid(luid)
                    .unwrap_or_else(|| panic!("{name}: no DXGI adapter of LUID {luid:02x?}"));
                let textures = PLANES
                    .each_ref()
                    .map(|p| d3d.shared_texture(p.width, p.height, p.dxgi, BOUND));
                let surface = import(name, &device, &textures);
                d3d_draw(&d3d, &textures, 1);
                surface.external_writes_done().expect("the acquire");
                let one = picture(1);
                let ahead = IN_FLIGHT as u64;
                let counters = device.counters();
                // Each case: its name, whether the second member reads the planes, whether the
                // group goes member by member, and the hand-overs, submissions and hand-overs of
                // parts sent apart it makes.
                let cases = [
                    ("A and B as one part", true, false, 1, 1, 0),
                    ("A and B apart", true, true, 2, 2, 2),
                    ("A and F apart", false, true, 1, 2, 1),
                ];
                for (case, reads, apart, hands, parts, part_hands) in cases {
                    let outs = [picture_buffer(&device), picture_buffer(&device)];
                    let from = picture_buffer(&device);
                    let queue = device.new_command_queue(64).expect("a queue");
                    behind_busy(name, &device, &queue, |busy, submitted, _| {
                        let before = counters.metrics();
                        let a = queue.command_buffer().expect("a command buffer");
                        encode_read(name, &a, &surface, &outs[0]);
                        a.commit().expect("a commit");
                        let second = queue.command_buffer().expect("a command buffer");
                        match reads {
                            true => encode_read(name, &second, &surface, &outs[1]),
                            false => {
                                let mut blit = second.blit_command_encoder().expect("an encoder");
                                (blit.copy_from_buffer_to_buffer(&from, 0, &outs[1], 0, 256))
                                    .expect("a copy");
                                blit.end_encoding();
                            }
                        }
                        second.commit().expect("a commit");
                        for command_buffer in [&a, &second] {
                            within(name, || {
                                command_buffer.status() != CommandBufferStatus::Committed
                            });
                        }
                        let sent = || counters.metrics().submissions - submitted;
                        assert_eq!(sent(), ahead, "{name}: {case}: held in one group");
                        match apart {
                            true => second.queue().ran(1_000_000_000),
                            false => {
                                queue.send_now(&second);
                                within(name, || sent() > ahead);
                            }
                        }
                        busy.open();
                        let got = [
                            read_bytes(name, (a.clone(), outs[0].clone())),
                            read_bytes(name, (second.clone(), outs[1].clone())),
                        ];
                        let metrics = counters.metrics();
                        let handed = [
                            metrics.d3d11_plane_submissions - before.d3d11_plane_submissions,
                            metrics.external_acquires - before.external_acquires,
                            metrics.external_releases - before.external_releases,
                        ];
                        let group = sent() - ahead;
                        let part_handed = metrics.d3d11_plane_parts - before.d3d11_plane_parts;
                        println!(
                            "d3d11-apart: {name}: {case}: the group's vkQueueSubmit2 calls {group}; \
                             plane-keeping submissions, acquires, releases {handed:?}; \
                             d3d11_plane_parts {part_handed}"
                        );
                        assert_eq!(
                            compare(&one, &got[0]).0,
                            None,
                            "{name}: {case}: A read picture 1"
                        );
                        if reads {
                            assert_eq!(
                                compare(&one, &got[1]).0,
                                None,
                                "{name}: {case}: B read picture 1"
                            );
                        }
                        if apart {
                            assert!(group >= 2, "{name}: {case}: went member by member");
                        }
                        assert_eq!(
                            handed, [hands; 3],
                            "{name}: {case}: one acquire and one release in each part that uses \
                             the planes"
                        );
                        assert_eq!(
                            group, parts,
                            "{name}: {case}: one vkQueueSubmit2 a part, and no more"
                        );
                        assert_eq!(
                            part_handed, part_hands,
                            "{name}: {case}: the parts of a group sent apart that hand planes over"
                        );
                    });
                }
            });
        }

        /// The old layout of each plane's acquire. A plane is made undefined
        /// (`VUID-VkImageCreateInfo-pNext-01443`) and is general only after Direct3D 11's access
        /// (section 12.5.2), which the consumer reports by `Surface::external_writes_done`. On a
        /// surface rt writes first, with no such call (a blit of picture 3 into both planes),
        /// each plane's first acquire names the undefined layout, and the acquire of rt's next
        /// read names the general layout; Direct3D 11 reads picture 3 exactly between them, and
        /// that read is picture 3's exactly. On a surface Direct3D 11 draws first, reported by
        /// `external_writes_done`, every acquire names the general layout, the first included,
        /// and rt reads picture 1 exactly. The acquires are read from the record the hand-over
        /// keeps for the device tests (`Work::acquired`).
        #[test]
        fn every_device_acquires_a_direct3d_11_plane_from_undefined_only_before_any_access() {
            each_device("import_d3d11_surface_first_use", |name, _, device| {
                let luid = luid(&device).unwrap_or_else(|| panic!("{name}: no valid LUID"));
                let d3d = D3d::on_luid(luid)
                    .unwrap_or_else(|| panic!("{name}: no DXGI adapter of LUID {luid:02x?}"));
                let core = device.core();
                // The old layouts of the acquires of `surface`'s planes recorded since `from`,
                // each plane's in order, the luma's first.
                let acquires = |surface: &Surface, from: usize| -> Vec<Vec<vk::ImageLayout>> {
                    let recorded = lock(&core.work.acquired);
                    (surface.planes().iter())
                        .map(|texture| {
                            let image = Arc::as_ptr(&texture.held_image()) as usize;
                            (recorded[from..].iter())
                                .filter(|&&(at, _)| at == image)
                                .map(|&(_, layout)| layout)
                                .collect()
                        })
                        .collect()
                };
                let mark = || lock(&core.work.acquired).len();
                let (undefined, general) = (vk::ImageLayout::UNDEFINED, vk::ImageLayout::GENERAL);

                // rt's first use, with no Direct3D 11 access before it.
                let textures = PLANES
                    .each_ref()
                    .map(|p| d3d.shared_texture(p.width, p.height, p.dxgi, BOUND));
                let surface = import(name, &device, &textures);
                let from = mark();
                let three = picture(3);
                rt_write(name, &device, &surface, &three);
                let written = acquires(&surface, from);
                let read = d3d_read(&d3d, &textures);
                let from = mark();
                let again = rt_read(name, &device, &surface);
                let later = acquires(&surface, from);
                println!(
                    "d3d11-first-use: {name}: written first by rt: acquires {written:?}, then \
                     after D3D11's read {later:?}"
                );
                assert_eq!(
                    compare(&three, &read).0,
                    None,
                    "{name}: picture 3 read by D3D11"
                );
                assert_eq!(
                    compare(&three, &again).0,
                    None,
                    "{name}: picture 3 read by rt"
                );
                assert_eq!(
                    written,
                    [vec![undefined], vec![undefined]],
                    "{name}: a plane's first acquire before any Direct3D 11 access names the \
                     undefined layout"
                );
                assert_eq!(
                    later,
                    [vec![general], vec![general]],
                    "{name}: a plane's later acquire names the general layout"
                );

                // Direct3D 11's first use, reported, then rt's.
                let textures = PLANES
                    .each_ref()
                    .map(|p| d3d.shared_texture(p.width, p.height, p.dxgi, BOUND));
                let surface = import(name, &device, &textures);
                d3d_draw(&d3d, &textures, 1);
                surface.external_writes_done().expect("the acquire");
                let from = mark();
                let first = rt_read(name, &device, &surface);
                let second = rt_read(name, &device, &surface);
                let reported = acquires(&surface, from);
                println!(
                    "d3d11-first-use: {name}: drawn first by D3D11 and reported: acquires \
                     {reported:?}"
                );
                let one = picture(1);
                assert_eq!(
                    compare(&one, &first).0,
                    None,
                    "{name}: picture 1 read by rt"
                );
                assert_eq!(
                    compare(&one, &second).0,
                    None,
                    "{name}: picture 1 read again"
                );
                assert_eq!(
                    reported,
                    [vec![general, general], vec![general, general]],
                    "{name}: after external_writes_done, every acquire names the general layout"
                );
            });
        }

        /// How many reads [`every_device_times_back_to_back_reads_of_direct3d_11_planes`] commits
        /// in a run.
        const READS: u32 = 64;

        /// `READS` reads of the planes Direct3D 11 drew into one Shared buffer, each a command
        /// buffer of one queue committed as soon as it is encoded, the last waited for: the wall
        /// time from the first's encoding to the last's completion over `READS`, in microseconds,
        /// for each of five runs, and the plane-keeping submissions each made; their median
        /// printed with them. On an instance without the validation layer, whose own cost would be
        /// the count. Printed only.
        #[test]
        fn every_device_times_back_to_back_reads_of_direct3d_11_planes() {
            let instance = crate::tests::validation::bare(&[]);
            on_each_device(
                "import_d3d11_surface_read_time",
                &instance,
                |name, _, device| {
                    let luid = luid(&device).unwrap_or_else(|| panic!("{name}: no valid LUID"));
                    let d3d = D3d::on_luid(luid)
                        .unwrap_or_else(|| panic!("{name}: no DXGI adapter of LUID {luid:02x?}"));
                    let textures = PLANES
                        .each_ref()
                        .map(|p| d3d.shared_texture(p.width, p.height, p.dxgi, BOUND));
                    let surface = import(name, &device, &textures);
                    d3d_draw(&d3d, &textures, 1);
                    surface.external_writes_done().expect("the acquire");
                    let out = picture_buffer(&device);
                    let queue = device.new_command_queue(READS).expect("a queue");
                    let counters = device.counters();
                    let mut runs = Vec::new();
                    for _ in 0..5 {
                        let before = counters.metrics().d3d11_plane_submissions;
                        let began = Instant::now();
                        let mut last = None;
                        for _ in 0..READS {
                            let command_buffer = queue.command_buffer().expect("a command buffer");
                            encode_read(name, &command_buffer, &surface, &out);
                            command_buffer.commit().expect("a commit");
                            last = Some(command_buffer);
                        }
                        let last = last.expect("a read");
                        finished(name, &last);
                        assert_eq!(last.error(), None, "{name}");
                        let micros = began.elapsed().as_micros() as u64 / u64::from(READS);
                        let kept = counters.metrics().d3d11_plane_submissions - before;
                        runs.push((micros, kept));
                    }
                    let mut sorted: Vec<u64> = runs.iter().map(|&(micros, _)| micros).collect();
                    sorted.sort_unstable();
                    println!(
                        "d3d11-read-time: {name}: {READS} reads committed back to back: {} us per \
                     command buffer, the median of five runs; each run's us and plane-keeping \
                     submissions {runs:?}",
                        sorted[2]
                    );
                },
            );
        }

        /// Three planes made with `ShaderRead`, `RenderTarget` and `PixelFormatView` usage, as a
        /// video's are, reached through their views alone: an 8-bit luma (`R8_UNORM`), a chroma
        /// (`R16_UINT` under `RG8Unorm`) and a 10-bit luma (`R16_UNORM`). Each image lists no
        /// format past its own, the descriptor's and the descriptor's sRGB sibling, and the pair
        /// where the two differ: the 10-bit luma's lists none and is not mutable. A swizzled view
        /// in a third format of the class (`R8Uint`, `RG8Uint`, `R16Uint`) is `Gap::TextureView` of
        /// `SurfacePixelFormat`. The lists are printed.
        #[test]
        fn every_device_lists_a_direct3d_11_plane_s_formats_and_no_third() {
            each_device("import_d3d11_surface_views", |name, _, device| {
                let luid = luid(&device).unwrap_or_else(|| panic!("{name}: no valid LUID"));
                let d3d = D3d::on_luid(luid)
                    .unwrap_or_else(|| panic!("{name}: no DXGI adapter of LUID {luid:02x?}"));
                // Each plane: the texture's format in rt's terms and DXGI's, the descriptor's, and
                // the third format a view asks.
                let shapes = [
                    (
                        PixelFormat::R8Unorm,
                        d3d::DXGI_FORMAT_R8_UNORM,
                        PixelFormat::R8Unorm,
                        PixelFormat::R8Uint,
                    ),
                    (
                        PixelFormat::R16Uint,
                        d3d::DXGI_FORMAT_R16_UINT,
                        PixelFormat::RG8Unorm,
                        PixelFormat::RG8Uint,
                    ),
                    (
                        PixelFormat::R16Unorm,
                        d3d::DXGI_FORMAT_R16_UNORM,
                        PixelFormat::R16Unorm,
                        PixelFormat::R16Uint,
                    ),
                ];
                let (width, height) = (64, 32);
                let textures =
                    shapes.map(|(_, dxgi, _, _)| d3d.shared_texture(width, height, dxgi, BOUND));
                let descriptors = shapes.map(|(_, _, view, _)| {
                    let mut d = TextureDescriptor::new(TextureType::Type2D, view, width, height);
                    d.usage = TextureUsage::SHADER_READ
                        | TextureUsage::RENDER_TARGET
                        | TextureUsage::PIXEL_FORMAT_VIEW;
                    d
                });
                let planes: Vec<D3d11Plane<'_>> = (textures.iter().zip(&shapes))
                    .zip(&descriptors)
                    .map(|((texture, &(image, ..)), descriptor)| D3d11Plane {
                        handle: texture.handle.as_handle(),
                        image_format: image,
                        bind_flags: BOUND,
                        descriptor: *descriptor,
                        index: None,
                    })
                    .collect();
                let surface = (device.import_d3d11_surface(&planes))
                    .unwrap_or_else(|error| panic!("{name}: the import: {error}"));
                let first = IndexClass::Texture.range().start;
                let _grant =
                    (device.grant_indices(IndexClass::Texture, first, 3)).expect("a grant");
                let swizzle = TextureSwizzleChannels {
                    red: TextureSwizzle::Red,
                    green: TextureSwizzle::Red,
                    blue: TextureSwizzle::Red,
                    alpha: TextureSwizzle::One,
                };
                for (((texture, d), &(image, _, view, third)), at) in
                    (surface.planes().iter().zip(&descriptors))
                        .zip(&shapes)
                        .zip(first..)
                {
                    let label = format!("{name}: {image:?} as {view:?}");
                    // The Vulkan format rt holds `pixel_format` in as a view of this plane's
                    // texture, where the device holds it.
                    let format = |pixel_format| {
                        let mut d = *d;
                        d.pixel_format = pixel_format;
                        device.plane_plan(&d).ok().map(|plan| plan.format.as_raw())
                    };
                    let listed: Vec<i32> = (texture.object().image_view_formats().iter())
                        .map(|format| format.as_raw())
                        .collect();
                    let own = [format(image), format(view)].map(|f| f.expect("the plane's format"));
                    let sibling = srgb_sibling(view).and_then(format);
                    let pair: Vec<i32> = match image == view {
                        true => Vec::new(),
                        false => own.to_vec(),
                    };
                    assert!(
                        (listed.iter()).all(|f| own.contains(f) || Some(*f) == sibling),
                        "{label}: the image lists {listed:?}, past its own, the descriptor's \
                         ({own:?}) and the sibling's ({sibling:?})"
                    );
                    assert!(
                        pair.iter().all(|f| listed.contains(f)),
                        "{label}: the image lists {listed:?}, not the pair {pair:?}"
                    );
                    if image == view && sibling.is_none() {
                        assert!(listed.is_empty(), "{label}: not mutable: {listed:?}");
                    }
                    let made = texture.new_texture_view_with_swizzle(
                        third,
                        TextureType::Type2D,
                        0..1,
                        0..1,
                        swizzle,
                        Some(at),
                    );
                    assert!(
                        matches!(
                            made.as_ref().err(),
                            Some(Error::Unsupported(Gap::TextureView {
                                kind: TextureViewKind::SurfacePixelFormat,
                            }))
                        ),
                        "{label}: the {third:?} view: {:?}",
                        made.err()
                    );
                    println!(
                        "d3d11-views: {label}: the image lists {listed:?} (own and descriptor's \
                         {own:?}, sibling {sibling:?}); the swizzled {third:?} view refused"
                    );
                }
            });
        }

        /// Each plane's image takes the usage its texture's bind flags give and no other. Made
        /// rendered and sampled (`D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE`, as a
        /// video's planes are), it is copied and sampled, and rendered at most beside, with no
        /// storage; made sampled alone, it is copied and sampled, and not rendered. A texture over
        /// a plane whose uses ask storage (`ShaderWrite`) is `Gap::SurfaceUsage` of storage, where
        /// the device stores the texture's format through a storage image without a format, as
        /// Direct3D 11 made the texture without `D3D11_BIND_UNORDERED_ACCESS`; where it does not,
        /// the uses ask no storage and the import is made.
        #[test]
        fn every_device_makes_a_direct3d_11_plane_of_the_usage_its_bind_flags_give() {
            each_device("import_d3d11_surface_usage", |name, _, device| {
                let luid = luid(&device).unwrap_or_else(|| panic!("{name}: no valid LUID"));
                let d3d = D3d::on_luid(luid)
                    .unwrap_or_else(|| panic!("{name}: no DXGI adapter of LUID {luid:02x?}"));
                let copied = vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST;
                let read = copied | vk::ImageUsageFlags::SAMPLED;
                let rendered = read | vk::ImageUsageFlags::COLOR_ATTACHMENT;
                for (bind, given) in [(BOUND, rendered), (SHADER_RESOURCE, read)] {
                    let textures = PLANES
                        .each_ref()
                        .map(|p| d3d.shared_texture(p.width, p.height, p.dxgi, bind));
                    let planes: Vec<D3d11Plane<'_>> = (textures.iter().zip(&PLANES))
                        .map(|(texture, plane)| D3d11Plane {
                            handle: texture.handle.as_handle(),
                            image_format: plane.image,
                            bind_flags: bind,
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
                    for (texture, plane) in surface.planes().iter().zip(&PLANES) {
                        let label = format!(
                            "{name}: {:?} as {:?}, bind flags {bind:#x}",
                            plane.image, plane.view
                        );
                        let usage = texture.object().image_usage();
                        println!(
                            "d3d11-usage: {label}: the image's usage {:#x}, the flags give {:#x}",
                            usage.as_raw(),
                            given.as_raw()
                        );
                        assert_eq!(
                            (usage & !given).as_raw(),
                            0,
                            "{label}: a usage the bind flags do not give, of {:#x}",
                            usage.as_raw()
                        );
                        assert!(
                            usage.contains(read),
                            "{label}: copied and sampled: {:#x}",
                            usage.as_raw()
                        );
                    }
                }
                let textures = PLANES
                    .each_ref()
                    .map(|p| d3d.shared_texture(p.width, p.height, p.dxgi, BOUND));
                for (texture, plane) in textures.iter().zip(&PLANES) {
                    let label = format!("{name}: {:?} as {:?} written", plane.image, plane.view);
                    let mut d = TextureDescriptor::new(
                        TextureType::Type2D,
                        plane.view,
                        plane.width,
                        plane.height,
                    );
                    d.usage = TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE;
                    let storage = vk::ImageUsageFlags::STORAGE;
                    let stores =
                        (device.plane_plan(&d).expect("the plane's plan").usage).contains(storage);
                    let made = device.import_d3d11_surface(&[D3d11Plane {
                        handle: texture.handle.as_handle(),
                        image_format: plane.image,
                        bind_flags: BOUND,
                        descriptor: d,
                        index: None,
                    }]);
                    println!(
                        "d3d11-usage: {label}: the uses ask storage: {stores}; the import: {:?}",
                        made.as_ref().err()
                    );
                    match stores {
                        true => assert!(
                            matches!(
                                made.as_ref().err(),
                                Some(Error::Unsupported(Gap::SurfaceUsage { missing, .. }))
                                    if missing & storage.as_raw() != 0
                            ),
                            "{label}: storage asked of a texture made without unordered access: \
                             {:?}",
                            made.err()
                        ),
                        false => assert!(made.is_ok(), "{label}: {:?}", made.err()),
                    }
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

        /// A surface of Direct3D 11 planes is freed at its drop with no other call and no
        /// submission: `external_writes_done` records nothing for it, each submission that uses
        /// the planes acquiring them itself, so no upload holds a plane. 32 pictures' planes, each
        /// pair imported and `external_writes_done` called, dropped together; a picture a command
        /// buffer read, dropped once that command buffer completed; one whose
        /// `external_writes_done` is called again after such a read, dropped; then 200 rounds of
        /// import, `external_writes_done`, a read on every other round and drop, each freed before
        /// the next, the gauges the same at rounds 100 and 200.
        #[test]
        fn every_device_frees_direct3d_11_planes_at_the_surface_drop() {
            each_device("import_d3d11_surface_drop", |name, _, device| {
                let luid = luid(&device).unwrap_or_else(|| panic!("{name}: no valid LUID"));
                let d3d = D3d::on_luid(luid)
                    .unwrap_or_else(|| panic!("{name}: no DXGI adapter of LUID {luid:02x?}"));
                let pair = || {
                    PLANES
                        .each_ref()
                        .map(|p| d3d.shared_texture(p.width, p.height, p.dxgi, BOUND))
                };
                let acquired = |textures: &[SharedTexture; 2]| -> Surface {
                    let planes: Vec<D3d11Plane<'_>> = (textures.iter().zip(&PLANES))
                        .map(|(texture, plane)| D3d11Plane {
                            handle: texture.handle.as_handle(),
                            image_format: plane.image,
                            bind_flags: BOUND,
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

                // 32 pictures never used, dropped together.
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
                assert_eq!(burst, [0, 0], "{name}: the 32 drops send nothing");
                drop(pictures);

                // Read by a command buffer: the drop sends nothing.
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

                // `external_writes_done` again after a read: the drop sends nothing.
                let surface = acquired(&textures);
                rt_read(name, &device, &surface);
                surface.external_writes_done().expect("the second acquire");
                let before = sends();
                drop(surface);
                freed("a picture whose writes were done again after its read");
                let again = since(before);
                assert_eq!(
                    again,
                    [0, 0],
                    "{name}: the drop after the second call sends nothing"
                );

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
                     after a second external_writes_done {again:?}; 200 rounds: surface imports, \
                     their bytes, \
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

    /// The real-size pictures' planes (`PICTURES` of `real_size.rs`) as Direct3D 11 makes them:
    /// shared textures of default usage, render target and shader resource, shared by NT handle,
    /// written by a Direct3D 11 draw through a render target view in the texture's own format,
    /// then imported by rt and read, every texel compared. Beside the import, two images of the
    /// form the import made before its images took their textures' bind flags, each over the same
    /// textures, sampled and printed, not compared to a bar.
    #[cfg(m2v_gpu)]
    mod real_size {
        use super::d3d::{self, D3d, SharedTexture};
        use super::picture::luid;
        use super::{RENDER_TARGET, SHADER_RESOURCE};
        use crate::d3d11::import_query;
        use crate::hazard::HeapId;
        use crate::tests::command::on_device::each_device;
        use crate::tests::real_size::on_device::{
            copy_out, differs, library, read, sampled, Cells, Plane, Readers, PICTURES,
        };
        use crate::texture::TextureObject;
        use crate::{
            D3d11Plane, Device, IndexClass, PixelFormat, TextureDescriptor, TextureType,
            TextureUsage,
        };
        use ash::vk;
        use std::os::windows::io::AsHandle;

        /// The planes' bind flags, as the consumer makes them.
        const BIND: u32 = SHADER_RESOURCE | RENDER_TARGET;

        /// The DXGI format of a plane's image.
        fn dxgi(plane: &Plane) -> u32 {
            match plane.image {
                PixelFormat::R16Unorm => d3d::DXGI_FORMAT_R16_UNORM,
                PixelFormat::R32Uint => d3d::DXGI_FORMAT_R32_UINT,
                PixelFormat::R8Unorm => d3d::DXGI_FORMAT_R8_UNORM,
                PixelFormat::R16Uint => d3d::DXGI_FORMAT_R16_UINT,
                other => panic!("no plane of {other:?}"),
            }
        }

        /// The pixel shader that draws `plane`'s words (`Plane::word`), complemented where
        /// `complemented`: a normalized image's as the word over its maximum, which its format
        /// stores as the word, and an integer image's as the word.
        fn pixel_shader(plane: &Plane, complemented: bool) -> String {
            let mask = plane.mask();
            let flip = if complemented { mask } else { 0 };
            let hash = "uint hash(uint x, uint y) {
    uint v = (x * 0x9e3779b1u) ^ (y * 0x85ebca77u);
    v ^= v >> 15;
    v *= 0x2c1b3c6du;
    return v ^ (v >> 12);
}
";
            let word = format!("((hash(q.x, q.y) & {mask}u) ^ {flip}u)");
            let main = match plane.image {
                PixelFormat::R16Unorm | PixelFormat::R8Unorm => format!(
                    "float main(float4 p : SV_Position) : SV_Target {{
    uint2 q = uint2(p.xy);
    return float({word}) / {mask}.0;
}}"
                ),
                _ => format!(
                    "uint main(float4 p : SV_Position) : SV_Target {{
    uint2 q = uint2(p.xy);
    return {word};
}}"
                ),
            };
            format!("{hash}{main}")
        }

        /// Direct3D 11 draws each plane's words, complemented where `complemented`, and waits.
        fn draw(d3d: &D3d, textures: &[SharedTexture; 2], planes: &[Plane; 2], complemented: bool) {
            for (texture, plane) in textures.iter().zip(planes) {
                let [width, height] = plane.size;
                d3d.draw(texture, (width, height), &pixel_shader(plane, complemented));
            }
            d3d.finish();
        }

        /// The image usage and format list the import gave a plane before its images took their
        /// textures' bind flags, recomputed from `import_d3d11_surface` in
        /// `crates/metal2vulkan-runtime/src/d3d11.rs` at commit 3073230664 (its `WANTED` usage
        /// and its `plane_list`): sampled, rendered,
        /// copied from and into, and storage, with the descriptor's own usage, each where the
        /// view's format's optimal tiling holds it and the import query takes it with the image's
        /// and the view's formats; the feedback usage where the plane asks it and the query takes
        /// it; and the list of the image's format, the view's and every format the descriptor's
        /// texture views as, where it is longer than the pair and the query takes it with that
        /// usage, else the pair where the two formats differ, else none.
        fn old_form(
            device: &Device,
            plane: &D3d11Plane<'_>,
        ) -> (vk::ImageUsageFlags, Vec<vk::Format>) {
            let core = device.core();
            let d = &plane.descriptor;
            let plan = device.plane_plan(d).expect("the plane's plan");
            let mut own = *d;
            own.pixel_format = plane.image_format;
            let format = device.plane_plan(&own).expect("the image's plan").format;
            let view = plan.format;
            let pair = match format == view {
                true => Vec::new(),
                false => vec![format, view],
            };
            let (_, held) = core.format_features(view);
            let holds = |bit: vk::ImageUsageFlags| {
                let feature = match bit {
                    vk::ImageUsageFlags::SAMPLED => vk::FormatFeatureFlags2::SAMPLED_IMAGE,
                    vk::ImageUsageFlags::COLOR_ATTACHMENT => {
                        vk::FormatFeatureFlags2::COLOR_ATTACHMENT
                    }
                    vk::ImageUsageFlags::TRANSFER_SRC => vk::FormatFeatureFlags2::TRANSFER_SRC,
                    vk::ImageUsageFlags::TRANSFER_DST => vk::FormatFeatureFlags2::TRANSFER_DST,
                    vk::ImageUsageFlags::STORAGE => vk::FormatFeatureFlags2::STORAGE_IMAGE,
                    _ => return false,
                };
                held.contains(feature)
            };
            let imports = |usage: vk::ImageUsageFlags, list: &[vk::Format]| {
                (import_query(core, format, usage, list)).is_ok_and(|(features, _)| {
                    features.contains(vk::ExternalMemoryFeatureFlags::IMPORTABLE)
                })
            };
            let wanted = vk::ImageUsageFlags::SAMPLED
                | vk::ImageUsageFlags::COLOR_ATTACHMENT
                | vk::ImageUsageFlags::TRANSFER_SRC
                | vk::ImageUsageFlags::TRANSFER_DST
                | vk::ImageUsageFlags::STORAGE
                | plan.usage;
            let usage = (0..32)
                .map(|bit| vk::ImageUsageFlags::from_raw(1u32 << bit))
                .filter(|&bit| wanted.contains(bit) && holds(bit) && imports(bit, &pair))
                .fold(vk::ImageUsageFlags::empty(), |all, bit| all | bit);
            let looped = usage | vk::ImageUsageFlags::ATTACHMENT_FEEDBACK_LOOP_EXT;
            let usage = match plan.looped && imports(looped, &pair) {
                true => looped,
                false => usage,
            };
            let mut full = vec![format];
            for &listed in [view].iter().chain(&plan.view_formats) {
                if !full.contains(&listed) {
                    full.push(listed);
                }
            }
            if full.len() == 1 {
                full.clear();
            }
            let list = match full.len() > pair.len() && imports(usage, &full) {
                true => full,
                false => pair,
            };
            (usage, list)
        }

        /// The planes of `textures`, as the consumer imports them, at the indices from `at`.
        fn imported<'a>(
            textures: &'a [SharedTexture; 2],
            planes: &[Plane; 2],
            at: u32,
        ) -> Vec<D3d11Plane<'a>> {
            (textures.iter().zip(planes).zip(at..))
                .map(|((texture, plane), index)| D3d11Plane {
                    handle: texture.handle.as_handle(),
                    image_format: plane.image,
                    bind_flags: BIND,
                    descriptor: descriptor(plane),
                    index: Some(index),
                })
                .collect()
        }

        /// The texture rt makes over `plane`, of the usage the consumer's planes take: read by
        /// shaders, rendered to and viewed in another format (`ShaderRead | RenderTarget |
        /// PixelFormatView`), so the old form's list is the long one.
        fn descriptor(plane: &Plane) -> TextureDescriptor {
            let mut d = TextureDescriptor::new(
                TextureType::Type2D,
                plane.view,
                plane.size[0],
                plane.size[1],
            );
            d.usage = TextureUsage::SHADER_READ
                | TextureUsage::RENDER_TARGET
                | TextureUsage::PIXEL_FORMAT_VIEW;
            d
        }

        /// The formats of a list, raw.
        fn raw(list: &[vk::Format]) -> Vec<i32> {
            list.iter().map(|format| format.as_raw()).collect()
        }

        /// Each picture's two planes made by Direct3D 11 and imported by rt (the product's form),
        /// drawn by Direct3D 11, first their complement and then their words; then, each read
        /// printed as it reads: rt samples each, reads each through `read()`, copies each out,
        /// samples each again; Direct3D 11 reads each back through a staging copy; rt samples each
        /// a third time; then Direct3D 11 draws the complement, and rt's sample reads it. Every
        /// one of these is compared, and the test fails on any texel wrong.
        ///
        /// Then two images of the old form over each of the same textures, each after Direct3D
        /// 11 drew the complement and the words again, so each image's first read follows a
        /// draw: OLD-A, the import's usage with storage where the old form took it, and the
        /// import's list; OLD-B, the import's usage with the old form's list. Each is sampled
        /// once, every texel, and printed, not compared to a bar: on a driver where the old form
        /// reads wrong and the import right, the line that differs names the half that does. The
        /// old images are only read while they live, never written, as the hazard sets do not
        /// know they alias the import.
        #[test]
        fn every_device_reads_real_size_direct3d_11_planes_every_texel() {
            let Some(library) = library() else {
                return;
            };
            each_device("real_size_d3d11", |name, _, device| {
                let luid = luid(&device).unwrap_or_else(|| panic!("{name}: no valid LUID"));
                let d3d = D3d::on_luid(luid)
                    .unwrap_or_else(|| panic!("{name}: no DXGI adapter of LUID {luid:02x?}"));
                let readers = Readers::new(name, &device, &library);
                let first = IndexClass::Texture.range().start;
                (device.grant_indices(IndexClass::Texture, first, 4))
                    .unwrap_or_else(|error| panic!("{name}: a texture grant: {error}"));
                let queue = device.new_command_queue(4).expect("a queue");
                let on = (&device, &queue);
                let mut cells = Cells::new(name);
                for planes in &PICTURES {
                    let textures = planes.each_ref().map(|plane| {
                        d3d.shared_texture(plane.size[0], plane.size[1], dxgi(plane), BIND)
                    });
                    let surface = (device
                        .import_d3d11_surface(&imported(&textures, planes, first)))
                    .unwrap_or_else(|error| panic!("{name}: the import: {error}"));
                    draw(&d3d, &textures, planes, true);
                    draw(&d3d, &textures, planes, false);
                    surface.external_writes_done().expect("the acquire");
                    let wants = planes
                        .each_ref()
                        .map(|plane| (plane.words(false, 4), plane.words(false, plane.texel)));
                    let cell = |plane: &Plane, how: &str| {
                        format!("Direct3D 11 {}, drawn by Direct3D 11: {how}", plane.what)
                    };
                    let sample = |cells: &mut Cells, how: &str| {
                        for ((texture, plane), (words, _)) in
                            (surface.planes().iter().zip(planes)).zip(&wants)
                        {
                            let at = (texture, plane.size);
                            let got = sampled(name, on, &readers, at, &plane.pack(), words);
                            cells.check(&cell(plane, how), got, words, (4, plane.size[0]));
                        }
                    };
                    sample(&mut cells, "rt sampled, first");
                    for ((texture, plane), (words, _)) in
                        (surface.planes().iter().zip(planes)).zip(&wants)
                    {
                        let at = (texture, plane.size);
                        let got = read(name, on, &readers, at, &plane.pack(), words);
                        cells.check(&cell(plane, "rt read()"), got, words, (4, plane.size[0]));
                    }
                    for ((texture, plane), (_, image)) in
                        (surface.planes().iter().zip(planes)).zip(&wants)
                    {
                        let got = copy_out(name, on, (texture, plane.size), plane.texel, image);
                        let texel = (plane.texel as usize, plane.size[0]);
                        cells.check(&cell(plane, "rt copied out"), got, image, texel);
                    }
                    sample(&mut cells, "rt sampled, second");
                    for ((texture, plane), (_, image)) in (textures.iter().zip(planes)).zip(&wants)
                    {
                        let [width, height] = plane.size;
                        let row = width * plane.texel;
                        let got = d3d.read(texture, width, height, dxgi(plane), row);
                        let texel = (plane.texel as usize, width);
                        cells.check(&cell(plane, "Direct3D 11 read back"), Ok(got), image, texel);
                    }
                    sample(&mut cells, "rt sampled, third");
                    // Direct3D 11's next draw, the complement, is what rt samples next.
                    draw(&d3d, &textures, planes, true);
                    surface.external_writes_done().expect("the acquire");
                    for (texture, plane) in surface.planes().iter().zip(planes) {
                        let words = plane.words(true, 4);
                        let at = (texture, plane.size);
                        let got = sampled(name, on, &readers, at, &plane.pack(), &words);
                        let how = "rt sampled after Direct3D 11 drew the complement";
                        cells.check(&cell(plane, how), got, &words, (4, plane.size[0]));
                    }

                    // The old form's two halves, each over the same textures, at the two indices
                    // past the import's.
                    let ours = imported(&textures, planes, first + 2);
                    let forms: Vec<_> = (surface.planes().iter().zip(&ours))
                        .map(|(texture, plane)| {
                            let object = texture.object();
                            let new = (object.image_usage(), object.image_view_formats().to_vec());
                            (new, old_form(&device, plane))
                        })
                        .collect();
                    for half in ["OLD-A", "OLD-B"] {
                        draw(&d3d, &textures, planes, true);
                        draw(&d3d, &textures, planes, false);
                        surface.external_writes_done().expect("the acquire");
                        for ((plane, ours), ((new_usage, new_list), (old_usage, old_list))) in
                            (planes.iter().zip(&ours)).zip(&forms)
                        {
                            let (usage, list) = match half {
                                "OLD-A" => (
                                    *new_usage | (*old_usage & vk::ImageUsageFlags::STORAGE),
                                    new_list,
                                ),
                                _ => (*new_usage, old_list),
                            };
                            let what = format!(
                                "Direct3D 11 {}, drawn by Direct3D 11: {half} (usage {:#x}, list {:?}; \
                                 the import's usage {:#x}, list {:?}): rt sampled first, PRINTED, \
                                 not asserted",
                                plane.what,
                                usage.as_raw(),
                                raw(list),
                                new_usage.as_raw(),
                                raw(new_list),
                            );
                            let unit = HeapId::unit(device.core());
                            let made = device.d3d11_plane_texture(ours, usage, list.clone(), unit);
                            let line = match made {
                                Err(error) => format!("not made: {error} ({error:?})"),
                                Ok(texture) => {
                                    let words = plane.words(false, 4);
                                    let at = (&texture, plane.size);
                                    match sampled(name, on, &readers, at, &plane.pack(), &words) {
                                        Err(why) => format!("REFUSED: {why}"),
                                        Ok(got) => match differs(&got, &words, 4, plane.size[0]) {
                                            None => "exact".to_owned(),
                                            Some(how) => format!("WRONG: {how}"),
                                        },
                                    }
                                }
                            };
                            println!("real-size: {name}: {what}: {line}");
                        }
                    }
                    drop(surface);
                }
                cells.all_exact();
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

        pub(super) const DXGI_FORMAT_R32_UINT: u32 = 42;
        pub(super) const DXGI_FORMAT_R16_UNORM: u32 = 56;
        pub(super) const DXGI_FORMAT_R16_UINT: u32 = 57;
        pub(super) const DXGI_FORMAT_R8_UNORM: u32 = 61;
        const D3D_DRIVER_TYPE_UNKNOWN: i32 = 0;
        const D3D11_SDK_VERSION: u32 = 7;
        const D3D11_USAGE_DEFAULT: u32 = 0;
        const D3D11_USAGE_STAGING: u32 = 3;
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

            /// A texture of `format` at `width` by `height`, default usage, of the bind flags
            /// `bind`, shared by an NT handle, and the handle.
            pub(super) fn shared_texture(
                &self,
                width: u32,
                height: u32,
                format: u32,
                bind: u32,
            ) -> SharedTexture {
                let desc = texture_desc(
                    (width, height, format),
                    D3D11_USAGE_DEFAULT,
                    (
                        bind,
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
