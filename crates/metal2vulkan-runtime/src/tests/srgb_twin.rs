//! An 8-bit sRGB texture's storage element is its UNORM twin's view, and a kernel's write at a
//! slot bound to it runs the kernel's sRGB variant, which encodes in the shader
//! (`LoweringInputs::srgb_storage`): one form on every device.
//!
//! Under a device cfg, on every device of the cfg's kind, through the owned fixture
//! `fpv_srgb_storage`'s `fpv_srgb_write` and `fpv_srgb_write_half` over a [`WIDTH`] by 1 texture
//! of `RGBA8Unorm_sRGB` and one of `BGRA8Unorm_sRGB`, of shader-write usage:
//!
//! - each state's precision notes hold `PrecisionNote::DevicePackConversion` from its creation;
//! - where the texture has its twin's storage view (the device stores the twin, and its image
//!   format query takes the image extended with the storage usage), each dispatch runs and the
//!   texture's bytes hold, in the format's channel order, the MSL transfer's encode of each value
//!   written within one code, and its alpha linear; elsewhere the dispatch is
//!   `Gap::StorageElement` of slot 0, `StorageElementCase::Absent`;
//! - a kernel's sRGB variant is lowered at its first dispatch onto a twin and kept in the
//!   device's map of contents: a dispatch onto the other format's twin, or onto an `RGBA8Unorm`
//!   texture, lowers nothing; filled past a cap of 2, the map holds 2, and a variant it forgot
//!   is lowered again at its kernel's next dispatch onto a twin (on a device whose textures have
//!   their twins' storage views). It prints the translation time of the float kernel's state and
//!   of its variant.
//!
//! `llvm-as` assembles the fixture; without it the tests skip visibly.

#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod on_device {
    use crate::tests::command::on_device::{each_device, finished};
    use crate::tests::draw::on_device::Buffers;
    use crate::texture::TextureObject;
    use crate::{
        Buffer, ComputePipelineDescriptor, ComputePipelineState, Device, DispatchType, Error,
        FunctionDescriptor, Gap, IndexClass, Library, PixelFormat, PrecisionNote, Region,
        StageSource, StorageElementCase, StorageMode, Texture, TextureDescriptor, TextureType,
        TextureUsage,
    };
    use ash::vk;
    use metal2vulkan_devtools::{engine_dir, llvm_assemble, AssembleError};

    /// The textures' width, a multiple of 16.
    const WIDTH: u32 = 64;

    /// The formats written, each with its twin and the byte each of red, green, blue and alpha
    /// takes in a texel.
    const FORMATS: [(PixelFormat, vk::Format, [usize; 4]); 2] = [
        (
            PixelFormat::RGBA8UnormSRGB,
            vk::Format::R8G8B8A8_UNORM,
            [0, 1, 2, 3],
        ),
        (
            PixelFormat::BGRA8UnormSRGB,
            vk::Format::B8G8R8A8_UNORM,
            [2, 1, 0, 3],
        ),
    ];

    /// The fixture's library, or None where `llvm-as` is missing.
    fn library() -> Option<Library> {
        match llvm_assemble(&engine_dir().join("fpv_srgb_storage.ll")) {
            Ok(bitcode) => Some(Library::from_air(&bitcode).expect("the library")),
            Err(AssembleError::Missing(why)) => {
                eprintln!("SKIPPED: {why}");
                None
            }
            Err(error) => panic!("fpv_srgb_storage.ll: {error:?}"),
        }
    }

    /// The kernel `kernel`'s state.
    fn state(device: &Device, library: &Library, kernel: &str) -> ComputePipelineState {
        let function = library.function(kernel).expect("the kernel");
        let descriptor = ComputePipelineDescriptor::new(StageSource::Function(
            FunctionDescriptor::new(function),
        ));
        (device.new_compute_pipeline_state(&descriptor, crate::tests::no_constexpr_samplers))
            .unwrap_or_else(|error| panic!("{kernel}: {error}"))
    }

    /// A [`WIDTH`] by 1 texture of `format` and shader-write usage at texture index `index`.
    fn texture(name: &str, device: &Device, format: PixelFormat, index: u32) -> Texture {
        let mut d = TextureDescriptor::new(TextureType::Type2D, format, WIDTH, 1);
        d.usage = TextureUsage::SHADER_WRITE;
        d.storage_mode = StorageMode::Shared;
        (device.new_texture(&d, Some(index))).unwrap_or_else(|error| panic!("{name}: {error}"))
    }

    /// The linear value the sRGB transfer decodes code `code` to.
    fn center(code: u32) -> f32 {
        let c = f64::from(code) / 255.0;
        let linear = match c <= 0.04045 {
            true => c / 12.92,
            false => ((c + 0.055) / 1.055).powf(2.4),
        };
        linear as f32
    }

    /// The values written, a texel each: two of channels past either end, then codes' values
    /// in three orders, alpha rising.
    fn values() -> Vec<[f32; 4]> {
        let mut out = vec![[-1.0, 2.0, 1.5, 1.25], [0.0, -0.5, 1.0, -0.25]];
        for i in 2..WIDTH {
            let code = i * 4;
            let rest = [255 - code, (code * 37) & 255].map(center);
            out.push([center(code), rest[0], rest[1], i as f32 / 63.0]);
        }
        out
    }

    /// `value`'s `half` bits, its significand cut: `value` is 0 or of a normal `half`'s size.
    fn half(value: f32) -> u16 {
        if value == 0.0 {
            return 0;
        }
        let bits = value.to_bits();
        let sign = ((bits >> 16) & 0x8000) as u16;
        let exponent = (((bits >> 23) & 0xff) + 15 - 127) as u16;
        sign | (exponent << 10) | ((bits >> 13) & 0x3ff) as u16
    }

    /// The value of `bits`, a `half` that [`half`] made.
    fn unhalf(bits: u16) -> f32 {
        let sign = u32::from(bits & 0x8000) << 16;
        if bits & 0x7fff == 0 {
            return f32::from_bits(sign);
        }
        let exponent = u32::from((bits >> 10) & 0x1f) + 127 - 15;
        f32::from_bits(sign | (exponent << 23) | (u32::from(bits & 0x3ff) << 13))
    }

    /// Buffer 0's bytes for `kernel`, and the values its write takes: each `float4`, or each
    /// `half4`'s.
    fn written(kernel: &str) -> (Vec<u8>, Vec<[f32; 4]>) {
        let mut bytes = Vec::new();
        let mut taken = Vec::new();
        for texel in values() {
            match kernel.ends_with("_half") {
                true => {
                    let bits = texel.map(half);
                    bytes.extend(bits.iter().flat_map(|bits| bits.to_le_bytes()));
                    taken.push(bits.map(unhalf));
                }
                false => {
                    bytes.extend(texel.iter().flat_map(|value| value.to_le_bytes()));
                    taken.push(texel);
                }
            }
        }
        (bytes, taken)
    }

    /// The code the MSL transfer encodes `value` to, clamped to `[0, 1]` first, or, for alpha,
    /// the linear code.
    fn code(value: f32, alpha: bool) -> f64 {
        let v = f64::from(value).clamp(0.0, 1.0);
        let c = match (alpha, v < 0.0031308) {
            (true, _) => v,
            (false, true) => 12.92 * v,
            (false, false) => 1.055 * v.powf(1.0 / 2.4) - 0.055,
        };
        (c * 255.0).round()
    }

    /// One dispatch of `state` over [`WIDTH`] threads with `texture` at slot 0, `values` at
    /// buffer 0 and `width` at buffer 1, committed and finished: the encoder's answer.
    fn dispatch(
        device: &Device,
        state: &ComputePipelineState,
        texture: &Texture,
        values: &Buffer,
        width: &Buffer,
    ) -> Result<(), Error> {
        let queue = device.new_command_queue(1).expect("a queue");
        let command_buffer = queue.command_buffer().expect("a command buffer");
        let mut encoder =
            (command_buffer.compute_command_encoder(DispatchType::Serial)).expect("an encoder");
        encoder
            .set_compute_pipeline_state(state)
            .expect("the state");
        encoder.set_texture(Some(texture), 0).expect("a texture");
        encoder.set_buffer(Some(values), 0, 0).expect("a buffer");
        encoder.set_buffer(Some(width), 0, 1).expect("a buffer");
        let dispatched = encoder.dispatch_threadgroups([WIDTH / 16, 1, 1], [16, 1, 1]);
        encoder.end_encoding();
        command_buffer.commit().expect("a commit");
        finished("sRGB twins", &command_buffer);
        assert_eq!(command_buffer.error(), None);
        dispatched
    }

    /// Whether the device stores `twin` through a storage image without a format.
    fn stores(device: &Device, twin: vk::Format) -> bool {
        let (_, optimal) = device.core().format_features(twin);
        optimal.contains(
            vk::FormatFeatureFlags2::STORAGE_IMAGE
                | vk::FormatFeatureFlags2::STORAGE_WRITE_WITHOUT_FORMAT,
        )
    }

    #[test]
    fn every_device_writes_an_8_bit_srgb_texture_through_its_unorm_twin_encoded() {
        let Some(library) = library() else {
            return;
        };
        each_device("sRGB twins", |name, _, device| {
            let _grant = (device.grant_indices(IndexClass::Texture, 1, 2))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let mut buffers = Buffers::new(&device);
            let width = buffers.of(&device, &WIDTH.to_le_bytes());
            let mut off = 0.0f64;
            for (at, (format, twin, order)) in FORMATS.into_iter().enumerate() {
                let texture = texture(name, &device, format, 1 + at as u32);
                let usage = texture.object().image_usage();
                let extended = usage.contains(vk::ImageUsageFlags::STORAGE);
                let stores = stores(&device, twin);
                assert_eq!(
                    texture.encodes(),
                    stores && extended,
                    "{name}: {format:?}: the twin, the device storing it {stores}"
                );
                for kernel in ["fpv_srgb_write", "fpv_srgb_write_half"] {
                    let state = state(&device, &library, kernel);
                    assert!(
                        (state.precision_notes()).contains(&PrecisionNote::DevicePackConversion),
                        "{name}: {kernel}: the note"
                    );
                    let (bytes, taken) = written(kernel);
                    let values = buffers.of(&device, &bytes);
                    match (
                        texture.encodes(),
                        dispatch(&device, &state, &texture, &values, &width),
                    ) {
                        (true, Ok(())) => {}
                        (
                            false,
                            Err(Error::Unsupported(Gap::StorageElement {
                                slot: Some(0),
                                pixel_format,
                                case: StorageElementCase::Absent,
                                ..
                            })),
                        ) if pixel_format == format => {
                            println!("{name}: {format:?}: {kernel}: no twin, refused");
                            continue;
                        }
                        (encodes, other) => {
                            panic!("{name}: {format:?}: {kernel}, a twin {encodes}: {other:?}")
                        }
                    }
                    let mut stored = vec![0; WIDTH as usize * 4];
                    let region = Region {
                        origin: [0; 3],
                        size: [WIDTH, 1, 1],
                    };
                    let pitch = u64::from(WIDTH) * 4;
                    (texture.get_bytes(&mut stored, pitch, pitch, region, 0, 0))
                        .unwrap_or_else(|error| panic!("{name}: {format:?}: {error}"));
                    for (i, value) in taken.iter().enumerate() {
                        for (channel, &byte) in order.iter().enumerate() {
                            let want = code(value[channel], channel == 3);
                            let got = f64::from(stored[i * 4 + byte]);
                            assert!(
                                (got - want).abs() <= 1.0,
                                "{name}: {format:?}: {kernel}: texel {i} channel {channel}: \
                                 {value:?} stored {got}, the encode {want}"
                            );
                            off = off.max((got - want).abs());
                        }
                    }
                }
            }
            println!(
                "{name}: 8-bit sRGB written through the UNORM twin, every code within {off} of \
                 the MSL encode"
            );
        });
    }

    #[test]
    fn every_device_keeps_a_kernels_srgb_variant_in_the_bounded_map_of_contents() {
        let Some(library) = library() else {
            return;
        };
        each_device("sRGB variants", |name, _, device| {
            let _grant = (device.grant_indices(IndexClass::Texture, 1, 3))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let [rgba, bgra] =
                [0, 1].map(|at| texture(name, &device, FORMATS[at].0, 1 + at as u32));
            if !(rgba.encodes() && bgra.encodes()) {
                println!("{name}: a texture without its twin's storage view; nothing to keep");
                return;
            }
            let unorm = texture(name, &device, PixelFormat::RGBA8Unorm, 3);
            device.core().kernels.set_cap(2);
            let counters = device.counters();
            let entries = || counters.gauges().kernel_entries;
            let lowered = || counters.metrics().translations;
            let nanos = || counters.metrics().translation_nanos;
            let mut buffers = Buffers::new(&device);
            let width = buffers.of(&device, &WIDTH.to_le_bytes());
            let begun = nanos();
            let float = state(&device, &library, "fpv_srgb_write");
            let plain = nanos() - begun;
            let values = buffers.of(&device, &written("fpv_srgb_write").0);
            let run = |state: &ComputePipelineState, texture: &Texture| {
                (dispatch(&device, state, texture, &values, &width))
                    .unwrap_or_else(|error| panic!("{name}: {error}"));
            };
            let (start, begun) = (lowered(), nanos());
            run(&float, &rgba);
            let (each, variant) = (lowered() - start, nanos() - begun);
            assert!(
                each > 0,
                "{name}: the variant lowered at its first dispatch"
            );
            run(&float, &bgra);
            run(&float, &unorm);
            assert_eq!(lowered() - start, each, "{name}: one variant, kept");
            assert_eq!(entries(), 2, "{name}: the state's content and its variant");
            // The half kernel's content and its variant fill the map past its cap: the float
            // kernel's content goes, then its variant.
            let half = state(&device, &library, "fpv_srgb_write_half");
            let made = lowered();
            (dispatch(&device, &half, &rgba, &values, &width))
                .unwrap_or_else(|error| panic!("{name}: half: {error}"));
            assert!(lowered() > made, "{name}: the half variant lowered");
            assert_eq!(entries(), 2, "{name}: the map holds its cap");
            let forgotten = lowered();
            run(&float, &rgba);
            assert_eq!(
                lowered() - forgotten,
                each,
                "{name}: a forgotten variant lowered again"
            );
            assert_eq!(entries(), 2, "{name}: the map holds its cap");
            drop((float, half, rgba, bgra, unorm));
            println!(
                "{name}: one sRGB variant a kernel, kept in a map of 2 filled past its cap, a \
                 forgotten one lowered again; fpv_srgb_write's state took {plain} ns of \
                 translation, its sRGB variant {variant} ns"
            );
        });
    }
}
