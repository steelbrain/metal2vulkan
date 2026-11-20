//! An opaque-black border sampled through a swizzled view, under a device cfg on every device of
//! the cfg's kind, through the owned fixture `a2_clamp_to_zero`'s runtime sampler: a 4 by 4
//! `RGBA8Unorm` texture whose every texel is `(0x40, 0x80, 0xc0, 0x20)`, a view of it made with
//! the swizzle green, alpha, zero, red, and a sampler state of nearest filters that clamps `s`
//! and `t` to an opaque-black border, sampled at a texel centre and a texel past each side.
//!
//! - Metal's border texel is the border colour in the format's channels, then the swizzle:
//!   `(0, 0, 0, 1)` through green, alpha, zero, red reads `(0, 1, 0, 0)`. A device that lists
//!   `VK_EXT_border_color_swizzle` and reports `borderColorSwizzleFromImage` is given the extension
//!   with both features, and reads that texel exactly; Vulkan leaves it undefined elsewhere, where
//!   the extension is not enabled and the test prints the device's answer.
//! - The texel inside reads through the swizzle on every device.
//! - The state carries `PrecisionNote::SwizzledOpaqueBlackBorder` on every device, decided by its
//!   descriptor: a state that clamps to zero with an opaque-black border carries it too, and one
//!   with a transparent-black border, or an opaque-black one that no address mode reads, none.
//! - A pipeline carries the note where its kernel samples through a constexpr sampler of an
//!   opaque-black border, decided by the sampler's own constants: the owned fixture
//!   `a2_border_color`'s state does (its `border_black`), `a2_clamp_to_zero`'s, whose constexpr
//!   samplers clamp to a transparent-black border, does not. `border_black` sampled past the left
//!   edge through the same view reads Metal's texel where the device reports it from the image.
//!
//! The work runs on every device. `llvm-as` assembles the fixtures; without it the test skips
//! visibly.

#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod on_device {
    use crate::tests::command::on_device::{each_device, finished, Arena};
    use crate::{
        ComputePipelineDescriptor, ComputePipelineState, Device, DispatchType, FunctionDescriptor,
        HostBytes, IndexClass, Library, PixelFormat, PrecisionNote, Region, SamplerAddressMode,
        SamplerBorderColor, SamplerDescriptor, StageSource, StorageMode, TextureDescriptor,
        TextureSwizzle, TextureSwizzleChannels, TextureType, TextureUsage,
    };
    use ash::vk;
    use metal2vulkan_devtools::{engine_dir, llvm_assemble, AssembleError};

    /// The texture's every texel, in bytes.
    const TEXEL: [u8; 4] = [0x40, 0x80, 0xc0, 0x20];
    /// The texture's side.
    const SIDE: u32 = 4;
    /// The points sampled: a texel centre inside, and a texel past the left and right edges.
    const COORDS: [[f32; 2]; 3] = [[0.375, 0.375], [-0.5, 0.375], [1.5, 0.375]];
    /// The `uint4`s the kernel writes a thread; the runtime sampler's is the third.
    const PER_THREAD: usize = 5;
    const RUNTIME_SAMPLER: usize = 2;
    /// `a2_border_color`'s `uint4` of its float texture through `border_black`.
    const BORDER_BLACK: usize = 1;

    /// The owned fixture `name`'s library, or None where `llvm-as` is missing.
    fn library(name: &str) -> Option<Library> {
        match llvm_assemble(&engine_dir().join(format!("{name}.ll"))) {
            Ok(bitcode) => Some(Library::from_air(&bitcode).expect("the library")),
            Err(AssembleError::Missing(why)) => {
                eprintln!("SKIPPED: {why}");
                None
            }
            Err(error) => panic!("{name}.ll: {error:?}"),
        }
    }

    /// A compute pipeline state of `library`'s kernel `name`, its numbered samplers at
    /// `first..`.
    fn kernel(device: &Device, library: &Library, name: &str, first: u32) -> ComputePipelineState {
        let function = library.function(name).expect("the kernel");
        let descriptor = ComputePipelineDescriptor::new(StageSource::Function(
            FunctionDescriptor::new(function),
        ));
        (device.new_compute_pipeline_state(&descriptor, |numbered| {
            Ok((first..).take(numbered.len()).collect())
        }))
        .unwrap_or_else(|error| panic!("{name}: the state: {error}"))
    }

    /// Whether the device lists `VK_EXT_border_color_swizzle` and reports
    /// `borderColorSwizzleFromImage`, read from the device itself.
    fn from_image(device: &Device) -> bool {
        let core = device.core();
        let instance = core.instance.vk();
        let name = ash::ext::border_color_swizzle::NAME;
        // SAFETY: the physical device is the instance's.
        let listed = unsafe { instance.enumerate_device_extension_properties(core.physical) }
            .expect("the device extensions")
            .iter()
            .any(|properties| properties.extension_name_as_c_str() == Ok(name));
        if !listed {
            return false;
        }
        let mut swizzle = vk::PhysicalDeviceBorderColorSwizzleFeaturesEXT::default();
        let mut features = vk::PhysicalDeviceFeatures2::default().push_next(&mut swizzle);
        // SAFETY: as above, the device lists the extension, and `swizzle` lives through the call.
        unsafe { instance.get_physical_device_features2(core.physical, &mut features) };
        swizzle.border_color_swizzle_from_image == vk::TRUE
    }

    /// A sampler descriptor of nearest filters whose `s` and `t` take `mode`, with `border`.
    fn descriptor(mode: SamplerAddressMode, border: SamplerBorderColor) -> SamplerDescriptor {
        let mut d = SamplerDescriptor::new();
        d.s_address_mode = mode;
        d.t_address_mode = mode;
        d.border_color = border;
        d
    }

    #[test]
    fn every_device_samples_an_opaque_black_border_through_a_swizzled_view() {
        let (Some(library), Some(constexprs)) =
            (library("a2_clamp_to_zero"), library("a2_border_color"))
        else {
            return;
        };
        each_device("border_swizzle", |name, _, device| {
            let textures = (device.grant_indices(IndexClass::Texture, 1, 2))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let samplers = (device.grant_indices(IndexClass::Sampler, 1, 8))
                .unwrap_or_else(|error| panic!("{name}: {error}"));

            // The note, decided by the descriptor on every device.
            use SamplerAddressMode::{ClampToBorderColor, ClampToEdge, ClampToZero};
            use SamplerBorderColor::{OpaqueBlack, TransparentBlack};
            let note = Some(PrecisionNote::SwizzledOpaqueBlackBorder);
            for (mode, border, expected) in [
                (ClampToBorderColor, OpaqueBlack, note),
                (ClampToZero, OpaqueBlack, note),
                (ClampToBorderColor, TransparentBlack, None),
                (ClampToEdge, OpaqueBlack, None),
            ] {
                let state = (device.new_sampler_state(&descriptor(mode, border), 2))
                    .unwrap_or_else(|error| panic!("{name}: {error}"));
                assert_eq!(
                    state.precision_note(),
                    expected,
                    "{name}: {mode:?} {border:?}"
                );
            }
            let sampler = (device
                .new_sampler_state(&descriptor(ClampToBorderColor, OpaqueBlack), 1))
            .unwrap_or_else(|error| panic!("{name}: {error}"));

            let mut d =
                TextureDescriptor::new(TextureType::Type2D, PixelFormat::RGBA8Unorm, SIDE, SIDE);
            d.usage = TextureUsage::SHADER_READ | TextureUsage::PIXEL_FORMAT_VIEW;
            d.storage_mode = StorageMode::Shared;
            let texture =
                (device.new_texture(&d, Some(1))).unwrap_or_else(|error| panic!("{name}: {error}"));
            let whole = Region {
                origin: [0; 3],
                size: [SIDE, SIDE, 1],
            };
            let bytes = TEXEL.repeat((SIDE * SIDE) as usize);
            (texture.replace_region(whole, 0, 0, HostBytes::from(&bytes[..]), 16, 64))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            use TextureSwizzle::{Alpha, Green, Red, Zero};
            let swizzle = TextureSwizzleChannels {
                red: Green,
                green: Alpha,
                blue: Zero,
                alpha: Red,
            };
            let view = (texture.new_texture_view_with_swizzle(
                PixelFormat::RGBA8Unorm,
                TextureType::Type2D,
                0..1,
                0..1,
                swizzle,
                Some(2),
            ))
            .unwrap_or_else(|error| panic!("{name}: the view: {error}"));

            // Each sampler a state numbers at an index of its own: `a2_clamp_to_zero`'s two
            // constexpr samplers at 3 and 4, `a2_border_color`'s four at 5 to 8.
            let state = kernel(&device, &library, "a2_clamp_to_zero", 3);
            let border_state = kernel(&device, &constexprs, "a2_border_color", 5);
            let noted = |state: &ComputePipelineState| {
                (state.precision_notes()).contains(&PrecisionNote::SwizzledOpaqueBlackBorder)
            };
            assert!(
                !noted(&state),
                "{name}: transparent-black constexpr borders"
            );
            assert!(
                noted(&border_state),
                "{name}: an opaque-black constexpr border"
            );

            let mut arena = Arena::new(&device);
            let coords = arena.buffer();
            let out = arena.buffer();
            let border_out = arena.buffer();
            let points: Vec<u8> = (COORDS.iter())
                .flatten()
                .flat_map(|v| v.to_le_bytes())
                .collect();
            let contents = coords.contents().expect("a Shared buffer");
            // SAFETY: the buffer holds 256 bytes, more than the 24 written, and no work is pending.
            unsafe { (points.as_ptr()).copy_to_nonoverlapping(contents.as_ptr(), points.len()) };

            let queue = device.new_command_queue(1).expect("a queue");
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let mut encoder =
                (command_buffer.compute_command_encoder(DispatchType::Serial)).expect("an encoder");
            encoder
                .set_compute_pipeline_state(&state)
                .expect("the state");
            // The integer textures' slots hold nil: their samples are not read.
            encoder.set_texture(Some(&view), 0).expect("the view");
            encoder
                .set_sampler_state(Some(&sampler), 0)
                .expect("the sampler");
            encoder.set_buffer(Some(&coords), 0, 0).expect("the points");
            encoder.set_buffer(Some(&out), 0, 1).expect("the output");
            (encoder.dispatch_threads([COORDS.len() as u32, 1, 1], [COORDS.len() as u32, 1, 1]))
                .unwrap_or_else(|error| panic!("{name}: the dispatch: {error}"));
            // One thread of `a2_border_color` at the point past the left edge: its ten `uint4`s
            // fit the buffer.
            (encoder.set_compute_pipeline_state(&border_state)).expect("the border state");
            encoder.set_texture(Some(&view), 0).expect("the view");
            encoder.set_buffer(Some(&coords), 8, 0).expect("the point");
            (encoder.set_buffer(Some(&border_out), 0, 1)).expect("the output");
            (encoder.dispatch_threads([1, 1, 1], [1, 1, 1]))
                .unwrap_or_else(|error| panic!("{name}: the border dispatch: {error}"));
            encoder.end_encoding();
            command_buffer.commit().expect("a commit");
            finished("border swizzle", &command_buffer);
            assert_eq!(command_buffer.error(), None, "{name}");

            let contents = out.contents().expect("a Shared buffer");
            // SAFETY: the dispatch that wrote the buffer finished, and its writes reach the host;
            // the buffer holds 256 bytes, the kernel's three threads write 240.
            let words = unsafe {
                std::slice::from_raw_parts(contents.as_ptr().cast::<f32>(), 4 * PER_THREAD * 3)
            };
            let sample = |thread: usize| -> [f32; 4] {
                let at = 4 * (PER_THREAD * thread + RUNTIME_SAMPLER);
                std::array::from_fn(|c| words[at + c])
            };
            let unorm = |byte: u8| f32::from(byte) / 255.0;
            let inside = [unorm(TEXEL[1]), unorm(TEXEL[3]), 0.0, unorm(TEXEL[0])];
            let got = sample(0);
            assert!(
                (got.iter().zip(inside)).all(|(g, w)| (g - w).abs() <= 1.0 / 512.0),
                "{name}: the texel inside through the swizzle: {got:?}, Metal's {inside:?}"
            );
            let metal = [0.0, 1.0, 0.0, 0.0];
            let contents = border_out.contents().expect("a Shared buffer");
            // SAFETY: as above; the one thread writes 160 of the buffer's 256 bytes.
            let constexpr = unsafe {
                let words = contents.as_ptr().cast::<f32>().add(4 * BORDER_BLACK);
                std::array::from_fn::<f32, 4, _>(|c| *words.add(c))
            };
            let borders = [sample(1), sample(2), constexpr];
            let reported = from_image(&device);
            let name_of =
                |n: &std::ffi::CString| n.as_c_str() == ash::ext::border_color_swizzle::NAME;
            let enabled = device.extensions().iter().any(name_of);
            let exact = borders.iter().all(|&border| border == metal);
            // Printed before the checks, so a failing run shows the device's answer.
            println!(
                "{name}: borderColorSwizzleFromImage {reported}, the extension enabled {enabled}: \
                 the border through the swizzle {borders:?}, Metal's {metal:?}, exact {exact}"
            );
            assert!(
                exact || !reported,
                "{name}: the border through the swizzle {borders:?}, Metal's {metal:?}"
            );
            assert_eq!(
                enabled, reported,
                "{name}: the extension is enabled where the device reports it from the image"
            );
            drop((
                command_buffer,
                queue,
                state,
                border_state,
                view,
                texture,
                sampler,
            ));
            for grant in [textures, samplers] {
                (device.release_indices(grant)).unwrap_or_else(|error| panic!("{name}: {error}"));
            }
        });
    }
}
