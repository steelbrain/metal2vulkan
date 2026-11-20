//! A texture view's swizzle in kernels, under a device cfg on every device of the cfg's kind,
//! through the owned fixture `rrp_swizzled_storage` over a 2 by 2 `RGBA8Unorm` texture of
//! shader-read and shader-write usage whose every texel is `(10, 20, 30, 40)` in bytes, and a view
//! of it made with the swizzle blue, green, red, one:
//!
//! - an `access::read` read through the view answers the texel through the swizzle,
//!   `(30, 20, 10, 255)`, as Apple defines the swizzle for reads;
//! - an `access::write` write through the view stores the value unswizzled at each texel;
//! - an `access::read_write` slot bound to the view is `Gap::SwizzledReadWrite`, found at the
//!   dispatch, with nothing recorded; the same kernel over the texture itself reads its texel.
//!
//! `llvm-as` assembles the kernels; without it the test skips visibly.

#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod on_device {
    use crate::tests::command::on_device::{each_device, finished, Arena};
    use crate::{
        ComputePipelineDescriptor, ComputePipelineState, Device, DispatchType, Error,
        FunctionDescriptor, Gap, IndexClass, Library, PixelFormat, Region, StageSource,
        StorageMode, Texture, TextureDescriptor, TextureSwizzle, TextureSwizzleChannels,
        TextureType, TextureUsage,
    };
    use metal2vulkan_devtools::{engine_dir, llvm_assemble, AssembleError};

    /// The texture's every texel, in bytes.
    const TEXEL: [u8; 4] = [10, 20, 30, 40];

    /// The fixture's library, or None where `llvm-as` is missing.
    fn library() -> Option<Library> {
        match llvm_assemble(&engine_dir().join("rrp_swizzled_storage.ll")) {
            Ok(bitcode) => Some(Library::from_air(&bitcode).expect("the library")),
            Err(AssembleError::Missing(why)) => {
                eprintln!("SKIPPED: {why}");
                None
            }
            Err(error) => panic!("rrp_swizzled_storage.ll: {error:?}"),
        }
    }

    /// The kernel `name`'s state, each sampler it numbers served by sampler 0.
    fn state(device: &Device, library: &Library, name: &str) -> ComputePipelineState {
        let function = library.function(name).expect("the kernel");
        let descriptor = ComputePipelineDescriptor::new(StageSource::Function(
            FunctionDescriptor::new(function),
        ));
        (device.new_compute_pipeline_state(&descriptor, |samplers| Ok(vec![0; samplers.len()])))
            .unwrap_or_else(|error| panic!("{name}: {error}"))
    }

    /// One dispatch of `state` over a 2 by 2 grid with `texture` at slot 0 and `buffer` at slots 0
    /// and 1, committed and finished where the encoder takes it.
    fn dispatch(
        device: &Device,
        state: &ComputePipelineState,
        texture: &Texture,
        buffer: &crate::Buffer,
    ) -> Result<(), Error> {
        let queue = device.new_command_queue(1).expect("a queue");
        let command_buffer = queue.command_buffer().expect("a command buffer");
        let mut encoder =
            (command_buffer.compute_command_encoder(DispatchType::Serial)).expect("an encoder");
        encoder
            .set_compute_pipeline_state(state)
            .expect("the state");
        encoder.set_texture(Some(texture), 0).expect("a texture");
        for index in [0, 1] {
            encoder
                .set_buffer(Some(buffer), 0, index)
                .expect("a buffer");
        }
        let dispatched = encoder.dispatch_threadgroups([1, 1, 1], [2, 2, 1]);
        encoder.end_encoding();
        command_buffer.commit().expect("a commit");
        finished("swizzled kernels", &command_buffer);
        assert_eq!(command_buffer.error(), None);
        dispatched
    }

    /// The first four floats of `buffer`, as bytes (x 255).
    fn texel_of(buffer: &crate::Buffer) -> [u8; 4] {
        let contents = buffer.contents().expect("a Shared buffer");
        // SAFETY: the dispatch that wrote the buffer finished, and its writes reach the host.
        let floats = unsafe { std::slice::from_raw_parts(contents.as_ptr().cast::<f32>(), 4) };
        std::array::from_fn(|c| (floats[c] * 255.0).round() as u8)
    }

    #[test]
    fn every_device_reads_through_a_view_s_swizzle_and_writes_past_it() {
        let Some(library) = library() else {
            return;
        };
        each_device("swizzled kernels", |name, _, device| {
            let grant = (device.grant_indices(IndexClass::Texture, 1, 2))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let mut d = TextureDescriptor::new(TextureType::Type2D, PixelFormat::RGBA8Unorm, 2, 2);
            d.usage = TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE;
            d.storage_mode = StorageMode::Shared;
            let texture =
                (device.new_texture(&d, Some(1))).unwrap_or_else(|error| panic!("{name}: {error}"));
            let whole = Region {
                origin: [0; 3],
                size: [2, 2, 1],
            };
            let fill = |texture: &Texture| {
                (texture.replace_region(
                    whole,
                    0,
                    0,
                    crate::HostBytes::from(&TEXEL.repeat(4)[..]),
                    8,
                    16,
                ))
                .unwrap_or_else(|error| panic!("{name}: {error}"))
            };
            fill(&texture);
            use TextureSwizzle::{Blue, Green, One, Red};
            let swizzle = TextureSwizzleChannels {
                red: Blue,
                green: Green,
                blue: Red,
                alpha: One,
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
            let mut arena = Arena::new(&device);
            let out = arena.buffer();

            let read = state(&device, &library, "rrp_sw_read_f");
            (dispatch(&device, &read, &view, &out)).unwrap_or_else(|e| panic!("{name}: {e}"));
            assert_eq!(
                texel_of(&out),
                [30, 20, 10, 255],
                "{name}: a read through the swizzle"
            );

            let read_write = state(&device, &library, "rrp_sw_rwread_f");
            match dispatch(&device, &read_write, &view, &out) {
                Err(Error::Unsupported(Gap::SwizzledReadWrite { slot: 0 })) => {}
                other => panic!("{name}: a read_write slot bound to the view: {other:?}"),
            }
            (dispatch(&device, &read_write, &texture, &out))
                .unwrap_or_else(|e| panic!("{name}: {e}"));
            assert_eq!(
                texel_of(&out),
                TEXEL,
                "{name}: a read_write read of the texture"
            );

            // The value buffer's first four floats are (0.2, 0.4, 0.6, 0.8).
            let value: Vec<u8> = [0.2f32, 0.4, 0.6, 0.8]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect();
            let contents = out.contents().expect("a Shared buffer");
            // SAFETY: the buffer holds 256 bytes, more than the 16 written, and no work is pending.
            unsafe { (value.as_ptr()).copy_to_nonoverlapping(contents.as_ptr(), value.len()) };
            let write = state(&device, &library, "rrp_sw_write_f");
            (dispatch(&device, &write, &view, &out)).unwrap_or_else(|e| panic!("{name}: {e}"));
            let mut bytes = [0u8; 16];
            (texture.get_bytes(&mut bytes, 8, 16, whole, 0, 0))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            assert_eq!(
                bytes.as_slice(),
                [51, 102, 153, 204].repeat(4).as_slice(),
                "{name}: a write through the view stores the value unswizzled"
            );
            drop((view, texture));
            (device.release_indices(grant)).unwrap_or_else(|error| panic!("{name}: {error}"));
            println!("{name}: a swizzled view read through, written past, read_write refused");
        });
    }
}
