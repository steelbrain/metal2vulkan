//! A texture with no storage element in kernels, under a device cfg on every device of the cfg's
//! kind, through the owned fixture `rrp_swizzled_storage` over a 2 by 2 `RGBA8Unorm_sRGB` texture
//! of shader-read and shader-write usage, a format Vulkan requires no device to store:
//!
//! - an `access::write` slot and an `access::read_write` slot bound to it are
//!   `Gap::StorageElement`, found at the dispatch, exactly where the device's optimal tiling holds
//!   neither the format's storage nor its storage writes without a format, and dispatch where it
//!   holds both;
//! - an `access::read` slot bound to it dispatches on every device.
//!
//! `llvm-as` assembles the kernels; without it the test skips visibly.

#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod on_device {
    use crate::tests::command::on_device::{each_device, finished, Arena};
    use crate::{
        ComputePipelineDescriptor, ComputePipelineState, Device, DispatchType, Error,
        FunctionDescriptor, Gap, IndexClass, Library, PixelFormat, StageSource, StorageMode,
        Texture, TextureDescriptor, TextureType, TextureUsage,
    };
    use ash::vk;
    use metal2vulkan_devtools::{engine_dir, llvm_assemble, AssembleError};

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

    /// The kernel `name`'s state.
    fn state(device: &Device, library: &Library, name: &str) -> ComputePipelineState {
        let function = library.function(name).expect("the kernel");
        let descriptor = ComputePipelineDescriptor::new(StageSource::Function(
            FunctionDescriptor::new(function),
        ));
        (device.new_compute_pipeline_state(&descriptor, |samplers| Ok(vec![0; samplers.len()])))
            .unwrap_or_else(|error| panic!("{name}: {error}"))
    }

    /// One dispatch of `state` over a 2 by 2 grid with `texture` at slot 0 and `buffer` at slots 0
    /// and 1, committed and finished: the encoder's answer.
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
        finished("storage elements", &command_buffer);
        assert_eq!(command_buffer.error(), None);
        dispatched
    }

    #[test]
    fn every_device_refuses_a_stored_slot_bound_to_a_texture_with_no_storage_element() {
        let Some(library) = library() else {
            return;
        };
        each_device("storage elements", |name, _, device| {
            let (_, optimal) = device.core().format_features(vk::Format::R8G8B8A8_SRGB);
            let stores = optimal.contains(
                vk::FormatFeatureFlags2::STORAGE_IMAGE
                    | vk::FormatFeatureFlags2::STORAGE_WRITE_WITHOUT_FORMAT,
            );
            let grant = (device.grant_indices(IndexClass::Texture, 1, 1))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let mut d =
                TextureDescriptor::new(TextureType::Type2D, PixelFormat::RGBA8UnormSRGB, 2, 2);
            d.usage = TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE;
            d.storage_mode = StorageMode::Shared;
            let texture =
                (device.new_texture(&d, Some(1))).unwrap_or_else(|error| panic!("{name}: {error}"));
            let mut arena = Arena::new(&device);
            let out = arena.buffer();
            for kernel in ["rrp_sw_write_f", "rrp_sw_rwread_f"] {
                let stored = state(&device, &library, kernel);
                match (stores, dispatch(&device, &stored, &texture, &out)) {
                    (true, Ok(())) => {}
                    (
                        false,
                        Err(Error::Unsupported(Gap::StorageElement {
                            slot: 0,
                            pixel_format: PixelFormat::RGBA8UnormSRGB,
                        })),
                    ) => {}
                    (stores, other) => {
                        panic!(
                            "{name}: {kernel}, the device storing the format {stores}: {other:?}"
                        )
                    }
                }
            }
            let read = state(&device, &library, "rrp_sw_read_f");
            (dispatch(&device, &read, &texture, &out)).unwrap_or_else(|e| panic!("{name}: {e}"));
            drop(texture);
            (device.release_indices(grant)).unwrap_or_else(|error| panic!("{name}: {error}"));
            println!("{name}: the device stores 8-bit sRGB {stores}; stored slots answered so");
        });
    }
}
