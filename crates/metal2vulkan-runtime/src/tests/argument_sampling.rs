//! Kernels that sample the textures an argument buffer holds, under a device cfg on every device of
//! the cfg's kind: `examples/heap_samples`' `hs_sample` samples a `half` and a `float` texture its
//! argument buffer holds through a constexpr nearest sampler, at the index the consumer gives it,
//! and through a nearest sampler state bound at a sampler slot. Each thread samples one texel at its
//! centre, and every answer is that texel's exact value.
//!
//! `llvm-as` assembles the kernel; without it the test skips visibly.

#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod on_device {
    use crate::tests::command::on_device::{each_device, finished};
    use crate::{
        Block, Buffer, ComputePipelineDescriptor, Device, DispatchType, FunctionDescriptor,
        IndexClass, Library, MemoryClass, NumberedSampler, PixelFormat, Region, Resource,
        ResourceUsage, SamplerDescriptor, StageSource, Texture, TextureDescriptor, TextureType,
        TextureUsage,
    };
    use metal2vulkan_devtools::{fixtures_dir, llvm_assemble, AssembleError};

    /// The side of each texture: one texel per thread of a 2 by 2 grid.
    const SIDE: u32 = 2;
    /// The constexpr sampler's index the test gives `hs_sample`.
    const CONSTEXPR: u32 = 2;
    /// The sampler state's index.
    const SLOT_SAMPLER: u32 = 1;

    /// Values a `half` holds exactly, with their `half` bits.
    const HALVES: [(f32, u16); 8] = [
        (0.125, 0x3000),
        (0.25, 0x3400),
        (0.5, 0x3800),
        (0.75, 0x3a00),
        (1.0, 0x3c00),
        (1.5, 0x3e00),
        (2.0, 0x4000),
        (4.0, 0x4400),
    ];

    /// Channel `c` of texel `i` (row by row) of the `half` texture, as an index of [`HALVES`].
    fn half_at(i: usize, c: usize) -> usize {
        (3 * i + c) % HALVES.len()
    }

    /// Channel `c` of texel `i` (row by row) of the `float` texture.
    fn float_at(i: usize, c: usize) -> f32 {
        -1.5 - (4 * i + c) as f32
    }

    /// The `hs_sample` kernel's library, or None where `llvm-as` is missing.
    fn library() -> Option<Library> {
        match llvm_assemble(&fixtures_dir().join("examples/heap_samples.ll")) {
            Ok(bitcode) => Some(Library::from_air(&bitcode).expect("the library")),
            Err(AssembleError::Missing(why)) => {
                eprintln!("SKIPPED: {why}");
                None
            }
            Err(error) => panic!("heap_samples.ll: {error:?}"),
        }
    }

    /// A 2D texture of `format` (`texel` bytes a texel), `SIDE` a side, shader-read, at texture
    /// index `index`, its bytes `bytes` row by row.
    fn texture(
        name: &str,
        device: &Device,
        (format, texel): (PixelFormat, u64),
        index: u32,
        bytes: &[u8],
    ) -> Texture {
        let mut d = TextureDescriptor::new(TextureType::Type2D, format, SIDE, SIDE);
        d.usage = TextureUsage::SHADER_READ;
        let texture =
            (device.new_texture(&d, Some(index))).unwrap_or_else(|error| panic!("{name}: {error}"));
        let whole = Region {
            origin: [0; 3],
            size: [SIDE, SIDE, 1],
        };
        let row = u64::from(SIDE) * texel;
        (texture.replace_region(
            whole,
            0,
            0,
            crate::HostBytes::from(bytes),
            row,
            row * u64::from(SIDE),
        ))
        .unwrap_or_else(|error| panic!("{name}: {error}"));
        texture
    }

    /// A Shared buffer of `bytes` at `offset` of `block`.
    fn shared(device: &Device, block: &Block, offset: u64, bytes: &[u8]) -> Buffer {
        let buffer =
            (device.new_buffer(block.id, offset, bytes.len() as u64, Some(0))).expect("a buffer");
        let contents = buffer.contents().expect("a Shared buffer");
        // SAFETY: the range lies in the buffer, which is alive, and no GPU work uses it.
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), contents.as_ptr(), bytes.len()) };
        buffer
    }

    /// The `count` float4s at the start of `buffer`, read once its writer completed.
    fn float4s(buffer: &Buffer, count: usize) -> Vec<[f32; 4]> {
        let contents = buffer.contents().expect("a Shared buffer");
        // SAFETY: the work that wrote the buffer completed, and its writes reach the host; the
        // buffer holds at least `count` float4s.
        let floats =
            unsafe { std::slice::from_raw_parts(contents.as_ptr().cast::<f32>(), 4 * count) };
        floats.chunks(4).map(|c| [c[0], c[1], c[2], c[3]]).collect()
    }

    /// Each thread of `hs_sample` samples texel `(x, y)` of the `half` and the `float` texture its
    /// argument buffer holds at its centre, through the constexpr nearest sampler and through the
    /// nearest sampler state at slot 0, and writes the four answers: each is the texel.
    #[test]
    fn every_device_samples_an_argument_buffers_textures_through_slot_and_constexpr_samplers() {
        let Some(library) = library() else {
            return;
        };
        each_device("argument_sampling", |name, _, device| {
            let grants = [
                (IndexClass::Texture, 1),
                (IndexClass::Sampler, SLOT_SAMPLER),
            ]
            .map(|(class, first)| {
                (device.grant_indices(class, first, 2))
                    .unwrap_or_else(|error| panic!("{name}: {error}"))
            });
            let function = library.function("hs_sample").expect("the kernel");
            let descriptor = ComputePipelineDescriptor::new(StageSource::Function(
                FunctionDescriptor::new(function),
            ));
            let state = (device.new_compute_pipeline_state(&descriptor, |samplers| {
                assert!(
                    matches!(samplers, [NumberedSampler::Constexpr(_)]),
                    "{name}: {samplers:?}"
                );
                Ok(vec![CONSTEXPR])
            }))
            .unwrap_or_else(|error| panic!("{name}: hs_sample: {error}"));
            let texels = (SIDE * SIDE) as usize;
            let halves: Vec<u8> = (0..texels)
                .flat_map(|i| (0..4).map(move |c| HALVES[half_at(i, c)].1))
                .flat_map(u16::to_le_bytes)
                .collect();
            let floats: Vec<u8> = (0..texels)
                .flat_map(|i| (0..4).map(move |c| float_at(i, c)))
                .flat_map(f32::to_le_bytes)
                .collect();
            let h = texture(name, &device, (PixelFormat::RGBA16Float, 8), 1, &halves);
            let f = texture(name, &device, (PixelFormat::RGBA32Float, 16), 2, &floats);
            let sampler = (device.new_sampler_state(&SamplerDescriptor::new(), SLOT_SAMPLER))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let block = (device.take_block(MemoryClass::Shared, 65_536)).expect("a Shared block");
            let ids: Vec<u8> = [&h, &f]
                .iter()
                .flat_map(|t| t.gpu_resource_id().to_bits().to_le_bytes())
                .collect();
            let held = shared(&device, &block, 0, &ids);
            let out = shared(&device, &block, 256, &[0; 16 * 16]);
            let queue = device.new_command_queue(1).expect("a queue");
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let mut encoder =
                (command_buffer.compute_command_encoder(DispatchType::Serial)).expect("an encoder");
            encoder.set_compute_pipeline_state(&state).expect("a state");
            encoder.set_buffer(Some(&held), 0, 0).expect("held");
            encoder.set_buffer(Some(&out), 0, 1).expect("out");
            (encoder.set_sampler_state(Some(&sampler), 0)).expect("the sampler");
            for texture in [&h, &f] {
                (encoder.use_resource(Resource::Texture(texture), ResourceUsage::READ))
                    .expect("a texture");
            }
            (encoder.dispatch_threadgroups([1, 1, 1], [SIDE, SIDE, 1]))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            encoder.end_encoding();
            command_buffer.commit().expect("a commit");
            finished(name, &command_buffer);
            assert_eq!(command_buffer.error(), None, "{name}");
            let read = float4s(&out, 4 * texels);
            for i in 0..texels {
                let half = [0, 1, 2, 3].map(|c| HALVES[half_at(i, c)].0);
                let float = [0, 1, 2, 3].map(|c| float_at(i, c));
                assert_eq!(
                    read[4 * i..4 * i + 4],
                    [half, float, half, float],
                    "{name}: texel {i}, through the constexpr sampler and the slot's"
                );
            }
            drop((command_buffer, queue, state, sampler, h, f));
            for grant in grants {
                (device.release_indices(grant)).unwrap_or_else(|error| panic!("{name}: {error}"));
            }
            println!(
                "{name}: an argument buffer's textures sampled through slot and constexpr samplers"
            );
        });
    }
}
