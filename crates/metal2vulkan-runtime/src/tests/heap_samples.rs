//! The samples of the textures an argument buffer holds, under a device cfg on every device of the
//! cfg's kind, checked exactly: the owned fixture `examples/heap_samples.ll` (its source
//! `heap_samples.metal`) holds `hs_sample`, whose thread `(x, y)` samples texel `(x, y)` of a
//! `half` texture and of a `float` one at its center, each through a constexpr nearest sampler and
//! through a sampler slot bound a nearest sampler state. Metal answers each texel's own value; a
//! `half` texture's sample of an `RGBA16Float` texel whose channels a `half` holds exactly is that
//! value too.
//!
//! `llvm-as` assembles the fixture; without it the test skips visibly.

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

    /// The first texture index the test grants, the constexpr sampler's index, and the sampler
    /// state's.
    const TEXTURES: u32 = 4096;
    const CONSTEXPR: u32 = 1;
    const SLOT: u32 = 2;

    /// Texel `(x, y)` of both textures, at `y * 2 + x`: each channel a different value a `half`
    /// holds exactly.
    const TEXELS: [[f32; 4]; 4] = [
        [0.25, 0.5, 0.75, 1.0],
        [1.5, 2.5, 3.5, 4.5],
        [-1.0, -2.0, -4.0, -8.0],
        [0.125, 16.0, 0.0625, 32.0],
    ];

    /// The fixture's library, or None where `llvm-as` is missing.
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

    /// `x`'s `half` bits, for a value a `half` holds exactly: zero, or a normal one.
    fn half(x: f32) -> u16 {
        if x == 0.0 {
            return 0;
        }
        let bits = x.to_bits();
        let exponent = ((bits >> 23) & 0xff) as i32 - 127 + 15;
        assert!((1..31).contains(&exponent) && bits & 0x1fff == 0, "{x}");
        ((bits >> 16) & 0x8000) as u16 | ((exponent as u16) << 10) | ((bits >> 13) & 0x3ff) as u16
    }

    /// A 2 by 2 texture of `format` at texture index `index`, holding `TEXELS`, each channel's
    /// bytes `channel` gives.
    fn texture(
        name: &str,
        device: &Device,
        (format, unit): (PixelFormat, u64),
        index: u32,
        channel: fn(f32) -> Vec<u8>,
    ) -> Texture {
        let mut d = TextureDescriptor::new(TextureType::Type2D, format, 2, 2);
        d.usage = TextureUsage::SHADER_READ;
        let texture =
            (device.new_texture(&d, Some(index))).unwrap_or_else(|error| panic!("{name}: {error}"));
        let bytes: Vec<u8> = TEXELS.iter().flatten().flat_map(|&v| channel(v)).collect();
        let region = Region {
            origin: [0; 3],
            size: [2, 2, 1],
        };
        (texture.replace_region(
            region,
            0,
            0,
            crate::HostBytes::from(&bytes[..]),
            2 * unit,
            4 * unit,
        ))
        .unwrap_or_else(|error| panic!("{name}: {error}"));
        texture
    }

    /// A Shared buffer at byte `at` of `block`, holding `bytes`.
    fn shared(device: &Device, block: &Block, at: u64, bytes: &[u8]) -> Buffer {
        let buffer =
            (device.new_buffer(block.id, at, bytes.len() as u64, Some(0))).expect("a buffer");
        let contents = buffer.contents().expect("a Shared buffer");
        // SAFETY: the range lies in the buffer, which is alive, and no GPU work uses it.
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), contents.as_ptr(), bytes.len()) };
        buffer
    }

    /// The `half` and the `float` texture an argument buffer holds, each sampled through a
    /// constexpr nearest sampler and through a sampler slot: every answer is its texel's value.
    #[test]
    fn every_device_samples_the_half_and_float_textures_an_argument_buffer_holds() {
        let Some(library) = library() else {
            return;
        };
        each_device("heap_samples", |name, _, device| {
            let _grants = [
                (IndexClass::Texture, TEXTURES, 2),
                (IndexClass::Sampler, CONSTEXPR, 2),
            ]
            .map(|(class, first, count)| {
                (device.grant_indices(class, first, count))
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
            .unwrap_or_else(|error| panic!("{name}: the translator made no pipeline: {error}"));
            let halves = texture(
                name,
                &device,
                (PixelFormat::RGBA16Float, 8),
                TEXTURES,
                |v| half(v).to_le_bytes().to_vec(),
            );
            let floats = texture(
                name,
                &device,
                (PixelFormat::RGBA32Float, 16),
                TEXTURES + 1,
                |v| v.to_le_bytes().to_vec(),
            );
            let sampler = (device.new_sampler_state(&SamplerDescriptor::new(), SLOT))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let block = (device.take_block(MemoryClass::Shared, 65_536)).expect("a Shared block");
            let ids: Vec<u8> = [&halves, &floats]
                .iter()
                .flat_map(|t| t.gpu_resource_id().to_bits().to_le_bytes())
                .collect();
            let held = shared(&device, &block, 0, &ids);
            let out = shared(&device, &block, 256, &[0; 256]);
            let queue = device.new_command_queue(1).expect("a queue");
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let mut encoder =
                (command_buffer.compute_command_encoder(DispatchType::Serial)).expect("an encoder");
            encoder.set_compute_pipeline_state(&state).expect("a state");
            encoder.set_buffer(Some(&held), 0, 0).expect("held");
            encoder.set_buffer(Some(&out), 0, 1).expect("out");
            encoder
                .set_sampler_state(Some(&sampler), 0)
                .expect("the slot's sampler");
            for texture in [&halves, &floats] {
                (encoder.use_resource(Resource::Texture(texture), ResourceUsage::READ))
                    .expect("a texture");
            }
            (encoder.dispatch_threadgroups([1, 1, 1], [2, 2, 1]))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            encoder.end_encoding();
            command_buffer.commit().expect("a commit");
            finished(name, &command_buffer);
            assert_eq!(command_buffer.error(), None, "{name}");
            let contents = out.contents().expect("a Shared buffer");
            // SAFETY: the dispatch that wrote the buffer completed, and its writes reach the host;
            // the buffer holds 16 float4s.
            let got =
                unsafe { std::slice::from_raw_parts(contents.as_ptr().cast::<[f32; 4]>(), 16) };
            let reads = [
                "half, constexpr",
                "float, constexpr",
                "half, slot",
                "float, slot",
            ];
            for (texel, want) in TEXELS.iter().enumerate() {
                for (k, read) in reads.iter().enumerate() {
                    assert_eq!(
                        &got[4 * texel + k],
                        want,
                        "{name}: texel {texel}'s {read} sample"
                    );
                }
            }
            println!("{name}: every sample of both textures answered its texel");
        });
    }
}
