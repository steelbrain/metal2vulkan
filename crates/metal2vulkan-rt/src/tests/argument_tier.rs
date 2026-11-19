//! Three of the things Metal's argument buffers tier 2 adds over tier 1, under a device cfg on
//! every device of the cfg's kind, one test a piece (Apple's "Improving CPU performance by using
//! argument buffers", read against its tier 1 limits):
//!
//! - writable textures in an argument buffer: `a8r_buffers`' `a8r_held` reads a `read_write`
//!   member's texels, then writes it and a `write` member;
//! - an argument buffer the GPU writes: `ab_tier2`'s `ab_copy` copies a texture member from one
//!   argument buffer to another, and `ab_read` reads through the copy;
//! - an argument buffer the CPU cannot reach: a Private buffer a blit filled, which `ab_read`
//!   reads through.
//!
//! The program writes each texture member itself, as the texture's `gpuResourceID`, as tier 2's
//! layout lets it.
//!
//! `llvm-as` assembles the fixtures; without it the tests skip visibly.

#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod on_device {
    use crate::tests::command::on_device::{each_device, finished};
    use crate::{
        Block, Buffer, ComputePipelineDescriptor, ComputePipelineState, Device, DispatchType,
        FunctionDescriptor, IndexClass, Library, MemoryClass, PixelFormat, Region, Resource,
        ResourceUsage, StageSource, Texture, TextureDescriptor, TextureType, TextureUsage,
    };
    use metal2vulkan_devtools::{engine_dir, llvm_assemble, AssembleError};

    /// The fixture `name`'s library, or None where `llvm-as` is missing.
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

    /// The state of `entry`, each texture read's sampler at index 0.
    fn state(name: &str, device: &Device, library: &Library, entry: &str) -> ComputePipelineState {
        let function = library.function(entry).expect("the kernel");
        let descriptor = ComputePipelineDescriptor::new(StageSource::Function(
            FunctionDescriptor::new(function),
        ));
        (device.new_compute_pipeline_state(&descriptor, |samplers| Ok(vec![0; samplers.len()])))
            .unwrap_or_else(|error| panic!("{name}: {entry}: {error}"))
    }

    /// Shared buffers in one block, 256 bytes apart.
    struct Shared {
        block: Block,
        next: u64,
    }

    impl Shared {
        fn new(device: &Device) -> Shared {
            Shared {
                block: (device.take_block(MemoryClass::Shared, 65_536)).expect("a Shared block"),
                next: 0,
            }
        }

        /// A buffer holding `bytes`.
        fn of(&mut self, device: &Device, bytes: &[u8]) -> Buffer {
            let len = bytes.len() as u64;
            let buffer =
                (device.new_buffer(self.block.id, self.next, len, Some(0))).expect("a buffer");
            self.next += len.next_multiple_of(256);
            let contents = buffer.contents().expect("a Shared buffer");
            // SAFETY: the range lies in the buffer, which is alive, and no GPU work uses it.
            unsafe {
                std::ptr::copy_nonoverlapping(bytes.as_ptr(), contents.as_ptr(), bytes.len())
            };
            buffer
        }
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

    /// The bytes of `values`.
    fn bytes(values: &[[f32; 4]]) -> Vec<u8> {
        values
            .iter()
            .flatten()
            .flat_map(|v| v.to_le_bytes())
            .collect()
    }

    /// A 2D `RGBA32Float` texture of `side` texels a side at texture index `index`, its texels
    /// `texels` row by row.
    fn texture(name: &str, device: &Device, side: u32, index: u32, texels: &[[f32; 4]]) -> Texture {
        let mut d =
            TextureDescriptor::new(TextureType::Type2D, PixelFormat::RGBA32Float, side, side);
        d.usage = TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE;
        let texture =
            (device.new_texture(&d, Some(index))).unwrap_or_else(|error| panic!("{name}: {error}"));
        let row = u64::from(side) * 16;
        (texture.replace_region(
            whole(side),
            0,
            0,
            crate::HostBytes::from(&bytes(texels)[..]),
            row,
            row * u64::from(side),
        ))
        .unwrap_or_else(|error| panic!("{name}: {error}"));
        texture
    }

    /// The region of a `side` by `side` texture.
    fn whole(side: u32) -> Region {
        Region {
            origin: [0; 3],
            size: [side, side, 1],
        }
    }

    /// The texels of a `side` by `side` `RGBA32Float` texture.
    fn texels(name: &str, texture: &Texture, side: u32) -> Vec<[f32; 4]> {
        let row = u64::from(side) * 16;
        let mut read = vec![0u8; (row * u64::from(side)) as usize];
        (texture.get_bytes(&mut read, row, row * u64::from(side), whole(side), 0, 0))
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        (read.chunks(16))
            .map(|t| {
                let f = |at: usize| f32::from_le_bytes(t[at..at + 4].try_into().expect("4 bytes"));
                [f(0), f(4), f(8), f(12)]
            })
            .collect()
    }

    /// One texel's value per texture of the tests, each channel a different exact float.
    const TEXELS: [[f32; 4]; 4] = [
        [0.25, 0.5, 0.75, 1.0],
        [1.5, 2.5, 3.5, 4.5],
        [-1.0, -2.0, -4.0, -8.0],
        [0.125, 16.0, 0.0625, 32.0],
    ];

    /// A `read_write` and a `write` texture held in an argument buffer: the kernel reads the
    /// former's texels, then writes `color[0]` into it and `color[1]` into the latter.
    #[test]
    fn every_device_reads_and_writes_the_writable_textures_an_argument_buffer_holds() {
        let Some(library) = library("a8r_buffers") else {
            return;
        };
        each_device("argument_tier_writable", |name, _, device| {
            let state = state(name, &device, &library, "a8r_held");
            let grant = (device.grant_indices(IndexClass::Texture, 1, 2))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let rw = texture(name, &device, 2, 1, &TEXELS);
            let w = texture(name, &device, 2, 2, &[[0.0; 4]; 4]);
            let mut shared = Shared::new(&device);
            let color = [[0.5, 0.25, 0.125, 1.0], [8.0, 4.0, 2.0, 1.0]];
            let colors = shared.of(&device, &bytes(&color));
            let ids: Vec<u8> = [&rw, &w]
                .iter()
                .flat_map(|t| t.gpu_resource_id().to_bits().to_le_bytes())
                .collect();
            let held = shared.of(&device, &ids);
            let out = shared.of(&device, &[0; 160]);
            let queue = device.new_command_queue(1).expect("a queue");
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let mut encoder =
                (command_buffer.compute_command_encoder(DispatchType::Serial)).expect("an encoder");
            encoder.set_compute_pipeline_state(&state).expect("a state");
            encoder.set_buffer(Some(&colors), 0, 0).expect("color");
            encoder.set_buffer(Some(&out), 0, 1).expect("out");
            encoder.set_buffer(Some(&held), 0, 2).expect("held");
            let usage = ResourceUsage::READ | ResourceUsage::WRITE;
            for texture in [&rw, &w] {
                (encoder.use_resource(Resource::Texture(texture), usage)).expect("a texture");
            }
            (encoder.dispatch_threadgroups([1, 1, 1], [2, 2, 1]))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            encoder.end_encoding();
            command_buffer.commit().expect("a commit");
            finished(name, &command_buffer);
            assert_eq!(command_buffer.error(), None, "{name}");
            let read = float4s(&out, 10);
            for (texel, want) in TEXELS.iter().enumerate() {
                let (x, y) = (texel % 2, texel / 2);
                assert_eq!(
                    &read[y * 8 + x],
                    want,
                    "{name}: the read of texel ({x}, {y})"
                );
            }
            assert_eq!(
                texels(name, &rw, 2),
                [color[0]; 4],
                "{name}: the read_write member"
            );
            assert_eq!(
                texels(name, &w, 2),
                [color[1]; 4],
                "{name}: the write member"
            );
            drop((rw, w, command_buffer));
            (device.release_indices(grant)).unwrap_or_else(|error| panic!("{name}: {error}"));
            println!("{name}: an argument buffer's read_write and write textures read and wrote");
        });
    }

    /// A kernel copies a texture member from one argument buffer to another, and the next
    /// dispatch reads the texture through the copy.
    #[test]
    fn every_device_reads_through_an_argument_buffer_the_gpu_wrote() {
        let Some(library) = library("ab_tier2") else {
            return;
        };
        each_device("argument_tier_gpu_written", |name, _, device| {
            let copy = state(name, &device, &library, "ab_copy");
            let read = state(name, &device, &library, "ab_read");
            let grant = (device.grant_indices(IndexClass::Texture, 1, 1))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let texture = texture(name, &device, 1, 1, &TEXELS[..1]);
            let mut shared = Shared::new(&device);
            let from = shared.of(&device, &texture.gpu_resource_id().to_bits().to_le_bytes());
            let to = shared.of(&device, &[0; 8]);
            let out = shared.of(&device, &[0; 16]);
            let queue = device.new_command_queue(1).expect("a queue");
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let mut encoder =
                (command_buffer.compute_command_encoder(DispatchType::Serial)).expect("an encoder");
            encoder.set_compute_pipeline_state(&copy).expect("a state");
            encoder.set_buffer(Some(&from), 0, 0).expect("from");
            encoder.set_buffer(Some(&to), 0, 1).expect("to");
            (encoder.dispatch_threadgroups([1, 1, 1], [1, 1, 1]))
                .unwrap_or_else(|error| panic!("{name}: the copy: {error}"));
            encoder.set_compute_pipeline_state(&read).expect("a state");
            encoder.set_buffer(Some(&to), 0, 0).expect("held");
            encoder.set_buffer(Some(&out), 0, 1).expect("out");
            (encoder.use_resource(Resource::Texture(&texture), ResourceUsage::READ))
                .expect("a texture");
            (encoder.dispatch_threadgroups([1, 1, 1], [1, 1, 1]))
                .unwrap_or_else(|error| panic!("{name}: the read: {error}"));
            encoder.end_encoding();
            command_buffer.commit().expect("a commit");
            finished(name, &command_buffer);
            assert_eq!(command_buffer.error(), None, "{name}");
            assert_eq!(float4s(&out, 1), [TEXELS[0]], "{name}");
            drop((texture, command_buffer));
            (device.release_indices(grant)).unwrap_or_else(|error| panic!("{name}: {error}"));
            println!("{name}: a texture read through an argument buffer a kernel wrote");
        });
    }

    /// An argument buffer in Private memory, which a blit filled from a Shared one: the kernel
    /// reads the texture through it.
    #[test]
    fn every_device_reads_through_a_private_argument_buffer() {
        let Some(library) = library("ab_tier2") else {
            return;
        };
        each_device("argument_tier_private", |name, _, device| {
            let read = state(name, &device, &library, "ab_read");
            let grant = (device.grant_indices(IndexClass::Texture, 1, 1))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let texture = texture(name, &device, 1, 1, &TEXELS[1..2]);
            let mut shared = Shared::new(&device);
            let staged = shared.of(&device, &texture.gpu_resource_id().to_bits().to_le_bytes());
            let out = shared.of(&device, &[0; 16]);
            let private_block =
                (device.take_block(MemoryClass::Private, 256)).expect("a Private block");
            let held = (device.new_buffer(private_block.id, 0, 8, Some(0))).expect("a buffer");
            assert!(
                held.contents().is_none(),
                "{name}: a Private buffer the host reaches"
            );
            let queue = device.new_command_queue(1).expect("a queue");
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let mut blit = command_buffer
                .blit_command_encoder()
                .expect("a blit encoder");
            (blit.copy_from_buffer_to_buffer(&staged, 0, &held, 0, 8))
                .unwrap_or_else(|error| panic!("{name}: the fill: {error}"));
            blit.end_encoding();
            let mut encoder =
                (command_buffer.compute_command_encoder(DispatchType::Serial)).expect("an encoder");
            encoder.set_compute_pipeline_state(&read).expect("a state");
            encoder.set_buffer(Some(&held), 0, 0).expect("held");
            encoder.set_buffer(Some(&out), 0, 1).expect("out");
            (encoder.use_resource(Resource::Texture(&texture), ResourceUsage::READ))
                .expect("a texture");
            (encoder.dispatch_threadgroups([1, 1, 1], [1, 1, 1]))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            encoder.end_encoding();
            command_buffer.commit().expect("a commit");
            finished(name, &command_buffer);
            assert_eq!(command_buffer.error(), None, "{name}");
            assert_eq!(float4s(&out, 1), [TEXELS[1]], "{name}");
            drop((texture, held, command_buffer));
            (device.release_indices(grant)).unwrap_or_else(|error| panic!("{name}: {error}"));
            println!("{name}: a texture read through a Private argument buffer");
        });
    }
}
