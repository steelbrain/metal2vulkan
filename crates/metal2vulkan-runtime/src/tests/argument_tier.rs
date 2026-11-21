//! What Metal's argument buffers tier 2 adds over tier 1, under a device cfg on every device of the
//! cfg's kind, one test a piece (Apple's "Improving CPU performance by using argument buffers",
//! read against its tier 1 limits):
//!
//! - writable textures in an argument buffer: `a8r_buffers`' `a8r_held` reads a `read_write`
//!   member's texels, then writes it and a `write` member;
//! - an argument buffer the GPU writes: `ab_tier2`'s `ab_copy` copies a texture member from one
//!   argument buffer to another, and `ab_read` reads through the copy;
//! - an argument buffer the CPU cannot reach: a Private buffer a blit filled, which `ab_read`
//!   reads through;
//! - pointer indexing, and an argument buffer that points at others: `ab_indexed`'s thread `g`
//!   reads the texture of element `g` of an array of argument buffers a pointer member reaches;
//! - buffers behind device addresses (`ab_pointers`): `ab_sum` adds two buffers an argument
//!   buffer's two pointer members reach, `ab_address` reads a buffer through a `gpuAddress` the
//!   CPU wrote into a constant buffer, and `ab_slot_address` stores a slot pointer converted to an
//!   integer, which is the slot's `gpuAddress` plus its offsets.
//!
//! A buffer read by address alone stays readable past its block's release until the read retires
//! (a GPU hazard), and a release under work committed before it waits for that work.
//!
//! The program writes each texture member itself, as the texture's `gpuResourceID`, and each
//! pointer member as the buffer's `gpu_address` plus the offset an encoder's `setBuffer:offset:`
//! names, as tier 2's layout lets it. The floor holds every feature tier 2 rests on (buffer device addresses, and the
//! set-1 arrays' runtime descriptor arrays, partial binding and update-after-bind of sampled and
//! storage images), so the tier is a floor fact.
//!
//! `llvm-as` assembles the fixtures; without it the tests skip visibly.

#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod on_device {
    use crate::tests::command::on_device::{each_device, finished};
    use crate::{
        ArgumentBuffersTier, Block, Buffer, ComputePipelineDescriptor, ComputePipelineState,
        Device, DispatchType, FunctionDescriptor, IndexClass, Library, MemoryClass, PixelFormat,
        Region, Resource, ResourceUsage, StageSource, Texture, TextureDescriptor, TextureType,
        TextureUsage,
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
            assert_eq!(
                device.argument_buffers_support(),
                ArgumentBuffersTier::Tier2,
                "{name}"
            );
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

    /// The bound argument buffer holds the address of an array of four argument buffers, each
    /// holding a texture, and thread `g` reads texel (0, 0) of the texture of element `g`
    /// (`ab_indexed`).
    #[test]
    fn every_device_reads_the_texture_an_argument_buffer_a_pointer_member_reaches_holds() {
        let Some(library) = library("ab_tier2") else {
            return;
        };
        each_device("argument_tier_indexed", |name, _, device| {
            let read = state(name, &device, &library, "ab_indexed");
            let grant = (device.grant_indices(IndexClass::Texture, 1, 4))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let textures: Vec<Texture> = (TEXELS.iter().zip(1..))
                .map(|(texel, index)| texture(name, &device, 1, index, std::slice::from_ref(texel)))
                .collect();
            let mut shared = Shared::new(&device);
            let ids: Vec<u8> = (textures.iter())
                .flat_map(|t| t.gpu_resource_id().to_bits().to_le_bytes())
                .collect();
            let held = shared.of(&device, &ids);
            let table = shared.of(&device, &held.gpu_address().to_le_bytes());
            let out = shared.of(&device, &[0; 64]);
            let queue = device.new_command_queue(1).expect("a queue");
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let mut encoder =
                (command_buffer.compute_command_encoder(DispatchType::Serial)).expect("an encoder");
            encoder.set_compute_pipeline_state(&read).expect("a state");
            encoder.set_buffer(Some(&table), 0, 0).expect("table");
            encoder.set_buffer(Some(&out), 0, 1).expect("out");
            (encoder.use_resource(Resource::Buffer(&held), ResourceUsage::READ))
                .expect("the held argument buffers");
            for texture in &textures {
                (encoder.use_resource(Resource::Texture(texture), ResourceUsage::READ))
                    .expect("a texture");
            }
            (encoder.dispatch_threadgroups([1, 1, 1], [4, 1, 1]))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            encoder.end_encoding();
            command_buffer.commit().expect("a commit");
            finished(name, &command_buffer);
            assert_eq!(command_buffer.error(), None, "{name}");
            assert_eq!(float4s(&out, 4), TEXELS, "{name}");
            drop((textures, command_buffer));
            (device.release_indices(grant)).unwrap_or_else(|error| panic!("{name}: {error}"));
            println!("{name}: the textures argument buffers a pointer member reaches hold, read");
        });
    }

    /// Threads of `ab_pointers`' kernels, in two threadgroups.
    const THREADS: u32 = 64;

    /// The `count` words at the start of `buffer`, read once its writer completed.
    fn words<T: Copy>(buffer: &Buffer, count: usize) -> Vec<T> {
        let contents = buffer.contents().expect("a Shared buffer");
        // SAFETY: the work that wrote the buffer completed, and its writes reach the host; the
        // buffer holds at least `count` words of `T`, at its start, which is 256-byte aligned.
        unsafe { std::slice::from_raw_parts(contents.as_ptr().cast::<T>(), count) }.to_vec()
    }

    /// The bytes of `values`.
    fn u32_bytes(values: impl IntoIterator<Item = u32>) -> Vec<u8> {
        values.into_iter().flat_map(u32::to_le_bytes).collect()
    }

    /// One dispatch of `state` over [`THREADS`] threads, with `bound` at buffer indices 0 and 1
    /// and `used` made resident (`useResource`, read).
    fn dispatch(
        name: &str,
        device: &Device,
        state: &ComputePipelineState,
        bound: [&Buffer; 2],
        used: &[&Buffer],
    ) {
        let queue = device.new_command_queue(1).expect("a queue");
        let command_buffer = queue.command_buffer().expect("a command buffer");
        let mut encoder =
            (command_buffer.compute_command_encoder(DispatchType::Serial)).expect("an encoder");
        encoder.set_compute_pipeline_state(state).expect("a state");
        for (index, buffer) in (0..).zip(bound) {
            encoder
                .set_buffer(Some(buffer), 0, index)
                .expect("a buffer");
        }
        for buffer in used {
            (encoder.use_resource(Resource::Buffer(buffer), ResourceUsage::READ))
                .expect("a buffer");
        }
        (encoder.dispatch_threadgroups([2, 1, 1], [THREADS / 2, 1, 1]))
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        encoder.end_encoding();
        command_buffer.commit().expect("a commit");
        finished(name, &command_buffer);
        assert_eq!(command_buffer.error(), None, "{name}");
    }

    /// An argument buffer whose two pointer members, as an argument encoder writes them at index
    /// 0, hold the addresses of two buffers, the second 16 bytes past its buffer's start: thread
    /// `i` stores `a[i] + b[i]` (`ab_sum`).
    #[test]
    fn every_device_sums_the_buffers_an_argument_buffers_pointer_members_reach() {
        let Some(library) = library("ab_pointers") else {
            return;
        };
        each_device("argument_tier_pointer_members", |name, _, device| {
            let sum = state(name, &device, &library, "ab_sum");
            let mut shared = Shared::new(&device);
            let a = shared.of(&device, &u32_bytes((0..THREADS).map(|k| 0x1000 + k)));
            let b = shared.of(
                &device,
                &u32_bytes((0..THREADS + 4).map(|k| 0x7000_0000 + 3 * k)),
            );
            let members: Vec<u8> = [a.gpu_address(), b.gpu_address() + 16]
                .iter()
                .flat_map(|address| address.to_le_bytes())
                .collect();
            let args = shared.of(&device, &members);
            let out = shared.of(&device, &u32_bytes((0..THREADS).map(|_| 0)));
            dispatch(name, &device, &sum, [&args, &out], &[&a, &b]);
            let want: Vec<u32> = (0..THREADS)
                .map(|i| (0x1000 + i) + (0x7000_0000 + 3 * (4 + i)))
                .collect();
            assert_eq!(words::<u32>(&out, THREADS as usize), want, "{name}");
            println!("{name}: two buffers summed through an argument buffer's pointer members");
        });
    }

    /// A constant buffer whose bytes hold a buffer's `gpuAddress` plus 8, as the CPU wrote it:
    /// thread `i` stores the word `i` past that address (`ab_address`).
    #[test]
    fn every_device_reads_a_buffer_through_a_gpu_address_the_cpu_wrote() {
        let Some(library) = library("ab_pointers") else {
            return;
        };
        each_device("argument_tier_stored_address", |name, _, device| {
            let read = state(name, &device, &library, "ab_address");
            let mut shared = Shared::new(&device);
            let read_from = shared.of(
                &device,
                &u32_bytes((0..THREADS + 2).map(|k| 0xA000_0000 | k)),
            );
            let held = shared.of(&device, &(read_from.gpu_address() + 8).to_le_bytes());
            let out = shared.of(&device, &u32_bytes((0..THREADS).map(|_| 0)));
            dispatch(name, &device, &read, [&held, &out], &[&read_from]);
            let want: Vec<u32> = (0..THREADS).map(|i| 0xA000_0000 | (2 + i)).collect();
            assert_eq!(words::<u32>(&out, THREADS as usize), want, "{name}");
            println!("{name}: a buffer read through a gpuAddress the CPU wrote");
        });
    }

    /// A GPU hazard (`m2v_gpu_hazards`, docs/VALIDATION.md; a CPU device runs it always): a read
    /// of freed memory where the runtime frees an idle allocation under work in flight. A Shared
    /// buffer of 17 MiB, in an allocation of its own, is read only by address, through a constant
    /// buffer whose bytes the CPU wrote (`ab_address`), by a command buffer that waits on an event
    /// first and names nothing of it. Its block's last handle drops right after the commit, and
    /// the event is set past twice the idle age. The allocation's free waits for that command
    /// buffer (`crate::memory::Wait`): it completes with no error, and the read sees the bytes the
    /// buffer held.
    #[cfg(any(m2v_cpu_icd, all(m2v_gpu, m2v_gpu_hazards)))]
    #[test]
    fn every_device_keeps_memory_read_by_address_alone_past_its_release_until_the_read_retires() {
        use crate::tests::command::on_device::shared_event;
        const LEN: u64 = 17 << 20;
        let Some(library) = library("ab_pointers") else {
            return;
        };
        each_device("argument_tier_idle_free", |name, _, device| {
            let read = state(name, &device, &library, "ab_address");
            let block = (device.take_block(MemoryClass::Shared, LEN)).expect("a block of its own");
            let far = (device.new_buffer(block.id, 0, LEN, Some(0))).expect("its buffer");
            let bytes = u32_bytes((0..THREADS + 2).map(|k| 0xA000_0000 | k));
            let contents = far.contents().expect("a Shared buffer");
            // SAFETY: the range lies in the buffer, which is alive, and no GPU work uses it.
            unsafe {
                std::ptr::copy_nonoverlapping(bytes.as_ptr(), contents.as_ptr(), bytes.len())
            };
            let start = far.gpu_address();
            let mut shared = Shared::new(&device);
            let held = shared.of(&device, &(start + 8).to_le_bytes());
            let out = shared.of(&device, &u32_bytes((0..THREADS).map(|_| 0)));
            let event = shared_event(&device);
            let queue = device.new_command_queue(1).expect("a queue");
            let command_buffer = queue.command_buffer().expect("a command buffer");
            (command_buffer.encode_wait_for_event(event.as_event(), 1)).expect("a wait");
            let mut encoder =
                (command_buffer.compute_command_encoder(DispatchType::Serial)).expect("an encoder");
            encoder.set_compute_pipeline_state(&read).expect("a state");
            encoder.set_buffer(Some(&held), 0, 0).expect("held");
            encoder.set_buffer(Some(&out), 0, 1).expect("out");
            (encoder.dispatch_threadgroups([2, 1, 1], [THREADS / 2, 1, 1]))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            encoder.end_encoding();
            command_buffer.commit().expect("a commit");
            drop((far, block));
            println!(
                "{name}: the buffer at {start:#x}..{:#x} released",
                start + LEN
            );
            std::thread::sleep(std::time::Duration::from_nanos(
                2 * crate::memory::IDLE_AGE_NANOS,
            ));
            event.set_signaled_value(1);
            finished(name, &command_buffer);
            assert_eq!(device.loss(), None, "{name}: the device");
            assert_eq!(
                command_buffer.error(),
                None,
                "{name}: the reading command buffer"
            );
            let want: Vec<u32> = (0..THREADS).map(|i| 0xA000_0000 | (2 + i)).collect();
            assert_eq!(
                words::<u32>(&out, THREADS as usize),
                want,
                "{name}: the bytes it held"
            );
        });
    }

    /// A release under work committed before it waits for that work (`crate::memory::Wait`): a
    /// command buffer that waits on an event, committed before the last handles of a 17 MiB
    /// Shared block (an allocation of its own) and of a texture at an index drop, holds both.
    /// Nothing reads either. Past twice the idle age the allocation is not freed, the index is in
    /// use and `releases_waiting` counts what waits; once the event is set and the command buffer
    /// retires, the allocation is freed, nothing waits and the grant may be returned. The reaper
    /// thread is held from before the releases until then, so the grant's return waits on the
    /// block-maker thread alone, as `releases_waiting` falls only after it vacated the index.
    #[test]
    fn every_device_holds_a_release_until_the_work_committed_before_it_retires() {
        use crate::tests::command::on_device::{shared_event, within};
        use crate::{Error, Refusal};
        const LEN: u64 = 17 << 20;
        each_device("argument_tier_held_release", |name, _, device| {
            let counters = device.counters();
            let grant = (device.grant_indices(IndexClass::Texture, 1, 1))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let mut d = TextureDescriptor::new(TextureType::Type2D, PixelFormat::RGBA8Unorm, 2, 2);
            d.usage = TextureUsage::SHADER_READ;
            let texture =
                (device.new_texture(&d, Some(1))).unwrap_or_else(|error| panic!("{name}: {error}"));
            let block = (device.take_block(MemoryClass::Shared, LEN)).expect("a block of its own");
            let event = shared_event(&device);
            let queue = device.new_command_queue(1).expect("a queue");
            let command_buffer = queue.command_buffer().expect("a command buffer");
            (command_buffer.encode_wait_for_event(event.as_event(), 1)).expect("a wait");
            command_buffer.commit().expect("a commit");
            // The reaper holds at this job until `resume` drops, at the end or in a panic.
            let (resume, parked) = std::sync::mpsc::channel::<()>();
            (device.core().reaper).give(crate::reaper::Job::Teardown(Box::new(
                move |_: &std::sync::Arc<crate::device::Core>| {
                    let _ = parked.recv();
                },
            )));
            let freed = counters.metrics().allocations_freed;
            drop((texture, block));
            std::thread::sleep(std::time::Duration::from_nanos(
                2 * crate::memory::IDLE_AGE_NANOS,
            ));
            let (held, waiting) = (
                counters.metrics().allocations_freed,
                counters.gauges().releases_waiting,
            );
            let in_use = device.release_indices(grant);
            event.set_signaled_value(1);
            finished(name, &command_buffer);
            assert_eq!(held, freed, "{name}: an allocation freed under the work");
            assert!(waiting >= 1, "{name}: {waiting} releases waiting");
            assert!(
                matches!(
                    in_use,
                    Err(Error::Refused(Refusal::IndicesInUse { index: 1, .. }))
                ),
                "{name}: the index under the work: {in_use:?}"
            );
            within("the releases once the work retired", || {
                counters.metrics().allocations_freed > freed
                    && counters.gauges().releases_waiting == 0
            });
            (device.release_indices(grant)).unwrap_or_else(|error| panic!("{name}: {error}"));
            drop(resume);
            println!("{name}: {waiting} releases held under the work, then gone");
        });
    }

    /// A slot pointer converted to an integer: thread `i` stores the address of element `i` of
    /// buffer 0, bound 16 bytes past its buffer's start, and increments the element through the
    /// same pointer (`ab_slot_address`). The address is the buffer's `gpuAddress` plus 16 plus
    /// `4 i`.
    #[test]
    fn every_device_answers_a_slot_pointer_converted_to_an_integer_as_its_gpu_address() {
        let Some(library) = library("ab_pointers") else {
            return;
        };
        each_device("argument_tier_slot_address", |name, _, device| {
            let convert = state(name, &device, &library, "ab_slot_address");
            let mut shared = Shared::new(&device);
            let elements: Vec<u32> = (0..THREADS + 4).map(|k| 0x500 + 7 * k).collect();
            let buf = shared.of(&device, &u32_bytes(elements.iter().copied()));
            let out = shared.of(&device, &vec![0; 8 * THREADS as usize]);
            let queue = device.new_command_queue(1).expect("a queue");
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let mut encoder =
                (command_buffer.compute_command_encoder(DispatchType::Serial)).expect("an encoder");
            encoder
                .set_compute_pipeline_state(&convert)
                .expect("a state");
            encoder.set_buffer(Some(&buf), 16, 0).expect("buf");
            encoder.set_buffer(Some(&out), 0, 1).expect("out");
            (encoder.dispatch_threadgroups([2, 1, 1], [THREADS / 2, 1, 1]))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            encoder.end_encoding();
            command_buffer.commit().expect("a commit");
            finished(name, &command_buffer);
            assert_eq!(command_buffer.error(), None, "{name}");
            let want: Vec<u64> = (0..u64::from(THREADS))
                .map(|i| buf.gpu_address() + 16 + 4 * i)
                .collect();
            assert_eq!(
                words::<u64>(&out, THREADS as usize),
                want,
                "{name}: the addresses"
            );
            let mut incremented = elements.clone();
            for e in &mut incremented[4..] {
                *e += 1;
            }
            assert_eq!(
                words::<u32>(&buf, elements.len()),
                incremented,
                "{name}: the elements"
            );
            println!("{name}: a slot pointer converted to an integer is its gpuAddress");
        });
    }
}
