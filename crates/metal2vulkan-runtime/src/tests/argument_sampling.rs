//! Kernels that sample the textures an argument buffer holds, under a device cfg on every device of
//! the cfg's kind: `examples/heap_samples`' `hs_sample` samples a `half` and a `float` texture its
//! argument buffer holds through a constexpr nearest sampler, at the index the consumer gives it,
//! and through a nearest sampler state bound at a sampler slot. Each thread samples one texel at its
//! centre, and every answer is that texel's exact value. `engine/tail_argument_samplers`'
//! `argument_sampler_edges` samples a texture through the sampler its argument buffer holds beside
//! it, at texel edges, so a linear sampler's answer differs from the default sampler's.
//! `engine/px_samplers`' `px_sample_argument` and `engine/xr_samples`' `xr_sample_runtime` sample
//! one texture through one normalized sampler state held in an argument buffer and bound at a
//! sampler slot, which read the same texels. `hs_sample`'s textures placed in a heap and reached
//! through their indices by a pass that names only the heap are read as they were where their
//! last handles drop right after the commit (a GPU hazard).
//!
//! `llvm-as` assembles the kernels; without it the tests skip visibly.

#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod on_device {
    use crate::tests::command::on_device::{each_device, finished};
    use crate::{
        Block, Buffer, ComputePipelineDescriptor, ComputePipelineState, Device, DispatchType,
        FunctionDescriptor, HeapRange, IndexClass, Library, MemoryClass, NumberedSampler,
        PixelFormat, Region, Resource, ResourceUsage, SamplerDescriptor, StageSource, Texture,
        TextureDescriptor, TextureType, TextureUsage,
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

    /// The library of fixture `path`, or None where `llvm-as` is missing.
    fn library(path: &str) -> Option<Library> {
        match llvm_assemble(&fixtures_dir().join(path)) {
            Ok(bitcode) => Some(Library::from_air(&bitcode).expect("the library")),
            Err(AssembleError::Missing(why)) => {
                eprintln!("SKIPPED: {why}");
                None
            }
            Err(error) => panic!("{path}: {error:?}"),
        }
    }

    /// A 2D texture of `format` (`texel` bytes a texel), `side` a side, shader-read, at texture
    /// index `index`, its bytes `bytes` row by row.
    fn texture(
        name: &str,
        device: &Device,
        (format, texel): (PixelFormat, u64),
        (index, side): (u32, u32),
        bytes: &[u8],
    ) -> Texture {
        let mut d = TextureDescriptor::new(TextureType::Type2D, format, side, side);
        d.usage = TextureUsage::SHADER_READ;
        let texture =
            (device.new_texture(&d, Some(index))).unwrap_or_else(|error| panic!("{name}: {error}"));
        let whole = Region {
            origin: [0; 3],
            size: [side, side, 1],
        };
        let row = u64::from(side) * texel;
        (texture.replace_region(
            whole,
            0,
            0,
            crate::HostBytes::from(bytes),
            row,
            row * u64::from(side),
        ))
        .unwrap_or_else(|error| panic!("{name}: {error}"));
        texture
    }

    /// `hs_sample`'s state, its constexpr sampler numbered [`CONSTEXPR`].
    fn hs_sample(name: &str, device: &Device, library: &Library) -> ComputePipelineState {
        let function = library.function("hs_sample").expect("the kernel");
        let descriptor = ComputePipelineDescriptor::new(StageSource::Function(
            FunctionDescriptor::new(function),
        ));
        (device.new_compute_pipeline_state(&descriptor, |samplers| {
            assert!(
                matches!(samplers, [NumberedSampler::Constexpr(_)]),
                "{name}: {samplers:?}"
            );
            Ok(vec![CONSTEXPR])
        }))
        .unwrap_or_else(|error| panic!("{name}: hs_sample: {error}"))
    }

    /// The bytes of `hs_sample`'s `half` texture and its `float` texture, row by row.
    fn hs_bytes() -> (Vec<u8>, Vec<u8>) {
        let texels = (SIDE * SIDE) as usize;
        let halves = (0..texels)
            .flat_map(|i| (0..4).map(move |c| HALVES[half_at(i, c)].1))
            .flat_map(u16::to_le_bytes)
            .collect();
        let floats = (0..texels)
            .flat_map(|i| (0..4).map(move |c| float_at(i, c)))
            .flat_map(f32::to_le_bytes)
            .collect();
        (halves, floats)
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
        let Some(library) = library("examples/heap_samples.ll") else {
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
            let state = hs_sample(name, &device, &library);
            let texels = (SIDE * SIDE) as usize;
            let (halves, floats) = hs_bytes();
            let h = texture(
                name,
                &device,
                (PixelFormat::RGBA16Float, 8),
                (1, SIDE),
                &halves,
            );
            let f = texture(
                name,
                &device,
                (PixelFormat::RGBA32Float, 16),
                (2, SIDE),
                &floats,
            );
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

    /// A GPU hazard (`m2v_gpu_hazards`, docs/VALIDATION.md; a CPU device runs it always): a read
    /// through set 1 of textures released under the work that reads them. `hs_sample`'s two
    /// textures sit in one heap at indices of their own, and its pass names only the heap
    /// (`use_heap`), reaching them through the resource IDs its argument buffer holds. Their last
    /// handles drop after the commit, in three forms: while the command buffer waits on an event
    /// (committed, not submitted), while long work committed ahead of it on its queue runs, and
    /// 20 ms into a pass of `SAMPLINGS` dispatches, while it samples them. The heap's range and
    /// each form's GPU span beside the drop's host time are printed, so a fault's address and
    /// time can be placed. A texture's going waits for the command buffers committed before its
    /// release (`crate::memory::Wait`): each completes with no error, and its pass reads the
    /// textures as they were.
    #[cfg(any(m2v_cpu_icd, all(m2v_gpu, m2v_gpu_hazards)))]
    #[test]
    fn every_device_keeps_heap_textures_read_through_their_indices_past_their_release() {
        /// The dispatches of the pass that samples while its textures drop: about 200 ms on an
        /// RTX 3090.
        const SAMPLINGS: u32 = 40_000;
        use crate::tests::command::on_device::shared_event;
        use crate::tests::compute::on_device::step;
        use crate::tests::gpu_span::on_device::{dispatches_for, long, words, WORDS};
        let Some(library) = library("examples/heap_samples.ll") else {
            return;
        };
        each_device("argument_sampling_released", |name, _, device| {
            let _grants = [
                (IndexClass::Texture, 1, 6),
                (IndexClass::Sampler, SLOT_SAMPLER, 2),
            ]
            .map(|(class, first, count)| {
                (device.grant_indices(class, first, count))
                    .unwrap_or_else(|error| panic!("{name}: {error}"))
            });
            let state = hs_sample(name, &device, &library);
            let texels = (SIDE * SIDE) as usize;
            let (halves, floats) = hs_bytes();
            let want: Vec<[f32; 4]> = (0..texels)
                .flat_map(|i| {
                    let half = [0, 1, 2, 3].map(|c| HALVES[half_at(i, c)].0);
                    let float = [0, 1, 2, 3].map(|c| float_at(i, c));
                    [half, float, half, float]
                })
                .collect();
            let sampler = (device.new_sampler_state(&SamplerDescriptor::new(), SLOT_SAMPLER))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let len = MemoryClass::Private.block_len();
            let heap_block = (device.take_block(MemoryClass::Private, len))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let heap = HeapRange {
                block: heap_block.id,
                offset: 0,
                len,
            };
            println!(
                "{name}: the heap's block at {:#x}..{:#x}",
                heap_block.gpu_address,
                heap_block.gpu_address + len
            );
            let descriptor = |format| {
                let mut d = TextureDescriptor::new(TextureType::Type2D, format, SIDE, SIDE);
                d.usage = TextureUsage::SHADER_READ;
                d
            };
            let formats = [
                (PixelFormat::RGBA16Float, 8),
                (PixelFormat::RGBA32Float, 16),
            ];
            let fits = formats.map(|(format, _)| {
                (device.heap_texture_size_and_align(&descriptor(format)))
                    .unwrap_or_else(|error| panic!("{name}: {error}"))
            });
            let align = fits.iter().map(|fit| fit.align).max().unwrap_or(1);
            let stride =
                (fits.iter().map(|fit| fit.size).max().unwrap_or(0)).next_multiple_of(align);
            // The texture of `format` (`texel` bytes a texel) at place `at` of the heap and texture
            // index `index`, its bytes `bytes`.
            let place = |at: u64, (format, texel): (PixelFormat, u64), index: u32, bytes: &[u8]| {
                let texture =
                    (device.new_heap_texture(&descriptor(format), Some(index), heap, at * stride))
                        .unwrap_or_else(|error| panic!("{name}: {error}"));
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
            };
            let block = (device.take_block(MemoryClass::Shared, 65_536)).expect("a Shared block");
            let ahead_state = step(&device);
            let busy = words(&device);
            let queue = device.new_command_queue(4).expect("a queue");
            let count = dispatches_for(name, &queue, &ahead_state, &busy, 200_000_000);
            let event = shared_event(&device);
            let mut unlike = Vec::new();
            for (round, form) in ["at an event wait", "behind long work", "while it runs"]
                .into_iter()
                .enumerate()
            {
                let first = 1 + 2 * round as u32;
                let at = 2 * round as u64;
                let h = place(at, formats[0], first, &halves);
                let f = place(at + 1, formats[1], first + 1, &floats);
                let ids: Vec<u8> = [&h, &f]
                    .iter()
                    .flat_map(|t| t.gpu_resource_id().to_bits().to_le_bytes())
                    .collect();
                let held = shared(&device, &block, 512 * at, &ids);
                let out = shared(&device, &block, 512 * at + 256, &[0; 16 * 16]);
                let ahead = (round == 1).then(|| long(&queue, &ahead_state, &busy, count, WORDS));
                let command_buffer = queue.command_buffer().expect("a command buffer");
                if round == 0 {
                    (command_buffer.encode_wait_for_event(event.as_event(), 1)).expect("a wait");
                }
                let mut encoder = (command_buffer.compute_command_encoder(DispatchType::Serial))
                    .expect("an encoder");
                encoder.set_compute_pipeline_state(&state).expect("a state");
                encoder.set_buffer(Some(&held), 0, 0).expect("held");
                encoder.set_buffer(Some(&out), 0, 1).expect("out");
                (encoder.set_sampler_state(Some(&sampler), 0)).expect("the sampler");
                (encoder.use_heap(heap.block)).expect("the heap");
                for _ in 0..if round == 2 { SAMPLINGS } else { 1 } {
                    (encoder.dispatch_threadgroups([1, 1, 1], [SIDE, SIDE, 1]))
                        .unwrap_or_else(|error| panic!("{name}: {error}"));
                }
                encoder.end_encoding();
                if let Some(ahead) = &ahead {
                    ahead.commit().expect("a commit");
                }
                command_buffer.commit().expect("a commit");
                if round == 2 {
                    std::thread::sleep(std::time::Duration::from_millis(20));
                }
                drop((h, f));
                let dropped = crate::host_time();
                println!(
                    "{name}: the textures at indices {first} and {} released {form}",
                    first + 1
                );
                if round == 0 {
                    std::thread::sleep(std::time::Duration::from_millis(50));
                    event.set_signaled_value(1);
                }
                for command_buffer in ahead.iter().chain([&command_buffer]) {
                    finished(name, command_buffer);
                }
                let (loss, error) = (device.loss(), command_buffer.error());
                let read = float4s(&out, 4 * texels);
                let (start, end) = (
                    command_buffer.gpu_start_time(),
                    command_buffer.gpu_end_time(),
                );
                println!(
                    "{name}: {form}: dropped at {dropped}, the pass's GPU span {start}..{end}; \
                     loss {loss:?}, error {error:?}, read {read:?}"
                );
                if loss.is_some() || error.is_some() || read != want {
                    unlike.push(form);
                }
                if loss.is_some() {
                    break;
                }
            }
            assert!(
                unlike.is_empty(),
                "{name}: the textures were not read as they were {unlike:?}"
            );
            println!("{name}: released heap textures read as they were through their indices");
        });
    }

    /// Each thread `i = 4c + s` of `engine/sampler_table`'s `sampler_table_held` samples row 0 of
    /// the 4 by 1 texture its argument buffer holds (red, green, blue, white) at `x = xs[i]`
    /// through `samplers[which[i]]`, a program-scope table of four constexpr nearest samplers
    /// (repeat, clamp to edge, mirrored repeat, clamp to zero) read at the index `which[i] = s`
    /// only the run knows, and writes the texel. Under the identity domains the four candidates
    /// are a switch over them, each arm sampling through its constexpr sampler's index. The
    /// answers are Metal's, measured on an Apple M3 Max under macOS 27.0.1
    /// (`metal2vulkan-conformance/probes/sampler_table.m`): by coordinate (texel units -0.5, 4.5,
    /// -1.5, 2.5 and 5.5), through the four samplers in order, white, red, red, zero; red, white,
    /// white, zero; blue, red, green, zero; blue four times; green, white, blue, zero.
    #[test]
    fn every_device_samples_an_argument_buffers_texture_through_a_table_of_constexpr_samplers() {
        const THREADS: usize = 20;
        const XS: [f32; 5] = [-0.125, 1.125, -0.375, 0.625, 1.375];
        let [red, green, blue, white, zero] = [
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
            [0.0; 4],
        ];
        let metal: [[[f32; 4]; 4]; 5] = [
            [white, red, red, zero],
            [red, white, white, zero],
            [blue, red, green, zero],
            [blue, blue, blue, blue],
            [green, white, blue, zero],
        ];
        let Some(library) = library("engine/sampler_table.ll") else {
            return;
        };
        each_device("argument_sampling", |name, _, device| {
            let grants = [(IndexClass::Texture, 1, 1), (IndexClass::Sampler, 1, 4)].map(
                |(class, first, count)| {
                    (device.grant_indices(class, first, count))
                        .unwrap_or_else(|error| panic!("{name}: {error}"))
                },
            );
            let function = library.function("sampler_table_held").expect("the kernel");
            let descriptor = ComputePipelineDescriptor::new(StageSource::Function(
                FunctionDescriptor::new(function),
            ));
            let state = (device.new_compute_pipeline_state(&descriptor, |samplers| {
                assert_eq!(samplers.len(), 4, "{name}: {samplers:?}");
                Ok((1..=4).collect())
            }))
            .unwrap_or_else(|error| panic!("{name}: sampler_table_held: {error}"));
            let mut d = TextureDescriptor::new(TextureType::Type2D, PixelFormat::RGBA8Unorm, 4, 1);
            d.usage = TextureUsage::SHADER_READ;
            let t =
                (device.new_texture(&d, Some(1))).unwrap_or_else(|error| panic!("{name}: {error}"));
            let texels: [u8; 16] = [
                255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 255, 255,
            ];
            let whole = Region {
                origin: [0; 3],
                size: [4, 1, 1],
            };
            (t.replace_region(whole, 0, 0, crate::HostBytes::from(&texels[..]), 16, 16))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let block = (device.take_block(MemoryClass::Shared, 65_536)).expect("a Shared block");
            let which: Vec<u8> = (0..THREADS as u32)
                .flat_map(|i| (i % 4).to_le_bytes())
                .collect();
            let xs: Vec<u8> = (0..THREADS).flat_map(|i| XS[i / 4].to_le_bytes()).collect();
            let id = t.gpu_resource_id().to_bits().to_le_bytes();
            let held = shared(&device, &block, 0, &id);
            let which = shared(&device, &block, 256, &which);
            let xs = shared(&device, &block, 512, &xs);
            let out = shared(&device, &block, 1024, &[0; 16 * THREADS]);
            let queue = device.new_command_queue(1).expect("a queue");
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let mut encoder =
                (command_buffer.compute_command_encoder(DispatchType::Serial)).expect("an encoder");
            encoder.set_compute_pipeline_state(&state).expect("a state");
            for (buffer, index) in [(&out, 0), (&which, 1), (&xs, 2), (&held, 3)] {
                encoder
                    .set_buffer(Some(buffer), 0, index)
                    .expect("a buffer");
            }
            (encoder.use_resource(Resource::Texture(&t), ResourceUsage::READ)).expect("a texture");
            (encoder.dispatch_threadgroups([1, 1, 1], [THREADS as u32, 1, 1]))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            encoder.end_encoding();
            command_buffer.commit().expect("a commit");
            finished(name, &command_buffer);
            assert_eq!(command_buffer.error(), None, "{name}");
            let read = float4s(&out, THREADS);
            for (i, texel) in read.iter().enumerate() {
                assert_eq!(
                    *texel,
                    metal[i / 4][i % 4],
                    "{name}: thread {i}, x {} through sampler {}",
                    XS[i / 4],
                    i % 4
                );
            }
            drop((command_buffer, queue, state, t));
            for grant in grants {
                (device.release_indices(grant)).unwrap_or_else(|error| panic!("{name}: {error}"));
            }
            println!(
                "{name}: an argument buffer's texture sampled through a table of constexpr samplers"
            );
        });
    }

    /// Each thread `g` of `argument_sampler_edges` samples the 4 by 4 texture its argument buffer
    /// holds, whose texel `(x, y)` has red `x + 4y`, at normalized `(g / 4, g / 4)`, a texel edge,
    /// through the sampler the argument buffer holds beside it. Through a linear sampler state at
    /// its own index each answer averages the four texels around the edge, clamped at the texture's
    /// edge for `g` 0: red 0, 2.5, 7.5, 12.5. Through a resource ID with that state's word and index
    /// 4,000, one past the sampler arrays, each samples through element 0, the default nearest
    /// sampler: red `5g`, the texel past the edge.
    #[test]
    fn every_device_samples_an_argument_buffers_texture_through_its_own_sampler() {
        use crate::SamplerMinMagFilter;
        const EDGE_SIDE: u32 = 4;
        let Some(library) = library("engine/tail_argument_samplers.ll") else {
            return;
        };
        each_device("argument_sampling", |name, _, device| {
            let grants = [
                (IndexClass::Texture, 1),
                (IndexClass::Sampler, SLOT_SAMPLER),
            ]
            .map(|(class, first)| {
                (device.grant_indices(class, first, 1))
                    .unwrap_or_else(|error| panic!("{name}: {error}"))
            });
            let function = library
                .function("argument_sampler_edges")
                .expect("the kernel");
            let descriptor = ComputePipelineDescriptor::new(StageSource::Function(
                FunctionDescriptor::new(function),
            ));
            let state = (device.new_compute_pipeline_state(&descriptor, |samplers| {
                assert!(samplers.is_empty(), "{name}: {samplers:?}");
                Ok(Vec::new())
            }))
            .unwrap_or_else(|error| panic!("{name}: argument_sampler_edges: {error}"));
            let texels: Vec<u8> = (0..EDGE_SIDE * EDGE_SIDE)
                .flat_map(|i| [i as f32, 0.0, 0.0, 1.0])
                .flat_map(f32::to_le_bytes)
                .collect();
            let t = texture(
                name,
                &device,
                (PixelFormat::RGBA32Float, 16),
                (1, EDGE_SIDE),
                &texels,
            );
            let mut linear = SamplerDescriptor::new();
            linear.min_filter = SamplerMinMagFilter::Linear;
            linear.mag_filter = SamplerMinMagFilter::Linear;
            let sampler = (device.new_sampler_state(&linear, SLOT_SAMPLER))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let id = sampler.gpu_resource_id().to_bits();
            assert_eq!(id & 0xffff_ffff, u64::from(SLOT_SAMPLER), "{name}");
            let past = (id & !0xffff_ffff) | 4_000;
            let block = (device.take_block(MemoryClass::Shared, 65_536)).expect("a Shared block");
            let material = |sampler: u64| -> Vec<u8> {
                let mut bytes = t.gpu_resource_id().to_bits().to_le_bytes().to_vec();
                bytes.extend(sampler.to_le_bytes());
                bytes
            };
            let held = [id, past]
                .iter()
                .enumerate()
                .map(|(k, s)| shared(&device, &block, 256 * k as u64, &material(*s)))
                .collect::<Vec<_>>();
            let outs = (0..2)
                .map(|k| shared(&device, &block, 1024 + 256 * k, &[0; 64]))
                .collect::<Vec<_>>();
            let queue = device.new_command_queue(1).expect("a queue");
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let mut encoder =
                (command_buffer.compute_command_encoder(DispatchType::Serial)).expect("an encoder");
            encoder.set_compute_pipeline_state(&state).expect("a state");
            (encoder.use_resource(Resource::Texture(&t), ResourceUsage::READ)).expect("a texture");
            for (held, out) in held.iter().zip(&outs) {
                encoder.set_buffer(Some(held), 0, 0).expect("held");
                encoder.set_buffer(Some(out), 0, 1).expect("out");
                (encoder.dispatch_threadgroups([1, 1, 1], [EDGE_SIDE, 1, 1]))
                    .unwrap_or_else(|error| panic!("{name}: {error}"));
            }
            encoder.end_encoding();
            command_buffer.commit().expect("a commit");
            finished(name, &command_buffer);
            assert_eq!(command_buffer.error(), None, "{name}");
            assert_eq!(
                float4s(&outs[0], 4),
                [0.0, 2.5, 7.5, 12.5].map(|r| [r, 0.0, 0.0, 1.0]),
                "{name}: through the linear sampler's resource ID"
            );
            assert_eq!(
                float4s(&outs[1], 4),
                [0.0, 5.0, 10.0, 15.0].map(|r| [r, 0.0, 0.0, 1.0]),
                "{name}: through a resource ID whose index is past the sampler arrays"
            );
            drop((command_buffer, queue, state, sampler, t));
            for grant in grants {
                (device.release_indices(grant)).unwrap_or_else(|error| panic!("{name}: {error}"));
            }
            println!(
                "{name}: an argument buffer's texture sampled through the argument buffer's sampler"
            );
        });
    }

    /// The conformance case `pixel-samplers`' coordinates over its 5 by 3 texture: per integer
    /// from -1 to 6 the texel edge and one ulp either side, `i + 1/512`, the quarter, the centre
    /// and one ulp either side, three quarters and `255/256`, across ten rows from -0.5 to 3.25,
    /// each divided by the size.
    fn normalized_grid(width: u32, height: u32) -> Vec<[f32; 2]> {
        let mut xs = Vec::new();
        for i in -1..=(width as i32 + 1) {
            let (b, h) = (i as f32, i as f32 + 0.5);
            let (lo, hi) = match i {
                0 => (-(2f32.powi(-24)), 2f32.powi(-24)),
                _ => (b.next_down(), b.next_up()),
            };
            xs.extend([
                lo,
                b,
                hi,
                b + 2f32.powi(-9),
                b + 0.25,
                h.next_down(),
                h,
                h.next_up(),
                b + 0.75,
                b + 255.0 / 256.0,
            ]);
        }
        let ys = [
            -0.5,
            0.0,
            0.5,
            1f32.next_down(),
            1.0,
            1.25,
            1.5f32.next_up(),
            2.5,
            3.0,
            3.25,
        ];
        (ys.iter())
            .flat_map(|&y| {
                xs.iter()
                    .map(move |&x| [x / width as f32, y / height as f32])
            })
            .collect()
    }

    /// One `RGBA32Float` texture of 5 by 3 texels, `100x + 10y + c` in channel `c`, sampled at the
    /// conformance case `pixel-samplers`' 800 coordinates over its size ([`normalized_grid`])
    /// through one normalized nearest sampler state under each address mode that case asks
    /// (clamp to edge, to zero, to an opaque white border, and mirror clamp to edge): bound at
    /// sampler slot 0 beside texture slot 0 (`xr_samples`' `xr_sample_runtime`), and held in an
    /// argument buffer beside the texture (`px_samplers`' `px_sample_argument`). The two read the
    /// same texels, bit for bit, at every coordinate under every mode.
    #[test]
    fn every_device_samples_alike_through_an_argument_buffers_sampler_and_a_slots() {
        use crate::{SamplerAddressMode, SamplerBorderColor};
        const WIDTH: u32 = 5;
        const HEIGHT: u32 = 3;
        let (Some(slots), Some(arguments)) = (
            library("engine/xr_samples.ll"),
            library("engine/px_samplers.ll"),
        ) else {
            return;
        };
        let modes = [
            SamplerAddressMode::ClampToEdge,
            SamplerAddressMode::ClampToZero,
            SamplerAddressMode::ClampToBorderColor,
            SamplerAddressMode::MirrorClampToEdge,
        ];
        let uv = normalized_grid(WIDTH, HEIGHT);
        let count = uv.len();
        each_device("argument_sampling", |name, _, device| {
            let grants = [
                (IndexClass::Texture, 1, 1),
                (IndexClass::Sampler, SLOT_SAMPLER, 4),
            ]
            .map(|(class, first, n)| {
                (device.grant_indices(class, first, n))
                    .unwrap_or_else(|error| panic!("{name}: {error}"))
            });
            let state = |library: &Library, entry: &str| {
                let function = library.function(entry).expect("the kernel");
                let descriptor = ComputePipelineDescriptor::new(StageSource::Function(
                    FunctionDescriptor::new(function),
                ));
                (device.new_compute_pipeline_state(&descriptor, |samplers| {
                    assert!(samplers.is_empty(), "{name}: {entry}: {samplers:?}");
                    Ok(Vec::new())
                }))
                .unwrap_or_else(|error| panic!("{name}: {entry}: {error}"))
            };
            let (slot_state, argument_state) = (
                state(&slots, "xr_sample_runtime"),
                state(&arguments, "px_sample_argument"),
            );
            let texels: Vec<u8> = (0..HEIGHT)
                .flat_map(|y| (0..WIDTH).map(move |x| (x, y)))
                .flat_map(|(x, y)| (0..4).map(move |c| (100 * x + 10 * y + c) as f32))
                .flat_map(f32::to_le_bytes)
                .collect();
            let mut d = TextureDescriptor::new(
                TextureType::Type2D,
                PixelFormat::RGBA32Float,
                WIDTH,
                HEIGHT,
            );
            d.usage = TextureUsage::SHADER_READ;
            let t =
                (device.new_texture(&d, Some(1))).unwrap_or_else(|error| panic!("{name}: {error}"));
            let whole = Region {
                origin: [0; 3],
                size: [WIDTH, HEIGHT, 1],
            };
            let row = u64::from(WIDTH) * 16;
            (t.replace_region(
                whole,
                0,
                0,
                crate::HostBytes::from(&texels[..]),
                row,
                row * u64::from(HEIGHT),
            ))
            .unwrap_or_else(|error| panic!("{name}: {error}"));
            let samplers: Vec<_> = (modes.iter().zip(SLOT_SAMPLER..))
                .map(|(mode, index)| {
                    let mut s = SamplerDescriptor::new();
                    (s.s_address_mode, s.t_address_mode, s.r_address_mode) = (*mode, *mode, *mode);
                    s.border_color = SamplerBorderColor::OpaqueWhite;
                    (device.new_sampler_state(&s, index))
                        .unwrap_or_else(|error| panic!("{name}: {mode:?}: {error}"))
                })
                .collect();
            let block = (device.take_block(MemoryClass::Shared, 1 << 20)).expect("a Shared block");
            let coordinates: Vec<u8> = (uv.iter())
                .flat_map(|[u, v]| [*u, *v])
                .flat_map(f32::to_le_bytes)
                .collect();
            let held_uv = shared(&device, &block, 0, &coordinates);
            let out_bytes = 16 * count as u64;
            let outs: Vec<[Buffer; 2]> = (0..modes.len() as u64)
                .map(|k| {
                    let at = 65_536 + 2 * k * out_bytes.next_multiple_of(256);
                    let next = at + out_bytes.next_multiple_of(256);
                    let blank = vec![0xab; out_bytes as usize];
                    [
                        shared(&device, &block, at, &blank),
                        shared(&device, &block, next, &blank),
                    ]
                })
                .collect();
            let materials: Vec<Buffer> = (samplers.iter().enumerate())
                .map(|(k, s)| {
                    let mut bytes = t.gpu_resource_id().to_bits().to_le_bytes().to_vec();
                    bytes.extend(s.gpu_resource_id().to_bits().to_le_bytes());
                    shared(&device, &block, 32_768 + 256 * k as u64, &bytes)
                })
                .collect();
            let queue = device.new_command_queue(1).expect("a queue");
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let mut encoder =
                (command_buffer.compute_command_encoder(DispatchType::Serial)).expect("an encoder");
            encoder.set_buffer(Some(&held_uv), 0, 0).expect("uv");
            (encoder.use_resource(Resource::Texture(&t), ResourceUsage::READ)).expect("a texture");
            let groups = [(count as u32).div_ceil(32), 1, 1];
            for ((sampler, material), [slot_out, argument_out]) in
                samplers.iter().zip(&materials).zip(&outs)
            {
                encoder
                    .set_compute_pipeline_state(&slot_state)
                    .expect("a state");
                encoder.set_texture(Some(&t), 0).expect("the texture");
                (encoder.set_sampler_state(Some(sampler), 0)).expect("the sampler");
                encoder.set_buffer(Some(slot_out), 0, 1).expect("out");
                (encoder.dispatch_threadgroups(groups, [32, 1, 1]))
                    .unwrap_or_else(|error| panic!("{name}: {error}"));
                encoder
                    .set_compute_pipeline_state(&argument_state)
                    .expect("a state");
                encoder
                    .set_buffer(Some(material), 0, 2)
                    .expect("the material");
                encoder.set_buffer(Some(argument_out), 0, 1).expect("out");
                (encoder.dispatch_threadgroups(groups, [32, 1, 1]))
                    .unwrap_or_else(|error| panic!("{name}: {error}"));
            }
            encoder.end_encoding();
            command_buffer.commit().expect("a commit");
            finished(name, &command_buffer);
            assert_eq!(command_buffer.error(), None, "{name}");
            for (mode, [slot_out, argument_out]) in modes.iter().zip(&outs) {
                let bits = |b: &Buffer| -> Vec<[u32; 4]> {
                    float4s(b, count)
                        .iter()
                        .map(|t| t.map(f32::to_bits))
                        .collect()
                };
                let (slot, argument) = (bits(slot_out), bits(argument_out));
                let differing: Vec<_> = (0..count)
                    .filter(|&g| slot[g] != argument[g])
                    .map(|g| {
                        (
                            uv[g],
                            slot[g].map(f32::from_bits),
                            argument[g].map(f32::from_bits),
                        )
                    })
                    .collect();
                assert!(
                    differing.is_empty(),
                    "{name}: {mode:?}: {} of {count} coordinates differ, the first (uv, slot, \
                     argument buffer) {:?}",
                    differing.len(),
                    differing.first()
                );
            }
            drop((
                command_buffer,
                queue,
                slot_state,
                argument_state,
                samplers,
                t,
            ));
            for grant in grants {
                (device.release_indices(grant)).unwrap_or_else(|error| panic!("{name}: {error}"));
            }
            println!(
                "{name}: one texture sampled alike through an argument buffer's sampler and a \
                 slot's, under {} address modes at {count} coordinates",
                modes.len()
            );
        });
    }
}
