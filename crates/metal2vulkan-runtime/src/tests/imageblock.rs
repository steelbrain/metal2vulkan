//! A kernel's explicit imageblock, under a device cfg on every device of the cfg's kind: an
//! encoder that never sets its size runs it 1 by 1, as Metal's does; one encoder runs two sizes
//! it sets, each dispatch at its own, every one a pipeline of its own made at its first use; the
//! setter refuses the sizes Metal's validation refuses; and a pre-pass's key follows the size.

#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod on_device {
    use crate::tests::command::on_device::{each_device, finished, Arena};
    use crate::{
        Buffer, ComputePipelineDescriptor, ComputePipelineState, Device, Dispatch, DispatchType,
        Error, FunctionDescriptor, Gap, NumberedSampler, Refusal, StageSource,
    };
    use metal2vulkan::library::Library;
    use metal2vulkan_devtools::{engine_dir, llvm_assemble, AssembleError};

    /// `a4_imageblock_extent`'s state, or None where `llvm-as` is missing: each thread stages its
    /// cell at its position, then writes its right-hand neighbour's cell (wrapping at the
    /// imageblock's width) and the extent it reads.
    fn extent_state(device: &Device) -> Option<ComputePipelineState> {
        let fixture = engine_dir().join("a4_imageblock_extent.ll");
        let bitcode = match llvm_assemble(&fixture) {
            Ok(bitcode) => bitcode,
            Err(AssembleError::Missing(why)) => {
                eprintln!("SKIPPED: {why}");
                return None;
            }
            Err(error) => panic!("a4_imageblock_extent: {error:?}"),
        };
        let library = Library::from_air(&bitcode).expect("the library");
        let function = library
            .function("a4_imageblock_extent")
            .expect("the kernel");
        let descriptor = ComputePipelineDescriptor::new(StageSource::Function(
            FunctionDescriptor::new(function),
        ));
        let ids = |samplers: &[NumberedSampler]| Ok((0..samplers.len() as u32).collect());
        Some((device.new_compute_pipeline_state(&descriptor, ids)).expect("the state"))
    }

    /// What `a4_imageblock_extent` writes for a `width` by `height` threadgroup over an imageblock
    /// of its size: for thread `(x, y)`, the cell of `((x + 1) % width, y)` and the extent.
    fn expected(width: u32, height: u32) -> Vec<u32> {
        (0..height)
            .flat_map(|y| {
                (0..width).flat_map(move |x| {
                    let next = (x + 1) % width;
                    let value = (next as f32 + 100.0 * y as f32).to_bits();
                    [value, next | (y << 16), width, height]
                })
            })
            .collect()
    }

    /// The first `count` words of `buffer`, whose work completed.
    fn words(buffer: &Buffer, count: usize) -> Vec<u32> {
        let contents = buffer.contents().expect("a Shared buffer").cast::<u32>();
        // SAFETY: the buffer is alive and 256 bytes long, `count` is at most 64, and its work
        // completed.
        (0..count)
            .map(|index| unsafe { contents.as_ptr().add(index).read_volatile() })
            .collect()
    }

    /// A kernel that reaches an explicit imageblock, dispatched by an encoder that never sets its
    /// size, runs it 1 by 1: one thread's threadgroup reads its own cell and an extent of 1 by 1,
    /// as Metal's does (job m3-max-70002, probe `imageblock_size_rules.m`'s `never-set`).
    #[test]
    fn an_imageblock_never_sized_runs_one_by_one() {
        each_device("imageblock_never_sized", |name, _, device| {
            let Some(state) = extent_state(&device) else {
                return;
            };
            let mut arena = Arena::new(&device);
            let out = arena.buffer();
            let queue = device.new_command_queue(1).expect("a queue");
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let mut encoder =
                (command_buffer.compute_command_encoder(DispatchType::Serial)).expect("an encoder");
            encoder
                .set_compute_pipeline_state(&state)
                .expect("the state");
            encoder.set_buffer(Some(&out), 0, 0).expect("the buffer");
            (encoder.dispatch_threadgroups([1, 1, 1], [1, 1, 1]))
                .unwrap_or_else(|error| panic!("{name}: the dispatch: {error}"));
            encoder.end_encoding();
            command_buffer.commit().expect("a commit");
            finished(name, &command_buffer);
            assert_eq!(command_buffer.error(), None, "{name}");
            assert_eq!(
                words(&out, 4),
                expected(1, 1),
                "{name}: the 1 by 1 imageblock"
            );
            println!("{name}: an imageblock never sized ran 1 by 1");
        });
    }

    /// One encoder sets two imageblock sizes, 4 by 2 and then 4 by 4, each before a dispatch of a
    /// threadgroup of its size: each dispatch reads the size set before it, through a pipeline of
    /// its own made at its first use (two first-use waits and two keys, the state having no main
    /// pipeline, as a kernel that takes a threadgroup memory length from the encoder has none).
    /// The state's imageblock length stays its typed gap.
    #[test]
    fn one_encoder_runs_each_imageblock_size_it_sets() {
        each_device("imageblock_two_sizes", |name, _, device| {
            let Some(state) = extent_state(&device) else {
                return;
            };
            let length = state.imageblock_memory_length_for_dimensions([8, 4, 3]);
            assert!(
                matches!(length, Err(Error::Unsupported(Gap::ImageblockLength))),
                "{name}: the imageblock length {length:?}"
            );
            state.finish_pipeline().expect("no main pipeline to make");
            let groups = Dispatch::Threadgroups([1, 1, 1]);
            let key = |size: [u32; 2]| {
                let threads = [size[0], size[1], 1];
                (state.dispatch_variant(groups, threads, &[], size)).expect("a key")
            };
            let (small, large) = (key([4, 2]), key([4, 4]));
            assert!(!small.1 && !large.1, "{name}: made before any dispatch");
            assert_ne!(small.0, large.0, "{name}: one key for two sizes");
            let wide = state.dispatch_variant(groups, [4, 2, 1], &[], [8, 2]);
            assert_ne!(
                wide.map(|(key, _)| key),
                Some(small.0),
                "{name}: one key for two sizes"
            );
            let refused = state.dispatch_variant(groups, [4, 2, 1], &[], [3, 2]);
            assert_eq!(
                refused, None,
                "{name}: a key for a size the encoder refuses"
            );
            let before = device.counters().metrics();
            let mut arena = Arena::new(&device);
            let (first, second) = (arena.buffer(), arena.buffer());
            let queue = device.new_command_queue(1).expect("a queue");
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let mut encoder =
                (command_buffer.compute_command_encoder(DispatchType::Serial)).expect("an encoder");
            encoder
                .set_compute_pipeline_state(&state)
                .expect("the state");
            for (size, out) in [([4, 2], &first), ([4, 4], &second)] {
                encoder.set_buffer(Some(out), 0, 0).expect("the buffer");
                encoder
                    .set_imageblock_size(size[0], size[1])
                    .expect("a size");
                (encoder.dispatch_threadgroups([1, 1, 1], [size[0], size[1], 1]))
                    .unwrap_or_else(|error| panic!("{name}: the dispatch at {size:?}: {error}"));
            }
            encoder.end_encoding();
            command_buffer.commit().expect("a commit");
            finished(name, &command_buffer);
            assert_eq!(command_buffer.error(), None, "{name}");
            assert_eq!(
                words(&first, 32),
                expected(4, 2),
                "{name}: the 4 by 2 imageblock"
            );
            assert_eq!(
                words(&second, 64),
                expected(4, 4),
                "{name}: the 4 by 4 imageblock"
            );
            let after = device.counters().metrics();
            let waits = |metrics: &crate::Metrics| {
                (
                    metrics.first_use_variant_waits,
                    metrics.first_use_unfinished_waits,
                )
            };
            let ((variant, unfinished), (variant_before, unfinished_before)) =
                (waits(&after), waits(&before));
            assert_eq!(
                (variant - variant_before, unfinished - unfinished_before),
                (2, 0),
                "{name}: a pipeline made at each size's first use, and no main pipeline"
            );
            assert!(
                key([4, 2]).1 && key([4, 4]).1,
                "{name}: each size's pipeline made"
            );
            println!("{name}: one encoder ran its imageblock at 4 by 2 and at 4 by 4");
        });
    }

    /// The encoder refuses, as Metal's validation layer does (job m3-max-70002), an imageblock side
    /// that is not a power of two, 0 included, and more than 1024 cells; it takes 1024 by 1 and 1
    /// by 1024.
    #[test]
    fn the_encoder_refuses_the_imageblock_sizes_metal_refuses() {
        each_device("imageblock_refusals", |name, _, device| {
            let queue = device.new_command_queue(1).expect("a queue");
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let mut encoder =
                (command_buffer.compute_command_encoder(DispatchType::Serial)).expect("an encoder");
            for [width, height] in [[24, 16], [16, 12], [0, 0], [0, 4], [64, 32]] {
                let refused = encoder.set_imageblock_size(width, height);
                assert!(
                    matches!(
                        refused,
                        Err(Error::Refused(Refusal::ImageblockSize { width: w, height: h }))
                            if [w, h] == [width, height]
                    ),
                    "{name}: {width} by {height}: {refused:?}"
                );
            }
            for [width, height] in [[1024, 1], [1, 1024], [32, 32], [1, 1]] {
                (encoder.set_imageblock_size(width, height))
                    .unwrap_or_else(|error| panic!("{name}: {width} by {height}: {error}"));
            }
            encoder.end_encoding();
        });
    }
}
