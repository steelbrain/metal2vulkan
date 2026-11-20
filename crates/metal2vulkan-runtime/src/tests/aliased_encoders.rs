//! Two textures over one buffer, and the buffer itself, across fence-ordered compute encoders of
//! one command buffer: under a device cfg, on every device of the cfg's kind. Encoder 1 writes
//! texture T2 over buffer B (`rows_write`), encoder 2 reads B through another texture T3 over B
//! (`rows_read`), and encoder 3 reads B as a buffer (`copy_words`); each waits for the fence the
//! one before it updated. The bytes are exact: B holds the texels encoder 1 wrote at its pitch and
//! the host's sentinels in every row remainder, encoder 2's output holds those texels, and encoder
//! 3's output holds B's bytes.
//!
//! It runs twice: with the textures in place over B (rows at the device's own pitch for the
//! format), and with each texture in an image of its own (rows at a wider pitch, which the
//! device's linear image does not take), whose copies the command buffer records. The run prints
//! which form each took.
//!
//! A second test bounds the copies where nothing aliases: one texture over a buffer in an image of
//! its own, written and read, takes the submission's leading copy in and its copy back, and a
//! second texture over bytes of the buffer the first's do not meet takes no copy for the first's
//! write. `llvm-as` assembles the kernels' libraries; without it the tests are skipped, visibly.

#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod on_device {
    use crate::tests::command::on_device::{each_device, finished};
    use crate::texture::TextureObject;
    use crate::{
        Buffer, ComputePipelineDescriptor, ComputePipelineState, Device, DispatchType,
        FunctionDescriptor, IndexClass, Library, MemoryClass, PixelFormat, StageSource, Texture,
        TextureDescriptor, TextureType, TextureUsage,
    };
    use metal2vulkan_devtools::{fixtures_dir, llvm_assemble, AssembleError};

    const WIDTH: u32 = 64;
    const HEIGHT: u32 = 16;
    const ROW: u64 = WIDTH as u64 * 4;
    /// Each buffer's place in its block: a multiple of every memory alignment.
    const AT: u64 = 65_536;

    /// The library of `ll`, or None where `llvm-as` is missing.
    fn library(ll: &str) -> Option<Library> {
        let path = fixtures_dir().join(ll);
        match llvm_assemble(&path) {
            Ok(bitcode) => Some(Library::from_air(&bitcode).expect("the library")),
            Err(AssembleError::Missing(why)) => {
                eprintln!("SKIPPED: aliased encoders: {why}");
                None
            }
            Err(error) => panic!("{}: {error:?}", path.display()),
        }
    }

    fn pipeline(
        name: &str,
        device: &Device,
        library: &Library,
        function: &str,
    ) -> ComputePipelineState {
        let source = StageSource::Function(FunctionDescriptor::new(
            library.function(function).expect("the function"),
        ));
        (device.new_compute_pipeline_state(
            &ComputePipelineDescriptor::new(source),
            crate::tests::no_constexpr_samplers,
        ))
        .unwrap_or_else(|error| panic!("{name}: {function}: {error}"))
    }

    /// The bytes of `buffer`, read by the host.
    fn held(buffer: &Buffer) -> Vec<u8> {
        let contents = buffer.contents().expect("Shared");
        // SAFETY: the buffer's bytes, which no command writes while the test reads them.
        unsafe { std::slice::from_raw_parts(contents.as_ptr(), buffer.length() as usize) }.to_vec()
    }

    /// Writes `bytes` over `buffer`'s from its start.
    fn write(buffer: &Buffer, bytes: &[u8]) {
        let contents = buffer.contents().expect("Shared");
        // SAFETY: the buffer holds `bytes.len()` bytes, and no command reads them meanwhile.
        unsafe {
            contents
                .as_ptr()
                .copy_from_nonoverlapping(bytes.as_ptr(), bytes.len())
        };
    }

    /// The bytes `rows_read` read into `buffer`: each channel's float, written as the byte it is
    /// over 255.
    fn channels(buffer: &Buffer) -> Vec<u8> {
        (held(buffer).as_chunks::<4>().0.iter())
            .map(|f| (f32::from_ne_bytes(*f) * 255.0).round().clamp(0.0, 255.0) as u8)
            .collect()
    }

    /// Texel `(x, y)`'s four channel bytes.
    fn texel(x: u32, y: u32) -> [u8; 4] {
        let at = |c: u32| ((x * 7 + y * 13 + c * 61 + 5) % 256) as u8;
        [at(0), at(1), at(2), at(3)]
    }

    /// The three kernels' states.
    struct Kernels {
        write: ComputePipelineState,
        read: ComputePipelineState,
        copy: ComputePipelineState,
    }

    /// One run of the three encoders over a buffer whose rows are `pitch` apart. True where both
    /// textures were in place.
    fn run(name: &str, device: &Device, kernels: &Kernels, pitch: u64) -> bool {
        let len = (u64::from(HEIGHT) * pitch).next_multiple_of(16);
        let texels = u64::from(WIDTH) * u64::from(HEIGHT);
        let block = (device.take_block(MemoryClass::Shared, 4 * AT + 2 * len + 16 * texels))
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        let place = |at: u64, length: u64| {
            (device.new_buffer(block.id, at, length, None))
                .unwrap_or_else(|error| panic!("{name}: {error}"))
        };
        let b = place(AT, len);
        let input = place(2 * AT, 16 * texels);
        let read_out = place(3 * AT, 16 * texels);
        let copy_out = place(4 * AT, len);

        // B: sentinels the host wrote, which the row remainders keep.
        let sentinels: Vec<u8> = (0..len as usize)
            .map(|i| ((i * 7 + 3) % 251) as u8 ^ 0xA5)
            .collect();
        write(&b, &sentinels);
        // The input, each channel's byte over 255: what `rows_write` stores as that byte.
        let mut floats = Vec::with_capacity(16 * texels as usize);
        for y in 0..HEIGHT {
            for x in 0..WIDTH {
                for c in texel(x, y) {
                    floats.extend_from_slice(&(f32::from(c) / 255.0).to_ne_bytes());
                }
            }
        }
        write(&input, &floats);
        write(&read_out, &vec![0; 16 * texels as usize]);
        write(&copy_out, &vec![0; len as usize]);

        let first = IndexClass::Texture.range().start;
        let mut d =
            TextureDescriptor::new(TextureType::Type2D, PixelFormat::RGBA8Unorm, WIDTH, HEIGHT);
        d.usage = TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE;
        let t2 = (b.new_texture(&d, Some(first), 0, pitch))
            .unwrap_or_else(|error| panic!("{name}: T2: {error}"));
        let t3 = (b.new_texture(&d, Some(first + 1), 0, pitch))
            .unwrap_or_else(|error| panic!("{name}: T3: {error}"));
        let in_place = t2.copied_over().is_none() && t3.copied_over().is_none();
        let copies = device.counters().metrics().over_copies;

        let queue = device.new_command_queue(4).expect("a queue");
        let fence = device.new_fence();
        let command_buffer = queue.command_buffer().expect("a command buffer");
        let done = |result: Result<(), crate::Error>| {
            result.unwrap_or_else(|error| panic!("{name}: {error}"))
        };
        let columns = WIDTH.to_ne_bytes();
        // E1: T2 written from the input.
        let mut e1 = (command_buffer.compute_command_encoder(DispatchType::Serial)).expect("E1");
        done(e1.set_compute_pipeline_state(&kernels.write));
        done(e1.set_buffer(Some(&input), 0, 0));
        done(e1.set_bytes(&columns, 1));
        done(e1.set_texture(Some(&t2), 0));
        done(e1.dispatch_threads([WIDTH, HEIGHT, 1], [8, 8, 1]));
        done(e1.update_fence(&fence));
        e1.end_encoding();
        // E2: B read through T3.
        let mut e2 = (command_buffer.compute_command_encoder(DispatchType::Serial)).expect("E2");
        done(e2.wait_for_fence(&fence));
        done(e2.set_compute_pipeline_state(&kernels.read));
        done(e2.set_texture(Some(&t3), 0));
        done(e2.set_buffer(Some(&read_out), 0, 0));
        done(e2.set_bytes(&columns, 1));
        done(e2.dispatch_threads([WIDTH, HEIGHT, 1], [8, 8, 1]));
        done(e2.update_fence(&fence));
        e2.end_encoding();
        // E3: B read as a buffer.
        let mut e3 = (command_buffer.compute_command_encoder(DispatchType::Serial)).expect("E3");
        done(e3.wait_for_fence(&fence));
        done(e3.set_compute_pipeline_state(&kernels.copy));
        done(e3.set_buffer(Some(&b), 0, 0));
        done(e3.set_buffer(Some(&copy_out), 0, 1));
        done(e3.dispatch_threads([(len / 16) as u32, 1, 1], [16, 1, 1]));
        e3.end_encoding();
        command_buffer.commit().expect("a commit");
        finished(name, &command_buffer);
        assert_eq!(command_buffer.error(), None, "{name}");

        let form = match in_place {
            true => "in place",
            false => "in images of their own",
        };
        let moved = device.counters().metrics().over_copies - copies;
        // B: the texels at their pitch, the sentinels in every row remainder.
        let mut expected = sentinels;
        for y in 0..HEIGHT {
            for x in 0..WIDTH {
                let at = (u64::from(y) * pitch + u64::from(x) * 4) as usize;
                expected[at..at + 4].copy_from_slice(&texel(x, y));
            }
        }
        let first_wrong = |what: &str, now: &[u8], want: &[u8]| {
            if let Some(i) = (0..want.len()).find(|&i| now[i] != want[i]) {
                let wrong = (0..want.len()).filter(|&i| now[i] != want[i]).count();
                panic!(
                    "{name}: textures {form} (pitch {pitch}, {moved} copies): {what}: byte {i} is \
                     {:#04x}, not {:#04x}; {wrong} bytes differ",
                    now[i], want[i]
                );
            }
        };
        first_wrong("B after encoder 1", &held(&b), &expected);
        first_wrong("encoder 3's copy of B", &held(&copy_out), &expected);
        // Encoder 2's output: each texel's channels as floats, each the byte over 255.
        let read = channels(&read_out);
        let mut want = Vec::with_capacity(read.len());
        for y in 0..HEIGHT {
            for x in 0..WIDTH {
                want.extend_from_slice(&texel(x, y));
            }
        }
        first_wrong("encoder 2's read through T3", &read, &want);
        println!(
            "aliased encoders: {name}: textures {form} (pitch {pitch}), {moved} copies: B, its \
             read through T3 and its copy as a buffer exact"
        );
        drop((t2, t3, b, input, read_out, copy_out, queue));
        (device.release_block(block)).unwrap_or_else(|error| panic!("{name}: {error}"));
        in_place
    }

    /// One command buffer over buffer B, whose rows are `pitch` apart, a pitch the device's linear
    /// image does not take: T2 over B's first `HEIGHT` rows, in an image of its own. Alone,
    /// encoder 1 writes T2 and encoder 2 reads it. With `beside`, T3 lies over the next `HEIGHT`
    /// rows, bytes T2's do not meet: encoder 1 reads T3, encoder 2 writes T2 and encoder 3 reads
    /// T3 again. B and the reads are exact; the answer is the copies the command buffer took.
    fn unaliased(name: &str, device: &Device, kernels: &Kernels, pitch: u64, beside: bool) -> u64 {
        let len = (2 * u64::from(HEIGHT) * pitch).next_multiple_of(16);
        let texels = u64::from(WIDTH) * u64::from(HEIGHT);
        let block = (device.take_block(MemoryClass::Shared, 5 * AT))
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        let place = |at: u64, length: u64| {
            (device.new_buffer(block.id, at, length, None))
                .unwrap_or_else(|error| panic!("{name}: {error}"))
        };
        let b = place(AT, len);
        let input = place(2 * AT, 16 * texels);
        let first_read = place(3 * AT, 16 * texels);
        let second_read = place(4 * AT, 16 * texels);
        let sentinels: Vec<u8> = (0..len as usize)
            .map(|i| ((i * 7 + 3) % 251) as u8 ^ 0xA5)
            .collect();
        write(&b, &sentinels);
        let mut floats = Vec::with_capacity(16 * texels as usize);
        for y in 0..HEIGHT {
            for x in 0..WIDTH {
                for c in texel(x, y) {
                    floats.extend_from_slice(&(f32::from(c) / 255.0).to_ne_bytes());
                }
            }
        }
        write(&input, &floats);

        let first = IndexClass::Texture.range().start;
        let mut d =
            TextureDescriptor::new(TextureType::Type2D, PixelFormat::RGBA8Unorm, WIDTH, HEIGHT);
        d.usage = TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE;
        let t2 = (b.new_texture(&d, Some(first), 0, pitch))
            .unwrap_or_else(|error| panic!("{name}: T2: {error}"));
        let t3 = (b.new_texture(&d, Some(first + 1), u64::from(HEIGHT) * pitch, pitch))
            .unwrap_or_else(|error| panic!("{name}: T3: {error}"));
        assert!(
            t2.copied_over().is_some() && t3.copied_over().is_some(),
            "{name}: pitch {pitch} in place"
        );
        let copies = device.counters().metrics().over_copies;

        let queue = device.new_command_queue(4).expect("a queue");
        let fence = device.new_fence();
        let command_buffer = queue.command_buffer().expect("a command buffer");
        let done = |result: Result<(), crate::Error>| {
            result.unwrap_or_else(|error| panic!("{name}: {error}"))
        };
        let columns = WIDTH.to_ne_bytes();
        let mut waits = false;
        let mut encode = |write: bool, texture: &Texture, out: &Buffer| {
            let mut e = (command_buffer.compute_command_encoder(DispatchType::Serial)).expect("E");
            if waits {
                done(e.wait_for_fence(&fence));
            }
            let (state, buffer) = match write {
                true => (&kernels.write, &input),
                false => (&kernels.read, out),
            };
            done(e.set_compute_pipeline_state(state));
            done(e.set_buffer(Some(buffer), 0, 0));
            done(e.set_bytes(&columns, 1));
            done(e.set_texture(Some(texture), 0));
            done(e.dispatch_threads([WIDTH, HEIGHT, 1], [8, 8, 1]));
            done(e.update_fence(&fence));
            e.end_encoding();
            waits = true;
        };
        match beside {
            false => {
                encode(true, &t2, &input);
                encode(false, &t2, &first_read);
            }
            true => {
                encode(false, &t3, &first_read);
                encode(true, &t2, &input);
                encode(false, &t3, &second_read);
            }
        }
        command_buffer.commit().expect("a commit");
        finished(name, &command_buffer);
        assert_eq!(command_buffer.error(), None, "{name}");
        let moved = device.counters().metrics().over_copies - copies;

        let first_wrong = |what: &str, now: &[u8], want: &[u8]| {
            if let Some(i) = (0..want.len()).find(|&i| now[i] != want[i]) {
                panic!(
                    "{name}: beside {beside} (pitch {pitch}, {moved} copies): {what}: byte {i} is \
                     {:#04x}, not {:#04x}",
                    now[i], want[i]
                );
            }
        };
        // B: T2's texels over its rows; the sentinels elsewhere.
        let mut expected = sentinels.clone();
        let mut written = Vec::with_capacity(4 * texels as usize);
        let mut under_t3 = Vec::with_capacity(4 * texels as usize);
        for y in 0..HEIGHT {
            for x in 0..WIDTH {
                let at = (u64::from(y) * pitch + u64::from(x) * 4) as usize;
                expected[at..at + 4].copy_from_slice(&texel(x, y));
                written.extend_from_slice(&texel(x, y));
                let below = at + (u64::from(HEIGHT) * pitch) as usize;
                under_t3.extend_from_slice(&sentinels[below..below + 4]);
            }
        }
        first_wrong("B", &held(&b), &expected);
        match beside {
            false => first_wrong("encoder 2's read of T2", &channels(&first_read), &written),
            true => {
                first_wrong("encoder 1's read of T3", &channels(&first_read), &under_t3);
                first_wrong("encoder 3's read of T3", &channels(&second_read), &under_t3);
            }
        }
        println!("unaliased overs: {name}: beside {beside} (pitch {pitch}): {moved} copies");
        drop((t2, t3, b, input, first_read, second_read, queue));
        (device.release_block(block)).unwrap_or_else(|error| panic!("{name}: {error}"));
        moved
    }

    /// Runs `body` on every device of the cfg's kind with the kernels, two texture indices granted
    /// and the device's linear texture alignment for the format; nothing where `llvm-as` is
    /// missing.
    fn each(test: &str, mut body: impl FnMut(&str, &Device, &Kernels, u64)) {
        let (Some(rows), Some(words)) = (
            library("examples/texel_rows.ll"),
            library("harness/workloads.ll"),
        ) else {
            return;
        };
        each_device(test, |name, _, device| {
            let kernels = Kernels {
                write: pipeline(name, &device, &rows, "rows_write"),
                read: pipeline(name, &device, &rows, "rows_read"),
                copy: pipeline(name, &device, &words, "copy_words"),
            };
            let first = IndexClass::Texture.range().start;
            (device.grant_indices(IndexClass::Texture, first, 2)).expect("a grant");
            let answer = (device.minimum_linear_texture_alignment(PixelFormat::RGBA8Unorm))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            body(name, &device, &kernels, answer);
        });
    }

    #[test]
    fn two_textures_over_a_buffer_and_the_buffer_across_fenced_encoders_read_what_was_written() {
        each("aliased encoders", |name, device, kernels, answer| {
            let own = ROW.next_multiple_of(answer);
            let in_place = run(name, device, kernels, own);
            // A pitch the device's linear image does not take: each texture in an image of its
            // own.
            let wider = own + answer.max(256);
            let copied = !run(name, device, kernels, wider);
            println!(
                "aliased encoders: {name}: forms reached: in place {in_place}, own images {copied}"
            );
        });
    }

    #[test]
    fn textures_over_a_buffer_in_images_of_their_own_with_no_alias_take_no_copy_more() {
        each("unaliased overs", |name, device, kernels, answer| {
            let wider = ROW.next_multiple_of(answer) + answer.max(256);
            // The submission's leading copy into T2, and T2's copy back at the end.
            assert_eq!(
                unaliased(name, device, kernels, wider, false),
                2,
                "{name}: alone"
            );
            // The leading copies into T3 and T2, and T2's copy back at the end: T2's write does
            // not stale T3, whose bytes do not meet its.
            assert_eq!(
                unaliased(name, device, kernels, wider, true),
                3,
                "{name}: beside"
            );
        });
    }
}
