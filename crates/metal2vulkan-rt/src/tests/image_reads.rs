//! `read()` of a texture out of range, under a device cfg on every device of the cfg's kind: the
//! engine fixture `oor_texture_reads` reads ten texels of an `RGBA8Unorm` texture of 4 x 4 with
//! two levels, of a view of its first level alone, and of an `R8Unorm` texture, each answering
//! what Metal answered on an Apple M3 Max under macOS 27.0.1 (job m3-max-47527, the same source
//! compiled at run time): the texel inside, and zero with alpha 0 past the width, the height, the
//! texture's levels, a level's own width and the view's levels, where the format has alpha, and
//! zero with alpha 1 where it has none. The device's robustness answers them, with no check in
//! the shader (`robustImageAccess2`).
//!
//! `llvm-as` assembles the kernel; without it the test skips visibly.

#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod on_device {
    use crate::tests::command::on_device::{each_device, finished, Arena};
    use crate::{
        ComputePipelineDescriptor, Device, DispatchType, FunctionDescriptor, IndexClass, Library,
        PixelFormat, Region, StageSource, Texture, TextureDescriptor, TextureType, TextureUsage,
    };
    use metal2vulkan_devtools::{engine_dir, llvm_assemble, AssembleError};

    /// Each read's x, y and level, as the fixture takes them.
    const AT: [[u32; 3]; 11] = [
        [1, 1, 0],
        [4, 0, 0],
        [0, 4, 0],
        [0, 0, 2],
        [2, 0, 1],
        [1, 1, 1],
        [u32::MAX, 0, 0],
        [0, 0, 1],
        [1, 1, 0],
        [4, 0, 0],
        [0, 0, 1],
    ];

    /// What Metal answered for each read, its channels' bytes (x 255) (job m3-max-47527).
    const METAL: [[u8; 4]; 11] = [
        [50, 60, 30, 200],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [55, 57, 99, 150],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [69, 0, 0, 255],
        [0, 0, 0, 255],
        [0, 0, 0, 255],
    ];

    /// The fixture's library, or None where `llvm-as` is missing.
    fn library() -> Option<Library> {
        match llvm_assemble(&engine_dir().join("oor_texture_reads.ll")) {
            Ok(bitcode) => Some(Library::from_air(&bitcode).expect("the library")),
            Err(AssembleError::Missing(why)) => {
                eprintln!("SKIPPED: {why}");
                None
            }
            Err(error) => panic!("oor_texture_reads.ll: {error:?}"),
        }
    }

    /// A 2D texture of `side` texels a side with `levels` levels, shader-read usage, at `index`,
    /// each level's texels `texel(level, x, y)`, `bytes` bytes a texel.
    fn texture(
        device: &Device,
        format: PixelFormat,
        (side, levels, index): (u32, u32, u32),
        bytes: u32,
        texel: fn(u32, u32, u32) -> Vec<u8>,
    ) -> Texture {
        let mut d = TextureDescriptor::new(TextureType::Type2D, format, side, side);
        d.mipmap_level_count = levels;
        d.usage = TextureUsage::SHADER_READ;
        let texture =
            (device.new_texture(&d, Some(index))).unwrap_or_else(|error| panic!("{d:?}: {error}"));
        for level in 0..levels {
            let side = side >> level;
            let texels: Vec<u8> = (0..side)
                .flat_map(|y| (0..side).flat_map(move |x| texel(level, x, y)))
                .collect();
            let whole = Region {
                origin: [0; 3],
                size: [side, side, 1],
            };
            let (row, image) = (u64::from(side * bytes), u64::from(side * side * bytes));
            (texture.replace_region(whole, level, 0, &texels, row, image))
                .unwrap_or_else(|error| panic!("level {level}: {error}"));
        }
        texture
    }

    #[test]
    fn every_device_reads_a_texture_out_of_range_as_metal_does() {
        let Some(library) = library() else {
            return;
        };
        each_device("image reads out of range", |name, _, device| {
            let function = library.function("oor_texture_reads").expect("the kernel");
            let descriptor = ComputePipelineDescriptor::new(StageSource::Function(
                FunctionDescriptor::new(function),
            ));
            let state = (device.new_compute_pipeline_state(&descriptor, |_| Ok(Vec::new())))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let grant = (device.grant_indices(IndexClass::Texture, 1, 3))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            // Level 0's texel (x, y) is (10 + 40x, 20 + 40y, 30, 200), level 1's (5 + 50x,
            // 7 + 50y, 99, 150); the red texture's 64 + x + 4y.
            let rgba = |level: u32, x: u32, y: u32| match level {
                0 => vec![(10 + 40 * x) as u8, (20 + 40 * y) as u8, 30, 200],
                _ => vec![(5 + 50 * x) as u8, (7 + 50 * y) as u8, 99, 150],
            };
            let two_levels = texture(&device, PixelFormat::RGBA8Unorm, (4, 2, 1), 4, rgba);
            let one_level = (two_levels.new_texture_view(
                PixelFormat::RGBA8Unorm,
                TextureType::Type2D,
                0..1,
                0..1,
                Some(2),
            ))
            .unwrap_or_else(|error| panic!("{name}: the view: {error}"));
            let red = |_: u32, x: u32, y: u32| vec![(64 + x + 4 * y) as u8];
            let red = texture(&device, PixelFormat::R8Unorm, (4, 1, 3), 1, red);
            let mut arena = Arena::new(&device);
            let (at, out) = (arena.buffer(), arena.buffer());
            let words: Vec<u8> = (AT.iter())
                .flat_map(|&[x, y, level]| [x, y, level, 0])
                .flat_map(u32::to_le_bytes)
                .collect();
            let contents = at.contents().expect("a Shared buffer");
            // SAFETY: the buffer holds 256 bytes, more than the 176 written, and nothing reads it
            // meanwhile.
            unsafe { (words.as_ptr()).copy_to_nonoverlapping(contents.as_ptr(), words.len()) };
            let queue = device.new_command_queue(1).expect("a queue");
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let mut encoder =
                (command_buffer.compute_command_encoder(DispatchType::Serial)).expect("an encoder");
            encoder
                .set_compute_pipeline_state(&state)
                .expect("the state");
            for (texture, index) in [(&two_levels, 0), (&one_level, 1), (&red, 2)] {
                encoder
                    .set_texture(Some(texture), index)
                    .expect("a texture");
            }
            encoder.set_buffer(Some(&at), 0, 0).expect("at");
            encoder.set_buffer(Some(&out), 0, 1).expect("out");
            (encoder.dispatch_threadgroups([1, 1, 1], [1, 1, 1]))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            encoder.end_encoding();
            command_buffer.commit().expect("a commit");
            finished(name, &command_buffer);
            assert_eq!(command_buffer.error(), None, "{name}");
            let contents = out.contents().expect("a Shared buffer");
            // SAFETY: the dispatch that wrote the buffer completed, and its writes reach the host.
            let floats = unsafe { std::slice::from_raw_parts(contents.as_ptr().cast::<f32>(), 44) };
            for (read, (texel, metal)) in floats.chunks(4).zip(METAL).enumerate() {
                let bytes = texel.iter().map(|&v| (v * 255.0).round() as u8);
                assert_eq!(
                    bytes.collect::<Vec<_>>(),
                    metal,
                    "{name}: read {read} at {:?}: {texel:?}",
                    AT[read]
                );
            }
            drop((two_levels, one_level, red, command_buffer));
            (device.release_indices(grant)).unwrap_or_else(|error| panic!("{name}: {error}"));
            println!("image reads out of range: {name}: the 11 reads answer Metal's texels");
        });
    }
}
