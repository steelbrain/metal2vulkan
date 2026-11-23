//! An upload into a 32-bit float depth, under a device cfg on every device of the cfg's kind, as
//! a guest fills a shadow map's texture: a 4 by 4 `Depth32Float` texture takes 16 depths within
//! `[0, 1]`, 0.5 among them, a Private one by a blit's copy from a buffer and a Shared one by
//! `replace_region`.
//!
//! - Each texture reads back bit for bit through a blit's copy into a buffer.
//! - The owned fixture `rrp_mirror_clamp`'s `rrp_mc_compare` compares each texel through a sampler
//!   state of nearest filters and `Less` at a reference below its depth, on it and above it,
//!   which pass, fail and fail, as Metal compares them: measured on an Apple M3 Max, an AMD Radeon
//!   Pro 5300M and an Intel UHD Graphics 630, which keep every uploaded `float` and compare it
//!   unclamped.
//! - Each upload carries `Texture::FLOAT_DEPTH_UPLOAD_NOTE`, `PrecisionNote::RestrictedDepthUpload`.
//!
//! `llvm-as` assembles the fixture; without it the test skips visibly.

#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod on_device {
    use crate::tests::command::on_device::{each_device, finished};
    use crate::tests::draw::on_device::{floats, Buffers};
    use crate::{
        Buffer, BufferLayout, CompareFunction, ComputePipelineDescriptor, DispatchType,
        FunctionDescriptor, HostBytes, IndexClass, Library, PixelFormat, Region, SamplerDescriptor,
        StageSource, StorageMode, TextureDescriptor, TexturePlace, TextureType, TextureUsage,
    };
    use metal2vulkan_devtools::{engine_dir, llvm_assemble, AssembleError};

    /// The texture's side.
    const SIDE: u32 = 4;
    /// How far a reference lies below or above a texel's depth: half the depths' step.
    const DELTA: f32 = 1.0 / 32.0;
    /// The `float4`s `rrp_mc_compare` writes a lane; the comparison with no offset is the first.
    const PER_LANE: usize = 5;

    /// Texel `i`'s depth: `i / 16`, and 1.0 at the last, each exact in a `float`.
    fn depth(i: u32) -> f32 {
        match i {
            15 => 1.0,
            _ => i as f32 / 16.0,
        }
    }

    /// The lanes compared: each texel's centre with a reference below its depth, on it and above
    /// it, those within `[0, 1]`, and whether `Less` passes there.
    fn lanes() -> Vec<([f32; 2], f32, bool)> {
        let mut out = Vec::new();
        for i in 0..SIDE * SIDE {
            let at = [i % SIDE, i / SIDE].map(|c| (c as f32 + 0.5) / SIDE as f32);
            let d = depth(i);
            for reference in [d - DELTA, d, d + DELTA] {
                if (0.0..=1.0).contains(&reference) {
                    out.push((at, reference, reference < d));
                }
            }
        }
        out
    }

    /// A Shared buffer's bytes, as the host reads them.
    fn read(buffer: &Buffer) -> Vec<u8> {
        let contents = buffer.contents().expect("a Shared buffer");
        // SAFETY: the buffer is alive, `length` bytes long, and no GPU work uses it.
        unsafe { std::slice::from_raw_parts(contents.as_ptr(), buffer.length() as usize) }.to_vec()
    }

    #[test]
    fn every_device_compares_and_reads_back_an_uploaded_float_depth() {
        let library = match llvm_assemble(&engine_dir().join("rrp_mirror_clamp.ll")) {
            Ok(bitcode) => Library::from_air(&bitcode).expect("the library"),
            Err(AssembleError::Missing(why)) => {
                eprintln!("SKIPPED: {why}");
                return;
            }
            Err(error) => panic!("rrp_mirror_clamp.ll: {error:?}"),
        };
        let depths: Vec<f32> = (0..SIDE * SIDE).map(depth).collect();
        let bytes = floats(&depths);
        let lanes = lanes();
        each_device("float_depth_upload", |name, _, device| {
            let ok = |result: Result<(), crate::Error>, what: &str| {
                result.unwrap_or_else(|error| panic!("{name}: {what}: {error}"))
            };
            let _textures = (device.grant_indices(IndexClass::Texture, 1, 2))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let _samplers = (device.grant_indices(IndexClass::Sampler, 1, 1))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let mut sampler = SamplerDescriptor::new();
            sampler.compare_function = CompareFunction::Less;
            let sampler = (device.new_sampler_state(&sampler, 1))
                .unwrap_or_else(|error| panic!("{name}: the sampler: {error}"));
            let function = library.function("rrp_mc_compare").expect("the kernel");
            let descriptor = ComputePipelineDescriptor::new(StageSource::Function(
                FunctionDescriptor::new(function),
            ));
            let state = (device.new_compute_pipeline_state(&descriptor, |numbered| {
                Ok((2..).take(numbered.len()).collect())
            }))
            .unwrap_or_else(|error| panic!("{name}: the state: {error}"));

            let make = |storage, index| {
                let mut d = TextureDescriptor::new(
                    TextureType::Type2D,
                    PixelFormat::Depth32Float,
                    SIDE,
                    SIDE,
                );
                d.usage = TextureUsage::SHADER_READ;
                d.storage_mode = storage;
                (device.new_texture(&d, Some(index)))
                    .unwrap_or_else(|error| panic!("{name}: the texture: {error}"))
            };
            let private = make(StorageMode::Private, 1);
            let shared = make(StorageMode::Shared, 2);
            let whole = Region {
                origin: [0; 3],
                size: [SIDE, SIDE, 1],
            };
            let row = u64::from(4 * SIDE);
            ok(
                shared.replace_region(whole, 0, 0, HostBytes::from(&bytes[..]), row, row * 4),
                "replace_region",
            );

            let mut buffers = Buffers::new(&device);
            let source = buffers.of(&device, &bytes);
            let backs = [0xee; 2].map(|fill| buffers.of(&device, &vec![fill; bytes.len()]));
            let coords: Vec<f32> = (lanes.iter())
                .flat_map(|&([s, t], reference, _)| [s, t, reference, 0.0])
                .collect();
            let coords = buffers.of(&device, &floats(&coords));
            let outs =
                [0, 1].map(|_| buffers.of(&device, &vec![0xee; 16 * PER_LANE * lanes.len()]));

            let queue = device.new_command_queue(1).expect("a queue");
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let layout = BufferLayout {
                offset: 0,
                bytes_per_row: row,
                bytes_per_image: row * u64::from(SIDE),
            };
            let at = TexturePlace {
                slice: 0,
                level: 0,
                origin: [0; 3],
            };
            let extent = [SIDE, SIDE, 1];
            let mut blit = (command_buffer.blit_command_encoder())
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            ok(
                blit.copy_from_buffer(&source, layout, extent, &private, at),
                "the copy in",
            );
            for (texture, back) in [&private, &shared].into_iter().zip(&backs) {
                ok(
                    blit.copy_from_texture_to_buffer(texture, at, extent, back, layout),
                    "the copy out",
                );
            }
            blit.end_encoding();
            let mut encoder =
                (command_buffer.compute_command_encoder(DispatchType::Serial)).expect("an encoder");
            encoder
                .set_compute_pipeline_state(&state)
                .expect("the state");
            encoder
                .set_sampler_state(Some(&sampler), 0)
                .expect("the sampler");
            encoder.set_buffer(Some(&coords), 0, 0).expect("the lanes");
            let count = lanes.len() as u32;
            for (texture, out) in [&private, &shared].into_iter().zip(&outs) {
                encoder.set_texture(Some(texture), 0).expect("the texture");
                encoder.set_buffer(Some(out), 0, 1).expect("the output");
                (encoder.dispatch_threads([count, 1, 1], [count.min(64), 1, 1]))
                    .unwrap_or_else(|error| panic!("{name}: the dispatch: {error}"));
            }
            encoder.end_encoding();
            command_buffer.commit().expect("a commit");
            finished("float depth upload", &command_buffer);
            assert_eq!(command_buffer.error(), None, "{name}");

            for (way, (back, out)) in ["blit", "replace_region"]
                .iter()
                .zip(backs.iter().zip(&outs))
            {
                assert_eq!(read(back), bytes, "{name}: {way}: the depths read back");
                let words = read(out);
                let compared = |lane: usize| {
                    let at = 16 * PER_LANE * lane;
                    f32::from_le_bytes(words[at..at + 4].try_into().expect("a word"))
                };
                for (lane, &(at, reference, passes)) in lanes.iter().enumerate() {
                    let want = if passes { 1.0 } else { 0.0 };
                    assert_eq!(
                        compared(lane),
                        want,
                        "{name}: {way}: reference {reference} at {at:?}"
                    );
                }
                println!(
                    "{name}: {way}: {} depths read back bit for bit, {} comparisons as Metal's, \
                     0.5 compared at 0.46875 reads 1",
                    depths.len(),
                    lanes.len()
                );
            }
        });
    }
}
