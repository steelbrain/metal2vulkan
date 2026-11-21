//! A fragment function that samples with a `min_lod_clamp` (`floor`'s `MIN_LOD_CLAMPS`), under a
//! device cfg on every device of the cfg's kind: where device creation enabled
//! `shaderResourceMinLod`, the translator's profile has it, the state is made and each drawn pixel
//! reads the levels the clamp selects; elsewhere creation refuses the state, naming `MinLodClamp`.
//!
//! `sample_min_lod` (engine `sampling`) samples a 2D texture through the bound sampler at a
//! sixteenth of the pixel's position, clamped to level of detail 1.5. Over a texture 16 texels a
//! side the position's derivatives make the level of detail 0, so the clamp alone moves it: under a
//! linear mip filter, every pixel reads level 1 and level 2 half and half, where an unclamped
//! sample reads level 0.
//!
//! `llvm-as` assembles the functions; without it the test skips visibly.

#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod on_device {
    use crate::tests::command::on_device::each_device;
    use crate::tests::draw::on_device::{
        descriptor, examples, floats, library, pass, read, run, target, Buffers, SIDE,
    };
    use crate::{
        Error, FunctionDescriptor, HostBytes, IndexClass, PixelFormat, PrimitiveType, Region,
        SamplerDescriptor, SamplerMipFilter, StageSource, StorageMode, TextureDescriptor,
        TextureType, TextureUsage,
    };
    use metal2vulkan::device::VkFeature;
    use metal2vulkan::error::Feature;
    use metal2vulkan_devtools::engine_dir;

    /// The sampled texture's side, and its levels (16 texels down to 1).
    const TEXELS: u32 = 16;
    const LEVELS: u32 = 5;
    /// Each level's one colour: level 0 red, 1 green, 2 blue, then white.
    const COLOURS: [[u8; 4]; LEVELS as usize] = [
        [255, 0, 0, 255],
        [0, 255, 0, 255],
        [0, 0, 255, 255],
        [255, 255, 255, 255],
        [255, 255, 255, 255],
    ];

    /// Where Vulkan offers `shaderResourceMinLod`, a sample's `min_lod_clamp` raises its level of
    /// detail as Metal's does; elsewhere the state is refused naming the clamp.
    #[test]
    fn every_device_clamps_a_sample_s_level_where_it_offers_min_lod() {
        let (Some(examples), Some(sampling)) =
            (examples(), library(&engine_dir().join("sampling.ll")))
        else {
            return;
        };
        let mut d = descriptor(&examples, "colored_fragment", PixelFormat::RGBA8Unorm);
        d.fragment = Some(StageSource::Function(FunctionDescriptor::new(
            sampling.function("sample_min_lod").expect("the function"),
        )));
        each_device("min_lod", |name, _, device| {
            let core = device.core();
            let on = (core.translator.profile().enabled().features)
                .contains(&VkFeature::ShaderResourceMinLod);
            let state =
                match device.new_render_pipeline_state(&d, crate::tests::no_constexpr_samplers) {
                    Ok(state) => state,
                    Err(Error::Translation(metal2vulkan::error::Error::Unsupported(u))) if !on => {
                        assert_eq!(u.feature, Feature::MinLodClamp, "{name}: {u:?}");
                        println!("min_lod: {name}: no shaderResourceMinLod; the state is refused");
                        return;
                    }
                    Err(error) => panic!("{name}: {error}"),
                };
            assert!(on, "{name}: a state made without shaderResourceMinLod");
            let _grants = [IndexClass::Texture, IndexClass::Sampler].map(|class| {
                (device.grant_indices(class, 1, 1))
                    .unwrap_or_else(|error| panic!("{name}: {error}"))
            });
            let mut td = TextureDescriptor::new(
                TextureType::Type2D,
                PixelFormat::RGBA8Unorm,
                TEXELS,
                TEXELS,
            );
            td.mipmap_level_count = LEVELS;
            td.usage = TextureUsage::SHADER_READ;
            td.storage_mode = StorageMode::Shared;
            let texture = (device.new_texture(&td, Some(1)))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            for (level, colour) in (0..LEVELS).zip(COLOURS) {
                let side = TEXELS >> level;
                let bytes = colour.repeat((side * side) as usize);
                let whole = Region {
                    origin: [0; 3],
                    size: [side, side, 1],
                };
                let row = u64::from(side) * 4;
                (texture.replace_region(
                    whole,
                    level,
                    0,
                    HostBytes::from(&bytes[..]),
                    row,
                    row * u64::from(side),
                ))
                .unwrap_or_else(|error| panic!("{name}: level {level}: {error}"));
            }
            let mut sd = SamplerDescriptor::new();
            sd.mip_filter = SamplerMipFilter::Linear;
            let sampler = (device.new_sampler_state(&sd, 1))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let mut buffers = Buffers::new(&device);
            // `colored_vertex`'s attributes: a triangle that covers the target, and its colours.
            let positions = buffers.of(&device, &floats(&[-1.0, -1.0, 3.0, -1.0, -1.0, 3.0]));
            let colors = buffers.of(&device, &floats(&[0.0; 12]));
            let color = target(
                &device,
                PixelFormat::RGBA8Unorm,
                TextureUsage::RENDER_TARGET,
            );
            let queue = device.new_command_queue(4).expect("a queue");
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let ok = |result: Result<(), Error>| {
                result.unwrap_or_else(|error| panic!("{name}: {error}"))
            };
            let mut encoder = (command_buffer.render_command_encoder(&pass(&color, None)))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            ok(encoder.set_render_pipeline_state(&state));
            ok(encoder.set_vertex_buffer(Some(&positions), 0, 0));
            ok(encoder.set_vertex_buffer(Some(&colors), 0, 1));
            ok(encoder.set_fragment_texture(Some(&texture), 0));
            ok(encoder.set_fragment_sampler_state(Some(&sampler), 0));
            ok(encoder.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0));
            encoder.end_encoding();
            run(name, &command_buffer);
            // Level 1 and level 2 half and half: green and blue each 127.5, within the device's
            // filtering of two codes.
            let texels = read(&color, 4);
            for (at, texel) in texels.as_chunks::<4>().0.iter().enumerate() {
                let [r, g, b, a] = *texel;
                let near = |code: u8| code.abs_diff(128) <= 2;
                assert!(
                    r == 0 && near(g) && near(b) && a == 255,
                    "{name}: pixel {at}: {texel:?}, where levels 1 and 2 half and half are \
                     [0, 128, 128, 255]"
                );
            }
            println!(
                "min_lod: {name}: shaderResourceMinLod enabled; {} pixels read levels 1 and 2 \
                 half and half under a clamp of 1.5",
                SIDE * SIDE
            );
        });
    }
}
