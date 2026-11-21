//! Layered rendering (`RenderPassDescriptor::render_target_array_length`), under a device cfg on
//! every device of the cfg's kind: the floor's `shaderOutputLayer` is enabled, a vertex function's
//! `[[render_target_array_index]]` `i` renders slice `slice + i` of a layered pass, an index at or
//! past the pass's length renders its first slice, the pass's clear covers every layer, a pass of
//! one layer renders its slice whatever index the vertex function writes, and the fragment
//! function reads the index as written, in each, over a 2D array, a cube and a cube array (its
//! faces counted across its cubes). A vertex function whose index may differ among a
//! primitive's vertices is refused at creation, naming `LayeredRendering`.
//!
//! `rt6_layered_vertex` (engine `rt6_layered`) draws the fullscreen triangle once per instance,
//! instance `i` writing index `layers[i]`; `rt6_layered_fragment` writes `(i + 1) << 8 | index`.
//! `rt6_first_vertex` writes each vertex's own index (`indices[vid]`).
//!
//! `llvm-as` assembles the functions; without it the test skips visibly.

#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod on_device {
    use crate::tests::command::on_device::each_device;
    use crate::tests::draw::on_device::{library, run, Buffers, SIDE};
    use crate::{
        ClearColor, ColorAttachmentDescriptor, Device, Error, FunctionDescriptor, HostBytes,
        LoadAction, PixelFormat, PrimitiveType, Region, RenderPassDescriptor,
        RenderPipelineDescriptor, RenderPipelineState, StageSource, StorageMode, Texture,
        TextureDescriptor, TextureType, TextureUsage,
    };
    use metal2vulkan::device::VkFeature;
    use metal2vulkan::error::Feature;
    use metal2vulkan_devtools::engine_dir;

    /// What priming leaves in every slice before a pass: no fragment's word and not the clear.
    const PRIOR: u32 = 99;
    /// What a pass's `Clear` load leaves in a layer it covers: no fragment's word.
    const CLEAR: u32 = 7;

    /// The word `rt6_layered_fragment` writes for instance `instance`, which wrote `index`.
    const fn word(instance: u32, index: u32) -> u32 {
        (instance + 1) << 8 | index
    }

    /// One pass and what each slice of its target holds after it.
    struct Layered {
        what: &'static str,
        texture_type: TextureType,
        slices: u32,
        slice: u32,
        length: u32,
        /// The index each instance writes.
        layers: &'static [u32],
        /// What each slice of the target holds after the pass.
        expect: &'static [u32],
    }

    const PASSES: [Layered; 5] = [
        // Slices 1 to 3 of four: instance 0 renders layer 1 (slice 2), instance 1 layer 0
        // (slice 1), and instance 2's index 5, past the length, layer 0 too, over instance 1;
        // layer 2 (slice 3) keeps the clear, slice 0 its prior.
        Layered {
            what: "a 2D array's slices 1 to 3",
            texture_type: TextureType::Type2DArray,
            slices: 4,
            slice: 1,
            length: 3,
            layers: &[1, 0, 5],
            expect: &[PRIOR, word(2, 5), word(0, 1), CLEAR],
        },
        // A cube's six faces: faces 5 and 2 drawn, the other four keep the clear.
        Layered {
            what: "a cube's six faces",
            texture_type: TextureType::Cube,
            slices: 6,
            slice: 0,
            length: 6,
            layers: &[5, 2],
            expect: &[CLEAR, CLEAR, word(1, 2), CLEAR, CLEAR, word(0, 5)],
        },
        // Six faces of a cube array of two cubes, slices 4 to 9, across both cubes: index 5
        // renders slice 9 (the second cube's face 3), index 0 slice 4 (the first cube's face 4);
        // slices 5 to 8 keep the clear, the four outside the pass their prior.
        Layered {
            what: "a cube array's slices 4 to 9",
            texture_type: TextureType::CubeArray,
            slices: 12,
            slice: 4,
            length: 6,
            layers: &[5, 0],
            expect: &[
                PRIOR,
                PRIOR,
                PRIOR,
                PRIOR,
                word(1, 0),
                CLEAR,
                CLEAR,
                CLEAR,
                CLEAR,
                word(0, 5),
                PRIOR,
                PRIOR,
            ],
        },
        // Passes of one layer, of Metal's default length 0 and of length 1: index 3 renders the
        // attachment's slice, and the fragment reads 3.
        Layered {
            what: "a 2D array's slice 1 alone, its vertex function writing index 3",
            texture_type: TextureType::Type2DArray,
            slices: 2,
            slice: 1,
            length: 0,
            layers: &[3],
            expect: &[PRIOR, word(0, 3)],
        },
        Layered {
            what: "a 2D texture in a pass of length 1, its vertex function writing index 3",
            texture_type: TextureType::Type2D,
            slices: 1,
            slice: 0,
            length: 1,
            layers: &[3],
            expect: &[word(0, 3)],
        },
    ];

    /// A Shared render target of `R32Uint`, `SIDE` a side, every slice primed to [`PRIOR`].
    fn target(device: &Device, texture_type: TextureType, slices: u32) -> Texture {
        let mut d = TextureDescriptor::new(texture_type, PixelFormat::R32Uint, SIDE, SIDE);
        match texture_type {
            TextureType::Type2DArray => d.array_length = slices,
            TextureType::CubeArray => d.array_length = slices / 6,
            _ => {}
        }
        d.usage = TextureUsage::RENDER_TARGET;
        d.storage_mode = StorageMode::Shared;
        let texture = (device.new_texture_for_views(&d)).expect("a target");
        let row = (SIDE * 4) as u64;
        let prior: Vec<u8> = (0..SIDE * SIDE).flat_map(|_| PRIOR.to_le_bytes()).collect();
        for slice in 0..slices {
            (texture.replace_region(whole(), 0, slice, HostBytes::from(&prior[..]), row, 0))
                .expect("a primed slice");
        }
        texture
    }

    fn whole() -> Region {
        Region {
            origin: [0; 3],
            size: [SIDE, SIDE, 1],
        }
    }

    /// The words of slice `slice` of `texture`.
    fn words(texture: &Texture, slice: u32) -> Vec<u32> {
        let row = (SIDE * 4) as u64;
        let mut bytes = vec![0; (row * u64::from(SIDE)) as usize];
        (texture.get_bytes(&mut bytes, row, 0, whole(), 0, slice)).expect("get_bytes");
        (bytes.as_chunks::<4>().0.iter())
            .map(|w| u32::from_le_bytes(*w))
            .collect()
    }

    /// Draws `pass` with `state` and checks every texel of every slice of its target.
    fn draw(name: &str, device: &Device, state: &RenderPipelineState, pass: &Layered) -> u32 {
        let ok = |result: Result<(), Error>| {
            result.unwrap_or_else(|error| panic!("{name}: {}: {error}", pass.what))
        };
        let texture = target(device, pass.texture_type, pass.slices);
        let mut d = RenderPassDescriptor::default();
        let c = &mut d.color_attachments[0];
        c.texture = Some(texture.clone());
        c.slice = pass.slice;
        c.load_action = LoadAction::Clear;
        c.clear_color = ClearColor {
            red: f64::from(CLEAR),
            green: 0.0,
            blue: 0.0,
            alpha: 1.0,
        };
        d.render_target_array_length = pass.length;
        let mut buffers = Buffers::new(device);
        let layers: Vec<u8> = (pass.layers.iter()).flat_map(|l| l.to_le_bytes()).collect();
        let layers = buffers.of(device, &layers);
        let queue = device.new_command_queue(4).expect("a queue");
        let command_buffer = queue.command_buffer().expect("a command buffer");
        let mut encoder = (command_buffer.render_command_encoder(&d))
            .unwrap_or_else(|error| panic!("{name}: {}: {error}", pass.what));
        ok(encoder.set_render_pipeline_state(state));
        ok(encoder.set_vertex_buffer(Some(&layers), 0, 0));
        let instances = pass.layers.len() as u32;
        ok(encoder.draw_primitives(PrimitiveType::Triangle, 0, 3, instances, 0));
        encoder.end_encoding();
        run(name, &command_buffer);
        for (slice, &want) in pass.expect.iter().enumerate() {
            let got = words(&texture, slice as u32);
            assert!(
                got.iter().all(|&w| w == want),
                "{name}: {}: slice {slice} holds {:?}, Metal's {want:#x}",
                pass.what,
                (got.iter().take(4))
                    .map(|w| format!("{w:#x}"))
                    .collect::<Vec<_>>()
            );
        }
        pass.expect.len() as u32 * SIDE * SIDE
    }

    /// Every device renders Metal's layers, from one form of the vertex function, and refuses a
    /// vertex function whose index may differ among a primitive's vertices.
    #[test]
    fn every_device_renders_metal_s_layers() {
        let Some(layered) = library(&engine_dir().join("rt6_layered.ll")) else {
            return;
        };
        let Some(first_vertex) = library(&engine_dir().join("rt6_first_vertex.ll")) else {
            return;
        };
        let function = |library: &crate::Library, name| {
            StageSource::Function(FunctionDescriptor::new(
                library.function(name).expect("the function"),
            ))
        };
        let descriptor = |library, vertex, fragment| {
            let mut d = RenderPipelineDescriptor::new(function(library, vertex));
            d.fragment = Some(function(library, fragment));
            d.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::R32Uint));
            d
        };
        each_device("layered", |name, _, device| {
            let core = device.core();
            let enabled = &core.translator.profile().enabled().features;
            assert!(
                enabled.contains(&VkFeature::ShaderOutputLayer),
                "{name}: shaderOutputLayer, which the floor requires"
            );
            let layered_state = (device.new_render_pipeline_state(
                &descriptor(&layered, "rt6_layered_vertex", "rt6_layered_fragment"),
                crate::tests::no_constexpr_samplers,
            ))
            .unwrap_or_else(|error| panic!("{name}: rt6_layered: {error}"));
            let texels: u32 = (PASSES.iter())
                .map(|pass| draw(name, &device, &layered_state, pass))
                .sum();
            let differing = device.new_render_pipeline_state(
                &descriptor(&first_vertex, "rt6_first_vertex", "rt6_first_vertex_read"),
                crate::tests::no_constexpr_samplers,
            );
            let refused = match differing {
                Err(Error::Translation {
                    error: metal2vulkan::error::Error::Unsupported(u),
                    ..
                }) => {
                    assert_eq!(u.feature, Feature::LayeredRendering, "{name}: {u:?}");
                    u
                }
                Err(error) => panic!("{name}: rt6_first_vertex: {error}"),
                Ok(_) => panic!("{name}: rt6_first_vertex's differing index made a state"),
            };
            println!(
                "layered: {name}: shaderOutputLayer enabled; {texels} texels of {} passes read \
                 Metal's layers; a differing index refused: {:?} needs {:?}",
                PASSES.len(),
                refused.feature,
                refused.needs
            );
        });
    }
}
