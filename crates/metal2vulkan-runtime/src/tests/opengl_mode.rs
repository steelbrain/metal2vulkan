//! OpenGL mode (`PrivateRenderState::open_gl_mode_enabled`) makes two render states Metal refuses
//! without it, as an Apple M3 Max draws them (probe `conformance/probes/opengl_mode.m`, macOS
//! 27.0.1): a fragment function writing depth without a depth attachment draws its colors as with
//! one; a rasterizing state whose vertex function returns `void` runs the vertex function and
//! rasterizes nothing. The runtime draws the first as Metal does, and makes it without the mode
//! too, where only the M3 Max's driver refuses it ("depthAttachmentPixelFormat is not valid and
//! shader writes to depth"; docs/API.md, accepted differences). It refuses the second in either
//! mode, as the translator's link does (`DescriptorError::RasterizedVoidVertex`): without the mode
//! as Metal does ("RasterizationEnabled is true but the vertex shader's return type is void"), and
//! with it a known gap (the runtime's README, Known gaps of render pipeline states).
//!
//! Device tests, under a device cfg on every device of the cfg's kind. The functions are those of
//! the owned fixture `validation/fixtures/public/examples/icb_linked_arguments.ll`; `llvm-as`
//! assembles them, and without it the tests skip visibly.
//!
//! Not ported from the earlier runtime's tests of this state, for want of the part each tested:
//! - `only_a_draw_of_points_announces_the_device_s_point_size`: the runtime announces no notice
//!   at a draw; `DevicePointSize` is the state's static note (`point_size_note`).
//!
//! Apple's private device properties, which the consumer answers and the runtime holds none of:
//! - `every_property_has_one_selector`: each property's selector.
//! - `the_answers_are_the_probes`: the M3 Max's values of some properties.
//! - `metals_answers_are_the_undocumented_properties`: which properties answer Metal's values.
//! - `table_counts_and_sizes_are_what_the_runtime_takes`: table counts and texture sizes.
//! - `the_device_draws_the_public_primitive_types`: `supportsPrimitiveType:`.
//! - `predication_answers_where_it_runs`: command predication's properties.
//! - `the_core_count_and_fp16_are_the_devices`: `gpuCoreCount` and native half floats.
//! - `the_alignments_atomics_and_formats_are_the_devices`: alignments, atomics and formats.

#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod on_device {
    use crate::tests::command::on_device::each_device;
    use crate::tests::draw::on_device::{library, read, run, state, target, Buffers, SIDE};
    use crate::{
        Buffer, ClearColor, ColorAttachmentDescriptor, Device, Error, FunctionDescriptor,
        IndexClass, Library, LoadAction, PixelFormat, PrimitiveType, Region, RenderPassDescriptor,
        RenderPipelineDescriptor, RenderPipelineState, StageSource, StorageMode, Texture,
        TextureDescriptor, TextureType, TextureUsage,
    };
    use metal2vulkan::error::DescriptorError;
    use metal2vulkan_devtools::fixtures_dir;

    /// The texel `tint` holds at (1, 0), which the depth-writing fragment function writes as its
    /// depth.
    const TEXEL: [f32; 4] = [0.25, 0.5, 0.75, 1.0];

    /// The index `tint` takes.
    const TINT: u32 = 4096;

    /// The fixture's library, or None where `llvm-as` is missing.
    fn examples() -> Option<Library> {
        library(&fixtures_dir().join("examples/icb_linked_arguments.ll"))
    }

    fn stage(library: &Library, name: &str) -> StageSource {
        StageSource::Function(FunctionDescriptor::new(
            library.function(name).expect("the function"),
        ))
    }

    /// A descriptor of `vertex` and `fragment` drawing into `RGBA8Unorm`, in OpenGL mode where
    /// `gl`.
    fn descriptor(
        library: &Library,
        (vertex, fragment): (&str, Option<&str>),
        gl: bool,
    ) -> RenderPipelineDescriptor {
        let mut d = RenderPipelineDescriptor::new(stage(library, vertex));
        d.fragment = fragment.map(|name| stage(library, name));
        d.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
        d.private.open_gl_mode_enabled = gl;
        d
    }

    /// A 2 x 1 `RGBA32Float` texture holding [`TEXEL`] at (1, 0), at index [`TINT`].
    fn tint(name: &str, device: &Device) -> Texture {
        (device.grant_indices(IndexClass::Texture, TINT, 1))
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        let mut d = TextureDescriptor::new(TextureType::Type2D, PixelFormat::RGBA32Float, 2, 1);
        d.usage = TextureUsage::SHADER_READ;
        d.storage_mode = StorageMode::Shared;
        let texture =
            (device.new_texture(&d, Some(TINT))).unwrap_or_else(|error| panic!("{name}: {error}"));
        let bytes: Vec<u8> = [0.0; 4]
            .iter()
            .chain(&TEXEL)
            .flat_map(|v: &f32| v.to_le_bytes())
            .collect();
        let region = Region {
            origin: [0; 3],
            size: [2, 1, 1],
        };
        (texture.replace_region(region, 0, 0, crate::HostBytes::from(&bytes[..]), 32, 0))
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        texture
    }

    /// The bytes of a [`SIDE`]-square `RGBA8Unorm` target cleared to zero after one draw of three
    /// vertices by `state`, `tint` bound at vertex and fragment texture 0 and `out` at vertex
    /// buffer 0, and of `depth` where given, cleared to 1.
    fn drawn(
        name: &str,
        device: &Device,
        state: &RenderPipelineState,
        (tint, out): (&Texture, &Buffer),
        depth: Option<&Texture>,
    ) -> Vec<u8> {
        let target = target(device, PixelFormat::RGBA8Unorm, TextureUsage::RENDER_TARGET);
        let mut pass = RenderPassDescriptor::default();
        let color = &mut pass.color_attachments[0];
        color.texture = Some(target.clone());
        color.load_action = LoadAction::Clear;
        color.clear_color = ClearColor::new(0.0, 0.0, 0.0, 0.0);
        if let Some(depth) = depth {
            pass.depth_attachment.texture = Some(depth.clone());
            pass.depth_attachment.load_action = LoadAction::Clear;
            pass.depth_attachment.clear_depth = 1.0;
        }
        let queue = device.new_command_queue(4).expect("a queue");
        let command_buffer = queue.command_buffer().expect("a command buffer");
        let mut encoder = (command_buffer.render_command_encoder(&pass))
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        let ok = |result: Result<(), Error>| result.unwrap_or_else(|e| panic!("{name}: {e}"));
        ok(encoder.set_render_pipeline_state(state));
        ok(encoder.set_vertex_texture(Some(tint), 0));
        ok(encoder.set_fragment_texture(Some(tint), 0));
        ok(encoder.set_vertex_buffer(Some(out), 0, 0));
        ok(encoder.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0));
        encoder.end_encoding();
        run(name, &command_buffer);
        read(&target, 4)
    }

    /// With the mode, a fragment function writing depth without a depth attachment makes a
    /// state, and its colors are those it draws with a depth attachment.
    #[test]
    fn opengl_mode_makes_a_depth_output_without_a_depth_attachment() {
        let Some(library) = examples() else {
            return;
        };
        let functions = ("plain_vertex", Some("depth_writing_fragment"));
        let gl = descriptor(&library, functions, true);
        let mut attached = descriptor(&library, functions, false);
        attached.depth_format = Some(PixelFormat::Depth32Float);
        each_device("opengl_mode_depth", |name, _, device| {
            let unattached = state(name, &device, &gl);
            let attached = state(name, &device, &attached);
            let mut d =
                TextureDescriptor::new(TextureType::Type2D, PixelFormat::Depth32Float, SIDE, SIDE);
            d.usage = TextureUsage::RENDER_TARGET;
            let depth = (device.new_texture_for_views(&d))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let tint = tint(name, &device);
            let mut buffers = Buffers::new(&device);
            let out = buffers.of(&device, &[0; 48]);
            let without = drawn(name, &device, &unattached, (&tint, &out), None);
            let with = drawn(name, &device, &attached, (&tint, &out), Some(&depth));
            assert!(
                without.iter().any(|&b| b != 0),
                "{name}: the state drew nothing"
            );
            assert_eq!(without, with, "{name}");
            println!("{name}: OpenGL mode drew a depth output without a depth attachment");
        });
    }

    /// Asserts that `device` refuses a rasterizing state of the `void` vertex function as the
    /// translator's link does, in OpenGL mode where `gl`, with or without a fragment function.
    fn refuses_the_void_vertex_function(name: &str, device: &Device, library: &Library, gl: bool) {
        for fragment in [None, Some("inputless_fragment")] {
            let functions = ("void_store_vertex", fragment);
            let refused = device.new_render_pipeline_state(
                &descriptor(library, functions, gl),
                crate::tests::no_constexpr_samplers,
            );
            match refused {
                Err(Error::Translation {
                    error:
                        metal2vulkan::error::Error::InvalidDescriptor(
                            DescriptorError::RasterizedVoidVertex,
                        ),
                    ..
                }) => {}
                Err(other) => panic!("{name}: {fragment:?}: another error: {other:?}"),
                Ok(_) => panic!("{name}: {fragment:?}: made"),
            }
        }
    }

    /// Without the mode, a rasterizing state whose vertex function returns `void` is refused
    /// with Metal's error, with or without a fragment function.
    #[test]
    fn without_opengl_mode_a_rasterizing_void_vertex_function_is_refused() {
        let Some(library) = examples() else {
            return;
        };
        each_device("opengl_mode_void_refused", |name, _, device| {
            refuses_the_void_vertex_function(name, &device, &library, false);
            println!("{name}: a rasterizing void vertex function refused without OpenGL mode");
        });
    }

    /// With the mode too, a rasterizing state whose vertex function returns `void` is refused,
    /// with or without a fragment function, as the translator's link refuses it whatever the
    /// mode. Metal makes it there (an Apple M3 Max, macOS 27.0.1, the probe): a state whose draw
    /// runs the vertex function, its three stores landing, and rasterizes nothing. The difference
    /// is a known gap (the runtime's README, Known gaps of render pipeline states).
    #[test]
    fn opengl_mode_refuses_a_rasterizing_void_vertex_function() {
        let Some(library) = examples() else {
            return;
        };
        each_device("opengl_mode_void", |name, _, device| {
            refuses_the_void_vertex_function(name, &device, &library, true);
            println!("{name}: a rasterizing void vertex function refused in OpenGL mode");
        });
    }
}
