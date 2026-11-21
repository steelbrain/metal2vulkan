//! A fragment function's raster order group on memory it reads and writes (`floor`'s
//! `PIXEL_INTERLOCK`), under a device cfg on every device of the cfg's kind: where device creation
//! enabled `VK_EXT_fragment_shader_interlock`, the translator's profile has
//! `fragmentShaderPixelInterlock`, the state is made and the draw answers Metal's values; elsewhere
//! creation refuses the state, naming `RasterOrderGroups`.
//!
//! `rog_buffer` (engine `fragment_rog`) adds 1 to its pixel's float in a buffer in raster order
//! group 0 and writes the sum, under four whole-target triangles in one draw, so each pixel's float
//! ends at 4 and its texel holds the fourth fragment's sum.
//!
//! `llvm-as` assembles the function; without it the test skips visibly.

#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod on_device {
    use crate::tests::command::on_device::each_device;
    use crate::tests::draw::on_device::{
        descriptor, examples, floats, library, pass, read, run, target, Buffers, SIDE,
    };
    use crate::{Error, FunctionDescriptor, PixelFormat, PrimitiveType, StageSource, TextureUsage};
    use metal2vulkan::device::VkFeature;
    use metal2vulkan::error::Feature;
    use metal2vulkan_devtools::engine_dir;

    /// Where Vulkan offers `fragmentShaderPixelInterlock`, a fragment function's raster order
    /// group orders its read-modify-write of its pixel's float, as Metal's does; elsewhere the
    /// state is refused naming raster order groups.
    #[test]
    fn every_device_orders_raster_order_groups_where_it_offers_the_pixel_interlock() {
        let (Some(examples), Some(groups)) =
            (examples(), library(&engine_dir().join("fragment_rog.ll")))
        else {
            return;
        };
        let mut d = descriptor(&examples, "colored_fragment", PixelFormat::RGBA32Float);
        d.fragment = Some(StageSource::Function(FunctionDescriptor::new(
            groups.function("rog_buffer").expect("the function"),
        )));
        each_device("pixel_interlock", |name, _, device| {
            // The translator's profile has the feature only where device creation enabled its
            // extension.
            let listed = (device.extensions().iter())
                .any(|enabled| enabled.as_c_str() == ash::ext::fragment_shader_interlock::NAME);
            let core = device.core();
            let enabled = &core.translator.profile().enabled().features;
            let on = enabled.contains(&VkFeature::FragmentShaderPixelInterlock);
            assert!(
                listed || !on,
                "{name}: fragmentShaderPixelInterlock without its extension"
            );
            let state = match device
                .new_render_pipeline_state(&d, crate::tests::no_constexpr_samplers)
            {
                Ok(state) => state,
                Err(Error::Translation {
                    error: metal2vulkan::error::Error::Unsupported(u),
                    ..
                }) if !on => {
                    assert_eq!(u.feature, Feature::RasterOrderGroups, "{name}: {u:?}");
                    println!("pixel_interlock: {name}: no pixel interlock; the state is refused");
                    return;
                }
                Err(error) => panic!("{name}: {error}"),
            };
            assert!(on, "{name}: a state made without the pixel interlock");
            const LAYERS: usize = 4;
            let mut buffers = Buffers::new(&device);
            let whole: [f32; 6] = [-1.0, -1.0, 3.0, -1.0, -1.0, 3.0];
            let positions = buffers.of(&device, &floats(&whole.repeat(LAYERS)));
            let colors = buffers.of(&device, &floats(&[0.0; 12 * LAYERS]));
            let pixels = (SIDE * SIDE) as usize;
            let sums = buffers.of(&device, &floats(&vec![0.0_f32; pixels]));
            let width = buffers.of(&device, &SIDE.to_le_bytes());
            let color = target(
                &device,
                PixelFormat::RGBA32Float,
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
            ok(encoder.set_fragment_buffer(Some(&sums), 0, 0));
            ok(encoder.set_fragment_buffer(Some(&width), 0, 1));
            ok(encoder.draw_primitives(PrimitiveType::Triangle, 0, 3 * LAYERS as u32, 1, 0));
            encoder.end_encoding();
            run(name, &command_buffer);
            let layers = LAYERS as f32;
            let contents = sums.contents().expect("a Shared buffer");
            // SAFETY: the buffer holds `pixels` floats, is alive, and the GPU work is done.
            let sums = unsafe { std::slice::from_raw_parts(contents.as_ptr(), pixels * 4) };
            let floats_of = |bytes: &[u8]| -> Vec<f32> {
                (bytes.as_chunks::<4>().0.iter())
                    .map(|w| f32::from_le_bytes(*w))
                    .collect()
            };
            assert_eq!(floats_of(sums), vec![layers; pixels], "{name}: the sums");
            let texels = floats_of(&read(&color, 16));
            assert_eq!(
                texels,
                [layers, 0.0, 0.0, 1.0].repeat(pixels),
                "{name}: the texels"
            );
            println!(
                "pixel_interlock: {name}: fragmentShaderPixelInterlock enabled; {pixels} pixels \
                 summed {LAYERS} fragments in order"
            );
        });
    }
}
