//! A texture made without an index, bound to a function's texture slot.
//!
//! Under a device cfg, on every device of the cfg's kind: a texture reached through its views
//! alone (`Device::new_texture_for_views`) bound to a compute encoder's texture slot, or to a
//! render encoder's vertex or fragment slot, is `Refusal::TextureWithoutIndex` with the slot's
//! index, and so is a depth target no shader reaches, which is made without an index too.
//! Unbinding the slot (`None`), a view of the texture at an index of its own, and a texture made
//! at an index each bind, and the command buffer completes with no error.

#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod on_device {
    use crate::error::{Error, Refusal};
    use crate::tests::command::on_device::each_device;
    use crate::tests::draw::on_device::{pass, run, target};
    use crate::{
        DispatchType, IndexClass, PixelFormat, StorageMode, TextureDescriptor, TextureType,
        TextureUsage,
    };

    /// What a bind answered: `None` where it bound, the refusal where it refused.
    fn answer(result: Result<(), Error>) -> Option<Refusal> {
        match result {
            Ok(()) => None,
            Err(Error::Refused(refusal)) => Some(refusal),
            Err(error) => panic!("not a refusal: {error}"),
        }
    }

    #[test]
    fn a_texture_made_without_an_index_binds_to_no_texture_slot() {
        each_device("bind_without_index", |name, _, device| {
            (device.grant_indices(IndexClass::Texture, 4096, 2))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let mut d = TextureDescriptor::new(TextureType::Type2D, PixelFormat::RGBA8Unorm, 4, 4);
            d.usage = TextureUsage::SHADER_READ;
            let unindexed =
                (device.new_texture_for_views(&d)).unwrap_or_else(|e| panic!("{name}: {e}"));
            let view = (unindexed.new_texture_view(
                PixelFormat::RGBA8Unorm,
                TextureType::Type2D,
                0..1,
                0..1,
                Some(4096),
            ))
            .unwrap_or_else(|error| panic!("{name}: {error}"));
            let indexed =
                (device.new_texture(&d, Some(4097))).unwrap_or_else(|e| panic!("{name}: {e}"));
            let mut dd =
                TextureDescriptor::new(TextureType::Type2D, PixelFormat::Depth32Float, 4, 4);
            dd.usage = TextureUsage::RENDER_TARGET;
            dd.storage_mode = StorageMode::Private;
            let depth = (device.new_texture(&dd, None)).unwrap_or_else(|e| panic!("{name}: {e}"));
            let without = |index| Some(Refusal::TextureWithoutIndex { index });
            let queue = device.new_command_queue(1).expect("a queue");
            let command_buffer = queue.command_buffer().expect("a command buffer");

            let mut compute = (command_buffer.compute_command_encoder(DispatchType::Serial))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let compute_answers = [
                answer(compute.set_texture(Some(&unindexed), 3)),
                answer(compute.set_texture(Some(&depth), 4)),
                answer(compute.set_texture(None, 3)),
                answer(compute.set_texture(Some(&view), 3)),
                answer(compute.set_texture(Some(&indexed), 4)),
            ];
            assert_eq!(
                compute_answers,
                [without(3), without(4), None, None, None],
                "{name}: compute"
            );
            compute.end_encoding();

            let color = target(
                &device,
                PixelFormat::RGBA8Unorm,
                TextureUsage::RENDER_TARGET,
            );
            let mut render = (command_buffer.render_command_encoder(&pass(&color, None)))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let render_answers = [
                answer(render.set_vertex_texture(Some(&unindexed), 5)),
                answer(render.set_fragment_texture(Some(&unindexed), 6)),
                answer(render.set_fragment_texture(Some(&depth), 7)),
                answer(render.set_vertex_texture(None, 5)),
                answer(render.set_fragment_texture(None, 6)),
                answer(render.set_vertex_texture(Some(&view), 5)),
                answer(render.set_fragment_texture(Some(&view), 6)),
                answer(render.set_fragment_texture(Some(&indexed), 7)),
            ];
            assert_eq!(
                render_answers,
                [
                    without(5),
                    without(6),
                    without(7),
                    None,
                    None,
                    None,
                    None,
                    None
                ],
                "{name}: render"
            );
            render.end_encoding();
            run(name, &command_buffer);
            println!("{name}: each bind of a texture made without an index refused with its slot");
            drop((
                command_buffer,
                queue,
                view,
                unindexed,
                indexed,
                depth,
                color,
            ));
        });
    }
}
