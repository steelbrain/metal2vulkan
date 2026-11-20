//! Multisample resolves in the pass.
//!
//! - A store action that resolves needs a resolve texture (`Refusal::ResolveTexture`), and
//!   `Unknown` resolves only where the attachment names one; no other action resolves.
//! - Under a device cfg, on every device of the cfg's kind: a 4-sample `RGBA8Unorm` attachment
//!   resolves into level 1 and slice 1 of a 2D array, each pixel the average of its samples (a
//!   pixel half covered within one code of the half's, the device's rounding, which the pass's
//!   static note `DeviceRoundedResolve` names), and every other level and slice of the resolve
//!   texture reads zero; a pass that resolves no color attachment carries no note. A resolving
//!   action with no resolve texture is refused, and so is an integer format's resolve; a resolve
//!   into another format or out of one sample is a gap; an `Unknown` store with a resolve texture
//!   resolves, and its encoder may then set only an action that resolves (`LateStore`). A
//!   4-sample `Depth32Float` attachment resolves by `Sample0` on every device, and by `Min` and
//!   `Max` where the device's depth resolve modes have them (else `ResolveFilter`): a pixel half
//!   covered keeps sample 0's depth, the least or the greatest. A stencil resolve by
//!   `DepthResolvedSample` is `ResolveFilter`.

use crate::error::{Error, Refusal};
use crate::pass::resolving;
use crate::StoreAction;

/// Which store actions resolve, and which need a resolve texture.
#[test]
fn a_resolving_action_needs_a_resolve_texture_and_unknown_resolves_where_one_is_named() {
    for store in [
        StoreAction::MultisampleResolve,
        StoreAction::StoreAndMultisampleResolve,
    ] {
        assert!(
            matches!(
                resolving(store, false),
                Err(Error::Refused(Refusal::ResolveTexture))
            ),
            "{store:?} without a resolve texture"
        );
        assert!(
            matches!(resolving(store, true), Ok(true)),
            "{store:?} with one"
        );
    }
    assert!(matches!(resolving(StoreAction::Unknown, true), Ok(true)));
    assert!(matches!(resolving(StoreAction::Unknown, false), Ok(false)));
    for store in [
        StoreAction::Store,
        StoreAction::DontCare,
        StoreAction::CustomSampleDepthStore,
    ] {
        for named in [false, true] {
            assert!(
                matches!(resolving(store, named), Ok(false)),
                "{store:?}, named {named}"
            );
        }
    }
}

#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod on_device {
    use crate::tests::command::on_device::each_device;
    use crate::tests::draw::on_device::{
        descriptor, examples, floats, pass, run, state, target, Buffers, SIDE,
    };
    use crate::texture::TextureObject;
    use crate::{
        CullMode, DepthStencilDescriptor, Device, Error, Gap, MultisampleDepthResolveFilter,
        MultisampleStencilResolveFilter, PixelFormat, PrecisionNote, PrimitiveType, Refusal,
        Region, RenderPassKind, StorageMode, StoreAction, Texture, TextureDescriptor, TextureType,
        TextureUsage,
    };

    const WHITE: [f32; 4] = [1.0; 4];

    /// A `Type2DMultisample` texture of 4 samples, `SIDE` a side.
    fn four(name: &str, device: &Device, format: PixelFormat) -> Result<Texture, Error> {
        let mut td = TextureDescriptor::new(TextureType::Type2DMultisample, format, SIDE, SIDE);
        td.sample_count = 4;
        td.usage = TextureUsage::RENDER_TARGET;
        td.storage_mode = StorageMode::Private;
        let made = device.new_texture(&td, None);
        if let Err(error) = &made {
            println!("{name}: no 4-sample {format:?}: {error}");
        }
        made
    }

    /// The bytes of `level` and `slice` of `texture`, `side` texels a side, `unit` bytes each.
    fn read_at(texture: &Texture, side: u32, unit: usize, level: u32, slice: u32) -> Vec<u8> {
        let row = side as usize * unit;
        let mut bytes = vec![0xee; row * side as usize];
        let len = bytes.len() as u64;
        let region = Region {
            origin: [0; 3],
            size: [side, side, 1],
        };
        (texture.get_bytes(&mut bytes, row as u64, len, region, level, slice)).expect("get_bytes");
        bytes
    }

    /// A white triangle over the lower-left half of the pass's 4-sample target: pixel `(x, y)`
    /// is wholly covered where `x < y`, not at all where `x > y`, and on the diagonal in samples
    /// 2 and 3 alone (`every_device_draws_four_samples_where_metal_places_them`).
    fn coverage(x: u32, y: u32) -> std::cmp::Ordering {
        x.cmp(&y)
    }

    /// Records the white triangle through `state` in a pass of `d`, and runs it.
    fn drawn(
        name: &str,
        device: &Device,
        d: &crate::RenderPassDescriptor,
        state: &crate::RenderPipelineState,
        depth: Option<&crate::DepthStencilState>,
    ) {
        let mut buffers = Buffers::new(device);
        let positions = buffers.of(device, &floats(&[-1.0, -1.0, 1.0, -1.0, -1.0, 1.0]));
        let colors = buffers.of(device, &floats(&WHITE.repeat(3)));
        let queue = device.new_command_queue(4).expect("a queue");
        let command_buffer = queue.command_buffer().expect("a command buffer");
        let mut encoder = (command_buffer.render_command_encoder(d))
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        let draw = |result: Result<(), Error>| result.unwrap_or_else(|e| panic!("{name}: {e}"));
        draw(encoder.set_render_pipeline_state(state));
        if let Some(depth) = depth {
            draw(encoder.set_depth_stencil_state(Some(depth)));
        }
        encoder.set_cull_mode(CullMode::None);
        draw(encoder.set_vertex_buffer(Some(&positions), 0, 0));
        draw(encoder.set_vertex_buffer(Some(&colors), 0, 1));
        draw(encoder.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0));
        encoder.end_encoding();
        run(name, &command_buffer);
    }

    /// The pass's encoder's error, where making it fails.
    fn refused(device: &Device, d: &crate::RenderPassDescriptor) -> Option<Error> {
        let queue = device.new_command_queue(1).expect("a queue");
        let command_buffer = queue.command_buffer().expect("a command buffer");
        command_buffer.render_command_encoder(d).err()
    }

    fn gap(kind: RenderPassKind) -> Error {
        Error::Unsupported(Gap::RenderPass { kind })
    }

    /// The module's color resolve and its refusals.
    #[test]
    fn every_device_resolves_four_samples_into_a_level_and_slice() {
        let Some(library) = examples() else {
            return;
        };
        let mut d = descriptor(&library, "colored_fragment", PixelFormat::RGBA8Unorm);
        d.raster_sample_count = 4;
        each_device("resolve_color", |name, _, device| {
            let four_state = state(name, &device, &d);
            let samples = four(name, &device, PixelFormat::RGBA8Unorm)
                .unwrap_or_else(|error| panic!("{name}: Vulkan requires 4 samples: {error}"));
            let mut rd = TextureDescriptor::new(
                TextureType::Type2DArray,
                PixelFormat::RGBA8Unorm,
                2 * SIDE,
                2 * SIDE,
            );
            rd.mipmap_level_count = 2;
            rd.array_length = 2;
            rd.usage = TextureUsage::RENDER_TARGET;
            rd.storage_mode = StorageMode::Shared;
            let into =
                (device.new_texture(&rd, None)).unwrap_or_else(|error| panic!("{name}: {error}"));
            let mut p = pass(&samples, None);
            assert_eq!(
                p.precision_note(),
                None,
                "{name}: a pass that resolves nothing"
            );
            let c = &mut p.color_attachments[0];
            c.store_action = StoreAction::MultisampleResolve;
            c.resolve_texture = Some(into.clone());
            c.resolve_level = 1;
            c.resolve_slice = 1;
            assert_eq!(
                p.precision_note(),
                Some(PrecisionNote::DeviceRoundedResolve),
                "{name}: the resolve's note"
            );
            let before = device.counters().metrics().resolves;
            drawn(name, &device, &p, &four_state, None);
            let resolves = device.counters().metrics().resolves - before;
            assert_eq!(resolves, 1, "{name}: the pass's one resolve");
            let got = read_at(&into, SIDE, 4, 1, 1);
            for y in 0..SIDE {
                for x in 0..SIDE {
                    let at = ((y * SIDE + x) * 4) as usize;
                    let texel = &got[at..at + 4];
                    match coverage(x, y) {
                        std::cmp::Ordering::Less => {
                            assert_eq!(texel, [255; 4], "{name}: ({x}, {y}) covered")
                        }
                        std::cmp::Ordering::Greater => {
                            assert_eq!(texel, [0, 0, 0, 255], "{name}: ({x}, {y}) uncovered")
                        }
                        std::cmp::Ordering::Equal => {
                            for channel in &texel[..3] {
                                assert!(
                                    (127..=128).contains(channel),
                                    "{name}: ({x}, {y}) half covered: {texel:?}"
                                );
                            }
                            assert_eq!(texel[3], 255, "{name}: ({x}, {y})'s alpha");
                        }
                    }
                }
            }
            assert!(
                read_at(&into, SIDE, 4, 1, 0).iter().all(|&b| b == 0),
                "{name}: level 1 of slice 0 untouched"
            );
            for slice in 0..2 {
                assert!(
                    read_at(&into, 2 * SIDE, 4, 0, slice)
                        .iter()
                        .all(|&b| b == 0),
                    "{name}: level 0 of slice {slice} untouched"
                );
            }

            // An action that resolves, with no resolve texture.
            let mut none = pass(&samples, None);
            none.color_attachments[0].store_action = StoreAction::StoreAndMultisampleResolve;
            assert!(
                matches!(
                    refused(&device, &none),
                    Some(Error::Refused(Refusal::ResolveTexture))
                ),
                "{name}: no resolve texture"
            );
            // Into another pixel format.
            let other = target(
                &device,
                PixelFormat::BGRA8Unorm,
                TextureUsage::RENDER_TARGET,
            );
            let mut p = pass(&samples, None);
            p.color_attachments[0].store_action = StoreAction::MultisampleResolve;
            p.color_attachments[0].resolve_texture = Some(other.clone());
            assert_eq!(
                refused(&device, &p).map(|e| e.to_string()),
                Some(gap(RenderPassKind::Resolve).to_string()),
                "{name}: another format"
            );
            // Out of one sample.
            let single = target(
                &device,
                PixelFormat::RGBA8Unorm,
                TextureUsage::RENDER_TARGET,
            );
            let flat = target(
                &device,
                PixelFormat::RGBA8Unorm,
                TextureUsage::RENDER_TARGET,
            );
            let mut p = pass(&single, None);
            p.color_attachments[0].store_action = StoreAction::MultisampleResolve;
            p.color_attachments[0].resolve_texture = Some(flat.clone());
            assert_eq!(
                refused(&device, &p).map(|e| e.to_string()),
                Some(gap(RenderPassKind::Resolve).to_string()),
                "{name}: one sample"
            );
            // An integer format's resolve, where the device makes a 4-sample one.
            if let Ok(ints) = four(name, &device, PixelFormat::RGBA8Uint) {
                let into = target(&device, PixelFormat::RGBA8Uint, TextureUsage::RENDER_TARGET);
                let mut p = pass(&ints, None);
                p.color_attachments[0].store_action = StoreAction::MultisampleResolve;
                p.color_attachments[0].resolve_texture = Some(into.clone());
                assert!(
                    matches!(
                        refused(&device, &p),
                        Some(Error::Refused(Refusal::ResolveFormat {
                            pixel_format: PixelFormat::RGBA8Uint
                        }))
                    ),
                    "{name}: an integer resolve"
                );
            }

            // `Unknown` with a resolve texture resolves; a later action must resolve too.
            let late = target(
                &device,
                PixelFormat::RGBA8Unorm,
                TextureUsage::RENDER_TARGET,
            );
            let mut p = pass(&samples, None);
            p.color_attachments[0].store_action = StoreAction::Unknown;
            p.color_attachments[0].resolve_texture = Some(late.clone());
            let queue = device.new_command_queue(1).expect("a queue");
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let mut encoder = (command_buffer.render_command_encoder(&p))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            assert_eq!(
                (encoder.set_color_store_action(StoreAction::Store, 0)).map_err(|e| e.to_string()),
                Err(gap(RenderPassKind::LateStore).to_string()),
                "{name}: a late store that does not resolve"
            );
            (encoder.set_color_store_action(StoreAction::MultisampleResolve, 0))
                .unwrap_or_else(|error| panic!("{name}: a late resolve: {error}"));
            encoder.end_encoding();
            run(name, &command_buffer);
            let cleared = read_at(&late, SIDE, 4, 0, 0);
            assert!(
                cleared.chunks(4).all(|t| t == [0, 0, 0, 255]),
                "{name}: the late resolve's cleared black"
            );
            drop((samples, into, other, single, flat, late));
            println!("{name}: four samples resolved into level 1 of slice 1");
        });
    }

    /// The module's depth resolves, and the stencil filter no Vulkan mode is.
    #[test]
    fn every_device_resolves_four_depth_samples_by_its_filters() {
        let Some(library) = examples() else {
            return;
        };
        let mut d = descriptor(&library, "colored_fragment", PixelFormat::RGBA8Unorm);
        d.raster_sample_count = 4;
        d.depth_format = Some(PixelFormat::Depth32Float);
        each_device("resolve_depth", |name, _, device| {
            let four_state = state(name, &device, &d);
            let mut writes = DepthStencilDescriptor::default();
            writes.depth_write_enabled = true;
            let writes = device.new_depth_stencil_state(&writes);
            let color = four(name, &device, PixelFormat::RGBA8Unorm)
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let depth = four(name, &device, PixelFormat::Depth32Float)
                .unwrap_or_else(|error| panic!("{name}: Vulkan requires 4 samples: {error}"));
            let filters = [
                MultisampleDepthResolveFilter::Sample0,
                MultisampleDepthResolveFilter::Min,
                MultisampleDepthResolveFilter::Max,
            ];
            for filter in filters {
                let into = target(
                    &device,
                    PixelFormat::Depth32Float,
                    TextureUsage::RENDER_TARGET,
                );
                let mut p = pass(&color, None);
                let z = &mut p.depth_attachment;
                z.texture = Some(depth.clone());
                z.clear_depth = 1.0;
                z.store_action = StoreAction::MultisampleResolve;
                z.resolve_texture = Some(into.clone());
                z.depth_resolve_filter = filter;
                if filter != MultisampleDepthResolveFilter::Sample0 {
                    if let Some(error) = refused(&device, &p) {
                        assert_eq!(
                            error.to_string(),
                            gap(RenderPassKind::ResolveFilter).to_string(),
                            "{name}: {filter:?}"
                        );
                        println!("{name}: no {filter:?} depth resolve on this device");
                        continue;
                    }
                }
                drawn(name, &device, &p, &four_state, Some(&writes));
                if into.unit().is_none() {
                    println!("{name}: {filter:?}: the depth's bytes are not copied on this device");
                    continue;
                }
                let got = read_at(&into, SIDE, 4, 0, 0);
                for y in 0..SIDE {
                    for x in 0..SIDE {
                        let at = ((y * SIDE + x) * 4) as usize;
                        let depth = f32::from_le_bytes(got[at..at + 4].try_into().unwrap());
                        let want = match (coverage(x, y), filter) {
                            (std::cmp::Ordering::Less, _) => 0.0,
                            (std::cmp::Ordering::Greater, _) => 1.0,
                            // Samples 2 and 3 drawn at depth 0, samples 0 and 1 cleared to 1.
                            (_, MultisampleDepthResolveFilter::Min) => 0.0,
                            (_, _) => 1.0,
                        };
                        assert_eq!(depth, want, "{name}: {filter:?}: ({x}, {y})");
                    }
                }
                println!("{name}: depth resolved by {filter:?}");
            }

            // A stencil resolve by the sample the depth resolve keeps.
            if let Ok(stencil) = four(name, &device, PixelFormat::Stencil8) {
                let into = target(&device, PixelFormat::Stencil8, TextureUsage::RENDER_TARGET);
                let mut p = pass(&color, Some(&stencil));
                let s = &mut p.stencil_attachment;
                s.store_action = StoreAction::MultisampleResolve;
                s.resolve_texture = Some(into.clone());
                s.stencil_resolve_filter = MultisampleStencilResolveFilter::DepthResolvedSample;
                assert_eq!(
                    refused(&device, &p).map(|e| e.to_string()),
                    Some(gap(RenderPassKind::ResolveFilter).to_string()),
                    "{name}: DepthResolvedSample"
                );
            }
        });
    }
}
