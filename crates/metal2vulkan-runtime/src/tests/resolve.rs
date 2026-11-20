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
//!   into another format or out of one sample is a gap; an `Unknown` store with a resolve
//!   texture resolves, and its encoder may then set only an action that resolves (`LateStore`). A
//!   4-sample `Depth32Float` attachment resolves by `Sample0` on every device, and by `Min` and
//!   `Max` where the device's depth resolve modes have them (else `ResolveFilter`): a pixel half
//!   covered keeps sample 0's depth, the least or the greatest. A stencil resolve by
//!   `DepthResolvedSample` is `ResolveFilter`. A pass that resolves nothing walks no resolve as
//!   its encoder begins and ends; one that resolves walks its resolves.
//! - Under a device cfg, on every device of the cfg's kind, as Metal resolves into a texture of
//!   any usage (probe `resolve_texture_usage.m`): a 4-sample `BGRA8Unorm_sRGB` attachment
//!   cleared to (0.25, 0.5, 0.75, 1) resolves into a one-sample Shared texture of `ShaderRead`
//!   usage, and of `ShaderRead | ShaderWrite`, whose image took the attachment usage, and reads
//!   the clear colour within one code; so does a 900 x 900 pass of every load `Load`, its colour
//!   beside a 4-sample `Depth32Float_Stencil8`, of what an earlier pass cleared. A depth resolve
//!   into a `Depth32Float` texture of `ShaderRead` usage is the gap `ResolveUsage`, and so is a
//!   resolve into a `ShaderRead` texture over a Shared buffer whose linear image lies in place,
//!   which stays in place, with no attachment usage.
//! - Under a device cfg, on every device of the cfg's kind, whether it has
//!   `shaderStorageImageMultisample` or not: a 4-sample `RGBA8Unorm` texture made for shader
//!   reads, shader writes and rendering, as Metal makes one, has no storage usage in its image and
//!   no storage view (Metal's shaders only read a multisampled texture), lacks no storage element,
//!   and draws and resolves as the target above.

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
        ClearColor, CullMode, DepthStencilDescriptor, Device, Error, Gap, IndexClass, LoadAction,
        MemoryClass, MultisampleDepthResolveFilter, MultisampleStencilResolveFilter, PixelFormat,
        PrecisionNote, PrimitiveType, Refusal, Region, RenderPassKind, StorageMode, StoreAction,
        Texture, TextureDescriptor, TextureType, TextureUsage,
    };
    use ash::vk;

    const WHITE: [f32; 4] = [1.0; 4];

    /// A `Type2DMultisample` texture of 4 samples, `SIDE` a side.
    fn four(name: &str, device: &Device, format: PixelFormat) -> Result<Texture, Error> {
        four_of(name, device, format, SIDE)
    }

    /// A `Type2DMultisample` render target of 4 samples, `side` a side.
    fn four_of(
        name: &str,
        device: &Device,
        format: PixelFormat,
        side: u32,
    ) -> Result<Texture, Error> {
        let mut td = TextureDescriptor::new(TextureType::Type2DMultisample, format, side, side);
        td.sample_count = 4;
        td.usage = TextureUsage::RENDER_TARGET;
        td.storage_mode = StorageMode::Private;
        let made = device.new_texture_for_views(&td);
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

    /// `got`, the `RGBA8Unorm` resolve of the white triangle over black, `SIDE` texels a side:
    /// white where the triangle covers a pixel, black where it does not, and on the diagonal
    /// within one code of half white, opaque.
    fn resolved(name: &str, got: &[u8]) {
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

    /// The walks of passes' resolves on this thread so far.
    fn walks() -> u64 {
        crate::pass::RESOLVE_WALKS.with(std::cell::Cell::get)
    }

    /// The module's walks: none for a pass that resolves nothing, as its encoder begins and ends.
    #[test]
    fn every_device_walks_no_resolve_for_a_pass_that_resolves_nothing() {
        each_device("resolve_walks", |name, _, device| {
            let queue = device.new_command_queue(1).expect("a queue");
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let encode = |d: &crate::RenderPassDescriptor| {
                let before = walks();
                let encoder = (command_buffer.render_command_encoder(d))
                    .unwrap_or_else(|error| panic!("{name}: {error}"));
                encoder.end_encoding();
                walks() - before
            };
            let single = target(
                &device,
                PixelFormat::RGBA8Unorm,
                TextureUsage::RENDER_TARGET,
            );
            assert_eq!(
                encode(&pass(&single, None)),
                0,
                "{name}: a pass that resolves nothing"
            );
            let samples = four(name, &device, PixelFormat::RGBA8Unorm)
                .unwrap_or_else(|error| panic!("{name}: Vulkan requires 4 samples: {error}"));
            let mut p = pass(&samples, None);
            assert_eq!(
                encode(&p),
                0,
                "{name}: a 4-sample pass that resolves nothing"
            );
            p.color_attachments[0].store_action = StoreAction::MultisampleResolve;
            p.color_attachments[0].resolve_texture = Some(single.clone());
            assert!(
                encode(&p) > 0,
                "{name}: a pass that resolves walks its resolve"
            );
            run(name, &command_buffer);
        });
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
            let into = (device.new_texture_for_views(&rd))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
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
            resolved(name, &read_at(&into, SIDE, 4, 1, 1));
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

    /// The module's multisampled texture made for shader writes.
    #[test]
    fn every_device_makes_a_written_multisample_target_with_no_storage_usage() {
        let Some(library) = examples() else {
            return;
        };
        let mut d = descriptor(&library, "colored_fragment", PixelFormat::RGBA8Unorm);
        d.raster_sample_count = 4;
        each_device("resolve_written", |name, _, device| {
            let four_state = state(name, &device, &d);
            let mut td = TextureDescriptor::new(
                TextureType::Type2DMultisample,
                PixelFormat::RGBA8Unorm,
                SIDE,
                SIDE,
            );
            td.sample_count = 4;
            td.usage = TextureUsage::SHADER_READ
                | TextureUsage::SHADER_WRITE
                | TextureUsage::RENDER_TARGET;
            td.storage_mode = StorageMode::Private;
            let samples = (device.new_texture_for_views(&td))
                .unwrap_or_else(|error| panic!("{name}: Vulkan samples 4 RGBA8 samples: {error}"));
            let usage = samples.object().image_usage();
            assert!(
                !usage.contains(vk::ImageUsageFlags::STORAGE),
                "{name}: the image's usage {:#x}",
                usage.as_raw()
            );
            assert!(
                !samples.object().has_storage_view(),
                "{name}: a storage view"
            );
            assert_eq!(
                samples.unstored(),
                None,
                "{name}: a missing storage element"
            );
            let into = target(
                &device,
                PixelFormat::RGBA8Unorm,
                TextureUsage::RENDER_TARGET,
            );
            let mut p = pass(&samples, None);
            p.color_attachments[0].store_action = StoreAction::MultisampleResolve;
            p.color_attachments[0].resolve_texture = Some(into.clone());
            drawn(name, &device, &p, &four_state, None);
            resolved(name, &read_at(&into, SIDE, 4, 0, 0));
            drop((samples, into));
            println!("{name}: a written 4-sample target, no storage usage, resolved");
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

    /// The clear colour of probe `resolve_texture_usage.m`.
    const CLEAR: ClearColor = ClearColor {
        red: 0.25,
        green: 0.5,
        blue: 0.75,
        alpha: 1.0,
    };

    /// sRGB's encoding of the linear channel `linear`, as a byte.
    fn srgb(linear: f64) -> u8 {
        let encoded = match linear <= 0.003_130_8 {
            true => 12.92 * linear,
            false => 1.055 * linear.powf(1.0 / 2.4) - 0.055,
        };
        (encoded * 255.0).round() as u8
    }

    /// `got`, `BGRA8Unorm_sRGB` texels, each within one code of [`CLEAR`]'s, the device's
    /// rounding (`PrecisionNote::DeviceRoundedResolve`).
    fn cleared(name: &str, got: &[u8]) {
        let want = [srgb(CLEAR.blue), srgb(CLEAR.green), srgb(CLEAR.red), 255];
        for (at, texel) in got.chunks(4).enumerate() {
            assert!(
                texel.iter().zip(want).all(|(&g, w)| g.abs_diff(w) <= 1),
                "{name}: texel {at}: {texel:?}, not within a code of {want:?}"
            );
        }
    }

    /// A one-sample 2D texture of `format`, `usage` and `mode`, `side` a side.
    fn one(
        name: &str,
        device: &Device,
        format: PixelFormat,
        usage: TextureUsage,
        mode: StorageMode,
        side: u32,
    ) -> Texture {
        let mut td = TextureDescriptor::new(TextureType::Type2D, format, side, side);
        td.usage = usage;
        td.storage_mode = mode;
        (device.new_texture_for_views(&td)).unwrap_or_else(|error| panic!("{name}: {error}"))
    }

    /// Encodes each pass of `passes` in one command buffer, with no draw, and runs it.
    fn passes(name: &str, device: &Device, passes: &[&crate::RenderPassDescriptor]) {
        let queue = device.new_command_queue(1).expect("a queue");
        let command_buffer = queue.command_buffer().expect("a command buffer");
        for p in passes {
            let encoder = (command_buffer.render_command_encoder(p))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            encoder.end_encoding();
        }
        run(name, &command_buffer);
    }

    /// Probe `resolve_texture_usage.m`'s case, which Metal resolves: a 4-sample
    /// `BGRA8Unorm_sRGB` attachment cleared to (0.25, 0.5, 0.75, 1) resolves into a one-sample
    /// Shared texture of `ShaderRead` usage, and of `ShaderRead | ShaderWrite`, whose image took
    /// the attachment usage. And the resolves the runtime does not build into a texture of no
    /// `RenderTarget` usage: a depth one, and one into a texture over a Shared buffer in place,
    /// whose linear image takes no attachment usage.
    #[test]
    fn every_device_resolves_into_a_texture_of_shader_read_usage() {
        each_device("resolve_usage", |name, _, device| {
            let samples = four(name, &device, PixelFormat::BGRA8UnormSRGB)
                .unwrap_or_else(|error| panic!("{name}: Vulkan requires 4 samples: {error}"));
            for usage in [
                TextureUsage::SHADER_READ,
                TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE,
            ] {
                let into = one(
                    name,
                    &device,
                    PixelFormat::BGRA8UnormSRGB,
                    usage,
                    StorageMode::Shared,
                    SIDE,
                );
                let image = into.object().image_usage();
                assert!(
                    image.contains(vk::ImageUsageFlags::COLOR_ATTACHMENT),
                    "{name}: {usage:?}: the image's usage {:#x}",
                    image.as_raw()
                );
                let mut p = pass(&samples, None);
                let c = &mut p.color_attachments[0];
                c.clear_color = CLEAR;
                c.store_action = StoreAction::MultisampleResolve;
                c.resolve_texture = Some(into.clone());
                let before = device.counters().metrics().resolves;
                passes(name, &device, &[&p]);
                let resolves = device.counters().metrics().resolves - before;
                assert_eq!(resolves, 1, "{name}: {usage:?}: the pass's one resolve");
                cleared(name, &read_at(&into, SIDE, 4, 0, 0));
                println!("{name}: resolved into a texture of {usage:?}");
            }

            // A depth resolve into a texture of no `RenderTarget` usage.
            let color = four(name, &device, PixelFormat::RGBA8Unorm)
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let depth = four(name, &device, PixelFormat::Depth32Float)
                .unwrap_or_else(|error| panic!("{name}: Vulkan requires 4 samples: {error}"));
            let into = one(
                name,
                &device,
                PixelFormat::Depth32Float,
                TextureUsage::SHADER_READ,
                StorageMode::Private,
                SIDE,
            );
            let mut p = pass(&color, None);
            let z = &mut p.depth_attachment;
            z.texture = Some(depth.clone());
            z.store_action = StoreAction::MultisampleResolve;
            z.resolve_texture = Some(into);
            assert_eq!(
                refused(&device, &p).map(|e| e.to_string()),
                Some(gap(RenderPassKind::ResolveUsage).to_string()),
                "{name}: a depth resolve into a ShaderRead texture"
            );

            // A resolve into a texture over a Shared buffer.
            let format = PixelFormat::RGBA8Unorm;
            let align = (device.minimum_linear_texture_alignment(format))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let pitch = u64::from(SIDE * 4).next_multiple_of(align);
            let len = u64::from(SIDE) * pitch;
            let block = (device.take_block(MemoryClass::Shared, len))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let buffer = (device.new_buffer(block.id, 0, len, None))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let mut td = TextureDescriptor::new(TextureType::Type2D, format, SIDE, SIDE);
            td.usage = TextureUsage::SHADER_READ;
            let index = Some(IndexClass::Texture.range().start);
            let over = (buffer.new_texture(&td, index, 0, pitch))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            if over.copied_over().is_some() {
                println!("{name}: the texture over the buffer is copied, not in place");
            } else {
                let image = over.object().image_usage();
                assert!(
                    !image.contains(vk::ImageUsageFlags::COLOR_ATTACHMENT),
                    "{name}: in place over the buffer, the image's usage {:#x}",
                    image.as_raw()
                );
                let mut p = pass(&color, None);
                p.color_attachments[0].store_action = StoreAction::MultisampleResolve;
                p.color_attachments[0].resolve_texture = Some(over.clone());
                assert_eq!(
                    refused(&device, &p).map(|e| e.to_string()),
                    Some(gap(RenderPassKind::ResolveUsage).to_string()),
                    "{name}: a resolve into a ShaderRead texture in place over a buffer"
                );
                println!("{name}: in place over a buffer, a resolve into it is ResolveUsage");
            }
            drop((samples, color, depth, over, buffer, block));
        });
    }

    /// The pass of a guest's walk, which Metal resolves: 900 x 900, a 4-sample
    /// `BGRA8Unorm_sRGB` colour beside a 4-sample `Depth32Float_Stencil8` depth and stencil,
    /// every load `Load`, the colour resolved into a one-sample `BGRA8Unorm_sRGB` Shared texture
    /// of `ShaderRead` usage, which reads what an earlier pass cleared the colour to.
    #[test]
    fn every_device_resolves_a_loaded_pass_beside_depth_and_stencil_into_a_shader_read_texture() {
        const WALK: u32 = 900;
        each_device("resolve_usage_walk", |name, _, device| {
            let color = four_of(name, &device, PixelFormat::BGRA8UnormSRGB, WALK)
                .unwrap_or_else(|error| panic!("{name}: Vulkan requires 4 samples: {error}"));
            let Ok(depth) = four_of(name, &device, PixelFormat::Depth32FloatStencil8, WALK) else {
                println!("{name}: no 4-sample Depth32Float_Stencil8");
                return;
            };
            let into = one(
                name,
                &device,
                PixelFormat::BGRA8UnormSRGB,
                TextureUsage::SHADER_READ,
                StorageMode::Shared,
                WALK,
            );
            let attached = |load: LoadAction| {
                let mut p = pass(&color, None);
                p.color_attachments[0].load_action = load;
                p.depth_attachment.texture = Some(depth.clone());
                p.depth_attachment.load_action = load;
                p.depth_attachment.store_action = StoreAction::Store;
                p.stencil_attachment.texture = Some(depth.clone());
                p.stencil_attachment.load_action = load;
                p.stencil_attachment.store_action = StoreAction::Store;
                p
            };
            let mut first = attached(LoadAction::Clear);
            first.color_attachments[0].clear_color = CLEAR;
            first.color_attachments[0].store_action = StoreAction::Store;
            let mut walk = attached(LoadAction::Load);
            walk.color_attachments[0].store_action = StoreAction::MultisampleResolve;
            walk.color_attachments[0].resolve_texture = Some(into.clone());
            let before = device.counters().metrics().resolves;
            passes(name, &device, &[&first, &walk]);
            let resolves = device.counters().metrics().resolves - before;
            assert_eq!(resolves, 1, "{name}: the walk's one resolve");
            cleared(name, &read_at(&into, WALK, 4, 0, 0));
            drop((color, depth, into));
            println!("{name}: the 900 x 900 loaded pass resolved into a ShaderRead texture");
        });
    }
}
