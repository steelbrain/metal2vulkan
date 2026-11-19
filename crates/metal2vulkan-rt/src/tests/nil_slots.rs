//! Texture slots that hold no texture, read by every stage of a draw.
//!
//! Under a device cfg, on every device of the cfg's kind, the owned fixture
//! `engine/nil_slots.ll`'s vertex, post-tessellation vertex and fragment functions, in a plain
//! draw (`nil_vertex`, `nil_fragment`) and a patch draw (`nil_post_tess`, `nil_fragment`, one
//! triangle patch whose function reads its three control points and its patch's tint), each
//! read and query the size and levels of five textures: texture slot 0 bound to nil, which they
//! also sample, and the four textures of an argument buffer whose IDs are 0, an index of the
//! consumer's grant that no texture ever took, the index of a texture a draw read and that then
//! went, and a live texture (not sampled: the runtime refuses a sample of an argument buffer's
//! texture, a known gap of its README). No command buffer ends in an error. Metal's answer, measured on an M3 Max under macOS
//! 27.0.1 for the nil slot and the ID 0 (the same in all three functions and both draws): a sample
//! and a read give the zero texel, and the width, the height and the level count are each 1. The
//! runtime's contract is that an element no live object owns answers as Metal's nil texture does
//! (`crate::arrays`), so the never-taken and the gone IDs are held to the same answer. The live
//! texture gives its texel, 4, 4 and 1 level; the patch's function its control points and tint as
//! bound. Metal's documents state no answer for a nil texture: `setVertexTexture:atIndex:` takes a
//! `nullable` texture, and the shading language gives `is_null_texture`.
//!
//! `llvm-as` assembles the fixture; without it the test skips visibly.

#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod on_device {
    use crate::tests::command::on_device::{each_device, within};
    use crate::tests::draw::on_device::{floats, library, pass, run, state, target, Buffers};
    use crate::{
        Buffer, ColorAttachmentDescriptor, Device, FunctionDescriptor, HostBytes, IndexClass,
        Library, PixelFormat, PrimitiveType, Region, RenderPipelineDescriptor, SamplerDescriptor,
        SamplerState, StageSource, Texture, TextureDescriptor, TextureType, TextureUsage,
    };
    use metal2vulkan::metal::{
        TessellationFactorStepFunction, TessellationPartitionMode, VertexFormat, VertexStepFunction,
    };
    use metal2vulkan::pipeline::{
        TessellationDescriptor, VertexAttributeDescriptor, VertexBufferLayoutDescriptor,
        VertexDescriptor,
    };
    use metal2vulkan_devtools::engine_dir;

    /// Words of a record, as the fixture's `RECORD`.
    const RECORD: usize = 80;
    /// Words of one texture's probe, as the fixture's `PROBE`.
    const PROBE: usize = 11;
    /// The live texture's texel, in every texel of its 4 x 4.
    const TEXEL: [f32; 4] = [0.5, 0.25, 0.125, 1.0];
    /// The patch's control points (the clip space a triangle covers, y up) and its tint.
    const POINTS: [f32; 12] = [
        -1.0, -1.0, 0.0, 1.0, 3.0, -1.0, 0.0, 1.0, -1.0, 3.0, 0.0, 1.0,
    ];
    const TINT: [f32; 4] = [0.25, 0.5, 0.75, 1.0];
    /// The texture indices of the consumer's grant: the live texture's, one no texture takes, and
    /// the gone texture's.
    const LIVE: u32 = 5;
    const NEVER: u32 = 6;
    const GONE: u32 = 7;

    /// The record a stage writes (`stage`: 1 vertex, 2 post-tessellation, 3 fragment), as Metal
    /// answers it: each of the four textures that hold none the zero texel at the sample and the
    /// read and 1, 1 and 1 level; the live one its texel at the read (its sample words 0, as the
    /// fixture samples no argument buffer's texture), 4, 4 and 1 level; the post-tessellation
    /// function its control points and tint.
    fn record(stage: u32) -> Vec<u32> {
        let mut r = vec![0; RECORD];
        for source in 0..4 {
            r[source * PROBE + 8..source * PROBE + 11].copy_from_slice(&[1, 1, 1]);
        }
        let live = 4 * PROBE;
        for c in 0..4 {
            r[live + 4 + c] = TEXEL[c].to_bits();
        }
        r[live + 8..live + 11].copy_from_slice(&[4, 4, 1]);
        if stage == 2 {
            for (i, v) in POINTS.iter().chain(&TINT).enumerate() {
                r[56 + i] = v.to_bits();
            }
        }
        r[72] = stage;
        r
    }

    /// Whether word `w` of a record is a size or level query's answer.
    fn query(w: usize) -> bool {
        w < 5 * PROBE && w % PROBE >= 8
    }

    /// The fixture's library, or None where `llvm-as` is missing.
    fn fixture() -> Option<Library> {
        library(&engine_dir().join("nil_slots.ll"))
    }

    fn function(library: &Library, name: &str) -> StageSource {
        StageSource::Function(FunctionDescriptor::new(
            library.function(name).expect("the function"),
        ))
    }

    /// `vertex` with `nil_fragment` into one RGBA8 attachment; for the post-tessellation function,
    /// its control points at buffer 0 (16 bytes each), its tint at buffer 3 per patch, integer
    /// partitions and factors per patch.
    fn descriptor(library: &Library, vertex: &str) -> RenderPipelineDescriptor {
        let mut d = RenderPipelineDescriptor::new(function(library, vertex));
        d.fragment = Some(function(library, "nil_fragment"));
        d.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
        if vertex == "nil_post_tess" {
            let mut v = VertexDescriptor::new();
            v.attributes.insert(
                0,
                VertexAttributeDescriptor::new(VertexFormat::Float4, 0, 0),
            );
            v.attributes.insert(
                5,
                VertexAttributeDescriptor::new(VertexFormat::Float4, 0, 3),
            );
            let mut points = VertexBufferLayoutDescriptor::new(Some(16));
            points.step_function = VertexStepFunction::PerPatchControlPoint;
            let mut tints = VertexBufferLayoutDescriptor::new(Some(16));
            tints.step_function = VertexStepFunction::PerPatch;
            v.layouts.insert(0, points);
            v.layouts.insert(3, tints);
            d.vertex_descriptor = Some(v);
            let mut t = TessellationDescriptor::default();
            t.factor_step_function = TessellationFactorStepFunction::PerPatch;
            t.partition_mode = TessellationPartitionMode::Integer;
            t.max_tessellation_factor = 16;
            d.tessellation = Some(t);
        }
        d
    }

    /// A 4 x 4 RGBA32Float texture of [`TEXEL`] at `index`.
    fn texture(name: &str, device: &Device, index: u32) -> Texture {
        let mut d = TextureDescriptor::new(TextureType::Type2D, PixelFormat::RGBA32Float, 4, 4);
        d.usage = TextureUsage::SHADER_READ;
        let texture =
            (device.new_texture(&d, Some(index))).unwrap_or_else(|error| panic!("{name}: {error}"));
        let bytes: Vec<u8> = (0..16).flat_map(|_| floats(&TEXEL)).collect();
        let region = Region {
            origin: [0, 0, 0],
            size: [4, 4, 1],
        };
        (texture.replace_region(region, 0, 0, HostBytes::from(&bytes[..]), 64, 256))
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        texture
    }

    /// The words of a Shared buffer.
    fn words(buffer: &Buffer, count: usize) -> Vec<u32> {
        let contents = buffer.contents().expect("a Shared buffer");
        (0..count)
            .map(|i| {
                // SAFETY: the word lies in the buffer, which is alive, and its work completed.
                unsafe { contents.as_ptr().add(4 * i).cast::<u32>().read_unaligned() }
            })
            .collect()
    }

    /// One draw of `state` into `color`, its texture slot 0 nil in both stages, `ab` the argument
    /// buffer and `out` the records, in a command buffer of its own; a patch draw where `patch`.
    fn draw(
        name: &str,
        device: &Device,
        state: &crate::RenderPipelineState,
        color: &Texture,
        sampler: &SamplerState,
        (ab, out): (&Buffer, &Buffer),
        factors: Option<&Buffer>,
    ) {
        let queue = device.new_command_queue(4).expect("a queue");
        let command_buffer = queue.command_buffer().expect("a command buffer");
        let mut e = (command_buffer.render_command_encoder(&pass(color, None)))
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        let ok =
            |result: Result<(), crate::Error>| result.unwrap_or_else(|e| panic!("{name}: {e}"));
        ok(e.set_render_pipeline_state(state));
        ok(e.set_vertex_texture(None, 0));
        ok(e.set_fragment_texture(None, 0));
        ok(e.set_vertex_sampler_state(Some(sampler), 0));
        ok(e.set_fragment_sampler_state(Some(sampler), 0));
        for index in [1, 2] {
            let buffer = if index == 1 { ab } else { out };
            ok(e.set_vertex_buffer(Some(buffer), 0, index));
            ok(e.set_fragment_buffer(Some(buffer), 0, index));
        }
        match factors {
            Some(factors) => {
                ok(e.set_vertex_bytes(&floats(&POINTS), 0));
                ok(e.set_vertex_bytes(&floats(&TINT), 3));
                ok(e.set_tessellation_factor_buffer(Some(factors), 0, 0));
                ok(e.draw_patches(3, 0, 1, None, 1, 0));
            }
            None => ok(e.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)),
        }
        e.end_encoding();
        run(name, &command_buffer);
    }

    /// The plain draw and the patch draw each answer [`record`] in both of their stages, with the
    /// argument buffer's IDs nil, never taken, gone and live; the gone texture's index answered
    /// its texel to a draw before it went.
    #[test]
    fn every_device_reads_zeros_from_texture_slots_that_hold_no_texture() {
        let Some(library) = fixture() else {
            return;
        };
        let plain = descriptor(&library, "nil_vertex");
        let patch = descriptor(&library, "nil_post_tess");
        each_device("nil_slots", |name, _, device| {
            let _textures = (device.grant_indices(IndexClass::Texture, 1, 8))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let _samplers = (device.grant_indices(IndexClass::Sampler, 1, 1))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let sampler = (device.new_sampler_state(&SamplerDescriptor::new(), 1))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let live = texture(name, &device, LIVE);
            let gone = texture(name, &device, GONE);
            let id = live.gpu_resource_id();
            let never = u64::from(NEVER) | u64::from(id.word) << 32;
            let ids: Vec<u8> = [0, never, gone.gpu_resource_id().to_bits(), id.to_bits()]
                .iter()
                .flat_map(|id| id.to_le_bytes())
                .collect();
            let mut buffers = Buffers::new(&device);
            let ab = buffers.of(&device, &ids);
            let factors: Vec<u8> = [0x3c00u16; 4]
                .iter()
                .flat_map(|f| f.to_le_bytes())
                .collect();
            let factors = buffers.of(&device, &factors);
            let color = target(
                &device,
                PixelFormat::RGBA8Unorm,
                TextureUsage::RENDER_TARGET,
            );
            let plain_state = state(name, &device, &plain);
            let patch_state = state(name, &device, &patch);
            // A draw reads the gone texture while it lives, then it goes and its index is free.
            let out = buffers.of(&device, &[0; 8 * RECORD]);
            draw(
                name,
                &device,
                &plain_state,
                &color,
                &sampler,
                (&ab, &out),
                None,
            );
            let third = 3 * PROBE..4 * PROBE;
            assert_eq!(
                words(&out, RECORD)[third.clone()],
                record(1)[4 * PROBE..5 * PROBE],
                "{name}: the gone texture's probe while it lived"
            );
            let counters = device.counters();
            let slots = counters.gauges().descriptor_slots;
            drop(gone);
            within("the gone texture's index", || {
                counters.gauges().descriptor_slots < slots
            });
            // Both draws run before anything is held, so a difference in one does not hide what
            // the other does.
            let mut got = Vec::new();
            for (what, state, factors, stage) in [
                ("the plain draw", &plain_state, None, 1),
                ("the patch draw", &patch_state, Some(&factors), 2),
            ] {
                let out = buffers.of(&device, &[0; 8 * RECORD]);
                draw(name, &device, state, &color, &sampler, (&ab, &out), factors);
                let both = words(&out, 2 * RECORD);
                let (first, second) = both.split_at(RECORD);
                for (record, stage) in [(first, stage), (second, 3)] {
                    let queries: Vec<u32> = (0..5 * PROBE)
                        .filter(|&w| query(w))
                        .map(|w| record[w])
                        .collect();
                    println!("{name}: {what}, stage {stage}: size and level answers {queries:?}");
                    got.push((what, stage, record.to_vec()));
                }
            }
            for (what, stage, record) in &got {
                let want = self::record(*stage);
                let texels = |r: &[u32]| -> Vec<u32> {
                    (0..RECORD).filter(|&w| !query(w)).map(|w| r[w]).collect()
                };
                assert_eq!(
                    texels(record),
                    texels(&want),
                    "{name}: {what}, stage {stage}: the texels, control points, tint and stage \
                     (11 words a texture: nil slot, ID 0, never taken, gone, live)"
                );
            }
            for (what, stage, record) in &got {
                assert_eq!(
                    record,
                    &self::record(*stage),
                    "{name}: {what}, stage {stage}: the size and level answers (Metal's nil \
                     texture answers 1, 1 and 1)"
                );
            }
            println!(
                "{name}: a plain draw and a patch draw read zeros in their vertex, \
                 post-tessellation and fragment functions from a nil texture slot and from \
                 argument-buffer IDs nil, never taken and gone, and answered Metal's sizes; the live \
                 texture answered its texel"
            );
        });
    }
}
