//! A draw finds and fills again only what changed: under a device cfg, on every device of the
//! cfg's kind, with the owned fixtures `examples/runtime_examples.ll` and
//! `engine/fca_vertex_unread_attribute.ll` and fragment functions this test writes:
//! - over 1,200 random calls of one encoder (pipeline states of 30 spilled buffers with two
//!   textures and a sampler, of 12 buffers, and of a fixed-function vertex binding; buffers,
//!   offsets, bytes and nothing bound at the vertex and fragment functions' slots; textures,
//!   sampler states, and draws), each draw's facts, push bytes and parameter blocks, found and
//!   filled slot by slot, equal a find and fill of every slot of what is bound, and its command
//!   buffer holds those push bytes; a draw refused is refused as a find of every slot is. The
//!   command buffer is never committed: some calls bind past a buffer's end.
//! - a draw pushes only the words its command buffer does not hold: the whole window at a
//!   command buffer's first, the words an offset or a buffer bound changed, none at a change of
//!   the pipeline state that changes no word (every pipeline has the device's one layout), and,
//!   in a pass after a dispatch in the same command buffer, only the words of the window the
//!   dispatch's push changed; every pixel reads the values its draw was bound.
//!
//! `llvm-as` assembles the fixtures; without it a device test skips visibly.

#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod on_device {
    use crate::tests::blit::on_device::settled;
    use crate::tests::command::on_device::each_device;
    use crate::tests::draw::on_device::{
        examples, floats, library, pass, read, run, state, target, vertices, Buffers, SIDE,
    };
    use crate::{
        Buffer, ColorAttachmentDescriptor, ComputePipelineDescriptor, Device, DispatchType, Error,
        FunctionDescriptor, IndexClass, Library, LoadAction, Metrics, PixelFormat, PrimitiveType,
        RenderPipelineDescriptor, RenderPipelineState, SamplerDescriptor, ScissorRect, StageSource,
        Texture, TextureDescriptor, TextureType, TextureUsage,
    };
    use metal2vulkan::library::{FunctionConstantValue, FunctionConstantValues};
    use metal2vulkan::metal::{DataType, VertexFormat};
    use metal2vulkan::pipeline::{
        VertexAttributeDescriptor, VertexBufferLayoutDescriptor, VertexDescriptor,
    };
    use metal2vulkan_devtools::{engine_dir, llvm_assemble_text, AssembleError};

    /// A module of fragment functions, each `(name, buffers, textures, wide, swapped)`: each returns
    /// (`b0[i]`, `b1[0]`, `b2[0]`, the sum of the other buffers' first words and of each texture's
    /// first channel sampled at (0.5, 0.5) through sampler 0), `b0` and `b1` swapped where
    /// `swapped`. A wide one reads `b0` at the index `b1`'s first word holds, a load the containment
    /// check takes, and stores its sum into its last buffer's second word; any other reads `b0[0]`.
    fn fragments(functions: &[(&str, u32, u32, bool, bool)]) -> String {
        let mut text = String::from(
            "; Owned synthetic fragment functions.\n\
    target datalayout = \"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32\"\n\
    target triple = \"air64_v28-apple-macosx26.0.0\"\n\n\
    declare { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i1, float, float, i32)\n\n",
        );
        // Metadata 1 to 5 are shared: the versions, the colour output and its list, the position.
        let mut meta = vec![
            "!1 = !{i32 2, i32 8, i32 0}".to_owned(),
            "!2 = !{!\"Metal\", i32 4, i32 0, i32 0}".to_owned(),
            "!3 = !{!\"air.render_target\", i32 0, i32 0, !\"air.arg_type_name\", !\"float4\"}"
                .to_owned(),
            "!4 = !{!3}".to_owned(),
            "!5 = !{i32 0, !\"air.position\", !\"air.center\", !\"air.no_perspective\", !\"air.arg_type_name\", !\"float4\", !\"air.arg_name\", !\"pos\"}".to_owned(),
        ];
        let mut next = 10;
        let mut entries = Vec::new();
        for &(name, buffers, textures, wide, swapped) in functions {
            let mut params = vec!["<4 x float> %pos".to_owned()];
            let mut args = vec!["!5".to_owned()];
            let entry = next;
            let list = next + 1;
            next += 2;
            for k in 0..buffers {
                params.push(format!("ptr addrspace(1) %b{k}"));
                let access = match wide && k == buffers - 1 {
                    true => "air.read_write",
                    false => "air.read",
                };
                meta.push(format!(
                    "!{next} = !{{i32 {}, !\"air.buffer\", !\"air.location_index\", i32 {k}, i32 1, !\"{access}\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"float\", !\"air.arg_name\", !\"b{k}\"}}",
                    k + 1
                ));
                args.push(format!("!{next}"));
                next += 1;
            }
            for t in 0..textures {
                params.push(format!("ptr addrspace(1) %t{t}"));
                meta.push(format!(
                    "!{next} = !{{i32 {}, !\"air.texture\", !\"air.location_index\", i32 {t}, i32 1, !\"air.sample\", !\"air.arg_type_name\", !\"texture2d<float, sample>\", !\"air.arg_name\", !\"t{t}\"}}",
                    buffers + 1 + t
                ));
                args.push(format!("!{next}"));
                next += 1;
            }
            if textures > 0 {
                params.push("ptr addrspace(2) %s".to_owned());
                meta.push(format!(
                    "!{next} = !{{i32 {}, !\"air.sampler\", !\"air.location_index\", i32 0, i32 1, !\"air.arg_type_name\", !\"sampler\", !\"air.arg_name\", !\"s\"}}",
                    buffers + 1 + textures
                ));
                args.push(format!("!{next}"));
                next += 1;
            }
            let mut body = String::new();
            for k in 1..buffers {
                body.push_str(&format!(
                    "  %v{k} = load float, ptr addrspace(1) %b{k}, align 4\n"
                ));
            }
            match wide {
                true => body.push_str(
                    "  %i = load i32, ptr addrspace(1) %b1, align 4\n  %iw = zext i32 %i to i64\n  %p0 = getelementptr inbounds float, ptr addrspace(1) %b0, i64 %iw\n  %v0 = load float, ptr addrspace(1) %p0, align 4\n",
                ),
                false => body.push_str("  %v0 = load float, ptr addrspace(1) %b0, align 4\n"),
            }
            body.push_str("  %a2 = fadd float 0.000000e+00, 0.000000e+00\n");
            let mut sum = 2;
            for k in 3..buffers {
                body.push_str(&format!("  %a{k} = fadd float %a{}, %v{k}\n", k - 1));
                sum = k;
            }
            let mut total = format!("%a{sum}");
            for t in 0..textures {
                body.push_str(&format!(
                    "  %r{t} = tail call {{ <4 x float>, i8 }} @air.sample_texture_2d.v4f32(ptr addrspace(1) %t{t}, ptr addrspace(2) %s, <2 x float> <float 5.000000e-01, float 5.000000e-01>, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)\n  %c{t} = extractvalue {{ <4 x float>, i8 }} %r{t}, 0\n  %e{t} = extractelement <4 x float> %c{t}, i64 0\n  %u{t} = fadd float {total}, %e{t}\n"
                ));
                total = format!("%u{t}");
            }
            if wide {
                body.push_str(&format!(
                    "  %q = getelementptr inbounds float, ptr addrspace(1) %b{}, i64 1\n  store float {total}, ptr addrspace(1) %q, align 4\n",
                    buffers - 1
                ));
            }
            let (x, y) = match swapped {
                true => ("%v1", "%v0"),
                false => ("%v0", "%v1"),
            };
            body.push_str(&format!(
                "  %o0 = insertelement <4 x float> undef, float {x}, i64 0\n  %o1 = insertelement <4 x float> %o0, float {y}, i64 1\n  %o2 = insertelement <4 x float> %o1, float %v2, i64 2\n  %o3 = insertelement <4 x float> %o2, float {total}, i64 3\n  ret <4 x float> %o3\n"
            ));
            text.push_str(&format!(
                "define <4 x float> @{name}({}) {{\nentry:\n{body}}}\n\n",
                params.join(", ")
            ));
            meta.push(format!("!{entry} = !{{ptr @{name}, !4, !{list}}}"));
            meta.push(format!("!{list} = !{{{}}}", args.join(", ")));
            entries.push(format!("!{entry}"));
        }
        text.push_str(&format!(
            "!air.fragment = !{{{}}}\n!air.version = !{{!1}}\n!air.language_version = !{{!2}}\n\n{}\n",
            entries.join(", "),
            meta.join("\n")
        ));
        text
    }

    /// The library of [`fragments`] of `functions`, or None where `llvm-as` is missing.
    fn fragment_library(functions: &[(&str, u32, u32, bool, bool)]) -> Option<Library> {
        match llvm_assemble_text(&fragments(functions)) {
            Ok(bitcode) => Some(Library::from_air(&bitcode).expect("the library")),
            Err(AssembleError::Missing(why)) => {
                eprintln!("SKIPPED: {why}");
                None
            }
            Err(error) => panic!("the fragment functions: {error:?}"),
        }
    }

    /// A pipeline state of `examples`' `colored_vertex` and `fragment`'s function `name`, into
    /// one colour attachment of `color`.
    fn colored(
        device: &Device,
        (examples, fragment): (&Library, &Library),
        name: &str,
        color: PixelFormat,
    ) -> RenderPipelineState {
        let function = |library: &Library, name| {
            StageSource::Function(FunctionDescriptor::new(
                library.function(name).expect("the function"),
            ))
        };
        let mut d = RenderPipelineDescriptor::new(function(examples, "colored_vertex"));
        d.fragment = Some(function(fragment, name));
        d.color_attachments[0] = Some(ColorAttachmentDescriptor::new(color));
        state(name, device, &d)
    }

    /// A deterministic sequence (xorshift64*).
    struct Sequence(u64);

    impl Sequence {
        /// The next value below `n`.
        fn below(&mut self, n: usize) -> usize {
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            (self.0.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 33) as usize % n
        }
    }

    /// Over 1,200 random calls of one encoder, every draw's facts and fields equal a find and fill
    /// of every slot, and its command buffer holds its push bytes; a refused draw is refused alike.
    #[test]
    fn every_device_refills_only_what_its_binds_change() {
        let Some(examples) = examples() else {
            return;
        };
        let Some(fragment) = fragment_library(&[("wide_fragment", 30, 2, true, false), TWELVE[0]])
        else {
            return;
        };
        let Some(stage_in) = library(&engine_dir().join("fca_vertex_unread_attribute.ll")) else {
            return;
        };
        // `fca_vertex_unread_attribute`'s two attributes from vertex binding 0, which reads
        // buffer slot 0.
        let constants = FunctionConstantValues::defaults()
            .with_index(0, FunctionConstantValue::new(DataType::Bool, [1]));
        let function = |name| {
            let mut f = FunctionDescriptor::new(stage_in.function(name).expect("the function"));
            f.constants = constants.clone();
            f.specialized = true;
            StageSource::Function(f)
        };
        let mut fetched = RenderPipelineDescriptor::new(function("fca_vertex_unread_attribute"));
        fetched.fragment = Some(function("fca_fragment_unread_attribute"));
        fetched.color_attachments[0] =
            Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
        let mut vertex = VertexDescriptor::new();
        for a in 0..2 {
            let attribute = VertexAttributeDescriptor::new(VertexFormat::Float4, 16 * a, 0);
            vertex.attributes.insert(a, attribute);
        }
        vertex
            .layouts
            .insert(0, VertexBufferLayoutDescriptor::new(Some(32)));
        fetched.vertex_descriptor = Some(vertex);
        each_device("refill", |name, _, device| {
            let libraries = (&examples, &fragment);
            let format = PixelFormat::RGBA8Unorm;
            let states = [
                colored(&device, libraries, "wide_fragment", format),
                colored(&device, libraries, "twelve_fragment", format),
                state(name, &device, &fetched),
            ];
            let spilled = &states[0].inner.compiled.draws.blocks;
            assert!(
                spilled[1].is_some(),
                "{name}: the wide fragment function's fields spill into a parameter block"
            );
            let _textures = (device.grant_indices(IndexClass::Texture, 1, 3))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let _samplers = (device.grant_indices(IndexClass::Sampler, 1, 2))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let textures: Vec<Texture> = (1..=3)
                .map(|index| {
                    let mut d =
                        TextureDescriptor::new(TextureType::Type2D, PixelFormat::RGBA8Unorm, 4, 4);
                    d.usage = TextureUsage::SHADER_READ;
                    (device.new_texture(&d, Some(index)))
                        .unwrap_or_else(|error| panic!("{name}: {error}"))
                })
                .collect();
            let samplers: Vec<_> = (1..=2)
                .map(|index| {
                    (device.new_sampler_state(&SamplerDescriptor::new(), index))
                        .unwrap_or_else(|error| panic!("{name}: {error}"))
                })
                .collect();
            let mut buffers = Buffers::new(&device);
            let bound: Vec<Buffer> = (0..4u8).map(|k| buffers.of(&device, &[k; 256])).collect();
            let color = target(&device, format, TextureUsage::RENDER_TARGET);
            let queue = device.new_command_queue(4).expect("a queue");
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let mut encoder = (command_buffer.render_command_encoder(&pass(&color, None)))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            // Offsets inside the buffers, one off every argument's alignment, one past their end.
            const OFFSETS: [u64; 5] = [0, 16, 48, 2, 4096];
            let mut sequence = Sequence(0x9e37_79b9_7f4a_7c15);
            let (mut drawn, mut refused) = (0, 0);
            let call = |result: Result<(), Error>| result.unwrap_or_else(|e| panic!("{name}: {e}"));
            call(encoder.set_render_pipeline_state(&states[0]));
            for step in 0..1_200 {
                let table = sequence.below(2);
                match sequence.below(12) {
                    0 => call(encoder.set_render_pipeline_state(&states[sequence.below(3)])),
                    1..=3 => {
                        let slot = sequence.below(31) as u32;
                        let buffer = (sequence.below(8) != 0).then(|| &bound[sequence.below(4)]);
                        let offset = OFFSETS[sequence.below(5)];
                        call(match table {
                            0 => encoder.set_vertex_buffer(buffer, offset, slot % 3),
                            _ => encoder.set_fragment_buffer(buffer, offset, slot),
                        });
                    }
                    4 => {
                        // Refused where the slot holds no buffer, as Metal's.
                        let (slot, offset) =
                            (sequence.below(31) as u32, OFFSETS[sequence.below(5)]);
                        let _ = match table {
                            0 => encoder.set_vertex_buffer_offset(offset, slot % 3),
                            _ => encoder.set_fragment_buffer_offset(offset, slot),
                        };
                    }
                    5 => {
                        let slot = sequence.below(31) as u32;
                        let bytes = [step as u8; 96];
                        call(match table {
                            0 => encoder.set_vertex_bytes(&bytes, slot % 3),
                            _ => encoder.set_fragment_bytes(&bytes, slot),
                        });
                    }
                    6 => {
                        let texture = &textures[sequence.below(3)];
                        call(encoder.set_fragment_texture(Some(texture), sequence.below(2) as u32));
                    }
                    7 => call(
                        encoder.set_fragment_sampler_state(Some(&samplers[sequence.below(2)]), 0),
                    ),
                    _ => {
                        let result = encoder.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0);
                        let refilled = encoder.refilled();
                        match result {
                            Ok(()) => {
                                assert_eq!(
                                    Ok(&refilled.slot_by_slot),
                                    refilled.whole.as_ref(),
                                    "{name}: step {step}: slot by slot and whole"
                                );
                                assert_eq!(
                                    refilled.held.get(..refilled.push.len()),
                                    Some(&refilled.push[..]),
                                    "{name}: step {step}: the command buffer's push bytes"
                                );
                                drawn += 1;
                            }
                            Err(error) => {
                                assert_eq!(
                                    refilled.whole.err(),
                                    Some(format!("{error:?}")),
                                    "{name}: step {step}: the refusal"
                                );
                                refused += 1;
                            }
                        }
                    }
                }
            }
            assert!(
                drawn > 100 && refused > 10,
                "{name}: {drawn} drawn, {refused} refused"
            );
            encoder.end_encoding();
            println!("{name}: {drawn} draws equal a whole find and fill, {refused} refused alike");
        });
    }

    /// Fragment functions of twelve buffers, the second with `b0` and `b1` swapped: the same fields
    /// in the same places, and a window longer than a dispatch's of three buffers.
    const TWELVE: [(&str, u32, u32, bool, bool); 2] = [
        ("twelve_fragment", 12, 0, false, false),
        ("swapped_fragment", 12, 0, false, true),
    ];

    /// The bytes a push of `push` records where the command buffer holds `held`: from the first
    /// word that differs to the last, and every word past `held`.
    fn changed(held: &[u8], push: &[u8]) -> u64 {
        let held = &held[..held.len().min(push.len())];
        let differs: Vec<usize> = (0..held.len())
            .step_by(4)
            .filter(|&at| held[at..at + 4] != push[at..at + 4])
            .collect();
        let first = differs.first().copied().unwrap_or(held.len());
        let end = match push.len() > held.len() {
            true => push.len(),
            false => differs.last().map_or(0, |at| at + 4),
        };
        end.saturating_sub(first) as u64
    }

    /// Pushes recorded, pushes held and push bytes since `before`.
    fn pushes(device: &Device, before: &Metrics) -> (u64, u64, u64) {
        let after = device.counters().metrics();
        (
            after.pushes - before.pushes,
            after.pushes_held - before.pushes_held,
            after.push_bytes - before.push_bytes,
        )
    }

    /// One command buffer: a pass whose first draw pushes its whole window, an offset change that
    /// pushes the changed words alone, a change of pipeline state that changes no word and pushes
    /// none, and a buffer bound that pushes the changed words alone; then a dispatch, and a pass
    /// whose draw pushes only the words of the window the dispatch's push changed. A second
    /// command buffer's first draw pushes the whole window again. Each draw writes its own
    /// column, which reads the values it was bound, and the dispatch writes its words.
    #[test]
    fn every_device_pushes_only_the_words_its_command_buffer_lacks() {
        let Some(examples) = examples() else {
            return;
        };
        let Some(fragment) = fragment_library(&TWELVE) else {
            return;
        };
        each_device("refill_push", |name, _, device| {
            let format = PixelFormat::RGBA32Float;
            let libraries = (&examples, &fragment);
            let twelve = colored(&device, libraries, "twelve_fragment", format);
            let swapped = colored(&device, libraries, "swapped_fragment", format);
            let places = |state: &RenderPipelineState| {
                let draws = &state.inner.compiled.draws;
                (draws.push_len, draws.fields.len(), draws.blocks)
            };
            assert_eq!(
                places(&twelve),
                places(&swapped),
                "{name}: the two states' push windows"
            );
            let scale = ComputePipelineDescriptor::new(StageSource::Function(
                FunctionDescriptor::new(examples.function("scale_and_offset").expect("a kernel")),
            ));
            let scale = (device
                .new_compute_pipeline_state(&scale, crate::tests::no_constexpr_samplers))
            .unwrap_or_else(|error| panic!("{name}: {error}"));
            let mut buffers = Buffers::new(&device);
            let (positions, colors) = vertices(&device, &mut buffers);
            let a = buffers.of(&device, &floats(&[0.25]));
            let b = buffers.of(&device, &floats(&[0.5, 0.75]));
            let c = buffers.of(&device, &floats(&[0.125]));
            let d = buffers.of(&device, &floats(&[0.875]));
            let e = buffers.of(&device, &floats(&[0.0625]));
            let input = buffers.of(&device, &floats(&[1.0, 2.0, 3.0, 4.0]));
            let output = buffers.of(&device, &[0; 16]);
            let color = target(&device, format, TextureUsage::RENDER_TARGET);
            let queue = device.new_command_queue(4).expect("a queue");
            let call = |result: Result<(), Error>| result.unwrap_or_else(|e| panic!("{name}: {e}"));
            let column = |encoder: &mut crate::RenderCommandEncoder, x: u64| {
                encoder.set_scissor_rect(ScissorRect {
                    x,
                    y: 0,
                    width: 1,
                    height: u64::from(SIDE),
                });
                call(encoder.draw_primitives(PrimitiveType::Triangle, 3, 3, 1, 0));
            };
            let loaded = || {
                let mut d = pass(&color, None);
                d.color_attachments[0].load_action = LoadAction::Load;
                d
            };
            settled(&device);
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let before = device.counters().metrics();
            let mut encoder = (command_buffer.render_command_encoder(&pass(&color, None)))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            call(encoder.set_render_pipeline_state(&twelve));
            call(encoder.set_vertex_buffer(Some(&positions), 0, 0));
            call(encoder.set_vertex_buffer(Some(&colors), 0, 1));
            for (slot, buffer) in [&a, &b, &c].into_iter().chain([&e; 9]).enumerate() {
                call(encoder.set_fragment_buffer(Some(buffer), 0, slot as u32));
            }
            // Each draw's bytes, against what the command buffer held before it.
            let mut want = Vec::new();
            let mut held = Vec::new();
            for step in 0..4 {
                match step {
                    1 => call(encoder.set_fragment_buffer_offset(4, 1)),
                    2 => call(encoder.set_render_pipeline_state(&swapped)),
                    3 => call(encoder.set_fragment_buffer(Some(&d), 0, 2)),
                    _ => {}
                }
                column(&mut encoder, step);
                let refilled = encoder.refilled();
                want.push(changed(&held, &refilled.push));
                held = refilled.held;
            }
            let window = want[0];
            assert_eq!(
                window,
                u64::from(twelve.inner.compiled.draws.push_len.next_multiple_of(4)),
                "{name}: the first draw's whole window"
            );
            assert!(
                (1..=8).contains(&want[1]) && (1..=8).contains(&want[3]),
                "{name}: one address's words at a bind: {want:?}"
            );
            assert_eq!(
                want[2], 0,
                "{name}: a change of pipeline state that changes no word"
            );
            encoder.end_encoding();
            assert_eq!(
                pushes(&device, &before),
                (3, 1, want.iter().sum()),
                "{name}: the first pass's pushes, pushes held and bytes ({want:?})"
            );
            let before = device.counters().metrics();
            let mut compute =
                (command_buffer.compute_command_encoder(DispatchType::Serial)).expect("an encoder");
            call(compute.set_compute_pipeline_state(&scale));
            call(compute.set_buffer(Some(&input), 0, 0));
            call(compute.set_buffer(Some(&output), 0, 1));
            call(compute.set_bytes(&floats(&[2.0]), 2));
            call(compute.dispatch_threads([4, 1, 1], [4, 1, 1]));
            compute.end_encoding();
            let (dispatched, _, _) = pushes(&device, &before);
            assert_eq!(dispatched, 1, "{name}: the dispatch's push");
            let before = device.counters().metrics();
            let mut encoder = (command_buffer.render_command_encoder(&loaded()))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            call(encoder.set_render_pipeline_state(&swapped));
            let after_dispatch = encoder.refilled().held;
            call(encoder.set_vertex_buffer(Some(&positions), 0, 0));
            call(encoder.set_vertex_buffer(Some(&colors), 0, 1));
            for (slot, buffer) in [&a, &b, &d].into_iter().chain([&e; 9]).enumerate() {
                let offset = if slot == 1 { 4 } else { 0 };
                call(encoder.set_fragment_buffer(Some(buffer), offset, slot as u32));
            }
            column(&mut encoder, 4);
            let refilled = encoder.refilled();
            let after = changed(&after_dispatch, &refilled.push);
            encoder.end_encoding();
            assert_eq!(
                pushes(&device, &before),
                (1, 0, after),
                "{name}: the pass after the dispatch"
            );
            assert!(
                after > 0 && after < window,
                "{name}: the words the dispatch changed, not the whole window: {after} of {window}"
            );
            run(name, &command_buffer);
            let before = device.counters().metrics();
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let mut encoder = (command_buffer.render_command_encoder(&loaded()))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            call(encoder.set_render_pipeline_state(&swapped));
            call(encoder.set_vertex_buffer(Some(&positions), 0, 0));
            call(encoder.set_vertex_buffer(Some(&colors), 0, 1));
            for (slot, buffer) in [&a, &b, &d].into_iter().chain([&e; 9]).enumerate() {
                let offset = if slot == 1 { 4 } else { 0 };
                call(encoder.set_fragment_buffer(Some(buffer), offset, slot as u32));
            }
            column(&mut encoder, 5);
            encoder.end_encoding();
            assert_eq!(
                pushes(&device, &before),
                (1, 0, window),
                "{name}: the next command buffer's first draw"
            );
            run(name, &command_buffer);
            // Each column's (b0 or b1, b1 or b0, b2, nine times 0.0625).
            let rest = 9.0 * 0.0625;
            let columns: [[f32; 4]; 8] = [
                [0.25, 0.5, 0.125, rest],
                [0.25, 0.75, 0.125, rest],
                [0.75, 0.25, 0.125, rest],
                [0.75, 0.25, 0.875, rest],
                [0.75, 0.25, 0.875, rest],
                [0.75, 0.25, 0.875, rest],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0],
            ];
            let texels = read(&color, 16);
            for (at, texel) in texels.chunks(16).enumerate() {
                let got: Vec<f32> = (texel.chunks(4))
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                let x = at % SIDE as usize;
                assert_eq!(got, columns[x], "{name}: pixel {at}, column {x}");
            }
            let contents = output.contents().expect("a Shared buffer");
            // SAFETY: the words lie in the buffer, which is alive, and its work completed.
            let words: Vec<f32> = (0..4)
                .map(|i| unsafe { contents.as_ptr().cast::<f32>().add(i).read_unaligned() })
                .collect();
            assert_eq!(words, [2.0, 5.0, 8.0, 11.0], "{name}: the dispatch's words");
            println!(
                "{name}: pushes of {want:?} bytes, then {after} of {window} after a dispatch, then {window} in a new command buffer"
            );
        });
    }
}
