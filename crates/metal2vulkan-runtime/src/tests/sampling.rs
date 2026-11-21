//! Kernels that sample textures, under a device cfg on every device of the cfg's kind: the
//! harness's `sample_down`, through a runtime sampler state bound with LOD clamps that read as its
//! own, under a stage-in region it ignores, and `sample_up`, through a constexpr sampler at the
//! index the consumer gives it, chained in one serial encoder so the second samples what the first
//! wrote. Each answer is checked against the linear filter's, and the constexpr sampler's index is
//! shared between the states that number it there and freed with the last.
//! A texture of any usage is made at any index of a texture grant: `sample_down` samples a texture
//! of shader-read usage alone at an index below 4,096 and writes one of shader-write usage above
//! it, in one dispatch. An index belongs to the texture placed there last: sampling and writing
//! through it reach that texture, a displaced texture's going touches nothing, a placement of a
//! texture with no storage view empties the storage element an earlier one wrote, and once the
//! last placed goes, the index samples zero. The rule holds at a constexpr sampler's index, and
//! over a texture and a sampler state the program released while a command buffer that used
//! them is still pending. A texture made for its views has no index of its own: `sample_down`
//! samples and writes it through views at their own indices, twins and a view of a view among
//! them, while a texture shaders reach made without an index is still refused.
//!
//! `llvm-as` assembles the kernels; without it the test skips visibly.

#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod on_device {
    use crate::device::lock;
    use crate::tests::command::on_device::{each_device, finished, shared_event, PATIENCE};
    use crate::{
        CommandBuffer, CommandBufferStatus, CommandQueue, ComputePipelineDescriptor,
        ComputePipelineState, Device, DispatchType, Error, FunctionDescriptor, IndexClass, Library,
        Limit, NumberedSampler, PixelFormat, Refusal, Region, SamplerDescriptor,
        SamplerMinMagFilter, SamplerState, StageSource, Texture, TextureDescriptor, TextureType,
        TextureUsage,
    };
    use metal2vulkan_devtools::{fixtures_dir, llvm_assemble, AssembleError};

    /// The source's side; the middle texture's is half of it.
    const SIDE: u32 = 8;
    /// The constexpr sampler's index the test gives `sample_up`.
    const CONSTEXPR: u32 = 2;

    /// The harness's library, or None where `llvm-as` is missing.
    fn library() -> Option<Library> {
        match llvm_assemble(&fixtures_dir().join("harness/workloads.ll")) {
            Ok(bitcode) => Some(Library::from_air(&bitcode).expect("the library")),
            Err(AssembleError::Missing(why)) => {
                eprintln!("SKIPPED: {why}");
                None
            }
            Err(error) => panic!("workloads.ll: {error:?}"),
        }
    }

    /// `x`'s `half` bits, for a value a `half` holds exactly: zero, or a normal one.
    fn half(x: f32) -> u16 {
        if x == 0.0 {
            return 0;
        }
        let bits = x.to_bits();
        let exponent = ((bits >> 23) & 0xff) as i32 - 127 + 15;
        assert!((1..31).contains(&exponent) && bits & 0x1fff == 0, "{x}");
        ((bits >> 16) & 0x8000) as u16 | ((exponent as u16) << 10) | ((bits >> 13) & 0x3ff) as u16
    }

    /// The `float` of `half` bits `h` of a zero or a normal value.
    fn float(h: u16) -> f32 {
        let exponent = u32::from((h >> 10) & 0x1f);
        let sign = u32::from(h & 0x8000) << 16;
        match exponent {
            0 => f32::from_bits(sign),
            _ => {
                f32::from_bits(sign | ((exponent + 127 - 15) << 23) | (u32::from(h & 0x3ff) << 13))
            }
        }
    }

    /// An `RGBA16Float` 2D texture of `side` texels a side.
    fn texture(device: &Device, side: u32, usage: TextureUsage, index: u32) -> Texture {
        let mut d =
            TextureDescriptor::new(TextureType::Type2D, PixelFormat::RGBA16Float, side, side);
        d.usage = usage;
        device
            .new_texture(&d, Some(index))
            .unwrap_or_else(|error| panic!("{d:?}: {error}"))
    }

    /// The texels of `texture`, `side` a side, as `float`s, four a texel.
    fn texels(texture: &Texture, side: u32) -> Vec<f32> {
        let mut bytes = vec![0u8; (side * side * 8) as usize];
        let whole = Region {
            origin: [0; 3],
            size: [side, side, 1],
        };
        texture
            .get_bytes(
                &mut bytes,
                u64::from(side * 8),
                u64::from(side * side * 8),
                whole,
                0,
                0,
            )
            .expect("the texels");
        bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|&h| float(u16::from_le_bytes(h)))
            .collect()
    }

    /// The state of the harness's kernel `name`, its numbered samplers given `ids`, after checking
    /// they are `numbered`.
    fn state(
        device: &Device,
        library: &Library,
        name: &str,
        numbered: fn(&[NumberedSampler]) -> bool,
        ids: Vec<u32>,
    ) -> Result<ComputePipelineState, Error> {
        let function = library.function(name).expect("the kernel");
        let descriptor = ComputePipelineDescriptor::new(StageSource::Function(
            FunctionDescriptor::new(function),
        ));
        device.new_compute_pipeline_state(&descriptor, |samplers| {
            assert!(numbered(samplers), "{name}: {samplers:?}");
            Ok(ids)
        })
    }

    /// Texel `(x, y)` of the source: its channels, in sixteenths a half holds exactly.
    fn texel(x: u32, y: u32) -> [f32; 4] {
        [x, y, x + y, 16].map(|k| k as f32 / 16.0)
    }

    /// Writes every texel of `source`, `SIDE` a side, [`texel`].
    fn fill(name: &str, source: &Texture) {
        fill_with(name, source, texel);
    }

    /// Writes every texel `(x, y)` of `source`, `SIDE` a side, `texel(x, y)`.
    fn fill_with(name: &str, source: &Texture, texel: fn(u32, u32) -> [f32; 4]) {
        let bytes: Vec<u8> = (0..SIDE)
            .flat_map(|y| (0..SIDE).flat_map(move |x| texel(x, y)))
            .flat_map(|v| half(v).to_le_bytes())
            .collect();
        let whole = Region {
            origin: [0; 3],
            size: [SIDE, SIDE, 1],
        };
        let (row, image) = (u64::from(SIDE * 8), u64::from(SIDE * SIDE * 8));
        (source.replace_region(whole, 0, 0, crate::HostBytes::from(&bytes[..]), row, image))
            .unwrap_or_else(|error| panic!("{name}: {error}"));
    }

    /// What `sample_down` writes of the source: each texel of a texture half its side the mean of
    /// its four source texels.
    fn means() -> Vec<f32> {
        let half_side = SIDE / 2;
        let at = |x: u32, y: u32, c: usize| texel(x, y)[c];
        (0..half_side)
            .flat_map(|y| (0..half_side).flat_map(move |x| (0..4).map(move |c| (x, y, c))))
            .map(|(x, y, c)| {
                let (x, y) = (2 * x, 2 * y);
                (at(x, y, c) + at(x + 1, y, c) + at(x, y + 1, c) + at(x + 1, y + 1, c)) / 4.0
            })
            .collect()
    }

    /// The state of the harness's `sample_down`, which numbers its texture write's sampler alone.
    fn down_state(name: &str, device: &Device, library: &Library) -> ComputePipelineState {
        let write = |s: &[NumberedSampler]| matches!(s, [NumberedSampler::Write]);
        state(device, library, "sample_down", write, vec![0])
            .unwrap_or_else(|error| panic!("{name}: sample_down: {error}"))
    }

    /// A linear sampler state at `index`.
    fn linear(name: &str, device: &Device, index: u32) -> SamplerState {
        let mut linear = SamplerDescriptor::new();
        linear.min_filter = SamplerMinMagFilter::Linear;
        linear.mag_filter = SamplerMinMagFilter::Linear;
        (device.new_sampler_state(&linear, index)).unwrap_or_else(|error| panic!("{name}: {error}"))
    }

    /// One dispatch of `sample_down` (`down`) in a command buffer of its own, sampling `source`,
    /// `SIDE` a side, through `sampler` into `target`'s first half of each side, committed and
    /// waited for.
    fn sample_down(
        name: &str,
        queue: &CommandQueue,
        pass: (&ComputePipelineState, &SamplerState),
        textures: [&Texture; 2],
    ) {
        let command_buffer = queue.command_buffer().expect("a command buffer");
        encode_down(name, &command_buffer, pass, textures);
        command_buffer.commit().expect("a commit");
        finished(name, &command_buffer);
        assert!(command_buffer.error().is_none(), "{name}");
    }

    /// Encodes one dispatch of `sample_down` (`down`) into `command_buffer`, sampling `source`,
    /// `SIDE` a side, through `sampler` into `target`'s first half of each side.
    fn encode_down(
        name: &str,
        command_buffer: &CommandBuffer,
        (down, sampler): (&ComputePipelineState, &SamplerState),
        [source, target]: [&Texture; 2],
    ) {
        let mut encoder = command_buffer
            .compute_command_encoder(DispatchType::Serial)
            .expect("an encoder");
        encoder.set_compute_pipeline_state(down).expect("down");
        encoder.set_texture(Some(source), 0).expect("source");
        encoder.set_texture(Some(target), 1).expect("target");
        // Bound with the clamps a replay stream binds it with, which read as its own.
        encoder
            .set_sampler_state_with_lod_clamps(Some(sampler), 0.0, 1000.0, 0)
            .expect("sampler");
        // A stage-in region, which a kernel without a `[[stage_in]]` argument ignores.
        encoder.set_stage_in_region(Region {
            origin: [1, 2, 0],
            size: [3, 4, 1],
        });
        encoder
            .dispatch_threads([SIDE / 2, SIDE / 2, 1], [4, 4, 1])
            .unwrap_or_else(|error| panic!("{name}: sample_down: {error}"));
        encoder.end_encoding();
    }

    /// Whether a state's numbered samplers are `sample_up`'s: its constexpr sampler, then its
    /// texture write's.
    fn up_samplers(s: &[NumberedSampler]) -> bool {
        matches!(s, [NumberedSampler::Constexpr(_), NumberedSampler::Write])
    }

    /// A state of the harness's `sample_up`, its constexpr sampler at `CONSTEXPR`.
    fn up_state(name: &str, device: &Device, library: &Library) -> ComputePipelineState {
        state(
            device,
            library,
            "sample_up",
            up_samplers,
            vec![CONSTEXPR, 0],
        )
        .unwrap_or_else(|error| panic!("{name}: sample_up: {error}"))
    }

    /// One dispatch of `sample_up` (`up`) in a command buffer of its own, sampling `middle`, half
    /// `SIDE` a side, through its constexpr sampler into `output`, `SIDE` a side, committed and
    /// waited for: `output`'s texels.
    fn sample_up(
        name: &str,
        queue: &CommandQueue,
        up: &ComputePipelineState,
        [middle, output]: [&Texture; 2],
    ) -> Vec<f32> {
        let command_buffer = queue.command_buffer().expect("a command buffer");
        let mut encoder = command_buffer
            .compute_command_encoder(DispatchType::Serial)
            .expect("an encoder");
        encoder.set_compute_pipeline_state(up).expect("up");
        encoder.set_texture(Some(middle), 0).expect("middle");
        encoder.set_texture(Some(output), 1).expect("output");
        encoder
            .dispatch_threads([SIDE, SIDE, 1], [4, 4, 1])
            .unwrap_or_else(|error| panic!("{name}: sample_up: {error}"));
        encoder.end_encoding();
        command_buffer.commit().expect("a commit");
        finished(name, &command_buffer);
        assert!(command_buffer.error().is_none(), "{name}");
        texels(output, SIDE)
    }

    /// What `sample_up` writes of `mean`, half `SIDE` a side, through a linear filter: each texel
    /// a quarter of a middle texel from a middle texel's center, clamped to the edge.
    fn upsampled(mean: &[f32]) -> Vec<f32> {
        let half_side = SIDE / 2;
        let taps = |x: u32| {
            let k = x / 2;
            match x % 2 {
                0 => (k.saturating_sub(1), k, 0.75f32),
                _ => (k, (k + 1).min(half_side - 1), 0.25f32),
            }
        };
        let mid = |x: u32, y: u32, c: u32| mean[(4 * (y * half_side + x) + c) as usize];
        (0..SIDE)
            .flat_map(|y| (0..SIDE).flat_map(move |x| (0..4).map(move |c| (x, y, c))))
            .map(|(x, y, c)| {
                let ((x0, x1, wx), (y0, y1, wy)) = (taps(x), taps(y));
                let row = |y| mid(x0, y, c) * (1.0 - wx) + mid(x1, y, c) * wx;
                row(y0) * (1.0 - wy) + row(y1) * wy
            })
            .collect()
    }

    /// Whether `got` is the linear filter's `want`, within what a device's filter weights round.
    fn near(got: &[f32], want: &[f32]) -> bool {
        got.len() == want.len()
            && got
                .iter()
                .zip(want)
                .all(|(g, w)| (g - w).abs() <= 1.0 / 256.0)
    }

    #[test]
    fn every_device_samples_through_runtime_and_constexpr_samplers() {
        let Some(library) = library() else {
            return;
        };
        each_device("sampling_runtime_and_constexpr", |name, _, device| {
            let counters = device.counters();
            let _grants = [
                (IndexClass::Texture, 4096),
                (IndexClass::Texture, 1),
                (IndexClass::Sampler, 1),
            ]
            .map(|(class, first)| {
                device
                    .grant_indices(class, first, 4)
                    .unwrap_or_else(|error| panic!("{name}: {error}"))
            });
            let read = TextureUsage::SHADER_READ;
            let source = texture(&device, SIDE, read, 4096);
            let middle = texture(&device, SIDE / 2, read | TextureUsage::SHADER_WRITE, 1);
            let output = texture(&device, SIDE, read | TextureUsage::SHADER_WRITE, 2);
            fill(name, &source);
            let mut linear = SamplerDescriptor::new();
            linear.min_filter = SamplerMinMagFilter::Linear;
            linear.mag_filter = SamplerMinMagFilter::Linear;
            let sampler = device
                .new_sampler_state(&linear, 1)
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let slots = counters.gauges().descriptor_slots;
            // `sample_down` numbers its texture write's sampler alone; `sample_up` its constexpr
            // sampler first, which takes an index of the grant.
            let down = state(
                &device,
                &library,
                "sample_down",
                |s| matches!(s, [NumberedSampler::Write]),
                vec![0],
            )
            .unwrap_or_else(|error| panic!("{name}: sample_down: {error}"));
            let constexpr = |s: &[NumberedSampler]| {
                matches!(s, [NumberedSampler::Constexpr(_), NumberedSampler::Write])
            };
            let up = state(
                &device,
                &library,
                "sample_up",
                constexpr,
                vec![CONSTEXPR, 0],
            )
            .unwrap_or_else(|error| panic!("{name}: sample_up: {error}"));
            assert_eq!(counters.gauges().descriptor_slots, slots + 1, "{name}");
            // A second state that numbers it at the same index shares it; an answer of another
            // count is refused.
            let again = state(
                &device,
                &library,
                "sample_up",
                constexpr,
                vec![CONSTEXPR, 0],
            )
            .unwrap_or_else(|error| panic!("{name}: sample_up again: {error}"));
            assert_eq!(counters.gauges().descriptor_slots, slots + 1, "{name}");
            assert!(
                matches!(
                    state(&device, &library, "sample_up", constexpr, vec![CONSTEXPR]),
                    Err(Error::Refused(Refusal::SamplerIds {
                        numbered: 2,
                        given: 1
                    }))
                ),
                "{name}"
            );
            drop(again);
            assert_eq!(counters.gauges().descriptor_slots, slots + 1, "{name}");

            let queue = device.new_command_queue(4).expect("a queue");
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let mut encoder = command_buffer
                .compute_command_encoder(DispatchType::Serial)
                .expect("an encoder");
            encoder.set_compute_pipeline_state(&down).expect("down");
            encoder.set_texture(Some(&source), 0).expect("source");
            encoder.set_texture(Some(&middle), 1).expect("middle");
            encoder
                .set_sampler_state(Some(&sampler), 0)
                .expect("sampler");
            let half_side = SIDE / 2;
            encoder
                .dispatch_threads([half_side, half_side, 1], [4, 4, 1])
                .unwrap_or_else(|error| panic!("{name}: sample_down: {error}"));
            // The middle texture, written above, is sampled next: the serial encoder orders them.
            encoder.set_compute_pipeline_state(&up).expect("up");
            encoder.set_texture(Some(&middle), 0).expect("middle");
            encoder.set_texture(Some(&output), 1).expect("output");
            encoder
                .dispatch_threads([SIDE, SIDE, 1], [4, 4, 1])
                .unwrap_or_else(|error| panic!("{name}: sample_up: {error}"));
            encoder.end_encoding();
            command_buffer.commit().expect("a commit");
            finished(name, &command_buffer);
            assert!(command_buffer.error().is_none(), "{name}");

            let mean = means();
            let got = texels(&middle, half_side);
            assert!(near(&got, &mean), "{name}: sample_down {got:?} {mean:?}");
            let upsampled = upsampled(&mean);
            let got = texels(&output, SIDE);
            assert!(
                near(&got, &upsampled),
                "{name}: sample_up {got:?} {upsampled:?}"
            );

            drop((command_buffer, queue, down, up));
            assert_eq!(counters.gauges().descriptor_slots, slots, "{name}");
            drop((sampler, source, middle, output));
        });
    }

    /// One texture class, one range: with the whole range granted, a texture of shader-read usage
    /// alone is made at index 2, below 4,096, and one of shader-write usage at 5,000, above it;
    /// one dispatch of `sample_down` samples the first and writes the second, and the second holds
    /// the linear filter's mean of the first.
    #[test]
    fn every_device_makes_a_texture_of_any_usage_at_any_index() {
        let Some(library) = library() else {
            return;
        };
        each_device("sampling_any_usage_at_any_index", |name, _, device| {
            let range = IndexClass::Texture.range();
            let _grants = [
                (IndexClass::Texture, range.start, range.end - range.start),
                (IndexClass::Sampler, 1, 1),
            ]
            .map(|(class, first, count)| {
                (device.grant_indices(class, first, count))
                    .unwrap_or_else(|error| panic!("{name}: {error}"))
            });
            let read = TextureUsage::SHADER_READ;
            let source = texture(&device, SIDE, read, 2);
            let written = texture(&device, SIDE / 2, read | TextureUsage::SHADER_WRITE, 5000);
            assert_eq!(
                [
                    source.gpu_resource_id().index,
                    written.gpu_resource_id().index
                ],
                [2, 5000],
                "{name}"
            );
            fill(name, &source);
            let sampler = linear(name, &device, 1);
            let down = down_state(name, &device, &library);
            let queue = device.new_command_queue(4).expect("a queue");
            sample_down(name, &queue, (&down, &sampler), [&source, &written]);
            let (got, want) = (texels(&written, SIDE / 2), means());
            assert!(near(&got, &want), "{name}: {got:?} {want:?}");
            drop((queue, down, sampler, source, written));
        });
    }

    /// Two textures reached only through their views (`Device::new_texture_for_views`), of
    /// Metal's unknown usage, each the one image under textures a consumer makes as its views:
    /// neither takes an index or a descriptor slot, and each answers index 0. The source's views
    /// at 4,096 and 4,097 (twins) and a view of the second at 4,098, and the target's at 1 and 2,
    /// each answer their own. The source's bytes go in through the source itself, and the target's
    /// come back through the target itself. One dispatch of `sample_down` samples the view at
    /// 4,096 and writes the target's view at 1; with the target zeroed through itself, a second
    /// samples the view of a view at 4,098 and writes the target's view at 2. Each time the target
    /// holds the linear filter's mean of the source. A texture of shader usage that
    /// `Device::new_texture` makes without an index, and a view of the source without one, are
    /// still `Refusal::TextureIndex`.
    #[test]
    fn every_device_samples_and_writes_a_texture_through_its_views() {
        let Some(library) = library() else {
            return;
        };
        each_device("sampling_through_views", |name, _, device| {
            let counters = device.counters();
            let _grants = [
                (IndexClass::Texture, 4096, 4),
                (IndexClass::Texture, 1, 4),
                (IndexClass::Sampler, 1, 1),
            ]
            .map(|(class, first, count)| {
                (device.grant_indices(class, first, count))
                    .unwrap_or_else(|error| panic!("{name}: {error}"))
            });
            let format = PixelFormat::RGBA16Float;
            let for_views = |side| {
                let mut d = TextureDescriptor::new(TextureType::Type2D, format, side, side);
                // No bits: Metal's unknown usage.
                d.usage = TextureUsage::empty();
                (device.new_texture_for_views(&d)).unwrap_or_else(|error| panic!("{name}: {error}"))
            };
            let slots = counters.gauges().descriptor_slots;
            let (source, target) = (for_views(SIDE), for_views(SIDE / 2));
            assert_eq!(counters.gauges().descriptor_slots, slots, "{name}: slots");
            let view = |texture: &Texture, index| {
                (texture.new_texture_view(format, TextureType::Type2D, 0..1, 0..1, Some(index)))
                    .unwrap_or_else(|error| panic!("{name}: a view at {index}: {error}"))
            };
            let (first, twin) = (view(&source, 4096), view(&source, 4097));
            let of_twin = view(&twin, 4098);
            let (written, written_twin) = (view(&target, 1), view(&target, 2));
            let views = [
                &source,
                &target,
                &first,
                &twin,
                &of_twin,
                &written,
                &written_twin,
            ];
            assert_eq!(
                views.map(|texture| texture.gpu_resource_id().index),
                [0, 0, 4096, 4097, 4098, 1, 2],
                "{name}: the indices"
            );
            fill(name, &source);
            let sampler = linear(name, &device, 1);
            let down = down_state(name, &device, &library);
            let queue = device.new_command_queue(4).expect("a queue");
            let want = means();
            sample_down(name, &queue, (&down, &sampler), [&first, &written]);
            let got = texels(&target, SIDE / 2);
            assert!(near(&got, &want), "{name}: 4096 into 1: {got:?} {want:?}");
            let half_side = SIDE / 2;
            let zeros = vec![0u8; (half_side * half_side * 8) as usize];
            let whole = Region {
                origin: [0; 3],
                size: [half_side, half_side, 1],
            };
            let (row, image) = (u64::from(half_side * 8), zeros.len() as u64);
            (target.replace_region(whole, 0, 0, crate::HostBytes::from(&zeros[..]), row, image))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let got = texels(&target, half_side);
            assert!(got.iter().all(|&v| v == 0.0), "{name}: zeroed {got:?}");
            sample_down(name, &queue, (&down, &sampler), [&of_twin, &written_twin]);
            let got = texels(&target, half_side);
            assert!(near(&got, &want), "{name}: 4098 into 2: {got:?} {want:?}");

            // A texture made with an index of its own, and a view, still take one where shaders
            // reach them.
            let mut d = TextureDescriptor::new(TextureType::Type2D, format, SIDE, SIDE);
            d.usage = TextureUsage::SHADER_READ;
            let unindexed = [
                device.new_texture(&d, None).map(drop),
                (source.new_texture_view(format, TextureType::Type2D, 0..1, 0..1, None)).map(drop),
            ];
            for result in unindexed {
                assert!(
                    matches!(
                        result,
                        Err(Error::Refused(Refusal::TextureIndex { index: None }))
                    ),
                    "{name}: {result:?}"
                );
            }
            drop((queue, down, sampler, first, twin, of_twin));
            drop((written, written_twin, source, target));
        });
    }

    /// The owner rule (`docs/RESOURCE-ID.md`), at one index: a texture of shader-write usage D,
    /// then one of shader-read usage alone A placed over it, whose placement empties D's storage
    /// element (a write through the index is dropped, and D keeps its texels); then one of
    /// shader-write usage B placed over A, and A goes, touching nothing. Sampling through the
    /// index reads B, and writing through it writes B. Once B goes, the index's elements are
    /// empty: sampling through D's ID, the same index, reads zero. The descriptor-slot gauge
    /// counts the index once throughout, and not after B.
    #[test]
    fn every_device_gives_an_index_to_the_texture_placed_there_last() {
        const AT: u32 = 3;
        let Some(library) = library() else {
            return;
        };
        each_device("sampling_index_placed_last", |name, _, device| {
            let counters = device.counters();
            let range = IndexClass::Texture.range();
            let _grants = [
                (IndexClass::Texture, range.start, range.end - range.start),
                (IndexClass::Sampler, 1, 1),
            ]
            .map(|(class, first, count)| {
                (device.grant_indices(class, first, count))
                    .unwrap_or_else(|error| panic!("{name}: {error}"))
            });
            let slots = counters.gauges().descriptor_slots;
            let read = TextureUsage::SHADER_READ;
            let write = read | TextureUsage::SHADER_WRITE;
            let source = texture(&device, SIDE, read, 5000);
            fill(name, &source);
            let target = texture(&device, SIDE / 2, write, range.end - 1);
            let sampler = linear(name, &device, 1);
            let down = down_state(name, &device, &library);
            let queue = device.new_command_queue(4).expect("a queue");
            let pass =
                |textures: [&Texture; 2]| sample_down(name, &queue, (&down, &sampler), textures);
            let pattern: Vec<f32> = (0..SIDE)
                .flat_map(|y| (0..SIDE).flat_map(move |x| texel(x, y)))
                .collect();

            let d = texture(&device, SIDE, write, AT);
            fill(name, &d);
            let a = texture(&device, SIDE, read, AT);
            assert_eq!(counters.gauges().descriptor_slots, slots + 4, "{name}");
            pass([&source, &a]);
            assert_eq!(
                texels(&d, SIDE),
                pattern,
                "{name}: D written through the index"
            );

            // B holds texels of its own, so a read through the index tells it from D and the
            // source.
            fn flat(_: u32, _: u32) -> [f32; 4] {
                [0.25, 0.5, 0.75, 1.0]
            }
            let b = texture(&device, SIDE, write, AT);
            fill_with(name, &b, flat);
            drop(a);
            assert_eq!(counters.gauges().descriptor_slots, slots + 4, "{name}");
            pass([&b, &target]);
            let half = SIDE / 2;
            let got = texels(&target, half);
            let flats: Vec<f32> = (0..half * half).flat_map(|_| flat(0, 0)).collect();
            assert!(
                near(&got, &flats),
                "{name}: sampled through the index: {got:?}"
            );
            // `sample_down` into a texture of the source's side samples each source texel at its
            // center, in the first half of each side.
            pass([&source, &b]);
            let want: Vec<f32> = (0..SIDE)
                .flat_map(|y| (0..SIDE).flat_map(move |x| (0..4).map(move |c| (x, y, c))))
                .map(|(x, y, c)| match x < half && y < half {
                    true => texel(x, y)[c as usize],
                    false => flat(x, y)[c as usize],
                })
                .collect();
            let got = texels(&b, SIDE);
            assert!(
                near(&got, &want),
                "{name}: written through the index: {got:?}"
            );

            drop(b);
            assert_eq!(counters.gauges().descriptor_slots, slots + 3, "{name}");
            pass([&d, &target]);
            let zero = vec![0.0; (half * half * 4) as usize];
            assert_eq!(texels(&target, half), zero, "{name}: sampled once B went");
            drop(d);
            assert_eq!(counters.gauges().descriptor_slots, slots + 3, "{name}");
            drop((queue, down, sampler, source, target));
        });
    }

    /// The owner rule at a constexpr sampler's index: a sampler state of nearest filter made at
    /// the index `sample_up`'s constexpr sampler (linear) holds takes it, and `sample_up` samples
    /// through the state's element; a later state that numbers the constexpr sampler there takes
    /// the index back, for every state that numbers it, and the sampler state's going touches
    /// nothing. The descriptor-slot gauge counts the index once throughout, and not after the
    /// last state that numbers the constexpr sampler.
    #[test]
    fn every_device_gives_a_constexpr_sampler_s_index_to_the_object_placed_there_last() {
        let Some(library) = library() else {
            return;
        };
        each_device("sampling_constexpr_index_placed_last", |name, _, device| {
            let counters = device.counters();
            let _grants = [IndexClass::Texture, IndexClass::Sampler].map(|class| {
                (device.grant_indices(class, 1, 4))
                    .unwrap_or_else(|error| panic!("{name}: {error}"))
            });
            let read = TextureUsage::SHADER_READ;
            let write = read | TextureUsage::SHADER_WRITE;
            let source = texture(&device, SIDE, read, 1);
            let middle = texture(&device, SIDE / 2, write, 2);
            let output = texture(&device, SIDE, write, 3);
            fill(name, &source);
            let sampler = linear(name, &device, 1);
            let down = down_state(name, &device, &library);
            let queue = device.new_command_queue(4).expect("a queue");
            sample_down(name, &queue, (&down, &sampler), [&source, &middle]);
            let mean = means();
            let linear_up = upsampled(&mean);
            // Nearest: each texel the middle texel it lies in.
            let half_side = SIDE / 2;
            let nearest_up: Vec<f32> = (0..SIDE)
                .flat_map(|y| (0..SIDE).flat_map(move |x| (0..4).map(move |c| (x, y, c))))
                .map(|(x, y, c)| mean[(4 * ((y / 2) * half_side + x / 2) + c) as usize])
                .collect();
            let up_pass =
                |up: &ComputePipelineState| sample_up(name, &queue, up, [&middle, &output]);

            let slots = counters.gauges().descriptor_slots;
            let up = up_state(name, &device, &library);
            assert_eq!(counters.gauges().descriptor_slots, slots + 1, "{name}");
            let got = up_pass(&up);
            assert!(
                near(&got, &linear_up),
                "{name}: the constexpr sampler {got:?}"
            );

            let nearest = (device.new_sampler_state(&SamplerDescriptor::new(), CONSTEXPR))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            assert_eq!(counters.gauges().descriptor_slots, slots + 1, "{name}");
            let got = up_pass(&up);
            assert!(
                near(&got, &nearest_up),
                "{name}: the sampler state made over it {got:?}"
            );

            let again = up_state(name, &device, &library);
            assert_eq!(counters.gauges().descriptor_slots, slots + 1, "{name}");
            let got = up_pass(&up);
            assert!(near(&got, &linear_up), "{name}: taken back {got:?}");
            drop(nearest);
            let got = up_pass(&again);
            assert!(
                near(&got, &linear_up),
                "{name}: once the sampler state went {got:?}"
            );
            drop((up, again));
            assert_eq!(counters.gauges().descriptor_slots, slots, "{name}");
            drop((queue, down, sampler, source, middle, output));
        });
    }

    /// The owner rule over objects the program released while work that used them is pending: a
    /// command buffer samples texture A through sampler state SA, signals once that is done, and
    /// waits on the host. A and SA are released, and texture B and sampler state SB are made at
    /// their indices at once, with the command buffer still pending; sampling through the indices
    /// on another queue reads B through SB. Once the command buffer has retired, and A and SA with
    /// it, sampling through the indices still reads B through SB. Nothing reads A or SA through
    /// the indices once B and SB take them: A's dispatch had completed (its signal), so the
    /// writes are valid under `UPDATE_UNUSED_WHILE_PENDING`. The descriptor-slot gauge counts each
    /// index once throughout.
    #[test]
    fn every_device_gives_a_released_object_s_index_to_the_next_at_once() {
        const AT: u32 = 3;
        let Some(library) = library() else {
            return;
        };
        each_device("sampling_released_index_next", |name, _, device| {
            let counters = device.counters();
            let _grants =
                [(IndexClass::Texture, 8), (IndexClass::Sampler, 1)].map(|(class, count)| {
                    (device.grant_indices(class, 1, count))
                        .unwrap_or_else(|error| panic!("{name}: {error}"))
                });
            let read = TextureUsage::SHADER_READ;
            let write = read | TextureUsage::SHADER_WRITE;
            fn flat(_: u32, _: u32) -> [f32; 4] {
                [0.5, 0.25, 0.75, 1.0]
            }
            let a = texture(&device, SIDE, read, AT);
            fill_with(name, &a, flat);
            let [first, second] = [5, 6].map(|index| texture(&device, SIDE / 2, write, index));
            let sa = linear(name, &device, 1);
            let down = down_state(name, &device, &library);
            let [queue, other] = [(); 2].map(|()| device.new_command_queue(4).expect("a queue"));
            let (signaled, release) = (shared_event(&device), shared_event(&device));
            let slots = counters.gauges().descriptor_slots;

            let pending = queue.command_buffer().expect("a command buffer");
            encode_down(name, &pending, (&down, &sa), [&a, &first]);
            (pending.encode_signal_event(signaled.as_event(), 1)).expect("a signal");
            (pending.encode_wait_for_event(release.as_event(), 1)).expect("a wait");
            pending.commit().expect("a commit");
            assert!(
                signaled
                    .wait_until_signaled_value(1, PATIENCE)
                    .expect("a wait outside a scheduled handler"),
                "{name}"
            );
            drop((a, sa));
            let b = texture(&device, SIDE, read, AT);
            fill(name, &b);
            let sb = linear(name, &device, 1);
            assert_eq!(counters.gauges().descriptor_slots, slots, "{name}");
            sample_down(name, &other, (&down, &sb), [&b, &second]);
            let status = pending.status();
            assert!(
                !matches!(
                    status,
                    CommandBufferStatus::Completed | CommandBufferStatus::Error
                ),
                "{name}: A's command buffer is pending: {status:?}"
            );
            let got = texels(&second, SIDE / 2);
            assert!(
                near(&got, &means()),
                "{name}: while A's work is pending {got:?}"
            );

            release.set_signaled_value(1);
            finished(name, &pending);
            assert!(pending.error().is_none(), "{name}");
            let want: Vec<f32> = (0..SIDE * SIDE / 4).flat_map(|_| flat(0, 0)).collect();
            let got = texels(&first, SIDE / 2);
            assert!(near(&got, &want), "{name}: A sampled {got:?}");
            assert_eq!(counters.gauges().descriptor_slots, slots, "{name}");
            sample_down(name, &queue, (&down, &sb), [&b, &first]);
            let got = texels(&first, SIDE / 2);
            assert!(
                near(&got, &means()),
                "{name}: once A's work retired {got:?}"
            );
            drop((b, sb));
            assert_eq!(counters.gauges().descriptor_slots, slots - 2, "{name}");
            drop((pending, queue, other, down, first, second));
        });
    }

    /// An equal sampler state made at the index of one the program released, while a command
    /// buffer that samples through that index is held from completing, writes no element of the
    /// sampler arrays and makes no device sampler: the command buffer waits on an event ahead of
    /// a dispatch that samples A through SA; SA is released and SB, of SA's descriptor, takes its
    /// index. Released, the held work samples through the index as SA would. A state of another
    /// descriptor at the index writes its elements.
    #[test]
    fn every_device_writes_no_element_for_an_equal_sampler_at_an_index_work_reads() {
        const AT: u32 = 3;
        let Some(library) = library() else {
            return;
        };
        each_device("sampling_equal_sampler_held", |name, _, device| {
            let counters = device.counters();
            let _grants =
                [(IndexClass::Texture, 8), (IndexClass::Sampler, 1)].map(|(class, count)| {
                    (device.grant_indices(class, 1, count))
                        .unwrap_or_else(|error| panic!("{name}: {error}"))
                });
            let read = TextureUsage::SHADER_READ;
            let a = texture(&device, SIDE, read, AT);
            fill(name, &a);
            let target = texture(&device, SIDE / 2, read | TextureUsage::SHADER_WRITE, 5);
            let sa = linear(name, &device, 1);
            let down = down_state(name, &device, &library);
            let queue = device.new_command_queue(4).expect("a queue");
            let release = shared_event(&device);
            let held = queue.command_buffer().expect("a command buffer");
            (held.encode_wait_for_event(release.as_event(), 1)).expect("a wait");
            encode_down(name, &held, (&down, &sa), [&a, &target]);
            held.commit().expect("a commit");
            let writes = counters.metrics().sampler_element_writes;
            let samplers = counters.gauges().device_samplers;
            drop(sa);
            let sb = linear(name, &device, 1);
            assert_eq!(
                counters.metrics().sampler_element_writes,
                writes,
                "{name}: an equal state at the index writes no element"
            );
            assert_eq!(counters.gauges().device_samplers, samplers, "{name}");
            let status = held.status();
            assert!(
                !matches!(
                    status,
                    CommandBufferStatus::Completed | CommandBufferStatus::Error
                ),
                "{name}: the command buffer is held: {status:?}"
            );
            release.set_signaled_value(1);
            finished(name, &held);
            assert!(held.error().is_none(), "{name}");
            let got = texels(&target, SIDE / 2);
            assert!(
                near(&got, &means()),
                "{name}: the held work sampled {got:?}"
            );
            // Another descriptor at the index writes its elements.
            let mut nearest = SamplerDescriptor::new();
            nearest.min_filter = SamplerMinMagFilter::Nearest;
            nearest.mag_filter = SamplerMinMagFilter::Nearest;
            let other = (device.new_sampler_state(&nearest, 1))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            assert!(
                counters.metrics().sampler_element_writes > writes,
                "{name}: another descriptor at the index writes its elements"
            );
            drop((sb, other, held, queue, down, a, target));
        });
    }

    /// A constexpr sampler past the device's samplers is refused, and the refusal lets go of what
    /// the state took: the same state is made once the samplers have room.
    #[test]
    fn every_device_refuses_a_constexpr_sampler_past_its_samplers() {
        let Some(library) = library() else {
            return;
        };
        each_device("sampling_constexpr_refused", |name, _, device| {
            let _grant = device
                .grant_indices(IndexClass::Sampler, 1, 4)
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let cap = lock(&device.core().samplers).set_cap(|held| held);
            let constexpr = |s: &[NumberedSampler]| {
                matches!(s, [NumberedSampler::Constexpr(_), NumberedSampler::Write])
            };
            let refused = state(
                &device,
                &library,
                "sample_up",
                constexpr,
                vec![CONSTEXPR, 0],
            )
            .err();
            assert!(
                matches!(refused, Some(Error::Limit(Limit::DeviceSamplers { .. }))),
                "{name}: {refused:?}"
            );
            lock(&device.core().samplers).set_cap(|_| cap);
            let up = state(
                &device,
                &library,
                "sample_up",
                constexpr,
                vec![CONSTEXPR, 0],
            )
            .unwrap_or_else(|error| panic!("{name}: sample_up: {error}"));
            drop(up);
        });
    }

    /// An image-only kernel that does not observe its threadgroup partition (`sample_down`) runs a
    /// grid one deep in 16x16x1 workgroups on the one `VkPipeline` its state made, testing its
    /// edge: over a grid of 20 by 12, ragged in both, a 32x32 destination holds the sample in the
    /// grid's texels and nothing past them, with no first-use wait. A grid two deep runs in
    /// workgroups of its tile on a `VkPipeline` made at that dispatch: one wait, the same
    /// texels. A column 256 deep, whose tile is 1 by 1 by 256, deeper than 64, runs its tile as
    /// one row of 256 threads on every device: one wait, its one texel. A grid eight deep whose
    /// dispatch `prepare_dispatch` prepared first finds its tile's `VkPipeline` made, the one the
    /// call made: no wait, the same texels.
    #[test]
    fn every_device_runs_an_image_only_kernel_in_square_workgroups() {
        let Some(library) = library() else {
            return;
        };
        each_device("square", |name, _, device| {
            let counters = device.counters();
            for (class, first) in [
                (IndexClass::Texture, 4096),
                (IndexClass::Texture, 1),
                (IndexClass::Sampler, 1),
            ] {
                device
                    .grant_indices(class, first, 4)
                    .unwrap_or_else(|error| panic!("{name}: {error}"));
            }
            let read = TextureUsage::SHADER_READ;
            let source = texture(&device, SIDE, read, 4096);
            const OUT: u32 = 32;
            let output = texture(&device, OUT, read | TextureUsage::SHADER_WRITE, 1);
            // One texel everywhere, so every linear sample is it exactly.
            let texel = [8.0, 4.0, 12.0, 16.0].map(|k: f32| k / 16.0);
            let fill = |texture: &Texture, side: u32, value: [f32; 4]| {
                let bytes: Vec<u8> = (0..side * side)
                    .flat_map(|_| value)
                    .flat_map(|v| half(v).to_le_bytes())
                    .collect();
                let whole = Region {
                    origin: [0; 3],
                    size: [side, side, 1],
                };
                texture
                    .replace_region(
                        whole,
                        0,
                        0,
                        crate::HostBytes::from(&bytes[..]),
                        u64::from(side * 8),
                        u64::from(side * side * 8),
                    )
                    .unwrap_or_else(|error| panic!("{name}: {error}"));
            };
            fill(&source, SIDE, texel);
            fill(&output, OUT, [0.0; 4]);
            let mut linear = SamplerDescriptor::new();
            linear.min_filter = SamplerMinMagFilter::Linear;
            linear.mag_filter = SamplerMinMagFilter::Linear;
            let sampler = device
                .new_sampler_state(&linear, 1)
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let made = counters.metrics();
            let down = state(
                &device,
                &library,
                "sample_down",
                |s| matches!(s, [NumberedSampler::Write]),
                vec![0],
            )
            .unwrap_or_else(|error| panic!("{name}: sample_down: {error}"));
            let created = counters.metrics();
            assert_eq!(
                created.pipelines_created - made.pipelines_created,
                1,
                "{name}: the square workgroups' pipeline, made with the state"
            );
            let queue = device.new_command_queue(4).expect("a queue");
            let runs = [
                ([20, 12], 1, 0, false),
                ([20, 12], 2, 1, false),
                ([1, 1], 256, 1, false),
                ([20, 12], 8, 0, true),
            ];
            for (grid, depth, waits, prepared) in runs {
                fill(&output, OUT, [0.0; 4]);
                if prepared {
                    let ahead = counters.metrics();
                    let threads = crate::Dispatch::Threads([grid[0], grid[1], depth]);
                    down.prepare_dispatch(threads, [4, 4, 1], &[]);
                    let made = counters.metrics().made_ahead_variants - ahead.made_ahead_variants;
                    assert_eq!(
                        made, 1,
                        "{name}: {depth} deep: the call made its tile's pipeline"
                    );
                }
                let before = counters.metrics();
                let command_buffer = queue.command_buffer().expect("a command buffer");
                let mut encoder = command_buffer
                    .compute_command_encoder(DispatchType::Serial)
                    .expect("an encoder");
                encoder.set_compute_pipeline_state(&down).expect("down");
                encoder.set_texture(Some(&source), 0).expect("source");
                encoder.set_texture(Some(&output), 1).expect("output");
                encoder
                    .set_sampler_state(Some(&sampler), 0)
                    .expect("sampler");
                encoder
                    .dispatch_threads([grid[0], grid[1], depth], [4, 4, 1])
                    .unwrap_or_else(|error| panic!("{name}: {depth} deep: {error}"));
                encoder.end_encoding();
                command_buffer.commit().expect("a commit");
                finished(name, &command_buffer);
                assert!(command_buffer.error().is_none(), "{name}: {depth} deep");
                let after = counters.metrics();
                assert_eq!(
                    after.first_use_waits - before.first_use_waits,
                    waits,
                    "{name}: {depth} deep"
                );
                let got = texels(&output, OUT);
                for y in 0..OUT {
                    for x in 0..OUT {
                        let at = (4 * (y * OUT + x)) as usize;
                        let inside = x < grid[0] && y < grid[1];
                        let want = if inside { texel } else { [0.0; 4] };
                        assert!(
                            near(&got[at..at + 4], &want),
                            "{name}: {depth} deep: texel {x}, {y}: {:?}",
                            &got[at..at + 4]
                        );
                    }
                }
            }
        });
    }
}
