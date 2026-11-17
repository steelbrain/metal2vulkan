//! Kernels that sample textures, under a device cfg on every device of the cfg's kind: the
//! harness's `sample_down`, through a runtime sampler state, and `sample_up`, through a constexpr
//! sampler at the index the consumer gives it, chained in one serial encoder so the second samples
//! what the first wrote. Each answer is checked against the linear filter's, and the constexpr
//! sampler's index is shared between the states that number it there and freed with the last.
//!
//! `llvm-as` assembles the kernels; without it the test skips visibly.

#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod on_device {
    use crate::tests::command::on_device::{each_device, finished};
    use crate::{
        ComputePipelineDescriptor, ComputePipelineState, Device, DispatchType, Error,
        FunctionDescriptor, IndexClass, Library, NumberedSampler, PixelFormat, Refusal, Region,
        SamplerDescriptor, SamplerMinMagFilter, StageSource, Texture, TextureDescriptor,
        TextureType, TextureUsage,
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
        each_device("sampling", |name, _, device| {
            let counters = device.counters();
            let grants = [
                (IndexClass::Texture, 4096),
                (IndexClass::StorageTexture, 1),
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
            // Each texel's channels, in sixteenths a half holds exactly.
            let texel = |x: u32, y: u32| [x, y, x + y, 16].map(|k| k as f32 / 16.0);
            let written: Vec<f32> = (0..SIDE)
                .flat_map(|y| (0..SIDE).flat_map(move |x| texel(x, y)))
                .collect();
            let bytes: Vec<u8> = written
                .iter()
                .flat_map(|&v| half(v).to_le_bytes())
                .collect();
            let whole = Region {
                origin: [0; 3],
                size: [SIDE, SIDE, 1],
            };
            source
                .replace_region(
                    whole,
                    0,
                    0,
                    &bytes,
                    u64::from(SIDE * 8),
                    u64::from(SIDE * SIDE * 8),
                )
                .unwrap_or_else(|error| panic!("{name}: {error}"));
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
            // A second state that numbers it at the same index shares it; a sampler state there,
            // or an answer of another count, is refused.
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
                    device.new_sampler_state(&linear, CONSTEXPR),
                    Err(Error::Refused(Refusal::IndexInUse {
                        class: IndexClass::Sampler,
                        index: CONSTEXPR
                    }))
                ),
                "{name}"
            );
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

            // Down: each texel the mean of its four source texels.
            let at = |x: u32, y: u32, c: usize| texel(x, y)[c];
            let mean: Vec<f32> = (0..half_side)
                .flat_map(|y| (0..half_side).flat_map(move |x| (0..4).map(move |c| (x, y, c))))
                .map(|(x, y, c)| {
                    let (x, y) = (2 * x, 2 * y);
                    (at(x, y, c) + at(x + 1, y, c) + at(x, y + 1, c) + at(x + 1, y + 1, c)) / 4.0
                })
                .collect();
            let got = texels(&middle, half_side);
            assert!(near(&got, &mean), "{name}: sample_down {got:?} {mean:?}");
            // Up: each texel a quarter of a middle texel from a middle texel's center, clamped to
            // the edge.
            let taps = |x: u32| {
                let k = x / 2;
                match x % 2 {
                    0 => (k.saturating_sub(1), k, 0.75f32),
                    _ => (k, (k + 1).min(half_side - 1), 0.25f32),
                }
            };
            let mid = |x: u32, y: u32, c: u32| mean[(4 * (y * half_side + x) + c) as usize];
            let upsampled: Vec<f32> = (0..SIDE)
                .flat_map(|y| (0..SIDE).flat_map(move |x| (0..4).map(move |c| (x, y, c))))
                .map(|(x, y, c)| {
                    let ((x0, x1, wx), (y0, y1, wy)) = (taps(x), taps(y));
                    let row = |y| mid(x0, y, c) * (1.0 - wx) + mid(x1, y, c) * wx;
                    row(y0) * (1.0 - wy) + row(y1) * wy
                })
                .collect();
            let got = texels(&output, SIDE);
            assert!(
                near(&got, &upsampled),
                "{name}: sample_up {got:?} {upsampled:?}"
            );

            drop((command_buffer, queue, down, up));
            assert_eq!(counters.gauges().descriptor_slots, slots, "{name}");
            drop((sampler, source, middle, output));
            for grant in grants {
                device
                    .release_indices(grant)
                    .unwrap_or_else(|error| panic!("{name}: {error}"));
            }
        });
    }
}
