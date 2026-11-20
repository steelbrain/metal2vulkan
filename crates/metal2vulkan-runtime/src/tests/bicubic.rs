//! A kernel that samples 2D textures through a bicubic constexpr sampler at normalized
//! coordinates (the owned fixture `engine/bcu_formats`, ten slots, the last two with `half`
//! results), under a device cfg on every device of the cfg's kind. Each dispatch pushes, for each
//! slot, how Metal's bicubic filter treats the format bound there, which selects the filter's arm
//! at run time: the linear inner taps for `R32Float`, a clamp to [0, 1] for `RGBA8Unorm`,
//! `RGBA16Unorm` and `R8Unorm`, a clamp to [-1, 1] for `RGBA8Snorm`, and no clamp for
//! `RGBA16Float`. Each format is bound at every slot over six dispatches; a slot bound to nothing
//! pushes the unbound word. A dispatch whose slot holds an `RGBA8Unorm_sRGB` texture is refused,
//! naming the slot and the format. The command buffer is never committed.
//!
//! `llvm-as` assembles the kernel; without it the test skips visibly.

#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod on_device {
    use crate::pipeline::{FieldValue, Place};
    use crate::tests::command::on_device::each_device;
    use crate::tests::draw::on_device::Buffers;
    use crate::{
        ComputePipelineDescriptor, DispatchType, Error, FunctionDescriptor, Gap, IndexClass,
        Library, NumberedSampler, PixelFormat, StageSource, Texture, TextureDescriptor,
        TextureType, TextureUsage,
    };
    use metal2vulkan::metal::format::{bicubic_texels_word, BICUBIC_TEXELS_UNBOUND};
    use metal2vulkan::metal::TextureSwizzleChannels;
    use metal2vulkan_devtools::{engine_dir, llvm_assemble, AssembleError};

    /// The kernel's texture slots.
    const SLOTS: usize = 10;
    /// The constexpr sampler's index the test gives the kernel.
    const CONSTEXPR: u32 = 1;

    /// The filter's arm, the word's low four bits (`metal::bicubic::BicubicTexels`): the linear
    /// inner taps, a clamp to [0, 1], a clamp to [-1, 1], and no clamp.
    const LINEAR: u32 = 1 << 0;
    const UNORM: u32 = 1 << 1;
    const SNORM: u32 = 1 << 2;
    const HALF: u32 = 1 << 3;

    /// Each format the test binds, with its arm.
    const ARMS: [(PixelFormat, u32); 6] = [
        (PixelFormat::R32Float, LINEAR),
        (PixelFormat::RGBA8Unorm, UNORM),
        (PixelFormat::RGBA8Snorm, SNORM),
        (PixelFormat::RGBA16Float, HALF),
        (PixelFormat::RGBA16Unorm, UNORM),
        (PixelFormat::R8Unorm, UNORM),
    ];

    /// A 2D texture of `format`, 4 texels a side, of shader-read usage at `index`.
    fn texture(name: &str, device: &crate::Device, format: PixelFormat, index: u32) -> Texture {
        let mut d = TextureDescriptor::new(TextureType::Type2D, format, 4, 4);
        d.usage = TextureUsage::SHADER_READ;
        (device.new_texture(&d, Some(index))).unwrap_or_else(|error| panic!("{name}: {error}"))
    }

    #[test]
    fn every_device_pushes_each_bicubic_slots_format_treatment() {
        let library = match llvm_assemble(&engine_dir().join("bcu_formats.ll")) {
            Ok(bitcode) => Library::from_air(&bitcode).expect("the library"),
            Err(AssembleError::Missing(why)) => {
                eprintln!("SKIPPED: {why}");
                return;
            }
            Err(error) => panic!("bcu_formats.ll: {error:?}"),
        };
        let function = library.function("bcu_formats").expect("the kernel");
        let descriptor = ComputePipelineDescriptor::new(StageSource::Function(
            FunctionDescriptor::new(function),
        ));
        each_device("bicubic_texels", |name, _, device| {
            let _grants = [
                (IndexClass::Texture, 1, ARMS.len() as u32 + 1),
                (IndexClass::Sampler, CONSTEXPR, 1),
            ]
            .map(|(class, first, count)| {
                (device.grant_indices(class, first, count))
                    .unwrap_or_else(|error| panic!("{name}: {error}"))
            });
            let state = device
                .new_compute_pipeline_state(&descriptor, |samplers| {
                    assert!(
                        matches!(samplers, [NumberedSampler::Constexpr(_)]),
                        "{name}: {samplers:?}"
                    );
                    Ok(vec![CONSTEXPR])
                })
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            // Each slot's field, in the push window.
            let mut at = [None; SLOTS];
            for field in &state.inner.compiled.whole.interface.fields {
                if let FieldValue::BicubicTexels(slot) = field.value {
                    let Place::Push(offset) = field.place else {
                        panic!("{name}: slot {slot}'s field in a parameter block");
                    };
                    assert!(
                        at[slot as usize].replace(offset as usize).is_none(),
                        "{name}"
                    );
                }
            }
            let at = at.map(|at| at.unwrap_or_else(|| panic!("{name}: a slot with no field")));
            let textures: Vec<Texture> = (ARMS.iter().zip(1..))
                .map(|(&(format, _), index)| texture(name, &device, format, index))
                .collect();
            let srgb = texture(
                name,
                &device,
                PixelFormat::RGBA8UnormSRGB,
                ARMS.len() as u32 + 1,
            );
            let mut buffers = Buffers::new(&device);
            let (coords, out) = (
                buffers.of(&device, &[0; 64]),
                buffers.of(&device, &[0; 4096]),
            );
            let queue = device.new_command_queue(1).expect("a queue");
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let mut encoder = (command_buffer.compute_command_encoder(DispatchType::Serial))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let call = |result: Result<(), Error>| result.unwrap_or_else(|e| panic!("{name}: {e}"));
            call(encoder.set_compute_pipeline_state(&state));
            call(encoder.set_buffer(Some(&coords), 0, 0));
            call(encoder.set_buffer(Some(&out), 0, 1));
            // Dispatch `d` binds format `(d + k) mod 6` at slot `k`, and nothing at the last slot
            // in the first dispatch.
            for d in 0..ARMS.len() {
                for (slot, k) in (0..SLOTS).zip(d..) {
                    let texture = (d, slot) != (0, SLOTS - 1);
                    let texture = texture.then(|| &textures[k % ARMS.len()]);
                    call(encoder.set_texture(texture, slot as u32));
                }
                call(encoder.dispatch_threads([4, 1, 1], [4, 1, 1]));
                let push = encoder.pushed().to_vec();
                for (slot, k) in (0..SLOTS).zip(d..) {
                    let word = u32::from_le_bytes(push[at[slot]..at[slot] + 4].try_into().unwrap());
                    if (d, slot) == (0, SLOTS - 1) {
                        assert_eq!(word, BICUBIC_TEXELS_UNBOUND, "{name}: slot {slot}, unbound");
                        continue;
                    }
                    let (format, arm) = ARMS[k % ARMS.len()];
                    let want = bicubic_texels_word(format, TextureSwizzleChannels::DEFAULT);
                    assert_eq!(
                        (word & 0xf, Some(word)),
                        (arm, want),
                        "{name}: slot {slot}, {format:?}"
                    );
                }
            }
            // An sRGB texture at slot 3 is refused, the slot and the format named.
            call(encoder.set_texture(Some(&srgb), 3));
            let refused = encoder.dispatch_threads([4, 1, 1], [4, 1, 1]);
            let Err(Error::Unsupported(gap)) = refused else {
                panic!("{name}: an sRGB texture's bicubic sample: {refused:?}");
            };
            assert_eq!(
                gap,
                Gap::BicubicTexels {
                    slot: 3,
                    pixel_format: PixelFormat::RGBA8UnormSRGB
                },
                "{name}"
            );
            let text = gap.to_string();
            assert!(
                text.contains("texture slot 3") && text.contains("RGBA8UnormSRGB"),
                "{name}: {text}"
            );
            encoder.end_encoding();
        });
    }
}
