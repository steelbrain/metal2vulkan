//! Metal's read-write texture tier 2 on the floor, under a device cfg on every device of the cfg's
//! kind: each of the 18 formats tier 2 names (tier 1's `R32Float`, `R32Uint` and `R32Sint`, and
//! the 15 it adds) is a texture of shader-read and shader-write usage that a kernel reads and
//! writes through `access::read_write`. Vulkan requires storage images of the four-channel and
//! the 32-bit formats on every device, and the floor's `shaderStorageImageExtendedFormats` the
//! six 8- and 16-bit single-channel ones, so the tier is a floor fact.
//!
//! The kernel reads each texel into a buffer, which checks the read's conversion and its missing
//! channels (0, 0 and 1), and writes back a function of it every channel moves: `1 - v` for a
//! float or normalized format, `v ^ 0x5a` for an integer one. A broken read, write or channel
//! order answers differently. The translator declares a format's storage image without a format
//! where the device reads and writes it so (`shaderStorageImageReadWithoutFormat`, or the format's
//! own bit), and with its format otherwise; the test runs whichever the device takes and says how
//! many formats took each.
//!
//! Beyond the tier, formats Metal does not promise a read of: each float format below of
//! shader-read and shader-write usage, bound to the float kernel, dispatches where the texture
//! has a storage element its device reads without a format (its view's format's
//! `VK_FORMAT_FEATURE_2_STORAGE_READ_WITHOUT_FORMAT_BIT`, the device's own answer), and is
//! `Gap::StorageElement` at the dispatch everywhere else: such a read is outside Vulkan's valid
//! usage (`VUID-vkCmdDispatch-OpTypeImage-07028`).
//!
//! `llvm-as` assembles the kernels; without it the tests skip visibly.

#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod on_device {
    use crate::tests::command::on_device::{each_device, finished, Arena};
    use crate::texture::TextureObject;
    use crate::{
        ComputePipelineDescriptor, ComputePipelineState, DispatchType, Error, FunctionDescriptor,
        Gap, IndexClass, Library, NumberedSampler, PixelFormat, ReadWriteTextureTier, Region,
        StageSource, TextureDescriptor, TextureType, TextureUsage,
    };
    use metal2vulkan::vk;
    use metal2vulkan_devtools::{llvm_assemble_text, AssembleError};

    /// The texture's side: 16 texels, one threadgroup of 4 by 4, and a read-back of 256 bytes.
    const SIDE: u32 = 4;

    /// What a format's channels hold.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum Kind {
        F32,
        F16,
        Unorm8,
        U32,
        U16,
        U8,
        S32,
        S16,
        S8,
    }

    impl Kind {
        /// The bytes one channel takes.
        fn bytes(self) -> usize {
            match self {
                Kind::F32 | Kind::U32 | Kind::S32 => 4,
                Kind::F16 | Kind::U16 | Kind::S16 => 2,
                Kind::Unorm8 | Kind::U8 | Kind::S8 => 1,
            }
        }

        /// The kernel of the format's class.
        fn kernel(self) -> &'static str {
            match self {
                Kind::F32 | Kind::F16 | Kind::Unorm8 => "rw_tier_f",
                Kind::U32 | Kind::U16 | Kind::U8 => "rw_tier_u",
                Kind::S32 | Kind::S16 | Kind::S8 => "rw_tier_s",
            }
        }
    }

    /// Tier 2's formats, tier 1's first: each with its channels' kind and count.
    const FORMATS: [(PixelFormat, Kind, usize); 18] = [
        (PixelFormat::R32Float, Kind::F32, 1),
        (PixelFormat::R32Uint, Kind::U32, 1),
        (PixelFormat::R32Sint, Kind::S32, 1),
        (PixelFormat::RGBA32Float, Kind::F32, 4),
        (PixelFormat::RGBA32Uint, Kind::U32, 4),
        (PixelFormat::RGBA32Sint, Kind::S32, 4),
        (PixelFormat::RGBA16Float, Kind::F16, 4),
        (PixelFormat::RGBA16Uint, Kind::U16, 4),
        (PixelFormat::RGBA16Sint, Kind::S16, 4),
        (PixelFormat::RGBA8Unorm, Kind::Unorm8, 4),
        (PixelFormat::RGBA8Uint, Kind::U8, 4),
        (PixelFormat::RGBA8Sint, Kind::S8, 4),
        (PixelFormat::R16Float, Kind::F16, 1),
        (PixelFormat::R16Uint, Kind::U16, 1),
        (PixelFormat::R16Sint, Kind::S16, 1),
        (PixelFormat::R8Unorm, Kind::Unorm8, 1),
        (PixelFormat::R8Uint, Kind::U8, 1),
        (PixelFormat::R8Sint, Kind::S8, 1),
    ];

    /// The mask an integer kernel writes its read through.
    const MASK: u32 = 0x5a;

    /// Three kernels over a `texture2d<T, access::read_write>` at texture 0 and a buffer of four
    /// words a texel at buffer 0, one per class: each reads texel `g` into `out[g.y * 4 + g.x]` and
    /// writes back `1 - v` (float) or `v ^ 0x5a` (integer).
    const KERNELS: &str = r#"target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

define void @rw_tier_f(ptr addrspace(1) %t, ptr addrspace(1) noundef "air-buffer-no-alias" %out, <2 x i32> noundef %g) {
entry:
  %s = tail call ptr addrspace(2) @air.get_read_sampler()
  %r = tail call { <4 x float>, i8 } @air.read_texture_2d.v4f32(ptr addrspace(1) %t, ptr addrspace(2) %s, <2 x i32> %g, <2 x i32> zeroinitializer, i32 0, i32 3)
  %v = extractvalue { <4 x float>, i8 } %r, 0
  %x = extractelement <2 x i32> %g, i64 0
  %y = extractelement <2 x i32> %g, i64 1
  %row = shl i32 %y, 2
  %i = add i32 %row, %x
  %at = zext i32 %i to i64
  %p = getelementptr inbounds <4 x float>, ptr addrspace(1) %out, i64 %at
  store <4 x float> %v, ptr addrspace(1) %p, align 16
  %w = fsub <4 x float> <float 1.0, float 1.0, float 1.0, float 1.0>, %v
  tail call void @air.write_texture_2d.v4f32(ptr addrspace(1) %t, <2 x i32> %g, <4 x float> %w, i32 0, i32 3)
  ret void
}

define void @rw_tier_u(ptr addrspace(1) %t, ptr addrspace(1) noundef "air-buffer-no-alias" %out, <2 x i32> noundef %g) {
entry:
  %s = tail call ptr addrspace(2) @air.get_read_sampler()
  %r = tail call { <4 x i32>, i8 } @air.read_texture_2d.u.v4i32(ptr addrspace(1) %t, ptr addrspace(2) %s, <2 x i32> %g, <2 x i32> zeroinitializer, i32 0, i32 3)
  %v = extractvalue { <4 x i32>, i8 } %r, 0
  %x = extractelement <2 x i32> %g, i64 0
  %y = extractelement <2 x i32> %g, i64 1
  %row = shl i32 %y, 2
  %i = add i32 %row, %x
  %at = zext i32 %i to i64
  %p = getelementptr inbounds <4 x i32>, ptr addrspace(1) %out, i64 %at
  store <4 x i32> %v, ptr addrspace(1) %p, align 16
  %w = xor <4 x i32> %v, <i32 90, i32 90, i32 90, i32 90>
  tail call void @air.write_texture_2d.u.v4i32(ptr addrspace(1) %t, <2 x i32> %g, <4 x i32> %w, i32 0, i32 3)
  ret void
}

define void @rw_tier_s(ptr addrspace(1) %t, ptr addrspace(1) noundef "air-buffer-no-alias" %out, <2 x i32> noundef %g) {
entry:
  %s = tail call ptr addrspace(2) @air.get_read_sampler()
  %r = tail call { <4 x i32>, i8 } @air.read_texture_2d.s.v4i32(ptr addrspace(1) %t, ptr addrspace(2) %s, <2 x i32> %g, <2 x i32> zeroinitializer, i32 0, i32 3)
  %v = extractvalue { <4 x i32>, i8 } %r, 0
  %x = extractelement <2 x i32> %g, i64 0
  %y = extractelement <2 x i32> %g, i64 1
  %row = shl i32 %y, 2
  %i = add i32 %row, %x
  %at = zext i32 %i to i64
  %p = getelementptr inbounds <4 x i32>, ptr addrspace(1) %out, i64 %at
  store <4 x i32> %v, ptr addrspace(1) %p, align 16
  %w = xor <4 x i32> %v, <i32 90, i32 90, i32 90, i32 90>
  tail call void @air.write_texture_2d.s.v4i32(ptr addrspace(1) %t, <2 x i32> %g, <4 x i32> %w, i32 0, i32 3)
  ret void
}

declare ptr addrspace(2) @air.get_read_sampler()
declare { <4 x float>, i8 } @air.read_texture_2d.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x i32>, <2 x i32>, i32, i32)
declare { <4 x i32>, i8 } @air.read_texture_2d.u.v4i32(ptr addrspace(1), ptr addrspace(2), <2 x i32>, <2 x i32>, i32, i32)
declare { <4 x i32>, i8 } @air.read_texture_2d.s.v4i32(ptr addrspace(1), ptr addrspace(2), <2 x i32>, <2 x i32>, i32, i32)
declare void @air.write_texture_2d.v4f32(ptr addrspace(1), <2 x i32>, <4 x float>, i32, i32)
declare void @air.write_texture_2d.u.v4i32(ptr addrspace(1), <2 x i32>, <4 x i32>, i32, i32)
declare void @air.write_texture_2d.s.v4i32(ptr addrspace(1), <2 x i32>, <4 x i32>, i32, i32)

!air.kernel = !{!0, !6, !10}
!air.version = !{!14}
!air.language_version = !{!15}
!0 = !{ptr @rw_tier_f, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5}
!3 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.arg_type_name", !"texture2d<float, read_write>", !"air.arg_name", !"t"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4", !"air.arg_name", !"out"}
!5 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint2", !"air.arg_name", !"g"}
!6 = !{ptr @rw_tier_u, !1, !7}
!7 = !{!8, !9, !5}
!8 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.arg_type_name", !"texture2d<uint, read_write>", !"air.arg_name", !"t"}
!9 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"uint4", !"air.arg_name", !"out"}
!10 = !{ptr @rw_tier_s, !1, !11}
!11 = !{!12, !13, !5}
!12 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.arg_type_name", !"texture2d<int, read_write>", !"air.arg_name", !"t"}
!13 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"int4", !"air.arg_name", !"out"}
!14 = !{i32 2, i32 8, i32 0}
!15 = !{!"Metal", i32 4, i32 0, i32 0}
"#;

    /// The value channel `channel` of texel `texel` holds, as the code the format stores: a
    /// float's or half's of `k / 256`, so `1 - v` is exact in both; a normalized byte's `k`; an
    /// integer's a value whose mask stays in the format's range.
    fn code(kind: Kind, texel: usize, channel: usize) -> u32 {
        let n = (texel * 4 + channel) as u32;
        match kind {
            Kind::F32 | Kind::F16 | Kind::Unorm8 | Kind::U8 => (n * 37 + 11) % 256,
            Kind::S8 => ((n * 37 + 11) % 256) as u8 as i8 as i32 as u32,
            Kind::U16 => (n * 4099 + 517) % 65_536,
            Kind::S16 => ((n * 4099 + 517) % 65_536) as u16 as i16 as i32 as u32,
            Kind::U32 | Kind::S32 => n.wrapping_mul(0x0123_4567).wrapping_add(0x89ab),
        }
    }

    /// The bits of the half of `value`, a multiple of 1/256 in [0, 1], which a half holds exactly.
    fn half_bits(value: f32) -> u16 {
        if value == 0.0 {
            return 0;
        }
        let bits = value.to_bits();
        let exponent = ((bits >> 23) & 0xff) as i32 - 127 + 15;
        ((exponent as u16) << 10) | ((bits >> 13) & 0x3ff) as u16
    }

    /// The bytes one channel stores for `code`.
    fn stored(kind: Kind, code: u32) -> Vec<u8> {
        match kind {
            Kind::F32 => (code as f32 / 256.0).to_le_bytes().to_vec(),
            Kind::F16 => half_bits(code as f32 / 256.0).to_le_bytes().to_vec(),
            Kind::U32 | Kind::S32 => code.to_le_bytes().to_vec(),
            Kind::U16 | Kind::S16 => (code as u16).to_le_bytes().to_vec(),
            Kind::Unorm8 | Kind::U8 | Kind::S8 => vec![code as u8],
        }
    }

    /// The word the kernel's read stores for `code`, as the read converts it.
    fn read_word(kind: Kind, code: u32) -> u32 {
        match kind {
            Kind::F32 | Kind::F16 => (code as f32 / 256.0).to_bits(),
            Kind::Unorm8 => (code as f32 / 255.0).to_bits(),
            _ => code,
        }
    }

    /// The code the kernel's write leaves for `code`.
    fn written(kind: Kind, code: u32) -> u32 {
        match kind {
            Kind::F32 | Kind::F16 => 256 - code,
            Kind::Unorm8 => 255 - code,
            _ => code ^ MASK,
        }
    }

    /// The library, or None where `llvm-as` is missing.
    fn library() -> Option<Library> {
        match llvm_assemble_text(KERNELS) {
            Ok(bitcode) => Some(Library::from_air(&bitcode).expect("the library")),
            Err(AssembleError::Missing(why)) => {
                eprintln!("SKIPPED: {why}");
                None
            }
            Err(error) => panic!("the read-write tier kernels: {error:?}"),
        }
    }

    #[test]
    fn every_device_reads_and_writes_each_format_of_read_write_tier_2() {
        let Some(library) = library() else {
            return;
        };
        each_device("read_write_tier", |name, _, device| {
            assert_eq!(
                device.read_write_texture_support(),
                ReadWriteTextureTier::Tier2,
                "{name}"
            );
            let profile = device.core().translator.profile();
            let states = |entry: &str| -> ComputePipelineState {
                let function = library.function(entry).expect("the kernel");
                let descriptor = ComputePipelineDescriptor::new(StageSource::Function(
                    FunctionDescriptor::new(function),
                ));
                // The texture's read and its write number a sampler each; sampler 0 serves both.
                (device.new_compute_pipeline_state(&descriptor, |samplers| {
                    assert_eq!(
                        samplers,
                        [NumberedSampler::Read, NumberedSampler::Write],
                        "{name}"
                    );
                    Ok(vec![0; samplers.len()])
                }))
                .unwrap_or_else(|error| panic!("{name}: {entry}: {error}"))
            };
            let kernels =
                ["rw_tier_f", "rw_tier_u", "rw_tier_s"].map(|entry| (entry, states(entry)));
            let grant = (device.grant_indices(IndexClass::Texture, 1, FORMATS.len() as u32))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let queue = device.new_command_queue(1).expect("a queue");
            let mut arena = Arena::new(&device);
            let (mut without_format, mut declared) = (0, 0);
            for (at, &(format, kind, channels)) in FORMATS.iter().enumerate() {
                let choice = metal2vulkan::metal::format::storage_image_choice(format, profile)
                    .unwrap_or_else(|error| panic!("{name}: {format:?}: {error}"));
                if choice.vk_format == vk::Format::UNDEFINED {
                    without_format += 1;
                } else {
                    declared += 1;
                }
                let texel_bytes = kind.bytes() * channels;
                let codes: Vec<Vec<u32>> = (0..(SIDE * SIDE) as usize)
                    .map(|texel| {
                        (0..channels)
                            .map(|channel| code(kind, texel, channel))
                            .collect()
                    })
                    .collect();
                let bytes: Vec<u8> = (codes.iter().flatten())
                    .flat_map(|&code| stored(kind, code))
                    .collect();
                let mut d = TextureDescriptor::new(TextureType::Type2D, format, SIDE, SIDE);
                d.usage = TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE;
                let texture = (device.new_texture(&d, Some(1 + at as u32)))
                    .unwrap_or_else(|error| panic!("{name}: {format:?}: {error}"));
                let whole = Region {
                    origin: [0; 3],
                    size: [SIDE, SIDE, 1],
                };
                let row = (SIDE as usize * texel_bytes) as u64;
                (texture.replace_region(
                    whole,
                    0,
                    0,
                    crate::HostBytes::from(&bytes[..]),
                    row,
                    row * u64::from(SIDE),
                ))
                .unwrap_or_else(|error| panic!("{name}: {format:?}: {error}"));
                let out = arena.buffer();
                let state = &(kernels.iter())
                    .find(|(entry, _)| *entry == kind.kernel())
                    .expect("the class's kernel")
                    .1;
                let command_buffer = queue.command_buffer().expect("a command buffer");
                let mut encoder = (command_buffer.compute_command_encoder(DispatchType::Serial))
                    .expect("an encoder");
                encoder
                    .set_compute_pipeline_state(state)
                    .expect("the state");
                encoder.set_texture(Some(&texture), 0).expect("the texture");
                encoder.set_buffer(Some(&out), 0, 0).expect("out");
                (encoder.dispatch_threadgroups([1, 1, 1], [SIDE, SIDE, 1]))
                    .unwrap_or_else(|error| panic!("{name}: {format:?}: {error}"));
                encoder.end_encoding();
                command_buffer.commit().expect("a commit");
                finished(name, &command_buffer);
                assert_eq!(command_buffer.error(), None, "{name}: {format:?}");
                let contents = out.contents().expect("a Shared buffer");
                // SAFETY: the dispatch that wrote the buffer's 256 bytes completed, and its writes
                // reach the host.
                let words =
                    unsafe { std::slice::from_raw_parts(contents.as_ptr().cast::<u32>(), 4 * 16) };
                let mut after = vec![0u8; bytes.len()];
                (texture.get_bytes(&mut after, row, row * u64::from(SIDE), whole, 0, 0))
                    .unwrap_or_else(|error| panic!("{name}: {format:?}: {error}"));
                let one = match kind {
                    Kind::F32 | Kind::F16 | Kind::Unorm8 => 1.0f32.to_bits(),
                    _ => 1,
                };
                for (texel, codes) in codes.iter().enumerate() {
                    let read = &words[texel * 4..texel * 4 + 4];
                    for (channel, &word) in read.iter().enumerate() {
                        let want = match codes.get(channel) {
                            Some(&code) => read_word(kind, code),
                            None if channel == 3 => one,
                            None => 0,
                        };
                        // A normalized byte's read is within a rounding of its code over 255.
                        let same = match (kind, codes.get(channel)) {
                            (Kind::Unorm8, Some(&code)) => {
                                (f32::from_bits(word) * 255.0).round() as u32 == code
                            }
                            _ => word == want,
                        };
                        assert!(
                            same,
                            "{name}: {format:?}: texel {texel} channel {channel}: read {word:#x}, \
                             Metal's {want:#x}"
                        );
                    }
                    let first = texel * texel_bytes;
                    let held = &after[first..first + texel_bytes];
                    for (channel, &code) in codes.iter().enumerate() {
                        let at = channel * kind.bytes();
                        let field = &held[at..at + kind.bytes()];
                        let want = stored(kind, written(kind, code));
                        // Vulkan's float-to-normalized conversion takes either integer beside a
                        // value that is not one, and `1 - code / 255` need not be exact.
                        let same = match kind {
                            Kind::Unorm8 => field[0].abs_diff(want[0]) <= 1,
                            _ => field == &want[..],
                        };
                        assert!(
                            same,
                            "{name}: {format:?}: texel {texel} channel {channel}: wrote \
                             {field:?}, Metal's {want:?}"
                        );
                    }
                }
                drop((texture, command_buffer));
            }
            (device.release_indices(grant)).unwrap_or_else(|error| panic!("{name}: {error}"));
            println!(
                "read_write_tier: {name}: the 18 formats read and write, {without_format} \
                 declared without a format, {declared} with it"
            );
        });
    }

    /// Float formats outside the tiers that a device may store: those whose storage reads without
    /// a format some device lacks (ANV before Gfx12.5: the first four), and four it has.
    const BEYOND: [PixelFormat; 8] = [
        PixelFormat::BGRA8Unorm,
        PixelFormat::BGR10A2Unorm,
        PixelFormat::RG11B10Float,
        PixelFormat::RGB10A2Unorm,
        PixelFormat::RGBA16Unorm,
        PixelFormat::RG16Float,
        PixelFormat::RG8Unorm,
        PixelFormat::RG32Float,
    ];

    #[test]
    fn every_device_refuses_a_storage_read_its_device_cannot_make_without_a_format() {
        let Some(library) = library() else {
            return;
        };
        each_device("read_write_beyond", |name, _, device| {
            let function = library.function("rw_tier_f").expect("the kernel");
            let descriptor = ComputePipelineDescriptor::new(StageSource::Function(
                FunctionDescriptor::new(function),
            ));
            let state = (device
                .new_compute_pipeline_state(&descriptor, |samplers| Ok(vec![0; samplers.len()])))
            .unwrap_or_else(|error| panic!("{name}: {error}"));
            let grant = (device.grant_indices(IndexClass::Texture, 1, BEYOND.len() as u32))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let queue = device.new_command_queue(1).expect("a queue");
            let mut arena = Arena::new(&device);
            let (mut read, mut refused) = (Vec::new(), Vec::new());
            for (at, &format) in BEYOND.iter().enumerate() {
                let mut d = TextureDescriptor::new(TextureType::Type2D, format, SIDE, SIDE);
                d.usage = TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE;
                let texture = match device.new_texture(&d, Some(1 + at as u32)) {
                    Ok(texture) => texture,
                    // A format the device holds no texture of is another gap's.
                    Err(Error::Unsupported(Gap::TextureFormat { .. })) => continue,
                    Err(error) => panic!("{name}: {format:?}: {error}"),
                };
                let (_, optimal) = device.core().format_features(texture.format());
                let reads = texture.unstored().is_none()
                    && optimal.contains(ash::vk::FormatFeatureFlags2::STORAGE_READ_WITHOUT_FORMAT);
                let out = arena.buffer();
                let command_buffer = queue.command_buffer().expect("a command buffer");
                let mut encoder = (command_buffer.compute_command_encoder(DispatchType::Serial))
                    .expect("an encoder");
                encoder
                    .set_compute_pipeline_state(&state)
                    .expect("the state");
                encoder.set_texture(Some(&texture), 0).expect("the texture");
                encoder.set_buffer(Some(&out), 0, 0).expect("out");
                let dispatched = encoder.dispatch_threadgroups([1, 1, 1], [SIDE, SIDE, 1]);
                encoder.end_encoding();
                command_buffer.commit().expect("a commit");
                finished(name, &command_buffer);
                assert_eq!(command_buffer.error(), None, "{name}: {format:?}");
                match (reads, dispatched) {
                    (true, Ok(())) => read.push(format),
                    (
                        false,
                        Err(Error::Unsupported(Gap::StorageElement {
                            slot: Some(0),
                            pixel_format,
                        })),
                    ) if pixel_format == format => refused.push(format),
                    (reads, other) => panic!(
                        "{name}: {format:?}, its storage read without a format {reads}: {other:?}"
                    ),
                }
                drop((texture, command_buffer));
            }
            (device.release_indices(grant)).unwrap_or_else(|error| panic!("{name}: {error}"));
            println!("read_write_beyond: {name}: read {read:?}; refused {refused:?}");
        });
    }
}
