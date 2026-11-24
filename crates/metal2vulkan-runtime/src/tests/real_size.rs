//! Textures at a picture's real size, every texel compared, and what reads them.
//!
//! Device-free: the colour pattern a real-size test writes ([`unorm_pattern`]) names its texel,
//! so a read of the wrong texel shows, and its complement ([`complement`], written before each
//! write as the stale baseline) differs from it in every channel.
//!
//! Under a device cfg, the readers the real-size tests share (`on_device`): two owned kernels
//! that read every texel of a 2D float texture, one through `read()` and one by sampling (nearest,
//! normalized, at the texel's centre), each packing the texel's channels into a word, one thread
//! a texel in threadgroups of 16x16; a blit of the whole texture into a buffer; and the per-cell
//! report, one line a cell, the count of wrong texels and the first, and how many are right and
//! where the first of them are. Each read writes into a buffer first filled with the complement
//! of the bytes it should read, so a word nothing wrote is wrong. A driver that runs one thread
//! a workgroup leaves 255 of every 256 words of each kernel read as filled, and fails.
//!
//! On every device of the cfg's kind, a picture's planes as rt's own textures at their real size
//! (`every_device_reads_real_size_planes_of_its_own_every_texel`): a 16-bit picture (luma
//! `R16Unorm` at 3840x2160, chroma `RG16Unorm` at 1920x1080) and an 8-bit one (`R8Unorm`,
//! `RG8Unorm`), each plane written by a blit of its complement and then of its pattern
//! ([`plane_word`]), then read by sampling, by a kernel's `read()`, by a blit out, and by
//! sampling again, every texel compared. The Direct3D 11 planes of the same pictures are
//! `d3d11.rs`'s.
//!
//! `llvm-as` assembles the kernels; without it the tests skip visibly.

/// The agreed pattern's channels at `(x, y)`, each below 4,096, in the order a shader reads
/// them (red, green, blue, alpha), for colour channels of `bits` (8 or 10) and an alpha of
/// `alpha_bits`: red `x`, green `y`, blue the bits of both above `bits`, alpha `x ^ y`, each cut
/// to its channel. Red, green and blue together name the texel.
pub(crate) fn unorm_pattern(bits: u32, alpha_bits: u32, x: u32, y: u32) -> [u32; 4] {
    let low = (1 << bits) - 1;
    [
        x & low,
        y & low,
        (x >> bits) | ((y >> bits) << (12 - bits)),
        (x ^ y) & ((1 << alpha_bits) - 1),
    ]
}

/// The stale baseline of a channel of `bits` holding `value`: its complement in the channel.
pub(crate) fn complement(bits: u32, value: u32) -> u32 {
    (1 << bits) - 1 - value
}

/// The agreed pattern names its texel at 8 and at 10 bits a colour channel, through the whole of
/// each axis below 4,096 (every `x` along rows of several `y`, every `y` along columns of several
/// `x`), and its complement differs from it in every channel.
#[test]
fn the_agreed_pattern_names_its_texel_and_its_complement_differs_in_every_channel() {
    let some = [
        0, 1, 255, 256, 1023, 1024, 1079, 1599, 2159, 2844, 3839, 4095,
    ];
    for (bits, alpha_bits) in [(8, 8), (10, 2)] {
        let cells = (0..4096)
            .flat_map(|x| some.map(|y| (x, y)))
            .chain(some.iter().flat_map(|&x| (0..4096).map(move |y| (x, y))));
        for (x, y) in cells {
            let [r, g, b, a] = unorm_pattern(bits, alpha_bits, x, y);
            let named = (
                r | ((b & ((1 << (12 - bits)) - 1)) << bits),
                g | ((b >> (12 - bits)) << bits),
            );
            assert_eq!(named, (x, y), "{bits} bits: ({x}, {y})");
            for (channel, value) in [r, g, b, a].into_iter().enumerate() {
                let width = if channel == 3 { alpha_bits } else { bits };
                assert!(
                    value < 1 << width,
                    "{bits} bits: ({x}, {y}) channel {channel}"
                );
                assert_ne!(complement(width, value), value, "{bits} bits: ({x}, {y})");
            }
        }
    }
}

#[cfg(any(m2v_gpu, m2v_cpu_icd))]
pub(crate) mod on_device {
    use crate::tests::command::on_device::{each_device, finished};
    use crate::{
        Buffer, BufferLayout, CommandBuffer, CommandQueue, ComputePipelineDescriptor,
        ComputePipelineState, Device, DispatchType, Error, FunctionDescriptor, IndexClass, Library,
        MemoryClass, PixelFormat, SamplerDescriptor, SamplerState, StageSource, StorageMode,
        Texture, TextureDescriptor, TexturePlace, TextureType,
    };
    use metal2vulkan_devtools::{llvm_assemble_text, AssembleError};

    /// The owned kernels: `packed_read` reads texel `g` of texture 0 through `read()`,
    /// `packed_sample` samples it through sampler 0 at its centre; each stores at `g.y * width +
    /// g.x` of buffer 0 the word that ORs, for each channel `c`, `(uint(v[c] * scale[c] + 0.5) <<
    /// shift[c]) & mask[c]`. Buffer 1 holds `uint4 {width, height, 0, 0}`, then `float4 scale`,
    /// `uint4 shift` and `uint4 mask`; threads past the width or height store nothing.
    const KERNELS: &str = r#"; Owned synthetic kernels.
source_filename = "real_size.metal"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

define void @packed_read(ptr addrspace(1) %t, ptr addrspace(1) noundef "air-buffer-no-alias" %out, ptr addrspace(2) noundef "air-buffer-no-alias" %p, <2 x i32> noundef %g) {
entry:
  %size = load <4 x i32>, ptr addrspace(2) %p, align 16
  %w = extractelement <4 x i32> %size, i64 0
  %h = extractelement <4 x i32> %size, i64 1
  %x = extractelement <2 x i32> %g, i64 0
  %y = extractelement <2 x i32> %g, i64 1
  %ix = icmp ult i32 %x, %w
  %iy = icmp ult i32 %y, %h
  %in = and i1 %ix, %iy
  br i1 %in, label %body, label %done

body:
  %s = tail call ptr addrspace(2) @air.get_read_sampler()
  %r = tail call { <4 x float>, i8 } @air.read_texture_2d.v4f32(ptr addrspace(1) %t, ptr addrspace(2) %s, <2 x i32> %g, <2 x i32> zeroinitializer, i32 0, i32 1)
  %v = extractvalue { <4 x float>, i8 } %r, 0
  %ps = getelementptr inbounds <4 x i32>, ptr addrspace(2) %p, i64 1
  %scale.bits = load <4 x i32>, ptr addrspace(2) %ps, align 16
  %scale = bitcast <4 x i32> %scale.bits to <4 x float>
  %pt = getelementptr inbounds <4 x i32>, ptr addrspace(2) %p, i64 2
  %shift = load <4 x i32>, ptr addrspace(2) %pt, align 16
  %pm = getelementptr inbounds <4 x i32>, ptr addrspace(2) %p, i64 3
  %mask = load <4 x i32>, ptr addrspace(2) %pm, align 16
  %sv = fmul <4 x float> %v, %scale
  %hv = fadd <4 x float> %sv, <float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01>
  %u = fptoui <4 x float> %hv to <4 x i32>
  %sh = shl <4 x i32> %u, %shift
  %m = and <4 x i32> %sh, %mask
  %c0 = extractelement <4 x i32> %m, i64 0
  %c1 = extractelement <4 x i32> %m, i64 1
  %c2 = extractelement <4 x i32> %m, i64 2
  %c3 = extractelement <4 x i32> %m, i64 3
  %o1 = or i32 %c0, %c1
  %o2 = or i32 %o1, %c2
  %word = or i32 %o2, %c3
  %row = mul i32 %y, %w
  %i = add i32 %row, %x
  %at = zext i32 %i to i64
  %q = getelementptr inbounds i32, ptr addrspace(1) %out, i64 %at
  store i32 %word, ptr addrspace(1) %q, align 4
  br label %done

done:
  ret void
}

define void @packed_sample(ptr addrspace(1) %t, ptr addrspace(2) %smp, ptr addrspace(1) noundef "air-buffer-no-alias" %out, ptr addrspace(2) noundef "air-buffer-no-alias" %p, <2 x i32> noundef %g) {
entry:
  %size = load <4 x i32>, ptr addrspace(2) %p, align 16
  %w = extractelement <4 x i32> %size, i64 0
  %h = extractelement <4 x i32> %size, i64 1
  %x = extractelement <2 x i32> %g, i64 0
  %y = extractelement <2 x i32> %g, i64 1
  %ix = icmp ult i32 %x, %w
  %iy = icmp ult i32 %y, %h
  %in = and i1 %ix, %iy
  br i1 %in, label %body, label %done

body:
  %wh = shufflevector <4 x i32> %size, <4 x i32> poison, <2 x i32> <i32 0, i32 1>
  %whf = tail call <2 x float> @air.convert.f.v2f32.u.v2i32(<2 x i32> %wh)
  %gf = tail call <2 x float> @air.convert.f.v2f32.u.v2i32(<2 x i32> %g)
  %c = fadd <2 x float> %gf, <float 5.000000e-01, float 5.000000e-01>
  %uv = fdiv <2 x float> %c, %whf
  %r = tail call { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) %t, ptr addrspace(2) %smp, <2 x float> %uv, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  %v = extractvalue { <4 x float>, i8 } %r, 0
  %ps = getelementptr inbounds <4 x i32>, ptr addrspace(2) %p, i64 1
  %scale.bits = load <4 x i32>, ptr addrspace(2) %ps, align 16
  %scale = bitcast <4 x i32> %scale.bits to <4 x float>
  %pt = getelementptr inbounds <4 x i32>, ptr addrspace(2) %p, i64 2
  %shift = load <4 x i32>, ptr addrspace(2) %pt, align 16
  %pm = getelementptr inbounds <4 x i32>, ptr addrspace(2) %p, i64 3
  %mask = load <4 x i32>, ptr addrspace(2) %pm, align 16
  %sv = fmul <4 x float> %v, %scale
  %hv = fadd <4 x float> %sv, <float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01>
  %u = fptoui <4 x float> %hv to <4 x i32>
  %sh = shl <4 x i32> %u, %shift
  %m = and <4 x i32> %sh, %mask
  %c0 = extractelement <4 x i32> %m, i64 0
  %c1 = extractelement <4 x i32> %m, i64 1
  %c2 = extractelement <4 x i32> %m, i64 2
  %c3 = extractelement <4 x i32> %m, i64 3
  %o1 = or i32 %c0, %c1
  %o2 = or i32 %o1, %c2
  %word = or i32 %o2, %c3
  %row = mul i32 %y, %w
  %i = add i32 %row, %x
  %at = zext i32 %i to i64
  %q = getelementptr inbounds i32, ptr addrspace(1) %out, i64 %at
  store i32 %word, ptr addrspace(1) %q, align 4
  br label %done

done:
  ret void
}

declare ptr addrspace(2) @air.get_read_sampler()
declare { <4 x float>, i8 } @air.read_texture_2d.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x i32>, <2 x i32>, i32, i32)
declare { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i1, float, float, i32)
declare <2 x float> @air.convert.f.v2f32.u.v2i32(<2 x i32>)

!air.kernel = !{!0, !6}
!air.version = !{!12}
!air.language_version = !{!13}
!0 = !{ptr @packed_read, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5, !11}
!3 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.read", !"air.arg_type_name", !"texture2d<float, read>", !"air.arg_name", !"t"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
!5 = !{i32 2, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 2, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"uint4", !"air.arg_name", !"p"}
!6 = !{ptr @packed_sample, !1, !7}
!7 = !{!8, !9, !10, !14, !15}
!8 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"t"}
!9 = !{i32 1, !"air.sampler", !"air.location_index", i32 0, i32 1, !"air.arg_type_name", !"sampler", !"air.arg_name", !"smp"}
!10 = !{i32 2, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
!11 = !{i32 3, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint2", !"air.arg_name", !"g"}
!12 = !{i32 2, i32 8, i32 0}
!13 = !{!"Metal", i32 4, i32 0, i32 0}
!14 = !{i32 3, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 2, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"uint4", !"air.arg_name", !"p"}
!15 = !{i32 4, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint2", !"air.arg_name", !"g"}
"#;

    /// The kernels' library, or None where `llvm-as` is missing.
    pub(crate) fn library() -> Option<Library> {
        match llvm_assemble_text(KERNELS) {
            Ok(bitcode) => Some(Library::from_air(&bitcode).expect("the library")),
            Err(AssembleError::Missing(why)) => {
                eprintln!("SKIPPED: {why}");
                None
            }
            Err(error) => panic!("the real-size kernels: {error:?}"),
        }
    }

    /// The state of `function`, a kernel of `library`, on `device`, its pipeline finished.
    pub(crate) fn kernel(
        name: &str,
        device: &Device,
        library: &Library,
        function: &str,
    ) -> ComputePipelineState {
        let function_handle = library.function(function).expect("the kernel");
        let descriptor = ComputePipelineDescriptor::new(StageSource::Function(
            FunctionDescriptor::new(function_handle),
        ));
        let state = (device
            .new_compute_pipeline_state(&descriptor, crate::tests::no_constexpr_samplers))
        .unwrap_or_else(|error| panic!("{name}: {function}: {error}"));
        (state.finish_pipeline()).unwrap_or_else(|error| panic!("{name}: {function}: {error}"));
        state
    }

    /// The two readers of [`KERNELS`] on a device, and the nearest sampler `packed_sample` reads
    /// through, at the first index of a sampler grant this makes.
    pub(crate) struct Readers {
        pub(crate) read: ComputePipelineState,
        pub(crate) sample: ComputePipelineState,
        pub(crate) sampler: SamplerState,
    }

    impl Readers {
        pub(crate) fn new(name: &str, device: &Device, library: &Library) -> Readers {
            let first = IndexClass::Sampler.range().start;
            (device.grant_indices(IndexClass::Sampler, first, 1))
                .unwrap_or_else(|error| panic!("{name}: a sampler grant: {error}"));
            let sampler = (device.new_sampler_state(&SamplerDescriptor::new(), first))
                .unwrap_or_else(|error| panic!("{name}: the sampler: {error}"));
            Readers {
                read: kernel(name, device, library, "packed_read"),
                sample: kernel(name, device, library, "packed_sample"),
                sampler,
            }
        }
    }

    /// How a reader packs a texel's channels into its word ([`KERNELS`]): channel `c` as
    /// `uint(v * scale[c] + 0.5) << shift[c]`, masked by `mask[c]`.
    #[derive(Clone, Copy, Debug)]
    pub(crate) struct Pack {
        pub(crate) scale: [f32; 4],
        pub(crate) shift: [u32; 4],
        pub(crate) mask: [u32; 4],
    }

    impl Pack {
        /// The first `channels` channels, each normalized of `bits`, packed from bit 0 up.
        pub(crate) fn unorm(bits: u32, channels: usize) -> Pack {
            let max = (1_u32 << bits) - 1;
            let packed = |c: usize| c < channels;
            Pack {
                scale: std::array::from_fn(|c| if packed(c) { max as f32 } else { 0.0 }),
                shift: std::array::from_fn(|c| if packed(c) { c as u32 * bits } else { 0 }),
                mask: std::array::from_fn(|c| {
                    if packed(c) {
                        max << (c as u32 * bits)
                    } else {
                        0
                    }
                }),
            }
        }

        /// Buffer 1's bytes for a texture of `[width, height]`.
        fn params(&self, [width, height]: [u32; 2]) -> Vec<u8> {
            [width, height, 0, 0]
                .into_iter()
                .chain(self.scale.map(f32::to_bits))
                .chain(self.shift)
                .chain(self.mask)
                .flat_map(u32::to_le_bytes)
                .collect()
        }
    }

    /// An error's text, with its variant.
    fn said(error: Error) -> String {
        format!("{error} ({error:?})")
    }

    /// A Shared buffer the length of `bytes` holding them, or their complement where
    /// `complemented`, and its block.
    fn filled(
        device: &Device,
        bytes: &[u8],
        complemented: bool,
    ) -> Result<(Buffer, crate::Block), String> {
        let len = bytes.len() as u64;
        let block = (device.take_block(MemoryClass::Shared, len)).map_err(said)?;
        let buffer = (device.new_buffer(block.id, 0, len, None)).map_err(said)?;
        let contents = buffer.contents().expect("a Shared buffer");
        // SAFETY: the buffer's bytes, which no command reaches yet and nothing else borrows.
        let held = unsafe { std::slice::from_raw_parts_mut(contents.as_ptr(), bytes.len()) };
        let flip = if complemented { 0xff } else { 0 };
        for (byte, from) in held.iter_mut().zip(bytes) {
            *byte = from ^ flip;
        }
        Ok((buffer, block))
    }

    /// Commits `command_buffer` and waits for it: what it failed with, if it did.
    fn completed(name: &str, command_buffer: &CommandBuffer) -> Result<(), String> {
        command_buffer.commit().map_err(said)?;
        finished(name, command_buffer);
        match command_buffer.error() {
            None => Ok(()),
            Some(error) => Err(format!("the command buffer: {error:?}")),
        }
    }

    /// The bytes of `buffer` (of `block`) once its command buffer completed, the block then
    /// released.
    fn taken(device: &Device, (buffer, block): (Buffer, crate::Block)) -> Vec<u8> {
        let len = buffer.length() as usize;
        let contents = buffer.contents().expect("a Shared buffer");
        // SAFETY: the buffer's bytes, which no pending command reaches.
        let bytes = unsafe { std::slice::from_raw_parts(contents.as_ptr(), len) }.to_vec();
        drop(buffer);
        (device.release_block(block)).expect("a release");
        bytes
    }

    /// Every texel of `texture`, of `size`, read by `state` (a kernel with [`KERNELS`]'
    /// bindings, sampling through `sampler` where one is given) packed by `pack`, into a buffer
    /// the length of `want` first filled with its complement: the buffer's bytes, or what refused
    /// or failed.
    pub(crate) fn kernel_read(
        name: &str,
        (device, queue): (&Device, &CommandQueue),
        (state, sampler): (&ComputePipelineState, Option<&SamplerState>),
        (texture, size): (&Texture, [u32; 2]),
        pack: &Pack,
        want: &[u8],
    ) -> Result<Vec<u8>, String> {
        let out = filled(device, want, true)?;
        let command_buffer = queue.command_buffer().map_err(said)?;
        let mut encoder =
            (command_buffer.compute_command_encoder(DispatchType::Serial)).map_err(said)?;
        encoder.set_compute_pipeline_state(state).map_err(said)?;
        encoder.set_texture(Some(texture), 0).map_err(said)?;
        if let Some(sampler) = sampler {
            encoder.set_sampler_state(Some(sampler), 0).map_err(said)?;
        }
        encoder.set_buffer(Some(&out.0), 0, 0).map_err(said)?;
        encoder.set_bytes(&pack.params(size), 1).map_err(said)?;
        (encoder.dispatch_threads([size[0], size[1], 1], [16, 16, 1])).map_err(said)?;
        encoder.end_encoding();
        completed(name, &command_buffer)?;
        Ok(taken(device, out))
    }

    /// The sampled read of `texture` ([`kernel_read`] through `packed_sample`).
    pub(crate) fn sampled(
        name: &str,
        on: (&Device, &CommandQueue),
        readers: &Readers,
        texture: (&Texture, [u32; 2]),
        pack: &Pack,
        want: &[u8],
    ) -> Result<Vec<u8>, String> {
        let through = (&readers.sample, Some(&readers.sampler));
        kernel_read(name, on, through, texture, pack, want)
    }

    /// The `read()` of `texture` ([`kernel_read`] through `packed_read`).
    pub(crate) fn read(
        name: &str,
        on: (&Device, &CommandQueue),
        readers: &Readers,
        texture: (&Texture, [u32; 2]),
        pack: &Pack,
        want: &[u8],
    ) -> Result<Vec<u8>, String> {
        kernel_read(name, on, (&readers.read, None), texture, pack, want)
    }

    /// Every texel of `texture`, of `size` and `texel` bytes a texel, copied out by a blit into a
    /// buffer of packed rows first filled with `want`'s complement: its bytes, or what refused or
    /// failed.
    pub(crate) fn copy_out(
        name: &str,
        (device, queue): (&Device, &CommandQueue),
        (texture, size): (&Texture, [u32; 2]),
        texel: u32,
        want: &[u8],
    ) -> Result<Vec<u8>, String> {
        let out = filled(device, want, true)?;
        let command_buffer = queue.command_buffer().map_err(said)?;
        let mut encoder = command_buffer.blit_command_encoder().map_err(said)?;
        let layout = BufferLayout {
            offset: 0,
            bytes_per_row: u64::from(size[0] * texel),
            bytes_per_image: 0,
        };
        let whole = [size[0], size[1], 1];
        (encoder.copy_from_texture_to_buffer(
            texture,
            TexturePlace::default(),
            whole,
            &out.0,
            layout,
        ))
        .map_err(said)?;
        encoder.end_encoding();
        completed(name, &command_buffer)?;
        Ok(taken(device, out))
    }

    /// Writes `bytes`, packed rows of `size` texels of `texel` bytes, into `texture` by a blit
    /// from a Shared buffer, and waits for it.
    pub(crate) fn copy_in(
        name: &str,
        (device, queue): (&Device, &CommandQueue),
        (texture, size): (&Texture, [u32; 2]),
        texel: u32,
        bytes: &[u8],
    ) -> Result<(), String> {
        let from = filled(device, bytes, false)?;
        let command_buffer = queue.command_buffer().map_err(said)?;
        let mut encoder = command_buffer.blit_command_encoder().map_err(said)?;
        let layout = BufferLayout {
            offset: 0,
            bytes_per_row: u64::from(size[0] * texel),
            bytes_per_image: 0,
        };
        let whole = [size[0], size[1], 1];
        (encoder.copy_from_buffer(&from.0, layout, whole, texture, TexturePlace::default()))
            .map_err(said)?;
        encoder.end_encoding();
        completed(name, &command_buffer)?;
        taken(device, from);
        Ok(())
    }

    /// How `got` differs from `want`, texels of `texel` bytes in rows of `width`: None where
    /// they are equal, else how many differ, the first, how many are right and where the first of
    /// those are.
    pub(crate) fn differs(got: &[u8], want: &[u8], texel: usize, width: u32) -> Option<String> {
        if got == want {
            return None;
        }
        if got.len() != want.len() {
            return Some(format!("{} bytes, not {}", got.len(), want.len()));
        }
        let at = |i: usize| (i as u32 % width, i as u32 / width);
        let (mut wrong, mut right, mut first, mut rights) = (0, 0, None, Vec::new());
        for (i, (g, w)) in (got.chunks_exact(texel).zip(want.chunks_exact(texel))).enumerate() {
            if g == w {
                right += 1;
                if rights.len() < 6 {
                    rights.push(at(i));
                }
            } else {
                wrong += 1;
                if first.is_none() {
                    first = Some((at(i), g, w));
                }
            }
        }
        let ((x, y), g, w) = first.expect("a texel differs");
        Some(format!(
            "{wrong} of {} texels wrong, the first at ({x}, {y}): {g:02x?}, not {w:02x?}; {right} \
             right, the first at {rights:?}",
            wrong + right
        ))
    }

    /// A real-size test's cells on one device: each printed as it reads, and the ones not exact
    /// kept for the end.
    pub(crate) struct Cells {
        name: String,
        failed: Vec<String>,
    }

    impl Cells {
        pub(crate) fn new(name: &str) -> Cells {
            Cells {
                name: name.to_owned(),
                failed: Vec::new(),
            }
        }

        /// Cell `what`: `read` against `want`, texels of `texel` bytes in rows of `width`.
        pub(crate) fn check(
            &mut self,
            what: &str,
            read: Result<Vec<u8>, String>,
            want: &[u8],
            (texel, width): (usize, u32),
        ) {
            let line = match read {
                Ok(got) => match differs(&got, want, texel, width) {
                    None => "exact".to_owned(),
                    Some(how) => format!("WRONG: {how}"),
                },
                Err(why) => format!("REFUSED: {why}"),
            };
            println!("real-size: {}: {what}: {line}", self.name);
            if line != "exact" {
                self.failed.push(format!("{what}: {line}"));
            }
        }

        /// Fails where a cell was not exact, naming each.
        pub(crate) fn all_exact(self) {
            assert!(
                self.failed.is_empty(),
                "{}: {} cells not exact:\n{}",
                self.name,
                self.failed.len(),
                self.failed.join("\n")
            );
        }
    }

    /// One plane of a picture: its size, its image's format and texel bytes, the format rt
    /// views it as, and that view's channels and their bits.
    pub(crate) struct Plane {
        pub(crate) what: &'static str,
        pub(crate) size: [u32; 2],
        pub(crate) image: PixelFormat,
        pub(crate) texel: u32,
        pub(crate) view: PixelFormat,
        pub(crate) channels: usize,
        pub(crate) bits: u32,
    }

    /// The two pictures, each a luma and a chroma plane, the chroma at half the luma's size on
    /// each axis: 16-bit (`R16Unorm`; an `R32Uint` image viewed `RG16Unorm`) and 8-bit
    /// (`R8Unorm`; an `R16Uint` image viewed `RG8Unorm`), each at a 3840x2160 picture's size.
    pub(crate) const PICTURES: [[Plane; 2]; 2] = [
        [
            Plane {
                what: "luma R16Unorm 3840x2160",
                size: [3840, 2160],
                image: PixelFormat::R16Unorm,
                texel: 2,
                view: PixelFormat::R16Unorm,
                channels: 1,
                bits: 16,
            },
            Plane {
                what: "chroma R32Uint 1920x1080 as RG16Unorm",
                size: [1920, 1080],
                image: PixelFormat::R32Uint,
                texel: 4,
                view: PixelFormat::RG16Unorm,
                channels: 2,
                bits: 16,
            },
        ],
        [
            Plane {
                what: "luma R8Unorm 3840x2160",
                size: [3840, 2160],
                image: PixelFormat::R8Unorm,
                texel: 1,
                view: PixelFormat::R8Unorm,
                channels: 1,
                bits: 8,
            },
            Plane {
                what: "chroma R16Uint 1920x1080 as RG8Unorm",
                size: [1920, 1080],
                image: PixelFormat::R16Uint,
                texel: 2,
                view: PixelFormat::RG8Unorm,
                channels: 2,
                bits: 8,
            },
        ],
    ];

    /// The position hash the planes' words are cut from (as Direct3D 11's pixel shaders compute
    /// it in `d3d11.rs`): two odd multipliers' products XORed, then mixed.
    pub(crate) fn hash(x: u32, y: u32) -> u32 {
        let mut v = x.wrapping_mul(0x9e37_79b1) ^ y.wrapping_mul(0x85eb_ca77);
        v ^= v >> 15;
        v = v.wrapping_mul(0x2c1b_3c6d);
        v ^ (v >> 12)
    }

    impl Plane {
        /// The texel's bits: all ones.
        pub(crate) fn mask(&self) -> u32 {
            u32::MAX >> (32 - 8 * self.texel)
        }

        /// The word the plane's image holds at `(x, y)`: the low bytes of [`hash`], complemented
        /// where `complemented` (the stale baseline).
        pub(crate) fn word(&self, x: u32, y: u32, complemented: bool) -> u32 {
            let word = hash(x, y) & self.mask();
            match complemented {
                true => !word & self.mask(),
                false => word,
            }
        }

        /// Every texel's word, rows from the top, each by `bytes` of it: the image's own bytes
        /// (`texel`) or a reader's word (4, which packs the view's channels into the image's
        /// word, its first channel lowest, as the image holds them).
        pub(crate) fn words(&self, complemented: bool, bytes: u32) -> Vec<u8> {
            let [width, height] = self.size;
            (0..height)
                .flat_map(|y| (0..width).map(move |x| (x, y)))
                .flat_map(|(x, y)| {
                    let word = self.word(x, y, complemented).to_le_bytes();
                    word.into_iter().take(bytes as usize)
                })
                .collect()
        }

        /// How a reader packs the view's channels.
        pub(crate) fn pack(&self) -> Pack {
            Pack::unorm(self.bits, self.channels)
        }
    }

    /// A picture's planes as rt's own textures, each written by a blit of its complement and then
    /// of its pattern, then read by sampling, by `read()`, by a blit out and by sampling again,
    /// every texel compared.
    #[test]
    fn every_device_reads_real_size_planes_of_its_own_every_texel() {
        let Some(library) = library() else {
            return;
        };
        each_device("real_size_planes", |name, _, device| {
            let readers = Readers::new(name, &device, &library);
            let first = IndexClass::Texture.range().start;
            (device.grant_indices(IndexClass::Texture, first, 4))
                .unwrap_or_else(|error| panic!("{name}: a texture grant: {error}"));
            let queue = device.new_command_queue(4).expect("a queue");
            let on = (&device, &queue);
            let mut cells = Cells::new(name);
            for (index, plane) in (first..).zip(PICTURES.iter().flatten()) {
                let mut d = TextureDescriptor::new(
                    TextureType::Type2D,
                    plane.view,
                    plane.size[0],
                    plane.size[1],
                );
                d.storage_mode = StorageMode::Private;
                let texture = (device.new_texture(&d, Some(index)))
                    .unwrap_or_else(|error| panic!("{name}: {}: {error}", plane.what));
                let at = (&texture, plane.size);
                let (texel, width) = (plane.texel as usize, plane.size[0]);
                let image = plane.words(false, plane.texel);
                let words = plane.words(false, 4);
                let pack = plane.pack();
                let wrote = copy_in(name, on, at, plane.texel, &plane.words(true, plane.texel))
                    .and_then(|()| copy_in(name, on, at, plane.texel, &image));
                if let Err(why) = wrote {
                    cells.check(
                        &format!("{}: the blits in", plane.what),
                        Err(why),
                        &[],
                        (1, 1),
                    );
                    continue;
                }
                let cell = |how: &str| format!("rt's own {}, written by blit: {how}", plane.what);
                let got = sampled(name, on, &readers, at, &pack, &words);
                cells.check(&cell("sampled"), got, &words, (4, width));
                let got = read(name, on, &readers, at, &pack, &words);
                cells.check(&cell("read()"), got, &words, (4, width));
                let got = copy_out(name, on, at, plane.texel, &image);
                cells.check(&cell("copied out"), got, &image, (texel, width));
                let got = sampled(name, on, &readers, at, &pack, &words);
                cells.check(&cell("sampled again"), got, &words, (4, width));
            }
            cells.all_exact();
        });
    }
}
