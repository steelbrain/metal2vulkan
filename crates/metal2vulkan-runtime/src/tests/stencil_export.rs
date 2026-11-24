//! A fragment function's `[[stencil]]` output, the reference of both faces' stencil test for its
//! fragment. Where the device lists `VK_EXT_shader_stencil_export` (`floor`'s
//! `OPTIONAL_EXTENSIONS`), device creation enables it and the function writes
//! `FragStencilRefEXT`; elsewhere each draw takes the draws of `depth_stencil::BitDraws`
//! (`RuntimeFact::StencilBitDraws`), counted in `Metrics::stencil_bit_draws`.
//!
//! Without a device: the draws `BitDraws` takes for each kind of depth-stencil state.
//!
//! Under a device cfg, on every device of the cfg's kind, each draw a quad of two triangles over
//! an 8 x 8 target, the encoder's reference `0x5a`, both faces `Always` and `Replace` under write
//! mask `0xff`:
//!
//! - the stencil alone (`stx_stencil_by_pixel`, a `Stencil8` attachment cleared to `0x3c`): every
//!   texel the low 8 bits of its fragment's output, the texels whose output's low bits are 0
//!   among them;
//! - the stencil and the depth (`stx_stencil_with_depth`, a `Depth32Float_Stencil8` attachment
//!   cleared to `0x3c` and 0.5): under a depth compare of `Always` that writes, every texel its
//!   fragment's stencil and depth, a texel whose output's low bits are 0 holding what the first
//!   draw alone wrote; under `Less` that writes, with a depth failure of `Invert`, a fragment that
//!   passes stores its stencil and its depth, and one that fails keeps 0.5 and inverts the stencil;
//! - the counter: 8 draws more for each of those draws, 9 under `Less`, on a device without the
//!   extension, and 0 on one with it;
//! - two triangles of one draw over one half of the target (`overlap_stencil`, outputs `0x35`,
//!   then `0x4a`): the half one fragment reaches stores `0x35`; the half both reach stores the
//!   later `0x4a` with the extension and their OR, `0x7f`, without it
//!   (`PrecisionNote::StencilOutputOverlap`), the row printed;
//! - refused without the extension, made and drawn with it: a function that writes a colour
//!   beside its stencil (`stx_stencil_ms_color`, at the state's creation, naming
//!   `StencilExport`), a draw under a stencil compare of `Equal` and one in a `Counting` visibility
//!   result's segment (`DrawKind::StencilOutput`); a draw in a `Boolean` segment is drawn.
//!
//! And the route a guest's GL layer takes to copy a depth and stencil texture by drawing: a
//! 4-sample `Depth32Float_Stencil8` source, each quadrant's stencil and each pixel's depth drawn
//! there, read at sample 0 by a function writing `[[depth(any)]]` and `[[stencil]]`
//! (`blit_stencil`) into a single-sample target through a triangle strip of 4 vertices, under
//! both faces `Always` and `Replace` under `0xff` and a depth compare of `Always` that writes,
//! beside a colour attachment 0 of `RGBA8Unorm` whose write mask is none and blending off: each
//! texel the source's stencil and depth, the colour as its clear left it, and 8 draws more
//! without the extension.
//!
//! `colored_vertex` (the runtime's examples) draws the geometry; `overlap_stencil` and
//! `blit_stencil` are owned AIR text ([`STENCIL_FRAGMENTS`]). `llvm-as` assembles the functions;
//! without it the tests skip visibly.

use crate::depth_stencil::{BitDraw, BitDraws, Tests, EVERY_FRAGMENT};
use ash::vk;
use metal2vulkan::metal::state::DepthStencilDescriptor;
use metal2vulkan::metal::{CompareFunction, StencilOperation};

/// Both faces of `d` replace on a pass.
fn replacing(mut d: DepthStencilDescriptor) -> DepthStencilDescriptor {
    for face in [&mut d.front_face_stencil, &mut d.back_face_stencil] {
        face.depth_stencil_pass_operation = StencilOperation::Replace;
    }
    d
}

/// A draw takes itself alone where its state replaces no bit, a zeroing draw and a draw per bit
/// its replacing faces' write masks hold where it does, and a last draw that writes the depth
/// where the state writes it under a compare that tests; none under a compare the reference
/// decides.
#[test]
fn a_draw_by_bits_takes_a_draw_for_each_bit_its_state_can_replace() {
    let reference = [0x5a, 0x5a];
    let draws = |d: &DepthStencilDescriptor| BitDraws::of(Tests::of(d), reference);
    let own = |d: &DepthStencilDescriptor| BitDraw {
        tests: Tests::of(d),
        reference,
        bit: EVERY_FRAGMENT,
    };
    // No stencil test, and a `Replace` under a write mask of 0: the draw alone.
    let plain = DepthStencilDescriptor::default();
    let alone = draws(&plain).expect("no stencil test");
    assert!(alone.draws() == [own(&plain)]);
    let mut unwritten = replacing(DepthStencilDescriptor::default());
    unwritten.front_face_stencil.write_mask = 0x100;
    unwritten.back_face_stencil.write_mask = 0;
    let alone = draws(&unwritten).expect("Always");
    assert!(alone.draws() == [own(&unwritten)]);

    // Both faces `Always` and `Replace` under every bit, a depth compare of `Always` that writes:
    // the zeroing draw, which writes the depth, then a draw for each bit.
    let mut replace = replacing(DepthStencilDescriptor::default());
    replace.depth_write_enabled = true;
    let all = draws(&replace).expect("Always");
    let zeroing = BitDraw {
        reference: [0; 2],
        ..own(&replace)
    };
    assert_eq!(all.draws().len(), 9);
    assert!(all.draws()[0] == zeroing);
    for (k, draw) in all.draws()[1..].iter().enumerate() {
        assert_eq!((draw.bit, draw.reference), (k as u32, [255; 2]), "bit {k}");
        assert!(!draw.tests.depth_write, "bit {k}");
        for face in draw.tests.faces {
            let ops = (face.fail, face.pass, face.depth_fail);
            let (keep, replace) = (vk::StencilOp::KEEP, vk::StencilOp::REPLACE);
            assert!(ops == (keep, replace, keep), "bit {k}");
            assert_eq!(face.write_mask, 1 << k, "bit {k}");
            assert!(face.compare == vk::CompareOp::ALWAYS, "bit {k}");
        }
    }

    // Under a depth compare that tests and writes: no draw before the last writes the depth, and
    // the last writes it under the encoder's reference and stencil write masks of 0.
    replace.depth_compare_function = CompareFunction::Less;
    let last = draws(&replace).expect("Always");
    assert_eq!(last.draws().len(), 10);
    assert!(last.draws()[..9].iter().all(|draw| !draw.tests.depth_write));
    let fin = last.draws()[9];
    assert_eq!((fin.bit, fin.reference), (EVERY_FRAGMENT, reference));
    let tests = Tests::of(&replace);
    assert!(fin.tests.depth() == tests.depth());
    assert!(fin.tests.faces.iter().all(|face| face.write_mask == 0));

    // Only a face that replaces counts, and only the low 8 bits of its write mask: a front face
    // that increments under `0xff0` beside a back face of `Never` that replaces on a failure under
    // `0x105`. The first draw keeps the increment; the bit draws keep none.
    let mut mixed = DepthStencilDescriptor::default();
    mixed.front_face_stencil.depth_stencil_pass_operation = StencilOperation::IncrementClamp;
    mixed.front_face_stencil.write_mask = 0xff0;
    let back = &mut mixed.back_face_stencil;
    back.stencil_compare_function = CompareFunction::Never;
    back.stencil_failure_operation = StencilOperation::Replace;
    back.write_mask = 0x105;
    let mixed = draws(&mixed).expect("Always and Never");
    let bits: Vec<u32> = mixed.draws().iter().map(|draw| draw.bit).collect();
    assert_eq!(bits, [EVERY_FRAGMENT, 0, 2]);
    let first = mixed.draws()[0].tests.faces;
    assert!(first[0].pass == vk::StencilOp::INCREMENT_AND_CLAMP);
    for draw in &mixed.draws()[1..] {
        let [front, back] = draw.tests.faces;
        assert!((front.pass, front.write_mask) == (vk::StencilOp::KEEP, 0));
        assert!(back.fail == vk::StencilOp::REPLACE);
        assert_eq!(back.write_mask, 1 << draw.bit);
    }

    // A compare the reference decides.
    let mut equal = replace;
    equal.back_face_stencil.stencil_compare_function = CompareFunction::Equal;
    assert!(draws(&equal).is_none());
}

/// Two fragment functions of our own: `overlap_stencil` returns as its `[[stencil]]` its flat
/// colour input's red channel, converted to `uint`; `blit_stencil` returns as its
/// `[[depth(any)]]` sample 0 of its `depth2d_ms<float>` (texture 0) and as its `[[stencil]]` the
/// first component of sample 0 of its `texture2d_ms<uint>` (texture 1), each at its pixel.
#[cfg(any(m2v_gpu, m2v_cpu_icd))]
const STENCIL_FRAGMENTS: &str = r#"target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

define <{ i32 }> @overlap_stencil(<4 x float> %position, <4 x float> %color) {
  %red = extractelement <4 x float> %color, i64 0
  %stencil = tail call i32 @air.convert.u.i32.f.f32(float %red)
  %out = insertvalue <{ i32 }> undef, i32 %stencil, 0
  ret <{ i32 }> %out
}

define <{ float, i32 }> @blit_stencil(<4 x float> %position, ptr addrspace(1) %depths, ptr addrspace(1) %stencils) {
  %x = extractelement <4 x float> %position, i64 0
  %y = extractelement <4 x float> %position, i64 1
  %xi = tail call i32 @air.convert.u.i32.f.f32(float %x)
  %yi = tail call i32 @air.convert.u.i32.f.f32(float %y)
  %p0 = insertelement <2 x i32> undef, i32 %xi, i64 0
  %p = insertelement <2 x i32> %p0, i32 %yi, i64 1
  %sampler = tail call ptr addrspace(2) @air.get_read_sampler()
  %d = tail call { float, i8 } @air.read_depth_2d_ms.f32(ptr addrspace(1) %depths, ptr addrspace(2) %sampler, i32 1, <2 x i32> %p, i32 0, i32 1)
  %depth = extractvalue { float, i8 } %d, 0
  %t = tail call { <4 x i32>, i8 } @air.read_texture_2d_ms.u.v4i32(ptr addrspace(1) %stencils, ptr addrspace(2) %sampler, <2 x i32> %p, i32 0, i32 1)
  %texel = extractvalue { <4 x i32>, i8 } %t, 0
  %stencil = extractelement <4 x i32> %texel, i64 0
  %o0 = insertvalue <{ float, i32 }> undef, float %depth, 0
  %out = insertvalue <{ float, i32 }> %o0, i32 %stencil, 1
  ret <{ float, i32 }> %out
}

declare i32 @air.convert.u.i32.f.f32(float)
declare ptr addrspace(2) @air.get_read_sampler()
declare { float, i8 } @air.read_depth_2d_ms.f32(ptr addrspace(1), ptr addrspace(2), i32, <2 x i32>, i32, i32)
declare { <4 x i32>, i8 } @air.read_texture_2d_ms.u.v4i32(ptr addrspace(1), ptr addrspace(2), <2 x i32>, i32, i32)

!air.fragment = !{!0, !6}
!air.version = !{!20}
!air.language_version = !{!21}

!0 = !{ptr @overlap_stencil, !1, !3}
!1 = !{!2}
!2 = !{!"air.stencil", !"air.arg_type_name", !"uint", !"air.arg_name", !"stencil"}
!3 = !{!4, !5}
!4 = !{i32 0, !"air.position", !"air.center", !"air.no_perspective", !"air.arg_type_name", !"float4", !"air.arg_name", !"position", !"air.arg_unused"}
!5 = !{i32 1, !"air.fragment_input", !"generated(5colorDv4_f)", !"air.flat", !"air.arg_type_name", !"float4", !"air.arg_name", !"color"}
!6 = !{ptr @blit_stencil, !7, !9}
!7 = !{!8, !2}
!8 = !{!"air.depth", !"air.depth_qualifier", !"air.any", !"air.arg_type_name", !"float", !"air.arg_name", !"depth"}
!9 = !{!10, !11, !12}
!10 = !{i32 0, !"air.position", !"air.center", !"air.no_perspective", !"air.arg_type_name", !"float4", !"air.arg_name", !"position"}
!11 = !{i32 1, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.read", !"air.arg_type_name", !"depth2d_ms<float, read>", !"air.arg_name", !"depths"}
!12 = !{i32 2, !"air.texture", !"air.location_index", i32 1, i32 1, !"air.read", !"air.arg_type_name", !"texture2d_ms<uint, read>", !"air.arg_name", !"stencils"}
!20 = !{i32 2, i32 8, i32 0}
!21 = !{!"Metal", i32 4, i32 0, i32 0}
"#;

#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod on_device {
    use super::{replacing, STENCIL_FRAGMENTS};
    use crate::tests::command::on_device::{each_device, finished};
    use crate::tests::draw::on_device::{examples, floats, library, Buffers, SIDE};
    use crate::{
        Buffer, BufferLayout, ClearColor, ColorAttachmentDescriptor, CompareFunction,
        DepthStencilDescriptor, DepthStencilState, Device, DrawKind, Error, FunctionDescriptor,
        Gap, IndexClass, Library, LoadAction, PixelFormat, PrimitiveType, RenderPassDescriptor,
        RenderPipelineDescriptor, RenderPipelineState, ScissorRect, StageSource, StencilOperation,
        StorageMode, StoreAction, Texture, TextureDescriptor, TexturePlace, TextureType,
        TextureUsage, VisibilityResultMode,
    };
    use metal2vulkan::error::Feature;
    use metal2vulkan::metal::ColorWriteMask;
    use metal2vulkan_devtools::{engine_dir, llvm_assemble_text, AssembleError};

    /// The stencil the attachments clear to, and the depth.
    const CLEARED: u32 = 0x3c;
    const CLEARED_DEPTH: f32 = 0.5;
    /// The encoder's reference, which each fragment's output replaces.
    const REFERENCE: u32 = 0x5a;
    /// The overlap row's outputs: the first triangle's, then the second's.
    const FIRST: u32 = 0x35;
    const SECOND: u32 = 0x4a;
    const TEXELS: usize = (SIDE * SIDE) as usize;

    /// The word pixel `i` (row-major) of the fragment function's buffer returns: bits above the
    /// low 8 on two of every three pixels, which Metal's stencil reference drops, and low bits of
    /// 0 on every fifth pixel, which only the zeroing draw reaches.
    fn value(i: u32) -> u32 {
        let low = match i % 5 {
            0 => 0,
            _ => (i * 29 + 7) % 256,
        };
        0x100 * (i % 3) + low
    }

    /// The depth pixel `i` returns: eighths along the diagonals, half of them below
    /// [`CLEARED_DEPTH`].
    fn depth(i: u32) -> f32 {
        ((i + i / SIDE) % 8) as f32 / 8.0
    }

    /// The library of [`STENCIL_FRAGMENTS`], or `None` where `llvm-as` is missing.
    fn ours() -> Option<Library> {
        match llvm_assemble_text(STENCIL_FRAGMENTS) {
            Ok(bitcode) => Some(Library::from_air(&bitcode).expect("the library")),
            Err(AssembleError::Missing(why)) => {
                eprintln!("SKIPPED: {why}");
                None
            }
            Err(error) => panic!("the stencil fragments: {error:?}"),
        }
    }

    fn stage(library: &Library, name: &str) -> StageSource {
        StageSource::Function(FunctionDescriptor::new(
            library.function(name).expect("the function"),
        ))
    }

    /// A pipeline of `colored_vertex` and `fragment`, drawing into a `format` attachment as its
    /// depth (where the format has it) and its stencil, and no colour attachment.
    fn descriptor(
        examples: &Library,
        (library, fragment): (&Library, &str),
        format: PixelFormat,
    ) -> RenderPipelineDescriptor {
        let mut d = RenderPipelineDescriptor::new(stage(examples, "colored_vertex"));
        d.fragment = Some(stage(library, fragment));
        d.stencil_format = Some(format);
        if format == PixelFormat::Depth32FloatStencil8 {
            d.depth_format = Some(format);
        }
        d
    }

    /// Whether `device` lists the extension, which its translator's profile then has.
    fn exports(name: &str, device: &Device) -> bool {
        let export = ash::ext::shader_stencil_export::NAME;
        let listed = device.extensions().iter().any(|e| e.as_c_str() == export);
        let profile = device.core().translator.profile().fragment().stencil_export;
        assert_eq!(profile, listed, "{name}: the translator's stencil export");
        listed
    }

    /// The bit draws the device counted.
    fn bit_draws(device: &Device) -> u64 {
        device.counters().metrics().stencil_bit_draws
    }

    /// A Shared buffer's bytes, as the host reads them.
    fn bytes(buffer: &Buffer) -> Vec<u8> {
        let contents = buffer.contents().expect("a Shared buffer");
        // SAFETY: the buffer is alive, `length` bytes long, and no GPU work uses it.
        unsafe { std::slice::from_raw_parts(contents.as_ptr(), buffer.length() as usize) }.to_vec()
    }

    fn floats_of(bytes: &[u8]) -> Vec<f32> {
        let (words, _) = bytes.as_chunks::<4>();
        words.iter().map(|word| f32::from_le_bytes(*word)).collect()
    }

    /// A Private `SIDE` x `SIDE` render target of `format` and `samples`, at texture index
    /// `index` where given.
    fn attachment(
        device: &Device,
        format: PixelFormat,
        samples: u32,
        index: Option<u32>,
    ) -> Texture {
        let kind = match samples {
            1 => TextureType::Type2D,
            _ => TextureType::Type2DMultisample,
        };
        let mut d = TextureDescriptor::new(kind, format, SIDE, SIDE);
        d.sample_count = samples;
        d.usage = TextureUsage::RENDER_TARGET;
        if index.is_some() {
            d.usage |= TextureUsage::SHADER_READ | TextureUsage::PIXEL_FORMAT_VIEW;
        }
        d.storage_mode = StorageMode::Private;
        let made = match index {
            Some(index) => device.new_texture(&d, Some(index)),
            None => device.new_texture_for_views(&d),
        };
        made.unwrap_or_else(|error| panic!("{format:?}: {error}"))
    }

    /// A pass of `attachment` as its stencil, and its depth where its format has one, cleared to
    /// [`CLEARED`] and [`CLEARED_DEPTH`] and stored.
    fn pass(attachment: &Texture) -> RenderPassDescriptor {
        let mut d = RenderPassDescriptor::default();
        let s = &mut d.stencil_attachment;
        s.texture = Some(attachment.clone());
        s.load_action = LoadAction::Clear;
        s.store_action = StoreAction::Store;
        s.clear_stencil = CLEARED;
        if attachment.descriptor().pixel_format == PixelFormat::Depth32FloatStencil8 {
            let z = &mut d.depth_attachment;
            z.texture = Some(attachment.clone());
            z.load_action = LoadAction::Clear;
            z.store_action = StoreAction::Store;
            z.clear_depth = f64::from(CLEARED_DEPTH);
        }
        d
    }

    /// The planes a blit reads back.
    #[derive(Clone, Copy, PartialEq)]
    enum BlitPlanes {
        Stencil,
        Both,
    }

    /// What a draw stored: each texel's stencil, its depth where the attachment has one, and the
    /// bit draws the device counted for it.
    struct Stored {
        stencil: Vec<u8>,
        depth: Vec<f32>,
        bit_draws: u64,
    }

    /// The vertices of `colored_vertex`: `positions`, two `float`s each, and `reds`, each
    /// vertex's colour's red channel.
    fn vertices(
        device: &Device,
        buffers: &mut Buffers,
        positions: &[f32],
        reds: &[f32],
    ) -> [Buffer; 2] {
        let colors: Vec<f32> = reds.iter().flat_map(|&red| [red, 0.0, 0.0, 0.0]).collect();
        [floats(positions), floats(&colors)].map(|bytes| buffers.of(device, &bytes))
    }

    /// The quad over the whole target, a triangle strip of 4 vertices.
    const QUAD: [f32; 8] = [-1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0];

    /// A draw of `state` under `tests` into a new `format` attachment ([`pass`]), its fragment
    /// buffers `fragment` from index 0 and the width at index 1, of `positions` as `primitive`
    /// with `reds` as their colours' red channels; then the attachment's `planes` read back.
    fn stored(
        name: &str,
        device: &Device,
        (state, tests): (&RenderPipelineState, &DepthStencilState),
        format: PixelFormat,
        fragment: [Option<&Buffer>; 3],
        (primitive, positions, reds): (PrimitiveType, &[f32], &[f32]),
    ) -> Stored {
        let mut buffers = Buffers::new(device);
        let [positions, colors] = vertices(device, &mut buffers, positions, reds);
        let count = positions.length() as u32 / 8;
        let target = attachment(device, format, 1, None);
        let planes = match format {
            PixelFormat::Depth32FloatStencil8 => BlitPlanes::Both,
            _ => BlitPlanes::Stencil,
        };
        let stencil_out = buffers.of(device, &[0xee; TEXELS]);
        let depth_out = buffers.of(device, &[0xee; 4 * TEXELS]);
        let before = bit_draws(device);
        let queue = device.new_command_queue(4).expect("a queue");
        let command_buffer = queue.command_buffer().expect("a command buffer");
        let ok = |result: Result<(), Error>| result.unwrap_or_else(|e| panic!("{name}: {e}"));
        let mut encoder = (command_buffer.render_command_encoder(&pass(&target)))
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        ok(encoder.set_render_pipeline_state(state));
        ok(encoder.set_vertex_buffer(Some(&positions), 0, 0));
        ok(encoder.set_vertex_buffer(Some(&colors), 0, 1));
        for (index, buffer) in fragment.into_iter().enumerate() {
            if let Some(buffer) = buffer {
                ok(encoder.set_fragment_buffer(Some(buffer), 0, index as u32));
            }
        }
        ok(encoder.set_fragment_bytes(&SIDE.to_le_bytes(), 1));
        encoder.set_stencil_reference_value(REFERENCE);
        ok(encoder.set_depth_stencil_state(Some(tests)));
        ok(encoder.draw_primitives(primitive, 0, count, 1, 0));
        encoder.end_encoding();
        read_back(
            name,
            &command_buffer,
            &target,
            planes,
            [&depth_out, &stencil_out],
        );
        Stored {
            stencil: bytes(&stencil_out),
            depth: floats_of(&bytes(&depth_out)),
            bit_draws: bit_draws(device) - before,
        }
    }

    /// Copies `texture`'s `planes` into `outs` (depth, then stencil), commits `command_buffer`,
    /// and waits for it.
    fn read_back(
        name: &str,
        command_buffer: &crate::CommandBuffer,
        texture: &Texture,
        planes: BlitPlanes,
        [depth_out, stencil_out]: [&Buffer; 2],
    ) {
        use crate::BlitOption::{DepthFromDepthStencil, StencilFromDepthStencil};
        let ok = |result: Result<(), Error>| result.unwrap_or_else(|e| panic!("{name}: {e}"));
        let mut blit = (command_buffer.blit_command_encoder())
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        let place = TexturePlace {
            slice: 0,
            level: 0,
            origin: [0; 3],
        };
        let whole = [SIDE, SIDE, 1];
        let tight = |bytes: u64| BufferLayout {
            offset: 0,
            bytes_per_row: bytes * u64::from(SIDE),
            bytes_per_image: bytes * u64::from(SIDE * SIDE),
        };
        match planes {
            BlitPlanes::Stencil => {
                ok(blit.copy_from_texture_to_buffer(texture, place, whole, stencil_out, tight(1)))
            }
            BlitPlanes::Both => {
                ok(blit.copy_from_texture_to_buffer_with_options(
                    texture,
                    place,
                    whole,
                    depth_out,
                    tight(4),
                    DepthFromDepthStencil,
                ));
                ok(blit.copy_from_texture_to_buffer_with_options(
                    texture,
                    place,
                    whole,
                    stencil_out,
                    tight(1),
                    StencilFromDepthStencil,
                ));
            }
        }
        blit.end_encoding();
        command_buffer.commit().expect("a commit");
        finished(name, command_buffer);
        assert_eq!(command_buffer.error(), None, "{name}");
    }

    /// Panics, naming the first texel and how many differ, where `got` is not `want`.
    fn exact<T: PartialEq + std::fmt::Debug>(name: &str, what: &str, got: &[T], want: &[T]) {
        assert_eq!(got.len(), want.len(), "{name}: {what}: the texels read");
        let differ: Vec<usize> = (0..got.len()).filter(|&i| got[i] != want[i]).collect();
        if let Some(&first) = differ.first() {
            let side = SIDE as usize;
            panic!(
                "{name}: {what}: {} of {} texels differ, the first at ({}, {}): {:?}, not {:?}",
                differ.len(),
                got.len(),
                first % side,
                first / side,
                got[first],
                want[first]
            );
        }
    }

    /// The bits of `depths`.
    fn bits(depths: &[f32]) -> Vec<u32> {
        depths.iter().map(|depth| depth.to_bits()).collect()
    }

    /// The stencil alone, and the stencil with the depth under `Always` and under `Less`: every
    /// texel exact, and the draws by bits counted where the device has no stencil export.
    #[test]
    fn every_device_stores_a_fragment_functions_stencil_output() {
        let (Some(examples), Some(stencils)) =
            (examples(), library(&engine_dir().join("stx_stencil.ll")))
        else {
            return;
        };
        let alone = descriptor(
            &examples,
            (&stencils, "stx_stencil_by_pixel"),
            PixelFormat::Stencil8,
        );
        let with_depth = descriptor(
            &examples,
            (&stencils, "stx_stencil_with_depth"),
            PixelFormat::Depth32FloatStencil8,
        );
        each_device("stencil_export", |name, _, device| {
            let export = exports(name, &device);
            let make = |d: &RenderPipelineDescriptor| {
                (device.new_render_pipeline_state(d, crate::tests::no_constexpr_samplers))
                    .unwrap_or_else(|error| panic!("{name}: {error}"))
            };
            let (alone, with_depth) = (make(&alone), make(&with_depth));
            let mut buffers = Buffers::new(&device);
            let words: Vec<u8> = (0..SIDE * SIDE)
                .flat_map(|i| value(i).to_le_bytes())
                .collect();
            let values = buffers.of(&device, &words);
            let depths: Vec<f32> = (0..SIDE * SIDE).map(depth).collect();
            let depth_buffer = buffers.of(&device, &floats(&depths));
            let quad = (PrimitiveType::TriangleStrip, &QUAD[..], &[0.0; 4][..]);
            let replace = replacing(DepthStencilDescriptor::default());
            let outputs: Vec<u8> = (0..SIDE * SIDE).map(|i| value(i) as u8).collect();
            let zeros = outputs.iter().filter(|&&s| s == 0).count();
            // What a device without the extension draws beyond each draw.
            let more = |n: u64| if export { 0 } else { n };
            let mut counted = Vec::new();

            // The stencil alone.
            let tests = device.new_depth_stencil_state(&replace);
            let fragment = [Some(&values), None, None];
            let got = stored(
                name,
                &device,
                (&alone, &tests),
                PixelFormat::Stencil8,
                fragment,
                quad,
            );
            exact(name, "the stencil alone", &got.stencil, &outputs);
            assert_eq!(
                got.bit_draws,
                more(8),
                "{name}: the stencil alone's bit draws"
            );
            counted.push(got.bit_draws);

            // With the depth, under `Always` that writes.
            let mut always = replace;
            always.depth_write_enabled = true;
            let tests = device.new_depth_stencil_state(&always);
            let fragment = [Some(&values), None, Some(&depth_buffer)];
            let format = PixelFormat::Depth32FloatStencil8;
            let got = stored(name, &device, (&with_depth, &tests), format, fragment, quad);
            exact(name, "the stencil beside the depth", &got.stencil, &outputs);
            exact(
                name,
                "the depth beside the stencil",
                &bits(&got.depth),
                &bits(&depths),
            );
            assert_eq!(got.bit_draws, more(8), "{name}: Always's bit draws");
            counted.push(got.bit_draws);

            // With the depth, under `Less` that writes, a depth failure inverting.
            let mut less = always;
            less.depth_compare_function = CompareFunction::Less;
            for face in [&mut less.front_face_stencil, &mut less.back_face_stencil] {
                face.depth_failure_operation = StencilOperation::Invert;
            }
            let tests = device.new_depth_stencil_state(&less);
            let got = stored(name, &device, (&with_depth, &tests), format, fragment, quad);
            let passes = |i: usize| depths[i] < CLEARED_DEPTH;
            let want_stencil: Vec<u8> = (0..TEXELS)
                .map(|i| match passes(i) {
                    true => outputs[i],
                    false => !(CLEARED as u8),
                })
                .collect();
            let want_depth: Vec<f32> = (0..TEXELS)
                .map(|i| match passes(i) {
                    true => depths[i],
                    false => CLEARED_DEPTH,
                })
                .collect();
            exact(name, "the stencil under Less", &got.stencil, &want_stencil);
            exact(
                name,
                "the depth under Less",
                &bits(&got.depth),
                &bits(&want_depth),
            );
            assert_eq!(got.bit_draws, more(9), "{name}: Less's bit draws");
            counted.push(got.bit_draws);
            let passing = (0..TEXELS).filter(|&i| passes(i)).count();
            let form = match export {
                true => "VK_EXT_shader_stencil_export enabled",
                false => "no VK_EXT_shader_stencil_export: drawn by bits",
            };
            println!(
                "{name}: {form}; {TEXELS} texels exact for the stencil alone, for the stencil and \
                 depth under Always, {zeros} of them of stencil 0 (the zeroing draw's alone), and \
                 under Less ({passing} passing, the rest inverted); bit draws {counted:?}"
            );
        });
    }

    /// Two triangles of one draw over one half of the target: the half they both reach stores
    /// the later output with the extension and the OR of both without it, the row printed.
    #[test]
    fn every_device_stores_its_answer_where_two_fragments_of_one_draw_overlap() {
        let (Some(examples), Some(ours)) = (examples(), ours()) else {
            return;
        };
        let d = descriptor(&examples, (&ours, "overlap_stencil"), PixelFormat::Stencil8);
        each_device("stencil_overlap", |name, _, device| {
            let export = exports(name, &device);
            let state = (device.new_render_pipeline_state(&d, crate::tests::no_constexpr_samplers))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let tests =
                device.new_depth_stencil_state(&replacing(DepthStencilDescriptor::default()));
            // The whole target, then its top half (clip space y up).
            let positions = [
                -1.0, -1.0, 3.0, -1.0, -1.0, 3.0, -1.0, 0.0, 3.0, 0.0, -1.0, 4.0,
            ];
            let reds = [FIRST, FIRST, FIRST, SECOND, SECOND, SECOND].map(|s| s as f32);
            let geometry = (PrimitiveType::Triangle, &positions[..], &reds[..]);
            let format = PixelFormat::Stencil8;
            let got = stored(name, &device, (&state, &tests), format, [None; 3], geometry);
            let both = match export {
                true => SECOND,
                false => FIRST | SECOND,
            };
            let want: Vec<u8> = (0..SIDE * SIDE)
                .map(|i| match i / SIDE < SIDE / 2 {
                    true => both as u8,
                    false => FIRST as u8,
                })
                .collect();
            let top: std::collections::BTreeSet<u8> =
                got.stencil[..TEXELS / 2].iter().copied().collect();
            println!(
                "{name}: overlap row: {} texels two fragments of one draw reach ({FIRST:#04x}, then \
                 {SECOND:#04x}) store {top:02x?}; the {} one reaches store {:02x?}; bit draws {}",
                TEXELS / 2,
                TEXELS / 2,
                (got.stencil[TEXELS / 2..].iter().copied()).collect::<std::collections::BTreeSet<u8>>(),
                got.bit_draws
            );
            exact(name, "the overlap", &got.stencil, &want);
            assert_eq!(
                got.bit_draws,
                if export { 0 } else { 8 },
                "{name}: bit draws"
            );
        });
    }

    /// Without the extension: a function that writes a colour beside its stencil is refused at
    /// the state's creation, a draw under a stencil compare of `Equal` and one in a `Counting`
    /// visibility result's segment are refused; a draw in a `Boolean` segment is drawn. With it,
    /// each is made and drawn.
    #[test]
    fn without_stencil_export_what_draws_by_bits_cannot_answer_is_refused() {
        let (Some(examples), Some(stencils), Some(colors)) = (
            examples(),
            library(&engine_dir().join("stx_stencil.ll")),
            library(&engine_dir().join("stx_stencil_ms.ll")),
        ) else {
            return;
        };
        let alone = descriptor(
            &examples,
            (&stencils, "stx_stencil_by_pixel"),
            PixelFormat::Stencil8,
        );
        let mut colored = descriptor(
            &examples,
            (&colors, "stx_stencil_ms_color"),
            PixelFormat::Stencil8,
        );
        colored.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::R32Uint));
        each_device("stencil_refusals", |name, _, device| {
            let export = exports(name, &device);
            let made =
                device.new_render_pipeline_state(&colored, crate::tests::no_constexpr_samplers);
            match (export, made) {
                (true, Ok(_)) => {}
                (
                    false,
                    Err(Error::Translation {
                        error: metal2vulkan::error::Error::Unsupported(u),
                        ..
                    }),
                ) => assert_eq!(u.feature, Feature::StencilExport, "{name}: {u:?}"),
                (_, made) => panic!("{name}: the colour beside the stencil: {:?}", made.err()),
            }
            let state = (device
                .new_render_pipeline_state(&alone, crate::tests::no_constexpr_samplers))
            .unwrap_or_else(|error| panic!("{name}: {error}"));
            let mut buffers = Buffers::new(&device);
            let words: Vec<u8> = (0..SIDE * SIDE)
                .flat_map(|i| value(i).to_le_bytes())
                .collect();
            let values = buffers.of(&device, &words);
            let [positions, colors] = vertices(&device, &mut buffers, &QUAD, &[0.0; 4]);
            let visibility = buffers.of(&device, &[0xaa; 32]);
            let mut equal = replacing(DepthStencilDescriptor::default());
            for face in [&mut equal.front_face_stencil, &mut equal.back_face_stencil] {
                face.stencil_compare_function = CompareFunction::Equal;
            }
            let equal = device.new_depth_stencil_state(&equal);
            let replace =
                device.new_depth_stencil_state(&replacing(DepthStencilDescriptor::default()));
            let target = attachment(&device, PixelFormat::Stencil8, 1, None);
            let mut d = pass(&target);
            d.visibility_result_buffer = Some(visibility.clone());
            let queue = device.new_command_queue(4).expect("a queue");
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let ok = |result: Result<(), Error>| result.unwrap_or_else(|e| panic!("{name}: {e}"));
            let mut encoder = (command_buffer.render_command_encoder(&d))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            ok(encoder.set_render_pipeline_state(&state));
            ok(encoder.set_vertex_buffer(Some(&positions), 0, 0));
            ok(encoder.set_vertex_buffer(Some(&colors), 0, 1));
            ok(encoder.set_fragment_buffer(Some(&values), 0, 0));
            ok(encoder.set_fragment_bytes(&SIDE.to_le_bytes(), 1));
            let refused = |result: Result<(), Error>, what: &str| match (export, result) {
                (true, Ok(())) => {}
                (
                    false,
                    Err(Error::Unsupported(Gap::Draw {
                        kind: DrawKind::StencilOutput,
                    })),
                ) => {}
                (_, result) => panic!("{name}: {what}: {result:?}"),
            };
            let quad = |encoder: &mut crate::RenderCommandEncoder| {
                encoder.draw_primitives(PrimitiveType::TriangleStrip, 0, 4, 1, 0)
            };
            ok(encoder.set_depth_stencil_state(Some(&equal)));
            refused(quad(&mut encoder), "under Equal");
            ok(encoder.set_depth_stencil_state(Some(&replace)));
            ok(encoder.set_visibility_result_mode(VisibilityResultMode::Counting, 0));
            refused(quad(&mut encoder), "in a Counting segment");
            ok(encoder.set_visibility_result_mode(VisibilityResultMode::Boolean, 8));
            ok(quad(&mut encoder));
            ok(encoder.set_visibility_result_mode(VisibilityResultMode::Disabled, 0));
            encoder.end_encoding();
            command_buffer.commit().expect("a commit");
            finished(name, &command_buffer);
            assert_eq!(command_buffer.error(), None, "{name}");
            let boolean =
                u64::from_le_bytes((&bytes(&visibility)[8..16]).try_into().expect("8 bytes"));
            assert_ne!(boolean, 0, "{name}: the Boolean segment saw no sample");
            let refusals = match export {
                true => "made and drawn: a colour beside the stencil, Equal, a Counting segment",
                false => {
                    "refused: a colour beside the stencil (StencilExport), Equal and a \
                          Counting segment (DrawKind::StencilOutput)"
                }
            };
            println!("{name}: {refusals}; a Boolean segment drawn, its result {boolean}");
        });
    }

    /// The route: a 4-sample source's stencil and depth, read at sample 0 by `blit_stencil`, into
    /// a single-sample target beside a colour attachment whose write mask is none.
    #[test]
    fn every_device_draws_a_multisampled_depth_and_stencil_into_a_single_sample_target() {
        let (Some(examples), Some(ours), Some(fill)) = (
            examples(),
            ours(),
            library(&engine_dir().join("depth_fill.ll")),
        ) else {
            return;
        };
        let format = PixelFormat::Depth32FloatStencil8;
        let mut filling = RenderPipelineDescriptor::new(stage(&fill, "depth_fill_vertex"));
        filling.fragment = Some(stage(&fill, "depth_fill_fragment"));
        filling.depth_format = Some(format);
        filling.stencil_format = Some(format);
        filling.raster_sample_count = 4;
        let mut blit = descriptor(&examples, (&ours, "blit_stencil"), format);
        let mut color = ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm);
        color.write_mask = ColorWriteMask::NONE;
        blit.color_attachments[0] = Some(color);
        // Each quadrant's stencil: top left, top right, bottom left, bottom right.
        let quadrants: [u32; 4] = [0x00, 0x5a, 0xa5, 0xff];
        let quadrant = |i: u32| {
            let (x, y) = (i % SIDE, i / SIDE);
            quadrants[usize::from(y >= SIDE / 2) * 2 + usize::from(x >= SIDE / 2)]
        };
        each_device("stencil_route", |name, _, device| {
            let export = exports(name, &device);
            let make = |d: &RenderPipelineDescriptor| {
                (device.new_render_pipeline_state(d, crate::tests::no_constexpr_samplers))
                    .unwrap_or_else(|error| panic!("{name}: {error}"))
            };
            let (filling, blit) = (make(&filling), make(&blit));
            let first = IndexClass::Texture.range().start;
            let _textures = (device.grant_indices(IndexClass::Texture, first, 2))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let source = attachment(&device, format, 4, Some(first));
            let stencil_view = (source.new_texture_view(
                PixelFormat::X32Stencil8,
                TextureType::Type2DMultisample,
                0..1,
                0..1,
                Some(first + 1),
            ))
            .unwrap_or_else(|error| panic!("{name}: the stencil view: {error}"));
            let target = attachment(&device, format, 1, None);
            let mut buffers = Buffers::new(&device);
            let depths: Vec<f32> = (0..SIDE * SIDE).map(depth).collect();
            let depth_buffer = buffers.of(&device, &floats(&depths));
            let [positions, colors] = vertices(&device, &mut buffers, &QUAD, &[0.0; 4]);
            let stencil_out = buffers.of(&device, &[0xee; TEXELS]);
            let depth_out = buffers.of(&device, &[0xee; 4 * TEXELS]);
            let colors_target = crate::tests::draw::on_device::target(
                &device,
                PixelFormat::RGBA8Unorm,
                TextureUsage::RENDER_TARGET,
            );
            let mut always = replacing(DepthStencilDescriptor::default());
            always.depth_write_enabled = true;
            for face in [
                &mut always.front_face_stencil,
                &mut always.back_face_stencil,
            ] {
                face.stencil_failure_operation = StencilOperation::Replace;
                face.depth_failure_operation = StencilOperation::Replace;
            }
            let always = device.new_depth_stencil_state(&always);
            let ok = |result: Result<(), Error>, what: &str| {
                result.unwrap_or_else(|error| panic!("{name}: {what}: {error}"))
            };
            let queue = device.new_command_queue(4).expect("a queue");
            let command_buffer = queue.command_buffer().expect("a command buffer");

            // The source: each pixel's depth, each quadrant's stencil through its reference.
            let mut fills = RenderPassDescriptor::default();
            let z = &mut fills.depth_attachment;
            z.texture = Some(source.clone());
            z.load_action = LoadAction::Clear;
            z.store_action = StoreAction::Store;
            z.clear_depth = 1.0;
            let s = &mut fills.stencil_attachment;
            s.texture = Some(source.clone());
            s.load_action = LoadAction::Clear;
            s.store_action = StoreAction::Store;
            let mut encoder = (command_buffer.render_command_encoder(&fills))
                .unwrap_or_else(|error| panic!("{name}: the source's pass: {error}"));
            ok(encoder.set_render_pipeline_state(&filling), "the fill");
            ok(
                encoder.set_fragment_buffer(Some(&depth_buffer), 0, 0),
                "the depths",
            );
            ok(
                encoder.set_fragment_bytes(&SIDE.to_le_bytes(), 1),
                "the width",
            );
            ok(
                encoder.set_depth_stencil_state(Some(&always)),
                "the fill's state",
            );
            let half = u64::from(SIDE / 2);
            for (q, &stencil) in quadrants.iter().enumerate() {
                encoder.set_scissor_rect(ScissorRect {
                    x: half * (q as u64 % 2),
                    y: half * (q as u64 / 2),
                    width: half,
                    height: half,
                });
                encoder.set_stencil_reference_value(stencil);
                ok(
                    encoder.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0),
                    "a quadrant",
                );
            }
            encoder.end_encoding();

            // The route.
            let mut route = RenderPassDescriptor::default();
            let c = &mut route.color_attachments[0];
            c.texture = Some(colors_target.clone());
            c.load_action = LoadAction::Clear;
            c.clear_color = ClearColor::new(1.0, 0.0, 1.0, 1.0);
            let z = &mut route.depth_attachment;
            z.texture = Some(target.clone());
            z.load_action = LoadAction::Clear;
            z.store_action = StoreAction::Store;
            z.clear_depth = f64::from(CLEARED_DEPTH);
            let s = &mut route.stencil_attachment;
            s.texture = Some(target.clone());
            s.load_action = LoadAction::Clear;
            s.store_action = StoreAction::Store;
            s.clear_stencil = CLEARED;
            let before = bit_draws(&device);
            let mut encoder = (command_buffer.render_command_encoder(&route))
                .unwrap_or_else(|error| panic!("{name}: the route's pass: {error}"));
            ok(encoder.set_render_pipeline_state(&blit), "the route");
            ok(
                encoder.set_vertex_buffer(Some(&positions), 0, 0),
                "the positions",
            );
            ok(
                encoder.set_vertex_buffer(Some(&colors), 0, 1),
                "the colours",
            );
            ok(encoder.set_fragment_texture(Some(&source), 0), "the source");
            ok(
                encoder.set_fragment_texture(Some(&stencil_view), 1),
                "the stencil view",
            );
            encoder.set_stencil_reference_value(REFERENCE);
            ok(
                encoder.set_depth_stencil_state(Some(&always)),
                "the route's state",
            );
            ok(
                encoder.draw_primitives(PrimitiveType::TriangleStrip, 0, 4, 1, 0),
                "the quad",
            );
            encoder.end_encoding();
            read_back(
                name,
                &command_buffer,
                &target,
                BlitPlanes::Both,
                [&depth_out, &stencil_out],
            );
            let counted = bit_draws(&device) - before;

            let want: Vec<u8> = (0..SIDE * SIDE).map(|i| quadrant(i) as u8).collect();
            exact(name, "the route's stencil", &bytes(&stencil_out), &want);
            exact(
                name,
                "the route's depth",
                &bits(&floats_of(&bytes(&depth_out))),
                &bits(&depths),
            );
            let colours = crate::tests::draw::on_device::read(&colors_target, 4);
            let cleared: Vec<u8> = [255, 0, 255, 255].repeat(TEXELS);
            exact(name, "the route's colour", &colours, &cleared);
            assert_eq!(counted, if export { 0 } else { 8 }, "{name}: bit draws");
            println!(
                "{name}: the route: a 4-sample Depth32Float_Stencil8 source read at sample 0 into a \
                 single-sample target, {TEXELS} stencils {quadrants:02x?} by quadrant and depths \
                 exact, the colour under write mask none as cleared; bit draws {counted}"
            );
        });
    }
}
