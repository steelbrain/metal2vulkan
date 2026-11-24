//! Render targets at an icon's real size, every texel compared: textures over buffers in one
//! Shared block of rt's own that a pass clears and draws into part of.
//!
//! Under a device cfg, on every device of the cfg's kind (`on_device`), with the readers and the
//! cells of `real_size.rs`:
//!
//! - `every_device_renders_icon_targets_over_one_shared_block_every_texel`: textures over buffers
//!   in one 16 MiB Shared block of rt's own, each made after the CPU wrote the complement of its
//!   pattern into its rows: `BGRA8Unorm` at 256x256 in rows of 1,024 bytes, read, written and a
//!   render target; `R8Unorm` at 256x256 in rows of 256 bytes, also a pixel format view, with an
//!   `A8Unorm` view; and `RGB10A2Unorm` at 258x258 in rows of 1,280 bytes, read, written and a
//!   render target. Four lie side by side at 4 KiB places in the block (the `RGB10A2Unorm` one
//!   twice); then, the `BGRA8Unorm` one released, two more over the bytes it rendered. Each is
//!   sampled (the complement), then drawn by a pass whose attachment 0 it is (`Clear`, `Store`)
//!   and whose attachment 1 is a Private `RG16Float` target of its size (`DontCare`, `DontCare`;
//!   for one `RGB10A2Unorm` placement `Clear`, `DontCare`), its one draw cut by a scissor
//!   rectangle that leaves 37, 41, 45 and 39 texels of the clear on the left, right, top and
//!   bottom, then by a blit encoder's `synchronizeResource` of it. Then three reads, each its own
//!   cell: the block's bytes as the CPU reads them, a blit of the texture into another buffer, and
//!   a sampled read (and the view's), every texel the pattern inside the rectangle and the clear
//!   outside it; and the block's bytes outside the texture's texels, which no pass may change. At
//!   the end the whole block, every byte. Each texture's line names its form (in place, with its
//!   linear image's row pitch, offset and size, or copied for its cause, with what the device's
//!   linear tiling lacks of its optimal tiling's features for its usage and the linear image's
//!   size and alignment), the layout the pass renders it in, its home between commands (where the
//!   CPU reads it), the pass's areas and the over-copies counted. A clear the bytes miss, a draw
//!   that lands where the clear did not, a copy back at the wrong pitch or place, or a stray write
//!   past the rows each shows as texels or bytes not exact.
//!
//! `llvm-as` assembles the shaders; without it the tests skip visibly.

#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod on_device {
    use crate::pass::Pass;
    use crate::tests::command::on_device::{each_device, finished};
    use crate::tests::real_size::on_device::{copy_out, sampled, Cells, Pack, Readers};
    use crate::tests::real_size::{complement, unorm_pattern};
    use crate::texture::{kept_features, TextureObject};
    use crate::{
        Block, ClearColor, ColorAttachmentDescriptor, CommandBuffer, CommandQueue, Device, Error,
        FunctionDescriptor, IndexClass, Library, LoadAction, MemoryClass, PixelFormat,
        PrimitiveType, RenderPassDescriptor, RenderPipelineDescriptor, RenderPipelineState,
        Resource, ScissorRect, StageSource, StorageMode, StoreAction, Texture, TextureDescriptor,
        TextureType, TextureUsage,
    };
    use ash::vk;
    use metal2vulkan_devtools::{llvm_assemble_text, AssembleError};
    use std::ptr::NonNull;

    /// The owned shaders. `cover` draws the triangle that covers clip space. Each fragment writes
    /// render target 0 and, to render target 1, its pixel's position: `unorm8_pair` the 8-bit
    /// pattern (`real_size.rs`'s `unorm_pattern(8, 8, x, y)`) as `(k + 0.25) / 255`,
    /// `alpha8_pair` that pattern's alpha in every channel, and `unorm10_pair`
    /// `unorm_pattern(10, 2, x, y)` as `(k + 0.25) / 1023`, its alpha `(k + 0.25) / 3`.
    /// `UNORM8` and `UNORM10` stand for the lines that make `%v`, the pattern at `(%x, %y)`.
    const SHADERS: &str = r#"; Owned synthetic shaders.
source_filename = "real_size_renders.metal"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

define <{ <4 x float>, <4 x float> }> @cover(i32 noundef %id) {
entry:
  %one = icmp eq i32 %id, 1
  %two = icmp eq i32 %id, 2
  %px = select i1 %one, float 3.000000e+00, float -1.000000e+00
  %py = select i1 %two, float 3.000000e+00, float -1.000000e+00
  %p0 = insertelement <4 x float> <float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 1.000000e+00>, float %px, i64 0
  %p = insertelement <4 x float> %p0, float %py, i64 1
  %o0 = insertvalue <{ <4 x float>, <4 x float> }> undef, <4 x float> %p, 0
  %o = insertvalue <{ <4 x float>, <4 x float> }> %o0, <4 x float> zeroinitializer, 1
  ret <{ <4 x float>, <4 x float> }> %o
}

define <{ <4 x float>, <4 x float> }> @unorm8_pair(<4 x float> %pos) {
entry:
  %xy = shufflevector <4 x float> %pos, <4 x float> poison, <2 x i32> <i32 0, i32 1>
  %pi = tail call <2 x i32> @air.convert.u.v2i32.f.v2f32(<2 x float> %xy)
  %x = extractelement <2 x i32> %pi, i64 0
  %y = extractelement <2 x i32> %pi, i64 1
UNORM8
  %o0 = insertvalue <{ <4 x float>, <4 x float> }> undef, <4 x float> %v, 0
  %o = insertvalue <{ <4 x float>, <4 x float> }> %o0, <4 x float> %pos, 1
  ret <{ <4 x float>, <4 x float> }> %o
}

define <{ <4 x float>, <4 x float> }> @alpha8_pair(<4 x float> %pos) {
entry:
  %xy = shufflevector <4 x float> %pos, <4 x float> poison, <2 x i32> <i32 0, i32 1>
  %pi = tail call <2 x i32> @air.convert.u.v2i32.f.v2f32(<2 x float> %xy)
  %x = extractelement <2 x i32> %pi, i64 0
  %y = extractelement <2 x i32> %pi, i64 1
UNORM8
  %a = shufflevector <4 x float> %v, <4 x float> poison, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
  %o0 = insertvalue <{ <4 x float>, <4 x float> }> undef, <4 x float> %a, 0
  %o = insertvalue <{ <4 x float>, <4 x float> }> %o0, <4 x float> %pos, 1
  ret <{ <4 x float>, <4 x float> }> %o
}

define <{ <4 x float>, <4 x float> }> @unorm10_pair(<4 x float> %pos) {
entry:
  %xy = shufflevector <4 x float> %pos, <4 x float> poison, <2 x i32> <i32 0, i32 1>
  %pi = tail call <2 x i32> @air.convert.u.v2i32.f.v2f32(<2 x float> %xy)
  %x = extractelement <2 x i32> %pi, i64 0
  %y = extractelement <2 x i32> %pi, i64 1
UNORM10
  %o0 = insertvalue <{ <4 x float>, <4 x float> }> undef, <4 x float> %v, 0
  %o = insertvalue <{ <4 x float>, <4 x float> }> %o0, <4 x float> %pos, 1
  ret <{ <4 x float>, <4 x float> }> %o
}

declare <2 x i32> @air.convert.u.v2i32.f.v2f32(<2 x float>)

!air.vertex = !{!0}
!air.fragment = !{!6, !12, !13}
!air.version = !{!14}
!air.language_version = !{!15}
!0 = !{ptr @cover, !1, !4}
!1 = !{!2, !3}
!2 = !{!"air.position", !"air.arg_type_name", !"float4", !"air.arg_name", !"position"}
!3 = !{!"air.vertex_output", !"generated(5colorDv4_f)", !"air.arg_type_name", !"float4", !"air.arg_name", !"color"}
!4 = !{!5}
!5 = !{i32 0, !"air.vertex_id", !"air.arg_type_name", !"uint", !"air.arg_name", !"id"}
!6 = !{ptr @unorm8_pair, !7, !10}
!7 = !{!8, !9}
!8 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4", !"air.arg_name", !"first"}
!9 = !{!"air.render_target", i32 1, i32 0, !"air.arg_type_name", !"float4", !"air.arg_name", !"second"}
!10 = !{!11}
!11 = !{i32 0, !"air.position", !"air.center", !"air.no_perspective", !"air.arg_type_name", !"float4", !"air.arg_name", !"pos"}
!12 = !{ptr @alpha8_pair, !7, !10}
!13 = !{ptr @unorm10_pair, !7, !10}
!14 = !{i32 2, i32 8, i32 0}
!15 = !{!"Metal", i32 4, i32 0, i32 0}
"#;

    /// The lines that make `%v`, the 8-bit pattern at `(%x, %y)`, each channel's `k` as
    /// `(k + 0.25) / 255`.
    const UNORM8: &str = r#"  %k0 = and i32 %x, 255
  %k1 = and i32 %y, 255
  %xh = lshr i32 %x, 8
  %yh = lshr i32 %y, 8
  %yh4 = shl i32 %yh, 4
  %k2 = or i32 %xh, %yh4
  %xy.x = xor i32 %x, %y
  %k3 = and i32 %xy.x, 255
  %f0 = uitofp i32 %k0 to float
  %f1 = uitofp i32 %k1 to float
  %f2 = uitofp i32 %k2 to float
  %f3 = uitofp i32 %k3 to float
  %kv0 = insertelement <4 x float> poison, float %f0, i64 0
  %kv1 = insertelement <4 x float> %kv0, float %f1, i64 1
  %kv2 = insertelement <4 x float> %kv1, float %f2, i64 2
  %kv = insertelement <4 x float> %kv2, float %f3, i64 3
  %kq = fadd <4 x float> %kv, <float 2.500000e-01, float 2.500000e-01, float 2.500000e-01, float 2.500000e-01>
  %v = fdiv <4 x float> %kq, <float 2.550000e+02, float 2.550000e+02, float 2.550000e+02, float 2.550000e+02>
"#;

    /// The lines that make `%v`, the 10-bit pattern at `(%x, %y)`, each colour channel's `k` as
    /// `(k + 0.25) / 1023` and its alpha's as `(k + 0.25) / 3`.
    const UNORM10: &str = r#"  %k0 = and i32 %x, 1023
  %k1 = and i32 %y, 1023
  %xh = lshr i32 %x, 10
  %yh = lshr i32 %y, 10
  %yh2 = shl i32 %yh, 2
  %k2 = or i32 %xh, %yh2
  %xy.x = xor i32 %x, %y
  %k3 = and i32 %xy.x, 3
  %f0 = uitofp i32 %k0 to float
  %f1 = uitofp i32 %k1 to float
  %f2 = uitofp i32 %k2 to float
  %f3 = uitofp i32 %k3 to float
  %kv0 = insertelement <4 x float> poison, float %f0, i64 0
  %kv1 = insertelement <4 x float> %kv0, float %f1, i64 1
  %kv2 = insertelement <4 x float> %kv1, float %f2, i64 2
  %kv = insertelement <4 x float> %kv2, float %f3, i64 3
  %kq = fadd <4 x float> %kv, <float 2.500000e-01, float 2.500000e-01, float 2.500000e-01, float 2.500000e-01>
  %v = fdiv <4 x float> %kq, <float 1.023000e+03, float 1.023000e+03, float 1.023000e+03, float 3.000000e+00>
"#;

    /// The two libraries the device tests take: `real_size.rs`'s readers and [`SHADERS`]; None
    /// where `llvm-as` is missing.
    fn libraries() -> Option<(Library, Library)> {
        let readers = crate::tests::real_size::on_device::library()?;
        let text = (SHADERS.replace("UNORM8\n", UNORM8)).replace("UNORM10\n", UNORM10);
        match llvm_assemble_text(&text) {
            Ok(bitcode) => Some((readers, Library::from_air(&bitcode).expect("the library"))),
            Err(AssembleError::Missing(why)) => {
                eprintln!("SKIPPED: {why}");
                None
            }
            Err(error) => panic!("the real-size render shaders: {error:?}"),
        }
    }

    /// How a render target's texels are held, drawn and read.
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Kind {
        /// `BGRA8Unorm`, of the 8-bit pattern.
        Bgra8,
        /// `R8Unorm`, of the 8-bit pattern's alpha, `(x ^ y) & 255`.
        R8,
        /// `RGB10A2Unorm`, of `unorm_pattern(10, 2, x, y)`.
        Rgb10a2,
    }

    impl Kind {
        fn format(self) -> PixelFormat {
            match self {
                Kind::Bgra8 => PixelFormat::BGRA8Unorm,
                Kind::R8 => PixelFormat::R8Unorm,
                Kind::Rgb10a2 => PixelFormat::RGB10A2Unorm,
            }
        }

        /// A texel's bytes.
        fn texel(self) -> u32 {
            match self {
                Kind::R8 => 1,
                Kind::Bgra8 | Kind::Rgb10a2 => 4,
            }
        }

        /// The fragment that draws the pattern.
        fn fragment(self) -> &'static str {
            match self {
                Kind::Bgra8 => "unorm8_pair",
                Kind::R8 => "alpha8_pair",
                Kind::Rgb10a2 => "unorm10_pair",
            }
        }

        /// The pattern's channels at `(x, y)`, or their complement, in shader order; an `R8Unorm`
        /// texel's red alone.
        fn channels(self, x: u32, y: u32, complemented: bool) -> [u32; 4] {
            let flip = |bits: u32, k: u32| if complemented { complement(bits, k) } else { k };
            match self {
                Kind::Bgra8 => unorm_pattern(8, 8, x, y).map(|k| flip(8, k)),
                Kind::R8 => [flip(8, unorm_pattern(8, 8, x, y)[3]), 0, 0, 0],
                Kind::Rgb10a2 => {
                    let [r, g, b, a] = unorm_pattern(10, 2, x, y);
                    [flip(10, r), flip(10, g), flip(10, b), flip(2, a)]
                }
            }
        }

        /// The channels the pass clears attachment 0 to: a `BGRA8Unorm` and an `RGB10A2Unorm`
        /// texel that neither the pattern nor its complement holds at these sizes; an `R8Unorm`
        /// texel's one channel takes every value, so its clear meets both at some texels.
        fn clear(self) -> [u32; 4] {
            match self {
                Kind::Bgra8 => [32, 96, 160, 224],
                Kind::R8 => [32, 0, 0, 0],
                Kind::Rgb10a2 => [32, 96, 160, 2],
            }
        }

        /// The clear colour of [`clear`](Self::clear): `(k + 0.25)` over each channel's largest.
        fn clear_color(self) -> ClearColor {
            let [r, g, b, a] = self.clear().map(|k| f64::from(k) + 0.25);
            match self {
                Kind::Rgb10a2 => ClearColor::new(r / 1023.0, g / 1023.0, b / 1023.0, a / 3.0),
                Kind::Bgra8 | Kind::R8 => {
                    ClearColor::new(r / 255.0, g / 255.0, b / 255.0, a / 255.0)
                }
            }
        }

        /// The bytes in memory of a texel of channels `k`: `BGRA8Unorm` blue, green, red, alpha;
        /// `R8Unorm` red; `RGB10A2Unorm` one little-endian word, red lowest.
        fn bytes(self, [r, g, b, a]: [u32; 4]) -> Vec<u8> {
            match self {
                Kind::Bgra8 => [b, g, r, a].map(|c| c as u8).to_vec(),
                Kind::R8 => vec![r as u8],
                Kind::Rgb10a2 => (r | (g << 10) | (b << 20) | (a << 30))
                    .to_le_bytes()
                    .to_vec(),
            }
        }

        /// A reader's word of a texel of channels `k`, as [`pack`](Self::pack) packs it.
        fn word(self, [r, g, b, a]: [u32; 4]) -> Vec<u8> {
            let word = match self {
                Kind::Bgra8 => r | (g << 8) | (b << 16) | (a << 24),
                Kind::R8 => r,
                Kind::Rgb10a2 => r | (g << 10) | (b << 20) | (a << 30),
            };
            word.to_le_bytes().to_vec()
        }

        /// How a reader packs a texel: each channel's bits from bit 0 up.
        fn pack(self) -> Pack {
            match self {
                Kind::Bgra8 => Pack::unorm(8, 4),
                Kind::R8 => Pack::unorm(8, 1),
                Kind::Rgb10a2 => Pack {
                    scale: [1023.0, 1023.0, 1023.0, 3.0],
                    shift: [0, 10, 20, 30],
                    mask: [0x3ff, 0x3ff << 10, 0x3ff << 20, 3 << 30],
                },
            }
        }
    }

    /// How a reader packs an `A8Unorm` view's alpha: bits 0 to 7.
    const ALPHA: Pack = Pack {
        scale: [0.0, 0.0, 0.0, 255.0],
        shift: [0; 4],
        mask: [0, 0, 0, 255],
    };

    /// A render target over a buffer: its texels, size, rows' bytes and usage.
    struct Shape {
        what: &'static str,
        kind: Kind,
        size: [u32; 2],
        row: u64,
        usage: TextureUsage,
    }

    /// Read, written and rendered into.
    const DRAWN: TextureUsage = TextureUsage::SHADER_READ
        .union(TextureUsage::SHADER_WRITE)
        .union(TextureUsage::RENDER_TARGET);

    const BGRA8: Shape = Shape {
        what: "BGRA8Unorm 256x256 rows 1024",
        kind: Kind::Bgra8,
        size: [256, 256],
        row: 1024,
        usage: DRAWN,
    };

    const R8: Shape = Shape {
        what: "R8Unorm 256x256 rows 256 (with an A8Unorm view)",
        kind: Kind::R8,
        size: [256, 256],
        row: 256,
        usage: DRAWN.union(TextureUsage::PIXEL_FORMAT_VIEW),
    };

    const RGB10A2: Shape = Shape {
        what: "RGB10A2Unorm 258x258 rows 1280",
        kind: Kind::Rgb10a2,
        size: [258, 258],
        row: 1280,
        usage: DRAWN,
    };

    /// The block the textures lie in: 16 MiB.
    const BLOCK: u64 = 16 << 20;

    /// The bytes of the block no texel holds.
    const PAD: u8 = 0x5a;

    /// A texture's place: its shape, its buffer's first byte in the block, and the load action
    /// of the pass's attachment 1.
    type Place = (&'static Shape, u64, LoadAction);

    /// The textures side by side in the block, at 4 KiB places, each made before any is drawn.
    const SIDE_BY_SIDE: [Place; 4] = [
        (&R8, 163_840, LoadAction::DontCare),
        (&BGRA8, 905_216, LoadAction::DontCare),
        (&RGB10A2, 2_744_320, LoadAction::DontCare),
        (&RGB10A2, 4_665_344, LoadAction::Clear),
    ];

    /// Which of [`SIDE_BY_SIDE`] is released before [`OVER_IT`] are made over its bytes.
    const RELEASED: usize = 1;

    /// The textures made over the bytes of [`SIDE_BY_SIDE`]'s `BGRA8Unorm` one once it is
    /// released, which they share between them.
    const OVER_IT: [Place; 2] = [
        (&R8, 892_928, LoadAction::DontCare),
        (&RGB10A2, 958_464, LoadAction::DontCare),
    ];

    /// The texels of the clear the draw leaves on the left, right, top and bottom: each past 32,
    /// none a multiple of it.
    const BORDER: [u32; 4] = [37, 41, 45, 39];

    /// The rectangle a pass's draw is cut to: `x`, `y`, width and height.
    fn drawn(size: [u32; 2]) -> [u32; 4] {
        let [left, right, top, bottom] = BORDER;
        [left, top, size[0] - left - right, size[1] - top - bottom]
    }

    /// What `(x, y)` holds after the pass: the pattern inside [`drawn`], the clear outside it.
    fn after(kind: Kind, size: [u32; 2], x: u32, y: u32) -> [u32; 4] {
        let [left, top, width, height] = drawn(size);
        let inside = (left..left + width).contains(&x) && (top..top + height).contains(&y);
        match inside {
            true => kind.channels(x, y, false),
            false => kind.clear(),
        }
    }

    /// Every texel's `of(x, y)`, rows from the top.
    fn each(size: [u32; 2], of: impl Fn(u32, u32) -> Vec<u8>) -> Vec<u8> {
        (0..size[1])
            .flat_map(|y| (0..size[0]).map(move |x| (x, y)))
            .flat_map(|(x, y)| of(x, y))
            .collect()
    }

    /// The block's bytes, written by the CPU through its mapping, and the bytes it should hold.
    struct Held {
        pointer: NonNull<u8>,
        expected: Vec<u8>,
    }

    impl Held {
        /// The block's bytes, every one [`PAD`].
        fn new(block: &Block) -> Held {
            let pointer = block.contents.expect("a Shared block");
            let held = Held {
                pointer,
                expected: vec![PAD; BLOCK as usize],
            };
            // SAFETY: the block's `BLOCK` bytes, which no command reaches yet.
            unsafe { std::ptr::write_bytes(pointer.as_ptr(), PAD, BLOCK as usize) };
            held
        }

        /// Writes `bytes` from `at`, through the mapping, while no command reaches them.
        fn write(&mut self, at: usize, bytes: &[u8]) {
            // SAFETY: `at` and the bytes lie in the block, and no command reaches them now.
            unsafe {
                let to = self.pointer.as_ptr().add(at);
                std::ptr::copy_nonoverlapping(bytes.as_ptr(), to, bytes.len());
            }
            self.expected[at..at + bytes.len()].copy_from_slice(bytes);
        }

        /// The block's bytes, once no command reaches them.
        fn read(&self) -> Vec<u8> {
            // SAFETY: the block's `BLOCK` bytes; every command that reached them completed.
            unsafe { std::slice::from_raw_parts(self.pointer.as_ptr(), BLOCK as usize) }.to_vec()
        }

        /// Sets the texels of `shape` at `at` to `of(x, y)`'s bytes: through the mapping where
        /// `written`, else only in what the block should hold, as a pass writes them.
        fn texels(
            &mut self,
            (shape, at): (&Shape, u64),
            of: impl Fn(u32, u32) -> [u32; 4],
            written: bool,
        ) {
            let kind = shape.kind;
            for y in 0..shape.size[1] {
                let row: Vec<u8> = (0..shape.size[0])
                    .flat_map(|x| kind.bytes(of(x, y)))
                    .collect();
                let start = (at + u64::from(y) * shape.row) as usize;
                match written {
                    true => self.write(start, &row),
                    false => self.expected[start..start + row.len()].copy_from_slice(&row),
                }
            }
        }
    }

    /// The texels of `shape` at `at` in `bytes`, a block's, in packed rows.
    fn texels_in(bytes: &[u8], (shape, at): (&Shape, u64)) -> Vec<u8> {
        let row = (shape.size[0] * shape.kind.texel()) as usize;
        (0..shape.size[1])
            .flat_map(|y| {
                let start = (at + u64::from(y) * shape.row) as usize;
                bytes[start..start + row].iter().copied()
            })
            .collect()
    }

    /// `bytes`, a block's, with the texels of `shape` at `at` zeroed: the bytes outside them.
    fn outside(mut bytes: Vec<u8>, (shape, at): (&Shape, u64)) -> Vec<u8> {
        let row = (shape.size[0] * shape.kind.texel()) as usize;
        for y in 0..shape.size[1] {
            let start = (at + u64::from(y) * shape.row) as usize;
            bytes[start..start + row].fill(0);
        }
        bytes
    }

    /// An error's text, with its variant.
    fn said(error: Error) -> String {
        format!("{error} ({error:?})")
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

    /// The render state of `cover` and `kind`'s fragment into attachment 0 of `kind`'s format
    /// and attachment 1 of `RG16Float`.
    fn drawing(name: &str, device: &Device, library: &Library, kind: Kind) -> RenderPipelineState {
        let function = |f: &str| {
            StageSource::Function(FunctionDescriptor::new(
                library.function(f).expect("the function"),
            ))
        };
        let fragment = kind.fragment();
        let mut d = RenderPipelineDescriptor::new(function("cover"));
        d.fragment = Some(function(fragment));
        d.color_attachments[0] = Some(ColorAttachmentDescriptor::new(kind.format()));
        d.color_attachments[1] = Some(ColorAttachmentDescriptor::new(PixelFormat::RG16Float));
        let state = (device.new_render_pipeline_state(&d, crate::tests::no_constexpr_samplers))
            .unwrap_or_else(|error| panic!("{name}: {fragment}: {error}"));
        (state.finish_pipeline()).unwrap_or_else(|error| panic!("{name}: {fragment}: {error}"));
        state
    }

    /// How `texture`, over a buffer, holds its texels: in place, with its linear image's row
    /// pitch, offset and size; or copied for its cause, with the format features the device's
    /// linear tiling lacks of its optimal tiling's for the image's usage, and the size and
    /// alignment of the linear image the device would make.
    fn held_as(device: &Device, texture: &Texture) -> String {
        let core = device.core();
        if let Some((_, copy)) = texture.copied_over() {
            let (linear, optimal) = core.format_features(texture.image_format());
            let kept = kept_features(texture.object().image_usage());
            let d = texture.descriptor();
            return format!(
                "copied for {:?} (the linear tiling lacks {:?}; a linear image's subresource size \
                 {:?}, memory alignment {:?})",
                copy.kind.cause,
                optimal & kept & !linear,
                device.linear_subresource_size(d),
                device.linear_memory_alignment(d)
            );
        }
        let color = vk::ImageSubresource {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            mip_level: 0,
            array_layer: 0,
        };
        // SAFETY: the texture's image, linear, of one level and layer of a colour format.
        let layout = unsafe { (core.device).get_image_subresource_layout(texture.image(), color) };
        format!(
            "in place, row pitch {} offset {} size {}",
            layout.row_pitch, layout.offset, layout.size
        )
    }

    /// The pass over `texture`: attachment 0 it, cleared to `kind`'s clear and stored;
    /// attachment 1 `second`, loaded by its action and not stored.
    fn pass_of(
        texture: &Texture,
        kind: Kind,
        (second, load): (&Texture, LoadAction),
    ) -> RenderPassDescriptor {
        let mut pass = RenderPassDescriptor::default();
        let first = &mut pass.color_attachments[0];
        first.texture = Some(texture.clone());
        first.load_action = LoadAction::Clear;
        first.store_action = StoreAction::Store;
        first.clear_color = kind.clear_color();
        let other = &mut pass.color_attachments[1];
        other.texture = Some(second.clone());
        other.load_action = load;
        other.store_action = StoreAction::DontCare;
        other.clear_color = ClearColor::new(0.5, 0.25, 0.0, 0.0);
        pass
    }

    /// The layout `pass` renders its attachment 0 in, that image's home between commands, and
    /// the pass's areas, as the runtime checks the pass.
    fn laid(device: &Device, pass: &RenderPassDescriptor) -> String {
        match Pass::check(device.core(), pass) {
            Ok(checked) => {
                let target = checked.colors[0].as_ref().expect("attachment 0");
                format!(
                    "rendered in {:?}, home {:?}, Metal's area {:?}, the load's and store's reach \
                     {:?}",
                    target.layout, target.attachment.home, checked.area, checked.reach
                )
            }
            Err(error) => format!("the pass refused: {}", said(error)),
        }
    }

    /// `pass`, its one draw by `state` cut to [`drawn`], then a blit encoder's
    /// `synchronizeResource` of `texture`, in one command buffer, committed and waited for.
    fn render(
        name: &str,
        queue: &CommandQueue,
        state: &RenderPipelineState,
        (pass, texture, size): (&RenderPassDescriptor, &Texture, [u32; 2]),
    ) -> Result<(), String> {
        let [x, y, width, height] = drawn(size).map(u64::from);
        let command_buffer = queue.command_buffer().map_err(said)?;
        let mut encoder = (command_buffer.render_command_encoder(pass)).map_err(said)?;
        encoder.set_render_pipeline_state(state).map_err(said)?;
        encoder.set_scissor_rect(ScissorRect {
            x,
            y,
            width,
            height,
        });
        (encoder.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)).map_err(said)?;
        encoder.end_encoding();
        let mut blit = command_buffer.blit_command_encoder().map_err(said)?;
        (blit.synchronize_resource(Resource::Texture(texture))).map_err(said)?;
        blit.end_encoding();
        completed(name, &command_buffer)
    }

    /// A texture over a buffer at its place in the block, which holds the buffer, and its
    /// `A8Unorm` view where it is `R8Unorm`, each at the next of `indices`.
    struct Made {
        place: Place,
        texture: Texture,
        view: Option<Texture>,
    }

    impl Made {
        fn new(
            name: &str,
            device: &Device,
            block: &Block,
            place: Place,
            indices: &mut impl Iterator<Item = u32>,
        ) -> Made {
            let (shape, at, _) = place;
            let what = format!("{} at byte {at}", shape.what);
            let len = shape.row * u64::from(shape.size[1]);
            let buffer = (device.new_buffer(block.id, at, len, None))
                .unwrap_or_else(|error| panic!("{name}: {what}: its buffer: {error}"));
            let [width, height] = shape.size;
            let format = shape.kind.format();
            let mut d = TextureDescriptor::new(TextureType::Type2D, format, width, height);
            d.usage = shape.usage;
            let texture = (buffer.new_texture(&d, indices.next(), 0, shape.row))
                .unwrap_or_else(|error| panic!("{name}: {what}: {error}"));
            let view = (shape.kind == Kind::R8).then(|| {
                let views = PixelFormat::A8Unorm;
                (texture.new_texture_view(views, TextureType::Type2D, 0..1, 0..1, indices.next()))
                    .unwrap_or_else(|error| panic!("{name}: {what}: its A8Unorm view: {error}"))
            });
            Made {
                place,
                texture,
                view,
            }
        }
    }

    /// What one texture's run takes: the readers, the queue, the render states by kind and the
    /// Private `RG16Float` attachments 1 by side.
    struct Rig<'a> {
        name: &'a str,
        device: &'a Device,
        queue: &'a CommandQueue,
        readers: Readers,
        states: [(Kind, RenderPipelineState); 3],
        seconds: [(u32, Texture); 2],
    }

    impl Rig<'_> {
        /// `made`'s texture sampled (the complement), drawn by its pass and synchronized, then
        /// read by the CPU from the block, by a blit out and by sampling, each a cell, the
        /// block's bytes outside its texels a cell too; what the block should hold updated.
        fn run(&self, made: &Made, held: &mut Held, cells: &mut Cells) {
            let (name, device) = (self.name, self.device);
            let on = (device, self.queue);
            let (shape, at, load) = made.place;
            let (kind, size) = (shape.kind, shape.size);
            let side = size[0];
            let second = &self
                .seconds
                .iter()
                .find(|(s, _)| *s == side)
                .expect("its side")
                .1;
            let state = &self
                .states
                .iter()
                .find(|(k, _)| *k == kind)
                .expect("its kind")
                .1;
            let form = held_as(device, &made.texture);
            let what = format!(
                "{} at byte {at}, attachment 1 RG16Float {side}x{side} {load:?}/DontCare ({form})",
                shape.what
            );
            let cell = |how: &str| format!("{what}: {how}");
            let width = size[0];
            let words = |of: &dyn Fn(u32, u32) -> [u32; 4]| each(size, |x, y| kind.word(of(x, y)));
            let stale = words(&|x, y| kind.channels(x, y, true));
            let at_size = (&made.texture, size);
            let got = sampled(name, on, &self.readers, at_size, &kind.pack(), &stale);
            cells.check(
                &cell("sampled first, the CPU's complement"),
                got,
                &stale,
                (4, width),
            );
            if let Some(view) = &made.view {
                let got = sampled(name, on, &self.readers, (view, size), &ALPHA, &stale);
                let how = "its A8Unorm view sampled first, the CPU's complement";
                cells.check(&cell(how), got, &stale, (4, width));
            }
            let pass = pass_of(&made.texture, kind, (second, load));
            let layouts = laid(device, &pass);
            let copies = || device.counters().metrics().over_copies;
            let before = copies();
            let rendered = render(name, self.queue, state, (&pass, &made.texture, size));
            println!(
                "real-size: {name}: {what}: {layouts}; {} over-copies counted in its pass",
                copies() - before
            );
            if let Err(why) = rendered {
                cells.check(&cell("the pass"), Err(why), &[], (1, 1));
                return;
            }
            held.texels((shape, at), |x, y| after(kind, size, x, y), false);
            let texel = kind.texel();
            let image = each(size, |x, y| kind.bytes(after(kind, size, x, y)));
            let block = held.read();
            let place = (shape, at);
            let got = Ok(texels_in(&block, place));
            let how = "(a) the block's bytes the CPU reads after synchronizeResource";
            cells.check(&cell(how), got, &image, (texel as usize, width));
            let got = copy_out(name, on, at_size, texel, &image);
            let how = "(b) a blit of the texture into another buffer";
            cells.check(&cell(how), got, &image, (texel as usize, width));
            let want = words(&|x, y| after(kind, size, x, y));
            let got = sampled(name, on, &self.readers, at_size, &kind.pack(), &want);
            cells.check(&cell("(c) sampled last"), got, &want, (4, width));
            if let Some(view) = &made.view {
                let got = sampled(name, on, &self.readers, (view, size), &ALPHA, &want);
                let how = "(c) its A8Unorm view sampled last";
                cells.check(&cell(how), got, &want, (4, width));
            }
            let got = Ok(outside(block, place));
            let want = outside(held.expected.clone(), place);
            let how = "the block's bytes outside its texels, each byte (x of 4 KiB page y)";
            cells.check(&cell(how), got, &want, (1, 4096));
        }
    }

    /// Render targets over buffers in one Shared block of rt's own, side by side and then over
    /// the bytes of one released, each sampled, drawn in part by a pass that clears it, and
    /// synchronized; then read by the CPU from the block, by a blit out and by sampling, every
    /// texel the pattern inside the draw's rectangle and the clear outside it, and the block's
    /// other bytes as they were; at the end the whole block. Each line names the texture's form.
    #[test]
    fn every_device_renders_icon_targets_over_one_shared_block_every_texel() {
        let Some((packed, own)) = libraries() else {
            return;
        };
        assert!(BORDER.iter().all(|b| *b > 32 && b % 32 != 0), "{BORDER:?}");
        each_device("real_size_block_renders", |name, _, device| {
            let first = IndexClass::Texture.range().start;
            (device.grant_indices(IndexClass::Texture, first, 10))
                .unwrap_or_else(|error| panic!("{name}: a texture grant: {error}"));
            let mut indices = first..first + 10;
            let queue = device.new_command_queue(4).expect("a queue");
            let second = |side: u32, index: Option<u32>| {
                let format = PixelFormat::RG16Float;
                let mut d = TextureDescriptor::new(TextureType::Type2D, format, side, side);
                d.usage = TextureUsage::SHADER_READ | TextureUsage::RENDER_TARGET;
                d.storage_mode = StorageMode::Private;
                (device.new_texture(&d, index))
                    .unwrap_or_else(|error| panic!("{name}: RG16Float {side}x{side}: {error}"))
            };
            let rig = Rig {
                name,
                device: &device,
                queue: &queue,
                readers: Readers::new(name, &device, &packed),
                states: [Kind::Bgra8, Kind::R8, Kind::Rgb10a2]
                    .map(|kind| (kind, drawing(name, &device, &own, kind))),
                seconds: [256, 258].map(|side| (side, second(side, indices.next()))),
            };
            let block = (device.take_block(MemoryClass::Shared, BLOCK))
                .unwrap_or_else(|error| panic!("{name}: a Shared block of {BLOCK}: {error}"));
            let mut held = Held::new(&block);
            let mut cells = Cells::new(name);
            let stale = |(shape, at, _): Place, held: &mut Held| {
                let kind = shape.kind;
                held.texels((shape, at), |x, y| kind.channels(x, y, true), true);
            };
            for place in SIDE_BY_SIDE {
                stale(place, &mut held);
            }
            let mut made: Vec<Made> = (SIDE_BY_SIDE.iter())
                .map(|&place| Made::new(name, &device, &block, place, &mut indices))
                .collect();
            for one in &made {
                rig.run(one, &mut held, &mut cells);
            }
            drop(made.remove(RELEASED));
            for place in OVER_IT {
                stale(place, &mut held);
            }
            for &place in &OVER_IT {
                let one = Made::new(name, &device, &block, place, &mut indices);
                rig.run(&one, &mut held, &mut cells);
                made.push(one);
            }
            let how = "the whole block after every pass, each byte (x of 4 KiB page y)";
            cells.check(how, Ok(held.read()), &held.expected, (1, 4096));
            drop((made, rig));
            (device.release_block(block)).expect("a release");
            cells.all_exact();
        });
    }
}
