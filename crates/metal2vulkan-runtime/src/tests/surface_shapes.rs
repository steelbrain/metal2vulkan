//! Textures over an IOSurface plane beyond a 2D texture of one texel a block: a BC texture over a
//! surface's rows, and a texture buffer, a 1D texture and a 2D array of one slice over a plane of
//! a surface with planes (`linear_plan.rs`'s **Block-compressed surfaces** and **Beyond the
//! plane's rows**). Each reads and writes the bytes the x86 Macs' Metal does
//! (`surface_texture_envelope`; each shape names the `oracle_surftex` case or the probe's shape it
//! stands for): a kernel's reads, the blits, `getBytes` and `replaceRegion` take those bytes, a
//! kernel after a blit or a CPU write reads what they wrote, and no other byte of the surface
//! changes.
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. Each surface is a buffer standing for
//! its memory, as the consumer's is: a `Shared` buffer, and a no-copy buffer over host pages where
//! the device imports them. A BC1 block is solid, its two colours one RGB565 value naming the
//! block, and every block of the surface outside the texture's names none; another texture's
//! bytes name their offset.

use super::blit_holds::{device, over_host, Host, Scratch, WAIT};
use crate::{
    BlitCommandEncoder, BlitOption, Buffer, ComputePipelineState, Device, Error, IoSurfacePlane,
    Origin, PipelineOptions, Region, ResourceOptions, Size, Texture, TextureDescriptor,
};
use metal2vulkan::metal::{PixelFormat, TextureType, TextureUsage};
use std::sync::atomic::AtomicUsize;
use std::sync::Arc;

/// How a kernel reads a texture of one type: its argument's type, the read's call and
/// declaration, and whether its grid is 2D.
struct Read {
    arg_type: &'static str,
    call: &'static str,
    declare: &'static str,
    two_d: bool,
}

/// The reads of a 2D texture, a 2D array's slice 0, a 1D texture and a texture buffer.
const READS: [(TextureType, Read); 4] = [
    (
        TextureType::Type2D,
        Read {
            arg_type: "texture2d<float, read>",
            call: "@air.read_texture_2d.v4f32(ptr addrspace(1) %t, ptr addrspace(2) %s, <2 x i32> %gid, <2 x i32> zeroinitializer, i32 0, i32 1)",
            declare: "@air.read_texture_2d.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x i32>, <2 x i32>, i32, i32)",
            two_d: true,
        },
    ),
    (
        TextureType::Type2DArray,
        Read {
            arg_type: "texture2d_array<float, read>",
            call: "@air.read_texture_2d_array.v4f32(ptr addrspace(1) %t, ptr addrspace(2) %s, <2 x i32> %gid, i32 0, <2 x i32> zeroinitializer, i32 0, i32 1)",
            declare: "@air.read_texture_2d_array.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x i32>, i32, <2 x i32>, i32, i32)",
            two_d: true,
        },
    ),
    (
        TextureType::Type1D,
        Read {
            arg_type: "texture1d<float, read>",
            call: "@air.read_texture_1d.v4f32(ptr addrspace(1) %t, ptr addrspace(2) %s, i32 %gid, i32 0, i32 1)",
            declare: "@air.read_texture_1d.v4f32(ptr addrspace(1), ptr addrspace(2), i32, i32, i32)",
            two_d: false,
        },
    ),
    (
        TextureType::TextureBuffer,
        Read {
            arg_type: "texture_buffer<float, read>",
            call: "@air.read_texture_buffer_1d.v4f32(ptr addrspace(1) %t, ptr addrspace(2) %s, i32 %gid, i32 1)",
            declare: "@air.read_texture_buffer_1d.v4f32(ptr addrspace(1), ptr addrspace(2), i32, i32)",
            two_d: false,
        },
    ),
];

/// A kernel that reads every texel of texture 0 into buffer 0 through `read`, `float4`s in rows
/// of the grid.
fn kernel(read: &Read) -> String {
    let (gid, index, size_arg, size_meta, gid_type) = match read.two_d {
        true => (
            "<2 x i32> %gid, <2 x i32> %size",
            "  %x = extractelement <2 x i32> %gid, i64 0\n  %y = extractelement <2 x i32> %gid, i64 1\n  %w = extractelement <2 x i32> %size, i64 0\n  %row = mul i32 %y, %w\n  %i = add i32 %row, %x\n",
            ", !6",
            "!6 = !{i32 3, !\"air.threads_per_grid\", !\"air.arg_type_name\", !\"uint2\", !\"air.arg_name\", !\"size\"}\n",
            "uint2",
        ),
        false => ("i32 %gid", "  %i = add i32 %gid, 0\n", "", "", "uint"),
    };
    format!(
        r#"; Owned synthetic kernel.
source_filename = "surface_shapes.metal"

define void @read_texels(ptr addrspace(1) %t, ptr addrspace(1) noalias %out, {gid}) {{
entry:
  %s = tail call ptr addrspace(2) @air.get_read_sampler()
  %r = tail call {{ <4 x float>, i8 }} {call}
  %v = extractvalue {{ <4 x float>, i8 }} %r, 0
{index}  %j = zext i32 %i to i64
  %p = getelementptr inbounds <4 x float>, ptr addrspace(1) %out, i64 %j
  store <4 x float> %v, ptr addrspace(1) %p, align 16
  ret void
}}

declare ptr addrspace(2) @air.get_read_sampler()
declare {{ <4 x float>, i8 }} {declare}

!air.kernel = !{{!0}}
!air.version = !{{!30}}
!air.language_version = !{{!31}}
!0 = !{{ptr @read_texels, !1, !2}}
!1 = !{{}}
!2 = !{{!3, !4, !5{size_arg}}}
!3 = !{{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.read", !"air.arg_type_name", !"{arg_type}", !"air.arg_name", !"t"}}
!4 = !{{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4", !"air.arg_name", !"out"}}
!5 = !{{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"{gid_type}", !"air.arg_name", !"gid"}}
{size_meta}!30 = !{{i32 2, i32 8, i32 0}}
!31 = !{{!"Metal", i32 4, i32 0, i32 0}}
"#,
        call = read.call,
        declare = read.declare,
        arg_type = read.arg_type,
    )
}

/// The read kernel's state for textures of `texture_type` on `device`.
fn read_state(device: &Device, texture_type: TextureType) -> ComputePipelineState {
    let (_, read) = READS.iter().find(|(t, _)| *t == texture_type).unwrap();
    let air = metal2vulkan_devtools::llvm_assemble_text(&kernel(read)).unwrap();
    let library = metal2vulkan::library::Library::from_air(&air).unwrap();
    let function = device
        .new_library_with_compile_library(library)
        .new_function_with_name("read_texels")
        .unwrap();
    device
        .new_compute_pipeline_state_with_function(&function, &PipelineOptions::default())
        .unwrap()
}

/// A surface: its allocation's bytes, and its plane the texture lies over (offset,
/// `bytesPerRow`, of a surface with planes or not).
#[derive(Clone, Copy, Debug)]
struct Surface {
    length: usize,
    plane: (u64, u64, bool),
}

/// `IOSurfaceCreate`'s 64 x 64 `'BGRA'` surface without planes.
const BGRA: Surface = Surface {
    length: 16384,
    plane: (0, 256, false),
};
/// The same at 512 bytes a row.
const BGRA_PAD: Surface = Surface {
    length: 32768,
    plane: (0, 512, false),
};
/// A 64 x 64 `'BGRA'` surface of one plane.
const BGRA1: Surface = Surface {
    length: 16384,
    plane: (0, 256, true),
};
/// A 64 x 64 `'BGRA'` surface of one plane in 32768 bytes.
const BGRA1X2: Surface = Surface {
    length: 32768,
    plane: (0, 256, true),
};
/// A 64 x 64 `420v` CVPixelBuffer's luma plane on the x86 Macs.
const P0: Surface = Surface {
    length: 6432,
    plane: (32, 64, true),
};
/// Its chroma plane.
const P1: Surface = Surface {
    length: 6432,
    plane: (4128, 64, true),
};

/// One texture over a surface.
#[derive(Clone, Copy, Debug)]
struct Shape {
    /// The `oracle_surftex` case (its number) or the probe's shape it stands for.
    name: &'static str,
    pf: PixelFormat,
    texture_type: TextureType,
    size: [u64; 2],
    surface: Surface,
    private: bool,
}

const fn shape(
    name: &'static str,
    pf: PixelFormat,
    texture_type: TextureType,
    size: [u64; 2],
    surface: Surface,
) -> Shape {
    Shape {
        name,
        pf,
        texture_type,
        size,
        surface,
        private: false,
    }
}

impl Shape {
    /// Its unit's bytes and texels across and down.
    fn unit(&self) -> (u64, [u64; 2]) {
        match self.pf {
            PixelFormat::BC1RGBA | PixelFormat::BC4RUnorm => (8, [4, 4]),
            PixelFormat::BC3RGBA | PixelFormat::BC7RGBAUnorm => (16, [4, 4]),
            PixelFormat::BGRA8Unorm => (4, [1, 1]),
            PixelFormat::R8Unorm => (1, [1, 1]),
            pf => panic!("{pf:?}"),
        }
    }

    /// Its units across and down, and the bytes of a row of them.
    fn units(&self) -> ([u64; 2], u64) {
        let (bytes, [w, h]) = self.unit();
        let across = self.size[0].div_ceil(w);
        ([across, self.size[1].div_ceil(h)], across * bytes)
    }

    /// The surface's byte where row `row` of units starts.
    fn row_at(&self, row: u64) -> usize {
        let (offset, bytes_per_row, _) = self.surface.plane;
        // A texture buffer's texels run from the plane's offset on, however wide.
        match self.texture_type {
            TextureType::TextureBuffer => offset as usize,
            _ => (offset + row * bytes_per_row) as usize,
        }
    }

    /// Whether a kernel's reads are checked: a BC1 texture's solid blocks, or 8-bit unsigned
    /// normalized texels.
    fn read_checked(&self) -> bool {
        matches!(
            self.pf,
            PixelFormat::BC1RGBA | PixelFormat::BGRA8Unorm | PixelFormat::R8Unorm
        )
    }

    fn descriptor(&self) -> TextureDescriptor {
        let mut d = TextureDescriptor::texture_2d(self.pf, self.size[0], self.size[1], false);
        d.texture_type = self.texture_type;
        d.usage = TextureUsage::SHADER_READ;
        d.resource_options = match self.private {
            true => ResourceOptions::private(),
            false => ResourceOptions::shared(),
        };
        d
    }
}

/// Every shape: the BC textures over a surface's rows the x86 Macs make, and the other types
/// over a planar surface's plane.
fn shapes() -> Vec<Shape> {
    use PixelFormat as P;
    use TextureType as T;
    vec![
        shape("66 bgra.fmt.bc1", P::BC1RGBA, T::Type2D, [64, 64], BGRA),
        shape(
            "67 bgra.fmt.bc7",
            P::BC7RGBAUnorm,
            T::Type2D,
            [64, 64],
            BGRA,
        ),
        shape(
            "131 bgra.fmt.bc1.128x64",
            P::BC1RGBA,
            T::Type2D,
            [128, 64],
            BGRA,
        ),
        shape(
            "147 bgra.fmt.bc1.62x64",
            P::BC1RGBA,
            T::Type2D,
            [62, 64],
            BGRA,
        ),
        Shape {
            private: true,
            ..shape(
                "148 bgra.fmt.bc1.private",
                P::BC1RGBA,
                T::Type2D,
                [64, 64],
                BGRA,
            )
        },
        shape("bgrapad.bc1", P::BC1RGBA, T::Type2D, [64, 64], BGRA_PAD),
        shape("bgra.bc3", P::BC3RGBA, T::Type2D, [64, 64], BGRA),
        shape("bgra.bc4", P::BC4RUnorm, T::Type2D, [64, 64], BGRA),
        shape("125 bgra1.fmt.bc1", P::BC1RGBA, T::Type2D, [64, 64], BGRA1),
        shape("420v.p0.bc1.16x64", P::BC1RGBA, T::Type2D, [16, 64], P0),
        shape("420v.p1.bc1.16x32", P::BC1RGBA, T::Type2D, [16, 32], P1),
        shape(
            "121 bgra1.typebuffer",
            P::BGRA8Unorm,
            T::TextureBuffer,
            [64, 1],
            BGRA1,
        ),
        shape(
            "bgra1x2.texbuf.128",
            P::BGRA8Unorm,
            T::TextureBuffer,
            [128, 1],
            BGRA1X2,
        ),
        shape(
            "129 420v.p0.typebuffer",
            P::R8Unorm,
            T::TextureBuffer,
            [64, 1],
            P0,
        ),
        shape("116 bgra1.type1d", P::BGRA8Unorm, T::Type1D, [64, 1], BGRA1),
        shape(
            "117 bgra1.type2darray",
            P::BGRA8Unorm,
            T::Type2DArray,
            [64, 64],
            BGRA1,
        ),
        shape(
            "127 420v.p0.type2darray",
            P::R8Unorm,
            T::Type2DArray,
            [64, 64],
            P0,
        ),
    ]
}

/// A BC1 block whose two colours are `color`, its texels all colour 0.
fn solid(color: u16) -> [u8; 8] {
    let [lo, hi] = color.to_le_bytes();
    [lo, hi, lo, hi, 0, 0, 0, 0]
}

/// The colour no texture's block takes: the surface's blocks outside it.
const OUTSIDE: u16 = 0xffff;

/// Byte `i` of a surface's bytes, of `seed`, naming its offset.
fn named(i: usize, seed: u8) -> u8 {
    (i as u8).wrapping_mul(37) ^ (i >> 8) as u8 ^ seed
}

/// The surface's first bytes before a texture of `shape` is made over them: `checked` of them.
fn surface_bytes(shape: &Shape, checked: usize) -> Vec<u8> {
    let mut bytes: Vec<u8> = match shape.pf {
        PixelFormat::BC1RGBA => (0..checked.div_ceil(8))
            .flat_map(|_| solid(OUTSIDE))
            .take(checked)
            .collect(),
        _ => (0..checked).map(|i| named(i, 0)).collect(),
    };
    if shape.pf == PixelFormat::BC1RGBA {
        let ([across, down], _) = shape.units();
        for j in 0..down {
            for i in 0..across {
                let at = shape.row_at(j) + 8 * i as usize;
                bytes[at..at + 8].copy_from_slice(&solid(block_color(shape, i, j, 0)));
            }
        }
    }
    bytes
}

/// The colour naming block `(i, j)` of `shape`'s, of generation `seed`.
fn block_color(shape: &Shape, i: u64, j: u64, seed: u16) -> u16 {
    let ([across, _], _) = shape.units();
    (j * across + i + 1) as u16 + 0x800 * seed
}

/// The texture's rows of units, packed, out of the surface's `bytes`.
fn rows_of(shape: &Shape, bytes: &[u8]) -> Vec<u8> {
    let ([_, down], row) = shape.units();
    (0..down)
        .flat_map(|j| {
            let at = shape.row_at(j);
            bytes[at..at + row as usize].to_vec()
        })
        .collect()
}

/// Write `rows` (packed rows of units) into the surface's `bytes` where the texture's lie.
fn set_rows(shape: &Shape, bytes: &mut [u8], rows: &[u8]) {
    let (_, row) = shape.units();
    for (j, packed) in rows.chunks(row as usize).enumerate() {
        let at = shape.row_at(j as u64);
        bytes[at..at + packed.len()].copy_from_slice(packed);
    }
}

/// What a kernel reads at texel `(x, y)` of the texture's packed `rows`, as the channels' codes
/// it compares (a BC1 block's RGB565 colour, else the texel's 8-bit codes in RGBA order).
fn expected(shape: &Shape, rows: &[u8], (x, y): (u64, u64)) -> [u32; 4] {
    let (bytes, [bw, bh]) = shape.unit();
    let (_, row) = shape.units();
    let at = ((y / bh) * row + (x / bw) * bytes) as usize;
    match shape.pf {
        PixelFormat::BC1RGBA => {
            let c = u32::from(u16::from_le_bytes([rows[at], rows[at + 1]]));
            [c >> 11, (c >> 5) & 63, c & 31, 1]
        }
        PixelFormat::BGRA8Unorm => {
            [rows[at + 2], rows[at + 1], rows[at], rows[at + 3]].map(u32::from)
        }
        PixelFormat::R8Unorm => [u32::from(rows[at]), 0, 0, 255],
        pf => panic!("{pf:?}"),
    }
}

/// A read value `v` as the codes [`expected`] gives.
fn codes(shape: &Shape, v: [f32; 4]) -> [u32; 4] {
    let max: [f32; 4] = match shape.pf {
        PixelFormat::BC1RGBA => [31.0, 63.0, 31.0, 1.0],
        _ => [255.0; 4],
    };
    [0, 1, 2, 3].map(|c| (v[c] * max[c]).round() as u32)
}

/// The buffer's first `length` bytes.
fn read_buffer(buffer: &Buffer, length: usize) -> Vec<u8> {
    let contents = buffer.contents().unwrap();
    // SAFETY: the buffer's bytes, which no command uses now.
    unsafe { std::slice::from_raw_parts(contents.as_ptr(), length) }.to_vec()
}

/// Set the buffer's first bytes to `bytes`.
fn write_buffer(buffer: &Buffer, bytes: &[u8]) {
    let contents = buffer.contents().unwrap();
    // SAFETY: as above; the buffer holds at least the surface's bytes.
    unsafe { std::slice::from_raw_parts_mut(contents.as_ptr(), bytes.len()) }
        .copy_from_slice(bytes);
}

/// Run one blit encoder `encode` fills, and wait for it.
fn blit(device: &Device, encode: impl FnOnce(&mut BlitCommandEncoder) -> Result<(), Error>) {
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    encode(&mut blit).unwrap();
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
}

/// The texels a kernel reads of `texture`, `float4`s in rows.
fn kernel_reads(
    device: &Device,
    state: &ComputePipelineState,
    shape: &Shape,
    texture: &Texture,
) -> Vec<[f32; 4]> {
    let [w, h] = shape.size;
    let out = device
        .new_buffer(16 * w * h, ResourceOptions::shared())
        .unwrap();
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(state).unwrap();
    encoder.set_texture(Some(texture), 0).unwrap();
    encoder.set_buffer(Some(&out), 0, 0).unwrap();
    let group = match h {
        1 => Size::new(w.min(64), 1, 1),
        _ => Size::new(8, 8, 1),
    };
    encoder.dispatch_threads(Size::new(w, h, 1), group).unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
    read_buffer(&out, (16 * w * h) as usize)
        .as_chunks::<16>()
        .0
        .iter()
        .map(|t| {
            [0, 1, 2, 3]
                .map(|c| f32::from_le_bytes([t[4 * c], t[4 * c + 1], t[4 * c + 2], t[4 * c + 3]]))
        })
        .collect()
}

/// Check one shape over `surface` (a buffer standing for it, `checked` bytes of it compared):
/// each failure into `wrong`. Returns the path its texture took.
fn check(
    device: &Device,
    states: &[(TextureType, ComputePipelineState)],
    shape: &Shape,
    surface: &Buffer,
    checked: usize,
    wrong: &mut Vec<String>,
) -> &'static str {
    let [w, h] = shape.size;
    let ([across, down], row) = shape.units();
    let (bytes, [bw, bh]) = shape.unit();
    let mut model = surface_bytes(shape, checked);
    write_buffer(surface, &model);
    let (offset, bytes_per_row, planar) = shape.surface.plane;
    let made = surface.new_texture_with_iosurface_plane(
        &shape.descriptor(),
        IoSurfacePlane::new(offset, bytes_per_row, planar),
    );
    let texture = match made {
        Ok(texture) => texture,
        Err(error) => {
            wrong.push(format!("refused: {error:?}"));
            return "refused";
        }
    };
    let path = match (texture.shadow().is_some(), texture.aliases_buffer()) {
        (true, _) => "shadow",
        (false, true) => "alias",
        (false, false) => "texels",
    };
    let state = states
        .iter()
        .find(|(t, _)| *t == shape.texture_type)
        .map(|(_, state)| state)
        .unwrap();
    let cpu = !shape.private;
    let blits = shape.texture_type != TextureType::TextureBuffer;
    let whole = match shape.texture_type {
        TextureType::Type1D | TextureType::TextureBuffer => Region::new_1d(0, w),
        _ => Region::new_2d(0, 0, w, h),
    };
    let image = row * down;
    let kernel = |when: &str, model: &[u8], wrong: &mut Vec<String>| {
        if !shape.read_checked() {
            return;
        }
        let rows = rows_of(shape, model);
        let read = kernel_reads(device, state, shape, &texture);
        let off: Vec<_> = (0..h)
            .flat_map(|y| (0..w).map(move |x| (x, y)))
            .filter(|&(x, y)| {
                codes(shape, read[(y * w + x) as usize]) != expected(shape, &rows, (x, y))
            })
            .collect();
        if let Some(&(x, y)) = off.first() {
            wrong.push(format!(
                "kernel read {when}: {} of {} texels off, first ({x}, {y}): {:?}, not {:?}",
                off.len(),
                w * h,
                codes(shape, read[(y * w + x) as usize]),
                expected(shape, &rows, (x, y)),
            ));
        }
    };
    let blit_read = |when: &str, model: &[u8], wrong: &mut Vec<String>| {
        if !blits {
            return;
        }
        let out = device.new_buffer(image, ResourceOptions::shared()).unwrap();
        blit(device, |blit| {
            blit.copy_from_texture_to_buffer(
                &texture,
                0,
                0,
                Origin::new(0, 0, 0),
                Size::new(w, h, 1),
                &out,
                0,
                row,
                0,
                BlitOption::None,
            )
        });
        if read_buffer(&out, image as usize) != rows_of(shape, model) {
            wrong.push(format!("blit read {when}: not the surface's rows"));
        }
    };
    let get_bytes = |when: &str, model: &[u8], wrong: &mut Vec<String>| {
        if !cpu {
            return;
        }
        let mut read = vec![0u8; image as usize];
        texture.get_bytes(&mut read, row, 0, whole, 0, 0).unwrap();
        if read != rows_of(shape, model) {
            wrong.push(format!("getBytes {when}: not the surface's rows"));
        }
    };
    let surface_holds = |when: &str, model: &[u8], wrong: &mut Vec<String>| {
        let now = read_buffer(surface, checked);
        let changed: Vec<usize> = (0..checked).filter(|&i| now[i] != model[i]).collect();
        if let (Some(first), Some(last)) = (changed.first(), changed.last()) {
            wrong.push(format!(
                "{when}: {} surface bytes not as Metal leaves them, {first} to {last}",
                changed.len()
            ));
        }
    };
    kernel("as made", &model, wrong);
    blit_read("as made", &model, wrong);
    get_bytes("as made", &model, wrong);
    // One unit at the last of the second row (the last texel of a texture buffer), from the CPU.
    if cpu {
        let (i, j) = (across - 1, (down - 1).min(1));
        let unit: Vec<u8> = match shape.pf {
            PixelFormat::BC1RGBA => solid(0x07ff).to_vec(),
            _ => (0..bytes as usize).map(|b| named(b, 0x5a)).collect(),
        };
        let region = match shape.texture_type {
            TextureType::Type1D | TextureType::TextureBuffer => Region::new_1d(i, 1),
            _ => Region::new_2d(i * bw, j * bh, bw.min(w - i * bw), bh.min(h - j * bh)),
        };
        texture
            .replace_region(region, 0, 0, &unit, bytes, 0)
            .unwrap();
        let at = shape.row_at(j) + (i * bytes) as usize;
        model[at..at + unit.len()].copy_from_slice(&unit);
        surface_holds("replaceRegion", &model, wrong);
        kernel("after replaceRegion", &model, wrong);
        blit_read("after replaceRegion", &model, wrong);
    }
    // Every unit, from a blit.
    if blits {
        let rows: Vec<u8> = match shape.pf {
            PixelFormat::BC1RGBA => (0..down)
                .flat_map(|j| (0..across).flat_map(move |i| solid(block_color(shape, i, j, 1))))
                .collect(),
            _ => (0..image as usize).map(|b| named(b, 0xa5)).collect(),
        };
        let source = device
            .new_buffer_with_bytes(&rows, ResourceOptions::shared())
            .unwrap();
        blit(device, |blit| {
            blit.copy_from_buffer_to_texture(
                &source,
                0,
                row,
                0,
                Size::new(w, h, 1),
                &texture,
                0,
                0,
                Origin::new(0, 0, 0),
                BlitOption::None,
            )
        });
        set_rows(shape, &mut model, &rows);
        surface_holds("blit write", &model, wrong);
        kernel("after the blit write", &model, wrong);
        get_bytes("after the blit write", &model, wrong);
    }
    path
}

/// Every shape over a `Shared` buffer and over host pages, on the device the cfg selects.
#[test]
fn textures_over_surfaces_take_the_bytes_metal_does() {
    let scratch = Scratch::new("surface-shapes");
    let host = Host::new();
    let runs = Arc::new(AtomicUsize::new(0));
    let device = device(&scratch.0);
    let name = device.name().to_owned();
    let states: Vec<(TextureType, ComputePipelineState)> = READS
        .iter()
        .map(|(t, _)| (*t, read_state(&device, *t)))
        .collect();
    let pages = over_host(&device, &host, &runs);
    let mut failures = Vec::new();
    for shape in shapes() {
        let shared = device
            .new_buffer(shape.surface.length as u64, ResourceOptions::shared())
            .unwrap();
        let mut over: Vec<(&str, &Buffer, usize)> = vec![("Shared", &shared, shape.surface.length)];
        // Over host pages the buffer runs past the surface: a page more of it is compared.
        if let Some(pages) = &pages {
            over.push(("host pages", pages, shape.surface.length + 4096));
        }
        for (what, buffer, checked) in over {
            let before = device.work_counts();
            let mut wrong = Vec::new();
            let path = check(&device, &states, &shape, buffer, checked, &mut wrong);
            let after = device.work_counts();
            eprintln!(
                "SURFACE-SHAPES {name} {} over {what}: {path}, shadowed_textures +{}, \
                 shadow_texels_copied_in +{}, {} wrong",
                shape.name,
                after.shadowed_textures - before.shadowed_textures,
                after.shadow_texels_copied_in - before.shadow_texels_copied_in,
                wrong.len()
            );
            failures.extend(
                wrong
                    .into_iter()
                    .map(|w| format!("{} over {what}: {w}", shape.name)),
            );
        }
    }
    drop(pages);
    for failure in &failures {
        eprintln!("SURFACE-SHAPES WRONG {failure}");
    }
    assert!(failures.is_empty(), "{} wrong", failures.len());
}
