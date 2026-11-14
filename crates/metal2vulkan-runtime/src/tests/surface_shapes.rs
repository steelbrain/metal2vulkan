//! Textures over an IOSurface plane beyond a 2D texture of one texel a block: a BC texture over a
//! surface's rows, and a texture buffer, a 1D texture, a 2D array of one slice, a 2D texture
//! wider than the plane's rows, and a cube and a cube array (each face, every cube's one after
//! another, one image on from the plane's offset) over a plane of a surface with planes
//! (`linear_plan.rs`'s **Block-compressed surfaces** and **Beyond the plane's rows**). A cube's
//! faces past the surface lie in the texture's tail, zero when made over a `Shared` buffer of the
//! surface's length, and over host pages in the buffer's bytes past the surface. Each reads and writes the bytes
//! the x86 Macs' Metal does, and a texture buffer at a plane offset off 16 and a texture wider
//! than its plane's rows the bytes the M3 Max's does (`surface_texture_envelope`; each shape
//! names the `oracle_surftex` case or the probe's shape it stands for): a kernel's reads, the
//! blits, `getBytes` and `replaceRegion` take those bytes, a kernel after a blit or a CPU write
//! reads what they wrote, and no other byte of the surface changes. A BC texture of 8-byte blocks
//! at a plane offset off 16 takes the bytes the x86 Macs' blits and samples and the 5300M's
//! `getBytes` take, at the plane's offset, on every path (the UHD 630's CPU calls take the plane
//! from byte 0). A blit into a 2D texture whose rows overlap is refused
//! (`NotImplemented(OverlappingSurfaceRows)`, `linear_plan.rs`'s **Rows that overlap**) and
//! changes no byte.
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. Each surface is a buffer standing for
//! its memory, as the consumer's is: a `Shared` buffer, and a no-copy buffer over host pages where
//! the device imports them. A BC1 block is solid, its two colours one RGB565 value naming the
//! block, and every block of the surface outside the texture's names none; another texture's
//! bytes name their offset. The device asks for the validation layer, which reports nothing
//! (`layer_claim.rs`).

use super::blit_holds::{over_host, Host, Scratch, WAIT};
use super::layer_claim::Watched;
use crate::{
    BlitCommandEncoder, BlitOption, Buffer, ComputePipelineState, Device, DeviceKind,
    DeviceOptions, Error, IoSurfacePlane, Origin, PipelineOptions, Region, ResourceOptions,
    RuntimeGap, ShaderCacheOptions, Size, Texture, TextureDescriptor,
};
use metal2vulkan::metal::{PixelFormat, TextureType, TextureUsage};
use std::path::Path;
use std::sync::atomic::AtomicUsize;
use std::sync::Arc;

/// The device the cfg selects, its shader cache in `cache`, asking for the validation layer, and
/// what its instance had the layer report.
fn watched(cache: &Path) -> (Device, Watched) {
    let watched = Watched::default();
    let options = DeviceOptions {
        kinds: match cfg!(m2v_cpu_icd) {
            true => vec![DeviceKind::Cpu],
            false => vec![
                DeviceKind::Discrete,
                DeviceKind::Integrated,
                DeviceKind::Virtual,
            ],
        },
        shader_cache: ShaderCacheOptions {
            directory: Some(cache.to_owned()),
            ..ShaderCacheOptions::default()
        },
        layers: watched.layers(),
        messages: Some(watched.sink(|_| {})),
        ..DeviceOptions::default()
    };
    (Device::system_default_with(&options).unwrap(), watched)
}

/// How a kernel reads a texture of one type: its argument's type, the read's call and
/// declaration, and its grid's dimensions (3: a cube's faces, every cube's one after another, a
/// face a plane of the grid).
struct Read {
    arg_type: &'static str,
    call: &'static str,
    declare: &'static str,
    grid: u32,
}

/// The reads of a 2D texture, a 2D array's slice 0, a 1D texture, a texture buffer, and every
/// face of a cube and a cube array.
const READS: [(TextureType, Read); 6] = [
    (
        TextureType::Type2D,
        Read {
            arg_type: "texture2d<float, read>",
            call: "@air.read_texture_2d.v4f32(ptr addrspace(1) %t, ptr addrspace(2) %s, <2 x i32> %gid, <2 x i32> zeroinitializer, i32 0, i32 1)",
            declare: "@air.read_texture_2d.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x i32>, <2 x i32>, i32, i32)",
            grid: 2,
        },
    ),
    (
        TextureType::Type2DArray,
        Read {
            arg_type: "texture2d_array<float, read>",
            call: "@air.read_texture_2d_array.v4f32(ptr addrspace(1) %t, ptr addrspace(2) %s, <2 x i32> %gid, i32 0, <2 x i32> zeroinitializer, i32 0, i32 1)",
            declare: "@air.read_texture_2d_array.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x i32>, i32, <2 x i32>, i32, i32)",
            grid: 2,
        },
    ),
    (
        TextureType::Type1D,
        Read {
            arg_type: "texture1d<float, read>",
            call: "@air.read_texture_1d.v4f32(ptr addrspace(1) %t, ptr addrspace(2) %s, i32 %gid, i32 0, i32 1)",
            declare: "@air.read_texture_1d.v4f32(ptr addrspace(1), ptr addrspace(2), i32, i32, i32)",
            grid: 1,
        },
    ),
    (
        TextureType::TextureBuffer,
        Read {
            arg_type: "texture_buffer<float, read>",
            call: "@air.read_texture_buffer_1d.v4f32(ptr addrspace(1) %t, ptr addrspace(2) %s, i32 %gid, i32 1)",
            declare: "@air.read_texture_buffer_1d.v4f32(ptr addrspace(1), ptr addrspace(2), i32, i32)",
            grid: 1,
        },
    ),
    (
        TextureType::Cube,
        Read {
            arg_type: "texturecube<float, read>",
            call: "@air.read_texture_cube.v4f32(ptr addrspace(1) %t, ptr addrspace(2) %s, <2 x i32> %coord, i32 %face, i32 0, i32 1)",
            declare: "@air.read_texture_cube.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x i32>, i32, i32, i32)",
            grid: 3,
        },
    ),
    (
        TextureType::CubeArray,
        Read {
            arg_type: "texturecube_array<float, read>",
            call: "@air.read_texture_cube_array.v4f32(ptr addrspace(1) %t, ptr addrspace(2) %s, <2 x i32> %coord, i32 %face, i32 %cube, i32 0, i32 1)",
            declare: "@air.read_texture_cube_array.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x i32>, i32, i32, i32, i32)",
            grid: 3,
        },
    ),
];

/// A kernel that reads every texel of texture 0 into buffer 0 through `read`, `float4`s in rows
/// of the grid (its planes one after another).
fn kernel(read: &Read) -> String {
    let (gid, index, size_arg, size_meta, gid_type) = match read.grid {
        2 => (
            "<2 x i32> %gid, <2 x i32> %size",
            "  %x = extractelement <2 x i32> %gid, i64 0\n  %y = extractelement <2 x i32> %gid, i64 1\n  %w = extractelement <2 x i32> %size, i64 0\n  %row = mul i32 %y, %w\n  %i = add i32 %row, %x\n",
            ", !6",
            "!6 = !{i32 3, !\"air.threads_per_grid\", !\"air.arg_type_name\", !\"uint2\", !\"air.arg_name\", !\"size\"}\n",
            "uint2",
        ),
        3 => (
            "<3 x i32> %gid, <3 x i32> %size",
            "  %x = extractelement <3 x i32> %gid, i64 0\n  %y = extractelement <3 x i32> %gid, i64 1\n  %z = extractelement <3 x i32> %gid, i64 2\n  %w = extractelement <3 x i32> %size, i64 0\n  %h = extractelement <3 x i32> %size, i64 1\n  %c0 = insertelement <2 x i32> poison, i32 %x, i64 0\n  %coord = insertelement <2 x i32> %c0, i32 %y, i64 1\n  %face = urem i32 %z, 6\n  %cube = udiv i32 %z, 6\n  %zh = mul i32 %z, %h\n  %zy = add i32 %zh, %y\n  %row = mul i32 %zy, %w\n  %i = add i32 %row, %x\n",
            ", !6",
            "!6 = !{i32 3, !\"air.threads_per_grid\", !\"air.arg_type_name\", !\"uint3\", !\"air.arg_name\", !\"size\"}\n",
            "uint3",
        ),
        _ => ("i32 %gid", "  %i = add i32 %gid, 0\n", "", "", "uint"),
    };
    format!(
        r#"; Owned synthetic kernel.
source_filename = "surface_shapes.metal"

define void @read_texels(ptr addrspace(1) %t, ptr addrspace(1) noalias %out, {gid}) {{
entry:
  %s = tail call ptr addrspace(2) @air.get_read_sampler()
{index}  %r = tail call {{ <4 x float>, i8 }} {call}
  %v = extractvalue {{ <4 x float>, i8 }} %r, 0
  %j = zext i32 %i to i64
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
/// A 64 x 64 `'BGRA'` surface of one plane at offset 8, in 32768 bytes (the probe's
/// `bgraat8`).
const BGRA_AT8: Surface = Surface {
    length: 32768,
    plane: (8, 256, true),
};
/// A 64 x 64 `'BGRA'` surface of one plane at 264 bytes a row, in 32768 bytes (the probe's
/// `bgrarow264`).
const BGRA_ROW264: Surface = Surface {
    length: 32768,
    plane: (0, 264, true),
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
/// A 64 x 64 `420v` surface's luma plane on the M3 Max (the probe's `420v`).
const M3_P0: Surface = Surface {
    length: 6720,
    plane: (512, 64, true),
};
/// Its chroma plane.
const M3_P1: Surface = Surface {
    length: 6720,
    plane: (4608, 64, true),
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
    /// A cube array's cubes (1 for every other type).
    cubes: u64,
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
        cubes: 1,
    }
}

impl Shape {
    /// Its unit's bytes and texels across and down.
    fn unit(&self) -> (u64, [u64; 2]) {
        match self.pf {
            PixelFormat::BC1RGBA | PixelFormat::BC4RUnorm => (8, [4, 4]),
            PixelFormat::BC3RGBA | PixelFormat::BC7RGBAUnorm => (16, [4, 4]),
            PixelFormat::BGRA8Unorm => (4, [1, 1]),
            PixelFormat::RG8Unorm => (2, [1, 1]),
            PixelFormat::R8Unorm => (1, [1, 1]),
            pf => panic!("{pf:?}"),
        }
    }

    /// Whether its rows share bytes: a 2D colour texture's rows wider than the plane's
    /// `bytesPerRow`.
    fn rows_overlap(&self) -> bool {
        let (_, row) = self.units();
        self.texture_type == TextureType::Type2D && row > self.surface.plane.1 && !self.blocks()
    }

    /// Whether it is a BC texture.
    fn blocks(&self) -> bool {
        let (_, [w, _]) = self.unit();
        w > 1
    }

    /// The bytes of each row of units that lie in place: a BC texture's rows of blocks wider
    /// than its plane's `bytesPerRow` keep to the plane's row, their blocks past it in no byte
    /// (reading zero, taking no write), as the x86 Macs' blits keep them; every other row whole.
    fn kept(&self) -> u64 {
        let (_, row) = self.units();
        let (_, bytes_per_row, planar) = self.surface.plane;
        match planar && self.blocks() && row > bytes_per_row {
            true => bytes_per_row,
            false => row,
        }
    }

    /// Its slices: a cube's six faces, every cube's one after another, else 1.
    fn layers(&self) -> u64 {
        match self.texture_type {
            TextureType::Cube | TextureType::CubeArray => 6 * self.cubes,
            _ => 1,
        }
    }

    /// Its units across and down, its slices' rows one after another, and the bytes of a row of
    /// them.
    fn units(&self) -> ([u64; 2], u64) {
        let (bytes, [w, h]) = self.unit();
        let across = self.size[0].div_ceil(w);
        let down = self.size[1].div_ceil(h) * self.layers();
        ([across, down], across * bytes)
    }

    /// The surface's byte past its last row of units: past the surface where its slices run
    /// past it, into the texture's tail.
    fn end(&self) -> usize {
        let ([_, down], row) = self.units();
        self.row_at(down - 1) + row as usize
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

    /// The surface's byte where a kernel reads row `row` of units from: a texture buffer's from
    /// the plane's offset rounded down to 16, as Metal's functions read one (`bgraat8.texbuf` on
    /// the M3 Max), every other shape's where its CPU calls do.
    fn gpu_row_at(&self, row: u64) -> usize {
        let (offset, _, _) = self.surface.plane;
        match self.texture_type {
            TextureType::TextureBuffer => (offset - offset % 16) as usize,
            _ => self.row_at(row),
        }
    }

    /// Whether a kernel's reads are checked: a BC1 texture's solid blocks, or 8-bit unsigned
    /// normalized texels.
    fn read_checked(&self) -> bool {
        matches!(
            self.pf,
            PixelFormat::BC1RGBA
                | PixelFormat::BGRA8Unorm
                | PixelFormat::RG8Unorm
                | PixelFormat::R8Unorm
        )
    }

    fn descriptor(&self) -> TextureDescriptor {
        let mut d = TextureDescriptor::texture_2d(self.pf, self.size[0], self.size[1], false);
        d.texture_type = self.texture_type;
        if self.texture_type == TextureType::CubeArray {
            d.array_length = self.cubes;
        }
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
        shape("420v.p0.bc1.32x64", P::BC1RGBA, T::Type2D, [32, 64], P0),
        // Rows of blocks wider than the plane's (187 420v.p0.fmt.bc1): each kept to the plane's
        // row, its blocks past it reading zero and taking no write.
        shape("187 420v.p0.fmt.bc1", P::BC1RGBA, T::Type2D, [64, 64], P0),
        shape("420v.p1.bc1.16x32", P::BC1RGBA, T::Type2D, [16, 32], P1),
        // 8-byte blocks over a plane at an offset or pitch off 16 (the x86 Macs' `bgraat8.bc1`
        // and `bgrarow264.bc1`): at the plane's own offset and pitch, unrounded.
        shape("bgraat8.bc1", P::BC1RGBA, T::Type2D, [64, 64], BGRA_AT8),
        shape(
            "bgrarow264.bc1",
            P::BC1RGBA,
            T::Type2D,
            [64, 64],
            BGRA_ROW264,
        ),
        shape("bgraat8.bc4", P::BC4RUnorm, T::Type2D, [64, 64], BGRA_AT8),
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
        // At a plane offset and a pitch off 16 (the M3 Max's `bgraat8.texbuf` and
        // `bgrarow264.texbuf`): its functions from the offset rounded down to 16, its CPU calls
        // from the offset.
        shape(
            "bgraat8.texbuf.64",
            P::BGRA8Unorm,
            T::TextureBuffer,
            [64, 1],
            BGRA_AT8,
        ),
        shape(
            "bgraat8.texbuf.128",
            P::BGRA8Unorm,
            T::TextureBuffer,
            [128, 1],
            BGRA_AT8,
        ),
        shape(
            "bgrarow264.texbuf.128",
            P::BGRA8Unorm,
            T::TextureBuffer,
            [128, 1],
            BGRA_ROW264,
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
        // Wider than the plane's rows (the M3 Max's): a 2D texture's rows `bytesPerRow` apart,
        // overlapping, and a 1D texture's one row from the plane's offset.
        shape("bgra1x2.65x64", P::BGRA8Unorm, T::Type2D, [65, 64], BGRA1X2),
        shape(
            "bgra1x2.128x32",
            P::BGRA8Unorm,
            T::Type2D,
            [128, 32],
            BGRA1X2,
        ),
        shape("420v.p0.65x64", P::R8Unorm, T::Type2D, [65, 64], M3_P0),
        shape("420v.p1.33x32", P::RG8Unorm, T::Type2D, [33, 32], M3_P1),
        shape(
            "bgra1x2.1d.128",
            P::BGRA8Unorm,
            T::Type1D,
            [128, 1],
            BGRA1X2,
        ),
        // A cube and a cube array (the x86 Macs'): face `k` one image on from the plane's
        // offset, the faces past the surface in the texture's tail, those past the plane but
        // inside the surface over its bytes there.
        shape(
            "119 bgra1.typecube",
            P::BGRA8Unorm,
            T::Cube,
            [64, 64],
            BGRA1,
        ),
        shape(
            "191 bgra1.typecube.32",
            P::BGRA8Unorm,
            T::Cube,
            [32, 32],
            BGRA1,
        ),
        shape(
            "188 bgra1.typecubearray",
            P::BGRA8Unorm,
            T::CubeArray,
            [64, 64],
            BGRA1,
        ),
        shape("189 420v.p0.typecube", P::R8Unorm, T::Cube, [64, 64], P0),
        shape(
            "190 420v.p0.typecubearray",
            P::R8Unorm,
            T::CubeArray,
            [64, 64],
            P0,
        ),
        shape("bgra1x2.cube.16", P::BGRA8Unorm, T::Cube, [16, 16], BGRA1X2),
        shape(
            "bgra1x2.cubearray.16",
            P::BGRA8Unorm,
            T::CubeArray,
            [16, 16],
            BGRA1X2,
        ),
        Shape {
            cubes: 2,
            ..shape(
                "bgra1x2.cubearray.8.a2",
                P::BGRA8Unorm,
                T::CubeArray,
                [8, 8],
                BGRA1X2,
            )
        },
        shape("420v.p0.cube.8", P::R8Unorm, T::Cube, [8, 8], P0),
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
        // The blocks in place alone: those past a row kept to the plane's lie in no byte.
        let across = across.min(shape.kept() / 8);
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

/// The texture's rows of units, packed, out of the surface's `bytes`: each row's bytes in place,
/// then zeros ([`Shape::kept`]).
fn rows_of(shape: &Shape, bytes: &[u8]) -> Vec<u8> {
    rows_from(shape, bytes, |j| shape.row_at(j))
}

/// The texture's rows of units as a kernel reads them, packed, out of the surface's `bytes`.
fn gpu_rows_of(shape: &Shape, bytes: &[u8]) -> Vec<u8> {
    rows_from(shape, bytes, |j| shape.gpu_row_at(j))
}

/// The texture's rows of units, row `j` at `at(j)` of the surface's `bytes`, packed: each row's
/// bytes in place, then zeros.
fn rows_from(shape: &Shape, bytes: &[u8], at: impl Fn(u64) -> usize) -> Vec<u8> {
    let ([_, down], row) = shape.units();
    let kept = shape.kept() as usize;
    (0..down)
        .flat_map(|j| {
            let at = at(j);
            let mut packed = bytes[at..at + kept].to_vec();
            packed.resize(row as usize, 0);
            packed
        })
        .collect()
}

/// Write `rows` (packed rows of units) into the surface's `bytes` where the texture's lie: each
/// row's bytes in place, those past them taking no write.
fn set_rows(shape: &Shape, bytes: &mut [u8], rows: &[u8]) {
    let (_, row) = shape.units();
    let kept = shape.kept() as usize;
    for (j, packed) in rows.chunks(row as usize).enumerate() {
        let at = shape.row_at(j as u64);
        let packed = &packed[..kept.min(packed.len())];
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
        PixelFormat::RG8Unorm => [u32::from(rows[at]), u32::from(rows[at + 1]), 0, 255],
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
    let layers = shape.layers();
    let out = device
        .new_buffer(16 * w * h * layers, ResourceOptions::shared())
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
    encoder
        .dispatch_threads(Size::new(w, h, layers), group)
        .unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
    read_buffer(&out, (16 * w * h * layers) as usize)
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
    // The surface's bytes and past them the buffer's; past the buffer the texture's tail, zero
    // when made.
    let held = (surface.length() as usize).min(checked.max(shape.end()));
    let mut model = surface_bytes(shape, held);
    write_buffer(surface, &model);
    model.resize(checked.max(shape.end()), 0);
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
    // A slice's bytes, its rows of units packed.
    let layers = shape.layers();
    let (face_rows, face_image) = (down / layers, image / layers);
    let kernel = |when: &str, model: &[u8], wrong: &mut Vec<String>| {
        if !shape.read_checked() {
            return;
        }
        let rows = gpu_rows_of(shape, model);
        let read = kernel_reads(device, state, shape, &texture);
        // Its slices' rows one after another, as the kernel reads them.
        let off: Vec<_> = (0..h * layers)
            .flat_map(|y| (0..w).map(move |x| (x, y)))
            .filter(|&(x, y)| {
                codes(shape, read[(y * w + x) as usize]) != expected(shape, &rows, (x, y))
            })
            .collect();
        if let Some(&(x, y)) = off.first() {
            wrong.push(format!(
                "kernel read {when}: {} of {} texels off, first ({x}, {y}): {:?}, not {:?}",
                off.len(),
                w * h * layers,
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
            (0..layers).try_for_each(|slice| {
                blit.copy_from_texture_to_buffer(
                    &texture,
                    slice,
                    0,
                    Origin::new(0, 0, 0),
                    Size::new(w, h, 1),
                    &out,
                    slice * face_image,
                    row,
                    0,
                    BlitOption::None,
                )
            })
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
        for (slice, face) in read.chunks_mut(face_image as usize).enumerate() {
            (texture.get_bytes(face, row, 0, whole, 0, slice as u64)).unwrap();
        }
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
    // One unit at the last of the second row (the last texel of a texture buffer) of the last
    // slice, from the CPU.
    if cpu {
        let (i, j, slice) = (across - 1, (face_rows - 1).min(1), layers - 1);
        let unit: Vec<u8> = match shape.pf {
            PixelFormat::BC1RGBA => solid(0x07ff).to_vec(),
            _ => (0..bytes as usize).map(|b| named(b, 0x5a)).collect(),
        };
        let region = match shape.texture_type {
            TextureType::Type1D | TextureType::TextureBuffer => Region::new_1d(i, 1),
            _ => Region::new_2d(i * bw, j * bh, bw.min(w - i * bw), bh.min(h - j * bh)),
        };
        texture
            .replace_region(region, 0, slice, &unit, bytes, 0)
            .unwrap();
        // A unit past a row's bytes in place takes no write.
        if i * bytes < shape.kept() {
            let at = shape.row_at(slice * face_rows + j) + (i * bytes) as usize;
            model[at..at + unit.len()].copy_from_slice(&unit);
        }
        surface_holds("replaceRegion", &model, wrong);
        kernel("after replaceRegion", &model, wrong);
        blit_read("after replaceRegion", &model, wrong);
    }
    // A blit into rows that share bytes, refused: no order a row copy writes them in is Metal's.
    if blits && shape.rows_overlap() {
        let source = device.new_buffer(image, ResourceOptions::shared()).unwrap();
        let mut gap = None;
        blit(device, |blit| {
            gap = blit
                .copy_from_buffer_to_texture(
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
                .err();
            Ok(())
        });
        if gap != Some(Error::NotImplemented(RuntimeGap::OverlappingSurfaceRows)) {
            wrong.push(format!("blit write: {gap:?}, not OverlappingSurfaceRows"));
        }
        surface_holds("a refused blit write", &model, wrong);
        return path;
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
            (0..layers).try_for_each(|slice| {
                blit.copy_from_buffer_to_texture(
                    &source,
                    slice * face_image,
                    row,
                    0,
                    Size::new(w, h, 1),
                    &texture,
                    slice,
                    0,
                    Origin::new(0, 0, 0),
                    BlitOption::None,
                )
            })
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
    let (device, watched) = watched(&scratch.0);
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
    watched.layer_reported_nothing("every surface shape's commands were valid");
}
