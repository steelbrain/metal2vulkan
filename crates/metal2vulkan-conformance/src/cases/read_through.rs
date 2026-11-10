//! What MSL 4.1's `read(sampler, coord[, offset][, lod])` answers through a sampler of each address
//! family: constexpr samplers, the runtime sampler at `[[sampler(0)]]`, an argument buffer's
//! sampler, and an argument buffer's texture.
//!
//! The owned fixture `kernel_read_through` (compiled with `-std=metal4.1`) holds one kernel per
//! sampler state; thread `i` reads at `p[i]` (x, y, then the layer or z, then the level) and stores
//! the texel. Each case binds one texture and runs the kernels that read its kind, every one over
//! the same positions: inside the texture, one and several texels past each edge, at a lower level,
//! past that level's extent, and at a level past the texture's last.
//!
//! What the M3 measured, and what these recordings hold the runtime to: the coordinate (plus the
//! offset) is addressed per axis by the sampler's mode, an edge mode clamping and a border mode
//! (`clamp_to_zero`, `clamp_to_border`) reading the border texel in the format's channels and
//! through the texture's swizzle; a sampler with a mip filter reads its own level and one without
//! reads level 0; a level or a layer past the texture reads zero in every lane. A texture buffer
//! read past its end reads texel 0 through an edge mode, except through a runtime sampler of
//! `clamp_to_edge`, which clamps.
//!
//! **One case per texture**, so a venue that cannot make one kind (a 1D texture, a texture buffer)
//! answers its own typed refusal at its own row. The two kernels that read an argument buffer's
//! texture through a border colour other than transparent black, or through a runtime sampler, are
//! cases of their own, so a venue that refuses one (a device without the handle heap's
//! descriptor-indexing features) stops only its own case at the pipeline.
//!
//! Every output word holds [`BLANK`] before its dispatch, so a word a kernel did not write is a
//! recorded value. Nothing here can fault a GPU: one compute pass a read, at most 20 threads, no
//! loop, and every coordinate outside a texture is the sampler's to address.

use crate::case::{
    call, made, Abort, ArgumentSlot, Bindings, Case, Failure, Handle, Input, MetalSource,
    SamplerSpec, Side,
};
use crate::fixture::{Fixture, METAL_TEXT, RUNTIME_TEXT};
use crate::recording::ShaderRef;
use crate::texture::{TextureRequest, Upload};
use crate::transcript::{ByteLayout, Digest, Element, Gaps, Node, Observed, Region, Sites};
use metal2vulkan::metal::{
    PixelFormat, SamplerAddressMode, SamplerBorderColor, SamplerMinMagFilter, SamplerMipFilter,
    TextureSwizzle, TextureSwizzleChannels, TextureType, TextureUsage,
};
use std::collections::BTreeMap;

/// The fixture every kernel lives in.
const FIXTURE: &str = "kernel_read_through";

/// The byte every output word holds before a dispatch.
pub const BLANK: u8 = 0xab;

/// The kinds of texture the fixture's kernels read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    /// A `texture2d`, 4 by 4, three levels.
    Plane,
    /// A `texture2d_array`, 4 by 4, three slices, two levels.
    Layers,
    /// A `texture3d`, 4 by 4 by 4, two levels.
    Volume,
    /// A `texture1d` of 8 texels.
    Line,
    /// A `texture2d<float, access::read_write>`, 4 by 4, one level, shader-read and -write.
    ReadWrite,
    /// A `texture_buffer` of 8 texels over a buffer, from its first byte.
    Buffer,
}

impl Kind {
    /// The extent of level 0: width, height, and the depth or the slices.
    const fn extent(self) -> [u32; 3] {
        match self {
            Kind::Plane | Kind::ReadWrite => [4, 4, 1],
            Kind::Layers => [4, 4, 3],
            Kind::Volume => [4, 4, 4],
            Kind::Line | Kind::Buffer => [8, 1, 1],
        }
    }

    /// The levels the texture holds.
    const fn levels(self) -> u32 {
        match self {
            Kind::Plane => 3,
            Kind::Layers | Kind::Volume => 2,
            Kind::Line | Kind::ReadWrite | Kind::Buffer => 1,
        }
    }

    /// The extent of `level`: a 3D texture's depth halves with its width, an array's slices stay.
    fn level_extent(self, level: u32) -> [u32; 3] {
        let [w, h, d] = self.extent();
        let halve = |n: u32| if n >> level == 0 { 1 } else { n >> level };
        match self {
            Kind::Volume => [halve(w), halve(h), halve(d)],
            _ => [halve(w), halve(h), d],
        }
    }

    /// The positions the kernels read at, `(x, y, layer or z, level)`; a kernel reads the parts
    /// its texture has.
    const fn positions(self) -> &'static [[i32; 4]] {
        match self {
            Kind::Plane | Kind::ReadWrite => &PLANE,
            Kind::Layers => &LAYERS,
            Kind::Volume => &VOLUME,
            Kind::Line | Kind::Buffer => &LINE,
        }
    }
}

/// A plane's positions: inside, past each edge by one and by more, past a corner, at level 1
/// inside and past its 2 by 2, at the last level, and at levels 3 and `-1` past the texture's.
const PLANE: [[i32; 4]; 20] = [
    [1, 2, 0, 0],
    [3, 3, 0, 0],
    [0, 0, 0, 0],
    [4, 1, 0, 0],
    [5, 1, 0, 0],
    [9, 2, 0, 0],
    [-1, 1, 0, 0],
    [-3, 2, 0, 0],
    [-6, 0, 0, 0],
    [1, 4, 0, 0],
    [2, 7, 0, 0],
    [2, -1, 0, 0],
    [-1, -1, 0, 0],
    [4, 4, 0, 0],
    [1, 1, 0, 1],
    [2, 0, 0, 1],
    [-1, 1, 0, 1],
    [0, 0, 0, 2],
    [0, 0, 0, 3],
    [0, 0, 0, -1],
];

/// An array's positions: inside two slices, past the last slice and at slice `-1`, past an edge,
/// at level 1 inside and past its extent, and at levels past the texture's.
const LAYERS: [[i32; 4]; 11] = [
    [1, 2, 0, 0],
    [1, 2, 2, 0],
    [1, 2, 3, 0],
    [1, 2, -1, 0],
    [4, 1, 1, 0],
    [-1, 1, 1, 0],
    [1, 1, 1, 1],
    [2, 1, 1, 1],
    [1, 1, 4, 1],
    [0, 0, 0, 2],
    [0, 0, 0, -1],
];

/// A volume's positions: inside, past each axis's edges, at level 1 inside and past each axis of
/// its 2 by 2 by 2, and at levels past the texture's.
const VOLUME: [[i32; 4]; 13] = [
    [1, 2, 3, 0],
    [5, 1, 1, 0],
    [-1, 1, 1, 0],
    [1, 4, 1, 0],
    [1, -2, 1, 0],
    [1, 1, 4, 0],
    [1, 1, -1, 0],
    [1, 1, 1, 1],
    [2, 1, 1, 1],
    [1, 2, 1, 1],
    [1, 1, 2, 1],
    [0, 0, 0, 2],
    [0, 0, 0, -1],
];

/// A line's or a texture buffer's positions: inside, at each end, and past each end.
const LINE: [[i32; 4]; 8] = [
    [0, 0, 0, 0],
    [3, 0, 0, 0],
    [7, 0, 0, 0],
    [8, 0, 0, 0],
    [9, 0, 0, 0],
    [-1, 0, 0, 0],
    [100, 0, 0, 0],
    [-8, 0, 0, 0],
];

/// What a kernel stores a thread.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Out {
    Float4,
    Int4,
    Half4,
    /// The texel's alpha alone.
    Float,
}

impl Out {
    /// The element a diff names, and the elements a thread stores.
    const fn element(self) -> (Element, u32) {
        match self {
            Out::Float4 => (Element::U32, 4),
            Out::Int4 => (Element::I32, 4),
            Out::Half4 => (Element::U16, 4),
            Out::Float => (Element::U32, 1),
        }
    }

    /// The bytes a thread stores.
    fn bytes(self) -> u32 {
        let (element, elements) = self.element();
        element.size() * elements
    }
}

/// The sampler a read goes through, beside the kernel's own constexpr one.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Through {
    /// The kernel's constexpr sampler.
    Constexpr,
    /// The runtime sampler at `[[sampler(0)]]`, of the named state.
    Slot(&'static str, SamplerSpec),
    /// An argument buffer's sampler at `[[buffer(2)]]`, of the named state.
    Table(&'static str, SamplerSpec),
    /// The kernel's constexpr sampler over an argument buffer's texture at `[[buffer(2)]]`.
    HeapTexture,
    /// The runtime sampler at `[[sampler(0)]]`, of the named state, over an argument buffer's
    /// texture at `[[buffer(2)]]`.
    HeapTextureSlot(&'static str, SamplerSpec),
}

/// One read: a kernel, what it reads through, and what it stores.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Read {
    entry: &'static str,
    through: Through,
    out: Out,
}

impl Read {
    /// The read's site.
    fn site(&self) -> String {
        match self.through {
            Through::Slot(state, _)
            | Through::Table(state, _)
            | Through::HeapTextureSlot(state, _) => format!("{}/{state}", self.entry),
            Through::Constexpr | Through::HeapTexture => self.entry.to_owned(),
        }
    }
}

/// The runtime samplers each case reads through: every address mode, each border colour, a mip
/// filter, a mixed state, and a linear filter (which changes nothing a read answers).
fn states() -> Vec<(&'static str, SamplerSpec)> {
    use SamplerAddressMode::{
        ClampToBorderColor as Border, ClampToZero as Zero, MirrorClampToEdge, MirrorRepeat, Repeat,
    };
    use SamplerBorderColor::{OpaqueBlack, OpaqueWhite};
    let all = |mode: SamplerAddressMode| {
        let mut spec = SamplerSpec::new().addressed(mode);
        spec.r_address_mode = mode;
        spec
    };
    let bordered = |mode, color| {
        let mut spec = all(mode);
        spec.border_color = color;
        spec
    };
    let mut mixed = bordered(Border, OpaqueWhite);
    mixed.t_address_mode = Repeat;
    mixed.r_address_mode = SamplerAddressMode::ClampToEdge;
    mixed.mip_filter = SamplerMipFilter::Nearest;
    let mut linear = SamplerSpec::filtered(SamplerMinMagFilter::Linear);
    linear.mip_filter = SamplerMipFilter::Linear;
    let mut mipped = all(Border);
    mipped.mip_filter = SamplerMipFilter::Nearest;
    vec![
        ("edge", SamplerSpec::new()),
        ("repeat", all(Repeat)),
        ("mirrored-repeat", all(MirrorRepeat)),
        ("mirror-clamp", all(MirrorClampToEdge)),
        ("zero", all(Zero)),
        ("zero-white", bordered(Zero, OpaqueWhite)),
        ("border-black", bordered(Border, OpaqueBlack)),
        ("border-white", bordered(Border, OpaqueWhite)),
        ("mixed-mip", mixed),
        ("linear-mip", linear),
        ("border-mip", mipped),
    ]
}

/// Runtime samplers that tell a texture buffer's axes apart: one axis's mode changed from
/// `clamp_to_edge` at a time, and pairs of them, every border opaque white. A texture buffer's
/// index is its `s` coordinate, so these ask which of the sampler's modes Metal's read past its end
/// answers by.
fn axis_states() -> Vec<(&'static str, SamplerSpec)> {
    use SamplerAddressMode::{
        ClampToBorderColor as B, ClampToEdge as E, ClampToZero as Z, MirrorRepeat as M, Repeat as R,
    };
    let state = |s, t, r| {
        let mut spec = SamplerSpec::new();
        spec.s_address_mode = s;
        spec.t_address_mode = t;
        spec.r_address_mode = r;
        spec.border_color = SamplerBorderColor::OpaqueWhite;
        spec
    };
    vec![
        ("s-border", state(B, E, E)),
        ("t-border", state(E, B, E)),
        ("r-border", state(E, E, B)),
        ("st-border", state(B, B, E)),
        ("sr-border", state(B, E, B)),
        ("tr-border", state(E, B, B)),
        ("s-zero", state(Z, E, E)),
        ("t-zero", state(E, Z, E)),
        ("s-repeat", state(R, E, E)),
        ("t-repeat", state(E, R, E)),
        ("r-repeat", state(E, E, R)),
        ("s-mirrored", state(M, E, E)),
        ("s-border-t-repeat", state(B, R, R)),
        ("s-border-t-mirrored", state(B, M, E)),
    ]
}

/// The reads of a `texture2d<float>`: every constexpr state, every runtime state through the slot
/// and through an argument buffer, and the argument buffer's texture through `clamp_to_zero`.
fn plane_reads() -> Vec<Read> {
    let own = |entry| Read {
        entry,
        through: Through::Constexpr,
        out: Out::Float4,
    };
    let mut reads: Vec<Read> = [
        "rt_repeat",
        "rt_mirrored_offset",
        "rt_zero",
        "rt_border_ob",
        "rt_border_ow",
        "rt_mixed",
        "rt_mip",
        "rt_mip_zero",
        "rt_filtered",
    ]
    .into_iter()
    .map(own)
    .collect();
    reads.push(Read {
        entry: "rt_alpha",
        through: Through::Constexpr,
        out: Out::Float,
    });
    for (state, spec) in states() {
        reads.push(Read {
            entry: "rt_arg",
            through: Through::Slot(state, spec),
            out: Out::Float4,
        });
    }
    for (state, spec) in states() {
        reads.push(Read {
            entry: "rt_table",
            through: Through::Table(state, spec.for_argument_buffer()),
            out: Out::Float4,
        });
    }
    reads.push(Read {
        entry: "rt_heap_zero",
        through: Through::HeapTexture,
        out: Out::Float4,
    });
    reads
}

/// One case: a texture, and the reads of it.
pub struct ReadThrough {
    name: &'static str,
    format: PixelFormat,
    kind: Kind,
    swizzle: TextureSwizzleChannels,
    reads: Vec<Read>,
}

/// Every case of the family.
pub fn all() -> Vec<ReadThrough> {
    use TextureSwizzle::{Green as G, One as O, Red as R, Zero as Z};
    let case = |name, format, kind, reads| ReadThrough {
        name,
        format,
        kind,
        swizzle: TextureSwizzleChannels::DEFAULT,
        reads,
    };
    let own = |entry, out| Read {
        entry,
        through: Through::Constexpr,
        out,
    };
    let slot = |entry| -> Vec<Read> {
        states()
            .into_iter()
            .map(|(state, spec)| Read {
                entry,
                through: Through::Slot(state, spec),
                out: Out::Float4,
            })
            .collect()
    };
    let mut swizzled = case(
        "read-through-rgba8-swizzled",
        PixelFormat::RGBA8Unorm,
        Kind::Plane,
        plane_reads(),
    );
    swizzled.swizzle = TextureSwizzleChannels::new(G, R, O, Z);
    let mut volume = vec![own("rt_volume", Out::Float4)];
    volume.extend(slot("rt_arg_volume"));
    let mut buffer = vec![
        own("rt_buffer", Out::Float4),
        own("rt_buffer_ow", Out::Float4),
    ];
    buffer.extend(slot("rt_arg_buffer"));
    buffer.extend(axis_states().into_iter().map(|(state, spec)| Read {
        entry: "rt_arg_buffer",
        through: Through::Slot(state, spec),
        out: Out::Float4,
    }));
    vec![
        case(
            "read-through-r32f",
            PixelFormat::R32Float,
            Kind::Plane,
            plane_reads(),
        ),
        case(
            "read-through-rgba8",
            PixelFormat::RGBA8Unorm,
            Kind::Plane,
            plane_reads(),
        ),
        swizzled,
        case(
            "read-through-a8",
            PixelFormat::A8Unorm,
            Kind::Plane,
            plane_reads(),
        ),
        case(
            "read-through-layers",
            PixelFormat::RGBA32Float,
            Kind::Layers,
            vec![own("rt_array", Out::Float4)],
        ),
        case(
            "read-through-volume",
            PixelFormat::RGBA32Float,
            Kind::Volume,
            volume,
        ),
        case(
            "read-through-line",
            PixelFormat::RGBA32Float,
            Kind::Line,
            vec![own("rt_line", Out::Float4)],
        ),
        case(
            "read-through-read-write",
            PixelFormat::RGBA32Float,
            Kind::ReadWrite,
            vec![own("rt_rw", Out::Float4)],
        ),
        case(
            "read-through-buffer",
            PixelFormat::RGBA32Float,
            Kind::Buffer,
            buffer,
        ),
        case(
            "read-through-int",
            PixelFormat::RGBA32Sint,
            Kind::Plane,
            vec![own("rt_int", Out::Int4)],
        ),
        case(
            "read-through-half",
            PixelFormat::RGBA16Float,
            Kind::Plane,
            vec![own("rt_half", Out::Half4)],
        ),
        case(
            "read-through-heap-border",
            PixelFormat::R32Float,
            Kind::Plane,
            vec![Read {
                entry: "rt_heap_ow",
                through: Through::HeapTexture,
                out: Out::Float4,
            }],
        ),
        case(
            "read-through-heap-sampler",
            PixelFormat::R32Float,
            Kind::Plane,
            vec![Read {
                entry: "rt_heap_arg",
                through: Through::HeapTextureSlot("edge", SamplerSpec::new()),
                out: Out::Float4,
            }],
        ),
    ]
}

/// The half-float bits of the integer `n`, below 2048 and above 0, which a half holds exactly.
fn half_of(n: u32) -> u16 {
    debug_assert!(n > 0 && n < 2048);
    let exponent = 31 - n.leading_zeros();
    let mantissa = (n << (10 - exponent)) & 0x3ff;
    ((exponent + 15) << 10 | mantissa) as u16
}

impl ReadThrough {
    /// The fixture under `entry`.
    fn fixture(entry: &'static str) -> Fixture {
        Fixture {
            name: FIXTURE,
            entry,
        }
    }

    /// The texel's value at `(x, y, layer or z)` of `level`: distinct over the whole texture, and
    /// small enough for every channel to hold it exactly.
    fn value(&self, level: u32, [x, y, z]: [u32; 3]) -> u32 {
        let [_, _, d] = self.kind.extent();
        1 + x + 4 * y + 16 * z + 16 * d * level
    }

    /// One texel's bytes in the case's format.
    fn texel(&self, v: u32) -> Vec<u8> {
        match self.format {
            PixelFormat::R32Float => (v as f32).to_le_bytes().to_vec(),
            PixelFormat::RGBA8Unorm => vec![v as u8, 64 + v as u8, 128 + v as u8, 192 + v as u8],
            PixelFormat::A8Unorm => vec![v as u8],
            PixelFormat::RGBA16Float => (0..4)
                .flat_map(|c| half_of(v + 256 * c).to_le_bytes())
                .collect(),
            PixelFormat::RGBA32Sint => [v as i32, -(v as i32), v as i32 + 256, -(v as i32) - 512]
                .into_iter()
                .flat_map(i32::to_le_bytes)
                .collect(),
            _ => (0..4)
                .flat_map(|c| ((v + 256 * c) as f32).to_le_bytes())
                .collect(),
        }
    }

    /// The bytes of one texel.
    fn texel_bytes(&self) -> u32 {
        self.texel(1).len() as u32
    }

    /// The bytes of each upload: one per level and, for an array, per slice.
    fn levels(&self) -> Vec<(u32, u32, Vec<u8>)> {
        let mut out = Vec::new();
        for level in 0..self.kind.levels() {
            let [w, h, d] = self.kind.level_extent(level);
            let slices = if self.kind == Kind::Layers { d } else { 1 };
            let depth = if self.kind == Kind::Layers { 1 } else { d };
            for slice in 0..slices {
                let mut bytes = Vec::new();
                for z in 0..depth {
                    for y in 0..h {
                        for x in 0..w {
                            bytes.extend(self.texel(self.value(level, [x, y, z.max(slice)])));
                        }
                    }
                }
                out.push((level, slice, bytes));
            }
        }
        out
    }

    /// The texture's uploads over `levels`' bytes.
    fn uploads<'a>(&self, levels: &'a [(u32, u32, Vec<u8>)]) -> Vec<Upload<'a>> {
        levels
            .iter()
            .map(|(level, slice, bytes)| {
                let [w, h, d] = self.kind.level_extent(*level);
                let depth = if self.kind == Kind::Volume { d } else { 1 };
                let row_pitch = w * self.texel_bytes();
                Upload {
                    level: *level,
                    slice: *slice,
                    origin: [0, 0, 0],
                    size: [w, h, depth],
                    bytes,
                    row_pitch,
                    rows: h * depth,
                    image_pitch: if depth > 1 { row_pitch * h } else { 0 },
                }
            })
            .collect()
    }

    /// The texture's request, loaded with `uploads`; a texture buffer's is over `buffer`.
    fn request<'a>(&self, uploads: &'a [Upload<'a>], buffer: Option<Handle>) -> TextureRequest<'a> {
        let [w, h, d] = self.kind.extent();
        let base = TextureRequest::read_2d(self.format.as_raw(), w, h)
            .levels(self.kind.levels())
            .swizzled(self.swizzle)
            .loaded(uploads);
        match (self.kind, buffer) {
            (Kind::Buffer, Some(buffer)) => TextureRequest::texel_buffer(
                self.format.as_raw(),
                w,
                buffer,
                0,
                u64::from(w * self.texel_bytes()),
            ),
            (Kind::Layers, _) => TextureRequest {
                texture_type: TextureType::Type2DArray,
                slices: d,
                ..base
            },
            (Kind::Volume, _) => TextureRequest {
                texture_type: TextureType::Type3D,
                size: [w, h, d],
                ..base
            },
            (Kind::Line, _) => TextureRequest {
                texture_type: TextureType::Type1D,
                ..base
            },
            (Kind::ReadWrite, _) => {
                base.used(TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE)
            }
            _ => base,
        }
    }

    /// The bytes the texture buffer is made over: its texels, in order.
    fn buffer_texels(&self) -> Vec<u8> {
        let [w, _, _] = self.kind.extent();
        (0..w)
            .flat_map(|x| self.texel(self.value(0, [x, 0, 0])))
            .collect()
    }

    /// Buffer 0: each thread's position, four `int`s.
    fn positions(&self) -> Vec<u8> {
        self.kind
            .positions()
            .iter()
            .flatten()
            .flat_map(|word| word.to_le_bytes())
            .collect()
    }

    /// The threads a dispatch runs: one a position.
    fn threads(&self) -> u32 {
        self.kind.positions().len() as u32
    }

    /// An output buffer before its dispatch.
    fn blank(&self, out: Out) -> Vec<u8> {
        vec![BLANK; (self.threads() * out.bytes()) as usize]
    }

    /// The layout a read-back compares under: one row a thread.
    fn layout(&self, out: Out) -> ByteLayout {
        let (element, elements) = out.element();
        let length = self.threads() * out.bytes();
        ByteLayout {
            region: Region::Buffer {
                offset: 0,
                length: u64::from(length),
            },
            element,
            row_offset: 0,
            row_elements: elements,
            row_pitch: out.bytes(),
            rows: self.threads(),
            image_pitch: length,
            images: 1,
            gaps: Gaps::Ignored,
        }
    }

    /// The distinct kernels the case runs, in the order it first runs each.
    fn entries(&self) -> Vec<&'static str> {
        let mut entries: Vec<&'static str> = Vec::new();
        for read in &self.reads {
            if !entries.contains(&read.entry) {
                entries.push(read.entry);
            }
        }
        entries
    }

    /// Run one read and answer the bytes it wrote.
    fn dispatch(
        &self,
        side: &mut dyn Side,
        sites: &mut Sites,
        read: &Read,
        pipeline: Handle,
        texture: Handle,
        positions: Handle,
    ) -> Result<Vec<u8>, Abort> {
        let site = read.site();
        let at = |step: &str| format!("{site}/{step}");
        let out = made(sites, &at("buffer"), side.buffer(&self.blank(read.out)))?;
        let (textures, samplers, table) = match read.through {
            Through::Constexpr => (vec![texture], Vec::new(), None),
            Through::Slot(_, spec) => {
                let sampler = made(sites, &at("sampler"), side.sampler(&spec))?;
                (vec![texture], vec![sampler], None)
            }
            Through::Table(_, spec) => {
                let sampler = made(sites, &at("sampler"), side.sampler(&spec))?;
                let table = made(
                    sites,
                    &at("argument-buffer"),
                    side.argument_buffer(&[ArgumentSlot::Sampler(sampler)]),
                )?;
                (vec![texture], Vec::new(), Some(table))
            }
            Through::HeapTexture => {
                let table = made(
                    sites,
                    &at("argument-buffer"),
                    side.argument_buffer(&[ArgumentSlot::Texture(texture)]),
                )?;
                (Vec::new(), Vec::new(), Some(table))
            }
            Through::HeapTextureSlot(_, spec) => {
                let sampler = made(sites, &at("sampler"), side.sampler(&spec))?;
                let table = made(
                    sites,
                    &at("argument-buffer"),
                    side.argument_buffer(&[ArgumentSlot::Texture(texture)]),
                )?;
                (Vec::new(), vec![sampler], Some(table))
            }
        };
        let mut buffers = vec![positions, out];
        buffers.extend(table);
        let threads = self.threads();
        let command_buffer = made(
            sites,
            &at("dispatch"),
            side.dispatch_threads(
                pipeline,
                &Bindings::new(&textures, &buffers).sampled(&samplers),
                &[],
                [threads, 1, 1],
                [threads, 1, 1],
            ),
        )?;
        call(
            sites,
            &at("status"),
            side.wait(command_buffer),
            Clone::clone,
        )?;
        call(
            sites,
            &at("read-back"),
            side.read_buffer(out, 0, self.blank(read.out).len()),
            |_| crate::transcript::Value::Absent,
        )
    }
}

impl Case for ReadThrough {
    fn name(&self) -> &'static str {
        self.name
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn allowances(&self) -> Vec<crate::allowance::Allowance> {
        // A read through an argument buffer's sampler or texture takes the handle heap, whose set
        // MoltenVK's layout query refuses though it runs it: measured on the M3 Max through
        // MoltenVK, where all four plane cases announced `set-past-layout-support` and nothing
        // else.
        let heap = self.reads.iter().any(|read| {
            matches!(
                read.through,
                Through::Table(..) | Through::HeapTexture | Through::HeapTextureSlot(..)
            )
        });
        match heap {
            true => vec![crate::replay::set_past_layout_notice()],
            false => Vec::new(),
        }
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        let mut shaders = Vec::new();
        for entry in self.entries() {
            for extension in [RUNTIME_TEXT, METAL_TEXT] {
                shaders.push(ShaderRef {
                    fixture: format!("{FIXTURE}{extension}"),
                    entry: entry.to_owned(),
                    source_sha256: crate::sha256_hex(
                        Self::fixture(entry).text(extension)?.as_bytes(),
                    ),
                });
            }
        }
        Ok(shaders)
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let fixture = Self::fixture(FIXTURE);
        let mut inputs = vec![
            Input {
                name: "air".to_owned(),
                digest: Digest::sha256(fixture.text(RUNTIME_TEXT)?.as_bytes()),
            },
            Input {
                name: "air-typed".to_owned(),
                digest: Digest::sha256(fixture.text(METAL_TEXT)?.as_bytes()),
            },
            Input::bytes("buffer:positions", &self.positions()),
            Input::bytes("buffer:blank", &[BLANK]),
        ];
        let levels = self.levels();
        let uploads = self.uploads(&levels);
        match self.kind {
            Kind::Buffer => {
                inputs.push(Input::bytes("buffer:texels", &self.buffer_texels()));
                let request = self.request(&[], Some(Handle(0)));
                inputs.push(Input::bytes("texture", &request.words()));
            }
            _ => inputs.push(Input::bytes(
                "texture",
                &self.request(&uploads, None).words(),
            )),
        }
        // Every read: its kernel, the state it reads through, and what it stores, so a read that
        // moves to another kernel or another sampler is unrecorded rather than compared against
        // Metal's answer for the first.
        for read in &self.reads {
            let mut words = read.entry.as_bytes().to_vec();
            match read.through {
                Through::Constexpr => words.push(0),
                Through::Slot(_, spec) => {
                    words.push(1);
                    words.extend(spec.words());
                }
                Through::Table(_, spec) => {
                    words.push(2);
                    words.extend(spec.words());
                }
                Through::HeapTexture => words.push(3),
                Through::HeapTextureSlot(_, spec) => {
                    words.push(4);
                    words.extend(spec.words());
                }
            }
            words.push(read.out.bytes() as u8);
            inputs.push(Input::bytes(format!("read:{}", read.site()), &words));
        }
        Ok(inputs)
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let stop = |at: &str, why: String| Abort {
            at: at.to_owned(),
            failure: Failure::named("FixtureNotRead", why),
        };
        let loaded = Self::fixture(FIXTURE)
            .loaded()
            .map_err(|why| stop("library", why))?;
        let library = made(sites, "library", side.library(&loaded.source()))?;
        let positions = made(sites, "buffer/positions", side.buffer(&self.positions()))?;
        let levels = self.levels();
        let uploads = self.uploads(&levels);
        let texture = match self.kind {
            Kind::Buffer => {
                let texels = made(sites, "buffer/texels", side.buffer(&self.buffer_texels()))?;
                made(
                    sites,
                    "texture",
                    side.texture(&self.request(&[], Some(texels))),
                )?
            }
            _ => made(
                sites,
                "texture",
                side.texture(&self.request(&uploads, None)),
            )?,
        };
        let mut pipelines = BTreeMap::new();
        for entry in self.entries() {
            let pipeline = made(
                sites,
                &format!("pipeline/{entry}"),
                side.compute_pipeline(library, entry),
            )?;
            pipelines.insert(entry, pipeline);
        }
        let mut words = Node::branch("words", Vec::new());
        for read in &self.reads {
            let bytes = self.dispatch(
                side,
                sites,
                read,
                pipelines[&read.entry],
                texture,
                positions,
            )?;
            words = words.with(Node::leaf(
                read.site(),
                Observed::bytes(bytes, self.layout(read.out)),
            ));
        }
        sites.push(words);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    /// The cases name distinct rows, and each case's reads distinct sites.
    #[test]
    fn the_cases_and_their_reads_are_distinct() {
        let names: BTreeSet<&str> = all().iter().map(|case| case.name).collect();
        assert_eq!(names.len(), all().len());
        for case in all() {
            let sites: BTreeSet<String> = case.reads.iter().map(Read::site).collect();
            assert_eq!(sites.len(), case.reads.len(), "{}", case.name);
        }
    }

    /// Every texel of a texture holds a distinct value, so a read that took the wrong texel, the
    /// wrong level or the wrong slice shows as a value rather than as a coincidence.
    #[test]
    fn every_texel_is_distinct() {
        for case in all() {
            let mut seen = BTreeSet::new();
            for (_, _, bytes) in case.levels() {
                for texel in bytes.chunks(case.texel_bytes() as usize) {
                    assert!(seen.insert(texel.to_vec()), "{}: {texel:?}", case.name);
                }
            }
        }
    }

    /// Each upload holds exactly the bytes its region takes.
    #[test]
    fn every_upload_holds_its_region() {
        for case in all() {
            let levels = case.levels();
            for upload in case.uploads(&levels) {
                assert_eq!(upload.bytes.len(), upload.needed(), "{}", case.name);
            }
        }
    }

    /// A half holds each channel's integer exactly.
    #[test]
    fn half_of_is_exact() {
        for (n, bits) in [
            (1, 0x3c00),
            (2, 0x4000),
            (3, 0x4200),
            (1024, 0x6400),
            (1027, 0x6403),
        ] {
            assert_eq!(half_of(n), bits, "{n}");
        }
    }

    /// Every kernel a case runs is one of the fixture's.
    #[test]
    fn every_kernel_is_the_fixtures() {
        let text = ReadThrough::fixture(FIXTURE)
            .text(RUNTIME_TEXT)
            .expect("the fixture's text");
        for case in all() {
            for entry in case.entries() {
                assert!(
                    text.contains(&format!("@{entry}(")),
                    "{}: {entry}",
                    case.name
                );
            }
        }
    }
}
