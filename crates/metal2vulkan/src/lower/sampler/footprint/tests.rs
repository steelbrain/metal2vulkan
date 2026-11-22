//! The footprint word, run on the interpreter. Each kernel reads a normalized coordinate per
//! invocation and writes [`counted`]'s word for a sparse image whose texels are unmapped where a
//! case left them so. Against Metal's own answers (apple-m3-max-macos26.7: cases 39503d65,
//! efe06aaf, 363e39a2, 95ca7dc5, 29f15128, 760fce8c, and fd292944 on 2D textures, 203caac1 on a 3D
//! one), and, for a 3D and a 2D-array texture, against the rule applied per axis. Each runs through
//! both of a slot's arms: the image's sparse fetches, and a tile map of the same texels over an
//! image resident everywhere.

use std::collections::BTreeSet;

use crate::spv::ImageFormat;

use super::{counted, Footprint, Kind, Site};
use crate::memory::views::View;
use crate::pipeline::sym::{CombinedSampler, ResourceKind, ResourceSlot, SamplerRef, Stage};
use crate::pipeline::tile_map;
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
use crate::spirv::error::BResult;
use crate::spirv::ids::{GlobalId, TypeId, Value};
use crate::spirv::interp::{self, Buffer, Config, Image, Resources};
use crate::spirv::ir::{
    natural_layout, ConstBits, Constant, ConvOp, Dim, ExecMode, ExecModel, ImageType, ImageUsage,
    IoDir, LayoutRule, MemAccess, StorageClass, Type, TypedModule,
};

fn slot(kind: ResourceKind, index: u32) -> ResourceSlot {
    ResourceSlot {
        stage: Stage::Kernel,
        kind,
        index,
    }
}

/// Texture 0 through sampler slot 0 (the kernel's sampler state is not used: it only fetches).
fn combined() -> ResourceSlot {
    let sampler = CombinedSampler::new(SamplerRef::Slot(0), false, false);
    slot(ResourceKind::CombinedTextureSampler(sampler), 0)
}

fn storage_buffer(
    b: &mut ModuleBuilder,
    index: u32,
    elem: TypeId,
    stride: u32,
) -> BResult<GlobalId> {
    let rta = b.ty(Type::RuntimeArray {
        elem,
        stride: Some(stride),
    })?;
    let layout = natural_layout(b.types(), LayoutRule::Std430, true, &[rta])?;
    let block = b.ty(Type::Struct {
        members: [rta].into(),
        layout: Some(layout),
    })?;
    b.resource(
        slot(ResourceKind::Buffer, index),
        StorageClass::StorageBuffer,
        block,
    )
}

fn element(f: &mut FunctionBuilder<'_>, g: GlobalId, i: Value) -> BResult<Value> {
    let u32_ = f.module().ty(Type::Int {
        width: 32,
        signed: false,
    })?;
    let c = f.module().constant(Constant::Scalar {
        ty: u32_,
        bits: ConstBits(0),
    })?;
    let zero = f.module().const_value(c)?;
    let base = f.module().global_value(g)?;
    f.access_chain(base, &[zero, i])
}

/// Where a kernel reads a texel's residency: a tile map in buffer 2 with tiles of the given texels
/// (x, y, z), made from the image's unmapped texels, which must be whole tiles; the image itself is
/// then resident everywhere, so only the map can answer.
#[derive(Clone, Copy, Debug)]
struct Arm([u32; 3]);

/// A kernel whose invocation `i` reads `vec4` probe `i` of buffer 0 (the coordinate in its first
/// `axes` lanes; the layer, as a float, in lane 3 of an array) and writes the word of `fp` at it,
/// through buffer 2's tile map, to `uint` element `i` of buffer 1.
fn kernel(fp: Footprint, dim: Dim, arrayed: bool) -> BResult<TypedModule> {
    let mut b = ModuleBuilder::new();
    let f32_ = b.ty(Type::Float { width: 32 })?;
    let u32_ = b.ty(Type::Int {
        width: 32,
        signed: false,
    })?;
    let vec4 = b.ty(Type::Vector { elem: f32_, len: 4 })?;
    let uvec3 = b.ty(Type::Vector { elem: u32_, len: 3 })?;
    let image = b.ty(Type::Image(ImageType {
        sampled_type: f32_,
        dim,
        depth: false,
        arrayed,
        multisampled: false,
        usage: ImageUsage::Sampled,
        format: ImageFormat::Unknown,
    }))?;
    let sampled = b.ty(Type::SampledImage { image })?;
    let texture = b.resource(combined(), StorageClass::UniformConstant, sampled)?;
    let probes = storage_buffer(&mut b, 0, vec4, 16)?;
    let words = storage_buffer(&mut b, 1, u32_, 4)?;
    let map = storage_buffer(&mut b, 2, u32_, 4)?;
    let map = View {
        global: map,
        u32_ty: u32_,
    };
    let id = b.builtin(crate::spv::BuiltIn::GlobalInvocationId, IoDir::Input, uvec3)?;
    let void = b.ty(Type::Void)?;
    let mut f = b.function(void)?;
    let gid = f.module().global_value(id)?;
    let gid = f.load(gid, MemAccess::default())?;
    let i = f.composite_extract(gid, &[0])?;
    let at = element(&mut f, probes, i)?;
    let p = f.load(at, MemAccess::default())?;
    let lanes = fp.axes;
    let pick: Vec<u32> = (0..u32::from(lanes)).collect();
    let coord = f.vector_shuffle(p, p, &pick)?;
    let layer = if arrayed {
        let l = f.composite_extract(p, &[3])?;
        Some(f.convert(ConvOp::ConvertFToU, l, u32_)?)
    } else {
        None
    };
    let site = Site {
        binding: crate::lower::sampler::Binding::Combined(texture),
        coord,
        lanes,
        layer,
    };
    let word = counted(&mut f, fp, map, site)?;
    let out = element(&mut f, words, i)?;
    f.store(out, word, MemAccess::default())?;
    f.ret(None)?;
    let main = f.finish()?;
    b.entry(
        ExecModel::GLCompute,
        main,
        vec![ExecMode::LocalSize { x: 1, y: 1, z: 1 }],
    )?;
    b.finalize()
}

/// A sparse image of `extent` and `layers` whose texel `(x, y, z)` of layer `l` is unmapped where
/// `unmapped(x, y, z, l)`.
fn image(
    dim: Dim,
    extent: [u32; 3],
    layers: u32,
    unmapped: impl Fn(u32, u32, u32, u32) -> bool,
) -> Image {
    let mut img = Image::new(combined(), extent, layers, 1, 1);
    let [w, h, d] = extent;
    let mut set = BTreeSet::new();
    let mut i = 0usize;
    for l in 0..layers {
        for z in 0..d {
            for y in 0..h {
                for x in 0..w {
                    if unmapped(x, y, z, l) {
                        set.insert(i);
                    }
                    i += 1;
                }
            }
        }
    }
    assert_eq!(i, img.texels.len(), "{dim:?} texel count");
    img.unmapped = set;
    img
}

/// The tile map of `img`'s level 0 with tiles of `tile` texels: a tile is mapped where none of its
/// texels is unmapped, and a tile only some of whose texels are unmapped is refused.
fn tile_map(img: &Image, tile: [u32; 3]) -> Vec<u8> {
    let [w, h, d] = img.extent;
    let [tw, th, td] = tile;
    tile_map::one_level_map(img.extent, img.layers, tile, |l, [tx, ty, tz]| {
        let (mut texels, mut unmapped) = (0, 0);
        for z in tz * td..((tz + 1) * td).min(d) {
            for y in ty * th..((ty + 1) * th).min(h) {
                for x in tx * tw..((tx + 1) * tw).min(w) {
                    let i = (((l * d + z) * h + y) * w + x) as usize;
                    texels += 1;
                    unmapped += usize::from(img.unmapped.contains(&i));
                }
            }
        }
        assert!(
            unmapped == 0 || unmapped == texels,
            "tile ({tx}, {ty}, {tz}) of layer {l} is partly unmapped"
        );
        unmapped == 0
    })
}

/// The words of `fp` at `probes` (each `[u, v, w, layer]`) through `arm`.
fn words(
    fp: Footprint,
    dim: Dim,
    arrayed: bool,
    mut img: Image,
    probes: &[[f32; 4]],
    arm: Arm,
) -> Vec<u32> {
    let m = kernel(fp, dim, arrayed).expect("kernel");
    let bytes: Vec<u8> = probes
        .iter()
        .flat_map(|p| p.iter().flat_map(|x| x.to_le_bytes()))
        .collect();
    let n = u32::try_from(probes.len()).expect("probe count");
    let mut buffers = vec![
        Buffer {
            slot: Some(slot(ResourceKind::Buffer, 0)),
            address: None,
            bytes,
        },
        Buffer {
            slot: Some(slot(ResourceKind::Buffer, 1)),
            address: None,
            bytes: vec![0xAB; probes.len() * 4],
        },
    ];
    let Arm(tile) = arm;
    buffers.push(Buffer {
        slot: Some(slot(ResourceKind::Buffer, 2)),
        address: None,
        bytes: tile_map(&img, tile),
    });
    img.unmapped.clear();
    let res = Resources {
        buffers,
        ..Resources::default()
    };
    let cfg = Config {
        workgroups: [n, 1, 1],
        ..Config::default()
    };
    let (out, _) = interp::run_with_images(&m, &cfg, res, vec![img]).expect("run");
    out.buffers
        .get(1)
        .expect("words")
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn footprint(kind: Kind, axes: u8, border: bool) -> Footprint {
    Footprint {
        kind,
        axes,
        border: [border; 3],
    }
}

/// One of Metal's answers: the access, the bits of `u` and `v`, and `resident()`.
type Reading = (Kind, u32, u32, bool);

const L: Kind = Kind::Linear;
const N: Kind = Kind::Nearest;
const G: Kind = Kind::Gather;

/// 256x128 r32_float, its left 128x128 mapped (39503d65, 95ca7dc5, 760fce8c, 29f15128; the last
/// samples at levels 1 to 3.5 of a one-level texture, which reads level 0).
const LEFT_256: &[Reading] = &[
    (L, 0x3eff0000, 0x3d880000, true),
    (L, 0x3eff0001, 0x3d880000, true),
    (L, 0x3eff0002, 0x3d880000, true),
    (L, 0x3eff0004, 0x3d880000, true),
    (L, 0x3eff0008, 0x3d880000, true),
    (L, 0x3eff0010, 0x3d880000, true),
    (L, 0x3eff0020, 0x3d880000, true),
    (L, 0x3eff0040, 0x3d880000, true),
    (L, 0x3eff0080, 0x3d880000, true),
    (L, 0x3eff00c0, 0x3d880000, true),
    (L, 0x3eff00ff, 0x3d880000, true),
    (L, 0x3eff0100, 0x3d880000, false),
    (L, 0x3eff0101, 0x3d880000, false),
    (L, 0x3eff0180, 0x3d880000, false),
    (L, 0x3eff01ff, 0x3d880000, false),
    (L, 0x3eff0200, 0x3d880000, false),
    (L, 0x3eff0201, 0x3d880000, false),
    (L, 0x3eff0300, 0x3d880000, false),
    (L, 0x3eff0500, 0x3d880000, false),
    (L, 0x3eff03ff, 0x3d880000, false),
    (L, 0x3eff0400, 0x3d880000, false),
    (L, 0x3eff0401, 0x3d880000, false),
    (L, 0x3eff0600, 0x3d880000, false),
    (L, 0x3eff0800, 0x3d880000, false),
    (L, 0x3eff1000, 0x3d880000, false),
    (L, 0x3eff2000, 0x3d880000, false),
    (L, 0x3eff8000, 0x3d880000, false),
    (L, 0x3eff0080, 0x3d880400, true),
    (L, 0x3eff0101, 0x3d880808, false),
    (L, 0x3eff0200, 0x3d881000, false),
    (N, 0x3effffff, 0x3d880000, true),
    (N, 0x3effffe0, 0x3d880000, true),
    (N, 0x3effff00, 0x3d880000, true),
    (N, 0x3efffe00, 0x3d880000, true),
    (L, 0x3e810000, 0x3f7f0020, true),
    (L, 0x3e810000, 0x3f7f0101, true),
    (L, 0x3e810000, 0x3f7f0201, true),
    (L, 0x3e810000, 0x3b7fe000, true),
    (L, 0x3e810000, 0x3b7eff00, true),
    (L, 0x3e810000, 0x3b7dff00, true),
    // 95ca7dc5: probe 2 is the texel center whose zero-weight neighbour is unmapped.
    (L, 0x3e800000, 0x3d800000, true),
    (L, 0x3f000000, 0x3d800000, false),
    (L, 0x3eff0000, 0x3d880000, true),
    (L, 0x3efd0000, 0x3d880000, true),
    (L, 0x3f008000, 0x3d880000, false),
    (L, 0x3eff8000, 0x3d880000, false),
    (G, 0x3f000000, 0x3d800000, false),
    (G, 0x3efe0000, 0x3d800000, true),
    (G, 0x3f010000, 0x3d800000, false),
    (N, 0x3eff8000, 0x3d880000, true),
    (N, 0x3f004000, 0x3d880000, false),
    (L, 0x00000000, 0x3d800000, true),
    (L, 0x3f800000, 0x3d800000, false),
    // 760fce8c.
    (N, 0x3cb00000, 0x3d700000, true),
    (N, 0x3f488000, 0x3d700000, false),
    (N, 0x3eff8000, 0x3d700000, true),
    (N, 0x3f004000, 0x3d700000, false),
    (N, 0xbe800000, 0x3ec00000, true),
    // 29f15128.
    (L, 0x3cb00000, 0x3d700000, true),
    (L, 0x3f488000, 0x3d700000, false),
    (G, 0x3cc00000, 0x3d800000, true),
    (G, 0x3f490000, 0x3d800000, false),
];

/// 256x128, its right 128x128 mapped (efe06aaf).
const RIGHT_256: &[Reading] = &[
    (L, 0x3f008000, 0x3d880000, true),
    (L, 0x3f007fff, 0x3d880000, true),
    (L, 0x3f007ffe, 0x3d880000, true),
    (L, 0x3f007ffc, 0x3d880000, true),
    (L, 0x3f007ff8, 0x3d880000, true),
    (L, 0x3f007ff0, 0x3d880000, true),
    (L, 0x3f007fe0, 0x3d880000, true),
    (L, 0x3f007fc0, 0x3d880000, true),
    (L, 0x3f007fa0, 0x3d880000, true),
    (L, 0x3f007f80, 0x3d880000, true),
    (L, 0x3f007f40, 0x3d880000, false),
    (L, 0x3f007f00, 0x3d880000, false),
    (L, 0x3f007e80, 0x3d880000, false),
    (L, 0x3f007d80, 0x3d880000, false),
    (L, 0x3f007e00, 0x3d880000, false),
    (L, 0x3f007d00, 0x3d880000, false),
    (L, 0x3f007c00, 0x3d880000, false),
    (L, 0x3f007800, 0x3d880000, false),
    (L, 0x3f007000, 0x3d880000, false),
    (L, 0x3f004000, 0x3d880000, false),
    (G, 0x3f008000, 0x3d880000, true),
    (G, 0x3f007ff0, 0x3d880000, true),
    (G, 0x3f007f80, 0x3d880000, true),
    (G, 0x3f007f00, 0x3d880000, false),
    (N, 0x3effffff, 0x3d880000, false),
    (N, 0x3effffe0, 0x3d880000, false),
    (N, 0x3effff00, 0x3d880000, false),
    (N, 0x3efffe00, 0x3d880000, false),
    // Beside the right edge: the clamp-to-zero border counts nothing.
    (L, 0x3f7f8010, 0x3d880000, true),
    (L, 0x3f7f8080, 0x3d880000, true),
    (L, 0x3f7f8100, 0x3d880000, true),
];

/// 384x128, x 128..256 mapped (363e39a2: off the sweep, where `u · 384` rounds).
const MIDDLE_384: &[Reading] = &[
    (L, 0x3f2a5560, 0x3d880000, true),
    (L, 0x3f2a5580, 0x3d880000, true),
    (L, 0x3f2a55ab, 0x3d880000, false),
    (L, 0x3f2a55d5, 0x3d880000, false),
    (L, 0x3f2a5600, 0x3d880000, false),
    (L, 0x3f2a5601, 0x3d880000, false),
    (L, 0x3f2a5655, 0x3d880000, false),
    (L, 0x3f2a56ab, 0x3d880000, false),
    (L, 0x3eab5540, 0x3d880000, true),
    (L, 0x3eab5500, 0x3d880000, true),
    (L, 0x3eab54ab, 0x3d880000, true),
    (L, 0x3eab54a9, 0x3d880000, false),
    (L, 0x3eab5455, 0x3d880000, false),
    (L, 0x3eab5400, 0x3d880000, false),
    (L, 0x3eab53ff, 0x3d880000, false),
    (L, 0x3eab5355, 0x3d880000, false),
    (L, 0x3eab52ab, 0x3d880000, false),
    (L, 0x3eff0000, 0x3d880000, true),
    (L, 0x3f2a8e39, 0x3d880000, false),
    (G, 0x3f2aaaab, 0x3d880000, false),
    (N, 0x3f2aaaab, 0x3d880000, false),
];

fn check(width: u32, mapped: std::ops::Range<u32>, border: bool, readings: &[Reading]) {
    check_arm(width, mapped, border, readings, Arm([64, 64, 1]));
}

fn check_arm(
    width: u32,
    mapped: std::ops::Range<u32>,
    border: bool,
    readings: &[Reading],
    arm: Arm,
) {
    let mut seen = 0;
    for kind in [L, N, G] {
        let probes: Vec<[f32; 4]> = readings
            .iter()
            .filter(|r| r.0 == kind)
            .map(|r| [f32::from_bits(r.1), f32::from_bits(r.2), 0.0, 0.0])
            .collect();
        let want: Vec<u32> = readings
            .iter()
            .filter(|r| r.0 == kind)
            .map(|r| u32::from(!r.3))
            .collect();
        if probes.is_empty() {
            continue;
        }
        let img = image(Dim::D2, [width, 128, 1], 1, |x, _, _, _| {
            !mapped.contains(&x)
        });
        let got = words(
            footprint(kind, 2, border),
            Dim::D2,
            false,
            img,
            &probes,
            arm,
        );
        for (k, (g, w)) in got.iter().zip(&want).enumerate() {
            assert_eq!(
                g,
                w,
                "{arm:?} {kind:?} at u = {:#010x}, v = {} on {width} texels mapped {mapped:?}",
                probes[k][0].to_bits(),
                probes[k][1]
            );
        }
        seen += got.len();
    }
    assert_eq!(seen, readings.len());
}

/// Metal's measured words at every probe its cases put beside an unmapped column: the linear
/// quantum on either side of the edge (ties up), the gather's same position counting all four
/// texels, the nearest filter's unquantized `floor`, the border counting nothing, and the
/// position `u · N` rounded to `float` before it is quantized (363e39a2 at 384 texels).
#[test]
fn the_footprint_word_answers_metal_beside_an_unmapped_column() {
    check(256, 0..128, true, LEFT_256);
    check(256, 128..256, true, RIGHT_256);
    check(384, 128..256, true, MIDDLE_384);
}

/// 256x128, its right 128x128 mapped, so the left edge column is unmapped (fd292944), through
/// clamp_to_edge samplers.
const EDGE: &[Reading] = &[
    (N, 0xbf000000, 0x3d880000, false),
    (N, 0x3fc00000, 0x3d880000, true),
    (N, 0x3f400000, 0xbf000000, true),
    (N, 0x3f400000, 0x3fc00000, true),
    (L, 0xbf000000, 0x3d880000, false),
    (L, 0x3fc00000, 0x3d880000, true),
    (L, 0x3f400000, 0xbf000000, true),
    (L, 0x3f008000, 0x3d880000, true),
    (L, 0x3f004000, 0x3d880000, false),
    (L, 0x3b000000, 0x3d880000, false),
    (L, 0xbb000000, 0x3d880000, false),
    (G, 0xbf000000, 0x3d880000, false),
    (G, 0x3fc00000, 0x3d880000, true),
    (G, 0x3f800000, 0x3d880000, true),
];

/// The same texture through clamp_to_border (opaque white) samplers (fd292944).
const BORDER: &[Reading] = &[
    (N, 0xbf000000, 0x3d880000, true),
    (N, 0x3b000000, 0x3d880000, false),
    (N, 0x3fc00000, 0x3d880000, true),
    (L, 0xbf000000, 0x3d880000, true),
    (L, 0x3a800000, 0x3d880000, false),
    (L, 0xba800000, 0x3d880000, false),
    (L, 0xbb800000, 0x3d880000, true),
    (L, 0xbb000000, 0x3d880000, true),
    (G, 0xbf000000, 0x3d880000, true),
    (G, 0x00000000, 0x3d880000, false),
    (G, 0x3fc00000, 0x3d880000, true),
];

/// Metal's measured words where the addressing meets an unmapped edge column: a clamp_to_edge
/// axis counts the edge texel it clamps to, and a clamp_to_border axis counts no border texel, at
/// negative positions and past the far edge alike.
#[test]
fn the_footprint_word_answers_metal_at_an_unmapped_edge() {
    check(256, 128..256, false, EDGE);
    check(256, 128..256, true, BORDER);
}

/// 64x64x32 r32_float 3D, slices z 0..16 mapped (203caac1): the quantum along z.
const VOLUME: &[(Kind, [u32; 3], bool)] = &[
    (N, [0x3db00000, 0x3dd00000, 0x3e700000], true),
    (N, [0x3db00000, 0x3dd00000, 0x3f240000], false),
    (N, [0x3db00000, 0x3dd00000, 0x3efffff8], true),
    (L, [0x3db00000, 0x3dd00000, 0x3f000000], false),
    (L, [0x3db00000, 0x3dd00000, 0x3ef80000], true),
    (L, [0x3db00000, 0x3dd00000, 0x3ef80800], false),
    (L, [0x3db00000, 0x3dd00000, 0x3ef807f8], true),
    (L, [0x3f000000, 0x3f000000, 0x3e800000], true),
    (L, [0x3db00000, 0x3dd00000, 0x3c7f0000], true),
];

/// Metal's measured words along a 3D texture's z, beside its unmapped upper slices: the same
/// quantum on the third axis (ties up at probe `0x3ef80800`, just below it at `0x3ef807f8`).
#[test]
fn the_footprint_word_answers_metal_along_a_volumes_depth() {
    let mut seen = 0;
    for (arm, kind) in [(Arm([64, 64, 1]), L), (Arm([64, 64, 1]), N)] {
        let rows: Vec<_> = VOLUME.iter().filter(|r| r.0 == kind).collect();
        let probes: Vec<[f32; 4]> = rows
            .iter()
            .map(|r| {
                let [u, v, w] = r.1.map(f32::from_bits);
                [u, v, w, 0.0]
            })
            .collect();
        let img = image(Dim::D3, [64, 64, 32], 1, |_, _, z, _| z >= 16);
        let got = words(footprint(kind, 3, true), Dim::D3, false, img, &probes, arm);
        for (r, g) in rows.iter().zip(&got) {
            assert_eq!(*g, u32::from(!r.2), "{arm:?} {kind:?} at {:#010x?}", r.1);
        }
        seen += got.len();
    }
    assert_eq!(seen, VOLUME.len());
}

/// The rule on one axis of `n` texels: the texels counted at `u`, and whether each is inside.
fn rule(kind: Kind, u: f32, n: u32) -> Vec<i64> {
    let x = u * n as f32;
    match kind {
        Kind::Nearest => vec![x.floor() as i64],
        _ => {
            let q = (256.0 * f64::from(x) + 0.5).floor() as i64 - 128;
            let i0 = q >> 8;
            if kind == Kind::Gather || q & 255 != 0 {
                vec![i0, i0 + 1]
            } else {
                vec![i0]
            }
        }
    }
}

/// A 3D texture's and a 2D array's words follow the rule on every axis: a 16x16x16 texture whose
/// texels with `z` 0 or at least 8 are unmapped, and a 16x16 array of 3 layers whose layer 1 has
/// those `x` unmapped, sampled at positions sweeping the quantum around each edge, from outside
/// the texture's left edge to past its right, with the border on one run and the edge clamp on
/// the other.
#[test]
fn the_footprint_word_applies_the_rule_on_every_axis() {
    let sweep: Vec<f32> = [-1i32, 0, 1, 2, 127, 128, 129, 255, 256]
        .iter()
        .flat_map(|&k| {
            // Positions `X = c + k / 256` around each edge of the unmapped texels, as normalized
            // coordinates.
            [-1.5f32, -0.5, 0.5, 7.5, 15.5].map(|c| (c + k as f32 / 256.0) / 16.0)
        })
        .collect();
    let unmapped = |i: i64| i == 0 || (8..16).contains(&i);
    let hit = |texels: &[i64], border: bool| {
        texels
            .iter()
            .any(|&i| unmapped(if border { i } else { i.clamp(0, 15) }))
    };
    let mut runs = 0;
    // The tile maps' tiles are whole runs of the unmapped texels: 8x8x1 for the 3D texture's
    // slices, 1x8x1 for the array's columns.
    let (volume, array) = (Arm([8, 8, 1]), Arm([1, 8, 1]));
    for border in [true, false] {
        for kind in [L, N] {
            let probes: Vec<[f32; 4]> = sweep.iter().map(|&w| [0.5, 0.25, w, 0.0]).collect();
            let img = image(Dim::D3, [16, 16, 16], 1, |_, _, z, _| {
                unmapped(i64::from(z))
            });
            let got = words(
                footprint(kind, 3, border),
                Dim::D3,
                false,
                img,
                &probes,
                volume,
            );
            for (p, g) in probes.iter().zip(&got) {
                let want = hit(&rule(kind, p[2], 16), border);
                assert_eq!(
                    *g,
                    u32::from(want),
                    "{volume:?} 3D {kind:?} border {border} w = {}",
                    p[2]
                );
                runs += 1;
            }
        }
        for kind in [L, N, G] {
            let probes: Vec<[f32; 4]> = sweep
                .iter()
                .flat_map(|&u| [0.0f32, 1.0, 2.0, 7.0].map(|l| [u, 0.5, 0.0, l]))
                .collect();
            let img = image(Dim::D2, [16, 16, 1], 3, |x, _, _, l| {
                l == 1 && unmapped(i64::from(x))
            });
            let got = words(
                footprint(kind, 2, border),
                Dim::D2,
                true,
                img,
                &probes,
                array,
            );
            for (p, g) in probes.iter().zip(&got) {
                // A layer past the array reads its last layer.
                let layer = (p[3] as u32).min(2);
                let want = layer == 1 && hit(&rule(kind, p[0], 16), border);
                assert_eq!(
                    *g,
                    u32::from(want),
                    "{array:?} array {kind:?} border {border} u = {} layer {}",
                    p[0],
                    p[3]
                );
                runs += 1;
            }
        }
    }
    assert!(runs > 0);
}
