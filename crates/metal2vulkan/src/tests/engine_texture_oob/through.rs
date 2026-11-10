//! Metal 4.1's reads through a sampler (`read(sampler s, coord[, offset][, lod])`, MSL 4.1 §6.13,
//! `lower::texture::Through`) on the CPU interpreter's image model, against what the M3 Max
//! measured (`lower::texture`'s module docs): the level a mip filter takes, zero past the level or
//! layer, the edge or the border texel per axis, a texture buffer's first texel, through
//! constexpr samplers of each state family (the owned fixture `engine/kernel_read_through`) and
//! runtime samplers of several words, with the border texels of a four-channel format and of a
//! one-channel one.

use super::*;
use crate::metal::sampler::{border_texels, SamplerWord};
use crate::metal::TextureSwizzleChannels;

/// What a row of a read through a sampler hands back.
#[derive(Clone, Copy, Debug)]
enum Want {
    /// The texel at `(level, layer, sample, xyz)`.
    At(At),
    /// Zero in every lane: a level or layer outside the texture.
    Zero,
    /// The border texel for border color `b` (`MTLSamplerBorderColor`'s value).
    Border(u32),
    /// A plain read's out-of-bounds texel, zero with the format's alpha: a sample index outside.
    Oob,
}

/// A kernel reading through a sampler: its texture (extent, layers, levels, samples), whether it
/// is a texel buffer, its output, the runtime sampler word bound at `[[sampler(0)]]` where it
/// reads through one, and its rows.
struct Case {
    entry: &'static str,
    texel: Texel,
    shape: ([u32; 3], u32, u32, u32),
    buffer: bool,
    out: Out,
    word: Option<u32>,
    rows: Vec<([u32; 4], Want)>,
}

const SAMPLER: SlotRef = SlotRef {
    stage: Stage::Kernel,
    class: SlotClass::Sampler,
    index: 0,
};

/// A runtime sampler's word ([`SamplerWord`]) of address modes `modes` (`s`, `t`, `r`, each
/// `MTLSamplerAddressMode`'s value), border color `border`, and mip filter `mip`.
fn word(modes: [u32; 3], border: u32, mip: u32) -> u32 {
    let mut w = mip << SamplerWord::MIP_FILTER_SHIFT | border << SamplerWord::BORDER_SHIFT;
    for (m, shift) in modes.into_iter().zip(SamplerWord::ADDRESS_SHIFT) {
        w |= m << shift;
    }
    w
}

const EDGE: u32 = 0;
const MIRROR_CLAMP: u32 = 1;
const REPEAT: u32 = 2;
const ZERO: u32 = 4;
const BORDER: u32 = 5;
const TB: u32 = 0;
const OB: u32 = 1;
const OW: u32 = 2;

impl Case {
    /// Whether the read may hand back the border texel: a runtime sampler's always.
    fn bordered(&self) -> bool {
        self.word.is_some() || self.rows.iter().any(|(_, w)| matches!(w, Want::Border(_)))
    }

    /// Run on every profile, with the border texels of a four-channel and a one-channel format
    /// and each out-of-bounds alpha, and check Metal's result; returns the classes it ran on.
    fn check(&self, bc: &[u8]) -> Vec<Class> {
        let (four, one) = match self.texel {
            Texel::Float => (PixelFormat::RGBA32Float, PixelFormat::R32Float),
            Texel::Int => (PixelFormat::RGBA32Sint, PixelFormat::R32Sint),
        };
        let formats = [four, one]
            .map(|pf| border_texels(pf, TextureSwizzleChannels::DEFAULT).expect("border texels"));
        let mut ran = Vec::new();
        for (stem, p) in profiles() {
            let Some((plan, m)) = module(bc, self.entry, &stem, &p) else {
                continue;
            };
            let class = class(&p, self.buffer);
            let fact = RuntimeFact::BorderTexels { slot: TEXTURE };
            assert_eq!(
                plan.facts.contains(&fact),
                self.bordered(),
                "{stem}: {}",
                self.entry
            );
            for bits in formats {
                for alpha in [0, self.texel.one()] {
                    let img = image(self.texel, self.shape, robust(class, alpha));
                    let p_rows: Vec<[u32; 4]> = self.rows.iter().map(|(r, _)| *r).collect();
                    let mut res = resources(
                        &plan,
                        &p_rows,
                        self.rows.len() * self.out.len(),
                        self.shape.2,
                    );
                    res.spec
                        .insert(SpecKey::OutOfBoundsAlpha(TEXTURE), u64::from(alpha != 0));
                    res.spec
                        .insert(SpecKey::BorderTexels(TEXTURE), u64::from(bits));
                    if let Some(w) = self.word {
                        let fields = SamplerWord::READ_STATE;
                        let key = SpecKey::SamplerState {
                            slot: SAMPLER,
                            fields,
                        };
                        res.spec.insert(key, u64::from(w & fields));
                    }
                    let images = bound(&plan, &img);
                    let (bufs, _) = run_both(&m, &res, &images, self.rows.len() as u32, &stem);
                    let want: Vec<u8> = self
                        .rows
                        .iter()
                        .flat_map(|(_, want)| {
                            let t = match want {
                                Want::At((level, layer, sample, xyz)) => {
                                    let i = img
                                        .index(*level, *layer, *sample, *xyz)
                                        .expect("in bounds");
                                    img.texels[i]
                                }
                                Want::Zero => [0; 4],
                                Want::Border(b) => {
                                    std::array::from_fn(|c| match bits >> (3 * c as u32 + b) & 1 {
                                        1 => self.texel.one(),
                                        _ => 0,
                                    })
                                }
                                Want::Oob => [0, 0, 0, alpha],
                            };
                            self.out.bytes(t)
                        })
                        .collect();
                    for (i, (got, want)) in bufs[1]
                        .chunks_exact(self.out.len())
                        .zip(want.chunks_exact(self.out.len()))
                        .enumerate()
                    {
                        assert_eq!(
                            got, want,
                            "{stem}: {} row {i} {:?}, border texels {bits:#x}, alpha {alpha:#x}",
                            self.entry, self.rows[i].0
                        );
                    }
                }
            }
            ran.push(class);
        }
        ran.sort();
        ran.dedup();
        ran
    }
}

/// The owned fixture `engine/kernel_read_through`'s module, assembled.
fn read_through() -> Option<Vec<u8>> {
    let path = engine_dir().join("kernel_read_through.ll");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path:?}: {e}"));
    assemble_text("kernel_read_through", &text)
}

/// A 2D texture of two levels, 4 x 4 and 2 x 2.
const PLANE: ([u32; 3], u32, u32, u32) = ([4, 4, 1], 1, 2, 1);

/// Texel `xyz` of level `level` of layer 0.
fn at(level: u32, xyz: [u32; 3]) -> Want {
    Want::At((level, 0, 0, xyz))
}

/// A float 2D read of `rows` through `entry`'s constexpr sampler.
fn plane(entry: &'static str, rows: Vec<([u32; 4], Want)>) -> Case {
    Case {
        entry,
        texel: Texel::Float,
        shape: PLANE,
        buffer: false,
        out: Out::Vec4,
        word: None,
        rows,
    }
}

/// Reads through constexpr samplers of each state family: `repeat` and `mirrored_repeat` (with
/// an offset) read the edge; `clamp_to_zero` and `clamp_to_border` the border texel, per axis
/// (`rt_mixed` borders on `s` alone); without a mip filter a level within the texture reads the
/// base level, under one (nearest or linear) its own; a level past the last reads zero in every
/// lane, the border's color or not; the filter, pixel coordinates and a LOD clamp change nothing.
/// 2D, 2D array (a layer outside reads zero), 3D (`r` borders alone), 1D, `read_write`, texture
/// buffer (an index outside reads the first texel under an edge mode), `int` and `half` texels,
/// and an alpha read alone.
#[test]
fn reads_through_constexpr_samplers_address_by_their_state() {
    let Some(bc) = read_through() else {
        return;
    };
    let cases = [
        plane(
            "rt_repeat",
            vec![
                ([1, 2, 0, 0], at(0, [1, 2, 0])),
                ([int(-1), 5, 0, 0], at(0, [0, 3, 0])),
                ([5, 1, 0, 1], at(0, [3, 1, 0])),
                ([0, 0, 0, 2], Want::Zero),
                ([1, 1, 0, MAX], Want::Zero),
            ],
        ),
        plane(
            "rt_mirrored_offset",
            vec![
                ([1, 1, 0, 0], at(0, [0, 2, 0])),
                ([3, 3, 0, 0], at(0, [1, 3, 0])),
                ([9, int(-9), 0, 1], at(0, [3, 0, 0])),
            ],
        ),
        plane(
            "rt_zero",
            vec![
                ([1, 2, 0, 0], at(0, [1, 2, 0])),
                ([int(-1), 2, 0, 0], Want::Border(TB)),
                ([1, 4, 0, 1], Want::Border(TB)),
                ([3, 3, 0, 1], at(0, [3, 3, 0])),
                ([2, 2, 0, 2], Want::Zero),
                ([9, 2, 0, 2], Want::Zero),
            ],
        ),
        plane(
            "rt_border_ob",
            vec![
                ([4, 0, 0, 0], Want::Border(OB)),
                ([0, 0, 0, 0], at(0, [0, 0, 0])),
                ([int(i32::MIN), 0, 0, 0], Want::Border(OB)),
            ],
        ),
        plane(
            "rt_border_ow",
            vec![
                ([0, int(-1), 0, 0], Want::Border(OW)),
                ([3, 3, 0, 1], at(0, [3, 3, 0])),
                ([int(i32::MAX), 0, 0, 0], Want::Border(OW)),
                ([4, 4, 0, MAX], Want::Zero),
            ],
        ),
        plane(
            "rt_mixed",
            vec![
                ([1, 9, 0, 0], at(0, [1, 3, 0])),
                ([int(-1), 1, 0, 0], Want::Border(OW)),
                ([5, int(-5), 0, 0], Want::Border(OW)),
                ([3, int(-5), 0, 0], at(0, [3, 0, 0])),
            ],
        ),
        plane(
            "rt_mip",
            vec![
                ([3, 3, 0, 1], at(1, [1, 1, 0])),
                ([1, 0, 0, 1], at(1, [1, 0, 0])),
                ([int(-1), 0, 0, 0], at(0, [0, 0, 0])),
                ([0, 0, 0, 2], Want::Zero),
            ],
        ),
        plane(
            "rt_mip_zero",
            vec![
                ([1, 1, 0, 1], at(1, [1, 1, 0])),
                ([2, 0, 0, 1], Want::Border(TB)),
                ([3, 3, 0, 0], at(0, [3, 3, 0])),
                ([0, 0, 0, 2], Want::Zero),
            ],
        ),
        plane(
            "rt_filtered",
            vec![
                ([9, 9, 0, 0], at(0, [3, 3, 0])),
                ([1, 1, 0, 1], at(0, [1, 1, 0])),
                ([1, 1, 0, 0], at(0, [1, 1, 0])),
            ],
        ),
        Case {
            entry: "rt_array",
            texel: Texel::Float,
            shape: ([4, 4, 1], 3, 2, 1),
            buffer: false,
            out: Out::Vec4,
            word: None,
            rows: vec![
                ([1, 1, 2, 0], Want::At((0, 2, 0, [1, 1, 0]))),
                ([1, 1, 3, 0], Want::Zero),
                ([int(-1), 1, 1, 0], Want::Border(TB)),
                ([1, 1, 1, 1], Want::At((1, 1, 0, [1, 1, 0]))),
                ([1, 1, MAX, 0], Want::Zero),
                ([5, 1, 3, 0], Want::Zero),
            ],
        },
        Case {
            entry: "rt_volume",
            texel: Texel::Float,
            shape: ([4, 4, 4], 1, 2, 1),
            buffer: false,
            out: Out::Vec4,
            word: None,
            rows: vec![
                ([int(-1), 9, 2, 0], at(0, [0, 3, 2])),
                ([1, 1, 4, 0], Want::Border(OW)),
                ([1, 1, int(-1), 0], Want::Border(OW)),
                ([1, 1, 1, 1], at(0, [1, 1, 1])),
                ([0, 0, 0, 2], Want::Zero),
            ],
        },
        Case {
            entry: "rt_line",
            texel: Texel::Float,
            shape: ([8, 1, 1], 1, 1, 1),
            buffer: false,
            out: Out::Vec4,
            word: None,
            rows: vec![
                ([3, 0, 0, 0], at(0, [3, 0, 0])),
                ([8, 0, 0, 0], Want::Border(TB)),
                ([int(-1), 0, 0, 0], Want::Border(TB)),
            ],
        },
        Case {
            entry: "rt_rw",
            texel: Texel::Float,
            shape: ([4, 4, 1], 1, 1, 1),
            buffer: false,
            out: Out::Vec4,
            word: None,
            rows: vec![
                ([1, 1, 0, 0], at(0, [1, 1, 0])),
                ([4, 0, 0, 0], Want::Border(OW)),
                ([0, int(-2), 0, 0], Want::Border(OW)),
            ],
        },
        Case {
            entry: "rt_buffer",
            texel: Texel::Float,
            shape: ([8, 1, 1], 1, 1, 1),
            buffer: true,
            out: Out::Vec4,
            word: None,
            rows: vec![
                ([3, 0, 0, 0], at(0, [3, 0, 0])),
                ([8, 0, 0, 0], at(0, [0, 0, 0])),
                ([int(-1), 0, 0, 0], at(0, [0, 0, 0])),
                ([MAX, 0, 0, 0], at(0, [0, 0, 0])),
            ],
        },
        Case {
            entry: "rt_buffer_ow",
            texel: Texel::Float,
            shape: ([8, 1, 1], 1, 1, 1),
            buffer: true,
            out: Out::Vec4,
            word: None,
            rows: vec![
                ([2, 0, 0, 0], at(0, [2, 0, 0])),
                ([8, 0, 0, 0], Want::Border(OW)),
                ([int(-1), 0, 0, 0], Want::Border(OW)),
            ],
        },
        Case {
            entry: "rt_int",
            texel: Texel::Int,
            shape: PLANE,
            buffer: false,
            out: Out::Vec4,
            word: None,
            rows: vec![
                ([int(-1), 0, 0, 0], Want::Border(OW)),
                ([1, 1, 0, 0], at(0, [1, 1, 0])),
                ([1, 1, 0, 2], Want::Zero),
            ],
        },
        Case {
            entry: "rt_half",
            texel: Texel::Float,
            shape: PLANE,
            buffer: false,
            out: Out::Half4,
            word: None,
            rows: vec![
                ([int(-1), 0, 0, 0], Want::Border(OW)),
                ([1, 1, 0, 0], at(0, [1, 1, 0])),
                ([1, 1, 0, 2], Want::Zero),
            ],
        },
        Case {
            entry: "rt_alpha",
            texel: Texel::Float,
            shape: PLANE,
            buffer: false,
            out: Out::Alpha,
            word: None,
            rows: vec![
                ([int(-1), 0, 0, 0], Want::Border(TB)),
                ([1, 1, 0, 0], at(0, [1, 1, 0])),
                ([1, 1, 0, 2], Want::Zero),
            ],
        },
    ];
    for case in &cases {
        let ran = case.check(&bc);
        assert!(ran.contains(&Class::Shader), "{}: {ran:?}", case.entry);
    }
}

/// Reads through a runtime `[[sampler(n)]]` address by its word (`SamplerWord::READ_STATE`'s
/// fields, the slot's `SamplerState` specialization), one module for every word: its address
/// modes per axis (`mirror_clamp_to_edge` reads the edge), its border color under
/// `clamp_to_zero` too, and its mip filter. A texture buffer's index outside clamps under its
/// `clamp_to_edge` and reads the first texel under the other edge modes.
#[test]
fn reads_through_runtime_samplers_address_by_their_word() {
    let Some(bc) = read_through() else {
        return;
    };
    let arg = |w: u32, rows| Case {
        entry: "rt_arg",
        texel: Texel::Float,
        shape: PLANE,
        buffer: false,
        out: Out::Vec4,
        word: Some(w),
        rows,
    };
    // `rt_arg` reads at the offset (1, -1).
    let cases = [
        arg(
            word([EDGE; 3], TB, 0),
            vec![
                ([3, 1, 0, 0], at(0, [3, 0, 0])),
                ([0, 0, 0, 1], at(0, [1, 0, 0])),
                ([0, 0, 0, 2], Want::Zero),
            ],
        ),
        arg(
            word([ZERO; 3], TB, 0),
            vec![
                ([3, 1, 0, 0], Want::Border(TB)),
                ([1, 2, 0, 0], at(0, [2, 1, 0])),
            ],
        ),
        arg(
            word([BORDER; 3], OW, 0),
            vec![([3, 1, 0, 0], Want::Border(OW)), ([0, 0, 0, 2], Want::Zero)],
        ),
        arg(
            word([ZERO, REPEAT, EDGE], OB, 0),
            vec![
                ([3, 3, 0, 0], Want::Border(OB)),
                ([1, 0, 0, 0], at(0, [2, 0, 0])),
            ],
        ),
        arg(
            word([EDGE; 3], TB, 1),
            vec![
                ([1, 1, 0, 1], at(1, [1, 0, 0])),
                ([0, 1, 0, 0], at(0, [1, 0, 0])),
            ],
        ),
        arg(
            word([MIRROR_CLAMP; 3], OW, 0),
            vec![([3, 1, 0, 0], at(0, [3, 0, 0]))],
        ),
    ];
    for case in &cases {
        let ran = case.check(&bc);
        assert!(
            ran.contains(&Class::Shader),
            "{:#x}: {ran:?}",
            case.word.unwrap_or(0)
        );
    }
    let buffer = |w: u32, rows| Case {
        entry: "rt_arg_buffer",
        texel: Texel::Float,
        shape: ([8, 1, 1], 1, 1, 1),
        buffer: true,
        out: Out::Vec4,
        word: Some(w),
        rows,
    };
    let buffers = [
        buffer(
            word([EDGE; 3], TB, 0),
            vec![
                ([8, 0, 0, 0], at(0, [7, 0, 0])),
                ([int(-1), 0, 0, 0], at(0, [0, 0, 0])),
                ([4, 0, 0, 0], at(0, [4, 0, 0])),
            ],
        ),
        buffer(
            word([REPEAT; 3], TB, 0),
            vec![
                ([9, 0, 0, 0], at(0, [0, 0, 0])),
                ([5, 0, 0, 0], at(0, [5, 0, 0])),
            ],
        ),
        buffer(
            word([MIRROR_CLAMP; 3], TB, 0),
            vec![([8, 0, 0, 0], at(0, [0, 0, 0]))],
        ),
        buffer(
            word([ZERO; 3], TB, 0),
            vec![([8, 0, 0, 0], Want::Border(TB))],
        ),
        // A texture buffer's border follows `t`'s mode, and `s`'s `clamp_to_edge` clamps first:
        // the M3's per-axis answers (the conformance case `read-through-buffer`).
        buffer(
            word([BORDER, EDGE, EDGE], OW, 0),
            vec![
                ([8, 0, 0, 0], at(0, [0, 0, 0])),
                ([int(-1), 0, 0, 0], at(0, [0, 0, 0])),
            ],
        ),
        buffer(
            word([BORDER, BORDER, EDGE], OW, 0),
            vec![([9, 0, 0, 0], Want::Border(OW))],
        ),
        buffer(
            word([REPEAT, ZERO, REPEAT], OW, 0),
            vec![([int(-1), 0, 0, 0], Want::Border(OW))],
        ),
        buffer(
            word([EDGE, BORDER, BORDER], OW, 0),
            vec![
                ([8, 0, 0, 0], at(0, [7, 0, 0])),
                ([int(-1), 0, 0, 0], at(0, [0, 0, 0])),
            ],
        ),
        buffer(
            word([BORDER, REPEAT, BORDER], OW, 0),
            vec![([100, 0, 0, 0], at(0, [0, 0, 0]))],
        ),
    ];
    for case in &buffers {
        let ran = case.check(&bc);
        assert!(
            ran.contains(&Class::Shader),
            "{:#x}: {ran:?}",
            case.word.unwrap_or(0)
        );
    }
    let volume = Case {
        entry: "rt_arg_volume",
        texel: Texel::Float,
        shape: ([4, 4, 4], 1, 2, 1),
        buffer: false,
        out: Out::Vec4,
        word: Some(word([EDGE, EDGE, ZERO], TB, 0)),
        rows: vec![
            ([1, 1, 4, 0], Want::Border(TB)),
            ([9, 1, 1, 0], at(0, [3, 1, 1])),
        ],
    };
    assert!(volume.check(&bc).contains(&Class::Shader));
}

/// The earlier clamp-to-edge reads ([`EDGE_READS`]) under the measured rules: a level past the
/// texture's, or a layer past its layers, reads zero in every lane, not the out-of-bounds texel;
/// a sample index outside is out of bounds as a plain read's; a texture buffer's index outside
/// reads its first texel through the constexpr sampler.
#[test]
fn a_clamp_to_edge_read_clamps_its_coordinate_into_the_texture() {
    let Some(bc) = assemble_text("edge_reads", EDGE_READS) else {
        return;
    };
    let case = |entry, texel, shape, rows| Case {
        entry,
        texel,
        shape,
        buffer: false,
        out: Out::Vec4,
        word: None,
        rows,
    };
    let cases = [
        case(
            "edge_2d",
            Texel::Float,
            ([4, 4, 1], 1, 2, 1),
            vec![
                ([1, 2, 0, 0], at(0, [1, 2, 0])),
                ([int(-1), 2, 0, 0], at(0, [0, 2, 0])),
                ([5, int(-3), 0, 0], at(0, [3, 0, 0])),
                ([MAX, 0, 0, 0], at(0, [0, 0, 0])),
                ([int(i32::MIN), 7, 0, 0], at(0, [0, 3, 0])),
                ([int(i32::MAX), 1, 0, 0], at(0, [3, 1, 0])),
                ([5, 5, 1, 0], at(0, [3, 3, 0])),
                ([int(-5), 9, 1, 0], at(0, [0, 3, 0])),
                ([0, 0, 2, 0], Want::Zero),
                ([1, 1, MAX, 0], Want::Zero),
            ],
        ),
        case(
            "edge_2d_offset",
            Texel::Int,
            ([4, 4, 1], 1, 1, 1),
            vec![
                ([2, 0, 0, 0], at(0, [0, 3, 0])),
                ([5, 0, 0, 0], at(0, [3, 3, 0])),
                ([1, int(-3), 0, 0], at(0, [0, 0, 0])),
                ([3, int(-1), 0, 0], at(0, [1, 2, 0])),
                ([MAX, 2, 0, 0], at(0, [0, 3, 0])),
            ],
        ),
        case(
            "edge_dynamic_offset",
            Texel::Float,
            ([4, 4, 1], 1, 1, 1),
            vec![
                ([1, 1, int(-2), 1], at(0, [0, 2, 0])),
                ([3, 0, 7, int(-8)], at(0, [3, 0, 0])),
                ([0, 3, 0, 0], at(0, [0, 3, 0])),
            ],
        ),
        case(
            "edge_2d_array",
            Texel::Int,
            ([4, 4, 1], 3, 2, 1),
            vec![
                ([int(-1), 5, 1, 0], Want::At((0, 1, 0, [0, 3, 0]))),
                ([2, 2, 3, 0], Want::Zero),
                ([9, int(-9), 2, 1], Want::At((0, 2, 0, [3, 0, 0]))),
                ([0, 0, 0, 2], Want::Zero),
                ([1, 1, MAX, 0], Want::Zero),
            ],
        ),
        case(
            "edge_3d",
            Texel::Float,
            ([4, 4, 4], 1, 2, 1),
            vec![
                ([int(-1), 4, 2, 0], at(0, [0, 3, 2])),
                ([7, 7, 7, 1], at(0, [3, 3, 3])),
                ([0, 0, 0, 2], Want::Zero),
            ],
        ),
        case(
            "edge_ms",
            Texel::Float,
            ([4, 4, 1], 1, 1, 4),
            vec![
                ([int(-2), 1, 3, 0], Want::At((0, 0, 3, [0, 1, 0]))),
                ([9, 9, 0, 0], at(0, [3, 3, 0])),
                ([1, 1, 4, 0], Want::Oob),
            ],
        ),
        case(
            "edge_rw",
            Texel::Int,
            ([4, 4, 1], 1, 1, 1),
            vec![
                ([int(-1), int(-1), 0, 0], at(0, [0, 0, 0])),
                ([4, 4, 0, 0], at(0, [3, 3, 0])),
                ([2, 1, 0, 0], at(0, [2, 1, 0])),
            ],
        ),
        case(
            "edge_1d",
            Texel::Float,
            ([8, 1, 1], 1, 1, 1),
            vec![
                ([int(-3), 0, 0, 0], at(0, [0, 0, 0])),
                ([100, 0, 0, 0], at(0, [7, 0, 0])),
                ([5, 0, 0, 0], at(0, [5, 0, 0])),
            ],
        ),
    ];
    for case in &cases {
        assert_eq!(case.check(&bc), ALL, "{}", case.entry);
    }
    // Every kernel links and passes `spirv-val` where it translates.
    let mut validated = 0;
    for (stem, p) in profiles() {
        for entry in cases.iter().map(|c| c.entry).chain(["edge_buffer"]) {
            match compile(&bc, entry, &p) {
                Ok(a) => {
                    link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{stem}: {entry}: {e}"));
                    validated += 1;
                }
                Err(Error::NeedsInput(_)) => {}
                Err(e) => panic!("{stem}: {entry}: {e}"),
            }
        }
    }
    assert!(validated > 0);
    let buffer = Case {
        entry: "edge_buffer",
        texel: Texel::Float,
        shape: ([8, 1, 1], 1, 1, 1),
        buffer: true,
        out: Out::Vec4,
        word: None,
        rows: vec![
            ([3, 0, 0, 0], at(0, [3, 0, 0])),
            ([int(-1), 0, 0, 0], at(0, [0, 0, 0])),
            ([8, 0, 0, 0], at(0, [0, 0, 0])),
            ([int(i32::MIN), 0, 0, 0], at(0, [0, 0, 0])),
        ],
    };
    assert_eq!(buffer.check(&bc), [Class::Device, Class::Shader]);
}

/// Every read through a sampler translates, links and passes `spirv-val` where it translates:
/// the constexpr, runtime and argument-buffer samplers' (`rt_table`), and an argument buffer's
/// texture through `clamp_to_zero` (`rt_heap_zero`), `clamp_to_border` with an opaque white border
/// (`rt_heap_ow`) and a runtime sampler (`rt_heap_arg`), whose border texels its handle-table entry
/// holds.
#[test]
fn every_read_through_a_sampler_validates_or_is_refused_typed() {
    let Some(bc) = read_through() else {
        return;
    };
    let entries = [
        "rt_repeat",
        "rt_mirrored_offset",
        "rt_zero",
        "rt_border_ob",
        "rt_border_ow",
        "rt_mixed",
        "rt_mip",
        "rt_mip_zero",
        "rt_filtered",
        "rt_array",
        "rt_volume",
        "rt_line",
        "rt_rw",
        "rt_buffer",
        "rt_buffer_ow",
        "rt_int",
        "rt_half",
        "rt_alpha",
        "rt_arg",
        "rt_arg_buffer",
        "rt_arg_volume",
        "rt_table",
        "rt_heap_zero",
        "rt_heap_ow",
        "rt_heap_arg",
    ];
    let mut validated = 0;
    let mut heap = 0;
    let not_implemented = Unsupported::not_implemented(Feature::ArgumentBufferTextures);
    for (stem, p) in profiles() {
        for entry in entries {
            match compile(&bc, entry, &p) {
                Ok(a) => {
                    link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{stem}: {entry}: {e}"));
                    validated += 1;
                    if entry.starts_with("rt_heap") {
                        heap += 1;
                    }
                }
                Err(Error::NeedsInput(_)) => {}
                // A device without the descriptor-indexing features a handle heap needs refuses
                // an argument buffer's texture whatever reads it.
                Err(Error::Unsupported(u))
                    if entry.starts_with("rt_heap")
                        && u.feature == Feature::ArgumentBufferTextures
                        && u.needs != not_implemented.needs => {}
                Err(e) => panic!("{stem}: {entry}: {e}"),
            }
        }
    }
    assert!(validated > 0);
    assert!(heap > 0, "no argument buffer's texture read translated");
}

/// The handle table's entry `(id, element, flags)` list of `bytes`.
fn table_entries(bytes: &[u8]) -> Vec<(u64, u32, u32)> {
    use crate::memory::handles::TABLE_ENTRY_BYTES;
    bytes
        .chunks_exact(TABLE_ENTRY_BYTES as usize)
        .map(|e| {
            let w = |i: usize| u32::from_le_bytes(e[4 * i..4 * i + 4].try_into().expect("word"));
            (u64::from(w(0)) | u64::from(w(1)) << 32, w(2), w(3))
        })
        .collect()
}

/// An argument buffer's texture read through a sampler hands back Metal's border texels of the
/// resident texture, its pixel format's and swizzle's, which the binder writes in the texture's
/// handle-table entry (`memory::handles::FLAG_BORDER_TEXELS`) and the read loads from the entry its
/// handle names (the search's indexed form, `memory::handles::Lookup::Indexed`). End to end on the
/// interpreter: the binder's table bytes for two resident textures of two levels, an `RGBA32Float`
/// made with the swizzle `(G, R, 1, 0)` and an `R32Float`, the emitted search, and reads in and
/// past each through `clamp_to_zero` (`rt_heap_zero`), an opaque white `clamp_to_border`
/// (`rt_heap_ow`), and a runtime sampler (`rt_heap_arg`) of each border color, of `clamp_to_edge`,
/// and of a border with a mip filter; a missed id reads zero in every lane. Before the entry held
/// them, the swizzled texture read `(0, 0, 0, 0)` past its edge through `clamp_to_zero`, where the
/// M3 reads `(0, 0, 1, 0)`, and the other two reads were refused.
#[test]
fn an_argument_buffer_texture_reads_its_border_texels_from_its_table_entry() {
    use crate::memory::handles::Lookup;
    use crate::metal::TextureSwizzle as S;
    let Some(bc) = read_through() else {
        return;
    };
    const SWIZZLED: u64 = 0x0000_0001_0000_0010;
    const RED: u64 = 0x0000_0001_0000_0020;
    const MISSED: u64 = 0x0000_0002_0000_0000;
    let swizzle = TextureSwizzleChannels {
        red: S::Green,
        green: S::Red,
        blue: S::One,
        alpha: S::Zero,
    };
    let textures = [
        (SWIZZLED, PixelFormat::RGBA32Float, swizzle),
        (RED, PixelFormat::R32Float, TextureSwizzleChannels::DEFAULT),
    ];
    let unrobust = [
        VkFeature::RobustImageAccess,
        VkFeature::RobustImageAccess2,
        VkFeature::RobustBufferAccess,
        VkFeature::RobustBufferAccess2,
    ];
    let devices = [
        ("moltenvk-m3", device("moltenvk-m3", &[])),
        (
            "moltenvk-m3 robust1",
            device("moltenvk-m3", &[VkFeature::RobustImageAccess2]),
        ),
        ("moltenvk-m3 unrobust", device("moltenvk-m3", &unrobust)),
        ("lavapipe", device("lavapipe", &[])),
        ("amd-780m-radv", device("amd-780m-radv", &[])),
    ];
    // Inside, past each edge, at level 1 (the base level's texel without a mip filter), past the
    // last level.
    let base = |border: Want| {
        vec![
            ([1, 2, 0, 0], at(0, [1, 2, 0])),
            ([4, 0, 0, 0], border),
            ([int(-1), 3, 0, 0], border),
            ([3, 3, 0, 1], at(0, [3, 3, 0])),
            ([0, 0, 0, 2], Want::Zero),
        ]
    };
    let edge = vec![
        ([1, 2, 0, 0], at(0, [1, 2, 0])),
        ([4, 0, 0, 0], at(0, [3, 0, 0])),
        ([int(-1), 3, 0, 0], at(0, [0, 3, 0])),
        ([0, 0, 0, 2], Want::Zero),
    ];
    // Under a mip filter the read reads its own level, 2 x 2.
    let mip = |border: Want| {
        vec![
            ([1, 0, 0, 1], at(1, [1, 0, 0])),
            ([3, 3, 0, 1], border),
            ([0, 0, 0, 2], Want::Zero),
        ]
    };
    let mut reads: Vec<(&str, Option<u32>, Vec<([u32; 4], Want)>)> = vec![
        ("rt_heap_zero", None, base(Want::Border(TB))),
        ("rt_heap_ow", None, base(Want::Border(OW))),
        ("rt_heap_arg", Some(word([EDGE; 3], OW, 0)), edge),
        (
            "rt_heap_arg",
            Some(word([ZERO; 3], OB, 0)),
            base(Want::Border(OB)),
        ),
        (
            "rt_heap_arg",
            Some(word([BORDER, ZERO, BORDER], OW, 1)),
            mip(Want::Border(OW)),
        ),
    ];
    for b in [TB, OB, OW] {
        reads.push((
            "rt_heap_arg",
            Some(word([BORDER; 3], b, 0)),
            base(Want::Border(b)),
        ));
    }
    let mut ran = Vec::new();
    for (stem, dev) in devices {
        for (entry, sampler, rows) in &reads {
            let lib = Library::from_air(&bc).expect("library");
            let f = lib.function(entry).expect("entry");
            let desc =
                ComputePipelineDescriptor::new(StageSource::Function(FunctionDescriptor::new(f)));
            let pipe = match dev.compile_compute_pipeline(&desc, &Budget::standard()) {
                Ok(p) => p,
                Err(Error::Unsupported(u)) if u.feature == Feature::ArgumentBufferTextures => {
                    continue
                }
                Err(e) => panic!("{stem}: {entry}: {e}"),
            };
            let used = |token, (id, pf, swizzle): (u64, PixelFormat, TextureSwizzleChannels)| {
                let mut facts = TextureFacts::new(TextureType::Type2D, pf);
                facts.mipmap_level_count = 2;
                facts.swizzle = swizzle;
                let mut u = UsedResource::new(ResourceClass::Texture(facts), token);
                u.resource_id = Some(id);
                u
            };
            let state = BindState {
                used_resources: vec![used(1, textures[0]), used(2, textures[1])],
                ..Default::default()
            };
            let bplan = pipe.plan(&dev, &state.shape()).expect("plan");
            let dispatch = Dispatch::Threadgroups {
                groups: [rows.len() as u32, 1, 1],
                threads_per_group: [1, 1, 1],
            };
            let cmds = bplan
                .instantiate_dispatch(&state, &dispatch)
                .expect("dispatch");
            let (set, binding) = bplan
                .descriptors
                .iter()
                .find_map(|d| {
                    matches!(d.source, WriteSource::HandleTable { .. })
                        .then_some((d.set, d.binding))
                })
                .expect("a handle table");
            let table = cmds
                .descriptor_bytes
                .iter()
                .find(|b| (b.set, b.binding) == (set, binding))
                .expect("table bytes")
                .bytes
                .clone();
            let entries = table_entries(&table);
            let element = |id: u64| {
                entries
                    .iter()
                    .find(|e| e.0 == id)
                    .map(|e| e.1)
                    .expect("a listed id")
            };

            let (plan, m) = typed_module(&bc, entry, dev.profile()).expect("module");
            assert!(
                matches!(plan.families.access.lookup(), Lookup::Indexed { .. }),
                "{stem}: {entry}: the search names each handle's entry"
            );
            let class = class(dev.profile(), false);
            let heap = plan
                .families
                .texture
                .heap
                .values()
                .find_map(|h| h.class)
                .expect("a heap class");
            let slot = ResourceSlot {
                stage: Stage::Kernel,
                kind: ResourceKind::Heap(HandleClass::Image(heap)),
                index: 0,
            };
            // Element 0 is the class's neutral; each texture's element views it (the swizzled
            // one's texels as its view reads them), 4 x 4 and 2 x 2, each texel naming itself.
            let image = |e: u32, pf: PixelFormat| {
                let mut img = Image::new(slot, [4, 4, 1], 1, 2, 1).at_element(e);
                for (i, t) in img.texels.iter_mut().enumerate() {
                    let i = i as f32;
                    *t = match pf {
                        PixelFormat::R32Float => [fl(1.0 + i), 0, 0, fl(1.0)],
                        _ => [fl(1.0 + i), fl(100.0 + i), fl(200.0 + i), fl(300.0 + i)],
                    };
                }
                let alpha = match pf {
                    PixelFormat::R32Float => fl(1.0),
                    _ => 0,
                };
                img.robust = robust(class, alpha);
                img
            };
            let mut neutral = Image::new(slot, [1, 1, 1], 1, 1, 1).at_element(0);
            neutral.robust = robust(class, 0);
            let mut images = vec![neutral];
            for (id, pf, _) in textures {
                images.push(image(element(id), pf));
            }
            images.sort_by_key(|i| i.element);
            let ids = textures
                .iter()
                .map(|&(id, pf, swizzle)| (id, Some((pf, swizzle))))
                .chain([(MISSED, None)]);
            for (id, texture) in ids {
                let bits = texture.map_or(0, |(pf, swizzle)| {
                    border_texels(pf, swizzle).expect("border texels")
                });
                let img = match texture {
                    Some(_) => images.iter().find(|i| i.element == element(id)),
                    None => None,
                };
                let mut spec = BTreeMap::new();
                for axis in [Axis::X, Axis::Y, Axis::Z] {
                    spec.insert(SpecKey::LocalSize(axis), 1);
                }
                spec.insert(SpecKey::FlatLocalSize, 1);
                if let Some(w) = sampler {
                    let fields = SamplerWord::READ_STATE;
                    let key = SpecKey::SamplerState {
                        slot: SAMPLER,
                        fields,
                    };
                    spec.insert(key, u64::from(w & fields));
                }
                let buffer = |kind, index, bytes| Buffer {
                    slot: Some(ResourceSlot {
                        stage: Stage::Kernel,
                        kind,
                        index,
                    }),
                    address: None,
                    bytes,
                };
                let p_rows: Vec<u8> = rows
                    .iter()
                    .flat_map(|(r, _)| r.iter().flat_map(|w| w.to_le_bytes()))
                    .collect();
                let res = Resources {
                    buffers: vec![
                        buffer(ResourceKind::Buffer, 0, p_rows),
                        buffer(ResourceKind::Buffer, 1, vec![0xAB; rows.len() * 16]),
                        buffer(ResourceKind::Buffer, 2, id.to_le_bytes().to_vec()),
                        buffer(
                            ResourceKind::Synthetic(SyntheticBinding::HandleTable),
                            0,
                            table.clone(),
                        ),
                    ],
                    push: vec![0u8; plan.push.push_range.map_or(0, |(_, end)| end as usize)],
                    spec,
                };
                let (bufs, _) = run_both(&m, &res, &images, rows.len() as u32, stem);
                let want: Vec<u8> = rows
                    .iter()
                    .flat_map(|(_, want)| {
                        let t = match (want, img) {
                            (_, None) | (Want::Zero, _) => [0; 4],
                            (Want::At((level, layer, sample, xyz)), Some(img)) => {
                                let i =
                                    img.index(*level, *layer, *sample, *xyz).expect("in bounds");
                                img.texels[i]
                            }
                            (Want::Border(b), Some(_)) => {
                                std::array::from_fn(|c| match bits >> (3 * c as u32 + b) & 1 {
                                    1 => fl(1.0),
                                    _ => 0,
                                })
                            }
                            (Want::Oob, Some(_)) => panic!("no sample index here"),
                        };
                        Out::Vec4.bytes(t)
                    })
                    .collect();
                for (i, (got, want)) in bufs[1]
                    .chunks_exact(16)
                    .zip(want.chunks_exact(16))
                    .enumerate()
                {
                    assert_eq!(
                        got, want,
                        "{stem}: {entry} {sampler:?} id {id:#x} row {i} {:?}",
                        rows[i].0
                    );
                }
            }
            ran.push(class);
        }
    }
    ran.sort();
    ran.dedup();
    assert_eq!(ran, ALL);
}

/// Metal's border texels per format (`metal::sampler::border_texels`), as the M3 Max read them
/// through `clamp_to_border` and `clamp_to_zero`: the color in the format's channels (a missing
/// red, green or blue 0, a missing alpha 1), `A8Unorm`'s alpha the color's red, a depth format
/// `(b, b, b, 1)`, then the swizzle (`(green, red, one, zero)` of `RGBA32Float` reads opaque white
/// as `(1, 1, 1, 0)` and transparent black as `(0, 0, 1, 0)`).
#[test]
fn border_texels_follow_the_format_and_swizzle() {
    use crate::metal::TextureSwizzle as S;
    // Each color's texel, `[r, g, b, a]` per color (transparent black, opaque black, opaque white).
    let texels = |pf, swizzle| -> [[u32; 4]; 3] {
        let bits = border_texels(pf, swizzle).expect("border texels");
        std::array::from_fn(|b| std::array::from_fn(|c| bits >> (3 * c + b) & 1))
    };
    let plain = TextureSwizzleChannels::DEFAULT;
    let cases = [
        (
            PixelFormat::RGBA32Float,
            [[0, 0, 0, 0], [0, 0, 0, 1], [1, 1, 1, 1]],
        ),
        (
            PixelFormat::R32Float,
            [[0, 0, 0, 1], [0, 0, 0, 1], [1, 0, 0, 1]],
        ),
        (
            PixelFormat::RG32Float,
            [[0, 0, 0, 1], [0, 0, 0, 1], [1, 1, 0, 1]],
        ),
        (
            PixelFormat::R32Uint,
            [[0, 0, 0, 1], [0, 0, 0, 1], [1, 0, 0, 1]],
        ),
        (
            PixelFormat::R8Unorm,
            [[0, 0, 0, 1], [0, 0, 0, 1], [1, 0, 0, 1]],
        ),
        (
            PixelFormat::A8Unorm,
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]],
        ),
        (
            PixelFormat::BGRA8Unorm,
            [[0, 0, 0, 0], [0, 0, 0, 1], [1, 1, 1, 1]],
        ),
        (
            PixelFormat::Depth32Float,
            [[0, 0, 0, 1], [0, 0, 0, 1], [1, 1, 1, 1]],
        ),
    ];
    for (pf, want) in cases {
        assert_eq!(texels(pf, plain), want, "{pf:?}");
    }
    let swizzled = |r, g, b, a| TextureSwizzleChannels {
        red: r,
        green: g,
        blue: b,
        alpha: a,
    };
    assert_eq!(
        texels(
            PixelFormat::RGBA32Float,
            swizzled(S::Green, S::Red, S::One, S::Zero)
        ),
        [[0, 0, 1, 0], [0, 0, 1, 0], [1, 1, 1, 0]]
    );
    assert_eq!(
        texels(
            PixelFormat::R32Float,
            swizzled(S::Red, S::One, S::Red, S::Zero)
        ),
        [[0, 1, 0, 0], [0, 1, 0, 0], [1, 1, 1, 0]]
    );
    assert_eq!(
        border_texels(PixelFormat::RGBA32Float, plain),
        Some(crate::metal::sampler::NEUTRAL_BORDER_TEXELS)
    );
}
