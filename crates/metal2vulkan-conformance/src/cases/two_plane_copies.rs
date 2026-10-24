//! Copies of Apple's two-plane extended-range textures, ported from
//! `runtime_lane::two_plane_copies`.
//!
//! Two cases, both from the halves of that lane whose Metal values were measured on **this** chip
//! under **this** release:
//!
//! - `two-plane-row-blits`, from the probe `bp.m` (an Apple M3 Max under macOS 27.0.1): a blit
//!   between a buffer and a one-level two-plane texture at a width that is not a multiple of 4, of
//!   the texture's own memory or a placement heap's. Metal's answer is **every word**, as the probe
//!   printed it, so the case records a typed list of words and a difference names the word that
//!   moved. The probe's `t<i>` and `t<s>.<i>` are the texel words `replace_region` set, which is
//!   why the case can hold Metal's answer as numbers rather than as a line of text.
//! - `two-plane-pad-fills`, from the probe `pad.m` (the same M3 Max, alike in two runs): a private
//!   two-plane texture at a width off 4, every color word set to `0xee` by a blit, one fill, and
//!   every slice read back by a blit, rows `4w` bytes apart. Metal's answer is the digest the probe
//!   kept of its printed lines, so the site is **digest-only** and says so: the bytes between the
//!   rows are inside the digest, and nobody wrote them out.
//!
//! **`two-plane-announced-row-blits` records on Metal; the other two carry their answer, and only
//! because of the fill.** The stated reason used to be that the two-plane formats are Apple's
//! private ones. That is wrong: the *format* is not private — `MTLTextureDescriptor`'s pixel format
//! is a raw value, and Metal makes, writes, blits and reads back 550 and 551 through entirely
//! public `objc2-metal` calls ([`crate::metal_work`]). What is private is `fillTexture:`. So the
//! announced blits, which fill nothing, run on Metal; `two-plane-row-blits` still carries its
//! answer because two of its twenty-two blits fill first, and `two-plane-pad-fills` because every
//! case of it is a fill.
//!
//! **Not ported, and why.** The lane's other three families stay the lane's:
//!
//! - its `METAL` table of 66 digests (`get`, `t2b`, `rep`, `b2t`, `t2t`) was measured on **an Apple
//!   M4** under macOS 27.0.1. A recording is keyed by chip, and this round records this M3 Max
//!   alone, so carrying an M4's digests into an `apple-m3-max` answer would key a measurement to a
//!   chip it was not taken on. The right fix is a carried answer that names the chip it was
//!   measured on, which the recording format does not hold today.
//! - its byte model (`bind::plane_copy`) is a check against our own model, not against Metal, so
//!   there is nothing for a Metal side to answer.
//! - its chained-blit case asserts a notice fired on the second copy and not the first, which is a
//!   per-device once rule rather than a Metal observation.

use crate::allowance::Allowance;
use crate::case::{answered, call, Abort, Case, Input, MetalSource, Side};
use crate::measured::hex_bytes;
use crate::replay::{notice_allowance, NOTICES_SITE};
use crate::transcript::{Digest, DigestKind, Node, Observed, Sites, Transcript, Value};
use crate::work::{Plane, Rect, Resource, Step, Storage, TextureSpec, Work};

/// Where the carried values were measured.
const MEASURED_IN: &str =
    "runtime_lane::two_plane_copies, from the probes bp.m and pad.m (an Apple M3 Max, \
     macOS 27.0.1)";

/// Why record mode must not run a case that fills on Metal.
///
/// The call, not the format: `fillTexture:` and `fillTextureColor:` are Apple SPI, and a case that
/// runs neither records on Metal whatever pixel format it names.
const PRIVATE_FILL: &str =
    "the fill goes through fillTexture:/fillTextureColor:, which are Apple SPI, so no public call \
     exists to record it from; the two-plane extended-range formats themselves are public, and \
     the cases here that run no fill do record on Metal";

/// The runs the row-blit table stands on: the probe ran each blit once, and the announced ones
/// three times alike.
const ROW_BLIT_RUNS: u32 = 1;

/// The runs the pad table stands on: the probe ran every case twice, alike.
const PAD_RUNS: u32 = 2;

/// `RGB10A8_2P_XR10` and its sRGB twin, by raw pixel format.
const P2: u32 = 550;
const P2S: u32 = 551;

/// `RGBA8Unorm`, which `pad.m`'s color fill packs its color in.
const RGBA8: u32 = 70;

/// One of `bp.m`'s blits: the texture's width, height and slices, whether it lies in a placement
/// heap of Metal's size, the slice the blit names, a fill of `a1 a2 a3 a4` first (x, y, width,
/// height), out of the texture or into it, the region, and Metal's words as the probe printed them.
struct RowBlit {
    extent: [u64; 3],
    heap: bool,
    slice: u64,
    fill: Option<[u64; 4]>,
    out: bool,
    region: [u64; 4],
    metal: &'static str,
}

const fn row_blit(
    (extent, heap, slice): ([u64; 3], bool, u64),
    fill: Option<[u64; 4]>,
    (out, region): (bool, [u64; 4]),
    metal: &'static str,
) -> RowBlit {
    RowBlit {
        extent,
        heap,
        slice,
        fill,
        out,
        region,
        metal,
    }
}

/// A texture of its own memory.
const OWN: bool = false;
/// A blit out of the texture, into a buffer.
const OUT: bool = true;
/// A blit into the texture, from a buffer.
const IN: bool = false;

/// Metal's blits that the runtime answers alike and announces nothing for, rows that meet into the
/// texture below a width of 32 among them.
const ROW_BLITS: [RowBlit; 22] = [
    row_blit(([5, 3, 1], OWN, 0), None, (OUT, [0, 0, 5, 3]),
        "t0 t1 t2 t3 t4 | t4 00000000 00000000 00000000 t5 | t5 t6 t7 t8 t9"),
    row_blit(([5, 3, 1], OWN, 0), None, (OUT, [1, 1, 3, 2]), "00000000 00000000 00000000 | t6 t7 t8"),
    row_blit(([7, 4, 1], OWN, 0), None, (OUT, [2, 1, 4, 3]),
        "t6 00000000 t7 t8 | t9 t10 t11 t12 | t13 00000000 t14 t15"),
    row_blit(([3, 4, 1], OWN, 0), None, (OUT, [0, 0, 3, 4]),
        "t0 t1 t2 | 00000000 00000000 00000000 | 00000000 00000000 00000000 | 00000000 00000000 00000000"),
    row_blit(([3, 4, 1], OWN, 0), None, (OUT, [1, 0, 2, 3]),
        "t1 t2 | 00000000 00000000 | 00000000 00000000"),
    row_blit(([1, 5, 1], OWN, 0), None, (OUT, [0, 0, 1, 5]),
        "t0 | 00000000 | 00000000 | 00000000 | 00000000"),
    row_blit(([5, 3, 2], OWN, 1), None, (OUT, [0, 0, 5, 3]),
        "t1.0 t1.1 t1.2 t1.3 t1.4 | t1.4 00000000 00000000 00000000 t1.5 | t1.5 t1.6 t1.7 t1.8 t1.9"),
    row_blit(([7, 4, 1], OWN, 0), None, (IN, [2, 1, 4, 3]),
        "t0 t1 t2 t3 t4 t5 b0000000 | b0000002 b0000003 b0000100 b0000101 b0000102 b0000103 b0000200 | b0000202 b0000203 t16 t17 t18 t19 t20 | t21 t22 t23 t24 t25 t26 t27"),
    row_blit(([3, 4, 1], OWN, 0), None, (IN, [0, 0, 3, 4]),
        "b0000000 b0000001 b0000002 | b0000100 b0000101 b0000102 | b0000200 b0000201 b0000202 | b0000300 b0000301 b0000302"),
    row_blit(([3, 4, 1], OWN, 0), None, (IN, [1, 1, 2, 2]),
        "t0 t1 t2 | t3 b0000000 b0000001 | t6 b0000100 b0000101 | t9 t10 t11"),
    row_blit(([5, 3, 1], true, 0), Some([1, 1, 2, 1]), (OUT, [0, 0, 5, 3]),
        "t0 t1 t2 t3 t4 | t4 a4a3a2a1 a4a3a2a1 00000000 t5 | t5 t6 t7 t8 t9"),
    row_blit(([7, 4, 1], true, 0), None, (IN, [2, 1, 4, 3]),
        "t0 t1 t2 t3 t4 t5 b0000000 | b0000002 b0000003 b0000100 b0000101 b0000102 b0000103 b0000200 | b0000202 b0000203 t16 t17 t18 t19 t20 | t21 t22 t23 t24 t25 t26 t27"),
    row_blit(([3, 4, 1], true, 0), None, (OUT, [0, 0, 3, 4]),
        "t0 t1 t2 | 00000000 00000000 00000000 | 00000000 00000000 00000000 | 00000000 00000000 00000000"),
    row_blit(([5, 3, 1], OWN, 0), Some([1, 1, 2, 1]), (OUT, [0, 0, 5, 3]),
        "t0 t1 t2 t3 t4 | t4 a4a3a2a1 a4a3a2a1 00000000 t5 | t5 t6 t7 t8 t9"),
    row_blit(([5, 6, 1], OWN, 0), None, (IN, [0, 1, 5, 4]),
        "t0 t1 t2 t3 b0000000 | b0000004 b0000101 b0000102 b0000103 b0000104 | b0000204 b0000301 b0000302 b0000303 b0000304 | t15 t16 t17 t18 t19 | t20 t21 t22 t23 t24 | t25 t26 t27 t28 t29"),
    row_blit(([5, 6, 1], true, 0), None, (IN, [0, 1, 5, 4]),
        "t0 t1 t2 t3 b0000000 | b0000004 b0000101 b0000102 b0000103 b0000104 | b0000204 b0000301 b0000302 b0000303 b0000304 | t15 t16 t17 t18 t19 | t20 t21 t22 t23 t24 | t25 t26 t27 t28 t29"),
    row_blit(([7, 4, 3], OWN, 2), None, (IN, [1, 1, 5, 3]),
        "t2.0 t2.1 t2.2 t2.3 t2.4 b0000000 b0000001 | b0000003 b0000004 b0000101 b0000102 b0000103 b0000104 b0000201 | b0000203 b0000204 t2.16 t2.17 t2.18 t2.19 t2.20 | t2.21 t2.22 t2.23 t2.24 t2.25 t2.26 t2.27"),
    row_blit(([7, 3, 1], OWN, 0), None, (IN, [0, 0, 7, 3]),
        "b0000000 b0000001 b0000002 b0000003 b0000004 b0000005 b0000006 | b0000200 b0000201 b0000202 b0000203 b0000204 b0000205 b0000206 | t14 t15 t16 t17 t18 t19 t20"),
    row_blit(([5, 8, 1], OWN, 0), None, (IN, [0, 1, 5, 6]),
        "t0 t1 t2 t3 b0000000 | b0000004 b0000101 b0000102 b0000103 b0000104 | b0000204 b0000301 b0000302 b0000303 b0000400 | b0000404 b0000501 b0000502 b0000503 b0000504 | t20 t21 t22 t23 t24 | t25 t26 t27 t28 t29 | t30 t31 t32 t33 t34 | t35 t36 t37 t38 t39"),
    row_blit(([5, 6, 1], OWN, 0), None, (IN, [0, 0, 5, 6]),
        "b0000000 b0000001 b0000002 b0000003 b0000004 | b0000200 b0000201 b0000202 b0000203 b0000204 | b0000400 b0000401 b0000402 b0000403 b0000404 | b0000504 t16 t17 t18 t19 | t20 t21 t22 t23 t24 | t25 t26 t27 t28 t29"),
    row_blit(([6, 5, 1], OWN, 0), None, (IN, [0, 0, 6, 5]),
        "b0000000 b0000001 b0000002 b0000003 b0000004 b0000005 | b0000200 b0000201 b0000202 b0000203 b0000204 b0000205 | b0000400 b0000401 b0000402 b0000403 b0000404 b0000405 | t18 t19 t20 t21 t22 t23 | t24 t25 t26 t27 t28 t29"),
    row_blit(([9, 4, 1], OWN, 0), None, (IN, [0, 0, 9, 4]),
        "b0000000 b0000001 b0000002 b0000003 b0000004 b0000005 b0000006 b0000007 b0000008 | b0000104 b0000105 b0000106 b0000107 b0000200 b0000201 b0000202 b0000203 b0000204 | b0000208 b0000301 b0000302 b0000303 b0000304 b0000305 b0000306 b0000307 b0000308 | t27 t28 t29 t30 t31 t32 t33 t34 t35"),
];

/// Metal's blits that the runtime announces: into the texture with rows that meet where Metal's
/// order varies from run to run, 32 texels wide or wider, and between groups of 32 rows, each
/// answered as Metal's in three runs, where the earlier row's bytes stood.
const ANNOUNCED_ROW_BLITS: [RowBlit; 2] = [
    row_blit(([41, 2, 1], OWN, 0), None, (IN, [0, 0, 41, 2]),
        "b0000000 b0000001 b0000002 b0000003 b0000004 b0000005 b0000006 b0000007 b0000008 b0000009 b000000a b000000b b000000c b000000d b000000e b000000f b0000010 b0000011 b0000012 b0000013 b0000014 b0000015 b0000016 b0000017 b0000018 b0000019 b000001a b000001b b000001c b000001d b000001e b000001f b0000020 b0000021 b0000022 b0000023 b0000024 b0000025 b0000026 b0000027 b0000028 | b0000104 b0000105 b0000106 b0000107 b0000108 b0000109 b000010a b000010b b000010c b000010d b000010e b000010f b0000110 b0000111 b0000112 b0000113 b0000114 b0000115 b0000116 b0000117 b0000118 b0000119 b000011a b000011b b000011c b000011d b000011e b000011f b0000120 b0000121 b0000122 b0000123 b0000124 b0000125 b0000126 b0000127 b0000128 t78 t79 t80 t81"),
    row_blit(([5, 33, 1], OWN, 0), None, (IN, [0, 0, 5, 33]),
        "b0000000 b0000001 b0000002 b0000003 b0000004 | b0000200 b0000201 b0000202 b0000203 b0000204 | b0000400 b0000401 b0000402 b0000403 b0000404 | b0000600 b0000601 b0000602 b0000603 b0000604 | b0000800 b0000801 b0000802 b0000803 b0000804 | b0000a00 b0000a01 b0000a02 b0000a03 b0000a04 | b0000c00 b0000c01 b0000c02 b0000c03 b0000c04 | b0000e00 b0000e01 b0000e02 b0000e03 b0000e04 | b0001000 b0001001 b0001002 b0001003 b0001004 | b0001200 b0001201 b0001202 b0001203 b0001204 | b0001400 b0001401 b0001402 b0001403 b0001404 | b0001600 b0001601 b0001602 b0001603 b0001604 | b0001800 b0001801 b0001802 b0001803 b0001804 | b0001a00 b0001a01 b0001a02 b0001a03 b0001a04 | b0001c00 b0001c01 b0001c02 b0001c03 b0001c04 | b0001e00 b0001e01 b0001e02 b0001e03 b0001e04 | b0001f04 b0002001 b0002002 b0002003 b0002004 | t85 t86 t87 t88 t89 | t90 t91 t92 t93 t94 | t95 t96 t97 t98 t99 | t100 t101 t102 t103 t104 | t105 t106 t107 t108 t109 | t110 t111 t112 t113 t114 | t115 t116 t117 t118 t119 | t120 t121 t122 t123 t124 | t125 t126 t127 t128 t129 | t130 t131 t132 t133 t134 | t135 t136 t137 t138 t139 | t140 t141 t142 t143 t144 | t145 t146 t147 t148 t149 | t150 t151 t152 t153 t154 | t155 t156 t157 t158 t159 | t160 t161 t162 t163 t164"),
];

/// The texel word `replace_region` sets for texel `index` of `slice`: what the probe's `t<s>.<i>`
/// names.
fn texel_word(slice: u64, index: u64) -> u32 {
    0x1000_0000 + ((slice as u32) << 16) + index as u32
}

/// The buffer word the probe's input holds at row `row`, column `column`: its `b<rrcc>`.
fn buffer_word(row: u64, column: u64) -> u32 {
    0xb000_0000 + ((row as u32) << 8) + column as u32
}

/// The words `printed` names, as `bp.m` printed them: a texel word by its slice and index, and
/// anything else a hexadecimal literal.
fn printed_words(printed: &str) -> Result<Vec<u32>, String> {
    printed
        .split_whitespace()
        .filter(|token| *token != "|")
        .map(|token| match token.strip_prefix('t') {
            Some(texel) => {
                let (slice, index) = texel.split_once('.').unwrap_or(("0", texel));
                let slice: u64 = slice.parse().map_err(|error| format!("{token}: {error}"))?;
                let index: u64 = index.parse().map_err(|error| format!("{token}: {error}"))?;
                Ok(texel_word(slice, index))
            }
            None => u32::from_str_radix(token, 16).map_err(|error| format!("{token}: {error}")),
        })
        .collect()
}

/// The words of `bytes`, little-endian.
fn words_of(bytes: &[u8]) -> Vec<u32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| u32::from_le_bytes(*word))
        .collect()
}

/// Words as a transcript value: a list, so a difference names the word that moved.
fn words_value(words: &[u32]) -> Value {
    Value::List(words.iter().map(|&word| Value::Uint(word.into())).collect())
}

/// The program one row blit runs.
fn row_blit_work(blit: &RowBlit) -> Work {
    let [w, h, slices] = blit.extent;
    let spec = {
        let mut spec = TextureSpec::texture_2d(P2, w, h).shared();
        if slices > 1 {
            spec = spec.array(slices).shared();
        }
        spec
    };
    let mut work = Work::new();
    let texture = if blit.heap {
        let heap = work.add(Resource::HeapForTexture {
            spec,
            storage: Storage::Shared,
        });
        work.add(Resource::HeapTexture {
            heap,
            spec,
            offset: 0,
        })
    } else {
        work.add(Resource::Texture { spec })
    };
    for slice in 0..slices {
        let words: Vec<u8> = (0..w * h)
            .flat_map(|index| texel_word(slice, index).to_le_bytes())
            .collect();
        work.step(Step::ReplaceRegion {
            texture,
            level: 0,
            slice,
            region: Rect::new(0, 0, w, h),
            bytes: words,
            // An uncompressed plane's rows are its texel rows.
            rows: h,
            bytes_per_row: 4 * w,
            bytes_per_image: 4 * w * h,
        });
    }
    if let Some([x, y, fw, fh]) = blit.fill {
        work.step(Step::FillTexture {
            texture,
            level: 0,
            slice: 0,
            region: Rect::new(x, y, fw, fh),
            pattern: vec![0xa1, 0xa2, 0xa3, 0xa4],
        });
        work.step(Step::Commit);
    }
    let [x, y, rw, rh] = blit.region;
    let row = 4 * rw;
    if blit.out {
        let buffer = work.buffer(vec![0xee; (row * rh) as usize]);
        work.step(Step::CopyTextureToBuffer {
            texture,
            slice: blit.slice,
            level: 0,
            origin: [x, y, 0],
            size: [rw, rh, 1],
            buffer,
            offset: 0,
            bytes_per_row: row,
            bytes_per_image: row * rh,
            plane: Plane::Whole,
        });
        work.step(Step::Commit);
        work.step(Step::ReadBuffer {
            buffer,
            offset: 0,
            length: (row * rh) as usize,
        });
        return work;
    }
    let input: Vec<u8> = (0..rh)
        .flat_map(|r| (0..rw).map(move |c| buffer_word(r, c)))
        .flat_map(u32::to_le_bytes)
        .collect();
    let buffer = work.buffer(input);
    work.step(Step::CopyBufferToTexture {
        buffer,
        offset: 0,
        bytes_per_row: row,
        bytes_per_image: row * rh,
        size: [rw, rh, 1],
        texture,
        slice: blit.slice,
        level: 0,
        origin: [x, y, 0],
        plane: Plane::Whole,
    });
    work.step(Step::Commit);
    work.step(Step::GetBytes {
        texture,
        level: 0,
        slice: blit.slice,
        region: Rect::new(0, 0, w, h),
        bytes_per_row: 4 * w,
        bytes_per_image: 4 * w * h,
        length: (4 * w * h) as usize,
        // The read starts from zero, as this case's reads always did: it asks
        // nothing about the padding between its rows.
        before: 0,
    });
    work
}

/// The row-blit case.
pub struct RowBlits {
    name: &'static str,
    /// Whether the case holds the blits the device announces.
    announced: bool,
}

/// The blits the runtime announces nothing for.
pub fn row_blits() -> RowBlits {
    RowBlits {
        name: "two-plane-row-blits",
        announced: false,
    }
}

/// The blits whose rows meet where Metal's own order varies from run to run, which the device
/// announces.
pub fn announced_row_blits() -> RowBlits {
    RowBlits {
        name: "two-plane-announced-row-blits",
        announced: true,
    }
}

impl RowBlits {
    /// The transcript the probe's own printed words state: every word of every blit, as `bp.m`
    /// printed it.
    ///
    /// For `two-plane-row-blits` this is the case's answer, because two of its blits fill. For
    /// `two-plane-announced-row-blits`, which record mode runs on Metal, it is the cross-check: a
    /// test holds the committed recording to it, so the probe's words and Metal's live answer must
    /// agree.
    pub fn printed_by_the_probe(&self) -> Result<Transcript, String> {
        let mut sites = Sites::new();
        sites.push(crate::case::inputs_node(&self.inputs()?));
        for (at, blit) in self.blits().iter().enumerate() {
            sites.leaf(format!("work/{at}"), answered());
            sites.push(Node::leaf(
                row_site(at, blit),
                Observed::value(words_value(&printed_words(blit.metal)?)),
            ));
        }
        sites.notices(NOTICES_SITE, &[] as &[&str]);
        Ok(sites.finish())
    }

    /// The case's blits.
    fn blits(&self) -> &'static [RowBlit] {
        if self.announced {
            &ANNOUNCED_ROW_BLITS
        } else {
            &ROW_BLITS
        }
    }
}

/// The site a row blit's words sit at.
fn row_site(at: usize, blit: &RowBlit) -> String {
    let [w, h, slices] = blit.extent;
    let way = if blit.out { "out" } else { "in" };
    format!("blit/{at}/{w}x{h}x{slices}-{way}")
}

impl Case for RowBlits {
    fn name(&self) -> &'static str {
        self.name
    }

    fn metal_source(&self) -> MetalSource {
        // The announced blits fill nothing, so every call they make is public and record mode
        // takes Metal's own answer. The other twenty-two include two that fill first.
        if self.announced {
            return MetalSource::Run;
        }
        MetalSource::carried_only(MEASURED_IN, PRIVATE_FILL)
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let mut geometry = String::new();
        let mut words = Vec::new();
        for blit in self.blits() {
            let [w, h, slices] = blit.extent;
            geometry.push_str(&format!(
                "{w}x{h}x{slices} heap {} slice {} fill {:?} out {} region {:?}\n",
                blit.heap, blit.slice, blit.fill, blit.out, blit.region
            ));
            // The texels the texture holds before the blit, and the buffer's words where the blit
            // reads one: what the probe fed, so a change to either makes the answer another run's.
            for slice in 0..slices {
                for index in 0..w * h {
                    words.extend(texel_word(slice, index).to_le_bytes());
                }
            }
            if !blit.out {
                let [_, _, rw, rh] = blit.region;
                for r in 0..rh {
                    for c in 0..rw {
                        words.extend(buffer_word(r, c).to_le_bytes());
                    }
                }
            }
        }
        Ok(vec![
            Input::bytes("blit-geometry", geometry.as_bytes()),
            Input::bytes("words-before", &words),
        ])
    }

    fn allowances(&self) -> Vec<Allowance> {
        // Every case declares the notice the venue's own memory model provokes: the buffer it
        // reads the copied planes back through is `Shared`, which a device with no unified memory
        // keeps in system memory because that is the only memory both sides can reach.
        let shared =
            notice_allowance(metal2vulkan_runtime::Degradation::SharedInSystemMemory.code());
        if !self.announced {
            // Three of the blits place their texture in a placement heap, where this device keeps
            // the two-plane texture's texels in the heap's bytes and announces it. Metal announces
            // nothing, so the notices site differs and that notice is what justifies it. Every
            // word still compares exactly.
            return vec![
                notice_allowance(metal2vulkan_runtime::Degradation::ShadowedTexture.code()),
                shared,
            ];
        }
        // Into the texture, where rows meet and Metal's own order varies from run to run, the
        // device says which row's bytes it kept. Metal announces nothing, so the notices site
        // differs; the two notices are what justify it. The words still compare exactly.
        vec![
            notice_allowance(metal2vulkan_runtime::Degradation::TwoPlaneBufferBlit.code()),
            notice_allowance(metal2vulkan_runtime::Degradation::TwoPlaneBufferBlitOrder.code()),
            shared,
        ]
    }

    fn carried_transcript(&self) -> Result<Option<(u32, Transcript)>, String> {
        if self.announced {
            // Record mode runs this one on Metal. Its probe's table stays as the cross-check
            // ([`RowBlits::printed_by_the_probe`]), held against the committed recording by a
            // test, so the words the probe printed still have to agree with what Metal answers
            // here.
            return Ok(None);
        }
        Ok(Some((ROW_BLIT_RUNS, self.printed_by_the_probe()?)))
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        for (at, blit) in self.blits().iter().enumerate() {
            let work = row_blit_work(blit);
            let done = call(sites, &format!("work/{at}"), side.work(&work), |_| {
                Value::Absent
            })?;
            let read = done.reads.first().cloned().unwrap_or_default();
            sites.push(Node::leaf(
                row_site(at, blit),
                Observed::value(words_value(&words_of(&read))),
            ));
        }
        Ok(())
    }
}

/// One of `pad.m`'s cases: the format, the texture's width, height and slices, the slice the fill
/// names, its region, the fill (a pattern's bytes in hexadecimal, or a color), and the digest the
/// probe printed of its lines.
type PadCase = (
    u32,
    [u64; 3],
    u64,
    [u64; 4],
    Result<&'static str, [f64; 4]>,
    &'static str,
);

const PAD_CASES: [PadCase; 9] = [
    (
        P2,
        [7, 3, 1],
        0,
        [0, 0, 7, 3],
        Ok("0102030405060708"),
        "a22dfa6fdd93570d",
    ),
    (
        P2,
        [7, 3, 1],
        0,
        [2, 1, 9, 5],
        Ok("090a0b"),
        "5b9fcd63521fb5dc",
    ),
    (
        P2S,
        [33, 5, 1],
        0,
        [3, 1, 20, 3],
        Err([1.0, 0.5, 0.25, 1.0]),
        "ce168b61643ecebb",
    ),
    (
        P2,
        [6, 2, 1],
        0,
        [0, 0, 6, 2],
        Ok("0102030405060708"),
        "dcc033fcbab11d58",
    ),
    (P2, [7, 3, 1], 0, [0, 0, 0, 0], Ok("01"), "37ba266b851a2342"),
    (
        P2,
        [13, 4, 1],
        0,
        [1, 1, 5, 2],
        Ok("a1b2c3d4"),
        "94b181522346b0df",
    ),
    (
        P2,
        [5, 3, 2],
        1,
        [0, 0, 5, 3],
        Ok("a1b2c3d4"),
        "fea04cfb6251b2ab",
    ),
    (
        P2,
        [3, 4, 1],
        0,
        [0, 0, 3, 4],
        Ok("a1b2c3d4"),
        "683193cef365dc59",
    ),
    (
        P2S,
        [9, 3, 1],
        0,
        [1, 0, 7, 3],
        Ok("c1c2c3c4"),
        "b6719f0cdbc1e202",
    ),
];

/// The program one pad case runs: a private texture of the format and extent, every color word set
/// to `0xee` by a blit from a buffer, one fill, and every slice read back by a blit into a buffer,
/// rows `4w` bytes apart.
fn pad_work(case: &PadCase) -> Result<Work, String> {
    let &(format, [w, h, slices], slice, [x, y, rw, rh], fill, _) = case;
    let mut spec = TextureSpec::texture_2d(format, w, h);
    if slices > 1 {
        spec = spec.array(slices);
    }
    let (row, image) = (4 * w, 4 * w * h);
    let mut work = Work::new();
    let texture = work.add(Resource::Texture { spec });
    let source = work.buffer(vec![0xee; image as usize]);
    let out = work.buffer(vec![0x55; (image * slices) as usize]);
    for s in 0..slices {
        work.step(Step::CopyBufferToTexture {
            buffer: source,
            offset: 0,
            bytes_per_row: row,
            bytes_per_image: image,
            size: [w, h, 1],
            texture,
            slice: s,
            level: 0,
            origin: [0, 0, 0],
            plane: Plane::Whole,
        });
    }
    let region = Rect::new(x, y, rw, rh);
    work.step(match fill {
        Ok(pattern) => Step::FillTexture {
            texture,
            level: 0,
            slice,
            region,
            pattern: hex_bytes(pattern)?,
        },
        Err([r, g, b, a]) => Step::FillTextureColor {
            texture,
            level: 0,
            slice,
            region,
            color: [r, g, b, a],
            packed_in: Some(RGBA8),
        },
    });
    for s in 0..slices {
        work.step(Step::CopyTextureToBuffer {
            texture,
            slice: s,
            level: 0,
            origin: [0, 0, 0],
            size: [w, h, 1],
            buffer: out,
            offset: s * image,
            bytes_per_row: row,
            bytes_per_image: image,
            plane: Plane::Whole,
        });
    }
    work.step(Step::Commit);
    work.step(Step::ReadBuffer {
        buffer: out,
        offset: 0,
        length: (image * slices) as usize,
    });
    Ok(work)
}

/// The probe's lines of `bytes`: a line a row of a slice, `s<slice> r<row>:` and the row's bytes.
fn pad_lines(bytes: &[u8], [w, h, slices]: [u64; 3]) -> Vec<String> {
    let (row, image) = (4 * w, 4 * w * h);
    (0..slices)
        .flat_map(|s| (0..h).map(move |r| (s, r)))
        .map(|(s, r)| {
            let at = (s * image + r * row) as usize;
            let hex: String = bytes
                .get(at..at + row as usize)
                .unwrap_or_default()
                .iter()
                .map(|byte| format!(" {byte:02x}"))
                .collect();
            format!("s{s} r{r}:{hex}")
        })
        .collect()
}

/// The pad-fill case.
pub struct PadFills {
    name: &'static str,
}

/// The ported case.
pub fn pad_fills() -> PadFills {
    PadFills {
        name: "two-plane-pad-fills",
    }
}

/// The site a pad case sits at.
fn pad_site(at: usize, case: &PadCase) -> String {
    let [w, h, slices] = case.1;
    format!("fill/{at}/{}-{w}x{h}x{slices}", case.0)
}

impl Case for PadFills {
    fn name(&self) -> &'static str {
        self.name
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::carried_only(MEASURED_IN, PRIVATE_FILL)
    }

    fn allowances(&self) -> Vec<Allowance> {
        // A fill into a two-plane texture's planes goes through the device's two-plane buffer
        // blit, which it announces. Metal announces nothing, so the notices site differs and that
        // notice is what justifies it; every digest still compares exactly. The sibling row blits
        // name this notice for the same reason, and the order notice besides, which a fill of
        // whole rows never meets.
        [
            metal2vulkan_runtime::Degradation::TwoPlaneBufferBlit,
            // The venue's own, for the `Shared` read-back buffer, as the row blits state.
            metal2vulkan_runtime::Degradation::SharedInSystemMemory,
        ]
        .iter()
        .map(|kind| notice_allowance(kind.code()))
        .collect()
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let mut rows = String::new();
        let mut patterns = Vec::new();
        for case in &PAD_CASES {
            rows.push_str(&format!(
                "{} {:?} slice {} region {:?} fill {:?}\n",
                case.0, case.1, case.2, case.3, case.4
            ));
            match case.4 {
                Ok(pattern) => patterns.extend(hex_bytes(pattern)?),
                Err(color) => {
                    for channel in color {
                        patterns.extend(channel.to_le_bytes());
                    }
                }
            }
        }
        Ok(vec![
            Input::bytes("cases", rows.as_bytes()),
            Input::bytes("fill-patterns", &patterns),
            // Every color word set to 0xee before the fill, and 0x55 in the buffer read into.
            Input::bytes("before", b"texture words 0xee, read-back buffer 0x55"),
        ])
    }

    fn carried_transcript(&self) -> Result<Option<(u32, Transcript)>, String> {
        let mut sites = Sites::new();
        sites.push(crate::case::inputs_node(&self.inputs()?));
        for (at, case) in PAD_CASES.iter().enumerate() {
            sites.leaf(format!("work/{at}"), answered());
            // The probe kept the digest of its printed lines and not the lines, so the site holds
            // the digest alone: the bytes between the rows are inside it and nowhere else.
            sites.push(Node::leaf(
                pad_site(at, case),
                Observed::absent().digested(Digest {
                    kind: DigestKind::Sha256Prefix16,
                    hex: case.5.to_owned(),
                }),
            ));
        }
        sites.notices(NOTICES_SITE, &[] as &[&str]);
        Ok(Some((PAD_RUNS, sites.finish())))
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        for (at, case) in PAD_CASES.iter().enumerate() {
            let work = pad_work(case).map_err(|why| Abort {
                at: format!("work/{at}"),
                failure: crate::case::Failure::named("BadCase", why),
            })?;
            let done = call(sites, &format!("work/{at}"), side.work(&work), |_| {
                Value::Absent
            })?;
            let read = done.reads.first().cloned().unwrap_or_default();
            let lines = pad_lines(&read, case.1);
            sites.push(Node::leaf(
                pad_site(at, case),
                Observed::absent().digested(Digest::sha256_prefix16(lines.join("\n").as_bytes())),
            ));
        }
        Ok(())
    }
}
