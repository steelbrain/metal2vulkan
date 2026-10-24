//! Metal's private texture fills, ported from `runtime_lane::texture_fills`.
//!
//! The probes (`tfl.swift`, `tfl3.swift` and `tfl4.swift`, an Apple M4 under macOS 27.0.1): a
//! private 2D or 2D array texture of the case's extent, levels and slices, every byte of every
//! level, slice and aspect set to `0xEE` by copies from a buffer, one fill, and every level, slice
//! and aspect read back by copies into a buffer; for a two-plane extended-range format, a one-level
//! texture at byte 0 of a shared placement heap of Metal's size, every byte of the heap set to
//! `0xEE` through a buffer over it, and the heap's changed bytes read back.
//!
//! **One case per format**, which is the port's one real gain over the lane. The lane walks the 50
//! formats in one test and, where the device holds no texture of one, prints a line and
//! `continue`s to the next format — so a format the venue cannot make leaves the test green with
//! nothing compared and nothing counted. Here the venue's typed refusal stops that format's case
//! alone, and the suite counts it as [`LacksFeature`](crate::verdict::Verdict::LacksFeature):
//! neither a pass nor a failure, and visible in the matrix. That matters on lavapipe, which holds
//! no texture of several of these formats.
//!
//! **The fidelity is the lane's and no better**: Metal's answer per format is the 16-hex-digit
//! digest of its 20 cases' lines, each line `<case> <answer>` where the answer is itself the digest
//! of that case's read-back lines. The probes kept the outer digest alone, so the case's site is
//! digest-only and the suite reports it as such. A difference names the format and no more — the
//! same reach the lane has, because a digest of digests is all anybody wrote down. Porting it does
//! not make it sharper; recording the per-case digests would, and that needs the probe run again.
//!
//! Metal's answer is **carried** and re-recording is **forbidden**: Metal fills a texture through
//! private API. The digests were measured on **an Apple M4**, not on this M3 Max, which the
//! recording's provenance says in words.

use crate::allowance::{Allowance, Justification, Loosening};
use crate::case::{answered, made, Abort, Case, Failure, Input, MetalSource, Side};
use crate::replay::{notice_allowance, NOTICES_SITE};
use crate::transcript::{
    ByteLayout, Digest, DigestKind, Element, Node, Observed, Sites, Transcript, Value,
};
use crate::work::{Plane, Rect, Resource, Step, Storage, TextureSpec, Work};

/// Where the carried values were measured.
const MEASURED_IN: &str =
    "runtime_lane::texture_fills, from the probes tfl.swift, tfl3.swift and tfl4.swift (an Apple \
     M4, macOS 27.0.1) -- another chip than this round's oracle";

/// Why record mode must not run these cases on Metal.
const NO_PUBLIC_CALL: &str =
    "Metal fills a texture through private API alone, so no public call exists to record it from";

/// The runs the carried table stands on: the probes ran each format once.
const MEASURED_RUNS: u32 = 1;

/// The byte every level, slice and aspect holds before the fill.
const BEFORE: u8 = 0xee;

/// Metal's answers: each format's raw value, its heap's bytes (a two-plane format's, else 0), and
/// the digest of its cases' lines.
const FORMATS: [(u32, u64, &str); 50] = [
    (10, 0, "5464fe66b33895e2"),
    (12, 0, "aa81065f10178e1f"),
    (13, 0, "ed9490278981257b"),
    (20, 0, "24fe976f77bb4506"),
    (23, 0, "3c66fbe4c17d400e"),
    (25, 0, "14accca524027699"),
    (30, 0, "b5ee494f91b9b3c6"),
    (40, 0, "be84d6a6cdce7ab4"),
    (41, 0, "9ec87b84ff4de073"),
    (42, 0, "6987da912b86071a"),
    (43, 0, "07cfccb40d827c06"),
    (53, 0, "4573cb785ba0b16a"),
    (55, 0, "f830ab1fa7dd3cc5"),
    (60, 0, "b3cf44f6777186d7"),
    (65, 0, "399c23bba36179e3"),
    (70, 0, "5e21f045d0204773"),
    (71, 0, "32cba3e03801ac38"),
    (80, 0, "8bcbb14846224039"),
    (81, 0, "fbb8049269874f64"),
    (90, 0, "1448888ff150253d"),
    (92, 0, "9067c0305af37729"),
    (93, 0, "f0aa4ed5a6cabe46"),
    (94, 0, "8f3866a50999be31"),
    (103, 0, "e78f560f4df1b5d0"),
    (105, 0, "d260d8c7f05ca8ff"),
    (110, 0, "2df2cb38c53448fb"),
    (115, 0, "27f9e93fc6475f47"),
    (123, 0, "87c65dbffa375059"),
    (125, 0, "a1a38a2bc59adc90"),
    (130, 0, "eee8a299143ef6c3"),
    (134, 0, "f96803ca3cf8b50a"),
    (152, 0, "f96803ca3cf8b50a"),
    (160, 0, "be7903fe2aeebda6"),
    (162, 0, "62f023563eaba5d6"),
    (170, 0, "eee8a299143ef6c3"),
    (178, 0, "f96803ca3cf8b50a"),
    (180, 0, "eee8a299143ef6c3"),
    (187, 0, "f96803ca3cf8b50a"),
    (198, 0, "a7cd635068daae8f"),
    (212, 0, "a7cd635068daae8f"),
    (236, 0, "a7cd635068daae8f"),
    (240, 0, "76db2674be6f822f"),
    (241, 0, "76db2674be6f822f"),
    (250, 0, "25b03997269e7e5f"),
    (252, 0, "91bf73db1b60d4f9"),
    (253, 0, "3e98e4db29723b7e"),
    (260, 0, "6bad089e9ea96a78"),
    (261, 0, "dbb9a8f56e387de0"),
    (550, 180224, "6e27fb9c47da9062"),
    (551, 180224, "6e27fb9c47da9062"),
];

/// Metal's answers for two-plane textures of their own memory, 8 texels wide (`tfl4.swift`).
const OWN_TWO_PLANES: [(u32, &str); 2] = [(550, "81c7070a27e42420"), (551, "81c7070a27e42420")];

/// The cases `tfl4.swift` ran.
const OWN_CASES: [&str; 15] = [
    "b0", "b1", "b2", "b3", "b4", "b5", "b6", "b7", "b8", "b9", "b10", "b11", "c2", "c3", "c6",
];

/// The cases `tfl.swift` and `tfl3.swift` ran.
const CASES: [&str; 20] = [
    "b0", "b1", "b2", "b3", "b4", "b5", "b6", "b7", "b8", "b9", "b10", "b11", "c0", "c1", "c2",
    "c3", "c4", "c5", "c6", "c7",
];

/// A format's unit as the probes count it: its block's bytes, width and height (a texel's for an
/// uncompressed format; a 4:2:2 format's 2 x 1 block).
fn block(raw: u32) -> (u64, u64, u64) {
    match raw {
        130 | 140 | 170 | 180 => (8, 4, 4),
        134 | 152 | 178 => (16, 4, 4),
        160 => (8, 8, 4),
        162 => (8, 4, 4),
        187 => (16, 5, 4),
        198 => (16, 10, 10),
        212 => (16, 8, 8),
        236 => (16, 12, 12),
        240 | 241 => (4, 2, 1),
        253 | 261 | 10 | 12 | 13 => (1, 1, 1),
        260 | 550 | 551 => (4, 1, 1),
        20 | 23 | 25 | 30 | 40 | 41 | 42 | 43 | 250 => (2, 1, 1),
        103 | 105 | 110 | 115 => (8, 1, 1),
        123 | 125 => (16, 1, 1),
        _ => (4, 1, 1),
    }
}

/// Whether a format's fills do not follow the color: a block-compressed or 4:2:2 one (a block of
/// more than one texel), depth or stencil, or two-plane extended-range. The probes' table records
/// `unpacked` for such a fill, because Metal's bytes there are its own stack's.
fn ignores_the_color(raw: u32) -> bool {
    let (_, bw, bh) = block(raw);
    bw * bh > 1 || matches!(raw, 250 | 252 | 253 | 260 | 261 | 550 | 551)
}

/// One probe case: the texture and the fill.
struct Probe {
    array: bool,
    extent: (u64, u64),
    levels: u64,
    slices: u64,
    level: u64,
    slice: u64,
    origin: (u64, u64),
    size: (u64, u64),
    fill: Fill,
}

/// What a case fills.
enum Fill {
    /// A pattern of `0` bytes, `0x80 + k` each.
    Bytes(usize),
    /// A color, packed in the format the option names, or in the texture's own.
    Color([f64; 4], Option<u32>),
}

/// The probes' case `name` of format `raw`; `wide`, at `tfl4.swift`'s extents 8 texels wide.
fn probe(raw: u32, name: &str, wide: bool) -> Result<Probe, String> {
    let (bb, bw, bh) = block(raw);
    let pvrtc = raw == 160 || raw == 162;
    let texels = bw == 1 && bh == 1;
    let side = 4 * bw.max(bh);
    let e = if wide {
        (8, 3)
    } else if pvrtc {
        (side, side)
    } else if texels {
        (5, 3)
    } else {
        (4 * bw, 3 * bh)
    };
    let odd = if wide {
        (12, 5)
    } else if pvrtc {
        e
    } else if texels {
        (7, 5)
    } else {
        (2 * bw + bw / 2 + 1, bh + bh / 2 + 1)
    };
    let mip = if wide {
        (8, 8)
    } else if pvrtc {
        (2 * side, 2 * side)
    } else {
        (4 * bw, 4 * bh)
    };
    let plain = |origin, size, fill| Probe {
        array: false,
        extent: e,
        levels: 1,
        slices: 1,
        level: 0,
        slice: 0,
        origin,
        size,
        fill,
    };
    let color = |r, g, b, a, format| Fill::Color([r, g, b, a], format);
    let bytes = |n: u64| Fill::Bytes(n as usize);
    Ok(match name {
        "b0" => plain((0, 0), e, bytes(bb)),
        "b1" => plain((bw, bh), (bw, bh), bytes(3)),
        "b2" => plain((1, 1), (1, 1), bytes(bb)),
        "b3" => plain((0, 0), (2 * bw + 1, bh), bytes(bb + 5)),
        "b4" => plain((3 * bw, 2 * bh), (100, 100), bytes(2 * bb)),
        "b5" => Probe {
            extent: odd,
            ..plain((0, 0), odd, bytes(bb))
        },
        "b6" => Probe {
            extent: mip,
            levels: 3,
            level: 1,
            ..plain((0, 0), (mip.0 / 2, mip.1 / 2), bytes(1))
        },
        "b7" => Probe {
            extent: mip,
            levels: 3,
            level: 2,
            ..plain((0, 0), (1, 1), bytes(bb))
        },
        "b8" => Probe {
            array: true,
            slices: 2,
            slice: 1,
            ..plain((0, 0), e, bytes(bb))
        },
        "b9" => plain((2, 1), (5, 3), bytes(7)),
        "b10" => plain((0, 0), e, bytes(64)),
        "b11" => plain((bw / 2, 0), (bw, 1), bytes(bb)),
        "c0" => plain((0, 0), e, color(0.25, 0.5, 0.75, 1.0, None)),
        "c1" => plain((0, 0), e, color(0.25, 0.5, 0.75, 1.0, Some(raw))),
        "c2" => plain((0, 0), e, color(0.25, 0.5, 0.75, 1.0, Some(70))),
        "c3" => plain((0, 0), e, color(0.25, 0.5, 0.75, 1.0, Some(115))),
        "c4" => plain((bw, bh), (bw, bh), color(0.25, 0.5, 0.75, 1.0, None)),
        "c5" => plain((0, 0), e, color(2.0, -1.0, 0.5, 0.3, None)),
        "c6" => plain((0, 0), e, color(0.25, 0.5, 0.75, 1.0, Some(10))),
        "c7" => plain((0, 0), e, color(0.0, 0.0, 0.0, 0.0, None)),
        _ => return Err(format!("no case {name}")),
    })
}

/// The pattern a `Fill::Bytes(n)` holds.
fn pattern(n: usize) -> Vec<u8> {
    (0..n).map(|k| 0x80u8.wrapping_add(k as u8)).collect()
}

/// The fill step of `probe` into resource `texture`, its pattern one zero byte where `zeros`.
fn fill_step(texture: usize, p: &Probe, zeros: bool) -> Step {
    let region = Rect::new(p.origin.0, p.origin.1, p.size.0, p.size.1);
    if zeros {
        return Step::FillTexture {
            texture,
            level: p.level,
            slice: p.slice,
            region,
            pattern: vec![0],
        };
    }
    match &p.fill {
        Fill::Bytes(n) => Step::FillTexture {
            texture,
            level: p.level,
            slice: p.slice,
            region,
            pattern: pattern(*n),
        },
        Fill::Color(color, packed_in) => Step::FillTextureColor {
            texture,
            level: p.level,
            slice: p.slice,
            region,
            color: *color,
            packed_in: *packed_in,
        },
    }
}

/// The spec of `probe`'s texture of `raw`.
fn spec(raw: u32, p: &Probe) -> TextureSpec {
    let mut spec = TextureSpec::texture_2d(raw, p.extent.0, p.extent.1).levels(p.levels);
    if p.array {
        spec = spec.array(p.slices).levels(p.levels);
    }
    spec
}

/// One part of a texture's read-back: a level, a slice and an aspect, with its extent, its blocks
/// across and down, the bytes of a block in that aspect, and where it sits in the buffer.
struct Part {
    level: u64,
    slice: u64,
    plane: Plane,
    unit: u64,
    aspect: &'static str,
    extent: (u64, u64),
    blocks: (u64, u64),
    at: u64,
}

/// Every part of `probe`'s texture of `raw`, and the buffer's bytes they need.
fn parts(raw: u32, p: &Probe) -> (Vec<Part>, u64) {
    let (bb, bw, bh) = block(raw);
    let aspects: &[(Plane, u64, &str)] = match raw {
        260 => &[(Plane::Depth, 4, "depth"), (Plane::Stencil, 1, "stencil")],
        261 => &[(Plane::Stencil, 1, "stencil")],
        _ => &[(Plane::Whole, bb, "all")],
    };
    let mut out = Vec::new();
    let mut total = 0;
    for level in 0..p.levels {
        let w = (p.extent.0 >> level).max(1);
        let h = (p.extent.1 >> level).max(1);
        let (across, down) = (w.div_ceil(bw), h.div_ceil(bh));
        for slice in 0..p.slices {
            for &(plane, unit, aspect) in aspects {
                // Each part at a multiple of 16 bytes, which every copy unit divides.
                total = u64::next_multiple_of(total, 16);
                out.push(Part {
                    level,
                    slice,
                    plane,
                    unit,
                    aspect,
                    extent: (w, h),
                    blocks: (across, down),
                    at: total,
                });
                total += across * down * unit;
            }
        }
    }
    (out, total)
}

/// The program one case of a texture of its own memory runs.
fn texture_work(raw: u32, p: &Probe, zeros: bool) -> Work {
    let (parts, total) = parts(raw, p);
    let mut work = Work::new();
    let texture = work.add(Resource::Texture { spec: spec(raw, p) });
    let prefill = work.buffer(vec![BEFORE; total as usize]);
    let readback = work.buffer(vec![0; total as usize]);
    for part in &parts {
        let row = part.blocks.0 * part.unit;
        work.step(Step::CopyBufferToTexture {
            buffer: prefill,
            offset: part.at,
            bytes_per_row: row,
            bytes_per_image: row * part.blocks.1,
            size: [part.extent.0, part.extent.1, 1],
            texture,
            slice: part.slice,
            level: part.level,
            origin: [0, 0, 0],
            plane: part.plane,
        });
    }
    work.step(fill_step(texture, p, zeros));
    for part in &parts {
        let row = part.blocks.0 * part.unit;
        work.step(Step::CopyTextureToBuffer {
            texture,
            slice: part.slice,
            level: part.level,
            origin: [0, 0, 0],
            size: [part.extent.0, part.extent.1, 1],
            buffer: readback,
            offset: part.at,
            bytes_per_row: row,
            bytes_per_image: row * part.blocks.1,
            plane: part.plane,
        });
    }
    work.step(Step::Commit);
    work.step(Step::ReadBuffer {
        buffer: readback,
        offset: 0,
        length: total as usize,
    });
    work
}

/// The probe's read-back lines of a texture's parts.
fn texture_lines(raw: u32, p: &Probe, bytes: &[u8]) -> Vec<String> {
    let (parts, _) = parts(raw, p);
    parts
        .iter()
        .map(|part| {
            let n = part.blocks.0 * part.blocks.1 * part.unit;
            let hex: String = bytes
                .get(part.at as usize..(part.at + n) as usize)
                .unwrap_or_default()
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect();
            format!(
                "level {} slice {} {} {}x{}x{}: {hex}",
                part.level, part.slice, part.aspect, part.blocks.0, part.blocks.1, part.unit
            )
        })
        .collect()
}

/// The program one case of a two-plane format in a shared placement heap of `heap` bytes runs.
fn heap_work(raw: u32, p: &Probe, heap: u64, zeros: bool) -> Work {
    let mut work = Work::new();
    let made = work.add(Resource::Heap {
        size: heap,
        storage: Storage::Shared,
    });
    let texture = work.add(Resource::HeapTexture {
        heap: made,
        spec: spec(raw, p).shared(),
        offset: 0,
    });
    let over_heap = work.add(Resource::HeapBuffer {
        heap: made,
        length: heap,
        offset: 0,
    });
    let before = work.buffer(vec![BEFORE; heap as usize]);
    let after = work.buffer(vec![0; heap as usize]);
    work.step(Step::CopyBufferToBuffer {
        from: before,
        from_offset: 0,
        to: over_heap,
        to_offset: 0,
        length: heap,
    });
    work.step(Step::Commit);
    work.step(fill_step(texture, p, zeros));
    work.step(Step::Commit);
    work.step(Step::CopyBufferToBuffer {
        from: over_heap,
        from_offset: 0,
        to: after,
        to_offset: 0,
        length: heap,
    });
    work.step(Step::Commit);
    work.step(Step::ReadBuffer {
        buffer: after,
        offset: 0,
        length: heap as usize,
    });
    work
}

/// The probe's `changed` line of a heap's bytes: the runs that are no longer the byte before.
fn heap_changed(bytes: &[u8]) -> String {
    let mut runs = Vec::new();
    let mut at = 0;
    while at < bytes.len() {
        if bytes[at] == BEFORE {
            at += 1;
            continue;
        }
        let mut end = at;
        while end < bytes.len() && bytes[end] != BEFORE {
            end += 1;
        }
        let shown: String = bytes[at..end.min(at + 64)]
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect();
        let more = if end - at > 64 {
            format!("...({})", end - at)
        } else {
            String::new()
        };
        runs.push(format!("{at}:{shown}{more}"));
        at = end;
    }
    format!("changed {} runs: {}", runs.len(), runs.join(" "))
}

/// The digest the probes took of a case's or a format's lines.
fn digest(lines: &[String]) -> String {
    Digest::sha256_prefix16(lines.join("\n").as_bytes()).hex
}

/// Which family a case belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Family {
    /// A private texture of the device's memory, or a two-plane format in a heap of `heap` bytes.
    Lane { heap: u64 },
    /// A two-plane texture of its own memory, 8 texels wide (`tfl4.swift`).
    Own,
}

/// Each lane-table case's name, in [`FORMATS`] order: a case names itself with a static string, so
/// nothing is allocated to answer [`Case::name`].
const LANE_NAMES: [&str; 50] = [
    "texture-fills-10",
    "texture-fills-12",
    "texture-fills-13",
    "texture-fills-20",
    "texture-fills-23",
    "texture-fills-25",
    "texture-fills-30",
    "texture-fills-40",
    "texture-fills-41",
    "texture-fills-42",
    "texture-fills-43",
    "texture-fills-53",
    "texture-fills-55",
    "texture-fills-60",
    "texture-fills-65",
    "texture-fills-70",
    "texture-fills-71",
    "texture-fills-80",
    "texture-fills-81",
    "texture-fills-90",
    "texture-fills-92",
    "texture-fills-93",
    "texture-fills-94",
    "texture-fills-103",
    "texture-fills-105",
    "texture-fills-110",
    "texture-fills-115",
    "texture-fills-123",
    "texture-fills-125",
    "texture-fills-130",
    "texture-fills-134",
    "texture-fills-152",
    "texture-fills-160",
    "texture-fills-162",
    "texture-fills-170",
    "texture-fills-178",
    "texture-fills-180",
    "texture-fills-187",
    "texture-fills-198",
    "texture-fills-212",
    "texture-fills-236",
    "texture-fills-240",
    "texture-fills-241",
    "texture-fills-250",
    "texture-fills-252",
    "texture-fills-253",
    "texture-fills-260",
    "texture-fills-261",
    "texture-fills-550",
    "texture-fills-551",
];

/// Each own-memory two-plane case's name, in [`OWN_TWO_PLANES`] order.
const OWN_NAMES: [&str; 2] = [
    "texture-fills-own-two-plane-550",
    "texture-fills-own-two-plane-551",
];

/// A case comparing one format's fills with Metal's digest.
pub struct TextureFills {
    name: &'static str,
    raw: u32,
    family: Family,
    metal: &'static str,
}

/// Every ported case: one per format of the lane's table, and one per format of the own-memory
/// two-plane table.
pub fn all() -> Vec<TextureFills> {
    let mut cases: Vec<TextureFills> = FORMATS
        .iter()
        .zip(LANE_NAMES)
        .map(|(&(raw, heap, metal), name)| TextureFills {
            name,
            raw,
            family: Family::Lane { heap },
            metal,
        })
        .collect();
    cases.extend(
        OWN_TWO_PLANES
            .iter()
            .zip(OWN_NAMES)
            .map(|(&(raw, metal), name)| TextureFills {
                name,
                raw,
                family: Family::Own,
                metal,
            }),
    );
    cases
}

impl TextureFills {
    /// The cases the family runs.
    fn cases(&self) -> &'static [&'static str] {
        match self.family {
            Family::Lane { .. } => &CASES,
            Family::Own => &OWN_CASES,
        }
    }

    /// Whether the family's extents are `tfl4.swift`'s, 8 texels wide.
    fn wide(&self) -> bool {
        self.family == Family::Own
    }

    /// One case's answer, as the probes record it: the digest of its read-back lines, or the word
    /// the probes wrote where the bytes do not follow the color or lie elsewhere.
    fn case_answer(&self, side: &mut dyn Side, name: &str) -> Result<String, Failure> {
        let run = |side: &mut dyn Side, zeros: bool| -> Result<Option<String>, Failure> {
            let p =
                probe(self.raw, name, self.wide()).map_err(|why| Failure::named("BadCase", why))?;
            match self.family {
                Family::Lane { heap } if heap > 0 => {
                    let done = side.work(&heap_work(self.raw, &p, heap, zeros))?;
                    // A texture of more levels or slices: its fill writes the heap's bytes at
                    // level 0's rows, which `two_plane_heap_fills.rs` checks byte for byte.
                    if p.levels > 1 || p.slices > 1 {
                        return Ok(Some("two-plane-heap-fills".to_owned()));
                    }
                    let read = done.reads.first().cloned().unwrap_or_default();
                    Ok(Some(digest(&[heap_changed(&read)])))
                }
                Family::Lane { .. } | Family::Own => {
                    // Metal makes no 4:2:2 texture of more than one level or slice.
                    if matches!(self.raw, 240 | 241) && (p.levels > 1 || p.slices > 1) {
                        return Ok(None);
                    }
                    if self.family == Family::Own && p.levels > 1 {
                        // Metal's blits read such a texture back in bytes that vary from run to
                        // run; `surface_fills.rs` checks its fills through `get_bytes`.
                        return Ok(Some("many-levels".to_owned()));
                    }
                    let done = side.work(&texture_work(self.raw, &p, zeros))?;
                    let read = done.reads.first().cloned().unwrap_or_default();
                    Ok(Some(digest(&texture_lines(self.raw, &p, &read))))
                }
            }
        };
        let answer = run(side, false)?;
        let Some(answer) = answer else {
            return Ok(String::new());
        };
        let p = probe(self.raw, name, self.wide()).map_err(|why| Failure::named("BadCase", why))?;
        let packed_in = match p.fill {
            Fill::Color(_, packed_in) => Some(packed_in.unwrap_or(self.raw)),
            _ => None,
        };
        // The rule is the probes': the format the color is packed in decides, not the texture's.
        let unpacked = packed_in.is_some_and(ignores_the_color);
        if !unpacked {
            return Ok(answer);
        }
        // A color fill packed where fills ignore the color: Metal's bytes are its stack's, and the
        // probes recorded `unpacked` where the fill wrote what a fill of one zero byte writes.
        match run(side, true)? {
            Some(zeros) if zeros == answer => Ok("unpacked".to_owned()),
            Some(zeros) => Ok(format!("unpacked-differs:{answer}/{zeros}")),
            None => Ok(answer),
        }
    }
}

/// The site a format's digest sits at.
fn site(raw: u32) -> String {
    format!("format/{raw}")
}

impl Case for TextureFills {
    fn name(&self) -> &'static str {
        self.name
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::carried_only(MEASURED_IN, NO_PUBLIC_CALL)
            .measured_on(crate::recording::APPLE_M4)
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let mut geometry = String::new();
        let mut patterns = Vec::new();
        for name in self.cases() {
            let p = probe(self.raw, name, self.wide())?;
            geometry.push_str(&format!(
                "{name} {}x{} levels {} slices {} level {} slice {} origin {:?} size {:?}\n",
                p.extent.0, p.extent.1, p.levels, p.slices, p.level, p.slice, p.origin, p.size
            ));
            match p.fill {
                Fill::Bytes(n) => patterns.extend(pattern(n)),
                Fill::Color(color, packed_in) => {
                    for channel in color {
                        patterns.extend(channel.to_le_bytes());
                    }
                    patterns.extend(packed_in.unwrap_or(self.raw).to_le_bytes());
                }
            }
        }
        Ok(vec![
            Input::bytes("case-geometry", geometry.as_bytes()),
            Input::bytes("fill-patterns", &patterns),
            Input::bytes("before", &[BEFORE]),
        ])
    }

    fn allowances(&self) -> Vec<Allowance> {
        let mut allowances = Vec::new();
        if self.raw == 260 {
            // `Depth32Float_Stencil8`'s patterns are depth outside [0, 1]. Where the device's
            // uploads announce that it may store such a value clamped, the format's bytes are not
            // Metal's to match: the lane skips the format there, and here the digest goes
            // unchecked, named, counted and reported, with that notice as the justification.
            allowances.push(Allowance::new(
                format!("case/{}", site(self.raw)),
                Loosening::Unchecked,
                Justification::Degradation(
                    metal2vulkan_runtime::Degradation::RestrictedDepthUpload
                        .code()
                        .to_owned(),
                ),
            ));
        }
        // What this work announces on a device that needs it, which is the lane's own list: a
        // fill of a color packed where fills ignore the color, a depth or block-compressed
        // texture's twin and its precision, a two-plane texture shadowed in a heap's bytes and a
        // mipmapped one whose later levels lie apart from the shadow, a float depth's denormals
        // and the combined format's clamp, and shared storage in system memory. Metal announces
        // none of them, so the notices site differs on such a device, and the difference is
        // justified where one of these fired. The format's own digest still compares, and only
        // `Depth32Float_Stencil8`'s goes unchecked, above.
        use metal2vulkan_runtime::Degradation as D;
        for kind in [
            D::UnpackedFillColor,
            D::ShadowedTexture,
            D::HeapTwoPlaneLayout,
            D::RestrictedDepthUpload,
            D::DepthTwin,
            D::DecodedBlockTwin,
            D::CarriedDecodedTwin,
            D::DecodedTexelPrecision,
            D::CompressedTexelDecode,
            D::DepthDenormals,
            D::SharedInSystemMemory,
        ] {
            allowances.push(notice_allowance(kind.code()));
        }
        allowances
    }

    fn carried_transcript(&self) -> Result<Option<(u32, Transcript)>, String> {
        let mut sites = Sites::new();
        sites.push(crate::case::inputs_node(&self.inputs()?));
        sites.leaf("work", answered());
        // The probes kept the digest of the format's lines and not the lines, so the site holds
        // the digest alone: a difference names the format and no more, which is the reach the
        // measurement has.
        sites.push(Node::leaf(
            site(self.raw),
            Observed::absent().digested(Digest {
                kind: DigestKind::Sha256Prefix16,
                hex: self.metal.to_owned(),
            }),
        ));
        sites.notices(NOTICES_SITE, &[] as &[&str]);
        Ok(Some((MEASURED_RUNS, sites.finish())))
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let mut answers = Vec::new();
        for name in self.cases() {
            match self.case_answer(side, name) {
                // A 4:2:2 texture Metal makes none of leaves no line, as the probes left none.
                Ok(answer) if answer.is_empty() => {}
                Ok(answer) => answers.push(format!("{name} {answer}")),
                Err(failure) => {
                    let at = format!("work/{}/{name}", self.raw);
                    sites.push(Node::leaf("work", Observed::value(failure.recorded())));
                    return Err(Abort { at, failure });
                }
            }
        }
        sites.leaf("work", answered());
        sites.push(Node::leaf(
            site(self.raw),
            Observed::absent().digested(Digest::sha256_prefix16(answers.join("\n").as_bytes())),
        ));
        Ok(())
    }
}

/// The two cases whose whole changed run this family measures on Metal itself.
///
/// `two_plane_case`'s formatter cut each run at 64 bytes and wrote `...({n})`, and exactly two
/// cases of the whole lane reach it: `b5` of 550 and of 551, whose texture is 7 by 5 texels, so the
/// fill's rows (16 bytes each, level 0's `4 w` rounded down to 16) overlap into **one 92-byte run**
/// and 28 of its bytes were in no comparison. Metal's bytes there had never been measured: the
/// probes printed the same truncation the lane did.
///
/// These two cases measure them. Each runs the same program the family's `b5` row runs — a 7 by 5
/// `RGB10A8_2P_XR10` (or its sRGB twin) at byte 0 of a shared placement heap of the probes' size,
/// every byte of the heap `0xEE` through a buffer over it, one `fillTexture:` of the four-byte
/// pattern, and the heap read back — but they record **every byte of the heap** and the whole run,
/// so nothing of the answer is a projection.
const HEAP_FILL_RUNS: [(&str, u32); 2] = [
    ("texture-fills-heap-run-550-b5", 550),
    ("texture-fills-heap-run-551-b5", 551),
];

/// The case name of [`HEAP_FILL_RUNS`]: the one case of the lane's twenty whose run is long enough
/// to have been elided.
const HEAP_FILL_CASE: &str = "b5";

/// The heap's bytes these cases place their texture in, as the probes sized it.
const HEAP_FILL_BYTES: u64 = 180_224;

/// The whole changed runs of a heap's bytes: one line a run, `<at>+<length> <every byte>`.
///
/// This is [`heap_changed`] without the 64-byte cut and without `...({n})`. The two are not two
/// spellings of one thing: `heap_changed` reproduces what the probes printed, which is what the
/// carried digests are of, and this prints the bytes themselves, which is what these cases record.
fn heap_run_lines(bytes: &[u8]) -> Vec<String> {
    let mut lines = Vec::new();
    let mut at = 0;
    while at < bytes.len() {
        if bytes[at] == BEFORE {
            at += 1;
            continue;
        }
        let mut end = at;
        while end < bytes.len() && bytes[end] != BEFORE {
            end += 1;
        }
        let hex: String = bytes[at..end]
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect();
        lines.push(format!("{at}+{} {hex}", end - at));
        at = end;
    }
    lines
}

/// A case that measures one two-plane heap fill on Metal itself, byte for byte.
pub struct TwoPlaneHeapFill {
    name: &'static str,
    raw: u32,
}

/// Both cases whose whole changed run this family measures on Metal itself.
pub fn heap_fill_runs() -> Vec<TwoPlaneHeapFill> {
    HEAP_FILL_RUNS
        .iter()
        .map(|&(name, raw)| TwoPlaneHeapFill { name, raw })
        .collect()
}

impl TwoPlaneHeapFill {
    /// The probe row the case runs.
    fn probe(&self) -> Result<Probe, String> {
        probe(self.raw, HEAP_FILL_CASE, false)
    }
}

impl Case for TwoPlaneHeapFill {
    fn name(&self) -> &'static str {
        self.name
    }

    /// Metal runs this. `fillTexture:` is Apple SPI rather than absent, and
    /// [`crate::metal_work`] reaches it by its selector after `respondsToSelector:`, so the answer
    /// is measured on this chip instead of carried from somebody's terminal.
    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    /// What the work announced on MoltenVK, measured: the two-plane texture is shadowed in the
    /// heap's bytes, which is the whole point of the case. Metal announces nothing, so the notices
    /// site differs and this is what names it; every byte of the heap still compares exactly.
    fn allowances(&self) -> Vec<Allowance> {
        // The second is the venue's own: the buffer this case reads the filled planes back
        // through is `Shared`, which a device with no unified memory keeps in system memory
        // because that is the only memory both sides can reach.
        [
            metal2vulkan_runtime::Degradation::ShadowedTexture,
            metal2vulkan_runtime::Degradation::SharedInSystemMemory,
        ]
        .iter()
        .map(|kind| notice_allowance(kind.code()))
        .collect()
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let p = self.probe()?;
        let Fill::Bytes(n) = p.fill else {
            return Err(format!("{} is not a pattern fill", self.name));
        };
        Ok(vec![
            Input::bytes(
                "case-geometry",
                format!(
                    "{HEAP_FILL_CASE} format {} {}x{} levels {} slices {} level {} slice {} \
                     origin {:?} size {:?} heap {HEAP_FILL_BYTES}\n",
                    self.raw,
                    p.extent.0,
                    p.extent.1,
                    p.levels,
                    p.slices,
                    p.level,
                    p.slice,
                    p.origin,
                    p.size
                )
                .as_bytes(),
            ),
            Input::bytes("fill-pattern", &pattern(n)),
            Input::bytes("before", &[BEFORE]),
        ])
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let p = self.probe().map_err(|why| Abort {
            at: "case".to_owned(),
            failure: Failure::named("BadCase", why),
        })?;
        let work = heap_work(self.raw, &p, HEAP_FILL_BYTES, false);
        let done = made(sites, "work", side.work(&work))?;
        let bytes = done.read(0).map(<[u8]>::to_vec).map_err(|why| Abort {
            at: "work".to_owned(),
            failure: Failure::named("UnansweredRead", why),
        })?;
        if bytes.len() != HEAP_FILL_BYTES as usize {
            return Err(Abort {
                at: "work".to_owned(),
                failure: Failure::named(
                    "ShortReadBack",
                    format!("{} bytes of {HEAP_FILL_BYTES}", bytes.len()),
                ),
            });
        }
        let digest = Digest::sha256(&bytes);
        sites.push(Node::leaf(
            "heap/changed",
            Observed::value(Value::Lines(heap_run_lines(&bytes))),
        ));
        sites.push(Node::leaf(
            "heap/bytes",
            Observed::bytes(bytes, ByteLayout::buffer(0, HEAP_FILL_BYTES, Element::U8))
                .digested(digest),
        ));
        Ok(())
    }
}
