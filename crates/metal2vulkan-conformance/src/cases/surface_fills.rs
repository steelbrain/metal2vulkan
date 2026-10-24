//! Metal's private texture fills over a surface's plane or a buffer: which bytes one fill changed.
//!
//! Ported from the runtime lane `runtime_lane/surface_fills.rs`'s first test, whose expectation is
//! a transcript of what our probes `surface.m` and `cases.sh` printed on this M3 Max under macOS
//! 27.0.1: for each case, the runs of bytes that are no longer `0xEE`, as ` [<at>+<n>: <bytes>]`.
//!
//! Metal's answer is **carried**, and re-recording it is **forbidden**: `fillTexture:` and
//! `fillTextureColor:` are Apple SPI, so there is no public call a Metal side could make. What the
//! case carries is our own measurement of this chip, stated in the shape the case emits.
//!
//! This port is where the typed form catches something the lane cannot. The lane compares the
//! *formatted* run string, and the formatter prints at most 24 bytes of a run and then ` ..`, so
//! every byte past the 24th of a run is never compared at all — 32 bytes of one case's answer. The
//! case compares the whole surface buffer byte for byte under a layout that names each plane's
//! rows and pitch, with [`Gaps::Compared`], so the bytes the fill must not have touched compare
//! too and a difference names the byte's row, its index and both values.
//!
//! The one row whose printed runs truncate, a color packed in `RGBA8Unorm` over the whole of an
//! 8x4 `RG10Unorm_X12_sRGB` texture on a buffer, is carried here from a **second** measurement:
//! `probes/surface_buffer_fills.m`, which calls the same SPI from Objective-C and prints every
//! byte of every run. That probe is committed, so the bytes it found can be taken again rather
//! than measured once and thrown away, which is how the gap arose. It re-measures the other three
//! buffer rows of the table as its own control, and reproduced their carried bytes exactly.
//!
//! Nothing here can fault a GPU: every fill is of a small region, and no pattern is longer than
//! four bytes (a texture-buffer fill pattern over 32752 bytes, which ends Metal's process, is a
//! case this round must not run).

use crate::allowance::{Allowance, Justification, Loosening};
use crate::case::{Abort, Case, Fill, Input, MetalSource, PlaneTexture, Side};
use crate::recording::ShaderRef;
use crate::replay::NOTICES_SITE;
use crate::transcript::{
    ByteLayout, Element, Gaps, Node, Observed, Region, Sites, Transcript, Value,
};
use metal2vulkan_runtime::Degradation;

/// The case's name.
pub const SURFACE_FILLS: &str = "surface-fills-over-planes-and-buffers";

/// Where the carried bytes were measured.
const MEASURED_IN: &str =
    "the runtime lane's surface-fill table, from the probes surface.m and cases.sh, and for the \
     RG10Unorm_X12_sRGB buffer row from probes/surface_buffer_fills.m (an Apple M3 Max, macOS \
     27.0.1)";

/// The runs the carried table stands on: the weakest row's, the lane probe's own single pass over
/// the cases. (`probes/surface_buffer_fills.m` took the `579 p70` row over three passes, which
/// printed the same bytes each time; the table's number is the one every row holds.)
pub const MEASURED_RUNS: u32 = 1;

/// Why record mode must not run the case on Metal.
const NO_PUBLIC_CALL: &str =
    "fillTexture: and fillTextureColor: are Apple SPI, so no public call exists to record them \
     from";

/// The byte every surface and buffer starts as, so a fill's bytes are the ones that are no longer
/// it.
const BEFORE: u8 = 0xee;

/// The pattern bytes a pattern fill takes, as the probe built them: `01 02 03 ..`, cut to the
/// case's length.
fn pattern(bytes: usize) -> Vec<u8> {
    (1..=bytes as u8).collect()
}

/// A surface's layout as CoreVideo made it: its length and its two planes' first byte and
/// `bytesPerRow` (the lane's `S8X4` and `S8X4_444`).
#[derive(Clone, Copy, Debug)]
struct Surface {
    length: u64,
    planes: [(u64, u64); 2],
}

/// The 4:2:0 and 4:2:2 surfaces' layout.
const S8X4: Surface = Surface {
    length: 1344,
    planes: [(512, 64), (1024, 64)],
};

/// The 4:4:4 surfaces' layout.
const S8X4_444: Surface = Surface {
    length: 1600,
    planes: [(512, 64), (1024, 64)],
};

/// One row of the case: the probe's name for it, the texture it fills, the fill, the buffer's
/// length, and the runs of changed bytes the probe printed.
struct Row {
    /// The probe's case word, which with the format names the row.
    case: &'static str,
    over: PlaneTextureOf,
    fill: FillOf,
    /// The buffer's length in bytes.
    length: u64,
    /// The runs Metal changed, as the probe printed them.
    metal: &'static str,
}

/// How a row's texture is made.
#[derive(Clone, Copy, Debug)]
enum PlaneTextureOf {
    /// Over `plane` of a surface of this OSType and layout.
    Surface {
        ostype: &'static [u8; 4],
        format: u32,
        size: [u32; 2],
        plane: u32,
        surface: Surface,
    },
    /// Over the buffer itself, at byte 0, with this `bytesPerRow`.
    Buffer {
        format: u32,
        size: [u32; 2],
        bytes_per_row: u64,
    },
}

/// A row's fill, as the probe's case word names it.
#[derive(Clone, Copy, Debug)]
enum FillOf {
    /// The whole texture, a pattern of this many bytes (the probe's `b<n>`).
    Whole(usize),
    /// The region at 1, 1 of 2 by 1, a pattern of this many bytes (the probe's `r<n>`).
    Small(usize),
    /// The region at `x`, `y` of 3 by 2, a pattern of this many bytes (the probe's `x<x>,<y>,<n>`).
    At(u32, u32, usize),
    /// The whole texture, the color (0.25, 0.5, 0.75, 1) packed in this raw `MTLPixelFormat` (the
    /// probe's `p<pf>`, whose number is decimal).
    Color(u32),
}

/// The color every color fill packs, as the probe asked for it.
const COLOR: [f64; 4] = [0.25, 0.5, 0.75, 1.0];

const fn surf(
    ostype: &'static [u8; 4],
    size: [u32; 2],
    (format, plane): (u32, u32),
    surface: Surface,
    case: &'static str,
    fill: FillOf,
    metal: &'static str,
) -> Row {
    Row {
        case,
        over: PlaneTextureOf::Surface {
            ostype,
            format,
            size,
            plane,
            surface,
        },
        fill,
        length: surface.length,
        metal,
    }
}

const fn buf(
    format: u32,
    size: [u32; 2],
    bytes_per_row: u64,
    case: &'static str,
    fill: FillOf,
    metal: &'static str,
) -> Row {
    Row {
        case,
        over: PlaneTextureOf::Buffer {
            format,
            size,
            bytes_per_row,
        },
        fill,
        length: bytes_per_row * size[1] as u64 + 256,
        metal,
    }
}

/// Metal's fills over a surface's plane or a buffer, as the probes measured them: the lane's
/// `OVER`, whole, its last row's 32-byte runs from `probes/surface_buffer_fills.m`.
const ROWS: &[Row] = &[
    surf(b"420v", [8, 4], (500, 0), S8X4, "b4", FillOf::Whole(4),
        " [0x200+8: 01 01 01 01 01 01 01 01] [0x240+8: 01 01 01 01 01 01 01 01] [0x280+8: 01 01 01 01 01 01 01 01] [0x2c0+8: 01 01 01 01 01 01 01 01]"),
    surf(b"420v", [8, 4], (500, 0), S8X4, "x5,2,3", FillOf::At(5, 2, 3),
        " [0x285+3: 01 01 01] [0x2c5+3: 01 01 01]"),
    surf(b"420v", [8, 4], (520, 0), S8X4, "p70", FillOf::Color(70),
        " [0x200+8: 40 40 40 40 40 40 40 40] [0x240+8: 40 40 40 40 40 40 40 40] [0x280+8: 40 40 40 40 40 40 40 40] [0x2c0+8: 40 40 40 40 40 40 40 40]"),
    surf(b"x420", [8, 4], (505, 0), S8X4, "b4", FillOf::Whole(4),
        " [0x200+16: 01 02 01 02 01 02 01 02 01 02 01 02 01 02 01 02] [0x240+16: 01 02 01 02 01 02 01 02 01 02 01 02 01 02 01 02] [0x280+16: 01 02 01 02 01 02 01 02 01 02 01 02 01 02 01 02] [0x2c0+16: 01 02 01 02 01 02 01 02 01 02 01 02 01 02 01 02]"),
    surf(b"x420", [8, 4], (505, 0), S8X4, "x5,2,3", FillOf::At(5, 2, 3),
        " [0x28a+6: 01 02 01 02 01 02] [0x2ca+6: 01 02 01 02 01 02]"),
    surf(b"x420", [8, 4], (505, 0), S8X4, "p70", FillOf::Color(70),
        " [0x200+16: 40 80 40 80 40 80 40 80 40 80 40 80 40 80 40 80] [0x240+16: 40 80 40 80 40 80 40 80 40 80 40 80 40 80 40 80] [0x280+16: 40 80 40 80 40 80 40 80 40 80 40 80 40 80 40 80] [0x2c0+16: 40 80 40 80 40 80 40 80 40 80 40 80 40 80 40 80]"),
    surf(b"x420", [8, 4], (543, 0), S8X4, "p20", FillOf::Color(20),
        " [0x200+16: 00 40 00 40 00 40 00 40 00 40 00 40 00 40 00 40] [0x240+16: 00 40 00 40 00 40 00 40 00 40 00 40 00 40 00 40] [0x280+16: 00 40 00 40 00 40 00 40 00 40 00 40 00 40 00 40] [0x2c0+16: 00 40 00 40 00 40 00 40 00 40 00 40 00 40 00 40]"),
    surf(b"x420", [38, 6], (505, 0),
        Surface { length: 2128, planes: [(512, 128), (1536, 128)] }, "x35,4,3", FillOf::At(35, 4, 3),
        " [0x446+6: 01 02 01 02 01 02] [0x4c6+6: 01 02 01 02 01 02]"),
    surf(b"xf20", [8, 4], (525, 0), S8X4, "x6,3,3", FillOf::At(6, 3, 3),
        " [0x2cc+4: 01 02 01 02]"),
    surf(b"444v", [8, 4], (503, 1), S8X4_444, "b4", FillOf::Whole(4),
        " [0x400+8: 01 01 01 01 01 01 01 01] [0x440+8: 01 01 01 01 01 01 01 01] [0x480+8: 01 01 01 01 01 01 01 01] [0x4c0+8: 01 01 01 01 01 01 01 01]"),
    surf(b"x444", [8, 4], (507, 1), S8X4_444, "x1,1,3", FillOf::At(1, 1, 3),
        " [0x442+6: 01 02 01 02 01 02] [0x482+6: 01 02 01 02 01 02]"),
    surf(b"x422", [8, 4], (544, 0), S8X4_444, "b3", FillOf::Whole(3),
        " [0x200+16: 01 02 01 02 01 02 01 02 01 02 01 02 01 02 01 02] [0x240+16: 01 02 01 02 01 02 01 02 01 02 01 02 01 02 01 02] [0x280+16: 01 02 01 02 01 02 01 02 01 02 01 02 01 02 01 02] [0x2c0+16: 01 02 01 02 01 02 01 02 01 02 01 02 01 02 01 02]"),
    surf(b"p420", [12, 4], (508, 0), S8X4, "x5,2,1", FillOf::At(5, 2, 1),
        " [0x285+3: 00 00 00] [0x2c5+3: 00 00 00]"),
    surf(b"p444", [12, 4], (510, 0), S8X4_444, "p10", FillOf::Color(10),
        " [0x200+12: 00 00 00 00 00 00 00 00 00 00 00 00] [0x240+12: 00 00 00 00 00 00 00 00 00 00 00 00] [0x280+12: 00 00 00 00 00 00 00 00 00 00 00 00] [0x2c0+12: 00 00 00 00 00 00 00 00 00 00 00 00]"),
    surf(b"p422", [8, 4], (529, 0), S8X4_444, "b4", FillOf::Whole(4),
        " [0x200+8: 00 00 00 00 00 00 00 00] [0x240+8: 00 00 00 00 00 00 00 00] [0x280+8: 00 00 00 00 00 00 00 00] [0x2c0+8: 00 00 00 00 00 00 00 00]"),
    buf(576, [8, 4], 64, "x5,2,3", FillOf::At(5, 2, 3),
        " [0x8a+6: 01 02 01 02 01 02] [0xca+6: 01 02 01 02 01 02]"),
    buf(577, [7, 3], 32, "b4", FillOf::Whole(4),
        " [0+14: 01 02 01 02 01 02 01 02 01 02 01 02 01 02] [0x20+14: 01 02 01 02 01 02 01 02 01 02 01 02 01 02] [0x40+14: 01 02 01 02 01 02 01 02 01 02 01 02 01 02]"),
    buf(578, [8, 4], 64, "r3", FillOf::Small(3), " [0x44+8: 01 02 03 01 01 02 03 01]"),
    // The row the lane's formatter truncated at 24 of each run's 32 bytes, measured whole by
    // `probes/surface_buffer_fills.m`: a 4-byte texel over eight texels a row, so each row's 32
    // bytes are the packed color eight times. The last eight bytes of each run are the ones no
    // passing test had ever compared.
    buf(579, [8, 4], 64, "p70", FillOf::Color(70),
        " [0+32: 40 80 bf ff 40 80 bf ff 40 80 bf ff 40 80 bf ff 40 80 bf ff 40 80 bf ff 40 80 bf ff 40 80 bf ff] [0x40+32: 40 80 bf ff 40 80 bf ff 40 80 bf ff 40 80 bf ff 40 80 bf ff 40 80 bf ff 40 80 bf ff 40 80 bf ff] [0x80+32: 40 80 bf ff 40 80 bf ff 40 80 bf ff 40 80 bf ff 40 80 bf ff 40 80 bf ff 40 80 bf ff 40 80 bf ff] [0xc0+32: 40 80 bf ff 40 80 bf ff 40 80 bf ff 40 80 bf ff 40 80 bf ff 40 80 bf ff 40 80 bf ff 40 80 bf ff]"),
];

/// The calls a row makes that answer an object or nothing of their own, by the site each is
/// emitted at under the row: both legs emit these, so the carried answer is the shape a run
/// produces and a site either side forgets fails loudly.
const MADE: [&str; 3] = ["buffer", "texture", "read-back"];

/// A case that compares the bytes one private texture fill changed with Metal's.
pub struct SurfaceFills {
    name: &'static str,
}

/// The case.
pub fn over_planes_and_buffers() -> SurfaceFills {
    SurfaceFills {
        name: SURFACE_FILLS,
    }
}

impl Row {
    /// The row's site name: its format and the probe's case word, as the lane printed a
    /// difference.
    fn site(&self) -> String {
        let format = match self.over {
            PlaneTextureOf::Surface { format, .. } | PlaneTextureOf::Buffer { format, .. } => {
                format
            }
        };
        format!("{format}/{}", self.case)
    }

    /// The texture the row makes.
    fn texture(&self) -> PlaneTexture {
        match self.over {
            PlaneTextureOf::Surface {
                ostype,
                format,
                size,
                plane,
                surface,
            } => PlaneTexture {
                ostype: Some(*ostype),
                format,
                size,
                plane,
                planes: surface.planes,
            },
            PlaneTextureOf::Buffer {
                format,
                size,
                bytes_per_row,
            } => PlaneTexture {
                ostype: None,
                format,
                size,
                plane: 0,
                planes: [(0, bytes_per_row), (0, 0)],
            },
        }
    }

    /// The fill the row encodes.
    fn fill(&self) -> Fill {
        let [width, height] = match self.over {
            PlaneTextureOf::Surface { size, .. } | PlaneTextureOf::Buffer { size, .. } => size,
        };
        match self.fill {
            FillOf::Whole(bytes) => Fill::Pattern {
                level: 0,
                slice: 0,
                region: [0, 0, width, height],
                pattern: pattern(bytes),
            },
            FillOf::Small(bytes) => Fill::Pattern {
                level: 0,
                slice: 0,
                region: [1, 1, 2, 1],
                pattern: pattern(bytes),
            },
            FillOf::At(x, y, bytes) => Fill::Pattern {
                level: 0,
                slice: 0,
                region: [x, y, 3, 2],
                pattern: pattern(bytes),
            },
            FillOf::Color(packed_into) => Fill::Color {
                level: 0,
                slice: 0,
                region: [0, 0, width, height],
                color: COLOR,
                packed_into,
            },
        }
    }

    /// The layout the row's buffer is read under: every byte of it, so the bytes the fill must not
    /// have touched compare as well as the ones it wrote.
    fn layout(&self) -> ByteLayout {
        let length = u32::try_from(self.length).unwrap_or(u32::MAX);
        ByteLayout {
            region: Region::Buffer {
                offset: 0,
                length: self.length,
            },
            element: Element::U8,
            row_offset: 0,
            row_elements: length,
            row_pitch: length,
            rows: 1,
            image_pitch: length,
            images: 1,
            gaps: Gaps::Compared,
        }
    }

    /// The bytes Metal left: `0xEE` everywhere but where the row's printed runs say otherwise.
    fn metal_bytes(&self) -> Result<Vec<u8>, String> {
        let mut bytes = vec![BEFORE; self.length as usize];
        for run in parse_runs(self.metal)? {
            let (at, written) = run;
            let end = at + written.len();
            if end > bytes.len() {
                return Err(format!(
                    "{}: a run of {} bytes at {at} is past the buffer's {}",
                    self.site(),
                    written.len(),
                    bytes.len()
                ));
            }
            bytes[at..end].copy_from_slice(&written);
        }
        Ok(bytes)
    }
}

/// The runs a printed answer states: each ` [<at>+<n>: <bytes>]`, as the byte offset and the bytes
/// written there.
///
/// A run whose printed bytes are fewer than its length is **rejected**, not guessed: the formatter
/// that wrote it stopped at 24 bytes, and a carried answer that filled the rest in would be a
/// measurement nobody took.
fn parse_runs(printed: &str) -> Result<Vec<(usize, Vec<u8>)>, String> {
    let mut runs = Vec::new();
    for piece in printed.split(" [").skip(1) {
        let piece = piece.trim_end();
        let body = piece
            .strip_suffix(']')
            .ok_or_else(|| format!("a run with no closing bracket: {piece}"))?;
        let (head, listed) = body
            .split_once(": ")
            .ok_or_else(|| format!("a run with no bytes: {body}"))?;
        let (at, length) = head
            .split_once('+')
            .ok_or_else(|| format!("a run with no length: {head}"))?;
        let at = parse_offset(at)?;
        let length: usize = length
            .parse()
            .map_err(|_| format!("a run of no length: {length}"))?;
        let mut written = Vec::with_capacity(length);
        for word in listed.split(' ') {
            if word == ".." {
                return Err(format!(
                    "the run at {at} is printed truncated, so Metal's bytes past the {}th were \
                     never written down",
                    written.len()
                ));
            }
            written.push(u8::from_str_radix(word, 16).map_err(|_| format!("not a byte: {word}"))?);
        }
        if written.len() != length {
            return Err(format!(
                "the run at {at} says {length} bytes and prints {}",
                written.len()
            ));
        }
        runs.push((at, written));
    }
    Ok(runs)
}

/// A run's offset, as the probe printed it: `0`, or hexadecimal with a `0x` prefix.
fn parse_offset(printed: &str) -> Result<usize, String> {
    match printed.strip_prefix("0x") {
        Some(hex) => usize::from_str_radix(hex, 16),
        None => printed.parse(),
    }
    .map_err(|_| format!("not an offset: {printed}"))
}

impl SurfaceFills {
    /// The sites one row emits, with `bytes` the buffer after the fill and `status` what the fill
    /// answered: the shape both the carried answer and a run are built through.
    fn row_node(row: &Row, bytes: Vec<u8>, status: Value) -> Node {
        Node::branch(row.site(), Vec::new())
            .with(Node::leaf(row.fill().call(), Observed::value(status)))
            .with(Node::leaf("bytes", Observed::bytes(bytes, row.layout())))
    }

    /// The transcript the carried measurement states, in the shape a run emits: the calls that
    /// made each row's buffer and texture and read it back, which answer an object and so record
    /// only that they answered, then the row's own node.
    pub fn measured_transcript(&self) -> Result<Transcript, String> {
        let mut sites = Sites::new();
        sites.push(crate::case::inputs_node(&self.inputs()?));
        for row in ROWS {
            let at = row.site();
            for call in MADE {
                sites.value(format!("{at}/{call}"), Value::ok(Value::Absent));
            }
            sites.push(Self::row_node(
                row,
                row.metal_bytes()?,
                Value::ok(Value::unit_variant("Completed")),
            ));
        }
        sites.notices(NOTICES_SITE, &[] as &[&str]);
        Ok(sites.finish())
    }
}

impl Case for SurfaceFills {
    fn name(&self) -> &'static str {
        self.name
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::carried_only(MEASURED_IN, NO_PUBLIC_CALL)
    }

    fn allowances(&self) -> Vec<Allowance> {
        // Metal announces nothing, so the notices site is the one field that may differ. A texture
        // over a buffer's padded rows is kept as an exact shadow copy, which the device announces;
        // a color packed in a format Metal packs no color in fills the command buffer's last
        // packed texel, which it announces too. Each names its own kind, and only the one that
        // fired on a run justifies that run's difference.
        [
            Degradation::ShadowedTexture,
            Degradation::ShadowedTextureView,
            Degradation::UnnamedShadowedTexture,
            Degradation::UnpackedFillColor,
            // The buffer the case reads the filled surfaces back through is `Shared`, which a
            // device with no unified memory keeps in system memory because that is the only
            // memory both sides can reach: the venue's memory model, not a fill.
            Degradation::SharedInSystemMemory,
        ]
        .into_iter()
        .map(|kind| {
            Allowance::new(
                format!("case/{NOTICES_SITE}/*"),
                Loosening::Unchecked,
                Justification::notice(kind),
            )
        })
        .collect()
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        Ok(Vec::new())
    }

    fn carried_transcript(&self) -> Result<Option<(u32, Transcript)>, String> {
        Ok(Some((MEASURED_RUNS, self.measured_transcript()?)))
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        // What the case feeds: every byte of every buffer before its fill (all `0xEE`), and each
        // row's texture and fill.
        let mut rows = Vec::new();
        for row in ROWS {
            rows.extend(row.site().into_bytes());
            rows.push(b'=');
            rows.extend(format!("{:?}", row.texture()).into_bytes());
            rows.extend(row.fill().words());
            rows.extend(row.length.to_le_bytes());
            rows.push(b'\n');
        }
        Ok(vec![
            Input::bytes("buffers-before", &[BEFORE]),
            Input::bytes("rows", &rows),
        ])
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        for row in ROWS {
            let at = row.site();
            let before = vec![BEFORE; row.length as usize];
            let buffer = crate::case::made(sites, &format!("{at}/buffer"), side.buffer(&before))?;
            let texture = crate::case::made(
                sites,
                &format!("{at}/texture"),
                side.plane_texture(buffer, &row.texture()),
            )?;
            let fill = row.fill();
            let status = match side.fill_texture(texture, &fill) {
                Ok(status) => Value::ok(status),
                Err(failure) => {
                    // The fill's own answer is the row's first field, so a refusal is emitted
                    // there and stops the run: the bytes after a refused fill are not the row's
                    // subject.
                    sites.push(
                        Node::branch(at.clone(), Vec::new())
                            .with(Node::leaf(fill.call(), Observed::value(failure.recorded()))),
                    );
                    return Err(Abort {
                        at: format!("{at}/{}", fill.call()),
                        failure,
                    });
                }
            };
            let after = crate::case::call(
                sites,
                &format!("{at}/read-back"),
                side.read_buffer(buffer, 0, row.length as usize),
                |_| Value::Absent,
            )?;
            sites.push(Self::row_node(row, after, status));
        }
        Ok(())
    }
}
