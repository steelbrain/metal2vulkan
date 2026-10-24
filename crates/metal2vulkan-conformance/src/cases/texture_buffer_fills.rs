//! Metal's private fills of a texture over a buffer, ported from
//! `runtime_lane::texture_buffer_fills`.
//!
//! The probes (`tb.m` and `tblf.m`, an Apple M3 Max under macOS 27.0.1): a texture buffer 12
//! texels wide of a format of 1 to 16 bytes a texel, at an offset into a shared buffer of 512
//! bytes whose byte `i` is `i * 37 mod 251`, rows 256 bytes apart; one or more fills in one
//! encoder; every byte of the buffer read back. The long-pattern probe fills whole texture buffers
//! up to 65,536 texels wide with patterns of 4,097 to 32,752 bytes.
//!
//! Three cases, because the Metal measurement has three fidelities and a case must not claim more
//! than its recording holds:
//!
//! - `texture-buffer-fill-streams` and `texture-buffer-fill-groups` compare **every byte**. The
//!   lane's committed runs (`fixtures/public/texture_buffer_fills/metal_runs.txt`, a line a case)
//!   name the bytes each fill changed, which over the probe's known first bytes reconstruct the
//!   whole 512-byte buffer. The site's layout is the buffer's two 256-byte rows, so a difference
//!   names the row and the byte inside it rather than a flat offset.
//! - `texture-buffer-long-fills` compares the **digest** the probe printed, because that is all
//!   `tblf.m` kept of buffers up to 262 KiB. The case says so at the site: a digest-only recording
//!   compares by its digest and the suite reports it as such, so the row never reads as exact over
//!   bytes nobody wrote down.
//!
//! Metal's answer is **carried**: Metal fills a texture through private API, so the Metal side has
//! no public call to record it from. Re-recording is forbidden for that reason, as for the
//! indirect-command seed case.
//!
//! **Not ported, and why**: `tblf.m`'s pattern one byte past Metal's longest (32,752) ends Metal's
//! process in the call. The runtime refuses it (`Problem::LongTextureBufferFillPattern`), and the
//! lane checks that refusal, but the suite has no outcome kind for "Metal ends the process" — the
//! plan leaves that unbuilt on purpose — so the row stays the lane's. Nothing here can fault or
//! hang a GPU: every fill is inside Metal's measured bounds.

use crate::case::{call, Abort, Case, Input, MetalSource, Side};
use crate::measured::{public_fixture, stripes};
use crate::replay::NOTICES_SITE;
use crate::transcript::{
    ByteLayout, Digest, DigestKind, Element, Gaps, Node, Observed, Region, Sites, Transcript, Value,
};
use crate::work::{Rect, Resource, Step, Storage, TextureKind, TextureSpec, Work};

/// Where the carried values were measured.
const MEASURED_IN: &str =
    "runtime_lane::texture_buffer_fills, from the probes tb.m and tblf.m (an Apple M3 Max, \
     macOS 27.0.1)";

/// Why record mode must not run these cases on Metal.
const NO_PUBLIC_CALL: &str =
    "Metal fills a texture through private API alone, so no public call exists to record it from";

/// The runs the carried tables stand on: the probes ran each case once.
const MEASURED_RUNS: u32 = 1;

/// The buffer's bytes, and the bytes of one of its rows: the texture's rows are 256 bytes apart,
/// so the buffer holds two.
const BYTES: usize = 512;
const ROW: u32 = 256;

/// The committed runs of bytes each stream and group case's fills changed, a line a case.
const METAL_RUNS: &str = "texture_buffer_fills/metal_runs.txt";

/// The longest pattern a texture buffer's fill takes, Metal's. One byte more ends Metal's process,
/// which is why no case here goes past it.
const LONGEST: usize = 32752;

/// One row of a case: how it is named, the format's raw value, the texture's offset into the
/// buffer, and the fills, as the probe's arguments.
struct Row {
    key: String,
    format: u32,
    offset: u64,
    fills: &'static [&'static str],
}

/// The streams, as the probe ran them: a name, the format, the offset and the fills.
const STREAMS: [(&str, u32, u64, &[&str]); 26] = [
    ("r8-8-whole", 10, 0, &["0,12:p8"]),
    ("r8-8-odd", 10, 4, &["3,5:p8"]),
    ("rgba8-8-whole", 70, 0, &["0,12:p8"]),
    ("rgba8-8-x4", 70, 64, &["4,8:p8"]),
    ("rgba8-c115", 70, 0, &["4,6:c115"]),
    ("r8-past-width", 10, 0, &["12,3:p1"]),
    ("rgba8-ending-past", 70, 0, &["10,5:p4"]),
    ("r8-no-rows", 10, 0, &["1,2,0,0,1:p1"]),
    ("rgba8-no-images", 70, 64, &["1,2,0,1,0:p4"]),
    ("rgba8-3-whole", 70, 0, &["0,12:p3"]),
    ("rgba8-9-after-8", 70, 0, &["0,12:p8", "0,12:p9@b1"]),
    ("rgba8-12-x1", 70, 4, &["1,10:p12"]),
    ("rgba32-17", 125, 0, &["0,12:p17"]),
    ("rgba32-24-x1", 125, 64, &["1,10:p24"]),
    ("rgba8-33-past", 70, 0, &["0,40:p33"]),
    ("r8-65-long", 10, 0, &["3,300:p65"]),
    ("rgba8-100", 70, 4, &["1,100:p100"]),
    ("rgba8-256", 70, 0, &["0,100:p256"]),
    ("rg8-5-x6", 30, 0, &["6,40:p5"]),
    ("rgba16-24", 115, 64, &["3,30:p24"]),
    ("rgba32-past-end", 125, 64, &["20,20:p16"]),
    ("r8-past-buffer", 10, 0, &["600,3:p1"]),
    ("rgba8-width-0", 70, 0, &["2,0:p3"]),
    ("rgba8-level-slice-z", 70, 4, &["2,4,5,3,2,3,7,4:p3"]),
    ("rgba8-c115-then-12", 70, 0, &["0,12:c115", "0,12:p12"]),
    ("r8-3-odd", 10, 4, &["1,30:p3"]),
];

/// The fills of a group of `r8` texels, one case each.
const R8: &[&[&str]] = &[
    &["0,12:p1"],
    &["3,5:p1"],
    &["7,5:p1"],
    &["2,4,5,3,2,3,7,4:p1"],
    &["1,6,4,2,1,1,2,0:c10"],
    &["0,12:p1", "5,0:p1", "12,0:p1", "9,2:c10"],
];

/// The fills of a group of `rg8` texels, one case each.
const RG8: &[&[&str]] = &[
    &["0,12:p1"],
    &["3,5:p1"],
    &["7,5:p1"],
    &["2,4,5,3,2,3,7,4:p1"],
    &["0,12:p2"],
    &["3,5:p2"],
    &["7,5:p2"],
    &["2,4,5,3,2,3,7,4:p2"],
    &["1,6,4,2,1,1,2,0:c10"],
    &["1,6,4,2,1,1,2,0:c30"],
    &["0,12:p1", "5,0:p1", "12,0:p1", "9,2:c10"],
];

/// The fills of a group of `rgba8` texels, one case each.
const RGBA8: &[&[&str]] = &[
    &["0,12:p1"],
    &["3,5:p1"],
    &["7,5:p1"],
    &["2,4,5,3,2,3,7,4:p1"],
    &["0,12:p2"],
    &["3,5:p2"],
    &["7,5:p2"],
    &["2,4,5,3,2,3,7,4:p2"],
    &["0,12:p4"],
    &["3,5:p4"],
    &["7,5:p4"],
    &["2,4,5,3,2,3,7,4:p4"],
    &["1,6,4,2,1,1,2,0:c10"],
    &["1,6,4,2,1,1,2,0:c30"],
    &["1,6,4,2,1,1,2,0:c70"],
    &["0,12:p1", "5,0:p1", "12,0:p1", "9,2:c10"],
];

/// The fills of a group of `rgba16` texels, one case each.
const RGBA16: &[&[&str]] = &[
    &["0,12:p1"],
    &["3,5:p1"],
    &["7,5:p1"],
    &["2,4,5,3,2,3,7,4:p1"],
    &["0,12:p2"],
    &["3,5:p2"],
    &["7,5:p2"],
    &["2,4,5,3,2,3,7,4:p2"],
    &["0,12:p4"],
    &["3,5:p4"],
    &["7,5:p4"],
    &["2,4,5,3,2,3,7,4:p4"],
    &["0,12:p8"],
    &["3,5:p8"],
    &["7,5:p8"],
    &["2,4,5,3,2,3,7,4:p8"],
    &["1,6,4,2,1,1,2,0:c10"],
    &["1,6,4,2,1,1,2,0:c30"],
    &["1,6,4,2,1,1,2,0:c70"],
    &["1,6,4,2,1,1,2,0:c115"],
    &["0,12:p1", "5,0:p1", "12,0:p1", "9,2:c10"],
];

/// The fills of a group of `rgba32` texels, one case each.
const RGBA32: &[&[&str]] = &[
    &["0,12:p1"],
    &["3,5:p1"],
    &["7,5:p1"],
    &["2,4,5,3,2,3,7,4:p1"],
    &["0,12:p2"],
    &["3,5:p2"],
    &["7,5:p2"],
    &["2,4,5,3,2,3,7,4:p2"],
    &["0,12:p4"],
    &["3,5:p4"],
    &["7,5:p4"],
    &["2,4,5,3,2,3,7,4:p4"],
    &["0,12:p8"],
    &["3,5:p8"],
    &["7,5:p8"],
    &["2,4,5,3,2,3,7,4:p8"],
    &["0,12:p16"],
    &["3,5:p16"],
    &["7,5:p16"],
    &["2,4,5,3,2,3,7,4:p16"],
    &["1,6,4,2,1,1,2,0:c10"],
    &["1,6,4,2,1,1,2,0:c30"],
    &["1,6,4,2,1,1,2,0:c70"],
    &["1,6,4,2,1,1,2,0:c115"],
    &["1,6,4,2,1,1,2,0:c125"],
    &["0,12:p1", "5,0:p1", "12,0:p1", "9,2:c10"],
];

/// The groups, as the probe ran them: a name, the format, the offset and the group's cases.
const GROUPS: [(&str, u32, u64, &[&[&str]]); 16] = [
    ("r8-at-0", 10, 0, R8),
    ("r8-at-4", 10, 4, R8),
    ("r8-at-64", 10, 64, R8),
    ("r8-at-68", 10, 68, R8),
    ("rg8-at-0", 30, 0, RG8),
    ("rg8-at-4", 30, 4, RG8),
    ("rg8-at-64", 30, 64, RG8),
    ("rg8-at-68", 30, 68, RG8),
    ("rgba8-at-0", 70, 0, RGBA8),
    ("rgba8-at-4", 70, 4, RGBA8),
    ("rgba8-at-64", 70, 64, RGBA8),
    ("rgba8-at-68", 70, 68, RGBA8),
    ("rgba16-at-0", 115, 0, RGBA16),
    ("rgba16-at-64", 115, 64, RGBA16),
    ("rgba32-at-0", 125, 0, RGBA32),
    ("rgba32-at-64", 125, 64, RGBA32),
];

/// The long patterns' fills (`tblf.m`): the format's raw value, its texel's bytes, the texture's
/// offset and width, the pattern's bytes, and the digest the probe printed of the buffer.
const LONG: [(u32, u64, u64, u64, usize, &str); 6] = [
    (13, 1, 4, 400, LONGEST, "fb41732d0bf8d61d"),
    (13, 1, 4, 12331, 4097, "270ffc1b247b7363"),
    (13, 1, 4, 49195, 16385, "b9e0af569f1c7db6"),
    (13, 1, 4, 65536, LONGEST, "4a510282aef3fc37"),
    (53, 4, 16, 16384, 30000, "154908c746d80c08"),
    (123, 16, 64, 4096, 5000, "8d4a10e63b972ade"),
];

/// The rows of the streams case.
fn stream_rows() -> Vec<Row> {
    STREAMS
        .iter()
        .map(|&(name, format, offset, fills)| Row {
            key: name.to_owned(),
            format,
            offset,
            fills,
        })
        .collect()
}

/// The rows of the groups case: a group's case `k` is named `<group>#<k>`, as the measurement
/// names it.
fn group_rows() -> Vec<Row> {
    let mut rows = Vec::new();
    for &(name, format, offset, cases) in &GROUPS {
        for (k, fills) in cases.iter().enumerate() {
            rows.push(Row {
                key: format!("{name}#{k}"),
                format,
                offset,
                fills,
            });
        }
    }
    rows
}

/// The `n` bytes of `tblf.m`'s pattern, a xorshift stream.
fn probe_stream(n: usize) -> Vec<u8> {
    let mut state: u64 = 0x9e37_79b9_7f4a_7c15;
    (0..n)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 24) as u8 | 1
        })
        .collect()
}

/// One fill as the probe's argument states it: where it writes, and what.
struct Fill {
    region: Rect,
    level: u64,
    slice: u64,
    /// The pattern's bytes, or the format a color is packed in.
    what: Result<Vec<u8>, Option<u32>>,
}

/// One fill, as the probe's argument `x,w[,y,h,d,level,slice,z]:<pattern>` states it: the region
/// and either the pattern's bytes or the color's packed format.
fn parse_fill(spec: &str) -> Result<Fill, String> {
    let (region, pattern) = spec
        .split_once(':')
        .ok_or_else(|| format!("{spec}: no pattern"))?;
    let mut axes = [0u64, 0, 0, 1, 1, 0, 0, 0];
    for (at, number) in region.split(',').enumerate() {
        let value: u64 = number
            .parse()
            .map_err(|error| format!("{spec}: {number}: {error}"))?;
        *axes
            .get_mut(at)
            .ok_or_else(|| format!("{spec}: too many numbers"))? = value;
    }
    let [x, w, y, h, depth, level, slice, z] = axes;
    let rect = Rect::new_3d([x, y, z], [w, h, depth]);
    if let Some(packed) = pattern.strip_prefix('c') {
        let format: u32 = packed
            .parse()
            .map_err(|error| format!("{spec}: {packed}: {error}"))?;
        return Ok(Fill {
            region: rect,
            level,
            slice,
            what: Err(Some(format)),
        });
    }
    let counted = pattern
        .strip_prefix('p')
        .ok_or_else(|| format!("{spec}: neither a pattern nor a color"))?;
    let (count, from) = counted.split_once('@').unwrap_or((counted, "a1"));
    let count: usize = count
        .parse()
        .map_err(|error| format!("{spec}: {count}: {error}"))?;
    let from = u8::from_str_radix(from, 16).map_err(|error| format!("{spec}: {from}: {error}"))?;
    let bytes = (0..count).map(|i| from.wrapping_add(i as u8)).collect();
    Ok(Fill {
        region: rect,
        level,
        slice,
        what: Ok(bytes),
    })
}

/// The color the probe's `c<pf>` fills.
const COLOR: [f64; 4] = [0.25, -0.5, 1.25, 0.75];

/// The program one row runs: the buffer's first bytes, a texture buffer 12 texels wide over them
/// at the row's offset, its fills in one encoder, and every byte read back.
fn row_work(work: &mut Work, row: &Row) -> Result<(), String> {
    let buffer = work.buffer(stripes(BYTES));
    let texture = work.add(Resource::TextureOverBuffer {
        buffer,
        spec: TextureSpec {
            kind: TextureKind::TextureBuffer,
            storage: Storage::Shared,
            ..TextureSpec::texture_2d(row.format, 12, 1)
        },
        offset: row.offset,
        bytes_per_row: ROW.into(),
    });
    for spec in row.fills {
        let fill = parse_fill(spec)?;
        let (region, level, slice) = (fill.region, fill.level, fill.slice);
        work.step(match fill.what {
            Ok(pattern) => Step::FillTexture {
                texture,
                level,
                slice,
                region,
                pattern,
            },
            Err(packed_in) => Step::FillTextureColor {
                texture,
                level,
                slice,
                region,
                color: COLOR,
                packed_in,
            },
        });
    }
    work.step(Step::Commit);
    work.step(Step::ReadBuffer {
        buffer,
        offset: 0,
        length: BYTES,
    });
    Ok(())
}

/// The buffer's two rows of bytes, as a site's layout: a difference names the row and the byte in
/// it, which a flat offset does not.
fn layout() -> ByteLayout {
    ByteLayout {
        region: Region::Buffer {
            offset: 0,
            length: BYTES as u64,
        },
        element: Element::U8,
        row_offset: 0,
        row_elements: ROW,
        row_pitch: ROW,
        rows: 2,
        image_pitch: BYTES as u32,
        images: 1,
        gaps: Gaps::Compared,
    }
}

/// Metal's bytes of the row `key`: the probe's first bytes with the runs its fills changed written
/// in ([`METAL_RUNS`]).
fn metal_bytes(runs: &str, key: &str) -> Result<Vec<u8>, String> {
    let line = runs
        .lines()
        .find(|line| {
            line.split_once(' ')
                .map_or(*line == key, |(name, _)| name == key)
        })
        .ok_or_else(|| format!("no Metal runs of {key}"))?;
    let mut bytes = stripes(BYTES);
    for run in line.split('|').skip(1) {
        let (at, values) = run
            .split_once(':')
            .ok_or_else(|| format!("{key}: a run with no offset"))?;
        let at = usize::from_str_radix(at.trim().trim_start_matches("0x"), 16)
            .map_err(|error| format!("{key}: {at}: {error}"))?;
        for (i, value) in values.split_whitespace().enumerate() {
            let byte = u8::from_str_radix(value, 16)
                .map_err(|error| format!("{key}: {value}: {error}"))?;
            *bytes
                .get_mut(at + i)
                .ok_or_else(|| format!("{key}: a run past the buffer"))? = byte;
        }
    }
    Ok(bytes)
}

/// Which family of rows a case compares.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Family {
    Streams,
    Groups,
}

/// A case comparing every byte a family's fills wrote.
pub struct ByteFills {
    name: &'static str,
    family: Family,
}

/// The streams case: patterns longer than the texel, regions past the width, the buffer's end and
/// a level, and a fill after a fill.
pub fn streams() -> ByteFills {
    ByteFills {
        name: "texture-buffer-fill-streams",
        family: Family::Streams,
    }
}

/// The groups case: each format at four offsets, every power of two of pattern bytes up to the
/// texel, each packed color, and fills of no texels between others.
pub fn groups() -> ByteFills {
    ByteFills {
        name: "texture-buffer-fill-groups",
        family: Family::Groups,
    }
}

impl ByteFills {
    /// The family's rows.
    fn rows(&self) -> Vec<Row> {
        match self.family {
            Family::Streams => stream_rows(),
            Family::Groups => group_rows(),
        }
    }

    /// The whole family's program.
    fn work(&self) -> Result<Work, String> {
        let mut work = Work::new();
        for row in &self.rows() {
            row_work(&mut work, row)?;
        }
        Ok(work)
    }
}

/// The site a row's bytes sit at.
fn row_site(key: &str) -> String {
    format!("fills/{key}")
}

impl Case for ByteFills {
    fn name(&self) -> &'static str {
        self.name
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::carried_only(MEASURED_IN, NO_PUBLIC_CALL)
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let fills: String = self
            .rows()
            .iter()
            .map(|row| {
                format!(
                    "{} {} {} {:?}\n",
                    row.key, row.format, row.offset, row.fills
                )
            })
            .collect();
        Ok(vec![
            Input::bytes("buffer-first-bytes", &stripes(BYTES)),
            Input::bytes("fills", fills.as_bytes()),
            Input::bytes("metal-runs", public_fixture(METAL_RUNS)?.as_bytes()),
        ])
    }

    fn carried_transcript(&self) -> Result<Option<(u32, Transcript)>, String> {
        let runs = public_fixture(METAL_RUNS)?;
        let mut sites = Sites::new();
        sites.push(crate::case::inputs_node(&self.inputs()?));
        sites.leaf("work", crate::case::answered());
        for row in &self.rows() {
            let bytes = metal_bytes(&runs, &row.key)?;
            sites.push(Node::leaf(
                row_site(&row.key),
                Observed::bytes(bytes, layout()),
            ));
            sites.push(Node::leaf(
                format!("status/{}", row.key),
                Observed::value(Value::unit_variant("Completed")),
            ));
        }
        sites.notices(NOTICES_SITE, &[] as &[&str]);
        Ok(Some((MEASURED_RUNS, sites.finish())))
    }

    fn allowances(&self) -> Vec<crate::allowance::Allowance> {
        // The fills whose last copy of the pattern is cut at the texel read the bytes past it and
        // write them as zeros, as Metal's do, and the device announces it once. Metal announces
        // nothing, so the notices site differs, and the notice is what justifies it.
        vec![
            crate::replay::notice_allowance(
                metal2vulkan_runtime::Degradation::TextureBufferFillLeftover.code(),
            ),
            // The texture buffer these fills write into is backed by a `Shared` buffer
            // ([`crate::replay::shared_memory_notice`]).
            crate::replay::shared_memory_notice(),
        ]
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let work = self.work().map_err(|why| Abort {
            at: "work".to_owned(),
            failure: crate::case::Failure::named("BadCase", why),
        })?;
        let done = call(sites, "work", side.work(&work), |_| Value::Absent)?;
        for (at, row) in self.rows().iter().enumerate() {
            let read = done.reads.get(at).cloned().unwrap_or_default();
            sites.push(Node::leaf(
                row_site(&row.key),
                Observed::bytes(read, layout()),
            ));
            let status = done
                .statuses
                .get(at)
                .cloned()
                .unwrap_or_else(|| "Missing".to_owned());
            sites.push(Node::leaf(
                format!("status/{}", row.key),
                Observed::value(Value::unit_variant(status)),
            ));
        }
        Ok(())
    }
}

/// The long-pattern case: the digest the probe kept of each whole buffer.
pub struct LongFills {
    name: &'static str,
}

/// The long-pattern case.
pub fn long_fills() -> LongFills {
    LongFills {
        name: "texture-buffer-long-fills",
    }
}

/// One long row's buffer bytes: the stripes of the texels plus 64 bytes past them.
fn long_length(bytes: u64, offset: u64, width: u64) -> u64 {
    offset + width * bytes + 64
}

impl LongFills {
    /// The program: a texture buffer of each row's width over the stripes, one fill of every
    /// texel, every byte read back.
    fn work(&self) -> Work {
        let mut work = Work::new();
        for &(format, bytes, offset, width, pattern, _) in &LONG {
            let length = long_length(bytes, offset, width);
            let buffer = work.buffer(stripes(length as usize));
            let texture = work.add(Resource::TextureOverBuffer {
                buffer,
                spec: TextureSpec {
                    kind: TextureKind::TextureBuffer,
                    storage: Storage::Shared,
                    ..TextureSpec::texture_2d(format, width, 1)
                },
                offset,
                bytes_per_row: (width * bytes).next_multiple_of(256),
            });
            work.step(Step::FillTexture {
                texture,
                level: 0,
                slice: 0,
                region: Rect::new_3d([0, 0, 0], [width, 1, 1]),
                pattern: probe_stream(pattern),
            });
            work.step(Step::Commit);
            work.step(Step::ReadBuffer {
                buffer,
                offset: 0,
                length: length as usize,
            });
        }
        work
    }
}

/// The site a long row sits at.
fn long_site(format: u32, width: u64, pattern: usize) -> String {
    format!("fill/{format}-{width}-{pattern}")
}

impl Case for LongFills {
    fn name(&self) -> &'static str {
        self.name
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::carried_only(MEASURED_IN, NO_PUBLIC_CALL)
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let rows: String = LONG
            .iter()
            .map(|&(format, bytes, offset, width, pattern, _)| {
                format!("{format} {bytes} {offset} {width} {pattern}\n")
            })
            .collect();
        Ok(vec![
            Input::bytes("rows", rows.as_bytes()),
            Input::bytes("longest-pattern", &probe_stream(LONGEST)),
        ])
    }

    fn carried_transcript(&self) -> Result<Option<(u32, Transcript)>, String> {
        let mut sites = Sites::new();
        sites.push(crate::case::inputs_node(&self.inputs()?));
        sites.leaf("work", crate::case::answered());
        for &(format, _, _, width, pattern, metal) in &LONG {
            // The probe kept the digest of the buffer and not its bytes, so the recording holds
            // the digest alone: the comparison checks the digest and says so.
            sites.push(Node::leaf(
                long_site(format, width, pattern),
                Observed::absent().digested(Digest {
                    kind: DigestKind::Sha256Prefix16,
                    hex: metal.to_owned(),
                }),
            ));
        }
        sites.notices(NOTICES_SITE, &[] as &[&str]);
        Ok(Some((MEASURED_RUNS, sites.finish())))
    }

    fn allowances(&self) -> Vec<crate::allowance::Allowance> {
        // The fills whose last copy of the pattern is cut at the texel read the bytes past it and
        // write them as zeros, as Metal's do, and the device announces it once. Metal announces
        // nothing, so the notices site differs, and the notice is what justifies it.
        vec![
            crate::replay::notice_allowance(
                metal2vulkan_runtime::Degradation::TextureBufferFillLeftover.code(),
            ),
            // The texture buffer these fills write into is backed by a `Shared` buffer
            // ([`crate::replay::shared_memory_notice`]).
            crate::replay::shared_memory_notice(),
        ]
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let done = call(sites, "work", side.work(&self.work()), |_| Value::Absent)?;
        for (at, &(format, _, _, width, pattern, _)) in LONG.iter().enumerate() {
            let read = done.reads.get(at).cloned().unwrap_or_default();
            sites.push(Node::leaf(
                long_site(format, width, pattern),
                Observed::absent().digested(Digest::sha256_prefix16(&read)),
            ));
        }
        Ok(())
    }
}
