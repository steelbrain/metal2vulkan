//! Metal's private texture fills of multisample textures, ported from
//! `runtime_lane::sample_fills`.
//!
//! The probe (`ms.m`, an Apple M3 Max under macOS 27.0.1): a private multisample texture of 5 x 3
//! texels (a two-slice array for one), every slice cleared by a render pass that draws nothing
//! (color `(0.25, 0.5, 0.75, 1)`, depth 0.25, stencil `0x5a`), one fill, and every sample read back
//! by a dispatch of one thread a sample — a color format's bits through an unsigned view of its
//! components, a depth's bits, a stencil's value, and `Depth32Float_Stencil8`'s stencil through its
//! `X32_Stencil8` view.
//!
//! **This is the sharpest fidelity in the batch**: Metal's answer is its read-back **lines**, as
//! the probe printed them, committed in `fixtures/public/sample_fills/metal.txt`. The case records
//! them as an ordered list of lines, so a difference names the first line that moved — the row of
//! the slice, with every texel's samples on it.
//!
//! **One case per row**, so a sample count the venue does not render stops that row alone and the
//! suite counts it as [`LacksFeature`](crate::verdict::Verdict::LacksFeature). That is not
//! cosmetic: lavapipe does not render 2 samples, which Vulkan leaves optional, and the lane's own
//! loop prints a line and carries on, leaving nothing compared and nothing counted.
//!
//! **The honest gap.** `d32-wide-4` fills a depth texture with the 16-byte pattern, so its samples
//! are depths outside `[0, 1]`. On a device whose depth copies go by a draw, the draw announces
//! that it changes such values (`drawn-depth-copy`), and the probe's lines are not what comes back
//! — so the lane committed that row's digest alone and compares the digest where the notice fired.
//! The ported case keeps that shape exactly: a digest-only site, with the notice named as what
//! justifies leaving it unchecked. It is not recoverable on this host and the port does not pretend
//! otherwise.
//!
//! Metal's answer is **carried** and re-recording is **forbidden**: Metal fills a texture through
//! private API.

use crate::allowance::{Allowance, Justification, Loosening};
use crate::case::{answered, call, Abort, Case, Failure, Input, MetalSource, Side};
use crate::fixture::{Fixture, METAL_TEXT, RUNTIME_TEXT};
use crate::measured::{hex_bytes, public_fixture};
use crate::recording::ShaderRef;
use crate::replay::{notice_allowance, NOTICES_SITE};
use crate::transcript::{Digest, DigestKind, Node, Observed, Sites, Transcript, Value};
use crate::work::{Clear, Rect, Resource, Step, TextureSpec, Work};

/// Where the carried values were measured.
const MEASURED_IN: &str =
    "runtime_lane::sample_fills, from the probe ms.m (an Apple M3 Max, macOS 27.0.1)";

/// Why record mode must not run these cases on Metal.
const NO_PUBLIC_CALL: &str =
    "Metal fills a texture through private API alone, so no public call exists to record it from";

/// The runs the carried table stands on: the probe ran each case once.
const MEASURED_RUNS: u32 = 1;

/// Metal's lines per case, each case's block headed `[<case>]`.
const METAL_LINES: &str = "sample_fills/metal.txt";

/// The owned fixture whose kernels read a multisample texture sample by sample.
const KERNELS: &str = "rt6_samples";

/// The texture's side.
const WIDTH: u32 = 5;
const HEIGHT: u32 = 3;

/// The probe's 16-byte pattern.
const P16: &str = "00112233445566778899aabbccddeeff";

/// The probe's 32-byte pattern.
const P32: &str = "00112233445566778899aabbccddeeff102132435465768798a9bacbdcedfe0f";

/// Depths in `[0, 1]`, one a sample: a denormal, 1, 0 and 0.25.
const DEPTHS: &str = "010000000000803f000000000000803e";

/// The one case whose depths lie outside `[0, 1]`, so a drawn depth copy changes them.
const WIDE_DEPTH: &str = "d32-wide-4";

/// The digest the probe printed of `d32-wide-4`'s lines, which is all the lane keeps of it.
const WIDE_DEPTH_DIGEST: &str = "9c5b7be066d1c9fc";

/// What a case fills: bytes in hexadecimal, or a color packed in the texture's own format.
#[derive(Clone, Copy, Debug)]
enum Fill {
    Bytes(&'static str),
    Color([f64; 4]),
}

/// One case: its name, the format's raw value, samples, the array's slices (0 for a 2D multisample
/// texture), the fill's slice, its region's origin and size, and what it fills.
struct Row {
    name: &'static str,
    format: u32,
    samples: u64,
    slices: u64,
    slice: u64,
    origin: [u64; 3],
    size: [u64; 3],
    fill: Fill,
}

const fn row(
    name: &'static str,
    format: u32,
    samples: u64,
    (slices, slice): (u64, u64),
    (origin, size): ([u64; 3], [u64; 3]),
    fill: Fill,
) -> Row {
    Row {
        name,
        format,
        samples,
        slices,
        slice,
        origin,
        size,
        fill,
    }
}

/// The region every case but the last two fills.
const ONE: ([u64; 3], [u64; 3]) = ([1, 1, 0], [2, 1, 1]);

/// The cases, as the probe ran them.
const ROWS: [Row; 17] = [
    row("r8-4", 10, 4, (0, 0), ONE, Fill::Bytes(P16)),
    row("rg8-4", 30, 4, (0, 0), ONE, Fill::Bytes(P16)),
    row("rgba8-4", 70, 4, (0, 0), ONE, Fill::Bytes(P16)),
    row("rgba8-2", 70, 2, (0, 0), ONE, Fill::Bytes(P32)),
    row("bgra8srgb-4", 81, 4, (0, 0), ONE, Fill::Bytes(P16)),
    row("rgba16f-4", 115, 4, (0, 0), ONE, Fill::Bytes(P16)),
    row("rgba16f-2", 115, 2, (0, 0), ONE, Fill::Bytes(P32)),
    row("rgba32f-2", 125, 2, (0, 0), ONE, Fill::Bytes(P32)),
    row("r32f-4", 55, 4, (0, 0), ONE, Fill::Bytes(P16)),
    row("d32-4", 252, 4, (0, 0), ONE, Fill::Bytes(DEPTHS)),
    row("d32-2", 252, 2, (0, 0), ONE, Fill::Bytes(DEPTHS)),
    row(WIDE_DEPTH, 252, 4, (0, 0), ONE, Fill::Bytes(P16)),
    row("s8-4", 253, 4, (0, 0), ONE, Fill::Bytes(P16)),
    row("d32s8-4", 260, 4, (0, 0), ONE, Fill::Bytes(DEPTHS)),
    row(
        "array-4",
        70,
        4,
        (2, 1),
        ([1, 0, 0], [3, 2, 1]),
        Fill::Bytes(P16),
    ),
    row(
        "color-4",
        70,
        4,
        (0, 0),
        ONE,
        Fill::Color([0.5, 0.25, 1.0, 0.75]),
    ),
    row(
        "whole-2",
        70,
        2,
        (0, 3),
        ([0, 0, 1], [9, 9, 2]),
        Fill::Bytes("a1b2c3d4"),
    ),
];

/// How a texture of `format` is read back, as the probe printed it: the kernel, the view's format
/// where the read goes through one, the bytes of each printed value, and the words the kernel
/// writes a sample.
fn reading(format: u32) -> (&'static str, Option<u32>, usize, usize) {
    match format {
        // R8Unorm, through R8Uint.
        10 => ("rt6_read_words1", Some(13), 1, 1),
        // RG8Unorm, through RG8Uint, read as one 16-bit word.
        30 => ("rt6_read_bytes4", Some(33), 2, 1),
        // RGBA8Unorm and BGRA8Unorm_sRGB, through RGBA8Uint, read as one 32-bit word.
        70 | 81 => ("rt6_read_bytes4", Some(73), 4, 1),
        // RGBA16Float, through RGBA16Uint, read as two words.
        115 => ("rt6_read_halves4", Some(113), 8, 2),
        // RGBA32Float, through RGBA32Uint, read as four words.
        125 => ("rt6_read_words4", Some(123), 16, 4),
        // R32Float, through R32Uint.
        55 => ("rt6_read_words1", Some(53), 4, 1),
        // Stencil8, read as its own value.
        253 => ("rt6_read_stencil", None, 1, 1),
        // Depth32Float and Depth32Float_Stencil8, read as the depth's bits.
        _ => ("rt6_read_depth", None, 4, 1),
    }
}

/// `X32_Stencil8`, the view `Depth32Float_Stencil8`'s stencil is read through.
const X32_STENCIL8: u32 = 261;

/// One value as the probe printed it: a texel of 1 or 2 bytes as one hex number, else its 32-bit
/// words in order.
fn printed(bytes: &[u8]) -> String {
    match bytes.len() {
        1 => format!("{:02x}", bytes[0]),
        2 => format!("{:04x}", u16::from_le_bytes([bytes[0], bytes[1]])),
        _ => bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|word| format!("{:08x}", u32::from_le_bytes(*word)))
            .collect(),
    }
}

/// The probe's lines of every sample `bytes` holds (texels in rows, each texel's samples, `unit`
/// bytes each) of slice `slice`, labelled `label`.
fn lines(label: &str, slice: u64, bytes: &[u8], samples: usize, unit: usize) -> Vec<String> {
    (0..HEIGHT as usize)
        .map(|y| {
            let mut line = format!("{label} s{slice} y{y}:");
            for x in 0..WIDTH as usize {
                line.push_str(" |");
                for s in 0..samples {
                    let at = ((y * WIDTH as usize + x) * samples + s) * unit;
                    line.push(' ');
                    line.push_str(&printed(bytes.get(at..at + unit).unwrap_or_default()));
                }
            }
            line
        })
        .collect()
}

/// Metal's lines of the case `name`, where the measurement holds them.
fn metal_lines(text: &str, name: &str) -> Option<Vec<String>> {
    let head = format!("[{name}]\n");
    let at = text.find(&head)? + head.len();
    let rest = &text[at..];
    let end = rest.find("\n[").map_or(rest.len(), |n| n + 1);
    Some(rest[..end].lines().map(str::to_owned).collect())
}

/// The program one row runs.
fn row_work(r: &Row) -> Result<Work, String> {
    let array = r.slices > 0;
    let layers = r.slices.max(1);
    let depth = matches!(r.format, 252 | 260);
    let stencil = matches!(r.format, 253 | 260);
    let mut work = Work::new();
    work.kernels = Some(KERNELS);
    let texture = work.add(Resource::Texture {
        spec: TextureSpec::texture_2d(r.format, WIDTH.into(), HEIGHT.into())
            .multisample(r.samples, layers),
    });
    for layer in 0..layers {
        let clear = match (depth, stencil) {
            (true, true) => Clear::DepthStencil(0.25, 0x5a),
            (true, false) => Clear::Depth(0.25),
            (false, true) => Clear::Stencil(0x5a),
            (false, false) => Clear::Color([0.25, 0.5, 0.75, 1.0]),
        };
        work.step(Step::ClearPass {
            texture,
            slice: layer,
            clear,
        });
    }
    let region = Rect::new_3d(r.origin, r.size);
    work.step(match r.fill {
        Fill::Bytes(pattern) => Step::FillTexture {
            texture,
            level: 0,
            slice: r.slice,
            region,
            pattern: hex_bytes(pattern)?,
        },
        Fill::Color(color) => Step::FillTextureColor {
            texture,
            level: 0,
            slice: r.slice,
            region,
            color,
            packed_in: None,
        },
    });
    let (kernel, view, texel_bytes, words) = reading(r.format);
    let samples = u32::try_from(r.samples).map_err(|error| format!("{}: {error}", r.name))?;
    let reads = |work: &mut Work, format: Option<u32>, kernel, texel_bytes, words, layer| {
        // A view where the read goes through another format, or where the texture is an array and
        // the kernel reads one slice as a 2D multisample texture.
        let source = match (format, array) {
            (None, false) => texture,
            (format, flatten) => work.add(Resource::View {
                texture,
                format: format.unwrap_or(r.format),
                level: 0,
                level_count: 1,
                slice: layer,
                slice_count: 1,
                flatten,
            }),
        };
        work.step(Step::ReadSamples {
            texture: source,
            kernel,
            width: WIDTH,
            height: HEIGHT,
            samples,
            texel_bytes,
            words,
        });
    };
    for layer in 0..layers {
        reads(&mut work, view, kernel, texel_bytes, words, layer);
    }
    if r.format == 260 {
        for layer in 0..layers {
            reads(
                &mut work,
                Some(X32_STENCIL8),
                "rt6_read_stencil",
                1,
                1,
                layer,
            );
        }
    }
    work.step(Step::Commit);
    Ok(work)
}

/// The lines one row's read-backs make, as the probe printed them: the color or depth or stencil
/// aspect first, every slice in order, and `Depth32Float_Stencil8`'s stencil after them as a word.
fn row_lines(r: &Row, reads: &[Vec<u8>]) -> Vec<String> {
    let layers = r.slices.max(1) as usize;
    let (_, _, texel_bytes, _) = reading(r.format);
    let samples = r.samples as usize;
    let mut out = Vec::new();
    for (layer, read) in reads.iter().take(layers).enumerate() {
        out.extend(lines("after ", layer as u64, read, samples, texel_bytes));
    }
    if r.format == 260 {
        for (layer, read) in reads.iter().skip(layers).take(layers).enumerate() {
            // The combined format's stencil, printed as a word.
            let words: Vec<u8> = read
                .iter()
                .flat_map(|&value| u32::from(value).to_le_bytes())
                .collect();
            out.extend(lines("afterS ", layer as u64, &words, samples, 4));
        }
    }
    out
}

/// A case comparing one row's samples with Metal's.
pub struct SampleFills {
    name: &'static str,
    row: &'static Row,
}

/// Each row's name as a case, in [`ROWS`] order.
const NAMES: [&str; 17] = [
    "sample-fills-r8-4",
    "sample-fills-rg8-4",
    "sample-fills-rgba8-4",
    "sample-fills-rgba8-2",
    "sample-fills-bgra8srgb-4",
    "sample-fills-rgba16f-4",
    "sample-fills-rgba16f-2",
    "sample-fills-rgba32f-2",
    "sample-fills-r32f-4",
    "sample-fills-d32-4",
    "sample-fills-d32-2",
    "sample-fills-d32-wide-4",
    "sample-fills-s8-4",
    "sample-fills-d32s8-4",
    "sample-fills-array-4",
    "sample-fills-color-4",
    "sample-fills-whole-2",
];

/// Every ported case: one per row.
pub fn all() -> Vec<SampleFills> {
    ROWS.iter()
        .zip(NAMES)
        .map(|(row, name)| SampleFills { name, row })
        .collect()
}

/// The site a row's lines sit at.
fn site(name: &str) -> String {
    format!("samples/{name}")
}

impl SampleFills {
    /// Whether the row is the one whose depths lie outside `[0, 1]`, so only its digest is Metal's.
    fn digest_only(&self) -> bool {
        self.row.name == WIDE_DEPTH
    }

    /// The fixture the row's read-backs run.
    fn fixture(&self) -> Fixture {
        Fixture::new(KERNELS)
    }
}

impl Case for SampleFills {
    fn name(&self) -> &'static str {
        self.name
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::carried_only(MEASURED_IN, NO_PUBLIC_CALL)
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        let fixture = self.fixture();
        let (kernel, _, _, _) = reading(self.row.format);
        Ok(vec![ShaderRef {
            fixture: KERNELS.to_owned(),
            entry: kernel.to_owned(),
            source_sha256: crate::sha256_hex(fixture.text(RUNTIME_TEXT)?.as_bytes()),
        }])
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let r = self.row;
        let geometry = format!(
            "{} format {} samples {} slices {} slice {} origin {:?} size {:?} clear \
             color (0.25, 0.5, 0.75, 1) depth 0.25 stencil 0x5a\n",
            r.name, r.format, r.samples, r.slices, r.slice, r.origin, r.size
        );
        let pattern = match r.fill {
            Fill::Bytes(pattern) => hex_bytes(pattern)?,
            Fill::Color(color) => color.iter().flat_map(|c| c.to_le_bytes()).collect(),
        };
        // Both committed texts of the read-back fixture: the one the runtime assembles and the one
        // Metal's own assembler reads. A change to either is a change to the case's inputs.
        let fixture = self.fixture();
        Ok(vec![
            Input::bytes("case-geometry", geometry.as_bytes()),
            Input::bytes("fill-pattern", &pattern),
            Input::bytes("kernels-ll", fixture.text(RUNTIME_TEXT)?.as_bytes()),
            Input::bytes("kernels-air-ll", fixture.text(METAL_TEXT)?.as_bytes()),
            Input::bytes("metal-lines", public_fixture(METAL_LINES)?.as_bytes()),
        ])
    }

    fn allowances(&self) -> Vec<Allowance> {
        let mut allowances = Vec::new();
        if self.digest_only() {
            // The depths outside [0, 1]: where the device's depth copies go by a draw, the draw
            // announces that it changes such values, and the probe's lines are not what comes
            // back. The lane keeps the digest alone there, and so does this case: the site goes
            // unchecked, named, counted and reported, with that notice as the justification.
            allowances.push(Allowance::new(
                format!("case/{}", site(self.row.name)),
                Loosening::Unchecked,
                Justification::Degradation(
                    metal2vulkan_runtime::Degradation::DrawnDepthCopy
                        .code()
                        .to_owned(),
                ),
            ));
        }
        // What a multisample fill and its read-back announce on a device that needs them: the
        // drawn depth copy, a shadowed texture view, and shared storage in system memory. Metal
        // announces none of them, so the notices site differs on such a device, and the
        // difference is justified where one of these fired. A depth texture's denormals and a
        // `Depth32Float` target's signed test (`depth-denormals`, `signed-depth-test`) are the
        // device's own answers, which the suite declares for every case
        // (`crate::replay::NATIVE_NOTES`); every depth word still compares exactly.
        use metal2vulkan_runtime::Degradation as D;
        for kind in [
            D::DrawnDepthCopy,
            D::SharedInSystemMemory,
            D::ShadowedTextureView,
        ] {
            allowances.push(notice_allowance(kind.code()));
        }
        // A 32-bit float depth row's fill reaches its depth image by an upload, which a device
        // without `VK_EXT_depth_range_unrestricted` (MoltenVK on the M3 Max) clamps into [0, 1]
        // and announces (`restricted-depth-upload`); measured there, the notices site is the
        // whole of the difference of each of the four such rows, and the wide row's digest is
        // unchecked above.
        if matches!(self.row.format, 252 | 260) {
            allowances.push(notice_allowance(D::RestrictedDepthUpload.code()));
        }
        allowances
    }

    fn carried_transcript(&self) -> Result<Option<(u32, Transcript)>, String> {
        let text = public_fixture(METAL_LINES)?;
        let mut sites = Sites::new();
        sites.push(crate::case::inputs_node(&self.inputs()?));
        sites.leaf("work", answered());
        let observed = match metal_lines(&text, self.row.name) {
            Some(lines) => Observed::value(Value::Lines(lines)),
            None if self.digest_only() => Observed::absent().digested(Digest {
                kind: DigestKind::Sha256Prefix16,
                hex: WIDE_DEPTH_DIGEST.to_owned(),
            }),
            None => {
                return Err(format!(
                    "{}: the measurement holds neither lines nor a digest",
                    self.row.name
                ))
            }
        };
        sites.push(Node::leaf(site(self.row.name), observed));
        sites.notices(NOTICES_SITE, &[] as &[&str]);
        Ok(Some((MEASURED_RUNS, sites.finish())))
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let work = row_work(self.row).map_err(|why| Abort {
            at: "work".to_owned(),
            failure: Failure::named("BadCase", why),
        })?;
        let done = call(sites, "work", side.work(&work), |_| Value::Absent)?;
        let lines = row_lines(self.row, &done.reads);
        let observed = if self.digest_only() {
            Observed::absent().digested(Digest::sha256_prefix16(lines.join("\n").as_bytes()))
        } else {
            Observed::value(Value::Lines(lines))
        };
        sites.push(Node::leaf(site(self.row.name), observed));
        Ok(())
    }
}
