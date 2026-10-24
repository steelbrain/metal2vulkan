//! Blits between Apple's two-plane extended-range textures placed in a heap, ported from
//! `runtime_lane::two_plane_heap_blits`.
//!
//! The probe (`tph`): a placement heap of Metal's size, shared or private, every byte of it a byte
//! of the case's seed written through a buffer over the whole heap; two one-level 2D textures of
//! `RGB10A8_2P_XR10` or its sRGB twin placed where the probe placed them; one blit of the case's
//! region from one into the other; the heap's bytes read back through the buffer.
//!
//! **The split between the run shape and the bytes is deliberate, and it is the lane's.** Each
//! case holds two sites:
//!
//! - `runs`, a **typed** list of the runs the blit changed — each an offset and a length — which is
//!   what a blit-geometry bug moves: a copy that reads or writes the wrong rows, lands on the
//!   wrong plane, or strides by the wrong pitch changes *where* the bytes moved, and that shows in
//!   the run shape whatever the bytes are. This is 2,062 runs over the 25 cases, 23 KiB
//!   (`fixtures/public/two_plane_heap_blits/metal_runs.txt`), and it is the site a reader reads
//!   first.
//! - `bytes`, under a **digest alone**, because the moved bytes are 356 KiB and the probe printed
//!   the digest. Committing them would be fifteen times the measurement's size for a fact the run
//!   shape already localises, and the suite's record shape says a digest-only site compares by its
//!   digest and reports it as such — so the row never reads as exact over bytes nobody wrote down.
//!   Upgrading the site to raw bytes later needs no rewrite: the same site gains a value beside its
//!   digest.
//!
//! **Record mode runs this case on Metal.** It used to carry its answer, on the stated reason that
//! the two-plane extended-range formats are Apple's private ones and so Metal has no public call
//! for them. That reason was wrong: the *format* is not private — `MTLTextureDescriptor`'s pixel
//! format is a raw value — and nothing this case does is Apple SPI. The heap, the placed textures,
//! the buffer over the heap, the texture-to-texture blit and the read-back are all public
//! `objc2-metal` calls, which is what the probe used to take the table in the first place
//! ([`crate::metal_work`]).
//!
//! The carried table stays, as a cross-check rather than as the answer
//! ([`HeapBlits::carried_from_the_m4`]): the digests the probe printed on an Apple M4 under macOS
//! 27.0.1 (each blit three times on the same bytes and every case twice, alike) and on an Apple M3
//! Max under macOS 26.7, with the run shape recovered on this M3 Max under MoltenVK. A test holds
//! the committed recording to it, so "this M3 Max agrees with the M4, run shape and digest, in all
//! 25 cases" is a checked fact instead of a sentence. Note what the run shape's provenance was:
//! recovered under MoltenVK, which is *our* answer, not Metal's — so until record mode ran, that
//! site's expectation was never tied to Metal at all.

use crate::allowance::Allowance;
use crate::case::{call, Abort, Case, Input, MetalSource, Side};
use crate::measured::public_fixture;
use crate::replay::{notice_allowance, NOTICES_SITE};
use crate::transcript::{Digest, DigestKind, Node, Observed, Sites, Transcript, Value};
use crate::work::{Resource, Step, Storage, TextureSpec, Work};

/// The runs the carried cross-check table stands on: the probe ran each blit three times on the
/// same bytes and every case twice, alike.
const MEASURED_RUNS: u32 = 6;

/// The committed runs of bytes each case's blit changed, as `<offset>+<length>` each, a line a
/// case in [`CASES`] order.
const METAL_RUNS: &str = "two_plane_heap_blits/metal_runs.txt";

/// `RGB10A8_2P_XR10` and its sRGB twin, by raw pixel format.
const P2: u32 = 550;
const P2S: u32 = 551;

/// One of the probe's cases.
struct Case2P {
    name: &'static str,
    private: bool,
    /// The source's and the destination's formats.
    formats: (u32, u32),
    /// The heap's bytes: Metal's size.
    heap: u64,
    /// The source's extent and its place in the heap.
    source: ([u64; 2], u64),
    /// The destination's extent and its place in the heap.
    destination: ([u64; 2], u64),
    /// The copy: its source origin, its destination origin and its size.
    copy: ([u64; 2], [u64; 2], [u64; 2]),
    seed: u64,
    /// The digest the probe printed of the heap's bytes after the blit.
    metal: &'static str,
}

impl Case2P {
    /// Where the case's heap and its textures live: the storage mode its `private` flag names.
    fn storage(&self) -> Storage {
        match self.private {
            true => Storage::Private,
            false => Storage::Shared,
        }
    }
}

/// A case of one format whose source lies at 0 and whose destination at `at`.
const fn case(
    name: &'static str,
    private: bool,
    (heap, at): (u64, u64),
    (source, destination): ([u64; 2], [u64; 2]),
    copy: ([u64; 2], [u64; 2], [u64; 2]),
    seed: u64,
    metal: &'static str,
) -> Case2P {
    Case2P {
        name,
        private,
        formats: (P2, P2),
        heap,
        source: (source, 0),
        destination: (destination, at),
        copy,
        seed,
        metal,
    }
}

const A: [u64; 2] = [48, 20];
const B: [u64; 2] = [63, 65];
const C: [u64; 2] = [257, 9];
const D: [u64; 2] = [255, 64];
const E: [u64; 2] = [100, 30];
const SMALL: (u64, u64) = (1_277_952, 114_688);
const WHOLE_A: ([u64; 2], [u64; 2], [u64; 2]) = ([0, 0], [0, 0], A);
const WHOLE_B: ([u64; 2], [u64; 2], [u64; 2]) = ([0, 0], [0, 0], B);

/// The probe's cases whose answer Metal gives alike in every run: rows that do not overlap in the
/// destination with other bytes than each other's, widths of 4 or more.
const CASES: [Case2P; 25] = [
    case("a48", false, SMALL, (A, A), WHOLE_A, 11, "cf7a721ed69f675d"),
    case(
        "r48",
        false,
        SMALL,
        (A, A),
        ([3, 5], [11, 2], [17, 9]),
        12,
        "bc30954fd2097534",
    ),
    case(
        "w64",
        false,
        (1_310_720, 131_072),
        ([64, 64], [64, 64]),
        ([0, 0], [0, 0], [64, 64]),
        13,
        "cf53bd41e01d0583",
    ),
    case(
        "w63",
        false,
        (1_507_328, 229_376),
        (B, B),
        WHOLE_B,
        14,
        "979517cd394734e8",
    ),
    case(
        "w5",
        false,
        SMALL,
        ([5, 5], [5, 5]),
        ([0, 0], [0, 0], [5, 5]),
        15,
        "854674c5652d7840",
    ),
    case(
        "w9",
        false,
        SMALL,
        ([9, 7], [9, 7]),
        ([0, 0], [0, 0], [9, 7]),
        16,
        "ae4beef74adc1996",
    ),
    case(
        "g20",
        false,
        SMALL,
        ([20, 8], [33, 9]),
        ([0, 0], [2, 1], [20, 8]),
        17,
        "52c8b953a5dd52c4",
    ),
    case(
        "w257",
        false,
        SMALL,
        (C, C),
        ([0, 0], [0, 0], C),
        18,
        "3498238e64c35fdd",
    ),
    case(
        "w255",
        false,
        (2_031_616, 491_520),
        (D, D),
        ([0, 0], [0, 0], D),
        19,
        "bdc838283ed53d30",
    ),
    case(
        "row48",
        false,
        SMALL,
        (A, A),
        ([0, 3], [0, 7], [48, 1]),
        20,
        "f49d5cc7392c2921",
    ),
    case(
        "row63a",
        false,
        (1_507_328, 229_376),
        (B, B),
        ([0, 3], [0, 0], [63, 1]),
        21,
        "6bb95ed142a8bf06",
    ),
    case(
        "row63b",
        false,
        (1_507_328, 229_376),
        (B, B),
        ([0, 3], [0, 64], [63, 1]),
        22,
        "6b76f907a836cbea",
    ),
    case(
        "w4",
        false,
        SMALL,
        ([4, 6], [4, 6]),
        ([0, 0], [0, 0], [4, 6]),
        23,
        "9d40da5989e9b6ad",
    ),
    case(
        "w16",
        false,
        SMALL,
        ([16, 16], [16, 16]),
        ([0, 0], [0, 0], [16, 16]),
        25,
        "07ccd0b17977890f",
    ),
    case(
        "n100",
        false,
        (1_310_720, 131_072),
        (E, E),
        ([7, 2], [13, 5], [50, 20]),
        26,
        "dc81d76a2bbc3690",
    ),
    Case2P {
        formats: (P2, P2S),
        ..case(
            "d551",
            false,
            SMALL,
            (A, A),
            WHOLE_A,
            27,
            "7b637b1f83c242ff",
        )
    },
    Case2P {
        formats: (P2S, P2S),
        ..case(
            "s551",
            false,
            (1_507_328, 229_376),
            (B, B),
            WHOLE_B,
            28,
            "b252db87f05339de",
        )
    },
    case("pa48", true, SMALL, (A, A), WHOLE_A, 31, "0dabf59cbe10ecd5"),
    case(
        "pw63",
        true,
        (1_507_328, 229_376),
        (B, B),
        WHOLE_B,
        32,
        "338f41ba61912259",
    ),
    case(
        "pw257",
        true,
        SMALL,
        (C, C),
        ([0, 0], [0, 0], C),
        33,
        "0d02eb803c5db59f",
    ),
    case(
        "pn100",
        true,
        (1_310_720, 131_072),
        (E, E),
        ([7, 2], [13, 5], [50, 20]),
        35,
        "db6df4fe38d47c8f",
    ),
    case(
        "o257",
        false,
        SMALL,
        ([256, 9], C),
        ([0, 0], [1, 0], [256, 9]),
        84,
        "582e0cde0ebcb3c0",
    ),
    // The source past the destination, where the destination's alpha bytes land.
    Case2P {
        source: (A, 17_408),
        destination: (A, 0),
        ..case(
            "ovl",
            false,
            (1_179_648, 0),
            (A, A),
            WHOLE_A,
            88,
            "46d3dbcde4772f1f",
        )
    },
    // The destination at the end of a heap only as large as it needs: its alpha bytes pass the
    // heap's end.
    case(
        "end255",
        false,
        (573_440, 491_520),
        (D, D),
        ([0, 0], [0, 0], D),
        90,
        "b7fb61ea05982c35",
    ),
    case(
        "pend255",
        true,
        (573_440, 491_520),
        (D, D),
        ([0, 0], [0, 0], D),
        91,
        "64e1fbb9343ef64f",
    ),
];

/// The probe's bytes of `seed`: each the high bits of a 64-bit linear congruential generator.
fn random_bytes(count: u64, seed: u64) -> Vec<u8> {
    let mut state = seed;
    (0..count)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 33) as u8
        })
        .collect()
}

/// The case's name.
pub const HEAP_BLITS: &str = "two-plane-heap-blits";

/// The case of the suite.
pub struct HeapBlits {
    name: &'static str,
}

/// The ported case.
pub fn heap_blits() -> HeapBlits {
    HeapBlits { name: HEAP_BLITS }
}

impl HeapBlits {
    /// The transcript the probe's own table states: the run shape it printed and the digest of
    /// each case's heap, over the runs behind them.
    ///
    /// This was the case's answer while it was carried, and it is kept as the **cross-check** now
    /// that record mode runs the case on Metal. A test holds the committed recording to it, so the
    /// agreement between this M3 Max under macOS 27.0.1 and the probe's Apple M4 is a checked fact
    /// rather than a claim in a comment — and the day Apple changes one of these blits, the
    /// re-record and this table disagree where a reader can see it.
    pub fn carried_from_the_m4(&self) -> Result<(u32, Transcript), String> {
        let runs = public_fixture(METAL_RUNS)?;
        let mut sites = Sites::new();
        sites.push(crate::case::inputs_node(&self.inputs()?));
        for case in &CASES {
            sites.leaf(format!("work/{}", case.name), crate::case::answered());
            let shape = metal_runs(&runs, case.name)?;
            for node in case_nodes(
                case.name,
                runs_value(&shape),
                Digest {
                    kind: DigestKind::Sha256Prefix16,
                    hex: case.metal.to_owned(),
                },
            ) {
                sites.push(node);
            }
        }
        sites.notices(NOTICES_SITE, &[] as &[&str]);
        Ok((MEASURED_RUNS, sites.finish()))
    }
}

/// The program one case runs: the heap, its two textures, the seed's bytes written through a
/// buffer over the whole heap, one blit, and the heap's bytes read back.
fn case_work(work: &mut Work, case: &Case2P) {
    let storage = case.storage();
    let heap = work.add(Resource::Heap {
        size: case.heap,
        storage,
    });
    let texture = |format: u32, [w, h]: [u64; 2], at: u64| Resource::HeapTexture {
        heap,
        spec: TextureSpec {
            storage,
            ..TextureSpec::texture_2d(format, w, h)
        },
        offset: at,
    };
    let source = work.add(texture(case.formats.0, case.source.0, case.source.1));
    let destination = work.add(texture(
        case.formats.1,
        case.destination.0,
        case.destination.1,
    ));
    let over_heap = work.add(Resource::HeapBuffer {
        heap,
        length: case.heap,
        offset: 0,
    });
    let before = work.buffer(random_bytes(case.heap, case.seed));
    let after = work.buffer(vec![0; case.heap as usize]);
    work.step(Step::CopyBufferToBuffer {
        from: before,
        from_offset: 0,
        to: over_heap,
        to_offset: 0,
        length: case.heap,
    });
    work.step(Step::Commit);
    let (from, to, [w, h]) = case.copy;
    work.step(Step::CopyTextureToTexture {
        from: source,
        from_slice: 0,
        from_level: 0,
        from_origin: [from[0], from[1], 0],
        size: [w, h, 1],
        to: destination,
        to_slice: 0,
        to_level: 0,
        to_origin: [to[0], to[1], 0],
    });
    work.step(Step::Commit);
    work.step(Step::CopyBufferToBuffer {
        from: over_heap,
        from_offset: 0,
        to: after,
        to_offset: 0,
        length: case.heap,
    });
    work.step(Step::Commit);
    work.step(Step::ReadBuffer {
        buffer: after,
        offset: 0,
        length: case.heap as usize,
    });
}

/// The runs of `after` that differ from `before`: each run's first byte and its length.
fn changed_runs(before: &[u8], after: &[u8]) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut at = 0;
    while at < after.len() {
        if after.get(at) == before.get(at) {
            at += 1;
            continue;
        }
        let mut end = at;
        while end < after.len() && after.get(end) != before.get(end) {
            end += 1;
        }
        out.push((at, end - at));
        at = end;
    }
    out
}

/// Metal's runs of the case `name` ([`METAL_RUNS`]).
fn metal_runs(runs: &str, name: &str) -> Result<Vec<(usize, usize)>, String> {
    let line = runs
        .lines()
        .find(|line| {
            line.split_once(' ')
                .map_or(*line == name, |(case, _)| case == name)
        })
        .ok_or_else(|| format!("no Metal runs of {name}"))?;
    line.split_whitespace()
        .skip(1)
        .map(|run| {
            let (at, length) = run
                .split_once('+')
                .ok_or_else(|| format!("{name}: {run}: a run with no length"))?;
            let at = usize::from_str_radix(at.trim_start_matches("0x"), 16)
                .map_err(|error| format!("{name}: {at}: {error}"))?;
            let length: usize = length
                .parse()
                .map_err(|error| format!("{name}: {length}: {error}"))?;
            Ok((at, length))
        })
        .collect()
}

/// The runs as a transcript value: a list of offsets and lengths, so a difference names the run
/// that moved and by how much rather than printing two thousand numbers.
fn runs_value(runs: &[(usize, usize)]) -> Value {
    Value::List(
        runs.iter()
            .map(|&(at, length)| {
                Value::fields([
                    ("at", Value::Uint(at as u64)),
                    ("length", Value::Uint(length as u64)),
                ])
            })
            .collect(),
    )
}

/// The sites one case emits: its run shape and its heap's digest.
fn case_nodes(name: &str, runs: Value, digest: Digest) -> Vec<Node> {
    vec![
        Node::leaf(format!("heap/{name}/runs"), Observed::value(runs)),
        Node::leaf(
            format!("heap/{name}/bytes"),
            Observed::absent().digested(digest),
        ),
    ]
}

impl Case for HeapBlits {
    fn name(&self) -> &'static str {
        self.name
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        // The bytes every case's heap holds before its blit, the cases in order, and the geometry
        // each blit runs: either moving makes the recording's answer another run's.
        let mut before = Vec::new();
        let mut geometry = String::new();
        for case in &CASES {
            before.extend(random_bytes(case.heap, case.seed));
            geometry.push_str(&format!(
                "{} {} {:?} {} {:?} {:?} {:?}\n",
                case.name,
                case.private,
                case.formats,
                case.heap,
                case.source,
                case.destination,
                case.copy
            ));
        }
        Ok(vec![
            Input::bytes("heap-bytes-before", &before),
            Input::bytes("blit-geometry", geometry.as_bytes()),
            Input::bytes("metal-runs", public_fixture(METAL_RUNS)?.as_bytes()),
        ])
    }

    fn allowances(&self) -> Vec<Allowance> {
        // A copy whose alpha bytes land past the destination's alpha plane, on the heap's other
        // bytes, and in `end255` past the heap's end, is what Metal's blit writes and what the
        // device announces. Metal announces nothing, so the notices site differs and that notice
        // is what justifies it.
        //
        // These textures are also shadowed, and the notice's own definition names this exact
        // shape: a heap's 2D texture of one level of `RGB10A8_2P_XR10` or its sRGB twin has an
        // image that is a shadow of its two planes in the heap's bytes, where Metal lays them out.
        // Declaring it loosens no byte -- every heap byte the case reads compares exactly.
        [
            metal2vulkan_runtime::Degradation::TwoPlaneTextureCopy,
            metal2vulkan_runtime::Degradation::ShadowedTexture,
            // The buffer the case reads the blitted planes back through is `Shared`, which a
            // device with no unified memory keeps in system memory because that is the only
            // memory both sides can reach: the venue's memory model, not this blit.
            metal2vulkan_runtime::Degradation::SharedInSystemMemory,
        ]
        .into_iter()
        .map(|kind| notice_allowance(kind.code()))
        .collect()
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        for (at, case) in CASES.iter().enumerate() {
            let mut work = Work::new();
            case_work(&mut work, case);
            let done = call(
                sites,
                &format!("work/{}", case.name),
                side.work(&work),
                |_| Value::Absent,
            )?;
            let after = done.reads.first().cloned().unwrap_or_default();
            let before = random_bytes(case.heap, case.seed);
            let _ = at;
            for node in case_nodes(
                case.name,
                runs_value(&changed_runs(&before, &after)),
                Digest::sha256_prefix16(&after),
            ) {
                sites.push(node);
            }
        }
        Ok(())
    }
}
