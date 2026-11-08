//! The shared draw A/B harness: draw a logged render pipeline's fragment function on a Vulkan
//! device through the runtime, on seeded inputs, and record what it writes and how long it takes,
//! so two builds of the translator (a base, and a change that must leave every answer as it was,
//! such as a shrink of the modules) can be compared bit for bit and timed.
//!
//! Five commands:
//!
//! - `corpus-draw-ab extract --corpus DIR --draws FILE --ids FILE --out DIR`: one pass over the
//!   corpus's source shards writes each fragment source the listed draws name to `DIR/<key>.bc`
//!   (its AIR bitcode, assembled from the row's text where the row retains no blob) and
//!   `DIR/<key>.entry` (its entry point's name). Nothing here translates, so either build
//!   extracts for both.
//! - `corpus-draw-ab run --draws FILE --id ID --sources DIR --device-type cpu|gpu --shader-cache DIR
//!   --out DIR [--runs N] [--side N] [--combos FORMAT:MODE,...]`:
//!   draws draw `ID` `N` times (12 by default) into `side`-square attachments (32 by default) and
//!   writes `DIR/ID.tsv`, one line per run, written as the run ends: `ID`, the run, `OK` with the
//!   SHA-256 and length of everything the draw wrote, or a typed outcome (`REFUSED` with the
//!   runtime's error; `SKIP` naming an input this harness does not seed), and the run's inputs.
//!   With `OK`, the bytes go to `DIR/ID/run-<n>.bin`, and a last column flags each color
//!   attachment whose image is no evidence of what the fragment computes (the coordinator's
//!   byte-claim rule of 2026-10-10): `uniform` (every texel the same bytes) and `unchanged` (the
//!   bytes it started as), or `drawn`, as `images=0:uniform+unchanged,1:drawn`; such a run also
//!   says so on stderr, and `compare` counts it. `--combos` gives run `n` the `n`-th combo in turn:
//!   every texture of a float data type the draw only reads is of `FORMAT` (a `MTLPixelFormat` name
//!   without its prefix; `BC1_RGBA` uploads its blocks), and every sampler filters by `MODE`:
//!   `nearest`, `linear`, `mip` (linear between the levels of a full mipmap chain, which those
//!   textures then carry, each level seeded) or `aniso` (`mip` with an anisotropy of 16), and
//!   `nearest-px` or `linear-px` (pixel coordinates); a combo `FORMAT:MODE@SLOT=FORMAT2` binds
//!   `FORMAT2` at texture slot `SLOT` in place of `FORMAT` (A pass that reads its own target's
//!   format at one slot and samples 8-bit textures at the others); `FORMAT2` a two-plane YCbCr
//!   format of 8-bit 4:2:0 codes (`YCBCR8_420_2P`, its `_sRGB` twin, `RGB8_420_2P`) binds a
//!   texture over plane 0 of a `420v` surface of seeded bytes ([`ycbcr_surface`]), as a video
//!   frame is bound, and a two-plane extended-range format (`RGB10A8_2P_XR10`, its `_sRGB` twin)
//!   a texture over plane 0 of a `'b3a8'` surface of seeded bytes ([`surface_planes`]), as the
//!   consumer's drawables are sampled. `--primitive TYPE` (`point`,
//!   `line`, `line-strip`, `triangle` or `triangle-strip`; `time` takes it too) draws the three
//!   vertices as `TYPE`, where the draws are of the template's class's list by default (a triangle
//!   for no class): a strip of three vertices covers what its list does, through a pipeline of the
//!   strip's topology. `--depth-test less` (`time` takes it too) sets a depth-stencil state
//!   comparing depth `Less` without writes on every draw (tests that drop fragments and write
//!   nothing, `FragmentTests::Drop`), where the draws set none by default. Both print the device's
//!   pipeline work counts on stderr at the end (`work counts: ...`), and its shader cache
//!   statistics (`shader cache: ...`).
//! - `corpus-draw-ab time --draws FILE --id ID --sources DIR --device-type cpu|gpu --shader-cache DIR
//!   --out DIR [--side N] [--repeats N] [--combo FORMAT:MODE]`:
//!   creates the draw's pipeline (its wall time, cold), draws once (the first draw's wall time,
//!   which includes any variant made at the draw), then draws `repeats` times (30 by default),
//!   one draw per command buffer into `side`-square attachments (512 by default), and writes
//!   `DIR/ID.time.tsv`: the creation and first-draw milliseconds, then each command buffer's GPU
//!   milliseconds (`GPUEndTime - GPUStartTime`) and wall milliseconds (from its commit until the
//!   host saw it complete, one command buffer in flight), each with their median and minimum.
//!   `--combo FORMAT:MODE` (one of `run`'s combos) binds that combo's textures and samplers, where
//!   by default every sampler is linear and each texture slot takes its turn of formats.
//!   `--then-lines yes` adds one more draw, of lines, after the first (its wall time,
//!   `line_draw_ms`: the pipeline of the other topology where the state's topology class allows
//!   lines), and `--then-lines together` draws the lines in the first draw's command buffer, so
//!   both pipelines are made at once; `--qos CLASS` sets the thread's quality of service first, as
//!   `storm`'s does. `--device-qos CLASS` opens the device at `CLASS` and runs the rest at `--qos`
//!   (the device's own threads start from the opening thread), and `--prime-qos CLASS` draws the
//!   state as points from a thread of its own at `CLASS` before the first draw, `prime_ms` (the
//!   pool thread that making starts is started from that thread). `--then TYPE` draws once more, as
//!   `TYPE` (`--primitive`'s names), after the first draw and the lines, `then_draw_ms`. `--again
//!   yes` makes the state a second time in the same process, right after the first (the first kept
//!   alive), and writes that creation's wall time as `created_again_ms`: the warm creation, against
//!   the runtime's in-process memos and whatever its cache kept. `--queue-first yes` makes the
//!   command queue before the state, as a Metal program makes it at its start, so the room the
//!   device keeps for its encodings is kept by the queue's creation, not the state's, and writes
//!   that creation's wall time as `queue_first_ms`. The wall time bounds the draw where the GPU
//!   times do not: on an M3 Max through MoltenVK a 2048x2048 draw's GPU time stayed at a 512x512
//!   one's, about 0.2 ms, where a Radeon 780M through RADV took about 5 ms for the same draw at
//!   512x512 (the runtime samples MoltenVK's timestamps in blit passes of their own; why they miss
//!   the draw there is not established).
//! - `corpus-draw-ab storm --storm a|b|c|d --air FILE --vertex NAME --fragment NAME --device-type
//!   cpu|gpu --shader-cache DIR --out FILE [--makings N] [--concurrency C] [--turns T --ahead A]
//!   [--tick-ms MS] [--tick-frame WxH:DRAWS] [--hold-ms MS] [--qos CLASS]`: the compile-storm benchmark of the runtime's budget of pipeline makings in
//!   flight, on an owned fixture (`draw_ab/storm.rs`, which documents it); `--storm e --draws
//!   FILE --ids ID,... --sources DIR` in place of the fixture makes the listed draws' states.
//! - `corpus-draw-ab compare --base DIR --change DIR --draws FILE --runs N [--expected-skips
//!   ID,...] [--out FILE]`: the A/B's verdict over two legs' `run` tables, each run of each draw
//!   same, different, refused alike, skipped or missing; only runs drawn in both legs count as
//!   compared, so a draw skipped in both legs is not (`draw_ab/compare.rs`, which documents it). It
//!   also counts the same runs that carry no evidence: every image flagged in both legs, or one
//!   digest across every run of a draw in a leg.
//!
//! The draws file holds one JSON object per line, `{"id", "vertex", "fragment", "template"}`, as
//! `corpus-pipeline --draws` reads it. A run draws the fragment function with the generated
//! passthrough vertex function (`companion::passthrough_vertex`, Metal's fullscreen triangle,
//! writing every varying the fragment reads) under the logged descriptor's color attachments
//! (formats, blending, write masks), depth and stencil formats and topology class, without its
//! vertex descriptor (the passthrough reads no attribute). Every argument the fragment's reflection
//! marks used is bound: each buffer to at least 64 KiB, each texture to a `side`-square texture (6
//! faces, 2 slices or 4 deep as its type needs) of the run's format for the texture's class, each
//! sampler to the run's filter.
//!
//! A two-plane extended-range attachment (`RGB10A8_2P_XR10` and its sRGB twin, the consumer's
//! drawables) stands over plane 0 of a `'b3a8'` surface, a shared buffer of its two planes
//! (2026-10-08): the surface starts as the palette's bytes, and a run reads it back whole, both
//! planes and the bytes between them.
//!
//! The bytes come from a palette seeded by the run, the slot and the byte's place, never by the
//! build: float specials (NaN of both signs, infinities, the smallest and largest denormals,
//! signed zeros, rounding ties 0.5, 1.5 and 2.5, half's largest value) and the same as half
//! pairs; small integers (0 to 23 in each byte, or a word of 0 to 31, so switch selectors and
//! indices land in range); random words; and, for timing, half small integers and half ordinary
//! floats. Runs take the first three in turn; every other triple samples with linear filtering;
//! each texture slot steps through its class's formats run by run. Attachments start as the
//! palette's bytes (blending and framebuffer reads see them). A run reads back every attachment,
//! then every buffer and texture the fragment may write, in slot order; that is what it hashes.
//!
//! `--policy NAMES` (`run` and `time`) opens the device with those `DevicePolicy` switches at their
//! accurate forms, where it opens at the default policy otherwise: a comma-separated list of
//! `exact-answers` (`DevicePolicy::exact_answers`), so a making under an application's exact forms
//! is timed as it makes them.
//!
//! `--device-type` names the device the draw runs on, as validation's other device commands do
//! (`cpu` matches only a CPU implementation such as lavapipe); the device opens through
//! `candidate::runtime_case::open_device`, which refuses a GPU while the hang tripwire is set.
//!
//! `--shader-cache DIR` is the runtime's translated-shader cache directory, which the harness never
//! leaves at the per-user default: a draw's translations are its inputs' (guest data stays where
//! the caller puts it), and a fresh directory per invocation makes `time`'s creation cold. The
//! caller removes it.
//!
//! A buffer whose type holds a pointer, a texture, a sampler or another resource (an argument
//! buffer), a texture buffer, a multisampled texture, and an argument of any other kind
//! (threadgroup memory, imageblocks, function tables, acceleration structures) make the draw a
//! `SKIP` naming it: seeded bytes cannot stand for them. Each invocation runs one draw, so a
//! fault or a hang stops that draw alone; the caller bounds it.
//!
//! Exit statuses (the coordinator's false-pass rule of 2026-10-08). A stopped draw never passes
//! silently: its row is written, its reason goes to stderr, and the status says what stopped it.
//! - **0**: every draw asked for ran. For `run`, a `REFUSED` run is a typed outcome, so it ends
//!   with status 0 (`compare` counts it as refused alike, or as different from a drawn run). For
//!   `compare`, every run was compared and the same, but the expected skips'.
//! - **2**: the harness failed, or the command line is wrong.
//! - **3**: `run` or `time` skipped its draw (`SKIP`: an input it does not seed), so the draw was
//!   not covered. `--skip-ok yes` accepts that, with status 0, for a survey that expects skips.
//! - **4**: `time` measured nothing of what it was asked. Either the runtime refused the state,
//!   its inputs or the first draw, or a draw `--then-lines`, `--then` or `--prime-qos` asked for
//!   (`REFUSED`); the other measurements are still written.
//! - **1**: `compare` found a run that differs between the legs.
//! - **5**: `compare` found no run that differs, but a run it did not compare (skipped, refused
//!   alike or missing) of a draw `--expected-skips` does not name.
//!
//! GUEST DATA: the corpus, the draws file and every output are the consumer's when run on the
//! consumer's shaders: they stay on the hosts the guest data rule allows and are never committed.

use metal2vulkan::bind::SamplerDescriptor;
use metal2vulkan::device::DevicePolicy;
use metal2vulkan::metal::state::DepthStencilDescriptor;
use metal2vulkan::metal::CompareFunction;
use metal2vulkan::metal::{
    BindingAccess, BindingType, DataType, PixelFormat, PrimitiveTopologyClass, PrimitiveType,
    SamplerMinMagFilter, SamplerMipFilter, TextureType, TextureUsage,
};
use metal2vulkan::pipeline::StageSource;
use metal2vulkan::reflection::{Binding, RenderPipelineReflection, StructType};
use metal2vulkan_runtime::{
    BlitOption, Buffer, CommandBuffer, CommandBufferStatus, CommandQueue, DepthStencilState,
    Device, DeviceOptions, IoSurfacePlane, IoSurfaceYCbCr, LoadAction, Origin, PipelineOption,
    PipelineOptions, RenderPassDescriptor, RenderPipelineState, ResourceOptions, SamplerState,
    ShaderCacheOptions, Size, StoreAction, Texture, TextureDescriptor,
};
use metal2vulkan_validation::acceptance::row::fragment_row;
use metal2vulkan_validation::candidate::engine::graphics::RenderTemplate;
use metal2vulkan_validation::candidate::runtime_case::open_device;
use metal2vulkan_validation::gpu::DeviceType;
use metal2vulkan_validation::source::{for_each_source_shard, source_shards_dir};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

#[path = "draw_ab/compare.rs"]
mod compare;
#[path = "draw_ab/storm.rs"]
mod storm;

const USAGE: &str =
    "usage: corpus-draw-ab extract --corpus DIR --draws FILE --ids FILE --out DIR\n\
     \x20      corpus-draw-ab run --draws FILE --id ID --sources DIR --device-type cpu|gpu\n\
     \x20                         --shader-cache DIR --out DIR [--runs N] [--side N]\n\
     \x20                         [--combos FORMAT:MODE,...] [--primitive TYPE] [--skip-ok yes]\n\
     \x20      corpus-draw-ab time --draws FILE --id ID --sources DIR --device-type cpu|gpu\n\
     \x20                          --shader-cache DIR --out DIR [--side N] [--repeats N]\n\
     \x20                          [--then-lines yes|together] [--qos CLASS]\n\
     \x20                          [--device-qos CLASS] [--prime-qos CLASS] [--then TYPE]\n\
     \x20                          [--skip-ok yes] [--combo FORMAT:MODE] [--again yes]\n\
     \x20      (run and time take [--policy NAMES])\n\
     \x20      corpus-draw-ab storm --storm a|b|c|d --air FILE --vertex NAME --fragment NAME\n\
     \x20                           --device-type cpu|gpu --shader-cache DIR --out FILE\n\
     \x20                           [--makings N] [--concurrency C] [--turns T --ahead A]\n\
     \x20                           [--tick-ms MS] [--tick-frame WxH:DRAWS] [--hold-ms MS]\n\
     \x20                           [--qos CLASS]\n\
     \x20      corpus-draw-ab storm --storm e --draws FILE --ids ID,... --sources DIR\n\
     \x20                           (then as storms a to d, from --device-type on)\n\
     \x20      corpus-draw-ab compare --base DIR --change DIR --draws FILE --runs N\n\
     \x20                             [--expected-skips ID,...] [--out FILE]";

/// Each bound buffer's least length.
const BUFFER_BYTES: u64 = 64 * 1024;
/// The longest a run waits for its command buffer.
const WAIT: Duration = Duration::from_secs(300);

/// One line of the draws file.
#[derive(serde::Deserialize)]
struct Draw {
    id: String,
    fragment: String,
    template: RenderTemplate,
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let result = match args.first().map(String::as_str) {
        Some("extract") => extract(&args[1..]).map_err(Failure::from),
        Some("run") => run(&args[1..]),
        Some("time") => time(&args[1..]),
        Some("storm") => storm::storm(&args[1..]).map_err(Failure::from),
        Some("compare") => compare::compare(&args[1..]),
        _ => Err(Failure::from(USAGE.to_string())),
    };
    if let Err(failure) = result {
        eprintln!("corpus-draw-ab: {}", failure.message());
        std::process::exit(failure.status());
    }
}

/// Why a command ends with a failing status (the module docs' exit statuses). A skipped or
/// untimed draw still writes its row first.
#[derive(Debug)]
enum Failure {
    /// The harness itself failed, or the command line is wrong: status 2.
    Harness(String),
    /// A draw stopped before it ran for an input this harness does not seed (`SKIP`), where
    /// `--skip-ok yes` did not accept it: status 3. The draw was not covered.
    Skipped(String),
    /// `time` measured nothing it was asked to: the runtime refused the draw, its inputs, or a
    /// draw `--then-lines`, `--then` or `--prime-qos` asked for (`REFUSED`): status 4.
    Untimed(String),
    /// `compare` found a run that differs between the legs: status 1.
    Differs(String),
    /// `compare` found no difference, but runs it did not compare of draws not expected to
    /// skip: status 5.
    Uncompared(String),
}

impl Failure {
    /// The process's exit status.
    fn status(&self) -> i32 {
        match self {
            Failure::Differs(_) => 1,
            Failure::Harness(_) => 2,
            Failure::Skipped(_) => 3,
            Failure::Untimed(_) => 4,
            Failure::Uncompared(_) => 5,
        }
    }

    /// The reason, for stderr.
    fn message(&self) -> &str {
        match self {
            Failure::Harness(why)
            | Failure::Skipped(why)
            | Failure::Untimed(why)
            | Failure::Differs(why)
            | Failure::Uncompared(why) => why,
        }
    }
}

impl From<String> for Failure {
    fn from(error: String) -> Failure {
        Failure::Harness(error)
    }
}

/// `--skip-ok yes`: a draw this harness cannot seed (`SKIP`) ends `run` or `time` with status 0,
/// for a survey that expects some. Without it (or with `no`), with status 3.
fn skip_ok(options: &BTreeMap<String, String>) -> Result<bool, String> {
    match options.get("skip-ok").map(String::as_str) {
        None | Some("no") => Ok(false),
        Some("yes") => Ok(true),
        Some(other) => Err(format!("--skip-ok {other:?}: yes or no")),
    }
}

/// The value of each `--name value` pair of `args`.
fn options(args: &[String]) -> Result<BTreeMap<String, String>, String> {
    let mut out = BTreeMap::new();
    let mut at = args.iter();
    while let Some(name) = at.next() {
        let key = name
            .strip_prefix("--")
            .ok_or_else(|| format!("unexpected argument {name:?}\n{USAGE}"))?;
        let value = at.next().ok_or_else(|| format!("{name} needs a value"))?;
        out.insert(key.to_string(), value.clone());
    }
    Ok(out)
}

/// The device policy `--policy` names (the module docs): the default policy with each named
/// switch at its accurate form.
fn policy(options: &BTreeMap<String, String>) -> Result<DevicePolicy, String> {
    let mut policy = DevicePolicy::default();
    let Some(names) = options.get("policy") else {
        return Ok(policy);
    };
    for name in names.split(',').filter(|name| !name.is_empty()) {
        match name {
            "exact-answers" => policy = policy.exact_answers(),
            _ => return Err(format!("--policy {name:?}: exact-answers")),
        }
    }
    Ok(policy)
}

fn required<'a>(options: &'a BTreeMap<String, String>, name: &str) -> Result<&'a str, String> {
    options
        .get(name)
        .map(String::as_str)
        .ok_or_else(|| format!("--{name} is required\n{USAGE}"))
}

fn number(options: &BTreeMap<String, String>, name: &str, default: u64) -> Result<u64, String> {
    options
        .get(name)
        .map_or(Ok(default), |value| value.parse())
        .map_err(|error| format!("--{name}: {error}"))
}

fn draws(path: &str) -> Result<Vec<Draw>, String> {
    let text = std::fs::read_to_string(path).map_err(|error| format!("read {path}: {error}"))?;
    text.lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| serde_json::from_str(line).map_err(|error| format!("{path}: {error}")))
        .collect()
}

fn write(path: &Path, bytes: &[u8]) -> Result<(), String> {
    std::fs::write(path, bytes).map_err(|error| format!("write {}: {error}", path.display()))
}

fn extract(args: &[String]) -> Result<(), String> {
    let options = options(args)?;
    let corpus = PathBuf::from(required(&options, "corpus")?);
    let ids: BTreeSet<String> = std::fs::read_to_string(required(&options, "ids")?)
        .map_err(|error| format!("read the ids: {error}"))?
        .split_whitespace()
        .map(String::from)
        .collect();
    let out = PathBuf::from(required(&options, "out")?);
    std::fs::create_dir_all(&out).map_err(|error| format!("create {}: {error}", out.display()))?;
    let wanted: BTreeSet<String> = draws(required(&options, "draws")?)?
        .into_iter()
        .filter(|draw| ids.contains(&draw.id))
        .map(|draw| draw.fragment)
        .collect();
    let dir = source_shards_dir(&corpus);
    let mut shards: Vec<PathBuf> = std::fs::read_dir(&dir)
        .map_err(|error| format!("read {}: {error}", dir.display()))?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.extension().is_some_and(|e| e == "jsonl"))
        .collect();
    shards.sort();
    let mut found = 0usize;
    for shard in shards {
        for_each_source_shard(&shard, |row| {
            let key = row.key().into_owned();
            if !wanted.contains(&key) {
                return Ok(());
            }
            let bitcode = match &row.blob_b64 {
                Some(b64) => {
                    base64::Engine::decode(&base64::engine::general_purpose::STANDARD, b64)
                        .map_err(|error| format!("{key}: base64: {error}"))?
                }
                None => metal2vulkan_validation::check::loop_free::assemble(&row.air_ll)
                    .map_err(|error| format!("{key}: assemble: {error}"))?,
            };
            write(&out.join(format!("{key}.bc")), &bitcode)?;
            write(&out.join(format!("{key}.entry")), row.entry.as_bytes())?;
            found += 1;
            Ok(())
        })?;
    }
    println!("extracted {found} of {} sources", wanted.len());
    Ok(())
}

/// A palette of seeded bytes.
#[derive(Clone, Copy, Debug)]
enum Palette {
    Specials,
    SmallIntegers,
    Random,
    Timing,
}

/// splitmix64 of `(stream, n)`.
fn mix(stream: u64, n: u64) -> u64 {
    let mut z = stream
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(n.wrapping_mul(0xBF58_476D_1CE4_E5B9))
        .wrapping_add(0x94D0_49BB_1331_11EB);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Float specials as single words, then as half pairs.
const SPECIALS: [u32; 24] = [
    0x7FC0_0000, // NaN
    0xFFC0_0001, // a negative NaN with a payload
    0x7F80_0000, // +inf
    0xFF80_0000, // -inf
    0x0000_0001, // the smallest denormal
    0x007F_FFFF, // the largest denormal
    0x8000_0001, // a negative denormal
    0x0000_0000, // +0
    0x8000_0000, // -0
    0x3F00_0000, // 0.5
    0x3FC0_0000, // 1.5
    0x4020_0000, // 2.5
    0x3F80_0000, // 1
    0xBF80_0000, // -1
    0x477F_E000, // 65504, half's largest
    0x3EAA_AAAB, // 1/3
    0x7E00_7C00, // half NaN, half +inf
    0xFC00_0001, // half -inf, half's smallest denormal
    0x03FF_8001, // half's largest denormal, a negative half denormal
    0x3800_3E00, // half 0.5, 1.5
    0x4100_3C00, // half 2.5, 1
    0x7BFF_0000, // half 65504, +0
    0x3555_8000, // half 1/3, -0
    0x3F7F_FFFF, // just under 1
];

/// The `n`-th word of `palette`'s stream `stream`, the same in every build.
fn word(palette: Palette, stream: u64, n: u64) -> u32 {
    let z = mix(stream, n);
    let small = |z: u64| {
        if z & 1 == 1 {
            ((z >> 8) % 32) as u32
        } else {
            let lane = |shift: u32| (((z >> shift) & 0xFF) % 24) as u32;
            lane(8) | lane(16) << 8 | lane(24) << 16 | lane(32) << 24
        }
    };
    match palette {
        Palette::Specials if z & 3 == 3 => (z >> 32) as u32,
        Palette::Specials => SPECIALS[((z >> 8) % SPECIALS.len() as u64) as usize],
        Palette::SmallIntegers => small(z),
        Palette::Random => (z >> 16) as u32,
        // A float in [-2, 2), or a small integer.
        Palette::Timing if z & 2 == 2 => {
            let sign = ((z >> 2) & 1) as u32;
            let exponent = 125 + ((z >> 3) % 3) as u32;
            sign << 31 | exponent << 23 | ((z >> 8) & 0x7F_FFFF) as u32
        }
        Palette::Timing => small(z),
    }
}

/// `len` bytes of `palette`'s stream `stream`.
fn bytes(palette: Palette, stream: u64, len: u64) -> Vec<u8> {
    (0..len.div_ceil(4))
        .flat_map(|n| word(palette, stream, n).to_le_bytes())
        .take(len as usize)
        .collect()
}

/// The formats a texture of `data` type takes, one per run in turn: every class the consumer's
/// textures come in.
fn formats(data: DataType, depth: bool) -> &'static [PixelFormat] {
    use PixelFormat as P;
    if depth {
        return &[P::Depth32Float];
    }
    match data {
        DataType::Int | DataType::Short | DataType::Char => {
            &[P::RGBA8Sint, P::RGBA16Sint, P::RGBA32Sint]
        }
        DataType::UInt | DataType::UShort | DataType::UChar => {
            &[P::RGBA8Uint, P::RGBA16Uint, P::RGBA32Uint]
        }
        _ => &[
            P::RGBA8Unorm,
            P::RGBA16Float,
            P::BGRA8Unorm,
            P::RGBA32Float,
            P::RGBA8Snorm,
            P::RGBA16Unorm,
            P::RGBA16Snorm,
            P::RGB10A2Unorm,
            P::RG11B10Float,
            P::B5G6R5Unorm,
            P::A1BGR5Unorm,
            P::ABGR4Unorm,
        ],
    }
}

/// The layout of a two-plane extended-range attachment's or sampled texture's surface
/// (`RGB10A8_2P_XR10` and its sRGB twin), a texture over plane 0 of a `'b3a8'` surface as the
/// consumer's drawables are: plane 0's
/// rows of `side` color words, then plane 1's rows of `side` alpha bytes, each row padded to 64
/// bytes. `None` for every other format. Returns plane 0's `bytesPerRow`, plane 1's offset and
/// `bytesPerRow`, and the surface's length.
fn surface_planes(format: PixelFormat, side: u64) -> Option<(u64, u64, u64, u64)> {
    use PixelFormat as P;
    if !matches!(format, P::RGB10A82PXR10 | P::RGB10A82PXR10SRGB) {
        return None;
    }
    let color_row = (4 * side).next_multiple_of(64);
    let alpha_at = color_row * side;
    let alpha_row = side.next_multiple_of(64);
    Some((color_row, alpha_at, alpha_row, alpha_at + alpha_row * side))
}

/// The OSType of the surface a two-plane extended-range attachment stands over (`'b3a8'`), the
/// one Metal's answers were measured over.
const B3A8: u32 = u32::from_be_bytes(*b"b3a8");

/// The OSType of the surface a sampled two-plane YCbCr texture stands over (`'420v'`, CoreVideo's
/// 8-bit 4:2:0 video-range surface, as a video frame's texture is).
const YUV420V: u32 = u32::from_be_bytes(*b"420v");

/// The layout of the `420v` surface a sampled two-plane YCbCr texture of `format` and `side`
/// stands over: the luma plane's rows of `side` codes, then the chroma plane's rows of `side /
/// 2` Cb and Cr pairs, each row padded to 64 bytes, for the 8-bit 4:2:0 formats. `None` for
/// every other format. Returns the planes' `bytesPerRow`, the chroma plane's offset and the
/// surface's length.
fn ycbcr_surface(format: PixelFormat, side: u64) -> Option<(u64, u64, u64)> {
    use PixelFormat as P;
    if !matches!(format, P::YCBCR84202P | P::YCBCR84202PSRGB | P::RGB84202P) {
        return None;
    }
    let row = side.next_multiple_of(64);
    let chroma_at = row * side;
    Some((row, chroma_at, chroma_at + row * side.div_ceil(2)))
}

/// The bytes of one texel of `format`, for the formats a draw may use here.
fn texel_bytes(format: PixelFormat) -> Option<u64> {
    use PixelFormat as P;
    Some(match format {
        P::A8Unorm | P::R8Unorm | P::R8UnormSRGB | P::R8Snorm | P::R8Uint | P::R8Sint => 1,
        P::R16Unorm
        | P::R16Snorm
        | P::R16Uint
        | P::R16Sint
        | P::R16Float
        | P::RG8Unorm
        | P::RG8UnormSRGB
        | P::RG8Snorm
        | P::RG8Uint
        | P::RG8Sint
        | P::B5G6R5Unorm
        | P::A1BGR5Unorm
        | P::ABGR4Unorm
        | P::BGR5A1Unorm => 2,
        P::R32Uint
        | P::R32Sint
        | P::R32Float
        | P::RG16Unorm
        | P::RG16Snorm
        | P::RG16Uint
        | P::RG16Sint
        | P::RG16Float
        | P::RGBA8Unorm
        | P::RGBA8UnormSRGB
        | P::RGBA8Snorm
        | P::RGBA8Uint
        | P::RGBA8Sint
        | P::BGRA8Unorm
        | P::BGRA8UnormSRGB
        | P::RGB10A2Unorm
        | P::RGB10A2Uint
        | P::RG11B10Float
        | P::RGB9E5Float
        | P::BGR10A2Unorm
        | P::Depth32Float => 4,
        P::RG32Uint
        | P::RG32Sint
        | P::RG32Float
        | P::RGBA16Unorm
        | P::RGBA16Snorm
        | P::RGBA16Uint
        | P::RGBA16Sint
        | P::RGBA16Float => 8,
        P::RGBA32Uint | P::RGBA32Sint | P::RGBA32Float => 16,
        _ => return None,
    })
}

/// Whether a value of `data` is a resource, not bytes.
fn resource(data: DataType) -> bool {
    matches!(
        data,
        DataType::Texture
            | DataType::Sampler
            | DataType::Pointer
            | DataType::RenderPipeline
            | DataType::ComputePipeline
            | DataType::IndirectCommandBuffer
            | DataType::VisibleFunctionTable
            | DataType::IntersectionFunctionTable
            | DataType::PrimitiveAccelerationStructure
            | DataType::InstanceAccelerationStructure
            | DataType::DepthStencilState
            | DataType::Tensor
    )
}

/// Whether `s` holds a resource at any depth.
fn holds_resource(s: &StructType) -> bool {
    s.members().iter().any(|member| {
        resource(member.data_type())
            || member.texture_reference_type().is_some()
            || member.pointer_type().is_some()
            || member.struct_type().is_some_and(holds_resource)
            || member.array_type().is_some_and(|array| {
                resource(array.element_type())
                    || array.element_texture_reference_type().is_some()
                    || array.element_pointer_type().is_some()
                    || array.element_struct_type().is_some_and(holds_resource)
            })
    })
}

/// What stops a draw before it runs.
enum Stop {
    /// An input this harness does not seed.
    Skip(String),
    /// The runtime refused the pipeline.
    Refused(String),
    /// The harness itself failed.
    Fatal(String),
}

impl From<String> for Stop {
    fn from(error: String) -> Stop {
        Stop::Fatal(error)
    }
}

/// A draw's pipeline, ready to draw.
struct Setup {
    device: Device,
    queue: CommandQueue,
    state: RenderPipelineState,
    reflection: RenderPipelineReflection,
    template: RenderTemplate,
    primitive: PrimitiveType,
    created: Duration,
    /// `--again yes`: the second creation's wall time, the first state alive.
    created_again: Option<Duration>,
    /// `--queue-first yes`: the command queue's creation's wall time, before the state's.
    queue_first: Option<Duration>,
    /// `--depth-test less`: the depth-stencil state every draw sets, if any.
    depth_stencil: Option<DepthStencilState>,
}

impl Setup {
    /// The fragment arguments the draw binds: those the function uses.
    fn bindings(&self) -> impl Iterator<Item = &Binding> {
        self.reflection
            .fragment_bindings()
            .iter()
            .filter(|binding| binding.is_used())
    }
}

/// A draw of the draws file, on a device: its logged template without its vertex descriptor,
/// its fragment function (from `sources`), and the passthrough vertex function generated for it.
struct DrawParts {
    template: RenderTemplate,
    vertex: metal2vulkan_runtime::Function,
    fragment: metal2vulkan_runtime::Function,
}

impl DrawParts {
    /// The render pipeline descriptor of `template` (this draw's, or a variant of it).
    fn descriptor(
        &self,
        template: &RenderTemplate,
    ) -> Result<metal2vulkan::pipeline::RenderPipelineDescriptor, String> {
        template
            .descriptor(
                StageSource::Function(self.vertex.function_descriptor()),
                Some(StageSource::Function(self.fragment.function_descriptor())),
            )
            .map_err(|error| error.to_string())
    }
}

/// Draw `id` of the draws file `draws_file` on `device`, its fragment source from `sources`.
fn draw_parts(
    device: &Device,
    draws_file: &str,
    id: &str,
    sources: &Path,
) -> Result<DrawParts, Stop> {
    let draw = draws(draws_file)?
        .into_iter()
        .find(|draw| draw.id == id)
        .ok_or_else(|| format!("no draw {id}"))?;
    let read = |what: String| -> Result<Vec<u8>, String> {
        std::fs::read(sources.join(&what)).map_err(|error| format!("read {what}: {error}"))
    };
    let bitcode = read(format!("{}.bc", draw.fragment))?;
    let entry = String::from_utf8(read(format!("{}.entry", draw.fragment))?)
        .map_err(|error| format!("the entry's name: {error}"))?;
    let row = fragment_row(
        &bitcode,
        &entry,
        &metal2vulkan::library::FunctionConstantValues::defaults(),
    )
    .map_err(|why| Stop::Skip(format!("passthrough vertex: {why}")))?;
    let companion = row.companion.opaque_bitcode()?;
    let library = |bitcode: &[u8]| -> Result<metal2vulkan_runtime::Library, String> {
        let parsed =
            metal2vulkan::library::Library::parse(bitcode).map_err(|error| error.to_string())?;
        Ok(device.new_library_with_compile_library(parsed))
    };
    let vertex = library(&companion)?
        .new_function_with_name(row.vertex_entry)
        .map_err(|error| error.to_string())?;
    let fragment = library(&bitcode)?
        .new_function_with_name(&entry)
        .map_err(|error| error.to_string())?;
    let mut template = draw.template;
    template.vertex_descriptor = None;
    Ok(DrawParts {
        template,
        vertex,
        fragment,
    })
}

/// The primitive type `--primitive` names.
fn primitive_type(name: &str) -> Result<PrimitiveType, String> {
    Ok(match name {
        "point" => PrimitiveType::Point,
        "line" => PrimitiveType::Line,
        "line-strip" => PrimitiveType::LineStrip,
        "triangle" => PrimitiveType::Triangle,
        "triangle-strip" => PrimitiveType::TriangleStrip,
        _ => {
            return Err(format!(
                "--primitive {name:?}: point, line, line-strip, triangle or triangle-strip"
            ))
        }
    })
}

fn setup(options: &BTreeMap<String, String>) -> Result<Setup, Stop> {
    let id = required(options, "id")?;
    let sources = PathBuf::from(required(options, "sources")?);
    let mut device_options = DeviceOptions::default();
    device_options.shader_cache =
        ShaderCacheOptions::in_directory(required(options, "shader-cache")?);
    device_options.policy = policy(options)?;
    // Through validation's tripwire-checked open: a GPU only while the hang tripwire is absent.
    let device_type = DeviceType::parse(required(options, "device-type")?)?;
    let device =
        open_device(device_type, device_options).map_err(|error| format!("device: {error}"))?;
    // `time --device-qos` opened the device at its class; the rest runs at `--qos`.
    if options.contains_key("device-qos") {
        if let Some(qos) = options.get("qos") {
            storm::set_qos(qos)?;
        }
    }
    let parts = draw_parts(&device, required(options, "draws")?, id, &sources)?;
    for attachment in parts.template.color_attachments.values() {
        let format = attachment.pixel_format;
        if texel_bytes(format).is_none() && surface_planes(format, 1).is_none() {
            return Err(Stop::Skip(format!(
                "color attachment format {}",
                attachment.pixel_format.name()
            )));
        }
    }
    let descriptor = parts.descriptor(&parts.template)?;
    let template = parts.template;
    // `--queue-first yes`: the command queue made before the state, as a Metal program makes it
    // at its start: the room the device keeps for its encodings is then kept by the queue's
    // creation, not by the state's.
    let (early_queue, queue_first) = match options.get("queue-first").map(String::as_str) {
        None => (None, None),
        Some("yes") => {
            let start = Instant::now();
            let queue = device
                .new_command_queue()
                .map_err(|error| error.to_string())?;
            (Some(queue), Some(start.elapsed()))
        }
        Some(other) => return Err(format!("--queue-first {other:?}: yes").into()),
    };
    let mut pipeline_options = PipelineOptions::default();
    pipeline_options.options = PipelineOption::BINDING_INFO;
    let start = Instant::now();
    let (state, reflection) = device
        .new_render_pipeline_state_with_reflection(&descriptor, &pipeline_options)
        .map_err(|error| Stop::Refused(format!("pipeline: {error}")))?;
    let created = start.elapsed();
    // `--again yes`: the same state again, in this process, the first one alive (warm).
    let created_again = match options.get("again").map(String::as_str) {
        None => None,
        Some("yes") => {
            let start = Instant::now();
            let again = device
                .new_render_pipeline_state_with_reflection(&descriptor, &pipeline_options)
                .map_err(|error| Stop::Refused(format!("pipeline, again: {error}")))?;
            let took = start.elapsed();
            drop(again);
            Some(took)
        }
        Some(other) => return Err(format!("--again {other:?}: yes").into()),
    };
    let reflection = reflection.ok_or_else(|| "the pipeline gave no reflection".to_string())?;
    for binding in reflection
        .fragment_bindings()
        .iter()
        .filter(|b| b.is_used())
    {
        let index = binding.index();
        let skip = match binding.binding_type() {
            BindingType::Buffer => binding
                .buffer()
                .is_some_and(|buffer| {
                    resource(buffer.buffer_data_type())
                        || buffer.buffer_struct_type().is_some_and(holds_resource)
                        || buffer
                            .buffer_pointer_type()
                            .is_some_and(|pointer| pointer.element_is_argument_buffer())
                })
                .then(|| format!("buffer {index} holds resources (an argument buffer)")),
            BindingType::Texture => match binding.texture().map(|t| t.texture_type()) {
                Some(TextureType::TextureBuffer) => {
                    Some(format!("texture {index} is a texture buffer"))
                }
                Some(TextureType::Type2DMultisample | TextureType::Type2DMultisampleArray) => {
                    Some(format!("texture {index} is multisampled"))
                }
                Some(_) => None,
                None => Some(format!("texture {index} has no texture type")),
            },
            BindingType::Sampler => None,
            other => Some(format!("{} at {index}", other.name())),
        };
        if let Some(why) = skip {
            return Err(Stop::Skip(why));
        }
    }
    let queue = match early_queue {
        Some(queue) => queue,
        None => device
            .new_command_queue()
            .map_err(|error| error.to_string())?,
    };
    let primitive = match options.get("primitive") {
        Some(name) => primitive_type(name)?,
        None => match template.input_primitive_topology {
            PrimitiveTopologyClass::Point => PrimitiveType::Point,
            PrimitiveTopologyClass::Line => PrimitiveType::Line,
            _ => PrimitiveType::Triangle,
        },
    };
    let depth_stencil = match options.get("depth-test").map(String::as_str) {
        None => None,
        Some("less") => {
            let mut less = DepthStencilDescriptor::default();
            less.depth_compare_function = CompareFunction::Less;
            less.depth_write_enabled = false;
            Some(
                device
                    .new_depth_stencil_state(&less)
                    .map_err(|error| format!("depth-stencil state: {error}"))?,
            )
        }
        Some(other) => return Err(format!("--depth-test {other:?}: less").into()),
    };
    Ok(Setup {
        device,
        queue,
        state,
        reflection,
        template,
        primitive,
        created,
        created_again,
        queue_first,
        depth_stencil,
    })
}

/// A texture a draw reads or writes, with the bytes it starts as (one upload per slice).
struct BoundTexture {
    index: u32,
    texture: Texture,
    written: bool,
    descriptor: TextureDescriptor,
    /// Each slice's and level's starting bytes: `(slice, level, bytes)`.
    uploads: Vec<(u64, u64, Buffer)>,
    /// For a two-plane YCbCr texture, the shared buffer standing for its surface
    /// ([`ycbcr_surface`]), which holds its starting bytes (it then has no uploads), kept alive
    /// with the texture.
    _surface: Option<Buffer>,
}

/// How a run's samplers filter.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SamplerMode {
    /// Nearest, one level.
    Nearest,
    /// Linear minification and magnification, one level.
    Linear,
    /// Linear, and linear between the levels of a full mipmap chain.
    Mip,
    /// [`SamplerMode::Mip`] with an anisotropy of 16.
    Anisotropic,
    /// Nearest, pixel coordinates, one level.
    NearestPixels,
    /// Linear, pixel coordinates, one level.
    LinearPixels,
}

impl SamplerMode {
    fn parse(text: &str) -> Result<SamplerMode, String> {
        match text {
            "nearest" => Ok(SamplerMode::Nearest),
            "linear" => Ok(SamplerMode::Linear),
            "mip" => Ok(SamplerMode::Mip),
            "aniso" => Ok(SamplerMode::Anisotropic),
            "nearest-px" => Ok(SamplerMode::NearestPixels),
            "linear-px" => Ok(SamplerMode::LinearPixels),
            _ => Err(format!(
                "sampler mode {text:?}: nearest, linear, mip, aniso, nearest-px or linear-px"
            )),
        }
    }

    fn name(self) -> &'static str {
        match self {
            SamplerMode::Nearest => "nearest",
            SamplerMode::Linear => "linear",
            SamplerMode::Mip => "mip",
            SamplerMode::Anisotropic => "aniso",
            SamplerMode::NearestPixels => "nearest-px",
            SamplerMode::LinearPixels => "linear-px",
        }
    }

    /// Whether the textures a draw only reads carry a full mipmap chain.
    fn mipmapped(self) -> bool {
        matches!(self, SamplerMode::Mip | SamplerMode::Anisotropic)
    }
}

/// How a run samples: the format of every texture of a float data type the draw only reads
/// (else each slot's turn among [`formats`]), and its samplers' filtering.
#[derive(Clone, Copy, Debug)]
struct Sampling {
    format: Option<PixelFormat>,
    mode: SamplerMode,
    /// A texture slot bound to its own format in place of `format`.
    slot: Option<(u32, PixelFormat)>,
}

impl Sampling {
    /// `--combos`: `FORMAT:MODE[@SLOT=FORMAT],...`, each format a `MTLPixelFormat` name without
    /// its prefix (`RGBA8Unorm_sRGB`, `BC1_RGBA`).
    fn combos(text: &str) -> Result<Vec<Sampling>, String> {
        let pixel_format = |combo: &str, format: &str| {
            PixelFormat::from_name(&format!("MTLPixelFormat{format}"))
                .ok_or_else(|| format!("combo {combo:?}: no pixel format {format}"))
        };
        text.split(',')
            .filter(|combo| !combo.is_empty())
            .map(|combo| {
                let (sampled, slot) = match combo.split_once('@') {
                    Some((sampled, slot)) => (sampled, Some(slot)),
                    None => (combo, None),
                };
                let (format, mode) = sampled
                    .split_once(':')
                    .ok_or_else(|| format!("combo {combo:?}: FORMAT:MODE"))?;
                let slot = match slot {
                    None => None,
                    Some(slot) => {
                        let (index, format) = slot
                            .split_once('=')
                            .ok_or_else(|| format!("combo {combo:?}: @SLOT=FORMAT"))?;
                        let index = index
                            .parse::<u32>()
                            .map_err(|_| format!("combo {combo:?}: slot {index}"))?;
                        Some((index, pixel_format(combo, format)?))
                    }
                };
                Ok(Sampling {
                    format: Some(pixel_format(combo, format)?),
                    mode: SamplerMode::parse(mode)?,
                    slot,
                })
            })
            .collect()
    }
}

/// The bytes of one row and the rows of a `width` x `height` image of `format`: texel rows, or
/// block rows of a block-compressed format.
fn layout(format: PixelFormat, width: u64, height: u64) -> Option<(u64, u64)> {
    use PixelFormat as P;
    match format {
        P::BC1RGBA | P::BC1RGBASRGB => Some((width.div_ceil(4) * 8, height.div_ceil(4))),
        _ => Some((texel_bytes(format)? * width, height)),
    }
}

/// Whether a texture of data type `data` takes the float formats of [`formats`].
fn float_data(data: DataType) -> bool {
    !matches!(
        data,
        DataType::Int
            | DataType::Short
            | DataType::Char
            | DataType::UInt
            | DataType::UShort
            | DataType::UChar
    )
}

/// A color attachment, with the bytes it starts as.
struct BoundAttachment {
    index: u32,
    texture: Texture,
    texel: u64,
    upload: Buffer,
    /// For a two-plane extended-range attachment, the shared buffer standing for its surface
    /// ([`surface_planes`]): it starts as the palette's bytes and is read back whole, both
    /// planes and the bytes between them. The attachment then has no upload or readback of its
    /// own.
    surface: Option<Buffer>,
}

/// One run's inputs.
struct Inputs {
    buffers: Vec<(u32, Buffer, bool)>,
    textures: Vec<BoundTexture>,
    samplers: Vec<(u32, SamplerState)>,
    attachments: Vec<BoundAttachment>,
    depth: Option<Texture>,
    stencil: Option<Texture>,
    side: u64,
    described: String,
}

/// The inputs of run `run` of `setup`'s draw.
fn inputs(
    setup: &Setup,
    palette: Palette,
    run: u64,
    sampling: Sampling,
    side: u64,
) -> Result<Inputs, String> {
    let device = &setup.device;
    let e = |what: String| move |error: metal2vulkan_runtime::Error| format!("{what}: {error}");
    let mut inputs = Inputs {
        buffers: Vec::new(),
        textures: Vec::new(),
        samplers: Vec::new(),
        attachments: Vec::new(),
        depth: None,
        stencil: None,
        side,
        described: format!("palette={palette:?} sampler={}", sampling.mode.name()),
    };
    let upload = |stream: u64, len: u64| {
        device
            .new_buffer_with_bytes(&bytes(palette, stream, len), ResourceOptions::shared())
            .map_err(e(format!("an upload of {len} bytes")))
    };
    for binding in setup.bindings() {
        let index = binding.index();
        let written = binding.access() != BindingAccess::ReadOnly;
        let kind: u64 = match binding.binding_type() {
            BindingType::Buffer => 1,
            BindingType::Texture => 2,
            _ => 3,
        };
        let stream = run << 32 | kind << 24 | u64::from(index) << 8;
        match binding.binding_type() {
            BindingType::Buffer => {
                let size = binding
                    .buffer()
                    .map_or(0, |buffer| buffer.buffer_data_size())
                    .next_multiple_of(4096)
                    .max(BUFFER_BYTES);
                inputs.buffers.push((index, upload(stream, size)?, written));
            }
            BindingType::Texture => {
                let reflected = binding.texture().ok_or("a texture without its type")?;
                let ty = reflected.texture_type();
                for element in 0..reflected.array_length().max(1) {
                    let slot = index + element;
                    let data = reflected.texture_data_type();
                    let depth = reflected.is_depth_texture();
                    let choices = formats(data, depth);
                    let chosen = !written && !depth && float_data(data);
                    let format = match (sampling.slot, sampling.format) {
                        (Some((at, format)), _) if chosen && at == slot => format,
                        (_, Some(format)) if chosen => format,
                        _ => choices[(run as usize + slot as usize) % choices.len()],
                    };
                    // A two-plane YCbCr texture stands over a surface of seeded bytes, as a
                    // video frame's does, one level.
                    if let Some((row, chroma_at, length)) = ycbcr_surface(format, side) {
                        if ty != TextureType::Type2D || !chosen {
                            return Err(format!(
                                "texture {slot}: {} at a slot not a sampled 2D one",
                                format.name()
                            ));
                        }
                        let surface = upload(stream | u64::from(element), length)?;
                        let mut d = TextureDescriptor::texture_2d(format, side, side, false);
                        d.usage = TextureUsage::SHADER_READ;
                        d.resource_options = ResourceOptions::shared();
                        let facts = IoSurfaceYCbCr::new(YUV420V, 0, [side, side], chroma_at, row);
                        let plane = IoSurfacePlane::new(0, row, true).with_ycbcr(facts);
                        let texture = surface
                            .new_texture_with_iosurface_plane(&d, plane)
                            .map_err(e(format!("texture {slot} of {}", format.name())))?;
                        inputs
                            .described
                            .push_str(&format!(" t{slot}={}", format.name()));
                        inputs.textures.push(BoundTexture {
                            index: slot,
                            texture,
                            written,
                            descriptor: d,
                            uploads: Vec::new(),
                            _surface: Some(surface),
                        });
                        continue;
                    }
                    // A two-plane extended-range texture stands over a `'b3a8'` surface of seeded
                    // bytes, as the consumer's drawables are sampled, one level.
                    if let Some((color_row, alpha_at, alpha_row, length)) =
                        surface_planes(format, side)
                    {
                        if ty != TextureType::Type2D || !chosen {
                            return Err(format!(
                                "texture {slot}: {} at a slot not a sampled 2D one",
                                format.name()
                            ));
                        }
                        let surface = upload(stream | u64::from(element), length)?;
                        let mut d = TextureDescriptor::texture_2d(format, side, side, false);
                        d.usage = TextureUsage::SHADER_READ;
                        d.resource_options = ResourceOptions::shared();
                        let facts = IoSurfaceYCbCr::new(B3A8, 0, [side, side], alpha_at, alpha_row);
                        let plane = IoSurfacePlane::new(0, color_row, true).with_ycbcr(facts);
                        let texture = surface
                            .new_texture_with_iosurface_plane(&d, plane)
                            .map_err(e(format!("texture {slot} of {}", format.name())))?;
                        inputs
                            .described
                            .push_str(&format!(" t{slot}={}", format.name()));
                        inputs.textures.push(BoundTexture {
                            index: slot,
                            texture,
                            written,
                            descriptor: d,
                            uploads: Vec::new(),
                            _surface: Some(surface),
                        });
                        continue;
                    }
                    let mipmapped = sampling.mode.mipmapped() && !written;
                    let mut d = TextureDescriptor::texture_2d(format, side, side, mipmapped);
                    d.texture_type = ty;
                    match ty {
                        TextureType::Type1D | TextureType::Type1DArray => d.height = 1,
                        TextureType::Type3D => d.depth = 4,
                        _ => {}
                    }
                    if matches!(
                        ty,
                        TextureType::Type1DArray
                            | TextureType::Type2DArray
                            | TextureType::CubeArray
                    ) {
                        d.array_length = 2;
                    }
                    d.usage = if written {
                        TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE
                    } else {
                        TextureUsage::SHADER_READ
                    };
                    d.resource_options = ResourceOptions::private();
                    let texture = device
                        .new_texture(&d)
                        .map_err(e(format!("texture {slot} of {}", format.name())))?;
                    let slices = match ty {
                        TextureType::Cube => 6,
                        TextureType::CubeArray => 6 * d.array_length,
                        _ => d.array_length,
                    };
                    let mut uploads = Vec::new();
                    for slice in 0..slices {
                        for level in 0..d.mipmap_level_count.max(1) {
                            let (row, rows) = layout(
                                format,
                                (d.width >> level).max(1),
                                (d.height >> level).max(1),
                            )
                            .ok_or("a texel's size")?;
                            let image = row * rows * (d.depth >> level).max(1);
                            let stream = stream | u64::from(element) | slice << 16 | level << 4;
                            uploads.push((slice, level, upload(stream, image)?));
                        }
                    }
                    inputs
                        .described
                        .push_str(&format!(" t{slot}={}", format.name()));
                    inputs.textures.push(BoundTexture {
                        index: slot,
                        texture,
                        written,
                        descriptor: d,
                        uploads,
                        _surface: None,
                    });
                }
            }
            BindingType::Sampler => {
                let mut d = SamplerDescriptor::new();
                if !matches!(
                    sampling.mode,
                    SamplerMode::Nearest | SamplerMode::NearestPixels
                ) {
                    d.min_filter = SamplerMinMagFilter::Linear;
                    d.mag_filter = SamplerMinMagFilter::Linear;
                }
                if matches!(
                    sampling.mode,
                    SamplerMode::NearestPixels | SamplerMode::LinearPixels
                ) {
                    d.normalized_coordinates = false;
                }
                if sampling.mode.mipmapped() {
                    d.mip_filter = SamplerMipFilter::Linear;
                }
                if sampling.mode == SamplerMode::Anisotropic {
                    d.max_anisotropy = 16;
                }
                let sampler = device
                    .new_sampler_state(&d)
                    .map_err(e(format!("sampler {index}")))?;
                inputs.samplers.push((index, sampler));
            }
            other => return Err(format!("{} at {index}", other.name())),
        }
    }
    for (&index, attachment) in &setup.template.color_attachments {
        let format = attachment.pixel_format;
        let stream = run << 32 | 4 << 24 | u64::from(index) << 8;
        // A two-plane extended-range attachment stands over a surface's planes, as the
        // consumer's drawables do; its surface starts as the palette's bytes.
        if let Some((color_row, alpha_at, alpha_row, length)) = surface_planes(format, side) {
            let surface = upload(stream, length)?;
            let mut d = TextureDescriptor::texture_2d(format, side, side, false);
            d.usage = TextureUsage::RENDER_TARGET;
            d.resource_options = ResourceOptions::shared();
            let plane = IoSurfacePlane::new(0, color_row, true).with_ycbcr(IoSurfaceYCbCr::new(
                B3A8,
                0,
                [side, side],
                alpha_at,
                alpha_row,
            ));
            let texture = surface
                .new_texture_with_iosurface_plane(&d, plane)
                .map_err(e(format!("attachment {index} of {}", format.name())))?;
            inputs.attachments.push(BoundAttachment {
                index,
                texture,
                texel: 4,
                upload: surface.clone(),
                surface: Some(surface),
            });
            continue;
        }
        let texel = texel_bytes(format).ok_or("a texel's size")?;
        let mut d = TextureDescriptor::texture_2d(format, side, side, false);
        d.usage = TextureUsage::RENDER_TARGET;
        d.resource_options = ResourceOptions::private();
        let texture = device
            .new_texture(&d)
            .map_err(e(format!("attachment {index} of {}", format.name())))?;
        inputs.attachments.push(BoundAttachment {
            index,
            texture,
            texel,
            upload: upload(stream, texel * side * side)?,
            surface: None,
        });
    }
    let target = |format: PixelFormat| {
        let mut d = TextureDescriptor::texture_2d(format, side, side, false);
        d.usage = TextureUsage::RENDER_TARGET;
        d.resource_options = ResourceOptions::private();
        device.new_texture(&d).map_err(e(format!(
            "a depth or stencil attachment of {}",
            format.name()
        )))
    };
    if let Some(format) = setup.template.depth_format {
        inputs.depth = Some(target(format)?);
    }
    if let Some(format) = setup.template.stencil_format {
        inputs.stencil = Some(match (&inputs.depth, setup.template.depth_format) {
            (Some(depth), Some(depth_format)) if depth_format == format => depth.clone(),
            _ => target(format)?,
        });
    }
    Ok(inputs)
}

/// Encode the uploads that give `inputs`'s textures and attachments their starting bytes.
fn encode_uploads(command_buffer: &CommandBuffer, inputs: &Inputs) -> Result<(), String> {
    let e = |error: metal2vulkan_runtime::Error| format!("upload: {error}");
    let mut blit = command_buffer.blit_command_encoder().map_err(e)?;
    for bound in &inputs.textures {
        let d = &bound.descriptor;
        for &(slice, level, ref upload) in &bound.uploads {
            let (width, height, depth) = (
                (d.width >> level).max(1),
                (d.height >> level).max(1),
                (d.depth >> level).max(1),
            );
            let (row, rows) = layout(d.pixel_format, width, height).ok_or("a texel's size")?;
            blit.copy_from_buffer_to_texture(
                upload,
                0,
                row,
                row * rows,
                Size::new(width, height, depth),
                &bound.texture,
                slice,
                level,
                Origin::new(0, 0, 0),
                BlitOption::None,
            )
            .map_err(e)?;
        }
    }
    for attachment in inputs.attachments.iter().filter(|a| a.surface.is_none()) {
        let row = attachment.texel * inputs.side;
        blit.copy_from_buffer_to_texture(
            &attachment.upload,
            0,
            row,
            row * inputs.side,
            Size::new(inputs.side, inputs.side, 1),
            &attachment.texture,
            0,
            0,
            Origin::new(0, 0, 0),
            BlitOption::None,
        )
        .map_err(e)?;
    }
    blit.end_encoding().map_err(e)
}

/// Encode one draw of `setup`'s pipeline with `inputs`.
fn encode_draw(
    command_buffer: &CommandBuffer,
    setup: &Setup,
    inputs: &Inputs,
) -> Result<(), String> {
    encode_draw_as(command_buffer, setup, inputs, setup.primitive)
}

/// Encode one draw of `setup`'s pipeline with `inputs`, of `primitive`.
fn encode_draw_as(
    command_buffer: &CommandBuffer,
    setup: &Setup,
    inputs: &Inputs,
    primitive: PrimitiveType,
) -> Result<(), String> {
    let e = |error: metal2vulkan_runtime::Error| format!("draw: {error}");
    let mut pass = RenderPassDescriptor::new();
    for attachment in &inputs.attachments {
        let color = pass
            .color_attachments
            .get_mut(attachment.index as usize)
            .ok_or("a color attachment index out of range")?;
        color.texture = Some(attachment.texture.clone());
        color.load_action = LoadAction::Load;
        color.store_action = StoreAction::Store;
    }
    if let Some(depth) = &inputs.depth {
        pass.depth_attachment.texture = Some(depth.clone());
        pass.depth_attachment.load_action = LoadAction::Clear;
        pass.depth_attachment.store_action = StoreAction::DontCare;
    }
    if let Some(stencil) = &inputs.stencil {
        pass.stencil_attachment.texture = Some(stencil.clone());
        pass.stencil_attachment.load_action = LoadAction::Clear;
        pass.stencil_attachment.store_action = StoreAction::DontCare;
    }
    let mut encoder = command_buffer.render_command_encoder(&pass).map_err(e)?;
    encoder.set_render_pipeline_state(&setup.state).map_err(e)?;
    if let Some(depth_stencil) = &setup.depth_stencil {
        encoder.set_depth_stencil_state(depth_stencil).map_err(e)?;
    }
    for (index, buffer, _) in &inputs.buffers {
        encoder
            .set_fragment_buffer(Some(buffer), 0, *index)
            .map_err(e)?;
    }
    for bound in &inputs.textures {
        encoder
            .set_fragment_texture(Some(&bound.texture), bound.index)
            .map_err(e)?;
    }
    for (index, sampler) in &inputs.samplers {
        encoder
            .set_fragment_sampler_state(Some(sampler), *index)
            .map_err(e)?;
    }
    encoder.draw_primitives(primitive, 0, 3, 1, 0).map_err(e)?;
    encoder.end_encoding().map_err(e)
}

/// Encode the copies of every attachment, then every texture the fragment may write, into new
/// shared buffers, in that order.
fn encode_readback(
    command_buffer: &CommandBuffer,
    device: &Device,
    inputs: &Inputs,
) -> Result<Vec<Buffer>, String> {
    let e = |error: metal2vulkan_runtime::Error| format!("readback: {error}");
    let mut blit = command_buffer.blit_command_encoder().map_err(e)?;
    let mut out = Vec::new();
    let mut copy = |texture: &Texture, texel: u64, size: Size, slices: u64| {
        let row = texel * size.width;
        let image = row * size.height * size.depth;
        for slice in 0..slices {
            let buffer = device
                .new_buffer(image, ResourceOptions::shared())
                .map_err(e)?;
            blit.copy_from_texture_to_buffer(
                texture,
                slice,
                0,
                Origin::new(0, 0, 0),
                size,
                &buffer,
                0,
                row,
                row * size.height,
                BlitOption::None,
            )
            .map_err(e)?;
            out.push(buffer);
        }
        Ok::<(), String>(())
    };
    for attachment in inputs.attachments.iter().filter(|a| a.surface.is_none()) {
        copy(
            &attachment.texture,
            attachment.texel,
            Size::new(inputs.side, inputs.side, 1),
            1,
        )?;
    }
    for bound in inputs.textures.iter().filter(|bound| bound.written) {
        let d = &bound.descriptor;
        let texel = texel_bytes(d.pixel_format).ok_or("a texel's size")?;
        copy(
            &bound.texture,
            texel,
            Size::new(d.width, d.height, d.depth),
            bound.uploads.len() as u64,
        )?;
    }
    blit.end_encoding().map_err(e)?;
    Ok(out)
}

/// Commit `command_buffer` and wait for it to complete.
fn finish(command_buffer: &CommandBuffer) -> Result<(), String> {
    command_buffer
        .commit()
        .map_err(|error| format!("commit: {error}"))?;
    command_buffer
        .wait_until_completed_for(WAIT)
        .map_err(|_| format!("the command buffer did not complete within {WAIT:?}"))?;
    match command_buffer.status() {
        CommandBufferStatus::Completed => Ok(()),
        status => Err(format!(
            "the command buffer ended {status:?}: {:?}",
            command_buffer.error()
        )),
    }
}

/// The bytes of `buffer`, which no command buffer writes any more.
fn contents(buffer: &Buffer) -> Result<Vec<u8>, String> {
    let at = buffer.contents().ok_or("a buffer without host contents")?;
    let length = usize::try_from(buffer.length()).map_err(|error| error.to_string())?;
    // SAFETY: the buffer is shared, `contents` holds its `length` bytes, and the command buffer
    // that wrote it completed.
    Ok(unsafe { std::slice::from_raw_parts(at.as_ptr(), length) }.to_vec())
}

/// Why a drawn image's bytes are no evidence of what the fragment computes. A draw that wrote
/// nothing, or one constant, hashes alike whatever its inputs and in every build, so two legs
/// agreeing on it says nothing about their answers (an A/B over the consumer's draws once read
/// such a hash, shared by every input, as agreement).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Blank {
    /// Every texel holds the same bytes.
    Uniform,
    /// The image holds the bytes it started as (attachments load the palette's bytes): the draw
    /// changed nothing in it.
    Unchanged,
}

impl Blank {
    fn name(self) -> &'static str {
        match self {
            Blank::Uniform => "uniform",
            Blank::Unchanged => "unchanged",
        }
    }
}

/// The [`Blank`] flags of one image, none where its bytes may carry evidence: `after` holds
/// `rows` rows of `width` texels of `texel` bytes, a row every `row` bytes (the bytes between
/// are padding), and `before` is what the image held before the draw, laid out alike.
fn blank(
    before: &[u8],
    after: &[u8],
    texel: usize,
    row: usize,
    width: usize,
    rows: usize,
) -> Vec<Blank> {
    let texel = texel.max(1);
    let mut texels = (0..rows).flat_map(|y| {
        after
            .get(y * row..y * row + width * texel)
            .unwrap_or_default()
            .chunks_exact(texel)
    });
    let mut flags = Vec::new();
    if let Some(first) = texels.next() {
        if texels.all(|other| other == first) {
            flags.push(Blank::Uniform);
        }
    }
    if before == after {
        flags.push(Blank::Unchanged);
    }
    flags
}

/// The run table's images column: `images=` and, for each color attachment, its index and its
/// flags joined by `+`, or `drawn` where it has none (`images=0:uniform+unchanged,1:drawn`);
/// `images=none` without a color attachment.
fn images_column(images: &[(u32, Vec<Blank>)]) -> String {
    if images.is_empty() {
        return "images=none".to_string();
    }
    let each: Vec<String> = images
        .iter()
        .map(|(index, flags)| match flags.as_slice() {
            [] => format!("{index}:drawn"),
            flags => {
                let names: Vec<&str> = flags.iter().map(|flag| flag.name()).collect();
                format!("{index}:{}", names.join("+"))
            }
        })
        .collect();
    format!("images={}", each.join(","))
}

/// What one run drew.
struct Drawn {
    /// Everything the draw wrote.
    bytes: Vec<u8>,
    /// Each color attachment's index and [`Blank`] flags, in the attachments' order.
    images: Vec<(u32, Vec<Blank>)>,
}

/// One run: upload, draw, read back; everything the draw wrote, and whether each image can
/// carry evidence.
fn draw_once(setup: &Setup, inputs: &Inputs) -> Result<Drawn, String> {
    let command_buffer = setup
        .queue
        .command_buffer()
        .map_err(|error| format!("command buffer: {error}"))?;
    // A two-plane attachment's surface is its own upload, which the draw writes: its bytes
    // before the draw, for its flags.
    let surfaces_before = inputs
        .attachments
        .iter()
        .filter_map(|a| a.surface.as_ref())
        .map(contents)
        .collect::<Result<Vec<_>, _>>()?;
    encode_uploads(&command_buffer, inputs)?;
    encode_draw(&command_buffer, setup, inputs)?;
    let readbacks = encode_readback(&command_buffer, &setup.device, inputs)?;
    finish(&command_buffer)?;
    let readbacks = readbacks
        .iter()
        .map(contents)
        .collect::<Result<Vec<_>, _>>()?;
    let written = inputs
        .buffers
        .iter()
        .filter(|(_, _, written)| *written)
        .map(|(_, buffer, _)| contents(buffer))
        .collect::<Result<Vec<_>, _>>()?;
    // Then each two-plane attachment's surface, whole.
    let surfaces = inputs
        .attachments
        .iter()
        .filter_map(|a| a.surface.as_ref())
        .map(contents)
        .collect::<Result<Vec<_>, _>>()?;
    // Each color attachment's flags: the readbacks start with the attachments' that have no
    // surface, in order; a surface's color plane rows are padded (`surface_planes`).
    let side = usize::try_from(inputs.side).map_err(|error| error.to_string())?;
    let (mut plain, mut over) = (readbacks.iter(), surfaces_before.iter().zip(&surfaces));
    let mut images = Vec::new();
    for attachment in &inputs.attachments {
        let texel = usize::try_from(attachment.texel).map_err(|error| error.to_string())?;
        let flags = match &attachment.surface {
            None => {
                let after = plain.next().ok_or("an attachment without its readback")?;
                let before = contents(&attachment.upload)?;
                blank(&before, after, texel, texel * side, side, side)
            }
            Some(_) => {
                let (before, after) = over.next().ok_or("a surface without its bytes")?;
                let (color_row, ..) =
                    surface_planes(attachment.texture.pixel_format(), inputs.side)
                        .ok_or("a surface attachment of a single-plane format")?;
                let row = usize::try_from(color_row).map_err(|error| error.to_string())?;
                blank(before, after, texel, row, side, side)
            }
        };
        images.push((attachment.index, flags));
    }
    let bytes = readbacks
        .iter()
        .chain(&written)
        .chain(&surfaces)
        .flatten()
        .copied()
        .collect();
    Ok(Drawn { bytes, images })
}

/// A table of one draw's lines, each written as it ends.
struct Table {
    id: String,
    file: std::fs::File,
}

impl Table {
    fn create(out: &Path, id: &str, suffix: &str) -> Result<Table, String> {
        std::fs::create_dir_all(out)
            .map_err(|error| format!("create {}: {error}", out.display()))?;
        let path = out.join(format!("{id}{suffix}"));
        let file = std::fs::File::create(&path)
            .map_err(|error| format!("create {}: {error}", path.display()))?;
        Ok(Table {
            id: id.to_string(),
            file,
        })
    }

    fn line(&mut self, fields: &[&str]) -> Result<(), String> {
        let fields: Vec<String> = fields
            .iter()
            .map(|field| field.replace(['\t', '\n'], " "))
            .collect();
        writeln!(self.file, "{}\t{}", self.id, fields.join("\t"))
            .map_err(|error| format!("write the table: {error}"))
    }

    /// The line of a draw that stopped before it ran. `Err` when the harness itself failed;
    /// when the draw was skipped, unless `skip_ok` (`--skip-ok yes`); and when the runtime
    /// refused it where `refusal_fails` (`time`, which then measured nothing). `run` keeps a
    /// refusal as its typed outcome, which the A/B compares.
    fn stopped(&mut self, stop: Stop, skip_ok: bool, refusal_fails: bool) -> Result<(), Failure> {
        match stop {
            Stop::Skip(why) => {
                self.line(&["-", "SKIP", &why])?;
                if skip_ok {
                    Ok(())
                } else {
                    Err(Failure::Skipped(format!("{}: SKIP: {why}", self.id)))
                }
            }
            Stop::Refused(why) => {
                self.line(&["-", "REFUSED", &why])?;
                if refusal_fails {
                    Err(Failure::Untimed(format!("{}: REFUSED: {why}", self.id)))
                } else {
                    Ok(())
                }
            }
            Stop::Fatal(error) => Err(Failure::Harness(error)),
        }
    }
}

fn run(args: &[String]) -> Result<(), Failure> {
    let options = options(args)?;
    let skip_ok = skip_ok(&options)?;
    let id = required(&options, "id")?.to_string();
    let out = PathBuf::from(required(&options, "out")?);
    let runs = number(&options, "runs", 12)?;
    let side = number(&options, "side", 32)?;
    let combos = options
        .get("combos")
        .map(|text| Sampling::combos(text))
        .transpose()?
        .unwrap_or_default();
    let mut table = Table::create(&out, &id, ".tsv")?;
    let setup = match setup(&options) {
        Ok(setup) => setup,
        Err(stop) => return table.stopped(stop, skip_ok, false),
    };
    let dir = out.join(&id);
    std::fs::create_dir_all(&dir).map_err(|error| format!("create {}: {error}", dir.display()))?;
    for r in 0..runs {
        let palette =
            [Palette::Specials, Palette::SmallIntegers, Palette::Random][(r % 3) as usize];
        // Each run its combo in turn; without `--combos`, every other triple filters linearly.
        let sampling = match combos.get(r as usize % combos.len().max(1)) {
            Some(&sampling) => sampling,
            None => Sampling {
                format: None,
                mode: if (r / 3) % 2 == 1 {
                    SamplerMode::Linear
                } else {
                    SamplerMode::Nearest
                },
                slot: None,
            },
        };
        let name = r.to_string();
        let inputs = match inputs(&setup, palette, r, sampling, side) {
            Ok(inputs) => inputs,
            Err(why) => {
                table.line(&[&name, "REFUSED", &format!("inputs: {why}")])?;
                continue;
            }
        };
        match draw_once(&setup, &inputs) {
            Ok(Drawn { bytes, images }) => {
                write(&dir.join(format!("run-{r}.bin")), &bytes)?;
                let digest = metal2vulkan_validation::hash::hex_encode(Sha256::digest(&bytes));
                let column = images_column(&images);
                if images.iter().any(|(_, flags)| !flags.is_empty()) {
                    eprintln!(
                        "run {r}: an image is uniform or unchanged ({column}): its bytes are no \
                         evidence of the fragment's answers"
                    );
                }
                table.line(&[
                    &name,
                    "OK",
                    &digest,
                    &bytes.len().to_string(),
                    &inputs.described,
                    &column,
                ])?;
            }
            Err(why) => table.line(&[&name, "REFUSED", &why, &inputs.described])?,
        }
    }
    work_counts(&setup);
    Ok(())
}

fn time(args: &[String]) -> Result<(), Failure> {
    let options = options(args)?;
    let skip_ok = skip_ok(&options)?;
    let id = required(&options, "id")?.to_string();
    let out = PathBuf::from(required(&options, "out")?);
    let side = number(&options, "side", 512)?;
    let repeats = number(&options, "repeats", 30)?;
    let mut table = Table::create(&out, &id, ".time.tsv")?;
    // The quality of service first, before the device and its threads.
    let qos_at_start = storm::qos_now();
    let qos = options.get("qos").map(String::as_str);
    // `--device-qos`: the device opens at its class, then the thread takes `--qos` (`setup`).
    let device_qos = options.get("device-qos").map(String::as_str);
    if let Some(class) = device_qos.or(qos) {
        storm::set_qos(class)?;
    }
    table.line(&["qos", &qos_at_start, qos.unwrap_or("not set")])?;
    table.line(&["device_qos", device_qos.unwrap_or("as qos")])?;
    let setup = match setup(&options) {
        Ok(setup) => setup,
        Err(stop) => return table.stopped(stop, skip_ok, true),
    };
    // What `time` was asked to measure and the runtime refused: the status is 4 at the end.
    let mut untimed = Vec::new();
    // `--combo FORMAT:MODE`: the textures and samplers `run --combos` binds for that combo; else
    // linear samplers and each slot's turn of formats.
    let linear = match options.get("combo") {
        Some(text) => match Sampling::combos(text)?.as_slice() {
            [one] => *one,
            _ => return Err(format!("--combo {text:?}: one combo").into()),
        },
        None => Sampling {
            format: None,
            mode: SamplerMode::Linear,
            slot: None,
        },
    };
    let inputs = match inputs(&setup, Palette::Timing, 0, linear, side) {
        Ok(inputs) => inputs,
        Err(why) => {
            let why = format!("inputs: {why}");
            table.line(&["-", "REFUSED", &why])?;
            return Err(Failure::Untimed(format!("{id}: REFUSED: {why}")));
        }
    };
    let ms = |d: Duration| format!("{:.3}", d.as_secs_f64() * 1e3);
    let command_buffer = |what: &str| {
        setup
            .queue
            .command_buffer()
            .map_err(|error| format!("{what}: {error}"))
    };
    // `--prime-qos CLASS`: before the first draw, a thread of its own at `CLASS` draws the state
    // as points and waits: a pipeline of another class, which the device's pool makes, so a pool
    // thread it starts is started from that thread.
    if let Some(class) = options.get("prime-qos") {
        let start = Instant::now();
        let primed = std::thread::scope(|scope| {
            scope
                .spawn(|| -> Result<(), String> {
                    storm::set_qos(class)?;
                    let points = setup
                        .queue
                        .command_buffer()
                        .map_err(|error| format!("prime: {error}"))?;
                    encode_draw_as(&points, &setup, &inputs, PrimitiveType::Point)?;
                    finish(&points)
                })
                .join()
                .unwrap_or_else(|_| Err("the prime thread panicked".to_string()))
        });
        match primed {
            Ok(()) => table.line(&["prime_ms", &ms(start.elapsed())])?,
            Err(why) => {
                table.line(&["prime", "REFUSED", &why])?;
                untimed.push(format!("prime: {why}"));
            }
        }
    }
    // The first draw: the uploads, then the draw, whose wall time includes any variant made at
    // the draw.
    let then_lines = options.get("then-lines").map(String::as_str);
    let first = command_buffer("first draw")?;
    encode_uploads(&first, &inputs)?;
    let start = Instant::now();
    // `--then-lines together`: the first command buffer draws lines too, so both pipelines are
    // made at once.
    let drawn = encode_draw(&first, &setup, &inputs).and_then(|()| match then_lines {
        Some("together") => encode_draw_as(&first, &setup, &inputs, PrimitiveType::Line),
        _ => Ok(()),
    });
    if let Err(why) = drawn.and_then(|()| finish(&first)) {
        table.line(&["-", "REFUSED", &why])?;
        return Err(Failure::Untimed(format!("{id}: REFUSED: {why}")));
    }
    let first_ms = ms(start.elapsed());
    table.line(&["created_ms", &ms(setup.created)])?;
    if let Some(again) = setup.created_again {
        table.line(&["created_again_ms", &ms(again)])?;
    }
    if let Some(queue) = setup.queue_first {
        table.line(&["queue_first_ms", &ms(queue)])?;
    }
    table.line(&["first_draw_ms", &first_ms])?;
    // `--then-lines yes`: one more draw, of lines, timed as the first was (the pipeline of the
    // other topology, where the state's topology class allows lines).
    if then_lines == Some("yes") {
        let lines = command_buffer("line draw")?;
        let start = Instant::now();
        match encode_draw_as(&lines, &setup, &inputs, PrimitiveType::Line)
            .and_then(|()| finish(&lines))
        {
            Ok(()) => table.line(&["line_draw_ms", &ms(start.elapsed())])?,
            Err(why) => {
                table.line(&["line_draw", "REFUSED", &why])?;
                untimed.push(format!("line draw: {why}"));
            }
        }
    }
    // `--then TYPE`: one more draw, of `TYPE`, timed as the first was: a strip after its class's
    // list takes the list's pipeline where the device makes the topology dynamic.
    if let Some(name) = options.get("then") {
        let then = primitive_type(name)?;
        let next = command_buffer("then draw")?;
        let start = Instant::now();
        match encode_draw_as(&next, &setup, &inputs, then).and_then(|()| finish(&next)) {
            Ok(()) => table.line(&["then_draw_ms", &ms(start.elapsed())])?,
            Err(why) => {
                table.line(&["then_draw", "REFUSED", &why])?;
                untimed.push(format!("then draw: {why}"));
            }
        }
    }
    let (mut gpu, mut wall) = (Vec::new(), Vec::new());
    for warm in 0..repeats + 3 {
        let timed = command_buffer("timed draw")?;
        encode_draw(&timed, &setup, &inputs)?;
        let start = Instant::now();
        finish(&timed)?;
        let waited = start.elapsed();
        if warm < 3 {
            continue;
        }
        let elapsed = timed
            .gpu_end_time()
            .and_then(|end| Ok(end - timed.gpu_start_time()?))
            .map_err(|error| format!("GPU time: {error}"))?;
        gpu.push(elapsed * 1e3);
        wall.push(waited.as_secs_f64() * 1e3);
    }
    for (name, mut values) in [("gpu_ms", gpu), ("wall_ms", wall)] {
        let text: Vec<String> = values.iter().map(|value| format!("{value:.4}")).collect();
        table.line(&[name, &text.join(" ")])?;
        values.sort_by(f64::total_cmp);
        if let (Some(min), Some(median)) = (values.first(), values.get(values.len() / 2)) {
            table.line(&[
                &format!("{name}_median_min"),
                &format!("{median:.4}"),
                &format!("{min:.4}"),
            ])?;
        }
    }
    work_counts(&setup);
    if untimed.is_empty() {
        Ok(())
    } else {
        Err(Failure::Untimed(format!(
            "{id}: REFUSED: {}",
            untimed.join("; ")
        )))
    }
}

/// Print the device's pipeline work counts on stderr, `work counts: name=value ...`: the
/// `VkPipeline`s variants made, those made at creations, and those of a creation a draw took;
/// the makings handed to the pool, the pipelines linked from libraries, and the libraries made,
/// those of fragment shaders among them.
fn work_counts(setup: &Setup) {
    let counts = setup.device.work_counts();
    eprintln!(
        "work counts: variant_pipelines_made={} creation_pipelines_made={} \
         creation_pipelines_drawn={} fills_recorded={} \
         command_pools_made={} attachment_clears_recorded={} image_blits_recorded={} \
         query_result_copies_recorded={} visibility_queries_begun={} indirect_draws_recorded={} \
         pipelines_pooled={} pipelines_linked={} pipeline_libraries_made={} \
         fragment_shader_libraries_made={}",
        counts.variant_pipelines_made,
        counts.creation_pipelines_made,
        counts.creation_pipelines_drawn,
        counts.fills_recorded,
        counts.command_pools_made,
        counts.attachment_clears_recorded,
        counts.image_blits_recorded,
        counts.query_result_copies_recorded,
        counts.visibility_queries_begun,
        counts.indirect_draws_recorded,
        counts.pipelines_pooled,
        counts.pipelines_linked,
        counts.pipeline_libraries_made,
        counts.fragment_shader_libraries_made
    );
    // And its shader cache's: the makings whose caches loaded their records, the records
    // written, and which pipelines the caches held.
    let cache = setup.device.shader_cache_statistics();
    eprintln!(
        "shader cache: hits={} misses={} writes={} driver_loads={} driver_saves={} \
         driver_pipeline_hits={} driver_pipeline_misses={} driver_data_truncated={} \
         driver_data_mismatched={} rejected={} write_failures={}",
        cache.hits,
        cache.misses,
        cache.writes,
        cache.driver_loads,
        cache.driver_saves,
        cache.driver_pipeline_hits,
        cache.driver_pipeline_misses,
        cache.driver_data_truncated,
        cache.driver_data_mismatched,
        cache.rejected,
        cache.write_failures,
    );
}

#[cfg(test)]
mod tests {
    use super::{
        blank, images_column, layout, primitive_type, Blank, Failure, PixelFormat, PrimitiveType,
        SamplerMode, Sampling, Stop, Table,
    };

    /// A stopped draw writes its row and fails the command: a skip with status 3 unless skips
    /// are accepted, a refusal with status 4 where it leaves nothing measured (`time`), and the
    /// harness's own failure with status 2. A refusal `run` compares passes.
    #[test]
    fn a_stopped_draw_writes_its_row_and_fails_unless_its_command_covers_it() {
        let dir =
            std::env::temp_dir().join(format!("corpus-draw-ab-stopped-{}", std::process::id()));
        let mut table = Table::create(&dir, "d", ".tsv").unwrap();
        let mut status = |stop: Stop, skip_ok: bool, refusal_fails: bool| {
            table
                .stopped(stop, skip_ok, refusal_fails)
                .err()
                .map(|failure: Failure| failure.status())
        };
        let why = || "why".to_string();
        assert_eq!(status(Stop::Skip(why()), false, false), Some(3));
        assert_eq!(status(Stop::Skip(why()), false, true), Some(3));
        assert_eq!(status(Stop::Skip(why()), true, false), None);
        assert_eq!(status(Stop::Refused(why()), false, false), None);
        assert_eq!(status(Stop::Refused(why()), false, true), Some(4));
        assert_eq!(status(Stop::Fatal(why()), true, false), Some(2));
        drop(table);
        let rows = std::fs::read_to_string(dir.join("d.tsv")).unwrap();
        std::fs::remove_dir_all(&dir).unwrap();
        let outcomes: Vec<&str> = rows
            .lines()
            .map(|row| row.split('\t').nth(2).unwrap())
            .collect();
        assert_eq!(outcomes, ["SKIP", "SKIP", "SKIP", "REFUSED", "REFUSED"]);
    }

    /// An image is flagged uniform when every texel holds one texel's bytes, whatever its rows'
    /// padding, and unchanged when it holds the bytes it started as; one texel that differs
    /// makes it drawn. The images column names each attachment's flags.
    #[test]
    fn uniform_and_unchanged_images_are_flagged() {
        // Two rows of two 4-byte texels, a row every 12 bytes (4 bytes of padding).
        let start: Vec<u8> = (0..24).collect();
        let zero = vec![0u8; 24];
        assert_eq!(
            blank(&start, &zero, 4, 12, 2, 2),
            [Blank::Uniform],
            "a blank image"
        );
        let mut padded = vec![7u8; 24];
        padded[8..12].copy_from_slice(&[1, 2, 3, 4]);
        padded[20..24].copy_from_slice(&[5, 6, 7, 8]);
        assert_eq!(blank(&start, &padded, 4, 12, 2, 2), [Blank::Uniform]);
        assert_eq!(blank(&start, &start, 4, 12, 2, 2), [Blank::Unchanged]);
        assert_eq!(
            blank(&zero, &zero, 4, 12, 2, 2),
            [Blank::Uniform, Blank::Unchanged]
        );
        let mut drawn = zero.clone();
        drawn[17] = 1;
        assert!(blank(&start, &drawn, 4, 12, 2, 2).is_empty());
        // A difference in the padding alone changes no texel.
        let mut pad_only = zero.clone();
        pad_only[9] = 1;
        assert_eq!(blank(&start, &pad_only, 4, 12, 2, 2), [Blank::Uniform]);
        assert_eq!(
            images_column(&[(0, vec![Blank::Uniform, Blank::Unchanged]), (2, vec![])]),
            "images=0:uniform+unchanged,2:drawn"
        );
        assert_eq!(images_column(&[]), "images=none");
    }

    /// `--primitive` names Metal's five primitive types, and nothing else.
    #[test]
    fn primitive_names() {
        let names = ["point", "line", "line-strip", "triangle", "triangle-strip"];
        let got: Vec<_> = names.iter().map(|n| primitive_type(n).unwrap()).collect();
        assert_eq!(got, PrimitiveType::ALL);
        assert!(primitive_type("triangle-fan").is_err());
    }

    /// `--combos` names formats as Metal does, without the prefix, and the four sampler modes.
    #[test]
    fn combos_name_formats_and_sampler_modes() {
        let combos = Sampling::combos(
            "RGBA8Unorm_sRGB:nearest,BC1_RGBA:linear,B5G6R5Unorm:mip,RGBA16Float:aniso,\
             A8Unorm:linear-px",
        )
        .expect("combos");
        let got: Vec<_> = combos.iter().map(|c| (c.format, c.mode)).collect();
        assert_eq!(
            got,
            [
                (Some(PixelFormat::RGBA8UnormSRGB), SamplerMode::Nearest),
                (Some(PixelFormat::BC1RGBA), SamplerMode::Linear),
                (Some(PixelFormat::B5G6R5Unorm), SamplerMode::Mip),
                (Some(PixelFormat::RGBA16Float), SamplerMode::Anisotropic),
                (Some(PixelFormat::A8Unorm), SamplerMode::LinearPixels),
            ]
        );
        assert!(Sampling::combos("RGBA8Unorm").is_err());
        assert!(Sampling::combos("NoSuchFormat:linear").is_err());
        assert!(Sampling::combos("RGBA8Unorm:bilinear").is_err());
        let slotted = Sampling::combos("RGBA8Unorm:nearest@0=RGBA16Float").expect("a slot");
        assert_eq!(slotted[0].format, Some(PixelFormat::RGBA8Unorm));
        assert_eq!(slotted[0].slot, Some((0, PixelFormat::RGBA16Float)));
        assert!(combos[0].slot.is_none());
        assert!(Sampling::combos("RGBA8Unorm:nearest@0").is_err());
        assert!(Sampling::combos("RGBA8Unorm:nearest@x=RGBA16Float").is_err());
        assert!(Sampling::combos("RGBA8Unorm:nearest@0=NoSuchFormat").is_err());
    }

    /// A block-compressed level is block rows of 8 bytes a block; any other, texel rows.
    #[test]
    fn layouts_of_texels_and_blocks() {
        assert_eq!(layout(PixelFormat::BC1RGBA, 32, 32), Some((64, 8)));
        assert_eq!(layout(PixelFormat::BC1RGBA, 2, 1), Some((8, 1)));
        assert_eq!(layout(PixelFormat::RGBA16Float, 4, 2), Some((32, 2)));
    }
}
