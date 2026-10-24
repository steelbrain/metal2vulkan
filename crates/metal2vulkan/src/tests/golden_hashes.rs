//! Golden output hashes over the public fixtures × the committed profiles.
//!
//! Every `.ll` fixture at the top level of `validation/fixtures/public/` and in its `engine/`
//! directory is assembled (`llvm-as`; a typed-pointer `.air.ll` through `xcrun metal-as`) and
//! loaded with `Library::from_air`. Every entry point is compiled with `Device::compile_function`
//! on every committed profile under the default policy, with no function constants and no
//! lowering inputs. A kernel then becomes a compute pipeline (`Device::compile_compute_pipeline`
//! of the compiled function); any other stage is linked alone and relocated, since a render
//! pipeline needs a partner stage. Some fixtures add pipelines of their own: the required pair
//! (`engine/storage_texture_write`'s storage texture bound as rgba8 and as bgra8, each under
//! the choice `metal::format::storage_image_choice` makes on the profile), a vertex + fragment
//! render pipeline (`engine/stage_io`) at one and four samples, the tile pipeline of each owned
//! tile fixture (`engine/k10ts_*`) under the attachment formats its cases use, whose staged
//! SPIR-V its entry alone, compiled without tile inputs, never reaches, and
//! `engine/multisample`'s render pipelines of four samples. A fragment function of a pipeline
//! of four samples compiles under `LoweringInputs::multisampled`. Each framebuffer-fetch fixture
//! (`engine/fragment_fetch`, `exf_*`, `exf2_fetch_unwritten`, `msf_fetch_ms`) adds its fragment
//! functions under the fetch state of a pipeline of the attachment formats its cases use, which the
//! interlocked storage form needs, and each whose cases draw multisampled (`exf_fetch_blend`,
//! `exf_fetch_count`, `msf_fetch_ms`) the same state at four samples, compiled under
//! `multisampled`: the per-sample forms.
//!
//! A cell is the first 64 bits of SHA-256 over the final SPIR-V words of the subject's stage
//! modules (each prefixed by its word count), or the typed outcome class of a refusal, so a new
//! refusal or a new success moves the table too. Each row records the SHA-256 prefix of its
//! fixture's source text and each profile its fingerprint: together with the subject they stand
//! for the cache key of the output, host-independently (the key's AIR hash covers bitcode that
//! differs between `llvm-as` versions).
//!
//! `golden_hashes.tsv` must match the observed table, cell for cell; it records no version, so a
//! `SEMANTICS_VERSION` bump alone leaves it as it is. A cell that changed while its fixture source
//! and profile fingerprint did not is output that changed under an unchanged key: it needs
//! `cache::SEMANTICS_VERSION` above its value at the table's last bless, and a bless. New, removed,
//! or re-keyed cells only need a bless. The bless (ignored) records what it observes and names the
//! cells that changed under an unchanged key; the master bless refuses those without the bump. It
//! needs every assembler, so it runs on macOS:
//!
//! ```text
//! cargo nextest run -p metal2vulkan --all-features --run-ignored ignored-only -E 'test(bless_golden_hashes)'
//! ```
//!
//! Rows of a fixture whose assembler is not installed (`metal-as` off macOS) are skipped visibly.
//!
//! A second table, `golden_hashes_exact.tsv`, pins the same cells of the fixtures that sample,
//! gather or compute a level of detail (their source declares an `air.sample_texture*`,
//! `air.gather_texture*` or `air.calculate_*` intrinsic) under every exact form
//! (`pipeline::testing::EXACT_SAMPLING`: Metal's sampler emulated, `SamplingPolicy::Exact`, and
//! `DevicePolicy::exact_answers`), whose output the default policy's native device sampling never
//! reaches. Its profiles' fingerprints are those of that policy. The bless writes both tables, under
//! the same rule.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use sha2::{Digest, Sha256};

use crate::device::{Device, DeviceDescription, DevicePolicy};
use crate::error::{Error, Needs};
use crate::library::{CompiledFunction, FunctionStage, Library};
use crate::metal::format::storage_image_choice;
use crate::metal::{PixelFormat, VertexFormat};
use crate::pipeline::testing::{
    engine_dir, fixtures_dir, link_standalone, llvm_assemble, metal_assemble, AssembleError,
    EXACT_SAMPLING,
};
use crate::pipeline::{
    ColorAttachmentDescriptor, ComputePipelineDescriptor, FramebufferFetchInputs,
    FunctionDescriptor, LoweringInputs, RenderPipelineDescriptor, SlotClass, SlotRef, Stage,
    StageSource, StorageImageClass, TileRenderPipelineDescriptor, VertexAttributeDescriptor,
    VertexBufferLayoutDescriptor, VertexDescriptor,
};
use crate::Budget;

/// One golden table: where it is committed, the policy its devices compile under, and the
/// fixtures it takes.
#[derive(Clone, Copy)]
struct Golden {
    /// The committed table, relative to the crate root.
    table: &'static str,
    /// The first lines of its header, before the bless command.
    title: &'static str,
    policy: DevicePolicy,
    /// Whether it takes a fixture, by its source text.
    takes: fn(&str) -> bool,
}

/// The table of every fixture under the default policy.
const DEFAULT: Golden = Golden {
    table: "tests/golden_hashes.tsv",
    title: "# Golden output hashes of the engine over the public fixtures x the committed\n\
            # profiles (src/tests/golden_hashes.rs). Written by the bless; do not edit:\n",
    policy: DevicePolicy::UNTRUSTED,
    takes: every,
};

/// The table of the fixtures that sample under every exact form.
const EXACT: Golden = Golden {
    table: "tests/golden_hashes_exact.tsv",
    title: "# Golden output hashes of the engine over the public fixtures that sample x the\n\
            # committed profiles under EXACT_SAMPLING, every exact form\n\
            # (src/tests/golden_hashes.rs). Written by the bless; do not edit:\n",
    policy: EXACT_SAMPLING,
    takes: samples,
};

fn every(_: &str) -> bool {
    true
}

/// Whether a fixture's source declares a sample, a gather or a level-of-detail query: the
/// intrinsics whose lowering `SamplingPolicy` chooses.
fn samples(text: &str) -> bool {
    [
        "@air.sample_texture",
        "@air.gather_texture",
        "@air.calculate_",
    ]
    .iter()
    .any(|family| text.contains(family))
}

const BLESS: &str = "cargo nextest run -p metal2vulkan --all-features --run-ignored ignored-only \
                     -E 'test(bless_golden_hashes)'";

/// How many lines of each kind of difference a failure prints.
const SHOWN: usize = 40;

// ---- Subjects ----

/// What one row compiles.
#[derive(Clone)]
enum Subject {
    /// Entry `index` of the library under no inputs: a compute pipeline for a kernel, else the
    /// function linked alone.
    Entry { index: u32, kernel: bool },
    /// Kernel `entry`'s storage texture 0 bound as `format`, as a compute pipeline.
    StorageTexture {
        entry: &'static str,
        format: PixelFormat,
    },
    /// A render pipeline of `vertex` and `fragment` at `samples` ([`render_descriptor`]).
    Render {
        vertex: &'static str,
        fragment: &'static str,
        samples: u32,
    },
    /// The tile pipeline of `entry` under color attachments of `formats`, by index.
    Tile {
        entry: &'static str,
        formats: &'static [PixelFormat],
    },
    /// A render pipeline of `vertex` and `fragment` at four samples
    /// ([`multisampled_descriptor`]).
    Multisampled {
        vertex: &'static str,
        fragment: &'static str,
    },
    /// Fragment `entry` compiled under the framebuffer-fetch state of a pipeline whose color
    /// attachments are `formats` (by index, each read), linked alone: the interlocked storage form
    /// on a device that orders fetch only through the pixel interlock, which the entry alone,
    /// compiled without the state, needs. Above one sample, compiled under `multisampled`: the
    /// per-sample forms.
    Fetch {
        entry: &'static str,
        formats: &'static [PixelFormat],
        samples: u32,
    },
    /// The module does not decode: its refusal class in every column.
    Undecodable(String),
    /// The installed assembler rejects the fixture. Every committed fixture assembles with the
    /// pinned LLVM (`validation/tests/fixtures_assemble_with_the_pinned_llvm.rs`, which allows no
    /// exception), so only an installed `llvm-as` of another version can reject one.
    Unassembled,
}

/// The pipelines a fixture adds beyond its entries, as (subject name, subject).
fn pipelines(fixture: &str) -> Vec<(String, Subject)> {
    match fixture {
        "engine/storage_texture_write.ll" => [PixelFormat::RGBA8Unorm, PixelFormat::BGRA8Unorm]
            .into_iter()
            .map(|format| {
                (
                    storage_subject(format),
                    Subject::StorageTexture {
                        entry: "storage_texture_write",
                        format,
                    },
                )
            })
            .collect(),
        "engine/stage_io.ll" => [1, 4]
            .into_iter()
            .map(|samples| {
                let at = if samples == 1 {
                    String::new()
                } else {
                    format!(" at {samples} samples")
                };
                (
                    format!("render stage_io_vertex + stage_io_fragment{at}"),
                    Subject::Render {
                        vertex: "stage_io_vertex",
                        fragment: "stage_io_fragment",
                        samples,
                    },
                )
            })
            .collect(),
        "engine/k10ts_tile_grid.ll" => vec![tile_subject(
            "k10ts_tile_grid",
            &[
                PixelFormat::R32Float,
                PixelFormat::R32Uint,
                PixelFormat::R32Uint,
            ],
        )],
        "engine/k10ts_alias_probe.ll" => vec![tile_subject(
            "k10ts_alias_probe",
            &[
                PixelFormat::R16Float,
                PixelFormat::R32Float,
                PixelFormat::RGBA16Float,
                PixelFormat::R16Float,
            ],
        )],
        "engine/k10ts_tile_order.ll" => {
            vec![tile_subject("k10ts_tile_order", &[PixelFormat::R32Float])]
        }
        "engine/fragment_fetch.ll" => vec![
            fetch_subject(
                "fetch_blend",
                &[
                    PixelFormat::RGBA8Unorm,
                    PixelFormat::RGBA32Uint,
                    PixelFormat::RG16Float,
                ],
                1,
            ),
            fetch_subject("fetch_scalar", &[PixelFormat::R32Float], 1),
            fetch_subject("fetch_ordered", &[PixelFormat::RGBA16Float], 1),
        ],
        "engine/exf_fetch_blend.ll" => [1, 4]
            .into_iter()
            .map(|samples| fetch_subject("exf_fetch_blend", &[PixelFormat::RGBA32Float], samples))
            .collect(),
        "engine/exf_fetch_count.ll" => [1, 4]
            .into_iter()
            .map(|samples| fetch_subject("exf_fetch_count", &[PixelFormat::RGBA32Uint], samples))
            .collect(),
        "engine/exf_fetch_three.ll" => vec![fetch_subject(
            "exf_fetch_three",
            &[
                PixelFormat::RGBA32Float,
                PixelFormat::R32Uint,
                PixelFormat::RGBA16Float,
            ],
            1,
        )],
        "engine/exf2_fetch_unwritten.ll" => vec![fetch_subject(
            "exf2_fetch_unwritten",
            &[PixelFormat::RGBA32Uint, PixelFormat::RGBA32Uint],
            1,
        )],
        "engine/msf_fetch_ms.ll" => [
            "msf_fetch_sample",
            "msf_fetch_mask_in",
            "msf_fetch_mask_out",
            "msf_fetch_discard",
        ]
        .into_iter()
        .flat_map(|entry| {
            [1, 4].map(|samples| fetch_subject(entry, &[PixelFormat::RGBA32Float], samples))
        })
        .collect(),
        "engine/multisample.ll" => [
            "multisample_center_at_sample",
            "multisample_sampled_varying",
            "multisample_counter",
        ]
        .into_iter()
        .map(|fragment| {
            (
                format!("render multisample_vertex + {fragment} at 4 samples"),
                Subject::Multisampled {
                    vertex: "multisample_vertex",
                    fragment,
                },
            )
        })
        .collect(),
        _ => Vec::new(),
    }
}

fn tile_subject(entry: &'static str, formats: &'static [PixelFormat]) -> (String, Subject) {
    (
        format!("tile {entry} as {formats:?}"),
        Subject::Tile { entry, formats },
    )
}

fn fetch_subject(
    entry: &'static str,
    formats: &'static [PixelFormat],
    samples: u32,
) -> (String, Subject) {
    let at = if samples == 1 {
        String::new()
    } else {
        format!(" at {samples} samples")
    };
    (
        format!("fetch {entry} as {formats:?}{at}"),
        Subject::Fetch {
            entry,
            formats,
            samples,
        },
    )
}

const STORAGE_FIXTURE: &str = "engine/storage_texture_write.ll";

fn storage_subject(format: PixelFormat) -> String {
    format!("compute storage_texture_write, texture 0 as {format:?}")
}

/// The render pipeline of the `stage_io` pair at `samples`: a 24-byte interleaved layout of the
/// four attributes the vertex function reads, and two color attachments of distinct numeric
/// classes with a depth attachment (as `tests/api_surface.rs`). At more than one sample it leaves
/// out the integer attachment, which most committed profiles cannot multisample, and the
/// fragment's output to it is discarded.
fn render_descriptor(
    vertex: FunctionDescriptor,
    fragment: FunctionDescriptor,
    samples: u32,
) -> RenderPipelineDescriptor {
    let mut layout = VertexDescriptor::new();
    layout
        .layouts
        .insert(0, VertexBufferLayoutDescriptor::new(Some(24)));
    for (index, format, offset) in [
        (0, VertexFormat::Float3, 0),
        (1, VertexFormat::Half2, 12),
        (4, VertexFormat::UShort2, 16),
        (7, VertexFormat::UChar4, 20),
    ] {
        layout
            .attributes
            .insert(index, VertexAttributeDescriptor::new(format, offset, 0));
    }
    let mut d = RenderPipelineDescriptor::new(StageSource::Function(vertex));
    d.fragment = Some(StageSource::Function(fragment));
    d.vertex_descriptor = Some(layout);
    d.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
    if samples == 1 {
        d.color_attachments[2] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA32Uint));
    }
    d.depth_format = Some(PixelFormat::Depth32Float);
    d.raster_sample_count = samples;
    d
}

/// A render pipeline of `vertex` and `fragment` over one `RGBA16Float` color attachment at four
/// samples (as `src/tests/engine_multisample.rs`).
fn multisampled_descriptor(
    vertex: FunctionDescriptor,
    fragment: FunctionDescriptor,
) -> RenderPipelineDescriptor {
    let mut d = RenderPipelineDescriptor::new(StageSource::Function(vertex));
    d.fragment = Some(StageSource::Function(fragment));
    d.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA16Float));
    d.raster_sample_count = 4;
    d
}

// ---- Inputs ----

/// Every committed profile as a device under `policy`, by stem.
fn devices(policy: DevicePolicy) -> Vec<(String, Device)> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../metal2vulkan-validation/profiles");
    let mut out = Vec::new();
    for entry in std::fs::read_dir(&dir).unwrap_or_else(|e| panic!("{}: {e}", dir.display())) {
        let path = entry.expect("profile directory entry").path();
        if path.extension().is_none_or(|x| x != "json") {
            continue;
        }
        let stem = path
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();
        let text = std::fs::read_to_string(&path).expect("read profile");
        let desc = DeviceDescription::from_khronos_json(&text, None)
            .unwrap_or_else(|e| panic!("{stem}: {e}"));
        let dev = Device::new(desc, policy).unwrap_or_else(|e| panic!("{stem}: {e}"));
        out.push((stem, dev));
    }
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

/// `.ll` files directly in `dir`, sorted by name.
fn ll_files(dir: &Path) -> Vec<PathBuf> {
    let mut paths = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("read {}: {e}", dir.display()))
        .map(|entry| entry.expect("read fixture directory entry").path())
        .filter(|p| p.extension().is_some_and(|x| x == "ll"))
        .collect::<Vec<_>>();
    paths.sort();
    paths
}

/// Every fixture `golden` takes as (table name, path), named as the engine inventory names them.
fn fixture_paths(golden: Golden) -> Vec<(String, PathBuf)> {
    let mut out = Vec::new();
    for (prefix, dir) in [("", fixtures_dir()), ("engine/", engine_dir())] {
        for p in ll_files(&dir) {
            let text =
                std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
            if !(golden.takes)(&text) {
                continue;
            }
            let name = p.file_name().unwrap().to_string_lossy().into_owned();
            out.push((format!("{prefix}{name}"), p));
        }
    }
    out
}

/// A fixture ready to compile.
struct Fixture {
    name: String,
    /// The SHA-256 prefix of the fixture's source text.
    source: String,
    /// `None` when the assembler is not installed (the reason is printed).
    library: Option<Library>,
    /// (subject name, subject); empty when skipped.
    subjects: Vec<(String, Subject)>,
}

fn load(name: String, path: &Path) -> Fixture {
    let text = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let source = hex16(&Sha256::digest(&text));
    let assembled = if name.ends_with(".air.ll") {
        metal_assemble(path)
    } else {
        llvm_assemble(path)
    };
    let bitcode = match assembled {
        Ok(b) => b,
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED {name}: {why}; its rows are not compared");
            return Fixture {
                name,
                source,
                library: None,
                subjects: Vec::new(),
            };
        }
        Err(AssembleError::Rejected(why)) => {
            eprintln!("{name}: the assembler rejected the fixture: {why}");
            return Fixture {
                name,
                source,
                library: None,
                subjects: vec![("-".to_owned(), Subject::Unassembled)],
            };
        }
    };
    let library = match Library::from_air(&bitcode) {
        Ok(l) => l,
        Err(e) => {
            let class =
                refusal(&e).unwrap_or_else(|| panic!("{name}: decode: translator bug: {e}"));
            return Fixture {
                name,
                source,
                library: None,
                subjects: vec![("-".to_owned(), Subject::Undecodable(class))],
            };
        }
    };
    let mut subjects = Vec::new();
    for (index, info) in (0u32..).zip(library.functions()) {
        if info.stage == FunctionStage::Visible {
            continue;
        }
        subjects.push((
            info.name.to_owned(),
            Subject::Entry {
                index,
                kernel: info.stage == FunctionStage::Kernel,
            },
        ));
    }
    assert!(!subjects.is_empty(), "{name}: no entry points");
    subjects.extend(pipelines(&name));
    let names: BTreeSet<&str> = subjects.iter().map(|(s, _)| s.as_str()).collect();
    assert_eq!(
        names.len(),
        subjects.len(),
        "{name}: two subjects share a name"
    );
    Fixture {
        name,
        source,
        library: Some(library),
        subjects,
    }
}

// ---- Cells ----

fn hex16(digest: &[u8]) -> String {
    digest[..8].iter().map(|b| format!("{b:02x}")).collect()
}

/// The cell of compiled stage modules: each module's word count, then its words, little-endian.
fn digest(modules: &[Vec<u32>]) -> String {
    let mut bytes = Vec::new();
    for words in modules {
        let len = u32::try_from(words.len()).expect("module length");
        bytes.extend_from_slice(&len.to_le_bytes());
        for w in words {
            bytes.extend_from_slice(&w.to_le_bytes());
        }
    }
    hex16(&Sha256::digest(&bytes))
}

/// The cell of a refusal, as the engine inventory writes it; `None` for an outcome that is a
/// translator bug.
fn refusal(e: &Error) -> Option<String> {
    Some(match e {
        Error::Unsupported(u) => {
            let needs = match &u.needs {
                Needs::AnyOf(alts) => format!("AnyOf{alts:?}"),
                Needs::NotImplemented => "NotImplemented".to_owned(),
                Needs::NotRepresentable => "NotRepresentable".to_owned(),
                other => format!("{other:?}"),
            };
            format!("Unsupported{{{:?}; {needs}}}", u.feature)
        }
        Error::NeedsInput(i) => format!("NeedsInput{{{i:?}}}"),
        Error::InvalidInput(i) => format!("InvalidInput{{{:?}}}", i.kind),
        Error::LimitExceeded(l) => format!("LimitExceeded{{{l:?}}}"),
        _ => return None,
    })
}

/// The final stage modules of `subject` on `dev`. No deadline: a cell must not depend on host
/// load (the deterministic caps still apply).
fn compiled(lib: &Library, subject: &Subject, dev: &Device) -> Result<Vec<Vec<u32>>, Error> {
    let b = Budget::default();
    let compute = |c: CompiledFunction| -> Result<Vec<Vec<u32>>, Error> {
        let d = ComputePipelineDescriptor::new(StageSource::Compiled(c));
        Ok(vec![dev
            .compile_compute_pipeline(&d, &b)?
            .stage
            .spirv
            .to_vec()])
    };
    match subject {
        Subject::Entry { index, kernel } => {
            let f = FunctionDescriptor::new(lib.function_at(*index)?);
            let c = dev.compile_function(&f, &LoweringInputs::new(), &b)?;
            if *kernel {
                compute(c)
            } else {
                Ok(vec![link_standalone(c.artifact(), dev.profile())?.1])
            }
        }
        Subject::StorageTexture { entry, format } => {
            let choice = storage_image_choice(*format, dev.profile())?;
            let mut inputs = LoweringInputs::new();
            inputs.storage_images.insert(storage_slot(), choice);
            let f = FunctionDescriptor::new(lib.function(entry)?);
            compute(dev.compile_function(&f, &inputs, &b)?)
        }
        Subject::Render {
            vertex,
            fragment,
            samples,
        } => {
            let d = render_descriptor(
                FunctionDescriptor::new(lib.function(vertex)?),
                FunctionDescriptor::new(lib.function(fragment)?),
                *samples,
            );
            let p = dev.compile_render_pipeline(&d, &b)?;
            Ok(p.stages.iter().map(|s| s.spirv.to_vec()).collect())
        }
        Subject::Tile { entry, formats } => {
            let mut d = TileRenderPipelineDescriptor::new(StageSource::Function(
                FunctionDescriptor::new(lib.function(entry)?),
            ));
            for (slot, format) in d.color_attachment_formats.iter_mut().zip(*formats) {
                *slot = Some(*format);
            }
            Ok(vec![dev
                .compile_tile_pipeline(&d, &b)?
                .stage
                .spirv
                .to_vec()])
        }
        Subject::Multisampled { vertex, fragment } => {
            let d = multisampled_descriptor(
                FunctionDescriptor::new(lib.function(vertex)?),
                FunctionDescriptor::new(lib.function(fragment)?),
            );
            let p = dev.compile_render_pipeline(&d, &b)?;
            Ok(p.stages.iter().map(|s| s.spirv.to_vec()).collect())
        }
        Subject::Fetch {
            entry,
            formats,
            samples,
        } => {
            let colors: Vec<Option<ColorAttachmentDescriptor>> = formats
                .iter()
                .map(|&f| Some(ColorAttachmentDescriptor::new(f)))
                .collect();
            let mut inputs = LoweringInputs::new();
            inputs.framebuffer_fetch = Some(FramebufferFetchInputs::of(
                0..formats.len() as u32,
                &colors,
                false,
                false,
                1,
            ));
            inputs.multisampled = *samples > 1;
            let f = FunctionDescriptor::new(lib.function(entry)?);
            let c = dev.compile_function(&f, &inputs, &b)?;
            Ok(vec![link_standalone(c.artifact(), dev.profile())?.1])
        }
        Subject::Undecodable(_) | Subject::Unassembled => {
            unreachable!("a subject without a library compiles nothing")
        }
    }
}

fn storage_slot() -> SlotRef {
    SlotRef {
        stage: Stage::Kernel,
        class: SlotClass::Texture,
        index: 0,
    }
}

fn cell(fixture: &Fixture, subject: &Subject, stem: &str, dev: &Device, what: &str) -> String {
    match subject {
        Subject::Undecodable(class) => class.clone(),
        Subject::Unassembled => "unassembled".to_owned(),
        _ => {
            let lib = fixture.library.as_ref().expect("a loaded library");
            match compiled(lib, subject, dev) {
                Ok(modules) => digest(&modules),
                Err(e) => refusal(&e).unwrap_or_else(|| {
                    panic!("{} {what} on {stem}: translator bug: {e}", fixture.name)
                }),
            }
        }
    }
}

/// Run `f` over `0..n` on every available CPU, results in index order.
fn parallel<T: Send>(n: usize, f: impl Fn(usize) -> T + Sync) -> Vec<T> {
    let threads = std::thread::available_parallelism().map_or(4, |t| t.get());
    let next = AtomicUsize::new(0);
    let mut out: Vec<(usize, T)> = std::thread::scope(|s| {
        let workers: Vec<_> = (0..threads.min(n.max(1)))
            .map(|_| {
                s.spawn(|| {
                    let mut done = Vec::new();
                    loop {
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        if i >= n {
                            return done;
                        }
                        done.push((i, f(i)));
                    }
                })
            })
            .collect();
        workers
            .into_iter()
            .flat_map(|w| w.join().unwrap_or_else(|p| std::panic::resume_unwind(p)))
            .collect()
    });
    out.sort_by_key(|(i, _)| *i);
    out.into_iter().map(|(_, t)| t).collect()
}

// ---- The table ----

/// One row: its fixture's source hash and one cell per profile.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Row {
    source: String,
    cells: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Table {
    /// (stem, fingerprint prefix), sorted by stem.
    profiles: Vec<(String, String)>,
    /// By (fixture, subject).
    rows: BTreeMap<(String, String), Row>,
}

impl Table {
    /// Every cell by (fixture, subject, profile stem), with its inputs (source, fingerprint).
    fn cells(&self) -> BTreeMap<(&str, &str, &str), (&str, &str, &str)> {
        let mut out = BTreeMap::new();
        for ((fixture, subject), row) in &self.rows {
            for ((stem, fp), cell) in self.profiles.iter().zip(&row.cells) {
                out.insert(
                    (fixture.as_str(), subject.as_str(), stem.as_str()),
                    (row.source.as_str(), fp.as_str(), cell.as_str()),
                );
            }
        }
        out
    }

    fn render(&self, golden: Golden) -> String {
        let mut s = String::new();
        s.push_str(golden.title);
        s.push_str(&format!("#   {BLESS}\n"));
        s.push_str(
            "# A cell is the first 64 bits of SHA-256 over the final SPIR-V of its subject, or its\n\
             # refusal class. A cell that changes under an unchanged fixture source and profile\n\
             # fingerprint needs cache::SEMANTICS_VERSION above its value at the last bless.\n",
        );
        for (stem, fp) in &self.profiles {
            s.push_str(&format!("profile\t{stem}\t{fp}\n"));
        }
        s.push_str("fixture\tsource\tsubject");
        for (stem, _) in &self.profiles {
            s.push('\t');
            s.push_str(stem);
        }
        s.push('\n');
        for ((fixture, subject), row) in &self.rows {
            s.push_str(&format!("{fixture}\t{}\t{subject}", row.source));
            for c in &row.cells {
                s.push('\t');
                s.push_str(c);
            }
            s.push('\n');
        }
        s
    }

    fn parse(text: &str) -> Result<Table, String> {
        let mut profiles = Vec::new();
        let mut header = false;
        let mut rows = BTreeMap::new();
        for (n, line) in (1..).zip(text.lines()) {
            if line.starts_with('#') || line.is_empty() {
                continue;
            }
            let f: Vec<&str> = line.split('\t').collect();
            match f.as_slice() {
                ["profile", stem, fp] => profiles.push(((*stem).to_owned(), (*fp).to_owned())),
                ["fixture", "source", "subject", stems @ ..] => {
                    let expected: Vec<&str> = profiles.iter().map(|(s, _)| s.as_str()).collect();
                    if stems != expected.as_slice() {
                        return Err(format!("line {n}: the header does not list the profiles"));
                    }
                    header = true;
                }
                [fixture, source, subject, cells @ ..] if header => {
                    if cells.len() != profiles.len() {
                        return Err(format!("line {n}: {} cells", cells.len()));
                    }
                    let row = Row {
                        source: (*source).to_owned(),
                        cells: cells.iter().map(|c| (*c).to_owned()).collect(),
                    };
                    if rows
                        .insert(((*fixture).to_owned(), (*subject).to_owned()), row)
                        .is_some()
                    {
                        return Err(format!("line {n}: a repeated row"));
                    }
                }
                _ => return Err(format!("line {n}: unexpected {line:?}")),
            }
        }
        Ok(Table { profiles, rows })
    }
}

fn table_path(golden: Golden) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(golden.table)
}

/// The committed table; `None` before the first bless.
fn committed(golden: Golden) -> Option<Table> {
    let text = std::fs::read_to_string(table_path(golden)).ok()?;
    Some(Table::parse(&text).unwrap_or_else(|e| panic!("{}: {e}", golden.table)))
}

/// The observed table and the fixtures that were skipped.
struct Observed {
    table: Table,
    skipped: BTreeSet<String>,
}

fn observe(golden: Golden, devices: &[(String, Device)]) -> Observed {
    let paths = fixture_paths(golden);
    let fixtures = parallel(paths.len(), |i| load(paths[i].0.clone(), &paths[i].1));
    let skipped = fixtures
        .iter()
        .filter(|f| f.subjects.is_empty())
        .map(|f| f.name.clone())
        .collect();
    let jobs: Vec<(usize, usize, usize)> = fixtures
        .iter()
        .enumerate()
        .flat_map(|(fi, f)| {
            (0..f.subjects.len()).flat_map(move |si| (0..devices.len()).map(move |d| (fi, si, d)))
        })
        .collect();
    let cells = parallel(jobs.len(), |j| {
        let (fi, si, d) = jobs[j];
        let f = &fixtures[fi];
        let (name, subject) = &f.subjects[si];
        let (stem, dev) = &devices[d];
        cell(f, subject, stem, dev, name)
    });
    let mut rows: BTreeMap<(String, String), Row> = BTreeMap::new();
    for ((fi, si, _), c) in jobs.iter().zip(cells) {
        let f = &fixtures[*fi];
        rows.entry((f.name.clone(), f.subjects[*si].0.clone()))
            .or_insert_with(|| Row {
                source: f.source.clone(),
                cells: Vec::new(),
            })
            .cells
            .push(c);
    }
    let profiles = devices
        .iter()
        .map(|(stem, dev)| (stem.clone(), hex16(&dev.profile().fingerprint())))
        .collect();
    Observed {
        table: Table { profiles, rows },
        skipped,
    }
}

/// The differences between a committed and an observed table, over the fixtures both compare.
#[derive(Default)]
struct Diff {
    /// Cells that changed under unchanged inputs: output changed under an unchanged key.
    changed: Vec<String>,
    /// Cells that are new, gone, or re-keyed (their fixture source or profile changed).
    stale: Vec<String>,
}

fn diff(committed: &Table, observed: &Observed) -> Diff {
    let old = committed.cells();
    let new = observed.table.cells();
    let mut d = Diff::default();
    for (&(fixture, subject, stem), &(source, fp, cell)) in &new {
        if observed.skipped.contains(fixture) {
            continue;
        }
        let at = format!("{fixture} | {subject} | {stem}");
        match old.get(&(fixture, subject, stem)) {
            None => d.stale.push(format!("new: {at}: {cell}")),
            Some(&(s, f, c)) => {
                if s == source && f == fp {
                    if c != cell {
                        d.changed.push(format!("{at}: {c} -> {cell}"));
                    }
                } else {
                    // Name the KEY that moved, not the cell. A re-keyed profile usually leaves the
                    // cell alone, and printing the cell then reads "profile changed: <hash> ->
                    // <same hash>", which says a cell moved when none did and sends the reader
                    // looking for a behaviour change that is not there.
                    let (what, was, now) = if s == source {
                        ("profile", f, fp)
                    } else {
                        ("fixture source", s, source)
                    };
                    let cell = if c == cell {
                        String::new()
                    } else {
                        format!(", cell {c} -> {cell}")
                    };
                    d.stale
                        .push(format!("{what} changed: {at}: {was} -> {now}{cell}"));
                }
            }
        }
    }
    for (&(fixture, subject, stem), &(_, _, cell)) in &old {
        if !observed.skipped.contains(fixture) && !new.contains_key(&(fixture, subject, stem)) {
            d.stale
                .push(format!("gone: {fixture} | {subject} | {stem}: {cell}"));
        }
    }
    d
}

fn shown(lines: &[String]) -> String {
    let mut s = String::new();
    for l in lines.iter().take(SHOWN) {
        s.push_str("  ");
        s.push_str(l);
        s.push('\n');
    }
    if lines.len() > SHOWN {
        s.push_str(&format!("  ... and {} more\n", lines.len() - SHOWN));
    }
    s
}

/// The required pair: wherever bgra8 is not storage-capable (its choice swizzles), the bgra8 and
/// rgba8 storage-texture pipelines both compile and differ.
fn check_storage_pair(observed: &Table, devices: &[(String, Device)]) -> usize {
    let row = |format| {
        observed
            .rows
            .get(&(STORAGE_FIXTURE.to_owned(), storage_subject(format)))
            .unwrap_or_else(|| panic!("no {STORAGE_FIXTURE} {format:?} row"))
    };
    let (rgba, bgra) = (row(PixelFormat::RGBA8Unorm), row(PixelFormat::BGRA8Unorm));
    let mut swizzled = 0;
    for (i, (stem, dev)) in devices.iter().enumerate() {
        let Ok(choice) = storage_image_choice(PixelFormat::BGRA8Unorm, dev.profile()) else {
            continue;
        };
        if choice.class != StorageImageClass::SwizzleBgra {
            continue;
        }
        let (r, b) = (&rgba.cells[i], &bgra.cells[i]);
        assert!(
            r.len() == 16 && b.len() == 16 && r != b,
            "{stem}: the bgra8 ({b}) and rgba8 ({r}) storage pipelines must both compile and differ"
        );
        swizzled += 1;
    }
    swizzled
}

#[test]
fn golden_hashes_match_the_table() {
    let devices = devices(DEFAULT.policy);
    let observed = observe(DEFAULT, &devices);
    if !observed.skipped.contains(STORAGE_FIXTURE) {
        let swizzled = check_storage_pair(&observed.table, &devices);
        assert!(
            swizzled > 0,
            "no committed profile swizzles a bgra8 storage texture"
        );
    }
    matches_the_table(DEFAULT, &devices, &observed);
}

/// The fixtures that sample, under every exact form: the sampler's emulation, which the default
/// table's native device sampling does not reach.
#[test]
fn exact_golden_hashes_match_the_table() {
    let devices = devices(EXACT.policy);
    let observed = observe(EXACT, &devices);
    matches_the_table(EXACT, &devices, &observed);
}

/// `golden`'s committed table holds `observed` cell for cell.
fn matches_the_table(golden: Golden, devices: &[(String, Device)], observed: &Observed) {
    let table = golden.table;
    if !observed.skipped.is_empty() {
        eprintln!(
            "SKIPPED {} fixture(s) whose assembler is not installed",
            observed.skipped.len()
        );
    }
    let Some(committed) = committed(golden) else {
        panic!("{table} is missing: bless it with\n  {BLESS}");
    };
    let d = diff(&committed, observed);
    if !d.changed.is_empty() {
        panic!(
            "{table}: output changed under an unchanged key: {} cell(s) changed while their \
             fixture source and profile fingerprint did not, so a cached artifact under the same \
             key would be stale unless cache::SEMANTICS_VERSION (src/cache/mod.rs) is above its \
             value at the table's last bless. Bump it if this change has not, and bless:\n  \
             {BLESS}\n{}",
            d.changed.len(),
            shown(&d.changed)
        );
    }
    if !d.stale.is_empty() {
        panic!(
            "{table} is stale: {} cell(s) are new, gone, or under a changed fixture source or \
             profile (their keys changed, so no SEMANTICS_VERSION bump is needed). Bless it \
             with\n  {BLESS}\n{}",
            d.stale.len(),
            shown(&d.stale)
        );
    }
    eprintln!(
        "{table}: {} rows x {} profiles match",
        observed.table.rows.len(),
        devices.len()
    );
}

/// Rewrite both tables from what this build observes: `BLESS` above. Refuses when an assembler is
/// missing; names the cells that changed under an unchanged key, which need a
/// `cache::SEMANTICS_VERSION` bump since the last bless (the master bless checks it).
#[test]
#[ignore]
fn bless_golden_hashes() {
    for golden in [DEFAULT, EXACT] {
        let table = golden.table;
        let devices = devices(golden.policy);
        let observed = observe(golden, &devices);
        assert!(
            observed.skipped.is_empty(),
            "refusing to bless without every assembler (skipped: {:?}); bless on macOS",
            observed.skipped
        );
        if let Some(committed) = committed(golden) {
            let d = diff(&committed, &observed);
            if !d.changed.is_empty() {
                eprintln!(
                    "{table}: {} cell(s) changed under an unchanged key: they need \
                     cache::SEMANTICS_VERSION above its value at the table's last bless\n{}",
                    d.changed.len(),
                    shown(&d.changed)
                );
            }
        }
        std::fs::write(table_path(golden), observed.table.render(golden)).expect("write the table");
        eprintln!(
            "{table}: blessed {} rows x {} profiles",
            observed.table.rows.len(),
            devices.len()
        );
    }
}

/// A stale line names the key that moved, and says nothing about a cell that did not move.
///
/// The line used to print the cell on a message announcing a key change, so a profile that
/// re-keyed while its output stayed put read `profile changed: <hash> -> <same hash>`: it says a
/// cell moved when none did, and sends the reader after a behaviour change that is not there. A
/// re-bless was diagnosed wrong that way, which is why this holds all four classes apart.
#[test]
fn a_stale_cell_names_the_key_that_moved_and_not_an_unmoved_cell() {
    let at = ("f.ll".to_owned(), "main".to_owned());
    let table = |fp: &str, source: &str, cell: &str| Table {
        profiles: vec![("a-gpu".to_owned(), fp.to_owned())],
        rows: BTreeMap::from([(
            at.clone(),
            Row {
                source: source.to_owned(),
                cells: vec![cell.to_owned()],
            },
        )]),
    };
    let observed = |fp: &str, cell: &str| Observed {
        table: table(fp, "src", cell),
        skipped: BTreeSet::new(),
    };

    // The profile re-keyed and the cell did not: the two fingerprints, and no word about a cell.
    let d = diff(&table("old-fp", "src", "cell"), &observed("new-fp", "cell"));
    assert!(d.changed.is_empty(), "{:?}", d.changed);
    assert_eq!(
        d.stale,
        ["profile changed: f.ll | main | a-gpu: old-fp -> new-fp"]
    );

    // The cell moved with it: the same line, with the cell named after the key.
    let d = diff(&table("old-fp", "src", "was"), &observed("new-fp", "now"));
    assert_eq!(
        d.stale,
        ["profile changed: f.ll | main | a-gpu: old-fp -> new-fp, cell was -> now"]
    );

    // A fixture source that moved names the sources, not the profile's fingerprint.
    let d = diff(&table("fp", "older", "cell"), &observed("fp", "cell"));
    assert_eq!(
        d.stale,
        ["fixture source changed: f.ll | main | a-gpu: older -> src"]
    );

    // Output that moved under an unchanged key is the other class, and still names the cell.
    let d = diff(&table("fp", "src", "was"), &observed("fp", "now"));
    assert!(d.stale.is_empty(), "{:?}", d.stale);
    assert_eq!(d.changed, ["f.ll | main | a-gpu: was -> now"]);
}

/// The exact table takes the fixtures whose sampling the policy changes, under the sampler's
/// emulation and Metal's other answers: its fingerprints are not the default table's.
#[test]
fn the_exact_table_takes_the_sampling_fixtures_under_the_emulation() {
    assert_eq!(EXACT.policy.sampling, crate::device::SamplingPolicy::Exact);
    assert_eq!(
        EXACT.policy,
        DevicePolicy {
            sampling: crate::device::SamplingPolicy::Exact,
            ..DevicePolicy::default().exact_answers()
        }
    );
    assert_eq!(DEFAULT.policy, DevicePolicy::default());
    let all = fixture_paths(DEFAULT);
    let taken = fixture_paths(EXACT);
    assert!(!taken.is_empty() && taken.len() < all.len());
    assert!(taken.iter().all(|f| all.contains(f)));
    for anisotropic in ["engine/ani_cube_grad.ll", "engine/a4_sample_grad.ll"] {
        assert!(
            taken.iter().any(|(name, _)| name == anisotropic),
            "{anisotropic}"
        );
    }
    let (stem, default) = &devices(DEFAULT.policy)[0];
    let (_, exact) = &devices(EXACT.policy)[0];
    assert_ne!(
        default.profile().fingerprint(),
        exact.profile().fingerprint(),
        "{stem}"
    );
}

#[test]
fn the_table_format_round_trips() {
    let mut rows = BTreeMap::new();
    rows.insert(
        ("a.ll".to_owned(), "k".to_owned()),
        Row {
            source: "00".repeat(8),
            cells: vec![
                "11".repeat(8),
                "Unsupported{Format; NotRepresentable}".to_owned(),
            ],
        },
    );
    let t = Table {
        profiles: vec![
            ("p".to_owned(), "22".repeat(8)),
            ("q".to_owned(), "33".repeat(8)),
        ],
        rows,
    };
    for golden in [DEFAULT, EXACT] {
        assert_eq!(Table::parse(&t.render(golden)), Ok(t.clone()));
    }
}

/// The rule itself on synthetic tables: a changed cell under unchanged inputs needs a bump; a
/// re-keyed, new, or gone cell only a bless; a skipped fixture is not compared.
#[test]
fn only_a_cell_changed_under_its_key_needs_a_bump() {
    let table = |source: &str, fp: &str, cells: &[(&str, &str)]| Table {
        profiles: vec![("p".to_owned(), fp.to_owned())],
        rows: cells
            .iter()
            .map(|(fixture, c)| {
                (
                    ((*fixture).to_owned(), "k".to_owned()),
                    Row {
                        source: source.to_owned(),
                        cells: vec![(*c).to_owned()],
                    },
                )
            })
            .collect(),
    };
    let observed = |t: Table, skipped: &[&str]| Observed {
        table: t,
        skipped: skipped.iter().map(|s| (*s).to_owned()).collect(),
    };
    let base = table("s", "f", &[("a.ll", "h1")]);
    let d = diff(&base, &observed(table("s", "f", &[("a.ll", "h2")]), &[]));
    assert_eq!((d.changed.len(), d.stale.len()), (1, 0));
    let d = diff(&base, &observed(table("s2", "f", &[("a.ll", "h2")]), &[]));
    assert_eq!((d.changed.len(), d.stale.len()), (0, 1));
    let d = diff(&base, &observed(table("s", "f2", &[("a.ll", "h2")]), &[]));
    assert_eq!((d.changed.len(), d.stale.len()), (0, 1));
    let d = diff(&base, &observed(table("s", "f", &[("b.ll", "h1")]), &[]));
    assert_eq!((d.changed.len(), d.stale.len()), (0, 2));
    let d = diff(&base, &observed(table("s", "f", &[]), &["a.ll"]));
    assert_eq!((d.changed.len(), d.stale.len()), (0, 0));
    let d = diff(&base, &observed(base.clone(), &[]));
    assert_eq!((d.changed.len(), d.stale.len()), (0, 0));
}
