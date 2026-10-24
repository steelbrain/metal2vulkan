//! A vertex function's capture form (`LoweringInputs::captured_positions`), over the owned fixtures
//! `engine/rch_vertex_math` (a position from a buffer at its vertex index through three
//! matrices, no point size) and `engine/stage_io` (`stage_io_vertex`: a position, a point size,
//! and the vertex and instance indices of its own) and `engine/vertex_amplification` (a
//! post-tessellation vertex function):
//!
//! - Captured, the function states `RuntimeFact::CapturedPositions`, reads
//!   `PushField::CaptureAddress` (a `u64` in its push window) and `PushField::CaptureLayout` (a
//!   `uvec4`), declares the vertex and instance indices, stores through physical storage buffer
//!   addresses (so it needs `bufferDeviceAddress` and `vertexPipelineStoresAndAtomics`), and still
//!   writes no memory of its own (`StageInterface::writes_memory`); every module passes L3 and
//!   the words the capture adds are bounded. Compiled without it the function is unchanged.
//! - A device without device addresses, or without the vertex stage's stores, refuses the capture
//!   `Feature::TileReach`, naming what it needs; a post-tessellation vertex function is
//!   `NotImplemented(TileReach)`.
//! - The capture variant of a render pipeline has its own keys, states its fact as a variant, its
//!   plan settles on its own key, and a direct draw leaves `PushHole::CaptureAddress` in the
//!   vertex window and writes the draw's first vertex, vertex count and first instance as its
//!   layout; an indexed draw of it asks for the indexed form's variant, and an indirect or
//!   amplified one is `NotImplemented(TileReach)`.
//! - The indexed form (`CaptureForm::Indexed`) reads each vertex from the draw's index buffer at
//!   `PushField::CaptureIndexAddress`, on the CPU interpreter: `u16` indices at 2- and 4-aligned
//!   addresses and `u32` ones, with a wrapping base vertex and instances, feed `[[vertex_id]]` and
//!   every pulled attribute, and each record's restart flag marks the all-ones index. Its
//!   projection pulls every attribute; its pipeline links, plans itself, and its indexed draws run
//!   without indices, reading the indices in the vertex stage.
//!
//! `llvm-as` assembles the fixtures; a missing tool skips visibly. No GPU is used.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use crate::bind::{
    AccessClass, BindState, BoundBuffer, ByteRange, DrawCall, IndexBufferRef, IndirectRef,
    PushHole, ResourceUse, StageTable, UseTarget, VertexAmplification, ViewMapping,
};
use crate::budget::Budget;
use crate::cache::VariantKey;
use crate::device::requirements::{SpirvCapability, VkFeature as F};
use crate::device::{Device, DeviceDescription, DevicePolicy, DeviceProfile};
use crate::error::{BindError, Capability, Error, Feature, Unsupported};
use crate::library::{FunctionArtifact, Library};
use crate::lower::select::Plan;
use crate::memory::contain;
use crate::metal::{self, IndexType, VertexFormat};
use crate::pipeline::descriptor::{
    FunctionDescriptor, RenderPipelineDescriptor, StageSource, VertexAttributeDescriptor,
    VertexBufferLayoutDescriptor, VertexDescriptor,
};
use crate::pipeline::facts::{Mechanism, RuntimeFact};
use crate::pipeline::interface::{AccessSet, PushLoc};
use crate::pipeline::link::vertex::vertex_fetch_projection_for;
use crate::pipeline::link::{link, LinkInputs};
use crate::pipeline::lower_inputs::{
    CaptureForm, LoweringInputs, VertexFetch, VertexFetchProjection,
};
use crate::pipeline::relocate::relocate;
use crate::pipeline::stage_io::{IoNumeric, StageBuiltin};
use crate::pipeline::sym::{PushField, ResourceKind, ResourceSlot, Stage};
use crate::pipeline::testing::{
    compile_with, engine_dir, llvm_assemble, typed_module_with, AssembleError,
};
use crate::spirv::interp::{run as interp_run, Buffer, Config, Resources};
use crate::spirv::ir::TypedModule;
use crate::spirv::testing::check_words_valid;
use crate::vk;

/// The bitcode of the owned fixture `stem` (`engine/<stem>.ll`), or `None` (printed) where
/// `llvm-as` is missing.
fn fixture(stem: &str) -> Option<Vec<u8>> {
    match llvm_assemble(&engine_dir().join(format!("{stem}.ll"))) {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected {stem}.ll: {why}"),
    }
}

fn description(stem: &str) -> DeviceDescription {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles")
        .join(format!("{stem}.json"));
    let text = std::fs::read_to_string(path).expect("profile");
    DeviceDescription::from_khronos_json(&text, None).expect("description")
}

fn device_of(desc: DeviceDescription) -> Device {
    Device::new(desc, DevicePolicy::default()).expect("device")
}

/// The lowering inputs of capture form `form`, or of none.
fn inputs(form: Option<CaptureForm>) -> LoweringInputs {
    let mut inputs = LoweringInputs::new();
    inputs.captured_positions = form;
    inputs
}

/// Link `a` alone, relocate it, and run L3; its relocated words.
fn validate(a: &FunctionArtifact, p: &DeviceProfile) -> Vec<u32> {
    let linked = link(&[a], &LinkInputs::standalone(), p).unwrap_or_else(|e| panic!("link: {e}"));
    let words = relocate(a, &linked.table).expect("relocate");
    check_words_valid(&words, &a.requirements, p).unwrap_or_else(|e| panic!("L3: {e}"));
    words
}

/// `e` without the location an `Unsupported` carries.
fn unlocated(e: Error) -> Error {
    match e {
        Error::Unsupported(u) => Error::Unsupported(Unsupported {
            location: None,
            ..u
        }),
        other => other,
    }
}

fn physical_addresses() -> SpirvCapability {
    SpirvCapability::from_raw(crate::spv::Capability::PhysicalStorageBufferAddresses as u32)
}

/// Each entry compiled without the capture and with it: the fact, the two push fields (the
/// address in the vertex window), the indices, the device-address stores and their needs, no
/// memory of the function's own, L3, and the words the capture adds (printed; bounded: measured
/// 246 to 259 on lavapipe and MoltenVK).
#[test]
fn a_captured_vertex_function_writes_its_records() {
    let entries = [
        ("rch_vertex_math", "rch_vertex_math"),
        ("stage_io", "stage_io_vertex"),
    ];
    for stem in ["lavapipe", "moltenvk-m3"] {
        let dev = device_of(description(stem));
        let p = dev.profile();
        for (fixture_stem, entry) in entries {
            let Some(bc) = fixture(fixture_stem) else {
                return;
            };
            let what = format!("{stem} {entry}");
            let plain = compile_with(&bc, entry, p, &inputs(None))
                .unwrap_or_else(|e| panic!("{what}: {e}"));
            let plain_words = validate(&plain, p).len();
            assert!(
                !plain
                    .facts
                    .iter()
                    .any(|f| matches!(f, RuntimeFact::CapturedPositions { .. })),
                "{what}"
            );
            assert!(
                !plain.interface.push.fields.iter().any(|f| matches!(
                    f.field,
                    PushField::CaptureAddress | PushField::CaptureLayout
                )),
                "{what}"
            );
            let a = compile_with(&bc, entry, p, &inputs(Some(CaptureForm::Vertices)))
                .unwrap_or_else(|e| panic!("{what} captured: {e}"));
            let words = validate(&a, p).len();
            assert!(
                a.facts.contains(&RuntimeFact::CapturedPositions {
                    form: CaptureForm::Vertices
                }),
                "{what}"
            );
            let field = |field: PushField| {
                a.interface
                    .push
                    .fields
                    .iter()
                    .find(|f| f.field == field)
                    .copied()
                    .unwrap_or_else(|| panic!("{what}: no {field:?}"))
            };
            let address = field(PushField::CaptureAddress);
            let PushLoc::Push { offset } = address.loc else {
                panic!("{what}: the capture address spilled: {address:?}");
            };
            assert_eq!(address.size, 8, "{what}");
            assert!(offset + 8 <= 64, "{what}: {offset}");
            assert_eq!(field(PushField::CaptureLayout).size, 16, "{what}");
            let builtins = &a.interface.io.builtins;
            assert!(builtins.contains(&StageBuiltin::VertexIndex), "{what}");
            assert!(builtins.contains(&StageBuiltin::InstanceIndex), "{what}");
            // Its records are stored through device addresses from the vertex stage.
            let r = &a.requirements;
            assert!(
                r.spirv_capabilities.contains(&physical_addresses()),
                "{what}"
            );
            assert!(r.features.contains(&F::BufferDeviceAddress), "{what}");
            assert!(
                r.features.contains(&F::VertexPipelineStoresAndAtomics),
                "{what}"
            );
            assert!(
                !plain
                    .requirements
                    .features
                    .contains(&F::VertexPipelineStoresAndAtomics),
                "{what}"
            );
            // The records are the product's: the function writes no memory of its own.
            assert!(!a.interface.writes_memory, "{what}");
            let added = words as i64 - plain_words as i64;
            println!("{what}: the capture adds {added} words to {plain_words}");
            assert!((0..=300).contains(&added), "{what}: {added} words");
        }
    }
}

/// The capture needs device addresses and the vertex stage's stores, refused `TileReach`
/// naming them; a post-tessellation vertex function's vertices are the tessellator's, so its
/// capture is not implemented.
#[test]
fn the_capture_needs_addresses_and_vertex_stores() {
    let Some(bc) = fixture("rch_vertex_math") else {
        return;
    };
    let Some(amplification) = fixture("vertex_amplification") else {
        return;
    };
    let floor = device_of(description("vk12-floor"));
    let p = floor.profile();
    compile_with(&bc, "rch_vertex_math", p, &inputs(None)).expect("without the capture");
    for form in [
        CaptureForm::Vertices,
        CaptureForm::Indexed(IndexType::UInt16),
    ] {
        let refusal = compile_with(&bc, "rch_vertex_math", p, &inputs(Some(form)))
            .map(|_| ())
            .map_err(unlocated);
        assert_eq!(
            refusal,
            Err(Error::Unsupported(Unsupported::needs_any_of(
                Feature::TileReach,
                vec![p.addressed_needs()]
            ))),
            "{form:?}"
        );
    }
    let mut desc = description("moltenvk-m3");
    assert!(desc.features.remove(&F::VertexPipelineStoresAndAtomics));
    let without = device_of(desc);
    let refusal = compile_with(
        &bc,
        "rch_vertex_math",
        without.profile(),
        &inputs(Some(CaptureForm::Vertices)),
    )
    .map(|_| ())
    .map_err(unlocated);
    assert_eq!(
        refusal,
        Err(Error::Unsupported(Unsupported::needs(
            Feature::TileReach,
            Capability::Feature(F::VertexPipelineStoresAndAtomics)
        )))
    );
    let lavapipe = device_of(description("lavapipe"));
    let refusal = compile_with(
        &amplification,
        "amp_post_tessellation",
        lavapipe.profile(),
        &inputs(Some(CaptureForm::Indexed(IndexType::UInt32))),
    )
    .map(|_| ())
    .map_err(unlocated);
    assert_eq!(
        refusal,
        Err(Error::Unsupported(Unsupported::not_implemented(
            Feature::TileReach
        )))
    );
}

/// `rch_vertex_math`'s buffers: its positions at 0 and its matrices at 1.
fn bound() -> BindState {
    let buffer = |length| BoundBuffer::Buffer {
        length,
        offset: 0,
        gpu_address: 0x1_0000,
        device_address: Some(0x7_0000_0000),
        attribute_stride: None,
    };
    let mut v = StageTable::default();
    v.buffers.insert(0, buffer(1024));
    v.buffers.insert(1, buffer(256));
    let mut s = BindState::default();
    s.stages.insert(Stage::Vertex, v);
    s
}

/// The capture variant of `rch_vertex_math`'s render pipeline: its own keys, its fact as a
/// variant, a plan that settles on its key, and a direct draw's hole and layout; the base
/// pipeline's plan asks for no capture. An indexed draw of the variant asks for the indexed form
/// of its index type; an indirect or amplified one is `NotImplemented(TileReach)`.
#[test]
fn a_capture_variant_plans_itself_and_writes_each_draws_layout() {
    let Some(bc) = fixture("rch_vertex_math") else {
        return;
    };
    let budget = Budget::standard();
    let lib = Library::from_air(&bc).expect("library");
    let function = FunctionDescriptor::new(lib.function("rch_vertex_math").expect("function"));
    let d = RenderPipelineDescriptor::new(StageSource::Function(function.clone()));
    let mut key = VariantKey::new();
    key.captured_positions = Some(CaptureForm::Vertices);
    assert!(!key.is_empty());
    assert_eq!(
        key.lowering_inputs(Stage::Vertex).captured_positions,
        Some(CaptureForm::Vertices)
    );
    assert_eq!(
        key.lowering_inputs(Stage::Fragment).captured_positions,
        None
    );
    for stem in ["lavapipe", "moltenvk-m3"] {
        let dev = device_of(description(stem));
        let base = dev
            .compile_render_pipeline(&d, &budget)
            .unwrap_or_else(|e| panic!("{stem}: {e}"));
        let keys = dev
            .render_variant_key(&d, &key, &budget, &crate::pipeline::StageEntries::new())
            .expect("keys");
        let base_keys = dev.render_pipeline_key(&d, &budget).expect("keys");
        assert_ne!(keys.vertex, base_keys.vertex, "{stem}");
        assert_ne!(keys.pipeline, base_keys.pipeline, "{stem}");
        let vertex_inputs = keys.vertex_inputs.clone().expect("vertex inputs");
        assert_eq!(
            vertex_inputs.captured_positions,
            Some(CaptureForm::Vertices),
            "{stem}"
        );
        assert_eq!(
            keys.vertex,
            dev.function_compile_key(&function, &vertex_inputs),
            "{stem}"
        );
        let p = dev
            .compile_render_variant(&d, &key, &budget)
            .unwrap_or_else(|e| panic!("{stem}: {e}"));
        assert_eq!(p.key, keys.pipeline, "{stem}");
        for m in &p.stages {
            check_words_valid(&m.spirv, &p.requirements, dev.profile())
                .unwrap_or_else(|e| panic!("{stem} {:?}: L3: {e}", m.stage));
        }
        assert!(
            p.dependencies.iter().any(|d| d.fact
                == RuntimeFact::CapturedPositions {
                    form: CaptureForm::Vertices
                }
                && d.mechanism == Mechanism::Variant),
            "{stem}"
        );
        assert!(!p.graphics.vertex_writes_memory, "{stem}");

        let state = bound();
        let plan = base.plan(&dev, &state.shape()).expect("base plan");
        assert!(plan.variant.is_empty(), "{stem}");
        let plan = p.plan(&dev, &state.shape()).expect("plan");
        assert_eq!(plan.variant, key, "{stem}");
        let draw = DrawCall::Primitives {
            vertex_start: 5,
            vertex_count: 7,
            instance_count: 2,
            base_instance: 3,
        };
        let commands = plan
            .instantiate_draw(&state, &draw)
            .unwrap_or_else(|e| panic!("{stem}: {e:?}"));
        let holes: Vec<_> = commands
            .push_holes
            .iter()
            .filter_map(|h| match *h {
                PushHole::CaptureAddress { stages, offset } => Some((stages, offset)),
                _ => None,
            })
            .collect();
        let [(stages, offset)] = holes[..] else {
            panic!("{stem}: {:?}", commands.push_holes);
        };
        assert_eq!(stages, vk::ShaderStageFlags::VERTEX, "{stem}");
        assert!(offset + 8 <= 64, "{stem}: {offset}");
        let layout = plan
            .push
            .fields
            .iter()
            .find(|f| f.stage == Stage::Vertex && f.field == PushField::CaptureLayout)
            .expect("placed");
        let crate::bind::FieldLocation::PushWindow { offset } = layout.location else {
            panic!("{stem}: the layout spilled: {layout:?}");
        };
        let write = commands
            .push_constants
            .iter()
            .find(|w| w.stages.contains(vk::ShaderStageFlags::VERTEX))
            .expect("vertex push");
        let at = (offset - write.offset) as usize;
        let words: Vec<u32> = write.bytes[at..at + 16]
            .chunks(4)
            .map(|w| u32::from_le_bytes(w.try_into().expect("a word")))
            .collect();
        assert_eq!(words, [5, 7, 3, 0], "{stem}");

        let not_implemented = Err(BindError::Unsupported(Unsupported::not_implemented(
            Feature::TileReach,
        )));
        let indexed = DrawCall::Indexed {
            index_type: vk::IndexType::UINT16,
            index_buffer: IndexBufferRef::new(1, 64),
            index_count: 3,
            index_offset: 0,
            instance_count: 1,
            base_vertex: 0,
            base_instance: 0,
        };
        assert_eq!(
            plan.instantiate_draw(&state, &indexed).map(|_| ()),
            Err(BindError::NeedsVariant {
                variant: Box::new(VariantKey {
                    captured_positions: Some(CaptureForm::Indexed(IndexType::UInt16)),
                    ..key.clone()
                }),
            }),
            "{stem}"
        );
        let indirect = DrawCall::Indirect {
            indirect: IndirectRef::new(2, 0),
            indexed: None,
        };
        assert_eq!(
            plan.instantiate_draw(&state, &indirect).map(|_| ()),
            not_implemented,
            "{stem}"
        );
        let mut amplified = bound();
        amplified.vertex_amplification =
            VertexAmplification::new(2, vec![ViewMapping::new(0, 0), ViewMapping::new(0, 1)]);
        assert_eq!(
            p.plan(&dev, &amplified.shape()).map(|_| ()),
            not_implemented,
            "{stem}"
        );
    }
}

// ---- The indexed form (CaptureForm::Indexed) -------------------------------------------------

/// Device addresses of the interpreter's address-backed buffers: the records, the indices, the
/// root table and its sink.
const RECORDS: u64 = 0x0000_0040_0000_0000;
const INDICES: u64 = 0x0000_0050_0000_0000;
const TABLE: u64 = 0x0000_0010_0000_0000;
const SINK: u64 = 0x0000_0011_0000_0000;
/// The byte every record is filled with before a run: the words a record leaves unwritten keep it.
const FILL: u8 = 0xAB;

/// One indexed capture run on the CPU interpreter: the draw's indices of `ty` (as `u32`s) placed
/// `skew` bytes past a 4-aligned address, its base vertex and its instances, and whether it
/// restarts (a strip; the layout's lane 0).
struct IndexedRun<'a> {
    ty: IndexType,
    indices: &'a [u32],
    skew: u64,
    base_vertex: i32,
    first_instance: u32,
    instances: u32,
    restarts: bool,
}

impl IndexedRun<'_> {
    fn count(&self) -> u32 {
        self.indices.len() as u32
    }

    /// The index buffer's bytes from its 4-aligned start, padded to whole words.
    fn index_bytes(&self) -> Vec<u8> {
        let mut out = vec![0xCDu8; self.skew as usize];
        for &i in self.indices {
            match self.ty {
                IndexType::UInt16 => out.extend_from_slice(&(i as u16).to_le_bytes()),
                IndexType::UInt32 => out.extend_from_slice(&i.to_le_bytes()),
            }
        }
        while !out.len().is_multiple_of(4) {
            out.push(0xCD);
        }
        out
    }

    /// The vertex record slot `s` holds: its index position's raw index plus the base vertex.
    fn vertex(&self, position: usize) -> u32 {
        self.indices[position].wrapping_add(self.base_vertex as u32)
    }

    fn restart(&self, position: usize) -> bool {
        self.indices[position]
            == match self.ty {
                IndexType::UInt16 => 0xffff,
                IndexType::UInt32 => u32::MAX,
            }
    }

    /// Whether index position `position` runs no body: a restart index of a draw that restarts.
    fn skipped(&self, position: usize) -> bool {
        self.restarts && self.restart(position)
    }
}

/// Whether `record` is a skipped restart vertex's: its flag 1 and every other word as it was.
fn skipped_record(record: &[u32; 6]) -> bool {
    let fill = u32::from_le_bytes([FILL; 4]);
    record[..5] == [fill; 5] && record[5] == 1
}

/// The root table with no slot entries (entry 1 the sink), for a module that reads one.
fn root_table() -> Vec<u8> {
    let mut t = vec![0u8; 32 * 2];
    t[32..40].copy_from_slice(&SINK.to_le_bytes());
    t[40..48].copy_from_slice(&2u64.to_le_bytes());
    t
}

fn addressed(address: u64, bytes: Vec<u8>) -> Buffer {
    Buffer {
        slot: None,
        address: Some(address),
        bytes,
    }
}

fn vertex_buffer(index: u32, bytes: Vec<u8>) -> Buffer {
    Buffer {
        slot: Some(ResourceSlot {
            stage: Stage::Vertex,
            kind: ResourceKind::Buffer,
            index,
        }),
        address: None,
        bytes,
    }
}

/// The push bytes of `plan` for `run`: the capture addresses and layout, zero bind residuals,
/// `stride` for every vertex buffer stride, and the root table's address.
fn capture_push(plan: &Plan, run: &IndexedRun<'_>, index_address: u64, stride: u32) -> Vec<u8> {
    let mut out = vec![0u8; 128];
    for f in &plan.push.fields {
        let PushLoc::Push { offset } = f.loc else {
            panic!("{:?} spilled to the ArgumentTable", f.field);
        };
        let at = offset as usize;
        let bytes: Vec<u8> = match f.field {
            PushField::CaptureAddress => RECORDS.to_le_bytes().to_vec(),
            PushField::CaptureIndexAddress => index_address.to_le_bytes().to_vec(),
            PushField::CaptureLayout => [
                u32::from(run.restarts),
                run.count(),
                run.first_instance,
                run.base_vertex as u32,
            ]
            .iter()
            .flat_map(|w| w.to_le_bytes())
            .collect(),
            PushField::RootTableAddress => TABLE.to_le_bytes().to_vec(),
            PushField::SlotBindOffset(_) => vec![0; 4],
            PushField::VertexBufferStride(_) => stride.to_le_bytes().to_vec(),
            other => panic!("unexpected push field {other:?}"),
        };
        out[at..at + bytes.len()].copy_from_slice(&bytes);
    }
    out
}

/// Run `m` (planned as `plan`) as `run`'s capture draw: `count` vertices from 0 of its instances,
/// with `extra` buffers bound; the records, as `[clip x, y, z, w, point size, restart flag]` bits
/// per slot, after checking that each record's last two words are unwritten.
fn run_capture(
    what: &str,
    plan: &Plan,
    m: &TypedModule,
    run: &IndexedRun<'_>,
    stride: u32,
    extra: Vec<Buffer>,
) -> Vec<[u32; 6]> {
    let slots = (run.count() * run.instances) as usize;
    let mut buffers = vec![
        addressed(RECORDS, vec![FILL; 32 * slots]),
        addressed(INDICES, run.index_bytes()),
        addressed(TABLE, root_table()),
        addressed(SINK, vec![0; 64]),
    ];
    buffers.extend(extra);
    let res = Resources {
        buffers,
        push: capture_push(plan, run, INDICES + run.skew, stride),
        spec: Default::default(),
    };
    let cfg = Config {
        workgroups: [run.count(), run.instances, 1],
        first: [0, run.first_instance],
        ..Config::default()
    };
    let out = interp_run(m, &cfg, res).unwrap_or_else(|e| panic!("{what}: {e}"));
    assert_eq!(
        out.buffers[1],
        run.index_bytes(),
        "{what}: the indices are read only"
    );
    out.buffers[0]
        .as_chunks::<32>()
        .0
        .iter()
        .map(|r| {
            assert_eq!(
                r[24..],
                [FILL; 8],
                "{what}: words 6 and 7 are left as they were"
            );
            let w = |k: usize| u32::from_le_bytes(r[4 * k..4 * k + 4].try_into().expect("word"));
            [w(0), w(1), w(2), w(3), w(4), w(5)]
        })
        .collect()
}

/// The lowering inputs of the indexed form of `ty` with projection `fetch`.
fn indexed_inputs(ty: IndexType, fetch: Option<VertexFetchProjection>) -> LoweringInputs {
    let mut i = inputs(Some(CaptureForm::Indexed(ty)));
    i.vertex_fetch = fetch;
    i
}

/// The indexed form reads each vertex from the draw's index buffer, on the CPU interpreter:
/// `stage_io_vertex_position` returns `(vid & 1, vid >> 1, 0, 1)`, so each record's clip
/// position names the vertex its `[[vertex_id]]` saw. For `u16` indices at a 2- and a 4-aligned
/// address and `u32` ones, with a negative and a wrapping base vertex and two instances from a
/// nonzero first one: record `p + (i - first instance) · count` holds index position `p`'s vertex
/// (its raw index plus the base vertex, wrapping), point size 1.0, and restart flag 1 exactly at
/// the type's all-ones index; the index read adds a bounded number of words; and the vertices
/// form writes flag 0 at `VertexIndex`'s own vertex.
#[test]
fn an_indexed_capture_reads_each_vertex_from_its_index() {
    let Some(bc) = fixture("stage_io") else {
        return;
    };
    let entry = "stage_io_vertex_position";
    let short = [5u32, 0, 0xffff, 3, 2, 0xfffe];
    let long = [7u32, u32::MAX, 1, 0x8000_0001];
    let runs = [
        IndexedRun {
            ty: IndexType::UInt16,
            indices: &short,
            skew: 2,
            base_vertex: -3,
            first_instance: 4,
            instances: 2,
            restarts: true,
        },
        IndexedRun {
            ty: IndexType::UInt16,
            indices: &short,
            skew: 0,
            base_vertex: 9,
            first_instance: 0,
            instances: 1,
            restarts: false,
        },
        IndexedRun {
            ty: IndexType::UInt32,
            indices: &long,
            skew: 0,
            base_vertex: 0x10,
            first_instance: 1,
            instances: 2,
            restarts: true,
        },
    ];
    for stem in ["lavapipe", "moltenvk-m3"] {
        let dev = device_of(description(stem));
        let p = dev.profile();
        let vertices = compile_with(&bc, entry, p, &inputs(Some(CaptureForm::Vertices)))
            .unwrap_or_else(|e| panic!("{stem}: {e}"));
        let vertices_words = validate(&vertices, p).len();
        for run in &runs {
            let what = format!("{stem} {:?} skew {}", run.ty, run.skew);
            let i = indexed_inputs(run.ty, None);
            let a = compile_with(&bc, entry, p, &i).unwrap_or_else(|e| panic!("{what}: {e}"));
            let words = validate(&a, p).len();
            assert!(
                a.facts.contains(&RuntimeFact::CapturedPositions {
                    form: CaptureForm::Indexed(run.ty)
                }),
                "{what}"
            );
            assert!(!a.interface.writes_memory, "{what}");
            let added = words as i64 - vertices_words as i64;
            println!("{what}: the index read adds {added} words to {vertices_words}");
            assert!((0..=300).contains(&added), "{what}: {added} words");
            let (plan, m) =
                typed_module_with(&bc, entry, p, &i).unwrap_or_else(|e| panic!("{what}: {e}"));
            let records = run_capture(&what, &plan, &m, run, 0, Vec::new());
            for (slot, record) in records.iter().enumerate() {
                let position = slot % run.count() as usize;
                if run.skipped(position) {
                    assert!(skipped_record(record), "{what}: slot {slot}: {record:x?}");
                    continue;
                }
                let vid = run.vertex(position);
                let clip = [(vid & 1) as f32, (vid >> 1) as f32, 0.0, 1.0];
                assert_eq!(
                    record[..4],
                    clip.map(f32::to_bits),
                    "{what}: slot {slot} sees vertex {vid:#x}"
                );
                assert_eq!(record[4], 1.0f32.to_bits(), "{what}: slot {slot}");
                assert_eq!(
                    record[5],
                    u32::from(run.restart(position)),
                    "{what}: {slot}"
                );
            }
        }
        // The vertices form: `VertexIndex` from the draw's first vertex, flag 0.
        let what = format!("{stem} vertices");
        let (plan, m) = typed_module_with(&bc, entry, p, &inputs(Some(CaptureForm::Vertices)))
            .unwrap_or_else(|e| panic!("{what}: {e}"));
        let mut push = vec![0u8; 128];
        for f in &plan.push.fields {
            let PushLoc::Push { offset } = f.loc else {
                panic!("{what}: {:?} spilled", f.field);
            };
            let at = offset as usize;
            let bytes: Vec<u8> = match f.field {
                PushField::CaptureAddress => RECORDS.to_le_bytes().to_vec(),
                PushField::CaptureLayout => [6u32, 3, 0, 0]
                    .iter()
                    .flat_map(|w| w.to_le_bytes())
                    .collect(),
                other => panic!("{what}: unexpected push field {other:?}"),
            };
            push[at..at + bytes.len()].copy_from_slice(&bytes);
        }
        let res = Resources {
            buffers: vec![addressed(RECORDS, vec![FILL; 32 * 3])],
            push,
            spec: Default::default(),
        };
        let cfg = Config {
            workgroups: [3, 1, 1],
            first: [6, 0],
            ..Config::default()
        };
        let out = interp_run(&m, &cfg, res).unwrap_or_else(|e| panic!("{what}: {e}"));
        for (slot, r) in out.buffers[0].as_chunks::<32>().0.iter().enumerate() {
            let vid = 6 + slot as u32;
            let w = |k: usize| u32::from_le_bytes(r[4 * k..4 * k + 4].try_into().expect("word"));
            assert_eq!(w(0), ((vid & 1) as f32).to_bits(), "{what}: {slot}");
            assert_eq!(w(1), ((vid >> 1) as f32).to_bits(), "{what}: {slot}");
            assert_eq!(w(5), 0, "{what}: {slot}");
            assert_eq!(r[24..], [FILL; 8], "{what}: {slot}");
        }
    }
}

/// `stage_io_vertex`'s attributes, interleaved per vertex in buffer 0 at stride 32.
fn interleaved() -> VertexDescriptor {
    let mut d = VertexDescriptor::new();
    d.layouts
        .insert(0, VertexBufferLayoutDescriptor::new(Some(32)));
    for (index, format, offset) in [
        (0, VertexFormat::Float3, 0),
        (1, VertexFormat::Half2, 12),
        (4, VertexFormat::UShort2, 16),
        (7, VertexFormat::UChar4, 20),
    ] {
        d.attributes
            .insert(index, VertexAttributeDescriptor::new(format, offset, 0));
    }
    d
}

/// The class `stage_io_vertex` declares each attribute it reads as.
fn classes() -> BTreeMap<u32, IoNumeric> {
    BTreeMap::from([
        (0, IoNumeric::Float),
        (1, IoNumeric::Float),
        (4, IoNumeric::Uint),
        (7, IoNumeric::Uint),
    ])
}

/// The indexed form pulls every attribute, at the vertex its index names: the projection pulls
/// each one (where the plain projection fetches some in fixed function, a hand-built projection
/// that does is refused `NotImplemented(TileReach)`), and on the CPU interpreter
/// `stage_io_vertex`'s position, `float4(attribute 0, 1) * scale`, is the attribute of the
/// vertex each record's index names, including at a list's restart index whose vertex the base
/// vertex wraps back into the buffer. A list's restart index whose vertex lies past the buffer
/// reads zero on a device without buffer robustness, where the function checks its reads itself;
/// a strip's runs no body at all, even on a robust device, where nothing in the shader would
/// bound its fetch.
#[test]
fn an_indexed_capture_pulls_every_attribute_at_its_vertex() {
    let Some(bc) = fixture("stage_io") else {
        return;
    };
    let entry = "stage_io_vertex";
    let desc = interleaved();
    let indices = [3u32, 0, 7, 5, u32::MAX, 2];
    let run = IndexedRun {
        ty: IndexType::UInt32,
        indices: &indices,
        skew: 0,
        base_vertex: 2,
        first_instance: 0,
        instances: 1,
        restarts: false,
    };
    let far = [1u32, u32::MAX, 4];
    let strip = IndexedRun {
        ty: IndexType::UInt32,
        indices: &far,
        skew: 0,
        base_vertex: 0,
        first_instance: 0,
        instances: 1,
        restarts: true,
    };
    // Vertex `k`'s attribute 0 is `(k, 2k, 3k)`; buffer 0 holds vertices 0 to 15.
    let mut source = vec![0u8; 32 * 16];
    for k in 0..16usize {
        for (lane, v) in [k as f32, 2.0 * k as f32, 3.0 * k as f32]
            .iter()
            .enumerate()
        {
            source[32 * k + 4 * lane..32 * k + 4 * lane + 4]
                .copy_from_slice(&v.to_bits().to_le_bytes());
        }
    }
    let mut scale = vec![0u8; 256];
    for lane in 0..4 {
        scale[4 * lane..4 * lane + 4].copy_from_slice(&1.0f32.to_bits().to_le_bytes());
    }
    let mut fixed_seen = false;
    for stem in ["lavapipe", "moltenvk-m3"] {
        let dev = device_of(description(stem));
        let p = dev.profile();
        let plain =
            vertex_fetch_projection_for(Some(&desc), &classes(), p, &BTreeSet::new(), false)
                .expect("projection");
        let every = vertex_fetch_projection_for(Some(&desc), &classes(), p, &BTreeSet::new(), true)
            .expect("projection");
        assert_eq!(every.attributes.len(), 4, "{stem}");
        assert!(
            every
                .attributes
                .values()
                .all(|f| matches!(f, VertexFetch::Pulled { .. })),
            "{stem}: {every:?}"
        );
        if plain
            .attributes
            .values()
            .any(|f| matches!(f, VertexFetch::FixedFunction { .. }))
        {
            fixed_seen = true;
            let refusal = compile_with(
                &bc,
                entry,
                p,
                &indexed_inputs(IndexType::UInt32, Some(plain)),
            )
            .map(|_| ())
            .map_err(unlocated);
            assert_eq!(
                refusal,
                Err(Error::Unsupported(Unsupported::not_implemented(
                    Feature::TileReach
                ))),
                "{stem}"
            );
        }
        let what = format!("{stem} pulled");
        let i = indexed_inputs(IndexType::UInt32, Some(every));
        let a = compile_with(&bc, entry, p, &i).unwrap_or_else(|e| panic!("{what}: {e}"));
        validate(&a, p);
        let (plan, m) =
            typed_module_with(&bc, entry, p, &i).unwrap_or_else(|e| panic!("{what}: {e}"));
        let records = run_capture(
            &what,
            &plan,
            &m,
            &run,
            32,
            vec![
                vertex_buffer(0, source.clone()),
                vertex_buffer(1, scale.clone()),
            ],
        );
        for (slot, record) in records.iter().enumerate() {
            let k = run.vertex(slot);
            assert!(k < 16, "{what}: vertex {k}");
            let clip = [k as f32, 2.0 * k as f32, 3.0 * k as f32, 1.0];
            assert_eq!(
                record[..4],
                clip.map(f32::to_bits),
                "{what}: slot {slot} reads vertex {k}'s attribute"
            );
            assert_eq!(record[4], 3.0f32.to_bits(), "{what}: slot {slot}");
            assert_eq!(
                record[5],
                u32::from(run.restart(slot)),
                "{what}: slot {slot}"
            );
        }
        // A strip's restart index, whose vertex lies far past the buffer: the body never runs,
        // so the interpreter, which traps on any access outside a buffer (a robust profile
        // checks no read in the shader), sees no fetch there.
        let what = format!("{stem} strip");
        let records = run_capture(
            &what,
            &plan,
            &m,
            &strip,
            32,
            vec![
                vertex_buffer(0, source.clone()),
                vertex_buffer(1, scale.clone()),
            ],
        );
        for (slot, record) in records.iter().enumerate() {
            if strip.skipped(slot) {
                assert!(skipped_record(record), "{what}: slot {slot}: {record:x?}");
                continue;
            }
            let k = strip.vertex(slot) as f32;
            let clip = [k, 2.0 * k, 3.0 * k, 1.0];
            assert_eq!(record[..4], clip.map(f32::to_bits), "{what}: slot {slot}");
            assert_eq!(record[5], 0, "{what}: slot {slot}");
        }
    }
    assert!(
        fixed_seen,
        "no profile fetches an attribute in fixed function"
    );
    // A restart index whose vertex lies far past the buffer, on a device without buffer
    // robustness, whose reads the function checks itself: every pulled read stays inside the buffer
    // and reads zero past it, so the interpreter, which traps on any access outside a buffer, sees
    // none, and the record holds the zero attribute's position.
    let mut fragile = description("moltenvk-m3");
    fragile.features.remove(&F::RobustBufferAccess);
    fragile.features.remove(&F::RobustBufferAccess2);
    let dev = device_of(fragile);
    let p = dev.profile();
    assert_eq!(contain::encoding(p), contain::Check::InShader);
    let run = IndexedRun {
        ty: IndexType::UInt32,
        indices: &far,
        skew: 0,
        base_vertex: 0,
        first_instance: 0,
        instances: 1,
        restarts: false,
    };
    let what = "moltenvk-m3 without robustness, a list";
    let every = vertex_fetch_projection_for(Some(&desc), &classes(), p, &BTreeSet::new(), true)
        .expect("projection");
    let i = indexed_inputs(IndexType::UInt32, Some(every));
    let a = compile_with(&bc, entry, p, &i).unwrap_or_else(|e| panic!("{what}: {e}"));
    validate(&a, p);
    let (plan, m) = typed_module_with(&bc, entry, p, &i).unwrap_or_else(|e| panic!("{what}: {e}"));
    let records = run_capture(
        what,
        &plan,
        &m,
        &run,
        32,
        vec![vertex_buffer(0, source), vertex_buffer(1, scale)],
    );
    for (slot, record) in records.iter().enumerate() {
        let k = match slot {
            1 => 0.0,
            _ => run.vertex(slot) as f32,
        };
        let clip = [k, 2.0 * k, 3.0 * k, 1.0];
        assert_eq!(record[..4], clip.map(f32::to_bits), "{what}: slot {slot}");
        assert_eq!(
            record[5],
            u32::from(run.restart(slot)),
            "{what}: slot {slot}"
        );
    }
}

/// `stage_io_vertex`'s buffers: its interleaved vertices at 0 and `scale` at 1.
fn indexed_state() -> BindState {
    let buffer = |length| BoundBuffer::Buffer {
        length,
        offset: 0,
        gpu_address: 0x1_0000,
        device_address: Some(0x7_0000_0000),
        attribute_stride: None,
    };
    let mut v = StageTable::default();
    v.buffers.insert(0, buffer(32 * 64));
    v.buffers.insert(1, buffer(256));
    let mut s = BindState::default();
    s.stages.insert(Stage::Vertex, v);
    s
}

/// The indexed capture variant of `stage_io_vertex`'s render pipeline: its keys name the index
/// type, the pipeline pulls every attribute (link accepts it), its plan settles on its key, and
/// an indexed draw of its type runs without indices as `index count` vertices from 0 with the
/// draw's instances, leaves both capture holes in the vertex window, writes `[restarts, index
/// count, base instance, base vertex]` (`restarts` from the state's primitive type: 1 for a
/// strip, 0 for a list, `MissingPrimitiveType` without one), and reads its indices in the
/// vertex stage as storage. A draw of
/// the other form asks for that form's variant; an indexed draw whose indices pass its buffer or
/// whose offset is not a multiple of the index size, and an indirect one, are
/// `NotImplemented(TileReach)`.
#[test]
fn an_indexed_capture_variant_draws_without_indices() {
    let Some(bc) = fixture("stage_io") else {
        return;
    };
    let budget = Budget::standard();
    let lib = Library::from_air(&bc).expect("library");
    let function = FunctionDescriptor::new(lib.function("stage_io_vertex").expect("function"));
    let mut d = RenderPipelineDescriptor::new(StageSource::Function(function));
    d.vertex_descriptor = Some(interleaved());
    let short = CaptureForm::Indexed(IndexType::UInt16);
    let mut key = VariantKey::new();
    key.captured_positions = Some(short);
    for stem in ["lavapipe", "moltenvk-m3"] {
        let dev = device_of(description(stem));
        let entries = crate::pipeline::StageEntries::new();
        let keys = dev
            .render_variant_key(&d, &key, &budget, &entries)
            .expect("keys");
        let mut long_key = VariantKey::new();
        long_key.captured_positions = Some(CaptureForm::Indexed(IndexType::UInt32));
        let long_keys = dev
            .render_variant_key(&d, &long_key, &budget, &entries)
            .expect("keys");
        assert_ne!(keys.vertex, long_keys.vertex, "{stem}");
        let vertex_inputs = keys.vertex_inputs.clone().expect("vertex inputs");
        assert_eq!(vertex_inputs.captured_positions, Some(short), "{stem}");
        let projection = vertex_inputs.vertex_fetch.expect("projection");
        assert!(
            projection
                .attributes
                .values()
                .all(|f| matches!(f, VertexFetch::Pulled { .. })),
            "{stem}: {projection:?}"
        );
        let p = dev
            .compile_render_variant(&d, &key, &budget)
            .unwrap_or_else(|e| panic!("{stem}: {e}"));
        assert!(p.vertex_input.bindings.is_empty(), "{stem}");
        assert!(p.vertex_input.attributes.is_empty(), "{stem}");
        assert_eq!(p.vertex_input.pulled.len(), 4, "{stem}");
        for m in &p.stages {
            check_words_valid(&m.spirv, &p.requirements, dev.profile())
                .unwrap_or_else(|e| panic!("{stem} {:?}: L3: {e}", m.stage));
        }
        assert!(
            p.dependencies
                .iter()
                .any(|d| d.fact == RuntimeFact::CapturedPositions { form: short }
                    && d.mechanism == Mechanism::Variant),
            "{stem}"
        );
        let mut state = indexed_state();
        state.primitive_type = Some(metal::PrimitiveType::TriangleStrip);
        let plan = p.plan(&dev, &state.shape()).expect("plan");
        assert_eq!(plan.variant, key, "{stem}");
        let indexed = |index_type, index_count, index_offset| DrawCall::Indexed {
            index_type,
            index_buffer: IndexBufferRef::new(5, 4096),
            index_count,
            index_offset,
            instance_count: 2,
            base_vertex: -3,
            base_instance: 1,
        };
        let commands = plan
            .instantiate_draw(&state, &indexed(vk::IndexType::UINT16, 6, 8))
            .unwrap_or_else(|e| panic!("{stem}: {e:?}"));
        assert_eq!(
            commands.draw,
            vk::DrawCommand::Draw {
                vertex_count: 6,
                instance_count: 2,
                first_vertex: 0,
                first_instance: 1,
            },
            "{stem}"
        );
        assert_eq!(commands.index_overrun, None, "{stem}");
        let mut holes: Vec<(&str, u32)> = commands
            .push_holes
            .iter()
            .filter_map(|h| match *h {
                PushHole::CaptureAddress { stages, offset } => {
                    assert_eq!(stages, vk::ShaderStageFlags::VERTEX, "{stem}");
                    Some(("records", offset))
                }
                PushHole::CaptureIndexAddress { stages, offset } => {
                    assert_eq!(stages, vk::ShaderStageFlags::VERTEX, "{stem}");
                    Some(("indices", offset))
                }
                _ => None,
            })
            .collect();
        holes.sort();
        let [("indices", indices), ("records", records)] = holes[..] else {
            panic!("{stem}: {:?}", commands.push_holes);
        };
        assert_eq!(
            indices,
            records + 8,
            "{stem}: the index address follows the records'"
        );
        assert!(indices + 8 <= 64, "{stem}: {indices}");
        let layout = plan
            .push
            .fields
            .iter()
            .find(|f| f.stage == Stage::Vertex && f.field == PushField::CaptureLayout)
            .expect("placed");
        let crate::bind::FieldLocation::PushWindow { offset } = layout.location else {
            panic!("{stem}: the layout spilled: {layout:?}");
        };
        let write = commands
            .push_constants
            .iter()
            .find(|w| w.stages.contains(vk::ShaderStageFlags::VERTEX))
            .expect("vertex push");
        let at = (offset - write.offset) as usize;
        let words: Vec<u32> = write.bytes[at..at + 16]
            .chunks(4)
            .map(|w| u32::from_le_bytes(w.try_into().expect("a word")))
            .collect();
        assert_eq!(words, [1, 6, 1, (-3i32) as u32], "{stem}");
        // A list shades its restart index: lane 0 is 0. Without a primitive type the draw
        // cannot say.
        let mut list = state.clone();
        list.primitive_type = Some(metal::PrimitiveType::Triangle);
        let commands = plan
            .instantiate_draw(&list, &indexed(vk::IndexType::UINT16, 6, 8))
            .unwrap_or_else(|e| panic!("{stem}: {e:?}"));
        let write = commands
            .push_constants
            .iter()
            .find(|w| w.stages.contains(vk::ShaderStageFlags::VERTEX))
            .expect("vertex push");
        let at = (offset - write.offset) as usize;
        assert_eq!(write.bytes[at..at + 4], [0; 4], "{stem}");
        let mut unsaid = state.clone();
        unsaid.primitive_type = None;
        assert_eq!(
            plan.instantiate_draw(&unsaid, &indexed(vk::IndexType::UINT16, 6, 8))
                .map(|_| ()),
            Err(BindError::MissingPrimitiveType),
            "{stem}"
        );
        assert_eq!(
            commands.uses,
            [ResourceUse {
                resource: UseTarget::IndexBuffer,
                access: AccessSet::READ,
                class: AccessClass::Storage,
                stages: vk::PipelineStageFlags2::VERTEX_SHADER,
                range: Some(ByteRange { offset: 0, len: 12 }),
            }],
            "{stem}"
        );

        // A draw of the other form asks for its variant.
        let needs = |form| {
            Err(BindError::NeedsVariant {
                variant: Box::new(VariantKey {
                    captured_positions: Some(form),
                    ..key.clone()
                }),
            })
        };
        let primitives = DrawCall::Primitives {
            vertex_start: 0,
            vertex_count: 3,
            instance_count: 1,
            base_instance: 0,
        };
        assert_eq!(
            plan.instantiate_draw(&state, &primitives).map(|_| ()),
            needs(CaptureForm::Vertices),
            "{stem}"
        );
        assert_eq!(
            plan.instantiate_draw(&state, &indexed(vk::IndexType::UINT32, 6, 8))
                .map(|_| ()),
            needs(CaptureForm::Indexed(IndexType::UInt32)),
            "{stem}"
        );
        let not_implemented = Err(BindError::Unsupported(Unsupported::not_implemented(
            Feature::TileReach,
        )));
        for (case, draw) in [
            ("misaligned", indexed(vk::IndexType::UINT16, 6, 3)),
            ("past the buffer", indexed(vk::IndexType::UINT16, 4000, 8)),
            (
                "indirect",
                DrawCall::Indirect {
                    indirect: IndirectRef::new(2, 0),
                    indexed: Some((vk::IndexType::UINT16, IndexBufferRef::new(5, 4096), 0)),
                },
            ),
        ] {
            assert_eq!(
                plan.instantiate_draw(&state, &draw).map(|_| ()),
                not_implemented,
                "{stem} {case}"
            );
        }
    }
}
