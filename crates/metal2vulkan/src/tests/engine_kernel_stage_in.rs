//! Kernel `[[stage_in]]` through the pulled fetch, end to end on the CPU interpreter. No device is
//! opened.
//!
//! Each kernel of the owned fixture `kernel_stage_in` goes through the public compile API under a
//! compute pipeline's stage-input descriptor, `ComputePipeline::plan`, and
//! `BindingPlan::instantiate_dispatch`; the harness runs the typed module on the interpreter with
//! exactly what the binder planned (descriptor writes, every region's push bytes, its
//! specialization and group count), in both scheduling orders. The source buffers hold one
//! distinct value per element, so an element taken from the wrong step, rate, origin, stride,
//! offset, or index lands a value that exists nowhere else.
//!
//! - Every step Metal defines for a compute layout: `Constant`, `ThreadPositionInGridX`/`Y` at
//!   rate 1 (from the stage-in region's origin) and above it, and `...Indexed` through 16- and
//!   32-bit indices; attributes sharing a layout, missing components, sign extension, and half
//!   bits.
//! - `[[grid_origin]]` and `[[grid_size]]` read the region the binder writes, as 32-bit vectors
//!   and 16-bit pairs.
//! - A layout of a fixed stride reads it whatever stride the buffer is bound with; one of
//!   `MTLBufferLayoutStrideDynamic` reads the bound stride, 0 without one.
//! - A thread past the stage-in region reads the region's last element on each axis, before an
//!   index lookup and with the origin added after it; a region of size 0 clamps nothing. The
//!   descriptor refusals are typed.
//! - A `Compiled` kernel's fetches must be the pipeline descriptor's; keys take the descriptor
//!   canonically.
//! - Every committed profile validates the modules (L3), and the size estimate bounds them.
//!
//! `llvm-as` assembles the fixture; without it the tests skip visibly.

use std::collections::BTreeMap;
use std::path::Path;

use crate::bind::{
    BindState, BoundBuffer, Dispatch, DispatchCommands, ExecutionStatus, PushHole, RootTable,
    StageInRegion, StageTable, WriteSource,
};
use crate::budget::{Budget, SPIRV_WORDS};
use crate::device::profile::BufferAccess;
use crate::device::requirements::VkFeature;
use crate::device::{Device, DeviceDescription, DevicePolicy};
use crate::error::{BindError, DescriptorError, Error, Feature, LowerFamily, Needs, RuntimeInput};
use crate::library::{FunctionConstantValues, Library};
use crate::metal::{AttributeFormat, IndexType, StepFunction};
use crate::pipeline::compile::{compile_bitcode, plan_bitcode, EntryRef};
use crate::pipeline::descriptor::{
    StageInputAttributeDescriptor, StageInputDescriptor, StageInputLayoutDescriptor,
};
use crate::pipeline::facts::RuntimeFact;
use crate::pipeline::layout::BindingSource;
use crate::pipeline::lower_inputs::LoweringInputs;
use crate::pipeline::sym::{ResourceKind, ResourceSlot, SlotClass, SlotRef, SpecKey, Stage};
use crate::pipeline::testing::{
    engine_dir, link_and_validate, llvm_assemble, meter, typed_module_with, AssembleError,
};
use crate::pipeline::{
    ComputePipeline, ComputePipelineDescriptor, FunctionDescriptor, StageSource,
};
use crate::spirv::interp::{run, Buffer, Config, Order, Resources};
use crate::spirv::ir::TypedModule;
use crate::spirv::testing::committed_profiles;

/// Device addresses of the consumer-owned objects the harness places.
const TABLE: u64 = 0x0000_0010_0000_0000;
const SINK: u64 = 0x0000_0011_0000_0000;
const STATUS: u64 = 0x0000_0012_0000_0000;

/// Profiles the dispatches run on, each also without buffer robustness (the in-shader check).
const PROFILES: [&str; 3] = ["moltenvk-m3", "lavapipe", "vk12-floor"];

fn fixture() -> Option<Vec<u8>> {
    match llvm_assemble(&engine_dir().join("kernel_stage_in.ll")) {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected kernel_stage_in.ll: {why}"),
    }
}

/// Committed profile `stem` as a device, without buffer robustness when `bare`.
fn device(stem: &str, bare: bool) -> Device {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles")
        .join(format!("{stem}.json"));
    let text = std::fs::read_to_string(path).expect("profile");
    let mut desc = DeviceDescription::from_khronos_json(&text, None).expect("description");
    if bare {
        desc.features.remove(&VkFeature::RobustBufferAccess);
        desc.features.remove(&VkFeature::RobustBufferAccess2);
    }
    Device::new(desc, DevicePolicy::default()).expect("device")
}

/// Each profile of [`PROFILES`], and each without buffer robustness (`true`).
fn devices() -> Vec<(String, Device, bool)> {
    let mut out = Vec::new();
    for stem in PROFILES {
        out.push((stem.to_owned(), device(stem, false), false));
        out.push((
            format!("{stem} without robustness"),
            device(stem, true),
            true,
        ));
    }
    out
}

/// The pipeline, or `None` when the device lacks a capability it needs (a descriptor limit).
fn compiled(what: &str, r: Result<ComputePipeline, Error>) -> Option<ComputePipeline> {
    match r {
        Ok(p) => Some(p),
        Err(Error::Unsupported(u)) if u.needs != Needs::NotImplemented => {
            eprintln!("{what}: {u}");
            None
        }
        Err(e) => panic!("{what}: {e}"),
    }
}

/// A stage-input descriptor: `attributes` as `(index, format, offset, buffer)`, `layouts` as
/// `(buffer, stride, step, rate)`, and the index buffer.
fn descriptor(
    attributes: &[(u32, AttributeFormat, u32, u32)],
    layouts: &[(u32, u32, StepFunction, u32)],
    index: (IndexType, u32),
) -> StageInputDescriptor {
    let mut d = StageInputDescriptor::new();
    for &(i, format, offset, buffer) in attributes {
        d.attributes.insert(
            i,
            StageInputAttributeDescriptor::new(format, offset, buffer),
        );
    }
    for &(buffer, stride, step, rate) in layouts {
        let mut l = StageInputLayoutDescriptor::new(Some(stride));
        l.step_function = step;
        l.step_rate = rate;
        d.layouts.insert(buffer, l);
    }
    (d.index_type, d.index_buffer_index) = index;
    d
}

/// The fetch kernel's attributes: `f` (float3 into float4) at offset 4 and `h` (half2) at offset
/// 16 of layout 1, `u` (ushort4) of layout 2, and `i` (char) at offset 5 of layout 3.
fn fetch_attributes() -> [(u32, AttributeFormat, u32, u32); 4] {
    [
        (0, AttributeFormat::Float3, 4, 1),
        (1, AttributeFormat::UShort4, 0, 2),
        (2, AttributeFormat::Char, 5, 3),
        (3, AttributeFormat::Half2, 16, 1),
    ]
}

/// Layout 1 of stride 24, layout 2 of stride 8 with steps `x` and `y`, and layout 3 constant
/// at a step rate of 0, as Metal requires of a constant step (its stride is not read).
fn fetch_descriptor(
    x: (StepFunction, u32),
    y: (StepFunction, u32),
    index: IndexType,
) -> StageInputDescriptor {
    descriptor(
        &fetch_attributes(),
        &[
            (1, 24, x.0, x.1),
            (2, 8, y.0, y.1),
            (3, 999, StepFunction::Constant, 0),
        ],
        (index, 4),
    )
}

fn kslot(index: u32) -> SlotRef {
    SlotRef {
        stage: Stage::Kernel,
        class: SlotClass::Buffer,
        index,
    }
}

fn inputs(d: &StageInputDescriptor) -> LoweringInputs {
    let mut i = LoweringInputs::new();
    i.stage_input = Some(d.clone());
    i
}

fn function(bc: &[u8], entry: &str) -> FunctionDescriptor {
    let lib = Library::from_air(bc).expect("library");
    FunctionDescriptor::new(lib.function(entry).expect("function"))
}

fn compute(f: FunctionDescriptor, d: Option<&StageInputDescriptor>) -> ComputePipelineDescriptor {
    let mut c = ComputePipelineDescriptor::new(StageSource::Function(f));
    c.stage_input = d.cloned();
    c
}

fn pipeline(
    dev: &Device,
    bc: &[u8],
    entry: &str,
    d: Option<&StageInputDescriptor>,
) -> Result<ComputePipeline, Error> {
    dev.compile_compute_pipeline(&compute(function(bc, entry), d), &Budget::standard())
}

/// A bound Metal buffer.
#[derive(Clone)]
struct Alloc {
    bytes: Vec<u8>,
    offset: u64,
}

/// What the consumer has: bound kernel buffers by index, the attribute strides they are bound
/// with (`setBuffer:offset:attributeStride:atIndex:`), the stage-in region the encoder set (if
/// any), the status buffer.
struct World {
    buffers: BTreeMap<u32, Alloc>,
    strides: BTreeMap<u32, u32>,
    region: Option<StageInRegion>,
    status: [u8; 32],
}

impl World {
    fn new(
        buffers: impl IntoIterator<Item = (u32, Alloc)>,
        region: impl Into<Option<StageInRegion>>,
    ) -> World {
        World {
            buffers: buffers.into_iter().collect(),
            strides: BTreeMap::new(),
            region: region.into(),
            status: ExecutionStatus::initial(),
        }
    }

    /// The world with each buffer of `strides` bound at its attribute stride.
    fn with_strides(mut self, strides: impl IntoIterator<Item = (u32, u32)>) -> World {
        self.strides = strides.into_iter().collect();
        self
    }

    fn state(&self) -> BindState {
        let mut t = StageTable::default();
        for (&i, a) in &self.buffers {
            t.buffers.insert(
                i,
                BoundBuffer::Buffer {
                    length: a.bytes.len() as u64,
                    offset: a.offset,
                    gpu_address: 0,
                    device_address: None,
                    attribute_stride: self.strides.get(&i).copied(),
                },
            );
        }
        let mut s = BindState::default();
        s.stages.insert(Stage::Kernel, t);
        s.stage_in_region = self.region;
        s
    }
}

fn padded(mut v: Vec<u8>) -> Vec<u8> {
    v.resize(v.len().next_multiple_of(16).max(16), 0);
    v
}

/// Where each interpreter buffer's final bytes go.
enum Back {
    Slice { slot: u32, offset: usize },
    Status,
    Nowhere,
}

fn symbol(source: BindingSource) -> ResourceSlot {
    match source {
        BindingSource::Buffer(slot) => ResourceSlot {
            stage: slot.stage,
            kind: ResourceKind::Buffer,
            index: slot.index,
        },
        BindingSource::Synthetic { stage, binding } => ResourceSlot {
            stage,
            kind: ResourceKind::Synthetic(binding),
            index: 0,
        },
        other => panic!("a compute binding {other:?}"),
    }
}

/// Plan, instantiate, and run `dispatch` of `pipe` (whose typed module is `m`) over `world`.
fn execute(
    dev: &Device,
    pipe: &ComputePipeline,
    m: &TypedModule,
    world: &mut World,
    dispatch: &Dispatch,
) -> Result<DispatchCommands, BindError> {
    let state = world.state();
    let plan = pipe.plan(dev, &state.shape())?;
    let cmds = plan.instantiate_dispatch(&state, dispatch)?;
    let layout: BTreeMap<(u32, u32), BindingSource> = pipe
        .layout
        .sets
        .iter()
        .flat_map(|s| {
            s.bindings
                .iter()
                .map(move |b| ((s.set, b.binding), b.source))
        })
        .collect();
    let ids: BTreeMap<u32, SpecKey> = pipe
        .stage
        .specialization
        .constants
        .iter()
        .map(|c| (c.id, c.key))
        .collect();
    for region in &cmds.regions {
        let mut buffers = Vec::new();
        let mut back = Vec::new();
        for w in &plan.descriptors {
            let sym = symbol(layout[&(w.set, w.binding)]);
            let range = cmds
                .descriptor_ranges
                .iter()
                .find(|r| (r.set, r.binding) == (w.set, w.binding))
                .map(|r| r.range);
            let (bytes, to) = match &w.source {
                WriteSource::Buffer { slot } => {
                    let r = range.expect("a range for every buffer write");
                    let all = padded(world.buffers[&slot.index].bytes.clone());
                    let (o, n) = (r.offset as usize, r.range as usize);
                    (
                        all[o..o + n].to_vec(),
                        Back::Slice {
                            slot: slot.index,
                            offset: o,
                        },
                    )
                }
                WriteSource::Neutral(_) => (vec![0; 4096], Back::Nowhere),
                WriteSource::ArgumentTable { .. } => {
                    let d = cmds
                        .descriptor_bytes
                        .iter()
                        .find(|d| (d.set, d.binding) == (w.set, w.binding))
                        .expect("argument table bytes");
                    (d.bytes.clone(), Back::Nowhere)
                }
                WriteSource::ConstantBlob { .. } => (
                    padded(pipe.stage.constant_blob.as_deref().expect("blob").to_vec()),
                    Back::Nowhere,
                ),
                other => panic!("unexpected write {other:?}"),
            };
            buffers.push(Buffer {
                slot: Some(sym),
                address: None,
                bytes,
            });
            back.push(to);
        }
        let mut push = Vec::new();
        if let Some(w) = region.push_constants.first() {
            push = vec![0u8; (w.offset as usize) + w.bytes.len()];
            push[w.offset as usize..].copy_from_slice(&w.bytes);
        }
        for hole in &cmds.push_holes {
            let (offset, address) = match *hole {
                PushHole::RootTableAddress { offset, .. } => (offset, TABLE),
                PushHole::ExecutionStatusAddress { offset, .. } => (offset, STATUS),
                other => panic!("{other:?}"),
            };
            let at = offset as usize;
            push[at..at + 8].copy_from_slice(&address.to_le_bytes());
            if address == STATUS {
                buffers.push(Buffer {
                    slot: None,
                    address: Some(STATUS),
                    bytes: world.status.to_vec(),
                });
                back.push(Back::Status);
            }
        }
        if let Some(RootTable { bytes, .. }) = &cmds.root_table {
            let mut bytes = bytes.clone();
            let at = RootTable::SINK_HOST_OFFSET as usize;
            bytes[at..at + 8].copy_from_slice(&SINK.to_le_bytes());
            buffers.push(Buffer {
                slot: None,
                address: Some(TABLE),
                bytes,
            });
            back.push(Back::Nowhere);
            buffers.push(Buffer {
                slot: None,
                address: Some(SINK),
                bytes: vec![0; 64],
            });
            back.push(Back::Nowhere);
        }
        let mut spec = BTreeMap::new();
        for e in &region.specialization.entries {
            let at = e.offset as usize;
            let mut v = [0u8; 8];
            v[..e.size as usize]
                .copy_from_slice(&region.specialization.data[at..at + e.size as usize]);
            spec.insert(ids[&e.constant_id], u64::from_le_bytes(v));
        }
        let res = Resources {
            buffers,
            push,
            spec,
        };
        let mut outs = Vec::new();
        for order in [Order::Forward, Order::Reverse] {
            let cfg = Config {
                workgroups: region.group_count,
                order,
                ..Config::default()
            };
            outs.push(run(m, &cfg, res.clone()).expect("run").buffers);
        }
        assert_eq!(outs[0], outs[1], "the two scheduling orders agree");
        for (bytes, to) in outs.swap_remove(0).into_iter().zip(back) {
            match to {
                Back::Slice { slot, offset } => {
                    let a = world.buffers.get_mut(&slot).expect("slot");
                    for (i, b) in bytes.into_iter().enumerate() {
                        if let Some(dst) = a.bytes.get_mut(offset + i) {
                            *dst = b;
                        }
                    }
                }
                Back::Status => world.status.copy_from_slice(&bytes[..32]),
                Back::Nowhere => {}
            }
        }
    }
    Ok(cmds)
}

fn words(bytes: &[u8]) -> Vec<u32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes(*c))
        .collect()
}

/// The dispatch grid of the fetch kernel: 5 x 3 threads in 2 x 2 threadgroups (edges on both
/// axes).
const W: u32 = 5;
const H: u32 = 3;

fn threads(w: u32, h: u32) -> Dispatch {
    Dispatch::Threads {
        threads: [w, h, 1],
        threads_per_group: [2, 2, 1],
    }
}

/// Elements the source buffers hold: enough for every step below, an index plus an origin
/// included.
const ELEMENTS: u32 = 24;

/// Indices the index buffer holds.
const INDICES: u32 = 16;

/// Layout 1's element `e` (stride 24): `f = (10e + 1, 10e + 2, 10e + 3)` at offset 4 and
/// `h = (bits 0x3c00 + e, 0x4000 + e)` at offset 16.
fn layout1() -> Vec<u8> {
    let mut b = vec![0xAA; (24 * ELEMENTS) as usize];
    for e in 0..ELEMENTS {
        let at = (24 * e) as usize;
        for k in 0..3u32 {
            let v = (10 * e + k + 1) as f32;
            b[at + 4 + 4 * k as usize..][..4].copy_from_slice(&v.to_le_bytes());
        }
        b[at + 16..][..2].copy_from_slice(&(0x3c00 + e as u16).to_le_bytes());
        b[at + 18..][..2].copy_from_slice(&(0x4000 + e as u16).to_le_bytes());
    }
    b
}

/// Layout 2's element `e` (stride 8): `u = (e, e + 100, e + 200, 0xffff - e)` as `ushort4`.
fn layout2() -> Vec<u8> {
    let mut b = Vec::new();
    for e in 0..ELEMENTS as u16 {
        for v in [e, e + 100, e + 200, 0xffff - e] {
            b.extend_from_slice(&v.to_le_bytes());
        }
    }
    b
}

/// Layout 3: the constant `i`, a `char` of -7 at offset 5.
fn layout3() -> Vec<u8> {
    let mut b = vec![0x11; 16];
    b[5] = (-7i8) as u8;
    b
}

/// The index buffer: `index[k] = (7k + 3) mod INDICES`, in 2- or 4-byte indices.
fn index_of(k: u32) -> u32 {
    (7 * k + 3) % INDICES
}

fn index_buffer(t: IndexType) -> Vec<u8> {
    let mut b = Vec::new();
    for k in 0..INDICES {
        match t {
            IndexType::UInt16 => b.extend_from_slice(&(index_of(k) as u16).to_le_bytes()),
            _ => b.extend_from_slice(&index_of(k).to_le_bytes()),
        }
    }
    b
}

/// The three `uint4` the fetch kernel writes for a thread that reads layout 1's element `ex` and
/// layout 2's element `ey`, at position `t`.
fn expected_fetch(ex: u32, ey: u32, t: u32) -> [u32; 12] {
    let f = |k: u32| ((10 * ex + k + 1) as f32).to_bits();
    [
        f(0),
        f(1),
        f(2),
        1.0f32.to_bits(),
        ey,
        ey + 100,
        ey + 200,
        0xffff - ey,
        (-7i32) as u32,
        0x3c00 + ex,
        0x4000 + ex,
        t,
    ]
}

/// The element a step of a layout reads for thread position `at` on its axis under a stage-in
/// region of `origin` and `size` on that axis, as an M4 and an M3 Max read it: the position divided
/// by the rate, or the index at the position, clamped to the region's last element (its size less
/// 1, which wraps at a size of 0 and clamps nothing), plus the origin.
fn element(step: StepFunction, rate: u32, origin: u32, size: u32, at: u32) -> u32 {
    let last = size.wrapping_sub(1);
    match step {
        StepFunction::ThreadPositionInGridX | StepFunction::ThreadPositionInGridY => {
            origin.wrapping_add((at / rate).min(last))
        }
        StepFunction::ThreadPositionInGridXIndexed | StepFunction::ThreadPositionInGridYIndexed => {
            index_of(at.min(last)).wrapping_add(origin)
        }
        other => panic!("{other:?}"),
    }
}

/// The stage-in region of the fetch grid from `origin`.
fn grid_region(origin: [u32; 3]) -> StageInRegion {
    StageInRegion::new(origin, [W, H, 1])
}

/// Run the fetch kernel under `x` (layout 1) and `y` (layout 2) with stage-in region `region` on
/// every device, and compare every thread's report.
fn run_fetch(
    bc: &[u8],
    x: (StepFunction, u32),
    y: (StepFunction, u32),
    index: IndexType,
    region: StageInRegion,
) {
    let d = fetch_descriptor(x, y, index);
    let mut ran = 0;
    for (what, dev, _) in devices() {
        let Some(pipe) = compiled(&what, pipeline(&dev, bc, "stage_in_fetch", Some(&d))) else {
            continue;
        };
        ran += 1;
        let (_, m) = typed_module_with(bc, "stage_in_fetch", dev.profile(), &inputs(&d))
            .unwrap_or_else(|e| panic!("{what}: {e}"));
        let mut world = World::new(
            [
                (
                    0,
                    Alloc {
                        bytes: vec![0; (48 * W * H) as usize],
                        offset: 0,
                    },
                ),
                (
                    1,
                    Alloc {
                        bytes: layout1(),
                        offset: 0,
                    },
                ),
                (
                    2,
                    Alloc {
                        bytes: layout2(),
                        offset: 0,
                    },
                ),
                (
                    3,
                    Alloc {
                        bytes: layout3(),
                        offset: 0,
                    },
                ),
                (
                    4,
                    Alloc {
                        bytes: index_buffer(index),
                        offset: 0,
                    },
                ),
            ],
            region,
        );
        execute(&dev, &pipe, &m, &mut world, &threads(W, H))
            .unwrap_or_else(|e| panic!("{what}: {e}"));
        let out = words(&world.buffers[&0].bytes);
        let (origin, size) = (region.origin, region.size);
        for gy in 0..H {
            for gx in 0..W {
                let t = gy * W + gx;
                let ex = element(x.0, x.1, origin[0], size[0], gx);
                let ey = element(y.0, y.1, origin[1], size[1], gy);
                let at = (12 * t) as usize;
                assert_eq!(
                    out[at..at + 12],
                    expected_fetch(ex, ey, t),
                    "{what} {x:?} {y:?} {index:?} {region:?}: thread ({gx}, {gy})"
                );
            }
        }
    }
    assert!(ran >= PROFILES.len(), "ran on {ran} devices");
}

/// Steps by thread position at rate 1 read the element at the region's origin plus the position, on
/// each axis; the constant step reads its offset alone; attributes sharing a layout read the same
/// element; `float3` into `float4` reads `w = 1`, `char` sign-extends, and half bits reach a half
/// attribute exactly.
#[test]
fn grid_steps_read_the_element_at_the_origin_plus_the_position() {
    let Some(bc) = fixture() else { return };
    let x = (StepFunction::ThreadPositionInGridX, 1);
    let y = (StepFunction::ThreadPositionInGridY, 1);
    for origin in [[0, 0, 0], [2, 1, 4]] {
        run_fetch(&bc, x, y, IndexType::UInt16, grid_region(origin));
    }
    // The steps swapped: layout 1 by y and layout 2 by x.
    let (sx, sy) = (
        (StepFunction::ThreadPositionInGridY, 1),
        (StepFunction::ThreadPositionInGridX, 1),
    );
    let d = fetch_descriptor(sx, sy, IndexType::UInt16);
    let dev = device("moltenvk-m3", false);
    let pipe = pipeline(&dev, &bc, "stage_in_fetch", Some(&d)).expect("pipeline");
    let (_, m) = typed_module_with(&bc, "stage_in_fetch", dev.profile(), &inputs(&d)).expect("m");
    let mut world = World::new(
        [
            (
                0,
                Alloc {
                    bytes: vec![0; (48 * W * H) as usize],
                    offset: 0,
                },
            ),
            (
                1,
                Alloc {
                    bytes: layout1(),
                    offset: 0,
                },
            ),
            (
                2,
                Alloc {
                    bytes: layout2(),
                    offset: 0,
                },
            ),
            (
                3,
                Alloc {
                    bytes: layout3(),
                    offset: 0,
                },
            ),
        ],
        StageInRegion::new([1, 3, 0], [W, H, 1]),
    );
    execute(&dev, &pipe, &m, &mut world, &threads(W, H)).expect("dispatch");
    let out = words(&world.buffers[&0].bytes);
    for gy in 0..H {
        for gx in 0..W {
            let t = gy * W + gx;
            let at = (12 * t) as usize;
            assert_eq!(
                out[at..at + 12],
                expected_fetch(3 + gy, 1 + gx, t),
                "({gx}, {gy})"
            );
        }
    }
}

/// A step rate `n > 1` reads a new element every `n` threads on its axis.
#[test]
fn step_rates_divide_the_position() {
    let Some(bc) = fixture() else { return };
    run_fetch(
        &bc,
        (StepFunction::ThreadPositionInGridX, 2),
        (StepFunction::ThreadPositionInGridY, 3),
        IndexType::UInt16,
        grid_region([0, 0, 0]),
    );
}

/// An indexed step reads the element the index buffer holds at the position, through 16- and 32-bit
/// indices; both axes read the one index buffer.
#[test]
fn indexed_steps_read_the_index_at_the_position() {
    let Some(bc) = fixture() else { return };
    for index in [IndexType::UInt16, IndexType::UInt32] {
        run_fetch(
            &bc,
            (StepFunction::ThreadPositionInGridXIndexed, 1),
            (StepFunction::ThreadPositionInGridYIndexed, 1),
            index,
            grid_region([0, 0, 0]),
        );
    }
    // An indexed step beside a step at rate 1 on the other axis, which adds its origin.
    run_fetch(
        &bc,
        (StepFunction::ThreadPositionInGridXIndexed, 1),
        (StepFunction::ThreadPositionInGridY, 1),
        IndexType::UInt32,
        grid_region([0, 2, 0]),
    );
}

/// A step at a rate above 1 adds the region's origin to the quotient, and an indexed step adds it
/// to the index it looks up at the position, not to the position looked up, as an M4 and an M3 Max
/// read; every step by thread position states that the kernel reads the region.
#[test]
fn a_nonzero_origin_adds_to_the_quotient_and_the_index() {
    let Some(bc) = fixture() else { return };
    let dev = device("moltenvk-m3", false);
    for x in [
        (StepFunction::ThreadPositionInGridX, 2),
        (StepFunction::ThreadPositionInGridXIndexed, 1),
        (StepFunction::ThreadPositionInGridX, 1),
    ] {
        let d = fetch_descriptor(x, (StepFunction::Constant, 0), IndexType::UInt16);
        let pipe = pipeline(&dev, &bc, "stage_in_fetch", Some(&d)).expect("pipeline");
        assert!(
            pipe.dependencies
                .iter()
                .any(|f| f.fact == RuntimeFact::StageInRegion),
            "{x:?}: {:?}",
            pipe.dependencies
        );
    }
    run_fetch(
        &bc,
        (StepFunction::ThreadPositionInGridX, 2),
        (StepFunction::ThreadPositionInGridY, 3),
        IndexType::UInt16,
        grid_region([2, 1, 4]),
    );
    for index in [IndexType::UInt16, IndexType::UInt32] {
        run_fetch(
            &bc,
            (StepFunction::ThreadPositionInGridXIndexed, 1),
            (StepFunction::ThreadPositionInGridYIndexed, 1),
            index,
            grid_region([1, 3, 0]),
        );
    }
}

/// A thread past the stage-in region reads the region's last element on its axis, each axis apart:
/// a step by thread position clamps its quotient, and an indexed step the position it looks up,
/// before the origin is added. A region of size 0 on an axis clamps nothing there (its size less 1
/// wraps), and a region of size 1 reads one element.
#[test]
fn a_thread_past_the_region_reads_its_last_element() {
    let Some(bc) = fixture() else { return };
    let grid = (
        (StepFunction::ThreadPositionInGridX, 1),
        (StepFunction::ThreadPositionInGridY, 1),
    );
    let rates = (
        (StepFunction::ThreadPositionInGridX, 2),
        (StepFunction::ThreadPositionInGridY, 3),
    );
    let indexed = (
        (StepFunction::ThreadPositionInGridXIndexed, 1),
        (StepFunction::ThreadPositionInGridYIndexed, 1),
    );
    for (x, y) in [grid, rates, indexed] {
        for region in [
            StageInRegion::new([0, 0, 0], [3, 2, 1]),
            StageInRegion::new([2, 1, 0], [3, 2, 1]),
            StageInRegion::new([1, 2, 0], [0, 0, 1]),
            StageInRegion::new([0, 3, 0], [1, 1, 1]),
            StageInRegion::new([2, 0, 0], [0, 1, 0]),
        ] {
            run_fetch(&bc, x, y, IndexType::UInt16, region);
        }
    }
}

/// The sixteen `MTLAttributeFormat` values that cross the numeric classes — the unnormalized 8-
/// and 16-bit integer codings, which Metal converts into a float attribute — each with its
/// component count, its component width in bits and its signedness.
const CROSSINGS: [(AttributeFormat, usize, u32, bool); 16] = [
    (AttributeFormat::UChar, 1, 8, false),
    (AttributeFormat::UChar2, 2, 8, false),
    (AttributeFormat::UChar3, 3, 8, false),
    (AttributeFormat::UChar4, 4, 8, false),
    (AttributeFormat::Char, 1, 8, true),
    (AttributeFormat::Char2, 2, 8, true),
    (AttributeFormat::Char3, 3, 8, true),
    (AttributeFormat::Char4, 4, 8, true),
    (AttributeFormat::UShort, 1, 16, false),
    (AttributeFormat::UShort2, 2, 16, false),
    (AttributeFormat::UShort3, 3, 16, false),
    (AttributeFormat::UShort4, 4, 16, false),
    (AttributeFormat::Short, 1, 16, true),
    (AttributeFormat::Short2, 2, 16, true),
    (AttributeFormat::Short3, 3, 16, true),
    (AttributeFormat::Short4, 4, 16, true),
];

/// The values a crossing format of `bits` bits and signedness `signed` carries on its components:
/// both ends of its range, zero and one, so a component read at the other signedness or the other
/// width lands a value none of the others holds. Each is exact in `f32`.
fn crossing_values(bits: u32, signed: bool) -> [i64; 4] {
    match (bits, signed) {
        (8, true) => [-128, -1, 0, 127],
        (8, false) => [0, 1, 200, 255],
        (_, true) => [-32768, -1, 0, 32767],
        (_, false) => [0, 1, 40000, 65535],
    }
}

/// Layout 1's element 0 holding `count` components of `bits` bits at byte `at` — the values
/// [`crossing_values`] gives them — over [`layout1`]'s own element 0, so every other attribute
/// of the element reads what the rest of this file expects of it.
fn crossing_element(at: usize, count: usize, bits: u32, signed: bool) -> Vec<u8> {
    let mut b = layout1();
    let width = (bits / 8) as usize;
    for (k, value) in crossing_values(bits, signed).iter().take(count).enumerate() {
        let raw = (*value as u64).to_le_bytes();
        b[at + k * width..][..width].copy_from_slice(&raw[..width]);
    }
    b
}

/// The twelve words the fetch kernel's one thread reports for a crossing format carried by
/// attribute `declared`: each component's integer value as the attribute's own float class, the
/// components the format lacks as `(0, 0, 0, 1)` on the `float4` and as zero on the `half2`, and
/// every other attribute as [`expected_fetch`]'s element 0.
fn expected_crossing(declared: u32, count: usize, bits: u32, signed: bool) -> [u32; 12] {
    let mut out = expected_fetch(0, 0, 0);
    let values = crossing_values(bits, signed);
    let value = |lane: usize| {
        if lane < count {
            values[lane] as f32
        } else {
            f32::from(lane == 3)
        }
    };
    if declared == 0 {
        for (lane, slot) in out[..4].iter_mut().enumerate() {
            *slot = value(lane).to_bits();
        }
    } else {
        for (lane, slot) in out[9..11].iter_mut().enumerate() {
            *slot = u32::from(crate::float16::f32_to_f16_bits(value(lane)));
        }
    }
    out
}

/// An unnormalized 8- or 16-bit integer format on a float-class `[[stage_in]]` attribute reads
/// each component's integer value, zero- or sign-extended by the format's signedness, as that
/// attribute's own float class — the conversion a pulled vertex attribute of the same format gets
/// (`fetch::class_match`'s `IntToFloat`), which Metal's stage-in fetch answers the same bits for as
/// its fixed-function vertex fetch. All sixteen crossing formats make the pipeline, on the `float4`
/// attribute and on the `half2` one, whose value is rounded to nearest even (65520 and above to
/// infinity).
#[test]
fn integer_formats_cross_into_a_float_attribute_as_their_values() {
    let Some(bc) = fixture() else { return };
    let dispatch = Dispatch::Threads {
        threads: [1, 1, 1],
        threads_per_group: [1, 1, 1],
    };
    let mut ran = 0;
    for (what, dev, _) in devices() {
        let mut here = 0;
        // Attribute 0 (`float4`) at byte 4 and attribute 3 (`half2`) at byte 16 of layout 1, each
        // in turn carrying the crossing format while the other three stay in their own class, so
        // nothing but the crossing is in question.
        for (declared, at) in [(0u32, 4usize), (3, 16)] {
            for (format, count, bits, signed) in CROSSINGS {
                let held = |index: u32, own: AttributeFormat| {
                    if index == declared {
                        format
                    } else {
                        own
                    }
                };
                let d = descriptor(
                    &[
                        (0, held(0, AttributeFormat::Float3), 4, 1),
                        (1, AttributeFormat::UShort4, 0, 2),
                        (2, AttributeFormat::Char, 5, 3),
                        (3, held(3, AttributeFormat::Half2), 16, 1),
                    ],
                    &[
                        (1, 24, StepFunction::ThreadPositionInGridX, 1),
                        (2, 8, StepFunction::ThreadPositionInGridY, 1),
                        (3, 999, StepFunction::Constant, 0),
                    ],
                    (IndexType::UInt16, 4),
                );
                let what = format!("{what} {format:?} on attribute {declared}");
                let Some(pipe) = compiled(&what, pipeline(&dev, &bc, "stage_in_fetch", Some(&d)))
                else {
                    continue;
                };
                let (_, m) = typed_module_with(&bc, "stage_in_fetch", dev.profile(), &inputs(&d))
                    .unwrap_or_else(|e| panic!("{what}: {e}"));
                let alloc = |bytes| Alloc { bytes, offset: 0 };
                let mut world = World::new(
                    [
                        (0, alloc(vec![0; 48])),
                        (1, alloc(crossing_element(at, count, bits, signed))),
                        (2, alloc(layout2())),
                        (3, alloc(layout3())),
                        (4, alloc(index_buffer(IndexType::UInt16))),
                    ],
                    StageInRegion::new([0; 3], [1, 1, 1]),
                );
                execute(&dev, &pipe, &m, &mut world, &dispatch)
                    .unwrap_or_else(|e| panic!("{what}: {e}"));
                assert_eq!(
                    words(&world.buffers[&0].bytes),
                    expected_crossing(declared, count, bits, signed),
                    "{what}"
                );
                here += 1;
            }
        }
        assert_eq!(here, 2 * CROSSINGS.len(), "{what}: made {here} pipelines");
        ran += 1;
    }
    assert!(ran >= PROFILES.len(), "ran on {ran} devices");
}

/// The fetch kernel's buffers: its output, layouts 1 to 3, and a 16-bit index buffer.
fn fetch_buffers() -> [(u32, Alloc); 5] {
    let alloc = |bytes| Alloc { bytes, offset: 0 };
    [
        (0, alloc(vec![0; (48 * W * H) as usize])),
        (1, alloc(layout1())),
        (2, alloc(layout2())),
        (3, alloc(layout3())),
        (4, alloc(index_buffer(IndexType::UInt16))),
    ]
}

/// What Metal reads as the stage-in region when the encoder never set one is not established, so a
/// kernel that reads the region (a step by thread position at any rate or through the index buffer,
/// `[[grid_origin]]`, `[[grid_size]]`) is not dispatched without it: `MissingStageInRegion`, before
/// anything is written. A kernel whose attributes all step `Constant` reads no region and
/// dispatches without one.
#[test]
fn a_kernel_that_reads_the_region_needs_the_encoders_region() {
    let Some(bc) = fixture() else { return };
    let dev = device("moltenvk-m3", false);
    let constant = (StepFunction::Constant, 0);
    for (x, y, reads) in [
        (constant, constant, false),
        ((StepFunction::ThreadPositionInGridX, 1), constant, true),
        ((StepFunction::ThreadPositionInGridX, 2), constant, true),
        (
            constant,
            (StepFunction::ThreadPositionInGridYIndexed, 1),
            true,
        ),
    ] {
        let d = fetch_descriptor(x, y, IndexType::UInt16);
        let pipe = pipeline(&dev, &bc, "stage_in_fetch", Some(&d)).expect("pipeline");
        let (_, m) =
            typed_module_with(&bc, "stage_in_fetch", dev.profile(), &inputs(&d)).expect("m");
        let mut world = World::new(fetch_buffers(), None);
        let r = execute(&dev, &pipe, &m, &mut world, &threads(W, H));
        if reads {
            assert_eq!(
                r.err(),
                Some(BindError::MissingStageInRegion),
                "{x:?} {y:?}"
            );
            let mut world = World::new(fetch_buffers(), StageInRegion::new([0; 3], [W, H, 1]));
            execute(&dev, &pipe, &m, &mut world, &threads(W, H))
                .unwrap_or_else(|e| panic!("{x:?} {y:?}: {e}"));
        } else {
            r.unwrap_or_else(|e| panic!("{x:?} {y:?}: {e}"));
            let out = words(&world.buffers[&0].bytes);
            // Every thread reads element 0 of layouts 1 and 2.
            let want = expected_fetch(0, 0, 0);
            for t in 0..W * H {
                let at = (12 * t) as usize;
                assert_eq!(out[at..at + 11], want[..11], "thread {t}");
            }
        }
    }
    let pipe = pipeline(&dev, &bc, "stage_in_region_short", None).expect("pipeline");
    let (_, m) = typed_module_with(
        &bc,
        "stage_in_region_short",
        dev.profile(),
        &LoweringInputs::new(),
    )
    .expect("m");
    let out = || {
        [(
            0,
            Alloc {
                bytes: vec![0; 8 * 4],
                offset: 0,
            },
        )]
    };
    let mut world = World::new(out(), None);
    assert_eq!(
        execute(&dev, &pipe, &m, &mut world, &threads(4, 1)).err(),
        Some(BindError::MissingStageInRegion)
    );
    let mut world = World::new(out(), StageInRegion::new([0; 3], [4, 1, 1]));
    execute(&dev, &pipe, &m, &mut world, &threads(4, 1)).expect("dispatch");
}

/// The three `uint4` the fetch kernel writes for a thread at position `t` that reads layout 1's
/// attributes at byte `a1` of [`layout1`] and layout 2's at byte `a2` of [`layout2`], decoded from
/// the bytes there whatever their alignment (an element boundary or not).
fn expected_at(a1: usize, a2: usize, t: u32) -> [u32; 12] {
    let (l1, l2) = (layout1(), layout2());
    let word = |b: &[u8], at: usize| u32::from_le_bytes(b[at..at + 4].try_into().expect("word"));
    let half = |b: &[u8], at: usize| u32::from(u16::from_le_bytes([b[at], b[at + 1]]));
    [
        word(&l1, a1 + 4),
        word(&l1, a1 + 8),
        word(&l1, a1 + 12),
        1.0f32.to_bits(),
        half(&l2, a2),
        half(&l2, a2 + 2),
        half(&l2, a2 + 4),
        half(&l2, a2 + 6),
        (-7i32) as u32,
        half(&l1, a1 + 16),
        half(&l1, a1 + 18),
        t,
    ]
}

/// Run the fetch kernel under `d` (layout 1 stepping by x, layout 2 by y, both at rate 1) with
/// the source buffers bound at `strides`, on every device, and compare every thread's report
/// with the bytes at `s1 · x` of layout 1 and `s2 · y` of layout 2.
fn run_strided(bc: &[u8], d: &StageInputDescriptor, strides: &[(u32, u32)], s1: u32, s2: u32) {
    let mut ran = 0;
    for (what, dev, _) in devices() {
        let Some(pipe) = compiled(&what, pipeline(&dev, bc, "stage_in_fetch", Some(d))) else {
            continue;
        };
        ran += 1;
        let (_, m) = typed_module_with(bc, "stage_in_fetch", dev.profile(), &inputs(d))
            .unwrap_or_else(|e| panic!("{what}: {e}"));
        let mut world = World::new(fetch_buffers(), StageInRegion::new([0; 3], [W, H, 1]))
            .with_strides(strides.iter().copied());
        execute(&dev, &pipe, &m, &mut world, &threads(W, H))
            .unwrap_or_else(|e| panic!("{what} {strides:?}: {e}"));
        let out = words(&world.buffers[&0].bytes);
        for gy in 0..H {
            for gx in 0..W {
                let t = gy * W + gx;
                let at = (12 * t) as usize;
                let (a1, a2) = ((s1 * gx) as usize, (s2 * gy) as usize);
                assert_eq!(
                    out[at..at + 12],
                    expected_at(a1, a2, t),
                    "{what} {strides:?}: thread ({gx}, {gy})"
                );
            }
        }
    }
    assert!(ran >= PROFILES.len(), "ran on {ran} devices");
}

/// A layout of a fixed stride fetches at it whatever stride the encoder binds its buffer with, as
/// an M4 reads one (a fixed 16 read at 16 under a bound 8): such a kernel
/// records no `RuntimeFact::StageInStride` and pushes no stride.
#[test]
fn a_fixed_layout_reads_its_own_stride_whatever_the_bind() {
    let Some(bc) = fixture() else { return };
    let d = fetch_descriptor(
        (StepFunction::ThreadPositionInGridX, 1),
        (StepFunction::ThreadPositionInGridY, 1),
        IndexType::UInt16,
    );
    let dev = device("moltenvk-m3", false);
    let pipe = pipeline(&dev, &bc, "stage_in_fetch", Some(&d)).expect("pipeline");
    assert!(
        !pipe
            .dependencies
            .iter()
            .any(|f| matches!(f.fact, RuntimeFact::StageInStride { .. })),
        "{:?}",
        pipe.dependencies
    );
    run_strided(&bc, &d, &[(1, 32), (2, 4), (3, 12)], 24, 8);
}

/// A layout of `MTLBufferLayoutStrideDynamic` fetches at the stride the encoder binds its buffer
/// with, as an M4 reads it (element `e` at `e · stride` from the bind offset
/// for strides 0, 2, 4, 6, 8, 12, 16 and 256, unaligned ones byte-exact); a buffer bound without
/// one reads at stride 0, as Metal does without its validation layer (which reads every element
/// past the first as zeros instead). The kernel records `RuntimeFact::StageInStride` for each
/// dynamic layout it reads, and a constant step's layout reads no stride.
#[test]
fn a_dynamic_layout_reads_the_bound_stride() {
    let Some(bc) = fixture() else { return };
    let mut d = fetch_descriptor(
        (StepFunction::ThreadPositionInGridX, 1),
        (StepFunction::ThreadPositionInGridY, 1),
        IndexType::UInt16,
    );
    for layout in [1, 2, 3] {
        d.layouts.get_mut(&layout).expect("layout").stride = None;
    }
    let dev = device("moltenvk-m3", false);
    let pipe = pipeline(&dev, &bc, "stage_in_fetch", Some(&d)).expect("pipeline");
    let strides: Vec<u32> = pipe
        .dependencies
        .iter()
        .filter_map(|f| match f.fact {
            RuntimeFact::StageInStride { layout } => Some(layout),
            _ => None,
        })
        .collect();
    assert_eq!(strides, [1, 2], "{:?}", pipe.dependencies);
    // The layouts' own strides, then others: none, unaligned, and a halfword one.
    for (s1, s2) in [(24, 8), (0, 16), (12, 2), (6, 0), (25, 3)] {
        run_strided(&bc, &d, &[(1, s1), (2, s2), (3, 7)], s1, s2);
    }
    // Bound without a stride: 0.
    run_strided(&bc, &d, &[(1, 24)], 24, 0);
    run_strided(&bc, &d, &[], 0, 0);
}

/// A source buffer's bind offset moves the fetch with it (the range and the residual), and without
/// buffer robustness one bound too short reads zero bytes past its end through the in-shader check
/// (the interpreter traps any access a robust device would contain, so it does not model the robust
/// side).
#[test]
fn bind_offsets_and_short_buffers_reach_the_right_bytes() {
    let Some(bc) = fixture() else { return };
    let d = fetch_descriptor(
        (StepFunction::ThreadPositionInGridX, 1),
        (StepFunction::ThreadPositionInGridY, 1),
        IndexType::UInt16,
    );
    for (what, dev, bare) in devices() {
        let pipe = pipeline(&dev, &bc, "stage_in_fetch", Some(&d)).expect("pipeline");
        let (_, m) =
            typed_module_with(&bc, "stage_in_fetch", dev.profile(), &inputs(&d)).expect("m");
        // Layout 1 bound 72 bytes (three elements) into its allocation; without robustness,
        // layout 2 holds two elements, so rows 2 and above read zeros.
        let mut shifted = vec![0x55; 72];
        shifted.extend(layout1());
        let short: Vec<u8> = if bare {
            layout2()[..16].to_vec()
        } else {
            layout2()
        };
        let mut world = World::new(
            [
                (
                    0,
                    Alloc {
                        bytes: vec![0; (48 * W * H) as usize],
                        offset: 0,
                    },
                ),
                (
                    1,
                    Alloc {
                        bytes: shifted,
                        offset: 72,
                    },
                ),
                (
                    2,
                    Alloc {
                        bytes: short,
                        offset: 0,
                    },
                ),
                (
                    3,
                    Alloc {
                        bytes: layout3(),
                        offset: 0,
                    },
                ),
            ],
            StageInRegion::new([0, 0, 0], [W, H, 1]),
        );
        execute(&dev, &pipe, &m, &mut world, &threads(W, H))
            .unwrap_or_else(|e| panic!("{what}: {e}"));
        let out = words(&world.buffers[&0].bytes);
        for gy in 0..H {
            for gx in 0..W {
                let t = gy * W + gx;
                let at = (12 * t) as usize;
                let mut want = expected_fetch(gx, gy, t);
                if bare && gy >= 2 {
                    want[4..8].copy_from_slice(&[0; 4]);
                }
                assert_eq!(out[at..at + 12], want, "{what}: thread ({gx}, {gy})");
            }
        }
    }
}

/// `[[grid_origin]]` and `[[grid_size]]` read the stage-in region the binder writes, as `uint3` and
/// as `ushort2`, beside a fetch at the origin.
#[test]
fn the_region_builtins_read_the_bound_region() {
    let Some(bc) = fixture() else { return };
    let d = descriptor(
        &[(0, AttributeFormat::Float, 0, 1)],
        &[(1, 4, StepFunction::ThreadPositionInGridX, 1)],
        (IndexType::UInt16, 0),
    );
    let samples: Vec<u8> = (0..ELEMENTS)
        .flat_map(|e| (e as f32 + 0.5).to_le_bytes())
        .collect();
    let (origin, size) = ([3, 0, 7], [10, 20, 30]);
    for (what, dev, _) in devices() {
        let pipe = pipeline(&dev, &bc, "stage_in_region", Some(&d)).expect("pipeline");
        let (_, m) =
            typed_module_with(&bc, "stage_in_region", dev.profile(), &inputs(&d)).expect("m");
        let mut world = World::new(
            [
                (
                    0,
                    Alloc {
                        bytes: vec![0; 32 * 8],
                        offset: 0,
                    },
                ),
                (
                    1,
                    Alloc {
                        bytes: samples.clone(),
                        offset: 0,
                    },
                ),
            ],
            StageInRegion::new(origin, size),
        );
        execute(&dev, &pipe, &m, &mut world, &threads(8, 1)).expect("dispatch");
        let out = words(&world.buffers[&0].bytes);
        for t in 0..8u32 {
            let at = (8 * t) as usize;
            let v = ((origin[0] + t) as f32 + 0.5).to_bits();
            assert_eq!(
                out[at..at + 8],
                [v, 3, 0, 7, 10, 20, 30, t],
                "{what}: thread {t}"
            );
        }

        // Without a stage input, as 16-bit pairs: no descriptor needed.
        let pipe = pipeline(&dev, &bc, "stage_in_region_short", None).expect("pipeline");
        let (_, m) = typed_module_with(
            &bc,
            "stage_in_region_short",
            dev.profile(),
            &LoweringInputs::new(),
        )
        .expect("m");
        let mut world = World::new(
            [(
                0,
                Alloc {
                    bytes: vec![0; 8 * 4],
                    offset: 0,
                },
            )],
            StageInRegion::new([1, 2, 3], [300, 40000, 5]),
        );
        execute(&dev, &pipe, &m, &mut world, &threads(4, 1)).expect("dispatch");
        let out = &world.buffers[&0].bytes;
        for t in 0..4usize {
            let h: Vec<u16> = out[8 * t..8 * t + 8]
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| u16::from_le_bytes([c[0], c[1]]))
                .collect();
            assert_eq!(h, [1, 2, 300, 40000], "{what}: thread {t}");
        }
    }
}

fn not_implemented(r: Result<ComputePipeline, Error>, feature: Feature) {
    match r {
        Err(Error::Unsupported(u)) => {
            assert_eq!(u.feature, feature);
            assert_eq!(u.needs, Needs::NotImplemented);
        }
        other => panic!("{feature:?}: {other:?}"),
    }
}

/// A kernel reading a stage input needs the descriptor; one that lacks the attribute or its layout,
/// or steps every 0th thread, is invalid, and so are a render-stage step and a constant step at a
/// rate other than 0; an indexed step at a rate other than 1, and a format of the other numeric
/// class, are not implemented.
#[test]
fn descriptor_refusals_are_typed() {
    let Some(bc) = fixture() else { return };
    let dev = device("moltenvk-m3", false);
    let fetch = |d: Option<&StageInputDescriptor>| pipeline(&dev, &bc, "stage_in_fetch", d);
    let grid = |x: (StepFunction, u32)| {
        fetch_descriptor(
            x,
            (StepFunction::ThreadPositionInGridY, 1),
            IndexType::UInt16,
        )
    };
    let valid = grid((StepFunction::ThreadPositionInGridX, 1));
    assert!(fetch(Some(&valid)).is_ok());

    assert!(matches!(
        fetch(None),
        Err(Error::NeedsInput(RuntimeInput::StageInputDescriptor {
            attribute: 0
        }))
    ));
    let mut d = valid.clone();
    d.attributes.remove(&3);
    assert!(matches!(
        fetch(Some(&d)),
        Err(Error::InvalidDescriptor(
            DescriptorError::StageInputAttribute { attribute: 3 }
        ))
    ));
    let mut d = valid.clone();
    d.layouts.remove(&2);
    assert!(matches!(
        fetch(Some(&d)),
        Err(Error::InvalidDescriptor(
            DescriptorError::StageInputLayout { layout: 2 }
        ))
    ));
    for step in [
        StepFunction::ThreadPositionInGridX,
        StepFunction::ThreadPositionInGridXIndexed,
    ] {
        assert!(matches!(
            fetch(Some(&grid((step, 0)))),
            Err(Error::InvalidDescriptor(
                DescriptorError::StageInputStepRate {
                    layout: 1,
                    step_rate: 0
                }
            ))
        ));
    }
    // A constant step at a step rate other than 0 stops Metal's pipeline creation at an
    // assertion, with and without API validation.
    let mut d = valid.clone();
    d.layouts.get_mut(&3).expect("layout 3").step_rate = 1;
    assert!(matches!(
        fetch(Some(&d)),
        Err(Error::InvalidDescriptor(
            DescriptorError::StageInputStepRate {
                layout: 3,
                step_rate: 1
            }
        ))
    ));

    not_implemented(
        fetch(Some(&grid((StepFunction::ThreadPositionInGridXIndexed, 2)))),
        Feature::KernelStageIn,
    );
    // A vertex stage's step function stops Metal's compute pipeline creation at an assertion, the
    // default per-vertex step of a fresh layout among them.
    for step in [
        StepFunction::PerVertex,
        StepFunction::PerInstance,
        StepFunction::PerPatch,
        StepFunction::PerPatchControlPoint,
    ] {
        assert!(
            matches!(
                fetch(Some(&grid((step, 1)))),
                Err(Error::InvalidDescriptor(
                    DescriptorError::StageInputStepFunction {
                        layout: 1,
                        step_function
                    }
                )) if step_function == step
            ),
            "{step:?}"
        );
    }
    let mut d = valid.clone();
    d.layouts
        .insert(1, StageInputLayoutDescriptor::new(Some(24)));
    assert!(matches!(
        fetch(Some(&d)),
        Err(Error::InvalidDescriptor(
            DescriptorError::StageInputStepFunction {
                layout: 1,
                step_function: StepFunction::PerVertex
            }
        ))
    ));
    // `u` is a `uint4`: a float format is a conversion whose Metal meaning is not established.
    let mut d = valid.clone();
    d.attributes.insert(
        1,
        StageInputAttributeDescriptor::new(AttributeFormat::Float4, 0, 2),
    );
    not_implemented(fetch(Some(&d)), Feature::Lowering(LowerFamily::StageIo));
    let mut d = valid.clone();
    d.attributes.insert(
        0,
        StageInputAttributeDescriptor::new(AttributeFormat::Invalid, 4, 1),
    );
    assert!(matches!(
        fetch(Some(&d)),
        Err(Error::InvalidDescriptor(
            DescriptorError::PlaceholderEnumValue { .. }
        ))
    ));
}

/// The kernel's key takes the descriptor canonically (a constant layout's stride and rate, and an
/// unread layout or index buffer, do not change it), the pipeline key is the one
/// `compute_pipeline_key` computes, and a `Compiled` kernel must fetch as the pipeline's
/// descriptor does.
#[test]
fn compiled_kernels_and_keys_follow_the_descriptor() {
    let Some(bc) = fixture() else { return };
    let dev = device("moltenvk-m3", false);
    let f = function(&bc, "stage_in_fetch");
    let a = fetch_descriptor(
        (StepFunction::ThreadPositionInGridX, 1),
        (StepFunction::ThreadPositionInGridY, 1),
        IndexType::UInt16,
    );
    let key = |d: &StageInputDescriptor| dev.function_compile_key(&f, &inputs(d));

    // Canonical: the constant layout's stride and rate, an extra layout, and the index buffer of
    // a descriptor without an indexed step (the key, which takes no check of the layouts).
    let mut same = a.clone();
    let l3 = same.layouts.get_mut(&3).expect("layout 3");
    (l3.stride, l3.step_rate) = (None, 1);
    same.layouts
        .insert(9, StageInputLayoutDescriptor::new(Some(64)));
    (same.index_type, same.index_buffer_index) = (IndexType::UInt32, 6);
    assert_eq!(key(&same), key(&a));
    let mut other = a.clone();
    other.layouts.get_mut(&2).expect("layout 2").stride = Some(12);
    assert_ne!(key(&other), key(&a));
    let mut dynamic = a.clone();
    dynamic.layouts.get_mut(&2).expect("layout 2").stride = None;
    assert_ne!(key(&dynamic), key(&a));
    assert_ne!(
        key(&a),
        dev.function_compile_key(&f, &LoweringInputs::new())
    );

    let pipe = dev
        .compile_compute_pipeline(&compute(f.clone(), Some(&a)), &Budget::standard())
        .expect("pipeline");
    assert_eq!(
        pipe.key,
        dev.compute_pipeline_key(&compute(f.clone(), Some(&a)))
    );

    let c = dev
        .compile_function(&f, &inputs(&a), &Budget::standard())
        .expect("compile");
    assert_eq!(*c.key(), key(&a));
    let compiled = |d: Option<&StageInputDescriptor>| {
        let mut p = ComputePipelineDescriptor::new(StageSource::Compiled(c.clone()));
        p.stage_input = d.cloned();
        dev.compile_compute_pipeline(&p, &Budget::standard())
    };
    let from_compiled = compiled(Some(&a)).expect("the descriptor it was compiled under");
    assert_eq!(from_compiled.stage.spirv, pipe.stage.spirv);
    // Metal refuses the extra layout of a nonzero stride that no attribute names, and takes it at
    // stride 0.
    assert!(matches!(
        compiled(Some(&same)),
        Err(Error::InvalidDescriptor(
            DescriptorError::StageInputLayout { layout: 9 }
        ))
    ));
    // At stride 0 the extra layout is Metal's unset one; the constant layout's rate goes back to
    // 0, as Metal requires of a pipeline.
    same.layouts
        .insert(9, StageInputLayoutDescriptor::new(Some(0)));
    same.layouts.get_mut(&3).expect("layout 3").step_rate = 0;
    assert!(compiled(Some(&same)).is_ok());
    assert!(matches!(
        compiled(Some(&other)),
        Err(Error::InvalidDescriptor(
            DescriptorError::StageInputAttribute { attribute: 1 }
        ))
    ));
    assert!(matches!(
        compiled(None),
        Err(Error::NeedsInput(RuntimeInput::StageInputDescriptor {
            attribute: 0
        }))
    ));
    // A kernel compiled on its own without the descriptor needs it.
    assert!(matches!(
        dev.compile_function(&f, &LoweringInputs::new(), &Budget::standard()),
        Err(Error::NeedsInput(RuntimeInput::StageInputDescriptor {
            attribute: 0
        }))
    ));
}

/// The descriptors the validation and estimate checks compile under: every step.
fn every_step() -> Vec<StageInputDescriptor> {
    use StepFunction as S;
    vec![
        fetch_descriptor(
            (S::ThreadPositionInGridX, 1),
            (S::ThreadPositionInGridY, 1),
            IndexType::UInt16,
        ),
        fetch_descriptor(
            (S::ThreadPositionInGridX, 3),
            (S::ThreadPositionInGridY, 2),
            IndexType::UInt16,
        ),
        fetch_descriptor(
            (S::ThreadPositionInGridXIndexed, 1),
            (S::ThreadPositionInGridYIndexed, 1),
            IndexType::UInt32,
        ),
        fetch_descriptor((S::Constant, 1), (S::Constant, 1), IndexType::UInt16),
    ]
}

/// L3 on every committed profile: each step's module links and validates wherever it compiles,
/// and the plan's size estimate bounds its words.
#[test]
fn every_committed_profile_validates_the_fetch() {
    let Some(bc) = fixture() else { return };
    let region = descriptor(
        &[(0, AttributeFormat::Float, 0, 1)],
        &[(1, 4, StepFunction::ThreadPositionInGridX, 1)],
        (IndexType::UInt16, 0),
    );
    let mut cases: Vec<(&str, StageInputDescriptor)> = every_step()
        .into_iter()
        .map(|d| ("stage_in_fetch", d))
        .collect();
    cases.push(("stage_in_region", region.clone()));
    cases.push(("stage_in_region_short", region));
    let mut compiled = 0;
    for (stem, p) in committed_profiles().expect("profiles") {
        // The `BufferRangeLimit` variant reaches the source and index buffers by device address
        // where the profile has them.
        let addressed = p.fields().buffer_access == BufferAccess::Addressed;
        let variants: &[bool] = if addressed { &[false, true] } else { &[false] };
        for ((entry, d), &psb) in cases
            .iter()
            .flat_map(|c| variants.iter().map(move |v| (c, v)))
        {
            let mut i = inputs(d);
            if psb {
                i.psb_slots = [1, 2, 4].into_iter().map(kslot).collect();
            }
            let consts = FunctionConstantValues::defaults();
            let a = match compile_bitcode(
                &bc,
                EntryRef::Name(entry),
                &consts,
                &p,
                &i,
                &mut meter(),
                SPIRV_WORDS,
            ) {
                Ok(a) => a,
                Err(Error::Unsupported(u)) if u.needs != Needs::NotImplemented => continue,
                Err(e) => panic!("{stem} {entry} psb={psb} {d:?}: {e}"),
            };
            link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{stem} {entry}: {e}"));
            let plan = plan_bitcode(&bc, EntryRef::Name(entry), &consts, &p, &i, &mut meter())
                .expect("plan");
            let words = a.module.words.len() as u64;
            assert!(
                plan.estimate.words >= words,
                "{stem} {entry} {d:?}: estimate {} < emitted {words}",
                plan.estimate.words
            );
            // The artifact lists every fetched attribute with its descriptor entries.
            let listed = a
                .interface
                .io
                .stage_in
                .iter()
                .map(|s| s.index)
                .collect::<Vec<_>>();
            let want: Vec<u32> = if *entry == "stage_in_fetch" {
                vec![0, 1, 2, 3]
            } else if *entry == "stage_in_region" {
                vec![0]
            } else {
                Vec::new()
            };
            assert_eq!(listed, want, "{stem} {entry}");
            compiled += 1;
        }
    }
    assert!(compiled >= cases.len(), "compiled {compiled}");
}

/// A kernel the stage-in region reaches is never a device-replay variant, so the replay, which
/// ignores a command's region, never runs one with its region dropped. Compiled as a replay variant
/// (`LoweringInputs::replay_bindings`) under a stage-input descriptor, each fixture kernel (the
/// fetch, and the `[[grid_origin]]` and `[[grid_size]]` readers) is refused as
/// `GpuEncodedIndirectCommands` on every committed profile with device addresses; a reader of the
/// region without a descriptor that compiles plans `PushField::StageInRegion`, which the binder
/// refuses to replay (`check_replay`: `ReplayRefusal::NotReplayVariant`).
#[test]
fn a_kernel_the_stage_in_region_reaches_is_no_replay_variant() {
    use crate::pipeline::sym::PushField;
    let Some(bc) = fixture() else { return };
    let region = descriptor(
        &[(0, AttributeFormat::Float, 0, 1)],
        &[(1, 4, StepFunction::ThreadPositionInGridX, 1)],
        (IndexType::UInt16, 0),
    );
    let fetch = fetch_descriptor(
        (StepFunction::ThreadPositionInGridX, 1),
        (StepFunction::ThreadPositionInGridY, 1),
        IndexType::UInt16,
    );
    let consts = FunctionConstantValues::defaults();
    let mut checked = 0;
    for (stem, p) in committed_profiles().expect("profiles") {
        if p.fields().buffer_access != BufferAccess::Addressed {
            continue;
        }
        for (entry, d) in [
            ("stage_in_fetch", &fetch),
            ("stage_in_region", &region),
            ("stage_in_region_short", &region),
        ] {
            let mut i = inputs(d);
            i.replay_bindings = true;
            let r = compile_bitcode(
                &bc,
                EntryRef::Name(entry),
                &consts,
                &p,
                &i,
                &mut meter(),
                SPIRV_WORDS,
            );
            match r {
                Err(Error::Unsupported(u)) => {
                    assert_eq!(
                        u.feature,
                        Feature::GpuEncodedIndirectCommands,
                        "{stem} {entry}"
                    )
                }
                other => panic!("{stem} {entry}: {:?}", other.map(|_| ())),
            }
            checked += 1;
        }
        let mut i = LoweringInputs::new();
        i.replay_bindings = true;
        if let Ok(plan) = plan_bitcode(
            &bc,
            EntryRef::Name("stage_in_region_short"),
            &consts,
            &p,
            &i,
            &mut meter(),
        ) {
            assert!(
                plan.push
                    .fields
                    .iter()
                    .any(|f| f.field == PushField::StageInRegion),
                "{stem}"
            );
        }
    }
    assert!(checked > 0, "no committed profile with device addresses");
}
