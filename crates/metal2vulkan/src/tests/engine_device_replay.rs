//! The device replay of GPU-encoded compute commands end to end on the CPU interpreter. No device
//! is opened.
//!
//! K14b's template (`icb_replay_template`: each thread stages `params[0] * (tid + 1)` in
//! threadgroup memory, waits for its group, and writes its staged word plus `params[1]` to
//! `out[tid]`) compiles as a replay variant (`LoweringInputs::replay_bindings`) through the public
//! API on every committed profile with device addresses, plans with the resident set and its
//! threadgroup length and no kernel buffer, and instantiates with
//! `BindingPlan::instantiate_replay`. The preprocess runs over host-encoded records (a dispatch of
//! threadgroups, an empty command, a dispatch of threads, each dispatch binding its buffers at
//! interior offsets), and each command's dispatch runs the variant with its arguments and its
//! slot table, as the consumer records them. The output is what Metal computes for each command,
//! and the status is clean. On the profiles without device addresses the variant's compile names
//! what they lack; a plan of the template compiled without the input, or a bind state that binds
//! a kernel buffer, is refused.
//!
//! `icb_replay_grid_template` observes every grid builtin of its region (its threadgroup's size and
//! position, the nominal size, the grid in threadgroups and threads) and its neighbour in
//! threadgroup memory. Ragged thread grids, and group counts past a device's limit (a description
//! limited to 3 x 2 x 64), replay on the device exactly as the host replay of the same records runs
//! them (the template compiled plainly, each command's own dispatch over the regions) and as a
//! model of the source computes them; a command with more regions than the execution records is the
//! typed `REGIONS` fault, beside a command that still runs.
//!
//! A kernel that reads only `thread_position_in_grid` and `threads_per_grid` (`flat`) compiles as a
//! flat variant (`LoweringInputs::flat_grid`), which every command of its pipeline selects: thread
//! and threadgroup grids, ragged or not, of any threadgroup size and lengths, run each grid thread
//! once with Metal's position and grid, in threadgroups of the variant's size spread over the
//! device's group-count limits; a threadgroup Metal runs nothing for runs nothing and reports
//! nothing, and a grid no group count holds is the typed `GROUP_COUNT` fault. A kernel that
//! observes its partition is refused under the input. Outside a replay (`VariantKey::flat_grid`),
//! the flat variant binds as any kernel does and runs each command as one indirect dispatch at its
//! flat group counts, reading its grid from the region record at its `ReplayRegionAddress`, as
//! Metal computes it.
//!
//! The grid template, which observes its partition, compiles as a folded variant
//! (`VariantKey::folded_groups`) and runs each threadgroup command as one indirect dispatch of
//! its threadgroup size at its group counts folded within the device's limits (its own, and
//! 3 x 2 x 64), each threadgroup at Metal's position, as Metal computes it.
//!
//! `llvm-as` assembles the fixture; without it the tests skip visibly.

use std::collections::BTreeMap;
use std::num::NonZeroU64;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::bind::replay::{
    replay_preprocess_generated, typed_module as preprocess_module, GeneratedLayout,
    GeneratedRefusal, GeneratedVariant, MAX_REPLAY_REGIONS, MAX_SEGMENT_COMMANDS,
};
use crate::bind::status::ReplayFaults;
use crate::bind::{
    replay_preprocess, BindState, BoundBuffer, Dispatch, ExecutionStatus, IndirectArgs,
    IndirectRef, PushHole, ReplayExecution, ReplayRefusal, ReplayVariant, ResourceClass, RootTable,
    StageTable, StatusReport, UsedResource, WriteSource,
};
use crate::budget::Budget;
use crate::cache::VariantKey;
use crate::device::profile::BufferAccess;
use crate::device::requirements::VkFeature;
use crate::device::{Device, DeviceDescription, DevicePolicy, GeneratedCommandsProperties};
use crate::error::{BindError, Error, Feature, Needs};
use crate::indirect_commands::{
    layout, Action, BufferBinding, BufferStage, CommandRecord, BINDING_COUNT,
    COMMAND_BUFFER_HEADER_BYTES, COMMAND_BYTES,
};
use crate::library::Library;
use crate::pipeline::facts::RuntimeFact;
use crate::pipeline::layout::BindingSource;
use crate::pipeline::sym::{PushField, ResourceKind, ResourceSlot, SpecKey, Stage};
use crate::pipeline::testing::{engine_dir, llvm_assemble, typed_module_with, AssembleError};
use crate::pipeline::{
    ComputePipeline, ComputePipelineDescriptor, FunctionDescriptor, LoweringInputs, StageSource,
};
use crate::spirv::interp::{run, Buffer, Config, Order, Resources};
use crate::spirv::ir::TypedModule;
use crate::spirv::testing::refuses_threadgroup_memory;

const ENTRY: &str = "icb_replay_template";

const HEADER: &str = "target datalayout = \"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32\"\ntarget triple = \"air64_v28-apple-macosx26.0.0\"\n\n";

/// Device addresses of the consumer's objects.
const SCRATCH: u64 = 0x0000_0030_0000_0000;
const RECORDS: u64 = 0x0000_0031_0000_0000;
const STATUS: u64 = 0x0000_0012_0000_0000;
const SINK: u64 = 0x0000_0011_0000_0000;

/// The two resident allocations `(guest, host)`, `BYTES` each: the parameters and the output.
const PARAMS: (u64, u64) = (0x0000_0001_0000_0000, 0x0000_0020_0000_0000);
const OUT: (u64, u64) = (0x0000_0001_0001_0000, 0x0000_0021_0000_0000);
const BYTES: usize = 512;

const TOKEN: u64 = 0x0123_4567_89ab_cdef;
const LOCAL: [u32; 3] = [8, 1, 1];
/// The template's threadgroup length: one word per thread of a group.
const LENGTH: u32 = 32;

fn fixture() -> Option<Vec<u8>> {
    match llvm_assemble(&engine_dir().join(format!("{ENTRY}.ll"))) {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected {ENTRY}.ll: {why}"),
    }
}

/// `text` (an AIR module body) assembled, with the AIR target header.
fn assemble(name: &str, text: &str) -> Option<Vec<u8>> {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let dir = std::env::temp_dir().join(format!(
        "m2v-device-replay-{}-{}-{name}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&dir).expect("scratch");
    let ll = dir.join("k.ll");
    std::fs::write(&ll, format!("{HEADER}{text}")).expect("write");
    let r = llvm_assemble(&ll);
    let _ = std::fs::remove_dir_all(&dir);
    match r {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected {name}: {why}\n{text}"),
    }
}

/// Every committed profile as a device, by stem.
fn devices() -> Vec<(String, Device)> {
    devices_with(|_| {})
}

/// Every committed profile as a device, and lavapipe without `bufferDeviceAddress`: the committed
/// profiles without device addresses take no threadgroup memory, which the replayed kernels use.
fn devices_and_unaddressed() -> Vec<(String, Device)> {
    let mut out = devices();
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles/lavapipe.json");
    let text = std::fs::read_to_string(path).expect("profile");
    let mut desc = DeviceDescription::from_khronos_json(&text, None).expect("description");
    desc.features.remove(&VkFeature::BufferDeviceAddress);
    let dev = Device::from_queries(desc, DevicePolicy::default()).expect("device");
    assert_ne!(
        dev.profile().fields().buffer_access,
        BufferAccess::Addressed
    );
    out.push(("lavapipe without device addresses".into(), dev));
    out
}

/// Every committed profile as a device, by stem, with its description changed by `tweak`.
fn devices_with(tweak: impl Fn(&mut DeviceDescription)) -> Vec<(String, Device)> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../metal2vulkan-validation/profiles");
    let mut out: Vec<(String, Device)> = std::fs::read_dir(&dir)
        .expect("profiles")
        .map(|e| e.expect("entry").path())
        .filter(|p| p.extension().is_some_and(|x| x == "json"))
        .map(|path| {
            let stem = path
                .file_stem()
                .expect("stem")
                .to_string_lossy()
                .into_owned();
            let text = std::fs::read_to_string(&path).expect("profile");
            let mut desc = DeviceDescription::from_khronos_json(&text, None).expect("description");
            tweak(&mut desc);
            (
                stem,
                Device::from_queries(desc, DevicePolicy::default()).expect("device"),
            )
        })
        .collect();
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

fn replay_inputs() -> LoweringInputs {
    let mut inputs = LoweringInputs::new();
    inputs.replay_bindings = true;
    inputs
}

/// Function `entry` of `bc` compiled under `inputs`, through the public API.
fn pipeline(
    dev: &Device,
    bc: &[u8],
    entry: &str,
    inputs: &LoweringInputs,
) -> Result<ComputePipeline, Error> {
    let lib = Library::from_air(bc).expect("library");
    let f = FunctionDescriptor::new(lib.function(entry).expect("function"));
    let compiled = dev.compile_function(&f, inputs, &Budget::standard())?;
    let d = ComputePipelineDescriptor::new(StageSource::Compiled(compiled));
    dev.compile_compute_pipeline(&d, &Budget::standard())
}

/// What every replayed command shares: threadgroup length 0 `length` (0: none) and the two
/// resident allocations; no kernel buffer.
fn shared_state(length: u32) -> BindState {
    let mut kernel = StageTable::default();
    if length != 0 {
        kernel.threadgroup_lengths.insert(0, length);
    }
    let mut s = BindState::default();
    s.stages.insert(Stage::Kernel, kernel);
    for (i, (guest, host)) in [PARAMS, OUT].into_iter().enumerate() {
        let mut u = UsedResource::new(ResourceClass::Buffer, i as u32);
        u.gpu_address = Some(guest);
        u.device_address = Some(host);
        u.length = BYTES as u64;
        s.used_resources.push(u);
    }
    s
}

/// A record of `action` with the template's pipeline and threadgroup length, kernel buffer 0 at
/// `params` and kernel buffer 1 at `out` (guest addresses).
fn record(action: Action, params: u64, out: u64) -> CommandRecord {
    let mut r = CommandRecord {
        action,
        ..CommandRecord::default()
    };
    r.bindings.pipeline = NonZeroU64::new(TOKEN);
    let kernel = r.bindings.buffers_mut(BufferStage::Kernel);
    for (slot, address) in [(0, params), (1, out)] {
        kernel[slot] = Some(BufferBinding {
            address,
            attribute_stride: None,
        });
    }
    r.threadgroup_memory_lengths[0] = LENGTH;
    r
}

fn words(bytes: &[u8]) -> Vec<u32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes(*c))
        .collect()
}

/// The consumer's memory: the scratch, the records, the two allocations, and the status.
struct Memory {
    scratch: Vec<u8>,
    records: Vec<u8>,
    params: Vec<u8>,
    out: Vec<u8>,
    status: [u8; ExecutionStatus::BYTES],
}

impl Memory {
    /// Every buffer the interpreter reaches by address, in a fixed order, with `extra` first.
    fn buffers(&self, extra: Vec<Buffer>) -> Vec<Buffer> {
        let at = |address, bytes: &[u8]| Buffer {
            slot: None,
            address: Some(address),
            bytes: bytes.to_vec(),
        };
        let mut out = extra;
        out.extend([
            at(SCRATCH, &self.scratch),
            at(RECORDS, &self.records),
            at(PARAMS.1, &self.params),
            at(OUT.1, &self.out),
            at(STATUS, &self.status),
            at(SINK, &[0; 64]),
        ]);
        out
    }

    /// Take back what a run left in the buffers of [`Memory::buffers`] after `extra` of them.
    fn take(&mut self, mut out: Vec<Vec<u8>>, extra: usize) {
        let mut rest = out.split_off(extra).into_iter();
        let mut next = || rest.next().expect("a buffer");
        self.scratch = next();
        assert_eq!(next(), self.records, "the records are only read");
        assert_eq!(next(), self.params, "the parameters are only read");
        self.out = next();
        self.status = next().try_into().expect("status");
    }

    fn report(&self) -> StatusReport {
        ExecutionStatus::decode(&self.status)
    }
}

/// Run `m` with `res`'s push and spec over `workgroups` in both scheduling orders; the buffers.
fn run_both(m: &TypedModule, workgroups: [u32; 3], res: Resources) -> Vec<Vec<u8>> {
    let mut outs = Vec::new();
    for order in [Order::Forward, Order::Reverse] {
        let cfg = Config {
            workgroups,
            order,
            ..Config::default()
        };
        outs.push(run(m, &cfg, res.clone()).expect("run").buffers);
    }
    assert_eq!(outs[0], outs[1], "the two scheduling orders agree");
    outs.swap_remove(0)
}

/// The typed-module symbol of a layout binding.
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

/// One device replay of `commands` (a header, then their records) over the variant of `entry`
/// (in `bc`) for `LOCAL` and threadgroup length 0 `length` (0: none), with the parameters
/// allocation holding `params`, on `dev`: the preprocess, then per command, region, and variant
/// the variant's dispatch over its arguments with the command's slot table and the region's
/// record, as the consumer records them. The memory it leaves, and the dispatches it made.
fn replay_run(
    stem: &str,
    dev: &Device,
    bc: &[u8],
    entry: &str,
    length: u32,
    commands: &[CommandRecord],
    params: Vec<u8>,
) -> (Memory, usize) {
    let count = commands.len() as u32;
    replay_variants_run(
        stem,
        (dev, bc, entry),
        (length, &[LOCAL], None),
        commands,
        (ReplayExecution::new(count, 0, count), Route::Arguments),
        params,
    )
}

/// Which of the device replay's paths a run takes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Route {
    /// K14c's: per command, region, and variant an indirect dispatch from the arguments.
    Arguments,
    /// The generated-commands path: per possible segment an execution of the sequences its count
    /// states, each binding its variant, pushing its bytes, and dispatching its count.
    Generated,
}

/// Whether `dev` runs the generated-commands path: its description states
/// `VK_EXT_device_generated_commands` with compute pipeline binding.
fn generates(dev: &Device) -> bool {
    dev.generated_commands().is_some_and(|g| {
        g.pipeline_binding_stages
            .contains(crate::vk::ShaderStageFlags::COMPUTE)
    })
}

/// [`replay_run`] over a variant of `entry` for each threadgroup size of `locals`, in that order,
/// all with threadgroup length 0 `length`, replaying `execution`; with `flat`, flat variants
/// for a state of that many threads per threadgroup at most.
fn replay_variants_run(
    stem: &str,
    (dev, bc, entry): (&Device, &[u8], &str),
    (length, locals, flat): (u32, &[[u32; 3]], Option<u32>),
    commands: &[CommandRecord],
    (execution, path): (ReplayExecution, Route),
    params: Vec<u8>,
) -> (Memory, usize) {
    let mut inputs = replay_inputs();
    inputs.flat_grid = flat.is_some();
    let pipe = pipeline(dev, bc, entry, &inputs).unwrap_or_else(|e| panic!("{stem}: {e}"));
    let (_, m) = typed_module_with(bc, entry, dev.profile(), &inputs)
        .unwrap_or_else(|e| panic!("{stem}: {e}"));
    let state = shared_state(length);
    let plan = pipe
        .plan(dev, &state.shape())
        .unwrap_or_else(|e| panic!("{stem}: {e:?}"));
    assert!(
        plan.diagnostics.is_empty(),
        "{stem}: {:?}",
        plan.diagnostics
    );
    let mut lengths = [0; 31];
    lengths[0] = length;
    let mut variants = Vec::new();
    let mut instantiated = Vec::new();
    for &local in locals {
        let cmds = plan
            .instantiate_replay(&state, local)
            .unwrap_or_else(|e| panic!("{stem}: {local:?}: {e:?}"));
        assert_eq!(cmds.regions.len(), 1, "{stem}");
        assert_eq!(
            cmds.regions[0].indirect,
            Some(IndirectArgs::Replay),
            "{stem}"
        );
        assert!(
            cmds.root_table.is_none() && cmds.indirect_clamp.is_none(),
            "{stem}"
        );
        let token = NonZeroU64::new(TOKEN).expect("token");
        variants.push(match flat {
            Some(max) => ReplayVariant::flat(token, local, max),
            None => ReplayVariant::new(token, local, lengths),
        });
        instantiated.push(cmds);
    }

    // The header as the runtime writes it (`CommandDescriptor::encode_header`): the capacity, and
    // the packed binding counts, which the replay reads the kernel bindings below
    // (`layout::COUNTS`), here every kernel binding.
    let mut records = vec![0u8; COMMAND_BUFFER_HEADER_BYTES];
    records[..4].copy_from_slice(&execution.capacity.to_le_bytes());
    let counts = layout::COUNTS as usize;
    records[counts..counts + 4].copy_from_slice(&(BINDING_COUNT as u32).to_le_bytes());
    for c in commands {
        let mut bytes = vec![0u8; COMMAND_BYTES];
        c.encode_into(&mut bytes).expect("record");
        records.extend_from_slice(&bytes);
    }
    let pre = match path {
        Route::Arguments => replay_preprocess(dev, &state, &execution, &variants),
        Route::Generated => {
            let generated: Vec<GeneratedVariant<'_>> = instantiated
                .iter()
                .map(|cmds| GeneratedVariant::new(&pipe, &plan, cmds))
                .collect();
            replay_preprocess_generated(dev, &state, &execution, &variants, &generated)
        }
    }
    .unwrap_or_else(|e| panic!("{stem}: {e:?}"));
    let layout = pre.layout;

    // The consumer's uploads.
    let mut scratch = vec![0u8; layout.bytes as usize];
    let mut put = |at: u64, bytes: &[u8]| {
        scratch[at as usize..at as usize + bytes.len()].copy_from_slice(bytes);
    };
    put(layout.params_offset(), &pre.params);
    let mut resident = pre.resident.bytes.clone();
    let sink = RootTable::SINK_HOST_OFFSET as usize;
    resident[sink..sink + 8].copy_from_slice(&SINK.to_le_bytes());
    put(layout.resident_offset, &resident);
    put(layout.extents_offset, &pre.extents);
    put(layout.variants_offset, &pre.variants);
    put(layout.parent_offset, &pre.parent);
    if let (Some(g), Some(replay)) = (layout.generated, &pre.generated) {
        put(g.templates_offset, &replay.templates);
    }
    let mut mem = Memory {
        scratch,
        records,
        params,
        out: vec![0; BYTES],
        status: ExecutionStatus::initial(),
    };

    // The preprocess.
    let mut push = pre.push_constants.clone();
    for hole in &pre.push_holes {
        let (offset, address) = match *hole {
            PushHole::RootTableAddress { offset, .. } => (offset, SCRATCH + layout.resident_offset),
            PushHole::ExecutionStatusAddress { offset, .. } => (offset, STATUS),
            PushHole::CommandRecordsAddress { offset, .. } => (offset, RECORDS),
            other => panic!("{other:?}"),
        };
        push[offset as usize..offset as usize + 8].copy_from_slice(&address.to_le_bytes());
    }
    let limits = dev.profile().fields().limits.max_compute_work_group_count;
    let prep = preprocess_module(limits).expect("preprocess module");
    let res = Resources {
        buffers: mem.buffers(Vec::new()),
        push,
        spec: BTreeMap::new(),
    };
    let out = run_both(&prep, pre.group_count, res);
    mem.take(out, 0);

    // The dispatches: per command, region, and variant, the variant over its arguments with the
    // command's slot table and the region's record.
    let layout_bindings: BTreeMap<(u32, u32), BindingSource> = pipe
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
    // What runs, in order: per dispatch its variant, push bytes, and group count.
    let mut jobs: Vec<(usize, Vec<u8>, [u32; 3])> = Vec::new();
    match (path, layout.generated) {
        (Route::Arguments, None) => {
            for s in 0..layout.slots {
                for r in 0..layout.regions {
                    for (v, cmds) in instantiated.iter().enumerate() {
                        let at = layout.argument_offset(s, r, v as u32) as usize;
                        let args: [u32; 3] = words(&mem.scratch[at..at + 12])
                            .try_into()
                            .expect("arguments");
                        let region = &cmds.regions[0];
                        let mut push = Vec::new();
                        if let Some(w) = region.push_constants.first() {
                            push = vec![0u8; (w.offset as usize) + w.bytes.len()];
                            push[w.offset as usize..].copy_from_slice(&w.bytes);
                        }
                        for hole in &cmds.push_holes {
                            let (offset, address) = match *hole {
                                PushHole::RootTableAddress { offset, .. } => {
                                    (offset, SCRATCH + layout.table_offset(s))
                                }
                                PushHole::ReplayRegionAddress { offset, .. } => {
                                    (offset, SCRATCH + layout.region_offset(s, r))
                                }
                                PushHole::ExecutionStatusAddress { offset, .. } => (offset, STATUS),
                                other => panic!("{other:?}"),
                            };
                            push[offset as usize..offset as usize + 8]
                                .copy_from_slice(&address.to_le_bytes());
                        }
                        jobs.push((v, push, args));
                    }
                }
            }
        }
        (Route::Generated, Some(g)) => {
            // Per possible segment, the sequences its count states, from its first.
            for s in 0..layout.slots {
                let at = g.count_offset(s) as usize;
                let count = words(&mem.scratch[at..at + 4])[0];
                assert!(count <= g.max_sequences(s), "{stem}: command {s}: {count}");
                for q in 0..u64::from(count) {
                    let seq = (g.sequence_offset(s, 0) + q * u64::from(g.stride)) as usize;
                    let index = words(&mem.scratch[seq..seq + 4])[0] as usize;
                    let push = seq + GeneratedLayout::PUSH_OFFSET as usize;
                    let push = mem.scratch[push..push + g.push_bytes as usize].to_vec();
                    let d = seq + g.dispatch_offset() as usize;
                    let args: [u32; 3] =
                        words(&mem.scratch[d..d + 12]).try_into().expect("dispatch");
                    assert!(index < instantiated.len(), "{stem}: index {index}");
                    jobs.push((index, push, args));
                }
            }
        }
        other => panic!("{stem}: {path:?} with {other:?}"),
    }
    let mut dispatched = 0;
    for (v, push, args) in jobs {
        let cmds = &instantiated[v];
        if args.contains(&0) {
            continue;
        }
        dispatched += 1;
        let region = &cmds.regions[0];
        let mut spec = BTreeMap::new();
        for e in &region.specialization.entries {
            let at = e.offset as usize;
            let mut x = [0u8; 8];
            x[..e.size as usize]
                .copy_from_slice(&region.specialization.data[at..at + e.size as usize]);
            spec.insert(ids[&e.constant_id], u64::from_le_bytes(x));
        }
        let mut descriptors = Vec::new();
        for w in &plan.descriptors {
            let bytes = match &w.source {
                WriteSource::Neutral(_) => vec![0; 4096],
                WriteSource::ArgumentTable { .. } => cmds
                    .descriptor_bytes
                    .iter()
                    .find(|d| (d.set, d.binding) == (w.set, w.binding))
                    .expect("argument table bytes")
                    .bytes
                    .clone(),
                other => panic!("{stem}: a replay variant's write {other:?}"),
            };
            descriptors.push(Buffer {
                slot: Some(symbol(layout_bindings[&(w.set, w.binding)])),
                address: None,
                bytes,
            });
        }
        let extra = descriptors.len();
        let res = Resources {
            buffers: mem.buffers(descriptors),
            push,
            spec,
        };
        let out = run_both(&m, args, res);
        mem.take(out, extra);
    }
    (mem, dispatched)
}

/// On every profile with device addresses, the device replay of the template's commands computes
/// what Metal computes for each, through the preprocess's arguments and slot tables.
#[test]
fn replayed_commands_compute_what_metal_computes() {
    let Some(bc) = fixture() else {
        return;
    };
    let mut ran = 0;
    for (stem, dev) in devices() {
        if dev.profile().fields().buffer_access != BufferAccess::Addressed {
            continue;
        }
        ran += 1;
        // Three commands: threadgroups with the parameters at 256 and the output at 0, an empty
        // command, and threads with the parameters at 0 and the output at 256.
        let commands = [
            record(
                Action::DispatchThreadgroups {
                    grid: [8, 1, 1],
                    local_size: LOCAL,
                },
                PARAMS.0 + 256,
                OUT.0,
            ),
            CommandRecord::default(),
            record(
                Action::DispatchThreads {
                    grid: [64, 1, 1],
                    local_size: LOCAL,
                },
                PARAMS.0,
                OUT.0 + 256,
            ),
        ];
        let mut params = vec![0u8; BYTES];
        for (at, w) in [(0, 3u32), (4, 100), (256, 5), (260, 7)] {
            params[at..at + 4].copy_from_slice(&w.to_le_bytes());
        }
        let (mem, dispatched) = replay_run(&stem, &dev, &bc, ENTRY, LENGTH, &commands, params);
        assert_eq!(
            dispatched, 2,
            "{stem}: the empty command dispatches nothing"
        );
        // What Metal computes: `out[tid] = params[0] * (tid + 1) + params[1]`, per command.
        let expected: Vec<u32> = (0..64u32)
            .map(|tid| 5 * (tid + 1) + 7)
            .chain((0..64).map(|tid| 3 * (tid + 1) + 100))
            .collect();
        assert_eq!(words(&mem.out), expected, "{stem}");
        let report = mem.report();
        assert!(
            !report.replay_fault && !report.overflow && !report.resident_miss,
            "{stem}: {report:?}"
        );
    }
    assert!(ran > 0, "no profile with device addresses");
}

/// `compare(b0, b1)`: each thread `t` writes `b0 == b1 + 16` and `b0 < b1 + 16` (as `uint`s) to
/// `b0[2t]` and `b0[2t + 1]`: pointer compares of two buffer slots, which read each slot's Metal
/// address.
const COMPARE: &str = r#"define void @compare(ptr addrspace(1) noundef "air-buffer-no-alias" %b0, ptr addrspace(1) noundef "air-buffer-no-alias" %b1, i32 %t) {
entry:
  %p = getelementptr inbounds i8, ptr addrspace(1) %b1, i64 16
  %eq = icmp eq ptr addrspace(1) %b0, %p
  %lt = icmp ult ptr addrspace(1) %b0, %p
  %e = zext i1 %eq to i32
  %l = zext i1 %lt to i32
  %t64 = zext i32 %t to i64
  %i = shl i64 %t64, 1
  %o0 = getelementptr inbounds i32, ptr addrspace(1) %b0, i64 %i
  store i32 %e, ptr addrspace(1) %o0, align 4
  %o1 = getelementptr inbounds i32, ptr addrspace(1) %o0, i64 1
  store i32 %l, ptr addrspace(1) %o1, align 4
  ret void
}

!air.kernel = !{!0}
!air.version = !{!20}
!air.language_version = !{!21}
!0 = !{ptr @compare, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"b0"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"b1"}
!5 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"t"}
!20 = !{i32 2, i32 8, i32 0}
!21 = !{!"Metal", i32 4, i32 0, i32 0}
"#;

/// A replay variant reads a slot's Metal address from the slot table the preprocess wrote (its
/// entry's `guest + bind_offset`), so pointer compares of two buffers see each command's own
/// addresses.
#[test]
fn a_replayed_pointer_compare_reads_each_commands_addresses() {
    let Some(bc) = assemble("compare", COMPARE) else {
        return;
    };
    let mut ran = 0;
    for (stem, dev) in devices() {
        if dev.profile().fields().buffer_access != BufferAccess::Addressed {
            continue;
        }
        ran += 1;
        // Compiled plainly, the compares read `SlotGuestBase` push fields; as a replay variant,
        // the root table instead.
        let guest_base = |inputs: &LoweringInputs| {
            let (plan, _) = typed_module_with(&bc, "compare", dev.profile(), inputs)
                .unwrap_or_else(|e| panic!("{stem}: {e}"));
            let fields: Vec<PushField> = plan.push.fields.iter().map(|f| f.field).collect();
            (
                fields
                    .iter()
                    .any(|f| matches!(f, PushField::SlotGuestBase(_))),
                fields.contains(&PushField::RootTableAddress),
            )
        };
        assert!(guest_base(&LoweringInputs::new()).0, "{stem}");
        assert_eq!(guest_base(&replay_inputs()), (false, true), "{stem}");
        // `b0` at 64 and `b1 + 16` at 64: equal; `b0` at 128 below `b1 + 16` at 272; `b0` at
        // 192 above `b1 + 16` in the parameters allocation (`PARAMS` is below `OUT`).
        let dispatch = |b0: u64, b1: u64| {
            let mut r = record(
                Action::DispatchThreadgroups {
                    grid: [1, 1, 1],
                    local_size: LOCAL,
                },
                b0,
                b1,
            );
            r.threadgroup_memory_lengths[0] = 0;
            r
        };
        let commands = [
            dispatch(OUT.0 + 64, OUT.0 + 48),
            dispatch(OUT.0 + 128, OUT.0 + 256),
            dispatch(OUT.0 + 192, PARAMS.0),
        ];
        let (mem, dispatched) =
            replay_run(&stem, &dev, &bc, "compare", 0, &commands, vec![0; BYTES]);
        assert_eq!(dispatched, 3, "{stem}");
        let out = words(&mem.out);
        for (command, at, eq, lt) in [(0, 16, 1, 0), (1, 32, 0, 1), (2, 48, 0, 0)] {
            for t in 0..LOCAL[0] as usize {
                assert_eq!(
                    [out[at + 2 * t], out[at + 2 * t + 1]],
                    [eq, lt],
                    "{stem}: command {command}, thread {t}"
                );
            }
        }
        assert!(!mem.report().replay_fault, "{stem}");
    }
    assert!(ran > 0, "no profile with device addresses");
}

/// The variant needs device addresses; the binder replays only a replay variant's plan, and only
/// with no kernel buffer bound.
#[test]
fn what_cannot_replay_is_refused_by_name() {
    let Some(bc) = fixture() else {
        return;
    };
    let mut unaddressed = 0;
    for (stem, dev) in devices_and_unaddressed() {
        if dev.profile().fields().buffer_access != BufferAccess::Addressed {
            match pipeline(&dev, &bc, ENTRY, &replay_inputs()) {
                Err(e) if refuses_threadgroup_memory(&e, dev.profile()) => {}
                Err(Error::Unsupported(u)) => {
                    assert_eq!(u.feature, Feature::GpuEncodedIndirectCommands, "{stem}");
                    unaddressed += 1;
                }
                other => panic!("{stem}: {:?}", other.map(|_| ())),
            }
            continue;
        }
        let state = shared_state(LENGTH);
        let plain = pipeline(&dev, &bc, ENTRY, &LoweringInputs::new()).expect("pipeline");
        let plan = plain.plan(&dev, &state.shape()).expect("plan");
        assert_eq!(
            plan.instantiate_replay(&state, LOCAL),
            Err(BindError::InvalidReplay {
                reason: ReplayRefusal::NotReplayVariant
            }),
            "{stem}"
        );
        let replay = pipeline(&dev, &bc, ENTRY, &replay_inputs()).expect("pipeline");
        let mut bound = shared_state(LENGTH);
        bound
            .stages
            .get_mut(&Stage::Kernel)
            .expect("kernel")
            .buffers
            .insert(
                1,
                BoundBuffer::Buffer {
                    length: BYTES as u64,
                    offset: 0,
                    gpu_address: OUT.0,
                    device_address: Some(OUT.1),
                    attribute_stride: None,
                },
            );
        let plan = replay.plan(&dev, &bound.shape()).expect("plan");
        assert_eq!(
            plan.instantiate_replay(&bound, LOCAL),
            Err(BindError::InvalidReplay {
                reason: ReplayRefusal::KernelBufferBound
            }),
            "{stem}"
        );
    }
    assert!(unaddressed > 0, "no device without device addresses");
}

/// The template that observes its regions: each thread adds a weighted sum of its grid builtins and
/// its neighbour's staged index to `out[index % params[1]]`.
const GRID: &str = "icb_replay_grid_template";
/// The grid template's threadgroup length: the 128-byte length its encoders set.
const GRID_LENGTH: u32 = 128;

fn grid_fixture() -> Option<Vec<u8>> {
    match llvm_assemble(&engine_dir().join(format!("{GRID}.ll"))) {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected {GRID}.ll: {why}"),
    }
}

/// A grid-template command: `action` with the parameters window at `params` and the output
/// window at `out` (byte offsets into the two allocations), and the 128-byte length.
fn grid_record(action: Action, params: u64, out: u64) -> CommandRecord {
    let mut r = record(action, PARAMS.0 + params, OUT.0 + out);
    r.threadgroup_memory_lengths[0] = GRID_LENGTH;
    r
}

/// The parameters allocation: `(word offset, params[0], params[1])` windows.
fn grid_params(windows: &[(usize, u32, u32)]) -> Vec<u8> {
    let mut params = vec![0u8; BYTES];
    for &(at, base, modulus) in windows {
        params[4 * at..4 * at + 4].copy_from_slice(&base.to_le_bytes());
        params[4 * at + 4..4 * at + 8].copy_from_slice(&modulus.to_le_bytes());
    }
    params
}

/// The window of kernel buffer `slot` of `c`: its allocation `(guest, host)` and its offset.
fn window(c: &CommandRecord, slot: usize) -> ((u64, u64), u64) {
    let address = c.bindings.buffers(BufferStage::Kernel)[slot]
        .expect("a buffer set")
        .address;
    let allocation = [PARAMS, OUT]
        .into_iter()
        .find(|&(guest, _)| (guest..guest + BYTES as u64).contains(&address))
        .expect("an address in an allocation");
    (allocation, address - allocation.0)
}

/// What Metal computes for the grid template's `commands` over the parameters `params`: per
/// threadgroup of each dispatch (an edge threadgroup of a thread grid has the remaining threads),
/// each thread's sum into its slot of the command's output window, as the source states it.
fn metal_grid(commands: &[CommandRecord], params: &[u8]) -> Vec<u8> {
    let params = words(params);
    let mut out = vec![0u32; BYTES / 4];
    for c in commands {
        let (groups, local, threads) = match c.action {
            Action::Empty => continue,
            Action::DispatchThreadgroups { grid, local_size } => (
                grid,
                local_size,
                std::array::from_fn(|i| grid[i] * local_size[i]),
            ),
            Action::DispatchThreads { grid, local_size } => (
                std::array::from_fn(|i| grid[i].div_ceil(local_size[i])),
                local_size,
                grid,
            ),
            other => panic!("{other:?}"),
        };
        let p = window(c, 0).1 as usize / 4;
        let o = window(c, 1).1 as usize / 4;
        let (base, modulus) = (params[p], params[p + 1]);
        for gz in 0..groups[2] {
            for gy in 0..groups[1] {
                for gx in 0..groups[0] {
                    let g = [gx, gy, gz];
                    let size: [u32; 3] =
                        std::array::from_fn(|i| local[i].min(threads[i] - g[i] * local[i]));
                    let n = size[0] * size[1] * size[2];
                    for lz in 0..size[2] {
                        for ly in 0..size[1] {
                            for lx in 0..size[0] {
                                let l = [lx, ly, lz];
                                let lin = lx + ly * size[0] + lz * size[0] * size[1];
                                let neighbour = (lin + 1) % n;
                                let tid: [u32; 3] = std::array::from_fn(|i| g[i] * local[i] + l[i]);
                                let terms = [
                                    (3, size[0]),
                                    (5, size[1]),
                                    (7, local[0]),
                                    (11, local[1]),
                                    (13, g[0]),
                                    (17, g[1]),
                                    (19, groups[0]),
                                    (23, groups[1]),
                                    (29, threads[0]),
                                    (31, threads[1]),
                                    (37, lx),
                                    (41, ly),
                                    (43, neighbour),
                                ];
                                let v = terms
                                    .iter()
                                    .fold(base, |v, &(k, x)| v.wrapping_add(x.wrapping_mul(k)));
                                let index =
                                    tid[0] + tid[1] * threads[0] + tid[2] * threads[0] * threads[1];
                                let at = o + (index % modulus) as usize;
                                out[at] = out[at].wrapping_add(v);
                            }
                        }
                    }
                }
            }
        }
    }
    out.iter().flat_map(|w| w.to_le_bytes()).collect()
}

/// The bind state of command `c`'s own dispatch: its kernel buffers bound in their allocations,
/// its threadgroup lengths, and the two resident allocations; and each bound slot's allocation
/// (its guest address).
fn command_state(c: &CommandRecord) -> (BindState, BTreeMap<u32, u64>) {
    let mut state = shared_state(0);
    let kernel = state.stages.get_mut(&Stage::Kernel).expect("kernel");
    let mut slots = BTreeMap::new();
    for slot in 0..2 {
        let ((guest, host), offset) = window(c, slot);
        kernel.buffers.insert(
            slot as u32,
            BoundBuffer::Buffer {
                length: BYTES as u64,
                offset,
                gpu_address: guest,
                device_address: Some(host),
                attribute_stride: None,
            },
        );
        slots.insert(slot as u32, guest);
    }
    for (i, &l) in c.threadgroup_memory_lengths.iter().enumerate() {
        if l != 0 {
            kernel.threadgroup_lengths.insert(i as u32, l);
        }
    }
    (state, slots)
}

/// How [`host_replay`] compiles `entry` and dispatches each command.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Run {
    /// Compiled plainly, each command its own direct dispatch.
    Plain,
    /// A flat variant of this many threads per threadgroup, dispatched indirectly.
    Flat([u32; 3]),
    /// A flat variant dispatched directly: each `dispatchThreads:` command in one region of its
    /// own threadgroup size, reading the record `instantiate_dispatch` states.
    FlatDirect,
    /// A folded variant of each command's threadgroup size.
    Folded,
}

/// The host replay of `commands` (K14b) over `entry` of `bc` compiled plainly, on `dev`, with the
/// parameters allocation holding `params`: each command its own dispatch, binding its kernel
/// buffers in their allocations and its threadgroup lengths, over the regions
/// `instantiate_dispatch` plans, in order. The output allocation it leaves.
///
/// With [`Run::Flat`] threads per threadgroup, `entry` compiles as a flat variant
/// (`VariantKey::flat_grid`) and each command runs as one indirect dispatch of that threadgroup
/// size at the group counts of its flat dispatch ([`flat_groups`]), reading its grid from the
/// region record at its `ReplayRegionAddress`, as a single-dispatch preprocess writes both. With
/// [`Run::FlatDirect`], `entry` compiles as a flat variant too and each `dispatchThreads:` command
/// runs as the one direct region `instantiate_dispatch` plans, of the command's own threadgroup
/// size, reading the record the plan states (`DispatchCommands::region_record`). With
/// [`Run::Folded`], `entry` compiles as a folded variant (`VariantKey::folded_groups`) and each
/// threadgroup command runs as one indirect dispatch of its threadgroup size at its folded group
/// counts ([`folded_counts`]), reading its group counts and grid from the region record at its
/// `ReplayRegionAddress`.
fn host_replay(
    stem: &str,
    (dev, bc, entry): (&Device, &[u8], &str),
    run: Run,
    commands: &[CommandRecord],
    params: &[u8],
) -> Vec<u8> {
    const ROOT: u64 = 0x0000_0040_0000_0000;
    const REGION: u64 = 0x0000_0041_0000_0000;
    // The variant's own threadgroup size where a flat variant's dispatches are indirect; a direct
    // one takes each command's.
    let flat = match run {
        Run::Flat(local) => Some(local),
        Run::FlatDirect | Run::Plain | Run::Folded => None,
    };
    let flat_grid = flat.is_some() || run == Run::FlatDirect;
    let folded = run == Run::Folded;
    let mut key = VariantKey::new();
    key.flat_grid = flat_grid;
    key.folded_groups = folded;
    let inputs = key.lowering_inputs(Stage::Kernel);
    let pipe = pipeline(dev, bc, entry, &inputs).unwrap_or_else(|e| panic!("{stem}: {e}"));
    let (_, m) = typed_module_with(bc, entry, dev.profile(), &inputs)
        .unwrap_or_else(|e| panic!("{stem}: {e}"));
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
    // The two allocations, by guest address.
    let mut memory: BTreeMap<u64, Vec<u8>> =
        BTreeMap::from([(PARAMS.0, params.to_vec()), (OUT.0, vec![0; BYTES])]);
    for c in commands {
        let dispatch = match c.action {
            Action::Empty => continue,
            Action::DispatchThreadgroups { grid, local_size } => Dispatch::Threadgroups {
                groups: grid,
                threads_per_group: local_size,
            },
            Action::DispatchThreads { grid, local_size } => Dispatch::Threads {
                threads: grid,
                threads_per_group: local_size,
            },
            other => panic!("{other:?}"),
        };
        // A flat or folded dispatch's group counts and region record, which its preprocess
        // writes; a direct dispatch on a flat variant keeps its own counts and reads the record
        // the plan states.
        let (dispatch, flat_counts, record) = match flat {
            None if run == Run::FlatDirect => {
                let volume = dispatch
                    .threads_per_group()
                    .iter()
                    .map(|&l| u64::from(l))
                    .product::<u64>();
                if volume == 0 || volume > u64::from(FLAT_MAX) {
                    continue;
                }
                (dispatch, None, Vec::new())
            }
            None if folded => {
                let Some((counts, record)) = folded_counts(dev, &dispatch) else {
                    continue;
                };
                let dispatch = Dispatch::ThreadgroupsIndirect {
                    indirect: IndirectRef::new(0, 0),
                    threads_per_group: dispatch.threads_per_group(),
                };
                (dispatch, Some(counts), record)
            }
            None => (dispatch, None, Vec::new()),
            Some(local) => {
                let Some((threads, groups)) = flat_groups(dev, &dispatch, local) else {
                    continue;
                };
                let mut record = vec![0u8; 32];
                for (i, t) in threads.into_iter().enumerate() {
                    record[16 + 4 * i..20 + 4 * i].copy_from_slice(&t.to_le_bytes());
                }
                let dispatch = Dispatch::ThreadgroupsIndirect {
                    indirect: IndirectRef::new(0, 0),
                    threads_per_group: local,
                };
                (dispatch, Some(groups), record)
            }
        };
        let (state, slots) = command_state(c);
        let plan = pipe
            .plan(dev, &state.shape())
            .unwrap_or_else(|e| panic!("{stem}: {e:?}"));
        assert_eq!(plan.variant, key, "{stem}");
        let cmds = plan
            .instantiate_dispatch(&state, &dispatch)
            .unwrap_or_else(|e| panic!("{stem}: {dispatch:?}: {e:?}"));
        // A direct dispatch on a flat variant: the record is the plan's, the consumer's to write.
        let record = match cmds.region_record {
            Some(bytes) => bytes.to_vec(),
            None => record,
        };
        if flat_grid || folded {
            assert_eq!(
                cmds.regions.len(),
                1,
                "{stem}: a flat or folded dispatch is one region"
            );
            assert!(cmds.indirect_clamp.is_none(), "{stem}: no clamp prepass");
        }
        for region in &cmds.regions {
            let expected =
                (flat.is_some() || folded).then_some(IndirectArgs::Buffer(IndirectRef::new(0, 0)));
            assert_eq!(region.indirect, expected, "{stem}");
            assert_eq!(
                cmds.region_record.is_some(),
                run == Run::FlatDirect,
                "{stem}: only a direct dispatch on a flat variant states its own record"
            );
            // Each buffer the region reaches, and the allocation and offset its bytes return to.
            let mut buffers = Vec::new();
            let mut back: Vec<Option<(u64, usize)>> = Vec::new();
            for w in &plan.descriptors {
                let sym = symbol(layout[&(w.set, w.binding)]);
                let (bytes, to) = match &w.source {
                    WriteSource::Buffer { slot } => {
                        let r = cmds
                            .descriptor_ranges
                            .iter()
                            .find(|r| (r.set, r.binding) == (w.set, w.binding))
                            .expect("a range for every buffer write")
                            .range;
                        let guest = slots[&slot.index];
                        let (o, n) = (r.offset as usize, r.range as usize);
                        (memory[&guest][o..o + n].to_vec(), Some((guest, o)))
                    }
                    WriteSource::Neutral(_) => (vec![0; 4096], None),
                    WriteSource::ArgumentTable { .. } => (
                        cmds.descriptor_bytes
                            .iter()
                            .find(|d| (d.set, d.binding) == (w.set, w.binding))
                            .expect("argument table bytes")
                            .bytes
                            .clone(),
                        None,
                    ),
                    other => panic!("{stem}: a host-replayed write {other:?}"),
                };
                buffers.push(Buffer {
                    slot: Some(sym),
                    address: None,
                    bytes,
                });
                back.push(to);
            }
            for (guest, host) in [PARAMS, OUT] {
                buffers.push(Buffer {
                    slot: None,
                    address: Some(host),
                    bytes: memory[&guest].clone(),
                });
                back.push(Some((guest, 0)));
            }
            let mut table = Vec::new();
            if let Some(t) = &cmds.root_table {
                table = t.bytes.clone();
                let at = RootTable::SINK_HOST_OFFSET as usize;
                table[at..at + 8].copy_from_slice(&SINK.to_le_bytes());
            }
            for (address, bytes) in [
                (ROOT, table),
                (REGION, record.clone()),
                (STATUS, ExecutionStatus::initial().to_vec()),
                (SINK, vec![0; 64]),
            ] {
                buffers.push(Buffer {
                    slot: None,
                    address: Some(address),
                    bytes,
                });
                back.push(None);
            }
            let mut push = Vec::new();
            if let Some(w) = region.push_constants.first() {
                push = vec![0u8; (w.offset as usize) + w.bytes.len()];
                push[w.offset as usize..].copy_from_slice(&w.bytes);
            }
            for hole in &cmds.push_holes {
                let (offset, address) = match *hole {
                    PushHole::RootTableAddress { offset, .. } => (offset, ROOT),
                    PushHole::ExecutionStatusAddress { offset, .. } => (offset, STATUS),
                    PushHole::ReplayRegionAddress { offset, .. } if flat_grid || folded => {
                        (offset, REGION)
                    }
                    other => panic!("{stem}: {other:?}"),
                };
                push[offset as usize..offset as usize + 8].copy_from_slice(&address.to_le_bytes());
            }
            let mut spec = BTreeMap::new();
            for e in &region.specialization.entries {
                let at = e.offset as usize;
                let mut x = [0u8; 8];
                x[..e.size as usize]
                    .copy_from_slice(&region.specialization.data[at..at + e.size as usize]);
                spec.insert(ids[&e.constant_id], u64::from_le_bytes(x));
            }
            let inputs: Vec<Vec<u8>> = buffers.iter().map(|b| b.bytes.clone()).collect();
            let res = Resources {
                buffers,
                push,
                spec,
            };
            let outs = run_both(&m, flat_counts.unwrap_or(region.group_count), res);
            // A region reaches each byte through one path: only the bytes it changed return.
            for ((after, before), to) in outs.into_iter().zip(inputs).zip(back) {
                let Some((guest, offset)) = to else {
                    continue;
                };
                let target = memory.get_mut(&guest).expect("allocation");
                for (i, (a, b)) in after.into_iter().zip(before).enumerate() {
                    if a != b {
                        target[offset + i] = a;
                    }
                }
            }
        }
    }
    assert_eq!(
        memory[&PARAMS.0], params,
        "{stem}: the parameters are only read"
    );
    memory.remove(&OUT.0).expect("output")
}

/// The device replay of the grid template's `commands` over a variant per threadgroup size of
/// `locals`, recording `regions` regions per command, on `dev`, by `path`.
fn grid_replay(
    stem: &str,
    (dev, bc, path): (&Device, &[u8], Route),
    locals: &[[u32; 3]],
    commands: &[CommandRecord],
    (regions, params): (u32, Vec<u8>),
) -> (Memory, usize) {
    let count = commands.len() as u32;
    let mut execution = ReplayExecution::new(count, 0, count);
    execution.regions = regions;
    replay_variants_run(
        stem,
        (dev, bc, GRID),
        (GRID_LENGTH, locals, None),
        commands,
        (execution, path),
        params,
    )
}

/// The paths `dev` runs: K14c's, and the generated-commands path where it states it.
fn paths(dev: &Device) -> Vec<Route> {
    let mut out = vec![Route::Arguments];
    if generates(dev) {
        out.push(Route::Generated);
    }
    out
}

/// On every profile with device addresses, ragged thread grids replay on the device as the host
/// replay runs them and as Metal computes them: each edge region on the variant of its size, with
/// the command's grid, nominal threadgroup size, and threadgroup positions.
#[test]
fn a_ragged_thread_grid_replays_as_the_host_replay_runs_it() {
    let Some(bc) = grid_fixture() else {
        return;
    };
    // Three ragged grids in threadgroups of 4 x 2 (two shapes of edge each, one with no whole
    // threadgroup along x), a uniform dispatch of threadgroups, and an empty command. The first
    // and the last dispatch share an output window.
    let local = [4, 2, 1];
    let commands = [
        grid_record(
            Action::DispatchThreads {
                grid: [10, 3, 1],
                local_size: local,
            },
            0,
            0,
        ),
        grid_record(
            Action::DispatchThreadgroups {
                grid: [2, 2, 1],
                local_size: local,
            },
            8,
            128,
        ),
        CommandRecord::default(),
        grid_record(
            Action::DispatchThreads {
                grid: [3, 5, 1],
                local_size: local,
            },
            16,
            0,
        ),
    ];
    let params = grid_params(&[(0, 1000, 24), (2, 7, 32), (4, 0x8000_0000, 24)]);
    // The command's size, and each edge's.
    let locals = [
        [4, 2, 1],
        [2, 2, 1],
        [4, 1, 1],
        [2, 1, 1],
        [3, 2, 1],
        [3, 1, 1],
    ];
    let metal = metal_grid(&commands, &params);
    let (mut ran, mut generated) = (0, 0);
    for (stem, dev) in devices() {
        if dev.profile().fields().buffer_access != BufferAccess::Addressed {
            continue;
        }
        ran += 1;
        let host = host_replay(&stem, (&dev, &bc, GRID), Run::Plain, &commands, &params);
        assert_eq!(words(&host), words(&metal), "{stem}: the host replay");
        for path in paths(&dev) {
            let (mem, dispatched) = grid_replay(
                &stem,
                (&dev, &bc, path),
                &locals,
                &commands,
                (MAX_REPLAY_REGIONS, params.clone()),
            );
            // 4 + 1 + 0 + 2 regions.
            assert_eq!(dispatched, 7, "{stem} {path:?}");
            assert_eq!(
                words(&mem.out),
                words(&host),
                "{stem} {path:?}: the device replay"
            );
            let report = mem.report();
            assert!(
                !report.replay_fault && !report.overflow && !report.resident_miss,
                "{stem} {path:?}: {report:?}"
            );
            generated += usize::from(path == Route::Generated);
        }
    }
    assert!(ran > 0, "no profile with device addresses");
    assert!(generated > 0, "no profile states compute pipeline binding");
}

/// On a device whose group counts are limited to 3 x 2 x 64, group counts past the limit replay as
/// the host replay runs them (the chunks, and a thread grid's edge after them) and as Metal
/// computes them; a command with more regions than the execution records is the typed `REGIONS`
/// fault and the execution dispatches nothing.
#[test]
fn a_group_count_past_the_limit_replays_as_the_host_replay_runs_it() {
    let Some(bc) = grid_fixture() else {
        return;
    };
    let commands = [
        // 3 x 2 chunks of at most 3 x 2 threadgroups.
        grid_record(
            Action::DispatchThreadgroups {
                grid: [7, 3, 1],
                local_size: [2, 1, 1],
            },
            0,
            0,
        ),
        // Four whole threadgroups in two chunks, and an edge of one thread.
        grid_record(
            Action::DispatchThreads {
                grid: [9, 1, 1],
                local_size: [2, 1, 1],
            },
            8,
            256,
        ),
    ];
    let params = grid_params(&[(0, 5, 40), (2, 9, 16)]);
    let locals = [[2, 1, 1], [1, 1, 1]];
    let metal = metal_grid(&commands, &params);
    let (mut ran, mut generated) = (0, 0);
    for (stem, dev) in devices_with(|d| d.limits.max_compute_work_group_count = [3, 2, 64]) {
        if dev.profile().fields().buffer_access != BufferAccess::Addressed {
            continue;
        }
        ran += 1;
        let host = host_replay(&stem, (&dev, &bc, GRID), Run::Plain, &commands, &params);
        assert_eq!(words(&host), words(&metal), "{stem}: the host replay");
        for path in paths(&dev) {
            let (mem, dispatched) = grid_replay(
                &stem,
                (&dev, &bc, path),
                &locals,
                &commands,
                (MAX_REPLAY_REGIONS, params.clone()),
            );
            assert_eq!(dispatched, 6 + 3, "{stem} {path:?}");
            assert_eq!(
                words(&mem.out),
                words(&host),
                "{stem} {path:?}: the device replay"
            );
            let report = mem.report();
            assert!(
                !report.replay_fault && !report.overflow && !report.resident_miss,
                "{stem} {path:?}: {report:?}"
            );

            // Recording five regions per command, the first command's six do not fit: it
            // dispatches nothing, and the second still runs.
            let (mem, dispatched) = grid_replay(
                &stem,
                (&dev, &bc, path),
                &locals,
                &commands,
                (5, params.clone()),
            );
            assert_eq!(
                dispatched, 3,
                "{stem} {path:?}: the faulted command dispatches nothing"
            );
            assert_eq!(
                words(&mem.out),
                words(&metal_grid(&commands[1..], &params)),
                "{stem} {path:?}"
            );
            let report = mem.report();
            assert!(report.replay_fault, "{stem} {path:?}: {report:?}");
            assert_eq!(
                report.replay_faults,
                ReplayFaults::REGIONS,
                "{stem} {path:?}"
            );
            generated += usize::from(path == Route::Generated);
        }
    }
    assert!(ran > 0, "no profile with device addresses");
    assert!(generated > 0, "no profile states compute pipeline binding");
}

/// `gate-lavapipe`'s description as a device, its generated-commands facts changed by `tweak`.
fn gate_lavapipe(tweak: impl Fn(&mut Option<GeneratedCommandsProperties>)) -> Device {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles/gate-lavapipe.json");
    let text = std::fs::read_to_string(path).expect("profile");
    let mut desc = DeviceDescription::from_khronos_json(&text, None).expect("description");
    tweak(&mut desc.generated_commands);
    Device::from_queries(desc, DevicePolicy::default()).expect("device")
}

/// The limit `set` changes on a description that states the facts.
fn limited(
    set: fn(&mut GeneratedCommandsProperties),
) -> impl Fn(&mut Option<GeneratedCommandsProperties>) {
    move |g| set(g.as_mut().expect("gate-lavapipe states generated commands"))
}

/// The generated-commands path runs where a committed profile states
/// `VK_EXT_device_generated_commands` with compute pipeline binding, and is held to every limit
/// the description states: a device that states none, binds no compute pipeline, or holds fewer
/// pipelines, tokens, token bytes, stride, or sequences than the replay needs is the typed
/// refusal naming it (K14c's path remains), and a sequence count below eight commands' regions
/// shortens the executions. Variants that bind other descriptors cannot share one execution.
#[test]
fn the_generated_commands_path_holds_to_what_the_device_states() {
    let Some(bc) = grid_fixture() else {
        return;
    };
    let generating: Vec<String> = devices()
        .into_iter()
        .filter(|(_, d)| generates(d))
        .map(|(stem, _)| stem)
        .collect();
    assert_eq!(
        generating,
        [
            "amd-780m-hawkpoint-radv",
            "amd-780m-radv",
            "amd-rdna2-apu-radv",
            "amd-rdna3-radv",
            "gate-lavapipe",
            "gate-lavapipe-x86_64",
            "intel-arl-anv",
            "nvidia-rtx50-linux",
            "nvidia-rtx5080-laptop"
        ]
    );
    let dev = gate_lavapipe(|_| {});
    let facts = *dev.generated_commands().expect("facts");
    assert_eq!(
        (
            facts.max_indirect_sequence_count,
            facts.max_indirect_commands_token_count,
            facts.max_indirect_commands_token_offset,
            facts.max_indirect_commands_indirect_stride,
            facts.max_indirect_pipeline_count,
        ),
        (1_048_576, 16, 2047, 2048, 4096),
        "gate-lavapipe's committed facts"
    );

    let inputs = replay_inputs();
    let pipe = pipeline(&dev, &bc, GRID, &inputs).expect("pipeline");
    let state = shared_state(GRID_LENGTH);
    let plan = pipe.plan(&dev, &state.shape()).expect("plan");
    let locals = [[4, 2, 1], [2, 2, 1]];
    let mut lengths = [0; 31];
    lengths[0] = GRID_LENGTH;
    let variants: Vec<ReplayVariant> = locals
        .iter()
        .map(|&l| ReplayVariant::new(NonZeroU64::new(TOKEN).expect("token"), l, lengths))
        .collect();
    let cmds: Vec<_> = locals
        .iter()
        .map(|&l| plan.instantiate_replay(&state, l).expect("instantiate"))
        .collect();
    let generated: Vec<GeneratedVariant<'_>> = cmds
        .iter()
        .map(|c| GeneratedVariant::new(&pipe, &plan, c))
        .collect();
    let execution = ReplayExecution::new(4, 0, 4);
    let run = |dev: &Device, generated: &[GeneratedVariant<'_>]| {
        replay_preprocess_generated(dev, &state, &execution, &variants, generated)
    };
    let refusal = |dev: &Device, generated: &[GeneratedVariant<'_>]| match run(dev, generated) {
        Err(BindError::InvalidReplay {
            reason: ReplayRefusal::GeneratedCommands { reason },
        }) => reason,
        other => panic!("{other:?}"),
    };

    let pre = run(&dev, &generated).expect("the generated-commands path");
    let g = pre.layout.generated.expect("its layout");
    assert_eq!(g.segment_commands, MAX_SEGMENT_COMMANDS);
    assert_eq!(g.stride, GeneratedLayout::PUSH_OFFSET + g.push_bytes + 12);
    let replay = pre.generated.expect("its templates");
    assert_eq!(
        replay.templates.len(),
        2 * (16 + g.push_bytes as usize),
        "a template per variant"
    );
    let range = &replay.pipeline_layout.push_constants;
    assert_eq!(range.len(), 1);
    assert_eq!((range[0].offset, range[0].size), (0, g.push_bytes));
    assert_eq!(replay.pipeline_layout.sets, pipe.layout.sets);
    assert_eq!(pre.layout.arguments_offset, pre.layout.bytes, "no argument");

    // Fewer sequences than eight commands' regions shorten the executions.
    let short = gate_lavapipe(limited(|g| g.max_indirect_sequence_count = 16));
    let pre = run(&short, &generated).expect("shorter executions");
    assert_eq!(pre.layout.generated.expect("layout").segment_commands, 2);

    let push = g.push_bytes;
    for (name, dev, expected) in [
        (
            "no facts",
            gate_lavapipe(|g| *g = None),
            GeneratedRefusal::NotEnabled,
        ),
        (
            "no compute binding",
            gate_lavapipe(limited(|g| {
                g.pipeline_binding_stages = crate::vk::ShaderStageFlags::FRAGMENT
            })),
            GeneratedRefusal::NoComputePipelineBinding,
        ),
        (
            "one pipeline",
            gate_lavapipe(limited(|g| g.max_indirect_pipeline_count = 1)),
            GeneratedRefusal::PipelineCount {
                variants: 2,
                limit: 1,
            },
        ),
        (
            "two tokens",
            gate_lavapipe(limited(|g| g.max_indirect_commands_token_count = 2)),
            GeneratedRefusal::TokenCount { limit: 2 },
        ),
        (
            "a token offset of 8",
            gate_lavapipe(limited(|g| g.max_indirect_commands_token_offset = 8)),
            GeneratedRefusal::TokenOffset {
                offset: 4 + push,
                limit: 8,
            },
        ),
        (
            "a stride of 16",
            gate_lavapipe(limited(|g| g.max_indirect_commands_indirect_stride = 16)),
            GeneratedRefusal::Stride {
                stride: 16 + push,
                limit: 16,
            },
        ),
        (
            "four sequences",
            gate_lavapipe(limited(|g| g.max_indirect_sequence_count = 4)),
            GeneratedRefusal::SequenceCount {
                sequences: MAX_REPLAY_REGIONS,
                limit: 4,
            },
        ),
    ] {
        assert_eq!(refusal(&dev, &generated), expected, "{name}");
    }
    // lavapipe's older profile states the extension but no stage.
    let (_, lavapipe) = devices()
        .into_iter()
        .find(|(stem, _)| stem == "lavapipe")
        .expect("lavapipe");
    assert!(lavapipe.generated_commands().is_some());
    assert_eq!(
        refusal(&lavapipe, &generated),
        GeneratedRefusal::NoComputePipelineBinding
    );
    assert_eq!(
        refusal(&dev, &generated[..1]),
        GeneratedRefusal::Variants {
            registered: 2,
            generated: 1
        }
    );
    // Each generated variant must be the registered one's size: swapped, index 0 runs 2 x 2 x 1
    // threads where the registered variant 0 is 4 x 2 x 1.
    assert_eq!(
        refusal(&dev, &[generated[1], generated[0]]),
        GeneratedRefusal::Unmatched { index: 0 }
    );
    // A variant whose pipeline layout differs binds other descriptors.
    let mut other = pipe.clone();
    other.layout.push_descriptor_eligible = !other.layout.push_descriptor_eligible;
    let mixed = [generated[0], GeneratedVariant::new(&other, &plan, &cmds[1])];
    assert_eq!(
        refusal(&dev, &mixed),
        GeneratedRefusal::Descriptors { index: 1 }
    );
}

/// `flat(b0, b1, p, g)`: each thread adds `b0[0] + 3 p.x + 5 p.y + 7 p.z + 11 g.x + 13 g.y +
/// 17 g.z` to `b1[p.x + p.y g.x + p.z g.x g.y]` by a relaxed atomic whose result it does not
/// read, reading only `thread_position_in_grid` and `threads_per_grid`: it does not observe its
/// threadgroup partition, and a thread run twice adds twice. (The same addition as a load and a
/// store is [`flat_rmw`], which takes a lockstep barrier and so has no flat variant.)
const FLAT: &str = r#"define void @flat(ptr addrspace(1) noundef "air-buffer-no-alias" %b0, ptr addrspace(1) noundef "air-buffer-no-alias" %b1, <3 x i32> noundef %p, <3 x i32> noundef %g) {
entry:
  %base = load i32, ptr addrspace(1) %b0, align 4
  %px = extractelement <3 x i32> %p, i64 0
  %py = extractelement <3 x i32> %p, i64 1
  %pz = extractelement <3 x i32> %p, i64 2
  %gx = extractelement <3 x i32> %g, i64 0
  %gy = extractelement <3 x i32> %g, i64 1
  %gz = extractelement <3 x i32> %g, i64 2
  %a0 = mul i32 %px, 3
  %a1 = mul i32 %py, 5
  %a2 = mul i32 %pz, 7
  %a3 = mul i32 %gx, 11
  %a4 = mul i32 %gy, 13
  %a5 = mul i32 %gz, 17
  %s0 = add i32 %base, %a0
  %s1 = add i32 %s0, %a1
  %s2 = add i32 %s1, %a2
  %s3 = add i32 %s2, %a3
  %s4 = add i32 %s3, %a4
  %v = add i32 %s4, %a5
  %row = mul i32 %py, %gx
  %plane = mul i32 %gx, %gy
  %layer = mul i32 %pz, %plane
  %i0 = add i32 %px, %row
  %i = add i32 %i0, %layer
  %i64 = zext i32 %i to i64
  %o = getelementptr inbounds i32, ptr addrspace(1) %b1, i64 %i64
  %old = atomicrmw add ptr addrspace(1) %o, i32 %v monotonic, align 4
  ret void
}

!air.kernel = !{!0}
!air.version = !{!20}
!air.language_version = !{!21}
!0 = !{ptr @flat, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5, !6}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"b0"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"b1"}
!5 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint3", !"air.arg_name", !"p"}
!6 = !{i32 3, !"air.threads_per_grid", !"air.arg_type_name", !"uint3", !"air.arg_name", !"g"}
!20 = !{i32 2, i32 8, i32 0}
!21 = !{!"Metal", i32 4, i32 0, i32 0}
"#;

/// [`FLAT`] adding to `b1[p.x + p.y g.x + p.z g.x g.y]` by a load and a store.
fn flat_rmw() -> String {
    FLAT.replace(
        "  %old = atomicrmw add ptr addrspace(1) %o, i32 %v monotonic, align 4\n",
        "  %cur = load i32, ptr addrspace(1) %o, align 4\n  %n = add i32 %cur, %v\n  store i32 \
         %n, ptr addrspace(1) %o, align 4\n",
    )
}

/// The most threads per threadgroup of the flat tests' state.
const FLAT_MAX: u32 = 1024;

/// What Metal computes for the flat kernel's `commands`: each command of a nonempty grid and a
/// threadgroup of 1 to `FLAT_MAX` threads adds each grid thread's value into its output window.
fn metal_flat(commands: &[CommandRecord], params: &[u8]) -> Vec<u8> {
    let params = words(params);
    let mut out = vec![0u32; BYTES / 4];
    for c in commands {
        let (threads, local) = match c.action {
            Action::Empty => continue,
            Action::DispatchThreadgroups { grid, local_size } => (
                std::array::from_fn::<u32, 3, _>(|i| grid[i] * local_size[i]),
                local_size,
            ),
            Action::DispatchThreads { grid, local_size } => (grid, local_size),
            other => panic!("{other:?}"),
        };
        let volume = local.iter().map(|&l| u64::from(l)).product::<u64>();
        if volume == 0 || volume > u64::from(FLAT_MAX) {
            continue;
        }
        let base = params[window(c, 0).1 as usize / 4];
        let o = window(c, 1).1 as usize / 4;
        for z in 0..threads[2] {
            for y in 0..threads[1] {
                for x in 0..threads[0] {
                    let terms = [
                        (3, x),
                        (5, y),
                        (7, z),
                        (11, threads[0]),
                        (13, threads[1]),
                        (17, threads[2]),
                    ];
                    let v = terms
                        .iter()
                        .fold(base, |v, &(k, a)| v.wrapping_add(a.wrapping_mul(k)));
                    let at = o + (x + y * threads[0] + z * threads[0] * threads[1]) as usize;
                    out[at] = out[at].wrapping_add(v);
                }
            }
        }
    }
    out.iter().flat_map(|w| w.to_le_bytes()).collect()
}

/// The device replay of the flat kernel's `commands` over one flat variant of `local` threads, on
/// `dev`, by `path`.
fn flat_replay(
    stem: &str,
    (dev, bc, path): (&Device, &[u8], Route),
    local: [u32; 3],
    commands: &[CommandRecord],
    params: Vec<u8>,
) -> (Memory, usize) {
    let count = commands.len() as u32;
    replay_variants_run(
        stem,
        (dev, bc, "flat"),
        (0, &[local], Some(FLAT_MAX)),
        commands,
        (ReplayExecution::new(count, 0, count), path),
        params,
    )
}

/// A flat-kernel command: `action` with its base word at `params` and its output window at
/// `out` (byte offsets into the two allocations), with the template record's 32-byte length,
/// which the flat variant ignores.
fn flat_record(action: Action, params: u64, out: u64) -> CommandRecord {
    record(action, PARAMS.0 + params, OUT.0 + out)
}

/// The grid of a flat kernel's `dispatch` (a `Threads` or `Threadgroups` one), by Metal's
/// arithmetic, and the group counts of its flat dispatch in threadgroups of `local` threads on
/// `dev`: `⌈threads / N⌉` threadgroups spread over `x` up to its limit, then `y`, then `z`.
/// `None` where Metal runs nothing: a zero axis of the grid, or a threadgroup with a zero axis or
/// above `FLAT_MAX` threads.
fn flat_groups(dev: &Device, dispatch: &Dispatch, local: [u32; 3]) -> Option<([u32; 3], [u32; 3])> {
    let (threads, size) = match *dispatch {
        Dispatch::Threads {
            threads,
            threads_per_group,
        } => (threads, threads_per_group),
        Dispatch::Threadgroups {
            groups,
            threads_per_group,
        } => (
            std::array::from_fn(|i| groups[i] * threads_per_group[i]),
            threads_per_group,
        ),
        Dispatch::ThreadgroupsIndirect { .. } | Dispatch::ThreadgroupRange { .. } => {
            panic!("a flat command is a whole direct dispatch")
        }
    };
    let volume = |v: [u32; 3]| v.iter().map(|&x| u64::from(x)).product::<u64>();
    if volume(size) == 0 || volume(size) > u64::from(FLAT_MAX) || volume(threads) == 0 {
        return None;
    }
    let max = dev.profile().fields().limits.max_compute_work_group_count;
    let groups = volume(threads).div_ceil(volume(local));
    let x = groups.min(u64::from(max[0]));
    let rest = groups.div_ceil(x);
    let y = rest.min(u64::from(max[1]));
    let z = rest.div_ceil(y);
    assert!(z <= u64::from(max[2]), "a flat grid past the limits");
    let count = |n: u64| u32::try_from(n).expect("a group count");
    Some((threads, [count(x), count(y), count(z)]))
}

/// The flat tests' commands, and their parameters: ragged thread grids and threadgroup grids of
/// threadgroup sizes other than a variant's, an empty command, and threadgroups with a zero axis
/// or above `FLAT_MAX` threads, which run nothing.
fn flat_commands() -> (Vec<CommandRecord>, Vec<u8>) {
    let commands = vec![
        flat_record(
            Action::DispatchThreads {
                grid: [10, 3, 1],
                local_size: [4, 2, 1],
            },
            0,
            0,
        ),
        flat_record(
            Action::DispatchThreadgroups {
                grid: [2, 2, 2],
                local_size: [3, 1, 1],
            },
            4,
            128,
        ),
        CommandRecord::default(),
        flat_record(
            Action::DispatchThreads {
                grid: [5, 1, 1],
                local_size: [32, 32, 2],
            },
            8,
            400,
        ),
        flat_record(
            Action::DispatchThreads {
                grid: [5, 1, 1],
                local_size: [0, 1, 1],
            },
            8,
            400,
        ),
        flat_record(
            Action::DispatchThreads {
                grid: [7, 2, 3],
                local_size: [1, 1, 1],
            },
            12,
            224,
        ),
    ];
    let mut params = vec![0u8; BYTES];
    for (at, w) in [(0, 1000u32), (4, 7), (8, 99), (12, 0x8000_0000)] {
        params[at..at + 4].copy_from_slice(&w.to_le_bytes());
    }
    (commands, params)
}

/// On every profile with device addresses, one flat variant runs every command of its pipeline as
/// Metal runs it: ragged thread grids and threadgroup grids of threadgroup sizes other than its
/// own, each thread once; a threadgroup with a zero axis or above the state's maximum runs nothing
/// and reports nothing.
#[test]
fn a_flat_variant_runs_every_grid_of_its_pipeline() {
    let Some(bc) = assemble("flat", FLAT) else {
        return;
    };
    let (commands, params) = flat_commands();
    let metal = metal_flat(&commands, &params);
    let (mut ran, mut generated) = (0, 0);
    for (stem, dev) in devices() {
        if dev.profile().fields().buffer_access != BufferAccess::Addressed {
            continue;
        }
        ran += 1;
        for path in paths(&dev) {
            let (mem, dispatched) = flat_replay(
                &stem,
                (&dev, &bc, path),
                [8, 1, 1],
                &commands,
                params.clone(),
            );
            assert_eq!(
                dispatched, 3,
                "{stem} {path:?}: one dispatch per running command"
            );
            assert_eq!(words(&mem.out), words(&metal), "{stem} {path:?}");
            let report = mem.report();
            assert!(
                !report.replay_fault && !report.overflow && !report.resident_miss,
                "{stem} {path:?}: {report:?}"
            );
            generated += usize::from(path == Route::Generated);
        }
    }
    assert!(ran > 0, "no profile with device addresses");
    assert!(generated > 0, "no profile states compute pipeline binding");
}

/// Outside a replay, on every profile with device addresses, the flat variant binds as any kernel
/// does, and `instantiate_dispatch` runs each command as one indirect region at its flat group
/// counts, with no clamp prepass, reading its grid from the region record at its
/// `ReplayRegionAddress`: each thread once, as Metal computes it, for threadgroup sizes other than
/// the commands'. A direct `dispatchThreads:` of a flat plan is one region of its own threadgroup
/// size, at group counts that cover the grid, reading the record the plan states; a direct
/// `dispatchThreadgroups:` of one is refused.
#[test]
fn a_flat_variant_dispatches_outside_a_replay() {
    let Some(bc) = assemble("flat", FLAT) else {
        return;
    };
    let (commands, params) = flat_commands();
    let metal = metal_flat(&commands, &params);
    let mut key = VariantKey::new();
    key.flat_grid = true;
    let mut ran = 0;
    for (stem, dev) in devices() {
        if dev.profile().fields().buffer_access != BufferAccess::Addressed {
            continue;
        }
        ran += 1;
        for local in [[8, 1, 1], [4, 2, 2]] {
            let out = host_replay(
                &stem,
                (&dev, &bc, "flat"),
                Run::Flat(local),
                &commands,
                &params,
            );
            assert_eq!(words(&out), words(&metal), "{stem} {local:?}");
        }
        let pipe = pipeline(&dev, &bc, "flat", &key.lowering_inputs(Stage::Kernel))
            .unwrap_or_else(|e| panic!("{stem}: {e}"));
        let (state, _) = command_state(&commands[0]);
        let plan = pipe
            .plan(&dev, &state.shape())
            .unwrap_or_else(|e| panic!("{stem}: {e:?}"));
        let direct = Dispatch::Threads {
            threads: [10, 3, 1],
            threads_per_group: [8, 1, 1],
        };
        let cmds = plan
            .instantiate_dispatch(&state, &direct)
            .unwrap_or_else(|e| panic!("{stem}: {e:?}"));
        assert_eq!(cmds.regions.len(), 1, "{stem}");
        assert_eq!(cmds.regions[0].indirect, None, "{stem}");
        assert!(cmds.indirect_clamp.is_none(), "{stem}");
        // 10 x 3 threads is 30, which 4 threadgroups of 8 cover.
        assert_eq!(cmds.regions[0].group_count, [4, 1, 1], "{stem}");
        assert_eq!(
            cmds.region_record,
            Some(
                crate::lower::dispatch::region_bytes([0; 3], [10, 3, 1], [8, 1, 1])
                    .expect("a packable threadgroup size")
            ),
            "{stem}"
        );
        // Its threadgroups are the shader's, so a `dispatchThreadgroups:` has no flat form.
        let groups = Dispatch::Threadgroups {
            groups: [2, 1, 1],
            threads_per_group: [8, 1, 1],
        };
        match plan.instantiate_dispatch(&state, &groups) {
            Err(BindError::Unsupported(u)) => {
                assert_eq!(u.feature, Feature::GpuEncodedIndirectCommands, "{stem}")
            }
            other => panic!("{stem}: {:?}", other.map(|_| ())),
        }
    }
    assert!(ran > 0, "no profile with device addresses");
}

/// A direct `dispatchThreads:` whose grid is not a whole number of threadgroups runs, on every
/// profile with device addresses, as one Vulkan dispatch of its own threadgroup size on the
/// kernel's flat variant, in place of the interior and ragged-edge regions a split records: the
/// kernel's own compiled SPIR-V, run over the plan's one region and the record the plan states,
/// adds exactly what Metal adds. The kernel accumulates into one slot per grid thread, so a thread
/// run twice, a thread not run, or an invocation past the grid taking a slot all move a word.
#[test]
fn a_direct_ragged_dispatch_on_a_flat_variant_runs_each_grid_thread_once() {
    let Some(bc) = assemble("flat", FLAT) else {
        return;
    };
    // Ragged on two axes (four split regions), ragged with no whole threadgroup along x (one),
    // ragged on all three (eight), and a grid that is a whole number of threadgroups (one), each
    // with its own output window inside the fixture's allocation.
    let commands = vec![
        flat_record(
            Action::DispatchThreads {
                grid: [10, 3, 1],
                local_size: [4, 2, 1],
            },
            0,
            0,
        ),
        flat_record(
            Action::DispatchThreads {
                grid: [5, 1, 1],
                local_size: [8, 1, 1],
            },
            4,
            120,
        ),
        flat_record(
            Action::DispatchThreads {
                grid: [5, 3, 5],
                local_size: [4, 2, 3],
            },
            8,
            144,
        ),
        flat_record(
            Action::DispatchThreads {
                grid: [4, 2, 1],
                local_size: [4, 2, 1],
            },
            12,
            448,
        ),
    ];
    let mut params = vec![0u8; BYTES];
    for (at, w) in [(0, 1000u32), (4, 7), (8, 99), (12, 0x8000_0000)] {
        params[at..at + 4].copy_from_slice(&w.to_le_bytes());
    }
    let metal = metal_flat(&commands, &params);
    let mut ran = 0;
    for (stem, dev) in devices() {
        if dev.profile().fields().buffer_access != BufferAccess::Addressed {
            continue;
        }
        ran += 1;
        let out = host_replay(
            &stem,
            (&dev, &bc, "flat"),
            Run::FlatDirect,
            &commands,
            &params,
        );
        assert_eq!(words(&out), words(&metal), "{stem}");
    }
    assert!(ran > 0, "no profile with device addresses");
}

/// On a device whose group counts are limited to 3 x 2 x 64, a flat dispatch spreads its
/// threadgroups over `x`, then `y`, then `z`, and its invocations past the grid return early; a
/// grid no group count holds, a threadgroup grid above `2^32 - 1` threads on an axis, and a grid of
/// more than `2^64 - 1` threads are the typed `GROUP_COUNT` fault, beside a command that still
/// runs.
#[test]
fn a_flat_grid_spreads_over_the_group_count_limits() {
    let Some(bc) = assemble("flat", FLAT) else {
        return;
    };
    let running = flat_record(
        Action::DispatchThreads {
            grid: [7, 2, 3],
            local_size: [7, 1, 1],
        },
        0,
        0,
    );
    let too_large = [
        // 500 threadgroups of 4: 3 x 2 x 84 past the limit of 64 along z.
        Action::DispatchThreads {
            grid: [2000, 1, 1],
            local_size: [1, 1, 1],
        },
        Action::DispatchThreadgroups {
            grid: [1 << 31, 1, 1],
            local_size: [2, 1, 1],
        },
        Action::DispatchThreads {
            grid: [u32::MAX, u32::MAX, 2],
            local_size: [1, 1, 1],
        },
    ];
    let mut params = vec![0u8; BYTES];
    params[..4].copy_from_slice(&5u32.to_le_bytes());
    let metal = metal_flat(std::slice::from_ref(&running), &params);
    let (mut ran, mut generated) = (0, 0);
    for (stem, dev) in devices_with(|d| d.limits.max_compute_work_group_count = [3, 2, 64]) {
        if dev.profile().fields().buffer_access != BufferAccess::Addressed {
            continue;
        }
        ran += 1;
        for path in paths(&dev) {
            // 42 threads in 11 threadgroups of 4: a dispatch of 3 x 2 x 2.
            let (mem, dispatched) = flat_replay(
                &stem,
                (&dev, &bc, path),
                [4, 1, 1],
                std::slice::from_ref(&running),
                params.clone(),
            );
            assert_eq!(dispatched, 1, "{stem} {path:?}");
            assert_eq!(words(&mem.out), words(&metal), "{stem} {path:?}");
            assert!(!mem.report().replay_fault, "{stem} {path:?}");
            for action in too_large {
                let commands = [running.clone(), flat_record(action, 0, 256)];
                let (mem, dispatched) = flat_replay(
                    &stem,
                    (&dev, &bc, path),
                    [4, 1, 1],
                    &commands,
                    params.clone(),
                );
                assert_eq!(dispatched, 1, "{stem} {path:?} {action:?}");
                assert_eq!(words(&mem.out), words(&metal), "{stem} {path:?} {action:?}");
                let report = mem.report();
                assert!(report.replay_fault, "{stem} {path:?} {action:?}");
                assert_eq!(
                    report.replay_faults,
                    ReplayFaults::GROUP_COUNT,
                    "{stem} {path:?} {action:?}"
                );
            }
            generated += usize::from(path == Route::Generated);
        }
    }
    assert!(ran > 0, "no profile with device addresses");
    assert!(generated > 0, "no profile states compute pipeline binding");
}

/// A kernel that observes its threadgroup partition (the grid template reads its threadgroup's size
/// and position) is refused as a flat variant, with the replay's bindings or without. The flat
/// kernel compiles under `VariantKey::flat_grid` as any other kernel binds, and reads its grid from
/// the region record at `ReplayRegionAddress` (no `DispatchRegion`); it states
/// `RuntimeFact::FlatGrid`, so its plan states the key, and it refuses emulated simdgroups. A
/// profile without device addresses names what it lacks.
#[test]
fn a_flat_grid_is_a_partition_oblivious_variant_s() {
    let (Some(grid), Some(flat)) = (grid_fixture(), assemble("flat", FLAT)) else {
        return;
    };
    let mut key = VariantKey::new();
    key.flat_grid = true;
    let plain = key.lowering_inputs(Stage::Kernel);
    let mut replayed = replay_inputs();
    replayed.flat_grid = true;
    let mut emulated = plain.clone();
    emulated.emulated_simdgroups = true;
    let mut ran = 0;
    for (stem, dev) in devices() {
        let refused = |r: Result<ComputePipeline, Error>| match r {
            Err(Error::Unsupported(u)) => {
                assert_eq!(u.feature, Feature::GpuEncodedIndirectCommands, "{stem}")
            }
            other => panic!("{stem}: {:?}", other.map(|_| ())),
        };
        if dev.profile().fields().buffer_access != BufferAccess::Addressed {
            refused(pipeline(&dev, &flat, "flat", &plain));
            continue;
        }
        ran += 1;
        refused(pipeline(&dev, &grid, GRID, &replayed));
        refused(pipeline(&dev, &grid, GRID, &plain));
        refused(pipeline(&dev, &flat, "flat", &emulated));
        assert!(pipeline(&dev, &flat, "flat", &replayed).is_ok(), "{stem}");
        let pipe = pipeline(&dev, &flat, "flat", &plain).unwrap_or_else(|e| panic!("{stem}: {e}"));
        assert!(
            pipe.dependencies
                .iter()
                .any(|d| d.fact == RuntimeFact::FlatGrid),
            "{stem}"
        );
        let (typed, _) = typed_module_with(&flat, "flat", dev.profile(), &plain)
            .unwrap_or_else(|e| panic!("{stem}: {e}"));
        let fields: Vec<PushField> = typed.push.fields.iter().map(|f| f.field).collect();
        assert!(
            fields.contains(&PushField::ReplayRegionAddress)
                && !fields.contains(&PushField::DispatchRegion)
                && !fields.contains(&PushField::RootTableAddress),
            "{stem}: {fields:?}"
        );
        let (state, _) = command_state(&flat_record(Action::Empty, 0, 0));
        let plan = pipe
            .plan(&dev, &state.shape())
            .unwrap_or_else(|e| panic!("{stem}: {e:?}"));
        assert_eq!(plan.variant, key, "{stem}");
    }
    assert!(ran > 0, "no profile with device addresses");
}

/// Known gap: [`flat_rmw`], the flat kernel's addition as a plain load and store of `b1[p.x + p.y
/// g.x + p.z g.x g.y]`, takes a device lockstep barrier between them. The analysis proves two
/// lanes' words apart only through their index in the simdgroup or the threadgroup, and two lanes
/// of one simdgroup do meet at that index in a grid of more than 2^32 threads, whose 32-bit index
/// wraps (Metal bounds each of the grid's dimensions, not their product): Apple's lockstep then
/// runs both loads before either store. The barrier is the conservative and exact answer; it
/// observes the threadgroup partition, so the kernel has no flat variant and its GPU-encoded
/// commands are refused, where [`FLAT`]'s atomic addition has one.
#[test]
fn a_flat_read_modify_write_of_its_grid_index_has_no_flat_variant() {
    let Some(rmw) = assemble("flat_rmw", &flat_rmw()) else {
        return;
    };
    let mut key = VariantKey::new();
    key.flat_grid = true;
    let plain = key.lowering_inputs(Stage::Kernel);
    let mut ran = 0;
    for (stem, dev) in devices() {
        if dev.profile().fields().buffer_access != BufferAccess::Addressed {
            continue;
        }
        ran += 1;
        match pipeline(&dev, &rmw, "flat", &plain) {
            Err(Error::Unsupported(u)) => {
                assert_eq!(u.feature, Feature::GpuEncodedIndirectCommands, "{stem}");
                assert_eq!(u.needs, Needs::NotRepresentable, "{stem}");
            }
            other => panic!("{stem}: {:?}", other.map(|_| ())),
        }
    }
    assert!(ran > 0, "no profile with device addresses");
}

/// `lanes(b0, p, l)`: each thread stores its index in its simdgroup (`thread_index_in_simdgroup`)
/// at `b0[p]`, `p` its position in the grid: it observes its threadgroup partition through its
/// simdgroups alone, with no subgroup operation (its lanes are counted from its thread's index).
const LANES: &str = r#"define void @lanes(ptr addrspace(1) noundef "air-buffer-no-alias" %b0, i32 noundef %p, i32 noundef %l) {
entry:
  %i = zext i32 %p to i64
  %o = getelementptr inbounds i32, ptr addrspace(1) %b0, i64 %i
  store i32 %l, ptr addrspace(1) %o, align 4
  ret void
}

!air.kernel = !{!0}
!air.version = !{!20}
!air.language_version = !{!21}
!0 = !{ptr @lanes, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"b0"}
!4 = !{i32 1, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"p"}
!5 = !{i32 2, !"air.thread_index_in_simdgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"l"}
!20 = !{i32 2, i32 8, i32 0}
!21 = !{!"Metal", i32 4, i32 0, i32 0}
"#;

/// [`LANES`] with a threadgroup barrier before its store: it observes its partition otherwise.
fn lanes_barrier() -> String {
    LANES
        .replace(
            "  store i32 %l,",
            "  call void @air.wg.barrier(i32 1, i32 1)\n  store i32 %l,",
        )
        .replace(
            "\n!air.kernel",
            "\ndeclare void @air.wg.barrier(i32, i32)\n\n!air.kernel",
        )
}

/// [`LANES`] storing the sum of its simdgroup's lanes (`simd_sum`) after a simdgroup barrier: it
/// observes its partition through its simdgroups alone, by subgroup operations.
fn lanes_sum() -> String {
    LANES
        .replace(
            "  store i32 %l,",
            "  call void @air.simdgroup.barrier(i32 1, i32 4)\n  %s = call i32 \
             @air.simd_sum.u.i32(i32 %l)\n  store i32 %s,",
        )
        .replace(
            "\n!air.kernel",
            "\ndeclare void @air.simdgroup.barrier(i32, i32)\ndeclare i32 \
             @air.simd_sum.u.i32(i32)\n\n!air.kernel",
        )
}

/// A kernel that observes its threadgroup partition through its simdgroups alone has no flat
/// variant of any threadgroups, and has one of the call's (`VariantKey::call_threadgroups`): a
/// lane's index in its simdgroup ([`LANES`]) on every profile with device addresses, and a
/// simdgroup barrier and reduction ([`lanes_sum`]) at least where the kernel's simdgroups are its
/// subgroups (full subgroups: `Direct`, `Cluster32`); elsewhere that barrier would be a workgroup
/// barrier or an emulated simdgroup's, and the variant is refused. The variant states
/// `RuntimeFact::FlatGrid` and `RuntimeFact::CallThreadgroups`. A kernel with a threadgroup
/// barrier has neither variant, and the input is refused without the flat grid and with the
/// replay's bindings.
#[test]
fn a_flat_grid_of_the_call_s_threadgroups_admits_a_kernel_of_simdgroups() {
    let (Some(lanes), Some(barrier), Some(sum)) = (
        assemble("lanes", LANES),
        assemble("lanes_barrier", &lanes_barrier()),
        assemble("lanes_sum", &lanes_sum()),
    ) else {
        return;
    };
    let mut key = VariantKey::new();
    key.flat_grid = true;
    let any = key.lowering_inputs(Stage::Kernel);
    key.call_threadgroups = true;
    let call = key.lowering_inputs(Stage::Kernel);
    let mut alone = LoweringInputs::new();
    alone.call_threadgroups = true;
    let mut replayed = call.clone();
    replayed.replay_bindings = true;
    let stated = |pipe: &ComputePipeline| {
        [RuntimeFact::FlatGrid, RuntimeFact::CallThreadgroups]
            .map(|fact| pipe.dependencies.iter().any(|d| d.fact == fact))
    };
    let (mut ran, mut sums) = (0, 0);
    for (stem, dev) in devices() {
        if dev.profile().fields().buffer_access != BufferAccess::Addressed {
            continue;
        }
        ran += 1;
        let refused = |r: Result<ComputePipeline, Error>, needs: Needs, what: &str| match r {
            Err(Error::Unsupported(u)) => {
                assert_eq!(
                    u.feature,
                    Feature::GpuEncodedIndirectCommands,
                    "{stem} {what}"
                );
                assert_eq!(u.needs, needs, "{stem} {what}");
            }
            other => panic!("{stem} {what}: {:?}", other.map(|_| ())),
        };
        for (bc, what) in [(&lanes, "lanes"), (&sum, "sum")] {
            refused(
                pipeline(&dev, bc, "lanes", &any),
                Needs::NotRepresentable,
                what,
            );
            refused(
                pipeline(&dev, bc, "lanes", &alone),
                Needs::NotImplemented,
                what,
            );
            refused(
                pipeline(&dev, bc, "lanes", &replayed),
                Needs::NotImplemented,
                what,
            );
        }
        let observed = pipeline(&dev, &barrier, "lanes", &call);
        refused(observed, Needs::NotRepresentable, "barrier");
        let pipe = pipeline(&dev, &lanes, "lanes", &call).unwrap_or_else(|e| panic!("{stem}: {e}"));
        assert_eq!(stated(&pipe), [true, true], "{stem}");
        let lib = Library::from_air(&sum).expect("library");
        let f = FunctionDescriptor::new(lib.function("lanes").expect("function"));
        let base = dev
            .compile_function(&f, &LoweringInputs::new(), &Budget::standard())
            .unwrap_or_else(|e| panic!("{stem}: {e}"));
        match pipeline(&dev, &sum, "lanes", &call) {
            Ok(pipe) => {
                assert_eq!(stated(&pipe), [true, true], "{stem}");
                sums += 1;
            }
            Err(e) if base.requires_full_subgroups() => panic!("{stem}: {e}"),
            refusal => refused(refusal, Needs::NotImplemented, "sum"),
        }
    }
    assert!(ran > 0, "no profile with device addresses");
    assert!(sums > 0, "no profile ran the simdgroup sum flat");
}

/// The folded group counts of a threadgroup `dispatch` on `dev`, and its region record: the product
/// of its group counts spread over `x` up to its limit, then `y`, then `z`, and the record of its
/// grid in threads with its threadgroup size packed, as the flat-dispatch module's folded form
/// writes both. `None` where Metal runs nothing (a zero group count).
fn folded_counts(dev: &Device, dispatch: &Dispatch) -> Option<([u32; 3], Vec<u8>)> {
    let Dispatch::Threadgroups {
        groups,
        threads_per_group,
    } = *dispatch
    else {
        panic!("a folded command is a threadgroup dispatch")
    };
    let total: u64 = groups.iter().map(|&g| u64::from(g)).product();
    if total == 0 {
        return None;
    }
    let max = dev.profile().fields().limits.max_compute_work_group_count;
    let x = total.min(u64::from(max[0]));
    let rest = total.div_ceil(x);
    let y = rest.min(u64::from(max[1]));
    let z = rest.div_ceil(y);
    assert!(z <= u64::from(max[2]), "folded groups past the limits");
    let count = |n: u64| u32::try_from(n).expect("a group count");
    let threads = std::array::from_fn(|i| groups[i] * threads_per_group[i]);
    let record = crate::lower::dispatch::region_bytes([0; 3], threads, threads_per_group)
        .expect("a packable threadgroup size")
        .to_vec();
    Some(([count(x), count(y), count(z)], record))
}

/// The folded tests' commands and parameters: threadgroup dispatches of the grid template,
/// three of them past a 3 x 2 x 64 device's `x` and `y`, and an empty command.
fn folded_commands() -> (Vec<CommandRecord>, Vec<u8>) {
    let commands = vec![
        grid_record(
            Action::DispatchThreadgroups {
                grid: [7, 3, 1],
                local_size: [2, 1, 1],
            },
            0,
            0,
        ),
        grid_record(
            Action::DispatchThreadgroups {
                grid: [2, 2, 1],
                local_size: [4, 2, 1],
            },
            8,
            128,
        ),
        CommandRecord::default(),
        grid_record(
            Action::DispatchThreadgroups {
                grid: [5, 1, 3],
                local_size: [1, 1, 1],
            },
            16,
            256,
        ),
        grid_record(
            Action::DispatchThreadgroups {
                grid: [1, 9, 2],
                local_size: [2, 2, 1],
            },
            24,
            384,
        ),
    ];
    let params = grid_params(&[(0, 5, 24), (2, 9, 32), (4, 0x8000_0000, 16), (6, 3, 32)]);
    (commands, params)
}

/// On every profile with device addresses, at its own group-count limits and at 3 x 2 x 64, a
/// folded variant of the grid template (which observes every grid builtin of its threadgroup and
/// its neighbour in threadgroup memory) runs each threadgroup dispatch as one indirect dispatch of
/// the command's threadgroup size at its folded group counts, reading its group counts and grid
/// from the region record at its `ReplayRegionAddress`: each threadgroup once, at Metal's position,
/// as Metal computes it.
#[test]
fn a_folded_variant_runs_every_group_count_as_metal_runs_it() {
    let Some(bc) = grid_fixture() else {
        return;
    };
    let (commands, params) = folded_commands();
    let metal = metal_grid(&commands, &params);
    let mut ran = 0;
    let limited = devices_with(|d| d.limits.max_compute_work_group_count = [3, 2, 64]);
    for (stem, dev) in devices().into_iter().chain(limited) {
        if dev.profile().fields().buffer_access != BufferAccess::Addressed {
            continue;
        }
        ran += 1;
        let out = host_replay(&stem, (&dev, &bc, GRID), Run::Folded, &commands, &params);
        assert_eq!(words(&out), words(&metal), "{stem}");
    }
    assert!(ran > 0, "no profile with device addresses");
}

/// A folded variant of a kernel that observes its partition compiles under
/// `VariantKey::folded_groups` as any other kernel binds, reads its group counts and grid from
/// the region record at `ReplayRegionAddress` (no `DispatchRegion`), and states
/// `RuntimeFact::FoldedGroups`, so its plan states the key; a direct dispatch of its plan is
/// refused. With a flat grid or a replay's bindings it is refused, and a profile without device
/// addresses names what it lacks.
#[test]
fn a_folded_variant_states_its_key_and_runs_only_indirect_threadgroups() {
    let Some(grid) = grid_fixture() else {
        return;
    };
    let mut key = VariantKey::new();
    key.folded_groups = true;
    let plain = key.lowering_inputs(Stage::Kernel);
    let mut flat = plain.clone();
    flat.flat_grid = true;
    let mut replayed = replay_inputs();
    replayed.folded_groups = true;
    let mut ran = 0;
    for (stem, dev) in devices_and_unaddressed() {
        let refused = |r: Result<ComputePipeline, Error>| match r {
            Err(e) if refuses_threadgroup_memory(&e, dev.profile()) => {}
            Err(Error::Unsupported(u)) => {
                assert_eq!(u.feature, Feature::GpuEncodedIndirectCommands, "{stem}")
            }
            other => panic!("{stem}: {:?}", other.map(|_| ())),
        };
        if dev.profile().fields().buffer_access != BufferAccess::Addressed {
            refused(pipeline(&dev, &grid, GRID, &plain));
            continue;
        }
        ran += 1;
        refused(pipeline(&dev, &grid, GRID, &flat));
        refused(pipeline(&dev, &grid, GRID, &replayed));
        let pipe = pipeline(&dev, &grid, GRID, &plain).unwrap_or_else(|e| panic!("{stem}: {e}"));
        assert!(
            pipe.dependencies
                .iter()
                .any(|d| d.fact == RuntimeFact::FoldedGroups),
            "{stem}"
        );
        let (typed, _) = typed_module_with(&grid, GRID, dev.profile(), &plain)
            .unwrap_or_else(|e| panic!("{stem}: {e}"));
        let fields: Vec<PushField> = typed.push.fields.iter().map(|f| f.field).collect();
        assert!(
            fields.contains(&PushField::ReplayRegionAddress)
                && !fields.contains(&PushField::DispatchRegion)
                && !fields.contains(&PushField::RootTableAddress),
            "{stem}: {fields:?}"
        );
        let (commands, _) = folded_commands();
        let (state, _) = command_state(&commands[0]);
        let plan = pipe
            .plan(&dev, &state.shape())
            .unwrap_or_else(|e| panic!("{stem}: {e:?}"));
        assert_eq!(plan.variant, key, "{stem}");
        let direct = Dispatch::Threadgroups {
            groups: [7, 3, 1],
            threads_per_group: [2, 1, 1],
        };
        match plan.instantiate_dispatch(&state, &direct) {
            Err(BindError::Unsupported(u)) => {
                assert_eq!(u.feature, Feature::GpuEncodedIndirectCommands, "{stem}")
            }
            other => panic!("{stem}: {:?}", other.map(|_| ())),
        }
    }
    assert!(ran > 0, "no profile with device addresses");
}
