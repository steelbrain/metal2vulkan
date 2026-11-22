//! The flat and folded compute variants end to end on the CPU interpreter. No device is opened.
//!
//! `icb_replay_grid_template` observes every grid builtin of its region (its threadgroup's size and
//! position, the nominal size, the grid in threadgroups and threads) and its neighbour in
//! threadgroup memory; `host_replay` runs a command list of it as the consumer records each
//! command's own dispatch, and a model of the source computes what Metal does.
//!
//! A kernel that reads only `thread_position_in_grid` and `threads_per_grid` (`flat`) compiles as a
//! flat variant (`LoweringInputs::flat_grid`, `VariantKey::flat_grid`), which binds as any kernel
//! does and reads its grid from the region record at its `ReplayRegionAddress`: a direct
//! `dispatchThreads:` of its plan, ragged or not, is one region that runs each grid thread once, as
//! Metal computes it. A kernel that observes its partition is refused under the input.
//!
//! The grid template, which observes its partition, compiles as a folded variant
//! (`VariantKey::folded_groups`), which states its key; a direct or indirect dispatch of its plan
//! is refused.
//!
//! `llvm-as` assembles the fixture; without it the tests skip visibly.

use std::collections::BTreeMap;
use std::num::NonZeroU64;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::bind::{
    BindState, BoundBuffer, Dispatch, ExecutionStatus, IndirectRef, PushHole, ResourceClass,
    RootTable, StageTable, UsedResource, WriteSource,
};
use crate::budget::Budget;
use crate::cache::VariantKey;
use crate::device::profile::BufferAccess;
use crate::device::requirements::VkFeature;
use crate::device::{Device, DeviceDescription, DevicePolicy};
use crate::error::{BindError, Error, Feature, Needs};
use crate::indirect_commands::{Action, BufferBinding, BufferStage, CommandRecord};
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

const HEADER: &str = "target datalayout = \"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32\"\ntarget triple = \"air64_v28-apple-macosx26.0.0\"\n\n";

/// Device addresses of the consumer's objects.
const STATUS: u64 = 0x0000_0012_0000_0000;
const SINK: u64 = 0x0000_0011_0000_0000;

/// The two resident allocations `(guest, host)`, `BYTES` each: the parameters and the output.
const PARAMS: (u64, u64) = (0x0000_0001_0000_0000, 0x0000_0020_0000_0000);
const OUT: (u64, u64) = (0x0000_0001_0001_0000, 0x0000_0021_0000_0000);
const BYTES: usize = 512;

const TOKEN: u64 = 0x0123_4567_89ab_cdef;
/// The template's threadgroup length: one word per thread of a group.
const LENGTH: u32 = 32;

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
    let dev = Device::from_queries(desc, DevicePolicy::RESIDENT_TABLES).expect("device");
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
                Device::from_queries(desc, DevicePolicy::RESIDENT_TABLES).expect("device"),
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

/// The host replay of `commands` over `entry` of `bc` compiled as a flat variant
/// (`VariantKey::flat_grid`), on `dev`, with the parameters allocation holding `params`: each
/// `dispatchThreads:` command runs as the one direct region `instantiate_dispatch` plans, of the
/// command's own threadgroup size, binding its kernel buffers in their allocations and its
/// threadgroup lengths and reading the record the plan states (`DispatchCommands::region_record`),
/// in order. The output allocation it leaves.
fn host_replay(
    stem: &str,
    (dev, bc, entry): (&Device, &[u8], &str),
    commands: &[CommandRecord],
    params: &[u8],
) -> Vec<u8> {
    const ROOT: u64 = 0x0000_0040_0000_0000;
    const REGION: u64 = 0x0000_0041_0000_0000;
    let mut key = VariantKey::new();
    key.flat_grid = true;
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
        // A direct dispatch on a flat variant keeps its own counts and reads the record the plan
        // states.
        let volume = dispatch
            .threads_per_group()
            .iter()
            .map(|&l| u64::from(l))
            .product::<u64>();
        if volume == 0 || volume > u64::from(FLAT_MAX) {
            continue;
        }
        let (state, slots) = command_state(c);
        let plan = pipe
            .plan(dev, &state.shape())
            .unwrap_or_else(|e| panic!("{stem}: {e:?}"));
        assert_eq!(plan.variant, key, "{stem}");
        let cmds = plan
            .instantiate_dispatch(&state, &dispatch)
            .unwrap_or_else(|e| panic!("{stem}: {dispatch:?}: {e:?}"));
        // A direct dispatch on a flat variant: the record is the plan's, the consumer's to write.
        let record = cmds
            .region_record
            .map_or_else(Vec::new, |bytes| bytes.to_vec());
        assert_eq!(
            cmds.regions.len(),
            1,
            "{stem}: a flat dispatch is one region"
        );
        for region in &cmds.regions {
            assert_eq!(region.indirect, None, "{stem}");
            assert!(
                cmds.region_record.is_some(),
                "{stem}: a direct dispatch on a flat variant states its own record"
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
                    PushHole::ReplayRegionAddress { offset, .. } => (offset, REGION),
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
            let outs = run_both(&m, region.group_count, res);
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

/// A flat-kernel command: `action` with its base word at `params` and its output window at
/// `out` (byte offsets into the two allocations), with the template record's 32-byte length,
/// which the flat variant ignores.
fn flat_record(action: Action, params: u64, out: u64) -> CommandRecord {
    record(action, PARAMS.0 + params, OUT.0 + out)
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

/// Outside a replay, on every profile with device addresses, the flat variant binds as any kernel
/// does: a direct `dispatchThreads:` of a flat plan is one region of its own threadgroup size, at
/// group counts that cover the grid, reading the record the plan states; a direct
/// `dispatchThreadgroups:` of one, and an indirect dispatch, whose record only a device replay's
/// preprocess writes, are refused.
#[test]
fn a_flat_variant_dispatches_outside_a_replay() {
    let Some(bc) = assemble("flat", FLAT) else {
        return;
    };
    let (commands, _) = flat_commands();
    let mut key = VariantKey::new();
    key.flat_grid = true;
    let mut ran = 0;
    for (stem, dev) in devices() {
        if dev.profile().fields().buffer_access != BufferAccess::Addressed {
            continue;
        }
        ran += 1;
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
        // Its threadgroups are the shader's, so a `dispatchThreadgroups:` has no flat form, and
        // only a device replay's preprocess writes an indirect dispatch's record.
        let groups = Dispatch::Threadgroups {
            groups: [2, 1, 1],
            threads_per_group: [8, 1, 1],
        };
        let indirect = Dispatch::ThreadgroupsIndirect {
            indirect: IndirectRef {
                token: 0,
                offset: 0,
            },
            threads_per_group: [8, 1, 1],
        };
        for refused in [groups, indirect] {
            match plan.instantiate_dispatch(&state, &refused) {
                Err(BindError::Unsupported(u)) => {
                    assert_eq!(u.feature, Feature::GpuEncodedIndirectCommands, "{stem}")
                }
                other => panic!("{stem}: {refused:?}: {:?}", other.map(|_| ())),
            }
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
        let out = host_replay(&stem, (&dev, &bc, "flat"), &commands, &params);
        assert_eq!(words(&out), words(&metal), "{stem}");
    }
    assert!(ran > 0, "no profile with device addresses");
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

/// A folded variant of a kernel that observes its partition compiles under
/// `VariantKey::folded_groups` as any other kernel binds, reads its group counts and grid from
/// the region record at `ReplayRegionAddress` (no `DispatchRegion`), and states
/// `RuntimeFact::FoldedGroups`, so its plan states the key; outside a device replay, whose
/// preprocess alone writes its record, a direct or indirect dispatch of its plan is refused. With a flat grid or a replay's bindings it is refused, and a profile without device
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
        let indirect = Dispatch::ThreadgroupsIndirect {
            indirect: IndirectRef {
                token: 0,
                offset: 0,
            },
            threads_per_group: [2, 1, 1],
        };
        for refused in [direct, indirect] {
            match plan.instantiate_dispatch(&state, &refused) {
                Err(BindError::Unsupported(u)) => {
                    assert_eq!(u.feature, Feature::GpuEncodedIndirectCommands, "{stem}")
                }
                other => panic!("{stem}: {refused:?}: {:?}", other.map(|_| ())),
            }
        }
    }
    assert!(ran > 0, "no profile with device addresses");
}
