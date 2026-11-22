//! The compute binder end to end on the CPU interpreter. No device is opened.
//!
//! Each kernel goes through the public compile API (`Library::from_air`,
//! `Device::compile_compute_pipeline`), `ComputePipeline::plan`, and
//! `BindingPlan::instantiate_dispatch`; the harness then runs the pipeline's typed module on the
//! interpreter with exactly what the binder planned: every descriptor write (a buffer slot's
//! descriptor range sliced out of its allocation, the `ZeroBuffer` neutral, the argument table
//! bytes, the status buffer), every region's push bytes (with the root-table and status address
//! holes filled), its specialization, and its group count, region after region. Results are
//! compared with what Metal computes.
//!
//! - Plans are total on every committed profile: one write per layout binding, or a typed error.
//! - `dispatchThreads` edges and group-count splits compute Metal's grid builtins exactly.
//! - Bind offsets reach the right bytes through the range and the residual, with and without robust
//!   descriptors (the in-shader check reads the bound range's length).
//! - `[[threadgroup(n)]]` lengths specialize the arena; a total above
//!   `maxComputeSharedMemorySize` is refused at plan.
//! - Loaded pointers resolve through the binder's root table (resident set from bound buffers and
//!   `used_resources`); a miss is counted in the `ExecutionStatus`; a parameter merged with a
//!   loaded pointer is reached through its root entry, which needs a device address.
//! - An indirect dispatch reads its group counts from Metal's buffer.
//!
//! `llvm-as` assembles the kernels; without it the tests skip visibly.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::bind::{
    BindState, BoundBuffer, Dispatch, DispatchCommands, ExecutionStatus, IndirectArgs, IndirectRef,
    PushHole, ResourceClass, RootTable, StageTable, StatusReport, StatusWord, UsedResource,
    WriteSource,
};
use crate::budget::Budget;
use crate::device::requirements::VkFeature;
use crate::device::{Device, DeviceDescription, DevicePolicy};
use crate::error::{BindError, Feature};
use crate::library::Library;
use crate::pipeline::interface::{BufferForm, SlotInterface};
use crate::pipeline::layout::BindingSource;
use crate::pipeline::sym::{
    PushField, ResourceKind, ResourceSlot, SlotClass, SlotRef, SpecKey, Stage,
};
use crate::pipeline::testing::{engine_dir, llvm_assemble, typed_module, AssembleError};
use crate::pipeline::{
    ComputePipeline, ComputePipelineDescriptor, FunctionDescriptor, LoweringInputs, StageSource,
};
use crate::spirv::interp::{run, Buffer, Config, Order, Resources};
use crate::spirv::ir::TypedModule;
use crate::spirv::testing::committed_profiles;

const HEADER: &str = "target datalayout = \"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32\"\ntarget triple = \"air64_v28-apple-macosx26.0.0\"\n\n";

/// Device addresses of the consumer-owned objects the harness places.
const TABLE: u64 = 0x0000_0010_0000_0000;
const SINK: u64 = 0x0000_0011_0000_0000;
const STATUS: u64 = 0x0000_0012_0000_0000;

pub(super) fn assemble_text(name: &str, text: &str) -> Option<Vec<u8>> {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let dir = std::env::temp_dir().join(format!(
        "m2v-binder-{}-{}-{name}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&dir).expect("scratch");
    let ll = dir.join("k.ll");
    std::fs::write(&ll, text).expect("write");
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

/// Committed profile `stem` as a device, after `tweak` edits its description.
pub(super) fn device(stem: &str, tweak: impl FnOnce(&mut DeviceDescription)) -> Device {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles")
        .join(format!("{stem}.json"));
    let text = std::fs::read_to_string(path).expect("profile");
    let mut desc = DeviceDescription::from_khronos_json(&text, None).expect("description");
    tweak(&mut desc);
    Device::from_queries(desc, DevicePolicy::RESIDENT_TABLES).expect("device")
}

/// Profile F: `runtime-floor` without device addresses, below the runtime's floor, whose buffers
/// are descriptors at its four storage buffers a stage.
fn no_addresses(desc: &mut DeviceDescription) {
    desc.features.remove(&VkFeature::BufferDeviceAddress);
}

fn no_robustness(desc: &mut DeviceDescription) {
    desc.features.remove(&VkFeature::RobustBufferAccess);
    desc.features.remove(&VkFeature::RobustBufferAccess2);
}

fn pipeline(dev: &Device, bc: &[u8], entry: &str) -> Result<ComputePipeline, String> {
    let lib = Library::from_air(bc).map_err(|e| format!("library: {e}"))?;
    let f = lib.function(entry).map_err(|e| format!("{entry}: {e}"))?;
    let d = ComputePipelineDescriptor::new(StageSource::Function(FunctionDescriptor::new(f)));
    dev.compile_compute_pipeline(&d, &Budget::standard())
        .map_err(|e| format!("{entry}: {e}"))
}

fn kslot(index: u32) -> SlotRef {
    SlotRef {
        stage: Stage::Kernel,
        class: SlotClass::Buffer,
        index,
    }
}

/// A bound Metal buffer: its allocation bytes and addresses.
#[derive(Clone)]
pub(super) struct Alloc {
    pub(super) bytes: Vec<u8>,
    offset: u64,
    guest: u64,
    host: Option<u64>,
}

impl Alloc {
    pub(super) fn new(len: usize, offset: u64) -> Alloc {
        Alloc {
            bytes: vec![0; len],
            offset,
            guest: 0,
            host: None,
        }
    }
}

/// What the consumer has: bound buffers by kernel slot, used resources (`(guest, host, bytes)`),
/// and the status buffer.
pub(super) struct World {
    pub(super) buffers: BTreeMap<u32, Alloc>,
    used: Vec<(u64, u64, Vec<u8>)>,
    status: [u8; ExecutionStatus::BYTES],
}

impl World {
    pub(super) fn new(buffers: impl IntoIterator<Item = (u32, Alloc)>) -> World {
        World {
            buffers: buffers.into_iter().collect(),
            used: Vec::new(),
            status: ExecutionStatus::initial(),
        }
    }

    pub(super) fn state(&self) -> BindState {
        let mut t = StageTable::default();
        for (&i, a) in &self.buffers {
            t.buffers.insert(
                i,
                BoundBuffer::Buffer {
                    length: a.bytes.len() as u64,
                    offset: a.offset,
                    gpu_address: a.guest,
                    device_address: a.host,
                    attribute_stride: None,
                },
            );
        }
        let mut s = BindState::default();
        s.stages.insert(Stage::Kernel, t);
        for (i, (guest, host, bytes)) in self.used.iter().enumerate() {
            let mut u = UsedResource::new(ResourceClass::Buffer, i as u32);
            u.gpu_address = Some(*guest);
            u.length = bytes.len() as u64;
            u.device_address = Some(*host);
            s.used_resources.push(u);
        }
        s
    }

    fn report(&self) -> StatusReport {
        ExecutionStatus::decode(&self.status)
    }
}

fn padded(mut v: Vec<u8>) -> Vec<u8> {
    v.resize(v.len().next_multiple_of(16).max(16), 0);
    v
}

/// Where each interpreter buffer's final bytes go.
enum Back {
    Slice { slot: u32, offset: usize },
    Whole { slot: u32 },
    Used(usize),
    Status,
    Nowhere,
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

/// Plan, instantiate, and run `dispatch` of `pipe` (whose typed module is `m`) over `world`.
fn execute(
    dev: &Device,
    pipe: &ComputePipeline,
    m: &TypedModule,
    world: &mut World,
    dispatch: &Dispatch,
) -> Result<DispatchCommands, BindError> {
    execute_with(dev, (pipe, m), world, dispatch, Config::default())
}

/// [`execute`] with the interpreter's subgroup width and barrier rule `cfg`'s.
pub(super) fn execute_with(
    dev: &Device,
    (pipe, m): (&ComputePipeline, &TypedModule),
    world: &mut World,
    dispatch: &Dispatch,
    cfg: Config,
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
    assert_eq!(
        plan.descriptors.len(),
        layout.len(),
        "one write per binding"
    );
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
                    let a = &world.buffers[&slot.index];
                    let all = padded(a.bytes.clone());
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
                WriteSource::ConstantBlob { stage } => (
                    padded({
                        assert_eq!(*stage, Stage::Kernel);
                        pipe.stage.constant_blob.as_deref().expect("blob").to_vec()
                    }),
                    Back::Nowhere,
                ),
                WriteSource::ExecutionStatus => (world.status.to_vec(), Back::Status),
                other => panic!("unexpected write {other:?}"),
            };
            buffers.push(Buffer {
                slot: Some(sym),
                address: None,
                bytes,
            });
            back.push(to);
        }
        for (&i, a) in &world.buffers {
            if let Some(host) = a.host {
                buffers.push(Buffer {
                    slot: None,
                    address: Some(host),
                    bytes: padded(a.bytes.clone()),
                });
                back.push(Back::Whole { slot: i });
            }
        }
        for (i, (_, host, bytes)) in world.used.iter().enumerate() {
            buffers.push(Buffer {
                slot: None,
                address: Some(*host),
                bytes: padded(bytes.clone()),
            });
            back.push(Back::Used(i));
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
        let workgroups = match region.indirect {
            None => region.group_count,
            Some(IndirectArgs::Buffer(_)) => INDIRECT_ARGS,
            Some(other) => panic!("{other:?}"),
        };
        let res = Resources {
            buffers,
            push,
            spec,
        };
        let mut outs = Vec::new();
        for order in [Order::Forward, Order::Reverse] {
            let cfg = Config {
                workgroups,
                order,
                ..cfg
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
                Back::Whole { slot } => {
                    let a = world.buffers.get_mut(&slot).expect("slot");
                    let n = a.bytes.len();
                    a.bytes.copy_from_slice(&bytes[..n]);
                }
                Back::Used(i) => {
                    let n = world.used[i].2.len();
                    world.used[i].2.copy_from_slice(&bytes[..n]);
                }
                Back::Status => world
                    .status
                    .copy_from_slice(&bytes[..ExecutionStatus::BYTES]),
                Back::Nowhere => {}
            }
        }
    }
    Ok(cmds)
}

/// The indirect arguments the harness keeps for `ThreadgroupsIndirect` (token 0).
static INDIRECT_ARGS: [u32; 3] = [3, 1, 1];

pub(super) fn words(bytes: &[u8]) -> Vec<u32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes(*c))
        .collect()
}

/// `out[thread_position_in_grid] = threads_per_grid << 16 | threadgroup_position_in_grid`.
fn grid_ids() -> String {
    format!(
        "{HEADER}define void @grid_ids(ptr addrspace(1) noundef \"air-buffer-no-alias\" %0, i32 noundef %tid, i32 noundef %grp, i32 noundef %tpg) {{\n\
entry:\n\
  %s = shl i32 %tpg, 16\n\
  %v = or i32 %s, %grp\n\
  %i = zext i32 %tid to i64\n\
  %p = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %i\n\
  store i32 %v, ptr addrspace(1) %p, align 4\n\
  ret void\n\
}}\n\n\
!air.kernel = !{{!0}}\n!air.version = !{{!6}}\n!air.language_version = !{{!7}}\n\
!0 = !{{ptr @grid_ids, !1, !2}}\n!1 = !{{}}\n!2 = !{{!3, !4, !5, !8}}\n\
!3 = !{{i32 0, !\"air.buffer\", !\"air.location_index\", i32 0, i32 1, !\"air.read_write\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"out\"}}\n\
!4 = !{{i32 1, !\"air.thread_position_in_grid\", !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"tid\"}}\n\
!5 = !{{i32 2, !\"air.threadgroup_position_in_grid\", !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"grp\"}}\n\
!8 = !{{i32 3, !\"air.threads_per_grid\", !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"tpg\"}}\n\
!6 = !{{i32 2, i32 8, i32 0}}\n!7 = !{{!\"Metal\", i32 4, i32 0, i32 0}}\n"
    )
}

/// Every profile with a compute-capable Vulkan 1.2 floor the M1 milestone names.
const PROFILES: [&str; 3] = ["moltenvk-m3", "lavapipe", "runtime-floor"];

#[test]
fn plans_are_total_on_every_committed_profile() {
    let Some(grid) = assemble_text("grid_ids", &grid_ids()) else {
        return;
    };
    let Some(load) = fixture("buffer_end_byte_load") else {
        return;
    };
    let Some(constant) = fixture("buffer_end_byte_load_constant") else {
        return;
    };
    let kernels = [
        (&grid, "grid_ids"),
        (&load, "buffer_end_byte_load"),
        (&constant, "buffer_end_byte_load_constant"),
    ];
    let (mut planned, mut refused) = (0, 0);
    for (stem, _) in committed_profiles().expect("profiles") {
        let dev = device(&stem, |_| {});
        for (bc, entry) in kernels {
            let Ok(pipe) = pipeline(&dev, bc, entry) else {
                refused += 1;
                continue;
            };
            let bindings: usize = pipe.layout.sets.iter().map(|s| s.bindings.len()).sum();
            let slots: Vec<u32> = pipe.interface.slots.keys().map(|s| s.index).collect();
            let shapes = [
                World::new(slots.iter().map(|&i| (i, Alloc::new(64, 0)))),
                World::new([]),
                World::new(slots.iter().map(|&i| (i, Alloc::new(64, 16)))),
            ];
            for w in &shapes {
                let state = w.state();
                let plan = pipe
                    .plan(&dev, &state.shape())
                    .unwrap_or_else(|e| panic!("{stem} {entry}: {e}"));
                assert_eq!(plan.descriptors.len(), bindings, "{stem} {entry}");
                for d in [
                    Dispatch::Threadgroups {
                        groups: [2, 1, 1],
                        threads_per_group: [4, 1, 1],
                    },
                    Dispatch::Threads {
                        threads: [9, 1, 1],
                        threads_per_group: [4, 1, 1],
                    },
                    Dispatch::ThreadgroupsIndirect {
                        indirect: IndirectRef {
                            token: 0,
                            offset: 0,
                        },
                        threads_per_group: [4, 1, 1],
                    },
                ] {
                    let c = plan
                        .instantiate_dispatch(&state, &d)
                        .unwrap_or_else(|e| panic!("{stem} {entry} {d:?}: {e}"));
                    assert!(!c.regions.is_empty());
                    for r in &c.regions {
                        let push_bytes: usize =
                            r.push_constants.iter().map(|p| p.bytes.len()).sum();
                        let range: u32 = pipe.layout.push_constants.iter().map(|p| p.size).sum();
                        assert_eq!(push_bytes as u32, range, "every push byte is written");
                    }
                }
                planned += 1;
            }
        }
    }
    assert!(planned > 0);
    eprintln!("{planned} plans, {refused} compiles refused");
}

#[test]
fn dispatch_threads_edges_and_group_splits_compute_metal_grid_builtins() {
    let Some(bc) = assemble_text("grid_ids", &grid_ids()) else {
        return;
    };
    for stem in PROFILES {
        // A small group-count limit makes the splits reachable with a small grid.
        let dev = device(stem, |d| d.limits.max_compute_work_group_count = [4, 4, 4]);
        let pipe = pipeline(&dev, &bc, "grid_ids").expect(stem);
        let (_, m) = typed_module(&bc, "grid_ids", dev.profile()).expect("module");
        let expect = |threads: u32, l: u32| -> Vec<u32> {
            (0..threads).map(|t| (threads << 16) | (t / l)).collect()
        };
        // 40 threads in groups of 16: an interior of two groups and an edge of 8 threads.
        let mut w = World::new([(0, Alloc::new(4 * 64, 0))]);
        let d = Dispatch::Threads {
            threads: [40, 1, 1],
            threads_per_group: [16, 1, 1],
        };
        let c = execute(&dev, &pipe, &m, &mut w, &d).expect(stem);
        assert_eq!(c.regions.len(), 2, "{stem}");
        assert_eq!(&words(&w.buffers[&0].bytes)[..40], expect(40, 16), "{stem}");
        assert!(words(&w.buffers[&0].bytes)[40..].iter().all(|&x| x == 0));
        // 10 groups of 4 with at most 4 groups per dispatch: three regions.
        let mut w = World::new([(0, Alloc::new(4 * 64, 0))]);
        let d = Dispatch::Threadgroups {
            groups: [10, 1, 1],
            threads_per_group: [4, 1, 1],
        };
        let c = execute(&dev, &pipe, &m, &mut w, &d).expect(stem);
        assert_eq!(c.regions.len(), 3, "{stem}");
        assert_eq!(&words(&w.buffers[&0].bytes)[..40], expect(40, 4), "{stem}");
        // Indirect: the dispatch reads Metal's 3 groups from its buffer.
        let mut w = World::new([(0, Alloc::new(4 * 64, 0))]);
        let d = Dispatch::ThreadgroupsIndirect {
            indirect: IndirectRef {
                token: 0,
                offset: 0,
            },
            threads_per_group: [4, 1, 1],
        };
        let c = execute(&dev, &pipe, &m, &mut w, &d).expect(stem);
        assert_eq!(
            c.regions[0].indirect,
            Some(IndirectArgs::Buffer(IndirectRef {
                token: 0,
                offset: 0
            })),
            "{stem}"
        );
        let got = words(&w.buffers[&0].bytes);
        let want: Vec<u32> = (0..12).map(|t| (12 << 16) | (t / 4)).collect();
        assert_eq!(
            &got[..12],
            want,
            "{stem}: NumWorkgroups grid of the buffer's counts"
        );
        assert!(got[12..].iter().all(|&x| x == 0));
    }
}

/// A kernel's `seq_cst` fence orders through the command buffer's status: on a device that does
/// not address buffers (Profile F, [`no_addresses`]) the plan binds it
/// (`WriteSource::ExecutionStatus`) and pushes no status address, and where the device addresses
/// buffers (`runtime-floor`, `moltenvk-m3`) it pushes the address and binds nothing. Either way each of the
/// dispatch's threads increments `StatusWord::SeqCstFences` of the one status the command buffer
/// keeps, and the store after the fence lands.
#[test]
fn a_seq_cst_fence_reaches_the_status_by_binding_or_address_as_the_device_allows() {
    let Some(bc) = fixture("dispatch_builtins") else {
        return;
    };
    let same = |_: &mut DeviceDescription| {};
    for (stem, tweak, bound) in [
        (
            "runtime-floor",
            no_addresses as fn(&mut DeviceDescription),
            true,
        ),
        ("runtime-floor", same, false),
        ("moltenvk-m3", same, false),
    ] {
        let dev = device(stem, tweak);
        let stem = &format!("{stem}{}", if bound { " without addresses" } else { "" });
        let pipe = pipeline(&dev, &bc, "device_fence_store").expect(stem);
        let (_, m) = typed_module(&bc, "device_fence_store", dev.profile()).expect("module");
        let mut w = World::new([(0, Alloc::new(16, 0))]);
        let d = Dispatch::Threads {
            threads: [24, 1, 1],
            threads_per_group: [8, 1, 1],
        };
        let c = execute(&dev, &pipe, &m, &mut w, &d).expect(stem);
        let plan = pipe.plan(&dev, &w.state().shape()).expect(stem);
        let writes = plan
            .descriptors
            .iter()
            .filter(|write| write.source == WriteSource::ExecutionStatus)
            .count();
        let holes = c
            .push_holes
            .iter()
            .filter(|hole| matches!(hole, PushHole::ExecutionStatusAddress { .. }))
            .count();
        assert_eq!(
            (writes, holes),
            (usize::from(bound), usize::from(!bound)),
            "{stem}"
        );
        let at = StatusWord::SeqCstFences.byte_offset() as usize;
        let fences = u32::from_le_bytes(w.status[at..at + 4].try_into().expect("a word"));
        assert_eq!(fences, 24, "{stem}: one increment a thread");
        assert_eq!(words(&w.buffers[&0].bytes)[0], 42, "{stem}");
    }
}

/// `in` at a nonzero bind offset in a larger allocation, `out` too: the range and the residual
/// reach exactly Metal's bytes, robust or not.
#[test]
fn bind_offsets_reach_metal_bytes_through_range_and_residual() {
    let Some(load) = fixture("buffer_end_byte_load") else {
        return;
    };
    let Some(constant) = fixture("buffer_end_byte_load_constant") else {
        return;
    };
    let Some(store) = fixture("buffer_end_byte_store") else {
        return;
    };
    let one = Dispatch::Threadgroups {
        groups: [1, 1, 1],
        threads_per_group: [1, 1, 1],
    };
    for stem in ["moltenvk-m3", "lavapipe", "runtime-floor"] {
        for robust in [true, false] {
            let dev = if robust {
                device(stem, |_| {})
            } else {
                device(stem, no_robustness)
            };
            for (bc, entry) in [
                (&load, "buffer_end_byte_load"),
                (&constant, "buffer_end_byte_load_constant"),
            ] {
                let pipe = pipeline(&dev, bc, entry).expect(entry);
                let (_, m) = typed_module(bc, entry, dev.profile()).expect("module");
                for (offset, out_offset) in [(0u64, 0u64), (100, 36), (250, 260), (1, 4)] {
                    let mut input = Alloc::new(offset as usize + 6, offset);
                    for (i, b) in input.bytes.iter_mut().enumerate() {
                        *b = (i as u8).wrapping_mul(7).wrapping_add(3);
                    }
                    let want = u32::from(input.bytes[offset as usize + 5]);
                    let mut w = World::new([
                        (0, input),
                        (1, Alloc::new(out_offset as usize + 4, out_offset)),
                    ]);
                    execute(&dev, &pipe, &m, &mut w, &one)
                        .unwrap_or_else(|e| panic!("{stem} {entry}: {e}"));
                    let out = &w.buffers[&1].bytes[out_offset as usize..];
                    assert_eq!(
                        words(out)[0],
                        want,
                        "{stem} robust={robust} {entry} at {offset}"
                    );
                }
            }
            let pipe = pipeline(&dev, &store, "buffer_end_byte_store").expect("store");
            let (_, m) = typed_module(&store, "buffer_end_byte_store", dev.profile()).expect("m");
            for offset in [0u64, 3, 100] {
                let mut a = Alloc::new(offset as usize + 6, offset);
                a.bytes.iter_mut().for_each(|b| *b = 0x11);
                let mut w = World::new([(0, a)]);
                execute(&dev, &pipe, &m, &mut w, &one).expect("store");
                let b = &w.buffers[&0].bytes;
                let at = offset as usize + 5;
                assert_eq!(b[at], 0xA5, "{stem} robust={robust} at {offset}");
                assert!(
                    b.iter().enumerate().all(|(i, &x)| i == at || x == 0x11),
                    "{stem}: neighbours kept"
                );
            }
        }
    }
}

/// One `[[threadgroup(0)]]` region after a 12-byte static global: the region starts at 16.
fn tg_region() -> String {
    format!(
        "{HEADER}@s = internal addrspace(3) global [3 x i32] undef, align 4\n\n\
define void @tg_region(ptr addrspace(3) %a, ptr addrspace(1) %out, i32 %t) {{\n\
  %pa = getelementptr inbounds i32, ptr addrspace(3) %a, i32 %t\n\
  store i32 %t, ptr addrspace(3) %pa, align 4\n\
  %first = icmp eq i32 %t, 0\n\
  br i1 %first, label %seed, label %sync\n\
seed:\n\
  %ps = getelementptr inbounds [3 x i32], ptr addrspace(3) @s, i32 0, i32 2\n\
  store i32 1000, ptr addrspace(3) %ps, align 4\n\
  br label %sync\n\
sync:\n\
  tail call void @air.wg.barrier(i32 2, i32 1)\n\
  %r = sub i32 63, %t\n\
  %qa = getelementptr inbounds i32, ptr addrspace(3) %a, i32 %r\n\
  %va = load i32, ptr addrspace(3) %qa, align 4\n\
  %qs = getelementptr inbounds [3 x i32], ptr addrspace(3) @s, i32 0, i32 2\n\
  %vs = load i32, ptr addrspace(3) %qs, align 4\n\
  %y = add i32 %va, %vs\n\
  %o = getelementptr inbounds i32, ptr addrspace(1) %out, i32 %t\n\
  store i32 %y, ptr addrspace(1) %o, align 4\n\
  ret void\n\
}}\n\n\
declare void @air.wg.barrier(i32, i32)\n\n\
!air.kernel = !{{!0}}\n!air.version = !{{!6}}\n!air.language_version = !{{!7}}\n\
!6 = !{{i32 2, i32 8, i32 0}}\n!7 = !{{!\"Metal\", i32 4, i32 0, i32 0}}\n\
!0 = !{{ptr @tg_region, !1, !2}}\n!1 = !{{}}\n!2 = !{{!3, !4, !5}}\n\
!3 = !{{i32 0, !\"air.buffer\", !\"air.location_index\", i32 0, i32 1, !\"air.read_write\", !\"air.address_space\", i32 3, !\"air.arg_type_size\", i32 4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"a\"}}\n\
!4 = !{{i32 1, !\"air.buffer\", !\"air.location_index\", i32 0, i32 1, !\"air.read_write\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"out\"}}\n\
!5 = !{{i32 2, !\"air.thread_index_in_threadgroup\", !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"t\"}}\n"
    )
}

#[test]
fn threadgroup_lengths_specialize_the_arena_and_are_checked() {
    let Some(bc) = assemble_text("tg_region", &tg_region()) else {
        return;
    };
    for stem in ["moltenvk-m3", "lavapipe", "runtime-floor"] {
        let dev = device(stem, |_| {});
        let pipe = pipeline(&dev, &bc, "tg_region").expect(stem);
        assert_eq!(
            pipe.metal_info.static_threadgroup_memory_length, 16,
            "{stem}: the regions start at 16"
        );
        let (_, m) = typed_module(&bc, "tg_region", dev.profile()).expect("module");
        let mut w = World::new([(0, Alloc::new(4 * 64, 0))]);
        let state_len = |w: &World, len: u32| {
            let mut s = w.state();
            s.stages
                .get_mut(&Stage::Kernel)
                .expect("kernel")
                .threadgroup_lengths
                .insert(0, len);
            s
        };
        let d = Dispatch::Threadgroups {
            groups: [1, 1, 1],
            threads_per_group: [64, 1, 1],
        };
        // A 250-byte length specializes `TgLength(0)` to Metal's 16-byte rounding, 256.
        let state = state_len(&w, 250);
        let plan = pipe.plan(&dev, &state.shape()).expect("plan");
        let c = plan.instantiate_dispatch(&state, &d).expect("dispatch");
        let tg = pipe
            .stage
            .specialization
            .constants
            .iter()
            .find(|k| k.key == SpecKey::TgLength(0))
            .expect("TgLength(0)");
        let e = c.regions[0]
            .specialization
            .entries
            .iter()
            .find(|e| e.constant_id == tg.id)
            .expect("specialized");
        let at = e.offset as usize;
        let data = &c.regions[0].specialization.data;
        assert_eq!(
            u32::from_le_bytes(data[at..at + 4].try_into().expect("u32")),
            256,
            "{stem}: Metal's 16-byte rounding"
        );
        run_regions(&pipe, &m, &mut w, &c);
        let want: Vec<u32> = (0..64).map(|t| 63 - t + 1000).collect();
        assert_eq!(words(&w.buffers[&0].bytes), want, "{stem}");
        // Above `maxComputeSharedMemorySize`: refused at plan, naming the limit.
        let max = dev.profile().fields().limits.max_compute_shared_memory_size;
        let state = state_len(&w, max - 16 + 1);
        match pipe.plan(&dev, &state.shape()) {
            Err(BindError::Unsupported(u)) => {
                assert_eq!(u.feature, Feature::ThreadgroupMemorySize);
                assert!(format!("{:?}", u.needs).contains("MaxComputeSharedMemorySize"));
            }
            other => panic!("{stem}: {other:?}"),
        }
        let state = state_len(&w, max - 16);
        assert!(
            pipe.plan(&dev, &state.shape()).is_ok(),
            "{stem}: exactly full"
        );
    }
}

/// Run already-instantiated commands of a descriptor-only kernel (no addresses) on `world`.
fn run_regions(pipe: &ComputePipeline, m: &TypedModule, world: &mut World, c: &DispatchCommands) {
    let ids: BTreeMap<u32, SpecKey> = pipe
        .stage
        .specialization
        .constants
        .iter()
        .map(|k| (k.id, k.key))
        .collect();
    for region in &c.regions {
        let mut spec = BTreeMap::new();
        for e in &region.specialization.entries {
            let at = e.offset as usize;
            let v = u32::from_le_bytes(
                region.specialization.data[at..at + 4]
                    .try_into()
                    .expect("u32"),
            );
            spec.insert(ids[&e.constant_id], u64::from(v));
        }
        let mut buffers = Vec::new();
        for r in &c.descriptor_ranges {
            let binding = pipe.layout.sets[r.set as usize]
                .bindings
                .iter()
                .find(|b| b.binding == r.binding)
                .expect("binding");
            let BindingSource::Buffer(slot) = binding.source else {
                panic!("a buffer range for {:?}", binding.source);
            };
            let all = padded(world.buffers[&slot.index].bytes.clone());
            let (o, n) = (r.range.offset as usize, r.range.range as usize);
            buffers.push(Buffer {
                slot: Some(symbol(binding.source)),
                address: None,
                bytes: all[o..o + n].to_vec(),
            });
        }
        let push = region
            .push_constants
            .first()
            .map(|w| {
                let mut p = vec![0u8; w.offset as usize];
                p.extend_from_slice(&w.bytes);
                p
            })
            .unwrap_or_default();
        let cfg = Config {
            workgroups: region.group_count,
            ..Config::default()
        };
        let res = Resources {
            buffers,
            push,
            spec,
        };
        let out = run(m, &cfg, res).expect("run").buffers;
        for (r, bytes) in c.descriptor_ranges.iter().zip(out) {
            let binding = pipe.layout.sets[r.set as usize]
                .bindings
                .iter()
                .find(|b| b.binding == r.binding)
                .expect("binding");
            let BindingSource::Buffer(slot) = binding.source else {
                unreachable!()
            };
            let a = world.buffers.get_mut(&slot.index).expect("slot");
            for (i, b) in bytes.into_iter().enumerate() {
                if let Some(dst) = a.bytes.get_mut(r.range.offset as usize + i) {
                    *dst = b;
                }
            }
        }
    }
}

/// Kernel metadata for `name` with three `device` buffers: 0 (read), 1 (read-write), 2 (read).
fn meta(name: &str) -> String {
    format!(
        "!air.kernel = !{{!9}}\n!air.version = !{{!17}}\n!air.language_version = !{{!18}}\n\
         !9 = !{{ptr @{name}, !10, !11}}\n!10 = !{{}}\n!11 = !{{!12, !13, !14}}\n\
         !12 = !{{i32 0, !\"air.buffer\", !\"air.location_index\", i32 0, i32 1, !\"air.read\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 8, !\"air.arg_type_align_size\", i32 8, !\"air.arg_type_name\", !\"ulong\", !\"air.arg_name\", !\"ptrs\"}}\n\
         !13 = !{{i32 1, !\"air.buffer\", !\"air.location_index\", i32 1, i32 1, !\"air.read_write\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"out\"}}\n\
         !14 = !{{i32 2, !\"air.buffer\", !\"air.location_index\", i32 2, i32 1, !\"air.read\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 8, !\"air.arg_type_align_size\", i32 8, !\"air.arg_type_name\", !\"ulong\", !\"air.arg_name\", !\"params\"}}\n\
         !17 = !{{i32 2, i32 8, i32 0}}\n!18 = !{{!\"Metal\", i32 4, i32 0, i32 0}}\n"
    )
}

fn kernel3(name: &str, body: &str) -> String {
    format!(
        "{HEADER}define void @{name}(ptr addrspace(1) noundef \"air-buffer-no-alias\" %0, ptr addrspace(1) noundef \"air-buffer-no-alias\" %1, ptr addrspace(1) noundef \"air-buffer-no-alias\" %2) {{\nentry:\n{body}  ret void\n}}\n\n{}",
        meta(name)
    )
}

/// The pointer at element `params[0]` of `ptrs`, displaced by the `i64` `params[1]`.
const CHASE: &str = "  %k = load i32, ptr addrspace(1) %2, align 8\n\
  %op = getelementptr i8, ptr addrspace(1) %2, i64 8\n\
  %off = load i64, ptr addrspace(1) %op, align 8\n\
  %pp = getelementptr ptr addrspace(1), ptr addrspace(1) %0, i32 %k\n\
  %p = load ptr addrspace(1), ptr addrspace(1) %pp, align 8\n\
  %q = getelementptr i8, ptr addrspace(1) %p, i64 %off\n";

fn params(k: u32, off: u64) -> Alloc {
    let mut a = Alloc::new(16, 0);
    a.bytes[..4].copy_from_slice(&k.to_le_bytes());
    a.bytes[8..].copy_from_slice(&off.to_le_bytes());
    a
}

fn table(entries: &[u64]) -> Alloc {
    let mut a = Alloc::new(8 * entries.len(), 0);
    for (i, e) in entries.iter().enumerate() {
        a.bytes[8 * i..8 * i + 8].copy_from_slice(&e.to_le_bytes());
    }
    a
}

/// Two resident targets: guest `0x1_0000_0000 + 0x100 k`, host far away.
fn targets(w: &mut World) {
    for k in 0..2u64 {
        let bytes: Vec<u8> = (0..64u8).map(|b| b.wrapping_add(0x40 * k as u8)).collect();
        w.used.push((
            0x1_0000_0000 + 0x100 * k,
            0x20_0000_0000 + 0x1_0000_0000 * k,
            bytes,
        ));
    }
}

#[test]
fn loaded_pointers_resolve_through_the_binders_root_table() {
    let body = format!(
        "{CHASE}  %v = load i32, ptr addrspace(1) %q, align 4\n  store i32 %v, ptr addrspace(1) %1, align 4\n"
    );
    let Some(bc) = assemble_text("dyn_load", &kernel3("dyn_load", &body)) else {
        return;
    };
    let one = Dispatch::Threadgroups {
        groups: [1, 1, 1],
        threads_per_group: [1, 1, 1],
    };
    for stem in ["moltenvk-m3", "lavapipe"] {
        for robust in [true, false] {
            let dev = if robust {
                device(stem, |_| {})
            } else {
                device(stem, no_robustness)
            };
            let pipe = pipeline(&dev, &bc, "dyn_load").expect(stem);
            assert!(pipe.interface.needs_pointer_translation);
            let (_, m) = typed_module(&bc, "dyn_load", dev.profile()).expect("module");
            let ptrs = [0x1_0000_0000u64, 0x1_0000_0100, 0x7_0000_0000];
            for (k, off, want, miss) in [
                (0, 4, u32::from_le_bytes([4, 5, 6, 7]), false),
                (1, 60, u32::from_le_bytes([124, 125, 126, 127]), false),
                (1, 64, 0, false),
                (2, 0, 0, true),
            ] {
                let mut w = World::new([
                    (0, table(&ptrs)),
                    (1, Alloc::new(16, 0)),
                    (2, params(k, off)),
                ]);
                targets(&mut w);
                let c = execute(&dev, &pipe, &m, &mut w, &one)
                    .unwrap_or_else(|e| panic!("{stem}: {e}"));
                let table = c.root_table.expect("a root table");
                let n = u64::from_le_bytes(table.bytes[48..56].try_into().expect("N"));
                assert_eq!(n, 2, "{stem}: the resident set is the two used resources");
                assert_eq!(
                    words(&w.buffers[&1].bytes)[0],
                    want,
                    "{stem} k={k} off={off}"
                );
                let r = w.report();
                assert_eq!(r.resident_miss, miss, "{stem} k={k}");
                assert_eq!(r.resident_misses, u32::from(miss), "{stem} k={k}");
            }
        }
    }
}

#[test]
fn a_parameter_merged_with_a_loaded_pointer_is_reached_through_its_root_entry() {
    let body = format!(
        "{CHASE}  %cp = getelementptr i8, ptr addrspace(1) %2, i64 4\n\
  %c = load i32, ptr addrspace(1) %cp, align 4\n\
  %use_out = icmp ne i32 %c, 0\n\
  %t = select i1 %use_out, ptr addrspace(1) %1, ptr addrspace(1) %q\n\
  store i32 305419896, ptr addrspace(1) %t, align 4\n"
    );
    let Some(bc) = assemble_text("merged", &kernel3("merged", &body)) else {
        return;
    };
    let one = Dispatch::Threadgroups {
        groups: [1, 1, 1],
        threads_per_group: [1, 1, 1],
    };
    for stem in ["moltenvk-m3", "lavapipe"] {
        let dev = device(stem, |_| {});
        let pipe = pipeline(&dev, &bc, "merged").expect(stem);
        let (_, m) = typed_module(&bc, "merged", dev.profile()).expect("module");
        let ptrs = [0x1_0000_0000u64, 0x1_0000_0100];
        let world = |use_out: bool, host: Option<u64>| {
            let mut p = params(1, 8);
            p.bytes[4] = u8::from(use_out);
            let mut out = Alloc::new(40, 24);
            out.guest = 0x3_0000_0000;
            out.host = host;
            let mut w = World::new([(0, table(&ptrs)), (1, out), (2, p)]);
            targets(&mut w);
            w
        };
        // `out` is a device-address slot: without an address the plan refuses.
        let w = world(true, None);
        assert_eq!(
            pipe.plan(&dev, &w.state().shape()),
            Err(BindError::MissingDeviceAddress { slot: kslot(1) }),
            "{stem}"
        );
        // Through its root entry `2 + 1`, at its bind offset.
        let mut w = world(true, Some(0x30_0000_0000));
        let c = execute(&dev, &pipe, &m, &mut w, &one).expect("out");
        let table = c.root_table.expect("table");
        let entry = |f: usize| {
            let at = 3 * 32 + 8 * f;
            u64::from_le_bytes(table.bytes[at..at + 8].try_into().expect("u64"))
        };
        assert_eq!(
            (entry(0), entry(1), entry(2), entry(3)),
            (0x30_0000_0000, 0x3_0000_0000, 48, 24),
            "{stem}: host, guest, padded size, bind offset"
        );
        let out = &w.buffers[&1].bytes;
        assert_eq!(words(&out[24..28])[0], 0x1234_5678, "{stem}");
        assert!(out[..24].iter().chain(&out[28..]).all(|&b| b == 0));
        // Through the loaded pointer into the second target.
        let mut w = world(false, Some(0x30_0000_0000));
        execute(&dev, &pipe, &m, &mut w, &one).expect("target");
        assert_eq!(words(&w.used[1].2[8..12])[0], 0x1234_5678, "{stem}");
        assert!(w.buffers[&1].bytes.iter().all(|&b| b == 0));
    }
}

/// A descriptor-only kernel on the Vulkan 1.2 floor needs no root table, no address holes, and
/// no device addresses.
#[test]
fn descriptor_only_kernels_need_no_root_table_or_addresses() {
    let Some(bc) = assemble_text("grid_ids", &grid_ids()) else {
        return;
    };
    let dev = device("runtime-floor", |_| {});
    let pipe = pipeline(&dev, &bc, "grid_ids").expect("runtime-floor");
    assert!(!pipe.interface.needs_pointer_translation);
    let w = World::new([(0, Alloc::new(64, 0))]);
    let state = w.state();
    let plan = pipe.plan(&dev, &state.shape()).expect("plan");
    assert!(plan
        .push
        .fields
        .iter()
        .all(|f| f.field != PushField::RootTableAddress));
    let c = plan
        .instantiate_dispatch(
            &state,
            &Dispatch::Threadgroups {
                groups: [1, 1, 1],
                threads_per_group: [1, 1, 1],
            },
        )
        .expect("dispatch");
    assert!(c.root_table.is_none() && c.push_holes.is_empty());
    // Stage modules bind no synthetic buffer: every write is a bound Metal buffer.
    for d in &plan.descriptors {
        assert!(
            matches!(d.source, WriteSource::Buffer { .. }),
            "{:?}",
            d.source
        );
    }
}

/// The binder writes each compared slot's `SlotGuestBase` (the bound buffer's Metal address plus
/// its bind offset), so `ptrs == params` holds exactly when the two are bound at one Metal address
/// and `params == null` never holds for a bound buffer; a `set*Bytes` binding has no Metal address,
/// and a pipeline that reads one is refused at plan as `MissingGuestAddress` rather than seeing it
/// compare equal to null.
#[test]
fn compared_slots_get_their_guest_bases_and_inline_bytes_are_refused() {
    let body = "  %e = icmp eq ptr addrspace(1) %0, %2\n\
  %n = icmp eq ptr addrspace(1) %2, null\n\
  %re = zext i1 %e to i32\n\
  store i32 %re, ptr addrspace(1) %1, align 4\n\
  %rn = zext i1 %n to i32\n\
  %o = getelementptr inbounds i32, ptr addrspace(1) %1, i64 1\n\
  store i32 %rn, ptr addrspace(1) %o, align 4\n";
    let Some(bc) = assemble_text("guest_bases", &kernel3("guest_bases", body)) else {
        return;
    };
    let one = Dispatch::Threadgroups {
        groups: [1, 1, 1],
        threads_per_group: [1, 1, 1],
    };
    for stem in ["moltenvk-m3", "runtime-floor"] {
        let dev = device(stem, |_| {});
        let pipe = pipeline(&dev, &bc, "guest_bases").expect(stem);
        let (_, m) = typed_module(&bc, "guest_bases", dev.profile()).expect("module");
        // (guest and bind offset of `ptrs`, of `params`, whether they are one address)
        for (a, b, same) in [
            ((0x1_0000_0000, 0), (0x2_0000_0000, 0), false),
            ((0x1_0000_0000, 16), (0x1_0000_0010, 0), true),
            ((0x1_0000_0000, 0), (0x1_0000_0000, 16), false),
        ] {
            let mut ptrs = Alloc::new(64, a.1);
            ptrs.guest = a.0;
            let mut params = Alloc::new(64, b.1);
            params.guest = b.0;
            let mut w = World::new([(0, ptrs), (1, Alloc::new(16, 0)), (2, params)]);
            execute(&dev, &pipe, &m, &mut w, &one).unwrap_or_else(|e| panic!("{stem}: {e}"));
            assert_eq!(
                words(&w.buffers[&1].bytes)[..2],
                [u32::from(same), 0],
                "{stem}: {a:?} {b:?}"
            );
        }
        let mut state = World::new([(0, Alloc::new(64, 0)), (1, Alloc::new(16, 0))]).state();
        if let Some(t) = state.stages.get_mut(&Stage::Kernel) {
            t.buffers.insert(
                2,
                BoundBuffer::Bytes {
                    len: 16,
                    attribute_stride: None,
                },
            );
        }
        assert_eq!(
            pipe.plan(&dev, &state.shape()).err(),
            Some(BindError::MissingGuestAddress { slot: kslot(2) }),
            "{stem}"
        );
    }
}

/// The `descriptor_limit_uniform_views` fixture (its stem) and its first entry, whose slots are
/// `a`, `b`, `c` (`constant`), `p`, `d`, `out`.
const LIMIT_ENTRY: &str = "descriptor_limit_uniform_views";

/// A world for the fixture: `n` threads, and each read-only array holding `len[k]` words (at
/// least `n`), bound at an offset (`c`'s, 532, is not a multiple of 16); `p = {scale 3, bias
/// 0x5A5A}`. Returns it with the expected `out`.
fn limit_world(n: usize, len: [usize; 4]) -> (World, Vec<u32>) {
    let arr = |seed: u32, words: usize, offset: u64| {
        let mut a = Alloc::new(offset as usize + words * 4, offset);
        for i in 0..words {
            let v = (i as u32).wrapping_mul(2654435761).wrapping_add(seed);
            a.bytes[offset as usize + 4 * i..][..4].copy_from_slice(&v.to_le_bytes());
        }
        a
    };
    let a = arr(1, len[0], 256);
    let b = arr(2, len[1], 0);
    let c = arr(3, len[2], 532);
    let d = arr(4, len[3], 256);
    let mut p = Alloc::new(272, 256);
    p.bytes[256..264].copy_from_slice(&[3, 0, 0, 0, 0x5A, 0x5A, 0, 0]);
    let at = |x: &Alloc, i: usize| words(&x.bytes[x.offset as usize..])[i];
    let want = (0..n)
        .map(|i| {
            at(&a, i)
                .wrapping_add(at(&b, i).wrapping_mul(3))
                .wrapping_add(at(&c, i))
                .wrapping_add(at(&d, i) ^ 0x5A5A)
        })
        .collect();
    let mut w = World::new([
        (0, a),
        (1, b),
        (2, c),
        (3, p),
        (4, d),
        (5, Alloc::new(n * 4, 0)),
    ]);
    // Distinct Metal addresses: nothing aliases.
    for (k, a) in w.buffers.iter_mut() {
        a.guest = 0x10_0000_0000 + u64::from(*k) * 0x1_0000_0000;
    }
    (w, want)
}

fn limit_dispatch(n: usize) -> Dispatch {
    Dispatch::Threadgroups {
        groups: [(n / 32) as u32, 1, 1],
        threads_per_group: [32, 1, 1],
    }
}

fn form(pipe: &ComputePipeline, index: u32) -> BufferForm {
    match pipe.interface.slots.get(&kslot(index)) {
        Some(SlotInterface::Buffer(b)) => b.form,
        other => panic!("slot {index}: {other:?}"),
    }
}

/// The fixture compiled for `dev` under `inputs` (a `VariantKey`'s kernel inputs).
fn limit_pipeline(
    dev: &Device,
    bc: &[u8],
    inputs: &LoweringInputs,
) -> Result<ComputePipeline, crate::error::Error> {
    let lib = Library::from_air(bc).expect("library");
    let f = lib.function(LIMIT_ENTRY).expect("function");
    let compiled =
        dev.compile_function(&FunctionDescriptor::new(f), inputs, &Budget::standard())?;
    dev.compile_compute_pipeline(
        &ComputePipelineDescriptor::new(StageSource::Compiled(compiled)),
        &Budget::standard(),
    )
}

/// On Profile F ([`no_addresses`]: four storage buffers per stage, no device addresses) the
/// fixture's six buffer slots exceed the stage's storage-buffer limit, and no slot can move to a
/// device address: the compile refuses the kernel (`DescriptorCount`). On a profile whose limit
/// holds them every slot is a storage view, and through the binder, on the interpreter, `out` is
/// Metal's.
#[test]
fn a_stage_past_the_storage_buffer_limit_without_device_addresses_is_refused() {
    let Some(bc) = fixture(LIMIT_ENTRY) else {
        return;
    };
    let dev = device("runtime-floor", no_addresses);
    match limit_pipeline(&dev, &bc, &LoweringInputs::new()) {
        Err(crate::error::Error::Unsupported(u)) => {
            assert_eq!(u.feature, Feature::DescriptorCount)
        }
        other => panic!("expected a DescriptorCount refusal, got {:?}", other.err()),
    }
    let n = 64;
    let dev = device("moltenvk-m3", |_| {});
    let pipe = pipeline(&dev, &bc, LIMIT_ENTRY).expect("moltenvk-m3");
    assert!(
        (0..6).all(|i| form(&pipe, i) == BufferForm::Descriptor),
        "every slot a storage view"
    );
    let (_, m) = typed_module(&bc, LIMIT_ENTRY, dev.profile()).expect("module");
    let (mut w, want) = limit_world(n, [n; 4]);
    let state = w.state();
    let plan = pipe.plan(&dev, &state.shape()).expect("plan");
    assert!(plan.variant.is_empty(), "small buffers need no variant");
    execute(&dev, &pipe, &m, &mut w, &limit_dispatch(n)).expect("dispatch");
    assert_eq!(words(&w.buffers[&5].bytes), want);
}
