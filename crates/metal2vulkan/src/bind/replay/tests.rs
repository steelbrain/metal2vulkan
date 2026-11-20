//! The device replay's preprocess: valid on every committed profile with device addresses, refused
//! by name on the others, and exact on the CPU interpreter. A command selects the one variant its
//! pipeline, threadgroup size, threadgroup lengths (clamped and rounded as the plan path takes
//! them) and imageblock size name, its group count lands in that variant's arguments only, and its
//! slot table holds each kernel buffer's resident allocation at the address's offset, from the
//! record or, where the descriptor inherits them, from the parent encoder. A ragged grid and a
//! group count past the device's limit split into the regions, each on the variant of its size with
//! its region record. A command no variant replays dispatches nothing and says why, beside a clean
//! command that still runs. The host side refuses what it cannot set up. One indirect dispatch on a
//! flat variant spreads as the replay's flat commands do, runs nothing where Metal does, and reads
//! no byte past its arguments.

use std::path::PathBuf;

use super::*;
use crate::bind::instantiate::padded;
use crate::bind::state::{BoundBuffer, ResourceClass, UsedResource};
use crate::bind::status::{ExecutionStatus, StatusReport};
use crate::device::{DeviceDescription, DevicePolicy};
use crate::indirect_commands::{Action, BufferBinding, CommandRecord};
use crate::spirv::interp::{run, Buffer, Config, Order, Resources};
use crate::spirv::testing::{assert_valid, int_constants};

/// Device addresses of the harness's objects.
const SCRATCH: u64 = 0x0000_0030_0000_0000;
const RECORDS: u64 = 0x0000_0031_0000_0000;
const STATUS: u64 = 0x0000_0012_0000_0000;
const SINK: u64 = 0x0000_0011_0000_0000;

/// The resident allocations `(guest, host, length)`.
const A: (u64, u64, u64) = (0x0000_0001_0000_1000, 0x0000_0020_0000_0000, 256);
const B: (u64, u64, u64) = (0x0000_0001_0000_8000, 0x0000_0021_0000_0000, 60);
/// An allocation outside the resident set `(guest, host, length)`.
const C: (u64, u64, u64) = (0x0000_0001_0004_0000, 0x0000_0022_0000_0000, 96);
/// The device address of the consumer's upload of inline bytes the commands inherit.
const BYTES: u64 = 0x0000_0023_0000_0040;

const TOKEN: u64 = 0x0123_4567_89ab_cdef;
const OTHER: u64 = 77;

fn token(t: u64) -> NonZeroU64 {
    NonZeroU64::new(t).expect("token")
}

/// Every committed profile as a device, by stem, then Profile F's.
fn devices() -> Vec<(String, Device)> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../metal2vulkan-validation/profiles");
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
            let desc = DeviceDescription::from_khronos_json(&text, None).expect("description");
            let dev = Device::from_queries(desc, DevicePolicy::default()).expect("device");
            (stem, dev)
        })
        .collect();
    out.sort_by(|a, b| a.0.cmp(&b.0));
    // Profile F, below the runtime's floor: the form of a device without device addresses.
    let f = crate::pipeline::testing::device_f(DevicePolicy::default()).expect("Profile F");
    out.push((crate::pipeline::testing::PROFILE_F.to_string(), f));
    out
}

fn addressed(dev: &Device) -> bool {
    dev.profile().fields().buffer_access == BufferAccess::Addressed
}

/// A device with device addresses.
fn device() -> Device {
    devices()
        .into_iter()
        .map(|(_, d)| d)
        .find(addressed)
        .expect("an addressed profile")
}

/// The resident set: `A` and `B`, made resident by `useResource`.
fn state() -> BindState {
    let mut s = BindState::default();
    for (i, (guest, host, length)) in [A, B].into_iter().enumerate() {
        let mut u = UsedResource::new(ResourceClass::Buffer, i as u32);
        u.gpu_address = Some(guest);
        u.device_address = Some(host);
        u.length = length;
        s.used_resources.push(u);
    }
    s
}

fn lengths(first: u32) -> [u32; THREADGROUP_COUNT] {
    let mut l = [0; THREADGROUP_COUNT];
    l[0] = first;
    l
}

fn variant(t: u64, local: [u32; 3], first_length: u32) -> ReplayVariant {
    ReplayVariant::new(token(t), local, lengths(first_length))
}

fn bind(address: u64) -> Option<BufferBinding> {
    Some(BufferBinding {
        address,
        attribute_stride: None,
    })
}

/// A record of `action` with pipeline `t`, kernel buffers `buffers` (`(index, address)`, an
/// address below 2^32 a nil buffer's offset), and threadgroup length 0 `first_length`.
fn record(action: Action, t: Option<u64>, buffers: &[(usize, u64)], first_length: u32) -> Vec<u8> {
    let mut r = CommandRecord {
        action,
        ..CommandRecord::default()
    };
    r.bindings.pipeline = t.map(token);
    for &(i, address) in buffers {
        r.bindings.buffers_mut(BufferStage::Kernel)[i] = bind(address);
    }
    r.threadgroup_memory_lengths = lengths(first_length);
    let mut bytes = vec![0u8; COMMAND_BYTES];
    r.encode_into(&mut bytes).expect("record");
    bytes
}

fn groups(grid: [u32; 3], local: [u32; 3]) -> Action {
    Action::DispatchThreadgroups {
        grid,
        local_size: local,
    }
}

fn threads(grid: [u32; 3], local: [u32; 3]) -> Action {
    Action::DispatchThreads {
        grid,
        local_size: local,
    }
}

/// What one preprocess left: the plan, the scratch, and the status; and what it took: its
/// lane-steps, and each invocation's loop-header visits.
struct Ran {
    pre: ReplayPreprocess,
    scratch: Vec<u8>,
    status: StatusReport,
    steps: u64,
    visits: Vec<u64>,
}

impl Ran {
    /// The arguments of the first region of command `slot` for `variant`.
    fn args(&self, slot: u32, variant: u32) -> [u32; 3] {
        self.region_args(slot, 0, variant)
    }

    fn region_args(&self, slot: u32, region: u32, variant: u32) -> [u32; 3] {
        let at = self.pre.layout.argument_offset(slot, region, variant) as usize;
        std::array::from_fn(|i| word(&self.scratch, at + 4 * i))
    }

    /// The record of `region` of command `slot`: the `DispatchRegion` words.
    fn region(&self, slot: u32, region: u32) -> [u32; 8] {
        let at = self.pre.layout.region_offset(slot, region) as usize;
        std::array::from_fn(|i| word(&self.scratch, at + 4 * i))
    }

    /// Every argument of command `slot` that dispatches: `(region, variant, groups)`.
    fn dispatched(&self, slot: u32) -> Vec<(u32, u32, [u32; 3])> {
        let l = self.pre.layout;
        let mut out = Vec::new();
        for r in 0..l.regions {
            for v in 0..l.variants {
                let a = self.region_args(slot, r, v);
                if a != [0; 3] {
                    out.push((r, v, a));
                }
            }
        }
        out
    }

    /// The generated-commands path's layout.
    fn generated(&self) -> GeneratedLayout {
        self.pre
            .layout
            .generated
            .expect("the generated-commands path")
    }

    /// The sequence of `region` of command `slot`: its execution-set index, push bytes, and
    /// group count.
    fn sequence(&self, slot: u32, region: u32) -> (u32, Vec<u8>, [u32; 3]) {
        let g = self.generated();
        let at = g.sequence_offset(slot, region) as usize;
        let push = at + GeneratedLayout::PUSH_OFFSET as usize;
        let dispatch = at + g.dispatch_offset() as usize;
        (
            word(&self.scratch, at),
            self.scratch[push..push + g.push_bytes as usize].to_vec(),
            std::array::from_fn(|i| word(&self.scratch, dispatch + 4 * i)),
        )
    }

    /// The sequence count of the execution recorded at command `slot`.
    fn count(&self, slot: u32) -> u32 {
        word(&self.scratch, self.generated().count_offset(slot) as usize)
    }

    /// Entry `root` of the slot table of `slot`: `[host, guest, size, bind_offset]`.
    fn entry(&self, slot: u32, root: u64) -> [u64; 4] {
        let at = (self.pre.layout.table_offset(slot) + root * roots::ENTRY_BYTES) as usize;
        std::array::from_fn(|i| {
            u64::from_le_bytes(
                self.scratch[at + 8 * i..at + 8 * i + 8]
                    .try_into()
                    .expect("u64"),
            )
        })
    }
}

fn word(bytes: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(bytes[at..at + 4].try_into().expect("word"))
}

/// Run the preprocess of `execution` over `variants` on `dev` against `records` (each
/// `COMMAND_BYTES`, after a header whose fault word is `fault`), in both scheduling orders.
fn preprocess(
    dev: &Device,
    execution: &ReplayExecution,
    variants: &[ReplayVariant],
    records: &[Vec<u8>],
    fault: u32,
) -> Ran {
    preprocess_storage(
        dev,
        execution,
        variants,
        records,
        (records.len() as u32, fault),
    )
}

/// [`preprocess`] with the header stating `capacity` commands and the fault word `fault`.
fn preprocess_storage(
    dev: &Device,
    execution: &ReplayExecution,
    variants: &[ReplayVariant],
    records: &[Vec<u8>],
    storage: (u32, u32),
) -> Ran {
    let limits = dev.profile().fields().limits.max_compute_work_group_count;
    preprocess_with_limits(dev, limits, execution, variants, records, storage)
}

/// [`preprocess_storage`] with the module built for the group-count limits `limits`.
fn preprocess_with_limits(
    dev: &Device,
    limits: [u32; 3],
    execution: &ReplayExecution,
    variants: &[ReplayVariant],
    records: &[Vec<u8>],
    storage: (u32, u32),
) -> Ran {
    run_preprocess(dev, limits, (execution, variants), records, storage, None)
}

/// [`preprocess_with_limits`], on the generated-commands path `planned` describes where it is
/// given.
fn run_preprocess(
    dev: &Device,
    limits: [u32; 3],
    (execution, variants): (&ReplayExecution, &[ReplayVariant]),
    records: &[Vec<u8>],
    (capacity, fault): (u32, u32),
    planned: Option<generated::Planned>,
) -> Ran {
    let storage = (capacity, fault, BINDING_COUNT as u32);
    run_preprocess_counted(
        dev,
        limits,
        (execution, variants),
        records,
        storage,
        planned,
    )
}

/// [`run_preprocess`] with the header's packed binding counts `counts` (`layout::COUNTS`: the
/// kernel buffers in byte 0); [`run_preprocess`]'s count every kernel binding.
fn run_preprocess_counted(
    dev: &Device,
    limits: [u32; 3],
    (execution, variants): (&ReplayExecution, &[ReplayVariant]),
    records: &[Vec<u8>],
    (capacity, fault, counts): (u32, u32, u32),
    planned: Option<generated::Planned>,
) -> Ran {
    let pre = match planned {
        None => replay_preprocess(dev, &state(), execution, variants),
        Some(p) => super::preprocess(dev, &state(), execution, variants, Some(&|| Ok(p.clone()))),
    }
    .expect("preprocess");
    let mut storage = vec![0u8; COMMAND_BUFFER_HEADER_BYTES];
    storage[..4].copy_from_slice(&capacity.to_le_bytes());
    storage[4..8].copy_from_slice(&fault.to_le_bytes());
    storage[12..16].copy_from_slice(&counts.to_le_bytes());
    for r in records {
        storage.extend_from_slice(r);
    }
    let l = pre.layout;
    let mut scratch = vec![0xEE; l.bytes as usize];
    let mut put = |at: u64, bytes: &[u8]| {
        scratch[at as usize..at as usize + bytes.len()].copy_from_slice(bytes);
    };
    put(l.params_offset(), &pre.params);
    let mut resident = pre.resident.bytes.clone();
    let sink = RootTable::SINK_HOST_OFFSET as usize;
    resident[sink..sink + 8].copy_from_slice(&SINK.to_le_bytes());
    put(l.resident_offset, &resident);
    put(l.extents_offset, &pre.extents);
    put(l.variants_offset, &pre.variants);
    put(l.parent_offset, &pre.parent);
    for bytes in &pre.inherited_bytes {
        put(bytes.offset, &BYTES.to_le_bytes());
    }
    if let (Some(g), Some(replay)) = (l.generated, &pre.generated) {
        put(g.templates_offset, &replay.templates);
    }
    let mut push = pre.push_constants.clone();
    for hole in &pre.push_holes {
        let (offset, address) = match *hole {
            PushHole::RootTableAddress { offset, .. } => (offset, SCRATCH + l.resident_offset),
            PushHole::ExecutionStatusAddress { offset, .. } => (offset, STATUS),
            PushHole::CommandRecordsAddress { offset, .. } => (offset, RECORDS),
            PushHole::ReplayRegionAddress { .. } => panic!("the preprocess reads no region"),
            PushHole::CaptureIndexAddress { .. } => panic!("the preprocess reads no index"),
            PushHole::SnapshotPhase { .. } => panic!("the preprocess draws no snapshot"),
            PushHole::MergedDrawAddress { .. } => panic!("the preprocess draws no merged draw"),
        };
        push[offset as usize..offset as usize + 8].copy_from_slice(&address.to_le_bytes());
    }
    let m = typed_module(limits).expect("module");
    let mut outs = Vec::new();
    let mut took = None;
    for order in [Order::Forward, Order::Reverse] {
        let res = Resources {
            buffers: vec![
                Buffer {
                    slot: None,
                    address: Some(SCRATCH),
                    bytes: scratch.clone(),
                },
                Buffer {
                    slot: None,
                    address: Some(RECORDS),
                    bytes: storage.clone(),
                },
                Buffer {
                    slot: None,
                    address: Some(STATUS),
                    bytes: ExecutionStatus::initial().to_vec(),
                },
            ],
            push: push.clone(),
            spec: Default::default(),
        };
        let cfg = Config {
            workgroups: pre.group_count,
            order,
            ..Config::default()
        };
        let outcome = run(&m, &cfg, res).expect("run");
        let visits = outcome
            .invocations
            .iter()
            .map(|i| i.header_visits.iter().sum())
            .collect::<Vec<u64>>();
        took.get_or_insert((outcome.steps, visits));
        outs.push(outcome.buffers);
    }
    assert_eq!(outs[0], outs[1], "the two scheduling orders agree");
    let (steps, visits) = took.expect("a run");
    let mut out = outs.swap_remove(0);
    assert_eq!(out[1], storage, "the records are only read");
    let status: [u8; ExecutionStatus::BYTES] = out[2].as_slice().try_into().expect("status");
    Ran {
        pre,
        scratch: out.swap_remove(0),
        status: ExecutionStatus::decode(&status),
        steps,
        visits,
    }
}

#[test]
fn the_preprocess_module_is_valid_where_device_addresses_are() {
    let mut seen = (0, 0);
    for (stem, dev) in devices() {
        let p = dev.profile();
        let e = Unsupported::needs_any_of(
            Feature::GpuEncodedIndirectCommands,
            vec![p.addressed_needs()],
        );
        if addressed(&dev) {
            seen.0 += 1;
            let limits = p.fields().limits.max_compute_work_group_count;
            assert_valid(typed_module(limits).expect("module"), p);
            let m = replay_preprocess_module(&dev).unwrap_or_else(|e| panic!("{stem}: {e}"));
            assert_eq!(m.limits, limits);
            assert!(p.check(&m.requirements).is_ok(), "{stem}");
            // No descriptor set; the push window covers exactly the three addresses.
            assert!(m.layout.sets.is_empty(), "{stem}");
            assert_eq!(m.layout.push_constants.len(), 1, "{stem}");
            let range = m.layout.push_constants[0];
            assert_eq!(
                (range.stages, range.offset, range.size),
                (
                    crate::vk::ShaderStageFlags::COMPUTE,
                    0,
                    ReplayPreprocessModule::PUSH_BYTES
                ),
                "{stem}"
            );
            assert_eq!(
                ReplayPreprocessModule::RECORDS_PUSH + 8,
                ReplayPreprocessModule::PUSH_BYTES
            );
        } else {
            seen.1 += 1;
            assert_eq!(
                replay_preprocess_module(&dev),
                Err(Error::Unsupported(e.clone())),
                "{stem}"
            );
            let execution = ReplayExecution::new(1, 0, 1);
            let variants = [variant(TOKEN, [8, 1, 1], 0)];
            assert_eq!(
                replay_preprocess(&dev, &state(), &execution, &variants),
                Err(BindError::Unsupported(e)),
                "{stem}"
            );
        }
    }
    assert!(seen.0 > 0 && seen.1 > 0, "{seen:?}");
}

#[test]
fn the_layout_keeps_every_section_apart() {
    let dev = device();
    let variants = [
        variant(TOKEN, [8, 1, 1], 32),
        variant(TOKEN, [4, 2, 1], 0),
        variant(OTHER, [8, 1, 1], 32),
    ];
    let mut execution = ReplayExecution::new(9, 2, 5);
    execution.regions = 2;
    let pre = replay_preprocess(&dev, &state(), &execution, &variants).expect("preprocess");
    let l = pre.layout;
    assert_eq!((l.slots, l.variants, l.regions), (5, 3, 2));
    assert_eq!(l.table_offset(4) + SLOT_TABLE_BYTES, l.params_offset());
    assert_eq!(pre.params.len() as u64, PARAMS_BYTES);
    // The extents right after the resident set, which ends the resident root table.
    assert_eq!(
        l.resident_offset + pre.resident.bytes.len() as u64,
        l.extents_offset
    );
    assert!(l.extents_offset + pre.extents.len() as u64 <= l.variants_offset);
    assert!(l.variants_offset + pre.variants.len() as u64 <= l.parent_offset);
    assert!(l.parent_offset + pre.parent.len() as u64 <= l.regions_offset);
    // Ten region records, then thirty arguments: (command, region, variant), variant fastest.
    assert_eq!(
        l.region_offset(4, 1) + REGION_BYTES,
        l.regions_offset + 10 * REGION_BYTES
    );
    assert!(l.region_offset(4, 1) + REGION_BYTES <= l.arguments_offset);
    assert_eq!(l.argument_offset(1, 0, 0), l.arguments_offset + 6 * 12);
    assert_eq!(l.argument_offset(1, 1, 2), l.arguments_offset + 11 * 12);
    assert_eq!(
        l.argument_offset(4, 1, 2) + ARGUMENT_BYTES,
        l.arguments_offset + 30 * 12
    );
    assert!(l.argument_offset(4, 1, 2) + ARGUMENT_BYTES <= l.bytes);
    for at in [
        l.variants_offset,
        l.parent_offset,
        l.regions_offset,
        l.arguments_offset,
        l.bytes,
    ] {
        assert_eq!(at % SECTION_ALIGN, 0);
    }
    // The resident root table's own entries: the resident set from root 2.
    assert_eq!(l.resident_offset % roots::ENTRY_BYTES, 0);
    assert_eq!(pre.resident.bytes.len() as u64, 4 * roots::ENTRY_BYTES);
    assert_eq!(pre.group_count, [5, 1, 1]);
    assert_eq!(
        &pre.params[..12],
        [2u32, 5, 3].map(u32::to_le_bytes).concat().as_slice()
    );
    // One extent per entry of the set, each allocation's own length (B's 60 bytes, not its padded
    // 64), then the zero a search that found no entry reads.
    let extents: Vec<u64> = pre
        .extents
        .as_chunks::<8>()
        .0
        .iter()
        .map(|c| u64::from_le_bytes(*c))
        .collect();
    assert_eq!(extents, [A.2, B.2, 0]);
    assert_eq!(word(&pre.params, P_REGIONS as usize), 2);
    assert_eq!(
        u64::from(word(&pre.params, P_REGIONS_AT as usize)),
        l.regions_offset - l.resident_offset
    );
}

/// Write into `bytes`, a record's, at binding `slot` of `stage`, what no write of the record's
/// writers puts there: an address no resident allocation holds and a dynamic stride, which the
/// descriptor does not enable.
fn garbage(bytes: &mut [u8], stage: BufferStage, slot: usize) {
    let at = (layout::buffers(stage) + slot as u64 * layout::BINDING) as usize;
    bytes[at..at + 8].copy_from_slice(&0xABAB_ABAB_ABAB_ABABu64.to_le_bytes());
    bytes[at + 8..at + 16].copy_from_slice(&7u64.to_le_bytes());
}

/// A command reads the kernel bindings below the header's count alone: of a descriptor of 2
/// kernel buffers, 2 of the record's 128 bindings. A binding past the count, or of another stage,
/// is zero in every compute record (the storage is made zero, an encoder's write past the count
/// faults and writes nothing, the CPU's setter refuses one, a compute command has no other
/// stage's setter, a copy takes a source of the same descriptor), so a record with garbage in
/// every such binding dispatches as one without, its slots null and no miss counted: those bytes
/// are not read. The negative control: the same garbage in a counted binding refuses the command.
#[test]
fn a_command_reads_only_the_kernel_bindings_its_descriptor_counts() {
    let dev = device();
    let variants = [variant(TOKEN, [8, 1, 1], 32)];
    let execution = ReplayExecution::new(1, 0, 1);
    let limits = dev.profile().fields().limits.max_compute_work_group_count;
    let run = |record: Vec<u8>| {
        let ev = (&execution, &variants[..]);
        run_preprocess_counted(&dev, limits, ev, &[record], (1, 0, 2), None)
    };
    let dispatched = record(
        groups([3, 2, 1], [8, 1, 1]),
        Some(TOKEN),
        &[(0, A.0), (1, B.0)],
        32,
    );
    let mut past = dispatched.clone();
    for stage in [
        BufferStage::Vertex,
        BufferStage::Fragment,
        BufferStage::Object,
    ] {
        for slot in 0..BINDING_COUNT {
            garbage(&mut past, stage, slot);
        }
    }
    for slot in 2..BINDING_COUNT {
        garbage(&mut past, BufferStage::Kernel, slot);
    }
    let ran = run(past);
    assert_eq!(ran.status.replay_faults, ReplayFaults::default());
    assert!(!ran.status.resident_miss);
    assert_eq!(ran.args(0, 0), [3, 2, 1]);
    for slot in 2..BINDING_COUNT as u64 {
        assert_eq!(ran.entry(0, 2 + slot), [0; 4], "kernel {slot}");
    }
    let mut counted = dispatched;
    garbage(&mut counted, BufferStage::Kernel, 1);
    let ran = run(counted);
    assert_eq!(ran.status.replay_faults, ReplayFaults::BUFFERS);
    assert_eq!(ran.args(0, 0), [0; 3]);
}

#[test]
fn a_dispatch_selects_its_variant_and_resolves_its_buffers() {
    let dev = device();
    let variants = [variant(TOKEN, [8, 1, 1], 32), variant(TOKEN, [8, 2, 1], 0)];
    let records = [
        record(
            groups([3, 2, 1], [8, 1, 1]),
            Some(TOKEN),
            &[(0, A.0 + 16), (3, B.0), (5, 0x30)],
            32,
        ),
        record(Action::Empty, None, &[(0, A.0)], 0),
        record(
            threads([16, 4, 1], [8, 2, 1]),
            Some(TOKEN),
            &[(31, B.0 + 59)],
            0,
        ),
    ];
    let ran = preprocess(&dev, &ReplayExecution::new(3, 0, 3), &variants, &records, 0);
    assert!(!ran.status.replay_fault);
    assert_eq!(ran.status.replay_faults, ReplayFaults::default());
    assert_eq!(ran.args(0, 0), [3, 2, 1]);
    assert_eq!(ran.args(0, 1), [0; 3]);
    assert_eq!(ran.args(1, 0), [0; 3]);
    assert_eq!(ran.args(1, 1), [0; 3]);
    assert_eq!(ran.args(2, 0), [0; 3]);
    assert_eq!(ran.args(2, 1), [2, 2, 1]);
    // Each slot table: the null entry, the sink naming the resident set, the bound buffers, and
    // a nil buffer's offset as its bind offset alone, the address its kernel sees.
    let resident_first = ran.pre.layout.resident_offset + 2 * roots::ENTRY_BYTES;
    for s in 0..3 {
        assert_eq!(ran.entry(s, roots::NULL_ROOT), [0; 4]);
        let r0 = (resident_first - ran.pre.layout.table_offset(s)) / roots::ENTRY_BYTES;
        assert_eq!(ran.entry(s, roots::SINK_ROOT), [SINK, r0, 2, 0]);
    }
    let a = [A.1, A.0, padded(A.2), 16];
    let b = |offset| [B.1, B.0, padded(B.2), offset];
    for n in 0..BINDING_COUNT as u64 {
        let expected = match n {
            0 => a,
            3 => b(0),
            5 => [0, 0, 0, 0x30],
            _ => [0; 4],
        };
        assert_eq!(ran.entry(0, 2 + n), expected, "command 0, buffer {n}");
        // An empty command binds nothing.
        assert_eq!(ran.entry(1, 2 + n), [0; 4], "command 1, buffer {n}");
        let expected = if n == 31 { b(59) } else { [0; 4] };
        assert_eq!(ran.entry(2, 2 + n), expected, "command 2, buffer {n}");
    }
}

/// Commands that inherit the encoder's state run its pipeline and bind its kernel buffers as the
/// host wrote their entries once, resolving nothing: a resident buffer at its offset, a buffer
/// outside the resident set (one the encoder alone names) at its offset, inline bytes at the
/// consumer's upload of them (bind offset 0), and an unbound slot as a buffer never set; no miss
/// is counted.
#[test]
fn inherited_state_comes_from_the_parent() {
    let dev = device();
    let mut execution = ReplayExecution::new(2, 0, 2);
    execution.inheritance = Inheritance {
        pipeline: StateSource::Parent,
        buffers: StateSource::Parent,
    };
    execution.parent_pipeline = Some(token(TOKEN));
    let buffer = |(guest, host, length): (u64, u64, u64), offset| BoundBuffer::Buffer {
        length,
        offset,
        gpu_address: guest,
        device_address: Some(host),
        attribute_stride: None,
    };
    let kernel = &mut execution.parent_buffers;
    kernel.insert(1, buffer(B, 32));
    kernel.insert(
        4,
        BoundBuffer::Bytes {
            len: 20,
            attribute_stride: None,
        },
    );
    kernel.insert(6, buffer(C, 8));
    let records = [
        record(groups([1, 1, 1], [4, 1, 1]), None, &[], 0),
        record(threads([8, 1, 1], [4, 1, 1]), None, &[], 0),
    ];
    let variants = [variant(TOKEN, [4, 1, 1], 0)];
    let ran = preprocess(&dev, &execution, &variants, &records, 0);
    assert!(!ran.status.replay_fault, "{:?}", ran.status);
    assert!(!ran.status.resident_miss, "{:?}", ran.status);
    let entries = ran.pre.layout.parent_offset + COMMAND_BYTES as u64;
    assert_eq!(
        ran.pre.inherited_bytes,
        [InheritedBytes {
            stage: Stage::Kernel,
            index: 4,
            offset: entries + 4 * roots::ENTRY_BYTES + roots::HOST,
        }]
    );
    assert_eq!(ran.args(0, 0), [1, 1, 1]);
    assert_eq!(ran.args(1, 0), [2, 1, 1]);
    for s in 0..2 {
        assert_eq!(ran.entry(s, 2 + 1), [B.1, B.0, padded(B.2), 32]);
        assert_eq!(ran.entry(s, 2 + 4), [BYTES, 0, padded(20), 0]);
        assert_eq!(ran.entry(s, 2 + 6), [C.1, C.0, padded(C.2), 8]);
        for n in [0, 2, 3, 5, 7, 30] {
            assert_eq!(ran.entry(s, 2 + n), [0; 4], "command {s}, buffer {n}");
        }
    }
}

/// A command's threadgroup lengths select a variant as the plan path takes them, clamped to the
/// device's `max_threadgroup_memory_length` and rounded up to 16 bytes; its stage-in region, which
/// no variant reads, is ignored; its imageblock size selects a variant that states one (an unset
/// size being 1 by 1) and is ignored by one that does not.
#[test]
fn lengths_imageblock_sizes_and_stage_in_regions_select_as_the_plan_path_takes_them() {
    let dev = device();
    let max = dev.metal_properties().max_threadgroup_memory_length;
    assert!(max.is_multiple_of(16) && max >= 64, "{max}");
    let local = [8, 1, 1];
    let run = |variants: &[ReplayVariant], r: CommandRecord| {
        let mut bytes = vec![0u8; COMMAND_BYTES];
        r.encode_into(&mut bytes).expect("record");
        preprocess(&dev, &ReplayExecution::new(1, 0, 1), variants, &[bytes], 0)
    };
    let command = |length: u32| {
        let mut r = CommandRecord {
            action: groups([2, 1, 1], local),
            ..CommandRecord::default()
        };
        r.bindings.pipeline = Some(token(TOKEN));
        r.threadgroup_memory_lengths = lengths(length);
        r
    };
    // Unaligned and over-maximum lengths.
    for (length, registered) in [(20, 32), (32, 32), (1, 16), (max + 1, max), (u32::MAX, max)] {
        let ran = run(&[variant(TOKEN, local, registered)], command(length));
        assert!(!ran.status.replay_fault, "{length}: {:?}", ran.status);
        assert_eq!(ran.args(0, 0), [2, 1, 1], "{length}");
    }
    // A stage-in region, and an imageblock size a variant without one ignores.
    let mut set = command(32);
    set.stage_in_region = [1, 2, 3, 4, 5, 6];
    set.imageblock_size = [16, 8];
    let ran = run(&[variant(TOKEN, local, 32)], set.clone());
    assert!(!ran.status.replay_fault, "{:?}", ran.status);
    assert_eq!(ran.args(0, 0), [2, 1, 1]);
    // A variant that states its imageblock size takes only that size; an unset one is 1 by 1.
    let sized = |size| variant(TOKEN, local, 32).with_imageblock_size(size);
    for (variants, r, runs) in [
        (vec![sized([16, 8])], set.clone(), true),
        (vec![sized([8, 16])], set.clone(), false),
        (vec![sized([1, 1])], command(32), true),
        (vec![sized([16, 8])], command(32), false),
        (vec![sized([8, 16]), sized([16, 8])], set, true),
    ] {
        let ran = run(&variants, r.clone());
        assert_eq!(
            ran.status.replay_fault, !runs,
            "{variants:?}: {:?}",
            ran.status
        );
        if runs {
            let selected = variants.len() as u32 - 1;
            assert_eq!(ran.args(0, selected), [2, 1, 1], "{variants:?}");
        } else {
            assert_eq!(ran.status.replay_faults, ReplayFaults::VARIANT);
        }
    }
}

/// One refused command: its name, how its record (or the execution) differs from a clean one, and
/// the reasons it reports.
struct Refused {
    name: &'static str,
    record: Vec<u8>,
    faults: u32,
}

#[test]
fn a_command_no_variant_replays_dispatches_nothing_and_says_why() {
    let dev = device();
    let limit = dev.profile().fields().limits.max_compute_work_group_count[0];
    let local = [8, 1, 1];
    let variants = [variant(TOKEN, local, 32)];
    let clean = || record(groups([2, 1, 1], local), Some(TOKEN), &[(0, A.0)], 32);
    let poke = |at: usize, value: u32| {
        let mut r = clean();
        r[at..at + 4].copy_from_slice(&value.to_le_bytes());
        r
    };
    let mut cases = vec![
        Refused {
            name: "an unregistered pipeline",
            record: record(groups([2, 1, 1], local), Some(OTHER), &[], 32),
            faults: ReplayFaults::VARIANT.0,
        },
        Refused {
            name: "another threadgroup size",
            record: record(groups([2, 1, 1], [4, 2, 1]), Some(TOKEN), &[], 32),
            faults: ReplayFaults::VARIANT.0,
        },
        Refused {
            name: "another threadgroup length",
            record: record(groups([2, 1, 1], local), Some(TOKEN), &[], 48),
            faults: ReplayFaults::VARIANT.0,
        },
        Refused {
            name: "no pipeline",
            record: record(groups([2, 1, 1], local), None, &[], 32),
            faults: ReplayFaults::PIPELINE.0 | ReplayFaults::VARIANT.0,
        },
        Refused {
            name: "a draw",
            record: record(
                Action::DrawPrimitives {
                    primitive_type: 3,
                    vertex_start: 0,
                    vertex_count: 3,
                    instance_count: 1,
                    base_instance: 0,
                },
                Some(TOKEN),
                &[],
                0,
            ),
            faults: ReplayFaults::ACTION.0,
        },
        // Its edge threadgroup of 2 threads has no registered variant.
        Refused {
            name: "a ragged thread grid without its edge's variant",
            record: record(threads([10, 1, 1], local), Some(TOKEN), &[], 32),
            faults: ReplayFaults::VARIANT.0,
        },
        Refused {
            name: "a threadgroup axis above 1,024",
            record: record(threads([2048, 1, 1], [1025, 1, 1]), Some(TOKEN), &[], 32),
            faults: ReplayFaults::VARIANT.0,
        },
        Refused {
            name: "a dynamic stride",
            record: {
                let mut r = CommandRecord {
                    action: groups([2, 1, 1], local),
                    ..CommandRecord::default()
                };
                r.bindings.pipeline = Some(token(TOKEN));
                r.bindings.buffers_mut(BufferStage::Kernel)[0] = Some(BufferBinding {
                    address: A.0,
                    attribute_stride: Some(16),
                });
                r.threadgroup_memory_lengths = lengths(32);
                let mut bytes = vec![0u8; COMMAND_BYTES];
                r.encode_into(&mut bytes).expect("record");
                bytes
            },
            faults: ReplayFaults::BUFFERS.0,
        },
        Refused {
            name: "a barrier flag of 2",
            record: poke(layout::BARRIER as usize, 2),
            faults: ReplayFaults::RECORD.0,
        },
        Refused {
            name: "an unknown action",
            record: poke(layout::ACTION as usize, 7),
            faults: ReplayFaults::RECORD.0,
        },
        Refused {
            name: "an imageblock size of one side",
            record: poke(layout::IMAGEBLOCK as usize, 1),
            faults: ReplayFaults::RECORD.0,
        },
        Refused {
            name: "a pipeline state word of 4",
            record: poke(layout::STATE as usize, 4),
            faults: ReplayFaults::RECORD.0,
        },
        Refused {
            name: "a retained pipeline flag beside no pipeline",
            record: {
                let mut r = record(Action::Empty, None, &[], 32);
                let at = layout::STATE as usize;
                r[at..at + 4].copy_from_slice(&layout::RETAINED.to_le_bytes());
                r
            },
            faults: ReplayFaults::RECORD.0,
        },
        Refused {
            name: "a kernel's pipeline flag beside no pipeline",
            record: {
                let mut r = record(Action::Empty, None, &[], 32);
                let at = layout::STATE as usize;
                r[at..at + 4].copy_from_slice(&layout::KERNEL_PIPELINE.to_le_bytes());
                r
            },
            faults: ReplayFaults::RECORD.0,
        },
        Refused {
            name: "a threadgroup length that rounds up to another variant's",
            record: poke(layout::THREADGROUP as usize, 33),
            faults: ReplayFaults::VARIANT.0,
        },
    ];
    // 2^29 groups of 8 threads: 2^32 threads; past 8 chunks of the limit as well where it is
    // below 2^26.
    let chunks = (1u32 << 29).div_ceil(limit);
    cases.push(Refused {
        name: "a grid above 2^32 - 1 threads",
        record: record(groups([1 << 29, 1, 1], local), Some(TOKEN), &[], 32),
        faults: ReplayFaults::GROUP_COUNT.0
            | if chunks > MAX_REPLAY_REGIONS {
                ReplayFaults::REGIONS.0
            } else {
                0
            },
    });
    for case in cases {
        let ran = preprocess(
            &dev,
            &ReplayExecution::new(2, 0, 2),
            &variants,
            &[case.record, clean()],
            0,
        );
        let name = case.name;
        assert_eq!(ran.dispatched(0), [], "{name}: dispatches nothing");
        assert_eq!(ran.args(1, 0), [2, 1, 1], "{name}: the clean command runs");
        assert!(ran.status.replay_fault, "{name}");
        assert_eq!(ran.status.replay_faults.0, case.faults, "{name}");
        assert_eq!(ran.entry(1, 2), [A.1, A.0, padded(A.2), 0], "{name}");
    }

    // A pipeline a reset retained runs, as the host's `resolve_bindings` runs it.
    let retained = {
        let mut r = clean();
        let at = layout::STATE as usize;
        r[at..at + 4].copy_from_slice(&layout::RETAINED.to_le_bytes());
        r
    };
    let decoded = CommandRecord::decode(&retained).expect("a retained pipeline decodes");
    assert_eq!(
        (decoded.bindings.pipeline, decoded.retained_pipeline),
        (None, Some(token(TOKEN)))
    );
    let ran = preprocess(
        &dev,
        &ReplayExecution::new(2, 0, 2),
        &variants,
        &[retained, clean()],
        0,
    );
    assert!(!ran.status.replay_fault, "{:?}", ran.status);
    assert_eq!((ran.args(0, 0), ran.args(1, 0)), ([2, 1, 1], [2, 1, 1]));

    // A pipeline a kernel wrote runs, set or retained.
    for word in [
        layout::KERNEL_PIPELINE,
        layout::RETAINED | layout::KERNEL_PIPELINE,
    ] {
        let mut written = clean();
        let at = layout::STATE as usize;
        written[at..at + 4].copy_from_slice(&word.to_le_bytes());
        assert!(CommandRecord::decode(&written).unwrap().kernel_pipeline);
        let ran = preprocess(
            &dev,
            &ReplayExecution::new(2, 0, 2),
            &variants,
            &[written, clean()],
            0,
        );
        assert!(!ran.status.replay_fault, "{word}: {:?}", ran.status);
        assert_eq!((ran.args(0, 0), ran.args(1, 0)), ([2, 1, 1], [2, 1, 1]));
    }

    // The producer's sticky encoding fault refuses every command.
    let ran = preprocess(
        &dev,
        &ReplayExecution::new(2, 0, 2),
        &variants,
        &[clean(), clean()],
        1,
    );
    assert_eq!((ran.args(0, 0), ran.args(1, 0)), ([0; 3], [0; 3]));
    assert_eq!(ran.status.replay_faults, ReplayFaults::ENCODING);

    // State set while inherited: a pipeline, or a buffer of any stage.
    let mut execution = ReplayExecution::new(2, 0, 2);
    execution.inheritance = Inheritance {
        pipeline: StateSource::Parent,
        buffers: StateSource::Parent,
    };
    execution.parent_pipeline = Some(token(TOKEN));
    let inheriting = || record(groups([2, 1, 1], local), None, &[], 32);
    for (name, bad, faults) in [
        (
            "a pipeline set while inherited",
            record(groups([2, 1, 1], local), Some(TOKEN), &[], 32),
            ReplayFaults::PIPELINE.0,
        ),
        (
            "a kernel buffer set while inherited",
            record(groups([2, 1, 1], local), None, &[(5, B.0)], 32),
            ReplayFaults::BUFFERS.0,
        ),
    ] {
        let ran = preprocess(&dev, &execution, &variants, &[bad, inheriting()], 0);
        assert_eq!(ran.args(0, 0), [0; 3], "{name}");
        assert_eq!(ran.args(1, 0), [2, 1, 1], "{name}");
        assert_eq!(ran.status.replay_faults.0, faults, "{name}");
    }
}

/// A kernel buffer whose address no resident allocation holds leaves its slot unbound (as a
/// buffer never set) and counts one resident miss, as the host's step does, and the command still
/// dispatches: an address in no allocation, at an allocation's end, and at or past an
/// allocation's own end inside its padding (B is 60 bytes, padded to 64), each beside a bound
/// buffer. A command that dispatches nothing counts none of its misses.
#[test]
fn an_address_no_allocation_holds_leaves_its_slot_unbound_and_counts_a_miss() {
    let dev = device();
    let local = [8, 1, 1];
    let variants = [variant(TOKEN, local, 32)];
    for (name, missed) in [
        ("an address in no allocation", 0x7_dead_0000),
        ("an address at an allocation's end", A.0 + 256),
        (
            "an address at an allocation's own end, inside its padding",
            B.0 + 60,
        ),
        (
            "an address past an allocation's own end, inside its padding",
            B.0 + 61,
        ),
    ] {
        let records = [
            record(
                groups([2, 1, 1], local),
                Some(TOKEN),
                &[(0, A.0), (2, missed)],
                32,
            ),
            record(Action::Empty, None, &[(2, missed)], 0),
        ];
        let ran = preprocess(&dev, &ReplayExecution::new(2, 0, 2), &variants, &records, 0);
        assert!(!ran.status.replay_fault, "{name}: {:?}", ran.status);
        assert!(ran.status.resident_miss, "{name}");
        assert_eq!(ran.status.resident_misses, 1, "{name}");
        assert_eq!(ran.args(0, 0), [2, 1, 1], "{name}");
        assert_eq!(ran.entry(0, 2), [A.1, A.0, padded(A.2), 0], "{name}");
        assert_eq!(
            ran.entry(0, 4),
            [0; 4],
            "{name}: the missed slot is unbound"
        );
        assert_eq!(ran.args(1, 0), [0; 3], "{name}");
    }
}

/// A range that starts past the first command reads the records from its start: slot `s` is
/// command `first + s`. Command 0, a draw, is outside the range and refuses nothing.
#[test]
fn a_range_reads_its_commands_from_its_start() {
    let dev = device();
    let local = [8, 1, 1];
    let variants = [variant(TOKEN, local, 32)];
    let draw = Action::DrawPrimitives {
        primitive_type: 3,
        vertex_start: 0,
        vertex_count: 3,
        instance_count: 1,
        base_instance: 0,
    };
    let records = [
        record(draw, Some(TOKEN), &[], 0),
        record(groups([3, 1, 1], local), Some(TOKEN), &[(0, A.0 + 16)], 32),
        record(threads([16, 1, 1], local), Some(TOKEN), &[(1, B.0)], 32),
    ];
    let ran = preprocess(&dev, &ReplayExecution::new(3, 1, 2), &variants, &records, 0);
    assert!(!ran.status.replay_fault, "{:?}", ran.status);
    assert_eq!(ran.pre.group_count, [2, 1, 1]);
    assert_eq!(ran.args(0, 0), [3, 1, 1]);
    assert_eq!(ran.args(1, 0), [2, 1, 1]);
    assert_eq!(ran.entry(0, 2), [A.1, A.0, padded(A.2), 16]);
    assert_eq!(ran.entry(0, 3), [0; 4]);
    assert_eq!(ran.entry(1, 2), [0; 4]);
    assert_eq!(ran.entry(1, 3), [B.1, B.0, padded(B.2), 0]);
}

/// A grid with a zero axis runs nothing, as in Metal and the host replay: it dispatches nothing
/// and refuses nothing, whatever its other axes, threadgroup size, or variant.
#[test]
fn an_empty_grid_dispatches_nothing_and_refuses_nothing() {
    let dev = device();
    let limit = dev.profile().fields().limits.max_compute_work_group_count[1];
    let variants = [variant(TOKEN, [8, 4, 1], 0)];
    let clean = || record(groups([2, 1, 1], [8, 4, 1]), Some(TOKEN), &[(0, A.0)], 0);
    for (name, action) in [
        // Ragged on y with a registered size.
        (
            "threads (0, 5, 1) in groups of (8, 4, 1)",
            threads([0, 5, 1], [8, 4, 1]),
        ),
        // An unregistered size, ragged on x.
        (
            "threads (5, 0, 1) in groups of (3, 3, 1)",
            threads([5, 0, 1], [3, 3, 1]),
        ),
        (
            "threadgroups (0, limit + 1, 1)",
            groups([0, limit.saturating_add(1), 1], [8, 4, 1]),
        ),
        (
            "threadgroups (1 << 29, 1, 0)",
            groups([1 << 29, 1, 0], [8, 4, 1]),
        ),
    ] {
        let ran = preprocess(
            &dev,
            &ReplayExecution::new(2, 0, 2),
            &variants,
            &[record(action, Some(TOKEN), &[], 0), clean()],
            0,
        );
        assert!(!ran.status.replay_fault, "{name}: {:?}", ran.status);
        assert_eq!(ran.args(0, 0), [0; 3], "{name}");
        assert_eq!(ran.args(1, 0), [2, 1, 1], "{name}");
    }
}

/// A command whose index is not below the capacity the record storage's header states is not
/// read: it dispatches nothing and reports `RANGE`; the commands below it run.
#[test]
fn a_command_past_the_storages_capacity_is_not_read() {
    let dev = device();
    let local = [8, 1, 1];
    let variants = [variant(TOKEN, local, 32)];
    let clean = || record(groups([2, 1, 1], local), Some(TOKEN), &[(0, A.0)], 32);
    // The execution states 3 commands; the storage's header states 2 and holds 2.
    let ran = preprocess_storage(
        &dev,
        &ReplayExecution::new(3, 1, 2),
        &variants,
        &[clean(), clean()],
        (2, 0),
    );
    assert_eq!(ran.args(0, 0), [2, 1, 1]);
    assert_eq!(ran.args(1, 0), [0; 3]);
    assert!(ran.status.replay_fault);
    assert_eq!(ran.status.replay_faults, ReplayFaults::RANGE);
}

#[test]
fn the_binder_refuses_what_it_cannot_set_up() {
    let dev = device();
    let ok = [variant(TOKEN, [8, 1, 1], 0)];
    let refused = |execution: &ReplayExecution, variants: &[ReplayVariant]| match replay_preprocess(
        &dev,
        &state(),
        execution,
        variants,
    ) {
        Err(BindError::InvalidReplay { reason }) => reason,
        other => panic!("{other:?}"),
    };
    assert_eq!(
        refused(&ReplayExecution::new(4, 3, 2), &ok),
        ReplayRefusal::RangeOutsideCapacity
    );
    assert_eq!(
        refused(&ReplayExecution::new(4, u32::MAX, 2), &ok),
        ReplayRefusal::RangeOutsideCapacity
    );
    assert_eq!(
        refused(
            &ReplayExecution::new(MAX_REPLAY_SLOTS + 1, 0, MAX_REPLAY_SLOTS + 1),
            &ok
        ),
        ReplayRefusal::TooManyCommands {
            count: MAX_REPLAY_SLOTS + 1
        }
    );
    let execution = ReplayExecution::new(1, 0, 1);
    assert_eq!(
        refused(&execution, &[]),
        ReplayRefusal::VariantCount { count: 0 }
    );
    let many: Vec<ReplayVariant> = (1..=u64::from(MAX_REPLAY_VARIANTS) + 1)
        .map(|t| variant(t, [8, 1, 1], 0))
        .collect();
    assert_eq!(
        refused(&execution, &many),
        ReplayRefusal::VariantCount { count: many.len() }
    );
    assert_eq!(
        refused(&execution, &[ok[0], ok[0]]),
        ReplayRefusal::InvalidVariant { index: 1 }
    );
    assert_eq!(
        refused(&execution, &[variant(TOKEN, [8, 0, 1], 0)]),
        ReplayRefusal::InvalidVariant { index: 0 }
    );
    assert_eq!(
        refused(&execution, &[ok[0], variant(TOKEN, [8, 1, 1], 20)]),
        ReplayRefusal::InvalidVariant { index: 1 }
    );
    // An imageblock side of 0 is a size no command selects.
    let sized = variant(TOKEN, [8, 1, 1], 0).with_imageblock_size([0, 8]);
    assert_eq!(
        refused(&execution, &[sized]),
        ReplayRefusal::InvalidVariant { index: 0 }
    );
    // A flat variant is its pipeline's only one, of a maximum of 1 to 1,024 threads, and states no
    // length or imageblock size.
    let flat = ReplayVariant::flat(token(TOKEN), [8, 1, 1], 1024);
    for (variants, index) in [
        (vec![ReplayVariant::flat(token(TOKEN), [8, 1, 1], 0)], 0),
        (vec![ReplayVariant::flat(token(TOKEN), [8, 1, 1], 1025)], 0),
        (vec![flat.with_imageblock_size([8, 8])], 0),
        (vec![ok[0], flat], 1),
        (vec![flat, ok[0]], 1),
        (
            vec![flat, ReplayVariant::flat(token(TOKEN), [16, 1, 1], 1024)],
            1,
        ),
    ] {
        assert_eq!(
            refused(&execution, &variants),
            ReplayRefusal::InvalidVariant { index },
            "{variants:?}"
        );
    }
    let mut lengths = flat;
    lengths.threadgroup_memory_lengths[0] = 16;
    assert_eq!(
        refused(&execution, &[lengths]),
        ReplayRefusal::InvalidVariant { index: 0 }
    );
    let other = ReplayVariant::flat(token(OTHER), [8, 1, 1], 64);
    assert!(replay_preprocess(&dev, &state(), &execution, &[flat, other]).is_ok());
    // An empty range is set up and runs nothing.
    let pre = replay_preprocess(&dev, &state(), &ReplayExecution::new(4, 4, 0), &ok)
        .expect("an empty range");
    assert_eq!(pre.group_count, [0, 1, 1]);
    // Regions per command: 1 to 8, and at most 2^20 recorded dispatches.
    for regions in [0, MAX_REPLAY_REGIONS + 1] {
        let mut e = ReplayExecution::new(1, 0, 1);
        e.regions = regions;
        assert_eq!(
            refused(&e, &ok),
            ReplayRefusal::RegionCount { count: regions }
        );
    }
    let all: Vec<ReplayVariant> = (1..=u64::from(MAX_REPLAY_VARIANTS))
        .map(|t| variant(t, [8, 1, 1], 0))
        .collect();
    let mut e = ReplayExecution::new(MAX_REPLAY_SLOTS, 0, MAX_REPLAY_SLOTS);
    e.regions = 2;
    assert_eq!(
        refused(&e, &all),
        ReplayRefusal::TooManyDispatches {
            dispatches: 2 * MAX_REPLAY_DISPATCHES
        }
    );
    e.regions = 1;
    assert!(replay_preprocess(&dev, &state(), &e, &all).is_ok());
}

/// A thread grid that is not a multiple of its threadgroup size runs the regions: the interior on
/// the command's size and each edge on its remainder's, each region on its own registered variant
/// with its record (its first threadgroup, the command's packed size, and the grid), in the order
/// (`x` outermost). The records are what the planner pushes for the same dispatch.
#[test]
fn a_ragged_grid_runs_each_region_on_the_variant_of_its_size() {
    let dev = device();
    let variants = [
        variant(TOKEN, [8, 4, 1], 0),
        variant(TOKEN, [5, 4, 1], 0),
        variant(TOKEN, [8, 1, 1], 0),
        variant(TOKEN, [5, 1, 1], 0),
    ];
    let clean = || record(groups([2, 1, 1], [8, 4, 1]), Some(TOKEN), &[(0, A.0)], 0);
    let records = [
        record(threads([13, 5, 1], [8, 4, 1]), Some(TOKEN), &[(0, A.0)], 0),
        clean(),
        // Fewer threads than one threadgroup on `x`: only the edge.
        record(threads([5, 4, 1], [8, 4, 1]), Some(TOKEN), &[], 0),
    ];
    let ran = preprocess(&dev, &ReplayExecution::new(3, 0, 3), &variants, &records, 0);
    assert!(!ran.status.replay_fault, "{:?}", ran.status);
    // (region, variant, groups): the interior, the `y` edge, the `x` edge, the corner.
    assert_eq!(
        ran.dispatched(0),
        [
            (0, 0, [1, 1, 1]),
            (1, 2, [1, 1, 1]),
            (2, 1, [1, 1, 1]),
            (3, 3, [1, 1, 1])
        ]
    );
    let nominal = crate::lower::dispatch::pack_nominal([8, 4, 1]).expect("packs");
    for (r, base) in [[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0]]
        .into_iter()
        .enumerate()
    {
        let expected =
            crate::lower::dispatch::region_bytes(base, [13, 5, 1], [8, 4, 1]).expect("bytes");
        let words: Vec<u32> = expected
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| word(c, 0))
            .collect();
        assert_eq!(ran.region(0, r as u32).to_vec(), words, "region {r}");
        assert_eq!(ran.region(0, r as u32)[3], nominal);
    }
    assert_eq!(ran.dispatched(1), [(0, 0, [2, 1, 1])]);
    assert_eq!(
        ran.region(1, 0),
        [0, 0, 0, nominal, 16, 4, 1, 0],
        "a uniform command's one region"
    );
    assert_eq!(ran.dispatched(2), [(0, 1, [1, 1, 1])]);
    assert_eq!(ran.region(2, 0), [0, 0, 0, nominal, 5, 4, 1, 0]);
}

/// A group count past the device's limit splits into chunks of at most the limit per axis, each a
/// region of the command's own variant whose record starts at the chunk's first threadgroup; a
/// thread grid both past the limit and ragged adds its edge. A command of more regions than the
/// execution records dispatches nothing and reports `REGIONS` (with no `OVERFLOW`: nothing is
/// clamped).
#[test]
fn a_group_count_past_the_limit_splits_into_chunks() {
    let dev = device();
    let limits = [2, 2, 1];
    let variants = [variant(TOKEN, [2, 1, 1], 0), variant(TOKEN, [1, 1, 1], 0)];
    let records = [
        record(groups([5, 3, 1], [2, 1, 1]), Some(TOKEN), &[], 0),
        // Five whole threadgroups of 2 on `x`, then one edge thread.
        record(threads([11, 1, 1], [2, 1, 1]), Some(TOKEN), &[], 0),
    ];
    let ran = preprocess_with_limits(
        &dev,
        limits,
        &ReplayExecution::new(2, 0, 2),
        &variants,
        &records,
        (2, 0),
    );
    assert!(!ran.status.replay_fault, "{:?}", ran.status);
    let chunks = [
        ([0, 0], [2, 2]),
        ([0, 2], [2, 1]),
        ([2, 0], [2, 2]),
        ([2, 2], [2, 1]),
        ([4, 0], [1, 2]),
        ([4, 2], [1, 1]),
    ];
    let expected: Vec<(u32, u32, [u32; 3])> = chunks
        .iter()
        .enumerate()
        .map(|(r, (_, n))| (r as u32, 0, [n[0], n[1], 1]))
        .collect();
    assert_eq!(ran.dispatched(0), expected);
    for (r, (base, _)) in chunks.iter().enumerate() {
        assert_eq!(
            ran.region(0, r as u32),
            [base[0], base[1], 0, 1, 10, 3, 1, 0],
            "region {r}"
        );
    }
    assert_eq!(
        ran.dispatched(1),
        [
            (0, 0, [2, 1, 1]),
            (1, 0, [2, 1, 1]),
            (2, 0, [1, 1, 1]),
            (3, 1, [1, 1, 1])
        ]
    );
    assert_eq!(ran.region(1, 2)[..3], [4, 0, 0]);
    assert_eq!(ran.region(1, 3)[..7], [5, 0, 0, 1, 11, 1, 1]);

    // Nine regions, and a ragged grid where the execution records one region per command.
    let mut one = ReplayExecution::new(2, 0, 2);
    one.regions = 1;
    for (name, execution, rec) in [
        (
            "3 x 3 chunks",
            ReplayExecution::new(2, 0, 2),
            record(groups([5, 5, 1], [2, 1, 1]), Some(TOKEN), &[], 0),
        ),
        (
            "an edge where one region is recorded",
            one,
            record(threads([3, 1, 1], [2, 1, 1]), Some(TOKEN), &[], 0),
        ),
    ] {
        let clean = record(groups([1, 1, 1], [2, 1, 1]), Some(TOKEN), &[], 0);
        let ran =
            preprocess_with_limits(&dev, limits, &execution, &variants, &[rec, clean], (2, 0));
        assert_eq!(ran.dispatched(0), [], "{name}");
        assert_eq!(ran.dispatched(1), [(0, 0, [1, 1, 1])], "{name}");
        assert_eq!(ran.status.replay_faults, ReplayFaults::REGIONS, "{name}");
    }
}

/// The preprocess's work for a command is bounded by the regions and variants the execution
/// records, not by its count. Over 16 and then 32 copies of a command that splits into the most
/// regions, every invocation visits its loop headers equally often, and twice the commands take at
/// most twice the lane-steps: a walk of the other commands per command (a segment search, a rescan
/// of the records) fails here.
#[test]
fn the_preprocess_meters_each_command_alone() {
    let dev = device();
    let local = [4, 2, 1];
    let variants = [
        variant(TOKEN, [4, 2, 1], 0),
        variant(TOKEN, [2, 2, 1], 0),
        variant(TOKEN, [4, 1, 1], 0),
        variant(TOKEN, [2, 1, 1], 0),
    ];
    let ran = |n: u32| {
        // On limits 3 x 2 x 64, 3 x 2 regions: two chunks and an edge along x, a chunk and an
        // edge along y.
        let records: Vec<Vec<u8>> = (0..n)
            .map(|_| {
                record(
                    threads([26, 5, 1], local),
                    Some(TOKEN),
                    &[(0, A.0), (1, B.0 + 4)],
                    0,
                )
            })
            .collect();
        let execution = ReplayExecution::new(n, 0, n);
        let ran = preprocess_with_limits(&dev, [3, 2, 64], &execution, &variants, &records, (n, 0));
        assert!(!ran.status.replay_fault, "{:?}", ran.status);
        assert_eq!(ran.dispatched(n - 1).len(), 6, "{n}");
        ran
    };
    let (one, two) = (ran(16), ran(32));
    let per_command = one.visits[0];
    assert!(per_command > 0);
    for (n, r) in [(16, &one), (32, &two)] {
        let live: Vec<u64> = r.visits.iter().copied().filter(|&v| v != 0).collect();
        assert_eq!(live.len(), n, "one invocation per command");
        assert!(
            live.iter().all(|&v| v == per_command),
            "{n}: loop-header visits per command {live:?}"
        );
    }
    assert!(
        two.steps <= 2 * one.steps,
        "{} lane-steps for 32 commands, {} for 16",
        two.steps,
        one.steps
    );
}

/// Push bytes of the synthetic templates.
const GENERATED_PUSH: u32 = 32;

/// The holes of synthetic template `v` (root table, region, status) and where its static bytes
/// lie: an even variant reads its slot table at 0, its region at 8, and the status at 16, with
/// static bytes at 24; an odd one the status at 0 and its slot table at 16, no region, with static
/// bytes at 8.
fn template_shape(v: usize) -> ([u32; 3], usize) {
    if v.is_multiple_of(2) {
        ([0, 8, 16], 24)
    } else {
        ([16, NO_HOLE, 0], 8)
    }
}

/// A generated-commands plan of `variants` synthetic templates ([`template_shape`]), executions
/// of at most `segment_commands` commands.
fn generated_plan(variants: usize, segment_commands: u32) -> generated::Planned {
    let body = TEMPLATE_HEAD_BYTES as usize;
    let mut templates = Vec::new();
    for v in 0..variants {
        let (holes, statics) = template_shape(v);
        let mut t = vec![0u8; body + GENERATED_PUSH as usize];
        for (k, h) in holes.iter().enumerate() {
            t[4 * k..4 * k + 4].copy_from_slice(&h.to_le_bytes());
        }
        for i in 0..8 {
            t[body + statics + i] = 0xA0 + 16 * v as u8 + i as u8;
        }
        templates.extend(t);
    }
    generated::Planned {
        push_bytes: GENERATED_PUSH,
        stride: GeneratedLayout::PUSH_OFFSET + GENERATED_PUSH + 12,
        segment_commands,
        replay: GeneratedReplay {
            templates,
            pipeline_layout: PipelineLayout::default(),
        },
    }
}

/// The push bytes K14c's path records for variant `v` of the synthetic templates at command
/// `slot`, region `region`: its template with the command's slot table, the region's record, and
/// the status.
fn expected_push(ran: &Ran, v: usize, slot: u32, region: u32) -> Vec<u8> {
    let body = TEMPLATE_HEAD_BYTES as usize;
    let size = body + GENERATED_PUSH as usize;
    let replay = ran.pre.generated.as_ref().expect("templates");
    let mut push = replay.templates[v * size + body..(v + 1) * size].to_vec();
    let l = ran.pre.layout;
    let (holes, _) = template_shape(v);
    for (hole, address) in holes.into_iter().zip([
        SCRATCH + l.table_offset(slot),
        SCRATCH + l.region_offset(slot, region),
        STATUS,
    ]) {
        if hole != NO_HOLE {
            push[hole as usize..hole as usize + 8].copy_from_slice(&address.to_le_bytes());
        }
    }
    push
}

/// `bytes` (an encoded record) with its barrier set: it waits for every earlier command.
fn barred(mut bytes: Vec<u8>) -> Vec<u8> {
    let at = crate::indirect_commands::layout::BARRIER as usize;
    bytes[at..at + 4].copy_from_slice(&1u32.to_le_bytes());
    bytes
}

/// On the generated-commands path the preprocess writes, per command and region, one sequence that
/// carries what K14c's arguments carry: the index of the variant whose argument dispatches (0 where
/// none does), that variant's push bytes with the command's slot table, the region's record, and
/// the status at its own holes, and the same group count. The slot tables, region records, and
/// status are K14c's; a refused command's sequences run nothing.
#[test]
fn the_generated_sequences_carry_what_the_arguments_carry() {
    let dev = device();
    let limits = dev.profile().fields().limits.max_compute_work_group_count;
    let variants = [
        variant(TOKEN, [8, 4, 1], 0),
        variant(TOKEN, [5, 4, 1], 0),
        variant(TOKEN, [8, 1, 1], 0),
        variant(TOKEN, [5, 1, 1], 0),
        variant(OTHER, [8, 4, 1], 0),
    ];
    let records = [
        // A ragged grid: its four regions on four variants.
        record(threads([13, 5, 1], [8, 4, 1]), Some(TOKEN), &[(0, A.0)], 0),
        record(
            groups([2, 1, 1], [8, 4, 1]),
            Some(OTHER),
            &[(1, B.0 + 4)],
            0,
        ),
        record(Action::Empty, None, &[], 0),
        // No variant runs a threadgroup of 3 x 3: refused.
        record(threads([3, 3, 1], [3, 3, 1]), Some(TOKEN), &[], 0),
        record(
            groups([1, 2, 1], [8, 4, 1]),
            Some(TOKEN),
            &[(0, A.0 + 8)],
            0,
        ),
    ];
    let execution = ReplayExecution::new(5, 0, 5);
    let argued = preprocess(&dev, &execution, &variants, &records, 0);
    let ran = run_preprocess(
        &dev,
        limits,
        (&execution, &variants),
        &records,
        (5, 0),
        Some(generated_plan(variants.len(), MAX_SEGMENT_COMMANDS)),
    );
    assert_eq!(argued.status.replay_faults, ReplayFaults::VARIANT);
    assert_eq!(ran.status, argued.status);
    let l = ran.pre.layout;
    assert!(argued.pre.layout.generated.is_none());
    for slot in 0..5 {
        for root in 0..SLOT_TABLE_ENTRIES {
            assert_eq!(
                ran.entry(slot, root),
                argued.entry(slot, root),
                "{slot} {root}"
            );
        }
        let dispatched = argued.dispatched(slot);
        for region in 0..l.regions {
            assert_eq!(ran.region(slot, region), argued.region(slot, region));
            let (index, push, groups) = ran.sequence(slot, region);
            match dispatched.iter().find(|(r, ..)| *r == region) {
                Some(&(_, v, g)) => {
                    assert_eq!((index, groups), (v, g), "command {slot} region {region}");
                    assert_eq!(push, expected_push(&ran, v as usize, slot, region));
                }
                None => assert_eq!((index, groups), (0, [0; 3]), "{slot} {region}"),
            }
        }
    }
    assert_eq!(argued.dispatched(0).len(), 4);
    assert_eq!(argued.dispatched(1), [(0, 4, [2, 1, 1])]);
    // One execution: the five commands' sequences.
    assert_eq!(ran.count(0), 5 * l.regions);
}

/// An execution starts at a command whose record sets the barrier, at each multiple of the most
/// commands of one execution, and at a command the storage does not hold (which runs nothing); it
/// runs the commands up to the next start, and no count exceeds what the execution recorded there
/// may run.
#[test]
fn executions_start_at_barriers() {
    let dev = device();
    let limits = dev.profile().fields().limits.max_compute_work_group_count;
    let variants = [variant(TOKEN, [8, 4, 1], 0)];
    let clean = || record(groups([1, 1, 1], [8, 4, 1]), Some(TOKEN), &[], 0);
    // Barriers at 2, 3, and 9 of 12 commands; the storage holds 11 of them.
    let records: Vec<Vec<u8>> = (0..11)
        .map(|i| {
            if [2, 3, 9].contains(&i) {
                barred(clean())
            } else {
                clean()
            }
        })
        .collect();
    let mut execution = ReplayExecution::new(12, 0, 12);
    execution.regions = 2;
    let ran = run_preprocess(
        &dev,
        limits,
        (&execution, &variants),
        &records,
        (11, 0),
        Some(generated_plan(1, 4)),
    );
    assert_eq!(ran.status.replay_faults, ReplayFaults::RANGE);
    let g = ran.generated();
    // Starts 0, 2, 3, 4 (a multiple of 4), 8 (one), 9, and 11 (past the storage's 11).
    let commands = [2, 0, 1, 1, 4, 0, 0, 0, 1, 2, 0, 1];
    for (slot, n) in commands.into_iter().enumerate() {
        let slot = slot as u32;
        assert_eq!(ran.count(slot), 2 * n, "command {slot}");
        assert!(ran.count(slot) <= g.max_sequences(slot), "command {slot}");
    }
    assert_eq!(
        (0..12).map(|s| g.max_sequences(s)).collect::<Vec<_>>(),
        [8, 6, 4, 2, 8, 6, 4, 2, 8, 6, 4, 2]
    );
    assert_eq!(
        ran.sequence(11, 0).2,
        [0; 3],
        "a command the storage lacks runs nothing"
    );
    assert_eq!(ran.sequence(10, 0).2, [1, 1, 1]);
    assert_eq!(ran.sequence(10, 1).2, [0; 3], "a region the command lacks");

    // A range that starts mid-buffer counts its commands from its own start.
    let shifted = ReplayExecution::new(11, 1, 4);
    let ran = run_preprocess(
        &dev,
        limits,
        (&shifted, &variants),
        &records,
        (11, 0),
        Some(generated_plan(1, 4)),
    );
    assert!(!ran.status.replay_fault, "{:?}", ran.status);
    // Commands 1..5: barriers at buffer commands 2 and 3, so starts 0, 1, 2.
    assert_eq!(
        (0..4).map(|s| ran.count(s)).collect::<Vec<_>>(),
        [
            ran.pre.layout.regions,
            ran.pre.layout.regions,
            2 * ran.pre.layout.regions,
            0
        ]
    );
}

/// On the generated-commands path, too, the preprocess's work for a command is bounded: the
/// segment's end is found in at most `segment - 1` later barrier words, so every invocation visits
/// its loop headers equally often and twice the commands take at most twice the lane-steps. A walk
/// to the segment's end over every later command fails here.
#[test]
fn the_generated_preprocess_meters_each_command_alone() {
    let dev = device();
    let local = [4, 2, 1];
    let variants = [
        variant(TOKEN, [4, 2, 1], 0),
        variant(TOKEN, [2, 2, 1], 0),
        variant(TOKEN, [4, 1, 1], 0),
        variant(TOKEN, [2, 1, 1], 0),
    ];
    let ran = |n: u32| {
        let records: Vec<Vec<u8>> = (0..n)
            .map(|_| {
                record(
                    threads([26, 5, 1], local),
                    Some(TOKEN),
                    &[(0, A.0), (1, B.0 + 4)],
                    0,
                )
            })
            .collect();
        let execution = ReplayExecution::new(n, 0, n);
        let ran = run_preprocess(
            &dev,
            [3, 2, 64],
            (&execution, &variants),
            &records,
            (n, 0),
            Some(generated_plan(variants.len(), MAX_SEGMENT_COMMANDS)),
        );
        assert!(!ran.status.replay_fault, "{:?}", ran.status);
        assert_eq!(ran.count(0), MAX_SEGMENT_COMMANDS * ran.pre.layout.regions);
        ran
    };
    let (one, two) = (ran(16), ran(32));
    let per_command = one.visits[0];
    assert!(per_command > 0);
    for (n, r) in [(16, &one), (32, &two)] {
        let live: Vec<u64> = r.visits.iter().copied().filter(|&v| v != 0).collect();
        assert_eq!(live.len(), n, "one invocation per command");
        assert!(
            live.iter().all(|&v| v == per_command),
            "{n}: loop-header visits per command {live:?}"
        );
    }
    assert!(
        two.steps <= 2 * one.steps,
        "{} lane-steps for 32 commands, {} for 16",
        two.steps,
        one.steps
    );
}

/// The replay's modules test a product's overflow only by `OpUMulExtended`'s high word, never by
/// the forms NVIDIA's compiler (615.71.09) refuses beside the preprocess's other arithmetic: a
/// division of the largest `u64` (`b > (2^64 - 1) / a`), or a comparison with `2^32 - 1` of a
/// product of two widened `u32`s (or a select of one). Every `OpUMulExtended` is 32-bit: a `u64`
/// product is four of its halves' (`FunctionBuilder::mul_extended_from_halves`). The preprocess
/// on every committed profile with device addresses.
#[test]
fn the_replay_modules_test_overflow_by_a_products_high_word() {
    const TYPE_INT: u32 = 21;
    const TYPE_VECTOR: u32 = 23;
    const TYPE_STRUCT: u32 = 30;
    const UCONVERT: u32 = 113;
    const IMUL: u32 = 132;
    const UDIV: u32 = 134;
    const UMUL_EXTENDED: u32 = 151;
    const SELECT: u32 = 169;
    const UGREATER_THAN: u32 = 172;
    let mut seen = 0;
    for (stem, dev) in devices().into_iter().filter(|(_, d)| addressed(d)) {
        seen += 1;
        let preprocess = replay_preprocess_module(&dev).expect("preprocess");
        let (what, words) = ("preprocess", &preprocess.spirv);
        let consts = int_constants(words);
        let is = |id: u32, width: u32, value: u64| consts.get(&id) == Some(&(width, value));
        let mut widened = std::collections::BTreeSet::new();
        let mut products = std::collections::BTreeSet::new();
        // Each integer, vector or struct type's (first member's) integer width.
        let mut widths = std::collections::BTreeMap::new();
        let mut extended = std::collections::BTreeMap::<u32, u32>::new();
        let mut i = 5;
        while let Some(&w) = words.get(i) {
            let (len, op) = ((w >> 16) as usize, w & 0xFFFF);
            let at = |k: usize| words.get(i + k).copied().unwrap_or(0);
            match op {
                TYPE_INT => {
                    widths.insert(at(1), at(2));
                }
                TYPE_VECTOR | TYPE_STRUCT => {
                    if let Some(&w) = widths.get(&at(2)) {
                        widths.insert(at(1), w);
                    }
                }
                UCONVERT => {
                    widened.insert(at(2));
                }
                IMUL if widened.contains(&at(3)) && widened.contains(&at(4)) => {
                    products.insert(at(2));
                }
                SELECT if products.contains(&at(4)) || products.contains(&at(5)) => {
                    products.insert(at(2));
                }
                // OpUDiv type, result, dividend, divisor.
                UDIV => assert!(!is(at(3), 64, u64::MAX), "{stem} {what}: (2^64 - 1) / x"),
                // OpUGreaterThan type, result, a, b.
                UGREATER_THAN => assert!(
                    !(products.contains(&at(3)) && is(at(4), 64, u64::from(u32::MAX))),
                    "{stem} {what}: a widened u32 product compared with 2^32 - 1"
                ),
                // OpUMulExtended type, result, a, b.
                UMUL_EXTENDED => {
                    let width = widths.get(&at(1)).copied().unwrap_or(0);
                    *extended.entry(width).or_default() += 1;
                }
                _ => {}
            }
            if len == 0 {
                break;
            }
            i += len;
        }
        // The flat dispatch's five `u64` products of four each and three `u32` ones
        // (`flat_dispatch`), and the preprocess's own three `u32` ones.
        let want = 26;
        assert_eq!(
            extended,
            std::collections::BTreeMap::from([(32, want)]),
            "{stem} {what}: OpUMulExtended by width"
        );
    }
    assert!(seen > 0);
}
