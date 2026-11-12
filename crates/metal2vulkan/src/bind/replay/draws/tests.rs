//! The draw replay's preprocess, on the CPU interpreter: valid where device addresses are and
//! refused by name elsewhere; each command's draw lands in the argument slot of its primitive type
//! with Metal's vertex start and base instance as Vulkan's first vertex and first instance; its
//! vertex and fragment slot tables hold its buffers' resident allocations (or the encoder's,
//! where the commands inherit buffers); an address no allocation holds leaves the slot unbound and
//! counts a resident miss while the command still draws; a command the replay cannot run draws
//! nothing and says why beside clean commands that draw; the host side refuses what it cannot set
//! up.

use std::num::NonZeroU64;
use std::path::PathBuf;

use super::*;
use crate::bind::state::{ResourceClass, UsedResource};
use crate::bind::status::{ExecutionStatus, StatusReport};
use crate::device::{DeviceDescription, DevicePolicy};
use crate::indirect_commands::{Action, BufferBinding, DRAW, INHERIT_PIPELINE, RENDER};
use crate::spirv::interp::{run, Buffer, Config, Order, Resources};
use crate::spirv::testing::assert_valid;

const SCRATCH: u64 = 0x0000_0030_0000_0000;
const RECORDS: u64 = 0x0000_0031_0000_0000;
const STATUS: u64 = 0x0000_0012_0000_0000;
const SINK: u64 = 0x0000_0011_0000_0000;

/// The resident allocations `(guest, host, length)`.
const A: (u64, u64, u64) = (0x0000_0001_0000_1000, 0x0000_0020_0000_0000, 256);
const B: (u64, u64, u64) = (0x0000_0001_0000_8000, 0x0000_0021_0000_0000, 60);
/// An address in no resident allocation.
const AWAY: u64 = 0x0000_0001_0004_0000;

/// Every committed profile as a device, by stem.
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
            (
                stem,
                Device::new(desc, DevicePolicy::default()).expect("device"),
            )
        })
        .collect();
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

fn addressed(dev: &Device) -> bool {
    dev.profile().fields().buffer_access == BufferAccess::Addressed
}

fn device() -> Device {
    devices()
        .into_iter()
        .map(|(_, d)| d)
        .find(addressed)
        .expect("an addressed profile")
}

/// The resident set: `A` and `B`.
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

fn bind(address: u64) -> Option<BufferBinding> {
    Some(BufferBinding {
        address,
        attribute_stride: None,
    })
}

fn draw(primitive_type: u32, vertex_start: u32, vertex_count: u32, instances: [u32; 2]) -> Action {
    Action::DrawPrimitives {
        primitive_type,
        vertex_start,
        vertex_count,
        instance_count: instances[0],
        base_instance: instances[1],
    }
}

/// A record of `action` with vertex buffers `vertex` and fragment buffers `fragment` (`(index,
/// address)`), its barrier set as `barrier` says.
fn record(action: Action, vertex: &[(usize, u64)], fragment: &[(usize, u64)]) -> CommandRecord {
    let mut r = CommandRecord {
        action,
        ..CommandRecord::default()
    };
    for &(i, address) in vertex {
        r.bindings.buffers_mut(BufferStage::Vertex)[i] = bind(address);
    }
    for &(i, address) in fragment {
        r.bindings.buffers_mut(BufferStage::Fragment)[i] = bind(address);
    }
    r
}

fn bytes(r: &CommandRecord) -> Vec<u8> {
    let mut out = vec![0u8; COMMAND_BYTES];
    r.encode_into(&mut out).expect("record");
    out
}

fn word(bytes: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(bytes[at..at + 4].try_into().expect("word"))
}

/// The storage header: `capacity`, the fault word, the policy bits of a render descriptor of
/// draws whose commands inherit the pipeline (and the buffers where `inherit_buffers`), and 4
/// vertex buffers and 2 fragment buffers.
struct Header {
    capacity: u32,
    fault: u32,
    inherit_buffers: bool,
}

/// What one preprocess left.
struct Ran {
    pre: DrawReplayPreprocess,
    scratch: Vec<u8>,
    status: StatusReport,
}

impl Ran {
    /// The `VkDrawIndirectCommand` of command `command` for type slot `slot`.
    fn args(&self, command: u32, slot: u32) -> [u32; 4] {
        let at = self.pre.layout.argument_offset(command, slot) as usize;
        std::array::from_fn(|i| word(&self.scratch, at + 4 * i))
    }

    /// Entry `root` of a slot table at `table`: `[host, guest, size, bind_offset]`.
    fn entry(&self, table: u64, root: u64) -> [u64; 4] {
        let at = (table + root * roots::ENTRY_BYTES) as usize;
        std::array::from_fn(|i| {
            u64::from_le_bytes(
                self.scratch[at + 8 * i..at + 8 * i + 8]
                    .try_into()
                    .expect("u64"),
            )
        })
    }

    fn vertex(&self, command: u32, slot: u64) -> [u64; 4] {
        self.entry(self.pre.layout.vertex_table_offset(command), 2 + slot)
    }

    fn fragment(&self, command: u32, slot: u64) -> [u64; 4] {
        self.entry(self.pre.layout.fragment_table_offset(command), 2 + slot)
    }
}

/// Run the preprocess of `replay` against `records` under `header`, in both scheduling orders.
fn preprocess(replay: &DrawReplay, records: &[CommandRecord], header: Header) -> Ran {
    preprocess_over(replay, records, header, None)
}

/// [`preprocess`], with `range` (`location`, `length`) in the parameters where a replay that
/// reads its range takes it, as the consumer copies it there.
fn preprocess_over(
    replay: &DrawReplay,
    records: &[CommandRecord],
    header: Header,
    range: Option<[u32; 2]>,
) -> Ran {
    let dev = device();
    let pre = draw_replay_preprocess(&dev, &state(), replay).expect("preprocess");
    let mut policy = RENDER | DRAW | INHERIT_PIPELINE;
    if replay.indexed {
        policy |= crate::indirect_commands::DRAW_INDEXED;
    }
    if header.inherit_buffers {
        policy |= crate::indirect_commands::INHERIT_BUFFERS;
    }
    let counts: u32 = (4 << 8) | (2 << 16);
    let mut storage = Vec::new();
    for w in [header.capacity, header.fault, policy, counts] {
        storage.extend_from_slice(&w.to_le_bytes());
    }
    for r in records {
        storage.extend_from_slice(&bytes(r));
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
    put(l.parent_offset, &pre.parent);
    if let Some([location, length]) = range {
        put(l.range_offset(), &location.to_le_bytes());
        put(l.range_offset() + 4, &length.to_le_bytes());
    }
    let mut push = pre.push_constants.clone();
    for hole in &pre.push_holes {
        let (offset, address) = match *hole {
            PushHole::RootTableAddress { offset, .. } => (offset, SCRATCH + l.resident_offset),
            PushHole::ExecutionStatusAddress { offset, .. } => (offset, STATUS),
            PushHole::CommandRecordsAddress { offset, .. } => (offset, RECORDS),
            _ => panic!("a hole the preprocess does not take"),
        };
        push[offset as usize..offset as usize + 8].copy_from_slice(&address.to_le_bytes());
    }
    let m = typed_module().expect("module");
    let mut outs = Vec::new();
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
        outs.push(run(&m, &cfg, res).expect("run").buffers);
    }
    assert_eq!(outs[0], outs[1], "the two scheduling orders agree");
    let mut out = outs.swap_remove(0);
    assert_eq!(out[1], storage, "the records are only read");
    let status: [u8; ExecutionStatus::BYTES] = out[2].as_slice().try_into().expect("status");
    Ran {
        pre,
        scratch: out.swap_remove(0),
        status: ExecutionStatus::decode(&status),
    }
}

fn clean(capacity: u32) -> Header {
    Header {
        capacity,
        fault: 0,
        inherit_buffers: false,
    }
}

#[test]
fn the_draw_preprocess_module_is_valid_where_device_addresses_are() {
    let mut seen = (0, 0);
    for (stem, dev) in devices() {
        let p = dev.profile();
        if addressed(&dev) {
            seen.0 += 1;
            assert_valid(typed_module().expect("module"), p);
            let m = draw_replay_module(&dev).unwrap_or_else(|e| panic!("{stem}: {e}"));
            assert!(p.check(&m.requirements).is_ok(), "{stem}");
            assert!(m.layout.sets.is_empty(), "{stem}");
            let range = m.layout.push_constants[0];
            assert_eq!(
                (range.stages, range.offset, range.size),
                (
                    crate::vk::ShaderStageFlags::COMPUTE,
                    0,
                    DrawReplayModule::PUSH_BYTES
                ),
                "{stem}"
            );
        } else {
            seen.1 += 1;
            let e = Unsupported::needs_any_of(
                Feature::GpuEncodedIndirectCommands,
                vec![p.addressed_needs()],
            );
            assert_eq!(
                draw_replay_module(&dev),
                Err(Error::Unsupported(e.clone())),
                "{stem}"
            );
            assert_eq!(
                draw_replay_preprocess(&dev, &state(), &DrawReplay::new(1, 0, 1, vec![3])),
                Err(BindError::Unsupported(e)),
                "{stem}"
            );
        }
    }
    assert!(seen.0 > 0 && seen.1 > 0, "{seen:?}");
}

/// A command's draw lands in the slot of its type, as Vulkan's draw of Metal's: vertex count,
/// instance count, vertex start as the first vertex, base instance as the first instance; the
/// other slot is zero. Its buffers resolve in the resident set at the address's offset, each
/// stage's in its own table; an unset slot and a nil buffer's offset hold no allocation, the nil
/// buffer's offset as the bind offset. An empty command draws nothing and reports nothing.
#[test]
fn each_command_draws_its_own_type_with_its_own_buffers() {
    let replay = DrawReplay::new(3, 0, 3, vec![3, 4]);
    let records = [
        record(
            draw(3, 6, 9, [2, 5]),
            &[(0, A.0 + 16), (2, 40)],
            &[(1, B.0 + 4)],
        ),
        record(draw(4, 0, 4, [1, 0]), &[(1, B.0)], &[]),
        record(Action::Empty, &[(0, A.0)], &[]),
    ];
    let ran = preprocess(&replay, &records, clean(3));
    assert!(!ran.status.replay_fault, "{:?}", ran.status);
    assert!(!ran.status.resident_miss);
    assert_eq!(ran.args(0, 0), [9, 2, 6, 5]);
    assert_eq!(ran.args(0, 1), [0; 4]);
    assert_eq!(ran.args(1, 0), [0; 4]);
    assert_eq!(ran.args(1, 1), [4, 1, 0, 0]);
    assert_eq!(ran.args(2, 0), [0; 4]);
    assert_eq!(ran.args(2, 1), [0; 4]);
    assert_eq!(ran.vertex(0, 0), [A.1, A.0, padded(A.2), 16]);
    assert_eq!(ran.vertex(0, 1), [0; 4]);
    assert_eq!(ran.vertex(0, 2), [0, 0, 0, 40]);
    assert_eq!(ran.fragment(0, 1), [B.1, B.0, padded(B.2), 4]);
    assert_eq!(ran.fragment(0, 0), [0; 4]);
    assert_eq!(ran.vertex(1, 1), [B.1, B.0, padded(B.2), 0]);
    // The sink entry names the shared resident set from each table.
    let sink = ran.entry(ran.pre.layout.vertex_table_offset(1), roots::SINK_ROOT);
    assert_eq!(sink[0], SINK);
    assert_eq!(sink[2], 2, "two resident allocations");
}

/// The resident root table pads each allocation as the binder does.
fn padded(length: u64) -> u64 {
    crate::bind::instantiate::padded(length)
}

/// A command the replay cannot run draws nothing and says why, and the clean commands beside it
/// draw: a draw of a type the execution does not name (`ACTION`), a set barrier (`BARRIER`), a
/// pipeline set while the commands inherit it (`PIPELINE`), a binding past its stage's count
/// (`RECORD`).
#[test]
fn a_command_the_replay_cannot_run_draws_nothing_and_says_why() {
    let replay = DrawReplay::new(5, 0, 5, vec![3]);
    let mut barrier = record(draw(3, 0, 3, [1, 0]), &[], &[]);
    barrier.barrier = true;
    let mut piped = record(draw(3, 0, 3, [1, 0]), &[], &[]);
    piped.bindings.pipeline = NonZeroU64::new(9);
    let cases = [
        (
            record(draw(0, 0, 3, [1, 0]), &[], &[]),
            ReplayFaults::ACTION,
        ),
        (barrier, ReplayFaults::BARRIER),
        (piped, ReplayFaults::PIPELINE),
        (
            record(draw(3, 0, 3, [1, 0]), &[(4, A.0)], &[]),
            ReplayFaults::RECORD,
        ),
    ];
    for (bad, why) in cases {
        let records = [
            record(draw(3, 0, 3, [1, 0]), &[], &[]),
            bad,
            record(draw(3, 3, 3, [1, 0]), &[], &[]),
            record(Action::Empty, &[], &[]),
            record(Action::Empty, &[], &[]),
        ];
        let ran = preprocess(&replay, &records, clean(5));
        assert!(ran.status.replay_fault, "{why:?}");
        assert_eq!(ran.status.replay_faults, why);
        assert_eq!(ran.args(0, 0), [3, 1, 0, 0], "{why:?}");
        assert_eq!(ran.args(1, 0), [0; 4], "{why:?}");
        assert_eq!(ran.args(2, 0), [3, 1, 3, 0], "{why:?}");
    }
}

/// An address no resident allocation holds leaves its slot unbound and counts one resident miss
/// per such buffer, and the command still draws.
#[test]
fn a_buffer_no_allocation_holds_is_unbound_and_counted() {
    let replay = DrawReplay::new(1, 0, 1, vec![3]);
    let records = [record(
        draw(3, 0, 3, [1, 0]),
        &[(0, AWAY), (1, A.0)],
        &[(0, AWAY)],
    )];
    let ran = preprocess(&replay, &records, clean(1));
    assert!(!ran.status.replay_fault);
    assert!(ran.status.resident_miss);
    assert_eq!(ran.status.resident_misses, 2);
    assert_eq!(ran.args(0, 0), [3, 1, 0, 0]);
    assert_eq!(ran.vertex(0, 0), [0; 4]);
    assert_eq!(ran.vertex(0, 1), [A.1, A.0, padded(A.2), 0]);
    assert_eq!(ran.fragment(0, 0), [0; 4]);
}

/// A buffer of the command's own that no slot table holds (past the slots its stage's function
/// reads, or of a stage that reads none) is resolved too: its miss counts, as the host step's
/// does, though no slot holds it. A nil buffer's offset counts none, nor does a command that draws
/// nothing.
#[test]
fn a_buffer_past_the_slots_read_counts_its_miss() {
    let mut replay = DrawReplay::new(2, 0, 2, vec![3]);
    replay.vertex_slots = 1;
    replay.fragment_slots = 0;
    let buffers = |action| {
        record(
            action,
            &[(0, A.0), (1, AWAY), (2, B.0), (3, 0x40)],
            &[(0, AWAY), (1, AWAY)],
        )
    };
    let records = [buffers(draw(3, 0, 3, [1, 0])), buffers(Action::Empty)];
    let ran = preprocess(&replay, &records, clean(2));
    assert!(!ran.status.replay_fault);
    assert!(ran.status.resident_miss);
    assert_eq!(ran.status.resident_misses, 3);
    assert_eq!(ran.args(0, 0), [3, 1, 0, 0]);
    assert_eq!(ran.vertex(0, 0), [A.1, A.0, padded(A.2), 0]);
    assert_eq!(ran.args(1, 0), [0; 4]);
}

/// Where the commands inherit buffers, every command's tables hold the encoder's, and a command
/// that sets one of its own draws nothing (`BUFFERS`).
#[test]
fn inherited_buffers_are_the_encoder_s() {
    let mut replay = DrawReplay::new(2, 0, 2, vec![3]);
    replay.inheritance.buffers = StateSource::Parent;
    replay.parent.buffers_mut(BufferStage::Vertex)[0] = bind(B.0 + 8);
    replay.parent.buffers_mut(BufferStage::Fragment)[1] = bind(A.0);
    let records = [
        record(draw(3, 0, 3, [1, 0]), &[], &[]),
        record(draw(3, 0, 3, [1, 0]), &[(0, A.0)], &[]),
    ];
    let header = Header {
        inherit_buffers: true,
        ..clean(2)
    };
    let ran = preprocess(&replay, &records, header);
    assert_eq!(ran.status.replay_faults, ReplayFaults::BUFFERS);
    assert_eq!(ran.args(0, 0), [3, 1, 0, 0]);
    assert_eq!(ran.args(1, 0), [0; 4]);
    assert_eq!(ran.vertex(0, 0), [B.1, B.0, padded(B.2), 8]);
    assert_eq!(ran.fragment(0, 1), [A.1, A.0, padded(A.2), 0]);
}

/// The producers' sticky encoding fault refuses every command; a command at or past the
/// capacity the header states is not read (`RANGE`), its range starting past the first command.
#[test]
fn a_faulted_or_short_storage_draws_nothing() {
    let replay = DrawReplay::new(2, 0, 2, vec![3]);
    let records = [
        record(draw(3, 0, 3, [1, 0]), &[], &[]),
        record(draw(3, 0, 3, [1, 0]), &[], &[]),
    ];
    let faulted = Header {
        fault: 1,
        ..clean(2)
    };
    let ran = preprocess(&replay, &records, faulted);
    assert_eq!(ran.status.replay_faults, ReplayFaults::ENCODING);
    assert_eq!(ran.args(0, 0), [0; 4]);
    assert_eq!(ran.args(1, 0), [0; 4]);
    let replay = DrawReplay::new(2, 1, 1, vec![3]);
    let ran = preprocess(&replay, &records, clean(1));
    assert_eq!(ran.status.replay_faults, ReplayFaults::RANGE);
    assert_eq!(ran.args(0, 0), [0; 4]);
}

/// The host side refuses a range past the capacity, commands that name their own pipelines, and
/// primitive types that are not one class's draws; tables hold the slots asked for.
#[test]
fn the_host_side_refuses_what_it_cannot_set_up() {
    let dev = device();
    let refused = |r: ReplayRefusal| Err(BindError::InvalidReplay { reason: r });
    let pre = |r: &DrawReplay| draw_replay_preprocess(&dev, &state(), r);
    assert_eq!(
        pre(&DrawReplay::new(2, 1, 2, vec![3])),
        refused(ReplayRefusal::RangeOutsideCapacity)
    );
    let mut own = DrawReplay::new(1, 0, 1, vec![3]);
    own.inheritance.pipeline = StateSource::Command;
    assert_eq!(pre(&own), refused(ReplayRefusal::OwnPipelines));
    for types in [
        vec![],
        vec![5],
        vec![3, 3],
        vec![0, 3],
        vec![2, 3],
        vec![1, 2, 3],
    ] {
        assert_eq!(
            pre(&DrawReplay::new(1, 0, 1, types.clone())),
            refused(ReplayRefusal::DrawPrimitiveTypes),
            "{types:?}"
        );
    }
    for types in [vec![0], vec![1, 2], vec![4, 3]] {
        assert!(pre(&DrawReplay::new(1, 0, 1, types)).is_ok());
    }
    let mut small = DrawReplay::new(4, 0, 4, vec![3]);
    small.vertex_slots = 3;
    small.fragment_slots = 0;
    let l = pre(&small).expect("small").layout;
    assert!(l.uploaded <= l.tables_offset);
    assert_eq!(l.vertex_table_offset(0), l.tables_offset);
    assert_eq!(
        l.fragment_table_offset(0),
        l.tables_offset + 5 * roots::ENTRY_BYTES
    );
    assert_eq!(
        l.vertex_table_offset(1),
        l.tables_offset + 7 * roots::ENTRY_BYTES
    );
    assert_eq!(
        l.arguments_offset,
        l.tables_offset + 28 * roots::ENTRY_BYTES
    );
}

// ---- A replay merged into one draw (`merged.rs`) ----

use super::super::merged::{merged_vertex, typed_module as merge_module, MergePhase};
use crate::bind::{draw_merge_module, DrawMergeModule, MERGED_REST_FIRST};

const OUT: u64 = 0x0000_0040_0000_0000;
const REST_OUT: u64 = 0x0000_0041_0000_0000;
/// Bytes of one decoded corner: Metal's vertex, instance, base vertex and base instance, the
/// slot table's address, and the command's index (then padding).
const DECODED: usize = 32;

/// `ran`'s scratch after the merged replay's header (its chunk replaced by `chunk`) and both
/// merge passes, in both scheduling orders, and the status the passes left.
fn merge(ran: &Ran, chunk: u32) -> (Vec<u8>, StatusReport) {
    let m = ran.pre.layout.merged.expect("a merged layout");
    let mut scratch = ran.scratch.clone();
    let at = m.header_offset as usize;
    scratch[at..at + ran.pre.header.len()].copy_from_slice(&ran.pre.header);
    scratch[at + 16..at + 20].copy_from_slice(&chunk.to_le_bytes());
    let push = DrawMergeModule::push_constants(SCRATCH + m.header_offset, STATUS);
    let mut status = ExecutionStatus::initial().to_vec();
    for phase in [MergePhase::Blocks, MergePhase::Starts] {
        let module = merge_module(phase).expect("merge module");
        let mut outs = Vec::new();
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
                        address: Some(STATUS),
                        bytes: status.clone(),
                    },
                ],
                push: push.clone(),
                spec: Default::default(),
            };
            let cfg = Config {
                workgroups: m.group_count(),
                order,
                ..Config::default()
            };
            outs.push(run(&module, &cfg, res).expect("run").buffers);
        }
        assert_eq!(
            outs[0], outs[1],
            "{phase:?}: the two scheduling orders agree"
        );
        let mut out = outs.swap_remove(0);
        status = out.swap_remove(1);
        scratch = out.swap_remove(0);
    }
    let status: [u8; ExecutionStatus::BYTES] = status.as_slice().try_into().expect("status");
    (scratch, ExecutionStatus::decode(&status))
}

/// A vertex entry that decodes its corner (`merged_vertex`) from the header at its
/// `MergedDrawAddress` and stores what it found at its corner's place: `OUT + 24 corner` in the
/// first draw, `REST_OUT + 24 (VertexIndex - MERGED_REST_FIRST)` in the second.
fn decoding_module(chunk: u32) -> TypedModule {
    let mut b = ModuleBuilder::new();
    let u32_ = b
        .ty(Type::Int {
            width: 32,
            signed: false,
        })
        .expect("u32");
    let u64_ = b
        .ty(Type::Int {
            width: 64,
            signed: false,
        })
        .expect("u64");
    let t = Types { u32_, u64_ };
    let m_header = b
        .push_member(PushField::MergedDrawAddress, u64_, 0)
        .expect("header");
    let push = b.push_global().expect("push");
    let vertex_index = b
        .builtin(crate::spv::BuiltIn::VertexIndex, IoDir::Input, u32_)
        .expect("vertex");
    let instance_index = b
        .builtin(crate::spv::BuiltIn::InstanceIndex, IoDir::Input, u32_)
        .expect("instance");
    let void = b.ty(Type::Void).expect("void");
    let mut f = b.function(void).expect("function");
    let fb = &mut f;
    let push_value = fb.module().global_value(push).expect("push value");
    let i = k32(fb, m_header.index).expect("index");
    let p = fb.access_chain(push_value, &[i]).expect("chain");
    let header = fb.load(p, MemAccess::default()).expect("header");
    let vi = fb.module().global_value(vertex_index).expect("vi");
    let vi = fb.load(vi, MemAccess::default()).expect("vi");
    let ii = fb.module().global_value(instance_index).expect("ii");
    let ii = fb.load(ii, MemAccess::default()).expect("ii");
    let v = merged_vertex(fb, header, (vi, ii)).expect("decode");
    let rest_first = k32(fb, MERGED_REST_FIRST).expect("k");
    let second = fb
        .binary(BinOp::UGreaterThanEqual, vi, rest_first)
        .expect("second");
    let chunk = k32(fb, chunk).expect("chunk");
    let first_slot = fb.binary(BinOp::IMul, ii, chunk).expect("mul");
    let first_slot = fb.binary(BinOp::IAdd, first_slot, vi).expect("add");
    let rest_slot = fb.binary(BinOp::ISub, vi, rest_first).expect("sub");
    let slot = fb.select(second, rest_slot, first_slot).expect("slot");
    let out = k64(fb, OUT).expect("out");
    let rest_out = k64(fb, REST_OUT).expect("rest out");
    let base = fb.select(second, rest_out, out).expect("base");
    let slot = wide(fb, t, slot).expect("wide");
    let bytes = k64(fb, DECODED as u64).expect("bytes");
    let rel = fb.binary(BinOp::IMul, slot, bytes).expect("rel");
    let at_slot = fb.binary(BinOp::IAdd, base, rel).expect("at");
    for (i, w) in [v.vertex, v.instance, v.base_vertex, v.base_instance]
        .into_iter()
        .enumerate()
    {
        let a = at(fb, at_slot, 4 * i as u64).expect("at");
        store32(fb, t, a, w).expect("store");
    }
    let a = at(fb, at_slot, 16).expect("at");
    store64(fb, t, a, v.table).expect("store");
    let a = at(fb, at_slot, 24).expect("at");
    store32(fb, t, a, v.command).expect("store");
    fb.ret(None).expect("ret");
    let id = f.finish().expect("finish");
    b.entry(ExecModel::Vertex, id, Vec::new()).expect("entry");
    b.finalize().expect("module")
}

/// What one corner of a merged draw runs: Metal's vertex, instance, base vertex and base
/// instance, the command's vertex slot table's address, and the command's index.
type Corner = (u32, u32, u32, u32, u64, u32);

/// Every corner the merged draws of `scratch` run (both draws, in their order), decoded by
/// `merged_vertex` with chunks of `chunk` vertices.
fn decoded(ran: &Ran, scratch: &[u8], chunk: u32) -> Vec<Corner> {
    decoded_over(ran, scratch, chunk, None)
}

/// [`decoded`], with `memory` (`(address, bytes)`: the indices the commands' index entries name)
/// beside the scratch.
fn decoded_over(
    ran: &Ran,
    scratch: &[u8],
    chunk: u32,
    memory: Option<(u64, &[u8])>,
) -> Vec<Corner> {
    let m = ran.pre.layout.merged.expect("merged");
    let draw = |i: u32| -> [u32; 4] {
        let at = m.draw_offset(i) as usize;
        std::array::from_fn(|k| word(scratch, at + 4 * k))
    };
    let [first_vertices, chunks, first_first, first_base] = draw(0);
    let [rest, rest_instances, rest_first, rest_base] = draw(1);
    assert_eq!((first_vertices, first_first, first_base), (chunk, 0, 0));
    assert_eq!(
        (rest_instances, rest_first, rest_base),
        (1, MERGED_REST_FIRST, 0)
    );
    let module = decoding_module(chunk);
    let mut push = vec![0u8; 8];
    push.copy_from_slice(&(SCRATCH + m.header_offset).to_le_bytes());
    let mut corners = Vec::new();
    for (out, workgroups, first, count) in [
        (
            OUT,
            [chunk, chunks, 1],
            [0, 0],
            chunk as usize * chunks as usize,
        ),
        (
            REST_OUT,
            [rest, 1, 1],
            [MERGED_REST_FIRST, 0],
            rest as usize,
        ),
    ] {
        if count == 0 {
            continue;
        }
        let mut buffers = vec![
            Buffer {
                slot: None,
                address: Some(SCRATCH),
                bytes: scratch.to_vec(),
            },
            Buffer {
                slot: None,
                address: Some(out),
                bytes: vec![0; count * DECODED],
            },
        ];
        if let Some((address, bytes)) = memory {
            buffers.push(Buffer {
                slot: None,
                address: Some(address),
                bytes: bytes.to_vec(),
            });
        }
        let res = Resources {
            buffers,
            push: push.clone(),
            spec: Default::default(),
        };
        let cfg = Config {
            workgroups,
            first,
            ..Config::default()
        };
        let buffers = run(&module, &cfg, res).expect("run").buffers;
        assert_eq!(buffers[0], scratch, "the decode only reads the scratch");
        for c in buffers[1].as_chunks::<DECODED>().0 {
            corners.push((
                word(c, 0),
                word(c, 4),
                word(c, 8),
                word(c, 12),
                u64::from_le_bytes(c[16..24].try_into().expect("u64")),
                word(c, 24),
            ));
        }
    }
    corners
}

/// What Metal's draws of `draws` (`(vertex start, vertex count, instance count, base
/// instance)`, `None` for a command that draws nothing) of `per` vertices per primitive run,
/// command by command, instance by instance, vertex by vertex, as a merged draw's corners.
fn metal_corners(ran: &Ran, draws: &[Option<[u32; 4]>], per: u32) -> Vec<Corner> {
    let mut out = Vec::new();
    for (c, d) in draws.iter().enumerate() {
        let Some([start, count, instances, base]) = *d else {
            continue;
        };
        let table = SCRATCH + ran.pre.layout.vertex_table_offset(c as u32);
        for i in 0..instances {
            for v in 0..count - count % per {
                out.push((start + v, base + i, start, base, table, c as u32));
            }
        }
    }
    out
}

/// A replay merged into one draw runs what Metal's draws of its commands run, in their order:
/// each command's whole primitives' vertices (a list's trailing vertices draw nothing), in each
/// of its instances, with its own vertex start, base instance and slot table; a command that
/// draws nothing (empty, no instance, no whole primitive, or refused) has no corner. Both of the
/// merged draws carry corners here (chunks of 6, so 3 whole chunks and 3 corners left), and both
/// passes run several blocks.
#[test]
fn a_merged_replay_runs_each_command_s_whole_primitives_in_order() {
    let mut replay = DrawReplay::new(9, 0, 9, vec![3]);
    replay.merged = true;
    replay.vertex_slots = 1;
    replay.fragment_slots = 0;
    let mut barrier = record(draw(3, 0, 3, [1, 0]), &[], &[]);
    barrier.barrier = true;
    let draws = [
        Some([0, 3, 1, 0]),
        Some([5, 7, 1, 2]),
        None,
        Some([1, 3, 2, 9]),
        Some([0, 2, 4, 0]),
        Some([0, 6, 0, 3]),
        Some([10, 4, 1, 7]),
        None,
        Some([2, 3, 1, 1]),
    ];
    let mut records: Vec<CommandRecord> = draws
        .iter()
        .map(|d| match d {
            Some([s, n, k, b]) => record(draw(3, *s, *n, [*k, *b]), &[(0, A.0)], &[]),
            None => record(Action::Empty, &[], &[]),
        })
        .collect();
    records[7] = barrier;
    let ran = preprocess(&replay, &records, clean(9));
    assert_eq!(ran.status.replay_faults, ReplayFaults::BARRIER);
    let m = ran.pre.layout.merged.expect("merged");
    assert_eq!((m.block_commands, m.blocks), (3, 3));
    let (scratch, status) = merge(&ran, 6);
    assert!(
        !status.replay_fault,
        "the passes report nothing: {status:?}"
    );
    let expected = metal_corners(&ran, &draws, 3);
    assert_eq!(expected.len(), 21);
    assert_eq!(decoded(&ran, &scratch, 6), expected);
    // With Metal's chunk, every corner is in the second draw.
    let (scratch, _) = merge(&ran, MERGED_CHUNK);
    let at = m.draw_offset(1) as usize;
    assert_eq!(word(&scratch, at), 21);
    assert_eq!(word(&scratch, m.draw_offset(0) as usize + 4), 0);
}

/// A replay that reads its range draws the commands inside it as the host's step takes the
/// range: from `location` to `location + length`, clamped at the size; none, and `RANGE` once,
/// for a location past the size or an end past `2^32 - 1`. Merged, the commands inside it draw
/// their corners in order.
#[test]
fn a_replay_that_reads_its_range_draws_the_commands_inside_it() {
    let mut replay = DrawReplay::new(4, 0, 4, vec![3]);
    replay.merged = true;
    replay.range_read = true;
    replay.vertex_slots = 0;
    replay.fragment_slots = 0;
    let draws = [[0, 3, 1, 0], [3, 6, 2, 1], [9, 3, 1, 4], [1, 3, 3, 0]];
    let records: Vec<CommandRecord> = draws
        .iter()
        .map(|[s, n, k, b]| record(draw(3, *s, *n, [*k, *b]), &[], &[]))
        .collect();
    let cases = [
        ([1, 2], 1..3, false),
        ([2, 5], 2..4, false),
        ([0, u32::MAX], 0..4, false),
        ([0, 0], 0..0, false),
        ([4, 1], 0..0, false),
        ([5, 0], 0..0, true),
        ([1, u32::MAX], 0..0, true),
    ];
    for (range, inside, faults) in cases {
        let ran = preprocess_over(&replay, &records, clean(4), Some(range));
        let why = match faults {
            true => ReplayFaults::RANGE,
            false => ReplayFaults::default(),
        };
        assert_eq!(ran.status.replay_faults, why, "{range:?}");
        let mut drawn = [None; 4];
        for (command, [s, n, k, b]) in (0..4).zip(draws) {
            let expected = match inside.contains(&command) {
                true => [n, k, s, b],
                false => [0; 4],
            };
            assert_eq!(ran.args(command, 0), expected, "{range:?} {command}");
            drawn[command as usize] = inside.contains(&command).then_some([s, n, k, b]);
        }
        let (scratch, status) = merge(&ran, 6);
        assert!(!status.replay_fault, "{range:?}: {status:?}");
        let expected = metal_corners(&ran, &drawn, 3);
        assert_eq!(decoded(&ran, &scratch, 6), expected, "{range:?}");
    }
    // A replay that reads its range runs merged over the whole buffer alone.
    let dev = device();
    let refused = Err(BindError::InvalidReplay {
        reason: ReplayRefusal::RangeRead,
    });
    let pre = |r: &DrawReplay| draw_replay_preprocess(&dev, &state(), r);
    let mut apart = replay.clone();
    apart.merged = false;
    assert_eq!(pre(&apart), refused);
    let mut part = DrawReplay::new(4, 1, 3, vec![3]);
    part.merged = true;
    part.range_read = true;
    part.fragment_slots = 0;
    assert_eq!(pre(&part), refused);
    assert!(pre(&replay).is_ok());
}

/// An indexed draw's record: `count` indices of `index_type` (0 `u16`, 1 `u32`) at `indices`, in
/// `instances[0]` instances from base instance `instances[1]`, from base vertex `base_vertex`.
fn indexed(
    index_type: u32,
    indices: u64,
    count: u32,
    instances: [u32; 2],
    base_vertex: i32,
) -> Action {
    Action::DrawIndexedPrimitives {
        primitive_type: 3,
        index_count: count,
        index_type,
        index_buffer: indices,
        instance_count: instances[0],
        base_vertex,
        base_instance: instances[1],
    }
}

/// A merged replay of indexed commands beside a draw of primitives draws each command's whole
/// triangles in order, an indexed command's vertices its indices plus its base vertex (wrapping),
/// from its index buffer's offset rounded down to a multiple of the index size, a position past
/// the buffer's whole indices reading index 0, as the host's step and the encoder's draw read
/// them. An index buffer no resident allocation holds draws nothing and counts as a resident
/// miss; an index type Metal has no value for, or no index buffer, is a refused record. Each
/// corner carries its command's index, which the vertex variant writes for the fragment variant.
#[test]
fn a_merged_replay_draws_indexed_commands_from_their_indices() {
    // `A`'s bytes: the `u16` at byte `2k` is `100 + k`.
    let memory: Vec<u8> = (0..A.2 / 2)
        .flat_map(|k| (100 + k as u16).to_le_bytes())
        .collect();
    let u16_at = |byte: u64| {
        u32::from(u16::from_le_bytes([
            memory[byte as usize],
            memory[byte as usize + 1],
        ]))
    };
    let u32_at = |byte: u64| {
        u32::from_le_bytes(
            memory[byte as usize..byte as usize + 4]
                .try_into()
                .expect("u32"),
        )
    };
    let records = [
        record(indexed(0, A.0 + 3, 3, [1, 0], 10), &[], &[]),
        record(draw(3, 4, 3, [1, 2]), &[], &[]),
        record(indexed(1, A.0 + 65, 6, [2, 5], -1), &[], &[]),
        record(indexed(1, A.0 + 248, 3, [1, 0], 0), &[], &[]),
        record(indexed(0, AWAY, 3, [1, 0], 0), &[], &[]),
        record(indexed(2, A.0, 3, [1, 0], 0), &[], &[]),
        record(indexed(0, 0, 3, [1, 0], 0), &[], &[]),
    ];
    let mut replay = DrawReplay::new(7, 0, 7, vec![3]);
    replay.merged = true;
    replay.indexed = true;
    replay.vertex_slots = 0;
    replay.fragment_slots = 0;
    let ran = preprocess(&replay, &records, clean(7));
    assert_eq!(ran.status.replay_faults, ReplayFaults::RECORD);
    assert_eq!(ran.status.resident_misses, 1, "the missing index buffer");
    let m = ran.pre.layout.merged.expect("merged");
    let entry = |c: u64| {
        let at = (m.indices_offset + 16 * c) as usize;
        let address = u64::from_le_bytes(ran.scratch[at..at + 8].try_into().expect("u64"));
        (
            address,
            word(&ran.scratch, at + 8),
            word(&ran.scratch, at + 12),
        )
    };
    assert_eq!(
        entry(0),
        (A.1 + 2, 127, 2),
        "rounded down to a multiple of 2"
    );
    assert_eq!(entry(1), (0, 0, 0), "a draw of primitives");
    assert_eq!(
        entry(2),
        (A.1 + 64, 48, 4),
        "rounded down to a multiple of 4"
    );
    assert_eq!(entry(3), (A.1 + 248, 2, 4), "two whole indices left");
    for c in 4..7 {
        assert_eq!(entry(c), (0, 0, 0), "command {c} draws nothing");
    }
    assert_eq!(ran.args(0, 0), [3, 1, 10, 0]);
    assert_eq!(ran.args(1, 0), [3, 1, 4, 2]);
    assert_eq!(ran.args(2, 0), [6, 2, u32::MAX, 5]);
    for c in 4..7 {
        assert_eq!(ran.args(c, 0), [0; 4], "command {c} draws nothing");
    }
    let (scratch, status) = merge(&ran, 6);
    assert!(!status.replay_fault, "{status:?}");
    let table = |c: u32| SCRATCH + ran.pre.layout.vertex_table_offset(c);
    let mut expected = Vec::new();
    for v in 0..3 {
        expected.push((u16_at(2 + 2 * v) + 10, 0, 10, 0, table(0), 0));
    }
    for v in 0..3 {
        expected.push((4 + v, 2, 4, 2, table(1), 1));
    }
    for i in 0..2 {
        for v in 0..6u64 {
            let raw = u32_at(64 + 4 * v);
            expected.push((raw.wrapping_sub(1), 5 + i, u32::MAX, 5, table(2), 2));
        }
    }
    for v in 0..3u64 {
        let raw = if v < 2 { u32_at(248 + 4 * v) } else { 0 };
        expected.push((raw, 0, 0, 0, table(3), 3));
    }
    let corners = decoded_over(&ran, &scratch, 6, Some((A.1, &memory)));
    assert_eq!(corners, expected);
}

/// Points and lines merge as triangles do, a corner per point and two per line.
#[test]
fn merged_points_and_lines_run_each_vertex_of_a_whole_primitive() {
    for (ty, per) in [(0u32, 1u32), (1, 2)] {
        let mut replay = DrawReplay::new(3, 0, 3, vec![ty]);
        replay.merged = true;
        replay.vertex_slots = 0;
        replay.fragment_slots = 0;
        let draws = [Some([4, 5, 2, 1]), Some([0, 1, 1, 0]), Some([7, 3, 1, 6])];
        let records: Vec<CommandRecord> = draws
            .iter()
            .flatten()
            .map(|[s, n, k, b]| record(draw(ty, *s, *n, [*k, *b]), &[], &[]))
            .collect();
        let ran = preprocess(&replay, &records, clean(3));
        assert!(!ran.status.replay_fault, "{ty}: {:?}", ran.status);
        let (scratch, _) = merge(&ran, 6);
        assert_eq!(
            decoded(&ran, &scratch, 6),
            metal_corners(&ran, &draws, per),
            "{ty}"
        );
    }
}

/// Corners past what two draws hold (`2^32` chunks) draw nothing and say why.
#[test]
fn a_merged_replay_past_its_draws_draws_nothing_and_says_why() {
    let mut replay = DrawReplay::new(1, 0, 1, vec![3]);
    replay.merged = true;
    replay.fragment_slots = 0;
    let records = [record(draw(3, 0, u32::MAX, [u32::MAX, 0]), &[], &[])];
    let ran = preprocess(&replay, &records, clean(1));
    assert!(!ran.status.replay_fault);
    let (scratch, status) = merge(&ran, MERGED_CHUNK);
    assert_eq!(status.replay_faults, ReplayFaults::VERTICES);
    let m = ran.pre.layout.merged.expect("merged");
    for i in 0..2 {
        let at = m.draw_offset(i) as usize;
        assert_eq!(
            [0, 4, 8, 12].map(|k| word(&scratch, at + k)),
            [0; 4],
            "draw {i}"
        );
    }
}

/// The most whole chunks the merged draw's first draw holds, `2^32 - 1`, draw, the second draw
/// drawing the corners left; a total of `2^32` chunks draws nothing and says why. With chunks
/// of 6, a command of `2^32 - 1` triangle vertices in 6 instances and one of 3 or 6 more corners:
/// `6 · 2^32 - 3` and `6 · 2^32` corners.
#[test]
fn a_merged_replay_of_the_most_chunks_its_first_draw_holds_draws_them() {
    for (more, fits) in [(3u32, true), (6, false)] {
        let mut replay = DrawReplay::new(2, 0, 2, vec![3]);
        replay.merged = true;
        replay.fragment_slots = 0;
        let records = [
            record(draw(3, 0, u32::MAX, [6, 0]), &[], &[]),
            record(draw(3, 0, more, [1, 0]), &[], &[]),
        ];
        let ran = preprocess(&replay, &records, clean(2));
        assert!(!ran.status.replay_fault);
        let (scratch, status) = merge(&ran, 6);
        let m = ran.pre.layout.merged.expect("merged");
        let draws = [0, 1].map(|i| {
            let at = m.draw_offset(i) as usize;
            [0, 4, 8, 12].map(|k| word(&scratch, at + k))
        });
        match fits {
            true => {
                assert!(!status.replay_fault, "{status:?}");
                assert_eq!(draws, [[6, u32::MAX, 0, 0], [3, 1, MERGED_REST_FIRST, 0]]);
            }
            false => {
                assert_eq!(status.replay_faults, ReplayFaults::VERTICES);
                assert_eq!(draws, [[0; 4]; 2]);
            }
        }
    }
}

/// The merge passes divide no `u64`: the last pass divides the total by a chunk in `u32` words
/// (`merged::divide_words`), where NVIDIA's compiler (615.71.09) ran its `u64` division wrong.
#[test]
fn the_merge_passes_divide_no_u64() {
    const TYPE_INT: u32 = 21;
    const UDIV: u32 = 134;
    const UMOD: u32 = 137;
    let dev = device();
    for phase in [MergePhase::Blocks, MergePhase::Starts] {
        let words = draw_merge_module(&dev, phase).expect("module").spirv;
        let mut wide = std::collections::BTreeSet::new();
        let mut divided = 0;
        let mut i = 5;
        while let Some(&w) = words.get(i) {
            let (len, op) = ((w >> 16) as usize, w & 0xFFFF);
            if op == TYPE_INT && words[i + 2] == 64 {
                wide.insert(words[i + 1]);
            }
            if (op == UDIV || op == UMOD) && wide.contains(&words[i + 1]) {
                divided += 1;
            }
            i += len.max(1);
        }
        assert_eq!(divided, 0, "{phase:?}: u64 divisions");
    }
}

/// The merge passes are valid where device addresses are; a merged replay refuses strips,
/// two types and fragment slots; its header and sections follow the arguments.
#[test]
fn merged_replays_are_set_up_for_one_list_type_or_points() {
    for (stem, dev) in devices() {
        if !addressed(&dev) {
            assert!(
                draw_merge_module(&dev, MergePhase::Blocks).is_err(),
                "{stem}"
            );
            continue;
        }
        for phase in [MergePhase::Blocks, MergePhase::Starts] {
            assert_valid(merge_module(phase).expect("module"), dev.profile());
            let m = draw_merge_module(&dev, phase).unwrap_or_else(|e| panic!("{stem}: {e}"));
            assert_eq!(
                m.layout.push_constants[0].size,
                DrawMergeModule::PUSH_BYTES,
                "{stem}"
            );
        }
    }
    let dev = device();
    let pre = |r: &DrawReplay| draw_replay_preprocess(&dev, &state(), r);
    let refused = Err(BindError::InvalidReplay {
        reason: ReplayRefusal::DrawPrimitiveTypes,
    });
    for types in [vec![2], vec![4], vec![3, 4]] {
        let mut r = DrawReplay::new(4, 0, 4, types.clone());
        r.merged = true;
        r.fragment_slots = 0;
        assert_eq!(pre(&r), refused, "{types:?}");
    }
    // A merged replay's fragment variant reads its command's fragment table, which the header
    // places past the vertex table.
    let mut r = DrawReplay::new(4, 0, 4, vec![3]);
    r.merged = true;
    r.vertex_slots = 2;
    r.fragment_slots = 1;
    let p = pre(&r).expect("fragment slots");
    let m = p.layout.merged.expect("layout");
    let header_word = |at: usize| word(&p.header, at);
    assert_eq!(
        u64::from(header_word(52)),
        p.layout.fragment_table_offset(0) - p.layout.vertex_table_offset(0)
    );
    assert_eq!(
        u64::from(header_word(48)),
        m.indices_offset - m.header_offset
    );
    // An indexed replay is merged alone.
    let mut r = DrawReplay::new(4, 0, 4, vec![3]);
    r.indexed = true;
    assert_eq!(
        pre(&r),
        Err(BindError::InvalidReplay {
            reason: ReplayRefusal::IndexedApart,
        })
    );
    let mut r = DrawReplay::new(10, 0, 10, vec![1]);
    r.merged = true;
    r.fragment_slots = 0;
    let p = pre(&r).expect("merged");
    let m = p.layout.merged.expect("layout");
    assert_eq!((m.block_commands, m.blocks), (4, 3));
    assert_eq!(p.header.len() as u64, MERGED_HEADER_BYTES);
    assert_eq!(p.layout.uploaded, m.header_offset + MERGED_HEADER_BYTES);
    assert!(p.layout.arguments_offset < m.prefix_offset);
    assert!(m.prefix_offset + 80 <= m.sums_offset);
    assert!(m.sums_offset + 24 <= m.draws_offset);
    assert_eq!(p.layout.bytes, (m.draws_offset + 32).next_multiple_of(64));
    assert!(pre(&DrawReplay::new(10, 0, 10, vec![1]))
        .expect("plain")
        .layout
        .merged
        .is_none());
}
