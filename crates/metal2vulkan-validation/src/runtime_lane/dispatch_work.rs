//! Device tests of a dispatch's set 0 on the device the cfg selects, under the validation layer:
//! pushed (`vkCmdPushDescriptorSet` at the compute bind point) where the device has the push
//! command, else allocated, written and bound, and the same words either way
//! (`WorkCounts::dispatches_pushing_set_zero`); and what a dispatch settles, taken again by the
//! encoder's next dispatch only where its inputs are the same (`WorkCounts::
//! dispatches_settled_as_before`, `set_zero_sets_reused`). Each runs both ways
//! (`draw_work::on_device_both_ways`). And what a dispatch settles with what the encoder made
//! resident: taken again only while the residency is unchanged, each dispatch's execution stating
//! the resources resident as it was encoded.
//!
//! The kernel is the owned fixture `a4_index_word` (`out[t] = in[3t + 1]`).

use super::draw_work::{on_device_both_ways, pushes};
use super::executions::on_device_with;
use super::indirect_host_steps::state;
use super::{buffer_with, read_buffer, WAIT};
use metal2vulkan::bind::{SamplerDescriptor, UsedClass, UsedShape};
use metal2vulkan::metal::{IndirectCommandType, ResourceUsage};
use metal2vulkan_runtime::{
    CommandBufferStatus, IdPlacement, IndirectCommandBufferDescriptor, ResourceOptions,
    ResourceRef, Size, WorkCounts,
};
use std::collections::BTreeMap;

/// `after`'s count of `count` less `before`'s.
fn grew(before: &WorkCounts, after: &WorkCounts, count: fn(&WorkCounts) -> u64) -> u64 {
    count(after) - count(before)
}

/// Threads a dispatch.
const THREADS: u64 = 16;
/// Dispatches, each into its own words of `out`, from its own words of `in`.
const DISPATCHES: u64 = 4;

/// Dispatches that each bind other offsets of the same buffers push their set 0 where the device
/// pushes it, and bind it where it does not: every dispatch's words are its own, both ways.
#[test]
fn dispatches_push_their_set_zero_where_the_device_pushes() {
    on_device_both_ways(|device| {
        let kernel = state(device, "a4_index_word", false);
        let words = 3 * THREADS * DISPATCHES;
        let input: Vec<u8> = (0..words as u32).flat_map(u32::to_le_bytes).collect();
        let input = buffer_with(device, &input).unwrap();
        let out = buffer_with(device, &vec![0; (4 * THREADS * DISPATCHES) as usize]).unwrap();
        let queue = device.new_command_queue().unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        encoder.set_compute_pipeline_state(&kernel).unwrap();
        let before = device.work_counts();
        for d in 0..DISPATCHES {
            encoder
                .set_buffer(Some(&input), 4 * 3 * THREADS * d, 0)
                .unwrap();
            encoder.set_buffer(Some(&out), 4 * THREADS * d, 1).unwrap();
            encoder
                .dispatch_threads(Size::new(THREADS, 1, 1), Size::new(THREADS, 1, 1))
                .unwrap();
        }
        let pushed =
            device.work_counts().dispatches_pushing_set_zero - before.dispatches_pushing_set_zero;
        encoder.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
        assert_eq!(pushed, if pushes(device) { DISPATCHES } else { 0 });
        let words: Vec<u32> = read_buffer(&out, 0, 4 * THREADS * DISPATCHES)
            .unwrap()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|word| u32::from_le_bytes(*word))
            .collect();
        let want: Vec<u32> = (0..(THREADS * DISPATCHES) as u32)
            .map(|t| 3 * t + 1)
            .collect();
        assert_eq!(words, want);
    });
}

/// What one dispatch of [`dispatches_settle_again_only_where_their_inputs_changed`] changed
/// before it ran, and the `out` word it writes from.
struct Step {
    change: &'static str,
    at: u64,
    threads: u64,
    /// Whether it takes what the dispatch before it settled.
    as_before: bool,
}

/// A compute encoder's dispatch takes the bind state, variant, plan, commands and made pipelines
/// its last dispatch settled only where it has the same inputs
/// (`WorkCounts::dispatches_settled_as_before`): a dispatch again of an unchanged state takes them;
/// one after a change of a buffer's offset, of the grid, after a sampler state was made, of another
/// pipeline state of the same kernel, or after a buffer was made resident, settles again
/// ([`dispatches_take_what_they_settled_only_with_the_same_residency`]). Set 0 is apart from that
/// memo: a dispatch binds the last set 0 wherever it writes the same descriptors into a set of the
/// same layout (`WorkCounts::set_zero_sets_reused`), so each dispatch that settles as before binds
/// it, where the device binds set 0, and the first does not; whether a changed offset or grid
/// changes the set's writes is the variant's (an offset may be push data). Every dispatch writes
/// its words. Both ways ([`on_device_both_ways`]).
#[test]
fn dispatches_settle_again_only_where_their_inputs_changed() {
    on_device_both_ways(|device| {
        let kernel = state(device, "a4_index_word", false);
        let other = state(device, "a4_index_word", false);
        let input: Vec<u8> = (0..3 * 4 * THREADS as u32)
            .flat_map(u32::to_le_bytes)
            .collect();
        let input = buffer_with(device, &input).unwrap();
        const WORDS: u64 = 4 * THREADS;
        let out = buffer_with(device, &vec![0; (4 * WORDS) as usize]).unwrap();
        let queue = device.new_command_queue().unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        encoder.set_compute_pipeline_state(&kernel).unwrap();
        encoder.set_buffer(Some(&input), 0, 0).unwrap();
        encoder.set_buffer(Some(&out), 0, 1).unwrap();
        let steps = [
            Step {
                change: "the first",
                at: 0,
                threads: THREADS,
                as_before: false,
            },
            Step {
                change: "nothing",
                at: 0,
                threads: THREADS,
                as_before: true,
            },
            Step {
                change: "out's offset",
                at: THREADS,
                threads: THREADS,
                as_before: false,
            },
            Step {
                change: "the grid",
                at: THREADS,
                threads: 2 * THREADS,
                as_before: false,
            },
            Step {
                change: "nothing again",
                at: THREADS,
                threads: 2 * THREADS,
                as_before: true,
            },
            Step {
                change: "a sampler state made",
                at: THREADS,
                threads: 2 * THREADS,
                as_before: false,
            },
            Step {
                change: "the pipeline state",
                at: THREADS,
                threads: 2 * THREADS,
                as_before: false,
            },
            Step {
                change: "a buffer made resident",
                at: THREADS,
                threads: 2 * THREADS,
                as_before: false,
            },
        ];
        let mut sampler = None;
        let mut settled = Vec::new();
        for step in &steps {
            match step.change {
                "out's offset" => encoder.set_buffer_offset(4 * step.at, 1).unwrap(),
                "the pipeline state" => encoder.set_compute_pipeline_state(&other).unwrap(),
                "a sampler state made" => {
                    sampler = Some(device.new_sampler_state(&SamplerDescriptor::new()).unwrap());
                }
                "a buffer made resident" => {
                    encoder
                        .use_resource(ResourceRef::Buffer(&input), ResourceUsage::READ)
                        .unwrap();
                }
                _ => {}
            }
            let before = device.work_counts();
            encoder
                .dispatch_threads(Size::new(step.threads, 1, 1), Size::new(THREADS, 1, 1))
                .unwrap();
            let after = device.work_counts();
            let as_before = grew(&before, &after, |c| c.dispatches_settled_as_before);
            let reused = grew(&before, &after, |c| c.set_zero_sets_reused);
            settled.push((step.change, as_before, reused));
        }
        drop(sampler);
        encoder.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
        let binds = !pushes(device);
        let as_before: Vec<_> = (settled.iter())
            .map(|&(change, as_before, _)| (change, as_before))
            .collect();
        let want: Vec<_> = (steps.iter())
            .map(|step| (step.change, u64::from(step.as_before)))
            .collect();
        assert_eq!(as_before, want);
        for (step, &(change, _, reused)) in steps.iter().zip(&settled) {
            match (step.change, step.as_before) {
                ("the first", _) => assert_eq!(reused, 0, "{change}"),
                (_, true) => assert_eq!(reused, u64::from(binds), "{change}"),
                (_, false) => assert!(reused <= u64::from(binds), "{change}"),
            }
        }
        let words: Vec<u32> = read_buffer(&out, 0, 4 * WORDS)
            .unwrap()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|word| u32::from_le_bytes(*word))
            .collect();
        let mut expected = vec![0u32; WORDS as usize];
        for step in &steps {
            for t in 0..step.threads {
                expected[(step.at + t) as usize] = 3 * t as u32 + 1;
            }
        }
        assert_eq!(words, expected);
    });
}

/// What one dispatch of [`dispatches_take_what_they_settled_only_with_the_same_residency`]
/// changed before it ran, the usages resident as it was encoded, by token, and whether it takes
/// what the dispatch before it settled.
struct Resident {
    change: &'static str,
    usages: &'static [ResourceUsage],
    as_before: bool,
}

/// A compute encoder's dispatch takes what its last dispatch settled with what the encoder made
/// resident only while that is unchanged (`WorkCounts::dispatches_settled_as_before`), and each
/// dispatch's execution states the resources resident as it was encoded: a dispatch after a
/// resource made resident, a usage widened, or a buffer made later at a resident indirect command
/// buffer's ID (which then reaches it no more) settles again with the residency of then, and the
/// next dispatch takes that; a resource made resident again with a usage it had changes nothing.
/// Every dispatch writes its words, those that settled as before as those that did not.
#[test]
fn dispatches_take_what_they_settled_only_with_the_same_residency() {
    const READ: ResourceUsage = ResourceUsage::READ;
    const BOTH: ResourceUsage = ResourceUsage::READ.union(ResourceUsage::WRITE);
    on_device_with(
        |options| options.record_executions = true,
        |device| {
            let kernel = state(device, "a4_index_word", false);
            let words = 3 * THREADS;
            let input: Vec<u8> = (0..words as u32).flat_map(u32::to_le_bytes).collect();
            let input = buffer_with(device, &input).unwrap();
            let out = buffer_with(device, &vec![0; (4 * THREADS) as usize]).unwrap();
            let first = buffer_with(device, &[0; 64]).unwrap();
            let second = buffer_with(device, &[0; 64]).unwrap();
            let mut descriptor = IndirectCommandBufferDescriptor::new();
            descriptor.command_types = IndirectCommandType::CONCURRENT_DISPATCH;
            descriptor.max_kernel_buffer_bind_count = 2;
            let icb = device
                .new_indirect_command_buffer(&descriptor, 4, ResourceOptions::shared())
                .unwrap();
            let mut taken = None;
            let queue = device.new_command_queue().unwrap();
            let command_buffer = queue.command_buffer().unwrap();
            let mut encoder = command_buffer.compute_command_encoder().unwrap();
            encoder.set_compute_pipeline_state(&kernel).unwrap();
            encoder.set_buffer(Some(&input), 0, 0).unwrap();
            encoder.set_buffer(Some(&out), 0, 1).unwrap();
            let steps = [
                Resident {
                    change: "the first",
                    usages: &[],
                    as_before: false,
                },
                Resident {
                    change: "nothing",
                    usages: &[],
                    as_before: true,
                },
                Resident {
                    change: "a buffer made resident",
                    usages: &[READ],
                    as_before: false,
                },
                Resident {
                    change: "nothing, it resident",
                    usages: &[READ],
                    as_before: true,
                },
                Resident {
                    change: "it made resident again with its usage",
                    usages: &[READ],
                    as_before: true,
                },
                Resident {
                    change: "its usage widened",
                    usages: &[BOTH],
                    as_before: false,
                },
                Resident {
                    change: "another buffer made resident",
                    usages: &[BOTH, READ],
                    as_before: false,
                },
                Resident {
                    change: "nothing, both resident",
                    usages: &[BOTH, READ],
                    as_before: true,
                },
                Resident {
                    change: "an indirect command buffer made resident",
                    usages: &[BOTH, READ, READ],
                    as_before: false,
                },
                Resident {
                    change: "nothing, it resident too",
                    usages: &[BOTH, READ, READ],
                    as_before: true,
                },
                Resident {
                    change: "a buffer made at its ID",
                    usages: &[BOTH, READ, READ],
                    as_before: false,
                },
                Resident {
                    change: "nothing, its ID taken",
                    usages: &[BOTH, READ, READ],
                    as_before: true,
                },
            ];
            let mut settled = Vec::new();
            for step in &steps {
                match step.change {
                    "a buffer made resident" | "it made resident again with its usage" => encoder
                        .use_resource(ResourceRef::Buffer(&first), READ)
                        .unwrap(),
                    "its usage widened" => encoder
                        .use_resource(ResourceRef::Buffer(&first), ResourceUsage::WRITE)
                        .unwrap(),
                    "another buffer made resident" => encoder
                        .use_resources(&[ResourceRef::Buffer(&second)], READ)
                        .unwrap(),
                    "an indirect command buffer made resident" => {
                        encoder.use_resource(icb.as_resource(), READ).unwrap();
                    }
                    "a buffer made at its ID" => {
                        let mut placed = descriptor;
                        placed.placement = IdPlacement::At(icb.gpu_resource_id());
                        let made = device
                            .new_indirect_command_buffer(&placed, 4, ResourceOptions::shared())
                            .unwrap();
                        taken = Some(made);
                    }
                    _ => {}
                }
                let before = device.work_counts();
                encoder
                    .dispatch_threads(Size::new(THREADS, 1, 1), Size::new(THREADS, 1, 1))
                    .unwrap();
                let after = device.work_counts();
                settled.push((
                    step.change,
                    grew(&before, &after, |c| c.dispatches_settled_as_before),
                ));
            }
            encoder.end_encoding().unwrap();
            command_buffer.commit().unwrap();
            command_buffer.wait_until_completed_for(WAIT).unwrap();
            assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
            drop(taken);
            let want: Vec<_> = (steps.iter())
                .map(|step| (step.change, u64::from(step.as_before)))
                .collect();
            assert_eq!(settled, want);
            let executions = command_buffer.executions();
            assert_eq!(executions.len(), steps.len());
            for (step, execution) in steps.iter().zip(&executions) {
                let want: BTreeMap<u32, (UsedClass, ResourceUsage)> = (0u32..)
                    .zip(step.usages)
                    .map(|(token, &usage)| (token, (UsedClass::Buffer, usage)))
                    .collect();
                let used: BTreeMap<u32, (UsedClass, ResourceUsage)> = (execution.shape().used)
                    .iter()
                    .map(|(&token, used): (&u32, &UsedShape)| (token, (used.class, used.usage)))
                    .collect();
                assert_eq!(used, want, "{}", step.change);
            }
            let words: Vec<u32> = read_buffer(&out, 0, 4 * THREADS)
                .unwrap()
                .as_chunks::<4>()
                .0
                .iter()
                .map(|word| u32::from_le_bytes(*word))
                .collect();
            let want: Vec<u32> = (0..THREADS as u32).map(|t| 3 * t + 1).collect();
            assert_eq!(words, want);
        },
    );
}

/// What one dispatch of [`dispatches_take_what_their_state_planned_for_the_same_shape`] changed
/// before it ran, where it writes, and whether it took what its state's last dispatch that planned
/// planned (`WorkCounts::dispatches_planned_as_before`) or what the encoder's last dispatch
/// settled (`dispatches_settled_as_before`).
struct Planning {
    change: &'static str,
    at: u64,
    threads: u64,
    planned: bool,
    settled: bool,
}

/// A dispatch whose bind state has the shape one of its pipeline state's latest direct dispatches
/// planned, of the same grid, takes that one's variant, plan and made pipelines: one whose buffer's
/// offset alone moved, and another encoder's first dispatch of the state. A grid or a resource
/// made resident plans again, and the next dispatch of the new shape takes that; and an encoder of
/// the first shape again, after one of the other planned, takes what that first shape planned, which
/// the state keeps beside the other (`pipeline::Planned`), where it kept one alone before. Every
/// dispatch writes its words. Both ways ([`on_device_both_ways`]).
#[test]
fn dispatches_take_what_their_state_planned_for_the_same_shape() {
    on_device_both_ways(|device| {
        let kernel = state(device, "a4_index_word", false);
        let input: Vec<u8> = (0..3 * 2 * THREADS as u32)
            .flat_map(u32::to_le_bytes)
            .collect();
        let input = buffer_with(device, &input).unwrap();
        const WORDS: u64 = 14 * THREADS;
        let out = buffer_with(device, &vec![0; (4 * WORDS) as usize]).unwrap();
        let queue = device.new_command_queue().unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        let step = |change, at, threads, (planned, settled)| Planning {
            change,
            at,
            threads,
            planned,
            settled,
        };
        let encoders = [
            vec![
                step("the first", 0, THREADS, (false, false)),
                step("out's offset", THREADS, THREADS, (true, false)),
                step("the grid", 2 * THREADS, 2 * THREADS, (false, false)),
                step(
                    "out's offset again",
                    4 * THREADS,
                    2 * THREADS,
                    (true, false),
                ),
            ],
            vec![
                step("another encoder", 6 * THREADS, 2 * THREADS, (true, false)),
                step(
                    "a buffer made resident",
                    8 * THREADS,
                    2 * THREADS,
                    (false, false),
                ),
                step(
                    "out's offset, resident",
                    10 * THREADS,
                    2 * THREADS,
                    (true, false),
                ),
                step("nothing", 10 * THREADS, 2 * THREADS, (false, true)),
            ],
            vec![step(
                "an encoder of the first shape",
                12 * THREADS,
                2 * THREADS,
                (true, false),
            )],
        ];
        let mut seen = Vec::new();
        for steps in &encoders {
            let mut encoder = command_buffer.compute_command_encoder().unwrap();
            encoder.set_compute_pipeline_state(&kernel).unwrap();
            encoder.set_buffer(Some(&input), 0, 0).unwrap();
            let mut bound = None;
            for step in steps {
                // Bound again only where it moved: a call that binds draws a new table stamp.
                if bound != Some(step.at) {
                    encoder.set_buffer(Some(&out), 4 * step.at, 1).unwrap();
                    bound = Some(step.at);
                }
                if step.change == "a buffer made resident" {
                    encoder
                        .use_resource(ResourceRef::Buffer(&input), ResourceUsage::READ)
                        .unwrap();
                }
                let before = device.work_counts();
                encoder
                    .dispatch_threads(Size::new(step.threads, 1, 1), Size::new(THREADS, 1, 1))
                    .unwrap();
                let after = device.work_counts();
                let planned = grew(&before, &after, |c| c.dispatches_planned_as_before);
                let settled = grew(&before, &after, |c| c.dispatches_settled_as_before);
                seen.push((step.change, planned, settled));
            }
            encoder.end_encoding().unwrap();
        }
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
        let steps: Vec<&Planning> = encoders.iter().flatten().collect();
        let want: Vec<_> = (steps.iter())
            .map(|step| {
                let (planned, settled) = (u64::from(step.planned), u64::from(step.settled));
                (step.change, planned, settled)
            })
            .collect();
        assert_eq!(seen, want);
        let words: Vec<u32> = read_buffer(&out, 0, 4 * WORDS)
            .unwrap()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|word| u32::from_le_bytes(*word))
            .collect();
        let mut expected = vec![0u32; WORDS as usize];
        for step in &steps {
            for t in 0..step.threads {
                expected[(step.at + t) as usize] = 3 * t as u32 + 1;
            }
        }
        assert_eq!(words, expected);
    });
}

/// The buffers [`dispatches_list_what_their_encoder_made_resident_once_a_change`] makes resident.
const RESIDENT: usize = 64;

/// A compute encoder's dispatches bring the bind state its last dispatch bound in place: where only
/// a buffer's offset moved between them, after the encoder made [`RESIDENT`] buffers resident, the
/// first lists them as its bind state's used resources and finds or makes its shape, and none after
/// it does either (`WorkCounts::dispatch_used_resources_listed`, `dispatch_shapes_made` and
/// `dispatch_shapes_found`), each taking what its pipeline state planned for that shape
/// (`dispatches_planned_as_before`); a buffer made resident lists them again at the next dispatch,
/// which finds or makes its shape again, and the dispatch after it neither. Every dispatch writes
/// its words. Both ways ([`on_device_both_ways`]).
#[test]
fn dispatches_list_what_their_encoder_made_resident_once_a_change() {
    const MOVES: u64 = 6;
    on_device_both_ways(|device| {
        let kernel = state(device, "a4_index_word", false);
        let input: Vec<u8> = (0..3 * THREADS as u32).flat_map(u32::to_le_bytes).collect();
        let input = buffer_with(device, &input).unwrap();
        let words = THREADS * (MOVES + 2);
        let out = buffer_with(device, &vec![0; (4 * words) as usize]).unwrap();
        let resident: Vec<_> = (0..=RESIDENT)
            .map(|_| buffer_with(device, &[0; 64]).unwrap())
            .collect();
        let queue = device.new_command_queue().unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        encoder.set_compute_pipeline_state(&kernel).unwrap();
        encoder.set_buffer(Some(&input), 0, 0).unwrap();
        let first: Vec<_> = resident[..RESIDENT]
            .iter()
            .map(ResourceRef::Buffer)
            .collect();
        encoder.use_resources(&first, ResourceUsage::READ).unwrap();
        // Per dispatch: what it changed, and whether it listed the resources, found or made its
        // shape, and took what its state planned.
        let mut seen = Vec::new();
        for at in 0..MOVES + 2 {
            let change = match at {
                0 => "the first",
                MOVES => "a buffer made resident",
                _ => "out's offset",
            };
            if at == MOVES {
                let last = ResourceRef::Buffer(&resident[RESIDENT]);
                encoder.use_resource(last, ResourceUsage::READ).unwrap();
            }
            encoder.set_buffer(Some(&out), 4 * THREADS * at, 1).unwrap();
            let before = device.work_counts();
            encoder
                .dispatch_threads(Size::new(THREADS, 1, 1), Size::new(THREADS, 1, 1))
                .unwrap();
            let after = device.work_counts();
            let listed = grew(&before, &after, |c| c.dispatch_used_resources_listed);
            let shaped = grew(&before, &after, |c| c.dispatch_shapes_made)
                + grew(&before, &after, |c| c.dispatch_shapes_found);
            let planned = grew(&before, &after, |c| c.dispatches_planned_as_before);
            seen.push((change, listed, shaped, planned));
        }
        encoder.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
        // The first of each residency lists it and shapes its state; whether it then plans is the
        // pipeline state's (none of the first shape planned yet; the second's neither).
        let want: Vec<_> = (0..MOVES + 2)
            .map(|at| match at {
                0 => ("the first", 1, 1, 0),
                MOVES => ("a buffer made resident", 1, 1, 0),
                _ => ("out's offset", 0, 0, 1),
            })
            .collect();
        assert_eq!(seen, want);
        let words: Vec<u32> = read_buffer(&out, 0, 4 * words)
            .unwrap()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|word| u32::from_le_bytes(*word))
            .collect();
        let want: Vec<u32> = (0..MOVES + 2)
            .flat_map(|_| (0..THREADS as u32).map(|t| 3 * t + 1))
            .collect();
        assert_eq!(words, want);
    });
}
