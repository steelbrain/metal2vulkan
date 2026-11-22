//! The `CpuInterp` executor of the generated-program fuzzers: the engine's typed module on the
//! CPU SPIR-V interpreter, and the ordinary gate's run of every generator over [`GATE_SEEDS`] on
//! it. The generators and their Rust reference models are the dev-tools crate's
//! (`metal2vulkan_devtools::fuzz`); validation runs the same programs on a Vulkan device. The
//! interpreter is this crate's test support, so the executor is one of this crate's tests;
//! `fuzz_soak` soaks it.
//!
//! For each kernel: `llvm-as` assembles the generated text into bitcode; the engine compiles
//! it ONCE for the executor's profile, keeping the finished typed module whose serialization the
//! artifact holds (`compile_bitcode_kept`); the artifact is linked, relocated, and validated (L3:
//! requirements within the profile, `spirv-val` under the profile's environments); and that same
//! module runs on the interpreter at every subgroup width the profile's kernel stage may run with
//! ([`widths`]), in both scheduling orders, all of which must agree.
//!
//! This module plays the binder's part for the one shape the generators produce -- one
//! threadgroup of [`Target::threads`] threads over plain `device` buffers bound at their start:
//! buffer `i` backs `ResourceSlot { Kernel, Buffer, i }`, every `SlotBindOffset` residual is
//! zero, and the local-size specialization constants are set to the dispatch. Every buffer is
//! also reachable at a device address and resident at its guest address ([`guest_address`]): a
//! function that reads the root table (`RootTableAddress`) gets one with an entry `2 + i` for
//! buffer `i` and every buffer in the resident set, and a function that counts resident-set misses
//! gets the `ExecutionStatus` at its device address. A static-root slot's `SlotGuestBase` (a
//! pointer compare) is its buffer's guest address, and a pointer-table input ([`Input::Addresses`])
//! holds the listed buffers' guest addresses. Any other push field is a failure, not a guess.
//!
//! **Every status word is checked** ([`check_status`]): after the run the `ExecutionStatus`
//! buffer must hold exactly what a run that misses nothing leaves, every word 0. No execution
//! bound stops a run: the interpreter's [`STEP_LIMIT`] is what fails a program that does not
//! terminate.

use std::collections::BTreeMap;

use crate::budget::SPIRV_WORDS;
use crate::device::{DevicePolicy, DeviceProfile, SubgroupSizes};
use crate::pipeline::compile::compile_bitcode_kept;
use crate::pipeline::testing::{link_and_validate, meter, profile};
use crate::pipeline::{
    Axis, EntryRef, LoweringInputs, PushField, PushLoc, ResourceKind, ResourceSlot, SlotClass,
    SlotRef, SpecKey, Stage,
};
use crate::spirv::interp::{
    run, BarrierRule, Buffer, Config, Order, Outcome, Resources, SubgroupSize,
};
use metal2vulkan_devtools::fuzz::{
    access, bytes_words, inline, irreducible, pointer_select, structured, word_bytes, workgroup,
    Execute, Input, Kernel, Run, Target, GATE_SEEDS, THREADS, THREAD_COUNTS,
};
use metal2vulkan_devtools::{llvm_assemble_text, AssembleError};

/// The committed device profile the ordinary gate compiles the generated programs for: a
/// desktop-class profile with 8/16/64-bit integers, so a refusal names a missing lowering rather
/// than a missing capability. The soak (`fuzz_soak`) may select another.
pub(super) const PROFILE: &str = "moltenvk-m3";

/// Lane-steps the interpreter may take per run before it traps (`TrapKind::StepLimit`). The
/// deepest generated program (a five-deep nest of four-trip loops, `structured`, at 64 threads)
/// takes far fewer; a program that reaches it does not terminate and fails.
const STEP_LIMIT: u64 = 1 << 26;

/// Device addresses of the stand-in binder's address-backed objects.
const TABLE: u64 = 0x0000_0010_0000_0000;
const SINK: u64 = 0x0000_0011_0000_0000;
const STATUS: u64 = 0x0000_0012_0000_0000;

/// The Metal (guest) address of kernel buffer `buffer` (0 is the output, `1..` the inputs) in
/// this binder stand-in: every buffer is resident at this address, so a program that loads it
/// from a pointer table reaches that buffer through the resident-set search.
fn guest_address(buffer: usize) -> u64 {
    0x0000_0001_0000_0000 + (buffer as u64) * 0x0100_0000
}

/// The device address of kernel buffer `buffer`.
fn host_address(buffer: usize) -> u64 {
    0x0000_0020_0000_0000 + (buffer as u64) * 0x1_0000_0000
}

/// `bytes` padded to a multiple of 16 (the allocation padding contract).
fn padded(mut bytes: Vec<u8>) -> Vec<u8> {
    bytes.resize(bytes.len().next_multiple_of(16), 0);
    bytes
}

/// The root table of `sizes` (padded buffer sizes by kernel buffer): entry 0 null, entry 1 the
/// sink with the resident range, `2 + i` buffer `i`, then every buffer again as the resident set
/// (sorted by guest address, which rises with `i`).
fn root_table(sizes: &[usize]) -> Vec<u8> {
    let first = 2 + sizes.len();
    let mut table = vec![0u8; 32 * (first + sizes.len())];
    let mut put = |root: usize, field: usize, value: u64| {
        let at = 32 * root + 8 * field;
        table[at..at + 8].copy_from_slice(&value.to_le_bytes());
    };
    put(1, 0, SINK);
    put(1, 1, first as u64);
    put(1, 2, sizes.len() as u64);
    for (i, size) in sizes.iter().enumerate() {
        for root in [2 + i, first + i] {
            put(root, 0, host_address(i));
            put(root, 1, guest_address(i));
            put(root, 2, *size as u64);
        }
    }
    table
}

/// The bytes of input `input` in this binder stand-in.
fn input_bytes(input: &Input) -> Vec<u8> {
    match input {
        Input::Words(words) => word_bytes(words),
        Input::Addresses(buffers) => buffers
            .iter()
            .flat_map(|&buffer| guest_address(buffer).to_le_bytes())
            .collect(),
    }
}

/// The subgroup widths the interpreter runs a module at: the width its requirement pins, else every
/// width the profile's kernel stage may run with (`sizes`, the powers of two from `min` to `max`)
/// that the interpreter models (8 to 64). A stage whose sizes are all narrower than 8 (`vk12-floor`
/// reports 1, SwiftShader 4) runs at the narrowest modelled width, `W8`; a pinned width the
/// interpreter does not model is an error.
fn widths(sizes: SubgroupSizes, pinned: Option<u32>) -> Result<Vec<SubgroupSize>, String> {
    const MODELLED: [SubgroupSize; 4] = [
        SubgroupSize::W8,
        SubgroupSize::W16,
        SubgroupSize::W32,
        SubgroupSize::W64,
    ];
    if let Some(size) = pinned {
        return MODELLED
            .into_iter()
            .find(|width| width.lanes() == size)
            .map(|width| vec![width])
            .ok_or_else(|| format!("the module pins subgroup size {size}, which is not modelled"));
    }
    let within: Vec<SubgroupSize> = MODELLED
        .into_iter()
        .filter(|width| (sizes.min..=sizes.max).contains(&width.lanes()))
        .collect();
    Ok(
        if within.is_empty() && sizes.max < SubgroupSize::W8.lanes() {
            vec![SubgroupSize::W8]
        } else {
            within
        },
    )
}

/// The `ExecutionStatus` buffer's 32-bit words.
const STATUS_WORDS: usize = crate::bind::status::StatusWord::SeqCstFences.index() as usize + 1;

/// Check every word of the `ExecutionStatus` buffer `bytes` after a run: a run that misses
/// nothing leaves every word 0 (no status bit, no resident-set miss, reserved words untouched).
/// Any other value fails the run.
fn check_status(bytes: &[u8]) -> Result<(), String> {
    let words = bytes_words(bytes);
    if words.len() != STATUS_WORDS {
        return Err(format!(
            "the status buffer holds {} words, wanted {}",
            words.len(),
            STATUS_WORDS
        ));
    }
    if words.iter().any(|&word| word != 0) {
        return Err(format!(
            "status words {words:?}: a run that misses nothing leaves every word 0"
        ));
    }
    Ok(())
}

/// Runs the engine's output on the CPU interpreter.
pub(super) struct CpuInterp {
    profile: DeviceProfile,
    threads: u32,
}

impl CpuInterp {
    /// The executor for committed profile `stem` (`validation/profiles/<stem>.json`, derived
    /// under the resident-table domains), dispatching [`THREADS`] threads. `Ok(None)` when
    /// `llvm-as` is not installed: the caller skips visibly, because there is no way to turn
    /// generated text into bitcode.
    pub(super) fn new(stem: &str) -> Result<Option<CpuInterp>, String> {
        let profile = profile(stem, DevicePolicy::RESIDENT_TABLES)?;
        match llvm_assemble_text("define void @probe() {\n  ret void\n}\n") {
            Ok(_) => Ok(Some(CpuInterp {
                profile,
                threads: THREADS,
            })),
            Err(AssembleError::Missing(why)) => {
                eprintln!("SKIPPED: {why}");
                Ok(None)
            }
            Err(AssembleError::Rejected(why)) => Err(format!("llvm-as probe: {why}")),
        }
    }

    /// This executor dispatching `threads` threads, one of [`THREAD_COUNTS`].
    pub(super) fn with_threads(self, threads: u32) -> Result<CpuInterp, String> {
        if !THREAD_COUNTS.contains(&threads) {
            return Err(format!(
                "{threads} threads: the generators build for {THREAD_COUNTS:?}"
            ));
        }
        Ok(CpuInterp { threads, ..self })
    }

    fn assemble(&self, kernel: &Kernel<'_>) -> Result<Vec<u8>, String> {
        llvm_assemble_text(kernel.air).map_err(|error| match error {
            AssembleError::Missing(why) => format!("llvm-as: {why}"),
            // The generator wrote text LLVM rejects: a generator bug, reported as such.
            AssembleError::Rejected(why) => format!("llvm-as rejected the generated text: {why}"),
        })
    }
}

fn slot(index: u32) -> ResourceSlot {
    ResourceSlot {
        stage: Stage::Kernel,
        kind: ResourceKind::Buffer,
        index,
    }
}

impl Execute for CpuInterp {
    fn target(&self) -> Target {
        Target {
            threads: self.threads,
            addressed: self.profile.addressed_needs().is_empty(),
        }
    }

    fn execute(&mut self, kernel: &Kernel<'_>) -> Result<Run, String> {
        if kernel.threads != self.threads {
            return Err(format!(
                "the kernel was built for {} threads, the executor dispatches {}",
                kernel.threads, self.threads
            ));
        }
        let bitcode = self.assemble(kernel)?;
        let profile = &self.profile;
        // One compile: the module the interpreter runs is the one whose serialization the
        // artifact holds and L3 validates.
        let (compiled, plan, module) = compile_bitcode_kept(
            &bitcode,
            EntryRef::Name(kernel.entry),
            profile,
            &LoweringInputs::new(),
            &mut meter(),
            SPIRV_WORDS,
        )
        .map_err(|error| format!("compile: {error}"))?;
        link_and_validate(compiled.artifact(), profile).map_err(|error| format!("L3: {error}"))?;
        let widths = widths(
            profile.subgroups().kernel.sizes,
            compiled.required_subgroup_size(),
        )?;

        let mut push = vec![0u8; plan.push.push_range.map_or(0, |(_, end)| end as usize)];
        let mut reads_table = false;
        let mut status_by_address = false;
        for field in &plan.push.fields {
            let address = match (field.field, field.loc) {
                (PushField::RootTableAddress, PushLoc::Push { offset }) => {
                    reads_table = true;
                    (offset, TABLE)
                }
                (PushField::ExecutionStatusAddress, PushLoc::Push { offset }) => {
                    status_by_address = true;
                    (offset, STATUS)
                }
                _ => continue,
            };
            let at = address.0 as usize;
            push.get_mut(at..at + 8)
                .ok_or_else(|| format!("address at {at} outside the push range"))?
                .copy_from_slice(&address.1.to_le_bytes());
        }
        for field in &plan.push.fields {
            match (field.field, field.loc) {
                (
                    PushField::RootTableAddress | PushField::ExecutionStatusAddress,
                    PushLoc::Push { .. },
                ) => {}
                // Every buffer is bound at its start: a zero residual.
                (PushField::SlotBindOffset(_), PushLoc::Push { offset }) => {
                    let at = offset as usize;
                    push.get_mut(at..at + 4)
                        .ok_or_else(|| format!("residual at {at} outside the push range"))?
                        .fill(0);
                }
                // A static-root slot's guest base: its buffer's guest address, bound at its start.
                (
                    PushField::SlotGuestBase(SlotRef {
                        stage: Stage::Kernel,
                        class: SlotClass::Buffer,
                        index,
                    }),
                    PushLoc::Push { offset },
                ) if (index as usize) <= kernel.inputs.len() => {
                    let at = offset as usize;
                    push.get_mut(at..at + 8)
                        .ok_or_else(|| format!("guest base at {at} outside the push range"))?
                        .copy_from_slice(&guest_address(index as usize).to_le_bytes());
                }
                // A single-region dispatch: `SpecKey::MultiRegion` keeps its default 0, so the
                // region words are never read.
                (PushField::DispatchRegion, PushLoc::Push { .. }) => {}
                other => return Err(format!("unexpected push field {other:?}")),
            }
        }
        let mut spec = BTreeMap::new();
        spec.insert(SpecKey::LocalSize(Axis::X), u64::from(self.threads));
        spec.insert(SpecKey::LocalSize(Axis::Y), 1);
        spec.insert(SpecKey::LocalSize(Axis::Z), 1);
        spec.insert(SpecKey::FlatLocalSize, u64::from(self.threads));

        // Every kernel buffer backs its slot and is reachable at its device address.
        let contents = std::iter::once(kernel.output_init.clone())
            .chain(kernel.inputs.iter().map(input_bytes))
            .map(padded);
        let mut buffers = Vec::new();
        for (index, bytes) in contents.enumerate() {
            buffers.push(Buffer {
                slot: Some(slot(
                    u32::try_from(index).map_err(|error| error.to_string())?,
                )),
                address: Some(host_address(index)),
                bytes,
            });
        }
        if reads_table {
            let sizes: Vec<usize> = buffers.iter().map(|b| b.bytes.len()).collect();
            buffers.push(Buffer {
                slot: None,
                address: Some(TABLE),
                bytes: root_table(&sizes),
            });
            buffers.push(Buffer {
                slot: None,
                address: Some(SINK),
                bytes: vec![0; 64],
            });
        }
        // A module that counts resident-set misses reads its `ExecutionStatus` by device address;
        // every word is checked below.
        let status = status_by_address.then(|| {
            buffers.push(Buffer {
                slot: None,
                address: Some(STATUS),
                bytes: vec![0; 4 * STATUS_WORDS],
            });
            buffers.len() - 1
        });
        let resources = Resources {
            buffers,
            push,
            spec,
        };

        // Every width the device may run the module at, each in both orders: one answer.
        let mut first: Option<(SubgroupSize, Order, Outcome)> = None;
        for &subgroup in &widths {
            for order in [Order::Forward, Order::Reverse] {
                let config = Config {
                    workgroups: [1, 1, 1],
                    subgroup,
                    order,
                    step_limit: STEP_LIMIT,
                    barriers: BarrierRule::Q3,
                    first: [0, 0],
                };
                let outcome = run(&module, &config, resources.clone()).map_err(|error| {
                    format!("interpreter ({subgroup:?}, {order:?} order): {error}")
                })?;
                match &first {
                    None => first = Some((subgroup, order, outcome)),
                    Some((width, earlier, agreed)) if agreed.buffers != outcome.buffers => {
                        return Err(format!(
                            "{width:?} in {earlier:?} order and {subgroup:?} in {order:?} order \
                             disagree: the program races or depends on the subgroup width"
                        ))
                    }
                    Some(_) => {}
                }
            }
        }
        let Some((_, _, Outcome { buffers, .. })) = first else {
            return Err(format!(
                "the profile's kernel stage runs no subgroup width the interpreter models \
                 ({:?})",
                profile.subgroups().kernel.sizes
            ));
        };
        if let Some(at) = status {
            check_status(&buffers[at])?;
        }
        let mut output = buffers.into_iter().next().unwrap_or_default();
        output.truncate(kernel.output_init.len());
        if output.len() != kernel.output_init.len() {
            return Err(format!(
                "read back {} bytes, wanted {}",
                output.len(),
                kernel.output_init.len()
            ));
        }
        Ok(Run {
            words: bytes_words(&output),
        })
    }
}

/// The executor for [`PROFILE`] at [`THREADS`] threads, or `None` (after a visible `SKIPPED`
/// line) when `llvm-as` is not installed.
fn gate_executor() -> Option<CpuInterp> {
    CpuInterp::new(PROFILE).expect("the CPU-interpreter executor")
}

/// Generated STRUCTURED control flow (`fuzz::structured`): nested loops, two-armed conditionals,
/// multi-way switches, multi-level breaks and continues, early returns, and a reducible cross-arm
/// edge, each compared against the program interpreted in Rust.
#[test]
fn generated_structured_control_flow_computes_what_the_program_says_on_the_cpu_interpreter() {
    let Some(mut executor) = gate_executor() else {
        return;
    };
    structured::check(&mut executor, GATE_SEEDS).assert_clean("structured (CpuInterp)", 12);
}

/// Generated IRREDUCIBLE control flow (`fuzz::irreducible`): an automaton whose single cycle is
/// entered at every one of its blocks, compared against the automaton stepped in Rust.
#[test]
fn generated_irreducible_cycles_step_the_same_automaton_on_the_cpu_interpreter() {
    let Some(mut executor) = gate_executor() else {
        return;
    };
    irreducible::check(&mut executor, GATE_SEEDS).assert_clean("irreducible (CpuInterp)", 12);
}

/// The same computation written across helpers and written flat (`fuzz::inline`), both compared
/// against the computation done in Rust.
#[test]
fn a_helper_computes_what_the_same_body_written_flat_computes_on_the_cpu_interpreter() {
    let Some(mut executor) = gate_executor() else {
        return;
    };
    inline::check(&mut executor, GATE_SEEDS).assert_clean("inline (CpuInterp)", 12);
}

/// Threadgroup memory, barriers, and threadgroup atomics (`fuzz::workgroup`), compared against the
/// same rounds computed in Rust.
#[test]
fn a_threadgroup_slot_holds_what_its_own_thread_wrote_on_the_cpu_interpreter() {
    let Some(mut executor) = gate_executor() else {
        return;
    };
    workgroup::check(&mut executor, GATE_SEEDS).assert_clean("workgroup (CpuInterp)", 12);
}

/// Two derivations of one address over generated aggregate layouts (`fuzz::access`), compared
/// byte for byte against the layout rules computed in Rust.
#[test]
fn a_typed_access_chain_and_a_raw_byte_offset_name_the_same_word_on_the_cpu_interpreter() {
    let Some(mut executor) = gate_executor() else {
        return;
    };
    access::check_load(&mut executor, GATE_SEEDS).assert_clean("access load (CpuInterp)", 12);
}

/// The whole-buffer store check (`fuzz::access`).
#[test]
fn a_stored_leaf_lands_on_its_own_bytes_and_no_others_on_the_cpu_interpreter() {
    let Some(mut executor) = gate_executor() else {
        return;
    };
    access::check_store(&mut executor, GATE_SEEDS).assert_clean("access store (CpuInterp)", 12);
}

/// The sub-word store race (`fuzz::access`).
#[test]
fn one_threads_subword_store_does_not_erase_another_threads_on_the_cpu_interpreter() {
    let Some(mut executor) = gate_executor() else {
        return;
    };
    access::check_race(&mut executor).assert_clean("access race (CpuInterp)", 12);
}

/// Pointers chosen at run time (`fuzz::pointer_select`): chains of `select`s, `phi`s, and
/// trailing GEPs between buffers, loaded through and finally stored through, compared word for
/// word over the whole output buffer against the chain resolved in Rust.
#[test]
fn a_selected_pointer_reads_and_writes_where_the_choice_says_on_the_cpu_interpreter() {
    let Some(mut executor) = gate_executor() else {
        return;
    };
    pointer_select::check(&mut executor, GATE_SEEDS).assert_clean("pointer select (CpuInterp)", 12);
}

/// Every word of the status buffer is checked: one nonzero word, whichever, fails the run.
#[test]
fn a_run_fails_on_any_nonzero_status_word() {
    let clean = [0u32; STATUS_WORDS];
    check_status(&word_bytes(&clean)).unwrap();
    for word in 0..STATUS_WORDS {
        let mut bad = clean;
        bad[word] = 1 << 7;
        assert!(
            check_status(&word_bytes(&bad)).is_err(),
            "word {word} is not checked"
        );
    }
    assert!(check_status(&word_bytes(&clean[1..])).is_err());
}

#[test]
fn the_interpreter_runs_every_modelled_width_the_stage_may_run_with() {
    let sizes = |min, max| SubgroupSizes { min, max };
    let free = None;
    let lanes = |widths: Vec<SubgroupSize>| widths.iter().map(|w| w.lanes()).collect::<Vec<_>>();
    assert_eq!(lanes(widths(sizes(4, 32), free).unwrap()), [8, 16, 32]);
    assert_eq!(lanes(widths(sizes(32, 64), free).unwrap()), [32, 64]);
    assert_eq!(lanes(widths(sizes(8, 8), free).unwrap()), [8]);
    assert_eq!(lanes(widths(sizes(1, 1), free).unwrap()), [8]);
    assert_eq!(lanes(widths(sizes(4, 4), free).unwrap()), [8]);
    let pinned = Some;
    assert_eq!(lanes(widths(sizes(4, 32), pinned(32)).unwrap()), [32]);
    assert!(widths(sizes(4, 4), pinned(4)).is_err());
}

/// The committed profiles' kernel stages, through [`widths`]: what the soak lanes run. A stage
/// without subgroup size control runs at the API's `subgroupSize` only (`moltenvk-m3`,
/// `lavapipe`); RDNA3 pins 32 or 64.
#[test]
fn the_soak_profiles_run_at_their_devices_widths() {
    let lanes = |stem: &str| {
        let profile = profile(stem, DevicePolicy::RESIDENT_TABLES).unwrap();
        widths(profile.subgroups().kernel.sizes, None)
            .unwrap()
            .iter()
            .map(|width| width.lanes())
            .collect::<Vec<_>>()
    };
    assert_eq!(lanes("moltenvk-m3"), [32]);
    assert_eq!(lanes("amd-780m-radv"), [32, 64]);
    assert_eq!(lanes("lavapipe"), [8]);
    assert_eq!(lanes("runtime-floor"), [8]);
}
