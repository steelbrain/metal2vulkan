//! A set-0 buffer binding an entry point uses through more than one variable: whether the device
//! takes it as a dynamic descriptor, and that what its pipelines do is what the variant chose.
//!
//! A slot declares one variable per view at its binding -- a word view, and the narrow or `uvec4`
//! unit views an access needs -- so a kernel whose stores move a narrow leaf has a binding of
//! several variables (`pipeline::objects::dynamic_bindings`). Vulkan bounds
//! the variables at a binding nowhere, but MoltenVK's SPIRV-Cross writes Metal that does not
//! compile for a **dynamic** one, so the device is asked (one pipeline of that shape created and
//! destroyed, `objects::takes_multi_variable_dynamic`) rather than every device paying for it.
//!
//! What each test holds, on any device:
//!
//! - the pipelines of such a kernel are made and run, whatever the device answered, and
//!   `record_stride_not_a_multiple_of_four` answers exactly what Metal's `sizeof` and its two
//!   narrow stores say, from its buffers' starts and from each dispatch's own offset where the
//!   offsets move: where its three-variable binding is dynamic, the dynamic offset reaches every
//!   variable at it;
//! - and the sets its dispatches take are the ones the variant's own choice calls for
//!   (`ComputePipelineState::set_zero_choice`): where it pushes set 0 every dispatch pushes; else
//!   every dispatch after the first binds its predecessor's set, its offsets moved or not, dynamic
//!   or not ([`Ran::holds_the_choice`]);
//! - and a self-check the driver refuses reaches the device's sink as at most one informational
//!   line, never as the driver's errors of the runtime's own question.
//!
//! The device's answer is read where it bounds that choice -- a device that does not take such a
//! binding has none of them dynamic -- and never as the expectation itself.
//!
//! On an Apple M3 Max under MoltenVK 1.4.2 the answer is no. On a Radeon 780M (RADV, Mesa 26.2.4)
//! and on lavapipe (Mesa, arm64) it is yes, and set 0 is pushed for the kernel here, which
//! reaches no dynamic binding either:
//! [`a_narrow_unit_kernel_without_the_push_command_takes_the_devices_answer`] withholds
//! `VK_KHR_push_descriptor` so that the choice is reached at all.
//!
//! A device test: compiled only under the GPU safety cfgs (`m2v_gpu`, `m2v_cpu_icd`).

use crate::pipeline::objects::SELF_CHECK_REFUSED;
use crate::pipeline::SetZeroChoice;
use crate::{
    DebugMessage, Device, DeviceKind, DeviceOptions, MessageSeverity, MessageSink, PipelineOptions,
    ResourceOptions, ShaderCacheOptions, Size,
};
use metal2vulkan::device::VkExtension;
use metal2vulkan::pipeline::{ComputePipelineDescriptor, StageSource};
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

struct Scratch(PathBuf);

impl Scratch {
    fn new() -> Scratch {
        let path = std::env::temp_dir().join(format!(
            "m2v-dynmulti-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(&path).unwrap();
        Scratch(path)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Whether a device of these tests leaves `VK_KHR_push_descriptor` disabled
/// ([`Pushing::Withheld`]), so that the variants of a layout a push descriptor set would hold
/// reach the dynamic-binding choice at all.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Pushing {
    /// The recommended enables, push descriptors among them where the device has them.
    Enabled,
    /// `VK_KHR_push_descriptor` withheld (`DeviceOptions::withheld_extensions`).
    Withheld,
}

fn device(cache: &Path, pushing: Pushing) -> Device {
    device_with(cache, pushing, None)
}

/// [`device`], its messages handed to `messages`.
fn device_with(cache: &Path, pushing: Pushing, messages: Option<MessageSink>) -> Device {
    let withheld: BTreeSet<VkExtension> = match pushing {
        Pushing::Withheld => [VkExtension::KhrPushDescriptor].into_iter().collect(),
        Pushing::Enabled => BTreeSet::new(),
    };
    let options = DeviceOptions {
        withheld_extensions: withheld,
        kinds: match cfg!(m2v_cpu_icd) {
            true => vec![DeviceKind::Cpu],
            false => vec![
                DeviceKind::Discrete,
                DeviceKind::Integrated,
                DeviceKind::Virtual,
            ],
        },
        shader_cache: ShaderCacheOptions {
            directory: Some(cache.to_owned()),
            ..ShaderCacheOptions::default()
        },
        messages,
        ..DeviceOptions::default()
    };
    Device::system_default_with(&options).unwrap()
}

/// How far a dispatch of [`run`] binds every buffer past its predecessor, where its offsets move:
/// the largest `minStorageBufferOffsetAlignment` Vulkan allows, so each dispatch's offset is one a
/// descriptor can start at on any device, and past what one dispatch of the narrow-unit kernel
/// writes (eight records of 18 bytes and eight words).
const MOVED: u64 = 256;

/// What a run of `entry` answered: the bytes of each bound buffer, what its variant chose for set
/// 0, whether its dispatches' offsets moved, and the counter steps.
struct Ran {
    buffers: Vec<Vec<u8>>,
    moved: bool,
    chose: SetZeroChoice,
    reused: u64,
    found: u64,
    pushed: u64,
}

/// A kernel these tests run: `entry` of fixture `file`, with `slots` buffers of `bytes` bytes
/// bound, each its own buffer, and `threads` threads a dispatch.
struct Kernel {
    file: &'static str,
    entry: &'static str,
    slots: u32,
    bytes: u64,
    threads: u64,
}

/// `record_stride_not_a_multiple_of_four`, over its two buffers, eight threads a dispatch.
const NARROW: Kernel = Kernel {
    file: "kernel_record_stride_not_a_multiple_of_four.ll",
    entry: "record_stride_not_a_multiple_of_four",
    slots: 2,
    bytes: 1024,
    threads: 8,
};

/// Dispatch `kernel` `dispatches` times on a device of this `pushing`. Where `moved`, dispatch `d`
/// binds every buffer at `d * MOVED` (each buffer as many bytes longer); else every dispatch binds
/// them at their starts.
fn run(kernel: &Kernel, dispatches: usize, pushing: Pushing, moved: bool) -> Result<Ran, String> {
    let Kernel {
        file,
        entry,
        slots,
        bytes,
        threads,
    } = *kernel;
    let scratch = Scratch::new();
    let device = device(&scratch.0, pushing);
    let queue = device.new_command_queue().unwrap();
    let ll = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/fixtures/public")
        .join(file);
    let bitcode = metal2vulkan_devtools::llvm_assemble(&ll).map_err(|e| format!("{e:?}"))?;
    let library = metal2vulkan::library::Library::from_air(&bitcode).map_err(|e| format!("{e}"))?;
    let function = device
        .new_library_with_compile_library(library)
        .new_function_with_name(entry)
        .map_err(|e| format!("{e}"))?;
    let descriptor =
        ComputePipelineDescriptor::new(StageSource::Function(function.function_descriptor()));
    let state = device
        .new_compute_pipeline_state(&descriptor, &PipelineOptions::default())
        .map_err(|e| format!("pipeline: {e}"))?;
    let chose = state.set_zero_choice();
    let step = if moved { MOVED } else { 0 };
    let length = bytes + step * dispatches as u64;
    let buffers: Vec<_> = (0..slots)
        .map(|_| {
            device
                .new_buffer(length, ResourceOptions::shared())
                .unwrap()
        })
        .collect();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(&state).unwrap();
    let before = device.work_counts();
    for dispatch in 0..dispatches {
        if dispatch == 0 || moved {
            for (index, buffer) in buffers.iter().enumerate() {
                let offset = step * dispatch as u64;
                encoder
                    .set_buffer(Some(buffer), offset, index as u32)
                    .unwrap();
            }
        }
        encoder
            .dispatch_threads(Size::new(threads, 1, 1), Size::new(threads.min(32), 1, 1))
            .unwrap();
    }
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer
        .wait_until_completed_for(std::time::Duration::from_secs(60))
        .unwrap();
    if let Some(e) = command_buffer.error() {
        return Err(format!("command buffer: {e:?}"));
    }
    let after = device.work_counts();
    Ok(Ran {
        buffers: buffers
            .iter()
            .map(|b| {
                let p = b.contents().unwrap();
                // SAFETY: the command buffer completed, so the GPU's writes are visible, and the
                // mapping holds the buffer's whole length.
                unsafe { std::slice::from_raw_parts(p.as_ptr(), b.length() as usize) }.to_vec()
            })
            .collect(),
        moved,
        chose,
        reused: after.set_zero_sets_reused - before.set_zero_sets_reused,
        found: after.set_zero_sets_found - before.set_zero_sets_found,
        pushed: after.dispatches_pushing_set_zero - before.dispatches_pushing_set_zero,
    })
}

/// `record_stride_not_a_multiple_of_four`'s answers, which Metal's `sizeof` and its two narrow
/// stores fix: `sizeof(Rec)` is 18 and `alignof` 2, so `sizes[tid] == 18002`, and its two narrow
/// stores land at `18 * tid + 6` (a `short`) and `18 * tid + 11` (a `uchar`), from each
/// dispatch's own offset where the offsets moved. A dynamic offset applied to one variable of
/// `recs`'s binding alone would answer other bytes.
fn holds_metals_values(ran: &Ran, threads: u64, dispatches: usize) {
    let recs = &ran.buffers[0];
    let sizes = &ran.buffers[1];
    let regions = if ran.moved { dispatches } else { 1 };
    for region in 0..regions {
        let at = region * MOVED as usize;
        for tid in 0..threads as usize {
            let size = at + tid * 4;
            let size = u32::from_le_bytes(sizes[size..size + 4].try_into().unwrap());
            let rec = at + 18 * tid;
            let short = i16::from_le_bytes(recs[rec + 6..rec + 8].try_into().unwrap());
            let byte = recs[rec + 11];
            assert_eq!(
                (size, short, byte),
                (18002, (tid as i16) * 7 + 1, (tid as u8) ^ 0xa5),
                "thread {tid} of dispatch {region} of {dispatches} answered ({size}, {short}, \
                 {byte}): Metal's `sizeof` is 18 and `alignof` 2, and the two narrow stores land \
                 at 18 * tid + 6 and 18 * tid + 11 from the dispatch's offset"
            );
        }
    }
}

/// `record_stride_not_a_multiple_of_four`: `recs` takes a `u16` and a `u8` unit view beside its
/// word view, so its binding is one of three variables. Its dispatches bind both buffers at their
/// starts, then at moving offsets.
#[test]
fn a_narrow_unit_kernel_runs_and_its_set_follows_the_variants_choice() {
    let answer = answered(Pushing::Enabled);
    for moved in [false, true] {
        for dispatches in [8usize, 16] {
            let ran = run(&NARROW, dispatches, Pushing::Enabled, moved)
                .unwrap_or_else(|e| panic!("{dispatches} dispatches, moved {moved}: {e}"));
            holds_metals_values(&ran, NARROW.threads, dispatches);
            ran.holds_the_choice(answer, dispatches, "record_stride_not_a_multiple_of_four");
        }
    }
}

/// The device's answer taken on its own: on a device that takes a binding of several variables
/// dynamically and has no push command, `record_stride_not_a_multiple_of_four`'s bindings are
/// dynamic, and its dispatches at moving offsets answer Metal's values through their dynamic
/// offsets, the three-variable binding's included; on a device that does not, they are static.
///
/// Why the extension is withheld rather than a device found without it: on every device measured
/// so far the choice is unreachable otherwise. MoltenVK and RADV both have
/// `VK_KHR_push_descriptor`, and this kernel's set 0 is two descriptors, well inside
/// `maxPushDescriptors`, so `objects::pushes_set_zero` is true and `dynamic_bindings` returns
/// before it reads a binding's variables at all. Withholding the extension
/// (`DeviceOptions::withheld_extensions`) is the one configuration reachable through the public
/// API that exercises the branch, and `pushed` is asserted zero here so that the test cannot go
/// quietly inert again if that changes.
///
#[test]
fn a_narrow_unit_kernel_without_the_push_command_takes_the_devices_answer() {
    let answer = answered(Pushing::Withheld);
    for dispatches in [8usize, 16] {
        let ran = run(&NARROW, dispatches, Pushing::Withheld, true)
            .unwrap_or_else(|e| panic!("{dispatches} dispatches: {e}"));
        assert_eq!(
            (ran.pushed, ran.chose.pushes),
            (0, false),
            "the device pushed set 0 for {dispatches} dispatches with \
             `VK_KHR_push_descriptor` withheld, so this test reached none of the choice it holds"
        );
        holds_metals_values(&ran, NARROW.threads, dispatches);
        ran.holds_the_choice(
            answer,
            dispatches,
            "record_stride_not_a_multiple_of_four, no push",
        );
        // Its set 0 is two storage buffers, and Vulkan's floor for
        // `maxDescriptorSetStorageBuffersDynamic` is four, so the budget covers every one of them
        // on any device: what the device answered is the whole of the choice here.
        assert_eq!(
            ran.all_dynamic(),
            answer,
            "the device answered {answer} and its set-0 bindings {:?} are dynamic, of {:?}: two \
             storage buffers against a floor of four dynamic ones, so every binding is dynamic \
             exactly where the device takes one of several variables",
            ran.chose.dynamic,
            ran.chose.bindings
        );
    }
}

/// The self-check's failed creation is the device's answer to the runtime's own question, not a
/// fault of the program's work: a device whose answer is no hands its sink none of the errors the
/// driver reported inside that creation, but at most one informational line carrying them
/// (`objects::SELF_CHECK_REFUSED`, none where the driver reported nothing); a device whose answer
/// is yes hands it no such line. MoltenVK 1.4.2 on an Apple M3 Max reports two errors there (the
/// Metal it could not compile, and the pipeline), which reached the sink as errors before, failing
/// every validation lane that made a kernel of such a binding. The `DYNMULTI` line says what the
/// device answered and what its sink heard.
#[test]
fn a_refused_self_check_reaches_the_sink_as_one_informational_line() {
    let heard: Arc<Mutex<Vec<DebugMessage>>> = Arc::default();
    let sink = {
        let heard = heard.clone();
        MessageSink::new(move |message| heard.lock().unwrap().push(message.clone()))
    };
    let scratch = Scratch::new();
    let answer = {
        let device = device_with(&scratch.0, Pushing::Enabled, Some(sink));
        // What the device's making reported is not the self-check's.
        heard.lock().unwrap().clear();
        crate::pipeline::objects::asks_multi_variable_dynamic(&device)
    };
    let heard = heard.lock().unwrap().clone();
    let refused: Vec<_> = (heard.iter())
        .filter(|message| message.id_name == SELF_CHECK_REFUSED)
        .collect();
    eprintln!(
        "DYNMULTI the device answered {answer}; its sink heard {} message(s), {} of them \
         `{SELF_CHECK_REFUSED}`: {:?}",
        heard.len(),
        refused.len(),
        refused.first().map(|message| &message.text)
    );
    let errors: Vec<_> = (heard.iter())
        .filter(|message| message.severity == MessageSeverity::Error)
        .collect();
    assert!(
        errors.is_empty(),
        "the self-check's creation reached the sink as errors: {errors:#?}"
    );
    match answer {
        true => assert!(
            refused.is_empty(),
            "a device that answered yes: {refused:#?}"
        ),
        false => assert!(
            refused.len() <= 1
                && (refused.iter()).all(|message| message.severity == MessageSeverity::Info),
            "a device that answered no hands its sink at most one informational line: \
             {refused:#?}"
        ),
    }
}

/// The device's own answer, from a device of this test's own making, of this `pushing`: what each
/// check here reads where the answer bounds the variant's choice.
fn answered(pushing: Pushing) -> bool {
    let scratch = Scratch::new();
    let device = device(&scratch.0, pushing);
    crate::pipeline::objects::asks_multi_variable_dynamic(&device)
}

impl Ran {
    /// Every set-0 binding of its variant is dynamic.
    fn all_dynamic(&self) -> bool {
        (self.chose.bindings.iter()).all(|b| self.chose.dynamic.contains(b))
    }

    /// Hold the sets the dispatches took against what the variant chose, never against the
    /// device's answer: the two come apart where the dynamic-buffer budget is spent before a
    /// binding the device would have taken.
    ///
    /// - Where its dispatches push set 0, all of them push and none allocates.
    /// - Else every dispatch binds the set its predecessor wrote, whatever its bindings' forms,
    ///   its offsets moved or not: the writes are the same (`binding::LastSetZero` compares them
    ///   by value). `record_stride_not_a_multiple_of_four` reads both its buffers by thread
    ///   through storage views of their whole windows, so each descriptor's range runs from the
    ///   buffer's space start to its end at every bind offset, the offset in the slot's residual
    ///   push field (`metal2vulkan`'s `bind::instantiate::planned_range`).
    ///
    /// And the device's answer bounds the choice in the one direction it implies: a device that
    /// does not take a binding of several variables dynamically has none of them dynamic.
    fn holds_the_choice(&self, answer: bool, dispatches: usize, what: &str) {
        let chose = &self.chose;
        assert!(
            !chose.stand_in,
            "{what}: its base variant is a stand-in for storage-texture formats, so its \
             dispatches run a variant whose choice this is not"
        );
        let dynamic_several: Vec<u32> = (chose.several_variables.iter())
            .copied()
            .filter(|b| chose.dynamic.contains(b))
            .collect();
        assert!(
            !chose.several_variables.is_empty(),
            "{what}: set 0 has no binding of several variables, and a kernel without one holds \
             nothing of the choice"
        );
        if !answer {
            assert!(
                dynamic_several.is_empty(),
                "{what}: the device does not take a binding of several variables dynamically, \
                 yet the variant made {dynamic_several:?} of them dynamic"
            );
        }
        if chose.pushes {
            assert_eq!(
                (self.pushed, self.reused, self.found),
                (dispatches as u64, 0, 0),
                "{what}: its dispatches push set 0, so all {dispatches} of them should push and \
                 none allocate; {} pushed, {} reused and {} found",
                self.pushed,
                self.reused,
                self.found
            );
            return;
        }
        assert_eq!(
            self.pushed, 0,
            "{what}: its variant pushes no set 0, yet {} of {dispatches} dispatches pushed one",
            self.pushed
        );
        let want = dispatches as u64 - 1;
        assert_eq!(
            (self.reused, self.found),
            (want, 0),
            "{what}: the variant made set-0 bindings {:?} of {:?} dynamic, of which {:?} are of \
             several variables, and the offsets moved: {}, so {dispatches} dispatches should have \
             reused {want} and found 0 of their predecessors' sets, not {} and {}",
            chose.dynamic,
            chose.bindings,
            chose.several_variables,
            self.moved,
            self.reused,
            self.found
        );
    }
}
