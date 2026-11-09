//! Device tests of textures and sampler states bound in kernels on the device the cfg selects,
//! under the validation layer: every committed public kernel case that binds a texture or a sampler
//! runs through the runtime against its committed Metal observation; sampler states are made or
//! refused typed; a bind's LOD clamps are used, and clamps Metal's sampling is not measured with
//! are refused where they are used; slots nothing is bound to read the device's neutral resources.
//!
//! The runtime runs the translation the engine candidate executor runs. Where the runtime's
//! output on this device is not Metal's (a translation or device difference C3 judges, such as a
//! device's NaN encoding, its filtering precision, or a result Metal leaves undefined), the census
//! runs the candidate executor on the case afresh, in the same process: the candidate first, then
//! the runtime, each on a device of its own opened once the other's is gone. The case passes only
//! when the runtime's bytes are the fresh candidate's, word for word, so the runtime adds no
//! difference of its own; the candidate's executor ABI is printed with the counts. Committed
//! candidate observations are not the evidence: a row may be an older executor's. Where the
//! candidate executor holds no form of the case on this device (`not_supported`: an image of a
//! format the device cannot make, such as lavapipe's `R8_SRGB`, which the runtime holds in its
//! UNORM twin), the runtime's differences from Metal are the device's answers under the precision
//! notes its executed stages record (`device-texel-conversion`, `device-sampling`,
//! `native-half-conversion` and their kin): the case passes only where those stages record one,
//! its differing words counted and printed with the notes. A case that gives a depth image a depth
//! outside `[0, 1]` on a device without `VK_EXT_depth_range_unrestricted` is one the candidate
//! does not run (`not_supported`, `ToolingRequirement::DepthOutsideUnitRange`): no such device
//! holds the depth (`VUID-vkCmdCopyBufferToImage-pRegions-07931`), so the runtime clamps the upload
//! and announces `restricted-depth-upload`, and the case passes where its run on a device of its
//! own announces that and completes clean with an output of Metal's length, its differing words
//! counted and printed (`read-a-cube-a-depth-slice-and-a-missing-volume`, whose `Depth32Float`
//! array holds 1000.0 to 1111.0, on MoltenVK and ANV).
//!
//! A device whose pipeline creation compile needs storage textures' formats makes such states
//! under stand-ins (`runtime::pipeline`'s creation): their query answers are held to the compile
//! of their cases' own formats.

use super::degrade_notices::counts;
use super::{
    buffer_with, encode_case, in_shard, layer_reported_nothing, layer_reported_nothing_at_all,
    pipeline_state, public_case, public_cases, read_buffer, run_case, validated_device,
    CaseResources, Messages, NamedCase, PublicCase, ScratchCache, WAIT,
};
use crate::candidate::engine::variants::compile_compute_for_shape;
use crate::candidate::engine::{compute_shape, ENGINE_EXECUTOR_ABI};
use crate::candidate::tests::engine_compute;
use crate::case::{AuthoredCase, Stage};
use crate::gpu::TEST_DEVICE;
use crate::requirement::ToolingRequirement;
use metal2vulkan::bind::{SamplerDescriptor, StatusReport};
use metal2vulkan::cache::VariantKey;
use metal2vulkan::error::Feature;
use metal2vulkan::library::Library;
use metal2vulkan::metal::{
    SamplerAddressMode, SamplerBorderColor, SamplerMinMagFilter, SamplerMipFilter,
};
use metal2vulkan::pipeline::{FunctionDescriptor, PrecisionNote};
use metal2vulkan::vk::DriverId;
use metal2vulkan::Budget;
use metal2vulkan_runtime::{
    Buffer, CommandBufferStatus, ComputeCommandEncoder, Degradation, Device, Error,
    ExecutedPipeline,
};
use std::collections::{BTreeMap, BTreeSet};

/// A case whose texture is sampled through a sampler (three 2D textures, one sampler, nearest
/// filtering, clamped to the edge, one level).
const SAMPLED: &str = "sample-each-of-three-textures-through-one-dynamic-array-index";

/// A case that writes one of two storage textures per thread.
const STORED: &str = "write-through-a-storage-texture-selected-per-thread";

/// Whether `why` is an honest refusal of a run: an input of a later runtime step (the lane's),
/// a gap the runtime names (`NotImplemented`), or what the device lacks (`Unsupported`, the
/// runtime's or the binder's: a bind state the translator cannot run, such as a sparse texture
/// the function writes), as the render census counts them.
pub(super) fn refusal(why: &str) -> bool {
    why.contains("later runtime step")
        || why.contains("not implemented in the runtime yet")
        || why.contains("unsupported:")
        || why.contains("the bind state was refused: Unsupported(")
}

/// Whether a completed run's execution report states only a read of an argument the function
/// constants removed, which the runtime answers zero and completes. Metal gives such a read no
/// value: its compiler drops a store of one, makes arithmetic on it NaN or folds it to zero by the
/// shape around it, so the output is no observation to compare with Metal's, and the candidate
/// executor records the same run as the typed refusal `FunctionConstantArgument`
/// (`NotRepresentable`): an honest refusal here too.
pub(super) fn absent_read_only(report: &StatusReport) -> bool {
    report.absent_argument_read && {
        let mut rest = *report;
        rest.absent_argument_read = false;
        rest == StatusReport::default()
    }
}

/// Whether `case` is a committed public kernel case that binds a texture, a texture array or a
/// sampler.
fn binds_textures(case: &AuthoredCase) -> bool {
    case.stage == Stage::Kernel
        && case.public_source.is_some()
        && !(case.textures.is_empty() && case.texture_arrays.is_empty() && case.samplers.is_empty())
}

/// The engine candidate executor's fresh run of `case` on the lane's device, compiled in the
/// device tests' compile worker: the device it opened, its output and its execution report, or why
/// it gave none.
pub(super) fn fresh_candidate(
    case: &PublicCase,
) -> Result<(String, Vec<u8>, StatusReport), String> {
    let run = engine_compute::run(&case.compute, &case.bitcode).map_err(|e| e.to_string())?;
    let device = run.environment["device"]
        .as_str()
        .unwrap_or("an unnamed device")
        .to_owned();
    Ok((device, run.output, run.status))
}

/// Whether `why`, the candidate's answer for a case, is its typed refusal of a depth outside `[0, 1]`
/// given to a depth image on a device without `VK_EXT_depth_range_unrestricted`
/// (`ToolingRequirement::DepthOutsideUnitRange`).
fn depth_range_refusal(why: &str) -> bool {
    why.starts_with("not_supported")
        && why.contains(&format!(
            "tooling_requirement:{}",
            ToolingRequirement::DepthOutsideUnitRange.as_str()
        ))
}

/// `case`'s run on a device of its own, which clamps its depths outside `[0, 1]`: the number of
/// its output's words that differ from Metal's, where the run completes with a clean execution
/// report and an output of Metal's length, announces `restricted-depth-upload`, and the
/// validation layer reports nothing; else what it did instead.
fn clamped_run(messages: &Messages, case: &PublicCase) -> Result<usize, String> {
    let cache = ScratchCache::new().unwrap();
    let device = validated_device(TEST_DEVICE, messages, cache.path()).unwrap();
    let _ = messages.take_notices();
    let run = run_case(&device, case);
    drop(device);
    let announced = counts(&messages.take_notices());
    let reported = messages.take();
    let run = run.map_err(|why| format!("the clamped run: {why}"))?;
    if !reported.is_empty() {
        return Err(format!(
            "the clamped run: the validation layer reported {reported:#?}"
        ));
    }
    if run.report != StatusReport::default() {
        return Err(format!(
            "the clamped run's execution report is {:?}",
            run.report
        ));
    }
    if !announced.contains_key(&Degradation::RestrictedDepthUpload) {
        return Err(format!(
            "the clamped run announced no restricted-depth-upload: {announced:?}"
        ));
    }
    if run.output.len() != case.metal_output.len() {
        return Err(format!(
            "the clamped run gave {} bytes, Metal {}",
            run.output.len(),
            case.metal_output.len()
        ));
    }
    Ok(run
        .output
        .chunks(4)
        .zip(case.metal_output.chunks(4))
        .filter(|(r, m)| r != m)
        .count())
}

/// The precision notes the stages `run` executed record, specialized ones included.
fn executed_notes(run: &super::RuntimeRun) -> BTreeSet<PrecisionNote> {
    let mut notes = BTreeSet::new();
    for execution in &run.executions {
        let stages = match execution.pipeline() {
            ExecutedPipeline::Compute(p) => std::slice::from_ref(&p.stage),
            ExecutedPipeline::Render(p) => &p.stages[..],
        };
        for stage in stages {
            notes.extend(stage.precision_notes.iter().copied());
            notes.extend(stage.specialized_notes.iter().map(|s| s.note));
        }
    }
    notes
}

/// Every committed public kernel case that binds a texture, a texture array or a sampler passes
/// the [`census`], in `TEXTURE_SHARDS` tests (`in_shard`), so nextest runs the pieces in parallel.
fn kernels_with_textures_and_samplers_match_metal(shard: usize) {
    let cases = public_cases(in_shard(binds_textures, shard, TEXTURE_SHARDS)).unwrap();
    assert!(
        !cases.is_empty(),
        "no committed case of shard {shard} binds a texture"
    );
    census(
        &format!("runtime texture kernels, shard {shard} of {TEXTURE_SHARDS}"),
        &cases,
    );
}

/// The pieces `kernels_with_textures_and_samplers_match_metal` is split into. As one test it took
/// 63-164 s on the M4 under the validation layer, past the Mac lane's 60 s a test.
const TEXTURE_SHARDS: usize = 16;

census_shards!(
    kernels_with_textures_and_samplers_match_metal, TEXTURE_SHARDS;
    kernels_with_textures_and_samplers_match_metal_0 = 0,
    kernels_with_textures_and_samplers_match_metal_1 = 1,
    kernels_with_textures_and_samplers_match_metal_2 = 2,
    kernels_with_textures_and_samplers_match_metal_3 = 3,
    kernels_with_textures_and_samplers_match_metal_4 = 4,
    kernels_with_textures_and_samplers_match_metal_5 = 5,
    kernels_with_textures_and_samplers_match_metal_6 = 6,
    kernels_with_textures_and_samplers_match_metal_7 = 7,
    kernels_with_textures_and_samplers_match_metal_8 = 8,
    kernels_with_textures_and_samplers_match_metal_9 = 9,
    kernels_with_textures_and_samplers_match_metal_10 = 10,
    kernels_with_textures_and_samplers_match_metal_11 = 11,
    kernels_with_textures_and_samplers_match_metal_12 = 12,
    kernels_with_textures_and_samplers_match_metal_13 = 13,
    kernels_with_textures_and_samplers_match_metal_14 = 14,
    kernels_with_textures_and_samplers_match_metal_15 = 15,
);

/// Each of `cases` runs through the runtime with a clean execution report and Metal's output, or
/// the output of the candidate executor run afresh on the case, or, where that executor holds no
/// form of the case on this device, an output of Metal's length under the precision notes its
/// executed stages record (the module's note), or is refused honestly (`refusal`); the validation
/// layer reports nothing, leaks included. The counts are printed as `label`'s census line.
pub(super) fn census(label: &str, cases: &[NamedCase]) {
    let messages = Messages::default();
    let (mut matched, mut differing, mut refused, mut failed) =
        (0usize, Vec::new(), Vec::new(), Vec::new());
    let device_name;
    {
        let cache = ScratchCache::new().unwrap();
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        device_name = device.name().to_owned();
        for (name, case) in cases {
            let run = case
                .as_ref()
                .map_err(Clone::clone)
                .and_then(|case| Ok((case, run_case(&device, case)?)));
            match run {
                Ok((_, run)) if absent_read_only(&run.report) => refused.push(format!(
                    "{name}: it read an argument its function constants removed \
                     (FunctionConstantArgument, not representable)"
                )),
                Ok((_, run)) if run.report != StatusReport::default() => {
                    failed.push(format!("{name}: the execution report is {:?}", run.report))
                }
                Ok((case, run)) if run.output == case.metal_output => matched += 1,
                Ok((case, run)) => {
                    let notes = executed_notes(&run);
                    differing.push((name, case, run.output, notes))
                }
                Err(why) if refusal(&why) => refused.push(format!("{name}: {why}")),
                Err(why) => failed.push(format!("{name}: {why}")),
            }
            let reported = messages.take();
            if !reported.is_empty() {
                failed.push(format!(
                    "{name}: the validation layer reported {reported:#?}"
                ));
            }
        }
    }
    let leaked = messages.take();
    // Where the runtime is not Metal: the candidate afresh, then the runtime on a device of its
    // own.
    let (mut as_candidate, mut native, mut clamped) = (Vec::new(), Vec::new(), Vec::new());
    for (name, case, output, notes) in differing {
        let (candidate_device, candidate, candidate_report) = match fresh_candidate(case) {
            Ok(run) => run,
            // The candidate executor holds no form of the case on this device: the runtime's
            // differences are the device's answers under the notes its stages record.
            Err(why)
                if why.starts_with("not_supported")
                    && !notes.is_empty()
                    && output.len() == case.metal_output.len() =>
            {
                let words = output
                    .chunks(4)
                    .zip(case.metal_output.chunks(4))
                    .filter(|(r, m)| r != m)
                    .count();
                native.push(format!(
                    "{name}: {words} of {} words differ from Metal's under {notes:?}; the \
                     candidate holds no form of it here: {why}",
                    output.len().div_ceil(4)
                ));
                continue;
            }
            // The candidate holds no form of a case that gives a depth image a depth outside
            // `[0, 1]` on this device: the runtime's clamp, announced (the module docs).
            Err(why) if depth_range_refusal(&why) => {
                match clamped_run(&messages, case) {
                    Ok(words) => clamped.push(format!(
                        "{name}: {words} of {} words differ from Metal's, its depths outside \
                         [0, 1] clamped and announced; the candidate holds no form of it here: \
                         {why}",
                        case.metal_output.len().div_ceil(4)
                    )),
                    Err(e) => failed.push(format!("{name}: {e}")),
                }
                continue;
            }
            Err(why) => {
                failed.push(format!("{name}: the fresh candidate gave no output: {why}"));
                continue;
            }
        };
        let rerun = {
            let cache = ScratchCache::new().unwrap();
            let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
            run_case(&device, case)
        };
        let reported = messages.take();
        if !reported.is_empty() {
            failed.push(format!(
                "{name}: the validation layer reported {reported:#?}"
            ));
        }
        match rerun {
            _ if candidate_device != device_name => failed.push(format!(
                "{name}: the candidate ran on {candidate_device}, the runtime on {device_name}"
            )),
            Ok(run) if run.report != StatusReport::default() => {
                failed.push(format!("{name}: the execution report is {:?}", run.report))
            }
            Ok(run) if run.output == candidate => as_candidate.push(format!(
                "{name}: the candidate's execution report {candidate_report:?}"
            )),
            Ok(run) => failed.push(format!(
                "{name}: differs from Metal and from the fresh candidate \
                 ({ENGINE_EXECUTOR_ABI}):\n runtime   {:02x?}\n candidate {:02x?}\n \
                 metal     {:02x?}",
                run.output, candidate, case.metal_output
            )),
            Err(why) => failed.push(format!("{name}: the run after the candidate's: {why}")),
        }
    }
    eprintln!(
        "{label} on {device_name}: {matched} match Metal, {} match the fresh candidate \
         ({ENGINE_EXECUTOR_ABI}), {} the device's under their notes, {} with their depths \
         clamped, {} refused, {} failed, of {}",
        as_candidate.len(),
        native.len(),
        clamped.len(),
        refused.len(),
        failed.len(),
        cases.len()
    );
    for line in &as_candidate {
        eprintln!("as the candidate: {line}");
    }
    for line in &native {
        eprintln!("the device's: {line}");
    }
    for line in &clamped {
        eprintln!("clamped: {line}");
    }
    for why in &refused {
        eprintln!("refused: {why}");
    }
    assert!(
        failed.is_empty() && leaked.is_empty(),
        "failed:\n{}\nat device destruction: {leaked:#?}",
        failed.join("\n")
    );
    layer_reported_nothing(
        &messages,
        "every case's commands were valid and left nothing behind",
    );
    assert!(matched > 0, "no case ran");
}

/// The answers to Metal's queries a compute pipeline gives, in the order the state's are read.
type Answers = (u32, u32, u32, bool);

/// On a device whose pipeline creation compile needs storage textures' formats, each state the
/// runtime makes under stand-ins answers Metal's queries as the compile of its case's own bound
/// formats does (`compile_compute_for_shape` without a dispatch: the bind's variant): every
/// committed public kernel case of the census whose function, compiled under no lowering inputs,
/// asks for a format. A case whose own formats do not compile is printed, not compared. In
/// `STAND_IN_SHARDS` tests (`in_shard`): as one, compiling every texture case's function took 60 s
/// on the M4 and 68 s on the 780M, at the Mac lane's 60 s a test.
fn stand_in_states_answer_queries_as_their_formats_compile(shard: usize) {
    let cases = public_cases(in_shard(binds_textures, shard, STAND_IN_SHARDS)).unwrap();
    assert!(
        !cases.is_empty(),
        "no committed case of shard {shard} binds a texture"
    );
    on_device(|device, _| {
        let compile = device.compile_device();
        let addressed = compile.profile().descriptors().buffer_device_address;
        let budget = Budget::standard();
        let (mut compared, mut uncompared, mut differ) = (0usize, Vec::new(), Vec::new());
        for (name, case) in &cases {
            let Ok(case) = case else { continue };
            if !case.compute.constants.is_empty() {
                continue;
            }
            let library = Library::from_air(&case.bitcode).unwrap();
            let function = FunctionDescriptor::new(library.function(&case.compute.entry).unwrap());
            let empty = VariantKey::new().lowering_inputs(metal2vulkan::pipeline::Stage::Kernel);
            let needs = matches!(
                compile.compile_function(&function, &empty, &budget),
                Err(metal2vulkan::error::Error::NeedsInput(
                    metal2vulkan::error::RuntimeInput::StorageTextureFormat { .. }
                ))
            );
            if !needs {
                continue;
            }
            let shape = compute_shape(&case.compute, addressed).unwrap();
            let own = compile_compute_for_shape(compile, &function, &shape, None, None, &budget);
            let state = pipeline_state(device, case);
            match (own, state) {
                (Ok((own, _)), Ok(state)) => {
                    compared += 1;
                    let info = &own.metal_info;
                    let expected: Answers = (
                        info.max_total_threads_per_threadgroup,
                        info.thread_execution_width,
                        info.static_threadgroup_memory_length,
                        info.support_indirect_command_buffers,
                    );
                    let answers: Answers = (
                        state.max_total_threads_per_threadgroup(),
                        state.thread_execution_width(),
                        state.static_threadgroup_memory_length(),
                        state.support_indirect_command_buffers(),
                    );
                    if answers != expected {
                        differ.push(format!(
                            "{name}: the stand-in state answers {answers:?}, its formats' \
                             compile {expected:?}"
                        ));
                    }
                }
                (Ok(_), Err(why)) => differ.push(format!(
                    "{name}: its formats compile, but the state is refused: {why}"
                )),
                (Err(why), _) => uncompared.push(format!("{name}: {why:?}")),
            }
        }
        eprintln!(
            "stand-in states on {}, shard {shard} of {STAND_IN_SHARDS}: {compared} answer as \
             their formats' compile, {} differ, {} of formats that do not compile",
            device.name(),
            differ.len(),
            uncompared.len()
        );
        for line in &uncompared {
            eprintln!("uncompared: {line}");
        }
        assert!(differ.is_empty(), "{}", differ.join("\n"));
    });
}

/// The pieces `stand_in_states_answer_queries_as_their_formats_compile` is split into.
const STAND_IN_SHARDS: usize = 8;

census_shards!(
    stand_in_states_answer_queries_as_their_formats_compile, STAND_IN_SHARDS;
    stand_in_states_answer_queries_as_their_formats_compile_0 = 0,
    stand_in_states_answer_queries_as_their_formats_compile_1 = 1,
    stand_in_states_answer_queries_as_their_formats_compile_2 = 2,
    stand_in_states_answer_queries_as_their_formats_compile_3 = 3,
    stand_in_states_answer_queries_as_their_formats_compile_4 = 4,
    stand_in_states_answer_queries_as_their_formats_compile_5 = 5,
    stand_in_states_answer_queries_as_their_formats_compile_6 = 6,
    stand_in_states_answer_queries_as_their_formats_compile_7 = 7,
);

/// Run `test` on a validated device of the lane, then fail on any validation message.
fn on_device(test: impl FnOnce(&Device, &Messages)) {
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        test(&device, &messages);
    }
    layer_reported_nothing_at_all(&messages);
}

/// A state the device can make is made, those Metal's validation stops among them, as plain
/// Metal makes them; what the device lacks is `Unsupported`.
#[test]
fn sampler_states_are_made_or_refused_typed() {
    on_device(|device, _| {
        // A `max_anisotropy` outside 1 to 16 is made, sampling as plain Metal samples it (0 as 1,
        // 17 as 32, 34 as 2, `u32::MAX` as 32; 32 at the device's largest, announced,
        // `degrade_notices`).
        for value in [0, 17, 34, u32::MAX] {
            let mut d = SamplerDescriptor::new();
            d.max_anisotropy = value;
            match device.new_sampler_state(&d) {
                Ok(_) => {}
                // A device without anisotropy refuses every state of more than 1.
                Err(Error::Unsupported(u)) if value != 0 => {
                    eprintln!("anisotropy {value}: {u:?}")
                }
                Err(other) => panic!("anisotropy {value}: {other}"),
            }
        }
        // Both border modes are made, as plain Metal makes them (`border_swizzles` samples through
        // them).
        let mut d = SamplerDescriptor::new();
        d.s_address_mode = SamplerAddressMode::ClampToZero;
        d.t_address_mode = SamplerAddressMode::ClampToBorderColor;
        d.border_color = SamplerBorderColor::OpaqueWhite;
        device.new_sampler_state(&d).unwrap();
        // A pixel-coordinate sampler Metal's validation stops is made, sampling its repeat axis
        // as `ClampToEdge`, as plain Metal does, and a mirror-clamp axis, which Metal mirrors once
        // (`mirror_clamp` samples through it), but on a device without the mode.
        let mut d = SamplerDescriptor::new();
        d.normalized_coordinates = false;
        d.s_address_mode = SamplerAddressMode::Repeat;
        d.mip_filter = SamplerMipFilter::Linear;
        device.new_sampler_state(&d).unwrap();
        d.t_address_mode = SamplerAddressMode::MirrorClampToEdge;
        match device.new_sampler_state(&d) {
            Ok(_) => {}
            Err(Error::Unsupported(u)) if u.feature == Feature::SamplerAddressMode => {
                eprintln!("a pixel mirror-clamp sampler: {u:?}")
            }
            Err(other) => panic!("a pixel mirror-clamp sampler: {other}"),
        }
        // Taken by Metal: made, or refused for what the device lacks.
        let mut linear = SamplerDescriptor::new();
        linear.min_filter = SamplerMinMagFilter::Linear;
        linear.mag_filter = SamplerMinMagFilter::Linear;
        linear.lod_min_clamp = f32::NAN;
        linear.lod_max_clamp = -1.0;
        let state = device.new_sampler_state(&linear).unwrap();
        assert_eq!(state.clone(), state);
        // Metal uniques sampler states by descriptor (probe 5): an equal descriptor gives the
        // live state, another descriptor another state.
        let mut repeat = SamplerDescriptor::new();
        repeat.min_filter = SamplerMinMagFilter::Linear;
        let first = device.new_sampler_state(&repeat).unwrap();
        assert_eq!(device.new_sampler_state(&repeat).unwrap(), first);
        let default = device.new_sampler_state(&SamplerDescriptor::new()).unwrap();
        assert_ne!(default, first);
        assert_ne!(default, state);
        for edit in [
            |d: &mut SamplerDescriptor| d.max_anisotropy = 16,
            |d: &mut SamplerDescriptor| d.s_address_mode = SamplerAddressMode::MirrorClampToEdge,
        ] {
            let mut d = SamplerDescriptor::new();
            edit(&mut d);
            match device.new_sampler_state(&d) {
                Ok(_) | Err(Error::Unsupported(_)) => {}
                Err(other) => panic!("{d:?}: {other}"),
            }
        }
    });
}

/// `case`'s buffers on `device`, holding the case's bytes.
fn case_buffers(device: &Device, case: &PublicCase) -> BTreeMap<u32, Buffer> {
    case.compute
        .buffers
        .iter()
        .map(|(&index, bytes)| (index, buffer_with(device, bytes).unwrap()))
        .collect()
}

/// Commit `encode`'s compute encoder in one command buffer on `device` and wait for it; the
/// command buffer completes.
fn run_encoder(device: &Device, encode: impl FnOnce(&mut ComputeCommandEncoder)) {
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encode(&mut encoder);
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
}

/// A bind's LOD clamps replace the state's: clamps Metal samples a one-level texture with alike
/// give Metal's output, inverted ones among them (they read the upper clamp's level); NaN clamps,
/// whose sampling Metal's answers are not measured for, are refused at the dispatch that uses them,
/// recording nothing.
#[test]
fn a_binds_lod_clamps_are_used_or_refused_where_used() {
    on_device(|device, _| {
        let case = public_case(SAMPLED).unwrap();
        let pipeline = pipeline_state(device, &case).unwrap();
        let buffers = case_buffers(device, &case);
        let resources = CaseResources::new(device, &case).unwrap();
        let (&slot, sampler) = resources.samplers.iter().next().unwrap();
        let (output, offset, length) = match case.compute.output {
            crate::candidate::engine::ComputeOutput::Buffer {
                binding,
                offset,
                length,
            } => (binding, offset, length),
            other => panic!("{SAMPLED}: output {other:?}"),
        };
        for (min, max) in [(f32::NAN, 1.0), (0.0, f32::NAN)] {
            run_encoder(device, |encoder| {
                let unsampled = CaseResources {
                    textures: resources.textures.clone(),
                    samplers: BTreeMap::new(),
                    ..CaseResources::default()
                };
                encoder.set_compute_pipeline_state(&pipeline).unwrap();
                encode_binds(encoder, &case, &buffers, &unsampled);
                encoder
                    .set_sampler_state_with_lod_clamps(Some(sampler), min, max, slot)
                    .unwrap();
                let error = dispatch(encoder, &case).unwrap_err();
                assert!(matches!(error, Error::Bind(_)), "{min}..{max}: {error}");
            });
        }
        for (min, max) in [(0.0, f32::MAX), (0.0, 0.0), (-4.0, 1000.0), (1.0, 0.0)] {
            let buffers = case_buffers(device, &case);
            run_encoder(device, |encoder| {
                let unsampled = CaseResources {
                    textures: resources.textures.clone(),
                    samplers: BTreeMap::new(),
                    ..CaseResources::default()
                };
                encoder.set_compute_pipeline_state(&pipeline).unwrap();
                encode_binds(encoder, &case, &buffers, &unsampled);
                encoder
                    .set_sampler_states_with_lod_clamps(&[Some(sampler)], &[min], &[max], slot)
                    .unwrap();
                dispatch(encoder, &case).unwrap();
            });
            let bytes = read_buffer(&buffers[&output], offset, length).unwrap();
            assert!(
                bytes == case.metal_output,
                "{min}..{max}: the output differs from Metal's"
            );
        }
    });
}

/// Bind `case`'s buffers, `resources` and threadgroup memory on `encoder`.
fn encode_binds(
    encoder: &mut ComputeCommandEncoder,
    case: &PublicCase,
    buffers: &BTreeMap<u32, Buffer>,
    resources: &CaseResources,
) {
    for (&index, buffer) in buffers {
        encoder.set_buffer(Some(buffer), 0, index).unwrap();
    }
    for (&index, texture) in &resources.textures {
        encoder.set_texture(Some(texture), index).unwrap();
    }
    for (&index, sampler) in &resources.samplers {
        encoder.set_sampler_state(Some(sampler), index).unwrap();
    }
    for (&index, &length) in &case.compute.threadgroup_lengths {
        encoder
            .set_threadgroup_memory_length(length, index)
            .unwrap();
    }
}

/// `case`'s dispatch on `encoder`, its state already bound.
fn dispatch(encoder: &mut ComputeCommandEncoder, case: &PublicCase) -> Result<(), Error> {
    use metal2vulkan::bind::Dispatch;
    use metal2vulkan_runtime::Size;
    let size = |axes: [u32; 3]| Size::new(axes[0].into(), axes[1].into(), axes[2].into());
    match case.compute.dispatch {
        Dispatch::Threads {
            threads,
            threads_per_group,
        } => encoder.dispatch_threads(size(threads), size(threads_per_group)),
        Dispatch::Threadgroups {
            groups,
            threads_per_group,
        } => encoder.dispatch_threadgroups(size(groups), size(threads_per_group)),
        other => panic!("{}: dispatch {other:?}", case.case.name),
    }
}

/// A kernel whose texture and sampler slots nothing is bound to reads and writes the device's
/// neutral resources: its dispatches complete with a clean execution report and the validation
/// layer reports nothing (what Metal reads there is undefined, so the bytes are not compared).
///
/// The neutral images its pipeline state's making made take no memory block of their own
/// (`WorkCounts::dedicated_memory_blocks_live` unchanged) where the driver binds memory that
/// queued work uses without waiting for it (ANV and RADV, the runtime's `memory.rs` `Fit`), and
/// blocks of their own elsewhere.
#[test]
fn unbound_texture_and_sampler_slots_use_neutrals() {
    on_device(|device, _| {
        let pooled = matches!(
            device.properties().description.driver_id,
            Some(DriverId::INTEL_OPEN_SOURCE_MESA | DriverId::MESA_RADV)
        );
        let dedicated = || device.work_counts().dedicated_memory_blocks_live;
        let before = dedicated();
        for name in [SAMPLED, STORED] {
            let case = public_case(name).unwrap();
            let pipeline = pipeline_state(device, &case).unwrap();
            let buffers = case_buffers(device, &case);
            let queue = device.new_command_queue().unwrap();
            let command_buffer = queue.command_buffer().unwrap();
            let mut encoder = command_buffer.compute_command_encoder().unwrap();
            encode_case(
                &mut encoder,
                &pipeline,
                &case.compute,
                &buffers,
                &CaseResources::default(),
            )
            .unwrap();
            encoder.end_encoding().unwrap();
            command_buffer.commit().unwrap();
            command_buffer.wait_until_completed_for(WAIT).unwrap();
            assert_eq!(
                command_buffer.status(),
                CommandBufferStatus::Completed,
                "{name}"
            );
            assert_eq!(
                command_buffer.execution_report(),
                Some(StatusReport::default()),
                "{name}"
            );
        }
        let made = dedicated() - before;
        println!("NEUTRAL-BLOCKS on {}: {made} dedicated", device.name());
        match pooled {
            true => assert_eq!(made, 0, "a neutral took a block of its own"),
            false => assert!(made > 0, "no neutral took a block of its own"),
        }
    });
}

/// The case run twice on one device, its textures made again: the second run matches Metal too,
/// so a texture's cached views and a state's kept samplers serve later commands.
#[test]
fn a_texture_case_runs_twice_alike() {
    on_device(|device, _| {
        for name in [SAMPLED, STORED] {
            let case = public_case(name).unwrap();
            for _ in 0..2 {
                let run = run_case(device, &case).unwrap();
                assert!(run.output == case.metal_output, "{name}");
            }
        }
    });
}
