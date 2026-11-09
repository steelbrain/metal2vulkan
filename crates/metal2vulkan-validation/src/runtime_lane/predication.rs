//! Command predication (`encode_start_if`, `encode_start_else`, `encode_end_if`, and the loops'
//! starts and ends) on the device the cfg selects, under the validation layer, against what Metal
//! answered on an Apple M3 Max under macOS 27.0.1 (the Metal platform's probe,
//! `metal::platform::predication_tests`, measures the same streams there). The owned synthetic
//! kernel `prd_mark` adds 1 to word `t` of a log, so the log shows which dispatches ran, each once:
//!
//! - streams of records and marks ([`run`]'s notation, the table `predication_streams.tsv`):
//!   blocks nested, with Elses, spanning a unit's encoders, killed by misused records, and
//!   unbalanced records dropping their ranges (the unit's dispatches before its first record
//!   included, in its encoder and the unit's earlier ones), with fences, setters and indirect
//!   dispatches among them, run the marks Metal ran and answer what Metal answered;
//! - the condition compares as Metal's codes say, reads the word at the offset rounded down to a
//!   multiple of 4, and reads what a dispatch of the encoder wrote just before, with no barrier,
//!   in a serial and a concurrent encoder;
//! - inside a failing block no dispatch runs, of any form: direct, indirect on the flat and folded
//!   paths, indirect by a host step, and an indirect command buffer's execution of a fixed and of
//!   an indirect range; inside a holding one each runs once; setters inside a skipped block take
//!   effect;
//! - loops run as Metal's ran, each pass with the encoder's state each command was encoded with,
//!   nested, around Ifs and every dispatch form, across their unit's encoders, and past the
//!   copies the runtime lays down, whose laying its check runs again, repeating nothing encoded
//!   before the loop (a render pass's clear, a blit); a loop without bound stops once the program
//!   holds no handle of its queue;
//! - a live block open where its unit ends fails the command buffer where its commands skip there,
//!   as Metal's GPU recovery does, and is harmless where they run;
//! - a word past its buffer reads as Metal reads it: zero within the 256 bytes Metal rounds a lone
//!   buffer up to, the heap's bytes past a buffer placed in a heap, and zero, announced, where
//!   Metal reads memory unrelated to the buffer.

use super::host_memory::Seen;
use super::rebind_reach::{buffer_len, buffer_over, deallocated, pages};
use super::rebinds::rebind;
use super::{
    buffer_with, layer_reported_nothing_at_all, read_buffer, validated_device,
    validated_device_with, Messages, ScratchCache, WAIT,
};
use crate::air_library::AirLibrary;
use crate::gpu::TEST_DEVICE;
use metal2vulkan::bind::PredicationStepForm;
use metal2vulkan::device::VkFeature;
use metal2vulkan::metal::{
    BarrierScope, DispatchType, IndirectCommandType, PixelFormat, ResourceUsage, TextureUsage,
};
use metal2vulkan::pipeline::{ComputePipelineDescriptor, StageSource};
use metal2vulkan_devtools::llvm_assemble_text;
use metal2vulkan_runtime::{
    BlitOption, Buffer, ClearColor, CommandBuffer, CommandBufferError, CommandBufferStatus,
    ComputeCommandEncoder, ComputePipelineState, Degradation, Device, Error, EventRef,
    HeapDescriptor, HeapType, IndirectCommandBuffer, IndirectCommandBufferDescriptor, LoadAction,
    Origin, PipelineOptions, RenderPassDescriptor, ResourceOptions, Size, StoreAction,
    TextureDescriptor,
};
use std::sync::Arc;

/// The kernels: `prd_mark` adds 1 to word `t[0]` of `log`; `prd_put` writes `v[0]` into word 0
/// of `w`; `prd_mark_group` adds its threadgroup's width to word `t[0]` of `log` (it observes
/// its partition, so its thread dispatches from GPU arguments take a host step).
const KERNELS: &str = r#"; Owned synthetic kernels: command predication's marks.
source_filename = "predication.metal"

define void @prd_mark(ptr addrspace(2) %t, ptr addrspace(1) %log) {
entry:
  %i = load i32, ptr addrspace(2) %t, align 4
  %i64 = zext i32 %i to i64
  %o = getelementptr inbounds i32, ptr addrspace(1) %log, i64 %i64
  %cur = load i32, ptr addrspace(1) %o, align 4
  %n = add i32 %cur, 1
  store i32 %n, ptr addrspace(1) %o, align 4
  ret void
}

define void @prd_put(ptr addrspace(2) %t, ptr addrspace(1) %log) {
entry:
  %v = load i32, ptr addrspace(2) %t, align 4
  store i32 %v, ptr addrspace(1) %log, align 4
  ret void
}

define void @prd_mark_group(ptr addrspace(2) %t, ptr addrspace(1) %log, <3 x i32> %tg) {
entry:
  %i = load i32, ptr addrspace(2) %t, align 4
  %i64 = zext i32 %i to i64
  %o = getelementptr inbounds i32, ptr addrspace(1) %log, i64 %i64
  %cur = load i32, ptr addrspace(1) %o, align 4
  %w = extractelement <3 x i32> %tg, i64 0
  %n = add i32 %cur, %w
  store i32 %n, ptr addrspace(1) %o, align 4
  ret void
}

!air.kernel = !{!0, !5, !6}
!air.version = !{!20}
!air.language_version = !{!21}
!0 = !{ptr @prd_mark, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"t"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"log"}
!5 = !{ptr @prd_put, !1, !2}
!6 = !{ptr @prd_mark_group, !1, !7}
!7 = !{!3, !4, !8}
!8 = !{i32 2, !"air.threads_per_threadgroup", !"air.arg_type_name", !"uint3", !"air.arg_name", !"tg"}
!20 = !{i32 2, i32 8, i32 0}
!21 = !{!"Metal", i32 4, i32 0, i32 0}
"#;

/// The log's words.
const LOG_WORDS: usize = 128;

/// Metal's comparison code `==` (0 `==`, 1 `!=`, 2 `>`, 3 `>=`, 4 `<`, 5 `<=`, and `==` for any
/// other code).
pub(super) const EQUAL: u64 = 0;

/// An offset past the end of a stream's condition words: Metal reads no word there for an
/// `encodeEndDoWhile:...` that closes no live DoWhile.
pub(crate) const PAST: u64 = 0x3f00;

/// Metal's comparison codes `!=` and `<`.
pub(super) const NOT_EQUAL: u64 = 1;
const LESS: u64 = 4;

/// The state of kernel `name` on `device`, made for indirect command buffers where `indirect`.
pub(super) fn kernel(device: &Device, name: &str, indirect: bool) -> ComputePipelineState {
    let library = device
        .new_library_with_air(&llvm_assemble_text(KERNELS).unwrap())
        .unwrap();
    let function = library.new_function_with_name(name).unwrap();
    let mut descriptor =
        ComputePipelineDescriptor::new(StageSource::Function(function.function_descriptor()));
    descriptor.support_indirect_command_buffers = indirect;
    device
        .new_compute_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap()
}

/// `words` as bytes.
fn bytes(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|w| w.to_le_bytes()).collect()
}

/// One thread, or a threadgroup of one.
fn one() -> Size {
    Size::new(1, 1, 1)
}

/// Write `words` from word 0 of `buffer` from the CPU.
fn write_words(buffer: &Buffer, words: &[u32]) {
    let contents = buffer
        .contents()
        .expect("a Shared buffer has contents")
        .as_ptr();
    for (i, &word) in words.iter().enumerate() {
        // SAFETY: the buffer holds at least `4 * words.len()` bytes at `contents`, alive with it,
        // and no command buffer of these tests runs while the CPU writes it.
        unsafe { contents.cast::<u32>().add(i).write_unaligned(word) }
    }
}

/// Write `value` into word 0 of `buffer` from the CPU.
fn write_word(buffer: &Buffer, value: u32) {
    let contents = buffer
        .contents()
        .expect("a Shared buffer has contents")
        .as_ptr();
    // SAFETY: the buffer holds at least 4 bytes at `contents`, alive with it, and no command
    // buffer of these tests runs while the CPU writes it.
    unsafe { contents.cast::<u32>().write_unaligned(value) }
}

/// What a test runs with: the marking kernel's state, the log, and the condition words (word 0
/// is 1).
struct Marks {
    mark: ComputePipelineState,
    log: Buffer,
    word: Buffer,
}

impl Marks {
    fn new(device: &Device) -> Marks {
        Marks {
            mark: kernel(device, "prd_mark", false),
            log: buffer_with(device, &[0; LOG_WORDS * 4]).unwrap(),
            word: buffer_with(device, &bytes(&[1, 0, 0, 0])).unwrap(),
        }
    }

    /// Bind the marking kernel to mark `tag` on `encoder`.
    fn bind(&self, encoder: &mut ComputeCommandEncoder, tag: u32) {
        encoder.set_compute_pipeline_state(&self.mark).unwrap();
        encoder.set_bytes(&tag.to_le_bytes(), 0).unwrap();
        encoder.set_buffer(Some(&self.log), 0, 1).unwrap();
    }

    /// Mark `tag` from `encoder`.
    fn mark(&self, encoder: &mut ComputeCommandEncoder, tag: u32) {
        self.bind(encoder, tag);
        encoder.dispatch_threads(one(), one()).unwrap();
    }

    /// Each tag's marks, by tag.
    fn counts(&self) -> Vec<u32> {
        read_buffer(&self.log, 0, self.log.length())
            .unwrap()
            .chunks(4)
            .map(|w| u32::from_le_bytes(w.try_into().unwrap()))
            .collect()
    }

    /// The tags that ran, ascending; a tag run more than once fails here.
    fn ran(&self) -> Vec<u32> {
        let log = self.counts();
        for (tag, &n) in log.iter().enumerate() {
            assert!(n <= 1, "mark {tag} ran {n} times");
        }
        (0..LOG_WORDS as u32)
            .filter(|&tag| log[tag as usize] == 1)
            .collect()
    }
}

/// Commit `command_buffer`, wait for it, and require it completed.
fn complete(command_buffer: &CommandBuffer) {
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
}

/// Run `test` on a validated device, then, where it has device addresses, on one made without
/// them (`bufferDeviceAddress` withheld), whose step module binds its places
/// (`PredicationStepForm::Bound`); after each, require that the layer reported nothing.
fn on_device(test: impl Fn(&Device)) {
    on_device_noted(|device, _| test(device));
}

/// [`on_device`], `test` given the devices' messages too (their degrade notices among them).
fn on_device_noted(test: impl Fn(&Device, &Messages)) {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    let addressed = {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let form = PredicationStepForm::of(device.compile_device());
        println!("PRD device {} {form:?}", device.name());
        test(&device, &messages);
        form == PredicationStepForm::Addressed
    };
    layer_reported_nothing_at_all(&messages);
    if !addressed {
        return;
    }
    let cache = ScratchCache::new().unwrap();
    {
        let without = |options: &mut metal2vulkan_runtime::DeviceOptions| {
            (options.withheld_features).insert(VkFeature::BufferDeviceAddress);
        };
        let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), without).unwrap();
        let form = PredicationStepForm::of(device.compile_device());
        assert_eq!(form, PredicationStepForm::Bound, "without device addresses");
        println!("PRD device {} {form:?}", device.name());
        test(&device, &messages);
    }
    layer_reported_nothing_at_all(&messages);
}

/// What [`run`] runs streams with: the marking kernel's state, the log, the condition words
/// (word 0 is 1), indirect dispatch arguments (one threadgroup of one at 0; a grid of one thread
/// in a group of one at 16), a fence and `prd_put`'s state.
pub(crate) struct Streams {
    marks: Marks,
    arguments: Buffer,
    fence: metal2vulkan_runtime::Fence,
    queue: metal2vulkan_runtime::CommandQueue,
    put: ComputePipelineState,
}

impl Streams {
    pub(crate) fn new(device: &Device) -> Streams {
        Streams {
            put: kernel(device, "prd_put", false),
            marks: Marks::new(device),
            arguments: buffer_with(device, &bytes(&[1, 1, 1, 0, 1, 1, 1, 1, 1, 1])).unwrap(),
            fence: device.new_fence(),
            queue: device.new_command_queue().unwrap(),
        }
    }
}

/// Run `script` in one command buffer, in the probes' notation: `Tn` marks `n` (its setters and
/// `dispatchThreads`), `Gn` and `Hn` by `dispatchThreadgroupsWithIndirectBuffer` and
/// `dispatchThreadsWithIndirectBuffer`, `bn` sets what `Tn` sets and dispatches nothing, `t`
/// dispatches with the state set; `I1` and `I0` start an If holding and failing (word 0 is 1,
/// compared `==` with 1 and with 0), `In` one on a nil buffer; `L` is an Else; `W1`, `W0` and
/// `Wn` start a While holding, failing and on a nil buffer, `D` a DoWhile; `E`, `V`, `Z0`, `Zn`
/// and `Zp` are an `encodeEndIf`, an `encodeEndWhile`, an `encodeEndDoWhile` failing, one on a nil
/// buffer and one at an offset past the words' buffer ([`PAST`]), whose answers are kept; `U`
/// makes the log resident, `Fu` and `Fw` update and wait for a
/// fence; `|` and `%` go on in a new serial and concurrent encoder, `B` ends the encoder and
/// encodes a blit (the next command opens a serial encoder, as one with no encoder open does).
/// The tags that ran, each once, and the answers, of a command buffer that completed.
pub(crate) fn run(streams: &Streams, script: &str) -> (Vec<u32>, String) {
    let ran = run_counted(streams, script);
    if let Some(error) = ran.refused {
        panic!("{script}: refused {error:?}");
    }
    assert_eq!(
        (ran.status, ran.error),
        (CommandBufferStatus::Completed, None),
        "{script}"
    );
    (streams.marks.ran(), ran.answers)
}

/// What a stream's command buffer ran ([`run_counted`]).
struct Counted {
    status: CommandBufferStatus,
    error: Option<CommandBufferError>,
    /// Each tag's marks, by tag.
    counts: Vec<u32>,
    answers: String,
    /// The first record the runtime refused, which recorded nothing.
    refused: Option<Error>,
}

/// [`run`]'s stream, with these for loops that count, as the Metal probe's: `w<n>[@k]` starts a
/// While on log word `k` (0 where none) `!= n`, `z<n>[@k]` ends a DoWhile on it `< n`, `i<n>[@k]`
/// starts an If on it `== n`, `Q<k>` writes 0 into log word `k`, `y<n>` sets the bytes to `n`,
/// `o<n>` sets buffer 1's offset to word `n`. Its command buffer's status and error once it ended,
/// each tag's marks, and the answers.
fn run_counted(streams: &Streams, script: &str) -> Counted {
    let marks = &streams.marks;
    write_words(&marks.log, &[0; LOG_WORDS]);
    write_words(&marks.word, &[1, 0, 0, 0]);
    let command_buffer = streams.queue.command_buffer().unwrap();
    let mut open: Option<ComputeCommandEncoder> = None;
    let mut answers = String::new();
    let mut refused = None;
    for token in script.split_whitespace() {
        match token {
            "|" | "%" => {
                if let Some(encoder) = open.take() {
                    encoder.end_encoding().unwrap();
                }
                let dispatch_type = match token {
                    "%" => DispatchType::Concurrent,
                    _ => DispatchType::Serial,
                };
                open = Some(
                    command_buffer
                        .compute_command_encoder_with_dispatch_type(dispatch_type)
                        .unwrap(),
                );
                continue;
            }
            "B" => {
                if let Some(encoder) = open.take() {
                    encoder.end_encoding().unwrap();
                }
                let mut blit = command_buffer.blit_command_encoder().unwrap();
                blit.fill_buffer(&marks.word, 8..12, 1).unwrap();
                blit.end_encoding().unwrap();
                continue;
            }
            _ => {}
        }
        let encoder = open.get_or_insert_with(|| command_buffer.compute_command_encoder().unwrap());
        let word = Some(&marks.word);
        let log = Some(&marks.log);
        let tag = |token: &str| token[1..].parse::<u32>().unwrap();
        // A record's answer, if it has one, is kept; its refusal too, the first one, and the
        // stream goes on.
        let mut record = |result: Result<Option<bool>, Error>| match result {
            Ok(Some(yes)) => answers.push(if yes { '1' } else { '0' }),
            Ok(None) => {}
            Err(error) => {
                refused.get_or_insert(error);
            }
        };
        let mut kept = |result: Result<(), Error>| record(result.map(|()| None));
        // A loop token's `n` and log word `k`.
        let number = |token: &str| -> (u32, u64) {
            let (n, k) = token[1..].split_once('@').unwrap_or((&token[1..], "0"));
            (n.parse().unwrap(), k.parse().unwrap())
        };
        match token {
            "I1" => kept(encoder.encode_start_if(word, 0, EQUAL, 1)),
            "I0" => kept(encoder.encode_start_if(word, 0, EQUAL, 0)),
            "In" => kept(encoder.encode_start_if(None, 0, EQUAL, 0)),
            "L" => kept(encoder.encode_start_else()),
            "W1" => kept(encoder.encode_start_while(word, 0, EQUAL, 1)),
            "W0" => kept(encoder.encode_start_while(word, 0, EQUAL, 0)),
            "Wn" => kept(encoder.encode_start_while(None, 0, EQUAL, 0)),
            "D" => kept(encoder.encode_start_do_while()),
            "E" => record(encoder.encode_end_if().map(Some)),
            "V" => record(encoder.encode_end_while().map(Some)),
            "Z0" => record(encoder.encode_end_do_while(word, 0, EQUAL, 0).map(Some)),
            "Zn" => record(encoder.encode_end_do_while(None, 0, EQUAL, 0).map(Some)),
            "Zp" => record(encoder.encode_end_do_while(word, PAST, EQUAL, 0).map(Some)),
            "U" => encoder
                .use_resource(
                    marks.log.as_resource(),
                    ResourceUsage::READ | ResourceUsage::WRITE,
                )
                .unwrap(),
            "Fu" => encoder.update_fence(&streams.fence).unwrap(),
            "Fw" => encoder.wait_for_fence(&streams.fence).unwrap(),
            "t" => encoder.dispatch_threads(one(), one()).unwrap(),
            _ if token.starts_with('T') => marks.mark(encoder, tag(token)),
            _ if token.starts_with('b') => marks.bind(encoder, tag(token)),
            _ if token.starts_with('G') => {
                marks.bind(encoder, tag(token));
                encoder
                    .dispatch_threadgroups_indirect(&streams.arguments, 0, one())
                    .unwrap();
            }
            _ if token.starts_with('H') => {
                marks.bind(encoder, tag(token));
                encoder
                    .dispatch_threads_indirect(&streams.arguments, 16)
                    .unwrap();
            }
            _ if token.starts_with('w') => {
                let (n, k) = number(token);
                kept(encoder.encode_start_while(log, 4 * k, NOT_EQUAL, n));
            }
            _ if token.starts_with('z') => {
                let (n, k) = number(token);
                record(encoder.encode_end_do_while(log, 4 * k, LESS, n).map(Some));
            }
            _ if token.starts_with('i') => {
                let (n, k) = number(token);
                kept(encoder.encode_start_if(log, 4 * k, EQUAL, n));
            }
            _ if token.starts_with('Q') => put(encoder, &streams.put, marks, tag(token).into()),
            _ if token.starts_with('y') => {
                encoder.set_bytes(&tag(token).to_le_bytes(), 0).unwrap();
            }
            _ if token.starts_with('o') => {
                encoder
                    .set_buffer_offset(4 * u64::from(tag(token)), 1)
                    .unwrap();
            }
            _ => panic!("no token {token}"),
        }
    }
    if let Some(encoder) = open {
        encoder.end_encoding().unwrap();
    }
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    Counted {
        status: command_buffer.status(),
        error: command_buffer.error(),
        counts: marks.counts(),
        answers,
        refused,
    }
}

/// A loop case: its stream ([`run_counted`]'s notation), each tag's count Metal ran, and its
/// Ends' answers.
pub(crate) type LoopCase = (String, Vec<(usize, u32)>, &'static str);

/// A fixed [`LoopCase`] as written: its script, what Metal ran and its answers.
type FixedLoopCase = (&'static str, &'static [(usize, u32)], &'static str);

/// The loops Metal ran on an M3 Max under macOS 27.0.1 (`metal_runs_loops_as_the_lane_expects`
/// on the Metal platform runs each there): the passes of Whiles and DoWhiles of 0 to 64, a While
/// in a concurrent encoder, three nested loops, each pass's bindings, an If in a DoWhile; loops
/// spanning a unit's encoders (serial, concurrent, three, around an If, inside a spanning If, with
/// an encoder's own bindings) and a serial encoder's While its concurrent encoder ends, which
/// kills it; four to six nested loops; and loops of 65 to 100000 passes, past the copies the
/// runtime lays down.
pub(crate) fn loop_cases() -> Vec<LoopCase> {
    let mut cases: Vec<LoopCase> = Vec::new();
    for n in [0u32, 1, 5, 64] {
        cases.push((
            format!("w{n} T0 T1 V T2"),
            vec![(0, n), (1, n), (2, 1)],
            "1",
        ));
        let runs = n.max(1);
        cases.push((
            format!("D T0 T1 z{n} T2"),
            vec![(0, runs), (1, runs), (2, 1)],
            "1",
        ));
    }
    let fixed: [FixedLoopCase; 28] = [
        // Past the copies laid down, the laying run again: 128 passes, a DoWhile of 65, and inner
        // loops of 17 and 40 passes in a While of 2, with marks after.
        ("w128 T0 V T1", &[(0, 128), (1, 1)], "1"),
        ("D T0 z65 T1", &[(0, 65), (1, 1)], "1"),
        (
            "w2 T0 Q1 w17@1 T1 T2 V V T3",
            &[(0, 2), (1, 17), (2, 34), (3, 1)],
            "11",
        ),
        (
            "w2 T0 Q1 w40@1 T1 T2 V V T3",
            &[(0, 2), (1, 40), (2, 80), (3, 1)],
            "11",
        ),
        ("% w7 T0 V", &[(0, 7)], "1"),
        (
            "w3 T0 Q1 w4@1 T1 Q2 D T2 T3 z5@2 V V",
            &[(0, 3), (1, 4), (2, 5), (3, 60)],
            "111",
        ),
        (
            "D T10 y0 o16 t T11 T0 z3 t",
            &[(0, 4), (10, 3), (11, 3), (16, 3)],
            "1",
        ),
        (
            "D T0 i2 T2 L T3 E T1 z4",
            &[(0, 4), (1, 4), (2, 1), (3, 3)],
            "11",
        ),
        ("w3 T0 | T5 V T6", &[(0, 3), (5, 3), (6, 1)], "1"),
        ("D T0 | T5 z3 T6", &[(0, 3), (5, 3), (6, 1)], "1"),
        (
            "w3 T0 | T5 | T7 V T6",
            &[(0, 3), (5, 3), (7, 3), (6, 1)],
            "1",
        ),
        ("% w3 T0 % T5 V T6", &[(0, 3), (5, 3), (6, 1)], "1"),
        ("w3 T0 % T5 V T6", &[(0, 1)], "0"),
        // An unbalanced End drops the unit's dispatches before it, in its encoder and the unit's
        // earlier one, and a loop after it runs, past its copies too.
        ("T1 E D T2 z3@2 T4", &[(2, 3), (4, 1)], "01"),
        ("T1 E D T2 z100@2 T4", &[(2, 100), (4, 1)], "01"),
        ("T1 | T3 E w70 T0 V T4", &[(0, 70), (4, 1)], "01"),
        (
            "D T0 | I1 T5 | E T7 z3 T6",
            &[(0, 3), (5, 3), (7, 3), (6, 1)],
            "11",
        ),
        (
            "I1 D T0 | T5 z3 | T7 E T6",
            &[(0, 3), (5, 3), (7, 1), (6, 1)],
            "11",
        ),
        (
            "D T10 | b10 y0 o16 t T0 z3 t",
            &[(0, 4), (10, 3), (16, 3)],
            "1",
        ),
        (
            "w2@1 T1 Q2 w2@2 T2 Q3 w2@3 T3 Q4 w2@4 T4 Q5 w2@5 T5 Q6 w2@6 T6 T50 V V V V V V",
            &[(1, 2), (2, 2), (3, 2), (4, 2), (5, 2), (6, 2), (50, 64)],
            "111111",
        ),
        (
            "D T1 Q2 D T2 Q3 D T3 Q4 D T4 T50 z2@4 z2@3 z2@2 z2@1",
            &[(1, 2), (2, 2), (3, 2), (4, 2), (50, 16)],
            "1111",
        ),
        ("w65 T0 T1 V T2", &[(0, 65), (1, 65), (2, 1)], "1"),
        ("w1000 T0 V T2", &[(0, 1000), (2, 1)], "1"),
        ("D T0 z100000 T2", &[(0, 100_000), (2, 1)], "1"),
        (
            "w2@1 T1 Q2 w17@2 T2 T50 V V",
            &[(1, 2), (2, 17), (50, 34)],
            "11",
        ),
        (
            "w3@1 T1 Q2 w100@2 T2 T50 V V",
            &[(1, 3), (2, 100), (50, 300)],
            "11",
        ),
        (
            "w2@1 T1 Q2 w2@2 T2 T50 V V",
            &[(1, 2), (2, 2), (50, 4)],
            "11",
        ),
        (
            "D T1 D T2 D T3 D T4 T50 Z0 Z0 Z0 Z0",
            &[(1, 1), (2, 1), (3, 1), (4, 1), (50, 1)],
            "1111",
        ),
    ];
    let fixed = fixed
        .iter()
        .map(|(script, want, answers)| (script.to_string(), want.to_vec(), *answers));
    cases.extend(fixed);
    cases
}

/// The streams Metal ran on an M3 Max under macOS 27.0.1, each with the tags that ran and what its
/// Ends answered (`predication_streams.tsv`).
pub(crate) fn measured_streams() -> Vec<(&'static str, &'static str, Vec<u32>, &'static str)> {
    include_str!("predication_streams.tsv")
        .lines()
        .filter(|line| !line.starts_with('#') && !line.is_empty())
        .map(|line| {
            let fields: Vec<&str> = line.split('\t').collect();
            let ran = (fields[2].split(',').filter(|tag| !tag.is_empty()))
                .map(|tag| tag.parse().unwrap())
                .collect();
            (fields[0], fields[1], ran, fields[3])
        })
        .collect()
}

/// Every measured stream runs the marks Metal ran, each once, and its Ends answer what Metal's
/// answered.
#[test]
fn streams_run_what_metal_ran() {
    on_device(|device| {
        let streams = Streams::new(device);
        let measured = measured_streams();
        assert!(measured.len() >= 600, "{}", measured.len());
        let mut wrong = Vec::new();
        for (name, script, ran, answers) in &measured {
            let got = run(&streams, script);
            if got != (ran.clone(), answers.to_string()) {
                wrong.push(format!(
                    "{name} `{script}`: ran {:?} answered {:?}, Metal ran {ran:?} answered \
                     {answers:?}",
                    got.0, got.1
                ));
            }
        }
        assert!(
            wrong.is_empty(),
            "{} of {}:\n{}",
            wrong.len(),
            measured.len(),
            wrong.join("\n")
        );
    });
}

/// A compute encoder whose first record is unbalanced with nothing dispatched before it drops its
/// unit from the unit's start, and the dispatches it encodes after that record lie in the dropped
/// range too (it runs to the next record): the range holds work Metal emitted, so the block the
/// next record opens is live rather than killed from its start. Such an encoder encodes again
/// exactly where a rebind encodes it again (`Predication::alone`: its unit laid no loop and left
/// no work out, and the work its `Resume` reads its own encoding laid down). The measured stream
/// `h147` (`In T1 T2 I1 T3 E T4`) marking into a no-copy buffer rebound after the commit and
/// before the first submission: the new pages hold the marks Metal ran (T3 and T4, not T1 or T2),
/// the old pages none, and the rebind encoded the encoder again, refusing nothing.
#[test]
fn a_drop_from_a_unit_s_start_then_a_dispatch_encodes_its_encoder_again_exactly() {
    on_device(|device| {
        let mark = kernel(device, "prd_mark", false);
        let word = buffer_with(device, &bytes(&[1, 0, 0, 0])).unwrap();
        // Each mapping lives with its deallocator, which notes its first 16 bytes as it runs.
        let seen = Arc::new(Seen::default());
        let Some(log) = buffer_over(device, pages(buffer_len(), 0, &[0; 16]), &seen) else {
            return;
        };
        let queue = device.new_command_queue().unwrap();
        let gate = device.new_shared_event().unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        command_buffer
            .encode_wait_for_event(EventRef::from(&gate), 1)
            .unwrap();
        let before = device.work_counts();
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        let mark_at = |encoder: &mut ComputeCommandEncoder, tag: u32| {
            encoder.set_compute_pipeline_state(&mark).unwrap();
            encoder.set_bytes(&tag.to_le_bytes(), 0).unwrap();
            encoder.set_buffer(Some(&log), 0, 1).unwrap();
            encoder.dispatch_threads(one(), one()).unwrap();
        };
        // `In T1 T2 I1 T3 E T4`: a Start on a nil buffer with no block open is unbalanced and
        // drops the unit from its start to the next record, T1 and T2 with it; the If then holds,
        // as the range held work, and T3 and T4 run.
        encoder.encode_start_if(None, 0, EQUAL, 0).unwrap();
        mark_at(&mut encoder, 1);
        mark_at(&mut encoder, 2);
        encoder.encode_start_if(Some(&word), 0, EQUAL, 1).unwrap();
        mark_at(&mut encoder, 3);
        assert_eq!(encoder.encode_end_if(), Ok(true));
        mark_at(&mut encoder, 4);
        encoder.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        let fresh = Arc::new(Seen::default());
        let rebound = rebind(&log, pages(buffer_len(), 0, &[0; 16]), &fresh);
        assert_eq!(rebound, Ok(()), "the rebind");
        gate.set_signaled_value(1);
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        let after = device.work_counts();
        let marks = |bytes: &[u8], tags: usize| -> Vec<u32> {
            (bytes.chunks(4).take(tags))
                .map(|w| u32::from_le_bytes(w.try_into().unwrap()))
                .collect()
        };
        let new_pages = marks(&read_buffer(&log, 0, 20).unwrap(), 5);
        // The old pages go once nothing holds them: the first encoding was never submitted, so
        // the 16 bytes the deallocator records hold none of its marks.
        deallocated(&seen);
        let old_pages = marks(&seen.bytes.lock().unwrap(), 4);
        assert_eq!(
            (
                command_buffer.status(),
                command_buffer.error(),
                new_pages,
                old_pages,
                after.encoders_reencoded - before.encoders_reencoded,
                after.reencodes_refused - before.reencodes_refused,
            ),
            (
                CommandBufferStatus::Completed,
                None,
                vec![0, 0, 0, 1, 1],
                vec![0, 0, 0, 0],
                1,
                0,
            ),
            "(status, error, the new pages' marks 0 to 4, the old pages' 0 to 3, encoders encoded \
             again, refused)"
        );
        eprintln!(
            "PRD-REENCODE {}: `In T1 T2 I1 T3 E T4` encoded again by a rebind ran T3 and T4, not \
             T1 or T2",
            device.name()
        );
    });
}

/// Which branch each of `cmp`'s rows took on Metal: `T` the If's, `E` the
/// Else's, for the rows [`cmp_rows`] lists, in order.
pub(crate) const CMP_BRANCHES: &str =
    "ETETETTEETTEEETETTETEETEETEETEETEETETEETEETEEETEETEETEETEEETEET";

/// `cmp`'s rows over the words `0x80000007, 0xffffffff, 5, 0`: each comparison code (and codes
/// past Metal's, which compare `==`) against references below, at and above word 0; then `==`
/// at each offset from 1 to 9, against each word.
pub(crate) fn cmp_rows() -> Vec<(u64, u64, u32)> {
    let codes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 255, 0x1_0000_0002, u64::MAX];
    let codes = codes
        .into_iter()
        .flat_map(|code| [0x8000_0006, 0x8000_0007, 0x8000_0008].map(|r| (code, 0, r)));
    let offsets = (1..=9).flat_map(|offset| [0x8000_0007, u32::MAX, 5].map(|r| (EQUAL, offset, r)));
    codes.chain(offsets).collect()
}

/// The condition compares the word as Metal's codes say, unsigned, the word at the offset
/// rounded down to a multiple of 4: each row runs the branch Metal ran, and its End answers YES.
#[test]
fn the_condition_compares_as_metal_does() {
    on_device(|device| {
        let marks = Marks::new(device);
        let words = buffer_with(device, &bytes(&[0x8000_0007, u32::MAX, 5, 0])).unwrap();
        let rows = cmp_rows();
        assert_eq!(rows.len(), CMP_BRANCHES.len());
        let queue = device.new_command_queue().unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        for (row, &(code, offset, reference)) in rows.iter().enumerate() {
            let row = row as u32;
            encoder
                .encode_start_if(Some(&words), offset, code, reference)
                .unwrap();
            marks.mark(&mut encoder, 2 * row);
            encoder.encode_start_else().unwrap();
            marks.mark(&mut encoder, 2 * row + 1);
            assert_eq!(encoder.encode_end_if(), Ok(true));
        }
        encoder.end_encoding().unwrap();
        complete(&command_buffer);
        let expected: Vec<u32> = CMP_BRANCHES
            .chars()
            .enumerate()
            .map(|(row, branch)| 2 * row as u32 + u32::from(branch == 'E'))
            .collect();
        assert_eq!(marks.ran(), expected);
    });
}

/// The word is read when the GPU reaches the record, as Metal reads it: after a dispatch just
/// before it wrote it, with no barrier, in a serial and in a concurrent encoder; after a CPU write
/// made between the encoding and the commit; and after a blit earlier in the command buffer filled
/// it.
#[test]
fn the_word_is_read_when_the_gpu_reaches_the_record() {
    on_device(|device| {
        let put = kernel(device, "prd_put", false);
        let queue = device.new_command_queue().unwrap();
        for dispatch_type in [DispatchType::Serial, DispatchType::Concurrent] {
            let marks = Marks::new(device);
            write_word(&marks.word, 3);
            let command_buffer = queue.command_buffer().unwrap();
            let mut encoder = command_buffer
                .compute_command_encoder_with_dispatch_type(dispatch_type)
                .unwrap();
            for i in 0..16u32 {
                let value: u32 = if i % 2 == 1 { 7 } else { 3 };
                encoder.set_compute_pipeline_state(&put).unwrap();
                encoder.set_bytes(&value.to_le_bytes(), 0).unwrap();
                encoder.set_buffer(Some(&marks.word), 0, 1).unwrap();
                encoder.dispatch_threads(one(), one()).unwrap();
                encoder
                    .encode_start_if(Some(&marks.word), 0, EQUAL, 7)
                    .unwrap();
                marks.mark(&mut encoder, 2 * i);
                encoder.encode_start_else().unwrap();
                marks.mark(&mut encoder, 2 * i + 1);
                assert_eq!(encoder.encode_end_if(), Ok(true));
                // Metal's probe orders the next write after this If's marks, as a concurrent
                // encoder needs.
                encoder.memory_barrier_with_scope(BarrierScope::BUFFERS);
            }
            encoder.end_encoding().unwrap();
            complete(&command_buffer);
            let expected: Vec<u32> = (0..16).map(|i| 2 * i + u32::from(i % 2 == 0)).collect();
            assert_eq!(marks.ran(), expected, "{dispatch_type:?}");
        }

        // A CPU write after the encoding, before the commit.
        let marks = Marks::new(device);
        write_word(&marks.word, 3);
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        encoder
            .encode_start_if(Some(&marks.word), 0, EQUAL, 7)
            .unwrap();
        marks.mark(&mut encoder, 4);
        encoder.encode_start_else().unwrap();
        marks.mark(&mut encoder, 5);
        assert_eq!(encoder.encode_end_if(), Ok(true));
        encoder.end_encoding().unwrap();
        write_word(&marks.word, 7);
        complete(&command_buffer);
        assert_eq!(marks.ran(), [4]);

        // A blit earlier in the command buffer.
        let marks = Marks::new(device);
        let command_buffer = queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.fill_buffer(&marks.word, 0..4, 7).unwrap();
        blit.end_encoding().unwrap();
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        encoder
            .encode_start_if(Some(&marks.word), 0, EQUAL, 0x0707_0707)
            .unwrap();
        marks.mark(&mut encoder, 1);
        encoder.encode_start_else().unwrap();
        marks.mark(&mut encoder, 2);
        assert_eq!(encoder.encode_end_if(), Ok(true));
        encoder.end_encoding().unwrap();
        complete(&command_buffer);
        assert_eq!(marks.ran(), [1]);
    });
}

/// How many dispatch forms [`Forms::encode`] encodes.
const FORMS: u32 = 8;

/// One of each dispatch form, of the marking kernel: direct of threads and of threadgroups;
/// indirect of threadgroups and of threads, of `prd_mark` (which may take the flat path) and of
/// `prd_mark_group` (which observes its partition, so takes a host step); and an indirect
/// command buffer's command executed over a fixed range and another over an indirect one.
struct Forms {
    group: ComputePipelineState,
    /// `MTLDispatchThreadsIndirectArguments` of one thread in a group of one, which begin with
    /// `MTLDispatchThreadgroupsIndirectArguments` of one group.
    arguments: Buffer,
    /// Command `c` of the indirect command buffer marks [`Forms::command_tag`]`(c)`, reading its
    /// tag from word `tag` of `tags`.
    icb: IndirectCommandBuffer,
    tags: Buffer,
    /// At `8 c` for an even `c`, the `MTLIndirectCommandBufferExecutionRange` of command `c + 1`.
    ranges: Buffer,
}

impl Forms {
    /// The commands of the indirect command buffer: two per block of the tests below.
    const COMMANDS: u64 = 8;

    /// The tag command `command` marks: the 7th and 8th forms of block `command / 2` of
    /// [`every_dispatch_form_runs_as_its_block_says`].
    fn command_tag(command: u64) -> u32 {
        (command / 2) as u32 * 10 + 6 + (command % 2) as u32
    }

    fn new(device: &Device, marks: &Marks) -> Forms {
        let mut descriptor = IndirectCommandBufferDescriptor::new();
        descriptor.command_types = IndirectCommandType::CONCURRENT_DISPATCH_THREADS;
        descriptor.max_kernel_buffer_bind_count = 2;
        let icb = device
            .new_indirect_command_buffer(&descriptor, Self::COMMANDS, ResourceOptions::shared())
            .unwrap();
        let tags: Vec<u32> = (0..LOG_WORDS as u32).collect();
        let tags = buffer_with(device, &bytes(&tags)).unwrap();
        let mark = kernel(device, "prd_mark", true);
        for command in 0..Self::COMMANDS {
            let at = icb.indirect_compute_command_at(command).unwrap();
            at.set_compute_pipeline_state(&mark).unwrap();
            let tag = u64::from(Self::command_tag(command));
            at.set_kernel_buffer(Some(&tags), 4 * tag, 0).unwrap();
            at.set_kernel_buffer(Some(&marks.log), 0, 1).unwrap();
            at.concurrent_dispatch_threads(one(), one()).unwrap();
        }
        let ranges: Vec<u32> = (0..Self::COMMANDS as u32)
            .flat_map(|c| match c % 2 {
                0 => [c + 1, 1],
                _ => [0, 0],
            })
            .collect();
        Forms {
            group: kernel(device, "prd_mark_group", false),
            arguments: buffer_with(device, &bytes(&[1; 6])).unwrap(),
            icb,
            tags,
            ranges: buffer_with(device, &bytes(&ranges)).unwrap(),
        }
    }

    /// Make the commands' buffers resident on `encoder`, as an indirect command's must be.
    fn resident(&self, encoder: &mut ComputeCommandEncoder, marks: &Marks) {
        encoder
            .use_resources(
                &[self.tags.as_resource(), marks.log.as_resource()],
                ResourceUsage::READ | ResourceUsage::WRITE,
            )
            .unwrap();
    }

    /// Encode each form on `encoder`, marking `base` to `base + FORMS - 1`; the executions run
    /// commands `command` (fixed) and `command + 1` (indirect).
    fn encode(&self, encoder: &mut ComputeCommandEncoder, marks: &Marks, command: u64) {
        let base = Self::command_tag(command) - 6;
        marks.bind(encoder, base);
        encoder.dispatch_threads(one(), one()).unwrap();
        marks.bind(encoder, base + 1);
        encoder.dispatch_threadgroups(one(), one()).unwrap();
        marks.bind(encoder, base + 2);
        encoder
            .dispatch_threadgroups_indirect(&self.arguments, 0, one())
            .unwrap();
        marks.bind(encoder, base + 3);
        encoder
            .dispatch_threads_indirect(&self.arguments, 0)
            .unwrap();
        encoder.set_compute_pipeline_state(&self.group).unwrap();
        encoder.set_bytes(&(base + 4).to_le_bytes(), 0).unwrap();
        encoder
            .dispatch_threadgroups_indirect(&self.arguments, 0, one())
            .unwrap();
        encoder.set_bytes(&(base + 5).to_le_bytes(), 0).unwrap();
        encoder
            .dispatch_threads_indirect(&self.arguments, 0)
            .unwrap();
        encoder
            .execute_commands_in_buffer(&self.icb, command..command + 1)
            .unwrap();
        encoder
            .execute_commands_in_buffer_indirect(&self.icb, &self.ranges, 8 * command)
            .unwrap();
    }
}

/// Inside a failing block no dispatch runs, of any form, and inside a holding one each runs once,
/// as on Metal: a failing If's block and a holding If's Else's run nothing; the failing If's Else's
/// and the holding If's run each form. A setter inside a failing block takes effect: the state and
/// tag it set mark after the block.
#[test]
fn every_dispatch_form_runs_as_its_block_says() {
    on_device(|device| {
        let marks = Marks::new(device);
        let forms = Forms::new(device, &marks);
        let queue = device.new_command_queue().unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        forms.resident(&mut encoder, &marks);
        for (holds, blocks) in [(false, [0, 2]), (true, [4, 6])] {
            let reference = u32::from(holds);
            encoder
                .encode_start_if(Some(&marks.word), 0, EQUAL, reference)
                .unwrap();
            forms.encode(&mut encoder, &marks, blocks[0]);
            encoder.encode_start_else().unwrap();
            forms.encode(&mut encoder, &marks, blocks[1]);
            assert_eq!(encoder.encode_end_if(), Ok(true));
        }
        encoder
            .encode_start_if(Some(&marks.word), 0, EQUAL, 0)
            .unwrap();
        marks.bind(&mut encoder, 50);
        assert_eq!(encoder.encode_end_if(), Ok(true));
        encoder.dispatch_threads(one(), one()).unwrap();
        encoder.end_encoding().unwrap();
        complete(&command_buffer);
        let ran: Vec<u32> = (10..10 + FORMS).chain(20..20 + FORMS).chain([50]).collect();
        assert_eq!(marks.ran(), ran);
    });
}

/// A condition word past its buffer's end reads as Metal read it on an M3 Max under macOS 27.0.1
/// (the Metal platform's `metal_reads_a_word_past_its_buffer_as_the_lane_expects` runs the same
/// records there): zero within the 256 bytes Metal rounds a lone buffer up to (offsets 16 and 252
/// of a buffer of 16); past a buffer placed at 0 of a placement heap, the bytes of the buffer
/// placed 65536 bytes on; and zero past the heap's end, at offset 2^32 of a lone buffer (Metal's
/// offsets do not wrap) and for a live DoWhile's End far past its buffer, which runs its body
/// once (`!= 0` fails). Those last three are memory unrelated to the buffer, which the device
/// announces once (`predication-word-past-buffer`); the first two are not.
#[test]
fn a_word_past_its_buffer_reads_as_metal_reads_it() {
    on_device_noted(|device, messages| {
        let marks = Marks::new(device);
        let heap_size = 1 << 20;
        let heap = (device.new_heap(&HeapDescriptor::new(
            HeapType::Placement,
            heap_size,
            ResourceOptions::shared(),
        )))
        .unwrap();
        let first = heap
            .new_buffer_at_offset(256, ResourceOptions::shared(), 0)
            .unwrap();
        let next = (heap.new_buffer_at_offset(256, ResourceOptions::shared(), 65536)).unwrap();
        write_words(&next, &[0xb0b0_b0b0]);
        let queue = device.new_command_queue().unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        let mut tag = 0;
        let mut holds = |encoder: &mut ComputeCommandEncoder, buffer: &Buffer, offset, word| {
            encoder
                .encode_start_if(Some(buffer), offset, EQUAL, word)
                .unwrap();
            marks.mark(encoder, tag);
            encoder.encode_start_else().unwrap();
            marks.mark(encoder, tag + 1);
            assert_eq!(encoder.encode_end_if(), Ok(true));
            tag += 2;
        };
        holds(&mut encoder, &marks.word, 16, 0);
        holds(&mut encoder, &marks.word, 252, 0);
        holds(&mut encoder, &first, 65536, 0xb0b0_b0b0);
        assert_eq!(noted(messages), 0, "noted before unrelated memory");
        holds(&mut encoder, &first, heap_size, 0);
        holds(&mut encoder, &marks.word, 1 << 32, 0);
        encoder.encode_start_do_while().unwrap();
        marks.mark(&mut encoder, 20);
        let end = encoder.encode_end_do_while(Some(&marks.word), PAST, NOT_EQUAL, 0);
        assert_eq!(end, Ok(true));
        encoder.end_encoding().unwrap();
        complete(&command_buffer);
        assert_eq!(marks.ran(), [0, 2, 4, 6, 8, 20]);
        assert_eq!(noted(messages), 1);
    });
}

/// The kind a word past its buffer that reads memory unrelated to it announces
/// ([`a_word_past_its_buffer_reads_as_metal_reads_it`], `degrade_notices.rs`).
pub(super) const WORD_PAST_BUFFER: [Degradation; 1] = [Degradation::PredicationWordPastBuffer];

/// How many notices of [`WORD_PAST_BUFFER`] `messages` delivered since this was last asked.
fn noted(messages: &Messages) -> usize {
    (messages.take_notices().into_iter())
        .filter_map(|message| message.degradation)
        .filter(|notice| WORD_PAST_BUFFER.contains(&notice.degradation))
        .count()
}

/// A block open where its unit ends, as Metal answers it on an M3 Max under macOS 27.0.1: where
/// the unit's commands skip there (a failed If, a failed While, a held If's Else, a failing If
/// inside a held one) Metal's GPU recovery fails the command buffer, `Internal`, code 1, with
/// Metal's description, at the command buffer's end, at a blit, and at the end of an encoder that
/// updated a fence, the error naming the record that opened the outermost block left open in the
/// first unit that failed so; where they run (a held If, a held While, which ran once, a DoWhile,
/// a failed If's Else) and in a killed block the command buffer completes, every mark Metal ran
/// run.
#[test]
fn a_block_open_where_its_unit_ends_fails_where_metals_recovery_does() {
    on_device(|device| {
        let streams = Streams::new(device);
        let failed = |record| Some(CommandBufferError::PredicationOpenAtUnitEnd { record });
        let cases: [(&str, Option<CommandBufferError>, &[u32]); 15] = [
            ("T1 I0 T2", failed(0), &[]),
            ("T1 I0 T2 B T3", failed(0), &[]),
            ("T1 W0 T2 B T3", failed(0), &[]),
            ("T1 I1 T2 L T3 B T4", failed(0), &[]),
            ("T1 I1 I0 T2 B T3", failed(0), &[]),
            ("T1 I0 T2 Fu | E T4", failed(0), &[]),
            // The record named: the outermost open block's, in the first unit that failed.
            ("I1 E T1 I0 T2", failed(2), &[]),
            ("I1 T1 B I0 T2", failed(1), &[]),
            ("T1 I0 T2 B I0 T3", failed(0), &[]),
            ("T5 B I1 T1 B T3", None, &[1, 3, 5]),
            ("T5 B W1 T1 B T3", None, &[1, 3, 5]),
            ("T5 B D T1 B T3", None, &[1, 3, 5]),
            ("T5 B I0 T1 L T2 B T3", None, &[2, 3, 5]),
            ("T5 B I1 V T1 B T3", None, &[3, 5]),
            ("T9 I1 T1 % E T3", None, &[1, 9]),
        ];
        for (script, error, ran) in cases {
            let (got, status, description) = run_to_end(&streams, script);
            assert_eq!(status, error, "{script}");
            if error.is_some() {
                assert_eq!(
                    description.as_deref(),
                    Some(
                        "Discarded (victim of GPU error/recovery) \
                         (00000005:kIOGPUCommandBufferCallbackErrorInnocentVictim)"
                    ),
                    "{script}"
                );
            } else {
                assert_eq!(got, ran, "{script}");
            }
        }
    });
}

/// [`run`] of `script` to its command buffer's end, completed or failed: the tags that ran, the
/// command buffer's error, and its description.
fn run_to_end(
    streams: &Streams,
    script: &str,
) -> (Vec<u32>, Option<CommandBufferError>, Option<String>) {
    let marks = &streams.marks;
    write_words(&marks.log, &[0; LOG_WORDS]);
    write_words(&marks.word, &[1, 0, 0, 0]);
    let command_buffer = streams.queue.command_buffer().unwrap();
    let mut open: Option<ComputeCommandEncoder> = None;
    for token in script.split_whitespace() {
        if matches!(token, "|" | "%" | "B") {
            if let Some(encoder) = open.take() {
                encoder.end_encoding().unwrap();
            }
        }
        match token {
            "|" => open = Some(command_buffer.compute_command_encoder().unwrap()),
            "%" => {
                open = Some(
                    command_buffer
                        .compute_command_encoder_with_dispatch_type(DispatchType::Concurrent)
                        .unwrap(),
                )
            }
            "B" => {
                let mut blit = command_buffer.blit_command_encoder().unwrap();
                blit.fill_buffer(&marks.word, 8..12, 1).unwrap();
                blit.end_encoding().unwrap();
            }
            _ => {
                let encoder =
                    open.get_or_insert_with(|| command_buffer.compute_command_encoder().unwrap());
                let word = Some(&marks.word);
                match token {
                    "I1" => encoder.encode_start_if(word, 0, EQUAL, 1).unwrap(),
                    "I0" => encoder.encode_start_if(word, 0, EQUAL, 0).unwrap(),
                    "W1" => encoder.encode_start_while(word, 0, EQUAL, 1).unwrap(),
                    "W0" => encoder.encode_start_while(word, 0, EQUAL, 0).unwrap(),
                    "D" => encoder.encode_start_do_while().unwrap(),
                    "L" => encoder.encode_start_else().unwrap(),
                    "E" => drop(encoder.encode_end_if().unwrap()),
                    "V" => drop(encoder.encode_end_while().unwrap()),
                    "Fu" => encoder.update_fence(&streams.fence).unwrap(),
                    _ => marks.mark(encoder, token[1..].parse().unwrap()),
                }
            }
        }
    }
    if let Some(encoder) = open {
        encoder.end_encoding().unwrap();
    }
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    let ran = marks.counts();
    let ran = (0..LOG_WORDS as u32)
        .filter(|&tag| ran[tag as usize] != 0)
        .collect();
    (
        ran,
        command_buffer.error(),
        command_buffer.error_description(),
    )
}

/// A loop spans its unit's encoders, as Metal's do on an M3 Max under macOS 27.0.1: its body's
/// commands in each encoder run each pass, serial and concurrent, across three encoders, and
/// around an If that spans them too, each with the bindings and bytes it was encoded with (an
/// earlier encoder's state included); a loop whose unit ends at a change of dispatch type ran
/// once, held, and its End in the next unit is unbalanced there.
#[test]
fn loops_span_their_units_encoders() {
    on_device(|device| {
        let cases: [SpanningCase; 6] = [
            (false, &[(0, 3), (5, 3), (6, 1)], |s| {
                s.start_while(3);
                s.mark(0);
                s.next(false);
                s.mark(5);
                s.end_while();
                s.mark(6);
            }),
            (false, &[(0, 3), (5, 3), (6, 1)], |s| {
                s.start_do();
                s.mark(0);
                s.next(false);
                s.mark(5);
                s.end_do(3);
                s.mark(6);
            }),
            (false, &[(0, 3), (5, 3), (7, 3), (6, 1)], |s| {
                s.start_while(3);
                s.mark(0);
                s.next(false);
                s.mark(5);
                s.next(false);
                s.mark(7);
                s.end_while();
                s.mark(6);
            }),
            (true, &[(0, 3), (5, 3), (6, 1)], |s| {
                s.start_while(3);
                s.mark(0);
                s.next(true);
                s.mark(5);
                s.end_while();
                s.mark(6);
            }),
            (false, &[(0, 3), (5, 3), (7, 3), (6, 1)], |s| {
                s.start_do();
                s.mark(0);
                s.next(false);
                let marks = s.marks;
                let word = Some(&marks.word);
                assert_eq!(s.encoder().encode_start_if(word, 0, EQUAL, 1), Ok(()));
                s.mark(5);
                s.next(false);
                assert_eq!(s.encoder().encode_end_if(), Ok(true));
                s.mark(7);
                s.end_do(3);
                s.mark(6);
            }),
            // `D T10 | b10 y0 o16 t T0 z3 t`: the bytes and offset set in the second encoder
            // after its own bindings, each pass.
            (false, &[(0, 4), (10, 3), (16, 3)], |s| {
                s.start_do();
                s.mark(10);
                s.next(false);
                let marks = s.marks;
                marks.bind(s.encoder(), 10);
                let encoder = s.encoder();
                encoder.set_bytes(&0u32.to_le_bytes(), 0).unwrap();
                encoder.set_buffer_offset(64, 1).unwrap();
                encoder.dispatch_threads(one(), one()).unwrap();
                s.mark(0);
                s.end_do(3);
                s.encoder().dispatch_threads(one(), one()).unwrap();
            }),
        ];
        for (case, (concurrent, want, encode)) in cases.into_iter().enumerate() {
            let marks = Marks::new(device);
            let queue = device.new_command_queue().unwrap();
            let command_buffer = queue.command_buffer().unwrap();
            let dispatch_type = match concurrent {
                true => DispatchType::Concurrent,
                false => DispatchType::Serial,
            };
            let encoder = command_buffer
                .compute_command_encoder_with_dispatch_type(dispatch_type)
                .unwrap();
            let mut spanning = Spanning {
                marks: &marks,
                command_buffer: &command_buffer,
                encoder: Some(encoder),
            };
            encode(&mut spanning);
            spanning.encoder.take().unwrap().end_encoding().unwrap();
            complete(&command_buffer);
            let mut counts = vec![0; LOG_WORDS];
            for &(tag, n) in want {
                counts[tag] = n;
            }
            assert_eq!(marks.counts(), counts, "case {case}");
        }
        let streams = Streams::new(device);
        assert_eq!(run(&streams, "W1 T1 % V T3"), (vec![1], "0".to_owned()));
    });
}

/// A case of [`loops_span_their_units_encoders`]: whether its first encoder is concurrent, the
/// marks Metal ran (tag and count), and its encoding.
type SpanningCase = (bool, &'static [(usize, u32)], fn(&mut Spanning));

/// A loop's commands across a unit's encoders ([`loops_span_their_units_encoders`]).
struct Spanning<'a> {
    marks: &'a Marks,
    command_buffer: &'a CommandBuffer,
    encoder: Option<ComputeCommandEncoder>,
}

impl Spanning<'_> {
    fn encoder(&mut self) -> &mut ComputeCommandEncoder {
        self.encoder.as_mut().unwrap()
    }

    fn mark(&mut self, tag: u32) {
        let marks = self.marks;
        marks.mark(self.encoder(), tag);
    }

    /// End the encoder and open the next of its unit, concurrent where `concurrent`.
    fn next(&mut self, concurrent: bool) {
        self.encoder.take().unwrap().end_encoding().unwrap();
        let dispatch_type = match concurrent {
            true => DispatchType::Concurrent,
            false => DispatchType::Serial,
        };
        self.encoder = Some(
            self.command_buffer
                .compute_command_encoder_with_dispatch_type(dispatch_type)
                .unwrap(),
        );
    }

    /// `while (log[0] != n)`.
    fn start_while(&mut self, n: u32) {
        let log = &self.marks.log;
        self.encoder
            .as_mut()
            .unwrap()
            .encode_start_while(Some(log), 0, NOT_EQUAL, n)
            .unwrap();
    }

    fn end_while(&mut self) {
        assert_eq!(self.encoder().encode_end_while(), Ok(true));
    }

    fn start_do(&mut self) {
        self.encoder().encode_start_do_while().unwrap();
    }

    /// `while (log[0] < n)`.
    fn end_do(&mut self, n: u32) {
        let log = &self.marks.log;
        let end = (self.encoder.as_mut().unwrap()).encode_end_do_while(Some(log), 0, LESS, n);
        assert_eq!(end, Ok(true));
    }
}

/// `loop`'s marks on `device`: a command buffer of one serial encoder (concurrent where
/// `concurrent`) that `encode` fills, run to its end; its status and error, and every tag's
/// marks.
fn looped(
    device: &Device,
    concurrent: bool,
    encode: impl FnOnce(&mut ComputeCommandEncoder, &Marks),
) -> (CommandBufferStatus, Option<CommandBufferError>, Vec<u32>) {
    let marks = Marks::new(device);
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let dispatch_type = match concurrent {
        true => DispatchType::Concurrent,
        false => DispatchType::Serial,
    };
    let mut encoder = command_buffer
        .compute_command_encoder_with_dispatch_type(dispatch_type)
        .unwrap();
    encode(&mut encoder, &marks);
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    (
        command_buffer.status(),
        command_buffer.error(),
        marks.counts(),
    )
}

/// Write `value` into log word `at` from `encoder`, by `prd_put`.
fn put(encoder: &mut ComputeCommandEncoder, put: &ComputePipelineState, marks: &Marks, at: u64) {
    encoder.set_compute_pipeline_state(put).unwrap();
    encoder.set_bytes(&0u32.to_le_bytes(), 0).unwrap();
    encoder.set_buffer(Some(&marks.log), 4 * at, 1).unwrap();
    encoder.dispatch_threads(one(), one()).unwrap();
}

/// Loops run as Metal's ran on an M3 Max under macOS 27.0.1 (and an M4), each counting in the
/// log what the probe counted: `while0` to `while5` (`while (log[0] != n) mark 0`), `do0` to
/// `do5` (`do mark 0 while (log[0] < n)`), at 64 runs, the copies the runtime lays down, too;
/// `whileconc` in a concurrent encoder (its body writes the word, no barrier); `loopnest3`
/// (`3 x 4 x 5` passes of a DoWhile in two Whiles, the inner counts reset by a dispatch in the
/// outer bodies, every End answering YES); `loopbind` (each pass runs with the bindings and bytes
/// each dispatch was encoded with, and the last stand after the loop); and `loopif` (an If and
/// its Else inside a DoWhile, per pass: Metal's log `3 1 2 1 3 1 3 1`). Every one completes.
#[test]
fn loops_run_as_metal_ran() {
    on_device(|device| {
        let ok = (CommandBufferStatus::Completed, None);
        let expect = |got: (CommandBufferStatus, Option<CommandBufferError>, Vec<u32>),
                      want: &[(usize, u32)],
                      case: &str| {
            assert_eq!((got.0, got.1), ok, "{case}");
            let mut counts = vec![0; LOG_WORDS];
            for &(tag, n) in want {
                counts[tag] = n;
            }
            assert_eq!(got.2, counts, "{case}");
        };
        for n in [0, 1, 5, 64] {
            let got = looped(device, false, |encoder, marks| {
                encoder
                    .encode_start_while(Some(&marks.log), 0, NOT_EQUAL, n)
                    .unwrap();
                marks.mark(encoder, 0);
                marks.mark(encoder, 1);
                assert_eq!(encoder.encode_end_while(), Ok(true));
                marks.mark(encoder, 2);
            });
            expect(got, &[(0, n), (1, n), (2, 1)], &format!("while{n}"));
            let got = looped(device, false, |encoder, marks| {
                encoder.encode_start_do_while().unwrap();
                marks.mark(encoder, 0);
                marks.mark(encoder, 1);
                let end = encoder.encode_end_do_while(Some(&marks.log), 0, LESS, n);
                assert_eq!(end, Ok(true));
                marks.mark(encoder, 2);
            });
            let runs = n.max(1);
            expect(got, &[(0, runs), (1, runs), (2, 1)], &format!("do{n}"));
        }
        let got = looped(device, true, |encoder, marks| {
            encoder
                .encode_start_while(Some(&marks.log), 0, NOT_EQUAL, 7)
                .unwrap();
            marks.mark(encoder, 0);
            assert_eq!(encoder.encode_end_while(), Ok(true));
        });
        expect(got, &[(0, 7)], "whileconc");

        let zero = kernel(device, "prd_put", false);
        let got = looped(device, false, |encoder, marks| {
            let log = Some(&marks.log);
            encoder.encode_start_while(log, 0, NOT_EQUAL, 3).unwrap();
            marks.mark(encoder, 0);
            put(encoder, &zero, marks, 1);
            encoder.encode_start_while(log, 4, NOT_EQUAL, 4).unwrap();
            marks.mark(encoder, 1);
            put(encoder, &zero, marks, 2);
            encoder.encode_start_do_while().unwrap();
            marks.mark(encoder, 2);
            marks.mark(encoder, 3);
            assert_eq!(encoder.encode_end_do_while(log, 8, LESS, 5), Ok(true));
            assert_eq!(encoder.encode_end_while(), Ok(true));
            assert_eq!(encoder.encode_end_while(), Ok(true));
        });
        expect(got, &[(0, 3), (1, 4), (2, 5), (3, 60)], "loopnest3");

        let got = looped(device, false, |encoder, marks| {
            encoder.encode_start_do_while().unwrap();
            marks.mark(encoder, 10);
            encoder.set_bytes(&0u32.to_le_bytes(), 0).unwrap();
            encoder.set_buffer_offset(64, 1).unwrap();
            encoder.dispatch_threads(one(), one()).unwrap();
            marks.mark(encoder, 11);
            marks.mark(encoder, 0);
            let end = encoder.encode_end_do_while(Some(&marks.log), 0, NOT_EQUAL, 3);
            assert_eq!(end, Ok(true));
            encoder.dispatch_threads(one(), one()).unwrap();
        });
        expect(got, &[(0, 4), (10, 3), (11, 3), (16, 3)], "loopbind");

        let got = looped(device, false, |encoder, marks| {
            encoder.encode_start_do_while().unwrap();
            marks.mark(encoder, 0);
            encoder
                .encode_start_if(Some(&marks.log), 0, EQUAL, 2)
                .unwrap();
            marks.mark(encoder, 2);
            encoder.encode_start_else().unwrap();
            marks.mark(encoder, 3);
            assert_eq!(encoder.encode_end_if(), Ok(true));
            marks.mark(encoder, 1);
            let end = encoder.encode_end_do_while(Some(&marks.log), 0, LESS, 4);
            assert_eq!(end, Ok(true));
        });
        expect(got, &[(0, 4), (1, 4), (2, 1), (3, 3)], "loopif");
    });
}

/// The loops Metal ran ([`loop_cases`]) run its passes, answer what its Ends answered and
/// complete, those that run past the copies the runtime lays down among them (their laying run
/// again, counted).
#[test]
fn loop_streams_run_as_metal_ran() {
    // Past the copies laid down: 2 x 2 x ... six deep, its laying cut at 64, 16, 16 and 1 copies.
    const PAST_COPIES: [&str; 13] = [
        "w128 T0 V T1",
        "D T0 z65 T1",
        "w2 T0 Q1 w17@1 T1 T2 V V T3",
        "w2 T0 Q1 w40@1 T1 T2 V V T3",
        "w2@1 T1 Q2 w2@2 T2 Q3 w2@3 T3 Q4 w2@4 T4 Q5 w2@5 T5 Q6 w2@6 T6 T50 V V V V V V",
        "D T1 Q2 D T2 Q3 D T3 Q4 D T4 T50 z2@4 z2@3 z2@2 z2@1",
        "w65 T0 T1 V T2",
        "w1000 T0 V T2",
        "D T0 z100000 T2",
        "w2@1 T1 Q2 w17@2 T2 T50 V V",
        "w3@1 T1 Q2 w100@2 T2 T50 V V",
        "T1 E D T2 z100@2 T4",
        "T1 | T3 E w70 T0 V T4",
    ];
    on_device(|device| {
        let streams = Streams::new(device);
        let mut wrong = Vec::new();
        let mut ran_again = device.work_counts().predicated_loop_runs_again;
        for (script, want, answers) in loop_cases() {
            let got = run_counted(&streams, &script);
            if let Some(error) = got.refused {
                wrong.push(format!("{script}: refused {error:?}"));
                continue;
            }
            let again = device.work_counts().predicated_loop_runs_again;
            if PAST_COPIES.contains(&script.as_str()) && again == ran_again {
                wrong.push(format!("{script}: its laying ran no time again"));
            }
            ran_again = again;
            let mut counts = vec![0; LOG_WORDS];
            for &(tag, n) in &want {
                counts[tag] = n;
            }
            let ok = (CommandBufferStatus::Completed, None);
            if (got.status, got.error) != ok || got.counts != counts || got.answers != answers {
                wrong.push(format!(
                    "{script}: {:?} {:?} answered {:?}, counts {:?}; Metal ran {want:?} \
                     answered {answers:?}",
                    got.status,
                    got.error,
                    got.answers,
                    (got.counts.iter().enumerate())
                        .filter(|(_, &n)| n > 0)
                        .collect::<Vec<_>>()
                ));
            }
        }
        assert!(wrong.is_empty(), "{}", wrong.join("\n"));
    });
}

/// Every dispatch form inside a loop runs in each pass: the forms of
/// [`every_dispatch_form_runs_as_its_block_says`] in a DoWhile of three passes mark three times
/// each.
#[test]
fn every_dispatch_form_runs_in_each_pass() {
    on_device(|device| {
        let marks = Marks::new(device);
        let forms = Forms::new(device, &marks);
        let queue = device.new_command_queue().unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        forms.resident(&mut encoder, &marks);
        encoder.encode_start_do_while().unwrap();
        forms.encode(&mut encoder, &marks, 2);
        marks.mark(&mut encoder, 0);
        let end = encoder.encode_end_do_while(Some(&marks.log), 0, LESS, 3);
        assert_eq!(end, Ok(true));
        encoder.end_encoding().unwrap();
        complete(&command_buffer);
        let mut counts = vec![0; LOG_WORDS];
        counts[0] = 3;
        let base = Forms::command_tag(2) - 6;
        for tag in base..base + FORMS {
            counts[tag as usize] = 3;
        }
        assert_eq!(marks.counts(), counts);
    });
}

/// Every dispatch form inside a loop that runs past its copies runs in each pass: the forms of
/// [`every_dispatch_form_runs_as_its_block_says`], whose indirect dispatches of `prd_mark_group`
/// and whose executions host steps record apart from the loop's command buffers, in a DoWhile of
/// 100 passes (64 copies, run again once), and in a DoWhile of 20 passes, then one more, inside a
/// DoWhile of two (16 copies of the inner loop). Each run again records the work apart again where
/// it lies, as its first run did.
#[test]
fn every_dispatch_form_runs_in_each_pass_past_the_copies() {
    on_device(|device| {
        let queue = device.new_command_queue().unwrap();
        let base = Forms::command_tag(2) - 6;
        let again = || device.work_counts().predicated_loop_runs_again;
        // One DoWhile of 100 passes.
        let marks = Marks::new(device);
        let forms = Forms::new(device, &marks);
        let before = again();
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        forms.resident(&mut encoder, &marks);
        encoder.encode_start_do_while().unwrap();
        forms.encode(&mut encoder, &marks, 2);
        marks.mark(&mut encoder, 0);
        let end = encoder.encode_end_do_while(Some(&marks.log), 0, LESS, 100);
        assert_eq!(end, Ok(true));
        encoder.end_encoding().unwrap();
        complete(&command_buffer);
        let mut counts = vec![0; LOG_WORDS];
        counts[0] = 100;
        for tag in base..base + FORMS {
            counts[tag as usize] = 100;
        }
        assert_eq!(marks.counts(), counts);
        assert_eq!(
            again() - before,
            1,
            "the 100 passes ran again past 64 copies"
        );
        // An inner DoWhile of 20 passes, then of one, in a DoWhile of two.
        let marks = Marks::new(device);
        let forms = Forms::new(device, &marks);
        let before = again();
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        forms.resident(&mut encoder, &marks);
        encoder.encode_start_do_while().unwrap();
        encoder.encode_start_do_while().unwrap();
        forms.encode(&mut encoder, &marks, 2);
        marks.mark(&mut encoder, 0);
        let end = encoder.encode_end_do_while(Some(&marks.log), 0, LESS, 20);
        assert_eq!(end, Ok(true));
        marks.mark(&mut encoder, 1);
        let end = encoder.encode_end_do_while(Some(&marks.log), 4, LESS, 2);
        assert_eq!(end, Ok(true));
        encoder.end_encoding().unwrap();
        complete(&command_buffer);
        let mut counts = vec![0; LOG_WORDS];
        counts[0] = 21;
        counts[1] = 2;
        for tag in base..base + FORMS {
            counts[tag as usize] = 21;
        }
        assert_eq!(marks.counts(), counts);
        assert!(again() > before, "the inner loop ran past its 16 copies");
    });
}

/// A loop that runs past the copies the runtime lays down (64, or 16 inside another loop) runs on
/// as Metal's runs: its check runs its laying again until it ends, as many times as the passes past
/// the copies need (counted), and its command buffer completes with every pass Metal ran (a While
/// of 65, 128 and 1000, a DoWhile of 65, an inner While of 17 and 40 in a While of 2 that resets
/// the inner count each pass).
#[test]
fn a_loop_past_its_copies_runs_on() {
    on_device(|device| {
        let ok = (CommandBufferStatus::Completed, None);
        let again = || device.work_counts().predicated_loop_runs_again;
        for (n, runs) in [(65u32, 1u64), (128, 1), (1000, 15)] {
            let before = again();
            let got = looped(device, false, |encoder, marks| {
                encoder
                    .encode_start_while(Some(&marks.log), 0, NOT_EQUAL, n)
                    .unwrap();
                marks.mark(encoder, 0);
                assert_eq!(encoder.encode_end_while(), Ok(true));
                marks.mark(encoder, 1);
            });
            assert_eq!((got.0, got.1), ok, "a While of {n}");
            assert_eq!(got.2[..2], [n, 1], "a While of {n}");
            assert_eq!(again() - before, runs, "a While of {n}: its runs again");
        }
        let before = again();
        let got = looped(device, false, |encoder, marks| {
            encoder.encode_start_do_while().unwrap();
            marks.mark(encoder, 0);
            let end = encoder.encode_end_do_while(Some(&marks.log), 0, LESS, 65);
            assert_eq!(end, Ok(true));
            marks.mark(encoder, 1);
        });
        assert_eq!((got.0, got.1), ok, "a DoWhile of 65");
        assert_eq!(got.2[..2], [65, 1], "a DoWhile of 65");
        assert_eq!(again() - before, 1, "a DoWhile of 65: its runs again");
        let zero = kernel(device, "prd_put", false);
        for (inner, runs) in [(16u32, 0u64), (17, 2), (40, 4)] {
            let before = again();
            let got = looped(device, false, |encoder, marks| {
                let log = Some(&marks.log);
                encoder.encode_start_while(log, 0, NOT_EQUAL, 2).unwrap();
                marks.mark(encoder, 0);
                put(encoder, &zero, marks, 1);
                encoder
                    .encode_start_while(log, 4, NOT_EQUAL, inner)
                    .unwrap();
                marks.mark(encoder, 1);
                marks.mark(encoder, 2);
                assert_eq!(encoder.encode_end_while(), Ok(true));
                assert_eq!(encoder.encode_end_while(), Ok(true));
                marks.mark(encoder, 3);
            });
            assert_eq!((got.0, got.1), ok, "an inner While of {inner}");
            let want = [2, inner, 2 * inner, 1];
            assert_eq!(got.2[..4], want, "an inner While of {inner}");
            assert_eq!(
                again() - before,
                runs,
                "an inner While of {inner}: its runs again"
            );
        }
    });
}

/// What a loop's check runs again is the loop's laying alone: a render pass that clears a texture,
/// a blit into it and a blit into the log, encoded before the loop in its command buffer, run
/// once, as Metal runs them. The loop (`while (log[3] != 105) { mark 3; mark 4 }`, from the blit's
/// 5) runs 100 passes, its laying once again; a run again of the blit would run more passes, and
/// one of the clear would leave the texture red.
#[test]
fn a_loop_run_again_repeats_nothing_before_it() {
    on_device(|device| {
        let marks = Marks::new(device);
        let queue = device.new_command_queue().unwrap();
        let mut descriptor = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, 1, 1, false);
        descriptor.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
        let texture = device.new_texture(&descriptor).unwrap();
        let green = buffer_with(device, &[0, 255, 0, 255]).unwrap();
        let five = buffer_with(device, &bytes(&[5])).unwrap();
        let out = buffer_with(device, &[0; 4]).unwrap();
        let before = device.work_counts().predicated_loop_runs_again;
        let command_buffer = queue.command_buffer().unwrap();
        let mut pass = RenderPassDescriptor::new();
        let color = &mut pass.color_attachments[0];
        color.texture = Some(texture.clone());
        color.load_action = LoadAction::Clear;
        color.store_action = StoreAction::Store;
        color.clear_color = ClearColor::new(1.0, 0.0, 0.0, 1.0);
        command_buffer
            .render_command_encoder(&pass)
            .unwrap()
            .end_encoding()
            .unwrap();
        let (origin, size) = (Origin::new(0, 0, 0), Size::new(1, 1, 1));
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.copy_from_buffer_to_texture(
            &green,
            0,
            4,
            4,
            size,
            &texture,
            0,
            0,
            origin,
            BlitOption::None,
        )
        .unwrap();
        blit.copy_from_buffer_to_buffer(&five, 0, &marks.log, 12, 4)
            .unwrap();
        blit.end_encoding().unwrap();
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        encoder
            .encode_start_while(Some(&marks.log), 12, NOT_EQUAL, 105)
            .unwrap();
        marks.mark(&mut encoder, 3);
        marks.mark(&mut encoder, 4);
        assert_eq!(encoder.encode_end_while(), Ok(true));
        encoder.end_encoding().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.copy_from_texture_to_buffer(
            &texture,
            0,
            0,
            origin,
            size,
            &out,
            0,
            4,
            4,
            BlitOption::None,
        )
        .unwrap();
        blit.end_encoding().unwrap();
        complete(&command_buffer);
        assert_eq!(marks.counts()[3..5], [105, 100]);
        assert_eq!(read_buffer(&out, 0, 4).unwrap(), [0, 255, 0, 255]);
        let again = device.work_counts().predicated_loop_runs_again - before;
        assert_eq!(again, 1, "the laying ran once again");
    });
}

/// A loop without bound (`while (log[1] != 1) mark 0`, log[1] staying 0) runs on, its laying run
/// again and again, as Metal's runs (an M3 Max with macOS 27.0.1 ran one 180 s with no error),
/// after the program let go of its queue, and of its command buffer too: Metal's GPU runs it on
/// until the process ends, the command buffer never completing. The process's exit, after the
/// test, ends it.
#[test]
fn a_loop_without_bound_runs_on_after_its_queue_is_let_go() {
    on_device(|device| {
        let marks = Marks::new(device);
        let queue = device.new_command_queue().unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        encoder
            .encode_start_while(Some(&marks.log), 4, NOT_EQUAL, 1)
            .unwrap();
        marks.mark(&mut encoder, 0);
        assert_eq!(encoder.encode_end_while(), Ok(true));
        marks.mark(&mut encoder, 2);
        encoder.end_encoding().unwrap();
        let again = || device.work_counts().predicated_loop_runs_again;
        let before = again();
        command_buffer.commit().unwrap();
        let ran = |more: u64| {
            let start = std::time::Instant::now();
            while again() < before + more {
                assert!(
                    start.elapsed() < WAIT,
                    "its laying ran {} times again",
                    again() - before
                );
                std::thread::yield_now();
            }
        };
        ran(3);
        drop(queue);
        ran(6);
        let running = !matches!(
            command_buffer.status(),
            CommandBufferStatus::Completed | CommandBufferStatus::Error
        );
        assert!(running, "{:?}", command_buffer.status());
        drop(command_buffer);
        ran(9);
        assert_eq!(marks.counts()[2], 0, "nothing after the loop ran");
    });
}

/// Commit on `device` a loop without bound, as
/// [`a_loop_without_bound_runs_on_after_its_queue_is_let_go`] does, and return once its laying ran
/// again `times` times; its queue and command buffer let go, it runs on to the process's end
/// (`runtime_lane::exits`).
pub(super) fn loop_without_bound(device: &Device, times: u64) {
    let marks = Marks::new(device);
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder
        .encode_start_while(Some(&marks.log), 4, NOT_EQUAL, 1)
        .unwrap();
    marks.mark(&mut encoder, 0);
    assert_eq!(encoder.encode_end_while(), Ok(true));
    encoder.end_encoding().unwrap();
    let again = || device.work_counts().predicated_loop_runs_again;
    let before = again();
    command_buffer.commit().unwrap();
    let start = std::time::Instant::now();
    while again() < before + times {
        assert!(
            start.elapsed() < WAIT,
            "its laying ran {} times again",
            again() - before
        );
        std::thread::yield_now();
    }
}

/// The shapes of [`a_runs_first_indirect_dispatch_of_another_type_runs_every_dispatch`]: each
/// the consumer's name for it and its script.
const INDIRECT_SHAPES: [(&str, &str); 18] = [
    ("C Direct, S G", "% T1 | G2"),
    ("C Direct, S T", "% T1 | H2"),
    ("S Direct, C G", "| T1 % G2"),
    ("S Direct, C T", "| T1 % H2"),
    ("C Direct, type 0 G", "% T1 | G2"),
    ("C nothing, S G", "% | G2"),
    ("C Direct, S Direct, S G", "% T1 | T2 | G3"),
    ("C G, S Direct", "% G1 | T2"),
    ("C Direct, C G", "% T1 % G2"),
    ("S Direct, S G", "| T1 | G2"),
    ("type 2 Direct, S G", "| T1 | G2"),
    ("C Direct, Blit, S G", "% T1 B | G2"),
    ("C Direct, Signal, S G", "% T1 S | G2"),
    ("C Direct, Wait, S G", "% T1 A | G2"),
    ("C G, S Direct, S G", "% G1 | T2 | G3"),
    ("S Direct, C Direct, S G", "| T1 % T2 | G3"),
    ("S G, C T", "| G1 % H2"),
    ("C G, S Direct, C G", "% G1 | T2 % G3"),
];

/// A run of compute encoders' first indirect dispatch in an encoder of another dispatch type than
/// the run's first encoder's, with no predication record, runs every dispatch: each of the
/// consumer's 18 shapes, as our M3 Max probe's 12 such shapes ran (macOS 27.0.1). `|` and `%` open
/// a serial and a concurrent encoder (Metal counts a raw type that is neither as serial, so the
/// consumer's types 0 and 2 are `|`); `Tn` marks `n` directly, `Gn` and `Hn` by the two indirect
/// dispatches; `B` encodes a blit, `S` an event signal and `A` an event wait between.
#[test]
fn a_runs_first_indirect_dispatch_of_another_type_runs_every_dispatch() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        println!("PRD device {}", device.name());
        let streams = Streams::new(&device);
        let event = device.new_shared_event().unwrap();
        event.set_signaled_value(5);
        for (name, script) in INDIRECT_SHAPES {
            let tags: Vec<u32> = (script.split_whitespace())
                .filter_map(|token| token[1..].parse().ok())
                .collect();
            assert_eq!(
                run_shape(&streams, &event, script),
                tags,
                "{name} `{script}`"
            );
        }
    }
    layer_reported_nothing_at_all(&messages);
}

/// [`a_runs_first_indirect_dispatch_of_another_type_runs_every_dispatch`]'s run of `script`: the
/// tags that ran.
fn run_shape(
    streams: &Streams,
    event: &metal2vulkan_runtime::SharedEvent,
    script: &str,
) -> Vec<u32> {
    let marks = &streams.marks;
    write_words(&marks.log, &[0; LOG_WORDS]);
    let command_buffer = streams.queue.command_buffer().unwrap();
    let mut open: Option<ComputeCommandEncoder> = None;
    for token in script.split_whitespace() {
        if matches!(token, "|" | "%" | "B" | "S" | "A") {
            if let Some(encoder) = open.take() {
                encoder.end_encoding().unwrap();
            }
        }
        match token {
            "|" | "%" => {
                let dispatch_type = match token {
                    "%" => DispatchType::Concurrent,
                    _ => DispatchType::Serial,
                };
                open = Some(
                    command_buffer
                        .compute_command_encoder_with_dispatch_type(dispatch_type)
                        .unwrap(),
                );
            }
            "B" => {
                let mut blit = command_buffer.blit_command_encoder().unwrap();
                blit.fill_buffer(&marks.word, 8..12, 1).unwrap();
                blit.end_encoding().unwrap();
            }
            "S" => command_buffer
                .encode_signal_event(EventRef::from(event), 6)
                .unwrap(),
            "A" => command_buffer
                .encode_wait_for_event(EventRef::from(event), 1)
                .unwrap(),
            _ => {
                let encoder = open.as_mut().unwrap();
                let tag = token[1..].parse().unwrap();
                marks.bind(encoder, tag);
                let dispatched = match &token[..1] {
                    "T" => encoder.dispatch_threads(one(), one()),
                    "G" => encoder.dispatch_threadgroups_indirect(&streams.arguments, 0, one()),
                    _ => encoder.dispatch_threads_indirect(&streams.arguments, 16),
                };
                dispatched.unwrap_or_else(|error| panic!("{script}: {error:?}"));
            }
        }
    }
    if let Some(encoder) = open {
        encoder.end_encoding().unwrap();
    }
    complete(&command_buffer);
    let counts = marks.counts();
    (0..LOG_WORDS as u32)
        .filter(|&tag| counts[tag as usize] != 0)
        .collect()
}
