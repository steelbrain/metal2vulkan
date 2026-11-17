//! A buffer slot's uniform view in its texel form (`DescriptorView::UniformTexel`,
//! `uniform_texels.rs`), on lavapipe under a device whose view check's facts the test fixes
//! (`DeviceOptions::view_facts`) to `DriverWorkaround::ExternalUniformReadsSkipACache`: a kernel
//! whose `constant` window is at most 4 KiB, read at an index (`tid` and `tid + 1023`, the one
//! thread's `tid` 0), reads it through a uniform texel buffer view made for each bind, and every
//! read is the bound bytes', at bind offsets the texel alignment divides (0, 16, 256) and does
//! not (4, 260: the residual pushed), from `set*Bytes` data, and for bytes written before the
//! dispatch was encoded, after it and before the commit, and by a blit before it. A window of
//! 4100 bytes keeps its uniform buffer. The counts are the evidence that each path ran
//! (`WorkCounts::uniform_texels_bound`, `uniform_buffers_bound_to_16k`).
//!
//! A device test on the CPU ICD alone (`m2v_cpu_icd`): lavapipe stands in for a device whose
//! check found the fact. The kernels are [`WINDOWS`] (`llvm-as`).

use super::blit_holds::{options, Scratch, WAIT};
use crate::{Buffer, ComputePipelineState, Device, PipelineOptions, ResourceOptions, Size};
use metal2vulkan::device::DriverWorkaround;

/// `utw_near` reads words `tid` and `tid + 1023` of a 4096-byte `constant` window into `out`;
/// `utw_far` words `tid` and `tid + 1024` of a 4100-byte one.
const WINDOWS: &str = r#"define void @utw_near(ptr addrspace(2) noundef "air-buffer-no-alias" %0, ptr addrspace(1) noundef "air-buffer-no-alias" %1, i32 noundef %tid) {
entry:
  %a = zext i32 %tid to i64
  %p = getelementptr inbounds [1024 x i32], ptr addrspace(2) %0, i64 0, i64 %a
  %v = load i32, ptr addrspace(2) %p, align 4
  store i32 %v, ptr addrspace(1) %1, align 4
  %b = add nuw i64 %a, 1023
  %q = getelementptr inbounds [1024 x i32], ptr addrspace(2) %0, i64 0, i64 %b
  %w = load i32, ptr addrspace(2) %q, align 4
  %o = getelementptr inbounds i32, ptr addrspace(1) %1, i64 1
  store i32 %w, ptr addrspace(1) %o, align 4
  ret void
}
define void @utw_far(ptr addrspace(2) noundef "air-buffer-no-alias" %0, ptr addrspace(1) noundef "air-buffer-no-alias" %1, i32 noundef %tid) {
entry:
  %a = zext i32 %tid to i64
  %p = getelementptr inbounds [1025 x i32], ptr addrspace(2) %0, i64 0, i64 %a
  %v = load i32, ptr addrspace(2) %p, align 4
  store i32 %v, ptr addrspace(1) %1, align 4
  %b = add nuw i64 %a, 1024
  %q = getelementptr inbounds [1025 x i32], ptr addrspace(2) %0, i64 0, i64 %b
  %w = load i32, ptr addrspace(2) %q, align 4
  %o = getelementptr inbounds i32, ptr addrspace(1) %1, i64 1
  store i32 %w, ptr addrspace(1) %o, align 4
  ret void
}
!air.kernel = !{!10, !20}
!air.version = !{!17}
!air.language_version = !{!18}
!10 = !{ptr @utw_near, !11, !12}
!11 = !{}
!12 = !{!13, !14, !15}
!13 = !{i32 0, !"air.buffer", !"air.buffer_size", i32 4096, !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.arg_type_size", i32 4096, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"Near", !"air.arg_name", !"c"}
!14 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
!15 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"tid"}
!20 = !{ptr @utw_far, !11, !21}
!21 = !{!22, !14, !15}
!22 = !{i32 0, !"air.buffer", !"air.buffer_size", i32 4100, !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.arg_type_size", i32 4100, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"Far", !"air.arg_name", !"c"}
!17 = !{i32 2, i32 8, i32 0}
!18 = !{!"Metal", i32 4, i32 0, i32 0}
"#;

/// The constants' words: word `i` is `3 i + 1`, so each word read names its place.
const WORDS: u32 = 4096;

fn word(i: u64) -> u32 {
    3 * i as u32 + 1
}

/// A device whose view check's facts the test fixes to the texel fact, and so its uniform texel
/// buffers.
fn device(cache: &std::path::Path) -> Device {
    let mut options = options(cache);
    let fact = DriverWorkaround::ExternalUniformReadsSkipACache;
    options.view_facts = Some([fact].into());
    let device = Device::system_default_with(&options).unwrap();
    let workarounds = device.compile_device().profile().workarounds();
    assert!(
        workarounds.contains(&fact),
        "the compile device's profile takes the texel form: {workarounds:?}"
    );
    device
}

/// The compute state of `name`, a kernel of [`WINDOWS`].
fn state(device: &Device, name: &str) -> ComputePipelineState {
    let bitcode = metal2vulkan_devtools::llvm_assemble_text(WINDOWS).unwrap();
    let library = metal2vulkan::library::Library::from_air(&bitcode).unwrap();
    let function = device
        .new_library_with_compile_library(library)
        .new_function_with_name(name)
        .unwrap();
    device
        .new_compute_pipeline_state_with_function(&function, &PipelineOptions::default())
        .unwrap()
}

/// The constants' bytes: [`WORDS`] words, then `extra` words more of the same rule.
fn constants(extra: u32) -> Vec<u8> {
    (0..u64::from(WORDS + extra))
        .flat_map(|i| word(i).to_le_bytes())
        .collect()
}

/// A `Shared` buffer holding `bytes`.
fn holding(device: &Device, bytes: &[u8]) -> Buffer {
    let buffer = device
        .new_buffer(bytes.len() as u64, ResourceOptions::shared())
        .unwrap();
    write(&buffer, 0, bytes);
    buffer
}

/// Write `bytes` into `buffer` from byte `at`, which no command reads or writes now.
fn write(buffer: &Buffer, at: usize, bytes: &[u8]) {
    assert!(at + bytes.len() <= buffer.length() as usize);
    let contents = buffer.contents().unwrap();
    // SAFETY: inside the buffer's bytes, which no command reads or writes now.
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), contents.as_ptr().add(at), bytes.len())
    };
}

/// The first two words of `buffer`, which no command uses now.
fn words(buffer: &Buffer) -> [u32; 2] {
    let contents = buffer.contents().unwrap();
    // SAFETY: the buffer's first 8 bytes, which no command uses now.
    let bytes = unsafe { std::slice::from_raw_parts(contents.as_ptr(), 8) };
    [0, 1].map(|w| u32::from_le_bytes(bytes[w * 4..w * 4 + 4].try_into().unwrap()))
}

/// Where a dispatch's constants are bound: a buffer at an offset, or `set*Bytes` data.
#[derive(Clone, Copy, Debug)]
enum Bound {
    At(u64),
    Bytes,
}

/// Dispatch `state` once per bind of `bound`, the constants bound from `constants` at slot 0 and
/// a fresh output at slot 1, in one command buffer; each dispatch's output words.
fn dispatch(
    device: &Device,
    state: &ComputePipelineState,
    constants: &Buffer,
    bound: &[Bound],
    bytes: &[u8],
) -> Vec<[u32; 2]> {
    let queue = device.new_command_queue().unwrap();
    let outputs: Vec<Buffer> = (bound.iter())
        .map(|_| device.new_buffer(16, ResourceOptions::shared()).unwrap())
        .collect();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(state).unwrap();
    for (bind, output) in bound.iter().zip(&outputs) {
        match *bind {
            Bound::At(at) => encoder.set_buffer(Some(constants), at, 0).unwrap(),
            Bound::Bytes => encoder.set_bytes(bytes, 0).unwrap(),
        }
        encoder.set_buffer(Some(output), 0, 1).unwrap();
        encoder
            .dispatch_threadgroups(Size::new(1, 1, 1), Size::new(1, 1, 1))
            .unwrap();
    }
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
    outputs.iter().map(words).collect()
}

/// A window of 4096 bytes reads through a uniform texel buffer at bind offsets 0, 4, 16, 256 and
/// 260, each read the bound bytes', a view bound for each dispatch, and from 4096 bytes of
/// `set*Bytes` data likewise.
#[test]
fn a_short_window_reads_through_a_texel_view_at_every_bind_offset() {
    let scratch = Scratch::new("uniform-texels-offsets");
    let device = device(&scratch.0);
    let near = state(&device, "utw_near");
    let data = constants(256);
    let constants = holding(&device, &data);
    let offsets = [0, 4, 16, 256, 260];
    let bound: Vec<Bound> = offsets.iter().map(|&at| Bound::At(at)).collect();
    let before = device.work_counts();
    let read = dispatch(&device, &near, &constants, &bound, &[]);
    let after = device.work_counts();
    for (at, read) in offsets.iter().zip(&read) {
        let first = at / 4;
        assert_eq!(*read, [word(first), word(first + 1023)], "offset {at}");
    }
    assert!(
        after.uniform_texels_bound >= before.uniform_texels_bound + offsets.len() as u64,
        "texel views bound: {} then {}",
        before.uniform_texels_bound,
        after.uniform_texels_bound
    );
    assert_eq!(
        after.uniform_buffers_bound_to_16k, before.uniform_buffers_bound_to_16k,
        "no short window bound as a uniform buffer"
    );
    // `set*Bytes` data, the constants' words 2 to 1025.
    let before = device.work_counts();
    let read = dispatch(&device, &near, &constants, &[Bound::Bytes], &data[8..4104]);
    let after = device.work_counts();
    assert_eq!(read[0], [word(2), word(1025)], "set*Bytes data");
    assert!(
        after.uniform_texels_bound > before.uniform_texels_bound,
        "a texel view bound for set*Bytes data"
    );
}

/// The bytes a dispatch reads through a texel view are those the buffer holds as it runs: written
/// before the dispatch was encoded, after it and before the commit, and by a blit encoded before
/// it in the same command buffer.
#[test]
fn a_texel_view_reads_the_bytes_written_before_its_dispatch_runs() {
    let scratch = Scratch::new("uniform-texels-writes");
    let device = device(&scratch.0);
    let near = state(&device, "utw_near");
    let queue = device.new_command_queue().unwrap();
    let data = constants(0);
    let constants = holding(&device, &data);
    let marks = |base: u32| -> Vec<u8> {
        [base, base + 1]
            .iter()
            .flat_map(|w| w.to_le_bytes())
            .collect()
    };
    // Words 0 and 1023 of each case: before encoding, after it, by a blit.
    let cases = [0x1000_0000u32, 0x2000_0000, 0x3000_0000];
    let outputs: Vec<Buffer> = (cases.iter())
        .map(|_| device.new_buffer(16, ResourceOptions::shared()).unwrap())
        .collect();
    let encode = |command_buffer: &crate::CommandBuffer, output: &Buffer| {
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        encoder.set_compute_pipeline_state(&near).unwrap();
        encoder.set_buffer(Some(&constants), 0, 0).unwrap();
        encoder.set_buffer(Some(output), 0, 1).unwrap();
        encoder
            .dispatch_threadgroups(Size::new(1, 1, 1), Size::new(1, 1, 1))
            .unwrap();
        encoder.end_encoding().unwrap();
    };
    let run = |command_buffer: crate::CommandBuffer| {
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.error(), None);
    };
    let put = |base: u32| {
        let [first, last] = [0, 1].map(|w| (base + w).to_le_bytes());
        write(&constants, 0, &first);
        write(&constants, 4092, &last);
    };
    // Before encoding.
    put(cases[0]);
    let command_buffer = queue.command_buffer().unwrap();
    encode(&command_buffer, &outputs[0]);
    run(command_buffer);
    // After encoding, before the commit.
    put(0);
    let command_buffer = queue.command_buffer().unwrap();
    encode(&command_buffer, &outputs[1]);
    put(cases[1]);
    run(command_buffer);
    // By a blit before the dispatch, from a buffer of the case's words.
    put(0);
    let source = holding(&device, &marks(cases[2]));
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.copy_from_buffer_to_buffer(&source, 0, &constants, 0, 4)
        .unwrap();
    blit.copy_from_buffer_to_buffer(&source, 4, &constants, 4092, 4)
        .unwrap();
    blit.end_encoding().unwrap();
    encode(&command_buffer, &outputs[2]);
    run(command_buffer);
    for ((base, output), how) in cases.iter().zip(&outputs).zip(["before", "after", "blit"]) {
        assert_eq!(words(output), [*base, base + 1], "written {how}");
    }
}

/// A window of 4100 bytes, past the texel form's 4096, keeps its uniform buffer: its reads are the
/// bound bytes', at bind offsets 0 and 4, and the device binds no texel view for it.
#[test]
fn a_window_past_the_bound_keeps_its_uniform_buffer() {
    let scratch = Scratch::new("uniform-texels-far");
    let device = device(&scratch.0);
    let far = state(&device, "utw_far");
    let data = constants(256);
    let constants = holding(&device, &data);
    let offsets = [0, 4];
    let bound: Vec<Bound> = offsets.iter().map(|&at| Bound::At(at)).collect();
    let before = device.work_counts();
    let read = dispatch(&device, &far, &constants, &bound, &[]);
    let after = device.work_counts();
    for (at, read) in offsets.iter().zip(&read) {
        let first = at / 4;
        assert_eq!(*read, [word(first), word(first + 1024)], "offset {at}");
    }
    assert_eq!(
        after.uniform_texels_bound, before.uniform_texels_bound,
        "texel views bound for a window past the bound"
    );
    assert!(
        after.uniform_buffers_bound_to_16k >= before.uniform_buffers_bound_to_16k + 2,
        "uniform buffers of windows to 16 KiB bound: {} then {}",
        before.uniform_buffers_bound_to_16k,
        after.uniform_buffers_bound_to_16k
    );
}
