//! Which threads of a 2D kernel's threadgroups run, at the shapes of a
//! picture conversion (a 3840x2160 picture, each thread writing 4x2 texels: a grid of 960x1088
//! threads in threadgroups of 32x32, 30x34 of them), for a kernel that reads its threadgroup's
//! position and size (`grid_words`) and one that reads only its position in the grid
//! (`grid_words_oblivious`), by `dispatchThreads` and by `dispatchThreadgroups`. Each thread
//! stores its index in the grid plus 1 at that index, `x` fastest, over a buffer filled with
//! 0xff. One line a case: every word right, or how many are, and where the right ones sit.
//!
//! `llvm-as` assembles the kernels; without it the test skips visibly.

#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod on_device {
    use crate::tests::command::on_device::{each_device, finished};
    use crate::{
        ComputePipelineDescriptor, DispatchType, FunctionDescriptor, Library, MemoryClass,
        StageSource,
    };
    use metal2vulkan_devtools::{llvm_assemble_text, AssembleError};

    /// Thread `group * size + pos` stores its index in the grid plus 1 there.
    const GRID_WORDS: &str = "; Owned synthetic kernel.
source_filename = \"grid_words.metal\"

define void @grid_words(ptr addrspace(1) %out, <3 x i32> %group, <3 x i32> %size, <3 x i32> %pos, <3 x i32> %grid) {
entry:
  %g = mul <3 x i32> %group, %size
  %q = add <3 x i32> %g, %pos
  %x = extractelement <3 x i32> %q, i64 0
  %y = extractelement <3 x i32> %q, i64 1
  %gx = extractelement <3 x i32> %grid, i64 0
  %rx = mul i32 %y, %gx
  %i = add i32 %rx, %x
  %v = add i32 %i, 1
  %w = zext i32 %i to i64
  %p = getelementptr inbounds i32, ptr addrspace(1) %out, i64 %w
  store i32 %v, ptr addrspace(1) %p, align 4
  ret void
}

define void @grid_words_oblivious(ptr addrspace(1) %out, <3 x i32> %q, <3 x i32> %grid) {
entry:
  %x = extractelement <3 x i32> %q, i64 0
  %y = extractelement <3 x i32> %q, i64 1
  %gx = extractelement <3 x i32> %grid, i64 0
  %rx = mul i32 %y, %gx
  %i = add i32 %rx, %x
  %v = add i32 %i, 1
  %w = zext i32 %i to i64
  %p = getelementptr inbounds i32, ptr addrspace(1) %out, i64 %w
  store i32 %v, ptr addrspace(1) %p, align 4
  ret void
}

!air.kernel = !{!0, !3}
!0 = !{ptr @grid_words, !1, !2}
!1 = !{}
!2 = !{!10, !11, !12, !13, !14}
!3 = !{ptr @grid_words_oblivious, !1, !4}
!4 = !{!10, !15, !16}
!10 = !{i32 0, !\"air.buffer\", !\"air.location_index\", i32 0, i32 1, !\"air.write\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"out\"}
!11 = !{i32 1, !\"air.threadgroup_position_in_grid\", !\"air.arg_type_name\", !\"uint3\", !\"air.arg_name\", !\"group\"}
!12 = !{i32 2, !\"air.threads_per_threadgroup\", !\"air.arg_type_name\", !\"uint3\", !\"air.arg_name\", !\"size\"}
!13 = !{i32 3, !\"air.thread_position_in_threadgroup\", !\"air.arg_type_name\", !\"uint3\", !\"air.arg_name\", !\"pos\"}
!14 = !{i32 4, !\"air.threads_per_grid\", !\"air.arg_type_name\", !\"uint3\", !\"air.arg_name\", !\"grid\"}
!15 = !{i32 1, !\"air.thread_position_in_grid\", !\"air.arg_type_name\", !\"uint3\", !\"air.arg_name\", !\"q\"}
!16 = !{i32 2, !\"air.threads_per_grid\", !\"air.arg_type_name\", !\"uint3\", !\"air.arg_name\", !\"grid\"}
";

    /// Every thread of a 2D grid writes its own word at the picture conversion's shapes, or the
    /// line says which threads did.
    #[test]
    fn every_device_runs_every_thread_of_a_picture_conversions_threadgroups() {
        let bitcode = match llvm_assemble_text(GRID_WORDS) {
            Ok(bitcode) => bitcode,
            Err(AssembleError::Missing(why)) => {
                eprintln!("SKIPPED: {why}");
                return;
            }
            Err(error) => panic!("grid_words: {error:?}"),
        };
        let library = Library::from_air(&bitcode).expect("the library");
        // (threadgroup, grid in threads, by dispatchThreadgroups)
        let cases: [([u32; 3], [u32; 3], bool); 5] = [
            ([32, 32, 1], [960, 1088, 1], false),
            ([32, 32, 1], [960, 1088, 1], true),
            ([16, 16, 1], [64, 64, 1], false),
            ([16, 16, 1], [64, 64, 1], true),
            ([8, 8, 1], [16, 16, 1], false),
        ];
        each_device("wg_lattice", |name, _, device| {
            let queue = device.new_command_queue(4).expect("a queue");
            let mut bad = 0;
            for kernel in ["grid_words", "grid_words_oblivious"] {
                let function = library.function(kernel).expect("the kernel");
                let descriptor = ComputePipelineDescriptor::new(StageSource::Function(
                    FunctionDescriptor::new(function),
                ));
                let state = device
                    .new_compute_pipeline_state(&descriptor, crate::tests::no_constexpr_samplers)
                    .unwrap_or_else(|error| panic!("{name}: {kernel}: {error}"));
                for (threadgroup, grid, by_groups) in cases {
                    let words = grid[0] * grid[1];
                    let len = u64::from(words * 4);
                    let block =
                        (device.take_block(MemoryClass::Shared, len)).expect("a Shared block");
                    let output =
                        (device.new_buffer(block.id, 0, len, Some(0))).expect("the output");
                    let contents = output.contents().expect("a Shared buffer");
                    // SAFETY: the buffer is alive and nothing else reaches its bytes.
                    unsafe { std::ptr::write_bytes(contents.as_ptr(), 0xff, len as usize) };
                    let command_buffer = queue.command_buffer().expect("a command buffer");
                    let mut encoder = command_buffer
                        .compute_command_encoder(DispatchType::Serial)
                        .expect("an encoder");
                    encoder.set_compute_pipeline_state(&state).expect("a state");
                    encoder.set_buffer(Some(&output), 0, 0).expect("a buffer");
                    let dispatched = match by_groups {
                        true => {
                            let groups = [grid[0] / threadgroup[0], grid[1] / threadgroup[1], 1];
                            encoder.dispatch_threadgroups(groups, threadgroup)
                        }
                        false => encoder.dispatch_threads(grid, threadgroup),
                    };
                    dispatched.unwrap_or_else(|error| panic!("{name}: the dispatch: {error}"));
                    encoder.end_encoding();
                    command_buffer.commit().expect("a commit");
                    finished(name, &command_buffer);
                    assert_eq!(command_buffer.error(), None, "{name}");
                    // SAFETY: the words lie in the buffer, which is alive, and its work completed.
                    let got = |i: u32| unsafe {
                        (contents.as_ptr().cast::<u32>())
                            .add(i as usize)
                            .read_unaligned()
                    };
                    let right: Vec<u32> = (0..words).filter(|&i| got(i) == i + 1).collect();
                    let how = if by_groups { "threadgroups" } else { "threads" };
                    let what = format!(
                        "{name}: {kernel}: {grid:?} by dispatch {how} in threadgroups of {threadgroup:?}"
                    );
                    if right.len() == words as usize {
                        println!("lattice: {what}: every word right ({words})");
                    } else {
                        bad += 1;
                        let mut xs: Vec<u32> = right.iter().map(|&i| i % grid[0]).collect();
                        let mut ys: Vec<u32> = right.iter().map(|&i| i / grid[0]).collect();
                        xs.sort_unstable();
                        xs.dedup();
                        ys.sort_unstable();
                        ys.dedup();
                        println!(
                            "lattice: {what}: WRONG, {} of {words} words right; right at x {:?} y {:?}",
                            right.len(),
                            &xs[..xs.len().min(8)],
                            &ys[..ys.len().min(8)]
                        );
                    }
                    drop(output);
                    (device.release_block(block)).unwrap_or_else(|e| panic!("{name}: {e}"));
                }
            }
            assert_eq!(bad, 0, "{name}: {bad} cases with threads that did not run");
        });
    }
}
