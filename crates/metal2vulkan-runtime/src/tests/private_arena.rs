//! The private arena (`crate::arena`), under a device cfg on every device of the cfg's kind: an
//! owned synthetic kernel, `arena`, keeps a private array of `N` words past the translator's
//! floor, fills it with `31 gid + i` and folds it back read at `(7 j + gid) mod N` as
//! `s = 3 s + a[k]`, so each thread's answer depends on every word of its own slot in an order no
//! other slot's words reproduce. Its thread's index is its threadgroup's position times 64 plus
//! its position in it, so it observes its threadgroups and runs in Metal's: a kernel that reads
//! only its position in the grid may run in workgroups of the runtime's own shape, whose pieces
//! differ.
//!
//! It runs at 7176 words (28,704 bytes a thread), 9216 (36 KiB) and 51,200 (200 KiB), over 8
//! threadgroups of 64, under the device's own budget and under a budget of a test's
//! (`Core::arena_budget`) that splits the dispatch: two threadgroups a piece at 28,704 bytes, four
//! at 36 KiB and one at 200 KiB. The device's own budget splits a dispatch too where its slots
//! pass it, into the pieces that budget holds. Every answer is the CPU's. The counts show the
//! path: one arena dispatch each, its pieces, a barrier before each piece but the first in a
//! block the dispatch took new, and none for a kernel without an arena, whose dispatches beside
//! it add no barrier.
//!
//! `llvm-as` assembles the kernels; without it the tests skip visibly.

#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod on_device {
    use crate::tests::command::on_device::{each_device, finished};
    use crate::{
        Buffer, ComputePipelineDescriptor, ComputePipelineState, Device, DispatchType,
        FunctionDescriptor, Library, MemoryClass, StageSource,
    };
    use metal2vulkan_devtools::{llvm_assemble_text, AssembleError};
    use std::sync::atomic::Ordering::Relaxed;

    /// Threadgroups of a dispatch, and threads of each.
    const GROUPS: u32 = 8;
    const THREADS: u32 = 64;

    /// The kernel `arena` of `n` private words and the kernel `plain`, which writes its thread's
    /// index and keeps no private memory.
    fn module(n: u32) -> String {
        let threads = THREADS;
        format!(
            r#"; Owned synthetic kernels: a private array of {n} words, and none.
source_filename = "private_arena.metal"

define void @arena(ptr addrspace(1) %out, i32 %tg, i32 %tid) {{
entry:
  %a = alloca [{n} x i32], align 4
  %first = mul i32 %tg, {threads}
  %gid = add i32 %first, %tid
  %m = mul i32 %gid, 31
  br label %fill
fill:
  %i = phi i32 [ 0, %entry ], [ %i1, %fill ]
  %v = add i32 %m, %i
  %p = getelementptr inbounds [{n} x i32], ptr %a, i32 0, i32 %i
  store i32 %v, ptr %p, align 4
  %i1 = add i32 %i, 1
  %more = icmp ult i32 %i1, {n}
  br i1 %more, label %fill, label %read
read:
  %j = phi i32 [ 0, %fill ], [ %j1, %read ]
  %s = phi i32 [ 0, %fill ], [ %s1, %read ]
  %j7 = mul i32 %j, 7
  %k0 = add i32 %j7, %gid
  %k = urem i32 %k0, {n}
  %q = getelementptr inbounds [{n} x i32], ptr %a, i32 0, i32 %k
  %x = load i32, ptr %q, align 4
  %s3 = mul i32 %s, 3
  %s1 = add i32 %s3, %x
  %j1 = add i32 %j, 1
  %again = icmp ult i32 %j1, {n}
  br i1 %again, label %read, label %done
done:
  %g = zext i32 %gid to i64
  %o = getelementptr inbounds i32, ptr addrspace(1) %out, i64 %g
  store i32 %s1, ptr addrspace(1) %o, align 4
  ret void
}}

define void @plain(ptr addrspace(1) %out, i32 %gid) {{
entry:
  %g = zext i32 %gid to i64
  %o = getelementptr inbounds i32, ptr addrspace(1) %out, i64 %g
  store i32 %gid, ptr addrspace(1) %o, align 4
  ret void
}}

!air.kernel = !{{!0, !3}}
!0 = !{{ptr @arena, !1, !4}}
!3 = !{{ptr @plain, !1, !2}}
!1 = !{{}}
!2 = !{{!10, !12}}
!4 = !{{!10, !13, !14}}
!10 = !{{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}}
!12 = !{{i32 1, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"gid"}}
!13 = !{{i32 1, !"air.threadgroup_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"tg"}}
!14 = !{{i32 2, !"air.thread_position_in_threadgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"tid"}}
"#
        )
    }

    /// What thread `gid` of `arena` of `n` words writes.
    fn reference(n: u32, gid: u32) -> u32 {
        let a: Vec<u32> = (0..n)
            .map(|i| gid.wrapping_mul(31).wrapping_add(i))
            .collect();
        (0..n).fold(0u32, |s, j| {
            let k = (j.wrapping_mul(7).wrapping_add(gid)) % n;
            s.wrapping_mul(3).wrapping_add(a[k as usize])
        })
    }

    /// The state of kernel `entry` of `library` on `device`.
    fn state(name: &str, device: &Device, library: &Library, entry: &str) -> ComputePipelineState {
        let function = library.function(entry).expect("the kernel");
        let descriptor = ComputePipelineDescriptor::new(StageSource::Function(
            FunctionDescriptor::new(function),
        ));
        (device.new_compute_pipeline_state(&descriptor, crate::tests::no_constexpr_samplers))
            .unwrap_or_else(|error| panic!("{name}: {entry}: {error}"))
    }

    /// The words of `buffer`, read once its writer completed.
    fn words(buffer: &Buffer, count: usize) -> Vec<u32> {
        let contents = buffer.contents().expect("a Shared buffer");
        // SAFETY: the work that wrote the buffer completed, and its writes reach the host; the
        // buffer holds at least `count` words.
        unsafe { std::slice::from_raw_parts(contents.as_ptr().cast::<u32>(), count) }.to_vec()
    }

    /// The barriers of one command buffer that dispatches `kernels` in turn, one encoder, each into
    /// its own of `outs`.
    fn barriers(
        name: &str,
        device: &Device,
        kernels: &[&ComputePipelineState],
        outs: &[Buffer],
    ) -> (u64, crate::Metrics) {
        let counters = device.counters();
        let before = counters.metrics();
        let queue = device.new_command_queue(1).expect("a queue");
        let command_buffer = queue.command_buffer().expect("a command buffer");
        let mut encoder =
            (command_buffer.compute_command_encoder(DispatchType::Serial)).expect("an encoder");
        for (kernel, out) in kernels.iter().zip(outs) {
            encoder.set_compute_pipeline_state(kernel).expect("a state");
            encoder.set_buffer(Some(out), 0, 0).expect("out");
            (encoder.dispatch_threadgroups([GROUPS, 1, 1], [THREADS, 1, 1]))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
        }
        encoder.end_encoding();
        command_buffer.commit().expect("a commit");
        finished(name, &command_buffer);
        assert_eq!(command_buffer.error(), None, "{name}");
        let after = counters.metrics();
        (after.barriers - before.barriers, after)
    }

    #[test]
    fn every_device_holds_private_memory_past_the_floor_in_its_arena_split_or_whole() {
        let threads = (GROUPS * THREADS) as usize;
        for n in [7176u32, 9216, 51_200] {
            let bitcode = match llvm_assemble_text(&module(n)) {
                Ok(bitcode) => bitcode,
                Err(AssembleError::Missing(why)) => {
                    eprintln!("SKIPPED: {why}");
                    return;
                }
                Err(error) => panic!("private_arena: {error:?}"),
            };
            let library = Library::from_air(&bitcode).expect("the library");
            let want: Vec<u32> = (0..threads as u32).map(|gid| reference(n, gid)).collect();
            let slot = u64::from(4 * n);
            let group = u64::from(THREADS) * slot;
            // The pieces of a dispatch under `budget`: as many whole threadgroups a piece as its
            // slots hold, at least one.
            let pieces_under = |budget: u64| {
                u64::from(GROUPS).div_ceil((budget / group).clamp(1, u64::from(GROUPS)))
            };
            // A budget of two threadgroups' slots at 28,704 bytes, four at 36 KiB, one at 200 KiB.
            let per_piece = match n {
                7176 => 2,
                9216 => 4,
                _ => 1,
            };
            let small = per_piece * group;
            each_device("private_arena", |name, _, device| {
                let arena = state(name, &device, &library, "arena");
                let plain = state(name, &device, &library, "plain");
                let block = (device.take_block(MemoryClass::Shared, 1 << 20)).expect("a block");
                let outs: Vec<Buffer> = (0..4)
                    .map(|k| {
                        (device.new_buffer(block.id, 4096 * k, 4096, Some(0))).expect("a buffer")
                    })
                    .collect();
                // Kernels without an arena: four dispatches into four buffers take the barriers one
                // does, no more.
                let (one, _) = barriers(name, &device, &[&plain], &outs[..1]);
                let (four, _) = barriers(name, &device, &[&plain, &plain, &plain, &plain], &outs);
                assert_eq!(
                    four, one,
                    "{name}: dispatches without an arena add no barrier"
                );
                for budget in [0, small] {
                    device.core().arena_budget.store(budget, Relaxed);
                    let pieces = pieces_under(device.core().arena_budget());
                    if budget != 0 {
                        assert_eq!(pieces, u64::from(GROUPS) / per_piece, "{name}");
                    }
                    let counters = device.counters();
                    let before = counters.metrics();
                    let kernels = [&plain, &arena, &arena, &plain];
                    let (barriers, after) = barriers(name, &device, &kernels, &outs);
                    let what =
                        format!("{name}: {n} words, budget {}", device.core().arena_budget());
                    for k in [1, 2] {
                        assert_eq!(words(&outs[k], threads), want, "{what}: arena dispatch {k}");
                    }
                    assert_eq!(
                        after.private_arena_dispatches - before.private_arena_dispatches,
                        2,
                        "{what}"
                    );
                    assert_eq!(
                        after.private_arena_pieces - before.private_arena_pieces,
                        2 * pieces,
                        "{what}"
                    );
                    // The first dispatch takes a block where none a command buffer keeps holds its
                    // slots; the second reaches the first's.
                    let taken = after.private_arena_blocks - before.private_arena_blocks;
                    assert!(taken <= 1, "{what}: {taken} blocks");
                    let ordered = 2 * pieces - taken;
                    assert_eq!(barriers, one + ordered, "{what}: barriers");
                    // Each counted in its source, the arena's pieces.
                    let pieced =
                        after.dispatch_arena_piece_barriers - before.dispatch_arena_piece_barriers;
                    assert_eq!(pieced, ordered, "{what}: barriers before arena pieces");
                    println!(
                        "{what}: {pieces} pieces a dispatch, {taken} block taken, {ordered} \
                         barriers before arena pieces"
                    );
                }
                device.core().arena_budget.store(0, Relaxed);
                drop((arena, plain, outs));
                (device.release_block(block)).unwrap_or_else(|error| panic!("{name}: {error}"));
            });
        }
    }
}
