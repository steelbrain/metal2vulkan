//! Stores into a thread's private arrays at an index only the run knows, under a device cfg on
//! every device of the cfg's kind: the owned synthetic kernel `sinks` fills two arrays of `N`
//! words, `a[i] = 31 t + i` and `b[i] = 17 t + 3 i`, at its loop's index, stores `t + 100000` at
//! byte offset `4 k` of `a` and `t + 200000` at `b[k]`, `k = in[t]`, then folds
//! `s = 3 (3 s + a[j]) + b[j]` over `j` and reads byte `(5 t) mod 4N` of `a`. The byte read at a
//! run-time offset places `a` in the private arena; `b` is a typed array. `kept` is the same kernel
//! without the two stores at `k`.
//!
//! At `N` 4 the arena is a `Private` array; at `N` 2100 the two arrays pass the translator's floor
//! and the arena is the runtime's device memory (`crate::arena`), which the counters show. Each
//! contained store always runs, at its index where it is in bounds; otherwise it rewrites the
//! arena's word 0, or the array's element 0, with the bits it holds (the translator's
//! `lower::access::local` and `memory::alloca`).
//!
//! - In bounds, `k = (5 t + 1) mod N`, every thread's two words are Metal's: the same steps in MSL
//!   (`metal2vulkan-conformance/probes/private_sinks.m`) answered `reference`'s on an Apple M3 Max
//!   for both sizes.
//! - Out of bounds, `k` past both arrays (2N, 2N + 1, 3N + 7 and 100N by turns), Metal's answer is
//!   not a defined one and none is claimed: on our side the stores change no word a thread
//!   reads, so `sinks` answers what `kept` answers, `reference` without the stores. (An index past
//!   `a` but within the arena reaches `b`, as Metal's own memory may: not asked.)
//! - Element 0 keeps a NaN's bits: the owned synthetic kernel `nan` stores a NaN with payload bits
//!   from `in` into element 0 of the punned `a: [4 x float]` (in the private arena) and of the
//!   typed `b: [4 x float]`, `1.0` into the rest, and `2.5` at byte offset `4 k` of `a` and at
//!   `b[k]`, `k` past both, then writes `a[0]`'s and `b[0]`'s bits and byte `(5 t) mod 16` of `a`.
//!   `nan_kept` is the same kernel without the two stores at `k`. Both answer the NaN's bits as
//!   `in` held them: the typed array's store selects between a float and its old value, and a
//!   device whose select or float move changed a NaN's payload would show it here.
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

    /// Threadgroups of a dispatch, and threads of each.
    const GROUPS: u32 = 2;
    const THREADS: u32 = 32;

    /// One kernel of [`module`]: `@name`, with the two stores at `k` where `stores`.
    fn kernel(name: &str, n: u32, stores: bool) -> String {
        let poke = if stores {
            "  %pk = getelementptr inbounds i32, ptr addrspace(1) %in, i32 %t
  %k = load i32, ptr addrspace(1) %pk, align 4
  %w = shl i32 %k, 2
  %pa2 = getelementptr inbounds i8, ptr %a, i32 %w
  %v1 = add i32 %t, 100000
  store i32 %v1, ptr %pa2, align 4
  %pb2 = getelementptr inbounds [N x i32], ptr %b, i32 0, i32 %k
  %v2 = add i32 %t, 200000
  store i32 %v2, ptr %pb2, align 4
"
        } else {
            ""
        };
        let threads = THREADS;
        let body = format!(
            r#"define void @{name}(ptr addrspace(1) %out, ptr addrspace(1) %in, i32 %tg, i32 %tid) {{
entry:
  %a = alloca [N x i32], align 4
  %b = alloca [N x i32], align 4
  %first = mul i32 %tg, {threads}
  %t = add i32 %first, %tid
  %ta = mul i32 %t, 31
  %tb = mul i32 %t, 17
  br label %fill
fill:
  %i = phi i32 [ 0, %entry ], [ %i1, %fill ]
  %va = add i32 %ta, %i
  %pa = getelementptr inbounds [N x i32], ptr %a, i32 0, i32 %i
  store i32 %va, ptr %pa, align 4
  %i3 = mul i32 %i, 3
  %vb = add i32 %tb, %i3
  %pb = getelementptr inbounds [N x i32], ptr %b, i32 0, i32 %i
  store i32 %vb, ptr %pb, align 4
  %i1 = add i32 %i, 1
  %more = icmp ult i32 %i1, N
  br i1 %more, label %fill, label %poke
poke:
{poke}  br label %fold
fold:
  %j = phi i32 [ 0, %poke ], [ %j1, %fold ]
  %s = phi i32 [ 0, %poke ], [ %s2, %fold ]
  %qa = getelementptr inbounds [N x i32], ptr %a, i32 0, i32 %j
  %xa = load i32, ptr %qa, align 4
  %qb = getelementptr inbounds [N x i32], ptr %b, i32 0, i32 %j
  %xb = load i32, ptr %qb, align 4
  %s3 = mul i32 %s, 3
  %s1 = add i32 %s3, %xa
  %s4 = mul i32 %s1, 3
  %s2 = add i32 %s4, %xb
  %j1 = add i32 %j, 1
  %again = icmp ult i32 %j1, N
  br i1 %again, label %fold, label %done
done:
  %t5 = mul i32 %t, 5
  %at = urem i32 %t5, {bytes}
  %py = getelementptr inbounds i8, ptr %a, i32 %at
  %y = load i8, ptr %py, align 1
  %yw = zext i8 %y to i32
  %o0 = shl i32 %t, 1
  %q0 = getelementptr inbounds i32, ptr addrspace(1) %out, i32 %o0
  store i32 %s2, ptr addrspace(1) %q0, align 4
  %o1 = or i32 %o0, 1
  %q1 = getelementptr inbounds i32, ptr addrspace(1) %out, i32 %o1
  store i32 %yw, ptr addrspace(1) %q1, align 4
  ret void
}}
"#,
            bytes = 4 * n
        );
        body.replace("N x i32", &format!("{n} x i32"))
            .replace("i32 %i1, N", &format!("i32 %i1, {n}"))
            .replace("i32 %j1, N", &format!("i32 %j1, {n}"))
    }

    /// The kernels `sinks` and `kept` of `n` words an array.
    fn module(n: u32) -> String {
        format!(
            r#"; Owned synthetic kernels: two private arrays of {n} words, stored at an index the run knows.
source_filename = "private_sinks.metal"

{}
{}
!air.kernel = !{{!0, !1}}
!0 = !{{ptr @sinks, !2, !3}}
!1 = !{{ptr @kept, !2, !3}}
!2 = !{{}}
!3 = !{{!10, !11, !12, !13}}
!10 = !{{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}}
!11 = !{{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"in"}}
!12 = !{{i32 2, !"air.threadgroup_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"tg"}}
!13 = !{{i32 3, !"air.thread_position_in_threadgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"tid"}}
"#,
            kernel("sinks", n, true),
            kernel("kept", n, false)
        )
    }

    /// Thread `t`'s two words with `n` words an array, its stores at `k` where `k` is inside both
    /// (none past them).
    fn reference(n: u32, t: u32, k: u32) -> [u32; 2] {
        let mut a: Vec<u32> = (0..n).map(|i| t.wrapping_mul(31).wrapping_add(i)).collect();
        let mut b: Vec<u32> = (0..n)
            .map(|i| t.wrapping_mul(17).wrapping_add(3 * i))
            .collect();
        if k < n {
            a[k as usize] = t + 100_000;
            b[k as usize] = t + 200_000;
        }
        let s = (0..n as usize).fold(0u32, |s, j| {
            let s = s.wrapping_mul(3).wrapping_add(a[j]);
            s.wrapping_mul(3).wrapping_add(b[j])
        });
        let at = (5 * t) % (4 * n);
        [s, (a[(at / 4) as usize] >> (8 * (at % 4))) & 0xff]
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

    /// Write `values` to `buffer`'s first words.
    fn fill(buffer: &Buffer, values: &[u32]) {
        let contents = buffer.contents().expect("a Shared buffer");
        // SAFETY: no work reaches the buffer yet; it holds at least `values.len()` words.
        let at = unsafe {
            std::slice::from_raw_parts_mut(contents.as_ptr().cast::<u32>(), values.len())
        };
        at.copy_from_slice(values);
    }

    /// The first `count` words `kernel` writes, one dispatch of [`GROUPS`] threadgroups of
    /// [`THREADS`] over the input words `ins`.
    fn run(
        name: &str,
        device: &Device,
        kernel: &ComputePipelineState,
        (input, out): (&Buffer, &Buffer),
        ins: &[u32],
        count: usize,
    ) -> Vec<u32> {
        fill(input, ins);
        fill(out, &vec![0xeeee_eeee; count]);
        let queue = device.new_command_queue(1).expect("a queue");
        let command_buffer = queue.command_buffer().expect("a command buffer");
        let mut encoder =
            (command_buffer.compute_command_encoder(DispatchType::Serial)).expect("an encoder");
        encoder.set_compute_pipeline_state(kernel).expect("a state");
        encoder.set_buffer(Some(out), 0, 0).expect("out");
        encoder.set_buffer(Some(input), 0, 1).expect("in");
        (encoder.dispatch_threadgroups([GROUPS, 1, 1], [THREADS, 1, 1]))
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        encoder.end_encoding();
        command_buffer.commit().expect("a commit");
        finished(name, &command_buffer);
        assert_eq!(command_buffer.error(), None, "{name}");
        words(out, count)
    }

    #[test]
    fn every_device_keeps_a_store_past_a_private_array_off_its_words() {
        let threads = GROUPS * THREADS;
        for n in [4u32, 2100] {
            let bitcode = match llvm_assemble_text(&module(n)) {
                Ok(bitcode) => bitcode,
                Err(AssembleError::Missing(why)) => {
                    eprintln!("SKIPPED: {why}");
                    return;
                }
                Err(error) => panic!("private_sinks: {error:?}"),
            };
            let library = Library::from_air(&bitcode).expect("the library");
            let inside: Vec<u32> = (0..threads).map(|t| (5 * t + 1) % n).collect();
            let past: Vec<u32> = (0..threads)
                .map(|t| [2 * n, 2 * n + 1, 3 * n + 7, 100 * n][(t % 4) as usize])
                .collect();
            let metal: Vec<u32> = (0..threads)
                .flat_map(|t| reference(n, t, inside[t as usize]))
                .collect();
            let untouched: Vec<u32> = (0..threads).flat_map(|t| reference(n, t, n)).collect();
            each_device("private_sinks", |name, _, device| {
                let sinks = state(name, &device, &library, "sinks");
                let kept = state(name, &device, &library, "kept");
                let slot = sinks.inner.compiled.whole.interface.private_arena;
                assert_eq!(
                    slot.is_some(),
                    n > 4,
                    "{name}: {n} words, arena slot {slot:?}"
                );
                let block = (device.take_block(MemoryClass::Shared, 1 << 16)).expect("a block");
                let input = (device.new_buffer(block.id, 0, 4096, Some(0))).expect("a buffer");
                let out = (device.new_buffer(block.id, 4096, 4096, Some(0))).expect("a buffer");
                let counters = device.counters();
                let before = counters.metrics().private_arena_dispatches;
                let what = format!("{name}: {n} words");
                let count = 2 * threads as usize;
                let read = run(name, &device, &sinks, (&input, &out), &inside, count);
                assert_eq!(read, metal, "{what}: stores inside the arrays");
                let read = run(name, &device, &sinks, (&input, &out), &past, count);
                let without = run(name, &device, &kept, (&input, &out), &past, count);
                assert_eq!(read, without, "{what}: stores past the arrays against none");
                assert_eq!(without, untouched, "{what}: no store");
                let arena = counters.metrics().private_arena_dispatches - before;
                assert_eq!(arena, if n > 4 { 3 } else { 0 }, "{what}: arena dispatches");
                println!(
                    "{what}: stores inside the arrays answer Metal's, past them change no word \
                     ({arena} dispatches in the runtime's arena, slot {slot:?})"
                );
                drop((sinks, kept, input, out));
                (device.release_block(block)).unwrap_or_else(|error| panic!("{name}: {error}"));
            });
        }
    }

    /// One kernel of [`nan_module`]: `@name`, with the two stores at `k` where `stores`.
    fn nan_kernel(name: &str, stores: bool) -> String {
        let poke = if stores {
            "  %i1 = or i32 %i0, 1
  %pk = getelementptr inbounds i32, ptr addrspace(1) %in, i32 %i1
  %k = load i32, ptr addrspace(1) %pk, align 4
  %w = shl i32 %k, 2
  %pa2 = getelementptr inbounds i8, ptr %a, i32 %w
  store float 2.500000e+00, ptr %pa2, align 4
  %pb2 = getelementptr inbounds [4 x float], ptr %b, i32 0, i32 %k
  store float 2.500000e+00, ptr %pb2, align 4
"
        } else {
            ""
        };
        let mut fill = String::new();
        for x in ["a", "b"] {
            fill += &format!(
                "  %p{x}0 = getelementptr inbounds [4 x float], ptr %{x}, i32 0, i32 0\n  store float %nan, ptr %p{x}0, align 4\n"
            );
            for i in 1..4 {
                fill += &format!(
                    "  %p{x}{i} = getelementptr inbounds [4 x float], ptr %{x}, i32 0, i32 {i}\n  store float 1.000000e+00, ptr %p{x}{i}, align 4\n"
                );
            }
        }
        format!(
            r#"define void @{name}(ptr addrspace(1) %out, ptr addrspace(1) %in, i32 %tg, i32 %tid) {{
entry:
  %a = alloca [4 x float], align 4
  %b = alloca [4 x float], align 4
  %first = mul i32 %tg, {THREADS}
  %t = add i32 %first, %tid
  %i0 = shl i32 %t, 1
  %pn = getelementptr inbounds float, ptr addrspace(1) %in, i32 %i0
  %nan = load float, ptr addrspace(1) %pn, align 4
{fill}{poke}  %ra = load float, ptr %pa0, align 4
  %rb = load float, ptr %pb0, align 4
  %t5 = mul i32 %t, 5
  %at = urem i32 %t5, 16
  %py = getelementptr inbounds i8, ptr %a, i32 %at
  %y = load i8, ptr %py, align 1
  %yw = zext i8 %y to i32
  %o0 = mul i32 %t, 3
  %q0 = getelementptr inbounds float, ptr addrspace(1) %out, i32 %o0
  store float %ra, ptr addrspace(1) %q0, align 4
  %o1 = add i32 %o0, 1
  %q1 = getelementptr inbounds float, ptr addrspace(1) %out, i32 %o1
  store float %rb, ptr addrspace(1) %q1, align 4
  %o2 = add i32 %o0, 2
  %q2 = getelementptr inbounds i32, ptr addrspace(1) %out, i32 %o2
  store i32 %yw, ptr addrspace(1) %q2, align 4
  ret void
}}
"#
        )
    }

    /// The kernels `nan` and `nan_kept`.
    fn nan_module() -> String {
        format!(
            r#"; Owned synthetic kernels: a NaN in element 0 of two private arrays, stored past them.
source_filename = "private_sinks_nan.metal"

{}
{}
!air.kernel = !{{!0, !1}}
!0 = !{{ptr @nan, !2, !3}}
!1 = !{{ptr @nan_kept, !2, !3}}
!2 = !{{}}
!3 = !{{!10, !11, !12, !13}}
!10 = !{{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}}
!11 = !{{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"in"}}
!12 = !{{i32 2, !"air.threadgroup_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"tg"}}
!13 = !{{i32 3, !"air.thread_position_in_threadgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"tid"}}
"#,
            nan_kernel("nan", true),
            nan_kernel("nan_kept", false)
        )
    }

    #[test]
    fn a_store_past_a_private_array_keeps_the_bits_of_a_nan_at_element_0() {
        let threads = GROUPS * THREADS;
        let bitcode = match llvm_assemble_text(&nan_module()) {
            Ok(bitcode) => bitcode,
            Err(AssembleError::Missing(why)) => {
                eprintln!("SKIPPED: {why}");
                return;
            }
            Err(error) => panic!("private_sinks nan: {error:?}"),
        };
        let library = Library::from_air(&bitcode).expect("the library");
        // A quiet NaN with payload bits, a signalling one, and a negative quiet one.
        let nans = [0x7fc0_1234u32, 0x7f80_1234, 0xffc5_a5a5];
        let ins: Vec<u32> = (0..threads)
            .flat_map(|t| [nans[(t % 3) as usize], [8, 9, 19, 400][(t % 4) as usize]])
            .collect();
        let want: Vec<u32> = (0..threads)
            .flat_map(|t| {
                let x = nans[(t % 3) as usize];
                let at = (5 * t) % 16;
                let word = if at < 4 { x } else { 0x3f80_0000 };
                [x, x, (word >> (8 * (at % 4))) & 0xff]
            })
            .collect();
        each_device("private_sinks nan", |name, _, device| {
            let nan = state(name, &device, &library, "nan");
            let kept = state(name, &device, &library, "nan_kept");
            let block = (device.take_block(MemoryClass::Shared, 1 << 16)).expect("a block");
            let input = (device.new_buffer(block.id, 0, 4096, Some(0))).expect("a buffer");
            let out = (device.new_buffer(block.id, 4096, 4096, Some(0))).expect("a buffer");
            let count = 3 * threads as usize;
            let without = run(name, &device, &kept, (&input, &out), &ins, count);
            let hex = |w: &[u32]| w.iter().map(|w| format!("{w:08x}")).collect::<Vec<_>>();
            assert_eq!(
                hex(&without),
                hex(&want),
                "{name}: a NaN with no store past it"
            );
            let read = run(name, &device, &nan, (&input, &out), &ins, count);
            assert_eq!(hex(&read), hex(&want), "{name}: a NaN with a store past it");
            println!("{name}: element 0 keeps each NaN's bits with a store past its array");
            drop((nan, kept, input, out));
            (device.release_block(block)).unwrap_or_else(|error| panic!("{name}: {error}"));
        });
    }
}
