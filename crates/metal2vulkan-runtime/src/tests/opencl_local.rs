//! Kernels of OpenCL mode (`ComputePipelineDescriptor::open_cl_mode_enabled`) whose threads share
//! `__local` memory through barriers, on every device of the cfg's kind. Each is written in the
//! shape Apple's OpenCL compiler writes OpenCL C: an entry `__entry_<name>` that loads each
//! argument passed by value from its constant pointer and calls the kernel, an internal function,
//! which calls the work-item functions and `barrier` through internal wrappers of `air.get_*` and
//! `air.wg.barrier`; a `__local` argument is a buffer of the threadgroup address space at its own
//! threadgroup index. Each is bound as a pipeline of OpenCL mode is: the dispatch's global offset
//! and dimension count (`__global_offset_and_num_dims`) by `setBytes` at index 0, the kernel's
//! buffers and values 3 past their indices, and each `__local` argument's length by
//! `setThreadgroupMemoryLength` at its own index.
//!
//! - **lsum**: each threadgroup of 64 threads sums its `in[i] = i` through `__local` memory and a
//!   barrier: 2016, 6112, 10208 and 14304, the M3 Max's answers (macOS 27.0.1).
//! - **tiled_matmul**: our own `c = a b` through two `__local` tiles of 32 by 32 floats, 4096
//!   bytes each at threadgroup indices 0 and 1, in threadgroups of 32 by 4 threads, each thread
//!   accumulating 8 rows of its column, with `barrier(CLK_LOCAL_MEM_FENCE)` after each tile's
//!   load and another before the next load; dispatched over 160 by 3200 threads, whole
//!   threadgroups, 8 times in one command buffer, each into its own `c`. Every element is
//!   compared with the CPU's product, exactly (`value` says why no tolerance is needed), and
//!   each repeat prints how many elements differ and by how much.
//!
//! Both pass OpenCL's barrier flags on to `air.wg.barrier` unchanged, as Apple's compiler does:
//! `CLK_LOCAL_MEM_FENCE` is 1 in OpenCL's encoding, which Metal's `mem_flags` read as
//! `mem_device`. lsum's `barrier` is the three-operand form the deleted test wrote
//! ([`BARRIER_THREE_OPERANDS`]), the multiply's the two-operand form Apple's OpenCL compiler
//! writes ([`BARRIER_TWO_OPERANDS`]).
//!
//! The modules are LLVM text written here (`llvm-as`); without it the tests skip visibly.

#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod on_device {
    use crate::tests::command::on_device::{each_device, finished};
    use crate::{
        Block, Buffer, ComputePipelineDescriptor, Device, DispatchType, FunctionDescriptor,
        Library, MemoryClass, StageSource,
    };
    use metal2vulkan_devtools::{llvm_assemble_text, AssembleError};

    /// The target, the wrappers of the work-item functions as Apple's OpenCL compiler writes
    /// them, and the module's versions; each kernel adds its `barrier`
    /// ([`BARRIER_THREE_OPERANDS`], [`BARRIER_TWO_OPERANDS`]).
    const HEAD: &str = "target datalayout = \"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32\"
target triple = \"air64_v29-apple-macosx27.0.0\"
define internal fastcc i64 @_Z13get_global_idj(i32 %0) #2 {
  %2 = tail call i32 @air.get_global_id.i32(i32 %0) #3
  %3 = zext i32 %2 to i64
  ret i64 %3
}
define internal fastcc i64 @_Z12get_local_idj(i32 %0) #2 {
  %2 = tail call i32 @air.get_local_id.i32(i32 %0) #3
  %3 = zext i32 %2 to i64
  ret i64 %3
}
define internal fastcc i64 @_Z12get_group_idj(i32 %0) #2 {
  %2 = tail call i32 @air.get_group_id.i32(i32 %0) #3
  %3 = zext i32 %2 to i64
  ret i64 %3
}
define internal fastcc i64 @_Z14get_local_sizej(i32 %0) #2 {
  %2 = tail call i32 @air.get_local_size.i32(i32 %0) #3
  %3 = zext i32 %2 to i64
  ret i64 %3
}
declare i32 @air.get_global_id.i32(i32) #3
declare i32 @air.get_local_id.i32(i32) #3
declare i32 @air.get_group_id.i32(i32) #3
declare i32 @air.get_local_size.i32(i32) #3
attributes #0 = { nounwind }
attributes #1 = { convergent nounwind }
attributes #2 = { nounwind optsize }
attributes #3 = { nounwind readnone willreturn }
attributes #4 = { convergent nounwind }
attributes #5 = { convergent nounwind willreturn }
!air.version = !{!90}
!air.language_version = !{!91}
!90 = !{i32 2, i32 9, i32 0}
!91 = !{!\"Metal\", i32 1, i32 1, i32 0}
";

    /// OpenCL C's `barrier(flags)` as the deleted runtime's OpenCL-mode tests wrote it: an
    /// internal function passing OpenCL's flags on to `air.wg.barrier` of three operands (flags,
    /// `seq_cst`, threadgroup) unchanged, so `CLK_LOCAL_MEM_FENCE` (1) reaches it as
    /// `mem_device`.
    const BARRIER_THREE_OPERANDS: &str = "define internal fastcc void @_Z7barrierj(i32 %0) #4 {
  call void @air.wg.barrier(i32 %0, i32 5, i32 1) #5
  ret void
}
declare void @air.wg.barrier(i32, i32, i32) #5
";

    /// OpenCL C's `barrier(flags)` as Apple's OpenCL compiler writes it (`llvm-dis` of a kernel
    /// it compiled): an internal function passing OpenCL's flags on to `air.wg.barrier` of two
    /// operands (flags, threadgroup) unchanged, so `CLK_LOCAL_MEM_FENCE` (1) reaches it as
    /// `mem_device`.
    const BARRIER_TWO_OPERANDS: &str = "define internal fastcc void @_Z7barrierj(i32 %0) #4 {
  tail call void @air.wg.barrier(i32 %0, i32 1) #5
  ret void
}
declare void @air.wg.barrier(i32, i32) #5
";

    /// The node of argument `n`, a buffer at `index` of address space `space` (1 global, 3
    /// threadgroup) holding `ty`, `access` one of `read`, `read_write`.
    fn buffer(n: u32, index: u32, access: &str, space: u32, ty: &str, name: &str) -> String {
        format!(
            "!{{i32 {n}, !\"air.buffer\", !\"air.location_index\", i32 {index}, i32 1, !\"air.{access}\", !\"air.address_space\", i32 {space}, !\"air.arg_type_size\", i32 4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"{ty}\", !\"air.arg_name\", !\"{name}\"}}"
        )
    }

    /// The node of argument `n`, a `uint` passed by value at `index`.
    fn uint_value(n: u32, index: u32, name: &str) -> String {
        format!(
            "!{{i32 {n}, !\"air.constant\", !\"air.location_index\", i32 {index}, i32 1, !\"air.arg_type_size\", i32 4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"{name}\"}}"
        )
    }

    /// The kernel list of `entry` with argument nodes `args`.
    fn kernel_list(entry: &str, args: &[String]) -> String {
        let refs: Vec<String> = (0..args.len()).map(|i| format!("!{}", 3 + i)).collect();
        let mut out = format!(
            "!air.kernel = !{{!0}}\n!0 = !{{ptr @{entry}, !1, !2}}\n!1 = !{{}}\n!2 = !{{{}}}\n",
            refs.join(", ")
        );
        for (i, a) in args.iter().enumerate() {
            out.push_str(&format!("!{} = {a}\n", 3 + i));
        }
        out
    }

    /// `__kernel void lsum(__global const float *in, __global float *out, __local float *tmp)`:
    /// `tmp[l] = in[i]`, `barrier(CLK_LOCAL_MEM_FENCE)`, and thread 0 of each group stores the
    /// sum of its group's `tmp` to `out[get_group_id(0)]`; its `barrier` is
    /// [`BARRIER_THREE_OPERANDS`], as the deleted test wrote it.
    fn lsum() -> String {
        let body = "define void @__entry_lsum(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(3) %2) #0 {
entry:
  tail call fastcc void @lsum(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(3) %2)
  ret void
}
define internal fastcc void @lsum(ptr addrspace(1) %in, ptr addrspace(1) %out, ptr addrspace(3) %tmp) #1 {
entry:
  %l = tail call fastcc i64 @_Z12get_local_idj(i32 0)
  %g = tail call fastcc i64 @_Z13get_global_idj(i32 0)
  %pi = getelementptr inbounds float, ptr addrspace(1) %in, i64 %g
  %v = load float, ptr addrspace(1) %pi, align 4
  %pt = getelementptr inbounds float, ptr addrspace(3) %tmp, i64 %l
  store float %v, ptr addrspace(3) %pt, align 4
  tail call fastcc void @_Z7barrierj(i32 1)
  %first = icmp eq i64 %l, 0
  br i1 %first, label %loop, label %done
loop:
  %s = phi float [ %s1, %body ], [ 0.000000e+00, %entry ]
  %j = phi i64 [ %j1, %body ], [ 0, %entry ]
  %n = tail call fastcc i64 @_Z14get_local_sizej(i32 0)
  %more = icmp ult i64 %j, %n
  br i1 %more, label %body, label %exit
body:
  %pj = getelementptr inbounds float, ptr addrspace(3) %tmp, i64 %j
  %vj = load float, ptr addrspace(3) %pj, align 4
  %s1 = fadd float %s, %vj
  %j1 = add nuw i64 %j, 1
  br label %loop
exit:
  %grp = tail call fastcc i64 @_Z12get_group_idj(i32 0)
  %po = getelementptr inbounds float, ptr addrspace(1) %out, i64 %grp
  store float %s, ptr addrspace(1) %po, align 4
  br label %done
done:
  ret void
}
";
        let args = [
            buffer(0, 0, "read", 1, "float", "in"),
            buffer(1, 1, "read_write", 1, "float", "out"),
            buffer(2, 0, "read_write", 3, "float", "tmp"),
        ];
        format!(
            "{BARRIER_THREE_OPERANDS}{body}{}",
            kernel_list("__entry_lsum", &args)
        )
    }

    /// The rows of `c` each thread computes, 4 apart, so a threadgroup of 32 by 4 threads
    /// computes a block of 32 by 32.
    const WORK: u32 = 8;

    /// Where the multiply's two tiles of 32 by 32 floats live.
    #[derive(Clone, Copy)]
    enum Tiles {
        /// Two `__local` arguments, `at` and `bt`, at threadgroup indices 0 and 1.
        Local,
        /// The threadgroup's own 2048 floats of a `__global float *scratch` argument at buffer
        /// index 5, from float `2048 (get_group_id(1) (n / 32) + get_group_id(0))`: `at` there
        /// and `bt` 1024 floats on.
        Global,
    }

    /// Our own tiled matrix multiply, `c = a b` with `a` of `m` rows by `k`, `b` of `k` rows by
    /// `n`, row-major, `k` a multiple of 32, in threadgroups of 32 by 4 threads over `n` by
    /// `m / 8` threads; written as the OpenCL compiler writes the OpenCL C below, its row loop
    /// unrolled and its loop invariants hoisted:
    ///
    /// ```c
    /// __kernel void tiled_matmul(__global const float *a, __global const float *b,
    ///                            __global float *c, uint n, uint k,
    ///                            __local float *at, __local float *bt) {
    ///     size_t x = get_local_id(0), y = get_local_id(1), col = get_global_id(0);
    ///     size_t row0 = get_group_id(1) * get_local_size(1) * 8;
    ///     float acc[8] = {0};
    ///     for (uint t = 0; t < k; t += 32) {
    ///         for (int w = 0; w < 8; w++) {
    ///             size_t r = y + 4 * w;
    ///             at[r * 32 + x] = a[(row0 + r) * k + t + x];
    ///             bt[r * 32 + x] = b[(t + r) * n + col];
    ///         }
    ///         barrier(CLK_LOCAL_MEM_FENCE);
    ///         for (int j = 0; j < 32; j++)
    ///             for (int w = 0; w < 8; w++)
    ///                 acc[w] += at[(y + 4 * w) * 32 + j] * bt[j * 32 + x];
    ///         barrier(CLK_LOCAL_MEM_FENCE);
    ///     }
    ///     for (int w = 0; w < 8; w++)
    ///         c[(row0 + y + 4 * w) * n + col] = acc[w];
    /// }
    /// ```
    ///
    /// Its `barrier` is [`BARRIER_TWO_OPERANDS`], called as Apple's compiler calls it, with
    /// `flags` in place of `CLK_LOCAL_MEM_FENCE`; `tiles` says where `at` and `bt` live.
    fn tiled_matmul(tiles: Tiles, flags: u32) -> String {
        let space = match tiles {
            Tiles::Local => 3,
            Tiles::Global => 1,
        };
        let (mut hoisted, mut tile_phis, mut loads) = (String::new(), String::new(), String::new());
        let (mut inner_phis, mut inner, mut exit_phis, mut stores) =
            (String::new(), String::new(), String::new(), String::new());
        for w in 0..WORK {
            // Row `r = y + 4w` of the block: its first element in a tile, and its row of `a` and
            // of `c`.
            hoisted.push_str(&format!(
                "  %r{w} = add i64 %y, {}
  %rt{w} = shl i64 %r{w}, 5
  %ra{w} = add i64 %row0, %r{w}
  %ak{w} = mul i64 %ra{w}, %k64
",
                4 * w
            ));
            tile_phis.push_str(&format!(
                "  %s{w} = phi float [ 0.000000e+00, %entry ], [ %e{w}, %next ]\n"
            ));
            loads.push_str(&format!(
                "  %ai{w}0 = add i64 %ak{w}, %t
  %ai{w} = add i64 %ai{w}0, %x
  %pa{w} = getelementptr inbounds float, ptr addrspace(1) %a, i64 %ai{w}
  %va{w} = load float, ptr addrspace(1) %pa{w}, align 4
  %ti{w} = add i64 %rt{w}, %x
  %pat{w} = getelementptr inbounds float, ptr addrspace({space}) %at, i64 %ti{w}
  store float %va{w}, ptr addrspace({space}) %pat{w}, align 4
  %bk{w} = add i64 %t, %r{w}
  %bi{w}0 = mul i64 %bk{w}, %n64
  %bi{w} = add i64 %bi{w}0, %col
  %pb{w} = getelementptr inbounds float, ptr addrspace(1) %b, i64 %bi{w}
  %vb{w} = load float, ptr addrspace(1) %pb{w}, align 4
  %pbt{w} = getelementptr inbounds float, ptr addrspace({space}) %bt, i64 %ti{w}
  store float %vb{w}, ptr addrspace({space}) %pbt{w}, align 4
"
            ));
            inner_phis.push_str(&format!(
                "  %q{w} = phi float [ %s{w}, %tile ], [ %e{w}, %inner ]\n"
            ));
            inner.push_str(&format!(
                "  %aj{w} = add i64 %rt{w}, %j
  %paj{w} = getelementptr inbounds float, ptr addrspace({space}) %at, i64 %aj{w}
  %av{w} = load float, ptr addrspace({space}) %paj{w}, align 4
  %m{w} = fmul float %av{w}, %bv
  %e{w} = fadd float %q{w}, %m{w}
"
            ));
            exit_phis.push_str(&format!(
                "  %f{w} = phi float [ 0.000000e+00, %entry ], [ %e{w}, %next ]\n"
            ));
            stores.push_str(&format!(
                "  %ci{w}0 = mul i64 %ra{w}, %n64
  %ci{w} = add i64 %ci{w}0, %col
  %pc{w} = getelementptr inbounds float, ptr addrspace(1) %c, i64 %ci{w}
  store float %f{w}, ptr addrspace(1) %pc{w}, align 4
"
            ));
        }
        // The entry's tile arguments, the kernel's, and the global form's tile addresses: the
        // group's own 2048 floats of `scratch`.
        let (entry_tiles, n, k, passed, params, addresses) = match tiles {
            Tiles::Local => (
                "ptr addrspace(3) %5, ptr addrspace(3) %6",
                "%7",
                "%8",
                "ptr addrspace(3) %5, ptr addrspace(3) %6",
                "ptr addrspace(3) %at, ptr addrspace(3) %bt",
                "",
            ),
            Tiles::Global => (
                "ptr addrspace(1) %5",
                "%6",
                "%7",
                "ptr addrspace(1) %5",
                "ptr addrspace(1) %scratch",
                "  %gx = tail call fastcc i64 @_Z12get_group_idj(i32 0)
  %across = lshr i64 %n64, 5
  %gl0 = mul i64 %gy, %across
  %gl = add i64 %gl0, %gx
  %tb = shl i64 %gl, 11
  %at = getelementptr inbounds float, ptr addrspace(1) %scratch, i64 %tb
  %bt = getelementptr inbounds float, ptr addrspace(1) %at, i64 1024
",
            ),
        };
        let body = format!(
            "define void @__entry_tiled_matmul(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(2) %3, ptr addrspace(2) %4, {entry_tiles}) #0 {{
entry:
  {n} = load i32, ptr addrspace(2) %3, align 4
  {k} = load i32, ptr addrspace(2) %4, align 4
  tail call fastcc void @tiled_matmul(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, i32 {n}, i32 {k}, {passed})
  ret void
}}
define internal fastcc void @tiled_matmul(ptr addrspace(1) %a, ptr addrspace(1) %b, ptr addrspace(1) %c, i32 %n, i32 %k, {params}) #1 {{
entry:
  %x = tail call fastcc i64 @_Z12get_local_idj(i32 0)
  %y = tail call fastcc i64 @_Z12get_local_idj(i32 1)
  %col = tail call fastcc i64 @_Z13get_global_idj(i32 0)
  %gy = tail call fastcc i64 @_Z12get_group_idj(i32 1)
  %ly = tail call fastcc i64 @_Z14get_local_sizej(i32 1)
  %rows = shl i64 %ly, 3
  %row0 = mul i64 %gy, %rows
  %n64 = zext i32 %n to i64
  %k64 = zext i32 %k to i64
{addresses}{hoisted}  %none = icmp eq i32 %k, 0
  br i1 %none, label %exit, label %tile
tile:
  %t = phi i64 [ 0, %entry ], [ %t1, %next ]
{tile_phis}{loads}  tail call fastcc void @_Z7barrierj(i32 noundef {flags})
  br label %inner
inner:
  %j = phi i64 [ 0, %tile ], [ %j1, %inner ]
{inner_phis}  %bj0 = shl i64 %j, 5
  %bj = add i64 %bj0, %x
  %pbj = getelementptr inbounds float, ptr addrspace({space}) %bt, i64 %bj
  %bv = load float, ptr addrspace({space}) %pbj, align 4
{inner}  %j1 = add nuw nsw i64 %j, 1
  %jend = icmp eq i64 %j1, 32
  br i1 %jend, label %next, label %inner
next:
  tail call fastcc void @_Z7barrierj(i32 noundef {flags})
  %t1 = add i64 %t, 32
  %tmore = icmp ult i64 %t1, %k64
  br i1 %tmore, label %tile, label %exit
exit:
{exit_phis}{stores}  ret void
}}
"
        );
        let mut args = vec![
            buffer(0, 0, "read", 1, "float", "a"),
            buffer(1, 1, "read", 1, "float", "b"),
            buffer(2, 2, "read_write", 1, "float", "c"),
            uint_value(3, 3, "n"),
            uint_value(4, 4, "k"),
        ];
        match tiles {
            Tiles::Local => {
                args.push(buffer(5, 0, "read_write", 3, "float", "at"));
                args.push(buffer(6, 1, "read_write", 3, "float", "bt"));
            }
            Tiles::Global => args.push(buffer(5, 5, "read_write", 1, "float", "scratch")),
        }
        format!(
            "{BARRIER_TWO_OPERANDS}{body}{}",
            kernel_list("__entry_tiled_matmul", &args)
        )
    }

    /// The descriptor of OpenCL mode of `entry` of the module `text` (after [`HEAD`]); `None`
    /// without `llvm-as`.
    fn descriptor(text: &str, entry: &str) -> Option<ComputePipelineDescriptor> {
        let air = match llvm_assemble_text(&format!("{HEAD}{text}")) {
            Ok(air) => air,
            Err(AssembleError::Missing(why)) => {
                eprintln!("SKIPPED: {why}");
                return None;
            }
            Err(error) => panic!("{entry}: {error:?}"),
        };
        let library = Library::from_air(&air).expect("the library");
        let function = library.function(entry).expect("the kernel");
        let mut descriptor = ComputePipelineDescriptor::new(StageSource::Function(
            FunctionDescriptor::new(function),
        ));
        descriptor.open_cl_mode_enabled = true;
        Some(descriptor)
    }

    /// `__global_offset_and_num_dims` of a dispatch of `dims` dimensions at no offset.
    fn offset_and_dims(dims: u32) -> Vec<u8> {
        [0, 0, 0, dims]
            .iter()
            .flat_map(|w| w.to_le_bytes())
            .collect()
    }

    /// The bytes of `floats`.
    fn bytes_of(floats: &[f32]) -> Vec<u8> {
        floats.iter().flat_map(|f| f.to_le_bytes()).collect()
    }

    /// A Shared buffer of `bytes`, and its block.
    fn shared(device: &Device, bytes: &[u8]) -> (Block, Buffer) {
        let len = bytes.len() as u64;
        let block = (device.take_block(MemoryClass::Shared, len)).expect("a Shared block");
        let buffer = (device.new_buffer(block.id, 0, len, None)).expect("a buffer");
        let contents = buffer.contents().expect("a Shared buffer");
        // SAFETY: the buffer is alive, `len` bytes long, and no command uses it yet.
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), contents.as_ptr(), bytes.len()) };
        (block, buffer)
    }

    /// The floats of `buffer`, whose work has completed.
    fn floats(buffer: &Buffer) -> Vec<f32> {
        let contents = buffer.contents().expect("a Shared buffer");
        // SAFETY: the buffer's bytes, which no command uses now.
        let bytes =
            unsafe { std::slice::from_raw_parts(contents.as_ptr(), buffer.length() as usize) };
        let words = bytes.as_chunks::<4>().0;
        words.iter().map(|w| f32::from_le_bytes(*w)).collect()
    }

    /// lsum over `groups` threadgroups of `group` threads, its `in[i] = i`, on every device of
    /// the cfg's kind; `check` judges each device's sums, one a threadgroup.
    fn sums(test: &str, group: u32, groups: u32, check: impl Fn(&str, &[f32])) {
        let Some(descriptor) = descriptor(&lsum(), "__entry_lsum") else {
            return;
        };
        let input: Vec<f32> = (0..group * groups).map(|i| i as f32).collect();
        each_device(test, |name, _, device| {
            let state = device
                .new_compute_pipeline_state(&descriptor, crate::tests::no_constexpr_samplers)
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let (_input_block, input) = shared(&device, &bytes_of(&input));
            let (_out_block, out) = shared(&device, &vec![0xff; groups as usize * 4]);
            let queue = device.new_command_queue(4).expect("a queue");
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let mut encoder = command_buffer
                .compute_command_encoder(DispatchType::Serial)
                .expect("an encoder");
            encoder.set_compute_pipeline_state(&state).expect("a state");
            (encoder.set_bytes(&offset_and_dims(1), 0)).expect("the offset");
            encoder.set_buffer(Some(&input), 0, 3).expect("in");
            encoder.set_buffer(Some(&out), 0, 4).expect("out");
            (encoder.set_threadgroup_memory_length(group * 4, 0)).expect("tmp");
            encoder
                .dispatch_threads([group * groups, 1, 1], [group, 1, 1])
                .unwrap_or_else(|error| panic!("{name}: the dispatch: {error}"));
            encoder.end_encoding();
            command_buffer.commit().expect("a commit");
            finished(name, &command_buffer);
            assert_eq!(command_buffer.error(), None, "{name}");
            check(name, &floats(&out));
        });
    }

    /// Four threadgroups of 64 each sum their `in[i] = i` through `__local` memory and a
    /// barrier, as the M3 Max sums them, on every device.
    #[test]
    fn every_device_sums_each_threadgroup_through_local_memory_in_opencl_mode() {
        sums("opencl_local lsum", 64, 4, |name, sums| {
            println!("{name}: lsum, each threadgroup's sum {sums:?}");
            assert_eq!(sums, [2016.0, 6112.0, 10208.0, 14304.0], "{name}: lsum");
        });
    }

    /// lsum in 64 threadgroups of 256 threads, more than one wave on every lab card, so each
    /// group's sum reads `__local` words other waves wrote: group `g` sums to
    /// `65536 g + 32640`, each partial sum an integer below 2^24 and so exact in `f32`, on every
    /// device.
    #[test]
    fn every_device_sums_threadgroups_of_256_through_local_memory_across_waves_in_opencl_mode() {
        sums("opencl_local lsum 256", 256, 64, |name, sums| {
            let wrong: Vec<(u32, f32, f32)> = (0..)
                .zip(sums)
                .map(|(g, &got)| (g, got, (65_536 * g + 32_640) as f32))
                .filter(|&(_, got, want)| got != want)
                .collect();
            println!(
                "{name}: lsum of 256 threads a group: {} of 64 sums differ, the first \
                 (group, got, want) {:?}",
                wrong.len(),
                &wrong[..wrong.len().min(8)]
            );
            assert!(wrong.is_empty(), "{name}: sums differ: {wrong:?}");
        });
    }

    /// The rows of `a` and `c`.
    const M: u32 = 25_600;
    /// The columns of `b` and `c`.
    const N: u32 = 160;
    /// The columns of `a` and the rows of `b`: four tiles.
    const K: u32 = 128;
    /// The period of `a`'s rows: row `r` holds row `r % PERIOD`'s values, so the CPU computes
    /// `PERIOD` rows of the product. It is past 32, so the 32 rows a threadgroup loads differ
    /// from each other.
    const PERIOD: u32 = 97;
    /// The dispatches of the multiply, each into its own `c`.
    const REPEATS: usize = 8;
    /// The seed of `b`'s first element, past `a`'s last.
    const B_SEEDS: u32 = 1 << 20;

    /// The element of seed `seed` of `a` or `b`: a multiple of 1/64 from -255/64 to 255/64.
    ///
    /// So each product of an element of `a` and one of `b` is a multiple of 2^-12 of at most
    /// 255² · 2^-12 in magnitude, and each partial sum of K = 128 of them a multiple of 2^-12 of
    /// at most 128 · 255² · 2^-12, about 2032: 8,323,200 steps of 2^-12, within the 2^24 an
    /// `f32` holds exactly. Every product and partial sum is exact, so `c` is the same in any
    /// order of summation, fused or not, and is compared exactly: an element that differs is
    /// wrong, not rounded.
    fn value(seed: u32) -> f32 {
        let h = seed.wrapping_mul(0x9e37_79b1);
        let h = (h ^ (h >> 15)).wrapping_mul(0x85eb_ca77);
        let h = h ^ (h >> 13);
        let steps = i16::try_from(h % 511).expect("below 511") - 255;
        f32::from(steps) / 64.0
    }

    /// How a `c` differs from the CPU's product: the elements that differ, the largest
    /// difference (an element left unwritten, NaN, counts as infinite), and the first 8 as
    /// `(row, column, got, want)`.
    struct Differences {
        count: usize,
        largest: f32,
        first: Vec<(u32, u32, f32, f32)>,
    }

    /// How `c` differs from `want`, the product's `PERIOD` rows.
    fn differences(c: &[f32], want: &[f32]) -> Differences {
        let mut found = Differences {
            count: 0,
            largest: 0.0,
            first: Vec::new(),
        };
        for row in 0..M {
            let got = &c[(row * N) as usize..((row + 1) * N) as usize];
            let at = row % PERIOD * N;
            let want = &want[at as usize..(at + N) as usize];
            for (column, (&got, &want)) in (0..).zip(got.iter().zip(want)) {
                if got == want {
                    continue;
                }
                found.count += 1;
                let by = match got.is_nan() {
                    true => f32::INFINITY,
                    false => (got - want).abs(),
                };
                found.largest = found.largest.max(by);
                if found.first.len() < 8 {
                    found.first.push((row, column, got, want));
                }
            }
        }
        found
    }

    /// Our tiled matrix multiply through two `__local` tiles of 4096 bytes at threadgroup
    /// indices 0 and 1, behind `barrier(CLK_LOCAL_MEM_FENCE)`, computes every element of each
    /// `c` as the CPU does, on every device ([`multiplies`]).
    #[test]
    fn every_device_multiplies_matrices_through_two_local_tiles_in_opencl_mode() {
        multiplies("opencl_local tiled_matmul", Tiles::Local, 1);
    }

    /// The same behind `barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE)`.
    #[test]
    fn every_device_multiplies_matrices_through_local_tiles_behind_both_fence_flags() {
        multiplies("opencl_local tiled_matmul both", Tiles::Local, 3);
    }

    /// The same with the tiles in the threadgroup's own part of a `__global` buffer, so what
    /// crosses waves goes through device memory, behind `barrier(CLK_GLOBAL_MEM_FENCE)` alone.
    #[test]
    fn every_device_multiplies_matrices_through_global_tiles_behind_the_global_fence_flag() {
        multiplies("opencl_local tiled_matmul global", Tiles::Global, 2);
    }

    /// Our tiled matrix multiply, its tiles where `tiles` says and its `barrier` called with
    /// `flags`, dispatched [`REPEATS`] times over 160 by 3200 threads in threadgroups of 32 by 4,
    /// computes every element of each `c` as the CPU does, on every device of the cfg's kind;
    /// `test` names the run.
    fn multiplies(test: &str, tiles: Tiles, flags: u32) {
        let Some(descriptor) = descriptor(&tiled_matmul(tiles, flags), "__entry_tiled_matmul")
        else {
            return;
        };
        let a: Vec<f32> = (0..M)
            .flat_map(|row| (0..K).map(move |k| value(row % PERIOD * K + k)))
            .collect();
        let b: Vec<f32> = (0..K * N).map(|i| value(B_SEEDS + i)).collect();
        let (a_rows, b_rows) = (&a, &b);
        let want: Vec<f32> = (0..PERIOD)
            .flat_map(|row| {
                (0..N).map(move |column| {
                    let sum: f64 = (0..K)
                        .map(|k| {
                            let x = a_rows[(row * K + k) as usize];
                            f64::from(x) * f64::from(b_rows[(k * N + column) as usize])
                        })
                        .sum();
                    let exact = sum as f32;
                    assert_eq!(f64::from(exact), sum, "the product is exact in f32");
                    exact
                })
            })
            .collect();
        let tile = 32 * 32 * 4;
        // The global form's scratch: each threadgroup's two tiles.
        let scratch_len = u64::from(N / 32 * (M / WORK / 4) * 2 * tile);
        each_device(test, |name, _, device| {
            let state = device
                .new_compute_pipeline_state(&descriptor, crate::tests::no_constexpr_samplers)
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let (_a_block, a) = shared(&device, &bytes_of(&a));
            let (_b_block, b) = shared(&device, &bytes_of(&b));
            let unwritten = vec![0xff; (M * N * 4) as usize];
            let outputs: Vec<(Block, Buffer)> =
                (0..REPEATS).map(|_| shared(&device, &unwritten)).collect();
            let queue = device.new_command_queue(4).expect("a queue");
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let mut encoder = command_buffer
                .compute_command_encoder(DispatchType::Serial)
                .expect("an encoder");
            encoder.set_compute_pipeline_state(&state).expect("a state");
            (encoder.set_bytes(&offset_and_dims(2), 0)).expect("the offset");
            encoder.set_buffer(Some(&a), 0, 3).expect("a");
            encoder.set_buffer(Some(&b), 0, 4).expect("b");
            encoder.set_bytes(&N.to_le_bytes(), 6).expect("n");
            encoder.set_bytes(&K.to_le_bytes(), 7).expect("k");
            let scratch = match tiles {
                Tiles::Local => {
                    (encoder.set_threadgroup_memory_length(tile, 0)).expect("at");
                    (encoder.set_threadgroup_memory_length(tile, 1)).expect("bt");
                    None
                }
                Tiles::Global => {
                    let block = (device.take_block(MemoryClass::Shared, scratch_len))
                        .expect("a Shared block");
                    let buffer = (device.new_buffer(block.id, 0, scratch_len, Some(0)))
                        .expect("the scratch");
                    encoder.set_buffer(Some(&buffer), 0, 8).expect("scratch");
                    Some((block, buffer))
                }
            };
            for (_, c) in &outputs {
                encoder.set_buffer(Some(c), 0, 5).expect("c");
                encoder
                    .dispatch_threads([N, M / WORK, 1], [32, 4, 1])
                    .unwrap_or_else(|error| panic!("{name}: the dispatch: {error}"));
            }
            encoder.end_encoding();
            command_buffer.commit().expect("a commit");
            finished(name, &command_buffer);
            assert_eq!(command_buffer.error(), None, "{name}");
            let mut wrong = Vec::new();
            for (repeat, (_, c)) in outputs.iter().enumerate() {
                let found = differences(&floats(c), &want);
                println!(
                    "{test} {name}: repeat {repeat}: {} of {} elements differ, by up to {}, \
                     the first (row, column, got, want) {:?}",
                    found.count,
                    M * N,
                    found.largest,
                    found.first
                );
                if found.count > 0 {
                    wrong.push((repeat, found.count, found.largest));
                }
            }
            assert!(
                wrong.is_empty(),
                "{test} {name}: elements differ, (repeat, elements, by up to): {wrong:?}"
            );
        });
    }
}
