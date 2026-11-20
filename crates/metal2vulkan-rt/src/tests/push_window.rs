//! The push window, under a device cfg on every device of the cfg's kind. Each kernel reads its
//! grid position and its buffers at a run-time offset, so it pushes a dispatch region, each
//! buffer's address, and the containment check's fields: the zero range's address and each
//! buffer's bounds (its bytes before the bind offset and its length, 8 bytes), which come last in
//! the order, so they are the first to go to a parameter block. A kernel of 12 buffers (232
//! bytes) pushes them all and writes nothing into the parameter ring, and its dispatches after
//! the first push nothing, their bytes held. One of 24 buffers pushes its block's address, the
//! region, 22 buffers' addresses and the zero tail word 24 bytes above them (`docs/API.md` section
//! 5), and the other 2 addresses, the zero range's and the 24 bounds go to a block; one of 30
//! pushes the same 22 addresses, and the other 8, the zero range's and the 30 bounds go to a
//! block. Each of those two writes one block a dispatch, whose new address each dispatch pushes.
//! All answer the fold of their inputs.
//!
//! `llvm-as` assembles the kernels; without it the test skips visibly.

#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod on_device {
    use crate::pipeline::{FieldValue, Place};
    use crate::tests::command::on_device::{each_device, finished};
    use crate::{
        ComputePipelineDescriptor, DispatchType, FunctionDescriptor, Library, MemoryClass,
        StageSource,
    };
    use metal2vulkan_devtools::{llvm_assemble_text, AssembleError};

    /// The threads of a dispatch, and the words of each buffer.
    const THREADS: u32 = 64;

    /// The dispatches of each kernel.
    const DISPATCHES: u64 = 3;

    /// A kernel of `buffers` buffers: thread `gid` folds word `gid` of the first `buffers - 1`
    /// (`s · 3 + v`) into word `gid` of the last.
    fn kernel(buffers: u32) -> String {
        let inputs = buffers - 1;
        let mut params = Vec::new();
        let mut meta = Vec::new();
        for k in 0..buffers {
            params.push(format!("ptr addrspace(1) %b{k}"));
            let access = if k == inputs {
                "air.read_write"
            } else {
                "air.read"
            };
            meta.push(format!(
                "!{} = !{{i32 {k}, !\"air.buffer\", !\"air.location_index\", i32 {k}, i32 1, !\"{access}\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"b{k}\"}}",
                10 + k
            ));
        }
        params.push("i32 %gid".to_owned());
        meta.push(format!(
            "!{} = !{{i32 {buffers}, !\"air.thread_position_in_grid\", !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"gid\"}}",
            10 + buffers
        ));
        let mut body = String::from("  %w = zext i32 %gid to i64\n");
        for k in 0..inputs {
            body.push_str(&format!(
                "  %p{k} = getelementptr inbounds i32, ptr addrspace(1) %b{k}, i64 %w\n  %v{k} = load i32, ptr addrspace(1) %p{k}, align 4\n"
            ));
        }
        body.push_str("  %s0 = add i32 %v0, 0\n");
        for k in 1..inputs {
            body.push_str(&format!(
                "  %m{k} = mul i32 %s{}, 3\n  %s{k} = add i32 %m{k}, %v{k}\n",
                k - 1
            ));
        }
        let args: Vec<String> = (0..=buffers).map(|k| format!("!{}", 10 + k)).collect();
        format!(
            "; Owned synthetic kernel.\nsource_filename = \"wide.metal\"\n\n\
define void @wide({}) {{\nentry:\n{body}  %out = getelementptr inbounds i32, ptr addrspace(1) %b{inputs}, i64 %w\n  store i32 %s{}, ptr addrspace(1) %out, align 4\n  ret void\n}}\n\n\
!air.kernel = !{{!0}}\n!0 = !{{ptr @wide, !1, !2}}\n!1 = !{{}}\n!2 = !{{{}}}\n{}\n",
            params.join(", "),
            inputs - 1,
            args.join(", "),
            meta.join("\n"),
        )
    }

    /// Word `gid` of input `k`.
    fn word(k: u32, gid: u32) -> u32 {
        (k + 1).wrapping_mul(0x9e37_79b9) ^ gid.wrapping_mul(0x85eb_ca6b)
    }

    /// The kernel of 12 buffers writes nothing into the ring, and pushes once for its three
    /// dispatches of the same bindings; the kernels of 24 and 30 push 22 addresses below their
    /// tail and have the other 2 and 8 in their block with every check field; each of those
    /// writes one block a dispatch, whose new address each dispatch pushes. Each answers the fold
    /// of its inputs.
    #[test]
    fn every_device_pushes_12_buffers_and_rings_24_and_30() {
        // Each kernel's buffers, its addresses and check fields in the block, its blocks a
        // dispatch, and its pushes over the dispatches.
        for (buffers, (addresses, checks), blocks, pushes) in [
            (12u32, (0usize, 0usize), 0u64, 1u64),
            (24, (2, 25), 1, DISPATCHES),
            (30, (8, 31), 1, DISPATCHES),
        ] {
            let bitcode = match llvm_assemble_text(&kernel(buffers)) {
                Ok(bitcode) => bitcode,
                Err(AssembleError::Missing(why)) => {
                    eprintln!("SKIPPED: {why}");
                    return;
                }
                Err(error) => panic!("wide: {error:?}"),
            };
            let library = Library::from_air(&bitcode).expect("the library");
            let function = library.function("wide").expect("the kernel");
            let descriptor = ComputePipelineDescriptor::new(StageSource::Function(
                FunctionDescriptor::new(function),
            ));
            let len = u64::from(THREADS * 4);
            each_device(&format!("push_window_{buffers}"), |name, _, device| {
                let what = format!("{name}, {buffers} buffers");
                let state = device
                    .new_compute_pipeline_state(&descriptor, crate::tests::no_constexpr_samplers)
                    .unwrap_or_else(|error| panic!("{what}: {error}"));
                // The fields in the block: addresses, and the check's (the zero range's address
                // and the bounds); nothing else goes there.
                let mut ringed = (0, 0);
                for field in &state.inner.compiled.whole.interface.fields {
                    match (field.place, &field.value) {
                        (Place::Push(_), _) => {}
                        (Place::Block(_), FieldValue::BufferAddress { .. }) => ringed.0 += 1,
                        (
                            Place::Block(_),
                            FieldValue::BufferBounds(_) | FieldValue::ZeroAddress,
                        ) => ringed.1 += 1,
                        (Place::Block(_), _) => panic!("{what}: another field in the block"),
                    }
                }
                assert_eq!(
                    ringed,
                    (addresses, checks),
                    "{what}: addresses and check fields in the block"
                );
                let block = device
                    .take_block(MemoryClass::Shared, u64::from(buffers) * len)
                    .expect("a Shared block");
                let bound: Vec<_> = (0..u64::from(buffers))
                    .map(|k| {
                        device
                            .new_buffer(block.id, k * len, len, Some(0))
                            .expect("a buffer")
                    })
                    .collect();
                for (k, buffer) in bound.iter().enumerate().take(buffers as usize - 1) {
                    let bytes: Vec<u8> = (0..THREADS)
                        .flat_map(|gid| word(k as u32, gid).to_le_bytes())
                        .collect();
                    let contents = buffer.contents().expect("a Shared buffer");
                    // SAFETY: the bytes lie in the buffer, which is alive, and no GPU work uses it.
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            bytes.as_ptr(),
                            contents.as_ptr(),
                            bytes.len(),
                        )
                    };
                }
                let queue = device.new_command_queue(4).expect("a queue");
                let command_buffer = queue.command_buffer().expect("a command buffer");
                let before = device.counters().metrics();
                let mut encoder = command_buffer
                    .compute_command_encoder(DispatchType::Serial)
                    .expect("an encoder");
                encoder.set_compute_pipeline_state(&state).expect("a state");
                for (k, buffer) in bound.iter().enumerate() {
                    encoder
                        .set_buffer(Some(buffer), 0, k as u32)
                        .expect("a buffer");
                }
                for _ in 0..DISPATCHES {
                    encoder
                        .dispatch_threads([THREADS, 1, 1], [THREADS, 1, 1])
                        .expect("a dispatch");
                }
                encoder.end_encoding();
                command_buffer.commit().expect("a commit");
                finished(name, &command_buffer);
                let after = device.counters().metrics();
                let writes = after.ring_writes - before.ring_writes;
                assert_eq!(command_buffer.error(), None, "{what}");
                assert_eq!(writes, blocks * DISPATCHES, "{what}: ring writes");
                let pushed = after.pushes - before.pushes;
                let held = after.pushes_held - before.pushes_held;
                assert_eq!(
                    (pushed, held),
                    (pushes, DISPATCHES - pushes),
                    "{what}: pushes, held"
                );
                let inputs = buffers - 1;
                let contents = bound[inputs as usize].contents().expect("a Shared buffer");
                for gid in 0..THREADS {
                    let want = (1..inputs).fold(word(0, gid), |s, k| {
                        s.wrapping_mul(3).wrapping_add(word(k, gid))
                    });
                    // SAFETY: the word lies in the buffer, which is alive, and its work completed.
                    let got = unsafe {
                        contents
                            .as_ptr()
                            .cast::<u32>()
                            .add(gid as usize)
                            .read_unaligned()
                    };
                    assert_eq!(got, want, "{what}: thread {gid}");
                }
                println!(
                    "{what}: {writes} ring writes and {pushed} pushes over {DISPATCHES} dispatches"
                );
            });
        }
    }
}
