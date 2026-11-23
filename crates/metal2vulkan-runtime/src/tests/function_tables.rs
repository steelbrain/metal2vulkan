//! Visible-function tables of compute pipeline states (`crate::function_table`), under a device
//! cfg on every device of the cfg's kind, through the owned engine fixtures
//! `visible_table_caller` and `visible_table_functions`: `visible_table_dispatch` calls entry
//! `tid % size` of the table at buffer 1 with `(tid, 3)` and writes the result at `out[tid]`,
//! 0xffffffff for a null table, 0xfffffffe for a table of no entries and 0xfffffffd for a null
//! entry. Its state links `table_add` and `table_mul` in `functions` and `table_sub` in
//! `private_functions`.
//!
//! - The device answers `supportsFunctionPointers` YES.
//! - The handles are the two callable functions', in their list's order; a private function and
//!   an unlinked one have none, Metal's nil.
//! - A dispatch calls the function each entry holds, exactly: a table of `[add, mul]` and one of
//!   `[mul, add]` (written by `setFunctions:withRange:`) write different words; a null entry, a
//!   table of no entries and an unbound slot write their markers; an entry set after the dispatch
//!   is encoded and before the commit is the one called.
//! - An entry past the table's size is `Refusal::FunctionTableIndex` and writes nothing; two
//!   tables answer two non-zero resource IDs.
//!
//! `llvm-as` assembles the fixtures; without it the test skips visibly.

#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod on_device {
    use crate::tests::command::on_device::{each_device, finished, Arena};
    use crate::{
        Buffer, ComputeCommandEncoder, ComputePipelineDescriptor, DispatchType, Error,
        FunctionDescriptor, FunctionHandle, Library, LinkedFunction, Refusal, StageSource,
        VisibleFunctionTable, VisibleFunctionTableDescriptor,
    };
    use metal2vulkan::pipeline::LinkedFunctions;
    use metal2vulkan_devtools::{engine_dir, llvm_assemble, AssembleError};

    /// The threads of each dispatch, and the words each writes.
    const THREADS: u32 = 12;

    /// A function a table entry calls, as the CPU computes it.
    type Op = fn(u32, u32) -> u32;

    /// The library of the engine fixture `fixture.ll`, or None where `llvm-as` is missing.
    fn library(fixture: &str) -> Option<Library> {
        match llvm_assemble(&engine_dir().join(format!("{fixture}.ll"))) {
            Ok(bitcode) => Some(Library::from_air(&bitcode).expect("the library")),
            Err(AssembleError::Missing(why)) => {
                eprintln!("SKIPPED: {why}");
                None
            }
            Err(error) => panic!("{fixture}: {error:?}"),
        }
    }

    /// The words of `buffer`, read once its writer completed.
    fn words(buffer: &Buffer) -> Vec<u32> {
        let contents = buffer.contents().expect("a Shared buffer");
        // SAFETY: the work that wrote the buffer completed; it holds at least `THREADS` words.
        let words = unsafe {
            std::slice::from_raw_parts(contents.as_ptr().cast::<u32>(), THREADS as usize)
        };
        words.to_vec()
    }

    /// What `visible_table_dispatch` writes for thread `tid` through a table of `entries`, each
    /// the function an entry calls (`None` for a null entry), at its index.
    fn expected(entries: &[Option<Op>]) -> Vec<u32> {
        (0..THREADS)
            .map(
                |tid| match entries.get((tid as usize) % entries.len().max(1)) {
                    None => 0xffff_fffe,
                    Some(None) => 0xffff_fffd,
                    Some(Some(f)) => f(tid, 3),
                },
            )
            .collect()
    }

    /// One dispatch of `THREADS` threads writing `out`, with `table` at buffer 1.
    fn dispatch(
        encoder: &mut ComputeCommandEncoder,
        out: &Buffer,
        table: Option<&VisibleFunctionTable>,
    ) {
        encoder.set_buffer(Some(out), 0, 0).expect("out");
        encoder
            .set_visible_function_table(table, 1)
            .expect("the table");
        (encoder.dispatch_threads([THREADS, 1, 1], [THREADS, 1, 1])).expect("a dispatch");
    }

    #[test]
    fn every_device_calls_the_function_each_entry_of_a_table_holds() {
        let (Some(caller), Some(functions)) = (
            library("visible_table_caller"),
            library("visible_table_functions"),
        ) else {
            return;
        };
        let function = |name: &str| functions.function(name).expect(name);
        let linked = LinkedFunctions::new()
            .with_function(function("table_add"))
            .with_function(function("table_mul"))
            .with_private_function(function("table_sub"));
        let kernel = caller
            .function("visible_table_dispatch")
            .expect("the kernel");
        let descriptor = ComputePipelineDescriptor::new(StageSource::Function(
            FunctionDescriptor::new(kernel).with_linked(linked),
        ));
        let add: Op = |a, b| a + b;
        let mul: Op = |a, b| a * b;
        each_device("function_tables", |name, _, device| {
            assert!(device.supports_function_pointers(), "{name}");
            let state = (device
                .new_compute_pipeline_state(&descriptor, crate::tests::no_constexpr_samplers))
            .unwrap_or_else(|error| panic!("{name}: {error}"));
            let handle = |f: &str| state.function_handle(&LinkedFunction::new(function(f)));
            let words_of = |f: &str| handle(f).map(FunctionHandle::word);
            assert_eq!(words_of("table_add"), Some(2), "{name}: table_add");
            assert_eq!(words_of("table_mul"), Some(3), "{name}: table_mul");
            assert_eq!(words_of("table_sub"), None, "{name}: a private function");
            assert_eq!(words_of("table_half"), None, "{name}: an unlinked function");
            let (h_add, h_mul) = (handle("table_add"), handle("table_mul"));
            let table = |count: u64| {
                (state.new_visible_function_table(&VisibleFunctionTableDescriptor::new(count)))
                    .unwrap_or_else(|error| panic!("{name}: a table: {error}"))
            };
            let pair = table(2);
            pair.set_function(h_add, 0).expect("entry 0");
            pair.set_function(h_mul, 1).expect("entry 1");
            let swapped = table(2);
            swapped
                .set_functions(&[h_mul, h_add], 0)
                .expect("the range");
            let holed = table(3);
            holed
                .set_functions(&[h_add, None, h_mul], 0)
                .expect("the range");
            let empty = table(0);
            let late = table(2);
            late.set_functions(&[h_add, h_add], 0).expect("the range");
            // Past the size: refused, and nothing written.
            for refused in [
                late.set_function(h_mul, 2),
                late.set_functions(&[h_mul; 2], 1),
            ] {
                assert!(
                    matches!(
                        refused,
                        Err(Error::Refused(Refusal::FunctionTableIndex { count: 2, .. }))
                    ),
                    "{name}: {refused:?}"
                );
            }
            let (a, b) = (pair.gpu_resource_id(), swapped.gpu_resource_id());
            assert!(
                a != 0 && b != 0 && a != b,
                "{name}: resource IDs {a:#x} {b:#x}"
            );

            let mut arena = Arena::new(&device);
            let outs: Vec<Buffer> = (0..6).map(|_| arena.buffer()).collect();
            let queue = device.new_command_queue(1).expect("a queue");
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let mut encoder =
                (command_buffer.compute_command_encoder(DispatchType::Serial)).expect("an encoder");
            encoder
                .set_compute_pipeline_state(&state)
                .expect("the state");
            let tables = [
                Some(&pair),
                Some(&swapped),
                Some(&holed),
                Some(&empty),
                None,
            ];
            for (out, table) in outs.iter().zip(tables) {
                dispatch(&mut encoder, out, table);
            }
            dispatch(&mut encoder, &outs[5], Some(&late));
            encoder.end_encoding();
            // Set after the encoding, before the commit: the entry the dispatch calls.
            late.set_function(h_mul, 1).expect("entry 1");
            command_buffer.commit().expect("a commit");
            finished(name, &command_buffer);
            assert_eq!(command_buffer.error(), None, "{name}");
            let expect = [
                expected(&[Some(add), Some(mul)]),
                expected(&[Some(mul), Some(add)]),
                expected(&[Some(add), None, Some(mul)]),
                expected(&[]),
                vec![0xffff_ffff; THREADS as usize],
                expected(&[Some(add), Some(mul)]),
            ];
            for (at, (out, expect)) in outs.iter().zip(expect).enumerate() {
                assert_eq!(words(out), expect, "{name}: dispatch {at}");
            }
            println!("{name}: every entry called the function it holds");
        });
    }
}
