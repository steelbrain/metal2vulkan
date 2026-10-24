//! Visible-function tables and function handles on the device the cfg selects: the runtime's tables
//! and handles answer as Metal's measured ones do (an M3 Max on macOS 26.7 and an M4 on 27.0.1, the
//! same on both), a kernel calls the functions its tables hold, a buffer bound where the kernel
//! reads a table reads as a null table, as plain Metal reads it, a table where it only reads a
//! buffer as an unbound buffer, zeros, as plain Metal reads it, and a table where it writes a
//! buffer is refused.
//!
//! The kernel is the owned `visible_table_caller` fixture's `visible_table_dispatch`: it calls
//! entry `tid % size` of the `uint(uint, uint)` table at buffer 1 with `(tid, 3)` and writes the
//! result to buffer 0 (0xfffffffd for a null entry, 0xfffffffe for an empty table, 0xffffffff for
//! a null table). The functions are `visible_table_functions`'s `table_add`, `table_mul` and
//! `table_sub` of that type, and `table_half` of another; its kernel links nothing.
//!
//! Tables an argument buffer holds are read by the owned `uni_argument_tables` fixture's kernels,
//! which call through a table member of the argument buffer at buffer 1 the same way: one member, a
//! member array indexed per thread, and a per-thread select of two members.
//!
//! A render state's vertex stage calls through its own tables: the owned `vft_vertex_caller`
//! fixture's vertex function `vft_vertex_dispatch` calls entry `vid % size` of the table at vertex
//! buffer 1 as the kernel does, writing word `vid` of vertex buffer 0.
//!
//! A tile state's tile stage calls through its own tables (the leftover): the owned
//! `vft_tile_caller` fixture's tile function `vft_tile_dispatch` calls entry `i % size` of the
//! table at tile buffer 1 as the kernel does, `i` its thread's index over the pass's tiles, writing
//! word `i` of tile buffer 0.

use super::executions::on_device_with;
use super::host_memory::{on_device, run};
use super::read_buffer;
use crate::air_library::AirLibrary;
use metal2vulkan::device::VkFeature;
use metal2vulkan::metal::{PixelFormat, PrimitiveType, RenderStages, TextureUsage};
use metal2vulkan::pipeline::{
    ColorAttachmentDescriptor, ComputePipelineDescriptor, LinkedFunctions,
    RenderPipelineDescriptor, StageSource, TileRenderPipelineDescriptor,
};
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use metal2vulkan_runtime::{
    Buffer, ComputeCommandEncoder, ComputePipelineState, CpuCacheMode, Device, Error,
    FunctionTableProblem, HazardTrackingMode, Invalid, Library, LoadAction, PipelineOptions,
    Problem, PurgeableState, RenderPassDescriptor, RenderPipelineState, ResourceId,
    ResourceOptions, RuntimeGap, Size, StorageMode, StoreAction, TextureDescriptor,
    VisibleFunctionTable, VisibleFunctionTableDescriptor,
};

/// Threads of each dispatch.
const N: u32 = 40;

fn invalid(call: &'static str, problem: Problem) -> Error {
    Error::Invalid(Invalid::new(call, problem))
}

fn table_problem(call: &'static str, problem: FunctionTableProblem) -> Error {
    invalid(call, Problem::FunctionTable(problem))
}

/// The owned fixtures' libraries: the caller's and the functions'.
fn libraries(device: &Device) -> (Library, Library) {
    let fixture = |stem: &str| llvm_assemble(&engine_dir().join(format!("{stem}.ll"))).unwrap();
    (
        device
            .new_library_with_air(&fixture("visible_table_caller"))
            .unwrap(),
        device
            .new_library_with_air(&fixture("visible_table_functions"))
            .unwrap(),
    )
}

/// The state of `kernel` of `caller`, linking `linked`.
fn state_of(caller: &Library, kernel: &str, linked: LinkedFunctions) -> ComputePipelineState {
    let kernel = caller.new_function_with_name(kernel).unwrap();
    caller
        .device()
        .new_compute_pipeline_state(
            &ComputePipelineDescriptor::new(StageSource::Function(
                kernel.function_descriptor().with_linked(linked),
            )),
            &PipelineOptions::default(),
        )
        .unwrap()
}

/// `visible_table_dispatch` linking the four functions in `functions`.
fn dispatch_state(caller: &Library, functions: &Library) -> ComputePipelineState {
    state_of(caller, "visible_table_dispatch", linked_four(functions))
}

fn table(state: &ComputePipelineState, count: u64) -> VisibleFunctionTable {
    state
        .new_visible_function_table(&VisibleFunctionTableDescriptor::new(count))
        .unwrap()
}

/// Run `state` over `N` threads, the output at buffer 0 and `table` at buffer 1 (none: unbound),
/// with `after` called once the dispatch is encoded and before the commit; the words written.
fn words_of(
    device: &Device,
    state: &ComputePipelineState,
    table: Option<&VisibleFunctionTable>,
    after: impl FnOnce(),
) -> Vec<u32> {
    let out = device
        .new_buffer(u64::from(N) * 4, ResourceOptions::shared())
        .unwrap();
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(state).unwrap();
    encoder.set_buffer(Some(&out), 0, 0).unwrap();
    if table.is_some() {
        encoder.set_visible_function_tables(&[table], 1).unwrap();
    }
    encoder
        .dispatch_threads(Size::new(N.into(), 1, 1), Size::new(N.into(), 1, 1))
        .unwrap();
    encoder.end_encoding().unwrap();
    after();
    run(&command_buffer);
    words(&out)
}

fn words(buffer: &Buffer) -> Vec<u32> {
    read_buffer(buffer, 0, buffer.length())
        .unwrap()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|w| u32::from_le_bytes(*w))
        .collect()
}

/// What Metal answers for tables and handles: a handle for each function the kernel links in
/// `functions` or `binary_functions`, the same handle for the same function, none for a private
/// function, an unlinked one or the kernel; a table is a `Shared`, tracked resource of 256 bytes
/// and 8 per entry outside any heap, always non-volatile, with an ID of the tables' own space from
/// 1; a kernel without linked functions makes tables too. Refused as Metal's validation stops it:
/// an entry past the table's size and a handle of another state; refused as no buffer holds its
/// words, a table past `max_buffer_length`.
#[test]
fn tables_and_handles_answer_as_metal_s_do() {
    on_device(|device| {
        let (caller, functions) = libraries(device);
        let function = |name: &str| functions.new_function_with_name(name).unwrap();
        let state = dispatch_state(&caller, &functions);
        let add = state.function_handle(&function("table_add")).unwrap();
        let add = add.expect("a linked function's handle");
        assert_eq!(add.name(), "table_add");
        assert_eq!(add.device().registry_id(), device.registry_id());
        assert_eq!(
            state.function_handle(&function("table_add")).unwrap(),
            Some(add.clone()),
            "one handle for one function"
        );
        let kernel = caller
            .new_function_with_name("visible_table_dispatch")
            .unwrap();
        assert_eq!(state.function_handle(&kernel).unwrap(), None);
        let private = state_of(
            &caller,
            "visible_table_dispatch",
            LinkedFunctions::new()
                .with_function(function("table_add").linked_function())
                .with_private_function(function("table_mul").linked_function()),
        );
        assert_eq!(
            private.function_handle(&function("table_mul")).unwrap(),
            None
        );
        assert_eq!(
            private.function_handle(&function("table_sub")).unwrap(),
            None
        );
        // A stage's binary functions are compiled to a binary, as Metal requires.
        let mut compiled = metal2vulkan_runtime::FunctionDescriptor::new("table_sub");
        compiled.options = metal2vulkan::metal::FunctionOptions::COMPILE_TO_BINARY;
        let mut binary = LinkedFunctions::new();
        binary.binary_functions = vec![functions
            .new_function_with_descriptor(&compiled)
            .unwrap()
            .linked_function()];
        let binary = state_of(&caller, "visible_table_dispatch", binary);
        let sub = binary.function_handle(&function("table_sub")).unwrap();
        assert_eq!(sub.map(|h| h.name().to_owned()), Some("table_sub".into()));

        let tables: Vec<VisibleFunctionTable> = [0, 1, 4].map(|n| table(&state, n)).into();
        let answers: Vec<_> = tables
            .iter()
            .map(|t| (t.allocated_size(), t.gpu_resource_id()))
            .collect();
        assert_eq!(
            answers,
            [
                (256, ResourceId(1)),
                (264, ResourceId(2)),
                (288, ResourceId(3))
            ]
        );
        let t = &tables[2];
        assert_eq!(t.storage_mode(), StorageMode::Shared);
        assert_eq!(t.cpu_cache_mode(), CpuCacheMode::DefaultCache);
        assert_eq!(t.hazard_tracking_mode(), HazardTrackingMode::Tracked);
        assert_eq!(t.resource_options().raw(), 512);
        assert!(t.heap().is_none() && t.heap_offset() == 0 && !t.is_aliasable());
        assert_eq!(t.label(), None);
        t.set_label("entries");
        assert_eq!(t.label().as_deref(), Some("entries"));
        for asked in [
            PurgeableState::KeepCurrent,
            PurgeableState::Volatile,
            PurgeableState::NonVolatile,
        ] {
            assert_eq!(t.set_purgeable_state(asked), PurgeableState::NonVolatile);
        }
        drop(tables);
        assert_eq!(table(&state, 2).gpu_resource_id(), ResourceId(1));

        let plain = function("visible_table_functions");
        let plain =
            device.new_compute_pipeline_state_with_function(&plain, &PipelineOptions::default());
        let plain = plain.unwrap_or_else(|e| panic!("{e}"));
        // A state that links nothing still makes a table of the size asked: its last index
        // takes a handle and the next one is out of range (`MTLVisibleFunctionTable` answers no
        // count, so its size is read through the calls that use it).
        let three = table(&plain, 3);
        three.set_function(None, 2).unwrap();
        assert_eq!(
            three.set_function(None, 3),
            Err(invalid(
                "set_function",
                Problem::IndexOutOfRange { index: 3, limit: 3 }
            ))
        );

        let two = table(&state, 2);
        assert_eq!(
            two.set_function(Some(&add), 2),
            Err(invalid(
                "set_function",
                Problem::IndexOutOfRange { index: 2, limit: 2 }
            ))
        );
        let other = dispatch_state(&caller, &functions);
        let foreign = other.function_handle(&function("table_add")).unwrap();
        assert_ne!(foreign, Some(add.clone()), "another state's handle");
        // Another kernel's handle is taken at its position (`a_handle_of_another_kernel_calls_
        // this_kernel_s_function_at_its_position`); a render state's, unmeasured on Metal, is
        // refused as Metal's validation stops it.
        two.set_functions(&[Some(&add), foreign.as_ref()], 0)
            .unwrap();
        let rendered = vertex_state(device, &functions)
            .function_handle(&function("table_add"), RenderStages::VERTEX)
            .unwrap();
        assert_eq!(
            two.set_functions(&[Some(&add), rendered.as_ref()], 0),
            Err(table_problem(
                "set_functions",
                FunctionTableProblem::HandleOfAnotherState
            ))
        );
        let max_buffer_length = device.properties().max_buffer_length;
        for count in [max_buffer_length / 4, 1 << 32, u64::MAX] {
            let refused =
                state.new_visible_function_table(&VisibleFunctionTableDescriptor::new(count));
            assert_eq!(
                refused.err(),
                Some(Error::NotImplemented(
                    RuntimeGap::FunctionTablePastMaxBufferLength {
                        count,
                        max_buffer_length
                    }
                )),
                "{count} entries"
            );
        }
    });
}

/// A kernel calls the functions its table holds: each thread calls entry `tid % 5` of a table
/// holding `table_add`, `table_mul`, a null entry, `table_sub` and `table_half` (another type,
/// which reaches the switch's default and yields 0). An empty table and an unbound one are observed
/// as on Metal. An entry set after the dispatch was encoded and before the commit is the one it
/// calls, as Metal's tables are read when the dispatch runs.
#[test]
fn a_kernel_calls_the_functions_its_table_holds() {
    on_device(|device| {
        let (caller, functions) = libraries(device);
        let function = |name: &str| functions.new_function_with_name(name).unwrap();
        let state = dispatch_state(&caller, &functions);
        let handle = |name: &str| state.function_handle(&function(name)).unwrap().unwrap();
        let five = table(&state, 5);
        let (add, mul, sub, half) = (
            handle("table_add"),
            handle("table_mul"),
            handle("table_sub"),
            handle("table_half"),
        );
        five.set_functions(&[Some(&add), Some(&mul)], 0).unwrap();
        five.set_function(Some(&sub), 3).unwrap();
        five.set_function(Some(&half), 4).unwrap();
        let expect = |tid: u32| match tid % 5 {
            0 => tid + 3,
            1 => tid * 3,
            2 => 0xffff_fffd,
            3 => tid.wrapping_sub(3),
            _ => 0,
        };
        let all: Vec<u32> = (0..N).map(expect).collect();
        assert_eq!(words_of(device, &state, Some(&five), || {}), all);
        assert_eq!(
            words_of(device, &state, Some(&table(&state, 0)), || {}),
            vec![0xffff_fffe; N as usize]
        );
        assert_eq!(
            words_of(device, &state, None, || {}),
            vec![0xffff_ffff; N as usize]
        );
        let late = words_of(device, &state, Some(&five), || {
            five.set_function(Some(&add), 2).unwrap();
        });
        let with_add: Vec<u32> = (0..N)
            .map(|tid| if tid % 5 == 2 { tid + 3 } else { expect(tid) })
            .collect();
        assert_eq!(late, with_add, "the entry set before the commit");
    });
}

/// A handle of another kernel's state set in a table calls the table's kernel's function at the
/// position the handle has in its own state, as plain Metal's does where its validation stops
/// the call (an M3 Max and an M4), Metal numbering a state's private functions first: `table_add`
/// second of the other state's callable functions calls `table_mul`, the table's state's second;
/// behind one private function, its third. A position past the table's state's functions calls its
/// last one, and is a null entry where it links none, as plain Metal's. A state that lists a
/// private function in a numbered list too, whose numbering Metal's probes did not measure, is
/// refused as Metal's validation stops it.
#[test]
fn a_handle_of_another_kernel_calls_this_kernel_s_function_at_its_position() {
    on_device(|device| {
        let (caller, functions) = libraries(device);
        let function = |name: &str| functions.new_function_with_name(name).unwrap();
        let linked = |names: &[&str], private: &[&str]| {
            let linked = names.iter().fold(LinkedFunctions::new(), |linked, name| {
                linked.with_function(function(name).linked_function())
            });
            private.iter().fold(linked, |linked, name| {
                linked.with_private_function(function(name).linked_function())
            })
        };
        let state = dispatch_state(&caller, &functions);
        let swapped = state_of(
            &caller,
            "visible_table_dispatch",
            linked(&["table_sub", "table_add"], &[]),
        );
        let behind = state_of(
            &caller,
            "visible_table_dispatch",
            linked(&["table_sub", "table_add"], &["table_mul"]),
        );
        let own = table(&state, 1);
        let add_of =
            |of: &ComputePipelineState| of.function_handle(&function("table_add")).unwrap();
        let calls = |table: &VisibleFunctionTable| words_of(device, &state, Some(table), || {});
        own.set_function(add_of(&swapped).as_ref(), 0).unwrap();
        assert_eq!(calls(&own), (0..N).map(|tid| tid * 3).collect::<Vec<_>>());
        own.set_function(add_of(&behind).as_ref(), 0).unwrap();
        assert_eq!(
            calls(&own),
            (0..N).map(|tid| tid.wrapping_sub(3)).collect::<Vec<_>>()
        );
        let refused = |handle: Option<metal2vulkan_runtime::FunctionHandle>| {
            assert_eq!(
                own.set_function(handle.as_ref(), 0),
                Err(table_problem(
                    "set_function",
                    FunctionTableProblem::HandleOfAnotherState
                ))
            );
        };
        let short = state_of(
            &caller,
            "visible_table_dispatch",
            linked(&["table_add"], &[]),
        );
        let short_table = table(&short, 1);
        let past = state.function_handle(&function("table_sub")).unwrap();
        short_table.set_function(past.as_ref(), 0).unwrap();
        assert_eq!(
            words_of(device, &short, Some(&short_table), || {}),
            (0..N).map(|tid| tid + 3).collect::<Vec<_>>()
        );
        // A kernel that links nothing takes the handle as a null entry. (A kernel that reads a
        // table and links nothing is refused by the compile layer, so this one reads none.)
        let bits = llvm_assemble(&engine_dir().join("a1_bool_from_float.ll")).unwrap();
        let library = device.new_library_with_air(&bits).unwrap();
        let unlinked = state_of(&library, "a1_bool_from_float", LinkedFunctions::new());
        table(&unlinked, 1).set_function(past.as_ref(), 0).unwrap();
        let listed_twice = state_of(
            &caller,
            "visible_table_dispatch",
            linked(&["table_sub", "table_add"], &["table_sub"]),
        );
        refused(add_of(&listed_twice));
    });
}

/// `vft_vertex_caller`'s `vft_vertex_dispatch` with rasterization disabled, its vertex stage
/// linking the four functions in `functions`.
fn vertex_state(device: &Device, functions: &Library) -> RenderPipelineState {
    vertex_state_linking(device, linked_four(functions))
}

/// The state of the owned `vft_vertex_caller` fixture's vertex function, linking `linked`.
fn vertex_state_linking(device: &Device, linked: LinkedFunctions) -> RenderPipelineState {
    vertex_state_adding(device, linked, false)
}

/// `vertex_state_linking`'s state, made with `support_adding_vertex_binary_functions` set to
/// `supported`.
fn vertex_state_adding(
    device: &Device,
    linked: LinkedFunctions,
    supported: bool,
) -> RenderPipelineState {
    let bitcode = llvm_assemble(&engine_dir().join("vft_vertex_caller.ll")).unwrap();
    let caller = device.new_library_with_air(&bitcode).unwrap();
    let vertex = caller
        .new_function_with_name("vft_vertex_dispatch")
        .unwrap();
    let source = vertex.function_descriptor().with_linked(linked);
    let mut descriptor = RenderPipelineDescriptor::new(StageSource::Function(source));
    descriptor.rasterization_enabled = false;
    descriptor.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
    descriptor.support_adding_vertex_binary_functions = supported;
    device
        .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap()
}

/// Draw `N` points of `state` in a pass of a 4x4 color target, the output at vertex buffer 0 and
/// `table` at vertex buffer 1 (none: unbound), with `after` called once the draw is encoded and
/// before the commit; the words written.
fn vertex_words_of(
    device: &Device,
    state: &RenderPipelineState,
    table: Option<&VisibleFunctionTable>,
    after: impl FnOnce(),
) -> Vec<u32> {
    vertex_words_with(device, state, At1::Table(table), after)
}

/// What a draw binds at vertex buffer 1: a table (none: unbound), or a buffer.
enum At1<'a> {
    Table(Option<&'a VisibleFunctionTable>),
    Buffer(&'a Buffer),
}

/// `vertex_words_of` with `at1` bound at vertex buffer 1.
fn vertex_words_with(
    device: &Device,
    state: &RenderPipelineState,
    at1: At1<'_>,
    after: impl FnOnce(),
) -> Vec<u32> {
    let out = device
        .new_buffer(u64::from(N) * 4, ResourceOptions::shared())
        .unwrap();
    let mut target = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, 4, 4, false);
    target.usage = TextureUsage::RENDER_TARGET;
    let mut pass = RenderPassDescriptor::new();
    pass.color_attachments[0].texture = Some(device.new_texture(&target).unwrap());
    pass.color_attachments[0].load_action = LoadAction::Clear;
    pass.color_attachments[0].store_action = StoreAction::Store;
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
    encoder.set_render_pipeline_state(state).unwrap();
    encoder.set_vertex_buffer(Some(&out), 0, 0).unwrap();
    match at1 {
        At1::Table(Some(table)) => encoder
            .set_vertex_visible_function_table(Some(table), 1)
            .unwrap(),
        At1::Table(None) => {}
        At1::Buffer(buffer) => encoder.set_vertex_buffer(Some(buffer), 0, 1).unwrap(),
    }
    encoder
        .draw_primitives(PrimitiveType::Point, 0, N, 1, 0)
        .unwrap();
    encoder.end_encoding().unwrap();
    after();
    run(&command_buffer);
    words(&out)
}

/// A buffer bound where a vertex function reads its stage's table reads as a null table, as a
/// kernel's does (`a_buffer_at_a_table_index_reads_as_a_null_table`) and as plain Metal reads it
/// where its validation stops the draw: every vertex writes 0xffffffff.
#[test]
fn a_buffer_at_a_vertex_table_index_reads_as_a_null_table() {
    on_device(|device| {
        let (_, functions) = libraries(device);
        let state = vertex_state(device, &functions);
        let filler = device
            .new_buffer_with_bytes(&[0xaa; 256], ResourceOptions::shared())
            .unwrap();
        assert_eq!(
            vertex_words_with(device, &state, At1::Buffer(&filler), || {}),
            vec![0xffff_ffff; N as usize]
        );
    });
}

/// A vertex function calls the functions its stage's table holds, as a kernel does and as
/// Metal's did (the M3 Max and the M4): vertex `vid` calls entry `vid % 5` of a vertex-stage table
/// holding `table_add`, `table_mul`, a null entry, `table_sub` and `table_half`. A call through an
/// entry of another type is undefined, so the words of entry 4 are not compared (Metal's were
/// `vid - 3`, the runtime's are 0). An empty table and an unbound one answer as plain Metal's did
/// (its debug layer stops both draws: "missing Visible Function Table binding at index 1 for
/// ops[0]"), and an entry set after the draw was encoded and before the commit is the one it calls,
/// as Metal's tables are read when the draw runs.
#[test]
fn a_vertex_function_calls_the_functions_its_stage_s_table_holds() {
    on_device(|device| {
        let (_, functions) = libraries(device);
        let function = |name: &str| functions.new_function_with_name(name).unwrap();
        let state = vertex_state(device, &functions);
        let vertex = RenderStages::VERTEX;
        let handle = |name: &str| {
            state
                .function_handle(&function(name), vertex)
                .unwrap()
                .unwrap()
        };
        let table = |count: u64| {
            state
                .new_visible_function_table(&VisibleFunctionTableDescriptor::new(count), vertex)
                .unwrap()
        };
        let five = table(5);
        let (add, mul, sub, half) = (
            handle("table_add"),
            handle("table_mul"),
            handle("table_sub"),
            handle("table_half"),
        );
        five.set_functions(&[Some(&add), Some(&mul)], 0).unwrap();
        five.set_function(Some(&sub), 3).unwrap();
        five.set_function(Some(&half), 4).unwrap();
        // Each vertex's word, `None` for the undefined call through entry 4.
        let expect = |vid: u32| match vid % 5 {
            0 => Some(vid + 3),
            1 => Some(vid * 3),
            2 => Some(0xffff_fffd),
            3 => Some(vid.wrapping_sub(3)),
            _ => None,
        };
        let defined = |words: Vec<u32>| -> Vec<Option<u32>> {
            (0..N)
                .zip(words)
                .map(|(vid, word)| (vid % 5 != 4).then_some(word))
                .collect()
        };
        let all: Vec<Option<u32>> = (0..N).map(expect).collect();
        assert_eq!(
            defined(vertex_words_of(device, &state, Some(&five), || {})),
            all
        );
        assert_eq!(
            vertex_words_of(device, &state, Some(&table(0)), || {}),
            vec![0xffff_fffe; N as usize]
        );
        assert_eq!(
            vertex_words_of(device, &state, None, || {}),
            vec![0xffff_ffff; N as usize]
        );
        let late = vertex_words_of(device, &state, Some(&five), || {
            five.set_function(Some(&add), 2).unwrap();
        });
        let with_add: Vec<Option<u32>> = (0..N)
            .map(|vid| {
                if vid % 5 == 2 {
                    Some(vid + 3)
                } else {
                    expect(vid)
                }
            })
            .collect();
        assert_eq!(defined(late), with_add, "the entry set before the commit");
    });
}

/// A buffer bound where the kernel reads a table reads as a null table, as plain Metal reads it
/// where its validation stops the dispatch ("incorrect Visible Function Table binding at index 1
/// ... found buffer"; an M3 Max and an M4): every thread writes 0xffffffff, whatever words the
/// buffer holds. A table where the kernel writes a buffer refuses the dispatch (plain Metal's
/// answer was not measured), recording nothing. A later binding of the other kind at the index
/// replaces the earlier one, as on Metal.
#[test]
fn a_buffer_at_a_table_index_reads_as_a_null_table() {
    on_device(|device| {
        let (caller, functions) = libraries(device);
        let state = dispatch_state(&caller, &functions);
        let one = table(&state, 1);
        let buffer = device.new_buffer(256, ResourceOptions::shared()).unwrap();
        let filler = device
            .new_buffer_with_bytes(&[0xaa; 256], ResourceOptions::shared())
            .unwrap();
        let queue = device.new_command_queue().unwrap();
        let grid = || (Size::new(N.into(), 1, 1), Size::new(N.into(), 1, 1));
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        encoder.set_compute_pipeline_state(&state).unwrap();
        encoder.set_buffer(Some(&buffer), 0, 0).unwrap();
        encoder.set_buffer(Some(&filler), 0, 1).unwrap();
        let (threads, group) = grid();
        encoder.dispatch_threads(threads, group).unwrap();
        encoder.end_encoding().unwrap();
        run(&command_buffer);
        assert_eq!(words(&buffer)[..N as usize], vec![0xffff_ffff; N as usize]);

        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        encoder.set_compute_pipeline_state(&state).unwrap();
        encoder.set_visible_function_table(Some(&one), 1).unwrap();
        encoder.set_visible_function_table(Some(&one), 0).unwrap();
        let (threads, group) = grid();
        assert_eq!(
            encoder.dispatch_threads(threads, group),
            Err(table_problem(
                "dispatch_threads",
                FunctionTableProblem::TableAtBufferIndex { index: 0 }
            ))
        );
        // An offset there makes the table read as the null table; the kernel's write of a
        // buffer at its index stays refused (not measured).
        encoder.set_buffer_offset(0, 0).unwrap();
        let (threads, group) = grid();
        assert_eq!(
            encoder.dispatch_threads(threads, group),
            Err(table_problem(
                "dispatch_threads",
                FunctionTableProblem::TableAtBufferIndex { index: 0 }
            ))
        );
        encoder.set_buffers(&[Some(&buffer)], &[0], 0).unwrap();
        let (threads, group) = grid();
        encoder.dispatch_threads(threads, group).unwrap();
        encoder.end_encoding().unwrap();
        run(&command_buffer);
        // The one entry is null: every thread wrote 0xfffffffd.
        assert_eq!(words(&buffer)[..N as usize], vec![0xffff_fffd; N as usize]);
    });
}

/// An offset set at an index holding a table makes the kernel read the null table there (size 0,
/// every entry null: every thread writes 0xffffffff), at any offset, 0 included, and a second
/// offset keeps it null, until the table is bound again, as plain Metal reads it (an M4 under
/// macOS 27.0.1; Metal's debug layer raises an exception at the offset).
#[test]
fn an_offset_at_a_table_index_reads_the_null_table() {
    on_device(|device| {
        let (caller, functions) = libraries(device);
        let state = dispatch_state(&caller, &functions);
        let one = table(&state, 1);
        let queue = device.new_command_queue().unwrap();
        let dispatched = |bind: &dyn Fn(&mut ComputeCommandEncoder)| {
            let buffer = device.new_buffer(256, ResourceOptions::shared()).unwrap();
            let command_buffer = queue.command_buffer().unwrap();
            let mut encoder = command_buffer.compute_command_encoder().unwrap();
            encoder.set_compute_pipeline_state(&state).unwrap();
            encoder.set_buffer(Some(&buffer), 0, 0).unwrap();
            encoder.set_visible_function_table(Some(&one), 1).unwrap();
            bind(&mut encoder);
            let size = Size::new(N.into(), 1, 1);
            encoder.dispatch_threads(size, size).unwrap();
            encoder.end_encoding().unwrap();
            run(&command_buffer);
            words(&buffer)[..N as usize].to_vec()
        };
        // The one entry is null: every thread writes 0xfffffffd; the null table, 0xffffffff.
        assert_eq!(dispatched(&|_| {}), vec![0xffff_fffd; N as usize]);
        for offset in [0, 8, 4096] {
            let words = dispatched(&|e| e.set_buffer_offset(offset, 1).unwrap());
            assert_eq!(words, vec![0xffff_ffff; N as usize], "offset {offset}");
        }
        let twice = dispatched(&|e| {
            e.set_buffer_offset(8, 1).unwrap();
            e.set_buffer_offset(0, 1).unwrap();
        });
        assert_eq!(twice, vec![0xffff_ffff; N as usize], "a second offset");
        let again = dispatched(&|e| {
            e.set_buffer_offset(8, 1).unwrap();
            e.set_visible_function_table(Some(&one), 1).unwrap();
        });
        assert_eq!(
            again,
            vec![0xffff_fffd; N as usize],
            "the table bound again"
        );
    });
}

/// The four functions of `functions`, linked.
fn linked_four(functions: &Library) -> LinkedFunctions {
    ["table_add", "table_mul", "table_sub", "table_half"]
        .iter()
        .fold(LinkedFunctions::new(), |linked, name| {
            linked.with_function(
                functions
                    .new_function_with_name(name)
                    .unwrap()
                    .linked_function(),
            )
        })
}

/// Run `state` over `N` threads, the output at buffer 0 and an argument buffer of `args` at
/// buffer 1, with `after` called once the dispatch is encoded and before the commit; the words
/// written.
fn words_with_arguments(
    device: &Device,
    state: &ComputePipelineState,
    args: &[u8],
    after: impl FnOnce(),
) -> Vec<u32> {
    let out = device
        .new_buffer(u64::from(N) * 4, ResourceOptions::shared())
        .unwrap();
    let args = device
        .new_buffer_with_bytes(args, ResourceOptions::shared())
        .unwrap();
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(state).unwrap();
    encoder.set_buffer(Some(&out), 0, 0).unwrap();
    encoder.set_buffer(Some(&args), 0, 1).unwrap();
    encoder
        .dispatch_threads(Size::new(N.into(), 1, 1), Size::new(N.into(), 1, 1))
        .unwrap();
    encoder.end_encoding().unwrap();
    after();
    run(&command_buffer);
    words(&out)
}

/// A kernel calls through the tables its argument buffer holds, by their `gpu_resource_id`s, as
/// it calls through a table bound at an index: the entry's function (zero for one of another
/// type), a null entry, an empty table, and the null table for id 0; per thread, through a member
/// array element and a select of two members. No `use_resource` is needed: every live table of
/// the state is readable.
#[test]
fn a_kernel_calls_through_the_tables_its_argument_buffer_holds() {
    on_device(|device| {
        argument_table_calls(device);
        an_entry_set_before_the_commit_is_called(device);
    });
}

/// [`a_kernel_calls_through_the_tables_its_argument_buffer_holds`]'s calls on `device`.
fn argument_table_calls(device: &Device) {
    let fixture = |stem: &str| llvm_assemble(&engine_dir().join(format!("{stem}.ll"))).unwrap();
    let caller = device
        .new_library_with_air(&fixture("uni_argument_tables"))
        .unwrap();
    let functions = device
        .new_library_with_air(&fixture("visible_table_functions"))
        .unwrap();
    // The functions in `table_add, table_mul, table_sub, table_half` order; `None` a null
    // entry.
    let t_entries = [Some(0), Some(1), None, Some(2), Some(3)];
    let u_entries = [Some(2), Some(0)];
    let ids = |list: &[u64]| -> Vec<u8> { list.iter().flat_map(|id| id.to_le_bytes()).collect() };
    for (kernel, k) in [
        ("argument_table_dispatch", 3u32),
        ("argument_table_array_dispatch", 5),
        ("argument_table_select", 7),
    ] {
        let state = state_of(&caller, kernel, linked_four(&functions));
        let handles: Vec<_> = ["table_add", "table_mul", "table_sub", "table_half"]
            .iter()
            .map(|name| {
                let f = functions.new_function_with_name(name).unwrap();
                state.function_handle(&f).unwrap().unwrap()
            })
            .collect();
        let filled = |entries: &[Option<usize>]| {
            let t = table(&state, entries.len() as u64);
            for (i, e) in entries.iter().enumerate() {
                t.set_function(e.map(|f| &handles[f]), i as u32).unwrap();
            }
            t
        };
        let (t, u, e) = (filled(&t_entries), filled(&u_entries), table(&state, 0));
        let (tid_, uid, eid) = (
            t.gpu_resource_id().0,
            u.gpu_resource_id().0,
            e.gpu_resource_id().0,
        );
        let expect = |id: u64, i: u32, tid: u32| -> u32 {
            let entries: &[Option<usize>] = if id == tid_ {
                &t_entries
            } else if id == uid {
                &u_entries
            } else if id == eid {
                &[]
            } else {
                return 0xffff_ffff;
            };
            if entries.is_empty() {
                return 0xffff_fffe;
            }
            match entries[i as usize % entries.len()] {
                None => 0xffff_fffd,
                Some(0) => tid.wrapping_add(k),
                Some(1) => tid.wrapping_mul(k),
                Some(2) => tid.wrapping_sub(k),
                Some(_) => 0,
            }
        };
        let cases: Vec<(Vec<u8>, Vec<u32>)> = match kernel {
            "argument_table_dispatch" => [tid_, uid, eid, 0]
                .iter()
                .map(|&id| (ids(&[id]), (0..N).map(|tid| expect(id, tid, tid)).collect()))
                .collect(),
            "argument_table_array_dispatch" => [[tid_, uid], [eid, 0]]
                .iter()
                .map(|pair| {
                    let mut args = vec![0; 8];
                    args.extend(ids(pair));
                    let want = (0..N)
                        .map(|tid| expect(pair[(tid & 1) as usize], tid >> 1, tid))
                        .collect();
                    (args, want)
                })
                .collect(),
            _ => [[uid, tid_], [0, eid]]
                .iter()
                .map(|&[a, b]| {
                    let want = (0..N)
                        .map(|tid| expect(if tid & 1 != 0 { a } else { b }, tid >> 1, tid))
                        .collect();
                    (ids(&[a, b]), want)
                })
                .collect(),
        };
        for (args, want) in cases {
            assert_eq!(
                words_with_arguments(device, &state, &args, || {}),
                want,
                "{kernel} {args:x?}"
            );
        }
    }
}

/// A table entry set after a dispatch that reads the table through its argument buffer was
/// encoded, and before the commit, is the one the dispatch calls, as Metal calls a bound
/// table's (measured on an M3 Max and an M4, `function_table.rs`).
fn an_entry_set_before_the_commit_is_called(device: &Device) {
    let fixture = |stem: &str| llvm_assemble(&engine_dir().join(format!("{stem}.ll"))).unwrap();
    let caller = device
        .new_library_with_air(&fixture("uni_argument_tables"))
        .unwrap();
    let functions = device
        .new_library_with_air(&fixture("visible_table_functions"))
        .unwrap();
    let state = state_of(&caller, "argument_table_dispatch", linked_four(&functions));
    let handle = |name: &str| {
        let f = functions.new_function_with_name(name).unwrap();
        state.function_handle(&f).unwrap().unwrap()
    };
    let (add, mul) = (handle("table_add"), handle("table_mul"));
    let t = table(&state, 2);
    t.set_functions(&[Some(&add), Some(&add)], 0).unwrap();
    let args = t.gpu_resource_id().0.to_le_bytes();
    let words = words_with_arguments(device, &state, &args, || {
        t.set_function(Some(&mul), 1).unwrap()
    });
    let want: Vec<u32> = (0..N)
        .map(|tid| match tid % 2 {
            0 => tid.wrapping_add(3),
            _ => tid.wrapping_mul(3),
        })
        .collect();
    assert_eq!(words, want);
}

/// A device that withholds `shaderStorageBufferArrayNonUniformIndexing`, one of the four
/// features the set-1 function-table array needs, compiles for none of it, and its kernels read
/// the tables their argument buffers name from the snapshot each command buffer writes as it
/// commits: the same answers as through the array, and an entry set after the dispatch was
/// encoded and before the commit is the one it calls.
#[test]
fn without_the_array_s_features_a_kernel_reads_its_tables_from_a_snapshot() {
    on_device_with(
        |options| {
            options
                .withheld_features
                .insert(VkFeature::ShaderStorageBufferArrayNonUniformIndexing);
        },
        |device| {
            let compile = device.compile_device();
            assert!(!compile.profile().descriptors().non_uniform_storage_buffer);
            argument_table_calls(device);
            an_entry_set_before_the_commit_is_called(device);
        },
    );
}

/// The entries of the wide tables of the tests below: their words, `4 (2^25 + 66)` bytes, pass
/// the 128-MiB `maxStorageBufferRange` of lavapipe and the Vulkan 1.2 floors.
const WIDE_ENTRIES: u64 = (1 << 25) + 64;

/// A table of `state` of [`WIDE_ENTRIES`] entries, the first `N` alternating `table_add` and
/// `table_mul` of `functions` and the rest null.
fn wide_table(state: &ComputePipelineState, functions: &Library) -> VisibleFunctionTable {
    let handle = |name: &str| {
        let f = functions.new_function_with_name(name).unwrap();
        state.function_handle(&f).unwrap().unwrap()
    };
    let (add, mul) = (handle("table_add"), handle("table_mul"));
    let t = table(state, WIDE_ENTRIES);
    let entries: Vec<_> = (0..N)
        .map(|i| Some(if i % 2 == 0 { &add } else { &mul }))
        .collect();
    t.set_functions(&entries, 0).unwrap();
    t
}

/// What each thread of [`wide_table`]'s callers writes: its entry `tid` called with `(tid, 3)`.
fn wide_words() -> Vec<u32> {
    (0..N)
        .map(|tid| match tid % 2 {
            0 => tid.wrapping_add(3),
            _ => tid.wrapping_mul(3),
        })
        .collect()
}

/// `uni_argument_tables`' `argument_table_dispatch` on `device`, linking the four functions,
/// and those functions' library.
fn argument_dispatch_state(device: &Device) -> (ComputePipelineState, Library) {
    let fixture = |stem: &str| llvm_assemble(&engine_dir().join(format!("{stem}.ll"))).unwrap();
    let (_, functions) = libraries(device);
    let caller = device
        .new_library_with_air(&fixture("uni_argument_tables"))
        .unwrap();
    let state = state_of(&caller, "argument_table_dispatch", linked_four(&functions));
    (state, functions)
}

/// Whether [`WIDE_ENTRIES`]' words pass `device`'s `maxStorageBufferRange`, where the plans of
/// the tests below ask for the variant that reads tables at their device addresses.
fn wide_passes_the_range(device: &Device) -> bool {
    let range = device
        .compile_device()
        .profile()
        .limits()
        .max_storage_buffer_range;
    4 * (WIDE_ENTRIES + 2) > u64::from(range)
}

/// A kernel calls the entries of a table of [`WIDE_ENTRIES`] entries through an argument buffer
/// naming it, as it calls a small table's: on a device whose `maxStorageBufferRange` the table's
/// words pass, through the variant that reads every listed table at its device address; elsewhere
/// bound whole as the function-table array's element.
#[test]
fn a_table_past_the_storage_range_is_called_through_its_address() {
    on_device(|device| {
        let (state, functions) = argument_dispatch_state(device);
        let t = wide_table(&state, &functions);
        let id = t.gpu_resource_id().0;
        assert_eq!(
            words_with_arguments(device, &state, &id.to_le_bytes(), || {}),
            wide_words()
        );
    });
}

/// Beside a table of [`WIDE_ENTRIES`] entries, an empty table's id reads the empty table, and
/// the null id and an id no table has read the null table, through the same variant where the
/// wide table's words pass the range: the directory's null entry, whose address is 0, reads
/// nothing.
#[test]
fn beside_a_table_past_the_storage_range_null_and_unknown_ids_read_the_null_table() {
    on_device(|device| {
        let (state, functions) = argument_dispatch_state(device);
        let _wide = wide_table(&state, &functions);
        let empty = table(&state, 0);
        for (id, want) in [
            (empty.gpu_resource_id().0, 0xffff_fffe),
            (0, 0xffff_ffff),
            ((1 << 40) | 7, 0xffff_ffff),
        ] {
            assert_eq!(
                words_with_arguments(device, &state, &id.to_le_bytes(), || {}),
                vec![want; N as usize],
                "id {id:#x}"
            );
        }
    });
}

/// A table of [`WIDE_ENTRIES`] entries bound at buffer 1 is called as a small one is: its words
/// are a buffer slot past the range where they pass it.
#[test]
fn a_table_past_the_storage_range_bound_at_an_index_is_called() {
    on_device(|device| {
        let (caller, functions) = libraries(device);
        let state = dispatch_state(&caller, &functions);
        let t = wide_table(&state, &functions);
        assert_eq!(words_of(device, &state, Some(&t), || {}), wide_words());
    });
}

/// On a device without the function-table array's features, whose snapshot of the listed tables
/// cannot hold a table whose words pass the range, the plan asks for the variant that reads the
/// tables at their addresses in place of the snapshot. Only where [`WIDE_ENTRIES`]' words pass
/// the range: elsewhere the snapshot holds them, copied whole at each bind and commit.
#[test]
fn without_the_array_s_features_a_table_past_the_storage_range_is_called_through_its_address() {
    on_device_with(
        |options| {
            options
                .withheld_features
                .insert(VkFeature::ShaderStorageBufferArrayNonUniformIndexing);
        },
        |device| {
            if !wide_passes_the_range(device) {
                eprintln!("SKIPPED: the device's maxStorageBufferRange holds the wide table");
                return;
            }
            let (state, functions) = argument_dispatch_state(device);
            let t = wide_table(&state, &functions);
            let id = t.gpu_resource_id().0;
            assert_eq!(
                words_with_arguments(device, &state, &id.to_le_bytes(), || {}),
                wide_words()
            );
        },
    );
}

/// What Metal answers for binary functions and required threadgroup sizes (the M3 Max and the M4
/// alike):
/// - A stage's `binary_functions` and a made state's additions must be compiled to a binary.
/// - Without the state's support flag, an added function's handle calls the kernel's last other
///   callable function.
/// - The added state has a handle for each of the base state's functions and each added one, and
///   calls both.
/// - A required size Metal refuses refuses the creation; a dispatch of another size runs the size
///   given, as plain Metal runs it.
/// - Every combination of a function descriptor's options is taken and answered back.
#[test]
fn binary_functions_and_required_threads_answer_as_metal_s_do() {
    use metal2vulkan::metal::FunctionOptions;
    use metal2vulkan_runtime::{BinaryFunctionProblem, FunctionDescriptor, RequiredThreadsProblem};
    on_device(|device| {
        let (caller, functions) = libraries(device);
        let with_options = |name: &str, options: FunctionOptions| {
            let mut descriptor = FunctionDescriptor::new(name);
            descriptor.options = options;
            functions.new_function_with_descriptor(&descriptor)
        };
        let compiled = |name: &str| with_options(name, FunctionOptions::COMPILE_TO_BINARY).unwrap();
        let plain = |name: &str| functions.new_function_with_name(name).unwrap();
        let kernel = caller
            .new_function_with_name("visible_table_dispatch")
            .unwrap();
        let descriptor = |linked: LinkedFunctions| {
            ComputePipelineDescriptor::new(StageSource::Function(
                kernel.function_descriptor().with_linked(linked),
            ))
        };
        let create = |descriptor: &ComputePipelineDescriptor| {
            device.new_compute_pipeline_state(descriptor, &PipelineOptions::default())
        };
        const CREATE: &str = "new_compute_pipeline_state";
        const ADD: &str = "new_compute_pipeline_state_with_additional_binary_functions";
        let binary_problem = |call, problem| invalid(call, Problem::BinaryFunctions(problem));
        let not_precompiled = |call, name: &str| {
            binary_problem(
                call,
                BinaryFunctionProblem::NotPrecompiled { name: name.into() },
            )
        };

        // The options a function descriptor takes: every combination of the four bits, each
        // answered back verbatim, as plain Metal answers it over a descriptor that lists no
        // binary archive (`library-facts-descriptor-*`).
        for raw in 0..16u32 {
            let options = FunctionOptions::try_from(raw).unwrap();
            let made = with_options("table_add", options).unwrap();
            assert_eq!(made.options(), options, "options {raw:#x}");
        }
        let stored = with_options(
            "table_add",
            FunctionOptions::COMPILE_TO_BINARY
                | FunctionOptions::STORE_FUNCTION_IN_METAL_PIPELINES_SCRIPT,
        )
        .unwrap();
        assert_eq!(
            stored.options(),
            FunctionOptions::COMPILE_TO_BINARY
                | FunctionOptions::STORE_FUNCTION_IN_METAL_PIPELINES_SCRIPT
        );

        // A stage's binary functions.
        let mut linked = LinkedFunctions::new();
        linked.binary_functions = vec![plain("table_add").linked_function()];
        assert_eq!(
            create(&descriptor(linked)).err(),
            Some(not_precompiled(CREATE, "table_add"))
        );

        // Functions added to a made state. Without the support flag, which Metal's validation
        // stops, plain Metal's handle of an added function calls the kernel's last other
        // callable function, also through a state made from that one.
        let base = LinkedFunctions::new().with_function(plain("table_mul").linked_function());
        let unsupported = create(&descriptor(base.clone()))
            .unwrap()
            .new_compute_pipeline_state_with_additional_binary_functions(&[
                compiled("table_add"),
                compiled("table_sub"),
            ])
            .unwrap();
        let handle = |name: &str| {
            unsupported
                .function_handle(&compiled(name))
                .unwrap()
                .expect("a handle for the added function")
        };
        let (add, sub) = (handle("table_add"), handle("table_sub"));
        let two = table(&unsupported, 2);
        two.set_functions(&[Some(&add), Some(&sub)], 0).unwrap();
        let calls_mul: Vec<u32> = (0..N).map(|tid| tid * 3).collect();
        assert_eq!(words_of(device, &unsupported, Some(&two), || {}), calls_mul);
        let again = unsupported
            .new_compute_pipeline_state_with_additional_binary_functions(&[compiled("table_half")])
            .unwrap();
        let again_handle = |name: &str| {
            again
                .function_handle(&compiled(name))
                .unwrap()
                .expect("a handle for an added function")
        };
        let two = table(&again, 2);
        two.set_functions(
            &[
                Some(&again_handle("table_add")),
                Some(&again_handle("table_half")),
            ],
            0,
        )
        .unwrap();
        assert_eq!(words_of(device, &again, Some(&two), || {}), calls_mul);
        let mut supported = descriptor(base);
        supported.support_adding_binary_functions = true;
        let state = create(&supported).unwrap();
        assert_eq!(
            state
                .new_compute_pipeline_state_with_additional_binary_functions(&[
                    compiled("table_add"),
                    plain("table_sub"),
                ])
                .err(),
            Some(not_precompiled(ADD, "table_sub"))
        );
        assert_eq!(state.function_handle(&compiled("table_add")).unwrap(), None);
        let added = state
            .new_compute_pipeline_state_with_additional_binary_functions(&[compiled("table_add")])
            .unwrap();
        let add = added
            .function_handle(&compiled("table_add"))
            .unwrap()
            .expect("a handle for the added function");
        let mul = added
            .function_handle(&plain("table_mul"))
            .unwrap()
            .expect("a handle for the base state's function");
        let two = table(&added, 2);
        two.set_functions(&[Some(&add), Some(&mul)], 0).unwrap();
        let expect: Vec<u32> = (0..N)
            .map(|tid| if tid % 2 == 0 { tid + 3 } else { tid * 3 })
            .collect();
        assert_eq!(words_of(device, &added, Some(&two), || {}), expect);
        added
            .new_compute_pipeline_state_with_additional_binary_functions(&[])
            .unwrap();

        // Required threads per threadgroup.
        let required = |axes: [u32; 3], max: Option<u32>| {
            let mut descriptor = descriptor(linked_four(&functions));
            descriptor.required_threads_per_threadgroup = Some(axes);
            descriptor.max_total_threads_per_threadgroup = max;
            create(&descriptor)
        };
        let threads_problem = |problem| invalid(CREATE, Problem::RequiredThreads(problem));
        assert_eq!(
            required([0, 1, 1], None).err(),
            Some(threads_problem(RequiredThreadsProblem::EmptyAxis {
                required: [0, 1, 1]
            }))
        );
        assert!(matches!(
            required([64, 32, 1], None).err(),
            Some(Error::Invalid(Invalid {
                problem: Problem::RequiredThreads(RequiredThreadsProblem::OverDeviceLimit {
                    threads: 2048,
                    ..
                }),
                ..
            }))
        ));
        assert_eq!(
            required([N, 1, 1], Some(32)).err(),
            Some(threads_problem(RequiredThreadsProblem::OverMaxTotal {
                threads: N.into(),
                max: 32
            }))
        );
        assert_eq!(
            required([N, 1, 1], Some(64)).err(),
            Some(threads_problem(RequiredThreadsProblem::MaxTotalDiffers {
                threads: N.into(),
                max: 64
            }))
        );
        let exact = required([N, 1, 1], Some(N)).unwrap();
        assert_eq!(
            exact.required_threads_per_threadgroup(),
            Size::new(N.into(), 1, 1)
        );
        let four = table(&exact, 4);
        assert_eq!(
            words_of(device, &exact, Some(&four), || {}).len(),
            N as usize
        );
        // A dispatch of another size runs the size given, as plain Metal runs it (Metal's API
        // validation stops it): as the state requiring that size runs it.
        let eight = required([8, 1, 1], None).unwrap();
        assert_eq!(
            words_of(device, &eight, Some(&table(&eight, 4)), || {}),
            words_of(device, &exact, Some(&four), || {})
        );
    });
}

/// A table bound where a kernel only reads a buffer reads as an unbound buffer, zeros, as plain
/// Metal read a table there (an M4), where its validation stops the dispatch ("incorrect Buffer
/// binding"). The kernel is the owned `a1_bool_from_float` fixture's: thread `i` converts the
/// float and half patterns of buffers 0 and 1 to `bool` and writes one word of the results to
/// buffer 2, 0 for zeros alone. The table's words (a null entry and a set one) would set bits.
#[test]
fn a_table_at_a_buffer_index_reads_as_an_unbound_buffer() {
    on_device(|device| {
        let (caller, functions) = libraries(device);
        let dispatch = dispatch_state(&caller, &functions);
        let add = functions.new_function_with_name("table_add").unwrap();
        let two = table(&dispatch, 2);
        two.set_function(dispatch.function_handle(&add).unwrap().as_ref(), 1)
            .unwrap();
        let bits = llvm_assemble(&engine_dir().join("a1_bool_from_float.ll")).unwrap();
        let library = device.new_library_with_air(&bits).unwrap();
        let state = state_of(&library, "a1_bool_from_float", LinkedFunctions::new());
        let ones = device
            .new_buffer_with_bytes(&[0x3f; 16 * N as usize], ResourceOptions::shared())
            .unwrap();
        let halves = device
            .new_buffer(8 * u64::from(N), ResourceOptions::shared())
            .unwrap();
        let queue = device.new_command_queue().unwrap();
        let run_with = |first: Option<&Buffer>| {
            let out = device
                .new_buffer_with_bytes(&[0xaa; 4 * N as usize], ResourceOptions::shared())
                .unwrap();
            let command_buffer = queue.command_buffer().unwrap();
            let mut encoder = command_buffer.compute_command_encoder().unwrap();
            encoder.set_compute_pipeline_state(&state).unwrap();
            match first {
                Some(buffer) => encoder.set_buffer(Some(buffer), 0, 0).unwrap(),
                None => encoder.set_visible_function_table(Some(&two), 0).unwrap(),
            }
            encoder.set_buffer(Some(&halves), 0, 1).unwrap();
            encoder.set_buffer(Some(&out), 0, 2).unwrap();
            encoder
                .dispatch_threads(Size::new(N.into(), 1, 1), Size::new(N.into(), 1, 1))
                .unwrap();
            encoder.end_encoding().unwrap();
            run(&command_buffer);
            words(&out)
        };
        // Nonzero floats set the scalar and vector bits of every lane.
        assert_eq!(run_with(Some(&ones)), vec![0xff; N as usize]);
        assert_eq!(run_with(None), vec![0; N as usize]);
    });
}

/// A handle of another drawing state's vertex function calls the table's state's vertex function
/// at the handle's position, numbered private functions first, as plain Metal's table does where
/// its validation stops `setFunction:atIndex:` (an M4). The other state links `table_mul` private,
/// then `table_sub` and `table_add`: its `table_add` is at position 2, where the table's state
/// links `table_sub`, so every vertex writes `vid - 3`. A kernel's handle stays refused
/// (unmeasured).
#[test]
fn a_handle_of_another_vertex_function_calls_this_state_s_function_at_its_position() {
    on_device(|device| {
        let (caller, functions) = libraries(device);
        let function = |name: &str| functions.new_function_with_name(name).unwrap();
        let state = vertex_state(device, &functions);
        let other = vertex_state_linking(
            device,
            LinkedFunctions::new()
                .with_private_function(function("table_mul").linked_function())
                .with_function(function("table_sub").linked_function())
                .with_function(function("table_add").linked_function()),
        );
        let own = state
            .new_visible_function_table(
                &VisibleFunctionTableDescriptor::new(1),
                RenderStages::VERTEX,
            )
            .unwrap();
        let add = other
            .function_handle(&function("table_add"), RenderStages::VERTEX)
            .unwrap();
        own.set_function(add.as_ref(), 0).unwrap();
        assert_eq!(
            vertex_words_of(device, &state, Some(&own), || {}),
            (0..N).map(|vid| vid.wrapping_sub(3)).collect::<Vec<_>>()
        );
        let kernel = dispatch_state(&caller, &functions)
            .function_handle(&function("table_add"))
            .unwrap();
        assert_eq!(
            own.set_function(kernel.as_ref(), 0).unwrap_err(),
            table_problem("set_function", FunctionTableProblem::HandleOfAnotherState)
        );
    });
}

/// Without `support_adding_vertex_binary_functions`, which Metal's validation stops, plain Metal
/// makes the state with functions added to its vertex stage, and its handle of an added function
/// calls the stage's last other callable function, as a kernel's does (and alike `frag_off_two`,
/// `tile_off_two`, an M4). The base links `table_mul`, then `table_add`: `table_sub`'s handle calls
/// `table_add`, so every vertex writes `vid + 3`, and so do `table_sub`'s and `table_half`'s
/// through a state made from that one; with the flag `table_sub`'s calls `table_sub`.
#[test]
fn functions_added_to_a_vertex_stage_without_its_flag_call_its_last_function() {
    use metal2vulkan::metal::FunctionOptions;
    use metal2vulkan_runtime::{FunctionDescriptor, RenderPipelineFunctionsDescriptor};
    on_device(|device| {
        let (_, functions) = libraries(device);
        let plain = |name: &str| functions.new_function_with_name(name).unwrap();
        let compiled = |name: &str| {
            let mut descriptor = FunctionDescriptor::new(name);
            descriptor.options = FunctionOptions::COMPILE_TO_BINARY;
            functions.new_function_with_descriptor(&descriptor).unwrap()
        };
        let base = || {
            LinkedFunctions::new()
                .with_function(plain("table_mul").linked_function())
                .with_function(plain("table_add").linked_function())
        };
        let adding = |state: &RenderPipelineState, name: &str| {
            let mut added = RenderPipelineFunctionsDescriptor::new();
            added.vertex_additional_binary_functions = vec![compiled(name)];
            state
                .new_render_pipeline_state_with_additional_binary_functions(&added)
                .unwrap()
        };
        let calls = |state: &RenderPipelineState, name: &str| {
            let handle = state
                .function_handle(&compiled(name), RenderStages::VERTEX)
                .unwrap()
                .expect("a handle for the added function");
            let table = state
                .new_visible_function_table(
                    &VisibleFunctionTableDescriptor::new(1),
                    RenderStages::VERTEX,
                )
                .unwrap();
            table.set_function(Some(&handle), 0).unwrap();
            vertex_words_of(device, state, Some(&table), || {})
        };
        let calls_add: Vec<u32> = (0..N).map(|vid| vid + 3).collect();
        let unsupported = adding(&vertex_state_adding(device, base(), false), "table_sub");
        assert_eq!(calls(&unsupported, "table_sub"), calls_add);
        let again = adding(&unsupported, "table_half");
        assert_eq!(calls(&again, "table_sub"), calls_add);
        assert_eq!(calls(&again, "table_half"), calls_add);
        let supported = adding(&vertex_state_adding(device, base(), true), "table_sub");
        assert_eq!(
            calls(&supported, "table_sub"),
            (0..N).map(|vid| vid.wrapping_sub(3)).collect::<Vec<_>>()
        );
    });
}

/// Functions added to a stage a drawing state lacks (the fragment stage of a state without a
/// fragment function, and the tile stage) are left out, with no handle, as plain Metal makes the
/// state where its validation stops it; the state's vertex function keeps its handle. A function
/// not compiled to a binary stays refused there, as it is for a stage the state has.
#[test]
fn functions_added_to_a_stage_a_drawing_state_lacks_are_left_out() {
    use metal2vulkan::metal::FunctionOptions;
    use metal2vulkan_runtime::{
        BinaryFunctionProblem, FunctionDescriptor, RenderPipelineFunctionsDescriptor,
    };
    on_device(|device| {
        let (_, functions) = libraries(device);
        let plain = |name: &str| functions.new_function_with_name(name).unwrap();
        let compiled = |name: &str| {
            let mut descriptor = FunctionDescriptor::new(name);
            descriptor.options = FunctionOptions::COMPILE_TO_BINARY;
            functions.new_function_with_descriptor(&descriptor).unwrap()
        };
        let state = vertex_state_linking(
            device,
            LinkedFunctions::new().with_function(plain("table_add").linked_function()),
        );
        let mut added = RenderPipelineFunctionsDescriptor::new();
        added.fragment_additional_binary_functions = vec![compiled("table_mul")];
        added.tile_additional_binary_functions = vec![compiled("table_sub")];
        let made = state
            .new_render_pipeline_state_with_additional_binary_functions(&added)
            .unwrap();
        for (name, stage) in [
            ("table_mul", RenderStages::FRAGMENT),
            ("table_mul", RenderStages::VERTEX),
            ("table_sub", RenderStages::TILE),
            ("table_sub", RenderStages::VERTEX),
        ] {
            assert_eq!(
                made.function_handle(&compiled(name), stage).unwrap(),
                None,
                "{name} {stage:?}"
            );
        }
        assert!(made
            .function_handle(&plain("table_add"), RenderStages::VERTEX)
            .unwrap()
            .is_some());
        added.tile_additional_binary_functions = vec![plain("table_sub")];
        assert_eq!(
            state
                .new_render_pipeline_state_with_additional_binary_functions(&added)
                .unwrap_err(),
            invalid(
                "new_render_pipeline_state_with_additional_binary_functions",
                Problem::BinaryFunctions(BinaryFunctionProblem::NotPrecompiled {
                    name: "table_sub".into()
                })
            )
        );
    });
}

/// The words a tile dispatch writes: two 16x16 tiles of 8x8 threads, a square threadgroup of a
/// measured layout, whose `thread_index_in_threadgroup` runs over its 2x2 quads.
const TILE_WORDS: u32 = 128;

/// The state of the owned `vft_tile_caller` fixture's tile function, in a tile pipeline of one
/// `R32Uint` attachment, its tile stage linking `linked`, made with
/// `support_adding_binary_functions` set to `supported`.
fn tile_state_adding(
    device: &Device,
    linked: LinkedFunctions,
    supported: bool,
) -> RenderPipelineState {
    let bitcode = llvm_assemble(&engine_dir().join("vft_tile_caller.ll")).unwrap();
    let caller = device.new_library_with_air(&bitcode).unwrap();
    let tile = caller.new_function_with_name("vft_tile_dispatch").unwrap();
    let source = tile.function_descriptor().with_linked(linked);
    let mut descriptor = TileRenderPipelineDescriptor::new(StageSource::Function(source));
    descriptor.color_attachment_formats[0] = Some(PixelFormat::R32Uint);
    descriptor.support_adding_binary_functions = supported;
    device
        .new_tile_render_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap()
}

/// Dispatch `state` in a pass of two 16x16 tiles (a 32x16 `R32Uint` target), 8x8 threads a
/// tile, the output at tile buffer 0 and `at1` at tile buffer 1, with `after` called once the
/// pass is encoded and before the commit; the words written.
fn tile_words_with(
    device: &Device,
    state: &RenderPipelineState,
    at1: At1<'_>,
    after: impl FnOnce(),
) -> Vec<u32> {
    let out = device
        .new_buffer(u64::from(TILE_WORDS) * 4, ResourceOptions::shared())
        .unwrap();
    let mut target = TextureDescriptor::texture_2d(PixelFormat::R32Uint, 32, 16, false);
    target.usage = TextureUsage::RENDER_TARGET;
    let mut pass = RenderPassDescriptor::new();
    pass.color_attachments[0].texture = Some(device.new_texture(&target).unwrap());
    pass.color_attachments[0].load_action = LoadAction::Clear;
    pass.color_attachments[0].store_action = StoreAction::DontCare;
    pass.tile_width = 16;
    pass.tile_height = 16;
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
    encoder.set_render_pipeline_state(state).unwrap();
    encoder.set_tile_buffer(Some(&out), 0, 0).unwrap();
    match at1 {
        At1::Table(Some(table)) => encoder
            .set_tile_visible_function_table(Some(table), 1)
            .unwrap(),
        At1::Table(None) => {}
        At1::Buffer(buffer) => encoder.set_tile_buffer(Some(buffer), 0, 1).unwrap(),
    }
    encoder
        .dispatch_threads_per_tile(Size::new(8, 8, 1))
        .unwrap();
    encoder.end_encoding().unwrap();
    after();
    run(&command_buffer);
    words(&out)
}

/// A tile function calls the functions its stage's table holds, as a kernel and a vertex
/// function do and as Metal's did (the M3 Max and an M4): thread `i` of the owned
/// `vft_tile_caller`'s two tiles calls entry `i % 5` of a tile-stage table holding `table_add`,
/// `table_mul`, a null entry, `table_sub` and `table_half`. A call through an entry of another type
/// is undefined, so the words of entry 4 are not compared. An entry set after the dispatch was
/// encoded and before the commit is the one it calls, as Metal's tables are read when the dispatch
/// runs. An empty table, an unbound one, and a buffer bound at the table's index answer as plain
/// Metal's did where its debug layer stops the dispatch: 0xfffffffe, and 0xffffffff for both
/// others.
#[test]
fn a_tile_function_calls_the_functions_its_stage_s_table_holds() {
    on_device(|device| {
        let (_, functions) = libraries(device);
        let function = |name: &str| functions.new_function_with_name(name).unwrap();
        let state = tile_state_adding(device, linked_four(&functions), false);
        let tile = RenderStages::TILE;
        let handle = |name: &str| {
            state
                .function_handle(&function(name), tile)
                .unwrap()
                .unwrap()
        };
        let table = |count: u64| {
            state
                .new_visible_function_table(&VisibleFunctionTableDescriptor::new(count), tile)
                .unwrap()
        };
        let five = table(5);
        let (add, mul, sub, half) = (
            handle("table_add"),
            handle("table_mul"),
            handle("table_sub"),
            handle("table_half"),
        );
        five.set_functions(&[Some(&add), Some(&mul)], 0).unwrap();
        five.set_function(Some(&sub), 3).unwrap();
        five.set_function(Some(&half), 4).unwrap();
        // Each thread's word, `None` for the undefined call through entry 4.
        let expect = |i: u32| match i % 5 {
            0 => Some(i + 3),
            1 => Some(i * 3),
            2 => Some(0xffff_fffd),
            3 => Some(i.wrapping_sub(3)),
            _ => None,
        };
        let defined = |words: Vec<u32>| -> Vec<Option<u32>> {
            (0..TILE_WORDS)
                .zip(words)
                .map(|(i, word)| (i % 5 != 4).then_some(word))
                .collect()
        };
        let words = |at1: At1<'_>| tile_words_with(device, &state, at1, || {});
        let all: Vec<Option<u32>> = (0..TILE_WORDS).map(expect).collect();
        assert_eq!(defined(words(At1::Table(Some(&five)))), all);
        let late = tile_words_with(device, &state, At1::Table(Some(&five)), || {
            five.set_function(Some(&add), 2).unwrap();
        });
        let with_add: Vec<Option<u32>> = (0..TILE_WORDS)
            .map(|i| if i % 5 == 2 { Some(i + 3) } else { expect(i) })
            .collect();
        assert_eq!(defined(late), with_add, "the entry set before the commit");
        let empty = table(0);
        assert_eq!(
            words(At1::Table(Some(&empty))),
            vec![0xffff_fffe; TILE_WORDS as usize]
        );
        assert_eq!(
            words(At1::Table(None)),
            vec![0xffff_ffff; TILE_WORDS as usize]
        );
        let filler = device
            .new_buffer_with_bytes(&[0xaa; 256], ResourceOptions::shared())
            .unwrap();
        assert_eq!(
            words(At1::Buffer(&filler)),
            vec![0xffff_ffff; TILE_WORDS as usize]
        );
    });
}

/// Without `support_adding_binary_functions`, which Metal's validation stops, plain Metal makes
/// a tile state with functions added to its tile stage, and its handle of an added function
/// calls the stage's last other callable function, as a vertex function's does (an M4;
/// `functions_added_to_a_vertex_stage_without_its_flag_call_its_last_function`). The base links
/// `table_mul`, then `table_add`: `table_sub`'s handle calls `table_add`, so every thread writes
/// `i + 3`, and so do `table_sub`'s and `table_half`'s through a state made from that one; with
/// the flag `table_sub`'s calls `table_sub` (`tile_on_two`).
#[test]
fn functions_added_to_a_tile_stage_without_its_flag_call_its_last_function() {
    use metal2vulkan::metal::FunctionOptions;
    use metal2vulkan_runtime::{FunctionDescriptor, RenderPipelineFunctionsDescriptor};
    on_device(|device| {
        let (_, functions) = libraries(device);
        let plain = |name: &str| functions.new_function_with_name(name).unwrap();
        let compiled = |name: &str| {
            let mut descriptor = FunctionDescriptor::new(name);
            descriptor.options = FunctionOptions::COMPILE_TO_BINARY;
            functions.new_function_with_descriptor(&descriptor).unwrap()
        };
        let base = || {
            LinkedFunctions::new()
                .with_function(plain("table_mul").linked_function())
                .with_function(plain("table_add").linked_function())
        };
        let adding = |state: &RenderPipelineState, name: &str| {
            let mut added = RenderPipelineFunctionsDescriptor::new();
            added.tile_additional_binary_functions = vec![compiled(name)];
            state
                .new_render_pipeline_state_with_additional_binary_functions(&added)
                .unwrap()
        };
        let calls = |state: &RenderPipelineState, name: &str| {
            let handle = state
                .function_handle(&compiled(name), RenderStages::TILE)
                .unwrap()
                .expect("a handle for the added function");
            let table = state
                .new_visible_function_table(
                    &VisibleFunctionTableDescriptor::new(1),
                    RenderStages::TILE,
                )
                .unwrap();
            table.set_function(Some(&handle), 0).unwrap();
            tile_words_with(device, state, At1::Table(Some(&table)), || {})
        };
        let calls_add: Vec<u32> = (0..TILE_WORDS).map(|i| i + 3).collect();
        let unsupported = adding(&tile_state_adding(device, base(), false), "table_sub");
        assert_eq!(calls(&unsupported, "table_sub"), calls_add);
        let again = adding(&unsupported, "table_half");
        assert_eq!(calls(&again, "table_sub"), calls_add);
        assert_eq!(calls(&again, "table_half"), calls_add);
        let supported = adding(&tile_state_adding(device, base(), true), "table_sub");
        assert_eq!(
            calls(&supported, "table_sub"),
            (0..TILE_WORDS)
                .map(|i| i.wrapping_sub(3))
                .collect::<Vec<_>>()
        );
    });
}
