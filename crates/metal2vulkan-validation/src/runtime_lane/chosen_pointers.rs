//! Function pointers chosen at run time on the device the cfg selects: a kernel calls through a
//! pointer among functions' addresses as Metal's does (an M4 on macOS 27.0.1, and an M3 Max on
//! 26.7, which ran the owned fixtures' own AIR).
//!
//! The kernels are the owned `fpc_chosen_pointers` fixture's. Each writes word `tid` of buffer 0
//! over 40 threads in one threadgroup: through a select of two addresses, a phi of three, a
//! pointer carried around a loop, a private array, comparisons, a threadgroup array, a 64-bit
//! integer written to device memory and read back by the neighbouring thread, a table's entry
//! merged with an address and stored beside one, and recursion through a pointer each function
//! chooses. The functions are `fpc_pointer_functions`' `fpc_inc`, `fpc_dbl` and `fpc_sq` of
//! `uint(uint)` and the recursive `fpc_rec_even` and `fpc_rec_odd`, which the pipeline links;
//! `fpc_own_*` choose among the caller's own functions, which are not visible, and link nothing.

use super::at_plan;
use super::host_memory::on_device_with;
use super::host_memory::{on_device, run};
use super::read_buffer;
use crate::air_library::AirLibrary;
use metal2vulkan::error::{Error as CompileError, RuntimeInput};
use metal2vulkan::pipeline::{ComputePipelineDescriptor, LinkedFunctions, StageSource};
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use metal2vulkan_runtime::{
    Buffer, ComputePipelineState, Device, Error, Library, PipelineOptions, ResourceOptions, Size,
    VisibleFunctionTable, VisibleFunctionTableDescriptor,
};

/// Threads of each dispatch, all in one threadgroup.
const N: u32 = 40;

/// The functions every pipeline but the `fpc_own_*` kernels' links.
const LINKED: [&str; 5] = [
    "fpc_inc",
    "fpc_dbl",
    "fpc_sq",
    "fpc_rec_even",
    "fpc_rec_odd",
];

/// The owned fixtures' libraries: the caller's and the functions'.
fn libraries(device: &Device) -> (Library, Library) {
    let fixture = |stem: &str| llvm_assemble(&engine_dir().join(format!("{stem}.ll"))).unwrap();
    (
        device
            .new_library_with_air(&fixture("fpc_chosen_pointers"))
            .unwrap(),
        device
            .new_library_with_air(&fixture("fpc_pointer_functions"))
            .unwrap(),
    )
}

/// The state of `kernel` of `caller` at `maxCallStackDepth` `depth`, linking `names` of
/// `functions`.
fn state_of(
    (caller, functions): &(Library, Library),
    kernel: &str,
    names: &[&str],
    depth: u32,
) -> Result<ComputePipelineState, Error> {
    let linked = names.iter().fold(LinkedFunctions::new(), |linked, name| {
        linked.with_function(
            functions
                .new_function_with_name(name)
                .unwrap()
                .linked_function(),
        )
    });
    let kernel = caller.new_function_with_name(kernel).unwrap();
    let mut descriptor = ComputePipelineDescriptor::new(StageSource::Function(
        kernel.function_descriptor().with_linked(linked),
    ));
    descriptor.max_call_stack_depth = depth;
    caller
        .device()
        .new_compute_pipeline_state(&descriptor, &PipelineOptions::default())
}

/// Run `state` over [`N`] threads in one threadgroup, `buffers` bound at their indices and
/// `table` at buffer 1.
fn dispatch(
    device: &Device,
    state: &ComputePipelineState,
    buffers: &[(u32, &Buffer)],
    table: Option<&VisibleFunctionTable>,
) {
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(state).unwrap();
    for &(index, buffer) in buffers {
        encoder.set_buffer(Some(buffer), 0, index).unwrap();
    }
    if table.is_some() {
        encoder.set_visible_function_tables(&[table], 1).unwrap();
    }
    encoder
        .dispatch_threads(Size::new(N.into(), 1, 1), Size::new(N.into(), 1, 1))
        .unwrap();
    encoder.end_encoding().unwrap();
    run(&command_buffer);
}

fn output(device: &Device) -> Buffer {
    device
        .new_buffer(u64::from(N) * 8, ResourceOptions::shared())
        .unwrap()
}

fn words(buffer: &Buffer, count: u32) -> Vec<u32> {
    read_buffer(buffer, 0, u64::from(count) * 4)
        .unwrap()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|w| u32::from_le_bytes(*w))
        .collect()
}

fn inc(x: u32) -> u32 {
    x.wrapping_add(1)
}

fn dbl(x: u32) -> u32 {
    x.wrapping_mul(2)
}

fn sq(x: u32) -> u32 {
    x.wrapping_mul(x)
}

const OPS: [fn(u32) -> u32; 3] = [inc, dbl, sq];

/// `fpc_compare`'s and `fpc_own_compare`'s word: of `[inc, dbl, sq, inc]`, `f` at `tid % 4` and
/// `g` at `(tid / 4) % 4`; bit 0 `f == g`, bit 1 `f` is `inc`, bit 2 `f` is not null, bit 3 `g`
/// is `sq`.
fn compared(tid: u32) -> u32 {
    let of = [0, 1, 2, 0];
    let (f, g) = (of[(tid % 4) as usize], of[((tid / 4) % 4) as usize]);
    4 | u32::from(f == g) | (u32::from(f == 0) << 1) | (u32::from(g == 2) << 3)
}

/// `fpc_loop`'s word: `inc` first, then `inc` after an odd result and `dbl` after an even one.
fn looped(tid: u32) -> u32 {
    let (mut f, mut acc) = (0, tid);
    for _ in 0..4 {
        acc = OPS[f](acc);
        f = if acc & 1 != 0 { 0 } else { 1 };
    }
    acc
}

/// A kernel that reads only `tid`, with the word Metal answers for it.
type Shape = (&'static str, fn(u32) -> u32);

/// Each kernel that reads only `tid`.
const SHAPES: [Shape; 9] = [
    ("fpc_select", |t| if t & 1 != 0 { inc(t) } else { dbl(t) }),
    ("fpc_phi", |t| OPS[(t % 3) as usize](t + 5)),
    ("fpc_loop", looped),
    ("fpc_private_array", |t| OPS[(t % 3) as usize](t)),
    ("fpc_compare", compared),
    (
        "fpc_threadgroup",
        |t| if t & 1 != 0 { inc(t) } else { sq(t) },
    ),
    (
        "fpc_own_select",
        |t| if t & 1 != 0 { inc(t) } else { dbl(t) },
    ),
    ("fpc_own_private_array", |t| OPS[(t % 3) as usize](t)),
    ("fpc_own_compare", compared),
];

/// A select, a phi, a loop, a private array, comparisons and a threadgroup array of functions'
/// addresses call and compare as Metal's do, linked functions' and the caller's own alike; the
/// phi's branches write their side words.
#[test]
fn pointers_chosen_at_run_time_call_the_functions_they_hold() {
    on_device_with(at_plan, |device| {
        let libraries = libraries(device);
        for (kernel, want) in SHAPES {
            let names: &[&str] = if kernel.starts_with("fpc_own_") {
                &[]
            } else {
                &LINKED
            };
            let state = state_of(&libraries, kernel, names, 1).unwrap();
            let (out, side) = (output(device), output(device));
            dispatch(device, &state, &[(0, &out), (1, &side)], None);
            let all: Vec<u32> = (0..N).map(want).collect();
            assert_eq!(words(&out, N), all, "{kernel}");
            if kernel == "fpc_phi" {
                let sides: Vec<u32> = (0..N).map(|t| 10 + t % 3).collect();
                assert_eq!(words(&side, N), sides, "{kernel} side");
            }
        }
    });
}

/// A pointer converted to a 64-bit integer, written to device memory, read back by the
/// neighbouring thread and converted back calls its function. The bits are each function's own,
/// the same in every thread; Metal's are its pipeline's and vary between pipelines,
/// so only that is compared.
#[test]
fn a_pointer_through_an_integer_calls_its_function() {
    on_device(|device| {
        let libraries = libraries(device);
        let state = state_of(&libraries, "fpc_int_back", &LINKED, 1).unwrap();
        let (out, bits) = (output(device), output(device));
        dispatch(device, &state, &[(0, &out), (1, &bits)], None);
        let all: Vec<u32> = (0..N)
            .map(|t| if t & 1 != 0 { dbl(t) } else { inc(t) })
            .collect();
        assert_eq!(words(&out, N), all);
        let bits: Vec<u64> = read_buffer(&bits, 0, u64::from(N) * 8)
            .unwrap()
            .as_chunks::<8>()
            .0
            .iter()
            .map(|w| u64::from_le_bytes(*w))
            .collect();
        assert!(bits.iter().step_by(2).all(|&b| b == bits[0]), "{bits:x?}");
        assert!(
            bits.iter().skip(1).step_by(2).all(|&b| b == bits[1]),
            "{bits:x?}"
        );
        assert_ne!(bits[0], bits[1]);
        assert!(bits[0] != 0 && bits[1] != 0, "{bits:x?}");
    });
}

/// A table's entry merged with a function's address, compared with one, and stored beside one
/// calls and compares as Metal's does: an entry holding `fpc_inc`'s handle equals `fpc_inc`'s
/// address, and one holding `fpc_sq`'s equals `fpc_sq`'s.
#[test]
fn table_entries_and_addresses_call_and_compare_alike() {
    on_device_with(at_plan, |device| {
        let libraries = libraries(device);
        for kernel in ["fpc_table_mix", "fpc_table_store"] {
            let state = state_of(&libraries, kernel, &LINKED, 1).unwrap();
            let handle = |name: &str| {
                let function = libraries.1.new_function_with_name(name).unwrap();
                state.function_handle(&function).unwrap().unwrap()
            };
            let table = state
                .new_visible_function_table(&VisibleFunctionTableDescriptor::new(3))
                .unwrap();
            let (i, d, s) = (handle("fpc_inc"), handle("fpc_dbl"), handle("fpc_sq"));
            table
                .set_functions(&[Some(&i), Some(&d), Some(&s)], 0)
                .unwrap();
            let out = output(device);
            dispatch(device, &state, &[(0, &out)], Some(&table));
            let all: Vec<u32> = (0..N)
                .map(|t| {
                    let e = (t % 3) as usize;
                    if kernel == "fpc_table_mix" {
                        let f = if t & 4 != 0 { e } else { 2 };
                        OPS[f](t) + if e == 0 { 1000 } else { 0 } + if f == 2 { 2000 } else { 0 }
                    } else {
                        OPS[if (t >> 1) & 1 != 0 { 2 } else { e }](t)
                    }
                })
                .collect();
            assert_eq!(words(&out, N), all, "{kernel}");
        }
    });
}

/// `fpc_rec_even(n)` (`odd` false) or `fpc_rec_odd(n)`: Metal's answer when the recursion fits.
fn recursed(odd: bool, n: u32) -> u32 {
    if n == 0 {
        return 0;
    }
    let next = if n & 2 != 0 { !odd } else { odd };
    (if odd { 2 * n } else { n }).wrapping_add(recursed(next, n - 1))
}

/// Recursion through pointers each function chooses between the two reaches depth `2k - 1` at
/// `maxCallStackDepth` `k`, as plain Metal's does: thread `t` runs depth
/// `t / 2`, up to that reach, through `fpc_rec_odd` for odd `t`.
#[test]
fn recursion_through_chosen_pointers_reaches_metal_s_depth() {
    on_device_with(at_plan, |device| {
        let libraries = libraries(device);
        for k in [1, 2, 4] {
            let state = state_of(&libraries, "fpc_recurse", &LINKED[3..], k).unwrap();
            let depths: Vec<u32> = (0..N).map(|t| (t / 2).min(2 * k - 1)).collect();
            let bytes: Vec<u8> = depths.iter().flat_map(|d| d.to_le_bytes()).collect();
            let depth = device
                .new_buffer_with_bytes(&bytes, ResourceOptions::shared())
                .unwrap();
            let out = output(device);
            dispatch(device, &state, &[(0, &out), (2, &depth)], None);
            let all: Vec<u32> = (0..N)
                .map(|t| recursed(t & 1 != 0, depths[t as usize]))
                .collect();
            assert_eq!(words(&out, N), all, "k {k}");
        }
    });
}

/// A visible function whose address a kernel takes and the pipeline does not link refuses the
/// state, as Metal refuses it ("unresolved visible function reference").
#[test]
fn an_unlinked_function_s_address_refuses_the_state() {
    on_device(|device| {
        let libraries = libraries(device);
        for kernel in ["fpc_select", "fpc_private_array", "fpc_threadgroup"] {
            match state_of(&libraries, kernel, &["fpc_inc"], 1) {
                Err(Error::Compile(CompileError::NeedsInput(RuntimeInput::LinkedFunction {
                    ..
                }))) => {}
                Err(other) => panic!("{kernel}: {other}"),
                Ok(_) => panic!("{kernel}: made"),
            }
        }
    });
}
