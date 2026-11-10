//! A recursion's frames on the device the cfg selects: the owned fixture `fpc_frames`' recursive
//! visible functions, linked into their kernels' pipelines, answer every depth Metal answers
//! correctly at `maxCallStackDepth` 1, 2 and 4 (on an M4 under macOS 27.0.1 ran the
//! fixture's own AIR).
//!
//! Each kernel `frames_*` calls its function at the depth word `tid` of buffer 2 holds, with the
//! mode word 0 of buffer 3 holds, and writes the result to word `tid` of buffer 0. Thread `t` runs
//! depth `t`, up to Metal's deepest correct depth: `k - 1` for a lone function, whose depth 0
//! answers zero (8 or 32 values live across the call) or a nonzero word (nested and
//! one-dimensional arrays), and further for the pairs of unequal frames.
//!
//! A stack past the device's Function storage is refused `Unsupported{Recursion}` needing more. The
//! lowering's stack holds a pair's frames uniformly, each sized for the larger, as many as `k L`
//! bytes hold of the smaller, so `frames_t` at `k` 4 (two stacks of 72 frames of 268 bytes, 38592
//! bytes) is refused on a device of 32 KiB a thread, where Metal runs it. The test accepts that
//! refusal past `k` 1 and prints it.

use super::at_plan;
use super::host_memory::on_device_with;
use super::host_memory::run;
use super::read_buffer;
use crate::air_library::AirLibrary;
use metal2vulkan::device::VkLimit;
use metal2vulkan::error::{Capability, Error as CompileError, Feature, Needs, Unsupported};
use metal2vulkan::pipeline::{ComputePipelineDescriptor, LinkedFunctions, StageSource};
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use metal2vulkan_runtime::{
    Buffer, ComputePipelineState, Device, Error, Library, PipelineOptions, ResourceOptions, Size,
};

/// Threads of each dispatch, all in one threadgroup.
const N: u32 = 32;

/// A function of the fixture, as its MSL computes it.
#[derive(Clone, Copy, Debug)]
enum Fr {
    /// `fr_live8`, `fr_live32`: this many hashed words live across the call.
    Live(u32),
    /// `fr_nest44`, `fr_nest88`: a square array of this side.
    Nest(u32),
    /// `fr_nestv`: a `[3][2]` array of `uint4`.
    NestV,
    /// `fr_w8` to `fr_w64`: a one-dimensional array of this many words.
    W(u32),
    /// `fr_pa` in this mode: only `n` across the call, to `fr_pb` (to itself in mode 2).
    Pa(u32),
    /// `fr_pb` in this mode: a 16-word array, to `fr_pa` in mode 0 and to itself otherwise.
    Pb(u32),
    /// `fr_qa`: a 16-word array, to `fr_qb`.
    Qa,
    /// `fr_qb`: a 64-word array, to `fr_qa`.
    Qb,
    /// `fr_sa`, `fr_ta`, `fr_ub`: only `n` across the call, to [`Fr::Big`].
    Small,
    /// `fr_sb`, `fr_tb`, `fr_ua`: a 64-word array, to [`Fr::Small`].
    Big,
}

/// `inner` plus the weighted sum of the `words`-word array `w[i] = n words + i`.
fn weighted(n: u32, words: u32, inner: u32) -> u32 {
    (0..words).fold(inner, |s, j| {
        let w = n
            .wrapping_mul(words)
            .wrapping_add(j.wrapping_mul(7).wrapping_add(n) & (words - 1));
        s.wrapping_add(w.wrapping_mul(j + 1))
    })
}

/// `f(n)`, Metal's answer where the recursion fits.
fn value(f: Fr, n: u32) -> u32 {
    let inner = |g: Fr| if n == 0 { 0 } else { value(g, n - 1) };
    match f {
        Fr::Live(count) => {
            if n == 0 {
                return 0;
            }
            let mut h = n.wrapping_mul(0x9E37_79B9).wrapping_add(0x7F4A_7C15);
            let a: Vec<u32> = (0..count)
                .map(|_| {
                    h = (h ^ (h >> 13)).wrapping_mul(0x85EB_CA6B);
                    h
                })
                .collect();
            (1u32..)
                .zip(a)
                .fold(inner(f), |s, (i, x)| s.wrapping_add(x.wrapping_mul(i)))
        }
        Fr::Nest(side) => (0..side * side).fold(inner(f), |s, j| {
            let i = j.wrapping_mul(7).wrapping_add(n) & (side - 1);
            let jj = j.wrapping_mul(5).wrapping_add(n) & (side - 1);
            let w = n
                .wrapping_mul(side * side)
                .wrapping_add(i * side)
                .wrapping_add(jj);
            s.wrapping_add(w.wrapping_mul(j + 1))
        }),
        Fr::NestV => (0..24u32).fold(inner(f), |s, j| {
            let i = j.wrapping_add(n) % 3;
            let jj = j.wrapping_mul(5).wrapping_add(n) & 1;
            let c = j.wrapping_mul(3).wrapping_add(n) & 3;
            let w = n.wrapping_mul(24).wrapping_add(i * 8 + jj * 4 + c);
            s.wrapping_add(w.wrapping_mul(j + 1))
        }),
        Fr::W(words) => weighted(n, words, inner(f)),
        Fr::Pa(mode) if n > 0 => {
            let next = if mode == 2 {
                Fr::Pa(mode)
            } else {
                Fr::Pb(mode)
            };
            n.wrapping_add(inner(next))
        }
        Fr::Pb(mode) => {
            let next = if mode == 0 {
                Fr::Pa(mode)
            } else {
                Fr::Pb(mode)
            };
            weighted(n, 16, inner(next))
        }
        Fr::Qa => weighted(n, 16, inner(Fr::Qb)),
        Fr::Qb => weighted(n, 64, inner(Fr::Qa)),
        Fr::Small if n > 0 => n.wrapping_add(inner(Fr::Big)),
        Fr::Big => weighted(n, 64, inner(Fr::Small)),
        Fr::Pa(_) | Fr::Small => 0,
    }
}

const PAIR: &[&str] = &["fr_pa", "fr_pb"];

/// A kernel's call in one mode: the kernel, the functions its pipeline links, the mode, the
/// function it calls, and Metal's deepest correct depth at `k`.
type Call = (
    &'static str,
    &'static [&'static str],
    u32,
    Fr,
    fn(u32) -> u32,
);

/// The calls the test makes.
fn calls() -> [Call; 17] {
    let lone: fn(u32) -> u32 = |k| k - 1;
    let twice: fn(u32) -> u32 = |k| 2 * k - 1;
    let four: fn(u32) -> u32 = |k| 4 * k - 1;
    [
        ("frames_live8", &["fr_live8"], 0, Fr::Live(8), lone),
        ("frames_live32", &["fr_live32"], 0, Fr::Live(32), lone),
        ("frames_nest44", &["fr_nest44"], 0, Fr::Nest(4), lone),
        ("frames_nest88", &["fr_nest88"], 0, Fr::Nest(8), lone),
        ("frames_nestv", &["fr_nestv"], 0, Fr::NestV, lone),
        ("frames_w8", &["fr_w8"], 0, Fr::W(8), lone),
        ("frames_w16", &["fr_w16"], 0, Fr::W(16), lone),
        ("frames_w32", &["fr_w32"], 0, Fr::W(32), lone),
        ("frames_w64", &["fr_w64"], 0, Fr::W(64), lone),
        ("frames_pair", PAIR, 0, Fr::Pa(0), twice),
        ("frames_pair", PAIR, 1, Fr::Pb(1), twice),
        ("frames_pair", PAIR, 2, Fr::Pa(2), twice),
        ("frames_quad", &["fr_qa", "fr_qb"], 0, Fr::Qa, four),
        ("frames_s", &["fr_sa", "fr_sb"], 0, Fr::Small, four),
        ("frames_t", &["fr_ta", "fr_tb"], 0, Fr::Small, |k| 2 * k),
        ("frames_t", &["fr_ta", "fr_tb"], 1, Fr::Big, twice),
        ("frames_u", &["fr_ua", "fr_ub"], 0, Fr::Big, twice),
    ]
}

/// The state of `kernel` at `maxCallStackDepth` `depth`, linking `names`.
fn state_of(
    library: &Library,
    kernel: &str,
    names: &[&str],
    depth: u32,
) -> Result<ComputePipelineState, Error> {
    let linked = names.iter().fold(LinkedFunctions::new(), |linked, name| {
        linked.with_function(
            library
                .new_function_with_name(name)
                .unwrap()
                .linked_function(),
        )
    });
    let function = library.new_function_with_name(kernel).unwrap();
    let mut descriptor = ComputePipelineDescriptor::new(StageSource::Function(
        function.function_descriptor().with_linked(linked),
    ));
    descriptor.max_call_stack_depth = depth;
    library
        .device()
        .new_compute_pipeline_state(&descriptor, &PipelineOptions::default())
}

/// Whether `u` refuses a recursion's stack past the device's Function storage.
fn past_function_storage(u: &Unsupported) -> bool {
    let Needs::AnyOf(alternatives) = &u.needs else {
        return false;
    };
    u.feature == Feature::Recursion
        && alternatives.iter().flatten().any(|c| {
            matches!(c, Capability::Limit {
                limit: VkLimit::FunctionStorageBytesPerThread,
                required,
                available,
            } if required > available)
        })
}

fn words_buffer(device: &Device, words: &[u32]) -> Buffer {
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    device
        .new_buffer_with_bytes(&bytes, ResourceOptions::shared())
        .unwrap()
}

/// Run `state` over [`N`] threads in one threadgroup with the depths `depths` and mode `mode`:
/// the words it writes.
fn dispatch(device: &Device, state: &ComputePipelineState, depths: &[u32], mode: u32) -> Vec<u32> {
    let out = words_buffer(device, &[0; N as usize]);
    let depth = words_buffer(device, depths);
    let mode = words_buffer(device, &[mode]);
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(state).unwrap();
    for (index, buffer) in [(0, &out), (2, &depth), (3, &mode)] {
        encoder.set_buffer(Some(buffer), 0, index).unwrap();
    }
    encoder
        .dispatch_threads(Size::new(N.into(), 1, 1), Size::new(N.into(), 1, 1))
        .unwrap();
    encoder.end_encoding().unwrap();
    run(&command_buffer);
    read_buffer(&out, 0, u64::from(N) * 4)
        .unwrap()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|w| u32::from_le_bytes(*w))
        .collect()
}

/// Every depth Metal answers correctly at `maxCallStackDepth` 1, 2 and 4 is answered as the MSL
/// computes it, for each kernel and mode of the fixture, where the device holds the stack.
#[test]
fn recursions_answer_every_depth_metal_answers() {
    on_device_with(at_plan, |device| {
        let air = llvm_assemble(&engine_dir().join("fpc_frames.ll")).unwrap();
        let library = device.new_library_with_air(&air).unwrap();
        for (kernel, linked, mode, root, metal) in calls() {
            for k in [1, 2, 4] {
                let reach = metal(k);
                let depths: Vec<u32> = (0..N).map(|t| t.min(reach)).collect();
                let state = match state_of(&library, kernel, linked, k) {
                    Ok(state) => state,
                    Err(Error::Unsupported(u) | Error::Compile(CompileError::Unsupported(u)))
                        if k > 1 && past_function_storage(&u) =>
                    {
                        eprintln!("REFUSED {kernel} mode {mode} at k {k}: {u:?}");
                        continue;
                    }
                    Err(e) => panic!("{kernel} at {k}: {e}"),
                };
                let want: Vec<u32> = depths.iter().map(|&d| value(root, d)).collect();
                assert_eq!(
                    dispatch(device, &state, &depths, mode),
                    want,
                    "{kernel} mode {mode} at k {k}"
                );
            }
        }
    });
}
