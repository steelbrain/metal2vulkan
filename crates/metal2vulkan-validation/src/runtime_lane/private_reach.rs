//! Kernels written for their largest tile through the runtime on the device the cfg selects, under
//! the validation layer, against what the probe `private_array_reach.m` (the conformance crate's
//! `probes/`) measured on an Apple M3 Max under macOS 27.0.1.
//!
//! The libraries are the owned fixtures `private_reach_tiles` (kernels each thread of which keeps
//! `float acc[32][32][4][4][4]`, 256 KiB, past every measured device's Function storage, and hands
//! each sum to the visible function `postfix`, which it declares) and `private_reach_functions`
//! (`postfix(x, w) = x * 2 + w`), linked. With the tile sizes 2 and 4 (function constants 0 and
//! 1), `private_reach_tiles` reaches `acc[0..2][0..4][0..4][0][0]`, 32 floats: over `in[i] = i`
//! thread `t` stores `2 * (120 + 16 t) * (w + 1) + w` at word `t * 128 + w` for `w < 32`, the
//! words the probe printed (thread 0: 240 481 722 ... 7711; thread 3: 336 673 1010 ... 10783).
//! `private_reach_rows`'s outer loop runs `in[64]` times, a count only the run knows: it keeps the
//! whole accumulator and is refused needing the device's Function storage, as the M3 refused its
//! creation ("Compute function exceeds available stack space").

use super::at_plan;
use super::host_memory::on_device_with;
use super::host_memory::run;
use super::read_buffer;
use crate::air_library::AirLibrary;
use metal2vulkan::device::VkLimit;
use metal2vulkan::error::{Capability, Error as CompileError, Feature, Needs, Unsupported};
use metal2vulkan::library::{FunctionConstantValue, FunctionConstantValues};
use metal2vulkan::metal::DataType;
use metal2vulkan::pipeline::{ComputePipelineDescriptor, LinkedFunctions, StageSource};
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use metal2vulkan_runtime::{
    ComputePipelineState, Device, Error, FunctionDescriptor, Library, PipelineOptions,
    ResourceOptions, Size,
};

/// Threads of the dispatch.
const THREADS: u32 = 4;

/// Words of `out` each thread owns.
const WORDS: u32 = 128;

/// Words a thread stores with the tile sizes 2 and 4.
const REACHED: u32 = 32;

/// The tile fixture's library and the functions' library.
fn libraries(device: &Device) -> (Library, Library) {
    let fixture = |stem: &str| llvm_assemble(&engine_dir().join(format!("{stem}.ll"))).unwrap();
    (
        device
            .new_library_with_air(&fixture("private_reach_tiles"))
            .unwrap(),
        device
            .new_library_with_air(&fixture("private_reach_functions"))
            .unwrap(),
    )
}

/// The state of `kernel` of `tiles` with the tile sizes 2 and 4, linking `postfix` of
/// `functions`.
fn state(
    tiles: &Library,
    functions: &Library,
    kernel: &str,
) -> Result<ComputePipelineState, Error> {
    let uint = |v: u32| FunctionConstantValue::new(DataType::UInt, v.to_le_bytes());
    let mut d = FunctionDescriptor::new(kernel);
    d.constant_values = Some(
        FunctionConstantValues::defaults()
            .with_index(0, uint(2))
            .with_index(1, uint(4)),
    );
    let kernel = tiles.new_function_with_descriptor(&d).unwrap();
    let postfix = functions.new_function_with_name("postfix").unwrap();
    let linked = LinkedFunctions::new().with_function(postfix.linked_function());
    tiles.device().new_compute_pipeline_state(
        &ComputePipelineDescriptor::new(StageSource::Function(
            kernel.function_descriptor().with_linked(linked),
        )),
        &PipelineOptions::default(),
    )
}

/// Whether `u` refuses a private array past the device's Function storage, needing `required`.
fn past_function_storage(u: &Unsupported, required: u64) -> bool {
    let Needs::AnyOf(alternatives) = &u.needs else {
        return false;
    };
    u.feature == Feature::PrivateMemory
        && alternatives.iter().flatten().any(|c| {
            matches!(c, Capability::Limit {
                limit: VkLimit::FunctionStorageBytesPerThread,
                required: r,
                available,
            } if *r == required && r > available)
        })
}

/// The words `state` stores over [`THREADS`] threads, `in[i] = i` for `i < 64` and `in[64] = 2`.
fn words(device: &Device, state: &ComputePipelineState) -> Vec<f32> {
    let input: Vec<u8> = (0..64)
        .map(|i| i as f32)
        .chain([2.0])
        .flat_map(f32::to_le_bytes)
        .collect();
    let input = device
        .new_buffer_with_bytes(&input, ResourceOptions::shared())
        .unwrap();
    let out = device
        .new_buffer_with_bytes(
            &vec![0; (THREADS * WORDS * 4) as usize],
            ResourceOptions::shared(),
        )
        .unwrap();
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(state).unwrap();
    encoder.set_buffer(Some(&out), 0, 0).unwrap();
    encoder.set_buffer(Some(&input), 0, 1).unwrap();
    encoder
        .dispatch_threads(
            Size::new(THREADS.into(), 1, 1),
            Size::new(THREADS.into(), 1, 1),
        )
        .unwrap();
    encoder.end_encoding().unwrap();
    run(&command_buffer);
    read_buffer(&out, 0, out.length())
        .unwrap()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|w| f32::from_le_bytes(*w))
        .collect()
}

/// What the probe printed: word `w < 32` of thread `t` is `2 * (120 + 16 t) * (w + 1) + w`; the
/// rest stay 0.
fn expected() -> Vec<f32> {
    (0..THREADS)
        .flat_map(|t| {
            (0..WORDS).map(move |w| {
                if w < REACHED {
                    (2 * (120 + 16 * t) * (w + 1) + w) as f32
                } else {
                    0.0
                }
            })
        })
        .collect()
}

/// A kernel whose tile sizes reach 128 bytes of its 256 KiB accumulator makes its state on every
/// venue (lavapipe's 48 KiB and the RX 7900 XTX's 96 KiB among them) and stores the M3's words;
/// one whose loop a buffer word bounds keeps the 256 KiB and is refused needing them. The probe's
/// `tiles` and `rows`.
#[test]
fn a_tile_kernel_holds_only_the_corner_its_tile_sizes_reach() {
    on_device_with(at_plan, |device| {
        let (tiles, functions) = libraries(device);
        let made = state(&tiles, &functions, "private_reach_tiles")
            .unwrap_or_else(|e| panic!("{}: {e}", device.name()));
        assert_eq!(words(device, &made), expected(), "{}", device.name());
        match state(&tiles, &functions, "private_reach_rows") {
            Err(Error::Unsupported(u) | Error::Compile(CompileError::Unsupported(u)))
                if past_function_storage(&u, 32 * 32 * 4 * 4 * 4 * 4) => {}
            Err(e) => panic!("{}: {e}", device.name()),
            Ok(_) => panic!("{}: the rows kernel made its state", device.name()),
        }
    });
}
