//! Kernel reads and writes of Apple's two-plane extended-range format on the device the cfg
//! selects, under the validation layer: a texture of `RGB10A8_2P_XR10` (or its sRGB twin), and a
//! consumer's two plane views of one over a buffer.
//!
//! The owned fixture `xr_two_planes` writes a buffer's `half4` or `float4` values to a texture
//! (`texture2d<half>` or `<float>`, `access::write`) and reads the texture back into a buffer
//! (`access::read`), one thread per texel of a grid whose rows are `w` texels apart, under 32 x 32
//! threadgroups:
//!
//! - a 256 x 256 texture of each two-plane format, one thread per texel: every `half` pattern in
//!   every channel (lane `c` of texel `n` holds pattern `(n + 16384 c) mod 65536`) and a dense
//!   `float` grid (`((n + 16384 c) mod 65536 - 32768) / 16384`), each read back as `half` and
//!   `float`;
//! - the consumer's shape: a 64 x 4 texture under 2 x 1 threadgroups (a 64 x 32 grid, rows 4 to
//!   31 outside the texture), lane `c` of grid thread `n` writing pattern `0x3000 + n + 2048 c`;
//! - the consumer's plane views: one shared buffer of 4096 bytes (byte `i` holding
//!   `(89 i + 7) mod 256`) with plane 0 at offset 0 as `BGR10_XR` (or its sRGB twin) and plane 1
//!   at offset 2048 as `ARGB8Unorm`, each a 2D texture made with `new_texture` over the buffer,
//!   64 x 4 with rows 256 bytes apart and 40 x 3 with rows 192 bytes apart, under the consumer's
//!   threadgroups: both views read, then written with `half` patterns
//!   (`(32 n + 16384 c + 7 c) mod 65536`) and read, then written with `float`s
//!   (`((32 n + 16384 c) mod 65536 - 32768) / 16384`) and read, with the buffer's bytes after each
//!   write.
//!
//! Metal's answer: the SHA-256 of each output, the same on an Apple M3 Max (macOS 26.7) and an
//! Apple M4 (macOS 27.0.1), plain and under Metal's validation layer alike: the textures' and the
//! plane views' from our probes, which keep the bytes. A differing output is
//! printed whole, on one `XR-TWO-PLANES-DIFFERS` line.

use super::degrade_notices::{counts, on_device};
use super::extended_range_targets::{half_bits, half_value};
use super::pixel_samplers::metal_sampler;
use super::xr_codes::{as_metal_reads, assert_announced, assert_counted, codes, texel_within};
use super::{buffer_with, converted_holds, half_within, read_buffer};
use crate::air_library::AirLibrary;
use metal2vulkan::error::Feature;
use metal2vulkan::metal::format::ExtendedRange;
use metal2vulkan::metal::{PixelFormat, TextureType, TextureUsage};
use metal2vulkan::pipeline::{ComputePipelineDescriptor, StageSource};
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use metal2vulkan_runtime::{
    BlitCommandEncoder, BlitOption, Buffer, ClearColor, CommandBufferStatus, CommandQueue,
    ComputePipelineState, CpuCacheMode, Degradation, Device, Error, HazardTrackingMode, Heap,
    HeapDescriptor, HeapType, Invalid, IoSurfacePlane, IoSurfaceYCbCr, LoadAction, Nil, Origin,
    PipelineOptions, Problem, Region, RenderPassDescriptor, ResourceOptions, Size, StorageMode,
    StoreAction, Texture, TextureDescriptor, TextureProblem,
};
use sha2::{Digest, Sha256};
use std::time::Duration;

/// The longest a command buffer may take.
const WAIT: Duration = Duration::from_secs(60);

/// Metal's outputs of a texture of each two-plane format: its format, the output (`HR`: reads
/// after the `half` writes, `FR`: after the `float` writes, `OR`: the consumer's shape; `H` and
/// `F`: the reads' result type) and the SHA-256 of its bytes.
const TEXTURES: [(PixelFormat, &str, &str); 12] = [
    (
        PixelFormat::RGB10A82PXR10,
        "HRH",
        "22cdc45b67da6527333067df72530e09e9dbcc84264693e68dbb931a7e71f390",
    ),
    (
        PixelFormat::RGB10A82PXR10,
        "HRF",
        "a35846dbd312d807c50310d22849a4a5ff789cd274a27ff926c59e562f2c00fa",
    ),
    (
        PixelFormat::RGB10A82PXR10,
        "FRH",
        "3d5a0a1f6040bc8fa020a49dc7fec306bab6a86722a6c324f347272db3c70ba6",
    ),
    (
        PixelFormat::RGB10A82PXR10,
        "FRF",
        "3ead12fa542258ee768c9cf388f96acca8cf94a761fbe6a82013f3146c13e33a",
    ),
    (
        PixelFormat::RGB10A82PXR10,
        "ORH",
        "957ef0211e3032b942c71651570ff625670198bfebe0f1c2978fdd21129d71d9",
    ),
    (
        PixelFormat::RGB10A82PXR10,
        "ORF",
        "eac8dabf8163dacdcf619a65681c3c980548e97c39d8ac35304b44208b8a73f3",
    ),
    (
        PixelFormat::RGB10A82PXR10SRGB,
        "HRH",
        "84b306e7871b19413062634947bfbdb8a0149c2aacf74f26ca3b9b6faccedc3d",
    ),
    (
        PixelFormat::RGB10A82PXR10SRGB,
        "HRF",
        "bbf1d6107835195d3daef224de56d24aa6b8b0221d3be8e36ca7de88dd09c0a3",
    ),
    (
        PixelFormat::RGB10A82PXR10SRGB,
        "FRH",
        "ea11a73fe55459ded35d32d872daf417b683fc9e3fdceb3416c74b61747e420f",
    ),
    (
        PixelFormat::RGB10A82PXR10SRGB,
        "FRF",
        "df70ed07347b993b43fdae512e9e9cc3e145a8fe2771e89772489a60eee88a19",
    ),
    (
        PixelFormat::RGB10A82PXR10SRGB,
        "ORH",
        "9ab95cab828e6aa443e91e0ed960269c33732dcf3e9364dd3944d0467fcb0780",
    ),
    (
        PixelFormat::RGB10A82PXR10SRGB,
        "ORF",
        "6760e2782c211b2b70bee680a6e9b59bd9c110e81955cf30d5697bfb38b37630",
    ),
];

/// Metal's outputs of the plane views: plane 0's format, the output (`G<g>`: geometry `g` of
/// [`GEOMETRIES`]; `R<s>P<p><H|F>`: the reads of plane `p` after step `s`, 0 before any write, 1
/// after the `half` writes, 2 after the `float` writes; `W<s>BUF`: the buffer's bytes after step
/// `s`'s writes) and the SHA-256 of its bytes.
const VIEWS: [(PixelFormat, &str, &str); 56] = [
    (
        PixelFormat::BGR10XR,
        "G0R0P0H",
        "b483b1d7b3b05b4d74a6e2327cf236c417db1eb2e2b81c8a531db95080afaf73",
    ),
    (
        PixelFormat::BGR10XR,
        "G0R0P0F",
        "6ff881356a5ca03fd54fca5ee6731e745a21bc69322b1b186db358479d29c355",
    ),
    (
        PixelFormat::BGR10XR,
        "G0R0P1H",
        "f761d8eb4502fd5c6f0e1c503ab49460daca7668c481afd9ff788d174e76431e",
    ),
    (
        PixelFormat::BGR10XR,
        "G0R0P1F",
        "15497052576b0dab3f2c9c0d62c35dbd4d5d6cca8050928cb89b8b13cee52a43",
    ),
    (
        PixelFormat::BGR10XR,
        "G0W1BUF",
        "19843f1b277fe8f5959d6972d0339c2aff91364b79c0ad161c190e37713f9dab",
    ),
    (
        PixelFormat::BGR10XR,
        "G0R1P0H",
        "1327e94fecd16989672c75481fd2002805fac2a4607c809e57134d6e35f015a9",
    ),
    (
        PixelFormat::BGR10XR,
        "G0R1P0F",
        "f59b9721689d9de08aa301336521a9ce2700167d5c139b41e5a4d65c807dd1ba",
    ),
    (
        PixelFormat::BGR10XR,
        "G0R1P1H",
        "86ac42b4c82d0a96aa8c98f82e937bedba235a00650dd392aa4d3847fba5f9ef",
    ),
    (
        PixelFormat::BGR10XR,
        "G0R1P1F",
        "4a14a7766350e5c557aaba4a0f3d87a56e336d8f97c17aff8c08abb3770a641b",
    ),
    (
        PixelFormat::BGR10XR,
        "G0W2BUF",
        "980d76185f2053779ab82ff89f72e136803dec9fbae6009b3243d32b72352956",
    ),
    (
        PixelFormat::BGR10XR,
        "G0R2P0H",
        "c57a627dc6d1385e29eaa3e2bd449b3279aa01d2d269d981e03f8e0c0da98647",
    ),
    (
        PixelFormat::BGR10XR,
        "G0R2P0F",
        "479fa0d8d1eb3cf3cf631008bc44319cbcd4722a98056eb8741eaeb42687e853",
    ),
    (
        PixelFormat::BGR10XR,
        "G0R2P1H",
        "ffa8a58c3b205876f527f34c43b995df9ca39830e12b118cc7c2943a796a9f27",
    ),
    (
        PixelFormat::BGR10XR,
        "G0R2P1F",
        "df728d6477a22251d213f356cf2cf34965e8b9e38cd41a371f37fe977ec5166f",
    ),
    (
        PixelFormat::BGR10XR,
        "G1R0P0H",
        "95ddef7435e9473baf8de155074df8785a3d8268ae88e9310c6b5e3a0cccf0d0",
    ),
    (
        PixelFormat::BGR10XR,
        "G1R0P0F",
        "c263b4f41df1b523b837482476b7e04971065b447ed842480f35b76759ab7262",
    ),
    (
        PixelFormat::BGR10XR,
        "G1R0P1H",
        "2d5828773550df7e6945fe80dcf5ab08f1e20308cb484fbf5b0ee1968e971bf8",
    ),
    (
        PixelFormat::BGR10XR,
        "G1R0P1F",
        "7c1b8e689df20bb0b2456482e340e8e093eee1f1a76c414e400cbb2ffeb825d7",
    ),
    (
        PixelFormat::BGR10XR,
        "G1W1BUF",
        "4050578c92bdd1c63170a3968bd318941322c06248d6896362654f9ded87a1d6",
    ),
    (
        PixelFormat::BGR10XR,
        "G1R1P0H",
        "985e576ba856b4c4745465c3e4057ad1c8cb75e03f4c4a00a96a1e35cc74f018",
    ),
    (
        PixelFormat::BGR10XR,
        "G1R1P0F",
        "6d0848522dfea8b6083689718ef76c64c14d36de097a7fda82e2436a6c020452",
    ),
    (
        PixelFormat::BGR10XR,
        "G1R1P1H",
        "36bd67d77aada00c5b064987f9c3adc9fcbd85f215b6bb170eb9ceaa75659b37",
    ),
    (
        PixelFormat::BGR10XR,
        "G1R1P1F",
        "18d70d1c735a4b56f9a769aae7d64539e46553bdb28858471e438fcbb0c8ec6e",
    ),
    (
        PixelFormat::BGR10XR,
        "G1W2BUF",
        "3954d3f5b58b05d9362952ee6aba2f74b215e34c2238c016ef1822746272dfea",
    ),
    (
        PixelFormat::BGR10XR,
        "G1R2P0H",
        "7d5eb36c1b5ffc414ae5d0473027e8174b4bb80fe434db33ff1de1b25cd493ca",
    ),
    (
        PixelFormat::BGR10XR,
        "G1R2P0F",
        "e756ae67309b05930f10e75ec06327b92540f9bc0e0596282e8c56a46700d0f7",
    ),
    (
        PixelFormat::BGR10XR,
        "G1R2P1H",
        "ed4e0e0fe4ee0b63d434722dd61fa4c79d180e76b6fe55af05c3c065df1de6b0",
    ),
    (
        PixelFormat::BGR10XR,
        "G1R2P1F",
        "0662049a1954905e032b58b76a78d3e55d76641af9f510912af40c9fc238d26d",
    ),
    (
        PixelFormat::BGR10XRSRGB,
        "G0R0P0H",
        "99286e79493d32759423220db6b7cec6253699d86d7155085c76d71f9305deef",
    ),
    (
        PixelFormat::BGR10XRSRGB,
        "G0R0P0F",
        "73a9165d9d5644f33d8b01cbe446361d38e402eee21836f0a4898f54f8b095a9",
    ),
    (
        PixelFormat::BGR10XRSRGB,
        "G0R0P1H",
        "f761d8eb4502fd5c6f0e1c503ab49460daca7668c481afd9ff788d174e76431e",
    ),
    (
        PixelFormat::BGR10XRSRGB,
        "G0R0P1F",
        "15497052576b0dab3f2c9c0d62c35dbd4d5d6cca8050928cb89b8b13cee52a43",
    ),
    (
        PixelFormat::BGR10XRSRGB,
        "G0W1BUF",
        "2282821c0cea3cf718ff030629dd403af227c8b7ae6d910c355a37cda912ce20",
    ),
    (
        PixelFormat::BGR10XRSRGB,
        "G0R1P0H",
        "090f5ac305e09d16af247f98ba3dd0bcf820608b52d3524ac677745d86d6d9fe",
    ),
    (
        PixelFormat::BGR10XRSRGB,
        "G0R1P0F",
        "eef6dd65dad347d096f2c1aedef847e28b72eb13922c9c8b58c3e5ba06c16a31",
    ),
    (
        PixelFormat::BGR10XRSRGB,
        "G0R1P1H",
        "86ac42b4c82d0a96aa8c98f82e937bedba235a00650dd392aa4d3847fba5f9ef",
    ),
    (
        PixelFormat::BGR10XRSRGB,
        "G0R1P1F",
        "4a14a7766350e5c557aaba4a0f3d87a56e336d8f97c17aff8c08abb3770a641b",
    ),
    (
        PixelFormat::BGR10XRSRGB,
        "G0W2BUF",
        "ad267fce676137bd9407833a970f47b4321d3d19a988a1433695d5ab36e8acb7",
    ),
    (
        PixelFormat::BGR10XRSRGB,
        "G0R2P0H",
        "40ea900c659d89ebf8d81fe59e1fcf8900aeb3c3417abf3871e88b5d158ee2f2",
    ),
    (
        PixelFormat::BGR10XRSRGB,
        "G0R2P0F",
        "a72c1e024c4d90eb62913677a024cbfc048a65a43fc08bb0b2d75f5659c63da9",
    ),
    (
        PixelFormat::BGR10XRSRGB,
        "G0R2P1H",
        "ffa8a58c3b205876f527f34c43b995df9ca39830e12b118cc7c2943a796a9f27",
    ),
    (
        PixelFormat::BGR10XRSRGB,
        "G0R2P1F",
        "df728d6477a22251d213f356cf2cf34965e8b9e38cd41a371f37fe977ec5166f",
    ),
    (
        PixelFormat::BGR10XRSRGB,
        "G1R0P0H",
        "2b7053f9b3a609596f3f7e43468277bd293e29596895506721b58694f895f8c1",
    ),
    (
        PixelFormat::BGR10XRSRGB,
        "G1R0P0F",
        "05a34979f30eff134b266d0e884c5d1f64bf80f8a54238e0dcfd84f9f867285b",
    ),
    (
        PixelFormat::BGR10XRSRGB,
        "G1R0P1H",
        "2d5828773550df7e6945fe80dcf5ab08f1e20308cb484fbf5b0ee1968e971bf8",
    ),
    (
        PixelFormat::BGR10XRSRGB,
        "G1R0P1F",
        "7c1b8e689df20bb0b2456482e340e8e093eee1f1a76c414e400cbb2ffeb825d7",
    ),
    (
        PixelFormat::BGR10XRSRGB,
        "G1W1BUF",
        "8a00511dc824517be1f3056c646e572b8ecebea55233e3dd37018016e0b696eb",
    ),
    (
        PixelFormat::BGR10XRSRGB,
        "G1R1P0H",
        "8b097d68c10282432a09df84f3efc6a8d43fdbac03204bddba62241e73da7fad",
    ),
    (
        PixelFormat::BGR10XRSRGB,
        "G1R1P0F",
        "d5cc080aa785dcb83058eec1e2ca10bded0647836bb119257dd8f36f5f3bb48f",
    ),
    (
        PixelFormat::BGR10XRSRGB,
        "G1R1P1H",
        "36bd67d77aada00c5b064987f9c3adc9fcbd85f215b6bb170eb9ceaa75659b37",
    ),
    (
        PixelFormat::BGR10XRSRGB,
        "G1R1P1F",
        "18d70d1c735a4b56f9a769aae7d64539e46553bdb28858471e438fcbb0c8ec6e",
    ),
    (
        PixelFormat::BGR10XRSRGB,
        "G1W2BUF",
        "44bdcba10e9ce1c7b826c1b2243bf96d9cf7507e9215b9cce8b885c83d0cb514",
    ),
    (
        PixelFormat::BGR10XRSRGB,
        "G1R2P0H",
        "98906c5cbe8d7e5c2029cf25254f492a255e56e9b4d889596cbd9ac43dc13671",
    ),
    (
        PixelFormat::BGR10XRSRGB,
        "G1R2P0F",
        "80741b2f62576719a80eacb03a7c8a6aafe4284e3392f0711c3e4de03a594ef9",
    ),
    (
        PixelFormat::BGR10XRSRGB,
        "G1R2P1H",
        "ed4e0e0fe4ee0b63d434722dd61fa4c79d180e76b6fe55af05c3c065df1de6b0",
    ),
    (
        PixelFormat::BGR10XRSRGB,
        "G1R2P1F",
        "0662049a1954905e032b58b76a78d3e55d76641af9f510912af40c9fc238d26d",
    ),
];

/// The plane views' geometries: width, height and bytes per row.
const GEOMETRIES: [(u64, u64, u64); 2] = [(64, 4, 256), (40, 3, 192)];

/// The plane views' buffer length, and plane 1's offset in it.
const VIEW_BUFFER: u64 = 4096;
const PLANE_1: u64 = 2048;

/// The consumer's grid: 2 x 1 threadgroups of 32 x 32, rows 64 threads wide.
const GRID_WIDTH: u64 = 64;
const GRID: usize = 64 * 32;

/// The fixture's kernel `name`.
fn state(device: &Device, name: &str) -> ComputePipelineState {
    let bitcode = llvm_assemble(&engine_dir().join("xr_two_planes.ll")).unwrap();
    let library = device.new_library_with_air(&bitcode).unwrap();
    let function = library.new_function_with_name(name).unwrap();
    device
        .new_compute_pipeline_state(
            &ComputePipelineDescriptor::new(StageSource::Function(function.function_descriptor())),
            &PipelineOptions::default(),
        )
        .unwrap_or_else(|e| panic!("{name}: {e}"))
}

/// The fixture's four kernels.
pub(super) struct Kernels {
    write_half: ComputePipelineState,
    write_float: ComputePipelineState,
    read_half: ComputePipelineState,
    read_float: ComputePipelineState,
}

impl Kernels {
    pub(super) fn new(device: &Device) -> Kernels {
        Kernels {
            write_half: state(device, "xr2p_write_half"),
            write_float: state(device, "xr2p_write_float"),
            read_half: state(device, "xr2p_read_half"),
            read_float: state(device, "xr2p_read_float"),
        }
    }
}

/// How a kernel is dispatched: one thread per texel of a `width` x `height` texture, or the
/// consumer's grid; 32 x 32 threads per threadgroup either way.
#[derive(Clone, Copy)]
pub(super) enum Dispatch {
    Texels(u64, u64),
    Consumer,
}

impl Dispatch {
    /// The grid's row width, `w`.
    fn width(self) -> u64 {
        match self {
            Dispatch::Texels(width, _) => width,
            Dispatch::Consumer => GRID_WIDTH,
        }
    }

    /// The grid's threads.
    fn threads(self) -> usize {
        match self {
            Dispatch::Texels(width, height) => (width * height) as usize,
            Dispatch::Consumer => GRID,
        }
    }
}

/// Runs `state` over `texture` with `buffer` at 0 and the grid's row width at 1.
fn run(
    device: &Device,
    queue: &CommandQueue,
    state: &ComputePipelineState,
    (texture, buffer): (&Texture, &Buffer),
    dispatch: Dispatch,
) {
    let width = buffer_with(device, &(dispatch.width() as u32).to_le_bytes()).unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(state).unwrap();
    encoder.set_texture(Some(texture), 0).unwrap();
    encoder.set_buffer(Some(buffer), 0, 0).unwrap();
    encoder.set_buffer(Some(&width), 0, 1).unwrap();
    let group = Size::new(32, 32, 1);
    match dispatch {
        Dispatch::Texels(width, height) => {
            encoder.dispatch_threads(Size::new(width, height, 1), group)
        }
        Dispatch::Consumer => encoder.dispatch_threadgroups(Size::new(2, 1, 1), group),
    }
    .unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    assert_eq!(command_buffer.error(), None);
}

/// Writes `values` (`half4`s where `half`, else `float4`s, one per grid thread) to `texture`.
pub(super) fn write(
    device: &Device,
    queue: &CommandQueue,
    kernels: &Kernels,
    texture: &Texture,
    (values, half): (&[u8], bool),
    dispatch: Dispatch,
) {
    let input = buffer_with(device, values).unwrap();
    let state = if half {
        &kernels.write_half
    } else {
        &kernels.write_float
    };
    run(device, queue, state, (texture, &input), dispatch);
}

/// What `texture` reads, as `half4`s and as `float4`s, one per grid thread, the output buffers
/// filled with `0xab` first.
pub(super) fn reads(
    device: &Device,
    queue: &CommandQueue,
    kernels: &Kernels,
    texture: &Texture,
    dispatch: Dispatch,
) -> [Vec<u8>; 2] {
    [(&kernels.read_half, 8u64), (&kernels.read_float, 16)].map(|(state, size)| {
        let length = dispatch.threads() as u64 * size;
        let out = buffer_with(device, &vec![0xab; length as usize]).unwrap();
        run(device, queue, state, (texture, &out), dispatch);
        read_buffer(&out, 0, length).unwrap()
    })
}

/// `n` grid threads' `half4`s: lane `c` of thread `n` holds pattern `pattern(n, c)`.
fn halves(n: usize, pattern: impl Fn(usize, usize) -> usize) -> Vec<u8> {
    (0..n)
        .flat_map(|t| (0..4).map(move |c| (t, c)))
        .flat_map(|(t, c)| ((pattern(t, c) & 0xffff) as u16).to_le_bytes())
        .collect()
}

/// `n` grid threads' `float4`s: lane `c` of thread `n` holds `(k mod 65536 - 32768) / 16384`,
/// `k = step n + 16384 c`.
fn floats(n: usize, step: usize) -> Vec<u8> {
    (0..n)
        .flat_map(|t| (0..4).map(move |c| (t, c)))
        .flat_map(|(t, c)| {
            let k = (step * t + 16384 * c) & 0xffff;
            ((k as f32 - 32768.0) / 16384.0).to_le_bytes()
        })
        .collect()
}

/// The codes Metal's kernel write of `values` (`half4`s where `half`, else `float4`s, one a grid
/// thread `w` wide) stores in a texture of `form`, by thread: those of the threads within the
/// `width` x `height` texture (`ExtendedRange::write_code`), `None` past it and for a channel the
/// form lacks (`BGR10_XR`'s alpha).
fn written_codes(
    form: ExtendedRange,
    (values, half): (&[u8], bool),
    w: usize,
    (width, height): (usize, usize),
) -> Vec<[Option<u32>; 4]> {
    let lanes: Vec<f32> = match half {
        true => values
            .as_chunks::<2>()
            .0
            .iter()
            .map(|h| half_value(u16::from_le_bytes(*h)))
            .collect(),
        false => values
            .as_chunks::<4>()
            .0
            .iter()
            .map(|f| f32::from_le_bytes(*f))
            .collect(),
    };
    let alpha = matches!(form, ExtendedRange::Planes { .. });
    lanes
        .as_chunks::<4>()
        .0
        .iter()
        .enumerate()
        .map(|(t, v)| {
            let inside = t % w < width && t / w < height;
            std::array::from_fn(|c| {
                (inside && (c < 3 || alpha)).then(|| form.write_code(v[c], c == 3).unwrap())
            })
        })
        .collect()
}

/// Checks the reads `h` and `f` of a texture of `form` that holds Metal's codes `codes` against
/// Metal's digests of output `what` (`{what}H` and `{what}F` by `metal`), each lane the device's
/// conversion within its bound replaced by Metal's read first ([`as_metal_reads`]), and counts the
/// texels it moved into `moved`.
fn check_reads(
    (wrong, moved): (&mut Vec<String>, &mut usize),
    (form, codes): (ExtendedRange, &[[Option<u32>; 4]]),
    (what, metal): (String, impl Fn(&str) -> &'static str),
    [h, f]: [Vec<u8>; 2],
) {
    let (f, h, n) = as_metal_reads(form, (&f, &h), codes);
    *moved += n;
    check(wrong, format!("{what}H"), metal("H"), &h);
    check(wrong, format!("{what}F"), metal("F"), &f);
}

/// The SHA-256 of `bytes`, in hex.
pub(super) fn digest(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// Checks `bytes` against Metal's digest `want` of output `what`, keeping a difference in `wrong`.
fn check(wrong: &mut Vec<String>, what: String, want: &str, bytes: &[u8]) {
    let got = digest(bytes);
    if got != want {
        let hex: String = bytes.iter().map(|b| format!("{b:02x}")).collect();
        wrong.push(format!("{what}: {got}, Metal {want}; bytes {hex}"));
    }
}

/// Prints `wrong`'s lines and fails when there is one.
fn report(device: &Device, wrong: &[String]) {
    for line in wrong {
        println!("XR-TWO-PLANES-DIFFERS {}: {line}", device.name());
    }
    assert!(
        wrong.is_empty(),
        "{} outputs differ from Metal",
        wrong.len()
    );
}

/// A `width` x `height` texture of `format` that kernels read and write.
pub(super) fn texture(device: &Device, format: PixelFormat, width: u64, height: u64) -> Texture {
    let mut descriptor = TextureDescriptor::texture_2d(format, width, height, false);
    descriptor.usage = TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE;
    descriptor.resource_options = ResourceOptions::shared();
    device
        .new_texture(&descriptor)
        .unwrap_or_else(|e| panic!("a {format:?} texture: {e}"))
}

/// A texture of each two-plane format reads back Metal's values after kernel writes of every
/// `half` pattern and of a dense `float` grid, and in the consumer's shape.
#[test]
fn two_plane_textures_read_and_write_as_metal_does() {
    on_device(|device, messages| {
        let queue = device.new_command_queue().unwrap();
        let kernels = Kernels::new(device);
        let (mut wrong, mut moved) = (Vec::new(), 0);
        let side = 256;
        let texels = Dispatch::Texels(side, side);
        let count = texels.threads();
        for format in [PixelFormat::RGB10A82PXR10, PixelFormat::RGB10A82PXR10SRGB] {
            let form = ExtendedRange::of(format).expect("a two-plane format");
            let metal = |tag: String| {
                move |lanes: &str| {
                    TEXTURES
                        .iter()
                        .find(|m| m.0 == format && m.1 == format!("{tag}{lanes}"))
                        .map(|m| m.2)
                        .unwrap()
                }
            };
            let t = texture(device, format, side, side);
            let steps = [
                ("HR", halves(count, |n, c| n + 16384 * c), true),
                ("FR", floats(count, 1), false),
            ];
            let all = (side as usize, side as usize);
            for (tag, values, half) in steps {
                write(device, &queue, &kernels, &t, (&values, half), texels);
                let codes = written_codes(form, (&values, half), side as usize, all);
                check_reads(
                    (&mut wrong, &mut moved),
                    (form, &codes),
                    (format!("{format:?} {tag}"), metal(tag.to_owned())),
                    reads(device, &queue, &kernels, &t, texels),
                );
            }
            let t = texture(device, format, 64, 4);
            let values = halves(GRID, |n, c| 0x3000 + n + 2048 * c);
            write(
                device,
                &queue,
                &kernels,
                &t,
                (&values, true),
                Dispatch::Consumer,
            );
            let codes = written_codes(form, (&values, true), GRID_WIDTH as usize, (64, 4));
            check_reads(
                (&mut wrong, &mut moved),
                (form, &codes),
                (format!("{format:?} OR"), metal("OR".to_owned())),
                reads(device, &queue, &kernels, &t, Dispatch::Consumer),
            );
        }
        println!(
            "XR-TWO-PLANES {}: {moved} texels' reads within the device's conversion of Metal's",
            device.name()
        );
        report(device, &wrong);
        assert_announced(messages, moved, "the two-plane textures' reads");
    });
}

/// The kinds a heap's two-plane texture announces: where it lies apart from the heap's bytes (one
/// whose moves pass the device's limits), and where its first level's planes lie in them and its
/// later levels apart (a mipmapped one).
pub(super) const HEAP_PLANES: [Degradation; 2] = [
    Degradation::HeapTwoPlaneTexture,
    Degradation::HeapTwoPlaneLayout,
];

/// A texture of each two-plane format in an automatic and a placement heap of each storage mode,
/// at the size and alignment the device answers, reads back Metal's values in the consumer's
/// shape as a device texture does: a one-level 2D one and a mipmapped one, shadows of their first
/// level's planes in the heap's bytes, announce `shadowed-texture`, the mipmapped one
/// `heap-two-plane-layout` once too; a 3D one of two depth slices, a shadow of its first slice's
/// planes there, never `heap-two-plane-texture`.
#[test]
fn two_plane_heap_textures_read_and_write_as_metal_does() {
    on_device(|device, messages| {
        let queue = device.new_command_queue().unwrap();
        let kernels = Kernels::new(device);
        let (mut wrong, mut moved) = (Vec::new(), 0);
        for format in [PixelFormat::RGB10A82PXR10, PixelFormat::RGB10A82PXR10SRGB] {
            let form = ExtendedRange::of(format).expect("a two-plane format");
            let metal = |lanes: &str| {
                TEXTURES
                    .iter()
                    .find(|m| m.0 == format && m.1 == format!("OR{lanes}"))
                    .map(|m| m.2)
                    .unwrap()
            };
            for options in [ResourceOptions::private(), ResourceOptions::shared()] {
                let mut descriptor = TextureDescriptor::texture_2d(format, 64, 4, false);
                descriptor.usage = TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE;
                descriptor.resource_options = options;
                let answer = device
                    .heap_texture_size_and_align(&descriptor)
                    .unwrap_or_else(|e| panic!("{format:?} {options:?} size and align: {e}"));
                assert!(answer.size >= 64 * 4 * 5 && answer.align.is_power_of_two());
                let mut mipmapped = descriptor;
                mipmapped.mipmap_level_count = 2;
                let kinds = [
                    (HeapType::Automatic, descriptor),
                    (HeapType::Placement, descriptor),
                    (HeapType::Placement, mipmapped),
                ];
                for (heap_type, descriptor) in kinds {
                    let answer = device.heap_texture_size_and_align(&descriptor).unwrap();
                    let heap = device
                        .new_heap(&HeapDescriptor::new(heap_type, answer.size, options))
                        .unwrap();
                    let t = match heap_type {
                        HeapType::Placement => heap.new_texture_at_offset(&descriptor, 0),
                        _ => heap.new_texture(&descriptor),
                    }
                    .unwrap_or_else(|e| panic!("{format:?} {options:?} {heap_type:?}: {e}"));
                    assert_eq!(t.pixel_format(), format);
                    let values = halves(GRID, |n, c| 0x3000 + n + 2048 * c);
                    write(
                        device,
                        &queue,
                        &kernels,
                        &t,
                        (&values, true),
                        Dispatch::Consumer,
                    );
                    let codes = written_codes(form, (&values, true), GRID_WIDTH as usize, (64, 4));
                    check_reads(
                        (&mut wrong, &mut moved),
                        (form, &codes),
                        (format!("{format:?} {options:?} {heap_type:?} OR"), metal),
                        reads(device, &queue, &kernels, &t, Dispatch::Consumer),
                    );
                }
            }
        }
        report(device, &wrong);
        let notices = counts(&messages.take_notices());
        assert_counted(&notices, moved, "the heap textures' reads");
        assert_eq!(notices.get(&HEAP_PLANES[0]), None, "{notices:?}");
        assert_eq!(
            notices.get(&HEAP_PLANES[1]).copied(),
            Some(1),
            "{notices:?}"
        );
        assert!(
            notices.contains_key(&Degradation::ShadowedTexture),
            "{notices:?}"
        );
        // A 3D texture of two depth slices keeps its first slice's planes in the heap's bytes
        // (`a_two_plane_volume_lies_over_its_first_slice_as_metal_s`).
        let mut volume = TextureDescriptor::texture_2d(PixelFormat::RGB10A82PXR10, 16, 8, false);
        volume.texture_type = TextureType::Type3D;
        volume.depth = 2;
        volume.usage = TextureUsage::SHADER_READ;
        volume.resource_options = ResourceOptions::private();
        let answer = device.heap_texture_size_and_align(&volume).unwrap();
        let heap = device
            .new_heap(&HeapDescriptor::new(
                HeapType::Placement,
                answer.size,
                ResourceOptions::private(),
            ))
            .unwrap();
        heap.new_texture_at_offset(&volume, 0).unwrap();
        let notices = counts(&messages.take_notices());
        assert_eq!(notices.get(&HEAP_PLANES[0]), None, "{notices:?}");
    });
}

/// The AIR module of [`a_two_plane_volume_lies_over_its_first_slice_as_metal_s`]'s kernels, over
/// a `texture3d<float>` at texture 0, a buffer at 0, and the grid's `w` and `h` (`uint`s) at buffer
/// 1, one thread per texel: `xr2pv_read` reads texel `g` into `out[(g.z h + g.y) w + g.x]`,
/// `xr2pv_write` writes `in[...]` there, and `xr2pv_queries` stores `get_width(l)`,
/// `get_height(l)`, `get_depth(l)` and `get_num_mip_levels()` for levels `l` 0 to 3 as `uint4`s.
const VOLUME_KERNELS: &str = r#"target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

define void @xr2pv_read(ptr addrspace(1) %t, ptr addrspace(1) %out, ptr addrspace(1) %shape, <3 x i32> %g) {
  %i = call i64 @texel(ptr addrspace(1) %shape, <3 x i32> %g)
  %rs = call ptr addrspace(2) @air.get_read_sampler()
  %r = call { <4 x float>, i8 } @air.read_texture_3d.v4f32(ptr addrspace(1) %t, ptr addrspace(2) %rs, <3 x i32> %g, <3 x i32> zeroinitializer, i32 0, i32 1)
  %v = extractvalue { <4 x float>, i8 } %r, 0
  %p = getelementptr inbounds <4 x float>, ptr addrspace(1) %out, i64 %i
  store <4 x float> %v, ptr addrspace(1) %p, align 16
  ret void
}

define void @xr2pv_write(ptr addrspace(1) %t, ptr addrspace(1) %in, ptr addrspace(1) %shape, <3 x i32> %g) {
  %i = call i64 @texel(ptr addrspace(1) %shape, <3 x i32> %g)
  %p = getelementptr inbounds <4 x float>, ptr addrspace(1) %in, i64 %i
  %v = load <4 x float>, ptr addrspace(1) %p, align 16
  call void @air.write_texture_3d.v4f32(ptr addrspace(1) %t, <3 x i32> %g, <4 x float> %v, i32 0, i32 2)
  ret void
}

define void @xr2pv_queries(ptr addrspace(1) %t, ptr addrspace(1) %out, ptr addrspace(1) %shape, <3 x i32> %g) {
  %n = call i32 @air.get_num_mip_levels_texture_3d(ptr addrspace(1) %t)
  call void @level(ptr addrspace(1) %t, ptr addrspace(1) %out, i32 0, i32 %n)
  call void @level(ptr addrspace(1) %t, ptr addrspace(1) %out, i32 1, i32 %n)
  call void @level(ptr addrspace(1) %t, ptr addrspace(1) %out, i32 2, i32 %n)
  call void @level(ptr addrspace(1) %t, ptr addrspace(1) %out, i32 3, i32 %n)
  ret void
}

define internal void @level(ptr addrspace(1) %t, ptr addrspace(1) %out, i32 %l, i32 %n) alwaysinline {
  %w = call i32 @air.get_width_texture_3d(ptr addrspace(1) %t, i32 %l)
  %h = call i32 @air.get_height_texture_3d(ptr addrspace(1) %t, i32 %l)
  %d = call i32 @air.get_depth_texture_3d(ptr addrspace(1) %t, i32 %l)
  %q0 = insertelement <4 x i32> zeroinitializer, i32 %w, i64 0
  %q1 = insertelement <4 x i32> %q0, i32 %h, i64 1
  %q2 = insertelement <4 x i32> %q1, i32 %d, i64 2
  %q3 = insertelement <4 x i32> %q2, i32 %n, i64 3
  %li = zext i32 %l to i64
  %p = getelementptr inbounds <4 x i32>, ptr addrspace(1) %out, i64 %li
  store <4 x i32> %q3, ptr addrspace(1) %p, align 16
  ret void
}

define internal i64 @texel(ptr addrspace(1) %shape, <3 x i32> %g) alwaysinline {
  %w = load i32, ptr addrspace(1) %shape, align 4
  %hp = getelementptr inbounds i32, ptr addrspace(1) %shape, i64 1
  %h = load i32, ptr addrspace(1) %hp, align 4
  %x = extractelement <3 x i32> %g, i64 0
  %y = extractelement <3 x i32> %g, i64 1
  %z = extractelement <3 x i32> %g, i64 2
  %zh = mul i32 %z, %h
  %zy = add i32 %zh, %y
  %row = mul i32 %zy, %w
  %n = add i32 %row, %x
  %i = zext i32 %n to i64
  ret i64 %i
}

declare ptr addrspace(2) @air.get_read_sampler()
declare { <4 x float>, i8 } @air.read_texture_3d.v4f32(ptr addrspace(1), ptr addrspace(2), <3 x i32>, <3 x i32>, i32, i32)
declare void @air.write_texture_3d.v4f32(ptr addrspace(1), <3 x i32>, <4 x float>, i32, i32)
declare i32 @air.get_width_texture_3d(ptr addrspace(1), i32)
declare i32 @air.get_height_texture_3d(ptr addrspace(1), i32)
declare i32 @air.get_depth_texture_3d(ptr addrspace(1), i32)
declare i32 @air.get_num_mip_levels_texture_3d(ptr addrspace(1))

!air.kernel = !{!0, !10, !20}
!0 = !{ptr @xr2pv_read, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5, !6}
!3 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.read", !"air.arg_type_name", !"texture3d<float, read>", !"air.arg_name", !"t"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4", !"air.arg_name", !"out"}
!5 = !{i32 2, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"shape"}
!6 = !{i32 3, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint3", !"air.arg_name", !"g"}
!10 = !{ptr @xr2pv_write, !1, !12}
!12 = !{!13, !4, !5, !6}
!13 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.write", !"air.arg_type_name", !"texture3d<float, write>", !"air.arg_name", !"t"}
!20 = !{ptr @xr2pv_queries, !1, !22}
!22 = !{!3, !24, !5, !6}
!24 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"uint4", !"air.arg_name", !"out"}
"#;

/// The pipeline of [`VOLUME_KERNELS`]'s kernel `name`, its module assembled in a scratch
/// directory removed at once.
fn volume_state(device: &Device, name: &str) -> ComputePipelineState {
    let dir = std::env::temp_dir().join(format!(
        "m2v-two-plane-volumes-{}-{name}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let ll = dir.join("volumes.ll");
    std::fs::write(&ll, VOLUME_KERNELS).unwrap();
    let bitcode = llvm_assemble(&ll);
    std::fs::remove_dir_all(&dir).unwrap();
    let library = device.new_library_with_air(&bitcode.unwrap()).unwrap();
    let function = library.new_function_with_name(name).unwrap();
    device
        .new_compute_pipeline_state(
            &ComputePipelineDescriptor::new(StageSource::Function(function.function_descriptor())),
            &PipelineOptions::default(),
        )
        .unwrap_or_else(|e| panic!("{name}: {e}"))
}

/// Runs [`VOLUME_KERNELS`]'s `state` over `texture` with `buffer` at 0, one thread per texel of
/// `[w, h, d]` (one thread where `one`).
fn run_volume(
    device: &Device,
    queue: &CommandQueue,
    state: &ComputePipelineState,
    (texture, buffer): (&Texture, &Buffer),
    [w, h, d]: [u64; 3],
    one: bool,
) {
    let shape: Vec<u8> = [w as u32, h as u32]
        .iter()
        .flat_map(|word| word.to_le_bytes())
        .collect();
    let shape = buffer_with(device, &shape).unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(state).unwrap();
    encoder.set_texture(Some(texture), 0).unwrap();
    encoder.set_buffer(Some(buffer), 0, 0).unwrap();
    encoder.set_buffer(Some(&shape), 0, 1).unwrap();
    let grid = match one {
        true => Size::new(1, 1, 1),
        false => Size::new(w, h, d),
    };
    encoder.dispatch_threads(grid, Size::new(4, 4, 4)).unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    assert_eq!(command_buffer.error(), None);
}

/// A one-level 3D texture of `RGB10A8_2P_XR10` in a `Shared` placement heap lies over its first
/// slice's planes as Metal's does (an Apple M3 Max under macOS 27.0.1, probes of a 16 x 8 x 4
/// texture, `Shared` and `Private`, in a heap and of its own memory): Metal answers the 2D
/// texture's size and alignment; every texel's read is texel `(0, 0, 0)`'s, the read of a 2D
/// texture over the same bytes at its `(0, 0)`; a write at `(x, y, z)` lands where the 2D
/// texture's write at `(x, y)` does; a blit of two images from a buffer at `z` 1 leaves the second
/// in the first slice's words, and one into a buffer reads texel `(0, 0, 0)`'s word at every
/// texel; `replaceRegion` of two images at `z` 1 writes the first there, and `getBytes` reads it
/// back into the first image, the second as it was; the queries answer its width and height, a
/// depth of `max(1, ⌈w / 4⌉ >> l)` at levels 0 to 3 and one level. The device announces
/// `shadowed-texture`, never `heap-two-plane-texture`.
#[test]
fn a_two_plane_volume_lies_over_its_first_slice_as_metal_s() {
    on_device(|device, messages| {
        let queue = device.new_command_queue().unwrap();
        let options = ResourceOptions::shared();
        let heap = device
            .new_heap(&HeapDescriptor::new(
                HeapType::Placement,
                PLANES_HEAP,
                options,
            ))
            .unwrap();
        let bytes = heap.new_buffer_at_offset(PLANES_HEAP, options, 0).unwrap();
        let [w, h, d] = [16u64, 8, 4];
        let usage = TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE;
        let mut descriptor = TextureDescriptor::texture_2d(PixelFormat::RGB10A82PXR10, w, h, false);
        descriptor.usage = usage;
        descriptor.resource_options = options;
        let flat_answer = device.heap_texture_size_and_align(&descriptor).unwrap();
        let flat_at = 0x8000;
        let flat = heap.new_texture_at_offset(&descriptor, flat_at).unwrap();
        descriptor.texture_type = TextureType::Type3D;
        descriptor.depth = d;
        let answer = device.heap_texture_size_and_align(&descriptor).unwrap();
        assert_eq!(answer, flat_answer, "the 2D texture's answer");
        let volume = heap.new_texture_at_offset(&descriptor, 0).unwrap();
        let span = answer.size as usize;
        let kernels = Kernels::new(device);
        let [read, written, queries] =
            ["xr2pv_read", "xr2pv_write", "xr2pv_queries"].map(|name| volume_state(device, name));
        let texels = (w * h * d) as usize;
        // Both textures' planes hold the same bytes; every read of the volume is the flat
        // texture's read of (0, 0).
        let mut pattern: Vec<u8> = (0..PLANES_HEAP).map(|i| (i * 89 + 7) as u8).collect();
        let first: Vec<u8> = pattern[..span].to_vec();
        pattern[flat_at as usize..flat_at as usize + span].copy_from_slice(&first);
        put(device, &queue, &bytes, &pattern);
        let out = buffer_with(device, &vec![0xab; 16 * texels]).unwrap();
        run_volume(device, &queue, &read, (&volume, &out), [w, h, d], false);
        let reads = read_buffer(&out, 0, 16 * texels as u64).unwrap();
        let [_, flat_reads] = super::extended_range_two_planes::reads(
            device,
            &queue,
            &kernels,
            &flat,
            Dispatch::Texels(w, h),
        );
        for n in 0..texels {
            assert_eq!(
                reads[16 * n..16 * n + 16],
                flat_reads[..16],
                "texel {n}'s read"
            );
        }
        // A write at (x, y, z) lands as the flat texture's at (x, y).
        let value = |x: u64, y: u64| -> [f32; 4] {
            [
                x as f32 / 16.0 - 0.5,
                y as f32 / 8.0,
                0.25,
                (x + y) as f32 / 32.0,
            ]
        };
        let values = |depth: u64| -> Vec<u8> {
            (0..w * h * depth)
                .flat_map(|n| value(n % w, n / w % h))
                .flat_map(f32::to_le_bytes)
                .collect()
        };
        let input = buffer_with(device, &values(d)).unwrap();
        run_volume(
            device,
            &queue,
            &written,
            (&volume, &input),
            [w, h, d],
            false,
        );
        write(
            device,
            &queue,
            &kernels,
            &flat,
            (&values(1), false),
            Dispatch::Texels(w, h),
        );
        let got = get(device, &queue, &bytes, PLANES_HEAP);
        assert!(
            got[..span] == got[flat_at as usize..flat_at as usize + span],
            "the volume's writes land where the flat texture's do"
        );
        assert!(
            got[span..flat_at as usize] == pattern[span..flat_at as usize],
            "and nowhere past its planes"
        );
        // A blit of two images from a buffer at z = 1: the second stays in the first slice.
        let image = (4 * w * h) as usize;
        let images: Vec<u8> = (0..2 * image).map(|i| (i * 7 + 3) as u8).collect();
        let source = buffer_with(device, &images).unwrap();
        blit_once(&queue, |blit| {
            blit.copy_from_buffer_to_texture(
                &source,
                0,
                4 * w,
                4 * w * h,
                Size::new(w, h, 2),
                &volume,
                0,
                0,
                Origin::new(0, 0, 1),
                BlitOption::None,
            )
        });
        let got = get(device, &queue, &bytes, PLANES_HEAP);
        assert!(got[..image] == images[image..], "the second image's words");
        // A blit of two images into a buffer at z = 1 reads texel (0, 0, 0)'s word everywhere.
        let target = buffer_with(device, &vec![0; 2 * image]).unwrap();
        blit_once(&queue, |blit| {
            blit.copy_from_texture_to_buffer(
                &volume,
                0,
                0,
                Origin::new(0, 0, 1),
                Size::new(w, h, 2),
                &target,
                0,
                4 * w,
                4 * w * h,
                BlitOption::None,
            )
        });
        let words = read_buffer(&target, 0, 2 * image as u64).unwrap();
        for (n, word) in words.chunks(4).enumerate() {
            assert_eq!(word, &got[..4], "word {n} of the blit out");
        }
        // CPU copies of two images at z = 1: the first image alone, at the first slice.
        let host: Vec<u8> = (0..2 * image).map(|i| (i * 11 + 5) as u8).collect();
        volume
            .replace_region(
                Region::new_3d(0, 0, 1, w, h, 2),
                0,
                0,
                &host,
                4 * w,
                4 * w * h,
            )
            .unwrap();
        let got = get(device, &queue, &bytes, PLANES_HEAP);
        assert!(got[..image] == host[..image], "replaceRegion's first image");
        let mut back = vec![0xab; 2 * image];
        volume
            .get_bytes(
                &mut back,
                4 * w,
                4 * w * h,
                Region::new_3d(0, 0, 1, w, h, 2),
                0,
                0,
            )
            .unwrap();
        assert!(back[..image] == host[..image], "getBytes' first image");
        assert!(
            back[image..].iter().all(|&b| b == 0xab),
            "getBytes' second image as it was"
        );
        // The queries.
        let out = buffer_with(device, &[0xab; 64]).unwrap();
        run_volume(device, &queue, &queries, (&volume, &out), [w, h, d], true);
        let words: Vec<u32> = read_buffer(&out, 0, 64)
            .unwrap()
            .chunks(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        // A width or height query at a level past the texture's last reads the last level here
        // (the translator's containment), where Metal halves the extent again: only the depth
        // and the level count are compared there.
        assert_eq!(words[..2], [w as u32, h as u32], "width and height");
        let depths: Vec<[u32; 2]> = (0..4)
            .map(|l| [words[4 * l + 2], words[4 * l + 3]])
            .collect();
        let want: Vec<[u32; 2]> = (0..4)
            .map(|l| [(w.div_ceil(4) >> l).max(1) as u32, 1])
            .collect();
        assert_eq!(depths, want, "depth and levels at levels 0 to 3");
        let notices = counts(&messages.take_notices());
        assert!(
            notices.contains_key(&Degradation::ShadowedTexture),
            "{notices:?}"
        );
        assert_eq!(notices.get(&HEAP_PLANES[0]), None, "{notices:?}");
    });
}

/// The size of the heaps of [`two_plane_heap_textures_keep_their_texels_in_the_heap_s_bytes`],
/// the texture's offset in them, and its extent.
const PLANES_HEAP: u64 = 64 << 10;
const PLANES_AT: u64 = 48;
const PLANES_EXTENT: [u64; 2] = [37, 5];

/// Where a heap's one-level 2D two-plane texture of `[w, h]` texels keeps its planes, from its
/// first byte, as an Apple M4 lays them (`Shared` and `Private` alike): plane
/// 0's rows `align(4 w, 16)` bytes apart, and plane 1 from `align(h · align(4 w, 16), 16384)`,
/// its rows `align(w, 16)` apart.
fn heap_planes([w, h]: [u64; 2]) -> (u64, u64, u64) {
    let color = (4 * w).next_multiple_of(16);
    (
        color,
        (h * color).next_multiple_of(16 << 10),
        w.next_multiple_of(16),
    )
}

/// `bytes` into `buffer` from its start: through a blit from a shared buffer where it is
/// `Private`.
pub(super) fn put(device: &Device, queue: &CommandQueue, buffer: &Buffer, bytes: &[u8]) {
    let staging = buffer_with(device, bytes).unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.copy_from_buffer_to_buffer(&staging, 0, buffer, 0, bytes.len() as u64)
        .unwrap();
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
}

/// `buffer`'s first `length` bytes, through a blit into a shared buffer.
pub(super) fn get(device: &Device, queue: &CommandQueue, buffer: &Buffer, length: u64) -> Vec<u8> {
    let staging = buffer_with(device, &vec![0; length as usize]).unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.copy_from_buffer_to_buffer(buffer, 0, &staging, 0, length)
        .unwrap();
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    read_buffer(&staging, 0, length).unwrap()
}

/// A one-level 2D texture of each two-plane format in a placement heap of each storage mode keeps
/// its texels in the heap's bytes where Metal does, as a buffer over the same
/// bytes sees them while the texture lives: a kernel's writes land on its two planes' rows, each
/// color word with bits 30 and 31 clear, and on no other byte; bytes written into the planes
/// reach the texture's reads; and its CPU copies move plane 0's words there. A texture of the
/// device's own, written alike, holds the same color words and alpha bytes, and, given the same
/// color words, reads the same colors. The device announces `shadowed-texture`, and neither
/// `heap-two-plane-texture` nor `heap-two-plane-layout`.
#[test]
fn two_plane_heap_textures_keep_their_texels_in_the_heap_s_bytes() {
    on_device(|device, messages| {
        let queue = device.new_command_queue().unwrap();
        let kernels = Kernels::new(device);
        let [w, h] = PLANES_EXTENT;
        let (color, alpha, alpha_row) = heap_planes(PLANES_EXTENT);
        let texels = (w * h) as usize;
        let dispatch = Dispatch::Texels(w, h);
        let at = |x: u64, y: u64| (PLANES_AT + y * color + 4 * x) as usize;
        let alpha_at = |x: u64, y: u64| (PLANES_AT + alpha + y * alpha_row + x) as usize;
        let alpha_of = |floats: &[u8], t: usize| {
            let a = f32::from_le_bytes(floats[16 * t + 12..16 * t + 16].try_into().unwrap());
            (a * 255.0).round() as u8
        };
        for format in [PixelFormat::RGB10A82PXR10, PixelFormat::RGB10A82PXR10SRGB] {
            let reference = texture(device, format, w, h);
            for options in [ResourceOptions::shared(), ResourceOptions::private()] {
                let what = format!("{format:?} {:?}", options.storage_mode().unwrap());
                let heap: Heap = device
                    .new_heap(&HeapDescriptor::new(
                        HeapType::Placement,
                        PLANES_HEAP,
                        options,
                    ))
                    .unwrap();
                let mut descriptor = TextureDescriptor::texture_2d(format, w, h, false);
                descriptor.usage = TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE;
                descriptor.resource_options = options;
                let t = heap
                    .new_texture_at_offset(&descriptor, PLANES_AT)
                    .unwrap_or_else(|e| panic!("{what}: {e}"));
                let bytes = heap
                    .new_buffer_at_offset(PLANES_HEAP, options, 0)
                    .unwrap_or_else(|e| panic!("{what}: a buffer over the texture: {e}"));
                // A kernel's writes land on the planes' rows alone.
                put(device, &queue, &bytes, &vec![0xee; PLANES_HEAP as usize]);
                let values = halves(texels, |n, c| 0x3100 + 37 * n + 2311 * c);
                write(device, &queue, &kernels, &t, (&values, true), dispatch);
                write(
                    device,
                    &queue,
                    &kernels,
                    &reference,
                    (&values, true),
                    dispatch,
                );
                let mut words = vec![0; 4 * texels];
                reference
                    .get_bytes(&mut words, 4 * w, 0, Region::new_2d(0, 0, w, h), 0, 0)
                    .unwrap();
                let [_, floats] = reads(device, &queue, &kernels, &reference, dispatch);
                let mut want = vec![0xee; PLANES_HEAP as usize];
                for y in 0..h {
                    for x in 0..w {
                        let i = (y * w + x) as usize;
                        want[at(x, y)..at(x, y) + 4].copy_from_slice(&words[4 * i..4 * i + 4]);
                        want[alpha_at(x, y)] = alpha_of(&floats, i);
                        assert_eq!(words[4 * i + 3] >> 6, 0, "{what}: bits 30 and 31 of {i}");
                    }
                }
                let got = get(device, &queue, &bytes, PLANES_HEAP);
                let differ = (0..got.len()).filter(|&k| got[k] != want[k]).count();
                assert_eq!(
                    differ, 0,
                    "{what}: {differ} bytes of the heap differ after writes"
                );
                // Bytes written into the planes reach the texture's reads.
                let pattern: Vec<u8> = (0..PLANES_HEAP).map(|k| (k * 131 + 7) as u8).collect();
                put(device, &queue, &bytes, &pattern);
                let [_, got] = reads(device, &queue, &kernels, &t, dispatch);
                let mut words = vec![0; 4 * texels];
                for y in 0..h {
                    for x in 0..w {
                        let i = (y * w + x) as usize;
                        words[4 * i..4 * i + 4].copy_from_slice(&pattern[at(x, y)..at(x, y) + 4]);
                        assert_eq!(
                            alpha_of(&got, i),
                            pattern[alpha_at(x, y)],
                            "{what}: alpha {i}"
                        );
                    }
                }
                reference
                    .replace_region(Region::new_2d(0, 0, w, h), 0, 0, &words, 4 * w, 0)
                    .unwrap();
                let [_, colors] = reads(device, &queue, &kernels, &reference, dispatch);
                for i in 0..texels {
                    assert_eq!(
                        got[16 * i..16 * i + 12],
                        colors[16 * i..16 * i + 12],
                        "{what}: colors of texel {i}"
                    );
                }
                // The CPU copies plane 0's words where they lie in the heap.
                if options.storage_mode().unwrap() == StorageMode::Shared {
                    let mut read = vec![0; 4 * texels];
                    t.get_bytes(&mut read, 4 * w, 0, Region::new_2d(0, 0, w, h), 0, 0)
                        .unwrap();
                    assert_eq!(read, words, "{what}: get_bytes");
                    let replaced: Vec<u8> = (0..4 * texels).map(|k| (k * 7 + 1) as u8).collect();
                    t.replace_region(Region::new_2d(0, 0, w, h), 0, 0, &replaced, 4 * w, 0)
                        .unwrap();
                    let got = read_buffer(&bytes, 0, PLANES_HEAP).unwrap();
                    let mut want = pattern.clone();
                    for y in 0..h {
                        for x in 0..w {
                            let i = (y * w + x) as usize;
                            want[at(x, y)..at(x, y) + 4]
                                .copy_from_slice(&replaced[4 * i..4 * i + 4]);
                        }
                    }
                    assert!(got == want, "{what}: replace_region");
                }
            }
        }
        let notices = counts(&messages.take_notices());
        assert!(
            notices.contains_key(&Degradation::ShadowedTexture),
            "{notices:?}"
        );
        for kind in HEAP_PLANES {
            assert!(!notices.contains_key(&kind), "{notices:?}");
        }
    });
}

/// The fixture `xr_two_plane_arrays`'s kernel `name`.
fn array_state(device: &Device, name: &str) -> ComputePipelineState {
    let bitcode = llvm_assemble(&engine_dir().join("xr_two_plane_arrays.ll")).unwrap();
    let library = device.new_library_with_air(&bitcode).unwrap();
    let function = library.new_function_with_name(name).unwrap();
    device
        .new_compute_pipeline_state(
            &ComputePipelineDescriptor::new(StageSource::Function(function.function_descriptor())),
            &PipelineOptions::default(),
        )
        .unwrap_or_else(|e| panic!("{name}: {e}"))
}

/// Runs the array kernel `state` over slice `slice` of `texture`, a `[w, h]` texture array, with
/// `buffer` at 0, one thread per texel.
fn run_slice(
    device: &Device,
    queue: &CommandQueue,
    state: &ComputePipelineState,
    (texture, buffer): (&Texture, &Buffer),
    [w, h]: [u64; 2],
    slice: u64,
) {
    let shape: Vec<u8> = [w as u32, slice as u32]
        .iter()
        .flat_map(|word| word.to_le_bytes())
        .collect();
    let shape = buffer_with(device, &shape).unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(state).unwrap();
    encoder.set_texture(Some(texture), 0).unwrap();
    encoder.set_buffer(Some(buffer), 0, 0).unwrap();
    encoder.set_buffer(Some(&shape), 0, 1).unwrap();
    encoder
        .dispatch_threads(Size::new(w, h, 1), Size::new(32, 32, 1))
        .unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    assert_eq!(command_buffer.error(), None);
}

/// The extent, slices and offset of the arrays of
/// [`two_plane_heap_arrays_keep_their_slices_in_the_heap_s_bytes`], and their slices apart in the
/// heap's bytes: `align(h · align(4 w, 16), 128)`.
const ARRAY_EXTENT: [u64; 2] = [37, 5];
const ARRAY_SLICES: u64 = 2;
const ARRAY_AT: u64 = 128;
const ARRAY_SLICE: u64 = 0x380;

/// A 2D array of each two-plane format in a placement heap of each storage mode keeps its slices'
/// color words in the heap's bytes as Metal does (an Apple M3 Max under macOS 27.0.1, a probe of
/// kernel writes and reads, `replaceRegion` and `getBytes` through a buffer over the heap's bytes):
/// slice `s`'s rows as a 2D texture's, `align(h · align(4 w, 16), 128)` bytes past slice `s - 1`'s,
/// and no alpha plane. A kernel's writes of a slice land on that slice's color words alone, as a
/// texture of the device's own written alike holds them, and no other byte; words written there
/// reach the texture's reads, whose alpha reads 0; the CPU copies move each slice's words there.
/// The device announces `shadowed-texture` and neither `heap-two-plane-texture` nor
/// `heap-two-plane-layout`.
#[test]
fn two_plane_heap_arrays_keep_their_slices_in_the_heap_s_bytes() {
    on_device(|device, messages| {
        let queue = device.new_command_queue().unwrap();
        let write = array_state(device, "xr2pa_write_float");
        let read = array_state(device, "xr2pa_read_float");
        let kernels = Kernels::new(device);
        let [w, h] = ARRAY_EXTENT;
        let texels = (w * h) as usize;
        let color = (4 * w).next_multiple_of(16);
        let at = |s: u64, x: u64, y: u64| (ARRAY_AT + s * ARRAY_SLICE + y * color + 4 * x) as usize;
        for format in [PixelFormat::RGB10A82PXR10, PixelFormat::RGB10A82PXR10SRGB] {
            let reference = texture(device, format, w, h);
            for options in [ResourceOptions::shared(), ResourceOptions::private()] {
                let what = format!("{format:?} {:?}", options.storage_mode().unwrap());
                let mut descriptor = TextureDescriptor::texture_2d(format, w, h, false);
                descriptor.texture_type = TextureType::Type2DArray;
                descriptor.array_length = ARRAY_SLICES;
                descriptor.usage = TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE;
                descriptor.resource_options = options;
                let answer = device.heap_texture_size_and_align(&descriptor).unwrap();
                // Metal's answer: plane 0's slices rounded up to 16 KiB, then plane 1's, at 128.
                assert_eq!((answer.size, answer.align), (0x4200, 128), "{what}");
                let heap: Heap = device
                    .new_heap(&HeapDescriptor::new(
                        HeapType::Placement,
                        PLANES_HEAP,
                        options,
                    ))
                    .unwrap();
                let t = heap
                    .new_texture_at_offset(&descriptor, ARRAY_AT)
                    .unwrap_or_else(|e| panic!("{what}: {e}"));
                let bytes = heap.new_buffer_at_offset(PLANES_HEAP, options, 0).unwrap();
                for slice in 0..ARRAY_SLICES {
                    // A kernel's writes of the slice land on its color words alone.
                    put(device, &queue, &bytes, &vec![0xee; PLANES_HEAP as usize]);
                    let values = floats(texels, 37 + 11 * slice as usize);
                    let input = buffer_with(device, &values).unwrap();
                    run_slice(device, &queue, &write, (&t, &input), [w, h], slice);
                    super::extended_range_two_planes::write(
                        device,
                        &queue,
                        &kernels,
                        &reference,
                        (&values, false),
                        Dispatch::Texels(w, h),
                    );
                    let mut words = vec![0; 4 * texels];
                    reference
                        .get_bytes(&mut words, 4 * w, 0, Region::new_2d(0, 0, w, h), 0, 0)
                        .unwrap();
                    let mut want = vec![0xee; PLANES_HEAP as usize];
                    for y in 0..h {
                        for x in 0..w {
                            let i = (y * w + x) as usize;
                            let o = at(slice, x, y);
                            want[o..o + 4].copy_from_slice(&words[4 * i..4 * i + 4]);
                        }
                    }
                    let got = get(device, &queue, &bytes, PLANES_HEAP);
                    let differ = (0..got.len()).filter(|&k| got[k] != want[k]).count();
                    assert_eq!(differ, 0, "{what}: slice {slice}: {differ} bytes differ");
                    // Words written there reach the slice's reads, its alpha 0.
                    let pattern: Vec<u8> = (0..PLANES_HEAP)
                        .map(|k| (k * 131 + 7 + slice) as u8)
                        .collect();
                    put(device, &queue, &bytes, &pattern);
                    let out = buffer_with(device, &vec![0xab; 16 * texels]).unwrap();
                    run_slice(device, &queue, &read, (&t, &out), [w, h], slice);
                    let got = read_buffer(&out, 0, 16 * texels as u64).unwrap();
                    let mut words = vec![0; 4 * texels];
                    for y in 0..h {
                        for x in 0..w {
                            let i = (y * w + x) as usize;
                            let o = at(slice, x, y);
                            words[4 * i..4 * i + 4].copy_from_slice(&pattern[o..o + 4]);
                        }
                    }
                    reference
                        .replace_region(Region::new_2d(0, 0, w, h), 0, 0, &words, 4 * w, 0)
                        .unwrap();
                    let [_, colors] =
                        reads(device, &queue, &kernels, &reference, Dispatch::Texels(w, h));
                    for i in 0..texels {
                        assert_eq!(
                            got[16 * i..16 * i + 12],
                            colors[16 * i..16 * i + 12],
                            "{what}: slice {slice}: colors of texel {i}"
                        );
                        assert_eq!(
                            got[16 * i + 12..16 * i + 16],
                            0f32.to_le_bytes(),
                            "{what}: slice {slice}: alpha of texel {i}"
                        );
                    }
                    // The CPU copies the slice's words where they lie in the heap.
                    if options.storage_mode().unwrap() == StorageMode::Shared {
                        let mut read = vec![0; 4 * texels];
                        let region = Region::new_2d(0, 0, w, h);
                        t.get_bytes(&mut read, 4 * w, 0, region, 0, slice).unwrap();
                        assert_eq!(read, words, "{what}: slice {slice}: get_bytes");
                        let replaced: Vec<u8> =
                            (0..4 * texels).map(|k| (k * 7 + 1) as u8).collect();
                        t.replace_region(region, 0, slice, &replaced, 4 * w, 0)
                            .unwrap();
                        let got = read_buffer(&bytes, 0, PLANES_HEAP).unwrap();
                        let mut want = pattern.clone();
                        for y in 0..h {
                            for x in 0..w {
                                let i = (y * w + x) as usize;
                                let o = at(slice, x, y);
                                want[o..o + 4].copy_from_slice(&replaced[4 * i..4 * i + 4]);
                            }
                        }
                        assert!(got == want, "{what}: slice {slice}: replace_region");
                    }
                }
            }
        }
        let notices = counts(&messages.take_notices());
        assert!(
            notices.contains_key(&Degradation::ShadowedTexture),
            "{notices:?}"
        );
        for kind in HEAP_PLANES {
            assert!(!notices.contains_key(&kind), "{notices:?}");
        }
    });
}

/// A mipmapped two-plane texture in a placement heap keeps its first level's color words in the
/// heap's bytes where Metal's CPU copies lay them (an Apple M3 Max under macOS 27.0.1: level 0's
/// rows as a one-level texture's), and its second level apart from them: a CPU copy of level 1
/// reads back what it wrote and changes none of the heap's bytes. The device announces
/// `heap-two-plane-layout` once.
#[test]
fn a_mipmapped_two_plane_heap_texture_keeps_its_first_level_in_the_heap_s_bytes() {
    on_device(|device, messages| {
        let [w, h] = [16u64, 8];
        let mut descriptor = TextureDescriptor::texture_2d(PixelFormat::RGB10A82PXR10, w, h, true);
        descriptor.mipmap_level_count = 2;
        descriptor.usage = TextureUsage::SHADER_READ;
        descriptor.resource_options = ResourceOptions::shared();
        let options = ResourceOptions::shared();
        let heap = device
            .new_heap(&HeapDescriptor::new(
                HeapType::Placement,
                PLANES_HEAP,
                options,
            ))
            .unwrap();
        let t = heap.new_texture_at_offset(&descriptor, 0).unwrap();
        let bytes = heap.new_buffer_at_offset(PLANES_HEAP, options, 0).unwrap();
        let before = read_buffer(&bytes, 0, PLANES_HEAP).unwrap();
        let level0: Vec<u8> = (0..4 * w * h).map(|k| (k * 3 + 1) as u8).collect();
        t.replace_region(Region::new_2d(0, 0, w, h), 0, 0, &level0, 4 * w, 0)
            .unwrap();
        let mut want = before.clone();
        want[..(4 * w * h) as usize].copy_from_slice(&level0);
        let got = read_buffer(&bytes, 0, PLANES_HEAP).unwrap();
        assert!(got == want, "level 0's words at its rows (64 bytes apart)");
        let level1: Vec<u8> = (0..w * h).map(|k| (k * 5 + 9) as u8).collect();
        let region = Region::new_2d(0, 0, w / 2, h / 2);
        t.replace_region(region, 1, 0, &level1, 2 * w, 0).unwrap();
        assert!(
            read_buffer(&bytes, 0, PLANES_HEAP).unwrap() == want,
            "level 1 changes none of the heap's bytes"
        );
        let mut read = vec![0; (w * h) as usize];
        t.get_bytes(&mut read, 2 * w, 0, region, 1, 0).unwrap();
        assert_eq!(read, level1, "level 1 reads back");
        let notices = counts(&messages.take_notices());
        assert_eq!(
            notices.get(&HEAP_PLANES[1]).copied(),
            Some(1),
            "{notices:?}"
        );
    });
}

/// A surface a two-plane texture is made over: the texture's extent, plane 0's offset and
/// `bytesPerRow`, plane 1's, and the surface's bytes.
type SurfaceShape = ([u64; 2], [u64; 2], [u64; 2], u64);

/// The surfaces a two-plane texture is made over: its extent, plane 0's offset and
/// `bytesPerRow`, plane 1's, and the surface's bytes. The consumer's `'b3a8'` surface of 384 x
/// 256 (plane 0 at 1536 bytes a row, plane 1 right after it at 512), and one Metal was measured
/// over at rows off 16 bytes (37 x 5 at 260 and 68 bytes a row, plane 1 4 bytes past plane 0;
/// its GPU reads them at 0 by 256 and 1296 by 64).
const SURFACES: [SurfaceShape; 2] = [
    ([384, 256], [0, 1536], [393_216, 512], 524_288),
    ([37, 5], [0, 260], [1304, 68], 1644),
];

/// `'b3a8'`.
const B3A8: u32 = u32::from_be_bytes(*b"b3a8");

/// A texture of `format` over plane 0 of `surface`, a two-plane `'b3a8'` surface laid out as
/// `shape` (an entry of [`SURFACES`]), of `options`.
fn surface_texture(
    surface: &Buffer,
    format: PixelFormat,
    shape: SurfaceShape,
    options: ResourceOptions,
) -> Result<Texture, Error> {
    let ([w, h], [offset, row], [alpha, alpha_row], _) = shape;
    let mut descriptor = TextureDescriptor::texture_2d(format, w, h, false);
    descriptor.usage =
        TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE | TextureUsage::RENDER_TARGET;
    descriptor.resource_options = options;
    let facts = IoSurfaceYCbCr::new(B3A8, 0, [w, h], alpha, alpha_row);
    surface.new_texture_with_iosurface_plane(
        &descriptor,
        IoSurfacePlane::new(offset, row, true).with_ycbcr(facts),
    )
}

/// `Managed` storage with the other modes at their defaults.
fn managed() -> ResourceOptions {
    ResourceOptions::new(
        StorageMode::Managed,
        CpuCacheMode::DefaultCache,
        HazardTrackingMode::Default,
    )
}

/// A `width` x `height` texture of `format` that kernels read and write and passes render into.
fn target(device: &Device, format: PixelFormat, width: u64, height: u64) -> Texture {
    let mut descriptor = TextureDescriptor::texture_2d(format, width, height, false);
    descriptor.usage =
        TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE | TextureUsage::RENDER_TARGET;
    descriptor.resource_options = ResourceOptions::shared();
    device
        .new_texture(&descriptor)
        .unwrap_or_else(|e| panic!("a {format:?} texture: {e}"))
}

/// Records one blit command through `record` and waits for it.
fn blit_once(
    queue: &CommandQueue,
    record: impl FnOnce(&mut BlitCommandEncoder) -> Result<(), Error>,
) {
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    record(&mut blit).unwrap();
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
}

/// A two-plane texture's planes in bytes as Metal's blit addresses them: plane 0's first byte,
/// the alpha plane's, and the rows both are addressed at.
type BlitPlanes = (usize, usize, usize);

/// Metal's blit of a copy of `w` x `h` texels from `source`'s planes into `destination`'s, each
/// at its own blit rows from its first bytes: each row's color words and alpha bytes in turn, a
/// byte read past the source's end 0 and one written past the destination's end dropped.
fn blit_planes(
    source: &[u8],
    from: BlitPlanes,
    destination: &mut [u8],
    to: BlitPlanes,
    [w, h]: [usize; 2],
) {
    let read = |at: usize| source.get(at).copied().unwrap_or(0);
    for r in 0..h {
        for c in 0..w {
            for k in 0..4 {
                let value = read(from.0 + r * from.2 + 4 * c + k);
                if let Some(byte) = destination.get_mut(to.0 + r * to.2 + 4 * c + k) {
                    *byte = value;
                }
            }
            let value = read(from.1 + r * from.2 + c);
            if let Some(byte) = destination.get_mut(to.1 + r * to.2 + c) {
                *byte = value;
            }
        }
    }
}

/// The planes of a `w` x `h` texture of the device's memory as Metal lays them out: `words` in rows
/// of `align(4 w, 16)` bytes, `alphas` from `align(h` rows`, 16384)` in rows of `align(w, 16)`,
/// every other byte 0; and their blit planes (rows `⌊4 w / 16⌋ · 16`).
fn own_planes(words: &[u8], alphas: &[u8], [w, h]: [usize; 2]) -> (Vec<u8>, BlitPlanes) {
    let (color, alpha) = ((4 * w).next_multiple_of(16), w.next_multiple_of(16));
    let alpha_at = (h * color).next_multiple_of(16384);
    let mut memory = vec![0; alpha_at + h * alpha];
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            memory[y * color + 4 * x..y * color + 4 * x + 4]
                .copy_from_slice(&words[4 * i..4 * i + 4]);
            memory[alpha_at + y * alpha + x] = alphas[i];
        }
    }
    (memory, (0, alpha_at, 4 * w / 16 * 16))
}

/// The color words and alpha bytes of a `w` x `h` texture's planes laid out as [`own_planes`].
fn own_texels(memory: &[u8], [w, h]: [usize; 2]) -> (Vec<u8>, Vec<u8>) {
    let (color, alpha) = ((4 * w).next_multiple_of(16), w.next_multiple_of(16));
    let alpha_at = (h * color).next_multiple_of(16384);
    let mut words = Vec::with_capacity(4 * w * h);
    let mut alphas = Vec::with_capacity(w * h);
    for y in 0..h {
        for x in 0..w {
            words.extend_from_slice(&memory[y * color + 4 * x..y * color + 4 * x + 4]);
            alphas.push(memory[alpha_at + y * alpha + x]);
        }
    }
    (words, alphas)
}

/// Clears `texture` through a render pass to `color`.
fn clear(queue: &CommandQueue, texture: &Texture, color: ClearColor) {
    let mut pass = RenderPassDescriptor::new();
    let attachment = &mut pass.color_attachments[0];
    attachment.texture = Some(texture.clone());
    attachment.load_action = LoadAction::Clear;
    attachment.store_action = StoreAction::Store;
    attachment.clear_color = color;
    let command_buffer = queue.command_buffer().unwrap();
    command_buffer
        .render_command_encoder(&pass)
        .unwrap()
        .end_encoding()
        .unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
}

/// A texture of each two-plane format over plane 0 of a two-plane surface, of `Shared` and
/// `Managed` storage, keeps its texels in the surface's two planes where Metal does: a kernel's
/// writes and a clear land on plane 0's and plane 1's GPU rows (their offsets and rows rounded down
/// to 16 bytes) and on no other byte, each color word with bits 30 and 31 clear; bytes written into
/// the planes reach the texture's reads; its CPU copies move plane 0's words at the plane's own
/// offset and rows, and its blits with a buffer at the GPU's, plane 1 untouched. A texture of the
/// device's own, written alike, holds the same color words and alpha bytes. A fill writes plane 0's
/// words at the GPU's rows, and a copy out of the surface into a texture of the device's memory,
/// into the surface from one, and into another surface's planes moves both planes' bytes at Metal's
/// blit rows (plane 0's GPU rows for a surface's, `⌊4 w / 16⌋ · 16` for the device's), past the
/// planes too but not past the surface.
#[test]
fn two_plane_surface_textures_keep_their_texels_in_the_surface_s_planes() {
    on_device(|device, messages| {
        let queue = device.new_command_queue().unwrap();
        let kernels = Kernels::new(device);
        let alpha_of = |floats: &[u8], t: usize| {
            let a = f32::from_le_bytes(floats[16 * t + 12..16 * t + 16].try_into().unwrap());
            (a * 255.0).round() as u8
        };
        for shape in SURFACES {
            let ([w, h], [offset, row], [alpha, alpha_row], length) = shape;
            let texels = (w * h) as usize;
            let dispatch = Dispatch::Texels(w, h);
            let down = |v: u64| v - v % 16;
            let gpu = |x: u64, y: u64| (down(offset) + y * down(row) + 4 * x) as usize;
            let cpu = |x: u64, y: u64| (offset + y * row + 4 * x) as usize;
            let alpha_at = |x: u64, y: u64| (down(alpha) + y * down(alpha_row) + x) as usize;
            for format in [PixelFormat::RGB10A82PXR10, PixelFormat::RGB10A82PXR10SRGB] {
                let reference = target(device, format, w, h);
                for options in [ResourceOptions::shared(), managed()] {
                    let what = format!("{format:?} {w}x{h} {:?}", options.storage_mode().unwrap());
                    let surface = buffer_with(device, &vec![0xee; length as usize]).unwrap();
                    let t = surface_texture(&surface, format, shape, options)
                        .unwrap_or_else(|e| panic!("{what}: {e}"));
                    // A kernel's writes land on the planes' GPU rows alone.
                    let values = halves(texels, |n, c| 0x3100 + 37 * n + 2311 * c);
                    write(device, &queue, &kernels, &t, (&values, true), dispatch);
                    write(
                        device,
                        &queue,
                        &kernels,
                        &reference,
                        (&values, true),
                        dispatch,
                    );
                    let mut words = vec![0; 4 * texels];
                    reference
                        .get_bytes(&mut words, 4 * w, 0, Region::new_2d(0, 0, w, h), 0, 0)
                        .unwrap();
                    let [_, floats] = reads(device, &queue, &kernels, &reference, dispatch);
                    let mut want = vec![0xee; length as usize];
                    for y in 0..h {
                        for x in 0..w {
                            let i = (y * w + x) as usize;
                            want[gpu(x, y)..gpu(x, y) + 4]
                                .copy_from_slice(&words[4 * i..4 * i + 4]);
                            want[alpha_at(x, y)] = alpha_of(&floats, i);
                            assert_eq!(words[4 * i + 3] >> 6, 0, "{what}: bits 30 and 31 of {i}");
                        }
                    }
                    let got = read_buffer(&surface, 0, length).unwrap();
                    let differ = (0..got.len()).filter(|&k| got[k] != want[k]).count();
                    assert_eq!(differ, 0, "{what}: {differ} bytes differ after writes");
                    // A clear lands on the same bytes as the reference's clear.
                    let color = ClearColor::new(1.25, -0.25, 0.5, 0.75);
                    clear(&queue, &t, color);
                    clear(&queue, &reference, color);
                    reference
                        .get_bytes(&mut words, 4 * w, 0, Region::new_2d(0, 0, w, h), 0, 0)
                        .unwrap();
                    let [_, floats] = reads(device, &queue, &kernels, &reference, dispatch);
                    for y in 0..h {
                        for x in 0..w {
                            let i = (y * w + x) as usize;
                            want[gpu(x, y)..gpu(x, y) + 4]
                                .copy_from_slice(&words[4 * i..4 * i + 4]);
                            want[alpha_at(x, y)] = alpha_of(&floats, i);
                        }
                    }
                    let got = read_buffer(&surface, 0, length).unwrap();
                    let differ = (0..got.len()).filter(|&k| got[k] != want[k]).count();
                    assert_eq!(differ, 0, "{what}: {differ} bytes differ after a clear");
                    // Bytes written into the planes reach the texture's reads.
                    let pattern: Vec<u8> = (0..length).map(|k| (k * 131 + 7) as u8).collect();
                    put(device, &queue, &surface, &pattern);
                    let [_, got] = reads(device, &queue, &kernels, &t, dispatch);
                    let mut words = vec![0; 4 * texels];
                    for y in 0..h {
                        for x in 0..w {
                            let i = (y * w + x) as usize;
                            words[4 * i..4 * i + 4]
                                .copy_from_slice(&pattern[gpu(x, y)..gpu(x, y) + 4]);
                            assert_eq!(
                                alpha_of(&got, i),
                                pattern[alpha_at(x, y)],
                                "{what}: alpha {i}"
                            );
                        }
                    }
                    reference
                        .replace_region(Region::new_2d(0, 0, w, h), 0, 0, &words, 4 * w, 0)
                        .unwrap();
                    let [_, colors] = reads(device, &queue, &kernels, &reference, dispatch);
                    for i in 0..texels {
                        assert_eq!(
                            got[16 * i..16 * i + 12],
                            colors[16 * i..16 * i + 12],
                            "{what}: colors of texel {i}"
                        );
                    }
                    // The CPU copies plane 0's words at the plane's own rows.
                    let mut read = vec![0; 4 * texels];
                    t.get_bytes(&mut read, 4 * w, 0, Region::new_2d(0, 0, w, h), 0, 0)
                        .unwrap();
                    for y in 0..h {
                        for x in 0..w {
                            let i = (y * w + x) as usize;
                            assert_eq!(
                                read[4 * i..4 * i + 4],
                                pattern[cpu(x, y)..cpu(x, y) + 4],
                                "{what}: get_bytes {i}"
                            );
                        }
                    }
                    let replaced: Vec<u8> = (0..4 * texels).map(|k| (k * 7 + 1) as u8).collect();
                    t.replace_region(Region::new_2d(0, 0, w, h), 0, 0, &replaced, 4 * w, 0)
                        .unwrap();
                    let mut want = pattern.clone();
                    for y in 0..h {
                        for x in 0..w {
                            let i = (y * w + x) as usize;
                            want[cpu(x, y)..cpu(x, y) + 4]
                                .copy_from_slice(&replaced[4 * i..4 * i + 4]);
                        }
                    }
                    let got = read_buffer(&surface, 0, length).unwrap();
                    assert!(got == want, "{what}: replace_region");
                    // Blits with a buffer move plane 0's words at the GPU's rows.
                    let out = buffer_with(device, &vec![0xcd; 4 * texels]).unwrap();
                    let command_buffer = queue.command_buffer().unwrap();
                    let mut blit = command_buffer.blit_command_encoder().unwrap();
                    blit.copy_from_texture_to_buffer(
                        &t,
                        0,
                        0,
                        Origin::new(0, 0, 0),
                        Size::new(w, h, 1),
                        &out,
                        0,
                        4 * w,
                        4 * w * h,
                        BlitOption::None,
                    )
                    .unwrap();
                    let into: Vec<u8> = (0..4 * texels).map(|k| (k * 13 + 5) as u8).collect();
                    let source = buffer_with(device, &into).unwrap();
                    blit.copy_from_buffer_to_texture(
                        &source,
                        0,
                        4 * w,
                        4 * w * h,
                        Size::new(w, h, 1),
                        &t,
                        0,
                        0,
                        Origin::new(0, 0, 0),
                        BlitOption::None,
                    )
                    .unwrap();
                    blit.end_encoding().unwrap();
                    command_buffer.commit().unwrap();
                    command_buffer.wait_until_completed_for(WAIT).unwrap();
                    let moved = read_buffer(&out, 0, 4 * texels as u64).unwrap();
                    let before = want.clone();
                    for y in 0..h {
                        for x in 0..w {
                            let i = (y * w + x) as usize;
                            assert_eq!(
                                moved[4 * i..4 * i + 4],
                                before[gpu(x, y)..gpu(x, y) + 4],
                                "{what}: blit out {i}"
                            );
                            want[gpu(x, y)..gpu(x, y) + 4].copy_from_slice(&into[4 * i..4 * i + 4]);
                        }
                    }
                    let got = read_buffer(&surface, 0, length).unwrap();
                    assert!(got == want, "{what}: blit in");
                    // A fill writes plane 0's words at the GPU's rows, cut at the level.
                    let mut want = read_buffer(&surface, 0, length).unwrap();
                    blit_once(&queue, |blit| {
                        blit.fill_texture(&t, 0, 0, Region::new_2d(1, 1, w, h), &[1, 2, 3])
                    });
                    for y in 1..h {
                        for x in 1..w {
                            want[gpu(x, y)..gpu(x, y) + 4].copy_from_slice(&[1, 2, 3, 1]);
                        }
                    }
                    let got = read_buffer(&surface, 0, length).unwrap();
                    assert!(got == want, "{what}: fill");
                    // Copies with a texture of the device's memory and with another surface's
                    // move both planes' bytes at the blit rows, as Metal's blit does.
                    let extent = [w as usize, h as usize];
                    let texels_of = |texture: &Texture| -> (Vec<u8>, Vec<u8>) {
                        let mut words = vec![0; 4 * texels];
                        texture
                            .get_bytes(&mut words, 4 * w, 0, Region::new_2d(0, 0, w, h), 0, 0)
                            .unwrap();
                        let [_, floats] = reads(device, &queue, &kernels, texture, dispatch);
                        (words, (0..texels).map(|i| alpha_of(&floats, i)).collect())
                    };
                    let planes = (
                        down(offset) as usize,
                        down(alpha) as usize,
                        down(row) as usize,
                    );
                    // Out of the surface into a texture of the device's memory.
                    let own = target(device, format, w, h);
                    write(device, &queue, &kernels, &own, (&values, true), dispatch);
                    let (words, alphas) = texels_of(&own);
                    let (mut memory, own_at) = own_planes(&words, &alphas, extent);
                    let bytes = read_buffer(&surface, 0, length).unwrap();
                    blit_once(&queue, |blit| blit.copy_texture(&t, &own));
                    blit_planes(&bytes, planes, &mut memory, own_at, extent);
                    assert!(
                        texels_of(&own) == own_texels(&memory, extent),
                        "{what}: copy out of the surface"
                    );
                    // Into the surface from a texture of the device's memory.
                    let fresh = target(device, format, w, h);
                    let others = halves(texels, |n, c| 0x2900 + 53 * n + 977 * c);
                    write(device, &queue, &kernels, &fresh, (&others, true), dispatch);
                    let (words, alphas) = texels_of(&fresh);
                    let (memory, fresh_at) = own_planes(&words, &alphas, extent);
                    let mut want = read_buffer(&surface, 0, length).unwrap();
                    blit_once(&queue, |blit| blit.copy_texture(&fresh, &t));
                    blit_planes(&memory, fresh_at, &mut want, planes, extent);
                    let got = read_buffer(&surface, 0, length).unwrap();
                    assert!(got == want, "{what}: copy into the surface");
                    // Into another surface's planes.
                    let pattern: Vec<u8> = (0..length).map(|k| (k * 29 + 7) as u8).collect();
                    let other = buffer_with(device, &pattern).unwrap();
                    let t2 = surface_texture(&other, format, shape, options)
                        .unwrap_or_else(|e| panic!("{what}: {e}"));
                    let mut want = pattern;
                    blit_once(&queue, |blit| blit.copy_texture(&t, &t2));
                    blit_planes(&got, planes, &mut want, planes, extent);
                    let into = read_buffer(&other, 0, length).unwrap();
                    assert!(into == want, "{what}: copy between surfaces");
                }
            }
        }
        let notices = counts(&messages.take_notices());
        assert!(
            notices.contains_key(&Degradation::ShadowedTexture),
            "{notices:?}"
        );
    });
}

/// A consumer's two plane views over one buffer, plane 0 in `BGR10_XR` (or its sRGB twin) and
/// plane 1 in `ARGB8Unorm`, read Metal's values and leave Metal's bytes in the buffer after
/// kernel writes, in both geometries. Where the device does not sample through Metal's own
/// texture unit, a read of plane 1 is the device's conversion of its codes (announced
/// `device-texel-conversion`): one that differs from Metal's holds where each value within the
/// conversion's bound of a code's Metal value, replaced by it, gives Metal's output
/// ([`as_metal_codes`]), counted.
#[test]
fn plane_views_over_a_buffer_read_and_write_as_metal_does() {
    on_device(|device, messages| {
        let queue = device.new_command_queue().unwrap();
        let kernels = Kernels::new(device);
        let exact = metal_sampler(device);
        let mut converted = Vec::new();
        let (mut wrong, mut moved) = (Vec::new(), 0);
        let steps = [
            (halves(GRID, |n, c| 32 * n + 16384 * c + 7 * c), true),
            (floats(GRID, 32), false),
        ];
        for plane0 in [PixelFormat::BGR10XR, PixelFormat::BGR10XRSRGB] {
            let form = ExtendedRange::of(plane0).expect("an extended-range format");
            for (g, (width, height, row)) in GEOMETRIES.into_iter().enumerate() {
                let view_texels = (width as usize, height as usize);
                let metal = |tag: &str| {
                    VIEWS
                        .iter()
                        .find(|m| m.0 == plane0 && m.1 == format!("G{g}{tag}"))
                        .map(|m| m.2)
                        .unwrap()
                };
                let initial: Vec<u8> = (0..VIEW_BUFFER).map(|i| (i * 89 + 7) as u8).collect();
                let buffer = buffer_with(device, &initial).unwrap();
                let view = |format: PixelFormat, offset: u64| {
                    let mut descriptor =
                        TextureDescriptor::texture_2d(format, width, height, false);
                    descriptor.usage = TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE;
                    descriptor.resource_options = ResourceOptions::shared();
                    buffer
                        .new_texture(&descriptor, offset, row)
                        .unwrap_or_else(|e| panic!("a {format:?} view: {e}"))
                };
                let planes = [view(plane0, 0), view(PixelFormat::ARGB8Unorm, PLANE_1)];
                let mut outputs = Vec::new();
                // Plane 0's reads, each lane the device's conversion within its bound of Metal's
                // codes `codes` replaced by Metal's read (`as_metal_reads`); plane 1's as read. The
                // texels the bound moved.
                let push_reads =
                    |outputs: &mut Vec<(String, Vec<u8>)>, s: usize, codes: &[[Option<u32>; 4]]| {
                        let mut moved = 0;
                        for (p, plane) in planes.iter().enumerate() {
                            let [h, f] = reads(device, &queue, &kernels, plane, Dispatch::Consumer);
                            let (f, h) = match p {
                                0 => {
                                    let (f, h, n) = as_metal_reads(form, (&f, &h), codes);
                                    moved += n;
                                    (f, h)
                                }
                                _ => (f, h),
                            };
                            outputs.push((format!("R{s}P{p}H"), h));
                            outputs.push((format!("R{s}P{p}F"), f));
                        }
                        moved
                    };
                moved += push_reads(
                    &mut outputs,
                    0,
                    &view_codes(form, &initial, view_texels, row),
                );
                for (s, (values, half)) in steps.iter().enumerate() {
                    let s = s + 1;
                    for plane in &planes {
                        write(
                            device,
                            &queue,
                            &kernels,
                            plane,
                            (values, *half),
                            Dispatch::Consumer,
                        );
                    }
                    // Metal's codes of plane 0 after the write; the buffer's plane 0 words each the
                    // device's encode within a code of Metal's replaced by Metal's word.
                    let codes =
                        written_codes(form, (values, *half), GRID_WIDTH as usize, view_texels);
                    let mut bytes = read_buffer(&buffer, 0, VIEW_BUFFER).unwrap();
                    for (t, texel) in codes.iter().enumerate() {
                        let [Some(r), Some(g), Some(b), _] = *texel else {
                            continue;
                        };
                        let (x, y) = (t as u64 % GRID_WIDTH, t as u64 / GRID_WIDTH);
                        let at = (y * row + 4 * x) as usize;
                        let metal = (0xc000_0000 | r << 20 | g << 10 | b).to_le_bytes();
                        let word = &mut bytes[at..at + 4];
                        if *word != metal && texel_within(form, word, &metal) {
                            word.copy_from_slice(&metal);
                            moved += 1;
                        }
                    }
                    outputs.push((format!("W{s}BUF"), bytes));
                    moved += push_reads(&mut outputs, s, &codes);
                }
                for (tag, bytes) in outputs {
                    let plane_1_read = tag.contains("P1");
                    if !exact
                        && plane_1_read
                        && digest(&bytes) != metal(&tag)
                        && digest(&as_metal_codes(&bytes, tag.ends_with('H'))) == metal(&tag)
                    {
                        converted.push(format!("{plane0:?} G{g}{tag}"));
                        continue;
                    }
                    check(
                        &mut wrong,
                        format!("{plane0:?} G{g}{tag}"),
                        metal(&tag),
                        &bytes,
                    );
                }
            }
        }
        println!(
            "XR-TWO-PLANES {}: {} plane 1 reads the device's conversion of Metal's codes: {converted:?}",
            device.name(),
            converted.len()
        );
        println!(
            "XR-TWO-PLANES {}: {moved} plane 0 texels within the device's conversion of Metal's",
            device.name()
        );
        let notices = counts(&messages.take_notices());
        if !converted.is_empty() {
            assert!(
                notices.contains_key(&Degradation::DeviceTexelConversion),
                "{notices:?}"
            );
        }
        report(device, &wrong);
        assert_counted(&notices, moved, "the plane views");
    });
}

/// The codes of a `BGR10_XR` (or `_sRGB`) view of `width` x `height` texels over `bytes` with rows
/// `row` bytes apart, by grid thread (`GRID_WIDTH` wide): red, green and blue of the texel a thread
/// within the view reads, `None` past it and for alpha.
fn view_codes(
    form: ExtendedRange,
    bytes: &[u8],
    (width, height): (usize, usize),
    row: u64,
) -> Vec<[Option<u32>; 4]> {
    (0..GRID)
        .map(|t| {
            let (x, y) = (t % GRID_WIDTH as usize, t / GRID_WIDTH as usize);
            if x >= width || y >= height {
                return [None; 4];
            }
            let at = y * row as usize + 4 * x;
            let (c, _) = codes(form, &bytes[at..at + 4]);
            [Some(c[0]), Some(c[1]), Some(c[2]), None]
        })
        .collect()
}

/// `bytes`, a read of an 8-bit normalized view (`half4`s where `half`, else `float4`s), with each
/// value in `[0, 1]` that lies within the device's conversion's bound of a code `c`'s Metal value
/// (`converted_holds` of `c / 255`, and `half_within` of its `half`) replaced by that value: what
/// Metal's read gives where the device read Metal's codes. Other bytes (the fill of threads past
/// the view, a value off every code) stay.
fn as_metal_codes(bytes: &[u8], half: bool) -> Vec<u8> {
    match half {
        true => bytes
            .as_chunks::<2>()
            .0
            .iter()
            .flat_map(|h| {
                let v = half_value(u16::from_le_bytes(*h));
                let metal = half_bits((v * 255.0).round() / 255.0);
                match (0.0..=1.0).contains(&v) && half_within(v, half_value(metal)) {
                    true => metal.to_le_bytes(),
                    false => *h,
                }
            })
            .collect(),
        false => bytes
            .as_chunks::<4>()
            .0
            .iter()
            .flat_map(|f| {
                let v = f32::from_le_bytes(*f);
                let metal = (v * 255.0).round() / 255.0;
                match (0.0..=1.0).contains(&v) && converted_holds(v, metal, false) {
                    true => metal.to_le_bytes(),
                    false => *f,
                }
            })
            .collect(),
    }
}

/// Each two-plane format makes what Metal makes and refuses what Metal stops (one case per
/// process): a 2D texture of each storage mode and usage, of 1 x 1 and 63 x 5, mipmapped, a 2D
/// array, 1D, 3D and `new_shared_texture`'s are made; a cube (Metal's
/// nil, `Error::Nil`) and a multisample texture (Metal asserts, `Error::Invalid`) are refused
/// `TypeForFormat`, and `ShaderAtomic` usage (Metal asserts) `UsageForFormat`.
#[test]
fn two_plane_textures_are_made_as_metal_makes_them() {
    on_device(|device, _| {
        let rw = TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE;
        for format in [PixelFormat::RGB10A82PXR10, PixelFormat::RGB10A82PXR10SRGB] {
            let of = |ty: TextureType, [width, height, depth]: [u64; 3], usage: TextureUsage| {
                let mut d = TextureDescriptor::texture_2d(format, width, height, false);
                d.texture_type = ty;
                d.depth = depth;
                d.usage = usage;
                d.resource_options = ResourceOptions::private();
                d
            };
            let mut made = vec![
                of(TextureType::Type2D, [64, 4, 1], rw),
                of(TextureType::Type2D, [64, 4, 1], TextureUsage::SHADER_READ),
                of(TextureType::Type2D, [64, 4, 1], TextureUsage::SHADER_WRITE),
                of(TextureType::Type2D, [64, 4, 1], TextureUsage::RENDER_TARGET),
                of(
                    TextureType::Type2D,
                    [64, 4, 1],
                    TextureUsage::SHADER_READ | TextureUsage::PIXEL_FORMAT_VIEW,
                ),
                of(TextureType::Type2D, [1, 1, 1], rw),
                of(TextureType::Type2D, [63, 5, 1], rw),
                of(TextureType::Type1D, [64, 1, 1], rw),
                of(TextureType::Type3D, [16, 16, 2], rw),
            ];
            let mut shared = of(TextureType::Type2D, [64, 4, 1], rw);
            shared.resource_options = ResourceOptions::shared();
            made.push(shared);
            let mut mipmapped = of(TextureType::Type2D, [64, 4, 1], rw);
            mipmapped.mipmap_level_count = 2;
            made.push(mipmapped);
            let mut array = of(TextureType::Type2DArray, [64, 4, 1], rw);
            array.array_length = 2;
            made.push(array);
            for d in &made {
                let t = device
                    .new_texture(d)
                    .unwrap_or_else(|e| panic!("{format:?} {d:?}: {e}"));
                assert_eq!(t.pixel_format(), format);
            }
            let t = device
                .new_shared_texture(&of(TextureType::Type2D, [64, 4, 1], rw))
                .unwrap_or_else(|e| panic!("{format:?} new_shared_texture: {e}"));
            assert_eq!(t.pixel_format(), format);
            let refused = |d: &TextureDescriptor, want: TextureProblem| match device
                .new_texture(d)
                .map(|_| ())
            {
                Err(Error::Invalid(Invalid {
                    problem: Problem::Texture(problem),
                    ..
                })) => assert_eq!(problem, want, "{format:?} {d:?}"),
                other => panic!("{format:?} {d:?}: {other:?}"),
            };
            let mut cube = of(TextureType::Cube, [16, 16, 1], rw);
            cube.height = 16;
            assert_eq!(
                device.new_texture(&cube).map(|_| ()).err(),
                Some(Error::Nil(Nil::new(
                    "new_texture",
                    Problem::Texture(TextureProblem::TypeForFormat)
                ))),
                "{format:?}"
            );
            let mut multisample = of(
                TextureType::Type2DMultisample,
                [64, 4, 1],
                TextureUsage::RENDER_TARGET,
            );
            multisample.sample_count = 2;
            refused(&multisample, TextureProblem::TypeForFormat);
            refused(
                &of(
                    TextureType::Type2D,
                    [64, 4, 1],
                    rw | TextureUsage::SHADER_ATOMIC,
                ),
                TextureProblem::UsageForFormat,
            );
            // A cube array of more than 341 cubes is stopped before the nil, and a memoryless
            // cube or cube array, which Metal makes, is refused not implemented.
            let mut cubes = of(TextureType::CubeArray, [16, 16, 1], rw);
            cubes.array_length = 342;
            refused(&cubes, TextureProblem::ArrayLength);
            for (ty, length) in [(TextureType::Cube, 1), (TextureType::CubeArray, 341)] {
                let mut d = of(ty, [16, 16, 1], TextureUsage::RENDER_TARGET);
                d.array_length = length;
                d.resource_options = ResourceOptions::new(
                    StorageMode::Memoryless,
                    CpuCacheMode::DefaultCache,
                    HazardTrackingMode::Default,
                );
                match device.new_texture(&d).map(|_| ()) {
                    Err(Error::Unsupported(gap)) => {
                        assert_eq!(gap.feature, Feature::Format, "{format:?} {ty:?}")
                    }
                    other => panic!("{format:?} memoryless {ty:?}: {other:?}"),
                }
            }
        }
        // Every format's cube array stops past 341 cubes.
        let mut cubes = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, 16, 16, false);
        cubes.texture_type = TextureType::CubeArray;
        cubes.resource_options = ResourceOptions::private();
        cubes.array_length = 341;
        device.new_texture(&cubes).unwrap();
        cubes.array_length = 342;
        match device.new_texture(&cubes).map(|_| ()) {
            Err(Error::Invalid(Invalid {
                problem: Problem::Texture(TextureProblem::ArrayLength),
                ..
            })) => {}
            other => panic!("RGBA8Unorm, 342 cubes: {other:?}"),
        }
    });
}
