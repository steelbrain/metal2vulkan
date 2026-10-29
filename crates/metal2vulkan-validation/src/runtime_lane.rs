//! The runtime lane: device tests of the Metal-like runtime, `metal2vulkan_runtime`, on the device
//! a GPU-safety cfg selects (`--cfg m2v_cpu_icd` or `--cfg m2v_gpu`), under the validation layer.
//!
//! The runtime creates its own instance and device here, with explicit options and no
//! environment: `VK_LAYER_KHRONOS_validation`, a message sink, and the physical-device kinds the
//! cfg names. A GPU device is opened only behind the hang tripwire check, as every device open in
//! this crate is (`gpu.rs`).
//!
//! Every device a lane test makes keeps its shader cache in a directory the test owns, empty when
//! made and removed after ([`ScratchCache`]): a warm cache never stands in for a translation a test
//! measures.
//!
//! An authored public kernel case runs through the runtime as a Metal program would run it
//! ([`run_case`]), by the case runner the engine executor's runtime path shares
//! (`candidate::runtime_case`). [`public_case`] gives the case as the engine maps it, with the
//! output its committed Metal observation records.

use crate::air_library::AirLibrary;
use crate::candidate::engine::{compute_case, ComputeCase};
pub use crate::candidate::runtime_case::{
    argument_texture_usage, bound_buffers, buffer_with, buffer_with_options, commit_case,
    compute_descriptor, encode_case, layer_reported_nothing, layer_reported_nothing_at_all, loader,
    nothing_reported_without_the_layer, open_device, open_devices, read_buffer, read_texture,
    texture_descriptor, texture_usage, texture_with, CaseResources, EnabledLayers, LayerClaim,
    Messages, ScratchCache, LAYERS_DROPPED, VALIDATION_LAYER,
};
use crate::candidate::runtime_case::{
    encoded_tile, report_each_message, tile_descriptor, Committed,
};
use crate::case::AuthoredCase;
use crate::check::interface::AirInterface;
use crate::check::loop_free::DecodedModule;
use crate::gpu::DeviceType;
use crate::metal::ORACLE_ABI;
use crate::observation::MetalObservation;
use crate::store::CorpusStore;
use ash::vk;
use base64::Engine as _;
use metal2vulkan::bind::StatusReport;
use metal2vulkan::metal::TextureUsage;
use metal2vulkan::pipeline::LinkedFunctions;
use metal2vulkan_devtools::{llvm_assemble, AssembleError};
use metal2vulkan_runtime::{
    CommandBufferStatus, ComputePipelineState, DebugMessage, Degradation, Device, DeviceOptions,
    Execution, Function, LayerSetting, LayerSettingValue, PipelineOptions,
};
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::sync::OnceLock;
use std::time::Duration;

/// The Metal environment whose committed observations the lane compares with.
pub const METAL_ENVIRONMENT: &str = "apple-m3-max-macos27.0.1";

/// The longest a lane command buffer may take.
const WAIT: Duration = Duration::from_secs(60);

/// `device`'s work counts once its pipeline pool holds no piece (`WorkCounts::pipeline_pool_busy`),
/// waiting at most [`WAIT`]: a draw that took its key's residual form does not wait for its own
/// key's making, which may outlast the draw's command buffer.
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
fn pipelines_settled(device: &Device) -> metal2vulkan_runtime::WorkCounts {
    let until = std::time::Instant::now() + WAIT;
    loop {
        let counts = device.work_counts();
        if counts.pipeline_pool_busy == 0 {
            return counts;
        }
        assert!(
            std::time::Instant::now() < until,
            "the device's pipeline pool still busy after {WAIT:?}"
        );
        std::thread::sleep(Duration::from_millis(1));
    }
}

/// What [`wait_for_device`] saw: whether the command buffer completed, the wall time it waited,
/// and the longest wait for a CPU that one of the process's threads had meanwhile.
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
#[derive(Clone, Copy, Debug)]
struct DeviceWait {
    completed: bool,
    wall: Duration,
    queued: Duration,
}

/// The longest [`wait_for_device`] waits by the wall clock, as a multiple of its bound of the
/// device's own time: a bound on the wait whatever the run-queue counts read.
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
const WALL_CAP: u32 = 5;

/// Wait until `command_buffer` completed, for at most `bound` of the device's own time: the wall
/// time less the longest wait for a CPU that any one of the process's threads had meanwhile, and
/// at most [`WALL_CAP`] times `bound` by the wall clock.
///
/// A CPU device (lavapipe) runs a command buffer on the process's own threads, which a loaded
/// host keeps off a CPU for much of the wall time: that wait is the host's, not the device's.
/// Every other wait stays in the reading: a device that hangs blocked, or spins, still crosses the
/// bound. On a GPU the threads wait for it blocked, and the reading is about the wall time.
///
/// Linux counts each thread's wait for a CPU: the second field of `/proc/self/task/*/schedstat`
/// (the kernel's `Documentation/scheduler/sched-stats.rst`); where it gives none, the reading is
/// the wall time. The counts are read after the start's instant and before the end's, so a wait
/// between the two reads stays in the reading.
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
fn wait_for_device(
    command_buffer: &metal2vulkan_runtime::CommandBuffer,
    bound: Duration,
) -> DeviceWait {
    let started = std::time::Instant::now();
    let before = run_queue_waits();
    let cap = bound * WALL_CAP;
    loop {
        let queued = longest_wait_since(&before);
        let wall = started.elapsed();
        let own = wall.saturating_sub(queued);
        if own >= bound || wall >= cap {
            return DeviceWait {
                completed: false,
                wall,
                queued,
            };
        }
        let slice = (bound - own).min(cap - wall);
        if command_buffer.wait_until_completed_for(slice).is_ok() {
            let queued = longest_wait_since(&before);
            return DeviceWait {
                completed: true,
                wall: started.elapsed(),
                queued,
            };
        }
    }
}

/// Each of the process's threads' wait for a CPU so far, by thread id, where Linux counts it
/// ([`wait_for_device`]).
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
fn run_queue_waits() -> BTreeMap<u64, Duration> {
    let Ok(tasks) = std::fs::read_dir("/proc/self/task") else {
        return BTreeMap::new();
    };
    let wait = |task: std::fs::DirEntry| {
        let id = task.file_name().to_str()?.parse().ok()?;
        let stat = std::fs::read_to_string(task.path().join("schedstat")).ok()?;
        let waited = stat.split_whitespace().nth(1)?.parse().ok()?;
        Some((id, Duration::from_nanos(waited)))
    };
    tasks.flatten().filter_map(wait).collect()
}

/// The longest wait for a CPU that one of the process's threads had since `before` (its counts
/// then, [`run_queue_waits`]); a thread started since counts all its wait.
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
fn longest_wait_since(before: &BTreeMap<u64, Duration>) -> Duration {
    let since = |(id, waited): (u64, Duration)| {
        waited.saturating_sub(before.get(&id).copied().unwrap_or_default())
    };
    run_queue_waits()
        .into_iter()
        .map(since)
        .max()
        .unwrap_or_default()
}

/// One lane test per shard of a census too long to run as one test (the Mac lane stops a test
/// at 60 s): `census_shards!(census, SHARDS; test_0 = 0, test_1 = 1, ...)` makes each `test_n` a
/// test that runs `census(n)`. It does not compile unless the shards listed are `0..SHARDS`, each
/// once and in order (`lists_each_shard`), so where `census(n)` takes its cases through
/// `in_shard(.., n, SHARDS)` the tests together run every case the census selects, each once.
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
macro_rules! census_shards {
    ($census:ident, $shards:expr; $($test:ident = $shard:literal),+ $(,)?) => {
        $(
            #[test]
            fn $test() {
                $census($shard);
            }
        )+
        const _: () = assert!(
            $crate::runtime_lane::lists_each_shard(&[$($shard),+], $shards),
            "the shard tests are not the shards 0..SHARDS, each once and in order"
        );
    };
}

/// `select` narrowed to shard `shard` of `shards`: of the cases `select` takes, in the order they
/// are offered, every `shards`th from the `shard`th. The shards `0..shards` together take each
/// case `select` takes exactly once. A shard maps only its own cases ([`public_cases`]).
#[cfg(test)]
fn in_shard<T>(select: impl Fn(&T) -> bool, shard: usize, shards: usize) -> impl Fn(&T) -> bool {
    let taken = std::cell::Cell::new(0usize);
    move |case: &T| {
        if !select(case) {
            return false;
        }
        let index = taken.get();
        taken.set(index + 1);
        index % shards == shard
    }
}

/// Metal's twelve-bit decode step of 8-bit sRGB code `c`: `round(decode(c / 255) · 4095)` for the
/// MSL transfer, every margin from a half step at least `1.6e-4` step, so `f64` rounds it as the
/// table does. Metal decodes the code to `step / 4095`, correctly rounded.
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
fn srgb_step(c: u32) -> u32 {
    let x = f64::from(c) / 255.0;
    let lin = if x <= 0.04045 {
        x / 12.92
    } else {
        ((x + 0.055) / 1.055).powf(2.4)
    };
    (lin * 4095.0).round() as u32
}

/// Whether `listed` is the shards `0..shards`, each once and in order (`census_shards!`).
#[cfg(test)]
const fn lists_each_shard(listed: &[usize], shards: usize) -> bool {
    if listed.len() != shards {
        return false;
    }
    let mut shard = 0;
    while shard < shards {
        if listed[shard] != shard {
            return false;
        }
        shard += 1;
    }
    true
}

#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod absent_reads;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod address_maps;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod alpha_coverage;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod alpha_in_red;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod alpha_tests;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod amplified_draws;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod anisotropic_filter;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod argument_buffers;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod argument_linear_samples;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod astc_textures;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod attachment_read_costs;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod attachment_reads;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod attachmentless;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod barycentric_weights;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod bfloat_targets;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod binary_archives;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod blits;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod border_swizzles;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod buffer_depth_samples;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod buffer_texture_blits;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod buffer_texture_order;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod buffer_texture_renders;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod buffer_textures;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod cache;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod capabilities;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod chosen_pointers;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod chroma_pairs;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod clear_reach;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod color_fills;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod color_groups;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod color_reformat;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod color_writes;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod command_queues;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod command_render_states;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod commands_recorded;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod commit_costs;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod commit_latency;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod commit_work;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod compressed_blits;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod compressed_decode;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod compressed_moves;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod compute_threadgroups;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod coverage_alpha_rounding;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod coverage_loop;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod coverage_outputs;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod creation_times;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod cross_class_reads;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod cross_class_storage;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod cubic_reach_costs;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod cubic_samplers;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod degrade_notices;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod denormal_word_fetches;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod depth_as_color;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod depth_compare_clamps;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod depth_compare_levels;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod depth_compares;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod depth_costs;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod depth_denormals;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod depth_never;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod depth_stencil_churn;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod depth_texel_reads;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod detach;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod detach_textures;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod device_commands;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod device_copies;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod device_lockstep;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod discard_counts;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod discard_hoist;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod dispatch_costs;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod dispatch_executions;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod dispatch_work;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd), target_os = "linux"))]
mod dma_buf_imports;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod dont_care_stores;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod draw_bases;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod draw_changes;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod draw_costs;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod draw_work;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod driver_deviations;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod dynamic_libraries;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod escaped_slots;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod etc_textures;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod event_round_trips;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod events;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod executions;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod exits;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod extended_range_bicubic;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod extended_range_blends;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod extended_range_planes;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod extended_range_resolve;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod extended_range_samples;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod extended_range_targets;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod extended_range_two_planes;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod extended_swizzles;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod extra_colors;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod failed_commits;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod families;
mod fatal_signals;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod fence_stages;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod fetch_costs;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod first_use;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod first_use_writes;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod first_vertex;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod format_copies;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod fragment_imageblocks;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod fragment_tile_memory;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod frame_costs;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod framebuffer_fetch;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod freed_memory;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod frees_in_flight;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod fresh_contents;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod fresh_costs;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod fresh_writes;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod function_tables;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod generated_executions;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod global_builtins;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod gpu_times;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod grouped_submissions;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod guarded_effects;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd), target_os = "linux"))]
mod guest_frame_costs;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod half_output_blends;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod half_read_nans;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod half_slice_nans;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod half_storage_writes;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod half_texel_nans;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod half_write_nans;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod handed_objects;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod handler_waits;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod handles;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod heap_device_samples;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod heap_extended_reads;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod heap_layouts;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod heap_metal_sizes;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod heap_sample_moves;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod heap_textures;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod heaps;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod helper_tangle;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod hidden_depth;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod host_bytes;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod host_contract;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod host_copied_textures;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod host_memory;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod in_place_textures;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod index_fetches;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod index_pools;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod indirect_command_buffers;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod indirect_dispatches;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod indirect_draws;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod indirect_executions;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod indirect_host_steps;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod indirect_patch_addresses;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod inline_bytes;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod inline_reasons;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod int_attributes;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod interlocked_draw_order;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod keep_alives;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod kept_apart;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod kept_buffer_bytes;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod kept_images;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod kept_resolves;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod kept_variants;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod kernel_imageblocks;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod kernel_lod;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod kernel_textures;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod late_resolves;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod layered_passes;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod level_sizes;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod level_writes_by_usage;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod linear_attachments;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod logic_ops;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod made_at_creation;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod math_routines;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod memoryless;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod memoryless_uses;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod mesh_barriers;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod mesh_commands;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod mesh_draws;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod mip_levels;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod mipmap_chains;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod mipmap_costs;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod mirror_clamp;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod moved_writers;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod msaa_depth_moves;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod nan_resolves;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod narrow_clamps;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod native_maths;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod nearest_positions;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod no_copy_heaps;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod no_copy_textures;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod norm_storage_writes;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod normalized_vertex_fetch;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod numbered_samplers;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod padded_loads;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod pass_start_reads;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod patch_culls;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod patch_indices;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod patch_primitive_ids;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod patches;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod pipeline_creation;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod pipeline_records;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod pitched_surfaces;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod pixel_downscales;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod pixel_samplers;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod placement;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod pointer_uploads;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod pool_churn;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod pool_slots;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod post_depth_coverage;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
pub(crate) mod predication;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod presenting;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod primitive_ids;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod private_memory;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod private_primitives;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod property_reads;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod protection_options;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod provoking;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod provoking_mesh;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod pvrtc_textures;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod queue_order;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod queue_storms;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod raster_orders;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod raster_sample_counts;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod rate_mapped_layers;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod rate_mapped_passes;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod rate_maps;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod raw_dispatch_types;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod read_through_views;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod rebind_reach;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod rebinds;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod recursion_frames;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod reflection;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod render_arguments;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod render_barriers;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod render_executions;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod render_gpu_executions;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod render_merged_executions;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod render_resident_writes;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod render_state;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod renders;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod resolve_rounding;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod resolves;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod resource_ids;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod restart;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod rgb10a2_srgb;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod rgbx_textures;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod rotated_targets;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod rotated_textures;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod row_copies;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod runtime_data;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod runtime_levels;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod same_texture_strips;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod sample_copies;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod sample_coverage;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod sample_fills;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod sample_imageblocks;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod sample_positions;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod separate_samples;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod seq_cst_fences;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod shader_stencil;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod shared_exponent_storage;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod single_sample_passes;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod snapshot_reads;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod snorm_tile_store;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod sparse_textures;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod srgb_decode;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod srgb_storage;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod stage_in_regions;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod stage_in_strides;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod state_ids;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod submission_checks;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod submission_handlers;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod submit_costs;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd), target_os = "linux"))]
mod submit_per_import;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod surface_aliases;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod surface_fills;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd), target_os = "linux"))]
mod surface_texture_costs;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod swizzled_storage;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
pub(crate) mod sync_coverage;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod teardown_holders;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod tests;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod texel_swizzles;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod texture_accesses;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod texture_blits;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod texture_buffer_fills;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod texture_creators;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod texture_fills;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod textures;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod tile_dispatches;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod tile_edges;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod tile_imageblock;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod tile_kept_bytes;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod tile_memory;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod tile_normalized;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod tile_path_costs;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod tile_shadows;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod tile_states;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod tile_threadgroups;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod topology_classes;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod two_plane_copies;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod two_plane_heap_blits;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod two_plane_heap_fills;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod two_plane_samples;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod uniform_varyings;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod unorm_depth_bias;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod used_resources;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod vertex_cost;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod vertex_strides;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod view_attachments;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod view_base_offsets;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod view_reinterpretation;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod viewport_ranges;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod visibility;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod wide_points;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod window_surfaces;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod word_samples;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod wrapped_buffers;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod ycbcr_device_reads;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod ycbcr_in_place;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod ycbcr_minify;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod ycbcr_passes;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod ycbcr_textures;

/// The runtime's default device among the physical devices of `device_type`, created with the
/// validation layer reporting to `messages` and its shader cache in `cache`
/// (`runtime_case::validated_options`). A GPU is opened only while the hang tripwire is absent
/// (`runtime_case::open_device`). The process first writes a native backtrace on a fatal signal
/// (`runtime_lane/fatal_signals.rs`), so a crash on a host that cannot be debugged says where.
pub fn validated_device(
    device_type: DeviceType,
    messages: &Messages,
    cache: &Path,
) -> Result<Device, String> {
    validated_device_with(device_type, messages, cache, |_| {})
}

/// [`validated_device`] with the options `adjust` changes beside the lane's own. The checks the
/// lane's switch enables ([`LaneChecks`]) hold over `adjust`'s settings.
///
/// The notices the device announces as it is made are taken apart and must be exactly
/// [`creation_kinds`]: a test reads only its own work's, and a device that announces another kind
/// as it is made fails here. Notices delivered before, which the test has not read, stay for it.
pub fn validated_device_with(
    device_type: DeviceType,
    messages: &Messages,
    cache: &Path,
    adjust: impl FnOnce(&mut DeviceOptions),
) -> Result<Device, String> {
    let pending = messages.take_notices();
    let device = made_lane_device(device_type, messages, cache, adjust);
    let made = messages.take_notices();
    messages.restore_notices(pending);
    messages.device_made(&made);
    let device = device?;
    creation_checked(&device, &made)?;
    Ok(device)
}

/// The device [`validated_device_with`] makes; the notices it announced as it was made are left in
/// its sink, `messages` or the one `adjust` set, for [`creation_checked`].
pub(crate) fn made_lane_device(
    device_type: DeviceType,
    messages: &Messages,
    cache: &Path,
    adjust: impl FnOnce(&mut DeviceOptions),
) -> Result<Device, String> {
    fatal_signals::install();
    let mut options = validated_options(messages, cache);
    adjust(&mut options);
    LaneChecks::switched().apply(&mut options);
    // The layers the device finally asks for, after `adjust`: what a layer-dependent claim of this
    // list holds is what the instance enables of them (`layer_reported_nothing`).
    messages.device_asked_layers(&options.layers);
    open_device(device_type, options).map_err(|error| {
        format!(
            "the runtime made no {} device: {error}",
            device_type.as_str()
        )
    })
}

/// Whether `made`, the messages `device`'s making delivered, are exactly its [`creation_kinds`],
/// one notice each; else what it announced.
pub(crate) fn creation_checked(device: &Device, made: &[DebugMessage]) -> Result<(), String> {
    let announced: BTreeSet<Degradation> = made
        .iter()
        .filter_map(|message| message.degradation.as_ref())
        .map(|notice| notice.degradation)
        .collect();
    let expected = creation_kinds(device);
    match announced == expected && made.len() == expected.len() {
        true => Ok(()),
        false => Err(format!(
            "{}: made announcing {:?}, not the device's own {expected:?}",
            device.name(),
            made.iter().map(|message| &message.text).collect::<Vec<_>>()
        )),
    }
}

/// The id name of the informational message a device sends as it is made where its driver keeps
/// its internal pipeline cache and that cache is Metal's own ([`metal_cache_kept`]); the runtime's
/// `device/create.rs` names it so.
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
const METAL_CACHE_KEPT: &str = "metal2vulkan-driver-pipeline-cache-kept";

/// [`creation_checked`] of `heard`, every message `device`'s making delivered to a sink that hears
/// every message (not the lane's, which keeps notices and failures alone): its notices exactly
/// its [`creation_kinds`], one each, and its other messages exactly the informational ones the
/// device sends as it is made, each once and in this order: [`LAYERS_DROPPED`] where the loader could not provide the instance layers the device asked for, which the device
/// alone can say (the instance's answer, not the loader's listing: a layer the loader lists and
/// cannot load answers the same); then [`METAL_CACHE_KEPT`] exactly where its driver keeps its
/// internal pipeline cache and that cache is Metal's own ([`metal_cache_kept`], read from the
/// driver apart from the runtime), an informational message naming the device and ending in its id
/// name; and nothing else.
///
/// What the device said of its layers goes to `messages`, the lane's list, which this device
/// reports nothing else to: a layer-dependent claim stated on that list afterwards holds only what
/// the instance enabled ([`layer_reported_nothing`]).
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
pub(crate) fn creation_heard_checked(
    device: &Device,
    heard: &[DebugMessage],
    messages: &Messages,
) -> Result<(), String> {
    let (notices, informed): (Vec<DebugMessage>, Vec<DebugMessage>) = heard
        .iter()
        .cloned()
        .partition(|message| message.degradation.is_some());
    creation_checked(device, &notices)?;
    for message in &informed {
        messages.heard_informational(&message.id_name, &message.text);
    }
    let dropped = messages.layers_dropped_message();
    let cache_kept = metal_cache_kept(device);
    let mut expected: Vec<String> = Vec::new();
    let mut rest = &informed[..];
    // Where the device was made without a layer it asked for, the loader said so first, at
    // `vkCreateInstance`: one message of its own per layer it could not provide (`layer_claim`).
    if dropped.is_some() {
        while let Some((first, after)) = rest.split_first() {
            if !crate::candidate::runtime_case::loader_layer_message(&first.id_name, &first.text) {
                break;
            }
            expected.push(first.text.clone());
            rest = after;
        }
    }
    if let Some(text) = &dropped {
        let tag = format!("[{LAYERS_DROPPED}]");
        expected.push(text.clone());
        match rest.split_first() {
            Some((first, after))
                if first.severity == metal2vulkan_runtime::MessageSeverity::Info
                    && first.id_name == LAYERS_DROPPED
                    && first.text.contains(device.name())
                    && first.text.ends_with(&tag) =>
            {
                rest = after;
            }
            _ => rest = &[],
        }
    }
    if cache_kept {
        let tag = format!("[{METAL_CACHE_KEPT}]");
        let on = format!("the device, on {}", device.name());
        expected.push(format!("an informational message on {on}, ending {tag}"));
        match rest.split_first() {
            Some((first, after))
                if first.severity == metal2vulkan_runtime::MessageSeverity::Info
                    && first.id_name == METAL_CACHE_KEPT
                    && first.text.contains(&on)
                    && first.text.ends_with(&tag) =>
            {
                rest = after;
            }
            _ => rest = &[],
        }
    }
    let whole = rest.is_empty() && informed.len() == expected.len();
    match whole {
        true => Ok(()),
        false => Err(format!(
            "{}: made saying {:?}, not the device's own {expected:?}",
            device.name(),
            informed
                .iter()
                .map(|message| &message.text)
                .collect::<Vec<_>>(),
        )),
    }
}

/// A macOS release as `[major, minor, patch]` (`26.7` is `[26, 7, 0]`), if it reads.
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
fn macos_release(text: &str) -> Option<[u32; 3]> {
    let mut parts = text.trim().split('.').map(str::parse::<u32>);
    let major = parts.next()?.ok()?;
    let minor = parts.next().unwrap_or(Ok(0)).ok()?;
    let patch = parts.next().unwrap_or(Ok(0)).ok()?;
    Some([major, minor, patch])
}

/// The macOS release this host runs (`sw_vers -productVersion`); `None` off macOS.
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
pub(crate) fn host_release() -> Option<[u32; 3]> {
    if !cfg!(target_os = "macos") {
        return None;
    }
    let out = std::process::Command::new("sw_vers")
        .arg("-productVersion")
        .output()
        .expect("sw_vers runs on macOS");
    assert!(out.status.success(), "sw_vers -productVersion failed");
    let text = String::from_utf8(out.stdout).expect("sw_vers prints UTF-8");
    Some(macos_release(&text).unwrap_or_else(|| panic!("macOS release {text:?} does not read")))
}

/// `VkPhysicalDevicePipelineBinaryPropertiesKHR`, laid out as the registry defines it (ash 0.38
/// predates `VK_KHR_pipeline_binary`).
#[repr(C)]
struct PipelineBinaryProperties {
    s_type: vk::StructureType,
    p_next: *mut std::ffi::c_void,
    internal_cache: vk::Bool32,
    internal_cache_control: vk::Bool32,
    prefers_internal_cache: vk::Bool32,
    precompiled_internal_cache: vk::Bool32,
    compressed_data: vk::Bool32,
}

// SAFETY: the layout is the registry's, beginning with `sType` and `pNext`, and this is its
// structure type (extension 484's fifth structure).
unsafe impl vk::TaggedStructure for PipelineBinaryProperties {
    const STRUCTURE_TYPE: vk::StructureType = vk::StructureType::from_raw(1_000_483_004);
}
// SAFETY: the registry lists `VkPhysicalDeviceProperties2` in the structure's `structextends`.
unsafe impl vk::ExtendsPhysicalDeviceProperties2 for PipelineBinaryProperties {}

/// Whether `device`'s driver keeps its internal pipeline cache, read from the driver: it lists
/// no `VK_KHR_pipeline_binary`, the device has no `VK_KHR_maintenance5` (core from Vulkan 1.4),
/// which that extension requires, or the driver's `pipelineBinaryInternalCacheControl` is false.
fn driver_keeps_pipeline_cache(device: &Device) -> bool {
    // SAFETY: instance-level queries of the device's physical device, made while it lives.
    let raw = unsafe { device.raw() };
    // SAFETY: as above.
    let listed = unsafe {
        raw.instance
            .enumerate_device_extension_properties(raw.physical_device)
    }
    .expect("the device's extensions");
    let lists = |name: &std::ffi::CStr| {
        listed
            .iter()
            .any(|extension| extension.extension_name_as_c_str() == Ok(name))
    };
    if !lists(c"VK_KHR_pipeline_binary") {
        return true;
    }
    let api = device.properties().description.api_version;
    let maintenance5 = (api.major, api.minor) >= (1, 4)
        || device
            .compile_device()
            .profile()
            .enabled()
            .extensions
            .contains(&metal2vulkan::device::VkExtension::KhrMaintenance5);
    if !maintenance5 {
        return true;
    }
    let mut binary = PipelineBinaryProperties {
        s_type: <PipelineBinaryProperties as vk::TaggedStructure>::STRUCTURE_TYPE,
        p_next: std::ptr::null_mut(),
        internal_cache: vk::FALSE,
        internal_cache_control: vk::FALSE,
        prefers_internal_cache: vk::FALSE,
        precompiled_internal_cache: vk::FALSE,
        compressed_data: vk::FALSE,
    };
    let mut properties = vk::PhysicalDeviceProperties2::default().push_next(&mut binary);
    // SAFETY: a physical device of the instance, which lists the extension; the chain lives for
    // the call.
    unsafe {
        raw.instance
            .get_physical_device_properties2(raw.physical_device, &mut properties)
    };
    binary.internal_cache_control != vk::TRUE
}

/// The MoltenVK releases (`driverVersion`) whose argument-buffer state the runtime reads from
/// their limits: 1.4.1 and 1.4.2, the releases whose source sets those limits as the runtime reads
/// them.
pub(crate) const MOLTENVK_ARGUMENT_BUFFERS_READ: [u32; 2] = [10401, 10402];

/// The kinds a device the lane makes announces as it is made, each from what the lane reads of
/// the device apart from the runtime, and nothing else: `driver-pipeline-cache-kept` where the
/// driver lets no device turn its internal pipeline cache off and that cache is not Metal's own
/// (MoltenVK's);
/// `moltenvk-argument-buffers-unread` on a MoltenVK whose argument-buffer state the runtime does
/// not read; and `float-tininess-after-rounding` where its profile's float functions are not
/// Metal's own (MoltenVK's).
pub fn creation_kinds(device: &Device) -> BTreeSet<Degradation> {
    let mut kinds = BTreeSet::new();
    let compile = device.compile_device();
    if !compile
        .profile()
        .takes_driver_deviation(metal2vulkan::error::DriverDeviation::MetalFloatFunctions)
    {
        kinds.insert(Degradation::FloatTininessAfterRounding);
    }
    let properties = device.properties();
    // The argument-buffer state of a release the runtime reads, from its limits: on, or off on an
    // Apple GPU.
    let description = &properties.description;
    let limits = &description.limits;
    let uab = limits.max_per_stage_descriptor_update_after_bind_sampled_images;
    let read = MOLTENVK_ARGUMENT_BUFFERS_READ.contains(&description.driver_version)
        && (uab == 1_000_000
            || (description.vendor_id == 0x106b
                && uab == limits.max_per_stage_descriptor_sampled_images));
    if description.driver_id == Some(metal2vulkan::vk::DriverId::MOLTENVK) && !read {
        kinds.insert(Degradation::MoltenVkArgumentBuffersUnread);
    }
    // MoltenVK's kept cache is Metal's own, said in an informational message, no notice
    // (`creation_heard_checked`).
    if driver_keeps_pipeline_cache(device) && !kept_cache_is_metals(device) {
        kinds.insert(Degradation::DriverPipelineCacheKept);
    }
    kinds
}

/// Whether the internal pipeline cache `device`'s driver keeps is Metal's own shader cache, as the
/// runtime decides it (`device/create.rs`'s `kept_cache_is_metals`): MoltenVK's, whose pipelines
/// are Metal pipeline states that Metal caches on the host.
fn kept_cache_is_metals(device: &Device) -> bool {
    device.properties().description.driver_id == Some(metal2vulkan::vk::DriverId::MOLTENVK)
}

/// Whether `device`, as it was made, said in an informational message ([`METAL_CACHE_KEPT`], no
/// notice) that its driver keeps its internal pipeline cache, which is Metal's own: the driver
/// keeps it ([`driver_keeps_pipeline_cache`], read from the driver: it lists no
/// `VK_KHR_pipeline_binary`, the device has no `VK_KHR_maintenance5`, or the driver does not let it
/// be turned off) and the cache is Metal's ([`kept_cache_is_metals`]).
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
fn metal_cache_kept(device: &Device) -> bool {
    driver_keeps_pipeline_cache(device) && kept_cache_is_metals(device)
}

/// The options of a lane device: the validation layer reporting to `messages`, its shader cache in
/// `cache` (`runtime_case::validated_options`, the executor's too), and the checks the lane's
/// switch enables ([`LaneChecks`]).
pub fn validated_options(messages: &Messages, cache: &Path) -> DeviceOptions {
    let mut options = crate::candidate::runtime_case::validated_options(messages, cache);
    LaneChecks::switched().apply(&mut options);
    options
}

/// Whether the runtime makes an untessellated vertex pipeline's topology and restart dynamic on
/// `device`: the profile's test (`pipeline::link::graphics`), both extended dynamic states, which
/// the runtime enables wherever the device has them. There a strip takes the pipeline its class's
/// list made.
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
fn dynamic_input_assembly(device: &Device) -> bool {
    use metal2vulkan::device::VkFeature;
    let enabled = &device.compile_device().profile().enabled().features;
    [
        VkFeature::ExtendedDynamicState,
        VkFeature::ExtendedDynamicState2,
    ]
    .iter()
    .all(|feature| enabled.contains(feature))
}

/// The validation layer's checks the runtime lane enables on every device it makes beside the
/// layer's defaults: an operational switch for diagnostics, `METAL2VULKAN_LANE_VALIDATION`
/// (`src/env_vars.rs`), default off, never a product path. Its value is a comma-separated list of
/// [`LaneChecks::WORDS`]; any other word, or a value that is not Unicode, stops the lane.
///
/// - `sync`: the layer's synchronization validation (`validate_sync`), which reports a hazard
///   between two accesses the runtime's barriers, waits and submissions leave unordered, inside a
///   command buffer and, at submission, across command buffers and queues. The layer validates
///   at submission by default: `syncval_submit_time_validation` (default on) through 1.4.357,
///   which 1.4.363 deprecates for `syncval_full_validation` (default on) and reports, when asked
///   for, as a settings warning, while an older layer reports the newer setting as unrecognized;
///   so the lane asks for neither and takes the default every release has. With the shaders'
///   accesses through descriptors
///   (`syncval_shader_accesses_heuristic`: the layer's default, off, sees no shader access at
///   all, only transfers, attachments and fixed-function reads). Even so, the layer sees no
///   access through a descriptor array of more than one element (the runtime's set 1, the heap
///   of argument-buffer resources) and none through a buffer device address (a pointer loaded
///   from memory, an argument buffer's buffer, a large-buffer slot), it takes a bound
///   descriptor's whole range from its offset as accessed (draws that write their own bytes of
///   one buffer bound at different offsets read as writing each other's: `int_attributes`,
///   `render_executions`), it reads no indirect draw's counts (a draw of none reads as accessing
///   what its shaders could: the index bounds pass's idle draw in `draw_bases`, `patches`' draws
///   of no patches), and it does not model a fragment shader interlock's critical sections; layer
///   1.4.341 also drops every write-after-write of a bound descriptor it finds while recording.
///   A clean run is evidence for the other accesses alone (`runtime_lane/sync_coverage.rs` holds
///   the controls, and asks the layer whether it sees a kernel's write:
///   `sync_coverage::layer_sees_kernel_writes`).
/// - `load-store`: a render pass's load operations checked against an earlier pass's stores
///   (`syncval_load_op_after_store_op_validation`, off by default), which a layer release that
///   does not know the setting reports as unrecognized (Ubuntu 26.04's 1.4.341 does not; the
///   1.4.357 of the RADV venue does): ask for it where the layer has it.
///
/// The switch takes no GPU-assisted validation (`gpuav_enable`): measured on x86 lavapipe (layer
/// 1.4.341), the layer reports it beside its core checks as a settings warning on every device,
/// then disables it at the first compute pipeline the runtime makes indirect-bindable
/// (`VK_PIPELINE_CREATE_2_INDIRECT_BINDABLE_BIT_EXT`, which it cannot instrument; a device with
/// device-generated commands makes one as it opens, `device/prime.rs`), so it checks nothing.
///
/// Where a check is on, every failing message a lane device reports is also written to stderr as a
/// `LANE-VALIDATION` line naming its test (`runtime_case::report_each_message`), including the
/// messages of a test that never reads its list, and the lane prints once which checks it runs
/// (`LANE-VALIDATION checks: ...`). The executor's own devices take none of this
/// (`runtime_run`: their settings are the executor's).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LaneChecks {
    /// `sync`: synchronization validation.
    pub synchronization: bool,
    /// `load-store`: load operations checked against earlier stores.
    pub load_after_store: bool,
}

impl LaneChecks {
    /// The words the switch takes, in its order.
    pub const WORDS: [&'static str; 2] = ["sync", "load-store"];

    /// The checks `value` names: comma-separated words of [`LaneChecks::WORDS`], spaces around
    /// them ignored, an empty value none.
    pub fn parse(value: &str) -> Result<LaneChecks, String> {
        let mut checks = LaneChecks::default();
        for word in value
            .split(',')
            .map(str::trim)
            .filter(|word| !word.is_empty())
        {
            match word {
                "sync" => checks.synchronization = true,
                "load-store" => checks.load_after_store = true,
                _ => {
                    return Err(format!(
                        "METAL2VULKAN_LANE_VALIDATION names `{word}`, no check of the lane's \
                         (it takes {}, comma-separated)",
                        LaneChecks::WORDS.join(", ")
                    ))
                }
            }
        }
        Ok(checks)
    }

    /// The checks the switch enables in this process, read once. A value the switch refuses
    /// stops the lane: a diagnostic run asked for checks it would not get.
    pub fn switched() -> LaneChecks {
        static SWITCHED: OnceLock<LaneChecks> = OnceLock::new();
        *SWITCHED.get_or_init(|| {
            let checks = match metal2vulkan::env_vars::lane_validation() {
                None => LaneChecks::default(),
                Some(value) => value
                    .to_str()
                    .ok_or_else(|| {
                        format!("METAL2VULKAN_LANE_VALIDATION is not Unicode: {value:?}")
                    })
                    .and_then(LaneChecks::parse)
                    .unwrap_or_else(|why| panic!("{why}")),
            };
            if checks != LaneChecks::default() {
                report_each_message();
                eprintln!(
                    "LANE-VALIDATION checks: synchronization {}, load after store {}",
                    checks.synchronization, checks.load_after_store
                );
            }
            checks
        })
    }

    /// The validation layer's settings the checks take.
    pub fn layer_settings(self) -> Vec<LayerSetting> {
        let synchronization = [c"validate_sync", c"syncval_shader_accesses_heuristic"];
        let load_after_store = [c"syncval_load_op_after_store_op_validation"];
        let on = |names: &[&std::ffi::CStr]| -> Vec<LayerSetting> {
            names
                .iter()
                .map(|name| {
                    LayerSetting::new(
                        VALIDATION_LAYER.to_owned(),
                        (*name).to_owned(),
                        LayerSettingValue::Bool(true),
                    )
                })
                .collect()
        };
        let mut settings = Vec::new();
        if self.synchronization {
            settings.extend(on(&synchronization));
        }
        if self.load_after_store {
            settings.extend(on(&load_after_store));
        }
        settings
    }

    /// `options` with the checks' settings, each in place of a setting of the same name, where
    /// `options` enables the validation layer (a device a test makes without it, to time it,
    /// stays without it, and takes no setting of a layer it lacks).
    pub fn apply(self, options: &mut DeviceOptions) {
        if !options
            .layers
            .iter()
            .any(|layer| layer.as_c_str() == VALIDATION_LAYER)
        {
            return;
        }
        for setting in self.layer_settings() {
            options
                .layer_settings
                .retain(|kept| kept.layer != setting.layer || kept.name != setting.name);
            options.layer_settings.push(setting);
        }
    }
}

/// The names of the committed cases whose AIR is a public fixture, in shard order.
pub fn public_case_names() -> Result<Vec<String>, String> {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("corpus");
    let store = CorpusStore::new(&root);
    let public: std::collections::HashSet<(String, String)> = crate::source::public_sources()?
        .into_iter()
        .map(|row| (row.air_sha256, row.entry))
        .collect();
    let mut names = Vec::new();
    for shard in 0..64 {
        for case in store.read_case_shard(shard)? {
            if public.contains(&(case.air_sha256.clone(), case.entry.clone())) {
                names.push(case.name);
            }
        }
    }
    Ok(names)
}

/// Whether `compute` needs a resource a later runtime step adds: anything but buffers,
/// threadgroup memory, an explicit imageblock, a tile pass, rasterization-rate maps, textures,
/// samplers, argument buffers of them, and stage inputs. ([`texture_with`] refuses the textures of
/// later steps: multisampled textures outside a render pass.) Visible-function tables run on the executor's runtime path, which links each entry's
/// module (`runtime_case::visible_function_tables`); a [`PublicCase`] carries no linked module, so
/// the lane leaves them to that path and binds none.
pub fn needs_later_steps(compute: &ComputeCase) -> bool {
    !compute.function_tables.is_empty()
        || !compute.acceleration_structures.is_empty()
        || !compute.command_buffers.is_empty()
}

/// A committed public kernel case for the runtime: the case, its AIR bitcode, its inputs as the
/// engine maps them, the usage the Metal oracle gives each texture, and the output its
/// committed Metal observation records.
pub struct PublicCase {
    pub case: AuthoredCase,
    pub bitcode: Vec<u8>,
    pub compute: ComputeCase,
    /// By Metal texture slot: the role united with the declared access (`texture_usage`).
    pub texture_usage: BTreeMap<u32, TextureUsage>,
    /// By argument-buffer texture, `(owner, field offset)`: the role united with the declared
    /// access of its AIR member (`argument_texture_usage`).
    pub argument_texture_usage: BTreeMap<(u32, u32), TextureUsage>,
    pub metal_output: Vec<u8>,
}

/// The committed case named `name` whose AIR is a public fixture, assembled with `llvm-as` and
/// mapped, with the output of its current Metal observation in [`METAL_ENVIRONMENT`].
pub fn public_case(name: &str) -> Result<PublicCase, String> {
    let mut found = public_cases(|case| case.name == name)?;
    match found.pop() {
        Some((_, case)) if found.is_empty() => case,
        Some(_) => Err(format!("several committed cases are named {name}")),
        None => Err(format!("no committed case is named {name}")),
    }
}

/// A committed case by name, as [`public_case`] gives it or why it cannot.
pub type NamedCase = (String, Result<PublicCase, String>);

/// Each committed case `select` takes, in shard order ([`NamedCase`]); one pass over the case
/// shards and the public fixtures.
pub fn public_cases(select: impl Fn(&AuthoredCase) -> bool) -> Result<Vec<NamedCase>, String> {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("corpus");
    let store = CorpusStore::new(&root);
    let mut selected = Vec::new();
    for shard in 0..64 {
        selected.extend(store.read_case_shard(shard)?.into_iter().filter(&select));
    }
    if selected.is_empty() {
        return Ok(Vec::new());
    }
    let rows = crate::source::public_sources()?;
    Ok(selected
        .into_iter()
        .map(|case| {
            let name = case.name.clone();
            let mapped = map_case(&store, &rows, case);
            (name, mapped)
        })
        .collect())
}

/// `case` as [`public_case`] gives it, its AIR among the public fixture `rows`.
fn map_case(
    store: &CorpusStore,
    rows: &[crate::source::SourceRow],
    case: AuthoredCase,
) -> Result<PublicCase, String> {
    let name = case.name.clone();
    let row = rows
        .iter()
        .find(|row| row.air_sha256 == case.air_sha256 && row.entry == case.entry)
        .ok_or_else(|| format!("{name}: its AIR is no public fixture"))?;
    let scratch = crate::ScratchDir::new("runtime-lane-case")?;
    let ll = scratch.path().join("case.ll");
    std::fs::write(&ll, &row.air_ll).map_err(|error| format!("{name}: {error}"))?;
    let bitcode = llvm_assemble(&ll).map_err(|error| match error {
        AssembleError::Missing(why) => format!("{name}: no llvm-as: {why}"),
        AssembleError::Rejected(why) => format!("{name}: llvm-as rejected the fixture: {why}"),
    })?;
    let decoded = DecodedModule::from_bitcode(&case.entry, &bitcode)
        .map_err(|error| format!("{name}: {error:?}"))?;
    let constants = crate::check::case_constants(&case)?;
    let interface = AirInterface::new(
        decoded.module(),
        decoded.abi().map_err(|error| format!("{name}: {error}"))?,
        &case.entry,
        &constants,
    )?;
    let compute = compute_case(&case, &interface).map_err(|error| format!("{name}: {error:?}"))?;
    let texture_usage =
        texture_usage(&case, &interface).map_err(|error| format!("{name}: {error}"))?;
    let argument_texture_usage =
        argument_texture_usage(&compute.argument_buffer_textures, &interface)
            .map_err(|error| format!("{name}: {error}"))?;
    let metal_output = metal_output(store, &case)?;
    Ok(PublicCase {
        case,
        bitcode,
        compute,
        texture_usage,
        argument_texture_usage,
        metal_output,
    })
}

/// The output of `case`'s current Metal observation in [`METAL_ENVIRONMENT`], from `store`.
pub(crate) fn metal_output(store: &CorpusStore, case: &AuthoredCase) -> Result<Vec<u8>, String> {
    let name = &case.name;
    let metal = MetalObservation::current(
        store.read_metal_of_case(case)?,
        case,
        METAL_ENVIRONMENT,
        ORACLE_ABI,
    )
    .map_err(|error| format!("{name}: {error:?}"))?;
    base64::engine::general_purpose::STANDARD
        .decode(
            metal
                .observed_output()
                .ok_or_else(|| format!("{name}: the Metal row records no output"))?,
        )
        .map_err(|error| format!("{name}: {error}"))
}

/// The pipeline state of `case`'s entry on `device`, specialized with the case's constant values
/// as the Metal oracle specializes it (`newFunctionWithName:constantValues:`), of the oracle's
/// descriptor ([`compute_descriptor`]: its stage-input descriptor).
pub fn pipeline_state(device: &Device, case: &PublicCase) -> Result<ComputePipelineState, String> {
    let function = library_function(device, case)?;
    device
        .new_compute_pipeline_state(
            &compute_descriptor(&function, LinkedFunctions::new(), &case.compute),
            &PipelineOptions::default(),
        )
        .map_err(|error| format!("{}: pipeline state: {error}", case.case.name))
}

impl CaseResources {
    /// `case`'s textures (holding their bytes) and sampler states on `device`, bound and in
    /// argument buffers, and its argument-buffer buffers.
    pub fn new(device: &Device, case: &PublicCase) -> Result<CaseResources, String> {
        Ok(CaseResources::of(
            device,
            &case.compute,
            &case.texture_usage,
            &case.argument_texture_usage,
        )?)
    }
}

/// What a case's run through the runtime observed: its output bytes, the command buffer's
/// execution report, and its execution record (what each dispatch and draw ran).
pub struct RuntimeRun {
    pub output: Vec<u8>,
    pub report: StatusReport,
    pub executions: Vec<Execution>,
}

/// Run `case` once on `device`: one command buffer with one compute encoder, or for a tile kernel
/// one render encoder of its pass and its tile dispatch (`runtime_case::encoded_tile`).
pub fn run_case(device: &Device, case: &PublicCase) -> Result<RuntimeRun, String> {
    if needs_later_steps(&case.compute) {
        return Err(format!(
            "{}: the case needs resources of later runtime steps",
            case.case.name
        ));
    }
    if let Some(pass) = &case.compute.tile {
        let function = library_function(device, case)?;
        let pipeline = device
            .new_tile_render_pipeline_state(
                &tile_descriptor(&function, LinkedFunctions::new(), pass),
                &PipelineOptions::default(),
            )
            .map_err(|error| format!("{}: tile pipeline state: {error}", case.case.name))?;
        let committed = encoded_tile(
            device,
            &case.compute,
            &case.texture_usage,
            &case.argument_texture_usage,
            &pipeline,
            BTreeMap::new(),
        )?
        .commit()?;
        return completed(case, committed);
    }
    let pipeline = pipeline_state(device, case)?;
    run_with(device, case, &pipeline)
}

/// [`run_case`] with `pipeline`, the pipeline state of `case`'s entry.
pub fn run_with(
    device: &Device,
    case: &PublicCase,
    pipeline: &ComputePipelineState,
) -> Result<RuntimeRun, String> {
    let committed = commit_case(
        device,
        &case.compute,
        &case.texture_usage,
        &case.argument_texture_usage,
        pipeline,
        BTreeMap::new(),
    )?;
    completed(case, committed)
}

/// What `case`'s `committed` run observed once its command buffer completed.
fn completed(case: &PublicCase, committed: Committed) -> Result<RuntimeRun, String> {
    let command_buffer = &committed.command_buffer;
    command_buffer
        .wait_until_completed_for(WAIT)
        .map_err(|_| format!("{}: the command buffer did not complete", case.case.name))?;
    if command_buffer.status() != CommandBufferStatus::Completed {
        return Err(format!(
            "{}: the command buffer failed: {:?}",
            case.case.name,
            command_buffer.error()
        ));
    }
    let report = command_buffer
        .execution_report()
        .ok_or("a completed command buffer has an execution report")?;
    let output = committed.output(&case.compute)?;
    Ok(RuntimeRun {
        output,
        report,
        executions: committed.executions,
    })
}

/// The kinds the making of `case`'s compute pipeline state ([`pipeline_state`]) announces where
/// the device has not announced them (`degrade_notices::made_compute_kinds` of its function).
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
fn case_making_kinds(device: &Device, case: &PublicCase) -> BTreeSet<Degradation> {
    let function = library_function(device, case).unwrap();
    degrade_notices::made_compute_kinds(device, function.function_descriptor())
}

/// The kinds a dispatch of `case`'s compute pipeline state ([`pipeline_state`]) may announce by
/// the values it pushes (`degrade_notices::bound_value_kinds` of its function).
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
fn case_bound_value_kinds(device: &Device, case: &PublicCase) -> BTreeSet<Degradation> {
    let function = library_function(device, case).unwrap();
    degrade_notices::bound_value_kinds(device, function.function_descriptor())
}

/// The kind a test's first `Shared` buffer or texture of the default CPU cache mode forces on
/// `device`: `shared-in-system-memory` where it has no unified memory
/// (`degrade_notices::shared_in_system_memory`).
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
fn shared_kinds(device: &Device) -> BTreeSet<Degradation> {
    let shared = degrade_notices::shared_in_system_memory(device);
    degrade_notices::forced(&[(Degradation::SharedInSystemMemory, shared)])
}

/// Whether `got`, a `float` channel a module converted from a texel's codes, holds as Metal's
/// `want`: bit for bit where `exact` (the device samples through Metal's own texture unit,
/// `pixel_samplers::metal_sampler`), else within one `f32` unit in the last place, the bound
/// `PrecisionNote::DeviceTexelConversion` records of an 8- or 16-bit normalized code (lavapipe
/// multiplies by the reciprocal); there a NaN holds any NaN (the note's half-float and
/// small-float NaN, the device's encoding).
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
fn converted_holds(got: f32, want: f32, exact: bool) -> bool {
    if got.to_bits() == want.to_bits() {
        return true;
    }
    if exact {
        return false;
    }
    let same_sign = got.is_sign_negative() == want.is_sign_negative();
    (got.is_nan() && want.is_nan())
        || (same_sign
            && got.is_finite()
            && want.is_finite()
            && got.to_bits().abs_diff(want.to_bits()) <= 1)
}

/// Whether `got`, a `float` channel a module read from a texel's 8-bit sRGB color code, holds as
/// Metal's `want` (the code's step of Metal's twelve-bit table over 4095, [`srgb_step`]): bit for
/// bit where `exact` (the device samples through Metal's own texture unit,
/// `pixel_samplers::metal_sampler`), else as the device's decode of the same code
/// (`PrecisionNote::SrgbTexelDecode`, whose precision Vulkan does not set): nearer `want` than
/// Metal's value of either neighbouring code.
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
fn srgb_decode_holds(got: f32, want: f32, exact: bool) -> bool {
    if got.to_bits() == want.to_bits() {
        return true;
    }
    if exact || !got.is_finite() {
        return false;
    }
    let value = |c: u32| (f64::from(srgb_step(c)) / 4095.0) as f32;
    let Some(code) = (0..=255).find(|&c| value(c).to_bits() == want.to_bits()) else {
        return false;
    };
    let off = |v: f32| (f64::from(got) - f64::from(v)).abs();
    [code.checked_sub(1), (code < 255).then_some(code + 1)]
        .into_iter()
        .flatten()
        .all(|n| off(want) < off(value(n)))
}

/// Whether `got` holds as Metal's `half` result `want` (a value on the `half` grid, widened) under
/// `PrecisionNote::NativeHalfConversion`'s recorded bound: within one unit in the last place of
/// the `half`, the device rounding otherwise or not at all, so that a value Metal takes to an
/// infinity may stay finite from 65504 up; a NaN holds any NaN.
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
fn half_within(got: f32, want: f32) -> bool {
    if got.to_bits() == want.to_bits() || (got.is_nan() && want.is_nan()) {
        return true;
    }
    if want.is_infinite() {
        let same_sign = got.is_sign_negative() == want.is_sign_negative();
        return same_sign && got.abs() >= 65504.0;
    }
    // A half's unit in the last place at `want`: 2^-24 among the denormals.
    let exponent = match want.abs() {
        w if w < 2f32.powi(-14) => -24,
        w => w.log2().floor() as i32 - 10,
    };
    want.is_finite() && (got - want).abs() <= 2f32.powi(exponent)
}

/// The function of `case`'s entry, made with its constant values, from a library of its AIR.
fn library_function(device: &Device, case: &PublicCase) -> Result<Function, String> {
    device
        .new_library_with_air(&case.bitcode)
        .map_err(|error| format!("{}: library: {error}", case.case.name))?
        .new_function_with_name_constant_values(&case.compute.entry, &case.compute.constants)
        .map_err(|error| format!("{}: function: {error}", case.case.name))
}

/// The lane's switch ([`LaneChecks`]) as it parses and applies, with no device.
#[cfg(test)]
mod lane_checks {
    use super::{LaneChecks, VALIDATION_LAYER};
    use metal2vulkan_runtime::{DeviceOptions, LayerSetting, LayerSettingValue};

    fn setting(name: &std::ffi::CStr, value: LayerSettingValue) -> LayerSetting {
        LayerSetting::new(VALIDATION_LAYER.to_owned(), name.to_owned(), value)
    }

    #[test]
    fn the_switch_takes_its_words_and_refuses_others() {
        let none = LaneChecks::default();
        let sync = LaneChecks {
            synchronization: true,
            ..none
        };
        let both = LaneChecks {
            synchronization: true,
            load_after_store: true,
        };
        assert_eq!(LaneChecks::parse(""), Ok(none));
        assert_eq!(LaneChecks::parse(" , "), Ok(none));
        assert_eq!(LaneChecks::parse("sync"), Ok(sync));
        assert_eq!(LaneChecks::parse("load-store, sync"), Ok(both));
        assert_eq!(LaneChecks::parse("sync,load-store,sync"), Ok(both));
        let refused = LaneChecks::parse("sync,gpu").unwrap_err();
        assert!(
            refused.contains("`gpu`") && refused.contains("sync, load-store,"),
            "{refused}"
        );
    }

    #[test]
    fn the_checks_replace_a_setting_of_their_name_on_a_validated_device_only() {
        let checks = LaneChecks {
            synchronization: true,
            load_after_store: true,
        };
        let mut options = DeviceOptions::default();
        options.layers.push(VALIDATION_LAYER.to_owned());
        options.layer_settings = vec![
            setting(c"enable_message_limit", LayerSettingValue::Bool(false)),
            setting(c"validate_sync", LayerSettingValue::Bool(false)),
        ];
        checks.apply(&mut options);
        let on = |name| setting(name, LayerSettingValue::Bool(true));
        assert_eq!(
            options.layer_settings,
            vec![
                setting(c"enable_message_limit", LayerSettingValue::Bool(false)),
                on(c"validate_sync"),
                on(c"syncval_shader_accesses_heuristic"),
                on(c"syncval_load_op_after_store_op_validation"),
            ]
        );
        // Applied twice, the same settings.
        let once = options.layer_settings.clone();
        checks.apply(&mut options);
        assert_eq!(options.layer_settings, once);
        // No validation layer (a timing device): no setting of a layer it lacks.
        let mut bare = DeviceOptions::default();
        checks.apply(&mut bare);
        assert!(bare.layer_settings.is_empty());
        // No check: nothing changes.
        let mut untouched = DeviceOptions::default();
        untouched.layers.push(VALIDATION_LAYER.to_owned());
        LaneChecks::default().apply(&mut untouched);
        assert!(untouched.layer_settings.is_empty());
    }
}

/// A census's shards (`in_shard`, `lists_each_shard`), with no device.
#[cfg(test)]
mod shards {
    use super::{in_shard, lists_each_shard};

    #[test]
    fn the_shards_of_a_selection_take_each_case_it_takes_once() {
        let select = |n: &usize| n % 3 != 1;
        for shards in [1, 2, 7, 32] {
            let mut taken: Vec<usize> = (0..shards)
                .flat_map(|shard| {
                    let take = in_shard(select, shard, shards);
                    (0..100).filter(|n| take(n)).collect::<Vec<_>>()
                })
                .collect();
            taken.sort();
            assert_eq!(taken, (0..100).filter(select).collect::<Vec<_>>());
        }
    }

    #[test]
    fn a_shard_list_is_each_shard_once_and_in_order() {
        assert!(lists_each_shard(&[0], 1));
        assert!(lists_each_shard(&[0, 1, 2], 3));
        assert!(!lists_each_shard(&[0, 1], 3));
        assert!(!lists_each_shard(&[0, 1, 2, 3], 3));
        assert!(!lists_each_shard(&[0, 2, 1], 3));
        assert!(!lists_each_shard(&[0, 1, 1], 3));
    }
}
