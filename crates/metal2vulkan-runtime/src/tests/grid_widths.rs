//! Direct `dispatchThreads:` of grids narrower than one threadgroup, at many widths: a kernel that
//! does not observe its threadgroup partition runs each on its flat variant's one `VkPipeline` of
//! the call's threadgroup size (`compute.rs`'s `flat_direct`), so a second pass over the same
//! widths makes none. Split instead, each width is a region of its own local size, a
//! `VkPipeline` each: the first pass makes 100.
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. The kernel is the owned fixture's
//! `scale_and_offset` (`tests::dispatch_pending`), which reads only `thread_position_in_grid`.

use super::dispatch_pending::{dispatch, state, Buffers, Grid};
use crate::{CommandBufferStatus, DeviceKind, DeviceOptions, ShaderCacheOptions, Size};
use std::time::Duration;

/// The widths dispatched, each pass: `1..=WIDTHS` threads.
const WIDTHS: u64 = 100;
/// The call's threadgroup: wider than every grid.
const GROUP: u64 = 128;
/// The longest a command buffer is waited for.
const WAIT: Duration = Duration::from_secs(60);

/// Two passes of [`WIDTHS`] dispatches, each of a grid of its width in threadgroups of [`GROUP`]:
/// every dispatch writes its grid's outputs, the first pass makes at most a pipeline or two, and
/// the second makes none.
#[test]
fn grids_narrower_than_a_threadgroup_take_one_pipeline_at_every_width() {
    let cache = std::env::temp_dir().join(format!("m2v-grid-widths-{}", std::process::id()));
    std::fs::create_dir_all(&cache).unwrap();
    let options = DeviceOptions {
        kinds: match cfg!(m2v_cpu_icd) {
            true => vec![DeviceKind::Cpu],
            false => vec![
                DeviceKind::Discrete,
                DeviceKind::Integrated,
                DeviceKind::Virtual,
            ],
        },
        shader_cache: ShaderCacheOptions {
            directory: Some(cache.clone()),
            ..ShaderCacheOptions::default()
        },
        ..DeviceOptions::default()
    };
    let device = crate::Device::system_default_with(&options).unwrap();
    let queue = device.new_command_queue().unwrap();
    let kernel = state(&device);
    let buffers = Buffers::new(&device);
    let made = || device.work_counts().variant_pipelines_made;
    let mut passes = Vec::new();
    for _ in 0..2 {
        let before = made();
        for width in 1..=WIDTHS {
            let grid = Grid::Threads(Size::new(width, 1, 1), Size::new(GROUP, 1, 1));
            let command_buffer = dispatch(&queue, &kernel, &buffers, grid);
            command_buffer.commit().unwrap();
            command_buffer.wait_until_completed_for(WAIT).unwrap();
            assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
            assert!(
                buffers.written(width),
                "a grid of {width} wrote its outputs"
            );
        }
        passes.push(made() - before);
    }
    println!(
        "GRID-WIDTHS {}: {WIDTHS} widths of 1 to {WIDTHS} threads in threadgroups of {GROUP}, \
         twice: VkPipelines made {passes:?}",
        device.name()
    );
    assert!(
        passes[0] <= 2,
        "the first pass made {} VkPipelines",
        passes[0]
    );
    assert_eq!(passes[1], 0, "the second pass made VkPipelines");
    drop((buffers, kernel, queue, device));
    let _ = std::fs::remove_dir_all(&cache);
}
