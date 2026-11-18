//! What runs a compute dispatch (`encoder::choose`), with no device. An image-only kernel that does
//! not observe its threadgroup partition runs a grid one deep in 16x16x1 workgroups at
//! `MultiRegion` 0, with no tile decode and its edge tested where they do not divide the grid. A
//! kernel that does not observe it and is not image-only runs such a grid in the tiles the region
//! carries. Either's grid more than one deep runs in tiles on a pipeline that folds the tile. An
//! observing kernel runs at its own size. Every workgroup a kernel that does not observe its
//! partition runs is within the floor's `maxComputeWorkGroupSize`, whatever its threadgroup size.

use crate::encoder::{choose, fits, Choice, Direct, Module};
use crate::pipeline::{Form, SQUARE};
use ash::vk;
use metal2vulkan::pipeline::{grid_tile, GRID_TILE_THREADS};

/// A `dispatchThreads` grid of `threads` in threadgroups of `size`.
fn threads_of(threads: [u32; 3], size: [u32; 3]) -> Direct {
    Direct {
        groups: std::array::from_fn(|axis| threads[axis].div_ceil(size[axis])),
        threads,
        ragged: (0..3).any(|axis| !threads[axis].is_multiple_of(size[axis])),
    }
}

/// An image-only kernel's grid one deep runs its whole module in 16x16x1 workgroups, which its
/// regions carry, so its pipeline's `MultiRegion` is 0 and the kernel reads no tile: whole where
/// they divide the grid, flagged where they do not.
#[test]
fn an_image_only_kernels_grid_one_deep_runs_in_square_workgroups() {
    for (threads, groups, ragged) in [
        ([1984, 1088, 1], [124, 68, 1], false),
        ([100, 60, 1], [7, 4, 1], true),
        ([300, 1, 1], [19, 1, 1], true),
    ] {
        let choice = choose(
            &Form::Square,
            Some(threads_of(threads, [8, 8, 1])),
            [8, 8, 1],
        );
        assert_eq!(
            choice,
            Choice {
                module: Module::Whole,
                local: SQUARE,
                nominal: SQUARE,
                fold: None,
                direct: Some((threads, groups, ragged)),
            },
            "{threads:?}"
        );
    }
}

/// A kernel that is not image-only runs a grid one deep in the tiles its regions carry, on its
/// tiles' pipeline, which folds none.
#[test]
fn a_tiling_kernels_grid_one_deep_runs_in_its_regions_tiles() {
    for threads in [[1984, 1088, 1], [100, 60, 1], [300, 1, 1]] {
        let tile = grid_tile(threads);
        let choice = choose(
            &Form::Tiled,
            Some(threads_of(threads, [32, 1, 1])),
            [32, 1, 1],
        );
        assert_eq!(choice.module, Module::Tiles, "{threads:?}");
        assert_eq!(choice.local, [GRID_TILE_THREADS, 1, 1], "{threads:?}");
        assert_eq!(choice.nominal, tile, "{threads:?}");
        assert_eq!(choice.fold, None, "{threads:?}");
    }
}

/// A grid more than one deep runs in its tiles on a pipeline that folds the tile, for an image-only
/// kernel and any other that does not observe its partition alike: one pipeline per tile.
#[test]
fn a_grid_more_than_one_deep_runs_on_a_pipeline_that_folds_its_tile() {
    for form in [Form::Tiled, Form::Square] {
        for threads in [[4, 16, 312], [24, 24, 252], [128, 128, 64], [7, 5, 3]] {
            let tile = grid_tile(threads);
            let choice = choose(&form, Some(threads_of(threads, [4, 4, 4])), [4, 4, 4]);
            let groups = std::array::from_fn(|axis| threads[axis].div_ceil(tile[axis]));
            let ragged = (0..3).any(|axis| !threads[axis].is_multiple_of(tile[axis]));
            assert_eq!(
                choice,
                Choice {
                    module: Module::Tiles,
                    local: [GRID_TILE_THREADS, 1, 1],
                    nominal: tile,
                    fold: Some(tile),
                    direct: Some((threads, groups, ragged)),
                },
                "{threads:?}"
            );
        }
    }
}

/// An observing kernel runs at its own size, a ragged grid on its partial-threadgroups variant;
/// an indirect dispatch of any kernel runs its whole module at its own size.
#[test]
fn an_observing_kernel_and_an_indirect_dispatch_run_at_their_own_size() {
    let sized = Form::Sized { partial: None };
    let whole = choose(&sized, Some(threads_of([64, 64, 1], [8, 8, 1])), [8, 8, 1]);
    assert_eq!(
        (whole.module, whole.local, whole.fold),
        (Module::Whole, [8, 8, 1], None)
    );
    let ragged = choose(&sized, Some(threads_of([60, 64, 1], [8, 8, 1])), [8, 8, 1]);
    assert_eq!(ragged.module, Module::Partial);
    for form in [Form::Tiled, Form::Square, Form::Sized { partial: None }] {
        let indirect = choose(&form, None, [4, 2, 1]);
        assert_eq!(
            indirect,
            Choice {
                module: Module::Whole,
                local: [4, 2, 1],
                nominal: [4, 2, 1],
                fold: None,
                direct: None,
            }
        );
    }
}

/// A threadgroup 320 deep, on a device whose `maxComputeWorkGroupSize` is 1,024 by 1,024 by 64 (the
/// floor's, and NVIDIA's): a kernel that does not observe its partition runs it in a workgroup
/// within the device's sizes, square or tiled, over a grid one deep and a deeper one alike; an
/// observing kernel's workgroup is the threadgroup, past the device's depth, which the encoder
/// refuses (`Limit::ThreadgroupShape`).
#[test]
fn a_threadgroup_deeper_than_the_device_runs_only_where_it_is_not_the_workgroup() {
    let limits = vk::PhysicalDeviceLimits {
        max_compute_work_group_size: [1024, 1024, 64],
        ..Default::default()
    };
    let max = limits.max_compute_work_group_size;
    let size = [1, 1, 320];
    for threads in [[4, 16, 320], [64, 64, 1]] {
        let direct = Some(threads_of(threads, size));
        for form in [Form::Tiled, Form::Square] {
            let choice = choose(&form, direct, size);
            assert!(fits(choice.local, max), "{threads:?}: {:?}", choice.local);
        }
        let sized = choose(&Form::Sized { partial: None }, direct, size);
        assert!(!fits(sized.local, max), "{threads:?}");
    }
}
