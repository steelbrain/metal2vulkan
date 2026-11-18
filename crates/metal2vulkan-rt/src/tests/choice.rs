//! What runs a compute dispatch (`encoder::choose`), with no device. An image-only kernel that does
//! not observe its threadgroup partition runs its grid at `MultiRegion` 0 in workgroups of 16x16x1
//! where it is one deep and of its tile where it is deeper, with no tile decode and its edge tested
//! where they do not divide the grid. A kernel that does not observe it and is not image-only runs
//! every grid in the tiles the region carries, as an image-only kernel does where its tile is past
//! the device's sizes. An observing kernel runs at its own size. Every workgroup a kernel that does not observe its
//! partition runs is within the device's `maxComputeWorkGroupSize`, whatever its threadgroup size.

use crate::encoder::{choose, fits, Choice, Direct, Module};
use crate::pipeline::{Form, SQUARE};
use ash::vk;
use metal2vulkan::pipeline::{grid_tile, GRID_TILE_THREADS};

/// A `maxComputeWorkGroupSize` of 1,024 by 1,024 by 64: the floor's, and NVIDIA's.
const MAX: [u32; 3] = [1024, 1024, 64];

/// A `dispatchThreads` grid of `threads` in threadgroups of `size`.
fn threads_of(threads: [u32; 3], size: [u32; 3]) -> Direct {
    Direct {
        groups: std::array::from_fn(|axis| threads[axis].div_ceil(size[axis])),
        threads,
        ragged: (0..3).any(|axis| !threads[axis].is_multiple_of(size[axis])),
    }
}

/// Metal's grid `threads` over workgroups of `shape`: the workgroups, and whether they do not
/// divide it.
fn over(threads: [u32; 3], shape: [u32; 3]) -> ([u32; 3], [u32; 3], bool) {
    (
        threads,
        std::array::from_fn(|axis| threads[axis].div_ceil(shape[axis])),
        (0..3).any(|axis| !threads[axis].is_multiple_of(shape[axis])),
    )
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
            MAX,
        );
        assert_eq!(
            choice,
            Choice {
                module: Module::Whole,
                local: SQUARE,
                nominal: SQUARE,
                direct: Some((threads, groups, ragged)),
            },
            "{threads:?}"
        );
    }
}

/// An image-only kernel's grid more than one deep runs its whole module in workgroups of the
/// grid's tile at `MultiRegion` 0, its regions carrying them: one pipeline per tile.
#[test]
fn an_image_only_kernels_deeper_grid_runs_in_workgroups_of_its_tile() {
    for threads in [[4, 16, 312], [24, 24, 252], [256, 256, 8], [7, 5, 3]] {
        let tile = grid_tile(threads);
        let choice = choose(
            &Form::Square,
            Some(threads_of(threads, [4, 4, 4])),
            [4, 4, 4],
            MAX,
        );
        assert_eq!(
            choice,
            Choice {
                module: Module::Whole,
                local: tile,
                nominal: tile,
                direct: Some(over(threads, tile)),
            },
            "{threads:?}"
        );
    }
}

/// A kernel that is not image-only runs every grid, one deep or deeper, in the tiles its regions
/// carry, on its tiles' one pipeline.
#[test]
fn a_tiling_kernel_runs_every_grid_in_its_regions_tiles() {
    for threads in [
        [1984, 1088, 1],
        [100, 60, 1],
        [300, 1, 1],
        [4, 16, 312],
        [24, 24, 252],
        [128, 128, 64],
        [7, 5, 3],
    ] {
        let tile = grid_tile(threads);
        let choice = choose(
            &Form::Tiled,
            Some(threads_of(threads, [32, 1, 1])),
            [32, 1, 1],
            MAX,
        );
        assert_eq!(
            choice,
            Choice {
                module: Module::Tiles,
                local: [GRID_TILE_THREADS, 1, 1],
                nominal: tile,
                direct: Some(over(threads, tile)),
            },
            "{threads:?}"
        );
    }
}

/// An observing kernel runs at its own size, a ragged grid on its partial-threadgroups variant;
/// an indirect dispatch of any kernel runs its whole module at its own size.
#[test]
fn an_observing_kernel_and_an_indirect_dispatch_run_at_their_own_size() {
    let sized = Form::Sized { partial: None };
    let whole = choose(
        &sized,
        Some(threads_of([64, 64, 1], [8, 8, 1])),
        [8, 8, 1],
        MAX,
    );
    assert_eq!((whole.module, whole.local), (Module::Whole, [8, 8, 1]));
    let ragged = choose(
        &sized,
        Some(threads_of([60, 64, 1], [8, 8, 1])),
        [8, 8, 1],
        MAX,
    );
    assert_eq!(ragged.module, Module::Partial);
    for form in [Form::Tiled, Form::Square, Form::Sized { partial: None }] {
        let indirect = choose(&form, None, [4, 2, 1], MAX);
        assert_eq!(
            indirect,
            Choice {
                module: Module::Whole,
                local: [4, 2, 1],
                nominal: [4, 2, 1],
                direct: None,
            }
        );
    }
}

/// A threadgroup 320 deep, on a device whose `maxComputeWorkGroupSize` is 1,024 by 1,024 by 64: a
/// kernel that does not observe its partition runs it in a workgroup within the device's sizes,
/// square, tiled or of its tile, over a grid one deep and a deeper one alike; an observing
/// kernel's workgroup is the threadgroup, past the device's depth, which the encoder refuses
/// (`Limit::ThreadgroupShape`). An image-only kernel whose grid's tile is past the device's depth
/// (a column 1,024 deep, whose tile is 1 by 1 by 256) runs it in the tiles its regions carry.
#[test]
fn a_threadgroup_deeper_than_the_device_runs_only_where_it_is_not_the_workgroup() {
    let limits = vk::PhysicalDeviceLimits {
        max_compute_work_group_size: MAX,
        ..Default::default()
    };
    let max = limits.max_compute_work_group_size;
    let size = [1, 1, 320];
    for threads in [[4, 16, 320], [64, 64, 1], [1, 1, 1024]] {
        let direct = Some(threads_of(threads, size));
        for form in [Form::Tiled, Form::Square] {
            let choice = choose(&form, direct, size, max);
            assert!(fits(choice.local, max), "{threads:?}: {:?}", choice.local);
        }
        let sized = choose(&Form::Sized { partial: None }, direct, size, max);
        assert!(!fits(sized.local, max), "{threads:?}");
    }
    let column = [1, 1, 1024];
    assert_eq!(grid_tile(column), [1, 1, 256]);
    let choice = choose(&Form::Square, Some(threads_of(column, size)), size, max);
    assert_eq!(
        (choice.module, choice.local, choice.nominal),
        (Module::Tiles, [GRID_TILE_THREADS, 1, 1], [1, 1, 256])
    );
}
