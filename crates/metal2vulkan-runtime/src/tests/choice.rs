//! What runs a compute dispatch (`encoder::choose`), with no device: the same on every device. An
//! image-only kernel that does not observe its threadgroup partition runs its grid at
//! `MultiRegion` 0 in workgroups of 16x16x1 where it is one deep and of its tile where it is
//! deeper, with no tile decode and its edge tested where they do not divide the grid. A kernel that
//! does not observe it and is not image-only runs every grid in the tiles the region carries. An
//! observing kernel runs at its own size, but one with its one-row module runs a grid whose
//! threadgroups keep each SIMD group in one row in that module's tiles. A size deeper than
//! `DEEP_THREADGROUP` runs as one row of
//! its threads, so every workgroup is within the floor's `maxComputeWorkGroupSize`, whatever the
//! threadgroup size.

use crate::encoder::{choose, image_tile, Choice, Direct, Dispatch, Module};
use crate::floor::{WORKGROUP_INVOCATIONS, WORKGROUP_SIZE};
use crate::pipeline::{Form, SQUARE};
use metal2vulkan::pipeline::{grid_tile, DEEP_THREADGROUP, GRID_TILE_THREADS};

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

/// An image-only kernel's grid more than one deep runs its whole module in workgroups of its tile
/// at `MultiRegion` 0, its regions carrying them: one pipeline per tile.
#[test]
fn an_image_only_kernels_deeper_grid_runs_in_workgroups_of_its_tile() {
    for threads in [[4, 16, 312], [24, 24, 252], [256, 256, 8], [7, 5, 3]] {
        let tile = image_tile(threads, [4, 4, 4]);
        let choice = choose(
            &Form::Square,
            Some(threads_of(threads, [4, 4, 4])),
            [4, 4, 4],
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

/// An image-only kernel's tile over a deeper grid: of the 256-thread tiles, those launching the
/// fewest threads, then with the fewest tiles partly past the grid, then the depth nearest the
/// threadgroup's (the shallower of two as near), then the most square, then the wider. Over a grid
/// every such tile divides, the threadgroup's depth decides: 8x8x4 for 8x8x4 and 4x4x4, the square
/// 16x16x1 for any one deep, as deep as the grid for a deeper one. Fewer threads launched comes
/// first: a column runs in one tile of its own, and a grid four wide in tiles four wide.
#[test]
fn an_image_only_kernels_tile_follows_the_threadgroups_depth() {
    for (threads, size, tile) in [
        ([512, 512, 16], [8, 8, 4], [8, 8, 4]),
        ([512, 512, 16], [4, 4, 4], [8, 8, 4]),
        ([512, 512, 16], [8, 8, 1], [16, 16, 1]),
        ([512, 512, 16], [16, 16, 1], [16, 16, 1]),
        ([512, 512, 16], [32, 32, 1], [16, 16, 1]),
        ([512, 512, 16], [8, 8, 2], [16, 8, 2]),
        ([512, 512, 16], [8, 8, 3], [16, 8, 2]),
        ([512, 512, 16], [1, 1, 64], [4, 4, 16]),
        ([64, 64, 16], [8, 8, 4], [8, 8, 4]),
        ([1, 1, 192], [8, 8, 4], [1, 1, 256]),
        ([4, 16, 252], [8, 8, 4], [4, 16, 4]),
        ([4, 16, 252], [8, 8, 1], [4, 16, 4]),
    ] {
        assert_eq!(image_tile(threads, size), tile, "{threads:?} at {size:?}");
    }
    // Every tile is 256 threads, whatever the grid and threadgroup.
    for threads in [
        [152, 152, 6],
        [40, 40, 90],
        [7, 5, 3],
        [1, 1, 2],
        [1000, 3, 7],
    ] {
        for size in [[8, 8, 4], [1, 1, 1], [32, 1, 32], [4, 4, 64]] {
            let tile = image_tile(threads, size);
            assert_eq!(
                tile.iter().product::<u32>(),
                GRID_TILE_THREADS,
                "{threads:?}"
            );
            assert!(
                tile.iter().all(|side| side.is_power_of_two()),
                "{threads:?}"
            );
        }
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
    let whole = choose(&sized, Some(threads_of([64, 64, 1], [8, 8, 1])), [8, 8, 1]);
    assert_eq!((whole.module, whole.local), (Module::Whole, [8, 8, 1]));
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
                direct: None,
            }
        );
    }
}

/// The workgroup a pipeline of local size `local` runs: one row of its threads where it is deeper
/// than `DEEP_THREADGROUP` (`DeepThreadgroup` 1), else the threadgroup.
fn workgroup(local: [u32; 3]) -> [u32; 3] {
    match local[2] > DEEP_THREADGROUP {
        true => [local.iter().product(), 1, 1],
        false => local,
    }
}

/// A threadgroup 320 deep: every kernel runs it in a workgroup within the floor's sizes, the same
/// on every device. One that does not observe its partition runs it square, tiled or of its tile,
/// over a grid one deep and a deeper one alike; an observing kernel at its own size, as one row of
/// its 320 threads, and so does an indirect dispatch. An image-only kernel whose grid's tile is
/// deeper than `DEEP_THREADGROUP` (a column 1,024 deep, whose tile is 1 by 1 by 256) runs it whole,
/// as one row of the tile's threads.
#[test]
fn a_threadgroup_deeper_than_the_floor_runs_as_one_row() {
    let within = |local: [u32; 3]| {
        let at = workgroup(local);
        (0..3).all(|axis| at[axis] <= WORKGROUP_SIZE[axis])
            && at.iter().product::<u32>() <= WORKGROUP_INVOCATIONS
    };
    let size = [1, 1, 320];
    for threads in [[4, 16, 320], [64, 64, 1], [1, 1, 1024]] {
        let direct = Some(threads_of(threads, size));
        for form in [Form::Tiled, Form::Square, Form::Sized { partial: None }] {
            let choice = choose(&form, direct, size);
            assert!(within(choice.local), "{threads:?}: {:?}", choice.local);
        }
        let sized = choose(&Form::Sized { partial: None }, direct, size);
        assert_eq!(workgroup(sized.local), [320, 1, 1], "{threads:?}");
    }
    let indirect = choose(&Form::Sized { partial: None }, None, size);
    assert_eq!(workgroup(indirect.local), [320, 1, 1]);
    let column = [1, 1, 1024];
    assert_eq!(grid_tile(column), [1, 1, 256]);
    let choice = choose(&Form::Square, Some(threads_of(column, size)), size);
    assert_eq!(
        (choice.module, choice.local, choice.nominal),
        (Module::Whole, [1, 1, 256], [1, 1, 256])
    );
    assert_eq!(workgroup(choice.local), [256, 1, 1]);
}

/// The grid `ComputePipelineState::prepare_dispatch` decides from (`Dispatch::direct`) is the
/// one each dispatch call forms, so the pre-pass and the dispatch choose alike: a
/// `dispatchThreadgroups` grid is its counts with their threads, a product that passes `u32::MAX`
/// saturating, and never ragged, so an observing kernel runs its whole module there; a
/// `dispatchThreads` grid of exactly `u32::MAX` threads on a size that does not divide it is
/// ragged, its partial-threadgroups variant; an indirect dispatch has no grid, its whole module
/// at its own size.
#[test]
fn a_prepared_dispatch_takes_the_grid_its_call_forms() {
    let sized = Form::Sized { partial: None };
    let size = [2, 1, 1];
    let saturated = Dispatch::Threadgroups([u32::MAX, 3, 1]).direct(size);
    let want = Direct {
        groups: [u32::MAX, 3, 1],
        threads: [u32::MAX, 3, 1],
        ragged: false,
    };
    assert_eq!(saturated, Some(want));
    assert_eq!(choose(&sized, saturated, size).module, Module::Whole);
    let max = Dispatch::Threads([u32::MAX, 1, 1]).direct(size);
    assert_eq!(max, Some(threads_of([u32::MAX, 1, 1], size)));
    assert_eq!(
        max.map(|max| (max.groups, max.ragged)),
        Some(([1 << 31, 1, 1], true))
    );
    assert_eq!(choose(&sized, max, size).module, Module::Partial);
    let whole = Dispatch::Threads([8, 4, 1]).direct([4, 2, 1]);
    assert_eq!(whole, Some(threads_of([8, 4, 1], [4, 2, 1])));
    assert_eq!(choose(&sized, whole, [4, 2, 1]).module, Module::Whole);
    assert_eq!(Dispatch::Indirect.direct(size), None);
    assert_eq!(choose(&sized, None, size).module, Module::Whole);
}
