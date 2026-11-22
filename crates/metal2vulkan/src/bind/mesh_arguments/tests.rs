//! Metal's object grid on the CPU, as Metal runs it.

use super::*;

/// Metal's object grids, as the M3 Max ran the owned fixture `mesh_object_grid` (macOS 27.0.1):
/// threads per grid wrap to 32 bits along each axis; an indirect draw runs every threadgroup, a
/// direct one those the wrapped threads fill, rounded up.
#[test]
fn metal_s_object_grid_wraps_its_threads_per_grid() {
    for (groups, threads, direct, runs, reported) in [
        (
            [70_000, 1, 1],
            [1, 1, 1],
            true,
            [70_000, 1, 1],
            [70_000, 1, 1],
        ),
        (
            [300, 300, 2],
            [4, 2, 1],
            true,
            [300, 300, 2],
            [1200, 600, 2],
        ),
        (
            [4_194_303, 1, 1],
            [1024, 1, 1],
            true,
            [4_194_303, 1, 1],
            [4_294_966_272, 1, 1],
        ),
        ([4_194_304, 1, 1], [1024, 1, 1], true, [0, 1, 1], [0, 1, 1]),
        (
            [4_194_304, 1, 1],
            [1024, 1, 1],
            false,
            [4_194_304, 1, 1],
            [0, 1, 1],
        ),
        (
            [4_194_305, 1, 1],
            [1024, 1, 1],
            true,
            [1, 1, 1],
            [1024, 1, 1],
        ),
        (
            [4_194_305, 1, 1],
            [1024, 1, 1],
            false,
            [4_194_305, 1, 1],
            [1024, 1, 1],
        ),
        (
            [4_294_968, 1, 1],
            [1000, 1, 1],
            true,
            [1, 1, 1],
            [704, 1, 1],
        ),
        (
            [4_294_970, 1, 1],
            [1000, 1, 1],
            true,
            [3, 1, 1],
            [2704, 1, 1],
        ),
        (
            [4_294_970, 1, 1],
            [1000, 1, 1],
            false,
            [4_294_970, 1, 1],
            [2704, 1, 1],
        ),
        (
            [2, 4_194_305, 1],
            [1, 1024, 1],
            true,
            [2, 1, 1],
            [2, 1024, 1],
        ),
        (
            [65536, 64, 1],
            [1024, 1, 1],
            true,
            [65536, 64, 1],
            [67_108_864, 64, 1],
        ),
    ] {
        assert_eq!(
            metal_object_grid(groups, threads, direct),
            Some((runs, reported)),
            "{groups:?} of {threads:?}, direct {direct}"
        );
    }
    assert_eq!(metal_object_grid([3, 1, 1], [0, 1, 1], true), None);
}
