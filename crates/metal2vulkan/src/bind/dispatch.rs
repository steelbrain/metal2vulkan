//! A Metal compute dispatch in Metal's terms ([`Dispatch`]), and the threadgroup bounds a tile
//! function's threadgroup takes. How a dispatch becomes Vulkan dispatches (its regions and their
//! `DispatchRegion` push field, `lower::dispatch::region_bytes`), after the pipeline's interface
//! and push fields, is the consumer's.

/// A Metal compute dispatch.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Dispatch {
    /// `dispatchThreadgroups:threadsPerThreadgroup:`.
    Threadgroups {
        groups: [u32; 3],
        threads_per_group: [u32; 3],
    },
    /// `dispatchThreads:threadsPerThreadgroup:`.
    Threads {
        threads: [u32; 3],
        threads_per_group: [u32; 3],
    },
    /// The threadgroups `first .. first + count` of `dispatchThreadgroups:` of `groups`, each
    /// axis's range cut to the grid: each runs as in the whole dispatch (its
    /// `threadgroup_position_in_grid`, and the grid of `groups * threads_per_group` threads),
    /// and no other runs. A tile dispatch over a region (`dispatchThreadsPerTile:inRegion:`)
    /// runs the tiles the region touches so. Its regions take their first threadgroup and the
    /// grid from the `DispatchRegion` push field (`MultiRegion` 1).
    ThreadgroupRange {
        groups: [u32; 3],
        first: [u32; 3],
        count: [u32; 3],
        threads_per_group: [u32; 3],
    },
}

impl Dispatch {
    /// Metal's `threads_per_threadgroup` of the dispatch.
    pub fn threads_per_group(&self) -> [u32; 3] {
        match *self {
            Dispatch::Threadgroups {
                threads_per_group, ..
            }
            | Dispatch::Threads {
                threads_per_group, ..
            }
            | Dispatch::ThreadgroupRange {
                threads_per_group, ..
            } => threads_per_group,
        }
    }
}

/// The widest tile threadgroup, and the tallest: Metal's largest tile, 32x32.
pub(crate) const TILE_THREADGROUP_SIDE: u32 = 32;
