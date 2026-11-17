//! Every read the runtime makes of memory the GPU writes. A guest shader holds raw device
//! addresses, so any value read here may be forged. Each function bounds what its value can
//! decide to the reading command buffer's own guest-visible results: it never fails, stalls or
//! ends anything else, and never indexes host memory with the value. A rule test holds every such
//! read to this module, and the runtime's README lists the sites.

use ash::vk;

/// A completed command buffer's two timestamps, from its queue's pool; None where the driver does
/// not report both, so the command buffer's GPU times stay 0.
///
/// Bound: the values only decide that command buffer's `GPUStartTime`, through arithmetic defined
/// for every value that gives a time no later than its `GPUEndTime`. The read does not wait.
///
/// # Safety
///
/// `pool` is a timestamp pool of `device` with the queries `first` and `first + 1`.
pub(crate) unsafe fn timestamps(
    device: &ash::Device,
    pool: vk::QueryPool,
    first: u32,
) -> Option<(u64, u64)> {
    let mut values = [0_u64; 2];
    // SAFETY: by the caller; `values` holds two 64-bit results.
    let read = unsafe {
        device.get_query_pool_results(pool, first, &mut values, vk::QueryResultFlags::TYPE_64)
    };
    read.ok().map(|()| (values[0], values[1]))
}
