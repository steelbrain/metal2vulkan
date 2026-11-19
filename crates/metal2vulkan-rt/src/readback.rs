//! Every read the runtime makes of memory the GPU writes. A guest shader holds raw device
//! addresses, so any value read here may be forged. Each function bounds what its value can
//! decide to the reading command buffer's own guest-visible results: it never fails, stalls or
//! ends anything else, and never indexes host memory with the value. A rule test holds every such
//! read to this module, and the runtime's README lists the sites.

use ash::vk;

/// A completed command buffer's two timestamps, from its pair in the device's pools; None where
/// the driver does not report both available, so the command buffer's GPU times stay 0.
///
/// The read does not wait (no `VK_QUERY_RESULT_WAIT_BIT`). A submission the device cancelled at a
/// reset may still signal its progress value having run nothing, its queries never made
/// available: a waiting read of them runs until the driver gives up (RADV: twice the kernel's
/// 60 s compute timeout), and the read without the wait reports no times at once
/// ("`vkGetQueryPoolResults` may return `VK_NOT_READY` if there are queries in the unavailable
/// state").
///
/// The caller reads a pair only once a host wait of the completion thread has seen its submission's
/// progress value (a wait with a timeout of 0 after the counter read, where the counter told it
/// first). The signal's first synchronization scope "includes every command in the same batch that
/// the signal operation is defined in, and all commands that occur earlier in submission order"
/// (`vkQueueSubmit2`), "a query is made available by the operation of ... `vkCmdWriteTimestamp2`",
/// and the host wait happens after the signal ("signal operations happening before host operations
/// that wait on them"): the pair of a submission that ran is available at the read. A driver that
/// makes timestamps available only after the signal (MoltenVK 1.4.2 on an M3 Max did, for most
/// command buffers; it is below the floor) would read none here and leave the GPU times 0 with no
/// other sign, so the caller counts each pair that reads as not available
/// (`Metrics::stamps_not_available`), and every play and suite leg reports the count.
///
/// Bound: the values only decide that command buffer's `GPUStartTime`, through arithmetic defined
/// for every value that gives a time no later than its `GPUEndTime`.
///
/// # Safety
///
/// `pool` is a timestamp pool of `device` with the queries `first` and `first + 1`, which no other
/// thread resets while it reads.
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

/// Copies `runs` (each an offset in `bytes`, one in `staging`, and a length) from a staging buffer
/// the GPU copied a texture's bytes into, for `Texture::get_bytes`.
///
/// Bound: the bytes only reach the caller's buffer as the texture's bytes; the offsets and the
/// length come from the call's checked region, never from the bytes, and the read does not wait.
///
/// # Safety
///
/// `staging` maps each run's staging offset plus its length, the copy into it completed and its
/// writes are visible to the host, and each run's offset plus its length lies in `bytes`.
pub(crate) unsafe fn texels(
    staging: *const u8,
    bytes: &mut [u8],
    runs: impl Iterator<Item = (usize, usize, usize)>,
) {
    for (to, from, len) in runs {
        // SAFETY: by the caller.
        unsafe {
            staging
                .add(from)
                .copy_to_nonoverlapping(bytes.as_mut_ptr().add(to), len)
        };
    }
}
