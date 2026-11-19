//! Every read the runtime makes of memory the GPU writes. A guest shader holds raw device
//! addresses, so any value read here may be forged. Each function bounds what its value can
//! decide to the reading command buffer's own guest-visible results: it never fails, stalls or
//! ends anything else, and never indexes host memory with the value. A rule test holds every such
//! read to this module, and the runtime's README lists the sites.

use ash::vk;

/// A completed command buffer's two timestamps, from its pair in the device's pools; None where
/// the driver does not report both available, so the command buffer's GPU times stay 0.
///
/// The read does not wait (no `VK_QUERY_RESULT_WAIT_BIT`). The caller reads a pair only once the
/// completion thread has seen its submission's progress value. The signal's first synchronization
/// scope "includes every command in the same batch that the signal operation is defined in, and
/// all commands that occur earlier in submission order" (`vkQueueSubmit2`, all commands' stages);
/// the host wait that sees the value happens after the signal ("signal operations happening
/// before host operations that wait on them"); and "a query is made available by the operation of
/// ... `vkCmdWriteTimestamp2`". So both queries of a submission that ran are available, and a
/// read without the wait reports them. A submission the device cancelled at a reset may still
/// signal its value having run nothing: its queries stay unavailable ("`vkGetQueryPoolResults`
/// may return `VK_NOT_READY` if there are queries in the unavailable state"), and the read reports
/// no times at once, where a waiting read runs until the driver gives up (RADV: twice the
/// kernel's 60 s compute timeout).
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
