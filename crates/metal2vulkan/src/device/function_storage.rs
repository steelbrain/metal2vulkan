//! The Function storage a thread holds correctly: [`ProfileFields::function_storage`].
//!
//! A thread's private memory, its `alloca`s (a recursion's call stack among them) and the constants
//! it holds as `Private` variables, is storage a driver backs with scratch memory. No Vulkan
//! property states how much of it a thread holds, and past it a driver can answer wrongly or hang
//! with no error: measured, the Radeon 780M and the RX 7900 XTX under RADV over 4096 threads at
//! 128 KiB a thread, the Iris Xe under ANV at 48 KiB, the RTX 5080 at 48 KiB and the RTX 3090 at
//! 192 KiB (both Xid 109).
//!
//! Every device holds [`FLOOR`], 16 KiB a thread: half the least any measured device held
//! (32 KiB), and below every size any of them failed at (48 KiB and up). Private memory within it
//! is taken as held on every device, and records nothing. Past it a vertex or fragment function
//! runs up to [`METAL`], 192 KiB a thread, the most a Metal device was measured to hold, and
//! records `PrecisionNote::FunctionStorageAssumed`: past the floor it rests on no measurement, and
//! a driver that holds less can answer wrongly or hang the GPU, as the RTX 5080 at 48 KiB and the
//! RTX 3090 at 192 KiB hung when probed. A kernel, an object or a mesh function past the floor is
//! refused, as is a vertex or fragment function past [`METAL`]: typed (`Unsupported{PrivateMemory}`,
//! or `Unsupported{Recursion}` for a stage with a call stack, needing
//! `VkLimit::FunctionStorageBytesPerThread`).
//!
//! [`ProfileFields::function_storage`]: crate::device::profile::ProfileFields::function_storage

use crate::device::profile::FunctionStorage;
use crate::pipeline::sym::Stage;

const KIB: u64 = 1024;

/// The Function storage every device holds, a thread: 16 KiB (the module docs).
pub(crate) const FLOOR: u64 = 16 * KIB;

/// The most Function storage a thread was measured to hold on Metal: the Apple M3 Max and M4 hold
/// 192 KiB and refuse 256 KiB at pipeline creation. A vertex or fragment function may keep this
/// much, past [`FLOOR`] with `PrecisionNote::FunctionStorageAssumed` (the module docs).
pub(crate) const METAL: u64 = 192 * KIB;

/// The private memory a thread of a `stage` function may keep on a device whose Function storage
/// is `storage`: the device's own value, but [`METAL`] for a vertex or fragment function on a
/// device not measured, which past [`FLOOR`] rests on no measurement (the module docs).
pub(crate) fn ceiling(storage: FunctionStorage, stage: Stage) -> u64 {
    match (storage, stage) {
        (FunctionStorage::Assumed(_), Stage::Vertex | Stage::Fragment) => METAL,
        _ => storage.bytes(),
    }
}
