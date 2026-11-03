//! Making a no-copy buffer looks at where its bytes lie once (`WorkCounts::no_copy_lies_read`):
//! each look takes its pages' lock (`buffer::HostCell`), and a buffer just made lies in its pages,
//! so the caller's log of their writes, their copy's name, the guest RAM runs they were resolved
//! to and the aliases over their bytes are all given those pages from one look.
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. A device that imports no host memory
//! says so and passes.

use super::blit_holds::{device, over_host, Host, Scratch};
use std::sync::atomic::AtomicUsize;
use std::sync::Arc;

/// A no-copy buffer over host memory, with no log of its writes and no guest RAM, is looked at
/// once as it is made.
#[test]
fn a_no_copy_buffer_is_looked_at_once_as_it_is_made() {
    let host = Host::new();
    let scratch = Scratch::new("no-copy-lies");
    let device = device(&scratch.0);
    let runs = Arc::new(AtomicUsize::new(0));
    let before = device.work_counts().no_copy_lies_read;
    let Some(buffer) = over_host(&device, &host, &runs) else {
        return;
    };
    let looked = device.work_counts().no_copy_lies_read - before;
    assert_eq!(looked, 1, "the making's looks at where the buffer lies");
    drop(buffer);
}
