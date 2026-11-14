//! Where an indirect command buffer's records lie as the device's policy says
//! (`DevicePolicy::indirect_records`, `buffer.rs`'s `Device::records_take_own_pages`), and what a
//! recovery's carry does with them there:
//!
//! - under the default, `IndirectRecordsPolicy::Untaxed`, on a device whose check found that a
//!   held host-pointer import costs every submission (the import tax, forced to hold here), the
//!   records lie in pooled host-visible memory, no fresh pages made for them, and a recovery
//!   moves them to new memory and resets every command, its notice counting the buffer as one
//!   whose records were lost;
//! - where the tax is forced to fail, and under `IndirectRecordsPolicy::Carried` whatever it
//!   answers, they lie in fresh pages of the runtime's own, imported by their pointer, which a
//!   recovery keeps, its notice counting none lost.
//!
//! Each case prints an `ICB-RECORDS` line: the policy, the verdict, where the records lie, the
//! fresh pages made for them and what the carry did.
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs, on Linux (the forced
//! verdict is a Linux test hook), on the kind of device the cfg selects (docs/VALIDATION.md). No
//! GPU fault: the loss is marked (`Device::recover_for_test`).

use super::device_recovery::{heard_device_with, recovery_notices, Heard, Scratch};
use super::device_recovery_live_table::dispatches;
use crate::submit_check::{Check, Verdict};
use crate::Device;
use metal2vulkan::device::{DevicePolicy, IndirectRecordsPolicy};
use std::time::Duration;

/// How long a case waits for the device's own submission checks to answer before it forces the
/// import tax's verdict over theirs.
const SETTLE: Duration = Duration::from_secs(60);

/// What one case found: whether the records lay in the runtime's own pages, whether the recovery
/// kept them there, and the recovery's notice.
struct Found {
    own_pages: bool,
    kept: bool,
    notice: String,
}

/// Whether `device` imports host pointers at all: where it does not, the records lie in pooled
/// memory under every policy and verdict.
fn imports_pointers(device: &Device) -> bool {
    (device.shared().host_import.as_ref()).is_ok_and(|importer| importer.pointers().is_some())
}

/// An indirect command buffer made on a device of `policy` whose import tax answers `verdict`,
/// and that device's recovery: `None` where the device imports no host pointer.
fn case(policy: IndirectRecordsPolicy, verdict: Verdict) -> Option<Found> {
    let scratch = Scratch::new();
    let heard = Heard::default();
    let mut device_policy = DevicePolicy::UNTRUSTED;
    device_policy.indirect_records = policy;
    let device = heard_device_with(&scratch.0, &heard, device_policy);
    if !imports_pointers(&device) {
        eprintln!(
            "ICB-RECORDS {}: the device imports no host pointer; not run",
            device.name()
        );
        return None;
    }
    // The device's own verdicts first, so that none lands over the forced one.
    device.settle_submit_checks(SETTLE);
    (device.shared().submit_checks).force(Check::ImportTax, verdict);
    let before = device.work_counts();
    let icb = dispatches(&device);
    let made = device.work_counts();
    let fresh = made.fresh_page_buffers_made - before.fresh_page_buffers_made;
    let own_pages = icb.records_in_own_pages();
    let records = icb.records_address();
    let from = heard.lock().unwrap().len();
    device.recover_for_test().unwrap();
    let kept = icb.records_address() == records;
    let said = recovery_notices(&heard, from, "the device");
    assert_eq!(said.len(), 1, "{said:?}");
    let notice = said[0].clone();
    eprintln!(
        "ICB-RECORDS {} {policy:?}, import tax forced {verdict:?}: records in the runtime's own \
         pages {own_pages}, {fresh} fresh-page buffers made, the carry {}",
        device.name(),
        match kept {
            true => "kept them",
            false => "moved them",
        }
    );
    assert_eq!(
        fresh > 0,
        own_pages,
        "fresh pages made for the records where, and only where, they lie in them"
    );
    Some(Found {
        own_pages,
        kept,
        notice,
    })
}

/// Under the default, a device whose held pointer import costs every submission lays the records
/// in pooled memory, which its recovery loses and announces; where the import costs nothing, in
/// the runtime's own pages, which its recovery keeps.
#[test]
fn records_lie_in_pooled_memory_where_a_held_import_taxes_submissions() {
    let Some(untaxed) = case(IndirectRecordsPolicy::Untaxed, Verdict::Fails) else {
        return;
    };
    assert!(
        untaxed.own_pages,
        "where a held import costs nothing, the records lie in the runtime's own pages"
    );
    assert!(
        untaxed.kept,
        "the carry keeps records in the runtime's own pages"
    );
    assert!(
        untaxed
            .notice
            .contains(", and 1 indirect command buffers, 0 of them whose records were lost"),
        "{}",
        untaxed.notice
    );
    let taxed = case(IndirectRecordsPolicy::Untaxed, Verdict::Holds).expect("the same device");
    assert!(
        !taxed.own_pages,
        "where a held import costs every submission, the records lie in pooled memory"
    );
    assert!(!taxed.kept, "the carry moves records in pooled memory");
    assert!(
        taxed
            .notice
            .contains(", and 1 indirect command buffers, 1 of them whose records were lost"),
        "{}",
        taxed.notice
    );
}

/// Under `IndirectRecordsPolicy::Carried`, which `DevicePolicy::exact_answers()` selects, the
/// records lie in the runtime's own pages whatever a held import costs, and the recovery keeps
/// them, as Metal keeps a buffer's records across its GPU's restart.
#[test]
fn carried_records_lie_in_the_runtime_s_own_pages_whatever_the_tax() {
    assert_eq!(
        DevicePolicy::UNTRUSTED.exact_answers().indirect_records,
        IndirectRecordsPolicy::Carried
    );
    let Some(taxed) = case(IndirectRecordsPolicy::Carried, Verdict::Holds) else {
        return;
    };
    assert!(
        taxed.own_pages,
        "carried records lie in the runtime's own pages"
    );
    assert!(taxed.kept, "the carry keeps them");
    assert!(
        taxed
            .notice
            .contains(", and 1 indirect command buffers, 0 of them whose records were lost"),
        "{}",
        taxed.notice
    );
}
