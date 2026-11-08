//! A sampler state going on one thread while another lists the device's sampler states
//! (`ResourceIds::samplers`, `Device::argument_samplers_at`): from its last handle's drop until its
//! ID's slot is given back, the state lives no more but still holds its slot, so no slot change is
//! counted yet. A listing taken then must not be at the count a listing taken while it lived was
//! taken at, or a bind state that kept that listing by its count (`compute/live.rs`, a render
//! encoder's `LiveBind::listed_at`) would keep it past the state's going, and a compute dispatch's
//! debug check, which lists the states again, would find the two apart.
//!
//! The test hook `sampler::going` holds the going thread at that point, so the order is the same
//! at every run.
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. The kernel is `scale_and_offset` of the
//! owned fixture `validation/fixtures/public/examples/runtime_examples.ll`, as
//! `dispatch_pending.rs` runs it.

use super::blit_holds::{device, Scratch, WAIT};
use super::dispatch_pending::{dispatch, state, Buffers, Grid};
use crate::sampler::going;
use crate::Size;
use metal2vulkan::bind::SamplerDescriptor;
use std::sync::mpsc;

/// A compute encoder's kept bind state listed a sampler state; the state then starts going on
/// another thread, held there before it gives its ID's slot back. The next encoder's dispatch,
/// which starts from that kept state, binds a listing without the state (its debug check lists
/// the states again and compares), and a listing taken then leaves the state out at a count
/// apart from the one the kept listing was taken at. Once the state is gone the count moved
/// again and no listing names it.
#[test]
fn a_sampler_state_going_meanwhile_leaves_no_listing_at_a_kept_count() {
    let scratch = Scratch::new("sampler-release");
    let device = device(&scratch.0);
    let queue = device.new_command_queue().unwrap();
    let pipeline = state(&device);
    let buffers = Buffers::new(&device);
    let sampler = device.new_sampler_state(&SamplerDescriptor::new()).unwrap();
    let id = sampler.gpu_resource_id();
    let lists = |states: &[crate::SamplerState]| states.iter().any(|s| s.gpu_resource_id() == id);
    let grid = || Grid::One(Size::new(64, 1, 1));
    // The encoder's dispatch lists the state, and its kept bind state goes to the device for the
    // next encoder to start from (`SpareLiveDispatches`).
    let first = dispatch(&queue, &pipeline, &buffers, grid());
    first.commit().unwrap();
    first.wait_until_completed().unwrap();
    let (before, listed) = device.argument_samplers_at();
    assert!(lists(&listed), "the live state listed");
    drop(listed);
    let weak = sampler.downgrade();
    assert_eq!(
        weak.strong_count(),
        1,
        "the test holds the state's one handle"
    );
    // The state goes on another thread, held where its last handle is gone and its ID's slot
    // still held.
    let (started, at_start) = mpsc::channel();
    let (go, to_go) = mpsc::channel::<()>();
    let dropper = std::thread::spawn(move || {
        going::at_next(move || {
            started.send(()).unwrap();
            // The test ending without a word lets it go too.
            let _ = to_go.recv();
        });
        drop(sampler);
    });
    at_start.recv_timeout(WAIT).unwrap();
    assert_eq!(weak.strong_count(), 0, "the state is going");
    assert_eq!(
        device.shared().sampler_slots_changed(),
        before,
        "no slot change is counted while the state still holds its slot"
    );
    // The next encoder starts from the kept bind state, which lists the state: its dispatch's
    // debug check finds the kept listing apart from the one it takes where the counts agree.
    let second = dispatch(&queue, &pipeline, &buffers, grid());
    let (meanwhile, listed) = device.argument_samplers_at();
    assert!(!lists(&listed), "a going state listed");
    assert_ne!(
        meanwhile, before,
        "a listing without the going state at the count a listing with it was taken at"
    );
    drop(listed);
    go.send(()).unwrap();
    dropper.join().unwrap();
    second.commit().unwrap();
    second.wait_until_completed().unwrap();
    assert!(buffers.written(64), "the second dispatch ran");
    let (after, listed) = device.argument_samplers_at();
    assert!(!lists(&listed), "a gone state listed");
    assert_ne!(after, meanwhile, "the slot given back is a change");
}
