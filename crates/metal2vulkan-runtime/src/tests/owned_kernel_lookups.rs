//! A lookup of one of the runtime's own kernels that the device made already waits for no other
//! kernel's making (`owned_kernel.rs`): an encoder's copy kernel, or the rate-map routes' kernel
//! an encoder takes only where it is made (`Device::route_counts_kernel`), is answered while the
//! device's pool or another thread makes an unrelated kernel, whose pipeline creation can take
//! seconds on some drivers. A second asker of a kernel being made waits for that making and
//! takes its kernel: the kernel is made once.
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects.

use super::blit_holds::{device, Scratch, WAIT};
use crate::freed_memory::held::Held;
use crate::owned_kernel::OwnedModule;
use std::sync::mpsc;
use std::time::{Duration, Instant};

/// How long a lookup of a made kernel may take while another kernel's making is held: a guard
/// against a wait for that making, which lasts as long as its hold; a lookup takes microseconds.
const LOOKUP: Duration = Duration::from_secs(5);

/// Lets a hold go when dropped, so that a failed assertion leaves no making held.
struct Release<'a>(&'a Held);

impl Drop for Release<'_> {
    fn drop(&mut self) {
        self.0.release();
    }
}

/// The rows copy's and the routes' kernels made, and the visibility combine's making held once
/// its module is built: a lookup of the rows copy's kernel on another thread returns while that
/// making is held, with the kernel made, and the routes' kernel is answered made; a second asker
/// of the visibility combine's kernel waits for the making held, making none itself; let go, the
/// held making makes its kernel, once, and both askers take it.
#[test]
fn a_made_kernel_s_lookup_waits_for_no_other_kernel_s_making() {
    let scratch = Scratch::new("owned-kernel-lookups");
    let device = device(&scratch.0);
    let rows = device.owned_kernel(OwnedModule::RowsCopy).unwrap();
    device.owned_kernel(OwnedModule::RouteCounts).unwrap();
    let held = device.hold_owned_making(OwnedModule::VisibilityCombine);
    let before = device.work_counts().owned_kernels_made;
    let device = &device;
    let (looked, routes, followers, holding, made, followed) = std::thread::scope(|scope| {
        let release = Release(&held);
        let making = scope.spawn(move || device.owned_kernel(OwnedModule::VisibilityCombine));
        held.until_waiting(1, WAIT);
        let (sent, got) = mpsc::channel();
        scope.spawn(move || {
            let asked = Instant::now();
            let kernel = device.owned_kernel(OwnedModule::RowsCopy);
            let _ = sent.send((kernel, asked.elapsed()));
        });
        let looked = got.recv_timeout(LOOKUP).ok();
        let routes = device.route_counts_kernel();
        let follower = scope.spawn(move || device.owned_kernel(OwnedModule::VisibilityCombine));
        let asked = Instant::now();
        while device.owned_followers(OwnedModule::VisibilityCombine) == 0 && asked.elapsed() < WAIT
        {
            std::thread::sleep(Duration::from_millis(1));
        }
        let followers = device.owned_followers(OwnedModule::VisibilityCombine);
        let holding = held.waiting();
        drop(release);
        let made = making.join().unwrap();
        (
            looked,
            routes,
            followers,
            holding,
            made,
            follower.join().unwrap(),
        )
    });
    let made_count = device.work_counts().owned_kernels_made - before;
    eprintln!(
        "OWNED-KERNEL-LOOKUPS: {}: the made kernel looked up in {:?} while another's making was \
         held; the routes' kernel answered {}; {followers} askers followed the held making, \
         {holding} held in it; {made_count} kernels made",
        device.name(),
        looked.as_ref().map(|(_, took)| *took),
        if routes.is_some() { "made" } else { "unmade" },
    );
    let Some((kernel, _)) = looked else {
        panic!("a made kernel's lookup waited past {LOOKUP:?} for another kernel's making");
    };
    assert_eq!(
        kernel.unwrap().pipeline,
        rows.pipeline,
        "the lookup answered the kernel made"
    );
    assert!(
        routes.is_some(),
        "the routes' kernel, made, was answered unmade while another kernel was being made"
    );
    assert_eq!(
        followers, 1,
        "the second asker waited for the making in flight"
    );
    assert_eq!(holding, 1, "the second asker made the kernel itself");
    assert_eq!(
        followed.unwrap().pipeline,
        made.unwrap().pipeline,
        "the second asker took the held making's kernel"
    );
    assert_eq!(made_count, 1, "the held making made its kernel once");
}
