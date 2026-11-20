//! Where a lane's shader cache lives, and why that is the whole of the fix.
//!
//! A [`ScratchCache`] is removed when it drops, which covers a return and an unwind and covers
//! neither a kill nor an exit that runs no destructor -- and lane processes are killed: a run gets
//! cancelled, and the memory cap kills a worker and its whole process group. The caches used to be
//! `$TMPDIR/m2v-lane-shader-cache-<pid>-<n>-<nanos>`, named for a process and recognisable to
//! nothing, so each one a killed lane left stayed there for good: 900 to 2200 of them on a host.
//!
//! The fix is not a second removal somewhere; it is the path. A cache in the project's scratch
//! root carries a lock the kernel releases when its owner dies, so the next run of any suite
//! removes it, and removes no live run's
//! ([`metal2vulkan_devtools::scratch`], whose own tests hold the sweep itself).

use metal2vulkan_validation::candidate::runtime_case::ScratchCache;

/// **A lane's shader cache is a path the next run can recognise**: in the scratch root, with a
/// lock beside it. Put it back under the OS temp directory on its own and this test fails naming
/// the directory it finds there, because nothing could ever sweep it.
#[test]
fn a_lane_shader_cache_lies_in_the_swept_scratch_root() {
    let root = metal2vulkan_devtools::scratch::root();
    let cache = ScratchCache::new().expect("a shader cache path");

    assert_eq!(
        cache.path().parent(),
        Some(root.as_path()),
        "the lane's shader cache {} is not in the scratch root {}, so no run can sweep it after \
         this one is killed",
        cache.path().display(),
        root.display()
    );

    let mut lock = cache.path().as_os_str().to_owned();
    lock.push(".lock");
    assert!(
        std::path::Path::new(&lock).is_file(),
        "{} has no lock beside it, so a sweep cannot tell its owner from a dead one",
        cache.path().display()
    );

    // The device makes it, so the path is a path and nothing else until then.
    assert!(
        !cache.path().exists(),
        "{} was created, so it is not the empty cache a lane measures against",
        cache.path().display()
    );
}
