//! What a killed run leaves in the suite's scratch root, and what the next run does about it.
//!
//! A drop guard covers a return, an `?` and an unwind; it cannot cover a kill, and a kill is
//! ordinary here (a cancelled run, the memory cap killing a worker's process group, a wedged Metal
//! shader compile that has to be killed). So the suite's scratch root carries a lock per run that
//! the kernel releases when its owner dies, and the next run removes the directory of every lock
//! it can take ([`metal2vulkan_devtools::scratch`]).
//!
//! This test is that sentence run for real, with three processes:
//!
//! - one run is killed with `SIGKILL` -- its own pid, from [`std::process::Child::kill`], never a
//!   pattern that would reach another stream's processes;
//! - a second run is started **before** the kill and is still live and still holding its lock;
//! - a third run then starts, which is "the next run", and sweeps.
//!
//! The assertion that matters is the third one: the sweep removes the dead run's directory and
//! leaves the live run's, with the file the live run is using still in it. A sweep that cannot
//! tell those two apart would destroy a concurrent run of the suite mid-case, which is why
//! recognition is a lock the kernel answers for and not a name.
//!
//! The three runs are given a temp directory of this test's own, so the scratch root they share is
//! this test's and the run that sweeps a directory of it can only be one of these three. Against
//! the machine's own root, a suite another stream started could be the sweeper instead -- which
//! would say nothing about the mechanism and would fail at random. That temp directory is itself a
//! scratch directory of the real root, so a kill of this test leaks nothing either.

use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};

/// The run the test starts and kills.
const HOLDER: &str = env!("CARGO_BIN_EXE_scratch-holder");

/// A holder process and the scratch directory it took, killed and reaped when this drops so a
/// failing assertion leaves no process of ours running.
struct Run {
    child: Child,
    dir: PathBuf,
}

impl Run {
    /// A started holder whose temp directory is `home`, with the scratch directory it printed. Its
    /// first scratch directory is what sweeps, so a holder that has printed its path has already
    /// swept.
    fn started(home: &Path) -> Run {
        let mut child = Command::new(HOLDER)
            .env("TMPDIR", home)
            .stdout(Stdio::piped())
            .spawn()
            .expect("the holder started");
        let mut line = String::new();
        BufReader::new(child.stdout.take().expect("the holder's output"))
            .read_line(&mut line)
            .expect("the holder's first line");
        let dir = PathBuf::from(line.trim());
        assert!(
            dir.is_dir(),
            "the holder printed {:?}, which is not a directory it took",
            line.trim()
        );
        Run { child, dir }
    }

    /// Kill it with its own pid and wait for the kernel to have taken its descriptors with it, so
    /// the lock it held is released before anything asks about it.
    fn kill(&mut self) {
        self.child.kill().expect("the holder killed");
        self.child.wait().expect("the holder reaped");
    }

    /// The file the run wrote in its scratch directory.
    fn file(&self) -> PathBuf {
        self.dir.join("0.metallib")
    }
}

impl Drop for Run {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// **A killed run's scratch directory is removed by the next run, and a live run's is not.**
///
/// Every step is a real process: the dead one was killed with `SIGKILL` and so ran no destructor,
/// the live one holds its lock from another process, and the sweep is the one a third run does on
/// its way to its own first scratch directory.
#[test]
fn the_next_run_sweeps_a_killed_runs_scratch_and_leaves_a_live_ones() {
    // Removed with everything under it when this test ends, however it ends.
    let home = metal2vulkan_devtools::scratch::Owned::made("killed-scratch-test")
        .expect("a temp directory of this test's own");

    // Live before the kill, so the sweep sees the two side by side.
    let live = Run::started(home.path());
    assert!(live.file().is_file(), "the live run wrote nothing");

    let mut dead = Run::started(home.path());
    let dead_file = dead.file();
    let dead_dir = dead.dir.clone();
    dead.kill();

    // What the kill left: a scratch directory, its contents, and no owner.
    assert!(
        dead_file.is_file(),
        "a killed run left no scratch behind, so there is nothing here to sweep"
    );

    // The next run. It has printed its own path, so it has already swept.
    let next = Run::started(home.path());

    assert!(
        !dead_dir.exists(),
        "{} outlived the next run, so a killed run still leaks its scratch",
        dead_dir.display()
    );
    assert!(
        live.file().is_file(),
        "the sweep removed {}, which a live run in another process is using",
        live.dir.display()
    );
    assert!(
        next.file().is_file(),
        "the sweep removed the sweeping run's own directory"
    );

    // Both remaining runs killed before the temp directory they wrote in goes with `home`.
    drop(next);
    drop(live);
}

/// **The program scratch a side keeps its metallibs in lies in the root the next run sweeps, with
/// a lock beside it.** It was `$TMPDIR/metal2vulkan-conformance-<pid>-<nanos>`, removed on drop
/// alone and recognisable to nothing, so a killed record or replay left it behind, still holding
/// its `0.air`, `0.air.ll` and `0.metallib`. Being in the root with a lock is the whole of what
/// makes [`the_next_run_sweeps_a_killed_runs_scratch_and_leaves_a_live_ones`] possible; move the
/// path back out and this test fails naming it.
#[test]
fn the_program_scratch_lies_in_the_swept_root() {
    let root = metal2vulkan_devtools::scratch::root();

    let program = metal2vulkan_conformance::archives::Scratch::new().expect("a program scratch");
    in_the_root("the program scratch", program.dir(), &root);
}

/// `what` at `path` is a run's directory in `root`, with the lock a sweep reads beside it.
fn in_the_root(what: &str, path: &Path, root: &Path) {
    assert_eq!(
        path.parent(),
        Some(root),
        "{what} {} is not in the scratch root {}, so no run can sweep it after this one is killed",
        path.display(),
        root.display()
    );
    let mut lock = path.as_os_str().to_owned();
    lock.push(".lock");
    assert!(
        Path::new(&lock).is_file(),
        "{what} {} has no lock beside it, so a sweep cannot tell its owner from a dead one",
        path.display()
    );
}
