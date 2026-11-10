//! The one place under the OS temp directory this project writes, and the lock that makes a
//! directory's owner knowable after the owner is gone.
//!
//! # Why a lock and not a name
//!
//! Every scratch directory of a run is removed when its owner drops ([`Owned`]), which covers a
//! return, an `?` and an unwind. It does not cover a kill, and a kill is ordinary here: a run gets
//! cancelled, the memory cap kills a worker and its whole process group, and MoltenVK can wedge in
//! Metal shader compile and has to be killed. A killed process runs no destructor, so the only
//! thing that can remove its directory is the *next* run.
//!
//! A next run cannot tell a dead run's directory from a live one by its name. Both are
//! `<kind>-<pid>-<nanos>-<serial>`, and several runs of this project share this machine with each
//! other and with the owner's own work, so a sweep that removed what it did not recognise would
//! destroy a live run's metallibs mid-case. Matching a name against a process is no better: a pid
//! is reused, `pgrep -x` compares only the first 16 characters of a process name on macOS, and a
//! `pgrep -f` pattern matches the shell that runs the `pgrep`.
//!
//! So ownership is held by the kernel instead of guessed. Beside each scratch directory sits
//! `<token>.lock`, and its owner holds that file's exclusive `flock` for its whole life. The
//! kernel releases an `flock` when the last descriptor on it closes, which happens on `SIGKILL`
//! exactly as it happens on a clean exit. A sweep therefore asks one question with one answer:
//! **can this lock be taken?** Taken means no owner is left and the directory is this sweep's to
//! remove; refused means a live run owns it and the sweep passes over it. The lock is `flock`,
//! which is held per open descriptor rather than per process, so a second run inside the same test
//! binary is protected by it too.
//!
//! The order of the three steps makes the answer safe to act on. A run opens its lock file, takes
//! the lock, **checks that the path still names the file it is holding**, and only then makes its
//! directory. The check is not a nicety: a lock file is visible for an instant between being
//! created and being locked, and a sweep that reaches it in that instant takes the lock and
//! removes the file -- and `flock` on a file with no name left answers yes, so a run without the
//! check would hold a lock nothing can see and keep a directory no sweep would ever recognise.
//! With it, a run that lost that race simply takes another name, and from the moment the check
//! passes the run holds its lock without a gap, so no sweep can take it and no sweep can remove
//! either the lock or the directory. A directory that exists at all therefore has a lock beside
//! it that its owner held when it was made.
//!
//! A sweep removes the directory first and the lock file second, so an interrupted sweep leaves
//! the lock to be taken again rather than an unowned directory nothing recognises. A lock with no
//! directory -- a reserved path nothing wrote, or a run killed before it made its directory -- is
//! swept the same way, so nothing of a dead run is left under the root at all. The root itself
//! stays, empty, because removing it raced every run that was making it ([`root`]).
//!
//! Nothing else is ever removed. An entry of the root that is not `<token>.lock`, or whose lock is
//! held, or whose lock cannot be opened, is left exactly as it is.

use std::fs::{File, TryLockError};
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

#[cfg(windows)]
mod windows;
#[cfg(windows)]
use windows::same_file;

/// The directory under the OS temp directory that holds every scratch directory of this project,
/// so a sweep reads one directory it owns rather than the whole temp directory.
const ROOT: &str = "metal2vulkan-scratch";

/// What marks a scratch directory's lock file.
const LOCK: &str = ".lock";

/// The root every scratch directory of this project sits in.
///
/// It is made when a run needs it and then left in place, empty, and **nothing removes it**. An
/// earlier form had the last run out remove it, which raced every other run: a run that had just
/// made the root found it gone again while opening its lock in it, and APFS answers that race with
/// `EINVAL` rather than with a missing file, so it was not even a condition to retry on by name.
/// An empty directory under the OS temp directory costs one inode, and it is what makes a sweep
/// recognisable in the first place rather than the long-lived dump the rule is about.
pub fn root() -> PathBuf {
    std::env::temp_dir().join(ROOT)
}

/// A scratch directory one run owns: removed when this value drops, and removable by the next run
/// of this project if this process is killed before that ([module docs](self)).
pub struct Owned {
    /// The directory itself. [`Owned::reserved`] does not create it.
    dir: PathBuf,
    /// The lock beside it, removed after the directory.
    lock_path: PathBuf,
    /// The held lock. Its descriptor is what the kernel closes when this process dies, which is
    /// what tells the next run that this directory has no owner left.
    held: File,
}

impl Owned {
    /// A directory of this run's own, made empty.
    pub fn made(kind: &str) -> io::Result<Self> {
        sweep_once();
        Self::claimed(&root(), kind, Make::Now)
    }

    /// A directory of this run's own, **not** made: the path is this run's, and whatever is asked
    /// to write there creates it. A device given an empty shader-cache path is the case -- the
    /// path has to be one nothing has written yet, and the device makes the directory itself.
    pub fn reserved(kind: &str) -> io::Result<Self> {
        sweep_once();
        Self::claimed(&root(), kind, Make::ByWhoeverWrites)
    }

    /// The directory's path.
    pub fn path(&self) -> &Path {
        &self.dir
    }

    /// The lock taken and the name claimed in `root`, which is the project's root
    /// ([`root`]) everywhere but in this module's own tests.
    ///
    /// Attempts, because one thing a concurrent sweep does is transient and is not a failure: it
    /// can reach this run's lock file in the instant after it was created and before it was
    /// locked, either taking the lock first or removing the file this run is holding. Either is
    /// answered by trying again under another name. Four attempts: one collision is a race, four
    /// is a defect to see rather than to spin on.
    fn claimed(root: &Path, kind: &str, make: Make) -> io::Result<Self> {
        for _ in 0..4 {
            if let Some(owned) = Self::claim_once(root, kind, make)? {
                return Ok(owned);
            }
        }
        Err(io::Error::other(format!(
            "no scratch name in {} could be claimed",
            root.display()
        )))
    }

    /// One attempt at [`Self::claimed`]: `Ok(None)` where a concurrent sweep took this attempt's
    /// name out from under it, which is for the caller to try again rather than to report.
    fn claim_once(root: &Path, kind: &str, make: Make) -> io::Result<Option<Self>> {
        std::fs::create_dir_all(root)?;
        let token = token(kind);
        let lock_path = root.join(format!("{token}{LOCK}"));
        let held = File::options()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&lock_path)?;
        match held.try_lock() {
            Ok(()) => {}
            // A sweep reached this file in the instant between its creation and this lock, and
            // holds it while it removes it. Another name, then.
            Err(TryLockError::WouldBlock) => return Ok(None),
            Err(TryLockError::Error(error)) => return Err(error),
        }
        // **The lock is only this run's if the path still names the file this run is holding.** A
        // sweep that reached the file before this lock did removes it, and `flock` on a file with
        // no name left answers yes -- so without this the run would hold a lock nothing can see
        // and keep a directory no sweep would ever recognise. The same check is what makes the
        // order below sound: from here on this run holds the lock continuously, so no sweep can
        // take it, so no sweep can remove either the lock or the directory.
        if !same_file(&held, &lock_path)? {
            return Ok(None);
        }
        // The lock before the directory, and that order is what a sweep acts on: a directory that
        // exists at all has a lock beside it that its owner held when it was made.
        let dir = root.join(token);
        if make == Make::Now {
            std::fs::create_dir_all(&dir)?;
        }
        Ok(Some(Self {
            dir,
            lock_path,
            held,
        }))
    }
}

/// Whether the open file `held` is still the file `path` names, by the one identity a file system
/// gives a file: its device and inode. A rename or an unlink of `path` makes this false, and
/// `held` keeps working on a file with no name.
#[cfg(unix)]
fn same_file(held: &File, path: &Path) -> io::Result<bool> {
    use std::os::unix::fs::MetadataExt;
    let held = held.metadata()?;
    match std::fs::metadata(path) {
        Ok(named) => Ok((held.dev(), held.ino()) == (named.dev(), named.ino())),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(error),
    }
}

/// Whether [`Owned`] creates its directory or leaves it to whatever writes there.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Make {
    /// Created empty, now.
    Now,
    /// Left for whatever is given the path.
    ByWhoeverWrites,
}

impl Drop for Owned {
    fn drop(&mut self) {
        // The directory before its lock, so an interrupted drop leaves a lock the next sweep can
        // take rather than a directory with nothing beside it.
        let _ = std::fs::remove_dir_all(&self.dir);
        let _ = std::fs::remove_file(&self.lock_path);
        // The root itself stays, empty ([`root`]).
        // The lock released last of all, after the files it stood for are gone. Closing its
        // descriptor would release it anyway -- that is the whole point of holding it -- but
        // saying so here keeps the order the sweep relies on in one place.
        let _ = self.held.unlock();
    }
}

/// A name no other run and no other call can take: the process, the time and a serial, so two
/// calls in one process at one clock reading still differ.
fn token(kind: &str) -> String {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |since| since.as_nanos());
    let kind: String = kind
        .chars()
        .map(|char| {
            if char.is_ascii_alphanumeric() {
                char
            } else {
                '-'
            }
        })
        .collect();
    format!(
        "{kind}-{}-{nanos}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    )
}

/// Sweep the project's root once per process, from the first scratch directory it takes, so every
/// entry point sweeps without any of them having to remember to.
fn sweep_once() {
    static SWEPT: std::sync::Once = std::sync::Once::new();
    SWEPT.call_once(|| {
        let _ = sweep(&root());
    });
}

/// Remove the scratch directory of every run under the project's root that has no owner left,
/// answering how many it removed.
///
/// It removes a directory only where this process can take that directory's lock, which no live
/// run's lock allows ([module docs](self)). It is public so a test can run it against a run it is
/// holding and see that run survive.
pub fn sweep_dead_runs() -> io::Result<usize> {
    sweep(&root())
}

/// [`sweep_dead_runs`] over `root`, which is the project's root everywhere but in this module's
/// own tests.
fn sweep(root: &Path) -> io::Result<usize> {
    let entries = match std::fs::read_dir(root) {
        Ok(entries) => entries,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(0),
        Err(error) => return Err(error),
    };
    let mut removed = 0;
    for entry in entries {
        let name = entry?.file_name();
        // Only a lock file of ours is even looked at: a directory with no lock beside it, and any
        // other file, is something this sweep does not recognise and so does not touch.
        let Some(name) = name.to_str() else { continue };
        let Some(token) = name.strip_suffix(LOCK) else {
            continue;
        };
        if token.is_empty() {
            continue;
        }
        let lock_path = root.join(name);
        let Ok(lock) = File::options().read(true).write(true).open(&lock_path) else {
            continue;
        };
        if lock.try_lock().is_err() {
            continue;
        }
        // Nothing holds this lock, and an owner holds its own for as long as it lives, so this
        // run is gone and its directory is this sweep's to remove.
        let dir = root.join(token);
        match std::fs::remove_dir_all(&dir) {
            Ok(()) => {}
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            // The directory stays, so the lock stays with it and the next sweep tries again.
            Err(_) => continue,
        }
        let _ = std::fs::remove_file(&lock_path);
        removed += 1;
    }
    Ok(removed)
}

#[cfg(test)]
mod dead_runs {
    use super::{sweep, Make, Owned, LOCK};

    /// A root of this test's own, inside a scratch directory of the project's real root.
    ///
    /// Two reasons, and neither is tidiness. A run of any suite anywhere on this machine sweeps
    /// the real root as it starts, so against the real root the sweeper that removes a directory
    /// this test is about to look at need not be this test's -- a flake, and one that says nothing
    /// about the mechanism. And a root inside an [`Owned`] is itself swept if this test process is
    /// killed, so the tests of the sweep do not leak what the sweep exists to stop.
    ///
    /// The mechanism under test is unchanged: `flock` is held per open descriptor rather than per
    /// process, so a second run inside this one process is held off exactly as a run in another
    /// process is. The cross-process case is run for real in the conformance suite's
    /// `tests/killed_scratch.rs`.
    struct Root {
        own: Owned,
    }

    impl Root {
        fn new() -> Root {
            Root {
                own: Owned::made("scratch-tests").expect("a directory of this test's own"),
            }
        }

        /// The root itself.
        fn path(&self) -> std::path::PathBuf {
            self.own.path().join("root")
        }

        /// A run of this root's.
        fn run(&self, kind: &str) -> Owned {
            Owned::claimed(&self.path(), kind, Make::Now).expect("a run's directory")
        }

        /// A dead run's directory, left exactly as a killed process leaves one: the directory and
        /// its lock file there, and no process holding the lock.
        fn dead_run(&self, token: &str) -> std::path::PathBuf {
            let dir = self.path().join(token);
            std::fs::create_dir_all(&dir).expect("the dead run's directory");
            std::fs::write(dir.join("0.metallib"), b"bitcode").expect("the file it was holding");
            std::fs::write(self.path().join(format!("{token}{LOCK}")), b"")
                .expect("the lock it left");
            dir
        }
    }

    /// **A killed run's directory is removed by the next run.** This is the case a drop guard
    /// cannot reach: the directory, its contents and its lock file are all there and nothing holds
    /// the lock, which is what `SIGKILL` leaves behind.
    #[test]
    fn a_dead_runs_directory_is_swept() {
        let root = Root::new();
        let dead = root.dead_run("dead-run");
        assert!(dead.is_dir(), "the dead run's directory was not made");

        let removed = sweep(&root.path()).expect("the sweep");

        assert!(
            !dead.exists(),
            "{} outlived a sweep, so a killed run still leaks its scratch",
            dead.display()
        );
        assert!(
            !root.path().join(format!("dead-run{LOCK}")).exists(),
            "the dead run's lock file outlived its directory"
        );
        assert_eq!(
            removed, 1,
            "the sweep counted something other than the one run"
        );
    }

    /// **A live run's directory survives the same sweep.** The one property that makes the sweep
    /// safe to run at all: several runs of these suites share this machine, and a sweep that
    /// removed what it did not recognise would take a live run's metallibs out from under it
    /// mid-case. Make the sweep remove every entry of the root instead of asking its lock and this
    /// test fails naming the directory it took.
    #[test]
    fn a_live_runs_directory_survives_the_sweep() {
        let root = Root::new();
        let live = root.run("live");
        std::fs::write(live.path().join("0.metallib"), b"bitcode").expect("the file it is using");
        let dead = root.dead_run("dead-run");

        let removed = sweep(&root.path()).expect("the sweep");

        assert!(
            live.path().join("0.metallib").is_file(),
            "the sweep removed {}, which a live run is using",
            live.path().display()
        );
        assert!(!dead.exists(), "the sweep did not remove the dead run");
        assert_eq!(
            removed, 1,
            "the sweep counted the live run as well as the dead one"
        );
    }

    /// **A directory the sweep does not recognise is never removed**, however much it looks like
    /// one of ours. Recognition is the lock file and nothing else, so a directory with no lock
    /// beside it -- another tool's, or one a half-finished sweep left -- stays.
    #[test]
    fn a_directory_with_no_lock_of_ours_is_left_alone() {
        let root = Root::new();
        let stranger = root.path().join("not-ours");
        std::fs::create_dir_all(&stranger).expect("the stranger's directory");
        std::fs::write(stranger.join("theirs"), b"theirs").expect("the stranger's file");

        let removed = sweep(&root.path()).expect("the sweep");

        assert!(
            stranger.join("theirs").is_file(),
            "the sweep removed {}, which has no lock of ours beside it",
            stranger.display()
        );
        assert_eq!(
            removed, 0,
            "the sweep counted a directory it does not recognise"
        );
    }

    /// A run's directory and its lock are gone when the run drops, and so is the root once the
    /// last run in it has gone.
    #[test]
    fn a_dropped_run_leaves_neither_its_directory_nor_its_lock() {
        let root = Root::new();
        let run = root.run("dropped");
        let dir = run.path().to_owned();
        let lock = root.path().join(format!(
            "{}{LOCK}",
            dir.file_name().expect("the name").to_string_lossy()
        ));
        assert!(dir.is_dir() && lock.is_file(), "the run's files were made");
        std::fs::write(dir.join("0.air"), b"bitcode").expect("a file in it");

        drop(run);

        assert!(!dir.exists(), "{} outlived its owner", dir.display());
        assert!(!lock.exists(), "{} outlived its owner", lock.display());
        assert_eq!(
            std::fs::read_dir(root.path()).expect("the root").count(),
            0,
            "the root holds something after the last run in it went"
        );
    }

    /// A reserved directory is a path and nothing more: the device that is given it as an empty
    /// shader cache is what creates it, and a path something has already written is not empty.
    #[test]
    fn a_reserved_directory_is_not_made() {
        let root = Root::new();
        let run = Owned::claimed(&root.path(), "reserved", Make::ByWhoeverWrites)
            .expect("a reserved path");
        assert!(
            !run.path().exists(),
            "{} was created, so it is not an empty cache path",
            run.path().display()
        );
    }
}
