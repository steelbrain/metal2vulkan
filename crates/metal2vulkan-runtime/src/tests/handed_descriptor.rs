//! The descriptor a dma-buf import hands its driver (`handed_descriptor.rs`), with the test as
//! the driver: a number the driver closes twice within one failed import, another thread's file
//! opened between the closes, closes nothing of that thread's (NVIDIA's 620.70.06 did so, and the
//! process lost a descriptor of another import); a failed import's clone the driver left open is
//! closed; imports in flight hold numbers of their own, above a file the process opens; and where
//! no number is kept (a descriptor limit too low, every number of the range held) the clone is an
//! ordinary one, which a failed import lets go of.

use crate::handed_descriptor::{range_below, Handed};
use std::fs::File;
use std::ops::Range;
use std::os::fd::{AsFd, AsRawFd, IntoRawFd, RawFd};

/// Whether descriptor `fd` is open.
fn open(fd: RawFd) -> bool {
    // SAFETY: `F_GETFD` reads the descriptor's flags and changes nothing.
    unsafe { libc::fcntl(fd, libc::F_GETFD) >= 0 }
}

/// Close `fd` as a driver does, ignoring whether it is open.
fn close(fd: RawFd) {
    // SAFETY: the test's own descriptor, or no descriptor at all.
    unsafe { libc::close(fd) };
}

/// The device, inode and device number of the file `fd` refers to, where it is open.
fn identity(fd: RawFd) -> Option<(libc::dev_t, libc::ino_t, libc::dev_t)> {
    let mut stat = std::mem::MaybeUninit::<libc::stat>::uninit();
    // SAFETY: `fstat` writes the structure where it succeeds.
    if unsafe { libc::fstat(fd, stat.as_mut_ptr()) } != 0 {
        return None;
    }
    // SAFETY: the call succeeded.
    let stat = unsafe { stat.assume_init() };
    Some((stat.st_dev, stat.st_ino, stat.st_rdev))
}

#[test]
fn a_number_its_driver_closes_twice_closes_nothing_else() {
    let source = File::open("/dev/zero").unwrap();
    let handed = Handed::of(source.as_fd()).unwrap();
    let number = handed.number();
    // The driver's import fails and closes the number.
    close(number);
    // Another thread opens a file meanwhile.
    let other = std::thread::spawn(|| File::open("/dev/null").unwrap().into_raw_fd())
        .join()
        .unwrap();
    let file = identity(other);
    assert!(file.is_some());
    // The driver closes the number again, still within the one call.
    close(number);
    handed.failed();
    let after = identity(other);
    if after == file {
        close(other);
    }
    assert_eq!(
        after, file,
        "the import's number {number}, closed twice, closed another thread's file {other}"
    );
}

#[test]
fn a_failed_import_closes_the_clone_its_driver_left_open() {
    let source = File::open("/dev/zero").unwrap();
    let handed = Handed::of(source.as_fd()).unwrap();
    let number = handed.number();
    assert!(open(number));
    handed.failed();
    assert!(
        !open(number),
        "the clone {number} a failed import's driver left open is still open"
    );
}

#[test]
fn imports_in_flight_hold_numbers_of_their_own() {
    let source = File::open("/dev/zero").unwrap();
    let first = Handed::of(source.as_fd()).unwrap();
    let second = Handed::of(source.as_fd()).unwrap();
    let file = File::open("/dev/null").unwrap().into_raw_fd();
    let numbers = [first.number(), second.number()];
    assert_ne!(numbers[0], numbers[1]);
    assert!(
        numbers.iter().all(|&number| number > file),
        "imports' numbers {numbers:?} not above a file the process opened, {file}"
    );
    close(file);
    // A successful import leaves its descriptor to the driver, which closes it.
    first.imported();
    assert!(open(numbers[0]));
    close(numbers[0]);
    second.failed();
    assert!(!open(numbers[1]));
}

#[test]
fn the_range_is_the_numbers_below_the_descriptor_limit_or_none_where_it_is_low() {
    assert_eq!(range_below(1024), Some(960..1024));
    assert_eq!(range_below(256), Some(192..256));
    assert_eq!(range_below(1 << 20), Some(65472..65536));
    assert_eq!(range_below(libc::RLIM_INFINITY), Some(65472..65536));
    assert_eq!(range_below(255), None);
    assert_eq!(range_below(0), None);
}

#[test]
fn with_no_number_kept_the_clone_is_an_ordinary_one_a_failed_import_lets_go_of() {
    let source = File::open("/dev/zero").unwrap();
    let handed = Handed::within(source.as_fd(), None).unwrap();
    let number = handed.number();
    assert!(!handed.kept());
    assert_eq!(identity(number), identity(source.as_raw_fd()));
    handed.failed();
    // Let go of, not closed: the number may be another's by the time the import returns.
    assert!(
        open(number),
        "a failed import closed an ordinary clone {number}"
    );
    close(number);
}

/// The two numbers at the top of the range the process's descriptor limit gives.
fn top_two() -> Range<RawFd> {
    let mut limit = libc::rlimit {
        rlim_cur: 0,
        rlim_max: 0,
    };
    // SAFETY: `getrlimit` writes the structure, which outlives the call.
    assert_eq!(
        unsafe { libc::getrlimit(libc::RLIMIT_NOFILE, &mut limit) },
        0
    );
    let range = range_below(limit.rlim_cur).expect("a descriptor limit that keeps numbers");
    range.end - 2..range.end
}

#[test]
fn a_number_the_process_holds_is_passed_over_and_past_the_range_the_clone_is_an_ordinary_one() {
    let source = File::open("/dev/zero").unwrap();
    let two = top_two();
    // The process holds the range's first number itself, taken the way the module takes one.
    let null = File::open("/dev/null").unwrap();
    // SAFETY: a clone of the test's own descriptor at the lowest free number from there up.
    let held = unsafe { libc::fcntl(null.as_raw_fd(), libc::F_DUPFD_CLOEXEC, two.start) };
    assert_eq!(held, two.start, "the range's first number is free");
    let held_file = identity(held);
    let first = Handed::within(source.as_fd(), Some(two.clone())).unwrap();
    assert!(first.kept());
    assert_eq!(first.number(), two.start + 1);
    assert_eq!(
        identity(held),
        held_file,
        "the process's own descriptor was taken over"
    );
    // Every number held: the next clone is an ordinary one.
    let second = Handed::within(source.as_fd(), Some(two.clone())).unwrap();
    assert!(!second.kept());
    assert!(!two.contains(&second.number()));
    let unkept = second.number();
    second.failed();
    close(unkept);
    first.failed();
    close(held);
    // The numbers free again, the range's first is the next import's.
    let third = Handed::within(source.as_fd(), Some(two.clone())).unwrap();
    assert_eq!((third.kept(), third.number()), (true, two.start));
    third.failed();
}
