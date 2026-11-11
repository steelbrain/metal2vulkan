//! The calling thread's identity (`this_thread.rs`), on threads the test starts. No device is
//! opened.

use crate::this_thread::{id, runs};
use std::sync::mpsc;
use std::thread::JoinHandle;

/// A thread reads the same identity each time, another live thread another one, and a thread's
/// handle runs on that thread alone.
#[test]
fn a_thread_tells_itself_from_another() {
    let here = id();
    assert_eq!(id(), here);
    let (give, take) = mpsc::channel::<JoinHandle<()>>();
    let (tell, told) = mpsc::channel();
    let handle = std::thread::spawn(move || {
        let own = take.recv().unwrap();
        tell.send((id(), runs(&own))).unwrap();
    });
    assert!(!runs(&handle));
    give.send(handle).unwrap();
    let (there, ran_there) = told.recv().unwrap();
    assert_ne!(there, here);
    assert!(ran_there);
}
