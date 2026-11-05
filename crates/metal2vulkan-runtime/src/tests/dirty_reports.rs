//! A command buffer's writes into host pages imported in place, as its commit and its last
//! batch's completion report them to the pages' logs (`dirty_pages.rs`,
//! `PagesLog::wrote_but`), on plain memory: over pages whose memory's own log and guest RAM's log
//! both keep them (the RAM's offsets host addresses, through the runs of the RAM the pages are),
//! a write a shadow's merge made is the shadow's own on the RAM's log (its writer named), and is
//! left out of the memory's log, which hears of it from the shadow's log; the bytes beside it
//! are the consumer's (writer 0) on both. A command's write no shadow made is the consumer's on
//! both.
//!
//! The logs are the tests' own (`copied_log::TestLog`).

use super::copied_log::TestLog;
use crate::dirty_pages::{hook, DirtySource, PagesLog};
use std::num::NonZeroU64;
use std::ops::Range;
use std::sync::Arc;

/// The writes `log` was told of, with their writers (0 the consumer's), in order.
fn told(log: &TestLog) -> Vec<(Range<u64>, u64)> {
    (log.writes.lock().unwrap().iter())
        .map(|w| (w.1.clone(), w.2))
        .collect()
}

#[test]
fn a_shadow_s_merge_is_its_own_write_on_every_log_of_the_pages() {
    // The pages: 0x3000 bytes at a host address of the memory's (its pointer), the RAM's window
    // two runs, the bytes 0x100 into it.
    let pointer = 0x7000_0000_u64;
    let length = 0x3000_u64;
    let own = Arc::new(TestLog::default());
    let ram = Arc::new(TestLog::default());
    let own_source = DirtySource::new(hook(Arc::clone(&own)), pointer);
    let ram_source = DirtySource::new(hook(Arc::clone(&ram)), 0);
    let (first, second) = (0x10_0000..0x10_2000, 0x40_0000..0x40_2000);
    let window = [first.clone(), second.clone()];
    let log = PagesLog::new(
        (pointer, length),
        Some(&own_source),
        Some((&ram_source, &window[..], 0x100)),
    )
    .unwrap();
    // The RAM's host address of the object's byte `byte`.
    let host = |byte: u64| match byte + 0x100 {
        w if w < 0x2000 => first.start + w,
        w => second.start + (w - 0x2000),
    };
    let writer = NonZeroU64::new(77).unwrap();
    // The shadow's merge wrote the object's bytes 0x1000..0x2000; the command buffer's commands
    // wrote 0x800..0x2800 in place (the merge among them).
    let merged = pointer + 0x1000..pointer + 0x2000;
    let shadows = [(writer, &own_source, merged)];
    log.wrote_but(std::slice::from_ref(&(0x800..0x2800)), &shadows);
    assert_eq!(
        told(&own),
        [(0x800..0x1000, 0), (0x2000..0x2800, 0)],
        "the memory's log: the bytes beside the merge, the consumer's; the merge left to the shadow's log"
    );
    assert_eq!(
        told(&ram),
        [
            (host(0x800)..host(0x1000), 0),
            (host(0x2000)..host(0x2800), 0),
            (host(0x1000)..first.end, 77),
            (second.start..host(0x2000), 77),
        ],
        "the RAM's log: the merge as the shadow's own, through the runs"
    );
    // The control: with no shadow's write among them, every byte is the consumer's on both.
    let (own, ram) = (Arc::new(TestLog::default()), Arc::new(TestLog::default()));
    let own_source = DirtySource::new(hook(Arc::clone(&own)), pointer);
    let ram_source = DirtySource::new(hook(Arc::clone(&ram)), 0);
    let log = PagesLog::new(
        (pointer, length),
        Some(&own_source),
        Some((&ram_source, &window[..], 0x100)),
    )
    .unwrap();
    log.wrote_but(std::slice::from_ref(&(0x800..0x2800)), &[]);
    assert_eq!(told(&own), [(0x800..0x2800, 0)]);
    assert_eq!(
        told(&ram),
        [(host(0x800)..first.end, 0), (second.start..host(0x2800), 0)]
    );
}
