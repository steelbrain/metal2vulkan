//! Readback rule: every read the runtime makes of memory the GPU writes is in `readback.rs`, and
//! each function there states the bound on what its value can decide. The check reads every
//! product source file with line comments removed.

use crate::tests::product_sources;

/// The file that holds every read of GPU-written memory.
const READBACK_FILE: &str = "readback.rs";

/// What reading GPU-written memory looks like in source: query results, and reads through a raw
/// pointer, which a mapping of device memory is.
const DEVICE_READS: &[&str] = &[
    "get_query_pool_results",
    "read_volatile",
    "read_unaligned",
    "ptr::read",
    "from_raw_parts",
];

/// The device reads in one source text, as (1-based line, form).
fn scan(text: &str) -> Vec<(usize, &'static str)> {
    let mut found = Vec::new();
    for (index, raw) in text.lines().enumerate() {
        let code = raw.split("//").next().unwrap_or("");
        for read in DEVICE_READS {
            if code.contains(read) {
                found.push((index + 1, *read));
            }
        }
    }
    found
}

/// The functions of a source text whose doc comment states no bound.
fn unbounded(text: &str) -> Vec<String> {
    let mut missing = Vec::new();
    let mut doc = String::new();
    for line in text.lines() {
        let line = line.trim();
        if let Some(comment) = line.strip_prefix("///") {
            doc.push_str(comment);
            continue;
        }
        if let Some(at) = line.find("fn ") {
            let name: String = line[at + 3..]
                .chars()
                .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
                .collect();
            if !doc.contains("Bound:") {
                missing.push(name);
            }
        }
        if !line.starts_with("#[") {
            doc.clear();
        }
    }
    missing
}

#[test]
fn every_device_read_is_in_the_readback_module_with_its_bound() {
    let sources = product_sources();
    let outside: Vec<(String, usize, &str)> = sources
        .iter()
        .filter(|(label, _)| label != READBACK_FILE)
        .flat_map(|(label, text)| {
            scan(text)
                .into_iter()
                .map(move |(line, read)| (label.clone(), line, read))
        })
        .collect();
    assert!(
        outside.is_empty(),
        "a read of GPU-written memory outside {READBACK_FILE}: {outside:#?}"
    );
    let (_, readback) = sources
        .iter()
        .find(|(label, _)| label == READBACK_FILE)
        .expect("the readback module");
    assert!(!scan(readback).is_empty(), "{READBACK_FILE} reads nothing");
    assert_eq!(
        unbounded(readback),
        Vec::<String>::new(),
        "a readback without its bound"
    );
}

#[test]
fn the_readback_scan_finds_each_form_and_passes_prose() {
    for (line, read) in [
        (
            "device.get_query_pool_results(pool, 0, &mut v, f)",
            "get_query_pool_results",
        ),
        ("let x = pointer.read_volatile();", "read_volatile"),
        ("let x = pointer.read_unaligned();", "read_unaligned"),
        ("let x = std::ptr::read(pointer);", "ptr::read"),
        (
            "let s = slice::from_raw_parts(pointer, len);",
            "from_raw_parts",
        ),
    ] {
        assert_eq!(scan(line), [(1, read)], "{line}");
    }
    assert!(scan("// read_volatile in prose").is_empty());
    assert_eq!(
        unbounded("/// Reads.\npub(crate) fn a() {}\n/// Bound: none.\n#[inline]\nfn b() {}\n"),
        ["a"]
    );
}

/// A timestamp pair no submission wrote, as one the device cancelled at a reset leaves it (its
/// progress value signalled, its commands never run): the read reports no times, and returns at
/// once rather than waiting for values that never come. RADV's waiting read of such a query spins
/// for twice the kernel's 60 s compute timeout before it reports the device lost; the bound here
/// turns that wait into a failure. On lavapipe a query no command wrote reads as not ready with or
/// without the wait flag, so there the test passes either way.
#[cfg(any(m2v_gpu, m2v_cpu_icd))]
#[test]
fn a_pair_no_submission_wrote_reads_as_no_times_at_once() {
    use crate::tests::command::on_device::{each_device, PATIENCE};
    use std::sync::{mpsc, Arc};
    each_device("readback_unwritten_pair", |name, _, device| {
        let core = Arc::clone(device.core());
        let pair = (core.stamps.take(&core.device, &core.counters)).expect("a pair");
        let (send, receive) = mpsc::channel();
        let reader = Arc::clone(&core);
        let thread = std::thread::spawn(move || {
            // SAFETY: the pair is the device's, both its queries reset, and no other thread resets
            // or reads them.
            let read =
                unsafe { crate::readback::timestamps(&reader.device, pair.pool, pair.first) };
            let _ = send.send(read);
        });
        let read = (receive.recv_timeout(PATIENCE)).unwrap_or_else(|_| {
            panic!("{name}: the read of a pair no submission wrote did not return in {PATIENCE:?}")
        });
        thread.join().expect("the reader");
        assert_eq!(read, None, "{name}: times from a pair no submission wrote");
        core.stamps.give_back(pair);
    });
}
