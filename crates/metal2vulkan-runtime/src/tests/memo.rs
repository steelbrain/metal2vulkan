//! The device's memo (`pipeline::memo`): its least-recently-used map keeps both of its bounds
//! through any churn and removes the least recently used first, and the memo of compiled pipelines
//! stays within its capacity while pipelines come and go. Boundedness checks, not timings: every
//! step is deterministic.

use crate::pipeline::memo::{Lru, PipelineMemo, Remembered, MEMO_ENTRIES};
use metal2vulkan::cache::{PipelineArtifact, VariantKey};
use metal2vulkan::device::{Device, DeviceDescription, DevicePolicy};
use metal2vulkan::library::{CompiledFunction, Library};
use metal2vulkan::pipeline::{ComputePipelineDescriptor, FunctionDescriptor, Stage, StageSource};
use metal2vulkan::Budget;
use std::collections::HashSet;
use std::path::Path;
use std::sync::Arc;

/// A fixed sequence of pseudo-random numbers (xorshift).
struct Sequence(u64);

impl Sequence {
    fn next(&mut self, below: u64) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0 % below
    }
}

#[test]
fn churn_keeps_both_bounds_and_the_latest_entry() {
    const MAX_WEIGHT: u64 = 1000;
    const MAX_ENTRIES: usize = 16;
    let mut lru: Lru<u64, u64> = Lru::new(MAX_WEIGHT, MAX_ENTRIES);
    let mut sequence = Sequence(0x9e37_79b9_7f4a_7c15);
    let mut full = false;
    for step in 0..20_000 {
        let key = sequence.next(64);
        if sequence.next(3) == 0 {
            if let Some(value) = lru.get(&key) {
                assert_eq!(*value, key * 7, "a hit is the value kept");
            }
        } else {
            let weight = sequence.next(MAX_WEIGHT / 3);
            lru.insert(key, key * 7, weight);
            assert_eq!(
                lru.get(&key),
                Some(&(key * 7)),
                "step {step}: the latest stays"
            );
        }
        let (weight, entries) = lru.footprint();
        assert!(weight <= MAX_WEIGHT, "step {step}: weight {weight}");
        assert!(entries <= MAX_ENTRIES, "step {step}: {entries} entries");
        full |= entries == MAX_ENTRIES || weight > MAX_WEIGHT - MAX_WEIGHT / 3;
    }
    assert!(full, "the churn reached the bounds");
}

#[test]
fn the_least_recently_used_goes_first() {
    let mut lru: Lru<&str, u32> = Lru::counted(3);
    lru.insert("a", 1, 1);
    lru.insert("b", 2, 1);
    lru.insert("c", 3, 1);
    assert_eq!(lru.get(&"a"), Some(&1));
    lru.insert("d", 4, 1);
    assert_eq!(lru.get(&"b"), None, "b was the least recently used");
    for (key, value) in [("a", 1), ("c", 3), ("d", 4)] {
        assert_eq!(lru.get(&key), Some(&value));
    }
    // By weight: the oldest go until the newest fits.
    let mut lru: Lru<u32, ()> = Lru::new(10, 8);
    for key in 0..5 {
        lru.insert(key, (), 2);
    }
    lru.insert(5, (), 5);
    assert_eq!(lru.footprint(), (9, 3));
    assert!(lru.get(&2).is_none() && lru.get(&3).is_some() && lru.get(&5).is_some());
}

/// A map counting its evictions is told of each entry an insert removed past its bounds, and of
/// nothing else: not an insert within them, not a key's replacement, not a removal by key.
#[test]
fn a_counting_map_tells_each_eviction_and_nothing_else() {
    let told = Arc::new(std::sync::atomic::AtomicU64::new(0));
    let counted = Arc::clone(&told);
    let tell = move |n| {
        counted.fetch_add(n, std::sync::atomic::Ordering::Relaxed);
    };
    let evicted = || told.load(std::sync::atomic::Ordering::Relaxed);
    let mut lru: Lru<u32, ()> = Lru::new(10, 3).counting_evictions(Box::new(tell));
    for key in 0..3 {
        lru.insert(key, (), 1);
    }
    lru.insert(1, (), 1);
    lru.remove(&2);
    assert_eq!(
        evicted(),
        0,
        "within the bounds, a replacement and a removal"
    );
    lru.insert(3, (), 1);
    lru.insert(4, (), 1);
    assert_eq!(evicted(), 1, "past the entries' bound: the oldest");
    lru.insert(5, (), 9);
    assert_eq!(
        evicted(),
        3,
        "past the weight's bound: the two oldest in one insert"
    );
    assert_eq!(lru.footprint(), (10, 2));
}

#[test]
fn replacing_counts_once_and_nothing_heavier_than_the_bound_is_kept() {
    let mut lru: Lru<u32, u32> = Lru::new(10, 4);
    lru.insert(1, 1, 4);
    lru.insert(1, 2, 6);
    assert_eq!(lru.footprint(), (6, 1));
    assert_eq!(lru.get(&1), Some(&2));
    lru.insert(2, 3, 11);
    assert_eq!(lru.get(&2), None, "heavier than the whole bound");
    assert_eq!(lru.footprint(), (6, 1), "and nothing else went for it");
    lru.insert(1, 4, 11);
    assert_eq!(
        lru.footprint(),
        (0, 0),
        "a heavy replacement removes the old value"
    );
    let mut none: Lru<u32, u32> = Lru::counted(0);
    none.insert(1, 1, 1);
    assert_eq!(none.footprint(), (0, 0));
}

#[test]
fn an_entry_found_again_where_it_lies_is_used_as_a_lookup_of_its_key_uses_it() {
    let mut lru: Lru<&str, u32> = Lru::counted(3);
    lru.insert("a", 1, 1);
    let (at, value) = lru.get_with_at(&"a").expect("a is kept");
    assert_eq!(*value, 1);
    lru.insert("b", 2, 1);
    lru.insert("c", 3, 1);
    assert_eq!(lru.get_at(at), Some(&1), "found again where it lies");
    lru.insert("d", 4, 1);
    assert_eq!(lru.get(&"b"), None, "b went first: a was used since");
    assert_eq!(lru.get(&"a"), Some(&1));
    lru.insert("a", 5, 1);
    assert_eq!(lru.get_at(at), None, "a replaced entry lies elsewhere");
    assert_eq!(lru.get(&"a"), Some(&5));
    assert_eq!(lru.footprint(), (3, 3));
}

/// Using the entry used last again moves nothing in the order (`Lru::reordered`), as no entry was
/// used since; using another moves it once. Over a fixed churn of which half the steps take the key
/// taken last, every lookup hits where a map that moves each entry used to the back holds the key,
/// and the two hold as many entries, so the same go first.
#[test]
fn using_the_latest_entry_again_keeps_the_order() {
    let mut lru: Lru<&str, u32> = Lru::counted(2);
    lru.insert("a", 1, 1);
    lru.insert("b", 2, 1);
    for _ in 0..8 {
        assert_eq!(lru.get(&"b"), Some(&2));
    }
    assert_eq!(lru.reordered(), 0, "b was the latest used");
    assert_eq!(lru.get(&"a"), Some(&1));
    assert_eq!(lru.get(&"a"), Some(&1));
    assert_eq!(lru.reordered(), 1, "a moved once");
    lru.insert("c", 3, 1);
    assert_eq!(lru.get(&"b"), None, "b went: the least recently used");
    const ENTRIES: usize = 8;
    let mut lru: Lru<u64, u64> = Lru::counted(ENTRIES);
    let mut model: Vec<u64> = Vec::new();
    let mut sequence = Sequence(0x2545_f491_4f6c_dd1d);
    let mut last = 0;
    let mut hits = 0;
    for step in 0..20_000 {
        let key = match sequence.next(2) {
            0 => last,
            _ => sequence.next(24),
        };
        last = key;
        if sequence.next(3) == 0 {
            lru.insert(key, key * 7, 1);
            model.retain(|&kept| kept != key);
            model.push(key);
            if model.len() > ENTRIES {
                model.remove(0);
            }
        } else {
            let hit = lru.get(&key).copied();
            let held = model.contains(&key);
            assert_eq!(hit, held.then_some(key * 7), "step {step}: key {key}");
            if held {
                hits += 1;
                model.retain(|&kept| kept != key);
                model.push(key);
            }
        }
        assert_eq!(lru.footprint().1, model.len(), "step {step}");
    }
    assert!(hits > 5_000 && lru.reordered() < hits, "{hits} hits");
}

/// The owned `ob3_texture_array` kernel, from its committed AIR, compiled for the committed
/// MoltenVK profile, and that device.
fn texture_array() -> (Device, CompiledFunction) {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let air = root.join("../metal2vulkan-validation/fixtures/public/engine/ob3_texture_array.air");
    let bitcode = std::fs::read(air).expect("the committed AIR");
    let text =
        std::fs::read_to_string(root.join("../metal2vulkan-validation/profiles/moltenvk-m3.json"))
            .expect("profile");
    let description = DeviceDescription::from_khronos_json(&text, None).expect("description");
    let device = Device::new(description, DevicePolicy::default()).expect("device");
    let function = Library::from_air(&bitcode)
        .expect("library")
        .function("ob3_texture_array")
        .expect("function");
    let compiled = device
        .compile_function(
            &FunctionDescriptor::new(function),
            &VariantKey::new().lowering_inputs(Stage::Kernel),
            &Budget::standard(),
        )
        .expect("compiles");
    (device, compiled)
}

#[test]
fn the_memo_of_compiled_pipelines_stays_within_its_capacity() {
    let (device, compiled) = texture_array();
    // Pipelines of one translation that differ in a keyed field: link-only compiles.
    let pipelines: Vec<(Arc<_>, u64)> = (1..=24)
        .map(|threads| {
            let mut descriptor =
                ComputePipelineDescriptor::new(StageSource::Compiled(compiled.clone()));
            descriptor.max_total_threads_per_threadgroup = Some(threads * 8);
            let pipeline = device
                .compile_compute_pipeline(&descriptor, &Budget::standard())
                .expect("links");
            let bytes = pipeline.to_bytes().len() as u64;
            (Arc::new(pipeline), bytes)
        })
        .collect();
    let function_bytes = compiled.to_bytes().len() as u64;
    let heaviest = pipelines.iter().map(|(_, bytes)| *bytes).max().unwrap();
    let capacity = 4 * heaviest + function_bytes;
    let memo = PipelineMemo::new(capacity);
    let mut sequence = Sequence(0x2545_f491_4f6c_dd1d);
    let mut kept = HashSet::new();
    let mut evicted = false;
    for step in 0..2_000 {
        let (pipeline, bytes) = &pipelines[sequence.next(pipelines.len() as u64) as usize];
        if sequence.next(8) == 0 {
            memo.keep_function(&compiled, function_bytes);
            assert!(memo.function(compiled.key()).is_some());
        }
        if memo.compute(&pipeline.key).is_none() {
            evicted |= !kept.insert(pipeline.key);
            memo.keep_compute(pipeline, *bytes);
        }
        match memo.compute(&pipeline.key) {
            Some(Remembered::Compiled(found)) => assert!(Arc::ptr_eq(&found, pipeline)),
            _ => panic!("step {step}: the pipeline just kept is the memo's"),
        }
        let (weight, entries) = memo.footprint();
        assert!(
            weight <= capacity,
            "step {step}: {weight} bytes of {capacity}"
        );
        assert!(entries <= MEMO_ENTRIES, "step {step}: {entries} entries");
    }
    assert!(evicted, "the churn evicted pipelines it needed again");
}
