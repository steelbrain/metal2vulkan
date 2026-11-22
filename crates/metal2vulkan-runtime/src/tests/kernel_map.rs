//! The device's compiled kernels by content (`kernel_map`).
//!
//! - The map of at most `cap` entries, filled past its cap, holds `cap`, forgets the least
//!   recently used (a lookup counts as a use), and keeps the entry it holds when a key is kept
//!   twice.
//! - Under a device cfg, on every device of the cfg's kind: two states of one content share one
//!   translation, one shader module and one `VkPipeline` per variant, for a kernel that tiles its
//!   grid and for one that observes its threadgroups, and a make whose equal landed in the map
//!   first is dropped and counted; states of other constant values, or of
//!   other linked functions, do not share; filled past a cap of 2, the map holds 2, a forgotten
//!   kernel lives while a state holds it and goes once that state does, and a content forgotten
//!   compiles again; every gauge reads 0 once the device is gone.

use crate::kernel_map::Lru;

/// The map filled past its cap: the least recently used goes first, a lookup is a use, and a key
/// kept twice keeps the first value and lets the second go.
#[test]
fn the_map_holds_its_cap_and_forgets_the_least_recently_used() {
    let mut map: Lru<u32, &str> = Lru::new(3);
    for (key, value) in [(1, "one"), (2, "two"), (3, "three")] {
        assert_eq!(map.keep(key, value), (value, None));
    }
    assert_eq!(map.get(&1), Some("one"));
    // 2 is now the least recently used.
    assert_eq!(map.keep(4, "four"), ("four", Some("two")));
    assert_eq!(map.len(), 3);
    assert_eq!(map.get(&2), None);
    assert_eq!(
        map.keep(1, "uno"),
        ("one", Some("uno")),
        "the entry held stays"
    );
    // 3 is now the least recently used, then 4.
    assert_eq!(map.keep(5, "five"), ("five", Some("three")));
    assert_eq!(map.keep(6, "six"), ("six", Some("four")));
    assert_eq!(map.len(), 3);
    for (key, value) in [(1, "one"), (5, "five"), (6, "six")] {
        assert_eq!(map.get(&key), Some(value));
    }
    for key in 7..100 {
        map.keep(key, "more");
    }
    assert_eq!(map.len(), 3);
    assert_eq!(map.clear().len(), 3);
    assert_eq!(map.len(), 0);
}

#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod on_device {
    use crate::kernel_map::KernelKey;
    use crate::tests::command::on_device::each_device;
    use crate::{
        ComputePipelineDescriptor, ComputePipelineState, Device, FunctionDescriptor, Library,
        StageSource,
    };
    use metal2vulkan::library::{FunctionConstantValue, FunctionConstantValues};
    use metal2vulkan::metal::DataType;
    use metal2vulkan::pipeline::LinkedFunctions;
    use metal2vulkan_devtools::{engine_dir, fixtures_dir, llvm_assemble, AssembleError};
    use std::path::Path;
    use std::sync::Arc;

    /// The library of the engine fixture `fixture.ll`, or None where `llvm-as` is missing.
    fn library(fixture: &str) -> Option<Library> {
        library_at(&engine_dir().join(format!("{fixture}.ll")))
    }

    /// The library of the fixture at `ll`, or None where `llvm-as` is missing.
    fn library_at(ll: &Path) -> Option<Library> {
        match llvm_assemble(ll) {
            Ok(bitcode) => Some(Library::from_air(&bitcode).expect("the library")),
            Err(AssembleError::Missing(why)) => {
                eprintln!("SKIPPED: {why}");
                None
            }
            Err(error) => panic!("{}: {error:?}", ll.display()),
        }
    }

    /// The harness's `signed_zero_products`, whose two `float` function constants are 0 and 1.
    fn products() -> Option<metal2vulkan::library::Function> {
        let library = library_at(&fixtures_dir().join("harness/workloads.ll"))?;
        Some(
            library
                .function("signed_zero_products")
                .expect("the kernel"),
        )
    }

    /// The descriptor of `function`.
    fn of(function: FunctionDescriptor) -> ComputePipelineDescriptor {
        ComputePipelineDescriptor::new(StageSource::Function(function))
    }

    fn state(
        name: &str,
        device: &Device,
        descriptor: &ComputePipelineDescriptor,
    ) -> ComputePipelineState {
        device
            .new_compute_pipeline_state(descriptor, crate::tests::no_constexpr_samplers)
            .unwrap_or_else(|error| panic!("{name}: {error}"))
    }

    /// Whether two states share their compiled kernel.
    fn shared(a: &ComputePipelineState, b: &ComputePipelineState) -> bool {
        Arc::ptr_eq(&a.inner.compiled, &b.inner.compiled)
    }

    /// `signed_zero_products` with its constants `value` and `-value`.
    fn vector(function: &metal2vulkan::library::Function, value: u32) -> ComputePipelineDescriptor {
        let float = |v: f32| FunctionConstantValue::new(DataType::Float, v.to_le_bytes());
        let constants = FunctionConstantValues::defaults()
            .with_index(0, float(value as f32))
            .with_index(1, float(-(value as f32)));
        let mut function = FunctionDescriptor::new(function.clone());
        function.constants = constants;
        function.specialized = true;
        of(function)
    }

    /// Two states of one content share one translation, one shader module and one `VkPipeline`
    /// per variant: `a1_bool_from_float`, which tiles its grid (one lowering, and its tiles'
    /// `VkPipeline` made with the first state alone), and `icb_replay_grid_template`, which
    /// observes its threadgroups (one lowering, its partial-threadgroups variant, with the first
    /// state alone). The map holds one
    /// entry each, and a kernel lives while a state or the map holds it. A make kept under a key
    /// the map holds already answers the map's kernel, and counts in
    /// `Metrics::kernel_makes_dropped`.
    #[test]
    fn every_device_shares_one_compiled_kernel_between_states_of_one_content() {
        let (Some(tiled), Some(observing)) = (
            library("a1_bool_from_float"),
            library("icb_replay_grid_template"),
        ) else {
            return;
        };
        let tiled = of(FunctionDescriptor::new(
            tiled.function("a1_bool_from_float").expect("the kernel"),
        ));
        let observing = of(FunctionDescriptor::new(
            observing
                .function("icb_replay_grid_template")
                .expect("the kernel"),
        ));
        each_device("kernel_map_shared", |name, _, device| {
            let counters = device.counters();
            for (kernel, descriptor, lowerings, pipelines) in
                [("tiled", &tiled, 1, 1), ("observing", &observing, 1, 0)]
            {
                let before = counters.metrics();
                let first = state(name, &device, descriptor);
                let made = counters.metrics();
                assert_eq!(
                    made.translations - before.translations,
                    lowerings,
                    "{name}: {kernel}: lowerings"
                );
                assert_eq!(
                    made.pipelines_created - before.pipelines_created,
                    pipelines,
                    "{name}: {kernel}: pipelines"
                );
                let second = state(name, &device, &descriptor.clone());
                let again = counters.metrics();
                assert_eq!(
                    again.translations, made.translations,
                    "{name}: {kernel}: a second lowering"
                );
                assert_eq!(
                    again.translations_loaded, made.translations_loaded,
                    "{name}: {kernel}"
                );
                assert_eq!(
                    again.pipelines_created, made.pipelines_created,
                    "{name}: {kernel}: a second pipeline"
                );
                assert!(
                    shared(&first, &second),
                    "{name}: {kernel}: one compiled kernel"
                );
                assert!(!first.is(&second), "{name}: {kernel}: two states");
                let compiled = Arc::clone(&first.inner.compiled);
                drop((first, second));
                // The map holds it still, beside this test's handle.
                assert_eq!(Arc::strong_count(&compiled), 2, "{name}: {kernel}");
            }
            // A make whose equal landed first: the observing kernel's content kept under the
            // tiled kernel's key, which the map holds.
            let core = device.core();
            let key = |descriptor: &ComputePipelineDescriptor| KernelKey {
                pipeline: core.translator.compute_pipeline_key(descriptor),
                required: descriptor.required_threads_per_threadgroup,
            };
            let held = (core.kernels.find(&key(&tiled))).expect("the tiled kernel");
            let ours = (core.kernels.find(&key(&observing))).expect("the observing kernel");
            let dropped = counters.metrics().kernel_makes_dropped;
            let kept = core.kernels.keep(key(&tiled), ours, &core.counters);
            assert!(
                Arc::ptr_eq(&kept, &held) && counters.metrics().kernel_makes_dropped == dropped + 1,
                "{name}: a make whose equal landed first, dropped and counted"
            );
            let gauges = counters.gauges();
            assert_eq!(
                (gauges.kernels, gauges.kernel_entries),
                (2, 2),
                "{name}: the map's kernels"
            );
            println!("{name}: two states of a content share its kernel, for a tiling and an observing kernel");
        });
    }

    /// States of other constant values, or of other linked functions, do not share; a state of
    /// the first constant values again shares the first's.
    #[test]
    fn every_device_compiles_other_constants_and_linked_functions_apart() {
        let (Some(constant), Some(caller), Some(functions), Some(other)) = (
            products(),
            library("linked_visible_caller"),
            library("linked_visible_functions"),
            library("linked_other_functions"),
        ) else {
            return;
        };
        let entry = caller
            .function("linked_visible_caller")
            .expect("the kernel");
        let offset = caller.function("linked_offset").expect("linked_offset");
        let blend = functions.function("linked_blend").expect("linked_blend");
        let linked = |scale| {
            of(FunctionDescriptor::new(entry.clone()).with_linked(
                LinkedFunctions::new()
                    .with_function(scale)
                    .with_function(blend.clone())
                    .with_private_function(offset.clone()),
            ))
        };
        let scale = functions.function("linked_scale").expect("linked_scale");
        let other_scale = other
            .function("linked_scale")
            .expect("the other linked_scale");
        each_device("kernel_map_apart", |name, _, device| {
            let counters = device.counters();
            let one = state(name, &device, &vector(&constant, 1));
            let two = state(name, &device, &vector(&constant, 2));
            let lowered = counters.metrics().translations;
            let one_again = state(name, &device, &vector(&constant, 1));
            assert!(!shared(&one, &two), "{name}: other constant values");
            assert!(shared(&one, &one_again), "{name}: the same constant values");
            assert_eq!(counters.metrics().translations, lowered, "{name}");
            let first = state(name, &device, &linked(scale.clone()));
            let second = state(name, &device, &linked(other_scale.clone()));
            let first_again = state(name, &device, &linked(scale.clone()));
            assert!(!shared(&first, &second), "{name}: other linked functions");
            assert!(
                shared(&first, &first_again),
                "{name}: the same linked functions"
            );
            assert_eq!(counters.gauges().kernel_entries, 4, "{name}");
            println!("{name}: other constants and other linked functions compile apart");
        });
    }

    /// Filled past a cap of 2, the map holds 2 and forgets the least recently used, which lives
    /// while a state holds it and goes once that state does; a content forgotten compiles again,
    /// and one the map holds does not.
    #[test]
    fn every_device_bounds_the_map_and_frees_a_forgotten_kernel_with_its_last_state() {
        let Some(constant) = products() else {
            return;
        };
        each_device("kernel_map_past_cap", |name, _, device| {
            device.core().kernels.set_cap(2);
            let counters = device.counters();
            let gauges = || {
                let gauges = counters.gauges();
                (gauges.kernels, gauges.kernel_entries)
            };
            let lowered = || counters.metrics().translations;
            let start = lowered();
            let one = state(name, &device, &vector(&constant, 1));
            let each = lowered() - start;
            assert!(each > 0, "{name}: a lowering");
            let two = state(name, &device, &vector(&constant, 2));
            assert_eq!(gauges(), (2, 2), "{name}");
            // 1 is the least recently used, and goes from the map; its state keeps it.
            let three = state(name, &device, &vector(&constant, 3));
            assert_eq!(
                gauges(),
                (3, 2),
                "{name}: a forgotten kernel its state holds"
            );
            let two_again = state(name, &device, &vector(&constant, 2));
            assert!(shared(&two, &two_again), "{name}: 2 is held");
            assert_eq!(lowered() - start, 3 * each, "{name}: 2 compiled once");
            drop(one);
            assert_eq!(
                gauges(),
                (2, 2),
                "{name}: the forgotten kernel went with its last state"
            );
            // 1 compiles again, and 3, now the least recently used, goes from the map.
            let one = state(name, &device, &vector(&constant, 1));
            assert_eq!(lowered() - start, 4 * each, "{name}: 1 compiled again");
            assert_eq!(gauges(), (3, 2), "{name}");
            drop(three);
            assert_eq!(gauges(), (2, 2), "{name}");
            drop((one, two, two_again));
            assert_eq!(gauges(), (2, 2), "{name}: the map holds 2");
            println!(
                "{name}: the map held 2 of 3 contents, and freed a forgotten kernel with its state"
            );
        });
    }
}
