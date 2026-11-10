//! Plain color writers a draw moves into the interlocked storage form.
//!
//! A render pass instance can keep a color out of its subpass for other draws' interlocked storage
//! form (`BindShape::interlocked_colors`), or hold an extended-range color's `half` shadow
//! (`BindShape::extended_shadows`); a draw there of a pipeline that writes the color as an
//! ordinary output runs the variant that writes it through the storage form, inside the fragment
//! function and so before the depth and stencil tests. The owned fixture `smw_writers` has one
//! plain writer of color 0 per way those tests bear on such a store: none (no discard, no other
//! write, or Metal's early tests), its discard alone (the early-tests variant where the tests
//! write nothing, the function's own depth or stencil test where they write), and a write of
//! other memory or a returned depth or sample mask (the function's own tests wherever the draw's
//! tests can drop a fragment).
//!
//! On every committed profile that orders fetch through the pixel interlock alone, in a pass with
//! a depth and stencil attachment, under each kind of draw tests: the plan of the pipeline as
//! compiled selects the variant the moved writer needs at once, and that variant, compiled, plans
//! itself (a fixed point, which the runtime requires of every variant it compiles), or both refuse
//! the draw. A writer whose write mask covers none of the color's channels stores nothing, so no
//! test bears on its write: its plan selects no tests under any draw tests, and its variant states
//! no store before them. A writer that declares the color as an input it never reads (the owned
//! fixture `smw_unread`) reads nothing there, so it takes no storage form as compiled and every draw
//! that moves no color runs it so. No GPU is used.

use std::path::Path;

use crate::bind::{BindState, FragmentTests};
use crate::budget::Budget;
use crate::cache::VariantKey;
use crate::device::{Device, DeviceDescription, DevicePolicy};
use crate::error::BindError;
use crate::library::Library;
use crate::lower::fragment::interlock_orders;
use crate::metal::{ColorWriteMask, PixelFormat};
use crate::pipeline::descriptor::{
    ColorAttachmentDescriptor, FunctionDescriptor, RenderPipelineDescriptor, StageSource,
};
use crate::pipeline::facts::RuntimeFact;
use crate::pipeline::testing::{engine_dir, llvm_assemble, AssembleError};
use crate::spirv::testing::{check_words_valid, committed_profiles};
use crate::vk;

/// How the depth and stencil tests bear on a writer's store before them.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Class {
    /// The variant declares early tests: no draw's tests constrain it.
    Free,
    /// Only its discard keeps it from early tests.
    EarlyVariant,
    /// It writes other memory or returns depth or a sample mask: only tests that drop nothing.
    Untested,
}

/// The writers of `smw_writers` and their classes.
const WRITERS: [(&str, Class); 6] = [
    ("smw_plain", Class::Free),
    ("smw_early", Class::Free),
    ("smw_discard", Class::EarlyVariant),
    ("smw_store", Class::Untested),
    ("smw_depth", Class::Untested),
    ("smw_mask", Class::Untested),
];

/// How a draw moves color 0 into the storage form: an interlocked `RGBA16Float` color, or a
/// `BGRA10_XR` color's `half` shadow.
#[derive(Clone, Copy, Debug)]
enum Move {
    Interlocked,
    Shadow,
}

const TESTS: [FragmentTests; 6] = [
    FragmentTests::Pass,
    FragmentTests::Drop,
    FragmentTests::Depth { writes: false },
    FragmentTests::Depth { writes: true },
    FragmentTests::Stencil,
    FragmentTests::Any,
];

/// The library of the owned fixture `name`, assembled with `llvm-as`; `None` where the tool is
/// missing (the test then skips visibly).
fn fixture_library(name: &str) -> Option<Library> {
    match llvm_assemble(&engine_dir().join(format!("{name}.ll"))) {
        Ok(bc) => Some(Library::from_air(&bc).expect("library")),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected {name}: {why}"),
    }
}

/// The function `name` of `library`, as a stage of a render pipeline descriptor.
fn stage(library: &Library, name: &str) -> StageSource {
    StageSource::Function(FunctionDescriptor::new(
        library.function(name).expect("function"),
    ))
}

/// The committed profiles that order fetch through the pixel interlock, each with its device.
fn ordering_devices() -> Vec<(String, Device)> {
    let profiles = committed_profiles().expect("profiles");
    profiles
        .iter()
        .filter(|(_, profile)| interlock_orders(profile))
        .map(|(stem, _)| (stem.clone(), device(stem)))
        .collect()
}

/// The bind state of a draw under `tests` that moves color 0 as `moved` does (none for `None`).
fn moving(tests: FragmentTests, moved: Option<Move>) -> BindState {
    let mut state = BindState {
        fragment_tests: tests,
        ..Default::default()
    };
    match moved {
        Some(Move::Interlocked) => {
            state.interlocked_colors.insert(0);
        }
        Some(Move::Shadow) => {
            state.extended_shadows.insert(0);
        }
        None => {}
    }
    state
}

/// The `early_variant` of the store before the tests `pipeline` states, if it states one.
fn stored_before_tests(pipeline: &crate::pipeline::artifact::RenderPipeline) -> Option<bool> {
    pipeline.dependencies.iter().find_map(|d| match d.fact {
        RuntimeFact::TestedStorageWrites { early_variant, .. } => Some(early_variant),
        _ => None,
    })
}

/// Whether `pipeline` writes color 0 through the storage form, and the channels that write keeps.
fn storage_write(
    pipeline: &crate::pipeline::artifact::RenderPipeline,
) -> Option<vk::ColorComponentFlags> {
    pipeline
        .graphics
        .storage_attachments
        .iter()
        .find(|a| a.color_index == 0 && a.writes)
        .map(|a| a.kept)
}

fn device(stem: &str) -> Device {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join(format!("../metal2vulkan-validation/profiles/{stem}.json"));
    let text = std::fs::read_to_string(path).expect("profile");
    let desc = DeviceDescription::from_khronos_json(&text, None).expect("description");
    Device::new(desc, DevicePolicy::default()).expect("device")
}

/// The early-tests variant (`read_only_tests`) and the function's own tests (`shader_tests`) a
/// writer of `class` needs under `tests`; `None` where the draw is refused (tests that may count
/// what they keep, or that the draw does not state).
fn wanted(class: Class, tests: FragmentTests) -> Option<(bool, bool)> {
    match (class, tests) {
        (Class::Free, _) | (_, FragmentTests::Pass) => Some((false, false)),
        (Class::EarlyVariant, FragmentTests::Drop | FragmentTests::Depth { writes: false }) => {
            Some((true, false))
        }
        (_, FragmentTests::Any) => None,
        _ => Some((false, true)),
    }
}

#[test]
fn a_moved_writer_selects_its_tests_in_its_first_plan() {
    let Some(library) = fixture_library("smw_writers") else {
        return;
    };
    let devices = ordering_devices();
    // The profiles checked, and the writers' variants that planned themselves, by the tests
    // they run (`read_only_tests`, `shader_tests`).
    let mut checked = Vec::new();
    let mut fixed_points = std::collections::BTreeSet::<(&str, (bool, bool))>::new();
    for (stem, dev) in &devices {
        checked.push(stem.as_str());
        for (writer, class) in WRITERS {
            for moved in [Move::Interlocked, Move::Shadow] {
                let mut d = RenderPipelineDescriptor::new(stage(&library, "smw_full"));
                d.fragment = Some(stage(&library, writer));
                d.color_attachments[0] = Some(ColorAttachmentDescriptor::new(match moved {
                    Move::Interlocked => PixelFormat::RGBA16Float,
                    Move::Shadow => PixelFormat::BGRA10XR,
                }));
                d.depth_format = Some(PixelFormat::Depth32FloatStencil8);
                d.stencil_format = Some(PixelFormat::Depth32FloatStencil8);
                let what = format!("{stem} {writer} {moved:?}");
                // A device short of what the writer needs (a fragment store) refuses it whole.
                let base = match dev.compile_render_pipeline(&d, &Budget::standard()) {
                    Ok(base) => base,
                    Err(e) => {
                        eprintln!("{what}: refused as compiled: {e}");
                        continue;
                    }
                };
                assert!(base.graphics.storage_attachments.is_empty(), "{what}");
                // The `early_variant` the function states, and its variant with it.
                let stated = match class {
                    Class::Free => None,
                    Class::EarlyVariant => Some(true),
                    Class::Untested => Some(false),
                };
                let movable = base.dependencies.iter().find_map(|d| match d.fact {
                    RuntimeFact::MovableStorageWrites { early_variant } => Some(early_variant),
                    _ => None,
                });
                assert_eq!(movable, stated, "{what}: the function's movable writes");
                for tests in TESTS {
                    let shape = moving(tests, Some(moved)).shape();
                    let what = format!("{what} {tests:?}");
                    let planned = base.plan(dev, &shape);
                    // The variant the plan names, compiled, plans itself: the runtime fails a draw
                    // whose compiled variant asks for another one.
                    let variant = planned.as_ref().ok().map(|plan| {
                        let variant = dev
                            .compile_render_variant(&d, &plan.variant, &Budget::standard())
                            .unwrap_or_else(|e| panic!("{what}: the variant: {e}"));
                        let again = variant.plan(dev, &shape).unwrap_or_else(|e| {
                            panic!("{what}: the variant planned refuses the draw: {e}")
                        });
                        assert_eq!(again.variant, plan.variant, "{what}: a fixed point");
                        variant
                    });
                    let plan = match (planned, wanted(class, tests)) {
                        (Ok(plan), Some(_)) => plan,
                        (Err(BindError::Unsupported(_)), None) => continue,
                        // The function's own tests the device does not ship.
                        (Err(BindError::Unsupported(u)), Some((_, true))) => {
                            eprintln!("{what}: its own tests refused: {u:?}");
                            continue;
                        }
                        (planned, want) => {
                            panic!(
                                "{what}: planned {:?}, wanted {want:?}",
                                planned.map(|p| p.variant)
                            )
                        }
                    };
                    let (read_only, shader) = wanted(class, tests).expect("planned");
                    assert_eq!(
                        (plan.variant.read_only_tests, plan.variant.shader_tests),
                        (read_only, shader),
                        "{what}"
                    );
                    let variant = variant.expect("compiled for the plan");
                    assert!(
                        storage_write(&variant).is_some(),
                        "{what}: the variant writes color 0 through the storage form"
                    );
                    assert_eq!(
                        stored_before_tests(&variant),
                        stated,
                        "{what}: the variant's store before the tests"
                    );
                    for m in &variant.stages {
                        check_words_valid(&m.spirv, &variant.requirements, dev.profile())
                            .unwrap_or_else(|e| panic!("{what}: L3: {e}"));
                    }
                    fixed_points.insert((writer, (read_only, shader)));
                }
            }
        }
    }
    // Not vacuous: each writer planned itself under every kind of tests it takes.
    eprintln!("moved writers: checked on {checked:?}");
    for (writer, class) in WRITERS {
        let kinds: &[(bool, bool)] = match class {
            Class::Free => &[(false, false)],
            Class::EarlyVariant => &[(false, false), (true, false), (false, true)],
            Class::Untested => &[(false, false), (false, true)],
        };
        for kind in kinds {
            assert!(
                fixed_points.contains(&(writer, *kind)),
                "{writer}: no fixed point with tests {kind:?}"
            );
        }
    }
}

/// A moved writer whose write mask covers none of color 0's channels stores nothing there, so the
/// depth and stencil tests bear on nothing it stores: under every kind of draw tests, `Any`
/// included, the plan of the pipeline as compiled names the variant that moves the color and
/// selects no tests, and that variant, compiled, writes color 0 through the storage form keeping
/// every channel, states no store before the tests, and plans itself. Every state is checked and
/// the misses are counted by kind before the test fails, so its output says how far a plan strays.
#[test]
fn a_moved_writer_that_stores_nothing_selects_no_tests() {
    let Some(library) = fixture_library("smw_writers") else {
        return;
    };
    let devices = ordering_devices();
    let mut planned = std::collections::BTreeSet::<(&str, String, String)>::new();
    // The states that miss, by kind of miss.
    let mut misses = std::collections::BTreeMap::<&str, Vec<String>>::new();
    let mut states = 0usize;
    for (stem, dev) in &devices {
        for (writer, _) in WRITERS {
            for moved in [Move::Interlocked, Move::Shadow] {
                let mut d = RenderPipelineDescriptor::new(stage(&library, "smw_full"));
                d.fragment = Some(stage(&library, writer));
                let mut color = ColorAttachmentDescriptor::new(match moved {
                    Move::Interlocked => PixelFormat::RGBA16Float,
                    Move::Shadow => PixelFormat::BGRA10XR,
                });
                color.write_mask = ColorWriteMask::NONE;
                d.color_attachments[0] = Some(color);
                d.depth_format = Some(PixelFormat::Depth32FloatStencil8);
                d.stencil_format = Some(PixelFormat::Depth32FloatStencil8);
                let what = format!("{stem} {writer} {moved:?} (write mask none)");
                let base = match dev.compile_render_pipeline(&d, &Budget::standard()) {
                    Ok(base) => base,
                    Err(e) => {
                        eprintln!("{what}: refused as compiled: {e}");
                        continue;
                    }
                };
                assert_eq!(stored_before_tests(&base), None, "{what}");
                for tests in TESTS {
                    states += 1;
                    let shape = moving(tests, Some(moved)).shape();
                    let what = format!("{what} {tests:?}");
                    let mut miss = |kind: &'static str| {
                        misses.entry(kind).or_default().push(what.clone());
                    };
                    let plan = match base.plan(dev, &shape) {
                        Ok(plan) => plan,
                        Err(e) => {
                            miss("the plan refuses the draw");
                            eprintln!("{what}: the plan refuses the draw: {e}");
                            continue;
                        }
                    };
                    let key = &plan.variant;
                    if key.read_only_tests || key.shader_tests || key.shader_stencil {
                        miss("the plan selects tests");
                    }
                    let variant = dev
                        .compile_render_variant(&d, key, &Budget::standard())
                        .unwrap_or_else(|e| panic!("{what}: the variant: {e}"));
                    assert_eq!(
                        storage_write(&variant),
                        Some(vk::ColorComponentFlags::known_bits()),
                        "{what}: the variant's write of color 0 keeps every channel"
                    );
                    if stored_before_tests(&variant).is_some() {
                        miss("the variant states a store before the tests");
                    }
                    match variant.plan(dev, &shape) {
                        Ok(again) if again.variant == *key => {}
                        Ok(_) => miss("the variant's plan asks for another variant"),
                        Err(_) => miss("the variant's plan refuses the draw"),
                    }
                    planned.insert((writer, format!("{moved:?}"), format!("{tests:?}")));
                }
            }
        }
    }
    eprintln!(
        "moved writers storing nothing: {states} states on {} profiles",
        devices.len()
    );
    for (kind, whats) in &misses {
        eprintln!("{} states: {kind} (first: {})", whats.len(), whats[0]);
    }
    assert!(misses.is_empty(), "{misses:#?}");
    // Not vacuous: every writer, move and kind of tests planned on some profile.
    assert_eq!(
        planned.len(),
        WRITERS.len() * 2 * TESTS.len(),
        "{planned:?}"
    );
}

/// A writer that declares color 0 as an input it never reads (`smw_unread`, which discards) reads
/// nothing there, so it takes no storage form as compiled, under a write mask that stores and one
/// that covers none of the color's channels: color 0 is a color attachment of the pipeline, the
/// pipeline states no store before the tests, and every draw that moves no color, one whose tests
/// may count what they keep (`Any`) included, runs it as compiled. The misses are counted by kind
/// before the test fails.
#[test]
fn an_unread_color_input_takes_no_storage_form() {
    let (Some(writers), Some(unread)) = (
        fixture_library("smw_writers"),
        fixture_library("smw_unread"),
    ) else {
        return;
    };
    let devices = ordering_devices();
    let mut misses = std::collections::BTreeMap::<&str, Vec<String>>::new();
    let mut states = 0usize;
    for (stem, dev) in &devices {
        for mask in [ColorWriteMask::ALL, ColorWriteMask::NONE] {
            let mut d = RenderPipelineDescriptor::new(stage(&writers, "smw_full"));
            d.fragment = Some(stage(&unread, "smw_unread"));
            let mut color = ColorAttachmentDescriptor::new(PixelFormat::RGBA16Float);
            color.write_mask = mask;
            d.color_attachments[0] = Some(color);
            d.depth_format = Some(PixelFormat::Depth32FloatStencil8);
            d.stencil_format = Some(PixelFormat::Depth32FloatStencil8);
            let what = format!("{stem} smw_unread under {mask:?}");
            let base = dev
                .compile_render_pipeline(&d, &Budget::standard())
                .unwrap_or_else(|e| panic!("{what}: refused as compiled: {e}"));
            if !base.graphics.storage_attachments.is_empty() {
                misses
                    .entry("it takes the storage form")
                    .or_default()
                    .push(what.clone());
            }
            if base.graphics.color_attachments[0].is_none() {
                misses
                    .entry("color 0 is no color attachment")
                    .or_default()
                    .push(what.clone());
            }
            if stored_before_tests(&base).is_some() {
                misses
                    .entry("it states a store before the tests")
                    .or_default()
                    .push(what.clone());
            }
            for tests in TESTS {
                states += 1;
                let what = format!("{what} {tests:?}");
                let kind = match base.plan(dev, &moving(tests, None).shape()) {
                    Ok(plan) if plan.variant == VariantKey::new() => continue,
                    Ok(_) => "the plan asks for a variant",
                    Err(_) => "the plan refuses the draw",
                };
                misses.entry(kind).or_default().push(what);
            }
        }
    }
    eprintln!(
        "unread color input: {states} draws on {} profiles",
        devices.len()
    );
    for (kind, whats) in &misses {
        eprintln!("{}: {kind} (first: {})", whats.len(), whats[0]);
    }
    assert!(misses.is_empty(), "{misses:#?}");
    assert!(states > 0, "no profile checked");
}
