use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Path, PathBuf};
mod build_support;
mod product_fingerprint;
use build_support::hash_files;
use product_fingerprint::{collect_files, product_fingerprint};

fn main() {
    // The product fingerprint (`product_fingerprint.rs`): the translator's, the runtime's, and the
    // dev-tools crate's sources beside this crate in the workspace's crates/, and nothing of this
    // crate.
    println!(
        "cargo:rustc-env=METAL2VULKAN_PRODUCT_FINGERPRINT={}",
        product_fingerprint(Path::new(".."))
    );

    // This crate's own sources (its manifest, this script and its fingerprint modules, and every
    // file under src/): the build
    // identity a device worker reports, so a parent refuses a worker rebuilt since it started. It
    // keys one cached fact: a case's reference facts (`c3::reference::CASE_FACTS_ABI`), which a
    // case's check derives through most of this crate.
    let validation = PathBuf::from(".");
    let mut validation_hash = Sha256::new();
    hash_files(&mut validation_hash, &validation, &crate_files(&validation));
    println!(
        "cargo:rustc-env=METAL2VULKAN_VALIDATION_FINGERPRINT={}",
        hex(&validation_hash.finalize())
    );

    // Authoring-capability classification (`triage::ANALYZER_ABI`) reads exactly these files of
    // this crate beside the product: its manifest (which selects the product's features) and the
    // modules its non-test code reaches. An edit to one invalidates the cached facts; an edit
    // anywhere else in the crate does not. `triage`'s key test holds the list closed over every
    // module file those sources reach.
    let classifier_inputs = [
        "Cargo.toml",
        "src/triage.rs",
        "src/air.rs",
        "src/air/calls.rs",
        "src/air/fc_specialize.rs",
        "src/air/meta.rs",
        "src/air/meta/embedded.rs",
        "src/air/meta/fragment.rs",
        "src/air/meta/function_constants.rs",
        "src/air/meta/globals.rs",
        "src/air/meta/intersections.rs",
        "src/air/meta/kernel.rs",
        "src/air/meta/mesh.rs",
        "src/air/meta/textures.rs",
        "src/air/meta/types.rs",
        "src/air/meta/vertex.rs",
        "src/air/projection.rs",
        "src/air/reflect.rs",
        "src/air/reflect/access.rs",
        "src/air/reflect/build.rs",
        "src/air/reflect/descriptors.rs",
        "src/air/reflect/samplers.rs",
        "src/air/sanitize.rs",
        "src/air/text.rs",
        "src/air/visible_references.rs",
        "src/executor_contract.rs",
        "src/executor_contract/mesh.rs",
        "src/executor_contract/min_lod_clamp.rs",
        "src/executor_contract/rate_map.rs",
        "src/executor_contract/rate_map/layout.rs",
        "src/executor_contract/rate_map/value.rs",
        "src/executor_contract/sample_rate.rs",
        "src/requirement.rs",
        "src/check/cfg_fact.rs",
        "src/observation_contract.rs",
        "src/observation_contract/comparison.rs",
        "src/case.rs",
        "src/hash.rs",
    ];
    let classifier_files = classifier_inputs
        .iter()
        .map(|path| validation.join(path))
        .collect::<Vec<_>>();
    let mut classifier_hash = Sha256::new();
    hash_files(&mut classifier_hash, &validation, &classifier_files);
    println!(
        "cargo:rustc-env=METAL2VULKAN_CLASSIFIER_INPUTS={}",
        classifier_inputs.join(":")
    );
    println!(
        "cargo:rustc-env=METAL2VULKAN_CLASSIFIER_FINGERPRINT={}",
        hex(&classifier_hash.finalize())
    );

    // The product's runtime: the modules that act on a compiled pipeline after the compile, whose
    // behaviour C3's currency holds to `metal2vulkan::cache::RUNTIME_VERSION` beside an artifact's
    // content digest. That is the binder (`src/bind`: planning, instantiation, the device replay,
    // resident sets, status decoding), the device a run opens and what it enables (`src/device`,
    // `src/ash_interop`), the Vulkan and Metal vocabularies the runtime maps (`src/vk`,
    // `src/metal`, and the AIR front end's `metal2vulkan-air/src/metal`, whose `MTLDataType`
    // `metal` re-exports), the record and table formats it writes (indirect commands, the rate and
    // tile maps, function tables; the rate map's and function table's words are the AIR front end's
    // `metal2vulkan-air/src/canon` formats), root tables, handles and the threadgroup regions'
    // alignment (`src/memory`), the dispatch-region encoding (`src/lower/dispatch.rs`), the
    // shader-side stencil's shadow format and state word (`src/lower/fragment/stencil.rs`), an
    // emulated mesh draw's work buffer layout (`src/lower/mesh/work.rs`), an emulated tessellation
    // draw's work buffer layout, factor reads and tessellator, which its scan runs
    // (`src/lower/tessellation/{work,factors,tessellator}.rs`),
    // the sRGB steps the ASTC and PVRTC decoders read (`src/lower/arith/srgb.rs`), the sRGB
    // transfer and half conversions the mipmap filter takes from `air.pack`'s lowering
    // (`src/lower/arith/pack.rs`), the blend color an
    // extended-range blend reads (`src/lower/arith/extended_range/blend_color.rs`), half-float
    // conversion, and the Metal-like runtime (`metal2vulkan-runtime/src`, with the measured answers
    // it reads), through which the executor runs the families that have moved onto it. The
    // translator's paths above are below its crate (`metal2vulkan/src/...`). Each path is a product
    // file below the workspace's crates/ directory, this crate's parent: those listed here, the
    // binder's Rust sources under `metal2vulkan/src/bind/`, and every file under
    // `metal2vulkan-runtime/src/`, test modules aside (`runtime_files`, so a file added under
    // either joins), all sorted as strings. The guard test in `candidate::engine::artifact`
    // requires this hash to be the one validation pins (`runtime-sources.sha256`), and the inputs
    // to be closed over the module files their non-test code reaches (each is an input, or listed
    // there with why a run does not depend on it beyond the artifact digest).
    let listed = [
        "metal2vulkan-air/src/canon/function_table.rs",
        "metal2vulkan-air/src/canon/rate_map_data.rs",
        "metal2vulkan-air/src/metal/generated.rs",
        "metal2vulkan-air/src/metal/mod.rs",
        "metal2vulkan/src/ash_interop/convert.rs",
        "metal2vulkan/src/ash_interop/mod.rs",
        "metal2vulkan/src/ash_interop/query.rs",
        "metal2vulkan/src/device/check.rs",
        "metal2vulkan/src/device/derive.rs",
        "metal2vulkan/src/device/description.rs",
        "metal2vulkan/src/device/driver_facts.rs",
        "metal2vulkan/src/device/enables.rs",
        "metal2vulkan/src/device/function_storage.rs",
        "metal2vulkan/src/device/gpus.rs",
        "metal2vulkan/src/device/khronos.rs",
        "metal2vulkan/src/device/mechanism.rs",
        "metal2vulkan/src/device/metal.rs",
        "metal2vulkan/src/device/mod.rs",
        "metal2vulkan/src/device/policy.rs",
        "metal2vulkan/src/device/profile.rs",
        "metal2vulkan/src/device/requirements.rs",
        "metal2vulkan/src/device/subgroup.rs",
        "metal2vulkan/src/float16.rs",
        "metal2vulkan/src/indirect_commands.rs",
        "metal2vulkan/src/indirect_commands/descriptor.rs",
        "metal2vulkan/src/indirect_commands/host.rs",
        "metal2vulkan/src/lower/dispatch.rs",
        "metal2vulkan/src/lower/fragment/stencil.rs",
        "metal2vulkan/src/lower/arith/exact_division.rs",
        "metal2vulkan/src/lower/arith/extended_range/blend_color.rs",
        "metal2vulkan/src/lower/arith/pack.rs",
        "metal2vulkan/src/lower/arith/srgb.rs",
        "metal2vulkan/src/lower/mesh/work.rs",
        "metal2vulkan/src/lower/tessellation/factors.rs",
        "metal2vulkan/src/lower/tessellation/tessellator.rs",
        "metal2vulkan/src/lower/tessellation/work.rs",
        "metal2vulkan/src/memory/arena.rs",
        "metal2vulkan/src/memory/handles.rs",
        "metal2vulkan/src/memory/roots.rs",
        "metal2vulkan/src/metal/bicubic.rs",
        "metal2vulkan/src/metal/format/extended_srgb.rs",
        "metal2vulkan/src/metal/format/footprint.rs",
        "metal2vulkan/src/metal/format/mod.rs",
        "metal2vulkan/src/metal/format/packed_codes.rs",
        "metal2vulkan/src/metal/format/padded_srgb.rs",
        "metal2vulkan/src/metal/format/word.rs",
        "metal2vulkan/src/metal/generated.rs",
        "metal2vulkan/src/metal/mod.rs",
        "metal2vulkan/src/metal/sampler.rs",
        "metal2vulkan/src/metal/state.rs",
        "metal2vulkan/src/metal/texel_format.rs",
        "metal2vulkan/src/metal/vertex.rs",
        "metal2vulkan/src/pipeline/table.rs",
        "metal2vulkan/src/pipeline/tile_map.rs",
        "metal2vulkan/src/vk/draw.rs",
        "metal2vulkan/src/vk/generated.rs",
        "metal2vulkan/src/vk/mod.rs",
        "metal2vulkan/src/vk/structs.rs",
    ];
    let product = PathBuf::from("..");
    let mut runtime_inputs = listed.map(String::from).to_vec();
    runtime_files(
        &product,
        Path::new("metal2vulkan/src/bind"),
        Walk::Sources,
        &mut runtime_inputs,
    );
    runtime_files(
        &product,
        Path::new("metal2vulkan-runtime/src"),
        Walk::Files,
        &mut runtime_inputs,
    );
    runtime_inputs.sort();
    let runtime_files = runtime_inputs
        .iter()
        .map(|path| product.join(path))
        .collect::<Vec<_>>();
    let mut runtime_hash = Sha256::new();
    hash_files(&mut runtime_hash, &product, &runtime_files);
    println!(
        "cargo:rustc-env=METAL2VULKAN_RUNTIME_INPUTS={}",
        runtime_inputs.join(":")
    );
    println!(
        "cargo:rustc-env=METAL2VULKAN_RUNTIME_FINGERPRINT={}",
        hex(&runtime_hash.finalize())
    );
}

/// This crate's manifest, build script and its fingerprint modules (`product_fingerprint.rs`, its
/// copy of `build_support.rs`), and every file under its `src/`, sorted.
fn crate_files(root: &Path) -> Vec<PathBuf> {
    let mut files = vec![
        root.join("Cargo.toml"),
        root.join("build.rs"),
        root.join("product_fingerprint.rs"),
        root.join("build_support.rs"),
    ];
    collect_files(&root.join("src"), &mut files);
    files.sort();
    files
}

/// What a walked directory gives the runtime's inputs.
#[derive(Clone, Copy, PartialEq)]
enum Walk {
    /// Every file: the runtime crate's sources and the measured answers they read.
    Files,
    /// The Rust sources alone: the binder (`metal2vulkan/src/bind/`), whose non-test code reads no data file
    /// (its `.txt` files are its tests' measured answers).
    Sources,
}

/// Every file below `relative` under `product` that `walk` takes, but test modules (a `tests`
/// directory, a `tests.rs` or `*_tests.rs` file) and hidden files, pushed as paths relative to
/// `product` with `/` separators. Cargo reruns this script when anything under the directory
/// changes, so a file added there joins the runtime's inputs.
fn runtime_files(product: &Path, relative: &Path, walk: Walk, output: &mut Vec<String>) {
    let directory = product.join(relative);
    println!("cargo:rerun-if-changed={}", directory.display());
    let entries = fs::read_dir(&directory)
        .unwrap_or_else(|error| panic!("read {}: {error}", directory.display()));
    for entry in entries {
        let entry = entry.expect("directory entry");
        let name = entry.file_name().to_string_lossy().into_owned();
        if name.starts_with('.') {
            continue;
        }
        let path = relative.join(&name);
        if entry.path().is_dir() {
            if name != "tests" {
                runtime_files(product, &path, walk, output);
            }
        } else if name != "tests.rs"
            && !name.ends_with("_tests.rs")
            && (walk == Walk::Files || name.ends_with(".rs"))
        {
            output.push(path.to_string_lossy().replace('\\', "/"));
        }
    }
}

/// `bytes` as lowercase hex, two digits a byte: the form of every digest string here.
fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}
