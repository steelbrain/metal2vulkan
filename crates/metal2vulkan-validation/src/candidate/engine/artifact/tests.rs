//! The case artifact digest and the runtime guard.

use std::collections::BTreeMap;
use std::path::Path;

use super::*;

/// The hash of the runtime's sources this build pins (the runtime guard): 64 lowercase hex digits
/// and a newline, which the orchestrator's merge tooling rewrites after a merge that leaves it
/// stale.
const RUNTIME_SOURCES: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/runtime-sources.sha256"
));

fn digest(byte: u8) -> Hash256 {
    [byte; 32]
}

/// The artifact names its parts by subject: the order a run recorded them in does not enter, a
/// subject recorded twice must agree with itself, and nothing recorded is no artifact.
#[test]
fn a_case_artifact_is_its_parts_by_subject() {
    let empty = CaseArtifact::default();
    assert_eq!(empty.sha256(), None);
    assert_eq!(
        empty.executed(),
        ExecutedArtifact {
            runtime_version: metal2vulkan::cache::RUNTIME_VERSION,
            sha256: None,
        }
    );

    let variant = subject_label(CompileSubject::ReplayVariant {
        token: 3,
        threads_per_group: [32, 1, 1],
    });
    assert_eq!(variant, "replay-variant/3/32x1x1");
    let mut a = CaseArtifact::default();
    a.insert("case".into(), digest(1)).unwrap();
    a.insert(variant.clone(), digest(2)).unwrap();
    let mut b = CaseArtifact::default();
    b.insert(variant.clone(), digest(2)).unwrap();
    b.insert("case".into(), digest(1)).unwrap();
    assert_eq!(a.sha256(), b.sha256());
    assert_eq!(a.sha256().unwrap().len(), 64);

    // The same subject again with the same digest is one part; with another, a failure.
    b.insert("case".into(), digest(1)).unwrap();
    assert_eq!(a.sha256(), b.sha256());
    assert!(b.insert("case".into(), digest(9)).is_err());

    let mut c = CaseArtifact::default();
    c.insert("case".into(), digest(1)).unwrap();
    assert_ne!(
        a.sha256(),
        c.sha256(),
        "a missing variant moves the artifact"
    );
    c.insert(variant, digest(3)).unwrap();
    assert_ne!(a.sha256(), c.sha256(), "a moved variant moves the artifact");

    // A run that compiled a host replay's pipeline has no digest, whatever else it compiled.
    a.record_uncovered();
    assert_eq!(a.sha256(), None);
    assert_eq!(a.executed().sha256, None);
}

/// The record's JSON: the runtime version always, the digest when there is one, nothing else.
#[test]
fn an_executed_artifact_round_trips() {
    for executed in [
        ExecutedArtifact {
            runtime_version: 1,
            sha256: Some("ab".repeat(32)),
        },
        ExecutedArtifact {
            runtime_version: 1,
            sha256: None,
        },
    ] {
        let json = serde_json::to_value(&executed).unwrap();
        assert_eq!(
            serde_json::from_value::<ExecutedArtifact>(json.clone()).unwrap(),
            executed
        );
        assert_eq!(json.get("sha256").is_some(), executed.sha256.is_some());
    }
    assert!(
        serde_json::from_value::<ExecutedArtifact>(serde_json::json!({
            "runtime_version": 1,
            "extra": true
        }))
        .is_err()
    );
}

/// The runtime guard. Validation pins the hash of the product's runtime sources
/// (`crates/metal2vulkan-validation/runtime-sources.sha256`) beside the product's `RUNTIME_VERSION`; this fails when
/// they differ, so an edit there is either declared free of
/// behaviour (the hash alone moves) or bumps the version. The build's hash is the one computed
/// here, and the inputs are closed over the module files their non-test code reaches: each such
/// file is an input, or is listed below with why a run's behaviour does not depend on it beyond
/// what the artifact digest already covers.
#[test]
fn the_runtime_sources_are_the_committed_ones() {
    use sha2::{Digest, Sha256};
    let product = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    let inputs = env!("METAL2VULKAN_RUNTIME_INPUTS")
        .split(':')
        .collect::<Vec<_>>();
    let files = inputs
        .iter()
        .map(|path| product.join(path))
        .collect::<Vec<_>>();
    let mut hash = Sha256::new();
    crate::build_support::hash_files(&mut hash, &product, &files);
    let now = crate::hash::hex_encode(hash.finalize());
    assert_eq!(
        now,
        env!("METAL2VULKAN_RUNTIME_FINGERPRINT"),
        "the test hashes the runtime's sources the way the build does"
    );
    assert_eq!(
        RUNTIME_SOURCES.trim(),
        now,
        "the product's runtime sources changed (the validation crate's build.rs runtime inputs). If the \
         edit changes no behaviour of a run, write {now} to crates/metal2vulkan-validation/runtime-sources.sha256 \
         and say why in the commit; otherwise also bump metal2vulkan::cache::RUNTIME_VERSION \
         (ask the orchestrator for the number). See docs/VALIDATION.md \"The runtime \
         guard\"."
    );

    let compile = "compile-time translation, reached only from the compile-time parts of mixed runtime files; what it produces is in the pipeline the artifact digests";
    let values = "the compiled pipeline's value types; the artifact digests their values";
    let spirv = "the SPIR-V builder: at run time it builds the device replay's preprocess module, whose words the artifact digests, the indirect clamp, which the executor refuses to run, and the visibility resolve, which the executor never runs";
    let packed = "compile-time translation, reached only through lower/arith/pack.rs's pack and unpack intrinsics, which the compile emits; the runtime's mipmap kernel takes that file's sRGB transfers (srgb_decode, srgb_encode), which reach none of it";
    let not_runtime = BTreeMap::from([
        ("metal2vulkan-air/src/abi/mod.rs", compile),
        ("metal2vulkan-air/src/budget/mod.rs", "the work budget and its meters: a run it stops ends in a refusal, and it changes no answer"),
        ("metal2vulkan-air/src/container/mod.rs", "the AIR container; the device profile reads its version constants at compile time"),
        ("metal2vulkan-air/src/error.rs", compile),
        ("metal2vulkan-air/src/intrinsics/mod.rs", packed),
        ("metal2vulkan-air/src/ir/mod.rs", compile),
        ("metal2vulkan-air/src/lib.rs", compile),
        ("metal2vulkan/src/cache/digest.rs", "the content digest, which defines the artifact rather than acting on it"),
        ("metal2vulkan/src/cache/mod.rs", "the plan's specialization and variant keys, whose values the plan digest holds, and RUNTIME_VERSION itself, which cannot be among the sources it versions"),
        ("metal2vulkan/src/env_vars.rs", "the operational variables' accessors: the runtime reads only METAL2VULKAN_LOG_PIPELINES and METAL2VULKAN_LOG_MEMORY, default-off diagnostics that print and change no run"),
        ("metal2vulkan/src/error/mod.rs", "the error vocabulary: the runtime's code decides which error a run reports, and a record keeps the error as reported"),
        ("metal2vulkan/src/library/mod.rs", compile),
        ("metal2vulkan/src/lower/access/imageblock.rs", compile),
        ("metal2vulkan/src/lower/access/tile.rs", "compile-time translation, reached only from the compile-time parts of mixed runtime files; what it produces is in the pipeline the artifact digests; the tile staging format reaches a run through the plan"),
        ("metal2vulkan/src/lower/arith/aggregate.rs", compile),
        ("metal2vulkan/src/lower/arith/extended_range.rs", "the extended-range arithmetic, compile-time translation: the runtime reaches it only through the half shadow's load (bind/extended_shadow.rs), which emits tile_value, the fetch's own decode that every pipeline reading such an attachment compiles, so a change to it is a translation change (SEMANTICS_VERSION) that moves the artifact digests"),
        // The block decode (`bind/block_decode.rs`) and its exact division
        // (`lower/arith/exact_division.rs`) reach the numeric forms and the constant emission at
        // run time.
        ("metal2vulkan/src/lower/arith/legal.rs", "the numeric forms (`Form`, `Held`, `FloatFmt`) every compiled pipeline's arithmetic is emitted in; the runtime's block decode and its exact division name their forms with them, so a change to what a form is is a translation change (SEMANTICS_VERSION) that moves the artifact digests"),
        ("metal2vulkan/src/lower/arith/math.rs", packed),
        ("metal2vulkan/src/lower/arith/mod.rs", compile),
        ("metal2vulkan/src/lower/arith/narrow.rs", packed),
        ("metal2vulkan/src/lower/arith/ops.rs", "the constant and type emission (`splat`, `ty`, `elem_ty`, `ones`) every compiled pipeline's arithmetic is made through; the runtime's exact division makes its constants with it, so a change to what it emits is a translation change (SEMANTICS_VERSION) that moves the artifact digests"),
        ("metal2vulkan/src/lower/emit.rs", compile),
        ("metal2vulkan/src/lower/fragment/fetch.rs", compile),
        ("metal2vulkan/src/lower/fragment/mod.rs", "the fragment lowering's module root: the runtime reaches it only through its re-exports of the shader-side stencil's shadow format and state word, whose module is an input"),
        ("metal2vulkan/src/lower/mesh/mod.rs", "compile-time translation, reached only from the compile-time part of lower/dispatch.rs (plan_kernel's mesh wrapper builtins) and as the parent of its work-buffer layout (lower/mesh/work.rs, an input); what it produces is in the pipeline the artifact digests"),
        ("metal2vulkan/src/lower/select.rs", compile),
        ("metal2vulkan/src/lower/simd/divergent.rs", "compile-time translation (the predicated walk); the runtime reads only VOTE_BYTES, the vote word's size, which the compile also adds to the static threadgroup length the artifact reports, so a change to it moves the artifact digest"),
        ("metal2vulkan/src/lower/simd/workgroup.rs","binding-plan construction; the artifact digests the plan it makes (the workgroup arena size)"),
        ("metal2vulkan/src/lower/stage_io/mod.rs", compile),
        ("metal2vulkan/src/lower/sync/memory_model.rs", compile),
        ("metal2vulkan/src/memory/abi_index.rs", compile),
        ("metal2vulkan/src/memory/contain.rs", "the containment encoding, a function of the device profile (the venue's profile key holds it), shared by compile and the vertex layout"),
        ("metal2vulkan/src/memory/leaf.rs", compile),
        ("metal2vulkan/src/memory/plan.rs", compile),
        ("metal2vulkan/src/memory/provenance.rs", compile),
        ("metal2vulkan/src/memory/union_find.rs", compile),
        ("metal2vulkan/src/memory/views.rs", compile),
        ("metal2vulkan/src/pipeline/artifact.rs", values),
        ("metal2vulkan/src/pipeline/compile.rs", compile),
        ("metal2vulkan/src/pipeline/compile/key.rs", "the compile's cache key, which the artifact digest excludes"),
        ("metal2vulkan/src/pipeline/descriptor.rs", "compile-time translation, reached only from the compile-time parts of mixed runtime files; what it produces is in the pipeline the artifact digests (the pipeline descriptors)"),
        ("metal2vulkan/src/pipeline/facts.rs", values),
        ("metal2vulkan/src/pipeline/graphics.rs", "the compiled pipeline's value types; the artifact digests their values (the graphics state)"),
        ("metal2vulkan/src/pipeline/interface.rs", values),
        ("metal2vulkan/src/pipeline/layout.rs", values),
        ("metal2vulkan/src/pipeline/lower_inputs.rs", "the compiled pipeline's value types; the artifact digests their values (the storage image classes)"),
        ("metal2vulkan/src/pipeline/mod.rs", values),
        ("metal2vulkan/src/pipeline/sym.rs", "the compiled pipeline's value types; the artifact digests their values (slots, push fields, handle classes)"),
        ("metal2vulkan/src/pipeline/vertex.rs", "the compiled pipeline's value types; the artifact digests their values (the vertex layout)"),
        ("metal2vulkan/src/reflection/mod.rs", "Metal's reflection answers, which the runtime returns beside a pipeline state it creates or for a library's function, and the bindings it lays an argument encoder out from (`new_argument_encoder_with_buffer_binding`); no run reads them or makes such an encoder"),
        ("metal2vulkan/src/spirv/builder/constructs.rs", spirv),
        ("metal2vulkan/src/spirv/builder/mod.rs", spirv),
        ("metal2vulkan/src/spirv/error.rs", spirv),
        ("metal2vulkan/src/spirv/ids.rs", spirv),
        ("metal2vulkan/src/spirv/ir/mod.rs", spirv),
        ("metal2vulkan/src/spirv/requirements/mod.rs", "the SPIR-V requirements tables: the runtime reads only `extended_format`, whether a storage image format needs `StorageImageExtendedFormats` (SPIR-V's fixed format table), through the view formats a texture lists (metal/format's `storage_view_formats`); the pipelines that write through those views are compiled with the same answer, which the artifact digests"),
        ("metal2vulkan/src/spirv/serialize/mod.rs", spirv),
        ("metal2vulkan/src/spirv/verify/mod.rs", spirv),
        ("metal2vulkan/src/spv.rs", "the SPIR-V grammar's opcodes and enumerants with their values, generated from Khronos's grammar: a value is fixed by the specification, and regenerating only adds or drops names, which no run can observe"),
    ]);
    // The resolver the classifier's key test holds its inputs closed with, across the product,
    // the AIR front end's crate, which the product imports as `crate::air`, and the runtime's
    // crate, whose inputs reach the product through `metal2vulkan::`, each by its directory in the
    // workspace's crates/.
    let members = [
        crate::triage::classifier_inputs::Member {
            name: "metal2vulkan_air",
            dir: "metal2vulkan-air",
        },
        crate::triage::classifier_inputs::Member {
            name: "metal2vulkan",
            dir: "metal2vulkan",
        },
        crate::triage::classifier_inputs::Member {
            name: "metal2vulkan_runtime",
            dir: "metal2vulkan-runtime",
        },
    ];
    let dependencies =
        crate::triage::classifier_inputs::dependencies(&product, &members, &inputs).unwrap();
    let unlisted = dependencies
        .iter()
        .filter(|(file, _)| {
            !inputs.contains(&file.as_str()) && !not_runtime.contains_key(file.as_str())
        })
        .collect::<Vec<_>>();
    assert!(
        unlisted.is_empty(),
        "the runtime inputs reach files that are neither inputs nor listed: {unlisted:#?}"
    );
    for file in not_runtime.keys() {
        assert!(
            dependencies.contains_key(*file),
            "{file} is listed but no runtime input reaches it"
        );
    }
}
