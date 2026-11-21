//! MoltenVK's MSL evaluates each subgroup operation where the module does.
//!
//! SPIRV-Cross forwards a single-reader value into its reader's text, writes `OpSelect` on a
//! scalar condition as `c ? t : f` and `OpLogicalAnd`/`OpLogicalOr` as `&&`/`||`. A subgroup
//! operation forwarded into a `?:` arm or a `&&`/`||` operand runs only in the invocations that
//! take it, and a shuffle there reads inactive lanes: authored case 948cb60a
//! (`kernel_simd_variant_leftovers`, whose `simd_shuffle_and_fill_down` lowers to two shuffles
//! and a select) read 0 in lanes 14-31 of its `half3` row on MoltenVK. The serializer writes such
//! a value with a copy SPIRV-Cross declares where it stands. This test converts the fixture's
//! `moltenvk-m3` module as MoltenVK does and requires every subgroup call of the kernel body
//! outside a `?:` arm and a `&&`/`||` operand.
//!
//! `llvm-as` and `spirv-cross` are external; without them the test skips visibly.

use crate::device::DevicePolicy;
use crate::pipeline::testing::{
    compile, fixtures_dir, link_and_validate, llvm_assemble, moltenvk_msl, profile, AssembleError,
};

fn skip_missing<T>(r: Result<T, AssembleError>, what: &str) -> Option<T> {
    match r {
        Ok(x) => Some(x),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("{what}: {why}"),
    }
}

/// The subgroup calls SPIRV-Cross writes in MSL (its `spvSubgroup*`/`spvQuad*` helpers and
/// Metal's `simd_*`/`quad_*` functions).
const SUBGROUP_CALLS: [&str; 4] = ["spvSubgroup", "spvQuad", "simd_", "quad_"];

/// The operator in front of each subgroup call of `body` that evaluates it lazily (`?`, `:`,
/// `&&`, `||`), with the call's line.
fn lazy_subgroup_calls(body: &str) -> Vec<String> {
    let mut found = Vec::new();
    for line in body.lines() {
        for call in SUBGROUP_CALLS {
            for (at, _) in line.match_indices(call) {
                let before = line[..at].trim_end_matches(|c: char| c == '(' || c.is_whitespace());
                if ["?", ":", "&&", "||"].iter().any(|op| before.ends_with(op))
                    && !before.ends_with("::")
                {
                    found.push(line.trim().to_owned());
                }
            }
        }
    }
    found
}

#[test]
fn moltenvk_msl_evaluates_subgroup_operations_eagerly() {
    let Some(bc) = skip_missing(
        llvm_assemble(&fixtures_dir().join("kernel_simd_variant_leftovers.ll")),
        "llvm-as",
    ) else {
        return;
    };
    let p = profile("moltenvk-m3", DevicePolicy::RESIDENT_TABLES).expect("profile");
    let a = compile(&bc, "kernel_simd_variant_leftovers", &p).expect("compile");
    let (_, words) = link_and_validate(&a, &p).expect("L3");
    let Some(msl) = skip_missing(moltenvk_msl(&words), "spirv-cross") else {
        return;
    };
    let body = msl
        .split_once("kernel void kernel_simd_variant_leftovers")
        .or_else(|| msl.split_once("kernel void main0"))
        .map(|(_, body)| body)
        .expect("the kernel");
    // The fill-down rows' four shuffles are declared, so the check has something to see.
    assert!(
        body.lines()
            .filter(|l| l.trim_start().starts_with("float") && l.contains("= spvSubgroupShuffle("))
            .count()
            >= 4,
        "{body}"
    );
    let lazy = lazy_subgroup_calls(body);
    assert!(
        lazy.is_empty(),
        "lazily evaluated subgroup calls: {lazy:#?}"
    );
}

#[test]
fn the_scan_finds_lazily_evaluated_subgroup_calls() {
    let body = "x = c ? spvSubgroupShuffle(a, i) : k;\n\
                y = c ? k : (spvSubgroupShuffle(b, i) + 1u);\n\
                z = c && simd_all(p);\n\
                w = c || (quad_any(p));\n\
                uint t = spvSubgroupShuffle(a, i);\n\
                v = metal::simd_sum(a);\n";
    assert_eq!(lazy_subgroup_calls(body).len(), 4);
}
