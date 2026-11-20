//! AIR 2.1's metadata records and texture declarations translate as today's.
//!
//! Apple's AIR 2.1 emitter writes three constructs in a shape its AIR 2.2 and later emitters do
//! not, each of which the front end once refused. The owned fixture `air_2_1_front_end` carries
//! all three: its deployment target (macOS 10.14) is what selects the AIR version, so the module
//! records `air64_v21` and `!air.version = !{i32 2, i32 1, i32 0}`.
//!
//! - The function-constant initializer global sits in no section, where AIR 2.2 and later place it
//!   in `air.fc_initializer` (`abi::check`). A module whose `air.function_constants` record is
//!   refused has no ABI at all.
//! - A vertex output restates its member's interpolation qualifier on `air.vertex_output`, and
//!   restates it partly: `uv`'s `[[centroid_perspective]]` writes `air.centroid` with no
//!   `air.perspective` (`abi::args`). A refused key there also costs the module its ABI.
//! - A sample, gather, or read states no residency status: its result is the texel alone rather
//!   than `{T, i8}`, and a sample carries no min-LOD-clamp operand
//!   (`intrinsics::Texture::status`). A read is also in the operand form without the sampler and
//!   offset positions, which AIR 2.8 added. An unmatched declaration is
//!   `Intrinsic(SignatureMismatch)`, so the module does not read.
//!
//! So every entry compiling on every committed profile is what holds the first two: without them
//! the fixture has no ABI and names no entry. The third is held by the entries that sample, gather
//! and read.
//!
//! Where the declaration states no status, the call's own result is the texel, so
//! `lower::texture::alpha` follows the lanes from the call rather than through an
//! `extractvalue 0`. `a21_blit_fs` takes lane 0 of its read and `a21_alpha_fs` the whole texel, so
//! on a profile whose reads the shader checks the first needs no `OutOfBoundsAlpha`
//! specialization and the second does; reading the result as the whole `{T, i8}` struct would ask
//! for it on both.
//!
//! `llvm-as` assembles the fixture; without it the tests skip visibly.

use crate::pipeline::facts::RuntimeFact;
use crate::pipeline::sym::{SlotClass, SlotRef, Stage};
use crate::pipeline::testing::{
    compile, engine_dir, link_and_validate, llvm_assemble, typed_module, AssembleError,
};
use crate::spirv::testing::committed_profiles;

/// The entry points the fixture declares.
const ENTRIES: &[&str] = &["a21_mip", "a21_blit_vs", "a21_blit_fs", "a21_alpha_fs"];

/// The fixture's bitcode, or `None` (skipped visibly) without `llvm-as`.
fn fixture() -> Option<Vec<u8>> {
    let path = engine_dir().join("air_2_1_front_end.ll");
    match llvm_assemble(&path) {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(t)) => {
            eprintln!("SKIPPED: {t}");
            None
        }
        Err(AssembleError::Rejected(e)) => panic!("llvm-as rejected air_2_1_front_end.ll: {e}"),
    }
}

/// The texture slot `raw` binds: the fragment functions' second texture.
const RAW: SlotRef = SlotRef {
    stage: Stage::Fragment,
    class: SlotClass::Texture,
    index: 1,
};

/// Every entry of an AIR 2.1 module compiles, so its ABI decoded (the function-constant record
/// and the vertex output's interpolation keys) and its status-less texture declarations matched.
/// A storage write whose format the profile needs stated is the one refusal admitted, the same one
/// the support inventory records for this kernel.
#[test]
fn every_entry_of_an_air_2_1_module_compiles() {
    let Some(bc) = fixture() else { return };
    let profiles = committed_profiles().expect("profiles");
    assert!(!profiles.is_empty());
    for (stem, p) in &profiles {
        for entry in ENTRIES {
            let what = format!("{entry} on {stem}");
            match compile(&bc, entry, p) {
                Ok(a) => {
                    link_and_validate(&a, p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
                }
                Err(crate::error::Error::NeedsInput(_)) if *entry == "a21_mip" => {}
                Err(e) => panic!("{what}: {e:?}"),
            }
        }
    }
}

/// A status-less read whose alpha the function never uses needs no `OutOfBoundsAlpha`
/// specialization; the same read with the whole texel used needs it. At least one profile must
/// ask for it, or the pair proves nothing: the runtime's floor without `robustImageAccess2` (a
/// device below that floor, whose reads the shader checks) does.
#[test]
fn a_status_less_reads_alpha_is_observed_only_where_the_function_uses_it() {
    let Some(bc) = fixture() else { return };
    let fact = RuntimeFact::OutOfBoundsAlpha { slot: RAW };
    let mut asked = 0;
    let mut profiles = committed_profiles().expect("profiles");
    let checked = crate::pipeline::testing::profile_without(
        "runtime-floor",
        &[crate::device::requirements::VkFeature::RobustImageAccess2],
        Default::default(),
    )
    .expect("profile");
    profiles.push((
        "runtime-floor without robustImageAccess2".to_string(),
        checked,
    ));
    for (stem, p) in &profiles {
        for (entry, uses_alpha) in [("a21_blit_fs", false), ("a21_alpha_fs", true)] {
            let (plan, _) =
                typed_module(&bc, entry, p).unwrap_or_else(|e| panic!("{entry}: {e:?}"));
            let has = plan.facts.contains(&fact);
            assert!(
                !has || uses_alpha,
                "{stem}: {entry} asks for the out-of-bounds alpha of a texel lane it never uses"
            );
            if has {
                asked += 1;
            }
        }
    }
    assert!(
        asked > 0,
        "no profile checks this read in the shader, so the pair proves nothing"
    );
}
