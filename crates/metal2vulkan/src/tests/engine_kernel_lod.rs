//! LOD queries outside a fragment function over the owned fixture `engine/rfs_kernel_lod`.
//!
//! Metal answers `calculate_unclamped_lod` there with 0 and `calculate_clamped_lod` with the
//! sampler's minimum level bounded by the texture's last level. A runtime sampler's minimum is its
//! slot's `SpecKey::SamplerMinLevel` specialization; a constexpr sampler's is a constant, in
//! sixty-fourths of its `half` clamp. No query asks the device (`OpImageQueryLod`). A depth
//! texture's query was not measured and is refused.
//!
//! The values are checked by the fixture's authored cases on Metal; here the structure. `llvm-as`
//! assembles the fixture; without it the tests skip visibly.

use crate::device::profile::DeviceProfile;
use crate::device::requirements::VkFeature;
use crate::device::DevicePolicy;
use crate::error::{Capability, Error, Feature, Needs};
use crate::library::FunctionArtifact;
use crate::pipeline::sym::{SlotClass, SlotRef, SpecKey, Stage};
use crate::pipeline::testing::{
    compile, engine_dir, link_and_validate, llvm_assemble, profile, AssembleError,
};
use crate::spirv::testing::committed_profiles;

/// The kernels, which compile on every profile.
const KERNELS: [&str; 11] = [
    "lod_2d",
    "lod_2d_array",
    "lod_3d",
    "lod_cube",
    "lod_2d_linear_clamp",
    "lod_2d_nearest_clamp",
    "lod_2d_unmipped_clamp",
    "lod_2d_linear_open",
    "lod_2d_linear_past",
    "lod_3d_nearest_tie",
    "lod_cube_linear_clamp",
];

fn fixture() -> Option<Vec<u8>> {
    match llvm_assemble(&engine_dir().join("rfs_kernel_lod.ll")) {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected rfs_kernel_lod: {why}"),
    }
}

fn m3() -> DeviceProfile {
    profile("moltenvk-m3", DevicePolicy::RESIDENT_TABLES).expect("profile")
}

/// The instructions of a SPIR-V module: opcode and word offset.
fn insts(words: &[u32]) -> Vec<(u16, usize)> {
    let mut out = Vec::new();
    let mut i = 5;
    while i < words.len() {
        let n = (words[i] >> 16) as usize;
        out.push(((words[i] & 0xFFFF) as u16, i));
        i += n.max(1);
    }
    out
}

fn count(a: &FunctionArtifact, op: u16) -> usize {
    insts(&a.module.words)
        .iter()
        .filter(|(o, _)| *o == op)
        .count()
}

/// Whether the module declares a 32-bit `OpConstant` of bits `bits`.
fn has_constant(a: &FunctionArtifact, bits: u32) -> bool {
    let w = &a.module.words;
    insts(w)
        .iter()
        .any(|(o, at)| *o == OP_CONSTANT && w[*at] >> 16 == 4 && w[at + 3] == bits)
}

fn uses(a: &FunctionArtifact, name: &str) -> u32 {
    a.strategies.entries.get(name).map_or(0, |u| u.uses)
}

fn spec_keys(a: &FunctionArtifact) -> Vec<SpecKey> {
    a.interface.read_keys()
}

const OP_CONSTANT: u16 = 43;
const OP_IMAGE_QUERY_LOD: u16 = 105;
const OP_IMAGE_QUERY_LEVELS: u16 = 106;

/// Every kernel compiles and passes L3 on every committed profile, and the vertex function where
/// the profile lets a vertex function store (it writes its levels to a buffer; elsewhere it is
/// refused for that, not for its query); the depth texture's query is refused, typed, on every
/// profile.
#[test]
fn every_query_compiles_and_a_depth_query_is_refused_on_every_profile() {
    let Some(bc) = fixture() else {
        return;
    };
    for (stem, p) in committed_profiles().expect("profiles") {
        for entry in KERNELS {
            let a = compile(&bc, entry, &p).unwrap_or_else(|e| panic!("{entry} on {stem}: {e}"));
            link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{entry} on {stem}: L3: {e}"));
            assert_eq!(count(&a, OP_IMAGE_QUERY_LOD), 0, "{entry} on {stem}");
        }
        match compile(&bc, "lod_vertex", &p) {
            Ok(a) => {
                link_and_validate(&a, &p)
                    .unwrap_or_else(|e| panic!("lod_vertex on {stem}: L3: {e}"));
                assert_eq!(count(&a, OP_IMAGE_QUERY_LOD), 0, "lod_vertex on {stem}");
            }
            Err(Error::Unsupported(u)) => assert_eq!(
                (u.feature, u.needs),
                (
                    Feature::StageStoresAndAtomics,
                    Needs::AnyOf(vec![vec![Capability::Feature(
                        VkFeature::VertexPipelineStoresAndAtomics
                    )]])
                ),
                "lod_vertex on {stem}"
            ),
            Err(e) => panic!("lod_vertex on {stem}: {e}"),
        }
        match compile(&bc, "lod_depth", &p) {
            Err(Error::Unsupported(u)) => assert_eq!(
                (u.feature, u.needs),
                (Feature::ImplicitLodOutsideFragment, Needs::NotImplemented),
                "{stem}"
            ),
            other => panic!("lod_depth on {stem}: {:?}", other.map(|_| ())),
        }
    }
}

/// A runtime sampler's clamped query reads its slot's `SamplerMinLevel` specialization against
/// the texture's level count, in kernels over every measured texture type and in a vertex
/// function; it reads no `SamplerState` word.
#[test]
fn a_runtime_samplers_query_reads_its_minimum_level_specialization() {
    let Some(bc) = fixture() else {
        return;
    };
    let p = m3();
    for (entry, stage) in [
        ("lod_2d", Stage::Kernel),
        ("lod_2d_array", Stage::Kernel),
        ("lod_3d", Stage::Kernel),
        ("lod_cube", Stage::Kernel),
        ("lod_vertex", Stage::Vertex),
    ] {
        let a = compile(&bc, entry, &p).unwrap_or_else(|e| panic!("{entry}: {e}"));
        let slot = SlotRef {
            stage,
            class: SlotClass::Sampler,
            index: 0,
        };
        let keys = spec_keys(&a);
        assert!(
            keys.contains(&SpecKey::SamplerMinLevel(slot)),
            "{entry}: {keys:?}"
        );
        assert!(
            !keys
                .iter()
                .any(|k| matches!(k, SpecKey::SamplerState { .. })),
            "{entry}: {keys:?}"
        );
        assert_eq!(uses(&a, "sampler.query_level"), 2, "{entry}");
        assert_eq!(count(&a, OP_IMAGE_QUERY_LEVELS), 1, "{entry}");
    }
}

/// A constexpr sampler's clamped query is its minimum level as a constant, in sixty-fourths of
/// its `half` clamp (0.3 reads 19/64; 1.1 reads 70/64; a nearest 2.7 reads level 3; a nearest
/// 1.5 level 1; 10 is bounded by the last level on the device), with no specialization; without
/// a mip filter or a clamp it is 0, and no level count is queried.
#[test]
fn a_constexpr_samplers_query_is_its_minimum_level() {
    let Some(bc) = fixture() else {
        return;
    };
    let p = m3();
    for (entry, level) in [
        ("lod_2d_linear_clamp", Some(19.0 / 64.0)),
        ("lod_2d_nearest_clamp", Some(3.0)),
        ("lod_2d_linear_past", Some(10.0)),
        ("lod_3d_nearest_tie", Some(1.0)),
        ("lod_cube_linear_clamp", Some(70.0 / 64.0)),
        ("lod_2d_unmipped_clamp", None),
        ("lod_2d_linear_open", None),
    ] {
        let a = compile(&bc, entry, &p).unwrap_or_else(|e| panic!("{entry}: {e}"));
        assert!(
            !spec_keys(&a)
                .iter()
                .any(|k| matches!(k, SpecKey::SamplerMinLevel(_))),
            "{entry}"
        );
        match level {
            Some(l) => {
                let bits = f32::to_bits(l);
                assert!(has_constant(&a, bits), "{entry}: no constant {l}");
                assert_eq!(count(&a, OP_IMAGE_QUERY_LEVELS), 1, "{entry}");
            }
            None => assert_eq!(count(&a, OP_IMAGE_QUERY_LEVELS), 0, "{entry}"),
        }
    }
}
