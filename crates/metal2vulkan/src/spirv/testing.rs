//! Shared SPIR-V test helpers: for the crate's tests, `spirv_val` for an explicit target
//! environment and the profile-driven `assert_valid` every test that produces SPIR-V goes through
//! (`G-val`), on the plan the profile gives (`DeviceProfile::spirv_validation`: its target
//! environments and the layout relaxations it enables).
//!
//! The product runs no external tool: the `spirv-val` and `spirv-dis` runs are the dev-tools
//! crate's (`metal2vulkan_devtools::spirv`), so the helpers that run them are the crate's tests'
//! alone, and validation runs its L3 through the same crate on the same public plan. They are test
//! and validation helpers, never on a product path.

pub mod barriers;
pub mod forwarding;

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock, PoisonError};

#[cfg(test)]
pub use metal2vulkan_devtools::spirv::{spirv_dis, L3Failure};

#[cfg(test)]
use crate::budget::{Budget, Meter};
use crate::device::profile::DeviceProfile;
use crate::device::requirements::{ApiVersion, Requirements, VkFeature};
use crate::device::{Device, DeviceDescription, DevicePolicy};
use crate::pipeline::sym::Symbol;
#[cfg(test)]
use crate::spirv::ir::TypedModule;
use crate::spirv::serialize::RelocatableModule;
#[cfg(test)]
use crate::spirv::serialize::{relocate, serialize_for};

/// A `spirv-val --target-env` value. Every profile emits SPIR-V 1.5, which both accept.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TargetEnv {
    Vulkan1_2,
    Vulkan1_3,
    Vulkan1_4,
}

impl TargetEnv {
    /// The `--target-env` argument.
    pub fn arg(self) -> &'static str {
        match self {
            TargetEnv::Vulkan1_2 => "vulkan1.2",
            TargetEnv::Vulkan1_3 => "vulkan1.3",
            TargetEnv::Vulkan1_4 => "vulkan1.4",
        }
    }

    /// The environment of a device's API version (the floor is 1.2; versions above the newest
    /// environment `spirv-val` knows use the newest).
    pub fn for_api(v: ApiVersion) -> TargetEnv {
        if v >= (ApiVersion { major: 1, minor: 4 }) {
            TargetEnv::Vulkan1_4
        } else if v >= ApiVersion::V1_3 {
            TargetEnv::Vulkan1_3
        } else {
            TargetEnv::Vulkan1_2
        }
    }

    /// The environments L3 validates a module for `profile` under: Vulkan 1.2 and the profile's own
    /// API environment.
    pub fn for_profile(profile: &DeviceProfile) -> Vec<TargetEnv> {
        let own = TargetEnv::for_api(profile.fields().api_version);
        if own == TargetEnv::Vulkan1_2 {
            vec![own]
        } else {
            vec![TargetEnv::Vulkan1_2, own]
        }
    }
}

/// `spirv-val` options that relax a check because a device feature is enabled.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct ValOptions {
    /// `--scalar-block-layout` (`scalarBlockLayout`).
    pub scalar_block_layout: bool,
    /// `--uniform-buffer-standard-layout` (`uniformBufferStandardLayout`).
    pub uniform_buffer_standard_layout: bool,
    /// `--allow-localsizeid` (`maintenance4`, needed for `LocalSizeId` under Vulkan 1.2).
    pub allow_localsizeid: bool,
    /// `--allow-offset-texture-operand` (`maintenance8`, needed for `Offset` outside a gather,
    /// VUID-RuntimeSpirv-Offset-10213).
    pub allow_offset_texture_operand: bool,
}

impl ValOptions {
    /// The options a module with these requirements needs: exactly the relaxations its
    /// required features enable. (L3 instead takes them from the profile,
    /// `DeviceProfile::spirv_validation`.)
    pub fn for_requirements(r: &Requirements) -> ValOptions {
        ValOptions::for_features(|f| r.features.contains(&f))
    }

    fn for_features(has: impl Fn(VkFeature) -> bool) -> ValOptions {
        ValOptions {
            scalar_block_layout: has(VkFeature::ScalarBlockLayout),
            uniform_buffer_standard_layout: has(VkFeature::UniformBufferStandardLayout),
            allow_localsizeid: has(VkFeature::Maintenance4),
            allow_offset_texture_operand: has(VkFeature::Maintenance8),
        }
    }

    #[cfg(test)]
    fn args(self) -> Vec<&'static str> {
        let mut out = Vec::new();
        for (on, arg) in [
            (self.scalar_block_layout, "--scalar-block-layout"),
            (
                self.uniform_buffer_standard_layout,
                "--uniform-buffer-standard-layout",
            ),
            (self.allow_localsizeid, "--allow-localsizeid"),
            (
                self.allow_offset_texture_operand,
                "--allow-offset-texture-operand",
            ),
        ] {
            if on {
                out.push(arg);
            }
        }
        out
    }
}

/// Run `spirv-val --target-env <env>` on `words` with no relaxations.
#[cfg(test)]
pub fn spirv_val(words: &[u32], env: TargetEnv) -> Result<(), String> {
    spirv_val_with(words, env, ValOptions::default())
}

/// Run `spirv-val --target-env <env>` with `opts` on `words`. The error is the validator's
/// diagnostic (or the launch failure).
#[cfg(test)]
pub fn spirv_val_with(words: &[u32], env: TargetEnv, opts: ValOptions) -> Result<(), String> {
    metal2vulkan_devtools::spirv::spirv_val(words, env.arg(), &opts.args())
        .map_err(L3Failure::into_text)
}

/// A link table numbering every symbol of `r` densely per kind, descriptor set 0 for every
/// resource and component 0 for every packed varying. Tests use it where no pipeline link
/// exists; it assigns one location per varying.
pub fn link_dense(r: &RelocatableModule) -> BTreeMap<Symbol, u32> {
    let mut table = BTreeMap::new();
    let (mut binding, mut location, mut spec) = (0u32, 0u32, 0u32);
    let bump = |n: &mut u32| {
        let v = *n;
        *n = n.saturating_add(1);
        v
    };
    for (sym, _) in &r.relocs {
        if table.contains_key(sym) {
            continue;
        }
        let n = match sym {
            Symbol::DescriptorSet(_) | Symbol::Component(_) => 0,
            Symbol::Binding(_) => bump(&mut binding),
            Symbol::Location(_) => bump(&mut location),
            Symbol::SpecId(_) => bump(&mut spec),
        };
        table.insert(sym.clone(), n);
    }
    table
}

/// L3 for a finished module on `profile`: the owned verifier (every rule, requirements ⊆ profile),
/// serialization for the profile (`serialize_for`), a dense link ([`link_dense`]), and `spirv-val`
/// under Vulkan 1.2 and the profile's environment with the profile's layout relaxations. Returns
/// the linked words, or the first failure with a disassembly.
#[cfg(test)]
pub fn check_valid(m: TypedModule, profile: &DeviceProfile) -> Result<Vec<u32>, String> {
    let mut meter = Meter::new(&Budget::default());
    crate::spirv::verify::check(&m, profile, &mut meter)
        .map_err(|e| format!("owned verifier: {e}"))?;
    let r = serialize_for(m, profile).map_err(|e| format!("serialize: {e}"))?;
    let words = relocate(&r, &link_dense(&r)).map_err(|e| format!("relocate: {e}"))?;
    check_words_valid(&words, &r.requirements, profile)?;
    Ok(words)
}

/// [`check_valid`] for already linked words: requirements ⊆ profile, then `spirv-val` under
/// Vulkan 1.2 and the profile's environment with the profile's layout relaxations.
#[cfg(test)]
pub fn check_words_valid(
    words: &[u32],
    requirements: &Requirements,
    profile: &DeviceProfile,
) -> Result<(), String> {
    check_words_l3(words, requirements, profile).map_err(|failure| failure.to_string())
}

/// [`check_words_valid`] with the failure typed, so a missing or broken `spirv-val` is not
/// reported as an invalid module.
#[cfg(test)]
pub fn check_words_l3(
    words: &[u32],
    requirements: &Requirements,
    profile: &DeviceProfile,
) -> Result<(), L3Failure> {
    profile
        .check(requirements)
        .map_err(|c| L3Failure::Requirements(format!("requirements exceed the profile: {c:?}")))?;
    let plan = profile.spirv_validation();
    metal2vulkan_devtools::spirv::check_l3(
        words,
        &plan.target_envs,
        &plan.relaxations,
        &plan.relaxations.join(" "),
        &format!("{requirements:?}"),
    )
}

/// [`check_valid`], panicking with the failure. Every test that produces SPIR-V for a profile
/// uses it (`G-val`).
#[cfg(test)]
#[track_caller]
pub fn assert_valid(m: TypedModule, profile: &DeviceProfile) -> Vec<u32> {
    match check_valid(m, profile) {
        Ok(words) => words,
        Err(e) => panic!("L3 rejected the module: {e}"),
    }
}

/// Every committed device profile (`validation/profiles/*.json`, read-only), derived through
/// `Device::new` with the default policy, sorted by file stem.
pub fn committed_profiles() -> Result<Vec<(String, DeviceProfile)>, String> {
    committed_profiles_under(DevicePolicy::default())
}

/// [`committed_profiles`] under `policy`.
pub fn committed_profiles_with(
    policy: DevicePolicy,
) -> Result<Vec<(String, DeviceProfile)>, String> {
    committed_profiles_under(policy)
}

/// Every committed device profile, derived through `Device::new` with `policy`, sorted by stem:
/// derived once per policy per test process, since deriving a device computes its fingerprint
/// and a test may ask for the profiles once per case it checks (a sixth of such a test's time).
fn committed_profiles_under(policy: DevicePolicy) -> Result<Vec<(String, DeviceProfile)>, String> {
    type Derived = Vec<(DevicePolicy, Vec<(String, DeviceProfile)>)>;
    static DERIVED: Mutex<Derived> = Mutex::new(Vec::new());
    let lock = || DERIVED.lock().unwrap_or_else(PoisonError::into_inner);
    if let Some((_, profiles)) = lock().iter().find(|(p, _)| *p == policy) {
        return Ok(profiles.clone());
    }
    let mut out = Vec::new();
    for (stem, desc) in committed_descriptions()? {
        let dev = Device::from_queries(desc.clone(), policy).map_err(|e| format!("{stem}: {e}"))?;
        out.push((stem.clone(), dev.profile().clone()));
    }
    lock().push((policy, out.clone()));
    Ok(out)
}

/// Every committed device description (`validation/profiles/*.json`, read-only) with its file
/// stem, sorted by stem, read and parsed once per test process: a test may derive the profiles
/// once per case it checks, and parsing the files each time was a fifth of such a test's time.
pub(crate) fn committed_descriptions() -> Result<&'static [(String, DeviceDescription)], String> {
    static DESCRIPTIONS: OnceLock<Result<Vec<(String, DeviceDescription)>, String>> =
        OnceLock::new();
    DESCRIPTIONS
        .get_or_init(|| {
            let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../metal2vulkan-validation/profiles");
            let entries = std::fs::read_dir(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
            let mut out = Vec::new();
            for e in entries {
                let path = e.map_err(|e| format!("{}: {e}", dir.display()))?.path();
                if path.extension().is_none_or(|x| x != "json") {
                    continue;
                }
                let stem = path
                    .file_stem()
                    .map(|s| s.to_string_lossy().into_owned())
                    .unwrap_or_default();
                let text = std::fs::read_to_string(&path)
                    .map_err(|e| format!("{}: {e}", path.display()))?;
                let desc = DeviceDescription::from_khronos_json(&text, None)
                    .map_err(|e| format!("{stem}: {e}"))?;
                out.push((stem, desc));
            }
            out.sort_by(|a, b| a.0.cmp(&b.0));
            Ok(out)
        })
        .as_deref()
        .map_err(String::clone)
}

/// The committed MoltenVK captures, MoltenVK 1.4.2 on the Apple M3 Max under macOS 27.0.1: a test
/// that singles out MoltenVK's answers among the committed profiles expects them of each.
pub const MOLTENVK_CAPTURES: [&str; 1] = ["moltenvk-m3"];

/// Whether the committed profile `stem` is one of [`MOLTENVK_CAPTURES`].
pub fn is_moltenvk_capture(stem: &str) -> bool {
    MOLTENVK_CAPTURES.contains(&stem)
}

/// One long check as several tests, so nextest runs them side by side: a suite's wall-clock is
/// at least its longest test's, and an exhaustive table run as one test was the longest.
///
/// `sharded_tests!(group = run(shard_0 => 0, shard_1 => 1, ...))` declares module `group` with
/// test `shard_k` calling `run(k, n)` (`run` in the module that invokes this), `n` the number of
/// shards listed; `run` checks the cases whose index is `k` modulo `n`. The shards must be
/// numbered `0, 1, ..., n - 1` in order, or the module does not compile: every case runs in
/// exactly one shard.
macro_rules! sharded_tests {
    ($group:ident = $run:ident($($test:ident => $k:literal),+ $(,)?)) => {
        mod $group {
            /// The number of shards.
            const SHARDS: usize = [$($k),+].len();
            // The shards are numbered 0, 1, ..., SHARDS - 1, in order.
            const _: () = {
                let shards: [usize; SHARDS] = [$($k),+];
                let mut i = 0;
                while i < SHARDS {
                    assert!(shards[i] == i, "shards are numbered 0, 1, ... in order");
                    i += 1;
                }
            };
            $(
                #[test]
                fn $test() {
                    super::$run($k, SHARDS);
                }
            )+
        }
    };
}
pub(crate) use sharded_tests;

/// The constants of SPIR-V `words` by id: `(width, value)` of each 32- or 64-bit integer
/// `OpConstant`.
pub fn int_constants(words: &[u32]) -> std::collections::BTreeMap<u32, (u32, u64)> {
    let mut widths = std::collections::BTreeMap::new();
    let mut out = std::collections::BTreeMap::new();
    let mut i = 5;
    while let Some(&w) = words.get(i) {
        let (len, op) = ((w >> 16) as usize, w & 0xFFFF);
        let at = |k: usize| words.get(i + k).copied().unwrap_or(0);
        match op {
            // OpTypeInt result, width.
            21 => {
                widths.insert(at(1), at(2));
            }
            // OpConstant type, result, value words.
            43 => match widths.get(&at(1)) {
                Some(&32) => {
                    out.insert(at(2), (32, u64::from(at(3))));
                }
                Some(&64) => {
                    out.insert(at(2), (64, u64::from(at(3)) | (u64::from(at(4)) << 32)));
                }
                _ => {}
            },
            _ => {}
        }
        if len == 0 {
            break;
        }
        i += len;
    }
    out
}

/// How many `OpUDiv`s of SPIR-V `words` divide the all-ones integer of their width (`2^32 - 1`
/// or `2^64 - 1`), the form of an overflow test (`b <= (2^32 - 1) / a`) NVIDIA's compiler
/// refuses in some modules (`bind::replay::checked_mul`).
pub fn all_ones_divisions(words: &[u32]) -> usize {
    const UDIV: u32 = 134;
    let consts = int_constants(words);
    let all_ones = |id: u32| match consts.get(&id) {
        Some(&(32, v)) => v == u64::from(u32::MAX),
        Some(&(64, v)) => v == u64::MAX,
        _ => false,
    };
    let mut n = 0;
    let mut i = 5;
    while let Some(&w) = words.get(i) {
        let (len, op) = ((w >> 16) as usize, w & 0xFFFF);
        // OpUDiv type, result, dividend, divisor.
        if op == UDIV && words.get(i + 3).is_some_and(|&id| all_ones(id)) {
            n += 1;
        }
        if len == 0 {
            break;
        }
        i += len;
    }
    n
}
