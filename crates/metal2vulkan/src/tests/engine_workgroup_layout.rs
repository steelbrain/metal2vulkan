//! No module the translator writes declares a capability of `VK_KHR_workgroup_memory_explicit_layout`
//! (`WorkgroupMemoryExplicitLayoutKHR`, `WorkgroupMemoryExplicitLayout8BitAccessKHR`,
//! `WorkgroupMemoryExplicitLayout16BitAccessKHR`) or requires the extension or one of its four
//! features: every `Workgroup` variable it declares is a plain array or scalar (the threadgroup
//! arena, typed shared objects, SIMD-group scratch, mesh and object state), never a `Block`, so no
//! device needs the extension.
//!
//! Each fixture of `validation/fixtures/public/` and its `engine/` directory whose text reaches
//! threadgroup memory, a SIMD-group or quad function, or a mesh or object stage (`addrspace(3)`,
//! `@air.simd_`, `@air.quad_`, `air.mesh`, `air.object`) is compiled, every entry its ABI declares,
//! on every committed profile under the default policy and under the identity domains (the
//! runtime's), each from `Device::from_queries`, and each module compiled
//! is read: it declares none of the three capabilities, its requirements name neither the
//! extension (Vulkan's or SPIR-V's) nor its features, and each part compiles modules that
//! declare `Workgroup` variables. Each part counts the modules whose threadgroup arena is sized at
//! run time (a `Workgroup` array whose length is a specialization constant: the
//! `[[threadgroup(n)]]` regions the binder sizes from the encoder's threadgroup memory lengths)
//! apart from those with only fixed-size `Workgroup` variables. A refusal is the fixture's on that
//! profile and is counted.
//!
//! `llvm-as` assembles the fixtures, `xcrun metal-as` the typed-pointer `.air.ll` ones (skipped
//! visibly off macOS); without `llvm-as` the parts skip visibly.

use std::path::{Path, PathBuf};

use super::engine_word_phases::identity_profiles;
use super::public_fixtures_engine::refusal;
use crate::device::{SpirvExtension, VkExtension, VkFeature};
use crate::pipeline::testing::{
    compile, engine_dir, fixtures_dir, llvm_assemble, metal_assemble, AssembleError,
};
use crate::spirv::testing::committed_profiles;
use crate::spirv_binary::{parse_bytes, Operand};
use crate::spv::{Capability, Op, StorageClass};

/// The capabilities no module may declare.
const EXPLICIT_LAYOUT: [Capability; 3] = [
    Capability::WorkgroupMemoryExplicitLayoutKHR,
    Capability::WorkgroupMemoryExplicitLayout8BitAccessKHR,
    Capability::WorkgroupMemoryExplicitLayout16BitAccessKHR,
];

/// The extension's features, which no module may require.
const EXPLICIT_LAYOUT_FEATURES: [VkFeature; 4] = [
    VkFeature::WorkgroupMemoryExplicitLayout,
    VkFeature::WorkgroupMemoryExplicitLayoutScalarBlockLayout,
    VkFeature::WorkgroupMemoryExplicitLayout8BitAccess,
    VkFeature::WorkgroupMemoryExplicitLayout16BitAccess,
];

/// The text a fixture that reaches `Workgroup` memory names.
const REACHES: [&str; 5] = [
    "addrspace(3)",
    "@air.simd_",
    "@air.quad_",
    "air.mesh",
    "air.object",
];

/// The fixtures, sorted by path, whose text names one of [`REACHES`].
fn fixtures() -> Vec<PathBuf> {
    let mut out = Vec::new();
    for dir in [fixtures_dir(), engine_dir()] {
        for entry in std::fs::read_dir(&dir).unwrap_or_else(|e| panic!("{}: {e}", dir.display())) {
            let path = entry.expect("a fixture directory entry").path();
            if path.extension().is_none_or(|x| x != "ll") {
                continue;
            }
            let text = std::fs::read_to_string(&path).expect("a fixture's text");
            if REACHES.iter().any(|r| text.contains(r)) {
                out.push(path);
            }
        }
    }
    out.sort();
    out
}

/// The `Workgroup` memory a module declares.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Workgroup {
    /// None.
    None,
    /// Only variables of a fixed size.
    Fixed,
    /// An array whose length is a specialization constant: the arena sized at run time.
    RunTime,
}

/// The capabilities module `words` declares, and the `Workgroup` memory it declares.
fn read(words: &[u32]) -> (Vec<u32>, Workgroup) {
    use std::collections::{HashMap, HashSet};
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let module = parse_bytes(&bytes).expect("a module");
    let mut caps = Vec::new();
    let (mut spec, mut arrays, mut pointers) = (HashSet::new(), HashMap::new(), HashMap::new());
    let mut workgroup = Workgroup::None;
    for i in &module.instructions {
        match (i.opcode, i.result_id, i.operands.as_slice()) {
            (Op::Capability, _, &[Operand::Word(cap), ..]) => caps.push(cap),
            (Op::SpecConstant | Op::SpecConstantOperation, Some(id), _) => {
                spec.insert(id);
            }
            (Op::TypeArray, Some(id), &[_, Operand::IdRef(len)]) => {
                arrays.insert(id, len);
            }
            (Op::TypePointer, Some(id), &[Operand::Word(_), Operand::IdRef(pointee)]) => {
                pointers.insert(id, pointee);
            }
            (Op::Variable, _, &[Operand::Word(class), ..])
                if class == StorageClass::Workgroup as u32 =>
            {
                let pointee = i.result_type.and_then(|t| pointers.get(&t));
                let len = pointee.and_then(|p| arrays.get(p));
                if len.is_some_and(|len| spec.contains(len)) {
                    workgroup = Workgroup::RunTime;
                } else if workgroup == Workgroup::None {
                    workgroup = Workgroup::Fixed;
                }
            }
            _ => {}
        }
    }
    (caps, workgroup)
}

/// The bitcode of `path`, or `None` where its assembler is not installed (a visible skip).
fn assemble(path: &Path) -> Option<Vec<u8>> {
    let typed = path.to_str().is_some_and(|s| s.ends_with(".air.ll"));
    let r = if typed {
        metal_assemble(path)
    } else {
        llvm_assemble(path)
    };
    match r {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED {}: {why}", path.display());
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("{}: rejected: {why}", path.display()),
    }
}

/// Part `part` of `of` of [`fixtures`]: each fixture whose position is `part` modulo `of`.
fn check_part(part: usize, of: usize) {
    let mut profiles = committed_profiles().expect("committed profiles");
    let committed = profiles.len();
    profiles.extend(
        identity_profiles()
            .into_iter()
            .map(|(stem, p)| (format!("{stem} (identity)"), p)),
    );
    let (mut modules, mut fixed, mut run_time, mut refused, mut skipped) = (0, 0, 0, 0, 0);
    let paths = fixtures();
    for path in paths.iter().skip(part).step_by(of) {
        let Some(bitcode) = assemble(path) else {
            skipped += 1;
            continue;
        };
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("?");
        let entries = match crate::air::decode(&bitcode, &mut crate::pipeline::testing::meter()) {
            Ok(d) => d
                .abi
                .entries
                .iter()
                .map(|e| e.name.to_string())
                .collect::<Vec<_>>(),
            Err(_) => {
                refused += 1;
                continue;
            }
        };
        for entry in &entries {
            for (stem, p) in &profiles {
                let a = match compile(&bitcode, entry, p) {
                    Ok(a) => a,
                    Err(e) => {
                        refusal(&e).unwrap_or_else(|| {
                            panic!("{name} {entry} on {stem}: translator bug: {e}")
                        });
                        refused += 1;
                        continue;
                    }
                };
                let (caps, wg) = read(&a.module.words);
                modules += 1;
                fixed += usize::from(wg == Workgroup::Fixed);
                run_time += usize::from(wg == Workgroup::RunTime);
                for cap in EXPLICIT_LAYOUT {
                    assert!(
                        !caps.contains(&(cap as u32)),
                        "{name} {entry} on {stem} declares capability {}",
                        cap as u32
                    );
                }
                let r = &a.module.requirements;
                for feature in EXPLICIT_LAYOUT_FEATURES {
                    assert!(
                        !r.features.contains(&feature),
                        "{name} {entry} on {stem} requires {feature:?}"
                    );
                }
                assert!(
                    !r.extensions
                        .contains(&VkExtension::KhrWorkgroupMemoryExplicitLayout)
                        && !r
                            .spirv_extensions
                            .contains(&SpirvExtension::KhrWorkgroupMemoryExplicitLayout),
                    "{name} {entry} on {stem} requires the explicit layout's extension"
                );
            }
        }
    }
    println!(
        "part {} of {of}: {} fixtures ({skipped} skipped), {} profiles ({committed} committed, the \
         rest identity): {modules} modules, {run_time} with a threadgroup arena sized at run time \
         and {fixed} more with fixed-size Workgroup variables, none declaring the explicit \
         layout's capabilities or requiring its extension or features; {refused} refused",
        part + 1,
        paths.iter().skip(part).step_by(of).count(),
        profiles.len(),
    );
    if skipped < paths.iter().skip(part).step_by(of).count() {
        assert!(
            run_time + fixed > 0,
            "part {}: no module declared a Workgroup variable",
            part + 1
        );
    }
}

macro_rules! parts {
    ($of:expr; $($test:ident = $part:expr),* $(,)?) => {
        $(
            #[test]
            fn $test() {
                check_part($part, $of);
            }
        )*
    };
}

parts!(8; part_1_of_8 = 0, part_2_of_8 = 1, part_3_of_8 = 2, part_4_of_8 = 3, part_5_of_8 = 4,
    part_6_of_8 = 5, part_7_of_8 = 6, part_8_of_8 = 7);
