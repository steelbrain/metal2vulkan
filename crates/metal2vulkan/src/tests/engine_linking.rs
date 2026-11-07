//! Static visible-function linking end to end.
//!
//! The owned fixtures `engine/linked_visible_caller` (a kernel calling `linked_scale` and
//! `linked_blend` through visible-function references, and defining the visible
//! `linked_offset`) and `engine/linked_visible_functions` (defining `linked_scale`, which
//! references `linked_offset`, and `linked_blend`) are what Apple's compiler emits for
//! `[[visible]]` functions. Through the public API (`Library::from_air`, `FunctionDescriptor`
//! with `LinkedFunctions`, `Device::compile_compute_pipeline`) the caller compiles once its
//! references resolve: a reference nothing supplies is `NeedsInput(LinkedFunction)`, a function
//! of another signature links with the arguments by position, a visible function is not a stage,
//! and the linked functions are part of the compile key. The linked kernel validates on every committed
//! profile and computes Metal's values on the CPU interpreter. No GPU is used.
//!
//! `llvm-as` assembles the fixtures; a missing tool skips visibly.

use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::air::ir::print::print_inst;
use crate::air::ir::InstId;
use crate::budget::{Budget, SPIRV_WORDS};
use crate::device::{Device, DeviceDescription, DevicePolicy};
use crate::error::{DescriptorError, Error, Feature, LowerFamily, RuntimeInput};
use crate::library::{FunctionConstantValue, FunctionConstantValues, FunctionStage, Library};
use crate::lower::select::LowerInputs;
use crate::metal::DataType;
use crate::pipeline::compile::{compile_bitcode_linked, EntryRef, LinkedSource};
use crate::pipeline::layout::BindingSource;
use crate::pipeline::link::{link, LinkInputs};
use crate::pipeline::sym::{ResourceKind, ResourceSlot, SamplerRef, Stage};
use crate::pipeline::testing::{
    engine_dir, link_and_validate, llvm_assemble, meter, profile, specialized_entry,
    typed_module_linked, AssembleError,
};
use crate::pipeline::{
    ComputePipelineDescriptor, FunctionDescriptor, LinkedFunction, LinkedFunctions, StageSource,
};
use crate::spirv::interp::{run, Buffer, Config, Resources};
use crate::spirv::testing::committed_profiles;

const CALLER: &str = "linked_visible_caller";
/// The caller module's second kernel, sampling through the linked `linked_sample`.
const SAMPLING: &str = "linked_visible_sampling";
const FUNCTIONS: &str = "linked_visible_functions";

fn assemble_ll(name: &str, ll: &Path) -> Option<Vec<u8>> {
    match llvm_assemble(ll) {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected {name}: {why}"),
    }
}

/// The fixture `stem` assembled.
fn fixture(stem: &str) -> Option<Vec<u8>> {
    assemble_ll(stem, &engine_dir().join(format!("{stem}.ll")))
}

/// The fixture `stem` with `edit` applied to its text, assembled.
fn edited(stem: &str, edit: impl FnOnce(String) -> String) -> Option<Vec<u8>> {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let text = std::fs::read_to_string(engine_dir().join(format!("{stem}.ll"))).expect("read");
    let dir = std::env::temp_dir().join(format!(
        "m2v-linking-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&dir).expect("scratch");
    let ll = dir.join("m.ll");
    std::fs::write(&ll, edit(text)).expect("write");
    let r = assemble_ll(stem, &ll);
    let _ = std::fs::remove_dir_all(&dir);
    r
}

/// Both fixtures assembled.
fn fixtures() -> Option<(Vec<u8>, Vec<u8>)> {
    Some((fixture(CALLER)?, fixture(FUNCTIONS)?))
}

fn device(stem: &str) -> Device {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles")
        .join(format!("{stem}.json"));
    let text = std::fs::read_to_string(path).expect("profile");
    let desc = DeviceDescription::from_khronos_json(&text, None).expect("description");
    Device::new(desc, DevicePolicy::default()).expect("device")
}

fn needs(symbol: &str) -> Error {
    Error::NeedsInput(RuntimeInput::LinkedFunction {
        symbol: crate::error::DiagName::new(symbol),
    })
}

/// The sources linking the caller completely: `linked_scale` and `linked_blend` of the
/// functions module, `linked_offset` of the caller's own module.
fn complete<'a>(
    caller: &'a [u8],
    functions: &'a [u8],
    none: &'a FunctionConstantValues,
) -> [LinkedSource<'a>; 3] {
    [
        LinkedSource::new("linked_scale", functions, none),
        LinkedSource::new("linked_blend", functions, none),
        LinkedSource::new("linked_offset", caller, none),
    ]
}

fn compile_linked(
    caller: &[u8],
    linked: &[LinkedSource<'_>],
    stem: &str,
) -> Result<std::sync::Arc<crate::library::FunctionArtifact>, Error> {
    let p = profile(stem, DevicePolicy::default()).expect("profile");
    compile_bitcode_linked(
        caller,
        EntryRef::Name(CALLER),
        (&FunctionConstantValues::defaults(), linked),
        &p,
        &LowerInputs::new(),
        &mut meter(),
        SPIRV_WORDS,
    )
}

/// A bitcode library lists each `[[visible]]` function a module defines as a visible function,
/// beside the module's stages.
#[test]
fn a_library_lists_visible_functions() {
    let Some((caller, functions)) = fixtures() else {
        return;
    };
    let listed = |bc: &[u8]| {
        let lib = Library::from_air(bc).expect("library");
        let mut fs = lib
            .functions()
            .map(|i| (i.name.to_owned(), i.stage))
            .collect::<Vec<_>>();
        fs.sort_by(|a, b| a.0.cmp(&b.0));
        fs
    };
    assert_eq!(
        listed(&caller),
        [
            ("linked_offset".to_owned(), FunctionStage::Visible),
            (CALLER.to_owned(), FunctionStage::Kernel),
            (SAMPLING.to_owned(), FunctionStage::Kernel),
        ]
    );
    assert_eq!(
        listed(&functions),
        [
            ("linked_blend".to_owned(), FunctionStage::Visible),
            ("linked_sample".to_owned(), FunctionStage::Visible),
            ("linked_scale".to_owned(), FunctionStage::Visible),
            (FUNCTIONS.to_owned(), FunctionStage::Kernel),
        ]
    );
}

/// Without linked functions the first reference a surviving call makes is a runtime input the
/// pipeline needs; a linked function's own references need their functions too.
#[test]
fn an_unresolved_reference_needs_its_linked_function() {
    let Some((caller, functions)) = fixtures() else {
        return;
    };
    let none = FunctionConstantValues::defaults();
    assert_eq!(
        compile_linked(&caller, &[], "moltenvk-m3").map(|_| ()),
        Err(needs("linked_scale"))
    );
    let [scale, blend, offset] = complete(&caller, &functions, &none);
    assert_eq!(
        compile_linked(&caller, &[scale, blend], "moltenvk-m3").map(|_| ()),
        Err(needs("linked_offset"))
    );
    assert_eq!(
        compile_linked(&caller, &[scale, offset], "moltenvk-m3").map(|_| ()),
        Err(needs("linked_blend"))
    );
}

/// With every reference resolved the kernel compiles, links, and passes L3 on every committed
/// profile.
#[test]
fn a_linked_kernel_validates_on_every_profile() {
    let Some((caller, functions)) = fixtures() else {
        return;
    };
    let none = FunctionConstantValues::defaults();
    let linked = complete(&caller, &functions, &none);
    for (stem, p) in committed_profiles().expect("profiles") {
        let a = compile_bitcode_linked(
            &caller,
            EntryRef::Name(CALLER),
            (&none, &linked),
            &p,
            &LowerInputs::new(),
            &mut meter(),
            SPIRV_WORDS,
        )
        .unwrap_or_else(|e| panic!("{stem}: {e}"));
        link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{stem}: L3: {e}"));
    }
}

/// The linked kernel computes what Metal does: `words[i] = 3i + (i & 7)` through `linked_scale`
/// and `linked_offset`, `floats[i] = i * 0.5 + 1` through `linked_blend`.
#[test]
fn a_linked_kernel_computes_its_linked_functions() {
    let Some((caller, functions)) = fixtures() else {
        return;
    };
    let none = FunctionConstantValues::defaults();
    let linked = complete(&caller, &functions, &none);
    let p = profile("moltenvk-m3", DevicePolicy::default()).expect("profile");
    let (_, m) = typed_module_linked(&caller, CALLER, &p, &linked).expect("module");
    const N: u32 = 32;
    let buffer = |index| Buffer {
        slot: Some(ResourceSlot {
            stage: Stage::Kernel,
            kind: ResourceKind::Buffer,
            index,
        }),
        address: None,
        bytes: vec![0; N as usize * 4],
    };
    let res = Resources {
        buffers: vec![buffer(0), buffer(1)],
        push: vec![0; 128],
        spec: Default::default(),
    };
    let cfg = Config {
        workgroups: [N, 1, 1],
        ..Config::default()
    };
    let out = run(&m, &cfg, res).unwrap_or_else(|e| panic!("{e:?}"));
    let words = |b: &[u8]| {
        b.as_chunks::<4>()
            .0
            .iter()
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect::<Vec<_>>()
    };
    let expect_words = (0..N).map(|i| 3 * i + (i & 7)).collect::<Vec<_>>();
    let expect_floats = (0..N)
        .map(|i| (i as f32 * 0.5 + 1.0).to_bits())
        .collect::<Vec<_>>();
    assert_eq!(words(&out.buffers[0]), expect_words);
    assert_eq!(words(&out.buffers[1]), expect_floats);
}

/// The words `words[i]` the caller computes over 32 threads with `linked`.
fn caller_words(caller: &[u8], linked: &[LinkedSource<'_>]) -> Vec<u32> {
    let p = profile("moltenvk-m3", DevicePolicy::default()).expect("profile");
    let (_, m) = typed_module_linked(caller, CALLER, &p, linked).expect("module");
    const N: u32 = 32;
    let buffer = |index| Buffer {
        slot: Some(ResourceSlot {
            stage: Stage::Kernel,
            kind: ResourceKind::Buffer,
            index,
        }),
        address: None,
        bytes: vec![0; N as usize * 4],
    };
    let res = Resources {
        buffers: vec![buffer(0), buffer(1)],
        push: vec![0; 128],
        spec: Default::default(),
    };
    let cfg = Config {
        workgroups: [N, 1, 1],
        ..Config::default()
    };
    let out = run(&m, &cfg, res).unwrap_or_else(|e| panic!("{e:?}"));
    out.buffers[0]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// `name` of `bitcode` in the linked list `list`.
fn in_list<'a>(
    name: &'a str,
    bitcode: &'a [u8],
    none: &'a FunctionConstantValues,
    list: crate::pipeline::LinkedList<'static>,
) -> LinkedSource<'a> {
    let mut source = LinkedSource::new(name, bitcode, none);
    source.key.list = list;
    source
}

/// No visible-function reference resolves to a function only `binary_functions` lists, as Metal
/// links them (a probe on an M3 Max under macOS 27.0.1, `binary_function_linking.m`): the caller's
/// `linked_scale` and `linked_scale`'s own `linked_offset` held there alone need their linked
/// function, as Metal's creation fails ("unresolved visible function reference"); beside a
/// `linked_scale` of `functions` (`engine/linked_other_functions`' is `x + 100`), the reference
/// calls that one, whichever module it comes from, as Metal's does.
#[test]
fn a_reference_resolves_to_no_binary_function() {
    use crate::pipeline::LinkedList::{BinaryFunctions, Functions, PrivateFunctions};
    let Some((caller, functions)) = fixtures() else {
        return;
    };
    let Some(other) = fixture("linked_other_functions") else {
        return;
    };
    let none = FunctionConstantValues::defaults();
    let blend = in_list("linked_blend", &functions, &none, Functions);
    let offset = in_list("linked_offset", &caller, &none, PrivateFunctions);
    let scale = |bitcode, list| in_list("linked_scale", bitcode, &none, list);
    let alone = [scale(&functions, BinaryFunctions), blend, offset];
    assert_eq!(
        compile_linked(&caller, &alone, "moltenvk-m3").map(|_| ()),
        Err(needs("linked_scale"))
    );
    let nested = [
        scale(&functions, Functions),
        blend,
        in_list("linked_offset", &caller, &none, BinaryFunctions),
    ];
    assert_eq!(
        compile_linked(&caller, &nested, "moltenvk-m3").map(|_| ()),
        Err(needs("linked_offset"))
    );
    let own: Vec<u32> = (0..32).map(|i| 3 * i + (i & 7)).collect();
    let others: Vec<u32> = (0..32).map(|i| i + 100).collect();
    for (linked, expect) in [
        (
            [scale(&functions, Functions), scale(&other, BinaryFunctions)],
            &own,
        ),
        (
            [scale(&other, BinaryFunctions), scale(&functions, Functions)],
            &own,
        ),
        (
            [scale(&other, Functions), scale(&functions, BinaryFunctions)],
            &others,
        ),
    ] {
        let all = [linked[0], linked[1], blend, offset];
        assert_eq!(&caller_words(&caller, &all), expect);
    }
}

/// Through the public API: `FunctionDescriptor.linked` resolves the references from
/// `functions` and `private_functions` alike, a function listed twice is one function, and
/// `groups` (function-pointer call sites) resolve no reference.
#[test]
fn the_public_api_links_functions_and_private_functions() {
    let Some((caller, functions)) = fixtures() else {
        return;
    };
    let dev = device("moltenvk-m3");
    let caller_lib = Library::from_air(&caller).expect("library");
    let functions_lib = Library::from_air(&functions).expect("library");
    let entry = caller_lib.function(CALLER).expect("entry");
    let scale = functions_lib.function("linked_scale").expect("scale");
    let blend = functions_lib.function("linked_blend").expect("blend");
    let offset = caller_lib.function("linked_offset").expect("offset");
    let compile = |linked: LinkedFunctions| {
        let d = ComputePipelineDescriptor::new(StageSource::Function(
            FunctionDescriptor::new(entry.clone()).with_linked(linked),
        ));
        dev.compile_compute_pipeline(&d, &Budget::standard())
            .map(|_| ())
    };
    assert_eq!(compile(LinkedFunctions::new()), Err(needs("linked_scale")));
    let linked = LinkedFunctions::new()
        .with_function(scale.clone())
        .with_function(blend.clone())
        .with_private_function(offset.clone());
    assert_eq!(compile(linked.clone()), Ok(()));
    assert_eq!(
        compile(linked.with_private_function(scale.clone())),
        Ok(()),
        "the same function listed twice is one function"
    );
    let mut grouped = LinkedFunctions::new();
    grouped.groups.insert(
        "table".to_owned(),
        vec![scale.clone().into(), blend.clone().into(), offset.into()],
    );
    assert_eq!(compile(grouped), Err(needs("linked_scale")));
}

/// A visible function is no pipeline stage: compiling it is `EntryNotFound`, as for a module
/// without that entry.
#[test]
fn a_visible_function_is_not_a_stage() {
    let Some((_, functions)) = fixtures() else {
        return;
    };
    let dev = device("moltenvk-m3");
    let lib = Library::from_air(&functions).expect("library");
    let scale = lib.function("linked_scale").expect("scale");
    let d = ComputePipelineDescriptor::new(StageSource::Function(FunctionDescriptor::new(scale)));
    assert_eq!(
        dev.compile_compute_pipeline(&d, &Budget::standard())
            .map(|_| ()),
        Err(Error::InvalidDescriptor(DescriptorError::EntryNotFound {
            requested: crate::error::DiagName::new("linked_scale"),
            available: 1,
        }))
    );
}

/// The linked functions are part of the compile key: the list each is in, and the constants it
/// was specialized with.
#[test]
fn linked_functions_are_part_of_the_key() {
    let Some((caller, functions)) = fixtures() else {
        return;
    };
    let dev = device("moltenvk-m3");
    let caller_lib = Library::from_air(&caller).expect("library");
    let functions_lib = Library::from_air(&functions).expect("library");
    let entry = caller_lib.function(CALLER).expect("entry");
    let scale = functions_lib.function("linked_scale").expect("scale");
    let key = |linked: LinkedFunctions| {
        dev.function_compile_key(
            &FunctionDescriptor::new(entry.clone()).with_linked(linked),
            &LowerInputs::new(),
        )
    };
    let none = key(LinkedFunctions::new());
    let public = key(LinkedFunctions::new().with_function(scale.clone()));
    let private = key(LinkedFunctions::new().with_private_function(scale.clone()));
    let one = FunctionConstantValues::defaults().with_index(
        0,
        FunctionConstantValue::new(DataType::UInt, 1u32.to_le_bytes()),
    );
    let specialized = key(LinkedFunctions::new()
        .with_function(LinkedFunction::new(scale.clone()).with_constants(one)));
    assert_eq!(none, key(LinkedFunctions::new()), "a key is deterministic");
    assert_eq!(public, key(LinkedFunctions::new().with_function(scale)));
    let keys = [none, public, private, specialized];
    for (i, a) in keys.iter().enumerate() {
        for b in &keys[i + 1..] {
            assert_ne!(a, b);
        }
    }
}

/// A linked function whose type is not the reference's links with the arguments by position, as
/// Metal links it (the probe `linked_function_signatures.m`, the conformance crate's `probes/`):
/// here `linked_scale`, referenced as `uint (uint)`, is `linked_blend`'s
/// `float (float a, float b) = a * b + 1` renamed onto it. Its `a` takes the bits of the
/// argument, its `b`, which the reference does not pass, reads zero, and its result passes back
/// as its bits: every word is 1.0's.
#[test]
fn a_linked_function_of_another_signature_links_by_position() {
    let Some((caller, functions)) = fixtures() else {
        return;
    };
    let Some(renamed) = edited(FUNCTIONS, |text| {
        text.replace("@linked_scale", "@linked_int")
            .replace("!\"linked_scale\"", "!\"linked_int\"")
            .replace("@linked_blend", "@linked_scale")
            .replace("!\"linked_blend\"", "!\"linked_scale\"")
    }) else {
        return;
    };
    let none = FunctionConstantValues::defaults();
    let linked = [
        LinkedSource::new("linked_scale", &renamed, &none),
        LinkedSource::new("linked_blend", &functions, &none),
    ];
    assert_eq!(caller_words(&caller, &linked), vec![1.0f32.to_bits(); 32]);
}

/// A constexpr sampler inside a linked function links with it: the linked module's
/// `!air.sampler_states` names the sampler's global, which is imported with the function. The
/// kernel samples through a constexpr combined binding that the layout bakes as an immutable
/// sampler, as for a constexpr sampler of its own module, and passes L3 on every committed
/// profile.
#[test]
fn a_constexpr_sampler_of_a_linked_function_links_with_it() {
    let Some((caller, functions)) = fixtures() else {
        return;
    };
    let none = FunctionConstantValues::defaults();
    let linked = [LinkedSource::new("linked_sample", &functions, &none)];
    let compile = |p: &crate::device::DeviceProfile, linked: &[LinkedSource<'_>]| {
        compile_bitcode_linked(
            &caller,
            EntryRef::Name(SAMPLING),
            (&none, linked),
            p,
            &LowerInputs::new(),
            &mut meter(),
            SPIRV_WORDS,
        )
    };
    for (stem, p) in committed_profiles().expect("profiles") {
        assert_eq!(
            compile(&p, &[]).map(|_| ()),
            Err(needs("linked_sample")),
            "{stem}"
        );
        let a = compile(&p, &linked).unwrap_or_else(|e| panic!("{stem}: {e}"));
        link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{stem}: L3: {e}"));
        let constexpr = a
            .interface
            .descriptors
            .iter()
            .filter(|d| {
                matches!(d.slot.kind, ResourceKind::CombinedTextureSampler(c)
                    if matches!(c.sampler, SamplerRef::Constexpr(_)))
            })
            .count();
        assert_eq!(constexpr, 1, "{stem}");
        let l = link(&[&a], &LinkInputs::compute(), &p).expect("link");
        let baked = l.layout.sets[0]
            .bindings
            .iter()
            .find(|b| matches!(b.source, BindingSource::CombinedTextureSampler { .. }))
            .and_then(|b| b.immutable_sampler)
            .unwrap_or_else(|| panic!("{stem}: no immutable sampler"));
        assert_eq!(
            baked.address_mode_u,
            crate::vk::SamplerAddressMode::REPEAT,
            "{stem}"
        );
        assert_eq!(baked.min_filter, crate::vk::Filter::LINEAR, "{stem}");
    }
}

/// A linked function whose module does not decode is refused only where a call to it survives: the
/// sampling kernel of the same module, which never calls `linked_scale`, compiles.
#[test]
fn a_linked_module_that_does_not_decode_is_refused_only_at_its_calls() {
    let Some((caller, functions)) = fixtures() else {
        return;
    };
    let none = FunctionConstantValues::defaults();
    let garbage = [0u8; 16];
    let linked = [
        LinkedSource::new("linked_scale", &garbage, &none),
        LinkedSource::new("linked_sample", &functions, &none),
    ];
    let p = profile("moltenvk-m3", DevicePolicy::default()).expect("profile");
    let compile = |entry| {
        compile_bitcode_linked(
            &caller,
            EntryRef::Name(entry),
            (&none, &linked),
            &p,
            &LowerInputs::new(),
            &mut meter(),
            SPIRV_WORDS,
        )
        .map(|_| ())
    };
    assert_eq!(compile(SAMPLING), Ok(()));
    assert!(matches!(compile(CALLER), Err(Error::InvalidInput(_))));
}

/// The entry `specialized_entry` builds is the one a linked compile selects from: a refusal raised
/// inside a linked function, once inlined, locates an instruction of that entry, the refused one.
/// Here `linked_blend`'s result feeds an observed uniformity test, refused as `Lowering(Simd)` at the `air.is_uniform` call.
#[test]
fn a_linked_refusal_locates_an_instruction_of_the_specialized_entry() {
    let Some(caller) = fixture(CALLER) else {
        return;
    };
    let Some(functions) = edited(FUNCTIONS, |text| {
        let blend = "  %4 = fadd fast float %3, 1.000000e+00\n  ret float %4\n}\n";
        assert_eq!(
            text.matches(blend).count(),
            1,
            "the fixture's linked_blend changed"
        );
        text.replace(
            blend,
            "  %4 = fadd fast float %3, 1.000000e+00\n  %b = fptoui float %4 to i32\n  \
             %u = tail call i1 @air.is_uniform.i32(i32 %b)\n  \
             %r = select i1 %u, float %4, float 0.000000e+00\n  ret float %r\n}\n\n\
             declare i1 @air.is_uniform.i32(i32)\n",
        )
    }) else {
        return;
    };
    let none = FunctionConstantValues::defaults();
    let linked = complete(&caller, &functions, &none);
    let p = profile("moltenvk-m3", DevicePolicy::default()).expect("profile");
    let refusal = match compile_linked(&caller, &linked, "moltenvk-m3") {
        Err(Error::Unsupported(u)) => u,
        other => panic!("not refused: {:?}", other.map(|_| ())),
    };
    assert_eq!(
        refusal.feature,
        Feature::Lowering(LowerFamily::Simd),
        "{refusal}"
    );
    let Some((function, _, Some(inst))) = refusal.location.as_ref().and_then(|l| l.in_function())
    else {
        panic!("not located at an instruction: {refusal}");
    };
    let spec = specialized_entry(&caller, CALLER, &p, &none, &linked, &mut meter()).expect("entry");
    assert_eq!(function.to_string(), spec.ep.name.to_string());
    let body = spec
        .module
        .func(spec.func)
        .and_then(|f| f.body.as_ref())
        .expect("a body");
    let id = InstId::new(inst);
    let at = body.insts.get(id).expect("the located instruction");
    let printed = print_inst(&spec.module, id, at);
    assert!(printed.contains("IsUniform"), "{printed}");
    // Without the linked functions the entry is not built: the reference needs its function.
    assert_eq!(
        specialized_entry(&caller, CALLER, &p, &none, &[], &mut meter()).map(|_| ()),
        Err(needs("linked_scale"))
    );
}

/// A pipeline state numbers the constexpr sampler of the one function the sampling kernel links
/// and calls, with no precision note. The module's other kernel, which never reaches
/// `linked_sample`, numbers nothing; linking it too, it numbers that function's sampler, as Metal
/// numbers a sampler of a function of `functions` called or not, with no note: one linked function
/// brings samplers (the case 2, where it was noted and numbered nothing). Through
/// `privateFunctions` it is numbered only where the kernel reaches it, with no note.
#[test]
fn a_linked_function_s_constexpr_sampler_is_numbered() {
    use crate::pipeline::interface::NumberedSampler;
    use crate::pipeline::PrecisionNote;
    let Some((caller, functions)) = fixtures() else {
        return;
    };
    let none = FunctionConstantValues::defaults();
    let p = profile("lavapipe", DevicePolicy::default()).expect("profile");
    let compile = |entry: &str, linked: &[LinkedSource<'_>]| {
        compile_bitcode_linked(
            &caller,
            EntryRef::Name(entry),
            (&none, linked),
            &p,
            &LowerInputs::new(),
            &mut meter(),
            SPIRV_WORDS,
        )
        .expect("compile")
    };
    let state = crate::air::abi::SamplerState::from_words([34901797601020562, 0]).expect("state");
    let sampler = NumberedSampler::Constexpr(crate::lower::sampler::state_words(&state));
    let noted = |a: &crate::library::FunctionArtifact| {
        a.precision.contains(&PrecisionNote::LinkedSamplerNumbering)
    };
    let sample = LinkedSource::new("linked_sample", &functions, &none);
    let a = compile(SAMPLING, &[sample]);
    assert_eq!(a.interface.numbered_samplers, [sampler]);
    assert!(!noted(&a), "{:?}", a.precision);
    let sources = complete(&caller, &functions, &none);
    let b = compile(CALLER, &sources);
    assert_eq!(b.interface.numbered_samplers, []);
    assert!(!noted(&b), "{:?}", b.precision);
    let [scale, blend, offset] = sources;
    let c = compile(CALLER, &[scale, blend, offset, sample]);
    assert_eq!(c.interface.numbered_samplers, [sampler]);
    assert!(!noted(&c), "{:?}", c.precision);
    // Through `privateFunctions`, the sampling kernel's call numbers it; the other kernel, whose
    // entry pruning took the module's call out of, numbers nothing, as Metal numbers nothing of a
    // private function whose call a function constant removed, and records no note.
    let private = LinkedSource {
        key: crate::pipeline::compile::key::LinkedKey {
            list: crate::pipeline::descriptor::LinkedList::PrivateFunctions,
            ..sample.key
        },
        bitcode: sample.bitcode,
    };
    let d = compile(SAMPLING, &[private]);
    assert_eq!(d.interface.numbered_samplers, [sampler]);
    assert!(!noted(&d), "{:?}", d.precision);
    let e = compile(CALLER, &[scale, blend, offset, private]);
    assert_eq!(e.interface.numbered_samplers, []);
    assert!(!noted(&e), "{:?}", e.precision);
}
