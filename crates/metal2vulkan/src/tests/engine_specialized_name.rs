//! Linked functions under a specialized name, and linked modules that declare function constants,
//! end to end against what the probe `linked_function_specialized_name.m` (the conformance
//! crate's `probes/`) measured on an Apple M3 Max under macOS 27.0.1.
//!
//! The owned fixtures: `engine/linked_specialized_caller` (kernels calling `custom_fn`, which no
//! module defines, and `base_fn`, `guarded_fn` and `plain_fn` by their own names) and
//! `engine/linked_specialized_functions` (those `float(float)` visible functions, in a module that
//! declares the float constant `scale` at index 0 and computes `twice = scale * 2` in its static
//! initializer: `base_fn(x) = x * scale + 1`, `derived_fn(x) = x * twice`, `guarded_fn(x) =
//! is_function_constant_defined(scale) ? x * scale : x - 1`, `plain_fn(x) = x + 2`). Each kernel
//! writes its callee's value at `tid + 5` to word `tid`. The kernel computes on the CPU
//! interpreter; no GPU is used.
//!
//! `llvm-as` assembles the fixtures; a missing tool skips visibly.

use crate::budget::{Budget, SPIRV_WORDS};
use crate::device::{Device, DeviceDescription, DevicePolicy};
use crate::error::{DescriptorError, DiagName, Error, LinkProblem, RuntimeInput};
use crate::library::{Function, FunctionConstantValue, FunctionConstantValues, Library};
use crate::lower::select::LowerInputs;
use crate::metal::DataType;
use crate::pipeline::compile::{compile_bitcode_linked, EntryRef, LinkedSource};
use crate::pipeline::sym::{ResourceKind, ResourceSlot, Stage};
use crate::pipeline::testing::{
    engine_dir, link_and_validate, llvm_assemble, meter, profile, typed_module_linked,
    AssembleError,
};
use crate::pipeline::{
    ComputePipelineDescriptor, FunctionDescriptor, LinkedFunction, LinkedFunctions, StageSource,
};
use crate::spirv::interp::{run, Buffer, Config, Resources};
use crate::spirv::testing::committed_profiles;

const CALLER: &str = "linked_specialized_caller";
const FUNCTIONS: &str = "linked_specialized_functions";

/// The fixture `stem` assembled, or `None` (skipped) without `llvm-as`.
fn fixture(stem: &str) -> Option<Vec<u8>> {
    match llvm_assemble(&engine_dir().join(format!("{stem}.ll"))) {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected {stem}: {why}"),
    }
}

/// Both fixtures' bitcode.
fn fixtures() -> Option<(Vec<u8>, Vec<u8>)> {
    Some((fixture(CALLER)?, fixture(FUNCTIONS)?))
}

fn device() -> Device {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles/moltenvk-m3.json");
    let text = std::fs::read_to_string(path).expect("profile");
    let desc = DeviceDescription::from_khronos_json(&text, None).expect("description");
    Device::new(desc, DevicePolicy::default()).expect("device")
}

/// `scale` set to `v`.
fn scale(v: f32) -> FunctionConstantValues {
    FunctionConstantValues::defaults().with_index(
        0,
        FunctionConstantValue::new(DataType::Float, v.to_le_bytes()),
    )
}

fn needs(symbol: &str) -> Error {
    Error::NeedsInput(RuntimeInput::LinkedFunction {
        symbol: DiagName::new(symbol),
    })
}

/// The caller's and the functions' libraries.
struct Libraries {
    caller: Library,
    functions: Library,
}

impl Libraries {
    fn new(caller: &[u8], functions: &[u8]) -> Libraries {
        Libraries {
            caller: Library::from_air(caller).expect("caller library"),
            functions: Library::from_air(functions).expect("functions library"),
        }
    }

    fn callee(&self, name: &str) -> Function {
        self.functions.function(name).expect("callee")
    }

    /// The compute pipeline of `kernel` linking `linked` through the public API.
    fn compile(&self, dev: &Device, kernel: &str, linked: LinkedFunctions) -> Result<(), Error> {
        let entry = self.caller.function(kernel).expect("kernel");
        let d = ComputePipelineDescriptor::new(StageSource::Function(
            FunctionDescriptor::new(entry).with_linked(linked),
        ));
        dev.compile_compute_pipeline(&d, &Budget::standard())
            .map(|_| ())
    }
}

/// A visible-function reference resolves to the linked function whose name it is: a function
/// made with the specialized name `custom_fn` answers that name and resolves `custom_fn`; made
/// without one (it answers `base_fn`) or with another, it does not, and the call needs its
/// function; linked beside one that does, the plain one changes nothing. The probe's
/// `specialized`, `plain`, `other` and `both`.
#[test]
fn a_reference_resolves_to_the_function_of_its_specialized_name() {
    let Some((caller, functions)) = fixtures() else {
        return;
    };
    let libs = Libraries::new(&caller, &functions);
    let dev = device();
    let base = || LinkedFunction::new(libs.callee("base_fn")).with_constants(scale(3.0));
    let specialized = base().with_specialized_name("custom_fn");
    assert_eq!(specialized.name(), "custom_fn");
    assert_eq!(base().name(), "base_fn");
    let compile = |linked| libs.compile(&dev, CALLER, linked);
    assert_eq!(
        compile(LinkedFunctions::new().with_function(specialized.clone())),
        Ok(())
    );
    assert_eq!(
        compile(LinkedFunctions::new().with_function(base())),
        Err(needs("custom_fn"))
    );
    assert_eq!(
        compile(
            LinkedFunctions::new()
                .with_function(specialized.clone())
                .with_function(base())
        ),
        Ok(())
    );
    assert_eq!(
        compile(LinkedFunctions::new().with_function(base().with_specialized_name("another_fn"))),
        Err(needs("custom_fn"))
    );
    // A private function resolves references by its specialized name too.
    assert_eq!(
        compile(LinkedFunctions::new().with_private_function(specialized)),
        Ok(())
    );
    // The specialized name names the reference; the definition is still `base_fn`'s.
    assert_eq!(
        libs.compile(
            &dev,
            "linked_specialized_base",
            LinkedFunctions::new().with_function(base().with_specialized_name("custom_fn"))
        ),
        Err(needs("base_fn"))
    );
}

/// The specialized name and the function's specialization are part of the compile key, as its
/// constant values are; a specialized name equal to the function's own is no other name.
#[test]
fn the_specialized_name_and_the_specialization_are_part_of_the_key() {
    let Some((caller, functions)) = fixtures() else {
        return;
    };
    let libs = Libraries::new(&caller, &functions);
    let dev = device();
    let entry = libs.caller.function(CALLER).expect("kernel");
    let key = |f: LinkedFunction| {
        dev.function_compile_key(
            &FunctionDescriptor::new(entry.clone())
                .with_linked(LinkedFunctions::new().with_function(f)),
            &LowerInputs::new(),
        )
    };
    let base = || LinkedFunction::new(libs.callee("base_fn"));
    let plain = key(base());
    let custom = key(base().with_specialized_name("custom_fn"));
    let another = key(base().with_specialized_name("another_fn"));
    let specialized = key(base().with_specialized(true));
    assert_eq!(custom, key(base().with_specialized_name("custom_fn")));
    assert_eq!(
        key(base().with_specialized_name("base_fn")),
        specialized,
        "its own name is no other name"
    );
    let keys = [plain, custom, another, specialized];
    for (i, a) in keys.iter().enumerate() {
        for b in &keys[i + 1..] {
            assert_ne!(a, b);
        }
    }
}

/// Words `0..N` the kernel `CALLER` writes linking `linked` (its reference `custom_fn`
/// resolved by one of them), on the CPU interpreter.
fn words(caller: &[u8], linked: &[LinkedSource<'_>]) -> Vec<f32> {
    const N: u32 = 8;
    let p = profile("moltenvk-m3", DevicePolicy::default()).expect("profile");
    let (_, m) = typed_module_linked(caller, CALLER, &p, linked).unwrap_or_else(|e| panic!("{e}"));
    let res = Resources {
        buffers: vec![Buffer {
            slot: Some(ResourceSlot {
                stage: Stage::Kernel,
                kind: ResourceKind::Buffer,
                index: 0,
            }),
            address: None,
            bytes: vec![0; N as usize * 4],
        }],
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
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}

/// A linked function computes with its own constant values, specialized before it is linked as
/// Metal specializes the `MTLFunction`: `base_fn` with `scale` 3 under the name `custom_fn` stores
/// the probe's 16 at word 0 (5 * 3 + 1), `derived_fn` reads the static initializer's `twice`
/// (the probe's `derived`, 30), `guarded_fn` takes its defined way with the value and the other
/// without one (the probe's `guarded-valued` 15 and `guarded` 4), and `plain_fn` reads no
/// constant (the probe's 7). Without a value, `base_fn` reads `scale` as an entry's undefined
/// constants read, as zero (Metal's read is undefined: its store did not land).
#[test]
fn linked_functions_compute_with_their_own_constant_values() {
    let Some((caller, functions)) = fixtures() else {
        return;
    };
    let (three, none) = (scale(3.0), FunctionConstantValues::defaults());
    let cases: [(&str, &FunctionConstantValues, fn(f32) -> f32); 6] = [
        ("base_fn", &three, |x| x * 3.0 + 1.0),
        ("derived_fn", &three, |x| x * 6.0),
        ("guarded_fn", &three, |x| x * 3.0),
        ("guarded_fn", &none, |x| x - 1.0),
        ("plain_fn", &none, |x| x + 2.0),
        ("base_fn", &none, |_| 1.0),
    ];
    for (name, constants, expect) in cases {
        let linked =
            [LinkedSource::new(name, &functions, constants).with_specialized_name("custom_fn")];
        let got = words(&caller, &linked);
        let want: Vec<f32> = (0..got.len()).map(|t| expect(t as f32 + 5.0)).collect();
        assert_eq!(got, want, "{name} with {constants:?}");
    }
    let probe =
        [LinkedSource::new("base_fn", &functions, &three).with_specialized_name("custom_fn")];
    assert_eq!(words(&caller, &probe)[0], 16.0);
}

/// A function made unspecialized (`newFunctionWithName:`) that reaches its module's function
/// constant refuses the pipeline, guarded by `is_function_constant_defined` or not, as Metal
/// fails the creation ("cannot lower module with unresolved function constants": the probe's
/// `byname` and `guarded-byname`); one that reaches none links (`constant-free-byname`). Made
/// specialized it links with a value and without one: `newFunctionWithName:constantValues:`
/// with empty values and `newFunctionWithDescriptor:` with none (`byname-empty`,
/// `descriptor-nil`), the constant read as undefined.
#[test]
fn an_unspecialized_function_that_reaches_a_constant_is_refused() {
    let Some((caller, functions)) = fixtures() else {
        return;
    };
    let libs = Libraries::new(&caller, &functions);
    let dev = device();
    let unresolved = |name: &str| {
        Err(Error::InvalidDescriptor(DescriptorError::LinkedFunction {
            name: DiagName::new(name),
            problem: LinkProblem::UnresolvedConstant { index: 0 },
        }))
    };
    let by_name = |name: &str| LinkedFunction::new(libs.callee(name));
    let compile = |kernel, f: LinkedFunction| {
        libs.compile(&dev, kernel, LinkedFunctions::new().with_function(f))
    };
    let base = "linked_specialized_base";
    assert_eq!(compile(base, by_name("base_fn")), unresolved("base_fn"));
    assert_eq!(
        compile("linked_specialized_guarded", by_name("guarded_fn")),
        unresolved("guarded_fn")
    );
    assert_eq!(
        compile("linked_specialized_free", by_name("plain_fn")),
        Ok(())
    );
    assert_eq!(
        compile(
            base,
            by_name("base_fn").with_constants(FunctionConstantValues::defaults())
        ),
        Ok(())
    );
    assert_eq!(
        compile(base, by_name("base_fn").with_specialized(true)),
        Ok(())
    );
    assert_eq!(
        compile(base, by_name("base_fn").with_constants(scale(3.0))),
        Ok(())
    );
    // A private function is refused the same way.
    assert_eq!(
        libs.compile(
            &dev,
            base,
            LinkedFunctions::new().with_private_function(by_name("base_fn"))
        ),
        unresolved("base_fn")
    );
}

/// The kernel linking `base_fn` under the name `custom_fn`, specialized with `scale` 3, passes L3
/// on every committed profile, and so does `guarded_fn` linked without a value.
#[test]
fn a_kernel_linking_a_specialized_function_validates_on_every_profile() {
    let Some((caller, functions)) = fixtures() else {
        return;
    };
    let (three, none) = (scale(3.0), FunctionConstantValues::defaults());
    let sets = [
        [LinkedSource::new("base_fn", &functions, &three).with_specialized_name("custom_fn")],
        [LinkedSource::new("guarded_fn", &functions, &none).with_specialized_name("custom_fn")],
    ];
    for (stem, p) in committed_profiles().expect("profiles") {
        for linked in &sets {
            let a = compile_bitcode_linked(
                &caller,
                EntryRef::Name(CALLER),
                (&none, linked),
                &p,
                &LowerInputs::new(),
                &mut meter(),
                SPIRV_WORDS,
            )
            .unwrap_or_else(|e| panic!("{stem}: {e}"));
            link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{stem}: L3: {e}"));
        }
    }
}
