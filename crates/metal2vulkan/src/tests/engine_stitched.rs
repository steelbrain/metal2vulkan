//! Functions Metal's stitcher makes (`newLibraryWithStitchedDescriptor:`), linked into a kernel
//! that calls them, end to end against what the probe `stitched_function_linked.m` (the
//! conformance crate's `probes/`) measured on an Apple M3 Max under macOS 27.0.1: `fused` linked
//! as a private function or a plain one stores `(2 + 3) * 4 = 20`, and `fused4` stores
//! `(2 + 3) * 4 + 5 = 25`.
//!
//! The caller is the owned fixture `engine/stitched_caller` (kernels `stitch_caller` and
//! `stitch_caller4` storing `fused(in[0], in[1], in[2])` and `fused4(in[0], .., in[3])` to word 0).
//! The stitched library is `metal2vulkan_devtools::stitched`'s, in the form the probe read back
//! from Metal: one module per stitched function, each carrying the stitching signature
//! (`!air.signature`) that describes the graph it was made from. The kernels compute on the CPU
//! interpreter; no GPU is used.
//!
//! `llvm-as` assembles the modules; a missing tool skips visibly.

use metal2vulkan_devtools::stitched::{self, Graph, PROBE_GRAPHS};

use crate::budget::{Budget, SPIRV_WORDS};
use crate::device::{Device, DeviceDescription, DevicePolicy};
use crate::error::Error;
use crate::library::{FunctionConstantValues, FunctionStage, Library};
use crate::lower::select::LowerInputs;
use crate::pipeline::compile::{compile_bitcode_linked, EntryRef, LinkedSource};
use crate::pipeline::sym::{ResourceKind, ResourceSlot, Stage};
use crate::pipeline::testing::{
    engine_dir, link_and_validate, llvm_assemble, llvm_assemble_text, meter, profile,
    typed_module_linked, AssembleError,
};
use crate::pipeline::{
    ComputePipelineDescriptor, FunctionDescriptor, LinkedFunction, LinkedFunctions, StageSource,
};
use crate::spirv::interp::{run, Buffer, Config, Resources};
use crate::spirv::testing::committed_profiles;

/// The kernel of the caller fixture that calls each of [`PROBE_GRAPHS`].
const KERNELS: [&str; 2] = ["stitch_caller", "stitch_caller4"];

/// The bitcode `assembled` gives, or `None` (skipped) without `llvm-as`.
fn assembled(what: &str, assembled: Result<Vec<u8>, AssembleError>) -> Option<Vec<u8>> {
    match assembled {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected {what}: {why}"),
    }
}

/// The caller fixture's bitcode and each stitched function's module, in [`PROBE_GRAPHS`]'s order.
fn fixtures() -> Option<(Vec<u8>, Vec<Vec<u8>>)> {
    let caller = assembled(
        "stitched_caller",
        llvm_assemble(&engine_dir().join("stitched_caller.ll")),
    )?;
    let modules = PROBE_GRAPHS
        .iter()
        .map(|g: &Graph<'_>| assembled(g.name, llvm_assemble_text(&stitched::module(g))))
        .collect::<Option<Vec<_>>>()?;
    Some((caller, modules))
}

fn device() -> Device {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles/moltenvk-m3.json");
    let text = std::fs::read_to_string(path).expect("profile");
    let desc = DeviceDescription::from_khronos_json(&text, None).expect("description");
    Device::new(desc, DevicePolicy::default()).expect("device")
}

/// The stitched library holds `fused` and `fused4` as visible functions, each in its own module,
/// and each kernel's compute pipeline links its function as Metal links it, in
/// `privateFunctions` and in `functions` alike (the probe's `private`, `functions` and `deep`).
#[test]
fn a_stitched_function_links_privately_or_not() {
    let Some((caller, modules)) = fixtures() else {
        return;
    };
    let named: Vec<(&str, &[u8])> = PROBE_GRAPHS
        .iter()
        .zip(&modules)
        .map(|(g, m)| (g.name, m.as_slice()))
        .collect();
    let stitched = Library::from_metallib(&stitched::library(&named)).expect("stitched library");
    let caller = Library::from_air(&caller).expect("caller library");
    let dev = device();
    for (kernel, g) in KERNELS.iter().zip(&PROBE_GRAPHS) {
        let f = stitched.function(g.name).expect("stitched function");
        assert_eq!(f.info().stage, FunctionStage::Visible, "{}", g.name);
        let f = LinkedFunction::new(f);
        for linked in [
            LinkedFunctions::new().with_private_function(f.clone()),
            LinkedFunctions::new().with_function(f.clone()),
        ] {
            let entry = caller.function(kernel).expect("kernel");
            let d = ComputePipelineDescriptor::new(StageSource::Function(
                FunctionDescriptor::new(entry).with_linked(linked.clone()),
            ));
            let made: Result<(), Error> = dev
                .compile_compute_pipeline(&d, &Budget::standard())
                .map(|_| ());
            assert_eq!(made, Ok(()), "{kernel} linking {}: {linked:?}", g.name);
        }
    }
}

/// Word 0 that `kernel` of `caller` stores linking the stitched `name` (`module`) on `input`,
/// on the CPU interpreter.
fn word(caller: &[u8], kernel: &str, (name, module): (&str, &[u8]), input: [f32; 4]) -> f32 {
    let none = FunctionConstantValues::defaults();
    let p = profile("moltenvk-m3", DevicePolicy::default()).expect("profile");
    let linked = [LinkedSource::new(name, module, &none)];
    let (_, m) = typed_module_linked(caller, kernel, &p, &linked).unwrap_or_else(|e| panic!("{e}"));
    let slot = |index| {
        Some(ResourceSlot {
            stage: Stage::Kernel,
            kind: ResourceKind::Buffer,
            index,
        })
    };
    let res = Resources {
        buffers: vec![
            Buffer {
                slot: slot(0),
                address: None,
                bytes: vec![0; 4],
            },
            Buffer {
                slot: slot(1),
                address: None,
                bytes: input.iter().flat_map(|v| v.to_le_bytes()).collect(),
            },
        ],
        push: vec![0; 128],
        spec: Default::default(),
    };
    let out = run(&m, &Config::default(), res).unwrap_or_else(|e| panic!("{e:?}"));
    f32::from_le_bytes(out.buffers[0][..4].try_into().expect("word 0"))
}

/// The stitched function computes its graph: on the probe's inputs 2, 3, 4 and 5, `fused` stores
/// Metal's 20 and `fused4` its 25, and on other inputs what the graph computes.
#[test]
fn a_stitched_function_computes_its_graph() {
    let Some((caller, modules)) = fixtures() else {
        return;
    };
    let fused = |[a, b, c, _]: [f32; 4]| (a + b) * c;
    let fused4 = |[a, b, c, d]: [f32; 4]| (a + b) * c + d;
    let graphs: [fn([f32; 4]) -> f32; 2] = [fused, fused4];
    let probe = [2.0, 3.0, 4.0, 5.0];
    for (i, (kernel, g)) in KERNELS.iter().zip(&PROBE_GRAPHS).enumerate() {
        let linked = (g.name, modules[i].as_slice());
        for input in [probe, [1.5, -2.0, 3.0, 0.25], [-1.0, 7.0, 0.5, 10.0]] {
            let got = word(&caller, kernel, linked, input);
            assert_eq!(got, graphs[i](input), "{} on {input:?}", g.name);
        }
    }
    let at_probe = |i: usize| {
        word(
            &caller,
            KERNELS[i],
            (PROBE_GRAPHS[i].name, &modules[i]),
            probe,
        )
    };
    assert_eq!([at_probe(0), at_probe(1)], [20.0, 25.0]);
}

/// Each kernel linking its stitched function passes L3 on every committed profile.
#[test]
fn a_kernel_linking_a_stitched_function_validates_on_every_profile() {
    let Some((caller, modules)) = fixtures() else {
        return;
    };
    let none = FunctionConstantValues::defaults();
    for (stem, p) in committed_profiles().expect("profiles") {
        for (i, (kernel, g)) in KERNELS.iter().zip(&PROBE_GRAPHS).enumerate() {
            let linked = [LinkedSource::new(g.name, &modules[i], &none)];
            let a = compile_bitcode_linked(
                &caller,
                EntryRef::Name(kernel),
                (&none, &linked),
                &p,
                &LowerInputs::new(),
                &mut meter(),
                SPIRV_WORDS,
            )
            .unwrap_or_else(|e| panic!("{stem}: {kernel}: {e}"));
            link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{stem}: {kernel}: L3: {e}"));
        }
    }
}
