//! Dynamic libraries end to end through the public API.
//!
//! The modules are written here as Apple's toolchain shapes them (probes, an M3 Max and an M4): a
//! dynamic library is a metallib of file type 2 whose exported functions (`TYPE` 3) share its one
//! module, named by its install name (`HDYN`); a client lists the declarations it imports in
//! `air.dyld_flat_table` and names the libraries it was linked against (`HDYN`'s `DYNL`s).
//! `llvm-as` assembles the modules (a missing tool skips visibly), and `metal2vulkan-devtools`
//! writes the containers around them. The answers checked are Metal's measured ones: the
//! library type and install name, `newDynamicLibrary:error:`'s refusals, a dynamic library's
//! function being no stage, the import resolved from a preloaded library or from the library an
//! install name loads (computed on the CPU interpreter), an import no library exports, an
//! install name not loaded failing even what a preloaded library exports, a dead import needing
//! nothing, and the universal file `serializeToURL:error:` writes. No GPU is used.

use metal2vulkan_devtools::metallib::{metallib_linked, universal, wrap, UniversalHeader};
use metal2vulkan_devtools::metallib::{MtlbFunction, MtlbLinkage};

use crate::budget::Budget;
use crate::device::{Device, DeviceDescription, DevicePolicy};
use crate::error::{DescriptorError, DiagName, DynamicLibraryProblem, Error, RuntimeInput};
use crate::library::{DynamicLibrary, FunctionStage, Library, LibraryType};
use crate::lower::select::LowerInputs;
use crate::pipeline::compile::compile_stage_kept;
use crate::pipeline::sym::{ResourceKind, ResourceSlot, Stage};
use crate::pipeline::testing::{llvm_assemble, meter, profile, AssembleError, Scratch};
use crate::pipeline::{
    ComputePipelineDescriptor, DynamicLibraries, FunctionDescriptor, StageSource,
};
use crate::spirv::interp::{run, Buffer, Config, Resources};

const HEADER: &str = "target datalayout = \"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32\"\ntarget triple = \"air64_v28-apple-macosx26.0.0\"\n\n";

/// The version metadata every module below ends with (the container's `VERS` states the same).
const VERSIONS: &str = "!air.version = !{!90}\n!air.language_version = !{!91}\n\
                        !90 = !{i32 2, i32 8, i32 0}\n!91 = !{!\"Metal\", i32 4, i32 0, i32 0}\n";

/// A device buffer `out` at parameter 0.
const OUT: &str = "!{i32 0, !\"air.buffer\", !\"air.location_index\", i32 0, i32 1, \
                   !\"air.read_write\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", \
                   i32 4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"uint\", \
                   !\"air.arg_name\", !\"out\"}";

/// The symbol of `uint dyl_scale(uint)`.
const SCALE: &str = "_Z9dyl_scalej";

/// `text` (after [`HEADER`]) assembled, or `None` when `llvm-as` is missing.
fn assemble(tag: &str, text: &str) -> Option<Vec<u8>> {
    let scratch = Scratch::new(&format!("m2v-dylib-{tag}"));
    let dir = scratch.path();
    let path = dir.join("module.ll");
    std::fs::write(&path, format!("{HEADER}{text}{VERSIONS}")).expect("write");
    let bc = llvm_assemble(&path);
    match bc {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected {tag}: {why}"),
    }
}

/// A dynamic library module: `dyl_scale(x) = x * mul`, and the exported variable
/// `constant uint dyl_k = k`; with `constant`, `dyl_scale` adds a function constant's value.
fn library_module(tag: &str, mul: u32, k: u32, constant: bool) -> Option<Vec<u8>> {
    let (fc_global, fc_use, fc_md) = if constant {
        (
            "@fc.MTL_FC_INIT_0_u = internal addrspace(2) externally_initialized constant i32 \
             undef, section \"air.fc_initializer\", align 4\n",
            "  %c = load i32, ptr addrspace(2) @fc.MTL_FC_INIT_0_u, align 4\n  \
             %r2 = add i32 %r, %c\n  ret i32 %r2\n",
            "!air.function_constants = !{!80}\n!80 = !{ptr addrspace(2) @fc.MTL_FC_INIT_0_u, \
             !\"uint\", !\"fc\", i32 0}\n",
        )
    } else {
        ("", "  ret i32 %r\n", "")
    };
    let text = format!(
        "@dyl_k = addrspace(2) constant i32 {k}, align 4\n{fc_global}\n\
         define i32 @{SCALE}(i32 %x) {{\n  %r = mul i32 %x, {mul}\n{fc_use}}}\n\n{fc_md}"
    );
    assemble(tag, &text)
}

/// A dynamic library of install name `name` over `module`, exporting `dyl_scale`.
fn dylib_bytes(name: &str, module: &[u8]) -> Vec<u8> {
    dylib_bytes_linked(name, module, &[])
}

/// [`dylib_bytes`] linked against the libraries of install names `dependencies`.
fn dylib_bytes_linked(name: &str, module: &[u8], dependencies: &[&str]) -> Vec<u8> {
    let f = MtlbFunction {
        name: SCALE,
        type_code: 3,
        bitcode: wrap(module),
        constants: Vec::new(),
        public: Vec::new(),
        tags: Vec::new(),
    };
    let linkage = MtlbLinkage {
        dynamic: true,
        install_name: Some(name),
        dependencies: dependencies.to_vec(),
        ..MtlbLinkage::default()
    };
    metallib_linked(&[f], &linkage)
}

fn dylib(name: &str, mul: u32, k: u32) -> Option<DynamicLibrary> {
    let module = library_module(name, mul, k, false)?;
    let lib = Library::parse(&dylib_bytes(name, &module)).expect("library");
    Some(DynamicLibrary::new(lib, &Budget::standard()).expect("dynamic library"))
}

/// The tables Apple's linker writes into a module that imports `imports` (constant
/// expressions of the declarations' addresses) and was linked against `dependencies` (the probe's
/// `libC`): the install names' offsets and lengths and the names themselves, the flat table of the
/// imports, and `llvm.used` keeping them. A module linked against nothing (which Metal's toolchain
/// cannot make) has the flat table alone.
fn dyld_tables(dependencies: &[&str], imports: &[String]) -> String {
    let flat = format!(
        "@air.dyld_flat_table = hidden global [{} x ptr] [{}], section \"air.dyld_flat_table\"\n",
        imports.len(),
        imports
            .iter()
            .map(|i| format!("ptr {i}"))
            .collect::<Vec<_>>()
            .join(", ")
    );
    if dependencies.is_empty() {
        return flat;
    }
    let (mut entries, mut names, mut length) = (Vec::new(), String::new(), 0usize);
    for d in dependencies {
        entries.push(format!(
            "{{ i32, i32 }} {{ i32 {length}, i32 {} }}",
            d.len()
        ));
        names.push_str(d);
        names.push_str("\\00");
        length += d.len() + 1;
    }
    format!(
        "@air.dyld_lib_table = hidden global [{} x {{ i32, i32 }}] [{}], section \
         \"air.dyld_lib_table\"\n@air.dyld_str_table = hidden global [{length} x i8] \
         c\"{names}\", section \"air.dyld_str_table\"\n{flat}@llvm.used = appending global \
         [3 x ptr] [ptr @air.dyld_lib_table, ptr @air.dyld_str_table, ptr \
         @air.dyld_flat_table], section \"llvm.metadata\"\n",
        entries.len(),
        entries.join(", ")
    )
}

/// A client kernel `k(device uint *out, uint tid)`: `out[tid] = dyl_scale(tid) + dyl_k`, or with
/// `dead` nothing (its imports behind `br false`), or without `variable` only `dyl_scale(tid)`;
/// linked against `dependencies`, with the linker's tables.
fn client_module(tag: &str, variable: bool, dead: bool, dependencies: &[&str]) -> Option<Vec<u8>> {
    let scale = format!("@{SCALE}");
    let (k_load, sum, table) = if variable {
        let k = "addrspacecast (ptr addrspace(2) @dyl_k to ptr)".to_owned();
        (
            "  %k = load i32, ptr addrspace(2) @dyl_k, align 4\n  %s = add i32 %v, %k\n",
            "%s",
            format!(
                "{}@dyl_k = external addrspace(2) constant i32, align 4\n",
                dyld_tables(dependencies, &[scale, k])
            ),
        )
    } else {
        ("", "%v", dyld_tables(dependencies, &[scale]))
    };
    let compute = format!(
        "  %v = call i32 @{SCALE}(i32 %1)\n{k_load}  %p = getelementptr i32, ptr addrspace(1) \
         %0, i32 %1\n  store i32 {sum}, ptr addrspace(1) %p, align 4\n"
    );
    let body = if dead {
        format!("  br i1 false, label %then, label %done\nthen:\n{compute}  br label %done\ndone:\n  ret void\n")
    } else {
        format!("{compute}  ret void\n")
    };
    let text = format!(
        "{table}\ndefine void @k(ptr addrspace(1) %0, i32 %1) {{\n{body}}}\n\n\
         declare i32 @{SCALE}(i32)\n\n\
         !air.kernel = !{{!0}}\n!0 = !{{ptr @k, !1, !2}}\n!1 = !{{}}\n!2 = !{{!3, !4}}\n!3 = {OUT}\n\
         !4 = !{{i32 1, !\"air.thread_position_in_grid\", !\"air.arg_type_name\", !\"uint\", \
         !\"air.arg_name\", !\"tid\"}}\n"
    );
    assemble(tag, &text)
}

/// The client kernel in a metallib whose `HDYN` names `dependencies`.
fn client(tag: &str, variable: bool, dependencies: &[&str]) -> Option<Library> {
    client_with(tag, variable, false, dependencies)
}

/// [`client`], with its imports behind `br false` when `dead`.
fn client_with(tag: &str, variable: bool, dead: bool, dependencies: &[&str]) -> Option<Library> {
    let module = client_module(tag, variable, dead, dependencies)?;
    let f = MtlbFunction {
        name: "k",
        type_code: 2,
        bitcode: wrap(&module),
        constants: Vec::new(),
        public: Vec::new(),
        tags: Vec::new(),
    };
    let linkage = MtlbLinkage {
        install_name: Some("client.metallib"),
        dependencies: dependencies.to_vec(),
        imports: vec![SCALE],
        ..MtlbLinkage::default()
    };
    Some(Library::parse(&metallib_linked(&[f], &linkage)).expect("client"))
}

fn device() -> Device {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles/moltenvk-m3.json");
    let text = std::fs::read_to_string(path).expect("profile");
    let desc = DeviceDescription::from_khronos_json(&text, None).expect("description");
    Device::new(desc, DevicePolicy::default()).expect("device")
}

fn stage(lib: &Library, libraries: DynamicLibraries) -> FunctionDescriptor {
    FunctionDescriptor::new(lib.function("k").expect("k")).with_libraries(libraries)
}

fn compile(fd: FunctionDescriptor) -> Result<(), Error> {
    let d = ComputePipelineDescriptor::new(StageSource::Function(fd));
    device()
        .compile_compute_pipeline(&d, &Budget::standard())
        .map(|_| ())
}

/// `out` after the stage `fd` runs over 16 threads on the CPU interpreter.
fn computed(fd: &FunctionDescriptor) -> Vec<u32> {
    let p = profile("moltenvk-m3", DevicePolicy::default()).expect("profile");
    let m = compile_stage_kept(fd, &p, &mut meter()).unwrap_or_else(|e| panic!("{e}"));
    const N: u32 = 16;
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
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn needs_symbol(symbol: &str) -> Error {
    Error::NeedsInput(RuntimeInput::DynamicSymbol {
        symbol: DiagName::new(symbol),
    })
}

fn needs_library(install_name: &str) -> Error {
    Error::NeedsInput(RuntimeInput::DynamicLibrary {
        install_name: DiagName::new(install_name),
    })
}

fn refused(problem: DynamicLibraryProblem) -> Error {
    Error::InvalidDescriptor(DescriptorError::DynamicLibrary { problem })
}

/// A dynamic metallib is a library of type dynamic with its install name, its exported function
/// no stage; `newDynamicLibrary:error:` takes it, and refuses an executable library, a bare
/// module, and a library whose code reads a function constant.
#[test]
fn a_dynamic_library_is_made_as_metal_makes_one() {
    let Some(module) = library_module("lib", 3, 7, false) else {
        return;
    };
    let lib = Library::parse(&dylib_bytes("@rpath/libA.metallib", &module)).expect("library");
    assert_eq!(lib.library_type(), LibraryType::Dynamic);
    assert_eq!(lib.install_name(), Some("@rpath/libA.metallib"));
    let stages: Vec<FunctionStage> = lib.functions().map(|f| f.stage).collect();
    assert_eq!(stages, [FunctionStage::Exported]);
    let made = DynamicLibrary::new(lib.clone(), &Budget::standard()).expect("dynamic");
    assert_eq!(made.install_name(), "@rpath/libA.metallib");
    assert_eq!(made.library(), &lib);

    // A dynamic library's function is no stage (Metal's `newFunctionWithName:` asserts).
    let d = ComputePipelineDescriptor::new(StageSource::Function(FunctionDescriptor::new(
        lib.function(SCALE).expect("exported"),
    )));
    assert_eq!(
        device()
            .compile_compute_pipeline(&d, &Budget::standard())
            .map(|_| ()),
        Err(Error::InvalidDescriptor(DescriptorError::EntryNotFound {
            requested: DiagName::new(SCALE),
            available: 0,
        }))
    );

    let Some(exe) = client("exe", false, &[]) else {
        return;
    };
    assert_eq!(exe.library_type(), LibraryType::Executable);
    assert_eq!(exe.install_name(), None);
    assert_eq!(
        DynamicLibrary::new(exe, &Budget::standard()),
        Err(refused(DynamicLibraryProblem::NotDynamic))
    );
    let bare = Library::from_air(&module).expect("bare");
    assert_eq!(
        DynamicLibrary::new(bare, &Budget::standard()),
        Err(refused(DynamicLibraryProblem::NotDynamic))
    );
    let Some(constant) = library_module("fc", 3, 7, true) else {
        return;
    };
    let lib = Library::parse(&dylib_bytes("libF.metallib", &constant)).expect("library");
    assert_eq!(
        DynamicLibrary::new(lib.clone(), &Budget::standard()),
        Err(refused(DynamicLibraryProblem::FunctionConstants))
    );

    // Loaded for an install name, a library is checked without decoding it: what cannot link
    // of it is refused only when a surviving import reaches it.
    let loaded = DynamicLibrary::loaded(lib).expect("loaded");
    let Some(user) = client("fc-user", false, &[]) else {
        return;
    };
    let reached = compile(stage(
        &user,
        DynamicLibraries::new().with_preloaded(loaded.clone()),
    ));
    assert!(
        matches!(&reached, Err(Error::Unsupported(_))),
        "{reached:?}"
    );
    let Some(dead) = client_module("fc-dead", true, true, &[]) else {
        return;
    };
    let dead = Library::from_air(&dead).expect("dead");
    assert_eq!(
        compile(stage(&dead, DynamicLibraries::new().with_preloaded(loaded))),
        Ok(())
    );
    let Some(exe) = client("loaded-exe", false, &[]) else {
        return;
    };
    assert_eq!(
        DynamicLibrary::loaded(exe),
        Err(refused(DynamicLibraryProblem::NotDynamic))
    );
}

/// A universal file, the one `serializeToURL:error:` writes or a Mach-O universal file of either
/// width, is the dynamic library its AIR slice holds: the library keeps that slice alone, hashed
/// as the bare metallib is, and loads and computes as it does.
#[test]
fn a_universal_file_is_the_dynamic_library_it_holds() {
    let Some(module) = library_module("universal", 3, 7, false) else {
        return;
    };
    let air = dylib_bytes("libA.metallib", &module);
    let bare = Library::parse(&air).expect("metallib");
    let gpu = [0xCFu8, 0xFA, 0xED, 0xFE, 0, 0, 0, 0];
    let Some(client) = client("universal-client", true, &[]) else {
        return;
    };
    let want: Vec<u32> = (0..16).map(|i| 3 * i + 7).collect();
    for header in UniversalHeader::ALL {
        let bytes = universal(header, &[(0x0100_0013, &gpu), (0x0100_0017, &air)]);
        let lib = Library::parse(&bytes).expect("universal");
        assert_eq!(lib.bytes(), &air[..], "{header:?}");
        assert_eq!(lib.content_hash(), bare.content_hash(), "{header:?}");
        assert_eq!(lib.library_type(), LibraryType::Dynamic, "{header:?}");
        assert_eq!(lib.install_name(), Some("libA.metallib"), "{header:?}");
        let made = DynamicLibrary::new(lib, &Budget::standard()).expect("dynamic");
        let fd = stage(&client, DynamicLibraries::new().with_preloaded(made));
        assert_eq!(computed(&fd), want, "{header:?}");
    }
}

/// A kernel's imports resolve from a preloaded library, or from the library its install name
/// loads, and compute what Metal computes: `out[i] = 3i + 7`. The first library exporting a
/// symbol supplies it.
#[test]
fn imports_resolve_from_preloaded_and_loaded_libraries() {
    let (Some(a), Some(b)) = (dylib("libA.metallib", 3, 7), dylib("libB.metallib", 5, 11)) else {
        return;
    };
    let Some(alone) = client("plain", true, &[]) else {
        return;
    };
    let want: Vec<u32> = (0..16).map(|i| 3 * i + 7).collect();
    let preloaded = DynamicLibraries::new().with_preloaded(a.clone());
    assert_eq!(computed(&stage(&alone, preloaded.clone())), want);
    assert_eq!(compile(stage(&alone, preloaded)), Ok(()));
    // Preloaded first, in order.
    let both = DynamicLibraries::new()
        .with_preloaded(a.clone())
        .with_preloaded(b.clone());
    assert_eq!(computed(&stage(&alone, both)), want);
    let swapped = DynamicLibraries::new()
        .with_preloaded(b.clone())
        .with_preloaded(a.clone());
    let other: Vec<u32> = (0..16).map(|i| 5 * i + 11).collect();
    assert_eq!(computed(&stage(&alone, swapped)), other);

    // Through the install name the client was linked against.
    let Some(linked) = client("linked", true, &["libA.metallib"]) else {
        return;
    };
    let loaded = DynamicLibraries::new().with_loaded("libA.metallib", a.clone());
    assert_eq!(computed(&stage(&linked, loaded.clone())), want);
    // Preloaded libraries come before the loaded ones.
    let first = loaded.with_preloaded(b);
    assert_eq!(computed(&stage(&linked, first)), other);
}

/// An import no library exports needs its symbol; an install name with no loaded library fails
/// every import, even one a preloaded library exports (Metal fails such a creation); and an
/// import whose use specialization deletes needs nothing at all.
#[test]
fn missing_imports_and_unloaded_install_names_are_refused_only_when_used() {
    let Some(a) = dylib("libA.metallib", 3, 7) else {
        return;
    };
    let Some(alone) = client("missing", false, &[]) else {
        return;
    };
    assert_eq!(
        compile(stage(&alone, DynamicLibraries::new())),
        Err(needs_symbol(SCALE))
    );
    let Some(linked) = client("unloaded", false, &["libA.metallib"]) else {
        return;
    };
    let preloaded = DynamicLibraries::new().with_preloaded(a);
    assert_eq!(
        compile(stage(&linked, preloaded)),
        Err(needs_library("libA.metallib"))
    );
    let Some(dead) = client_module("dead", true, true, &[]) else {
        return;
    };
    let lib = Library::from_air(&dead).expect("dead");
    assert_eq!(compile(stage(&lib, DynamicLibraries::new())), Ok(()));
    // Nor an install name not loaded (Metal's CLnone-Agone).
    let Some(dead) = client_with("dead-linked", true, true, &["libA.metallib"]) else {
        return;
    };
    assert_eq!(compile(stage(&dead, DynamicLibraries::new())), Ok(()));
}

/// The libraries a stage's imports are searched in are part of its compile key, and a stage
/// with none keeps the key it had.
#[test]
fn the_searched_libraries_are_part_of_the_key() {
    let (Some(a), Some(b)) = (dylib("libA.metallib", 3, 7), dylib("libB.metallib", 5, 11)) else {
        return;
    };
    let Some(linked) = client("key", true, &["libA.metallib"]) else {
        return;
    };
    let dev = device();
    let inputs = LowerInputs::new();
    let key =
        |libraries: DynamicLibraries| dev.function_compile_key(&stage(&linked, libraries), &inputs);
    let plain = dev.function_compile_key(
        &FunctionDescriptor::new(linked.function("k").expect("k")),
        &inputs,
    );
    let unloaded = key(DynamicLibraries::new());
    assert_eq!(plain, unloaded, "no libraries given is none given");
    let loaded = key(DynamicLibraries::new().with_loaded("libA.metallib", a.clone()));
    assert_ne!(loaded, unloaded);
    assert_eq!(
        loaded,
        key(DynamicLibraries::new().with_loaded("libA.metallib", a.clone()))
    );
    let other = key(DynamicLibraries::new().with_loaded("libA.metallib", b));
    assert_ne!(loaded, other, "another library at the install name");
    // A bare client names no install name: without libraries, its key is the one it had.
    let Some(bare) = client_module("bare-key", true, false, &[]) else {
        return;
    };
    let lib = Library::from_air(&bare).expect("bare");
    let f = lib.function("k").expect("k");
    assert_eq!(
        dev.function_compile_key(&FunctionDescriptor::new(f.clone()), &inputs),
        dev.function_compile_key(
            &FunctionDescriptor::new(f).with_libraries(DynamicLibraries::new()),
            &inputs
        )
    );
}

/// Loading a stage's dependencies asks for each install name in the search's order, a loaded
/// library's own after those already reached, skips one already loaded, and stops at the first
/// that does not load, whose imports the compile then refuses.
#[test]
fn dependencies_load_in_the_search_order() {
    let (Some(a), Some(b)) = (dylib("libA.metallib", 3, 7), dylib("libB.metallib", 5, 11)) else {
        return;
    };
    let Some(module) = library_module("libC", 9, 13, false) else {
        return;
    };
    let c = Library::parse(&dylib_bytes_linked(
        "libC.metallib",
        &module,
        &["libA.metallib"],
    ))
    .expect("library");
    let c = DynamicLibrary::new(c, &Budget::standard()).expect("dynamic");
    let Some(client) = client("load", true, &["libC.metallib", "libB.metallib"]) else {
        return;
    };
    let file = |name: &str| match name {
        "libA.metallib" => Some(a.clone()),
        "libB.metallib" => Some(b.clone()),
        "libC.metallib" => Some(c.clone()),
        _ => None,
    };
    let mut asked = Vec::new();
    let fd = stage(&client, DynamicLibraries::new()).with_loaded_dependencies(|name| {
        asked.push(name.to_owned());
        file(name)
    });
    assert_eq!(asked, ["libC.metallib", "libB.metallib", "libA.metallib"]);
    // The client's first dependency exports both symbols.
    let want: Vec<u32> = (0..16).map(|i| 9 * i + 13).collect();
    assert_eq!(computed(&fd), want);

    // A library already loaded is not asked for.
    let mut asked = Vec::new();
    let given = DynamicLibraries::new().with_loaded("libC.metallib", c.clone());
    stage(&client, given).with_loaded_dependencies(|name| {
        asked.push(name.to_owned());
        file(name)
    });
    assert_eq!(asked, ["libB.metallib", "libA.metallib"]);

    // An install name that does not load ends the loading, and the import is refused.
    let mut asked = Vec::new();
    let fd = stage(&client, DynamicLibraries::new()).with_loaded_dependencies(|name| {
        asked.push(name.to_owned());
        (name == "libC.metallib").then(|| c.clone())
    });
    assert_eq!(asked, ["libC.metallib", "libB.metallib"]);
    assert_eq!(compile(fd), Err(needs_library("libB.metallib")));
}
