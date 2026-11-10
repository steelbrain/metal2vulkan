//! Dynamic libraries through the runtime, on the device the cfg selects, under the validation
//! layer, against the answers Metal probes measured (an M3 Max on macOS 26.7 and an M4 on 27.0.1,
//! the same on both).
//!
//! The libraries are the probes' MSL functions written here as the AIR Apple's toolchain makes of
//! them (`llvm-as` assembles it), in the containers it writes (`metal2vulkan-devtools`): a dynamic
//! library is a metallib of file type 2 named by its install name, an absolute path into a
//! directory the test owns, and a library linked against dynamic libraries names their install
//! names (`HDYN`) and lists its imports in `air.dyld_flat_table`, beside the linker's tables of
//! those names. Each dynamic library is made by `Device::new_dynamic_library` and serialized to
//! the file its install name names, as the probes made them:
//!
//! - A: `dyl_scale(x) = 3x + 1`, `dyl_add(a, b) = a + b + 100`, `dyl_twice(x) = 2 dyl_scale(x)`;
//!   A-other: `9x`, `a + b + 900` and the same `dyl_twice`; B: `dyl_scale(x) = 5x`; B2:
//!   `dyl_scale(x) = 6x`, `dyl_add(a, b) = a + b + 500`; C, linked against A:
//!   `dyl_chain(x) = dyl_scale(x) + 0.25`; V: the variable `dyl_k = 7.5` and `dyl_get_k()`.
//! - The clients' kernels write three floats a thread `i` (of 8): `cl1` writes
//!   `dyl_scale(i)`, `dyl_add(i, 2)` and `dyl_twice(i)` (the probe's columns 0, 1 and 4),
//!   `cl_none` writes `i`, `cl_chain` `dyl_chain(i)`, and `cl_v` `dyl_k` and `dyl_get_k()`.
//! - The render client's vertex function writes `dyl_add(1, 2)` to a flat varying, and its
//!   fragment function stores `dyl_scale(x) + v` at pixel `x` of a 4x1 `R32Float` target (the
//!   probe interpolated the varying, whose value is the same at every vertex).

use super::at_plan;
use super::host_memory::on_device_with;
use super::host_memory::{on_device, run};
use super::read_buffer;
use super::render_state::texels;
use crate::air_library::AirLibrary;
use metal2vulkan::error::{
    ContainerError, DescriptorError, DiagName, DynamicLibraryProblem, Error as CompileError,
    InputErrorKind, RuntimeInput,
};
use metal2vulkan::library::LibraryType;
use metal2vulkan::metal::{PixelFormat, PrimitiveType, TextureUsage};
use metal2vulkan::pipeline::{
    ColorAttachmentDescriptor, ComputePipelineDescriptor, DynamicLibraries,
    RenderPipelineDescriptor, StageSource,
};
use metal2vulkan_devtools::llvm_assemble_text;
use metal2vulkan_devtools::metallib::{metallib_linked, wrap, MtlbFunction, MtlbLinkage};
use metal2vulkan_runtime::{
    ArchivedPipeline, BinaryArchiveDescriptor, BinaryArchiveMiss, ClearColor, Device,
    DynamicLibrary, DynamicLibraryError, Error, Invalid, Library, LoadAction, PipelineOption,
    PipelineOptions, Problem, RenderPassDescriptor, ResourceOptions, RuntimeGap, Size, StoreAction,
    TextureDescriptor,
};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

/// Threads of each dispatch.
const N: u32 = 8;

const HEADER: &str = "target datalayout = \"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32\"\ntarget triple = \"air64_v28-apple-macosx26.0.0\"\n\n";

/// The version metadata every module ends with.
const VERSIONS: &str = "!air.version = !{!90}\n!air.language_version = !{!91}\n\
                        !90 = !{i32 2, i32 8, i32 0}\n!91 = !{!\"Metal\", i32 4, i32 0, i32 0}\n";

const SCALE: &str = "_Z9dyl_scalef";
const ADD: &str = "_Z7dyl_addii";
const TWICE: &str = "_Z9dyl_twicef";
const CHAIN: &str = "_Z9dyl_chainf";
const GET_K: &str = "_Z9dyl_get_kv";

/// A directory the test owns: made empty, removed when dropped.
struct Scratch(PathBuf);

impl Scratch {
    fn new() -> Scratch {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let dir = std::env::temp_dir().join(format!(
            "m2v-lane-dynamic-libraries-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        Scratch(dir)
    }

    /// The install name (and file) of library `stem`.
    fn name(&self, stem: &str) -> String {
        self.0
            .join(format!("lib{stem}.metallib"))
            .to_string_lossy()
            .into_owned()
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// `text` (after [`HEADER`], before [`VERSIONS`]) assembled.
fn assemble(text: &str) -> Vec<u8> {
    llvm_assemble_text(&format!("{HEADER}{text}{VERSIONS}")).unwrap()
}

/// The tables Apple's linker writes into a module linked against `dependencies` (by install
/// name) that imports `imports` (each a constant expression of a declaration's address), as
/// probe 2's `libC` holds them: the names' offsets and lengths, the names, the imports, and
/// `llvm.used` keeping them.
fn dyld_tables(dependencies: &[&str], imports: &[&str]) -> String {
    let (mut entries, mut strings, mut length) = (Vec::new(), String::new(), 0usize);
    for name in dependencies {
        entries.push(format!(
            "{{ i32, i32 }} {{ i32 {length}, i32 {} }}",
            name.len()
        ));
        strings.push_str(name);
        strings.push_str("\\00");
        length += name.len() + 1;
    }
    let flat: Vec<String> = imports.iter().map(|i| format!("ptr {i}")).collect();
    format!(
        "@air.dyld_lib_table = hidden global [{} x {{ i32, i32 }}] [{}], section \
         \"air.dyld_lib_table\"\n\
         @air.dyld_str_table = hidden global [{length} x i8] c\"{strings}\", section \
         \"air.dyld_str_table\"\n\
         @air.dyld_flat_table = hidden global [{} x ptr] [{}], section \"air.dyld_flat_table\"\n\
         @llvm.used = appending global [3 x ptr] [ptr @air.dyld_lib_table, ptr \
         @air.dyld_str_table, ptr @air.dyld_flat_table], section \"llvm.metadata\"\n\n",
        entries.len(),
        entries.join(", "),
        flat.len(),
        flat.join(", "),
    )
}

/// The bytes of a metallib of `functions` (each a name, a type code and the module its bitcode
/// holds), with `linkage`.
fn container(functions: &[(&str, u8, &[u8])], linkage: &MtlbLinkage<'_>) -> Vec<u8> {
    let functions: Vec<MtlbFunction<'_>> = functions
        .iter()
        .map(|&(name, type_code, module)| MtlbFunction {
            name,
            type_code,
            bitcode: wrap(module),
            constants: Vec::new(),
            public: Vec::new(),
            tags: Vec::new(),
        })
        .collect();
    metallib_linked(&functions, linkage)
}

/// The dynamic library `stem` of `text`, exporting `exports`, linked against `dependencies`
/// and importing `imports`: made on `device` and serialized to the file of its install name.
fn dylib(
    device: &Device,
    dir: &Scratch,
    stem: &str,
    text: &str,
    exports: &[&str],
    (dependencies, imports): (&[&str], &[&str]),
) -> DynamicLibrary {
    let name = dir.name(stem);
    let module = assemble(text);
    // A dynamic library's exports share its one module (probe 1).
    let functions: Vec<(&str, u8, &[u8])> = exports.iter().map(|&e| (e, 3, &module[..])).collect();
    let linkage = MtlbLinkage {
        dynamic: true,
        install_name: Some(&name),
        dependencies: dependencies.to_vec(),
        imports: imports.to_vec(),
    };
    let library = device
        .new_library_with_data(&container(&functions, &linkage))
        .unwrap();
    assert_eq!(library.library_type(), LibraryType::Dynamic);
    assert_eq!(library.install_name(), Some(name.as_str()));
    let dynamic = device.new_dynamic_library(&library).unwrap();
    dynamic.serialize_to_url(Path::new(&name)).unwrap();
    dynamic
}

/// The functions of library A, or of A-other with `other`.
fn library_a(other: bool) -> String {
    let (mul, add, base) = if other {
        ("9.000000e+00", "0.000000e+00", 900)
    } else {
        ("3.000000e+00", "1.000000e+00", 100)
    };
    format!(
        "define float @{SCALE}(float %x) {{\n  %m = fmul float %x, {mul}\n  %r = fadd float %m, \
         {add}\n  ret float %r\n}}\n\n\
         define i32 @{ADD}(i32 %a, i32 %b) {{\n  %s = add i32 %a, %b\n  %r = add i32 %s, {base}\n  \
         ret i32 %r\n}}\n\n\
         define float @{TWICE}(float %x) {{\n  %s = call float @{SCALE}(float %x)\n  %r = fmul \
         float %s, 2.000000e+00\n  ret float %r\n}}\n\n"
    )
}

/// The functions of library B, or of B2 with `two`.
fn library_b(two: bool) -> String {
    let scale = format!(
        "define float @{SCALE}(float %x) {{\n  %r = fmul float %x, {}\n  ret float %r\n}}\n\n",
        if two { "6.000000e+00" } else { "5.000000e+00" }
    );
    match two {
        true => format!(
            "{scale}define i32 @{ADD}(i32 %a, i32 %b) {{\n  %s = add i32 %a, %b\n  %r = add i32 \
             %s, 500\n  ret i32 %r\n}}\n\n"
        ),
        false => scale,
    }
}

/// The `air.kernel` metadata of the kernel `entry`, taking a float buffer at index 0 and the
/// thread's position (metadata numbers 0 to 4).
fn kernel_metadata(entry: &str) -> String {
    format!(
        "!air.kernel = !{{!0}}\n!0 = !{{ptr @{entry}, !1, !2}}\n!1 = !{{}}\n!2 = !{{!3, !4}}\n\
         !3 = !{{i32 0, !\"air.buffer\", !\"air.location_index\", i32 0, i32 1, \
         !\"air.read_write\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 4, \
         !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"float\", \
         !\"air.arg_name\", !\"o\"}}\n\
         !4 = !{{i32 1, !\"air.thread_position_in_grid\", !\"air.arg_type_name\", !\"uint\", \
         !\"air.arg_name\", !\"i\"}}\n"
    )
}

/// A kernel `name` writing `values` (three float SSA names, computed by `body` from `%x`, the
/// thread's index as a float) at `o[3i]`.
fn kernel(name: &str, body: &str, values: [&str; 3]) -> String {
    let mut text = format!(
        "define void @{name}(ptr addrspace(1) %o, i32 %i) {{\n  %x = call float \
         @air.convert.f.f32.u.i32(i32 %i)\n{body}  %b0 = mul i32 %i, 3\n"
    );
    for (k, value) in values.iter().enumerate() {
        text.push_str(&format!(
            "  %b{k}x = add i32 %b0, {k}\n  %p{k} = getelementptr float, ptr addrspace(1) %o, \
             i32 %b{k}x\n  store float {value}, ptr addrspace(1) %p{k}, align 4\n"
        ));
    }
    text.push_str("  ret void\n}\n\n");
    text
}

/// The declarations the kernels below call.
fn declarations() -> String {
    format!(
        "declare float @air.convert.f.f32.u.i32(i32)\ndeclare float @air.convert.f.f32.s.i32(i32)\n\
         declare i32 @air.convert.u.i32.f.f32(float)\n\
         declare float @{SCALE}(float)\ndeclare i32 @{ADD}(i32, i32)\n\
         declare float @{TWICE}(float)\ndeclare float @{CHAIN}(float)\n\
         declare float @{GET_K}()\n\n"
    )
}

/// The client of `cl1` and `cl_none`, linked against `dependencies`.
fn client_cl1(device: &Device, dependencies: &[&str]) -> Library {
    let imports = [format!("@{SCALE}"), format!("@{ADD}"), format!("@{TWICE}")];
    let imports: Vec<&str> = imports.iter().map(String::as_str).collect();
    let cl1 = kernel(
        "cl1",
        &format!(
            "  %s = call float @{SCALE}(float %x)\n  %a = call i32 @{ADD}(i32 %i, i32 2)\n  %af = \
             call float @air.convert.f.f32.s.i32(i32 %a)\n  %t = call float @{TWICE}(float %x)\n"
        ),
        ["%s", "%af", "%t"],
    );
    let none = kernel("cl_none", "", ["%x", "0.000000e+00", "0.000000e+00"]);
    let module = |name: &str, kernel: &str| {
        format!(
            "{}{kernel}{}{}",
            dyld_tables(dependencies, &imports),
            declarations(),
            kernel_metadata(name)
        )
    };
    client(
        device,
        &[
            ("cl1", module("cl1", &cl1)),
            ("cl_none", module("cl_none", &none)),
        ],
        dependencies,
        &[SCALE, ADD, TWICE],
    )
}

/// An executable library of the kernels `kernels`, each a name and the text of its own module (a
/// metallib function's bitcode holds that entry alone), linked against `dependencies` and importing
/// `imports`.
fn client(
    device: &Device,
    kernels: &[(&str, String)],
    dependencies: &[&str],
    imports: &[&str],
) -> Library {
    let modules: Vec<Vec<u8>> = kernels.iter().map(|(_, text)| assemble(text)).collect();
    let functions: Vec<(&str, u8, &[u8])> = kernels
        .iter()
        .zip(&modules)
        .map(|((name, _), module)| (*name, 2, &module[..]))
        .collect();
    let linkage = MtlbLinkage {
        dynamic: false,
        install_name: Some("default.metallib"),
        dependencies: dependencies.to_vec(),
        imports: imports.to_vec(),
    };
    let library = device
        .new_library_with_data(&container(&functions, &linkage))
        .unwrap();
    assert_eq!(library.library_type(), LibraryType::Executable);
    assert_eq!(library.install_name(), None);
    library
}

/// The libraries `preloaded` as a stage preloads them.
fn preloading(preloaded: &[&DynamicLibrary]) -> DynamicLibraries {
    preloaded
        .iter()
        .fold(DynamicLibraries::new(), |libraries, l| {
            libraries.with_preloaded(l.preloaded_library())
        })
}

/// The compute descriptor of `kernel` of `library`, preloading `preloaded`.
fn descriptor(
    library: &Library,
    kernel: &str,
    preloaded: &[&DynamicLibrary],
) -> ComputePipelineDescriptor {
    let function = library.new_function_with_name(kernel).unwrap();
    ComputePipelineDescriptor::new(StageSource::Function(
        function
            .function_descriptor()
            .with_libraries(preloading(preloaded)),
    ))
}

/// What `kernel` of `library`, preloading `preloaded`, writes over `N` threads under `options`,
/// or the creation's error.
fn computed(
    library: &Library,
    kernel: &str,
    preloaded: &[&DynamicLibrary],
    options: &PipelineOptions,
) -> Result<Vec<f32>, Error> {
    let device = library.device();
    let state =
        device.new_compute_pipeline_state(&descriptor(library, kernel, preloaded), options)?;
    let out = device
        .new_buffer(u64::from(N) * 12, ResourceOptions::shared())
        .unwrap();
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(&state).unwrap();
    encoder.set_buffer(Some(&out), 0, 0).unwrap();
    encoder
        .dispatch_threads(Size::new(N.into(), 1, 1), Size::new(N.into(), 1, 1))
        .unwrap();
    encoder.end_encoding().unwrap();
    run(&command_buffer);
    Ok(read_buffer(&out, 0, out.length())
        .unwrap()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|w| f32::from_le_bytes(*w))
        .collect())
}

/// The plain creation's [`computed`].
fn plain(library: &Library, kernel: &str, preloaded: &[&DynamicLibrary]) -> Vec<f32> {
    computed(library, kernel, preloaded, &PipelineOptions::default())
        .unwrap_or_else(|e| panic!("{kernel}: {e}"))
}

/// The three values of each thread `i`.
fn rows(row: impl Fn(f32) -> [f32; 3]) -> Vec<f32> {
    (0..N).flat_map(|i| row(i as f32)).collect()
}

/// The creation's refusal for an install name that loaded nothing.
fn unloaded(install_name: &str) -> Error {
    Error::Compile(CompileError::NeedsInput(RuntimeInput::DynamicLibrary {
        install_name: DiagName::new(install_name),
    }))
}

/// Making, serializing and loading dynamic libraries answer as Metal's do: the type and install
/// name, a label set and none loaded, the file errors, and the refusals of an executable library
/// and of a dynamic library's functions.
#[test]
fn dynamic_libraries_are_made_and_loaded_as_metal_s_are() {
    on_device(|device| {
        let dir = Scratch::new();
        let a = dylib(
            device,
            &dir,
            "A",
            &library_a(false),
            &[SCALE, ADD, TWICE],
            (&[], &[]),
        );
        assert_eq!(a.install_name(), dir.name("A"));
        assert_eq!(a.label(), None);
        a.set_label("A");
        assert_eq!(a.label().as_deref(), Some("A"));
        assert_eq!(a.device().registry_id(), device.registry_id());

        // Loaded back from its file: its install name, no label, and it links as it did.
        let loaded = device
            .new_dynamic_library_with_url(Path::new(&dir.name("A")))
            .unwrap();
        assert_eq!(loaded.install_name(), dir.name("A"));
        assert_eq!(loaded.label(), None);
        let cl1 = client_cl1(device, &[&dir.name("A")]);
        let want = rows(|x| [3.0 * x + 1.0, 102.0 + x, 2.0 * (3.0 * x + 1.0)]);
        assert_eq!(plain(&cl1, "cl1", &[&loaded]), want);

        // The file errors: missing, not a library, a directory missing to serialize into.
        let missing = dir.0.join("missing.metallib");
        match device.new_dynamic_library_with_url(&missing) {
            Err(Error::DynamicLibrary(DynamicLibraryError::Read { path, kind, .. })) => {
                assert_eq!((path, kind), (missing, std::io::ErrorKind::NotFound))
            }
            other => panic!("a missing file: {other:?}"),
        }
        let garbage = dir.0.join("garbage.metallib");
        std::fs::write(&garbage, b"garbage").unwrap();
        assert_eq!(
            device.new_dynamic_library_with_url(&garbage).map(|_| ()),
            Err(Error::DynamicLibrary(DynamicLibraryError::InvalidFile {
                path: garbage
            }))
        );
        let nowhere = dir.0.join("missing").join("libA.metallib");
        assert_eq!(
            a.serialize_to_url(&nowhere),
            Err(Error::DynamicLibrary(DynamicLibraryError::InvalidFile {
                path: nowhere
            }))
        );

        // An executable library is no dynamic library, and a dynamic library's functions are no
        // functions to make.
        assert_eq!(
            device.new_dynamic_library(&cl1).map(|_| ()),
            Err(Error::Compile(CompileError::InvalidDescriptor(
                DescriptorError::DynamicLibrary {
                    problem: DynamicLibraryProblem::NotDynamic
                }
            )))
        );
        let a_library = device
            .new_library_with_data(&std::fs::read(dir.name("A")).unwrap())
            .unwrap();
        assert_eq!(a_library.library_type(), LibraryType::Dynamic);
        assert_eq!(
            a_library.new_function_with_name(SCALE).map(|_| ()),
            Err(Error::Invalid(Invalid::new(
                "new_function_with_name",
                Problem::DynamicLibraryFunction
            )))
        );
    });
}

/// `new_library_with_data` takes a metallib alone, as Metal's `newLibraryWithData:` does: a
/// kernel's bare AIR, raw or in a bitcode wrapper, is refused as any other
/// bytes that are no metallib are, and the translator's parse of the same AIR is a library the
/// device takes (`new_library_with_compile_library`), whose kernel runs as the metallib's does. A
/// file of bare AIR holds no dynamic library (`InvalidFile`), as Metal answers. Metal source is
/// refused typed (`NotImplemented(LibrarySource)`): the runtime builds no MSL compiler.
#[test]
fn a_library_comes_from_a_metallib_alone_as_metal_s_does() {
    on_device_with(at_plan, |device| {
        let text = format!(
            "{}declare float @air.convert.f.f32.u.i32(i32)\n\n{}",
            kernel("cl_none", "", ["%x", "0.000000e+00", "0.000000e+00"]),
            kernel_metadata("cl_none")
        );
        let module = assemble(&text);
        for bytes in [module.clone(), wrap(&module), b"garbage!".to_vec()] {
            match device.new_library_with_data(&bytes) {
                Err(Error::Compile(CompileError::InvalidInput(error))) => assert_eq!(
                    error.kind,
                    InputErrorKind::Container(ContainerError::BadMagic)
                ),
                other => panic!("bytes that are no metallib: {:?}", other.map(|_| ())),
            }
        }
        let linkage = MtlbLinkage {
            dynamic: false,
            install_name: Some("default.metallib"),
            dependencies: Vec::new(),
            imports: Vec::new(),
        };
        let metallib = device
            .new_library_with_data(&container(&[("cl_none", 2, &module[..])], &linkage))
            .unwrap();
        let air = device.new_library_with_air(&module).unwrap();
        // No MSL compiler: Metal source is refused typed (`newLibraryWithSource:`).
        assert_eq!(
            device
                .new_library_with_source("kernel void k() {}")
                .map(|_| ()),
            Err(Error::NotImplemented(RuntimeGap::LibrarySource))
        );
        let want = rows(|x| [x, 0.0, 0.0]);
        assert_eq!(plain(&metallib, "cl_none", &[]), want);
        assert_eq!(plain(&air, "cl_none", &[]), want);

        let dir = Scratch::new();
        for (stem, bytes) in [("raw", module.clone()), ("wrapped", wrap(&module))] {
            let path = PathBuf::from(dir.name(stem));
            std::fs::write(&path, bytes).unwrap();
            assert_eq!(
                device.new_dynamic_library_with_url(&path).map(|_| ()),
                Err(Error::DynamicLibrary(DynamicLibraryError::InvalidFile {
                    path
                }))
            );
        }
    });
}

/// A kernel's imports resolve in Metal's order, to Metal's values: the preloaded libraries in
/// order, then the libraries of the install names the client and they were linked against; a
/// library's own calls stay its own; an exported variable links; and a kernel whose imports
/// specialization deletes needs none of its dependencies' files.
#[test]
fn kernels_link_their_imports_as_metal_s_do() {
    on_device_with(at_plan, |device| {
        let dir = Scratch::new();
        let (na, nc, nv) = (dir.name("A"), dir.name("C"), dir.name("V"));
        let a = dylib(
            device,
            &dir,
            "A",
            &library_a(false),
            &[SCALE, ADD, TWICE],
            (&[], &[]),
        );
        let a_other = dylib(
            device,
            &dir,
            "Aother",
            &library_a(true),
            &[SCALE, ADD, TWICE],
            (&[], &[]),
        );
        let b = dylib(device, &dir, "B", &library_b(false), &[SCALE], (&[], &[]));
        let b2 = dylib(
            device,
            &dir,
            "B2",
            &library_b(true),
            &[SCALE, ADD],
            (&[], &[]),
        );
        let scale_import = format!("@{SCALE}");
        let c_text = format!(
            "{}define float @{CHAIN}(float %x) {{\n  %s = call float @{SCALE}(float %x)\n  %r = \
             fadd float %s, 2.500000e-01\n  ret float %r\n}}\n\ndeclare float @{SCALE}(float)\n\n",
            dyld_tables(&[&na], &[&scale_import])
        );
        let _c = dylib(device, &dir, "C", &c_text, &[CHAIN], (&[&na], &[SCALE]));
        let v_text = format!(
            "@dyl_k = addrspace(2) constant float 7.500000e+00, align 4\n\n\
             define float @{GET_K}() {{\n  %k = load float, ptr addrspace(2) @dyl_k, align 4\n  \
             ret float %k\n}}\n\n"
        );
        let _v = dylib(device, &dir, "V", &v_text, &[GET_K], (&[], &[]));

        // The client linked against A (probe 2b's CL1-A).
        let cl1 = client_cl1(device, &[&na]);
        let a_rows = |scale: fn(f32) -> f32| rows(move |x| [scale(x), 102.0 + x, 6.0 * x + 2.0]);
        assert_eq!(plain(&cl1, "cl1", &[]), a_rows(|x| 3.0 * x + 1.0));
        assert_eq!(plain(&cl1, "cl1", &[&a]), a_rows(|x| 3.0 * x + 1.0));
        // Preloaded first, the first that exports a symbol supplying it; A's `dyl_twice` calls
        // A's `dyl_scale` whatever supplies the kernel's.
        assert_eq!(plain(&cl1, "cl1", &[&b]), a_rows(|x| 5.0 * x));
        assert_eq!(
            plain(&cl1, "cl1", &[&b, &a_other]),
            rows(|x| [5.0 * x, 902.0 + x, 18.0 * x])
        );
        assert_eq!(
            plain(&cl1, "cl1", &[&a_other, &b]),
            rows(|x| [9.0 * x, 902.0 + x, 18.0 * x])
        );
        assert_eq!(
            plain(&cl1, "cl1", &[&b2, &b]),
            rows(|x| [6.0 * x, 502.0 + x, 6.0 * x + 2.0])
        );
        assert_eq!(plain(&cl1, "cl_none", &[]), rows(|x| [x, 0.0, 0.0]));
        // Linked against B, then A (CL1-B-A): B's `dyl_scale`, A's `dyl_add`.
        let cl1_ba = client_cl1(device, &[&dir.name("B"), &na]);
        assert_eq!(plain(&cl1_ba, "cl1", &[]), a_rows(|x| 5.0 * x));

        // A dependency's own import is searched as the kernel's are, preloaded first.
        let chain_text = format!(
            "{}{}{}{}",
            dyld_tables(&[&nc], &[&format!("@{CHAIN}")]),
            kernel(
                "cl_chain",
                &format!("  %c = call float @{CHAIN}(float %x)\n"),
                ["%c", "0.000000e+00", "0.000000e+00"]
            ),
            declarations(),
            kernel_metadata("cl_chain")
        );
        let chain = client(device, &[("cl_chain", chain_text)], &[&nc], &[CHAIN]);
        let chained = |add: f32, mul: f32| rows(move |x| [mul * x + add, 0.0, 0.0]);
        assert_eq!(plain(&chain, "cl_chain", &[]), chained(1.25, 3.0));
        assert_eq!(plain(&chain, "cl_chain", &[&b]), chained(0.25, 5.0));
        assert_eq!(plain(&chain, "cl_chain", &[&a_other]), chained(0.25, 9.0));

        // An exported variable is one object with the library's value.
        let k_import = "addrspacecast (ptr addrspace(2) @dyl_k to ptr)".to_owned();
        let v_client = format!(
            "{}@dyl_k = external addrspace(2) constant float, align 4\n\n{}{}{}",
            dyld_tables(&[&nv], &[&k_import, &format!("@{GET_K}")]),
            kernel(
                "cl_v",
                &format!(
                    "  %k = load float, ptr addrspace(2) @dyl_k, align 4\n  %g = call float \
                     @{GET_K}()\n"
                ),
                ["%k", "%g", "0.000000e+00"]
            ),
            declarations(),
            kernel_metadata("cl_v")
        );
        let v_lib = client(device, &[("cl_v", v_client)], &[&nv], &["dyl_k", GET_K]);
        assert_eq!(plain(&v_lib, "cl_v", &[]), rows(|_| [7.5, 7.5, 0.0]));

        // Dependency files at creation (probe 2b's section 5). With A's file gone, a kernel that
        // imports nothing is made, and one whose import survives is refused, even with A
        // preloaded.
        std::fs::remove_file(&na).unwrap();
        assert_eq!(plain(&cl1, "cl_none", &[]), rows(|x| [x, 0.0, 0.0]));
        let options = PipelineOptions::default();
        assert_eq!(
            computed(&chain, "cl_chain", &[], &options),
            Err(unloaded(&na))
        );
        assert_eq!(
            computed(&chain, "cl_chain", &[&a], &options),
            Err(unloaded(&na))
        );
        a.serialize_to_url(Path::new(&na)).unwrap();
        // A file of another library at an install name is loaded and used; one that is no
        // library loads nothing.
        let nb = dir.name("B");
        std::fs::copy(dir.name("B2"), &nb).unwrap();
        assert_eq!(
            plain(&cl1_ba, "cl1", &[]),
            rows(|x| [6.0 * x, 502.0 + x, 6.0 * x + 2.0])
        );
        std::fs::write(&nb, b"garbage").unwrap();
        assert_eq!(computed(&cl1_ba, "cl1", &[], &options), Err(unloaded(&nb)));
        b.serialize_to_url(Path::new(&nb)).unwrap();
        assert_eq!(plain(&cl1_ba, "cl1", &[]), a_rows(|x| 5.0 * x));
    });
}

/// Neither a preloaded library nor a dependency's contents is in a binary archive's key, as on
/// Metal (probe 2b's section 4): a creation that preloads other libraries hits an archive the
/// kernel was added to without them, and links them; a client linked against other install
/// names misses, naming its kernel.
#[test]
fn archives_key_no_dynamic_library() {
    on_device_with(at_plan, |device| {
        let dir = Scratch::new();
        let na = dir.name("A");
        let _a = dylib(
            device,
            &dir,
            "A",
            &library_a(false),
            &[SCALE, ADD, TWICE],
            (&[], &[]),
        );
        let b = dylib(device, &dir, "B", &library_b(false), &[SCALE], (&[], &[]));
        let cl1 = client_cl1(device, &[&na]);
        let archive = device
            .new_binary_archive(&BinaryArchiveDescriptor::default())
            .unwrap();
        archive
            .add_compute_pipeline_functions(&descriptor(&cl1, "cl1", &[]))
            .unwrap();
        let mut options = PipelineOptions::new(PipelineOption::FAIL_ON_BINARY_ARCHIVE_MISS);
        options.binary_archives = vec![archive];
        assert_eq!(
            computed(&cl1, "cl1", &[], &options).unwrap(),
            rows(|x| [3.0 * x + 1.0, 102.0 + x, 6.0 * x + 2.0])
        );
        assert_eq!(
            computed(&cl1, "cl1", &[&b], &options).unwrap(),
            rows(|x| [5.0 * x, 102.0 + x, 6.0 * x + 2.0])
        );
        let cl1_ba = client_cl1(device, &[&dir.name("B"), &na]);
        assert_eq!(
            computed(&cl1_ba, "cl1", &[], &options),
            Err(Error::BinaryArchiveMiss(BinaryArchiveMiss::new(
                "cl1",
                ArchivedPipeline::Compute
            )))
        );
    });
}

/// The render client: `rv_calls` and `rf_calls` calling the libraries, `rv_plain` and
/// `rf_plain` calling nothing, linked against `dependencies`.
fn render_client(device: &Device, dependencies: &[&str]) -> Library {
    let imports = [format!("@{SCALE}"), format!("@{ADD}")];
    let imports: Vec<&str> = imports.iter().map(String::as_str).collect();
    let vertex = |name: &str, varying: &str| {
        format!(
            "define <{{ <4 x float>, float }}> @{name}(i32 %vid) {{\n  %s = shl i32 %vid, 1\n  \
             %xi = and i32 %s, 2\n  %yi = and i32 %vid, 2\n  %xf = call float \
             @air.convert.f.f32.u.i32(i32 %xi)\n  %yf = call float \
             @air.convert.f.f32.u.i32(i32 %yi)\n  %x2 = fmul float %xf, 2.000000e+00\n  %x = \
             fadd float %x2, -1.000000e+00\n  %y2 = fmul float %yf, 2.000000e+00\n  %y = fadd \
             float %y2, -1.000000e+00\n  %p0 = insertelement <4 x float> <float 0.000000e+00, \
             float 0.000000e+00, float 0.000000e+00, float 1.000000e+00>, float %x, i64 0\n  %p \
             = insertelement <4 x float> %p0, float %y, i64 1\n{varying}  %r0 = insertvalue <{{ \
             <4 x float>, float }}> undef, <4 x float> %p, 0\n  %r = insertvalue <{{ <4 x \
             float>, float }}> %r0, float %v, 1\n  ret <{{ <4 x float>, float }}> %r\n}}\n\n"
        )
    };
    let fragment = |name: &str, color: &str| {
        format!(
            "define <4 x float> @{name}(<4 x float> %p, float %v) {{\n{color}  %r = \
             insertelement <4 x float> <float 0.000000e+00, float 0.000000e+00, float \
             0.000000e+00, float 1.000000e+00>, float %c, i64 0\n  ret <4 x float> %r\n}}\n\n"
        )
    };
    let tables = dyld_tables(dependencies, &imports);
    let module = |function: String, metadata: String| {
        assemble(&format!("{tables}{function}{}{metadata}", declarations()))
    };
    // Each function's bitcode holds that entry alone.
    let vertex_metadata = |name: &str| {
        format!(
            "!air.vertex = !{{!10}}\n!10 = !{{ptr @{name}, !12, !15}}\n!12 = !{{!13, !14}}\n\
             !13 = !{{!\"air.position\", !\"air.arg_type_name\", !\"float4\", !\"air.arg_name\", \
             !\"p\"}}\n\
             !14 = !{{!\"air.vertex_output\", !\"generated(1vf)\", !\"air.arg_type_name\", \
             !\"float\", !\"air.arg_name\", !\"v\"}}\n\
             !15 = !{{!16}}\n\
             !16 = !{{i32 0, !\"air.vertex_id\", !\"air.arg_type_name\", !\"uint\", \
             !\"air.arg_name\", !\"vid\"}}\n"
        )
    };
    let fragment_metadata = |name: &str| {
        format!(
            "!air.fragment = !{{!20}}\n!20 = !{{ptr @{name}, !22, !24}}\n!22 = !{{!23}}\n\
             !23 = !{{!\"air.render_target\", i32 0, i32 0, !\"air.arg_type_name\", !\"float4\", \
             !\"air.arg_name\", !\"color\"}}\n\
             !24 = !{{!25, !26}}\n\
             !25 = !{{i32 0, !\"air.position\", !\"air.center\", !\"air.no_perspective\", \
             !\"air.arg_type_name\", !\"float4\", !\"air.arg_name\", !\"p\"}}\n\
             !26 = !{{i32 1, !\"air.fragment_input\", !\"generated(1vf)\", !\"air.flat\", \
             !\"air.arg_type_name\", !\"float\", !\"air.arg_name\", !\"v\"}}\n"
        )
    };
    let modules = [
        module(
            vertex(
                "rv_calls",
                &format!(
                    "  %a = call i32 @{ADD}(i32 1, i32 2)\n  %v = call float \
                     @air.convert.f.f32.s.i32(i32 %a)\n"
                ),
            ),
            vertex_metadata("rv_calls"),
        ),
        module(
            vertex("rv_plain", "  %v = fadd float 0.000000e+00, 0.000000e+00\n"),
            vertex_metadata("rv_plain"),
        ),
        module(
            fragment(
                "rf_calls",
                &format!(
                    "  %px = extractelement <4 x float> %p, i64 0\n  %xi = call i32 \
                     @air.convert.u.i32.f.f32(float %px)\n  %xf = call float \
                     @air.convert.f.f32.u.i32(i32 %xi)\n  %s = call float @{SCALE}(float %xf)\n  \
                     %c = fadd float %s, %v\n"
                ),
            ),
            fragment_metadata("rf_calls"),
        ),
        module(
            fragment("rf_plain", "  %c = fadd float %v, 0.000000e+00\n"),
            fragment_metadata("rf_plain"),
        ),
    ];
    let functions = [
        ("rv_calls", 0, &modules[0][..]),
        ("rv_plain", 0, &modules[1][..]),
        ("rf_calls", 1, &modules[2][..]),
        ("rf_plain", 1, &modules[3][..]),
    ];
    let linkage = MtlbLinkage {
        dynamic: false,
        install_name: Some("default.metallib"),
        dependencies: dependencies.to_vec(),
        imports: vec![SCALE, ADD],
    };
    device
        .new_library_with_data(&container(&functions, &linkage))
        .unwrap()
}

/// The four texels a draw of `vertex` and `fragment` of `library` stores, each stage preloading
/// its own libraries.
fn drawn(
    library: &Library,
    (vertex, vertex_preloaded): (&str, &[&DynamicLibrary]),
    (fragment, fragment_preloaded): (&str, &[&DynamicLibrary]),
) -> Vec<f32> {
    let device = library.device();
    let stage = |name: &str, preloaded: &[&DynamicLibrary]| {
        StageSource::Function(
            library
                .new_function_with_name(name)
                .unwrap()
                .function_descriptor()
                .with_libraries(preloading(preloaded)),
        )
    };
    let mut pipeline = RenderPipelineDescriptor::new(stage(vertex, vertex_preloaded));
    pipeline.fragment = Some(stage(fragment, fragment_preloaded));
    pipeline.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::R32Float));
    let state = device
        .new_render_pipeline_state(&pipeline, &PipelineOptions::default())
        .unwrap();
    let mut target = TextureDescriptor::texture_2d(PixelFormat::R32Float, 4, 1, false);
    target.usage = TextureUsage::RENDER_TARGET;
    target.resource_options = ResourceOptions::shared();
    let target = device.new_texture(&target).unwrap();
    let mut pass = RenderPassDescriptor::new();
    let color = &mut pass.color_attachments[0];
    color.texture = Some(target.clone());
    color.load_action = LoadAction::Clear;
    color.clear_color = ClearColor::new(-7.0, 0.0, 0.0, 0.0);
    color.store_action = StoreAction::Store;
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
    encoder.set_render_pipeline_state(&state).unwrap();
    encoder
        .draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
        .unwrap();
    encoder.end_encoding().unwrap();
    run(&command_buffer);
    texels(&target, 4)
        .as_chunks::<4>()
        .0
        .iter()
        .map(|w| f32::from_le_bytes(*w))
        .collect()
}

/// Each render stage searches its own preloaded libraries, then the client's dependencies
/// (probe 2b's section 3): the vertex stage's `dyl_add` and the fragment stage's `dyl_scale`.
#[test]
fn render_stages_preload_their_own_libraries() {
    on_device_with(at_plan, |device| {
        let dir = Scratch::new();
        let _a = dylib(
            device,
            &dir,
            "A",
            &library_a(false),
            &[SCALE, ADD, TWICE],
            (&[], &[]),
        );
        let b = dylib(device, &dir, "B", &library_b(false), &[SCALE], (&[], &[]));
        let b2 = dylib(
            device,
            &dir,
            "B2",
            &library_b(true),
            &[SCALE, ADD],
            (&[], &[]),
        );
        let library = render_client(device, &[&dir.name("A")]);
        let calls = |v: &[&DynamicLibrary], f: &[&DynamicLibrary]| {
            drawn(&library, ("rv_calls", v), ("rf_calls", f))
        };
        assert_eq!(calls(&[], &[]), [104.0, 107.0, 110.0, 113.0]);
        assert_eq!(calls(&[&b2], &[]), [504.0, 507.0, 510.0, 513.0]);
        assert_eq!(calls(&[], &[&b2]), [103.0, 109.0, 115.0, 121.0]);
        assert_eq!(calls(&[&b2], &[&b]), [503.0, 508.0, 513.0, 518.0]);
        assert_eq!(
            drawn(&library, ("rv_plain", &[]), ("rf_plain", &[&b2])),
            [0.0; 4]
        );
    });
}
