//! The `metal2vulkan` binary (feature `cli`) on owned public fixtures and committed device
//! profiles: it writes what the library compiles, its stages, a refusal as `FALLBACK` with exit
//! status 2 and a repro bundle, its `spirv-val` and `llvm-as` steps, and no scratch file.
#![cfg(feature = "cli")]

use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicU64, Ordering};

use metal2vulkan::device::{Device, DeviceDescription, DevicePolicy};
use metal2vulkan::library::Library;
use metal2vulkan::metal::VertexFormat;
use metal2vulkan::pipeline::{
    ComputePipelineDescriptor, FunctionDescriptor, StageSource, VertexAttributeDescriptor,
    VertexBufferLayoutDescriptor, VertexDescriptor,
};
use metal2vulkan::Budget;

fn repo() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

fn fixture(name: &str) -> PathBuf {
    repo()
        .join("../metal2vulkan-validation/fixtures/public/engine")
        .join(name)
}

fn profile(stem: &str) -> PathBuf {
    repo().join(format!("../metal2vulkan-validation/profiles/{stem}.json"))
}

/// A private directory, removed when dropped.
struct Scratch(PathBuf);

impl Scratch {
    fn new() -> Scratch {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "m2v-cli-test-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&path).unwrap();
        Scratch(path)
    }

    fn join(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }

    /// The names in the directory, sorted.
    fn names(&self) -> Vec<String> {
        let mut names: Vec<String> = std::fs::read_dir(&self.0)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        names.sort();
        names
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// The binary with `args`, its repro bundles under `scratch/repros`, and its tools from the
/// environment unless `env` overrides them.
fn cli(scratch: &Scratch, args: &[&OsStr], env: &[(&str, &Path)]) -> Output {
    let mut command = Command::new(env!("CARGO_BIN_EXE_metal2vulkan"));
    command
        .args(args)
        .env("METAL2VULKAN_REPRO_DIR", scratch.join("repros"));
    for (name, value) in env {
        command.env(name, value);
    }
    command.output().unwrap()
}

fn os(s: &str) -> &OsStr {
    OsStr::new(s)
}

fn text(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

/// Whether the binary finds `spirv-val` as it would (the override, else `PATH`).
fn spirv_val_found() -> bool {
    let program = std::env::var_os("METAL2VULKAN_SPIRV_VAL").unwrap_or_else(|| "spirv-val".into());
    Command::new(program).arg("--version").output().is_ok()
}

/// The `llvm-as` the binary would run, when there is one.
fn llvm_as() -> Option<PathBuf> {
    let program: PathBuf = std::env::var_os("METAL2VULKAN_LLVM_AS")
        .unwrap_or_else(|| "llvm-as".into())
        .into();
    Command::new(&program)
        .arg("--version")
        .output()
        .ok()
        .map(|_| program)
}

/// A vertex descriptor file in `scratch` giving attribute 0 a `uint` from buffer 0.
fn vertex_descriptor(scratch: &Scratch) -> PathBuf {
    let mut descriptor = VertexDescriptor::new();
    descriptor
        .layouts
        .insert(0, VertexBufferLayoutDescriptor::new(Some(4)));
    descriptor
        .attributes
        .insert(0, VertexAttributeDescriptor::new(VertexFormat::UInt, 0, 0));
    let path = scratch.join("vertex-descriptor.json");
    std::fs::write(&path, serde_json::to_vec(&descriptor).unwrap()).unwrap();
    path
}

/// The one repro bundle under `repros`.
fn only_bundle(repros: &Path) -> PathBuf {
    let bundles = std::fs::read_dir(repros)
        .unwrap()
        .map(|e| e.unwrap().path())
        .collect::<Vec<_>>();
    let [bundle] = &bundles[..] else {
        panic!("one repro bundle: {bundles:?}");
    };
    bundle.clone()
}

#[test]
fn a_kernel_writes_the_pipeline_the_library_compiles() {
    let scratch = Scratch::new();
    let input = fixture("ob3_texture_array.air");
    let output = scratch.join("kernel.spv");
    let meta = scratch.join("kernel.json");
    let result = cli(
        &scratch,
        &[
            input.as_os_str(),
            output.as_os_str(),
            os("--profile"),
            profile("moltenvk-m3").as_os_str(),
            os("--emit-meta"),
            meta.as_os_str(),
        ],
        &[],
    );
    let stdout = text(&result.stdout);
    assert!(result.status.success(), "{}", text(&result.stderr));
    assert!(stdout.contains("verified: PASS"), "{stdout}");
    if spirv_val_found() {
        assert!(
            stdout.contains("spirv-val vulkan1.2, vulkan1.4: PASS"),
            "{stdout}"
        );
    } else {
        assert!(stdout.contains("spirv-val: not found"), "{stdout}");
    }

    let json = std::fs::read_to_string(profile("moltenvk-m3")).unwrap();
    let description = DeviceDescription::from_khronos_json(&json, None).unwrap();
    let device = Device::new(description, DevicePolicy::default()).unwrap();
    let library = Library::parse(&std::fs::read(&input).unwrap()).unwrap();
    let kernel = FunctionDescriptor::new(library.function("ob3_texture_array").unwrap());
    let descriptor = ComputePipelineDescriptor::new(StageSource::Function(kernel));
    let pipeline = device
        .compile_compute_pipeline(&descriptor, &Budget::standard())
        .unwrap();
    assert_eq!(
        std::fs::read(&output).unwrap(),
        pipeline.stage.spirv_bytes()
    );

    let meta: serde_json::Value = serde_json::from_slice(&std::fs::read(&meta).unwrap()).unwrap();
    assert_eq!(meta["kind"], "compute");
    assert_eq!(meta["stages"][0]["stage"], "comp");
    assert_eq!(meta["stages"][0]["entry_point"], pipeline.stage.entry_point);
    assert_eq!(
        meta["layout"],
        serde_json::to_value(&pipeline.layout).unwrap()
    );
    assert_eq!(
        meta["interface"],
        serde_json::to_value(&pipeline.interface).unwrap()
    );
    assert_eq!(scratch.names(), ["kernel.json", "kernel.spv"]);
}

#[test]
fn a_render_pipeline_writes_a_module_per_stage_from_its_vertex_descriptor() {
    let scratch = Scratch::new();
    let descriptor = vertex_descriptor(&scratch);
    let input = fixture("ob3_vertex_input.air");
    let output = scratch.join("pipeline.spv");
    let profile = profile("moltenvk-m3");
    let args = [
        input.as_os_str(),
        output.as_os_str(),
        os("--profile"),
        profile.as_os_str(),
    ];

    // Without its vertex descriptor, the vertex function's attribute has no format.
    let result = cli(&scratch, &args, &[]);
    assert_eq!(result.status.code(), Some(2), "{}", text(&result.stdout));
    assert!(text(&result.stderr).contains("--vertex-descriptor"));

    let mut with_descriptor = args.to_vec();
    with_descriptor.extend([os("--vertex-descriptor"), descriptor.as_os_str()]);
    let result = cli(&scratch, &with_descriptor, &[]);
    assert!(result.status.success(), "{}", text(&result.stderr));
    assert!(text(&result.stdout).contains("pipeline.vert.spv"));
    let module = std::fs::read(scratch.join("pipeline.vert.spv")).unwrap();
    assert_eq!(module.get(..4), Some(&0x0723_0203u32.to_le_bytes()[..]));
}

#[test]
fn a_refusal_is_a_fallback_with_a_repro_bundle_and_no_output() {
    let scratch = Scratch::new();
    let input = fixture("ob3_stage_input.air");
    let output = scratch.join("kernel.spv");
    let result = cli(
        &scratch,
        &[
            input.as_os_str(),
            output.as_os_str(),
            os("--profile"),
            profile("moltenvk-m3").as_os_str(),
        ],
        &[],
    );
    let stderr = text(&result.stderr);
    assert_eq!(result.status.code(), Some(2), "{stderr}");
    assert!(stderr.contains("FALLBACK"), "{stderr}");
    assert!(stderr.contains("StageInputDescriptor"), "{stderr}");
    assert!(stderr.contains("--stage-input"), "{stderr}");
    assert!(!output.exists());

    let bundle = only_bundle(&scratch.join("repros"));
    let mut names: Vec<String> = std::fs::read_dir(&bundle)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    assert_eq!(
        names,
        [
            "failure.txt",
            "input.air",
            "profile.json",
            "repro-command.txt",
            "repro.sh"
        ]
    );
    assert_eq!(
        std::fs::read(bundle.join("input.air")).unwrap(),
        std::fs::read(&input).unwrap()
    );
    let command = std::fs::read_to_string(bundle.join("repro-command.txt")).unwrap();
    assert_eq!(
        command,
        "metal2vulkan input.air out.spv --profile profile.json\n"
    );
    assert!(std::fs::read_to_string(bundle.join("failure.txt"))
        .unwrap()
        .contains("StageInputDescriptor"));
    assert_eq!(scratch.names(), ["repros"]);
}

/// The built binary runs and prints its help: the usage line, and the `METAL2VULKAN_*` registry
/// it reads.
#[test]
fn the_binary_prints_its_help() {
    let scratch = Scratch::new();
    for flag in ["--help", "-h"] {
        let result = cli(&scratch, &[os(flag)], &[]);
        let stdout = text(&result.stdout);
        assert_eq!(result.status.code(), Some(0), "{}", text(&result.stderr));
        assert!(stdout.starts_with("usage: metal2vulkan "), "{stdout}");
        assert!(stdout.contains("METAL2VULKAN_REPRO_DIR"), "{stdout}");
    }
    assert!(scratch.names().is_empty());
}

#[test]
fn usage_errors_are_fallbacks_without_a_bundle() {
    let scratch = Scratch::new();
    let input = fixture("ob3_texture_array.air");
    for (args, says) in [
        (
            vec![input.as_os_str()],
            "crates/metal2vulkan-validation/profiles",
        ),
        (
            vec![input.as_os_str(), os("--stag"), os("kernel")],
            "unrecognized option `--stag`",
        ),
        (
            vec![input.as_os_str(), os("--constant"), os("0:uchar:256")],
            "0:uchar:256",
        ),
    ] {
        let result = cli(&scratch, &args, &[]);
        let stderr = text(&result.stderr);
        assert_eq!(result.status.code(), Some(2), "{stderr}");
        assert!(stderr.contains("FALLBACK"), "{stderr}");
        assert!(stderr.contains(says), "{stderr}");
    }
    assert!(scratch.names().is_empty());
}

#[cfg(unix)]
#[test]
fn spirv_val_runs_unless_skipped_and_a_rejection_is_a_fallback() {
    use std::os::unix::fs::PermissionsExt;
    let scratch = Scratch::new();
    let rejecting = scratch.join("rejecting-spirv-val");
    std::fs::write(
        &rejecting,
        "#!/bin/sh\necho 'error: rejected' >&2\nexit 1\n",
    )
    .unwrap();
    std::fs::set_permissions(&rejecting, std::fs::Permissions::from_mode(0o700)).unwrap();
    let input = fixture("ob3_texture_array.air");
    let output = scratch.join("kernel.spv");
    let profile = profile("vk12-floor");
    let args = [
        input.as_os_str(),
        output.as_os_str(),
        os("--profile"),
        profile.as_os_str(),
    ];

    let result = cli(&scratch, &args, &[("METAL2VULKAN_SPIRV_VAL", &rejecting)]);
    let stderr = text(&result.stderr);
    assert_eq!(result.status.code(), Some(2), "{stderr}");
    assert!(
        stderr.contains("FALLBACK (spirv-val vulkan1.2 rejected"),
        "{stderr}"
    );
    assert!(stderr.contains("error: rejected"), "{stderr}");
    assert!(!output.exists());

    let missing = scratch.join("no-such-spirv-val");
    let result = cli(&scratch, &args, &[("METAL2VULKAN_SPIRV_VAL", &missing)]);
    assert!(result.status.success(), "{}", text(&result.stderr));
    assert!(text(&result.stdout).ends_with("verified: PASS; spirv-val: not found\n"));

    let mut skipped = args.to_vec();
    skipped.push(os("--no-spirv-val"));
    let result = cli(
        &scratch,
        &skipped,
        &[("METAL2VULKAN_SPIRV_VAL", &rejecting)],
    );
    assert!(result.status.success(), "{}", text(&result.stderr));
    assert!(text(&result.stdout).ends_with("verified: PASS\n"));
}

#[test]
fn llvm_text_is_assembled_by_llvm_as() {
    let scratch = Scratch::new();
    let input = fixture("kernel_vector_function_constant.ll");
    let output = scratch.join("kernel.spv");
    let profile = profile("moltenvk-m3");
    let args = [
        input.as_os_str(),
        output.as_os_str(),
        os("--profile"),
        profile.as_os_str(),
    ];

    let missing = scratch.join("no-such-llvm-as");
    let result = cli(&scratch, &args, &[("METAL2VULKAN_LLVM_AS", &missing)]);
    let stderr = text(&result.stderr);
    assert_eq!(result.status.code(), Some(2), "{stderr}");
    assert!(stderr.contains("needs llvm-as"), "{stderr}");

    let Some(llvm_as) = llvm_as() else {
        eprintln!("SKIPPED: llvm-as not on PATH and METAL2VULKAN_LLVM_AS unset");
        return;
    };
    // The kernel reads a function constant: it compiles with the constant undefined, as Metal's
    // does, and with its value.
    let result = cli(&scratch, &args, &[("METAL2VULKAN_LLVM_AS", &llvm_as)]);
    assert!(result.status.success(), "{}", text(&result.stderr));
    assert!(output.exists());
    std::fs::remove_file(&output).unwrap();
    let mut supplied = args.to_vec();
    supplied.extend([os("--constant"), os("0:uint4:1,2,3,4")]);
    let result = cli(&scratch, &supplied, &[("METAL2VULKAN_LLVM_AS", &llvm_as)]);
    assert!(result.status.success(), "{}", text(&result.stderr));
    assert!(output.exists());
}

#[cfg(unix)]
#[test]
fn a_bundle_carries_its_descriptor_files_and_replays_the_failure() {
    let scratch = Scratch::new();
    let descriptor = vertex_descriptor(&scratch);
    // The post-tessellation vertex function reads attribute 1 too, which the descriptor lacks.
    let input = fixture("ob3_control_point.air");
    let output = scratch.join("pipeline.spv");
    let profile = profile("moltenvk-m3");
    let result = cli(
        &scratch,
        &[
            input.as_os_str(),
            output.as_os_str(),
            os("--profile"),
            profile.as_os_str(),
            os("--vertex-descriptor"),
            descriptor.as_os_str(),
            os("--no-spirv-val"),
        ],
        &[],
    );
    let stderr = text(&result.stderr);
    assert_eq!(result.status.code(), Some(2), "{stderr}");
    assert!(
        stderr.contains("VertexAttribute { attribute: 1 }"),
        "{stderr}"
    );
    let bundle = only_bundle(&scratch.join("repros"));
    assert_eq!(
        std::fs::read(bundle.join("vertex-descriptor.json")).unwrap(),
        std::fs::read(&descriptor).unwrap()
    );
    assert_eq!(
        std::fs::read_to_string(bundle.join("repro-command.txt")).unwrap(),
        "metal2vulkan input.air out.spv --profile profile.json --vertex-descriptor \
         vertex-descriptor.json --no-spirv-val\n"
    );

    // The bundle's script replays the failure from the bundle alone.
    let binary = Path::new(env!("CARGO_BIN_EXE_metal2vulkan"));
    let path = std::env::join_paths(
        std::iter::once(binary.parent().unwrap().to_path_buf()).chain(std::env::split_paths(
            &std::env::var_os("PATH").unwrap_or_default(),
        )),
    )
    .unwrap();
    let replay = Command::new(bundle.join("repro.sh"))
        .env("PATH", path)
        .env("METAL2VULKAN_REPRO_DIR", scratch.join("replays"))
        .output()
        .unwrap();
    let replayed = text(&replay.stderr);
    assert_eq!(replay.status.code(), Some(2), "{replayed}");
    assert!(
        replayed.contains("VertexAttribute { attribute: 1 }"),
        "{replayed}"
    );
}
