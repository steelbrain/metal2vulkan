//! The pipeline creation log's descriptor and its SPIR-V dump (`pipeline/creation_log.rs`,
//! `METAL2VULKAN_LOG_PIPELINES`, `METAL2VULKAN_DUMP_PIPELINES`), on the device the cfg selects. A
//! child process of this test binary ([`dump_child`]) makes one render pipeline state of the owned
//! fixture `unspecialized_stage_functions`, its fragment function made with a constant and its
//! color blended, which makes its predicted `VkPipeline` at its creation
//! (`PipelineWaitPolicy::Exact`). With the dump directory set and the log variable unset, the
//! state's first line names each function, its bitcode's SHA-256 and its constant's type and
//! bytes, and the color's format, blend and write mask, and the directory holds each module the
//! line names, as `<hash>.spv`, of the words the line counts, and SPIR-V modules alone. With the dump variable empty and the
//! log variable unset, the child prints no line and writes no file.
//!
//! A device test: compiled only under the GPU safety cfgs (`m2v_gpu`, `m2v_cpu_icd`).

use super::blit_holds::{exact_waits_device, Scratch};
use super::kept_variants::library;
use crate::PipelineOptions;
use metal2vulkan::library::{FunctionConstantValue, FunctionConstantValues};
use metal2vulkan::metal::{BlendFactor, DataType, PixelFormat};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use std::collections::BTreeSet;
use std::ffi::OsStr;
use std::path::Path;
use std::process::Command;

/// The dump directory's variable.
const DUMP: &str = "METAL2VULKAN_DUMP_PIPELINES";

/// The log's variable.
const LOG: &str = "METAL2VULKAN_LOG_PIPELINES";

/// Where the child writes what the parent checks the line against: each function's name and its
/// bitcode's SHA-256, one per line.
const FACTS: &str = "M2V_PIPELINE_DUMP_FACTS";

/// What a line of the log starts with.
const LINE: &str = "metal2vulkan: pipeline creation: ";

/// Run [`dump_child`] in `cwd` with `dump` as the dump variable and the log variable unset, its
/// facts written to `facts`: its stderr.
fn child(cwd: &Path, dump: &OsStr, facts: &Path) -> String {
    let output = Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "tests::pipeline_dump::dump_child",
            "--ignored",
            "--nocapture",
            "--test-threads=1",
        ])
        .current_dir(cwd)
        .env(DUMP, dump)
        .env_remove(LOG)
        .env(FACTS, facts)
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
    assert!(output.status.success(), "the child failed: {stderr}");
    stderr
}

/// The files in `dir`, by name.
fn files(dir: &Path) -> BTreeSet<String> {
    std::fs::read_dir(dir)
        .unwrap()
        .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
        .collect()
}

/// With the dump directory set, the state's first line names its descriptor, and the directory
/// holds every module the lines name; with the dump variable empty and the log unset, no line and
/// no file (the module docs).
#[test]
fn the_log_names_the_descriptor_and_the_dump_holds_each_module() {
    let scratch = Scratch::new("pipeline-dump");
    let (on, off) = (scratch.0.join("on"), scratch.0.join("off"));
    std::fs::create_dir_all(&on).unwrap();
    std::fs::create_dir_all(&off).unwrap();
    let dir = scratch.0.join("dump");
    let facts = scratch.0.join("facts");

    let stderr = child(&on, dir.as_os_str(), &facts);
    let lines: Vec<&str> = stderr.lines().filter(|l| l.starts_with(LINE)).collect();
    let first = lines
        .iter()
        .find(|l| l.contains("; descriptor: "))
        .unwrap_or_else(|| panic!("no state's first line: {stderr}"));
    let facts = std::fs::read_to_string(&facts).unwrap();
    for function in facts.lines() {
        let (name, air) = function.split_once(' ').unwrap();
        assert!(
            first.contains(&format!("`{name}` air {air} constants {{")),
            "the line names {name} and its bitcode: {first}"
        );
    }
    for named in [
        "constants {fragment_tint: Float 00004040}",
        "colors [0: BGRA8Unorm blended: color One Add OneMinusSourceAlpha, alpha One Add \
         OneMinusSourceAlpha, write mask ",
        "; depth None; stencil None; raster samples 1",
    ] {
        assert!(first.contains(named), "the line names {named:?}: {first}");
    }
    // Every module a line names in full (`<hash> (<n> words)`) is in the directory, of its
    // words, and the directory holds SPIR-V modules alone.
    let mut named = BTreeSet::new();
    for line in &lines {
        for (at, _) in line.match_indices("spirv VK_SHADER_STAGE_") {
            let field = &line[at..];
            let value = &field[field.find(": ").unwrap() + 2..];
            let Some((hash, rest)) = value.split_at_checked(16) else {
                continue;
            };
            let Some(words) = rest
                .strip_prefix(" (")
                .and_then(|rest| rest.split_once(" words)"))
                .and_then(|(words, _)| words.parse::<usize>().ok())
            else {
                continue;
            };
            if !hash.chars().all(|c| c.is_ascii_hexdigit()) {
                continue;
            }
            let bytes = std::fs::read(dir.join(format!("{hash}.spv"))).unwrap();
            assert_eq!(
                bytes.len(),
                words * 4,
                "{hash}.spv holds the module's words"
            );
            named.insert(format!("{hash}.spv"));
        }
    }
    assert!(
        named.len() >= 2,
        "the lines name the stages' modules: {first}"
    );
    let dumped = files(&dir);
    assert!(named.is_subset(&dumped), "{named:?} among {dumped:?}");
    for file in &dumped {
        let bytes = std::fs::read(dir.join(file)).unwrap();
        assert!(file.ends_with(".spv"), "{file}: a module's file");
        assert_eq!(bytes[..4], 0x0723_0203u32.to_le_bytes(), "{file} is SPIR-V");
    }
    println!(
        "PIPELINE-DUMP: {} line(s), {} module(s) dumped; first: {first}",
        lines.len(),
        dumped.len()
    );

    let stderr = child(&off, OsStr::new(""), &scratch.0.join("facts-off"));
    assert!(
        !stderr.lines().any(|l| l.starts_with(LINE)),
        "no line with the log off: {stderr}"
    );
    assert!(files(&off).is_empty(), "no file with the dump off");
}

/// The child of [`the_log_names_the_descriptor_and_the_dump_holds_each_module`], run as its own
/// process: makes one state (the module docs) and writes its functions' names and their bitcode's
/// SHA-256 to the file the parent names.
#[test]
#[ignore = "a child process of the_log_names_the_descriptor_and_the_dump_holds_each_module"]
fn dump_child() {
    let facts = std::env::var_os(FACTS).expect("the parent names the facts' file");
    let scratch = Scratch::new("pipeline-dump-child");
    let device = exact_waits_device(&scratch.0);
    let library = library(&device, "unspecialized_stage_functions");
    let vertex = library
        .new_function_with_name("unspecialized_vertex_free")
        .unwrap();
    let tint = FunctionConstantValue::new(DataType::Float, 3.0f32.to_le_bytes().to_vec());
    let fragment = library
        .new_function_with_name_constant_values(
            "unspecialized_fragment",
            &FunctionConstantValues::defaults().with_name("fragment_tint", tint),
        )
        .unwrap();
    let (vertex, fragment) = (vertex.function_descriptor(), fragment.function_descriptor());
    let hex = |bytes: &[u8]| bytes.iter().map(|b| format!("{b:02x}")).collect::<String>();
    let written: String = [&vertex, &fragment]
        .iter()
        .map(|d| {
            format!(
                "{} {}\n",
                d.function.info().name,
                hex(&d.function.info().air_hash)
            )
        })
        .collect();
    std::fs::write(facts, written).unwrap();
    let mut descriptor = RenderPipelineDescriptor::new(StageSource::Function(vertex));
    descriptor.fragment = Some(StageSource::Function(fragment));
    let mut color = ColorAttachmentDescriptor::new(PixelFormat::BGRA8Unorm);
    color.blending_enabled = true;
    color.source_rgb_blend_factor = BlendFactor::One;
    color.destination_rgb_blend_factor = BlendFactor::OneMinusSourceAlpha;
    color.source_alpha_blend_factor = BlendFactor::One;
    color.destination_alpha_blend_factor = BlendFactor::OneMinusSourceAlpha;
    descriptor.color_attachments[0] = Some(color);
    let state = device
        .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap();
    drop(state);
    drop(device);
}
