//! A checkout's corpus tools read the tracked corpus of that checkout and the private sources from
//! another corpus root (`--private-corpus DIR`). The tracked corpus is a commit's data, written in
//! its schema; the private corpus is untracked and the same for every checkout. Reading another
//! commit's tracked shards fails naming the shard, the mismatch, and the remedy.

use metal2vulkan_validation::hash::sha256_bytes;
use metal2vulkan_validation::source::{public_source_set, write_source_shards};
use metal2vulkan_validation::ScratchDir;
use rusqlite::Connection;
use std::fs;
use std::path::Path;
use std::process::{Command, Output};

fn corpus_index(args: &[&str]) -> (Output, String) {
    let output = Command::new(bin_exe!("corpus-index"))
        .args(args)
        .output()
        .unwrap();
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    (output, text)
}

fn path(path: &Path) -> &str {
    path.to_str().unwrap()
}

#[test]
fn tools_read_this_checkouts_tracked_shards_and_another_roots_private_sources() {
    let scratch = ScratchDir::new("private-corpus-root").unwrap();
    // Another checkout's corpus: the private sources, and a case shard a newer commit wrote (a
    // committed case with a field this checkout's schema does not know).
    let other = scratch.path().join("other");
    let mut private = public_source_set()
        .unwrap()
        .rows
        .into_iter()
        .find(|row| row.stage == "Kernel")
        .unwrap();
    private.air_ll.push_str("\n; a private source\n");
    private.air_sha256 = sha256_bytes(private.air_ll.as_bytes());
    private.label = "private/source".into();
    write_source_shards(&other, [private.clone()]).unwrap();
    let committed = metal2vulkan_devtools::validation_dir().join("corpus/cases");
    let (shard, line) = fs::read_dir(&committed)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| {
            path.extension()
                .is_some_and(|extension| extension == "jsonl")
        })
        .find_map(|path| {
            let line = fs::read_to_string(&path)
                .unwrap()
                .lines()
                .next()?
                .to_owned();
            Some((path.file_name().unwrap().to_owned(), line))
        })
        .unwrap();
    let newer = format!(
        "{},\"a_field_of_a_newer_commit\":1}}\n",
        line.strip_suffix('}').unwrap()
    );
    fs::create_dir_all(other.join("cases")).unwrap();
    fs::write(other.join("cases").join(&shard), newer).unwrap();
    // This checkout's corpus: its tracked shards (none here) and no private sources.
    let tracked = scratch.path().join("tracked");
    fs::create_dir_all(&tracked).unwrap();
    let index = scratch.path().join("index.sqlite");

    // Reading the other commit's tracked shards fails, naming the shard and the remedy.
    let (output, text) = corpus_index(&["--corpus", path(&other), "--index", path(&index)]);
    assert!(!output.status.success(), "{text}");
    assert!(
        text.contains(&format!("cases/{}:1", shard.to_str().unwrap())),
        "{text}"
    );
    assert!(
        text.contains("unknown field `a_field_of_a_newer_commit`"),
        "{text}"
    );
    assert!(text.contains("--private-corpus DIR"), "{text}");
    assert!(!index.exists());

    // The checkout's own tracked corpus with the other root's private sources indexes the
    // private row at its byte location, from the other root's shard.
    let (output, text) = corpus_index(&[
        "--corpus",
        path(&tracked),
        "--private-corpus",
        path(&other),
        "--index",
        path(&index),
    ]);
    assert!(output.status.success(), "{text}");
    let location = Connection::open(&index)
        .unwrap()
        .query_row(
            "SELECT label, source_offset, source_length FROM sources WHERE air_sha256=?1",
            [&private.air_sha256],
            |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, Option<i64>>(1)?,
                    row.get::<_, Option<i64>>(2)?,
                ))
            },
        )
        .unwrap();
    assert_eq!(location.0, "private/source");
    assert!(location.1.is_some() && location.2.is_some(), "{location:?}");
    assert!(!tracked.join("local").exists());

    // A private root must hold private data.
    let (output, text) = corpus_index(&[
        "--corpus",
        path(&tracked),
        "--private-corpus",
        path(&tracked),
        "--index",
        path(&index),
    ]);
    assert!(!output.status.success(), "{text}");
    assert!(text.contains("no local/ directory"), "{text}");
}

/// A worker is a new process that never parsed its parent's `--private-corpus`: the authored-compile
/// census's case worker reads an authored case's source and linked library module from the private
/// root its parent was given, not from the checkout's own (empty) `local/`.
#[test]
fn a_worker_reads_private_data_from_its_parents_private_corpus() {
    use metal2vulkan_validation::case::{
        AuthoredCase, BufferResource, Comparison, Dispatch, ExecutionSafety,
        LinkedFunctionResource, OutputSelection, ResourceRole, Stage,
    };
    use metal2vulkan_validation::library_module::{merge_library_module_shards, LibraryModuleRow};
    use metal2vulkan_validation::source::SourceRow;

    let scratch = ScratchDir::new("private-corpus-worker").unwrap();
    // The private root holds the source, which references a linked function, and the library
    // module that defines it.
    let other = scratch.path().join("other");
    let air_ll = r#"
define void @main(ptr addrspace(1) %output) {
entry:
  %value = call i32 @linked.MTL_VISIBLE_FN_REF()
  store i32 %value, ptr addrspace(1) %output, align 4
  ret void
}
declare i32 @linked.MTL_VISIBLE_FN_REF() section "air.externally_defined"
!air.kernel = !{!0}
!air.visible_function_references = !{!4}
!0 = !{ptr @main, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_name", !"uint"}
!4 = !{!"air.visible_function_reference", ptr @linked.MTL_VISIBLE_FN_REF, !"linked"}
"#;
    let source = SourceRow {
        air_sha256: sha256_bytes(air_ll.as_bytes()),
        stage: "Kernel".into(),
        entry: "main".into(),
        air_ll: air_ll.into(),
        blob_b64: None,
        lib_sha256s: vec!["11".repeat(32)],
        label: "local/authored-reference.ll".into(),
    };
    write_source_shards(&other, [source.clone()]).unwrap();
    // `!air.visible` declares `linked` a visible function, which the engine links by name.
    let module_ll = r#"define i32 @linked() {
entry:
  ret i32 42
}
!air.visible = !{!0}
!0 = !{ptr @linked, !1, !3}
!1 = !{!2}
!2 = !{!"air.visible_output", !"air.arg_type_name", !"uint"}
!3 = !{}
"#;
    let module_sha256 = sha256_bytes(module_ll.as_bytes());
    // The checker's loop-free classifier decodes the retained blob, so it must be bitcode. CI
    // installs LLVM on both runners, so a missing `llvm-as` is a failure, not a skip.
    let bitcode = metal2vulkan_validation::check::loop_free::assemble(module_ll)
        .unwrap_or_else(|why| panic!("assemble the linked module: {why}"));
    let module_blob = base64::Engine::encode(&base64::engine::general_purpose::STANDARD, bitcode);
    merge_library_module_shards(
        &other,
        [LibraryModuleRow {
            module_sha256: module_sha256.clone(),
            air_ll: module_ll.into(),
            blob_b64: module_blob,
            lib_sha256s: vec!["22".repeat(32)],
            label: "local/linked.ll".into(),
        }],
    )
    .unwrap();
    // The checkout's tracked corpus holds the authored case that links the module.
    let tracked = scratch.path().join("tracked");
    fs::create_dir_all(&tracked).unwrap();
    let mut case = AuthoredCase {
        air_sha256: source.air_sha256.clone(),
        case_id: String::new(),
        name: "authored-reference".into(),
        entry: "main".into(),
        stage: Stage::Kernel,
        buffers: vec![BufferResource {
            binding: 0,
            role: ResourceRole::Output,
            bytes_b64: None,
            initial_bytes_b64: Some("q6urqw==".into()),
            rasterization_rate_map: None,
        }],
        argument_buffer_buffers: vec![],
        device_buffer_arrays: vec![],
        command_buffers: vec![],
        pipeline_states: vec![],
        threadgroup_memory: vec![],
        imageblock: None,
        fragment_imageblock: None,
        acceleration_structures: vec![],
        visible_function_references: vec![LinkedFunctionResource {
            module_sha256,
            function: "linked".into(),
        }],
        visible_function_tables: vec![],
        intersection_function_tables: vec![],
        argument_buffer_intersection_function_tables: vec![],
        textures: vec![],
        texture_arrays: vec![],
        argument_buffer_textures: vec![],
        argument_buffer_samplers: vec![],
        samplers: vec![],
        render_targets: vec![],
        depth_stencil: None,
        vertex_inputs: vec![],
        vertex_observation: None,
        varying_values: Vec::new(),
        kernel_stage_inputs: vec![],
        function_constants: vec![],
        dispatch: Some(Dispatch {
            grid: [1, 1, 1],
            threads_per_threadgroup: [1, 1, 1],
            stage_in_region: None,
        }),
        draw: None,
        tessellation: None,
        mesh_draw: None,
        output: OutputSelection::Buffer {
            binding: 0,
            offset: 0,
            length: 4,
        },
        compare: Comparison::Exact,
        execution_safety: ExecutionSafety::LoopFree,
        fuel: None,
        out_of_bounds: None,
        render_pass_rate_map: None,
        raster_sample_count: None,
        rationale: None,
        authored_by: Some("test".into()),
        public_source: None,
        no_effect: None,
        creation_bound: None,
    };
    case.case_id = case.computed_case_id().unwrap();
    let case_id = case.case_id.clone();
    metal2vulkan_validation::store::CorpusStore::new(&tracked)
        .put_case(case)
        .unwrap();
    let index = scratch.path().join("index.sqlite");
    let roots = [
        "--corpus",
        path(&tracked),
        "--private-corpus",
        path(&other),
        "--index",
        path(&index),
    ];
    let (output, text) = corpus_index(&roots);
    assert!(output.status.success(), "{text}");
    let output = Command::new(bin_exe!("corpus-triage"))
        .args(roots)
        .args(["--audit", "authored-compile", "--case"])
        .arg(&case_id)
        .args(["--profile", "vk12-floor"])
        .output()
        .unwrap();
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(output.status.success(), "{text}");
    assert!(
        text.contains(&format!(
            "authored-compile-cell\t{case_id}\tprofile=vk12-floor\toutcome=compiled\t"
        )),
        "{text}"
    );
    assert!(!tracked.join("local").exists());
}

/// The root a parent hands its workers is a worker's alone: a tool run on its own ignores a stray
/// `M2V_PRIVATE_CORPUS_ROOT`, and a worker refuses an inherited root without `local/`, as
/// `--private-corpus` refuses it.
#[test]
fn only_a_worker_adopts_an_inherited_private_corpus_root_and_validates_it() {
    let scratch = ScratchDir::new("private-corpus-inherited").unwrap();
    let other = scratch.path().join("other");
    let mut private = public_source_set()
        .unwrap()
        .rows
        .into_iter()
        .find(|row| row.stage == "Kernel")
        .unwrap();
    private
        .air_ll
        .push_str("\n; a private source another root holds\n");
    private.air_sha256 = sha256_bytes(private.air_ll.as_bytes());
    private.label = "private/inherited".into();
    write_source_shards(&other, [private.clone()]).unwrap();
    let tracked = scratch.path().join("tracked");
    fs::create_dir_all(&tracked).unwrap();
    let index = scratch.path().join("index.sqlite");

    // A tool run on its own reads the tracked root's own private data, whatever the variable says.
    for stray in [other.as_path(), tracked.as_path()] {
        if index.exists() {
            fs::remove_file(&index).unwrap();
        }
        let output = Command::new(bin_exe!("corpus-index"))
            .args(["--corpus", path(&tracked), "--index", path(&index)])
            .env("M2V_PRIVATE_CORPUS_ROOT", stray)
            .output()
            .unwrap();
        assert!(output.status.success(), "{output:?}");
        let indexed = Connection::open(&index)
            .unwrap()
            .query_row(
                "SELECT count(*) FROM sources WHERE air_sha256=?1",
                [&private.air_sha256],
                |row| row.get::<_, i64>(0),
            )
            .unwrap();
        assert_eq!(
            indexed,
            0,
            "a stray export of {} redirected the tool",
            stray.display()
        );
    }

    // A worker adopts the root its parent hands it, refused as the flag refuses it.
    let output = Command::new(bin_exe!("corpus-triage"))
        .arg("--translation-worker")
        .args([path(scratch.path()), path(&tracked), path(&index)])
        .env("M2V_PRIVATE_CORPUS_ROOT", &tracked)
        .stdin(std::process::Stdio::null())
        .output()
        .unwrap();
    let text = String::from_utf8_lossy(&output.stderr);
    assert!(!output.status.success(), "{text}");
    assert!(
        text.contains(&format!(
            "M2V_PRIVATE_CORPUS_ROOT {}: no local/ directory",
            tracked.display()
        )),
        "{text}"
    );
}
