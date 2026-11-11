use metal2vulkan_validation::{hash::sha256_bytes, source, ScratchDir};

/// A corpus whose every shard is invalid JSON: any read of an unrelated shard fails the command.
fn poisoned_corpus() -> ScratchDir {
    let scratch = ScratchDir::new("pipeline-selection").unwrap();
    let shards = source::source_shards_dir(scratch.path());
    std::fs::create_dir_all(&shards).unwrap();
    for index in 0..source::SHARD_COUNT {
        std::fs::write(
            source::source_shard_path(scratch.path(), index),
            "invalid JSON\n",
        )
        .unwrap();
    }
    scratch
}

#[test]
fn engine_selection_reads_only_the_selected_row_and_names_its_engine() {
    let scratch = poisoned_corpus();
    // A public fixture retains no bitcode: the compile worker assembles its AIR text and compiles
    // it, and `--compile-only` opens no device.
    let hash = sha256_bytes(include_bytes!(
        "../../fixtures/public/kernel_threadgroup_word.ll"
    ));
    let run = |args: &[&str]| {
        std::process::Command::new(env!("CARGO_BIN_EXE_corpus-pipeline"))
            .arg("--corpus")
            .arg(scratch.path())
            .args(args)
            .output()
            .unwrap()
    };
    let output = run(&["--profile", "vk12-floor", "--hash", &hash, "--compile-only"]);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert_eq!(stdout.lines().count(), 1, "{stdout}");
    assert!(stdout.starts_with(&hash), "{stdout}");
    let assembler = metal2vulkan_devtools::tool_version("llvm-as");
    if let Err(metal2vulkan_devtools::AssembleError::Missing(why)) = assembler {
        // Without an assembler the row is a tooling error, not a defect of the selection.
        eprintln!("SKIPPED the compile check: {why}");
        assert!(stdout.contains("\tERROR\t"), "{stdout}");
    } else {
        assert!(output.status.success(), "{stderr}");
        assert!(stdout.contains("\tCOMPILED\t"), "{stdout}");
        assert!(stderr.contains("compiled_only=1"), "{stderr}");
    }

    // The sweep needs a profile; --compile-only opens no device.
    for args in [
        &["--hash", hash.as_str(), "--pulled"][..],
        &["--hash", hash.as_str(), "--device-type", "cpu"][..],
        &["--hash", hash.as_str(), "--compile-only"][..],
        &[
            "--profile",
            "vk12-floor",
            "--hash",
            hash.as_str(),
            "--compile-only",
            "--device-type",
            "cpu",
        ][..],
    ] {
        assert!(!run(args).status.success(), "{args:?}");
    }
}

/// A fragment function that fetches its color attachment compiles on a profile with no ordered
/// form of the fetch (neither rasterization-order attachment access nor the pixel interlock):
/// through input attachments the consumer orders between draws alone, its row naming
/// `PrecisionNote::UnorderedFetchWithinDraw`.
#[test]
fn a_fetch_without_an_ordered_form_compiles_noted() {
    let scratch = poisoned_corpus();
    let hash = sha256_bytes(include_bytes!(
        "../../fixtures/public/engine/exf_fetch_blend.ll"
    ));
    let run = |args: &[&str]| {
        std::process::Command::new(env!("CARGO_BIN_EXE_corpus-pipeline"))
            .arg("--corpus")
            .arg(scratch.path())
            .args(args)
            .output()
            .unwrap()
    };
    if let Err(metal2vulkan_devtools::AssembleError::Missing(why)) =
        metal2vulkan_devtools::tool_version("llvm-as")
    {
        eprintln!("SKIPPED the compile check: {why}");
        return;
    }
    let args = ["--profile", "vk12-floor", "--hash", &hash, "--compile-only"];
    let compiled = String::from_utf8(run(&args).stdout).unwrap();
    assert_eq!(compiled.lines().count(), 1, "{compiled}");
    assert!(compiled.contains("\tCOMPILED\t"), "{compiled}");
    assert!(compiled.contains("UnorderedFetchWithinDraw"), "{compiled}");
}
