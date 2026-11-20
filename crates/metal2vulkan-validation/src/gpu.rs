//! GPU isolation: the hang tripwire, finite waits, and explicit device selection.
//!
//! The ordinary `cargo nextest run -p metal2vulkan-validation` never opens a device: every test
//! that would is compiled only under the non-default `--cfg m2v_gpu` (a real GPU) or
//! `--cfg m2v_cpu_icd` (a CPU Vulkan ICD such as lavapipe). They are cfgs, not features, because
//! the ordinary gate builds with `--all-features`.

pub mod compile;
mod device_type;
pub mod tripwire;
pub mod wait;

pub use device_type::DeviceType;

#[cfg(all(m2v_gpu, m2v_cpu_icd))]
compile_error!(
    "`--cfg m2v_gpu` and `--cfg m2v_cpu_icd` select different Vulkan devices for the same tests; \
     set exactly one"
);

/// The Vulkan device the tests compiled under `m2v_gpu` or `m2v_cpu_icd` execute on.
#[cfg(all(test, m2v_gpu))]
pub(crate) const TEST_DEVICE: DeviceType = DeviceType::Gpu;

/// The Vulkan device the tests compiled under `m2v_gpu` or `m2v_cpu_icd` execute on.
#[cfg(all(test, m2v_cpu_icd))]
pub(crate) const TEST_DEVICE: DeviceType = DeviceType::Cpu;

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};

    fn rust_sources(dir: &Path, out: &mut Vec<PathBuf>) {
        let mut entries = std::fs::read_dir(dir)
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .collect::<Vec<_>>();
        entries.sort();
        for path in entries {
            if path.is_dir() {
                rust_sources(&path, out);
            } else if path.extension().is_some_and(|extension| extension == "rs") {
                out.push(path);
            }
        }
    }

    /// Every device open goes through one of the functions that check the hang tripwire first:
    /// validation's Vulkan context, the Metal platform, and the runtime's device in the player
    /// and in the replay tests, each after the tripwire check in its own file. A new raw open
    /// anywhere else in the crate fails here, before it can run, and so does a runtime open
    /// ahead of its check.
    #[test]
    fn device_opens_exist_only_behind_the_tripwire_check() {
        let root = metal2vulkan_devtools::validation_dir();
        let mut files = Vec::new();
        rust_sources(&root.join("src"), &mut files);
        rust_sources(&root.join("tests"), &mut files);
        // Spelled in pieces so this file does not match itself.
        let opens = [
            concat!("MTLCreate", "SystemDefaultDevice("),
            concat!("MTLCopy", "AllDevices"),
            concat!(".create_", "instance("),
            concat!(".enumerate_", "physical_devices("),
            concat!(".create_", "device("),
            // The runtime's device creation (`metal2vulkan_runtime`).
            concat!("metal2vulkan_runtime::Device", "::new("),
        ];
        let allowed: &[(&str, &str, usize)] = &[
            ("src/candidate/platform/device.rs", opens[2], 1),
            ("src/candidate/platform/device.rs", opens[3], 1),
            ("src/candidate/platform/device.rs", opens[4], 1),
            ("src/metal/platform.rs", opens[0], 1),
            // The runtime's device: the player's and the replay tests'.
            ("src/bin/command-replay.rs", opens[5], 1),
            ("src/replay/rt_replay.rs", opens[5], 1),
        ];
        let checked = concat!("refuse_if", "_tripped(");
        for (file, open, _) in allowed.iter().filter(|(_, open, _)| *open == opens[5]) {
            let text = std::fs::read_to_string(root.join(file)).unwrap();
            let check = text.find(checked);
            assert!(
                check.is_some_and(|at| text.find(open).is_some_and(|open| at < open)),
                "{file} opens the runtime's device without the hang tripwire check before it"
            );
        }
        let mut found = Vec::new();
        for file in files {
            let text = std::fs::read_to_string(&file).unwrap();
            let relative = file
                .strip_prefix(&root)
                .unwrap()
                .to_string_lossy()
                .replace('\\', "/");
            for open in opens {
                let count = text.matches(open).count();
                if count != 0 {
                    found.push((relative.clone(), open, count));
                }
            }
        }
        let mut expected = allowed
            .iter()
            .map(|(file, open, count)| (file.to_string(), *open, *count))
            .collect::<Vec<_>>();
        expected.sort();
        found.sort();
        assert_eq!(found, expected);
    }

    /// Every Metal framework call that compiles and cannot be cancelled (pipeline creation,
    /// function specialization, a library from MSL source) runs under the compile deadline of
    /// `gpu/compile.rs`: its call opens at most two lines after a `compile::bounded(`. A new
    /// unbounded one anywhere in the crate fails here.
    #[test]
    fn metal_compilations_run_under_the_compile_deadline() {
        let root = metal2vulkan_devtools::validation_dir();
        let mut files = Vec::new();
        rust_sources(&root.join("src"), &mut files);
        rust_sources(&root.join("tests"), &mut files);
        // Spelled in pieces so this file does not match itself.
        let creations = [
            concat!(".newCompute", "PipelineState"),
            concat!(".newRender", "PipelineState"),
            concat!(".newFunctionWithName_", "constantValues"),
            concat!(".newLibraryWith", "Source"),
        ];
        let bounded = concat!("compile::", "bounded(");
        let (mut creations_seen, mut unbounded) = (0, Vec::new());
        for file in &files {
            let text = std::fs::read_to_string(file).unwrap();
            let lines = text.lines().collect::<Vec<_>>();
            for (at, line) in lines.iter().enumerate() {
                if !creations.iter().any(|creation| line.contains(creation)) {
                    continue;
                }
                creations_seen += 1;
                if !lines[at.saturating_sub(2)..=at]
                    .iter()
                    .any(|line| line.contains(bounded))
                {
                    unbounded.push(format!("{}:{}", file.display(), at + 1));
                }
            }
        }
        assert!(creations_seen > 0, "no Metal compilation found");
        assert_eq!(unbounded, Vec::<String>::new());
    }

    /// Every host `xcrun` tool runs in its own process group under a deadline: through
    /// `gpu/compile.rs`, under the compile deadline, or through a runner listed here with the
    /// bound it enforces. A direct spawn anywhere else in the crate (sources, tests, binaries)
    /// fails here, and so does a listed runner that no longer spawns `xcrun` or no longer
    /// supervises its process group.
    #[test]
    fn xcrun_runs_only_under_the_compile_deadline() {
        let root = metal2vulkan_devtools::validation_dir();
        let mut files = Vec::new();
        rust_sources(&root.join("src"), &mut files);
        rust_sources(&root.join("tests"), &mut files);
        // Spelled in pieces so this file does not match itself.
        let spawn = concat!("Command::new(\"", "xcrun\")");
        let supervised = concat!("MonitoredChild", "::spawn(");
        let runner = root.join("src/gpu/compile.rs");
        let text = std::fs::read_to_string(&runner).unwrap();
        assert!(
            text.contains(spawn) && text.contains(supervised),
            "{} no longer runs xcrun in a supervised process group",
            runner.display()
        );
        let spawning = files
            .iter()
            .filter(|file| **file != runner)
            .filter(|file| std::fs::read_to_string(file).unwrap().contains(spawn))
            .map(|file| file.display().to_string())
            .collect::<Vec<_>>();
        assert_eq!(spawning, Vec::<String>::new());
    }

    /// Every host wait on device work is bounded and goes through `gpu/wait.rs`. An unbounded
    /// wait anywhere else in the crate (sources, tests, binaries) fails here.
    #[test]
    fn device_waits_exist_only_in_the_bounded_wait_module() {
        let root = metal2vulkan_devtools::validation_dir();
        let mut files = Vec::new();
        rust_sources(&root.join("src"), &mut files);
        rust_sources(&root.join("tests"), &mut files);
        // Spelled in pieces so this file does not match itself.
        let waits = [
            concat!("wait_for", "_fences("),
            concat!("device_wait", "_idle("),
            concat!("queue_wait", "_idle("),
            concat!("waitUntil", "Completed"),
            concat!("waitUntil", "Scheduled"),
        ];
        let wait_module = root.join("src/gpu/wait.rs");
        let mut found = Vec::new();
        for file in &files {
            let text = std::fs::read_to_string(file).unwrap();
            if *file == wait_module {
                assert!(
                    !text.contains(concat!("u64", "::MAX")),
                    "the bounded wait module must not spell an infinite timeout"
                );
                continue;
            }
            for wait in waits {
                if text.contains(wait) {
                    found.push(format!("{} uses {wait}", file.display()));
                }
            }
        }
        assert!(files.contains(&wait_module));
        assert_eq!(found, Vec::<String>::new());
        let wait_source = std::fs::read_to_string(&wait_module).unwrap();
        assert_eq!(wait_source.matches(waits[0]).count(), 1);
    }
}
