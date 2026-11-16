//! The drm-shim recipe's two scripts, without docker or Mesa: `scripts/drm-shim-exec` preloads a
//! shim only inside the drm-shim image and refuses everywhere else, and `scripts/linux-gate
//! --recipe drm-shim` hands the command container the shim, chip variable, and ICD manifest the
//! pinned Mesa installs for the selected driver, refusing a chip spelling its shim cannot parse
//! before it touches docker.

use metal2vulkan_validation::ScratchDir;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn repo() -> PathBuf {
    fs::canonicalize(metal2vulkan_devtools::workspace_dir()).unwrap()
}

#[cfg(unix)]
fn executable(path: &Path, text: &str) {
    use std::os::unix::fs::PermissionsExt;
    fs::write(path, text).unwrap();
    fs::set_permissions(path, fs::Permissions::from_mode(0o755)).unwrap();
}

fn verdict(output: &Output) -> (Option<i32>, String) {
    (
        output.status.code(),
        String::from_utf8_lossy(&output.stderr).into_owned(),
    )
}

/// The first of `scripts/drm-shim-exec`'s conditions this process's environment fails, in the
/// wrapper's order and words, or `None` inside the drm-shim image (where it runs the command).
fn first_unmet_condition() -> Option<&'static str> {
    if std::env::consts::OS != "linux" {
        Some("not Linux")
    } else if !Path::new("/.dockerenv").exists() && !Path::new("/run/.containerenv").exists() {
        Some("not inside a container")
    } else if !Path::new("/etc/m2v-drm-shim-image").is_file() {
        Some("not the drm-shim image")
    } else if Path::new("/dev/dri").exists() {
        Some("/dev/dri exists")
    } else {
        None
    }
}

/// Runs `scripts/drm-shim-exec` with `vars` (and no other `M2V_DRM_SHIM_*` variable) on a command
/// that writes its environment to `ran`.
fn drm_shim_exec(vars: &[(&str, String)], ran: &Path) -> Output {
    let mut command = Command::new("bash");
    command
        .arg(repo().join("scripts/drm-shim-exec"))
        .args(["sh", "-c", "env > \"$1\"", "sh"])
        .arg(ran);
    for var in [
        "M2V_DRM_SHIM_LIBRARY",
        "M2V_DRM_SHIM_CHIP_VAR",
        "M2V_DRM_SHIM_CHIP",
        "M2V_DRM_SHIM_ICD",
    ] {
        command.env_remove(var);
    }
    for (key, value) in vars {
        command.env(key, value);
    }
    command.output().unwrap()
}

/// Outside the drm-shim image the wrapper refuses with exit 69, names the first condition the
/// environment fails, and never starts the command, whether the `M2V_DRM_SHIM_*` variables name
/// real files or are unset: the environment is checked before them. Here that is macOS on a
/// developer's host or the macOS runner, "not inside a container" on the Linux runner, and "not
/// the drm-shim image" in the gate container. Inside the drm-shim image the command runs with
/// exactly the shim, the chip variable, the ICD, and the shader cache off, and a variable that
/// selects no shim GPU is refused.
#[cfg(unix)]
#[test]
fn drm_shim_exec_preloads_only_inside_the_drm_shim_image() {
    let scratch = ScratchDir::new("drm-shim-exec").unwrap();
    let library = scratch.path().join("libnoop_drm_shim.so");
    let icd = scratch.path().join("icd.json");
    fs::write(&library, "not a library\n").unwrap();
    fs::write(&icd, "{}\n").unwrap();
    let vars = [
        ("M2V_DRM_SHIM_LIBRARY", library.display().to_string()),
        ("M2V_DRM_SHIM_CHIP_VAR", "AMDGPU_GPU_ID".to_string()),
        ("M2V_DRM_SHIM_CHIP", "navi31".to_string()),
        ("M2V_DRM_SHIM_ICD", icd.display().to_string()),
    ];
    let ran = scratch.path().join("ran");
    match first_unmet_condition() {
        Some(unmet) => {
            let unset: &[(&str, String)] = &[];
            for given in [&vars[..], unset] {
                let (code, stderr) = verdict(&drm_shim_exec(given, &ran));
                assert_eq!(code, Some(69), "{stderr}");
                assert!(stderr.contains(unmet), "expected {unmet:?}: {stderr}");
                assert!(!ran.exists(), "the command ran: {stderr}");
            }
        }
        None => {
            let (code, stderr) = verdict(&drm_shim_exec(&vars, &ran));
            assert_eq!(code, Some(0), "{stderr}");
            let env = fs::read_to_string(&ran).unwrap();
            for want in [
                format!("LD_PRELOAD={}", library.display()),
                "AMDGPU_GPU_ID=navi31".to_string(),
                format!("VK_DRIVER_FILES={}", icd.display()),
                "MESA_SHADER_CACHE_DISABLE=true".to_string(),
            ] {
                assert!(
                    env.lines().any(|line| line == want),
                    "{want} missing: {env}"
                );
            }
            fs::remove_file(&ran).unwrap();
            let mut other = vars.clone();
            other[1].1 = "LD_LIBRARY_PATH".to_string();
            let (code, stderr) = verdict(&drm_shim_exec(&other, &ran));
            assert_eq!(code, Some(69), "{stderr}");
            assert!(stderr.contains("selects no drm-shim GPU"), "{stderr}");
            assert!(!ran.exists(), "the command ran: {stderr}");
        }
    }
}

/// A stand-in `docker` for `scripts/linux-gate`: the daemon runs, the image exists, `create`
/// logs its arguments one per line after a `CREATE` line, `container inspect` reports what the
/// policy accepts for a run without a private corpus, and `start` drains the streamed tree.
const STUB_DOCKER: &str = r#"#!/bin/sh
case "$1" in
  create)
    echo CREATE >> "$STUB_LOG"
    for arg in "$@"; do printf '%s\n' "$arg" >> "$STUB_LOG"; done ;;
  container)
    case "$5" in *-cmd) role=cmd ;; *) role=fetch ;; esac
    case "$4" in
      *NetworkMode*) if [ "$role" = cmd ]; then echo none; else echo bridge; fi ;;
      *AutoRemove*) echo true ;;
      *Mounts*)
        printf 'volume\037%s\037/var/lib/docker/volumes/%s/_data\037/cache\037true\n' \
          "$STUB_VOLUME" "$STUB_VOLUME" ;;
    esac ;;
  start) cat > /dev/null ;;
esac
exit 0
"#;

/// Makes `command`'s git calls (its own, or `scripts/linux-gate`'s) read only the repository's
/// configuration, as `linux_gate_policy.rs`'s `own_git` does.
fn own_git<'a>(command: &'a mut Command, home: &Path) -> &'a mut Command {
    command
        .env("GIT_CONFIG_NOSYSTEM", "1")
        .env("GIT_CONFIG_GLOBAL", "/dev/null")
        .env("HOME", home)
        .env("XDG_CONFIG_HOME", home.join(".config"))
        .env_remove("GIT_CONFIG_PARAMETERS")
        .env_remove("GIT_CONFIG_COUNT")
        .env_remove("GIT_DIR")
        .env_remove("GIT_WORK_TREE")
        .env_remove("GIT_INDEX_FILE")
        .env_remove("GIT_COMMON_DIR")
}

/// `scripts/linux-gate --recipe drm-shim` gives both containers the driver's shim under
/// /opt/mesa/lib, its chip variable and chip, and its ICD manifest's pattern under
/// /opt/mesa/share/vulkan/icd.d (the names the pinned Mesa installs), from the drm-shim image; a
/// chip its shim cannot parse (an Intel name in capitals, a chipset id that is not hexadecimal, a
/// name with a space or a line break) exits 64 before docker is touched. The scripts run from a
/// throwaway repository holding them and the Dockerfile, so the test needs git but no checkout of
/// its own.
#[cfg(unix)]
#[test]
fn linux_gate_hands_the_command_the_selected_drivers_shim() {
    let scratch = ScratchDir::new("linux-gate-drm-shim").unwrap();
    let base = fs::canonicalize(scratch.path()).unwrap();
    let checkout = base.join("checkout");
    let stub = base.join("bin");
    let home = base.join("home");
    for dir in [checkout.join("scripts"), stub.clone(), home.clone()] {
        fs::create_dir_all(dir).unwrap();
    }
    for script in ["linux-gate", "linux-gate-policy"] {
        let text = fs::read_to_string(repo().join("scripts").join(script)).unwrap();
        executable(&checkout.join("scripts").join(script), &text);
    }
    fs::copy(
        repo().join("scripts/drm-shim.Dockerfile"),
        checkout.join("scripts/drm-shim.Dockerfile"),
    )
    .unwrap();
    for args in [
        &["init", "-q"][..],
        &["add", "."],
        &[
            "-c",
            "user.name=drm-shim test",
            "-c",
            "user.email=test@invalid",
            "-c",
            "commit.gpgsign=false",
            "-c",
            "core.hooksPath=/dev/null",
            "commit",
            "-q",
            "-m",
            "scripts",
        ],
    ] {
        let output = own_git(&mut Command::new("git"), &home)
            .arg("-C")
            .arg(&checkout)
            .args(args)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "git {args:?}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
    executable(&stub.join("docker"), STUB_DOCKER);
    let path = format!(
        "{}:{}",
        stub.display(),
        std::env::var("PATH").unwrap_or_default()
    );
    let log = base.join("docker.log");
    let gate = |driver: &str, chip: &str| {
        let _ = fs::remove_file(&log);
        let output = own_git(&mut Command::new("bash"), &home)
            .arg(checkout.join("scripts/linux-gate"))
            .args(["--cpus", "1", "--slot", "stub", "--platform", "arm64"])
            .args(["--recipe", "drm-shim", "--driver", driver, "--chip", chip])
            .args(["--", "true"])
            .env("PATH", &path)
            .env_remove("CARGO_TARGET_DIR")
            .env("STUB_LOG", &log)
            .env("STUB_VOLUME", "m2v-linux-gate-stub-arm64")
            .output()
            .unwrap();
        let (code, stderr) = verdict(&output);
        (code, stderr, fs::read_to_string(&log).unwrap_or_default())
    };
    for (driver, chip, shim, variable, icd) in [
        ("radv", "navi31", "amdgpu", "AMDGPU_GPU_ID", "radeon"),
        ("radv", "rembrandt", "amdgpu", "AMDGPU_GPU_ID", "radeon"),
        ("anv", "tgl", "intel", "INTEL_STUB_GPU_PLATFORM", "intel"),
        ("anv", "arl", "intel", "INTEL_STUB_GPU_PLATFORM", "intel"),
        ("nvk", "1b3", "nouveau", "NOUVEAU_CHIPSET", "nouveau"),
        ("nvk", "136", "nouveau", "NOUVEAU_CHIPSET", "nouveau"),
    ] {
        let (code, stderr, log) = gate(driver, chip);
        assert_eq!(code, Some(0), "{driver} {chip}: {stderr}");
        // The summary names the ICD drm-shim-exec selects, not only the command's lavapipe.
        let summary = format!(
            "driver={driver} chip={chip} icd={icd} under scripts/drm-shim-exec, lavapipe otherwise"
        );
        assert!(stderr.contains(&summary), "{driver} {chip}: {stderr}");
        let created: Vec<&str> = log.split("CREATE\n").filter(|c| !c.is_empty()).collect();
        assert_eq!(created.len(), 2, "{driver} {chip}: {log}");
        for container in created {
            let args: Vec<&str> = container.lines().collect();
            let image = args
                .iter()
                .find(|arg| arg.starts_with("m2v-linux-gate:"))
                .unwrap_or_else(|| panic!("no image: {container}"));
            assert!(
                image.starts_with("m2v-linux-gate:drm-shim-") && image.ends_with("-arm64"),
                "{image}"
            );
            for want in [
                format!("M2V_DRM_SHIM_LIBRARY=/opt/mesa/lib/lib{shim}_noop_drm_shim.so"),
                format!("M2V_DRM_SHIM_CHIP_VAR={variable}"),
                format!("M2V_DRM_SHIM_CHIP={chip}"),
                format!("M2V_DRM_SHIM_ICD_GLOB=/opt/mesa/share/vulkan/icd.d/{icd}_icd.*.json"),
            ] {
                assert!(
                    args.windows(2).any(|pair| pair == ["-e", want.as_str()]),
                    "{driver} {chip}: no -e {want} in {container}"
                );
            }
        }
    }
    for (driver, chip) in [
        ("anv", "TGL"),
        ("nvk", "GB203"),
        ("nvk", "0x1b3"),
        ("radv", "navi 31"),
        ("radv", "navi31\nx"),
        ("amdgpu", "navi31"),
    ] {
        let (code, stderr, log) = gate(driver, chip);
        assert_eq!(code, Some(64), "{driver} {chip:?} was accepted: {stderr}");
        assert!(log.is_empty(), "{driver} {chip:?} reached docker: {log}");
    }
}
