//! What `scripts/linux-gate` lets a container see, without docker: `scripts/linux-gate-policy`'s
//! verdicts on the scratch directory and on the mounts a daemon records, and `scripts/linux-gate`'s
//! own refusal of a scratch directory that overlaps the checkout or the private corpus before it
//! touches docker, the containers it never starts, and the files it streams. Every path is
//! synthetic: `scripts/linux-gate` runs from a throwaway git repository holding a copy of it
//! ([`checkout`]), so the tests pass from any copy of the crate (a worktree, a tarball, or
//! `scripts/linux-gate`'s own container copy, which has no `.git`).

use metal2vulkan_validation::ScratchDir;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

const VOLUME: &str = "m2v-linux-gate-test-arm64";
const PRIVATE_VOLUME: &str = "m2v-linux-gate-test-arm64-private";
const PRIVATE_DEST: &str = "/work/m2v/crates/metal2vulkan-validation/corpus/local";

fn repo() -> PathBuf {
    fs::canonicalize(metal2vulkan_devtools::workspace_dir()).unwrap()
}

/// A synthetic layout: a main checkout holding a linked worktree (the checkout), a private
/// corpus beside it, and a scratch directory apart from all three.
struct Layout {
    _scratch: ScratchDir,
    base: PathBuf,
    main: PathBuf,
    root: PathBuf,
    private: PathBuf,
    apart: PathBuf,
}

impl Layout {
    fn new(purpose: &str) -> Self {
        let scratch = ScratchDir::new(purpose).unwrap();
        let base = fs::canonicalize(scratch.path()).unwrap();
        let main = base.join("main");
        let root = main.join(".claude/worktrees/ev-test");
        let private = base.join("corpus");
        let apart = base.join("scratch");
        for dir in [
            root.clone(),
            private.join("local/sources"),
            private.join("local/library-modules"),
            apart.clone(),
        ] {
            fs::create_dir_all(dir).unwrap();
        }
        Self {
            _scratch: scratch,
            base,
            main,
            root,
            private,
            apart,
        }
    }

    fn args(&self, private: bool, scratch: Option<&Path>) -> Vec<String> {
        let mut args = vec![
            "--root".into(),
            self.root.display().to_string(),
            "--main-root".into(),
            self.main.display().to_string(),
        ];
        if private {
            args.extend(["--private".into(), self.private.display().to_string()]);
        }
        if let Some(scratch) = scratch {
            args.extend(["--scratch".into(), scratch.display().to_string()]);
        }
        args
    }
}

fn policy(args: &[String], stdin: &str) -> Output {
    use std::io::Write;
    let mut child = Command::new("bash")
        .arg(repo().join("scripts/linux-gate-policy"))
        .args(args)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .unwrap();
    // The policy may refuse on its arguments and exit before it reads its input; the write then
    // meets a closed pipe, and the refusal is in the output the caller checks.
    match child.stdin.take().unwrap().write_all(stdin.as_bytes()) {
        Err(e) if e.kind() == std::io::ErrorKind::BrokenPipe => {}
        written => written.unwrap(),
    }
    child.wait_with_output().unwrap()
}

/// `path` with the ASCII case of its last component's first letter flipped, when that spelling
/// names a directory (a case-insensitive file system, as macOS's default); `None` otherwise.
fn case_variant(path: &Path) -> Option<PathBuf> {
    let name = path.file_name()?.to_str()?;
    let at = name.find(|c: char| c.is_ascii_alphabetic())?;
    let mut flipped = name.to_string();
    let c = name[at..].chars().next()?;
    let other = if c.is_ascii_lowercase() {
        c.to_ascii_uppercase()
    } else {
        c.to_ascii_lowercase()
    };
    flipped.replace_range(at..at + 1, &other.to_string());
    let variant = path.with_file_name(flipped);
    variant.is_dir().then_some(variant)
}

#[cfg(unix)]
fn symlink(target: &Path, link: &Path) -> PathBuf {
    std::os::unix::fs::symlink(target, link).unwrap();
    link.to_path_buf()
}

#[cfg(unix)]
fn executable(path: &Path, text: &str) {
    use std::os::unix::fs::PermissionsExt;
    fs::write(path, text).unwrap();
    fs::set_permissions(path, fs::Permissions::from_mode(0o755)).unwrap();
}

/// Makes `command`'s git calls (its own, or `scripts/linux-gate`'s) read only the repository's
/// configuration: no system or global config, no default global excludes file (git looks for one
/// under `$XDG_CONFIG_HOME` or `~`, both pointed at the synthetic `home`), and no repository or
/// config override from the environment. A developer's global `*.ll` ignore rule would otherwise
/// decide what the stream test sees.
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

/// Runs `git args` in `checkout` with only its own configuration ([`own_git`]), a fixed identity,
/// and no signing or hooks; panics when git fails.
fn git(checkout: &Path, home: &Path, args: &[&str]) {
    let output = own_git(&mut Command::new("git"), home)
        .arg("-C")
        .arg(checkout)
        .args([
            "-c",
            "user.name=linux-gate test",
            "-c",
            "user.email=test@invalid",
        ])
        .args([
            "-c",
            "commit.gpgsign=false",
            "-c",
            "core.hooksPath=/dev/null",
        ])
        .args(args)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "git {args:?}: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

/// A throwaway git repository at `layout.base/checkout` for `scripts/linux-gate` to run from: a
/// copy of `scripts/linux-gate` and `scripts/linux-gate-policy`, a one-line
/// `scripts/linux-gate.Dockerfile`, a `.gitignore` of `*.ignored`, and `kept.txt`, all committed.
/// The `home` that [`own_git`] points git at is `layout.base/home`.
#[cfg(unix)]
fn checkout(layout: &Layout) -> (PathBuf, PathBuf) {
    let checkout = layout.base.join("checkout");
    fs::create_dir_all(checkout.join("scripts")).unwrap();
    for script in ["linux-gate", "linux-gate-policy"] {
        let text = fs::read_to_string(repo().join("scripts").join(script)).unwrap();
        executable(&checkout.join("scripts").join(script), &text);
    }
    fs::write(
        checkout.join("scripts/linux-gate.Dockerfile"),
        "FROM scratch\n",
    )
    .unwrap();
    fs::write(checkout.join(".gitignore"), "*.ignored\n").unwrap();
    fs::write(checkout.join("kept.txt"), "committed\n").unwrap();
    let home = layout.base.join("home");
    fs::create_dir_all(&home).unwrap();
    git(&checkout, &home, &["init", "-q"]);
    git(&checkout, &home, &["add", "."]);
    git(&checkout, &home, &["commit", "-q", "-m", "committed"]);
    (checkout, home)
}

fn verdict(output: &Output) -> (Option<i32>, String) {
    (
        output.status.code(),
        String::from_utf8_lossy(&output.stderr).into_owned(),
    )
}

/// One recorded mount, as `docker container inspect` writes it for the script.
fn mount(kind: &str, name: &str, source: &str, dest: &str, rw: bool) -> String {
    format!("{kind}\x1f{name}\x1f{source}\x1f{dest}\x1f{rw}\n")
}

fn volume(name: &str, dest: &str, rw: bool) -> String {
    mount(
        "volume",
        name,
        &format!("/var/lib/docker/volumes/{name}/_data"),
        dest,
        rw,
    )
}

fn bind(source: &Path, dest: &str, rw: bool) -> String {
    mount("bind", "", &source.display().to_string(), dest, rw)
}

fn mounts(
    layout: &Layout,
    role: &str,
    network: &str,
    private: bool,
    scratch: Option<&Path>,
    recorded: &[String],
) -> (Option<i32>, String) {
    let mut args = vec![
        "mounts".to_string(),
        "--role".into(),
        role.into(),
        "--network".into(),
        network.into(),
        "--autoremove".into(),
        "true".into(),
    ];
    args.extend(layout.args(private, scratch));
    args.extend([
        "--volume".into(),
        VOLUME.into(),
        "--private-volume".into(),
        PRIVATE_VOLUME.into(),
    ]);
    verdict(&policy(&args, &recorded.concat()))
}

/// The scratch directory is refused when it equals, contains, or lies inside the checkout, the
/// main checkout, or the private corpus, by file identity: also through a symlink, and through a
/// case variant where the file system ignores case. It is accepted apart from all three, through
/// a symlink too.
#[cfg(unix)]
#[test]
fn a_scratch_directory_may_not_overlap_a_protected_path() {
    let layout = Layout::new("linux-gate-paths");
    let paths = |scratch: &Path| {
        let mut args = vec!["paths".to_string()];
        args.extend(layout.args(true, Some(scratch)));
        verdict(&policy(&args, ""))
    };
    fs::create_dir_all(layout.main.join("validation")).unwrap();
    fs::create_dir_all(layout.root.join("target")).unwrap();
    let mut refused = vec![
        layout.root.clone(),
        layout.root.join("target"),
        layout.main.clone(),
        layout.main.join("validation"),
        layout.private.clone(),
        layout.private.join("local/sources"),
        layout.base.clone(),
        symlink(&layout.root, &layout.base.join("link-to-root")),
        symlink(
            &layout.main.join("validation"),
            &layout.base.join("link-into-main"),
        ),
        symlink(&layout.base, &layout.apart.join("link-to-base")),
    ];
    refused.extend(case_variant(&layout.main));
    refused.extend(case_variant(&layout.private));
    for refused in refused {
        let (code, stderr) = paths(&refused);
        assert_eq!(
            code,
            Some(64),
            "{} was accepted: {stderr}",
            refused.display()
        );
        assert!(stderr.contains("overlaps"), "{stderr}");
    }
    // A path that cannot be read counts as overlapping (it would be apart if it existed).
    let (code, stderr) = paths(&layout.base.join("missing"));
    assert_eq!(code, Some(64), "a missing scratch was accepted: {stderr}");
    assert_eq!(paths(&layout.apart), (Some(0), String::new()));
    let elsewhere = symlink(&layout.apart, &layout.base.join("link-to-apart"));
    assert_eq!(paths(&elsewhere), (Some(0), String::new()));
}

/// The containers of a run pass with exactly the mounts linux-gate gives them.
#[test]
fn the_mounts_a_run_gives_its_containers_pass() {
    let layout = Layout::new("linux-gate-mounts-pass");
    let scratch = Some(layout.apart.as_path());
    let fetch = [volume(VOLUME, "/cache", true)];
    assert_eq!(
        mounts(&layout, "fetch", "bridge", true, scratch, &fetch),
        (Some(0), String::new())
    );
    let plain = [
        volume(VOLUME, "/cache", true),
        bind(&layout.apart, "/scratch", true),
    ];
    assert_eq!(
        mounts(&layout, "cmd", "none", false, scratch, &plain),
        (Some(0), String::new())
    );
    let private = private_command(&layout, &layout.private);
    assert_eq!(
        mounts(&layout, "cmd", "none", true, scratch, &private),
        (Some(0), String::new())
    );
    // Docker Desktop records a macOS bind source under /host_mnt.
    let prefixed = private_command(
        &layout,
        &Path::new("/host_mnt").join(layout.private.strip_prefix("/").unwrap()),
    );
    assert_eq!(
        mounts(&layout, "cmd", "none", true, scratch, &prefixed),
        (Some(0), String::new())
    );
}

/// The command container of a --private-corpus run, with the private directories bound from
/// `source` (the private corpus as the daemon spells it).
fn private_command(layout: &Layout, source: &Path) -> Vec<String> {
    vec![
        volume(PRIVATE_VOLUME, "/cache", true),
        volume(VOLUME, "/cache-net", false),
        bind(
            &source.join("local/sources"),
            &format!("{PRIVATE_DEST}/sources"),
            false,
        ),
        bind(
            &source.join("local/library-modules"),
            &format!("{PRIVATE_DEST}/library-modules"),
            false,
        ),
        bind(&layout.apart, "/scratch", true),
    ]
}

/// Every mount outside that set is refused from the daemon's record: a bind of the checkout in
/// any container, any bind in a networked one, the private volume where a networked container
/// could reach it, the shared volume written by a private command, the private directories
/// read-write, networked, or one short, and a scratch bind that overlaps the private corpus or
/// the checkout.
#[test]
fn every_other_mount_is_refused() {
    let layout = Layout::new("linux-gate-mounts-refused");
    let apart = Some(layout.apart.as_path());
    let refused = |role: &str,
                   network: &str,
                   private: bool,
                   scratch: Option<&Path>,
                   recorded: Vec<String>| {
        let (code, stderr) = mounts(&layout, role, network, private, scratch, &recorded);
        assert_eq!(code, Some(70), "accepted {role} {network}: {recorded:?}");
        assert!(stderr.contains("refused"), "{stderr}");
    };
    let fetch = || vec![volume(VOLUME, "/cache", true)];
    let private = || private_command(&layout, &layout.private);

    // The checkout, bound read-only (the old /src), in either container.
    let mut with_src = fetch();
    with_src.push(bind(&layout.root, "/src", false));
    refused("fetch", "bridge", false, None, with_src);
    let mut with_src = private();
    with_src.push(bind(&layout.root, "/src", false));
    refused("cmd", "none", true, apart, with_src);
    let mut with_target = private();
    with_target.push(bind(&layout.root.join("target"), "/work/m2v/target", true));
    refused("cmd", "none", true, apart, with_target);
    // Any bind in a networked container, and the private volume there.
    let mut networked = fetch();
    networked.push(bind(&layout.apart, "/scratch", true));
    refused("fetch", "bridge", false, apart, networked);
    refused(
        "fetch",
        "bridge",
        true,
        None,
        vec![volume(PRIVATE_VOLUME, "/cache", true)],
    );
    // The command container networked, with or without the private corpus.
    refused("cmd", "bridge", true, apart, private());
    refused("cmd", "host", false, None, fetch());
    // A private command writing the shared volume, or the shared volume writable at /cache-net.
    let mut shared = private();
    shared[0] = volume(VOLUME, "/cache", true);
    refused("cmd", "none", true, apart, shared);
    let mut writable = private();
    writable[1] = volume(VOLUME, "/cache-net", true);
    refused("cmd", "none", true, apart, writable);
    // The private directories read-write, from another source, or one short.
    let mut rw = private();
    rw[2] = bind(
        &layout.private.join("local/sources"),
        &format!("{PRIVATE_DEST}/sources"),
        true,
    );
    refused("cmd", "none", true, apart, rw);
    let mut elsewhere = private();
    elsewhere[3] = bind(
        &layout.apart,
        &format!("{PRIVATE_DEST}/library-modules"),
        false,
    );
    refused("cmd", "none", true, apart, elsewhere);
    // Under any prefix but Docker Desktop's /host_mnt, the source is another directory.
    let prefixed = private_command(
        &layout,
        &Path::new("/mnt/other").join(layout.private.strip_prefix("/").unwrap()),
    );
    refused("cmd", "none", true, apart, prefixed);
    let mut short = private();
    short.remove(3);
    refused("cmd", "none", true, apart, short);
    // A private mount without --private-corpus.
    let mut unasked = vec![volume(VOLUME, "/cache", true)];
    unasked.push(bind(
        &layout.private.join("local/sources"),
        &format!("{PRIVATE_DEST}/sources"),
        false,
    ));
    refused("cmd", "none", false, None, unasked);
    // A scratch bind of a directory holding the private corpus, or of the main checkout, even
    // when the scratch directory was named so.
    let mut over_private = private();
    over_private[4] = bind(&layout.base, "/scratch", true);
    refused(
        "cmd",
        "none",
        true,
        Some(layout.base.as_path()),
        over_private,
    );
    refused(
        "cmd",
        "none",
        false,
        Some(layout.main.as_path()),
        vec![
            volume(VOLUME, "/cache", true),
            bind(&layout.main, "/scratch", true),
        ],
    );
    // A scratch bind of a directory other than the one named.
    let mut other = private();
    other[4] = bind(&layout.private, "/scratch", true);
    refused("cmd", "none", true, apart, other);
}

/// `scripts/linux-gate` refuses (exit 64) a scratch directory that overlaps the checkout it runs
/// from (a throwaway repository, [`checkout`]) or the private corpus before it looks for docker; a
/// scratch directory apart from both gets as far as docker (a stub that answers nothing, so the
/// script stops at exit 69 and starts no container).
#[cfg(unix)]
#[test]
fn linux_gate_refuses_an_overlapping_scratch_before_docker() {
    let layout = Layout::new("linux-gate-cli");
    let (repo, home) = checkout(&layout);
    let stub = layout.base.join("bin");
    fs::create_dir_all(&stub).unwrap();
    let docker = stub.join("docker");
    fs::write(&docker, "#!/bin/sh\nexit 99\n").unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(&docker, fs::Permissions::from_mode(0o755)).unwrap();
    }
    let path = format!(
        "{}:{}",
        stub.display(),
        std::env::var("PATH").unwrap_or_default()
    );
    let gate = |scratch: &Path| {
        let output = own_git(&mut Command::new("bash"), &home)
            .arg(repo.join("scripts/linux-gate"))
            .args(["--cpus", "1", "--private-corpus"])
            .arg(&layout.private)
            .arg("--scratch")
            .arg(scratch)
            .args(["--", "true"])
            .env("PATH", &path)
            .env_remove("CARGO_TARGET_DIR")
            .output()
            .unwrap();
        verdict(&output)
    };
    let mut refused = vec![
        repo.clone(),
        repo.join("scripts"),
        repo.parent().unwrap().to_path_buf(),
        layout.private.clone(),
        layout.private.join("local/sources"),
        layout.base.clone(),
        symlink(&repo, &layout.base.join("link-to-repo")),
    ];
    refused.extend(case_variant(&repo));
    for refused in refused {
        let (code, stderr) = gate(&refused);
        assert_eq!(
            code,
            Some(64),
            "{} was accepted: {stderr}",
            refused.display()
        );
        assert!(stderr.contains("overlaps"), "{stderr}");
    }
    let (code, stderr) = gate(&layout.apart);
    assert_eq!(code, Some(69), "{stderr}");
    assert!(stderr.contains("docker daemon is not running"), "{stderr}");
}

/// A stand-in `docker` for `scripts/linux-gate`: the image exists, creation succeeds, `container
/// inspect` reports the slot's volume at `/cache` plus what the scenario adds (a `/src` bind of
/// the checkout in the fetch or command container, or a networked command container), and
/// `start` reads the streamed tree (keeping it as `$STUB_ARCHIVES/<role>.tar` when that is set)
/// and logs the container it started.
const STUB_DOCKER: &str = r#"#!/bin/sh
case "$1" in
  container)
    case "$5" in *-cmd) role=cmd ;; *) role=fetch ;; esac
    case "$4" in
      *NetworkMode*)
        if [ "$role" = cmd ]; then echo "${STUB_CMD_NETWORK:-none}"; else echo bridge; fi ;;
      *AutoRemove*) echo true ;;
      *Mounts*)
        printf 'volume\037%s\037/var/lib/docker/volumes/%s/_data\037/cache\037true\n' \
          "$STUB_VOLUME" "$STUB_VOLUME"
        if [ "$role" = cmd ] && [ -n "${STUB_CMD_SRC:-}" ]; then
          printf 'bind\037\037%s\037/src\037false\n' "$STUB_CMD_SRC"
        fi
        if [ "$role" = fetch ] && [ -n "${STUB_FETCH_SRC:-}" ]; then
          printf 'bind\037\037%s\037/src\037false\n' "$STUB_FETCH_SRC"
        fi ;;
    esac ;;
  start)
    for arg in "$@"; do name=$arg; done
    case "$name" in *-cmd) role=cmd ;; *) role=fetch ;; esac
    if [ -n "${STUB_ARCHIVES:-}" ]; then
      cat > "$STUB_ARCHIVES/$role.tar"
    else
      cat > /dev/null
    fi
    echo "STARTED $name" >> "$STUB_LOG" ;;
esac
exit 0
"#;

/// A container the policy refuses is never started, and the script exits 70, whether it is the
/// fetch container or the command container (whose run sits in an `||` list, where set -e does
/// not stop a refusal); a run whose containers pass starts both. The script runs from a throwaway
/// repository ([`checkout`]), whose path is the `/src` bind the refused scenarios add.
#[cfg(unix)]
#[test]
fn linux_gate_never_starts_a_refused_container() {
    let layout = Layout::new("linux-gate-stub");
    let (repo, home) = checkout(&layout);
    let stub = layout.base.join("bin");
    fs::create_dir_all(&stub).unwrap();
    executable(&stub.join("docker"), STUB_DOCKER);
    let path = format!(
        "{}:{}",
        stub.display(),
        std::env::var("PATH").unwrap_or_default()
    );
    let run = |scenario: &[(&str, String)]| {
        let log = layout.base.join("starts.log");
        let _ = fs::remove_file(&log);
        let mut command = Command::new("bash");
        own_git(&mut command, &home)
            .arg(repo.join("scripts/linux-gate"))
            .args([
                "--cpus",
                "1",
                "--slot",
                "stub",
                "--platform",
                "arm64",
                "--",
                "true",
            ])
            .env("PATH", &path)
            .env_remove("CARGO_TARGET_DIR")
            .env("STUB_LOG", &log)
            .env("STUB_VOLUME", "m2v-linux-gate-stub-arm64");
        for (key, value) in scenario {
            command.env(key, value);
        }
        let output = command.output().unwrap();
        let starts = fs::read_to_string(&log).unwrap_or_default();
        let roles: Vec<&str> = starts
            .lines()
            .map(|line| line.rsplit('-').next().unwrap_or(""))
            .collect::<Vec<_>>();
        let (code, stderr) = verdict(&output);
        (code, stderr, roles.join(" "))
    };
    let src = repo.display().to_string();
    let (code, stderr, started) = run(&[]);
    assert_eq!((code, started.as_str()), (Some(0), "fetch cmd"), "{stderr}");
    for (scenario, want) in [
        (vec![("STUB_CMD_SRC", src.clone())], "fetch"),
        (vec![("STUB_CMD_NETWORK", "bridge".to_string())], "fetch"),
        (vec![("STUB_FETCH_SRC", src.clone())], ""),
    ] {
        let (code, stderr, started) = run(&scenario);
        assert_eq!(code, Some(70), "{scenario:?} started {started:?}: {stderr}");
        assert_eq!(started, want, "{scenario:?}: {stderr}");
        assert!(stderr.contains("refused"), "{stderr}");
    }
}

/// `scripts/linux-gate` streams the files in git's index and nothing else, to the networked fetch
/// container and the command container alike: a committed file and one opted in with
/// `git add -N`, never an untracked file that no ignore rule covers (an `.ll` dump at the root)
/// or an ignored one. The scripts run from a throwaway repository ([`checkout`]), and a stand-in
/// `docker` keeps each container's stdin. Every git call reads only that repository's
/// configuration (`own_git`), so no ignore rule of the developer's decides the outcome.
#[cfg(unix)]
#[test]
fn linux_gate_streams_only_the_index() {
    let layout = Layout::new("linux-gate-index");
    let (checkout, home) = checkout(&layout);
    fs::write(checkout.join("intent.txt"), "opted in\n").unwrap();
    git(&checkout, &home, &["add", "-N", "intent.txt"]);
    fs::write(checkout.join("x.ll"), "an untracked dump\n").unwrap();
    fs::write(checkout.join("local.ignored"), "an ignored file\n").unwrap();

    let stub = layout.base.join("bin");
    fs::create_dir_all(&stub).unwrap();
    executable(&stub.join("docker"), STUB_DOCKER);
    let archives = layout.base.join("archives");
    fs::create_dir_all(&archives).unwrap();
    let path = format!(
        "{}:{}",
        stub.display(),
        std::env::var("PATH").unwrap_or_default()
    );
    let output = own_git(&mut Command::new("bash"), &home)
        .arg(checkout.join("scripts/linux-gate"))
        .args([
            "--cpus",
            "1",
            "--slot",
            "stub",
            "--platform",
            "arm64",
            "--",
            "true",
        ])
        .env("PATH", &path)
        .env_remove("CARGO_TARGET_DIR")
        .env("STUB_LOG", layout.base.join("starts.log"))
        .env("STUB_VOLUME", "m2v-linux-gate-stub-arm64")
        .env("STUB_ARCHIVES", &archives)
        .output()
        .unwrap();
    let (code, stderr) = verdict(&output);
    assert_eq!(code, Some(0), "{stderr}");
    let want = [
        ".gitignore",
        "intent.txt",
        "kept.txt",
        "scripts/linux-gate",
        "scripts/linux-gate-policy",
        "scripts/linux-gate.Dockerfile",
    ];
    for role in ["fetch", "cmd"] {
        let listed = Command::new("tar")
            .arg("-t")
            .arg("-f")
            .arg(archives.join(format!("{role}.tar")))
            .output()
            .unwrap();
        assert!(
            listed.status.success(),
            "the {role} container's stream: {}",
            String::from_utf8_lossy(&listed.stderr)
        );
        let mut names: Vec<String> = String::from_utf8_lossy(&listed.stdout)
            .lines()
            .map(str::to_string)
            .collect();
        names.sort();
        assert_eq!(
            names, want,
            "the {role} container's stream (never x.ll or local.ignored)"
        );
    }
}
