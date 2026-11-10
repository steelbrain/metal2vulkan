//! The external-tool runner of the product's tests and of validation: `spirv-val`, `spirv-dis`,
//! `llvm-as` and the like, each resolved through its `METAL2VULKAN_<TOOL>` override, then the
//! usual LLVM install directories, then `PATH`. The product runs no external tool, and (S4) holds
//! no runner either: this crate's [`crate::spirv`] runs the L3 checks for the product's tests and
//! for validation alike.

use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

/// A tool that could not be launched, or that ran and failed.
#[derive(Debug)]
pub enum ToolError {
    Launch {
        tool: String,
        executable: PathBuf,
        source: std::io::Error,
    },
    Exit {
        tool: String,
        output: std::process::Output,
    },
}

impl std::fmt::Display for ToolError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Launch {
                tool,
                executable,
                source,
            } => write!(
                formatter,
                "cannot run {tool} ({}): {source}",
                executable.display()
            ),
            Self::Exit { tool, output } if output.status.code().is_none() => write!(
                formatter,
                "{tool} killed by signal:\n{}",
                String::from_utf8_lossy(&output.stderr)
            ),
            Self::Exit { tool, output } => write!(
                formatter,
                "{tool} failed ({}):\n{}",
                output.status,
                String::from_utf8_lossy(&output.stderr)
            ),
        }
    }
}

impl std::error::Error for ToolError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Launch { source, .. } => Some(source),
            Self::Exit { .. } => None,
        }
    }
}

/// Run a tool and collect stdout and stderr. Failed launches retain their I/O source;
/// unsuccessful processes retain status and both streams.
pub fn run<S: AsRef<OsStr>>(cmd: &str, args: &[S]) -> Result<(Vec<u8>, Vec<u8>), ToolError> {
    run_with_stdin(cmd, args, Stdio::null())
}

/// [`run`] with `stdin` as the tool's standard input.
pub fn run_with_stdin<S: AsRef<OsStr>>(
    cmd: &str,
    args: &[S],
    stdin: Stdio,
) -> Result<(Vec<u8>, Vec<u8>), ToolError> {
    let executable = tool_bin(cmd);
    let output = Command::new(&executable)
        .args(args)
        .stdin(stdin)
        .output()
        .map_err(|source| ToolError::Launch {
            tool: cmd.to_owned(),
            executable,
            source,
        })?;
    if output.status.success() {
        Ok((output.stdout, output.stderr))
    } else {
        Err(ToolError::Exit {
            tool: cmd.to_owned(),
            output,
        })
    }
}

/// The path override of external tool `cmd` (`llvm-as`, `spirv-val`, ...): `METAL2VULKAN_<CMD>`,
/// uppercased with `-` as `_`. That is the `METAL2VULKAN_<TOOL>` family the product's `env_vars`
/// registers and reads the same way (its `tool_path_override`); this crate builds from its own
/// directory, so it spells the name here, and the product's tests hold the two spellings together.
pub fn tool_path_override(cmd: &str) -> Option<std::ffi::OsString> {
    std::env::var_os(format!(
        "METAL2VULKAN_{}",
        cmd.replace('-', "_").to_ascii_uppercase()
    ))
}

fn tool_bin(cmd: &str) -> PathBuf {
    if let Some(path) = tool_path_override(cmd) {
        return PathBuf::from(path);
    }
    for dir in [
        "/opt/homebrew/opt/llvm/bin",
        "/usr/local/opt/llvm/bin",
        "/opt/homebrew/bin",
        "/usr/local/bin",
    ] {
        let candidate = Path::new(dir).join(cmd);
        if candidate.is_file() {
            return candidate;
        }
    }
    PathBuf::from(cmd)
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;

    #[test]
    fn tool_runner_reports_output_exit_and_signal_without_retrying() {
        let (stdout, stderr) =
            run("/bin/sh", &["-c", "printf output; printf diagnostic >&2"]).unwrap();
        assert_eq!(stdout, b"output");
        assert_eq!(stderr, b"diagnostic");
        let failure = run(
            "/bin/sh",
            &["-c", "printf partial; printf rejected >&2; exit 7"],
        )
        .unwrap_err();
        assert!(failure.to_string().contains("exit status: 7"), "{failure}");
        let ToolError::Exit { output, .. } = &failure else {
            panic!("{failure}");
        };
        assert_eq!(output.status.code(), Some(7));
        assert_eq!(output.stdout, b"partial");
        assert_eq!(output.stderr, b"rejected");
        let signal = run("/bin/sh", &["-c", "kill -TERM $$"]).unwrap_err();
        let ToolError::Exit { output, .. } = signal else {
            panic!("expected process termination");
        };
        use std::os::unix::process::ExitStatusExt;
        assert_eq!(output.status.signal(), Some(15));
        let missing = std::env::temp_dir().join(format!("m2v-missing-tool-{}", std::process::id()));
        std::fs::create_dir(&missing).unwrap();
        let executable = missing.join("absent");
        let error = run(executable.to_str().unwrap(), &[] as &[&str]).unwrap_err();
        std::fs::remove_dir(missing).unwrap();
        let ToolError::Launch {
            source,
            executable: reported,
            ..
        } = &error
        else {
            panic!("{error}");
        };
        assert_eq!(reported, &executable);
        assert_eq!(source.kind(), std::io::ErrorKind::NotFound);
        assert_eq!(
            std::error::Error::source(&error)
                .unwrap()
                .downcast_ref::<std::io::Error>()
                .unwrap()
                .kind(),
            source.kind()
        );
    }
}
