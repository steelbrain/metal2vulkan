//! FALLBACK repro bundles (feature `serde`): a directory that replays one refusal with the
//! `metal2vulkan` command line. The command line writes one for a failure after it read its
//! inputs; a runtime may write one for a library or pipeline creation it refuses, where
//! `METAL2VULKAN_REPRO_DIR` names a directory (`env_vars`). Not consumer API: the binary's and a
//! runtime's, hidden as `env_vars` is.
//!
//! A bundle holds the input (`input.metallib`, `input.air`, `input.ll` or `input.bin`), the device
//! file (`profile.json`, which `--profile` reads, or `device.json`, which `--device` reads: a
//! [`ReproDevice`]), the descriptor files the command names (`stage-input.json`,
//! `vertex-descriptor.json`), the command (`repro-command.txt`, and `repro.sh`, which runs it from
//! the bundle's directory) and `failure.txt`: why it failed, and each part of the request the
//! command does not replay ([`ReproBundle::unreplayed`]).
//!
//! A bundle's directory is named by its content ([`ReproBundle::key`]):
//! `metal2vulkan-repro-<16 hex digits>`. It is written under a temporary name and renamed, so a
//! directory of that name is a whole bundle, and a bundle whose directory exists is not written
//! again ([`ReproBundle::write`]).

use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use sha2::{Digest, Sha256};

use crate::cache::Hash256;
use crate::device::{DeviceDescription, DevicePolicy};
use crate::library::{FunctionConstantValues, Library};
use crate::metal::PixelFormat;
use crate::pipeline::FunctionDescriptor;

/// The device a runtime bundle's command compiles for, as `device.json` holds it and the command
/// line's `--device` reads it: the description and policy the runtime's compile device was built
/// from, which build the same `Device` (`Device::from_queries`).
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ReproDevice {
    pub description: DeviceDescription,
    pub policy: DevicePolicy,
}

impl ReproDevice {
    /// The device of `description` and `policy`.
    pub fn new(description: DeviceDescription, policy: DevicePolicy) -> ReproDevice {
        ReproDevice {
            description,
            policy,
        }
    }

    /// The device's `device.json`.
    pub fn to_json(&self) -> Result<Vec<u8>, String> {
        serde_json::to_vec_pretty(self).map_err(|e| e.to_string())
    }

    /// The device a `device.json` holds.
    pub fn from_json(json: &str) -> Result<ReproDevice, String> {
        serde_json::from_str(json).map_err(|e| e.to_string())
    }
}

/// One bundle, built in memory and written by [`write`](Self::write).
#[derive(Clone, Debug)]
pub struct ReproBundle<'a> {
    input_name: String,
    input: &'a [u8],
    input_hash: Hash256,
    files: Vec<(String, Vec<u8>)>,
    options: Vec<String>,
    unreplayed: Vec<String>,
    failure: String,
}

impl<'a> ReproBundle<'a> {
    /// A bundle of `input`, its file named `input_name`, and nothing else yet.
    pub fn new(input_name: impl Into<String>, input: &'a [u8]) -> ReproBundle<'a> {
        ReproBundle::hashed(input_name.into(), input, Sha256::digest(input).into())
    }

    fn hashed(input_name: String, input: &'a [u8], input_hash: Hash256) -> ReproBundle<'a> {
        ReproBundle {
            input_name,
            input,
            input_hash,
            files: Vec::new(),
            options: Vec::new(),
            unreplayed: Vec::new(),
            failure: String::new(),
        }
    }

    /// This bundle with the file `name` holding `bytes`.
    pub fn file(mut self, name: impl Into<String>, bytes: Vec<u8>) -> ReproBundle<'a> {
        self.files.push((name.into(), bytes));
        self
    }

    /// This bundle with `options` after the command's input and output.
    pub fn options<S: Into<String>>(
        mut self,
        options: impl IntoIterator<Item = S>,
    ) -> ReproBundle<'a> {
        self.options.extend(options.into_iter().map(Into::into));
        self
    }

    /// This bundle with `text` its `failure.txt`, before the list of what the command does not
    /// replay.
    pub fn failure(mut self, text: impl Into<String>) -> ReproBundle<'a> {
        self.failure = text.into();
        self
    }

    /// The command that replays the failure, run in the bundle's directory.
    pub fn command(&self) -> String {
        let mut words = vec![
            "metal2vulkan".to_owned(),
            self.input_name.clone(),
            "out.spv".to_owned(),
        ];
        words.extend(self.options.iter().cloned());
        shell_join(&words)
    }

    /// What the command does not replay of the request: each a field and its value, or a note.
    pub fn unreplayed(&self) -> &[String] {
        &self.unreplayed
    }

    /// The bytes the bundle writes.
    pub fn size(&self) -> u64 {
        let files: usize = self.files.iter().map(|(_, bytes)| bytes.len()).sum();
        (self.input.len() + files + self.failure_text().len() + 2 * self.command().len() + 64)
            as u64
    }

    /// The SHA-256 of what the bundle holds: its input's (by the input's own hash), each file's,
    /// its command's and its `failure.txt`'s, each with its name and length.
    pub fn key(&self) -> Hash256 {
        let mut hash = Sha256::new();
        let mut part = |name: &[u8], bytes: &[u8]| {
            hash.update((name.len() as u64).to_le_bytes());
            hash.update(name);
            hash.update((bytes.len() as u64).to_le_bytes());
            hash.update(bytes);
        };
        part(self.input_name.as_bytes(), &self.input_hash);
        for (name, bytes) in &self.files {
            part(name.as_bytes(), bytes);
        }
        part(b"repro-command.txt", self.command().as_bytes());
        part(b"failure.txt", self.failure_text().as_bytes());
        hash.finalize().into()
    }

    /// The name of the bundle's directory: `metal2vulkan-repro-` and the first 16 hex digits of
    /// its [`key`](Self::key).
    pub fn directory_name(&self) -> String {
        let key = self.key();
        let digits: String = key[..8].iter().map(|b| format!("{b:02x}")).collect();
        format!("metal2vulkan-repro-{digits}")
    }

    /// `failure.txt`: the failure text, then what the command does not replay.
    fn failure_text(&self) -> String {
        let mut text = self.failure.clone();
        if !text.is_empty() && !text.ends_with('\n') {
            text.push('\n');
        }
        if !self.unreplayed.is_empty() {
            text.push_str("\nnot replayed by the command:\n");
            for line in &self.unreplayed {
                text.push_str("  ");
                text.push_str(line);
                text.push('\n');
            }
        }
        text
    }

    /// Write the bundle into its directory under `base` (made where missing): the directory, and
    /// whether this call wrote it (`false` where a directory of its name is there already, which
    /// holds the same bundle). It is written under a temporary name in `base` and renamed, so a
    /// directory of its name is never partly written; a failure removes what it wrote.
    pub fn write(&self, base: &Path) -> io::Result<(PathBuf, bool)> {
        let name = self.directory_name();
        let dir = base.join(&name);
        if dir.exists() {
            return Ok((dir, false));
        }
        fs::create_dir_all(base)?;
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let temporary = base.join(format!(
            ".{name}-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&temporary)?;
        let written = self.write_into(&temporary).and_then(|()| {
            match fs::rename(&temporary, &dir) {
                Ok(()) => Ok(true),
                // Another writer's rename came first: its bundle is this one.
                Err(_) if dir.exists() => Ok(false),
                Err(e) => Err(e),
            }
        });
        if !matches!(written, Ok(true)) {
            let _ = fs::remove_dir_all(&temporary);
        }
        written.map(|new| (dir, new))
    }

    /// Write the bundle's files into `dir`.
    fn write_into(&self, dir: &Path) -> io::Result<()> {
        fs::write(dir.join(&self.input_name), self.input)?;
        for (name, bytes) in &self.files {
            fs::write(dir.join(name), bytes)?;
        }
        let command = self.command();
        fs::write(dir.join("repro-command.txt"), format!("{command}\n"))?;
        let script = dir.join("repro.sh");
        fs::write(
            &script,
            format!("#!/usr/bin/env sh\nset -eu\ncd -- \"$(dirname -- \"$0\")\"\n{command}\n"),
        )?;
        make_executable(&script)?;
        fs::write(dir.join("failure.txt"), self.failure_text())
    }
}

/// `args` as one shell command line, each quoted where it needs it.
pub fn shell_join(args: &[String]) -> String {
    args.iter()
        .map(|arg| shell_quote(arg))
        .collect::<Vec<_>>()
        .join(" ")
}

fn shell_quote(s: &str) -> String {
    if s.is_empty() {
        return "''".to_string();
    }
    if s.bytes()
        .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-' | b'.' | b'/' | b':' | b'='))
    {
        return s.to_string();
    }
    format!("'{}'", s.replace('\'', "'\\''"))
}

#[cfg(unix)]
fn make_executable(path: &Path) -> io::Result<()> {
    use std::os::unix::fs::PermissionsExt;
    fs::set_permissions(path, fs::Permissions::from_mode(0o755))
}

#[cfg(not(unix))]
fn make_executable(_path: &Path) -> io::Result<()> {
    Ok(())
}
