//! External-tool helpers for metal2vulkan's own tests and its validation tools: the fixture
//! assemblers (`llvm-as`, `xcrun metal-as`, `spirv-as`), the committed fixture directories, a
//! synthetic metallib writer ([`metallib`]), a tool's identity, the MoltenVK toolchain the MSL
//! checks reproduce, and the generated-program fuzzers ([`fuzz`]) the product's CPU-interpreter
//! tests and validation's device tests both run, the extended-range bicubic table both check
//! ([`xr_bicubic`]), and the store-buffering litmus kernel both compile
//! ([`litmus`]).
//!
//! Scratch directories every suite's temporaries go in ([`scratch`]): one root under the OS temp
//! directory, each run's directory removed when its owner drops and, when its owner is killed
//! instead, by the next run that reads the root.
//!
//! The product runs no external tool and takes this crate for its tests alone: it is the
//! product's path-only dev-dependency, which `cargo publish` drops, and a dependency of
//! `metal2vulkan-validation`. The tool runner ([`tools`]) reads the `METAL2VULKAN_<TOOL>`
//! overrides the product's `env_vars` registers (`tools::tool_path_override`, which spells them as
//! the product does; the product's tests hold the two together): each tool resolves through its
//! override, then the usual LLVM install directories, then `PATH`. [`spirv`] runs `spirv-val` and
//! `spirv-dis`, the L3 checks the product's tests and validation share.

use std::ffi::OsStr;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Stdio;

pub mod fuzz;
pub mod litmus;
pub mod metallib;
pub mod scratch;
pub mod spirv;
pub mod tools;
pub mod xr_bicubic;

/// The validation crate's `fixtures/public` beside this crate
/// (`crates/metal2vulkan-validation/fixtures/public`, read-only).
pub fn fixtures_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("the dev-tools crate lies in the workspace's crates/")
        .join("metal2vulkan-validation/fixtures/public")
}

/// [`fixtures_dir`]'s `engine` (read-only; the engine's own fixtures).
pub fn engine_dir() -> PathBuf {
    fixtures_dir().join("engine")
}

/// Why an external tool could not assemble a fixture or compile a module.
#[derive(Debug)]
pub enum AssembleError {
    /// The tool is not installed (a test skips visibly).
    Missing(String),
    /// The tool ran and rejected the input (a test fails).
    Rejected(String),
}

/// The bitcode `cmd` (after `pre`) assembles from the text on `input`: a function of the text and
/// the tool alone, wherever the text lies.
///
/// An LLVM assembler names a module after the file it reads, and a text without a
/// `source_filename` line takes that name as its `source_filename`, a record the bitcode carries.
/// Read from its path, one text assembled to different bytes in each directory: a census row
/// assembled in scratch directories named for process ids and counters carried them, so its work
/// units and ledger peak (the bitstream reader meters a record by its operands) and the detail of
/// a rejection changed from run to run. So the text goes in on standard input, where the module is
/// `<stdin>`, and the bitcode comes back on standard output.
fn assemble(cmd: &str, pre: &[&OsStr], input: Stdio) -> Result<Vec<u8>, AssembleError> {
    let mut args: Vec<&OsStr> = pre.to_vec();
    args.extend(["-", "-o", "-"].map(OsStr::new));
    match tools::run_with_stdin(cmd, &args, input) {
        Ok((bitcode, _)) => Ok(bitcode),
        Err(tools::ToolError::Launch { executable, .. }) => Err(AssembleError::Missing(format!(
            "{cmd} not found ({})",
            executable.display()
        ))),
        Err(e) => Err(AssembleError::Rejected(e.to_string())),
    }
}

/// [`assemble`] of the text in the file `input`.
fn assemble_file(cmd: &str, pre: &[&OsStr], input: &Path) -> Result<Vec<u8>, AssembleError> {
    let text = std::fs::File::open(input)
        .map_err(|e| AssembleError::Rejected(format!("read {}: {e}", input.display())))?;
    assemble(cmd, pre, text.into())
}

/// Opaque-pointer bitcode of an `.ll` fixture (`llvm-as`).
pub fn llvm_assemble(ll: &Path) -> Result<Vec<u8>, AssembleError> {
    assemble_file("llvm-as", &[], ll)
}

/// Opaque-pointer bitcode of the LLVM text `text` (`llvm-as`), which a test generates rather than
/// reads from a fixture: the same bitcode [`llvm_assemble`] makes of that text in a file.
pub fn llvm_assemble_text(text: &str) -> Result<Vec<u8>, AssembleError> {
    let (input, mut feed) =
        std::io::pipe().map_err(|e| AssembleError::Rejected(format!("pipe: {e}")))?;
    std::thread::scope(|scope| {
        // Written beside the tool's run, so neither its input nor its output pipe fills while the
        // other waits. The writer's end closes when the text is written, which ends the input.
        let writer = scope.spawn(move || feed.write_all(text.as_bytes()));
        let bitcode = assemble("llvm-as", &[], input.into());
        match (bitcode, writer.join().expect("the input writer")) {
            (Ok(_), Err(e)) => Err(AssembleError::Rejected(format!(
                "llvm-as stopped reading its input: {e}"
            ))),
            (bitcode, _) => bitcode,
        }
    })
}

/// The SPIR-V words of a `.spvasm` fixture (`spirv-as`, SPIR-V 1.5), every numeric id kept as the
/// text writes it (`--preserve-numeric-ids`), so a module `spirv-dis --raw-id` wrote assembles back
/// to its own words but the header's generator.
pub fn spirv_assemble(spvasm: &Path) -> Result<Vec<u32>, AssembleError> {
    let args = ["--preserve-numeric-ids", "--target-env", "spv1.5"].map(OsStr::new);
    let bytes = assemble_file("spirv-as", &args, spvasm)?;
    if bytes.len() % 4 != 0 {
        return Err(AssembleError::Rejected(format!(
            "spirv-as wrote {} bytes for {}, not whole words",
            bytes.len(),
            spvasm.display()
        )));
    }
    Ok(bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| u32::from_le_bytes(*word))
        .collect())
}

/// Typed-pointer AIR bitcode of an `.air.ll` companion (`xcrun metal-as`, macOS only).
pub fn metal_assemble(air_ll: &Path) -> Result<Vec<u8>, AssembleError> {
    if !cfg!(target_os = "macos") {
        return Err(AssembleError::Missing("metal-as needs macOS".to_owned()));
    }
    assemble_file("xcrun", &["metal-as".as_ref()], air_ll)
}

/// The `--version` output of external tool `cmd`, resolved as [`llvm_assemble`] and the L3 check
/// resolve it (`METAL2VULKAN_<TOOL>`, the Homebrew LLVM prefixes, then `PATH`): what a validation
/// cache keys that tool's behavior by. `Missing` when it cannot be launched, `Rejected` when it
/// runs and fails.
pub fn tool_version(cmd: &str) -> Result<String, AssembleError> {
    match tools::run(cmd, &["--version"]) {
        Ok((stdout, _)) => Ok(String::from_utf8_lossy(&stdout).into_owned()),
        Err(tools::ToolError::Launch { executable, .. }) => Err(AssembleError::Missing(format!(
            "{cmd} not found ({})",
            executable.display()
        ))),
        Err(e) => Err(AssembleError::Rejected(e.to_string())),
    }
}

/// The pipeline creation the product tests' `pipeline::testing::moltenvk_msl_args` reproduces:
/// MoltenVK 1.4.1 (the Homebrew bottle, built with Xcode 26.2) on an Apple M3 Max under macOS 26,
/// the device of the committed `moltenvk-m3` profile.
pub const MOLTENVK_MSL_TARGET: &str = "MoltenVK 1.4.1 on Apple M3 Max, macOS 26: MSL 4.0";

/// The SPIRV-Cross commit MoltenVK 1.4.1 links (its `ExternalRevisions/SPIRV-Cross_repo_revision`).
/// A `spirv-cross` built from another commit converts as that commit does.
pub const MOLTENVK_SPIRV_CROSS_REVISION: &str = "adec7acbf41a988713cdb85f93f26c8ca5ea863e";
