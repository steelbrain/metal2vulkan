//! A worker's L3 stage, run in a fresh image of the worker's executable.
//!
//! The 500-MiB ceiling covers a whole attempt, `spirv-val` included: the parent samples the
//! worker's process group, so a validator run beside the translation's resident memory is
//! charged both at once. A 232K-instruction kernel translated in 303 MB, and `spirv-val` alone
//! held 427 MB on its 3.6M-word module; run from the translating process, the group passed the
//! ceiling at 527 MB. An attempt's peak is the larger of its translation's and its validation's,
//! never their sum.
//!
//! So a worker whose attempt linked a module ([`crate::engine::translate_unvalidated`]) does not
//! run `spirv-val` itself: [`report_after_l3`] writes the module's words and the provisional
//! result to the worker's scratch and replaces the process image with this executable's L3 stage
//! (`exec`). Nothing of the translation survives the exec: no IR, no ledger, no allocator arena.
//! The stage keeps the worker's pid and process group, so the parent's deadline and memory
//! sampling go on uninterrupted, runs `spirv-val` on the module's file without reading it
//! ([`serve`]), folds L3's result into the attempt ([`crate::engine::Attempt::settle_l3`]), and
//! writes the worker's one report, its allocator peak and resident high water each the larger of
//! the translation's and its own.

use crate::engine::{Attempt, L3Plan, PendingL3};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// The first argument that selects the L3 stage; the second names the worker's result type, the
/// third the pending file.
pub const STAGE_ARG: &str = "--l3-stage";

/// What the translation hands the L3 stage, in the pending file.
#[derive(Serialize, Deserialize)]
struct Pending<R> {
    /// The translation's counting-allocator process peak and resident high water
    /// ([`super::Peaks::now`]), which the exec ends.
    peaks: super::Peaks,
    /// The worker's result, its attempt as it stands if `spirv-val` accepts the module.
    result: R,
    plan: L3Plan,
    /// The module's words, little-endian.
    words: PathBuf,
}

/// Report `result` once L3 has settled its attempt, whose outcome stands if `spirv-val` accepts
/// the module `pending`, in this executable's L3 stage for `stage` (the result type's name, which
/// the executable's `main` maps to [`serve`]): the module's words and the provisional result go to
/// the worker's scratch (`std::env::temp_dir()`, which the parent owns and removes on every exit
/// path), and the stage replaces this process's image. Returns only when that failed.
#[cfg(unix)]
pub fn report_after_l3<R: Serialize>(
    stage: &str,
    result: &R,
    pending: PendingL3,
) -> Result<(), String> {
    use std::os::unix::process::CommandExt as _;
    let dir = std::env::temp_dir();
    let words_path = dir.join(format!("l3-words-{}.spv", std::process::id()));
    let pending_path = dir.join(format!("l3-pending-{}.json", std::process::id()));
    let PendingL3 { words, plan, .. } = pending;
    let write = |path: &Path, words: Vec<u32>| {
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        drop(words);
        std::fs::write(path, bytes).map_err(|error| format!("write {}: {error}", path.display()))
    };
    write(&words_path, words)?;
    let text = serde_json::to_vec(&Pending {
        peaks: super::Peaks::now(),
        result,
        plan,
        words: words_path,
    })
    .map_err(|error| format!("encode the L3 stage's input: {error}"))?;
    std::fs::write(&pending_path, text)
        .map_err(|error| format!("write {}: {error}", pending_path.display()))?;
    let executable = std::env::current_exe()
        .map_err(|error| format!("resolve the worker's executable: {error}"))?;
    let error = std::process::Command::new(executable)
        .arg(STAGE_ARG)
        .arg(stage)
        .arg(&pending_path)
        .exec();
    Err(format!("exec the L3 stage: {error}"))
}

/// No image replaces a worker's off unix, where no worker runs ([`super::MonitoredChild::spawn`]).
#[cfg(not(unix))]
pub fn report_after_l3<R: Serialize>(
    _stage: &str,
    _result: &R,
    _pending: PendingL3,
) -> Result<(), String> {
    Err("the L3 stage replaces the worker's image, which needs unix".into())
}

/// The L3 stage of a worker whose result is an `R`: read the pending file `pending`, run
/// `spirv-val` on the module, fold its result into the attempt `attempt` finds in the result, and
/// [`super::report`] it, its allocator peak and resident high water each the larger of the
/// translation's and this stage's.
pub fn serve<R: Serialize + DeserializeOwned>(
    pending: &Path,
    attempt: impl FnOnce(&mut R) -> &mut Attempt,
) -> Result<(), String> {
    super::alloc::enable_budget(super::MEMORY_LIMIT_BYTES);
    let text =
        std::fs::read(pending).map_err(|error| format!("read {}: {error}", pending.display()))?;
    let _ = std::fs::remove_file(pending);
    let Pending {
        peaks,
        mut result,
        plan,
        words,
    } = serde_json::from_slice::<Pending<R>>(&text)
        .map_err(|error| format!("decode {}: {error}", pending.display()))?;
    drop(text);
    // The tools read the module from its file; this image never holds it.
    let checked = std::panic::catch_unwind(|| plan.check_file(&words));
    let _ = std::fs::remove_file(&words);
    attempt(&mut result).settle_l3(checked);
    super::report_with_peaks(&result, peaks.max(super::Peaks::now()))
}
