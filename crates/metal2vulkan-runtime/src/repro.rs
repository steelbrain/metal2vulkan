//! FALLBACK repro bundles of the creations a device refuses: a default-off diagnostic,
//! `METAL2VULKAN_REPRO_DIR` (`metal2vulkan::env_vars`), that writes what replays a refusal with
//! the `metal2vulkan` command line, the bundle its own failures write (`metal2vulkan::repro`).
//!
//! - **Where.** Each creation that compiles a guest's functions, where the refusal returns: a
//!   library of a metallib (`Device::new_library_with_data`), a library's function by name
//!   (`Library::new_function_with_name` and the forms beside it), and every compute, tile and
//!   render pipeline state's (a state's creation, its additions of binary functions, a binary
//!   archive's add). Not a mesh state's (the command line compiles no mesh pipeline), a dispatch's
//!   or draw's later variant (its lowering inputs, bound formats and grid forms among them, are
//!   no options of the command line), a driver's `VkPipeline` making (a Vulkan failure;
//!   `METAL2VULKAN_DUMP_PIPELINES` keeps its SPIR-V), nor a library of Metal source (no AIR).
//! - **Which refusals** ([`bundled`]): `Unsupported`, `NotImplemented`, and `Compile` but a
//!   cancelled compile and a name the library lacks (Metal's own answer, nil: a program that
//!   probes for its functions by name spends no bundle). Never Metal's own answers to a misuse
//!   (`Invalid`, `Nil`, `MetalError`).
//! - **What it holds.** The library's bytes, `device.json` (the description and policy the
//!   device's compile layer was built from, which the command line's `--device` reads), the
//!   request as the command line's options and descriptor files, and `failure.txt`: the call,
//!   the refusal, and what of the request the command does not replay.
//! - **Once per distinct refusal, bounded.** A bundle's directory is named by its content, so an
//!   identical refusal (on this run or an earlier one) writes nothing. A process writes at most
//!   [`BUNDLES`] bundles and [`BYTES`] bytes; past either it writes no more, and says so once.
//! - **The call is unchanged.** The bundle is written on the refusing thread (a refusal already
//!   spent a translation, and the shader cache memoizes refusals), and the call returns the same
//!   refusal: a write that fails is said once and fails nothing. Each bundle written is named in
//!   one line through the device's message sink at `Info` severity (id name
//!   `metal2vulkan-repro`), else on stderr: the bundle's directory, the call, and the refusal.
//!   The refusal itself carries no path: a host path reaches no guest's error.

use crate::{Device, Error};
use metal2vulkan::error::{DescriptorError, Error as CompileError};
use metal2vulkan::library::Library;
use metal2vulkan::pipeline::{
    ComputePipelineDescriptor, RenderPipelineDescriptor, TileRenderPipelineDescriptor,
};
use metal2vulkan::repro::{ReproBundle, ReproDevice};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

/// The bundles a process writes at most.
pub(crate) const BUNDLES: u32 = 32;

/// The bytes a process's bundles hold at most.
pub(crate) const BYTES: u64 = 256 << 20;

/// What a refused creation was asked to make.
#[derive(Clone, Copy)]
pub(crate) enum Request<'a> {
    /// A library of these bytes.
    Library(&'a [u8]),
    /// The function of this name of the library.
    Function(&'a Library, &'a str),
    Compute(&'a ComputePipelineDescriptor),
    Tile(&'a TileRenderPipelineDescriptor),
    Render(&'a RenderPipelineDescriptor),
}

/// Hand back `error`, `device`'s refusal of `call`'s `request`, having written its bundle where
/// the diagnostic is on and the refusal is one a bundle replays ([`bundled`]).
pub(crate) fn refused(device: &Device, call: &str, request: Request<'_>, error: Error) -> Error {
    if let Some(base) = directory() {
        if bundled(&error) {
            let mut say = |line: String| device.inform("metal2vulkan-repro", line);
            let what = format!("{call} refused: {error}");
            TALLY.write(
                &base,
                &what,
                || bundle(device, call, request, &error),
                &mut say,
            );
        }
    }
    error
}

/// Whether a bundle replays `error`, a creation's refusal (the module's documentation, "Which
/// refusals").
fn bundled(error: &Error) -> bool {
    match error {
        Error::Unsupported(_) | Error::NotImplemented(_) => true,
        Error::Compile(CompileError::Cancelled)
        | Error::Compile(CompileError::InvalidDescriptor(DescriptorError::EntryNotFound {
            ..
        })) => false,
        Error::Compile(_) => true,
        _ => false,
    }
}

/// The bundle of `device`'s refusal `error` of `call`'s `request`: none where a stage is one the
/// compile layer already compiled, which the command line cannot take; an error where the device's
/// file cannot be written.
fn bundle<'a>(
    device: &Device,
    call: &str,
    request: Request<'a>,
    error: &Error,
) -> Result<Option<ReproBundle<'a>>, String> {
    let compile = device.compile_device();
    let json = ReproDevice::new(device.properties().description.clone(), *compile.policy())
        .to_json()
        .map_err(|e| format!("device.json not serialized: {e}"))?;
    let bundle = match request {
        Request::Library(bytes) => Some(ReproBundle::for_library(&json, bytes)),
        Request::Function(library, name) => Some(ReproBundle::for_function(&json, library, name)),
        Request::Compute(descriptor) => ReproBundle::for_compute(&json, descriptor),
        Request::Tile(descriptor) => ReproBundle::for_tile(&json, descriptor),
        Request::Render(descriptor) => ReproBundle::for_render(&json, descriptor),
    };
    Ok(bundle.map(|bundle| {
        bundle.failure(format!(
            "why: {error}\ncall: {call}\n\ndetail:\n{error:?}\n"
        ))
    }))
}

/// What a process wrote, held to [`BUNDLES`] and [`BYTES`].
pub(crate) struct Tally {
    bundles: u32,
    bytes: u64,
    limits: (u32, u64),
    /// The bound was met, and said: nothing more is written.
    full: bool,
    /// A write failed, and was said.
    failure_said: bool,
}

/// The process's tally.
static TALLY: Bounded = Bounded(Mutex::new(Tally::new(BUNDLES, BYTES)));

/// A tally behind its lock: one bundle is written at a time.
pub(crate) struct Bounded(pub(crate) Mutex<Tally>);

impl Tally {
    /// A tally of no bundles, held to `bundles` bundles and `bytes` bytes.
    pub(crate) const fn new(bundles: u32, bytes: u64) -> Tally {
        Tally {
            bundles: 0,
            bytes: 0,
            limits: (bundles, bytes),
            full: false,
            failure_said: false,
        }
    }

    /// The bound is met at `base`: nothing more is written, and that is said once.
    fn bound(&mut self, base: &Path, say: &mut dyn FnMut(String)) {
        if !self.full {
            self.full = true;
            say(format!(
                "metal2vulkan: FALLBACK repro bundles: the process wrote {} of at most {} bundles \
                 and {} of at most {} bytes under {}; no more are written",
                self.bundles,
                self.limits.0,
                self.bytes,
                self.limits.1,
                base.display()
            ));
        }
    }
}

impl Bounded {
    /// Write the bundle `bundle` gives of the refusal `what` under `base` unless its directory
    /// is there already or the bound is met, saying through `say` the bundle written (its
    /// directory and `what`), the bound met (once) or a failure (once).
    pub(crate) fn write<'a>(
        &self,
        base: &Path,
        what: &str,
        bundle: impl FnOnce() -> Result<Option<ReproBundle<'a>>, String>,
        say: &mut dyn FnMut(String),
    ) {
        let mut tally = self
            .0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if tally.full || tally.bundles >= tally.limits.0 {
            tally.bound(base, say);
            return;
        }
        let failed = |tally: &mut Tally, say: &mut dyn FnMut(String), why: String| {
            if !tally.failure_said {
                tally.failure_said = true;
                say(format!(
                    "metal2vulkan: FALLBACK repro bundle not written under {}: {why}; a later \
                     failure is not said",
                    base.display()
                ));
            }
        };
        let bundle = match bundle() {
            Ok(Some(bundle)) => bundle,
            Ok(None) => return,
            Err(why) => return failed(&mut tally, say, why),
        };
        if base.join(bundle.directory_name()).exists() {
            return;
        }
        let size = bundle.size();
        if tally.bytes.saturating_add(size) > tally.limits.1 {
            tally.bound(base, say);
            return;
        }
        match bundle.write(base) {
            Ok((dir, true)) => {
                tally.bundles += 1;
                tally.bytes += size;
                say(format!(
                    "metal2vulkan: FALLBACK repro {}: {what}",
                    dir.display()
                ));
            }
            Ok((_, false)) => {}
            Err(e) => failed(&mut tally, say, e.to_string()),
        }
    }
}

/// The directory bundles go into: `METAL2VULKAN_REPRO_DIR`, read once, where set and neither
/// empty nor `0` (a device test's own directory under test, `TestDirectory`).
fn directory() -> Option<PathBuf> {
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    {
        let dir = TEST_DIRECTORY.with(|dir| dir.borrow().clone());
        if dir.is_some() {
            return dir;
        }
    }
    static DIR: OnceLock<Option<PathBuf>> = OnceLock::new();
    DIR.get_or_init(|| {
        metal2vulkan::env_vars::repro_dir()
            .filter(|value| !value.is_empty() && value != "0")
            .map(PathBuf::from)
    })
    .clone()
}

#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
thread_local! {
    /// The directory a test's creations on this thread write their bundles into, in place of the
    /// environment's ([`TestDirectory`]).
    static TEST_DIRECTORY: std::cell::RefCell<Option<PathBuf>> = const {
        std::cell::RefCell::new(None)
    };
}

/// The bundles of the creations this thread refuses go into a directory of the test's while it
/// lives, whatever the environment says: the test seam, which touches no other test's
/// environment.
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
pub(crate) struct TestDirectory;

#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
impl TestDirectory {
    pub(crate) fn set(dir: &Path) -> TestDirectory {
        TEST_DIRECTORY.with(|slot| *slot.borrow_mut() = Some(dir.to_path_buf()));
        TestDirectory
    }
}

#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
impl Drop for TestDirectory {
    fn drop(&mut self) {
        TEST_DIRECTORY.with(|slot| *slot.borrow_mut() = None);
    }
}
