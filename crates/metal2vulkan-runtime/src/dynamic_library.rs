//! `DynamicLibrary` (`MTLDynamicLibrary`): `Device::new_dynamic_library` and
//! `new_dynamic_library_with_url`, its install name and file, and the loading of the dynamic
//! libraries a pipeline stage depends on when a pipeline state is created.
//!
//! - **Making one.** A dynamic library is the compile layer's
//!   [`DynamicLibrary`](metal2vulkan::library::DynamicLibrary) held by its device: a library of
//!   type dynamic, with an install name and no function constant its code reads, as
//!   `newDynamicLibrary:error:` checks it (probes, an M3 Max and an M4).
//! - **Linking it.** A stage preloads it by its compile-layer form
//!   ([`DynamicLibrary::preloaded_library`]) in its function descriptor's
//!   `libraries.preloaded` (`preloadedLibraries`, and a render pipeline's
//!   `vertexPreloadedLibraries` and `fragmentPreloadedLibraries`). A library names the dynamic
//!   libraries it was linked against by install name; creating a pipeline state loads, for each
//!   install name its stage's search reaches, the library in the file it names
//!   (`stage_with_dependencies`), as Metal loads them at creation: the file is read at each
//!   creation, so a replaced file links its new code. A name that is not an absolute path, or
//!   whose file does not hold a dynamic library, loads nothing, and the creation fails only where
//!   an import survives specialization, as Metal's does.

use super::binary_archive::container::{write_file, WriteFailure};
use super::device::{device_error, lock};
use super::error::{invalid, DynamicLibraryError, Error, Problem};
use super::library::Library;
use super::Device;
use metal2vulkan::pipeline::StageSource;
use metal2vulkan::Budget;
use std::path::Path;
use std::sync::{Arc, Mutex};

/// A Metal dynamic library (`MTLDynamicLibrary`). Cheap to clone; `Send + Sync`.
#[derive(Clone, Debug)]
pub struct DynamicLibrary(Arc<Shared>);

#[derive(Debug)]
struct Shared {
    device: Device,
    library: metal2vulkan::library::DynamicLibrary,
    label: Mutex<Option<String>>,
}

impl Device {
    /// `newDynamicLibrary:error:`: `library` as a dynamic library, checked as the compile layer's
    /// [`DynamicLibrary::new`](metal2vulkan::library::DynamicLibrary::new) checks it under the
    /// standard budget: a library of another type, one without an install name, or one whose
    /// code reads a function constant is refused (`Compile(InvalidDescriptor(DynamicLibrary))`),
    /// as Metal refuses each. A library of another device is refused (`Invalid(ForeignObject)`).
    pub fn new_dynamic_library(&self, library: &Library) -> Result<DynamicLibrary, Error> {
        if !library.device().same(self) {
            return Err(invalid("new_dynamic_library", Problem::ForeignObject));
        }
        self.dynamic_library(library.compile_library().clone())
    }

    /// `newDynamicLibraryWithURL:error:`: the dynamic library in the file `url`, which
    /// [`DynamicLibrary::serialize_to_url`] writes (a universal file, as Metal writes, or a
    /// metallib, as the offline `metal -dynamiclib` writes, load alike). Its label is none, as
    /// Metal's; its install name is the file's. A file that cannot be read is
    /// `DynamicLibrary(Read)` (a missing one is Metal's `NSCocoaErrorDomain` 260), and one that
    /// holds no library `DynamicLibrary(InvalidFile)`, as Metal answers (a probe), bare AIR bitcode
    /// included, raw or wrapped ("its AIR slice is missing"). A library that is not one
    /// `new_dynamic_library` takes is refused as it refuses it, where Metal's answer for a file is
    /// not measured.
    pub fn new_dynamic_library_with_url(&self, url: &Path) -> Result<DynamicLibrary, Error> {
        let bytes = std::fs::read(url).map_err(|error| {
            Error::DynamicLibrary(DynamicLibraryError::Read {
                path: url.to_path_buf(),
                kind: error.kind(),
                message: error.to_string(),
            })
        })?;
        let library = metal2vulkan::library::Library::from_metallib(&bytes).map_err(|_| {
            Error::DynamicLibrary(DynamicLibraryError::InvalidFile {
                path: url.to_path_buf(),
            })
        })?;
        self.dynamic_library(library)
    }

    /// `library` checked as a dynamic library, held by this device.
    fn dynamic_library(
        &self,
        library: metal2vulkan::library::Library,
    ) -> Result<DynamicLibrary, Error> {
        let library = metal2vulkan::library::DynamicLibrary::new(library, &Budget::standard())
            .map_err(device_error)?;
        Ok(DynamicLibrary(Arc::new(Shared {
            device: self.clone(),
            library,
            label: Mutex::new(None),
        })))
    }
}

impl DynamicLibrary {
    /// `device`.
    pub fn device(&self) -> &Device {
        &self.0.device
    }

    /// `setLabel:`.
    pub fn set_label(&self, label: &str) {
        *lock(&self.0.label) = Some(label.to_owned());
    }

    /// `label`: `None` until one is set, and for a library loaded from a file, as Metal's.
    pub fn label(&self) -> Option<String> {
        lock(&self.0.label).clone()
    }

    /// `installName`: the name a library linked against this one loads it by.
    pub fn install_name(&self) -> &str {
        self.0.library.install_name()
    }

    /// `serializeToURL:error:`: write the library to the file `url`, replacing what is there,
    /// through a temporary file beside it and a rename. The file holds the library's metallib
    /// (of one loaded from a universal file, that file's AIR slice), which
    /// [`Device::new_dynamic_library_with_url`] and the loading of an install name read back,
    /// and which Metal loads as it loads the offline `metal -dynamiclib` output: Metal's own
    /// file is a universal file with a slice of its GPU's code beside the AIR. A missing directory
    /// is `DynamicLibrary(InvalidFile)`, Metal's "Invalid URL", and any other failure to write the
    /// file (a directory at `url`, for one) `DynamicLibrary(Write)`.
    pub fn serialize_to_url(&self, url: &Path) -> Result<(), Error> {
        write_file(self.0.library.library().bytes(), url).map_err(|failure| {
            Error::DynamicLibrary(match failure {
                WriteFailure::NoDirectory => DynamicLibraryError::InvalidFile {
                    path: url.to_path_buf(),
                },
                WriteFailure::Io(error) => DynamicLibraryError::Write {
                    path: url.to_path_buf(),
                    kind: error.kind(),
                    message: error.to_string(),
                },
            })
        })
    }

    /// The library as a stage preloads it: an entry of its function descriptor's
    /// `libraries.preloaded` ([`DynamicLibraries`](metal2vulkan::pipeline::DynamicLibraries)),
    /// `MTLComputePipelineDescriptor.preloadedLibraries` and a render pipeline's
    /// `vertexPreloadedLibraries` and `fragmentPreloadedLibraries`.
    pub fn preloaded_library(&self) -> metal2vulkan::library::DynamicLibrary {
        self.0.library.clone()
    }
}

/// `stage` with the dynamic libraries its search reaches loaded from the files their install
/// names name (`FunctionDescriptor::with_loaded_dependencies`), as Metal loads them when it
/// creates a pipeline state: read again at each creation, a library already in
/// `libraries.loaded` kept. Each is taken as
/// [`DynamicLibrary::loaded`](metal2vulkan::library::DynamicLibrary::loaded) takes it, its module
/// not decoded until a search reaches it. A name that is not an absolute path (one relative to a
/// loaded file, `@loader_path` or `@executable_path`, which a library made from bytes has no
/// directory for), or a file that cannot be read or holds no dynamic library with an install
/// name, loads nothing: the compile then refuses an import that survives specialization
/// (`NeedsInput(DynamicLibrary)`), as Metal's creation fails, and makes the others. A stage the
/// compile layer already compiled links nothing more.
pub(crate) fn stage_with_dependencies(stage: &StageSource) -> StageSource {
    match stage {
        StageSource::Function(function) => {
            StageSource::Function(function.clone().with_loaded_dependencies(load))
        }
        StageSource::Compiled(_) => stage.clone(),
    }
}

/// The dynamic library in the file `name` names, or `None` where none loads: as
/// `new_dynamic_library_with_url`, a file of bare AIR holds none.
fn load(name: &str) -> Option<metal2vulkan::library::DynamicLibrary> {
    let path = Path::new(name);
    if !path.is_absolute() {
        return None;
    }
    let bytes = std::fs::read(path).ok()?;
    let library = metal2vulkan::library::Library::from_metallib(&bytes).ok()?;
    metal2vulkan::library::DynamicLibrary::loaded(library).ok()
}
