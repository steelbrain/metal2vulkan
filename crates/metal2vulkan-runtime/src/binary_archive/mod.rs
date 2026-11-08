//! `BinaryArchive` (`MTLBinaryArchive`) and `Device::new_binary_archive`.
//!
//! - **What it holds.** The compile layer's self-checking artifacts, not driver binaries: each
//!   translated function and linked pipeline a creation added to it used, under its kind, key and
//!   profile; and the archive key (`metal_key`) of each descriptor added. The driver's pipeline
//!   cache data stays in the device's shader cache.
//! - **Where creations look.** A creation that lists archives in its [`PipelineOptions`] looks
//!   in the device's memo, then in its archives in order, then in the shader cache on disk,
//!   before compiling (`pipeline/sources.rs`). What an archive gives skips the translation, as
//!   Metal's archives skip the compile. A variant a dispatch or draw needs later is not looked
//!   up in archives: no add compiled it.
//! - **Failing on a miss.** Under `PipelineOption::FAIL_ON_BINARY_ARCHIVE_MISS`, a creation
//!   that lists at least one archive is refused ([`Error::BinaryArchiveMiss`]) unless the
//!   archives hold the archive key of each of its stages (`metal_key`), as Metal's creation is,
//!   naming the first stage's function they miss. Without archives, or without the option,
//!   archives only make creation faster.
//! - **Adding.** `add_compute_pipeline_functions`, `add_render_pipeline_functions`,
//!   `add_tile_render_pipeline_functions` and `add_mesh_render_pipeline_functions` make the state
//!   their descriptor describes, as the creation would, and answer that creation's error; then
//!   they keep what the creation used and the archive keys of its stages. Adding twice keeps one
//!   copy. A mesh creation compiles its pipeline whole, looking in no archive, so a mesh add keeps
//!   its key alone, which `FAIL_ON_BINARY_ARCHIVE_MISS` finds.
//! - **Files.** An archive is read whole from its descriptor's file (`container.rs`) and written
//!   through a temporary file and a rename. A metallib, which Metal takes as an archive, loads as
//!   an empty one, whose lookups miss as Metal's do. An archive of another profile or build
//!   loads, and its entries miss, as Metal's archive of another device or OS does. Labels are not
//!   written, as Metal's are not.
//! - **Differences from Metal**, accepted (docs/API.md §10): the file is read whole at load, so an
//!   archive serializes after its file was deleted, where Metal's fails; and a file is a path,
//!   so a URL that is not a file cannot be named, where Metal answers it as an invalid URL.

pub(super) mod container;
pub(super) mod metal_key;

use super::device::lock;
use super::error::{ArchivedPipeline, BinaryArchiveError, BinaryArchiveMiss, Error};
use super::pipeline::{
    ComputePipelineState, GraphicsState, PipelineOption, PipelineOptions, TileState,
};
use super::shader_cache::EntryKind;
use super::Device;
use container::{Contents, Place};
use metal2vulkan::cache::{Hash256, PipelineArtifact, PipelineKey};
use metal2vulkan::library::{CompiledFunction, FunctionCompileKey};
use metal2vulkan::pipeline::{
    ComputePipeline, ComputePipelineDescriptor, MeshPipelineDescriptor, RenderPipeline,
    RenderPipelineDescriptor, StageSource, TileRenderPipelineDescriptor,
};
use metal_key::VertexKind;
use std::cell::RefCell;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

/// `MTLBinaryArchiveDescriptor`.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct BinaryArchiveDescriptor {
    /// `url`: the file to load the archive from; `None` makes an empty archive.
    pub url: Option<PathBuf>,
}

impl BinaryArchiveDescriptor {
    /// The descriptor of the archive in the file `url`.
    pub fn new(url: Option<PathBuf>) -> BinaryArchiveDescriptor {
        BinaryArchiveDescriptor { url }
    }
}

/// A Metal binary archive (`MTLBinaryArchive`). Cheap to clone (a retain); `Send + Sync`.
#[derive(Clone)]
pub struct BinaryArchive(Arc<Shared>);

struct Shared {
    device: Device,
    label: Mutex<Option<String>>,
    contents: Mutex<Contents>,
}

impl std::fmt::Debug for BinaryArchive {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let contents = lock(&self.0.contents);
        f.debug_struct("BinaryArchive")
            .field("keys", &contents.keys.len())
            .field("entries", &contents.entries.len())
            .finish()
    }
}

/// An artifact a creation used, which an add keeps.
pub(crate) enum Artifact {
    Function(CompiledFunction),
    Compute(Arc<ComputePipeline>),
    Render(Arc<RenderPipeline>),
}

/// The artifacts a creation for an add used, in order.
pub(crate) type Used = RefCell<Vec<Artifact>>;

impl Device {
    /// `newBinaryArchiveWithDescriptor:error:`: an empty archive, or the archive in the
    /// descriptor's file. A file that is missing or neither an archive nor a metallib is
    /// `BinaryArchive(InvalidFile)`, as Metal answers for a missing, empty, garbage or text file
    /// and for a directory (measured on an M3 Max and an M4). A metallib, which Metal takes as an
    /// archive (measured on an M3 Max), holds nothing of ours: it loads as an empty archive,
    /// whose lookups miss and which serializes as one nothing was added to. Metal's archive of a
    /// metallib misses too: a compute pipeline of a kernel the metallib holds, listing it with
    /// `FAIL_ON_BINARY_ARCHIVE_MISS`, is refused, and without the bit it is created (measured on
    /// an M3 Max under macOS 27.0.1, `binary-archive-sources`).
    pub fn new_binary_archive(
        &self,
        descriptor: &BinaryArchiveDescriptor,
    ) -> Result<BinaryArchive, Error> {
        self.current().new_binary_archive_now(descriptor)
    }

    /// [`Device::new_binary_archive`] on the generation that serves (`device/line.rs`).
    pub(crate) fn new_binary_archive_now(
        &self,
        descriptor: &BinaryArchiveDescriptor,
    ) -> Result<BinaryArchive, Error> {
        let contents = match &descriptor.url {
            None => Contents::default(),
            Some(path) => container::read(path)?,
        };
        Ok(BinaryArchive(Arc::new(Shared {
            device: self.clone(),
            label: Mutex::new(None),
            contents: Mutex::new(contents),
        })))
    }
}

impl BinaryArchive {
    /// `device`.
    pub fn device(&self) -> &Device {
        &self.0.device
    }

    /// `setLabel:`.
    pub fn set_label(&self, label: &str) {
        *lock(&self.0.label) = Some(label.to_owned());
    }

    /// `label`: `None` until one is set.
    pub fn label(&self) -> Option<String> {
        lock(&self.0.label).clone()
    }

    /// `addComputePipelineFunctionsWithDescriptor:error:`: make the state `descriptor`
    /// describes, as `new_compute_pipeline_state` makes it with the default options, answering
    /// its error; then keep what that creation used and the descriptor's archive key.
    pub fn add_compute_pipeline_functions(
        &self,
        descriptor: &ComputePipelineDescriptor,
    ) -> Result<(), Error> {
        let used = Used::default();
        ComputePipelineState::create(
            self.device(),
            descriptor,
            &PipelineOptions::default(),
            Some(&used),
        )?;
        let profile = self.profile();
        self.keep(
            used.into_inner(),
            &profile,
            [metal_key::compute(&profile, descriptor)],
        );
        Ok(())
    }

    /// `addRenderPipelineFunctionsWithDescriptor:error:`: make the state `descriptor`
    /// describes, as `new_render_pipeline_state` makes it with the default options, answering
    /// its error; then keep what that creation used and the archive keys of its stages.
    pub fn add_render_pipeline_functions(
        &self,
        descriptor: &RenderPipelineDescriptor,
    ) -> Result<(), Error> {
        let used = Used::default();
        let state = GraphicsState::create(
            self.device(),
            descriptor,
            &PipelineOptions::default(),
            Some(&used),
        )?;
        let profile = self.profile();
        let keys = metal_key::render(&profile, descriptor);
        let kind = if crate::tessellation::patches(&state.base().graphics).is_some() {
            VertexKind::Patch
        } else {
            VertexKind::Plain
        };
        let fragment = keys.fragment.map(|fragment| fragment.of(kind));
        self.keep(
            used.into_inner(),
            &profile,
            fragment
                .into_iter()
                .chain([keys.vertex.of(kind)])
                .chain(keys.pair(kind)),
        );
        Ok(())
    }

    /// `addTileRenderPipelineFunctionsWithDescriptor:error:`: make the state `descriptor`
    /// describes, as `new_tile_render_pipeline_state` makes it with the default options,
    /// answering its error; then keep what that creation used and the descriptor's archive key.
    pub fn add_tile_render_pipeline_functions(
        &self,
        descriptor: &TileRenderPipelineDescriptor,
    ) -> Result<(), Error> {
        let used = Used::default();
        TileState::create(
            self.device(),
            descriptor,
            &PipelineOptions::default(),
            Some(&used),
        )?;
        let profile = self.profile();
        self.keep(
            used.into_inner(),
            &profile,
            [metal_key::tile(&profile, descriptor)],
        );
        Ok(())
    }

    /// `addMeshRenderPipelineFunctionsWithDescriptor:error:`: make the state `descriptor`
    /// describes, as `new_mesh_render_pipeline_state` makes it with the default options,
    /// answering its error; then keep the descriptor's archive key. A mesh creation compiles its
    /// pipeline whole and looks in no archive, so the add keeps no compiled artifact and makes no
    /// later creation faster. Which of the descriptor's fields Metal keys a mesh pipeline by is
    /// not measured: the key holds every field (`metal_key`), so under
    /// `FAIL_ON_BINARY_ARCHIVE_MISS` a mesh creation hits only an add of an equal descriptor, and
    /// one that differs from every add in a field Metal does not key is `BinaryArchiveMiss` where
    /// Metal's creation may hit.
    pub fn add_mesh_render_pipeline_functions(
        &self,
        descriptor: &MeshPipelineDescriptor,
    ) -> Result<(), Error> {
        GraphicsState::create_mesh(self.device(), descriptor, &PipelineOptions::default())?;
        let profile = self.profile();
        self.keep(
            Vec::new(),
            &profile,
            [metal_key::mesh(&profile, descriptor)],
        );
        Ok(())
    }

    /// `serializeToURL:error:`: write the archive to the file `url`, replacing what is there,
    /// through a temporary file beside it and a rename. An archive nothing was added to or
    /// loaded into is `BinaryArchive(NothingToSerialize)`; a missing directory
    /// `BinaryArchive(InvalidFile)`, and any other failure to write the file (a directory at
    /// `url`, for one) `BinaryArchive(Write)`, as Metal answers each (measured on an M3 Max and
    /// an M4).
    pub fn serialize_to_url(&self, url: &Path) -> Result<(), Error> {
        let bytes = {
            let contents = lock(&self.0.contents);
            if contents.is_empty() {
                return Err(Error::BinaryArchive(BinaryArchiveError::NothingToSerialize));
            }
            container::encode(&contents)
        };
        container::write(&bytes, url)
    }

    /// Whether the archive holds the archive key `key`.
    pub(crate) fn holds(&self, key: &Hash256) -> bool {
        lock(&self.0.contents).keys.contains(key)
    }

    /// The function `compile` compiled under `key`, from the archive, and its encoded length.
    pub(crate) fn function(
        &self,
        compile: &metal2vulkan::device::Device,
        key: &FunctionCompileKey,
    ) -> Option<(CompiledFunction, u64)> {
        let bytes = self.entry(EntryKind::Function, key.digest(), compile)?;
        let function = CompiledFunction::from_bytes_for_key(&bytes, key, compile).ok()?;
        Some((function, bytes.len() as u64))
    }

    /// The compute pipeline `compile` compiled under `key`, from the archive, and its encoded
    /// length.
    pub(crate) fn compute_pipeline(
        &self,
        compile: &metal2vulkan::device::Device,
        key: &PipelineKey,
    ) -> Option<(ComputePipeline, u64)> {
        let bytes = self.entry(EntryKind::ComputePipeline, key.digest(), compile)?;
        let pipeline = ComputePipeline::from_bytes_for_key(&bytes, key, compile).ok()?;
        Some((pipeline, bytes.len() as u64))
    }

    /// The render pipeline `compile` compiled under `key`, from the archive, and its encoded
    /// length.
    pub(crate) fn render_pipeline(
        &self,
        compile: &metal2vulkan::device::Device,
        key: &PipelineKey,
    ) -> Option<(RenderPipeline, u64)> {
        let bytes = self.entry(EntryKind::RenderPipeline, key.digest(), compile)?;
        let pipeline = RenderPipeline::from_bytes_for_key(&bytes, key, compile).ok()?;
        Some((pipeline, bytes.len() as u64))
    }

    /// The encoding of the artifact of `kind` under `key` for `compile`'s profile.
    fn entry(
        &self,
        kind: EntryKind,
        key: &Hash256,
        compile: &metal2vulkan::device::Device,
    ) -> Option<Arc<[u8]>> {
        let place = Place::new(kind, *key, compile.profile_fingerprint());
        lock(&self.0.contents).entries.get(&place).cloned()
    }

    /// The fingerprint of the device's profile.
    fn profile(&self) -> Hash256 {
        self.device().compile_device().profile_fingerprint()
    }

    /// Keep `used`, compiled for the profile of fingerprint `profile`, and the archive keys
    /// `keys`. An artifact already held is not encoded again.
    fn keep(
        &self,
        used: Vec<Artifact>,
        profile: &Hash256,
        keys: impl IntoIterator<Item = Hash256>,
    ) {
        let mut contents = lock(&self.0.contents);
        for artifact in used {
            let (kind, artifact_key) = match &artifact {
                Artifact::Function(function) => (EntryKind::Function, *function.key().digest()),
                Artifact::Compute(pipeline) => {
                    (EntryKind::ComputePipeline, *pipeline.key().digest())
                }
                Artifact::Render(pipeline) => (EntryKind::RenderPipeline, *pipeline.key().digest()),
            };
            let place = Place::new(kind, artifact_key, *profile);
            contents.entries.entry(place).or_insert_with(|| {
                Arc::from(match &artifact {
                    Artifact::Function(function) => function.to_bytes(),
                    Artifact::Compute(pipeline) => pipeline.to_bytes(),
                    Artifact::Render(pipeline) => pipeline.to_bytes(),
                })
            });
        }
        contents.keys.extend(keys);
    }
}

/// Refuse a creation under `options` that lists another device's archive
/// (`Invalid(ForeignObject)` in `call`), or that fails on an archive miss and has a stage whose
/// archive key no listed archive holds: `missing` answers, from the device's profile and whether
/// a listed archive holds a key, the function of the first such stage, which the miss names with
/// the kind of `pipeline`.
pub(crate) fn check<'a>(
    device: &Device,
    options: &PipelineOptions,
    call: &'static str,
    pipeline: ArchivedPipeline,
    missing: impl FnOnce(&Hash256, &dyn Fn(&Hash256) -> bool) -> Option<&'a StageSource>,
) -> Result<(), Error> {
    let archives = &options.binary_archives;
    if let Some(archive) = archives
        .iter()
        .find(|archive| !archive.device().same(device))
    {
        return Err(archive.device().foreign(call, device));
    }
    if archives.is_empty()
        || !options
            .options
            .contains(PipelineOption::FAIL_ON_BINARY_ARCHIVE_MISS)
    {
        return Ok(());
    }
    let held = |key: &Hash256| archives.iter().any(|archive| archive.holds(key));
    let Some(function) = missing(&device.compile_device().profile_fingerprint(), &held) else {
        return Ok(());
    };
    let name = match function {
        StageSource::Function(descriptor) => descriptor.function.info().name.to_string(),
        StageSource::Compiled(compiled) => format!("{:?}", compiled.key()),
    };
    Err(Error::BinaryArchiveMiss(BinaryArchiveMiss::new(
        &name, pipeline,
    )))
}

/// The kernel of a compute creation of `descriptor`, when no listed archive holds its key.
pub(crate) fn compute_missing<'a>(
    descriptor: &'a ComputePipelineDescriptor,
) -> impl FnOnce(&Hash256, &dyn Fn(&Hash256) -> bool) -> Option<&'a StageSource> + 'a {
    move |profile, held| {
        (!held(&metal_key::compute(profile, descriptor))).then_some(&descriptor.kernel)
    }
}

/// The mesh function of a mesh creation of `descriptor`, when no listed archive holds its key.
pub(crate) fn mesh_missing<'a>(
    descriptor: &'a MeshPipelineDescriptor,
) -> impl FnOnce(&Hash256, &dyn Fn(&Hash256) -> bool) -> Option<&'a StageSource> + 'a {
    move |profile, held| (!held(&metal_key::mesh(profile, descriptor))).then_some(&descriptor.mesh)
}

/// The tile function of a tile creation of `descriptor`, when no listed archive holds its key.
pub(crate) fn tile_missing<'a>(
    descriptor: &'a TileRenderPipelineDescriptor,
) -> impl FnOnce(&Hash256, &dyn Fn(&Hash256) -> bool) -> Option<&'a StageSource> + 'a {
    move |profile, held| (!held(&metal_key::tile(profile, descriptor))).then_some(&descriptor.tile)
}

/// Whether a render creation of `descriptor` under `options`, which fails on an archive miss,
/// found both its stages' keys but not the two from one add: one add's vertex stage with another
/// add's fragment stage, under enabled rasterization. Metal's vertex key holds the vertex outputs
/// its compiler finds the fragment reading, which this lookup does not decode, so Metal misses
/// such a creation where the two fragments read different outputs (`metal_key`'s module docs).
pub(crate) fn combines_stages(
    device: &Device,
    options: &PipelineOptions,
    descriptor: &RenderPipelineDescriptor,
) -> bool {
    let archives = &options.binary_archives;
    if archives.is_empty()
        || !options
            .options
            .contains(PipelineOption::FAIL_ON_BINARY_ARCHIVE_MISS)
    {
        return false;
    }
    let held = |key: &Hash256| archives.iter().any(|archive| archive.holds(key));
    render_combined(descriptor)(&device.compile_device().profile_fingerprint(), &held)
}

/// Whether a render lookup of `descriptor` found both its stages but no add's pair of them
/// ([`combines_stages`]); never for a creation without a fragment stage that reads.
pub(crate) fn render_combined(
    descriptor: &RenderPipelineDescriptor,
) -> impl FnOnce(&Hash256, &dyn Fn(&Hash256) -> bool) -> bool + '_ {
    move |profile, held| {
        if descriptor.fragment.is_none() || !descriptor.rasterization_enabled {
            return false;
        }
        let keys = metal_key::render(profile, descriptor);
        let Some(kind) = keys.vertex.held(held) else {
            return false;
        };
        let found = keys
            .fragment
            .is_some_and(|fragment| held(&fragment.of(kind)));
        found && keys.pair(kind).is_some_and(|pair| !held(&pair))
    }
}

/// The function of the first stage of a render creation of `descriptor`, fragment then vertex,
/// whose key no listed archive holds (`metal_key`). An archive holds only the vertex key that
/// fits its function, so the one held says which of the fragment's keys to look for; with
/// neither held, the vertex misses, after the fragment if neither of its keys is held.
pub(crate) fn render_missing<'a>(
    descriptor: &'a RenderPipelineDescriptor,
) -> impl FnOnce(&Hash256, &dyn Fn(&Hash256) -> bool) -> Option<&'a StageSource> + 'a {
    move |profile, held| {
        let keys = metal_key::render(profile, descriptor);
        let vertex = keys.vertex.held(held);
        if let (Some(fragment), Some(keys)) = (&descriptor.fragment, keys.fragment) {
            let found = match vertex {
                Some(kind) => held(&keys.of(kind)),
                None => keys.held(held).is_some(),
            };
            if !found {
                return Some(fragment);
            }
        }
        vertex.is_none().then_some(&descriptor.vertex)
    }
}
