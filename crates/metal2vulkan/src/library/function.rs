//! `Library`, `Function`, `FunctionInfo`, and function-constant values.
//!
//! A [`Library`] owns the container bytes and a function table; a [`Function`] is a library plus
//! a table index. Neither retains decoded AIR: each compile decodes, lowers, and drops it.
//!
//! - [`Library::parse`] accepts a metallib, a bitcode wrapper, or a raw bitcode module. A
//!   metallib's table comes from its container tags (every function `HASH` is verified, linear in
//!   the input); its bitcode and AIR metadata are decoded only when a function is compiled.
//! - A wrapper or raw module ([`Library::from_air`]) is one bitcode module that may hold several
//!   entry points. Enumerating them needs the module's AIR metadata, so construction decodes the
//!   module once under the standard budget and records **every** entry.
//!
//! Selecting a function by name is Metal's `newFunctionWithName:` contract, not a name-keyed
//! heuristic.

use std::collections::BTreeMap;
use std::ops::Range;
use std::sync::Arc;

use crate::air::abi::Stage as AbiStage;
use crate::air::container::{self, ContainerConstant, Library as Container, LibraryFunction};
pub use crate::air::container::{FunctionStage, LibraryType};
use crate::air::error::{ContainerError, InputError, InputErrorKind, Location};
use crate::bind::{ArgumentBufferLayout, ArgumentLayoutError};
use crate::budget::{Budget, Meter};
use crate::cache::Hash256;
use crate::error::{DescriptorError, DiagName, Error, InternalError, InternalKind};
use crate::metal;
use crate::reflection::{
    FunctionConstant as ReflectedConstant, FunctionProperties, FunctionReflection,
};

/// A metallib, bitcode wrapper, or raw bitcode module, as owned bytes with its function table.
/// Cheap to clone; `Send + Sync`.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Library(Arc<LibraryInner>);

#[derive(Debug, PartialEq, Eq)]
struct LibraryInner {
    bytes: Box<[u8]>,
    content_hash: Hash256,
    functions: Vec<FunctionEntry>,
    /// What a metallib says of dynamic libraries; a bare module says nothing.
    dynamic: Dynamic,
}

/// A library's type, install name, dependencies, and (a dynamic library's) module.
#[derive(Debug, PartialEq, Eq)]
struct Dynamic {
    library_type: LibraryType,
    /// `HDYN`'s `NAME`, kept for a dynamic library only.
    install_name: Option<String>,
    /// `HDYN`'s `DYNL`s, in link order.
    dependencies: Vec<String>,
    /// A dynamic library's module inside the library bytes.
    module: Option<Range<usize>>,
}

impl Dynamic {
    /// An executable library naming no dynamic library: a bare module, which has no container to
    /// name one.
    fn none() -> Dynamic {
        Dynamic {
            library_type: LibraryType::Executable,
            install_name: None,
            dependencies: Vec::new(),
            module: None,
        }
    }
}

/// One row of the function table.
#[derive(Debug, PartialEq, Eq)]
struct FunctionEntry {
    name: String,
    stage: FunctionStage,
    air_hash: Hash256,
    constants: Vec<FunctionConstant>,
    /// The function's bitcode inside the library bytes: a metallib function's wrapped module, or
    /// the whole module of a wrapper or raw input.
    bitcode: Range<usize>,
    source: EntrySource,
}

/// Where an entry's AIR comes from, with what its compile needs besides the bitcode.
#[derive(Debug, PartialEq, Eq)]
enum EntrySource {
    /// A metallib function: its container metadata, which compile cross-checks against the
    /// decoded AIR metadata (`AirAbi::check_container`).
    Metallib(MetallibTags),
    /// Entry `index` of the module's AIR metadata (`AirAbi::entries` order).
    Module { index: u32 },
    /// Visible function `index` the module exports (`!air.visible`, `AirAbi::visible_functions`
    /// order): linked into pipelines, never compiled as a stage.
    Visible { index: u32 },
}

/// The container tags of a metallib function that its compile cross-checks, and those Metal
/// answers the function's reflection and properties from.
#[derive(Debug, PartialEq, Eq)]
struct MetallibTags {
    air_version: (u16, u16),
    language_version: (u16, u16),
    /// `CNST`: (name, raw `MTLDataType`, index, raw flags).
    constants: Vec<(String, u8, u16, u8)>,
    /// Whether the function carries an `RFLT` tag: the reflection Metal's library answers from,
    /// which the compiler driver writes and `metallib` of `.air` files does not.
    reflection: bool,
    /// `VATT`: each attribute's name and packed index and flags.
    vertex_attributes: Vec<(String, u16)>,
    /// `VATY`: each attribute's raw `MTLDataType`.
    vertex_attribute_types: Vec<u8>,
    /// `TESS`, raw: a post-tessellation vertex function's patch.
    tessellation: Option<Box<[u8]>>,
}

impl Library {
    /// Parse a metallib, a bitcode wrapper (`0x0b17c0de`), or a raw bitcode module. A metallib is
    /// parsed from its container tags alone (no bitcode is decoded); a wrapper or raw module is
    /// decoded once to enumerate its entry points, as [`Library::from_air`]. A universal file
    /// (Metal's own, or a Mach-O universal file) is the metallib its AIR slice holds: the library
    /// keeps and hashes that slice alone ([`Library::bytes`]), and its slices of GPU code are not
    /// read.
    pub fn parse(bytes: &[u8]) -> Result<Library, Error> {
        crate::pipeline::compile::guarded(|| match container::parse(bytes)? {
            Container::Metallib(ml) => from_metallib(&ml),
            Container::Module(module) => from_module(bytes, module.hash),
        })
    }

    /// Bare AIR bitcode (wrapper or raw) as a library of every entry point it declares, in the
    /// order of its AIR metadata (`!air.kernel`, `!air.vertex`, `!air.fragment`), then every
    /// visible function it exports (`!air.visible`, [`FunctionStage::Visible`]), which a pipeline
    /// links (`FunctionDescriptor::linked`). Decodes the module once under the standard budget;
    /// nothing decoded is retained. A metallib is refused (`Container(BadMagic)`);
    /// [`Library::parse`] takes every form.
    pub fn from_air(bytes: &[u8]) -> Result<Library, Error> {
        crate::pipeline::compile::guarded(|| match container::parse(bytes)? {
            Container::Module(module) => from_module(bytes, module.hash),
            Container::Metallib(_) => Err(InputError::new(
                InputErrorKind::Container(ContainerError::BadMagic),
                Location::container(0),
            )
            .into()),
        })
    }

    /// A metallib as a library, parsed from its container tags alone, as [`Library::parse`]
    /// parses one. Bare AIR bitcode (wrapper or raw) is refused (`Container(BadMagic)`), as
    /// Metal's `newLibraryWithData:error:` refuses it, alike with any other bytes that are no
    /// metallib; [`Library::parse`] and [`Library::from_air`] take it.
    pub fn from_metallib(bytes: &[u8]) -> Result<Library, Error> {
        crate::pipeline::compile::guarded(|| match container::parse(bytes)? {
            Container::Metallib(ml) => from_metallib(&ml),
            Container::Module(_) => Err(InputError::new(
                InputErrorKind::Container(ContainerError::BadMagic),
                Location::container(0),
            )
            .into()),
        })
    }

    /// The container bytes the library was parsed from: of a universal file, its AIR slice.
    pub fn bytes(&self) -> &[u8] {
        &self.0.bytes
    }

    /// SHA-256 of [`Library::bytes`].
    pub fn content_hash(&self) -> &Hash256 {
        &self.0.content_hash
    }

    /// The number of functions in the table.
    pub fn len(&self) -> u32 {
        u32::try_from(self.0.functions.len()).unwrap_or(u32::MAX)
    }

    /// Whether the table is empty.
    pub fn is_empty(&self) -> bool {
        self.0.functions.is_empty()
    }

    /// Every function, in table order.
    pub fn functions(&self) -> impl Iterator<Item = FunctionInfo<'_>> {
        self.0.functions.iter().map(FunctionEntry::info)
    }

    /// The function named `name` (`newFunctionWithName:`). No match is
    /// `InvalidDescriptor(EntryNotFound)`; more than one is `InvalidDescriptor(AmbiguousEntry)`.
    pub fn function(&self, name: &str) -> Result<Function, Error> {
        let mut found = (0u32..)
            .zip(&self.0.functions)
            .filter(|(_, f)| f.name == name);
        match (found.next(), found.next()) {
            (Some((index, _)), None) => Ok(Function {
                library: self.clone(),
                index,
            }),
            (Some(_), Some(_)) => Err(Error::InvalidDescriptor(DescriptorError::AmbiguousEntry {
                requested: DiagName::new(name),
            })),
            (None, _) => Err(self.not_found(DiagName::new(name))),
        }
    }

    /// The function at `index` in the table.
    pub fn function_at(&self, index: u32) -> Result<Function, Error> {
        match self.entry(index) {
            Some(_) => Ok(Function {
                library: self.clone(),
                index,
            }),
            None => Err(self.not_found(DiagName::new(format!("#{index}")))),
        }
    }

    /// `MTLLibrary.type`: [`LibraryType::Dynamic`] for a metallib of file type 2 (a library
    /// [`DynamicLibrary::new`](crate::library::DynamicLibrary::new) takes), else
    /// [`LibraryType::Executable`]. A bare module is executable.
    pub fn library_type(&self) -> LibraryType {
        self.0.dynamic.library_type
    }

    /// `MTLLibrary.installName`: a dynamic library's install name, the name another library's
    /// dependency on it gives; `None` for an executable library.
    pub fn install_name(&self) -> Option<&str> {
        self.0.dynamic.install_name.as_deref()
    }

    /// The install names of the dynamic libraries this library was linked against
    /// (`MTLCompileOptions.libraries`), in link order: a pipeline stage searches the libraries
    /// they name, after its preloaded libraries, for the symbols the stage imports. Empty for a
    /// bare module, which has no container to name them.
    pub fn dependencies(&self) -> impl Iterator<Item = &str> {
        self.0.dynamic.dependencies.iter().map(String::as_str)
    }

    /// A dynamic library's module: the bitcode every function it exports shares.
    pub(crate) fn dynamic_module(&self) -> Option<&[u8]> {
        self.0
            .dynamic
            .module
            .clone()
            .and_then(|r| self.0.bytes.get(r))
    }

    fn not_found(&self, requested: DiagName) -> Error {
        Error::InvalidDescriptor(DescriptorError::EntryNotFound {
            requested,
            available: self.len(),
        })
    }

    fn entry(&self, index: u32) -> Option<&FunctionEntry> {
        usize::try_from(index)
            .ok()
            .and_then(|i| self.0.functions.get(i))
    }
}

fn invariant(detail: &str) -> Error {
    Error::Internal(InternalError::new(
        InternalKind::Invariant,
        detail.to_owned(),
    ))
}

/// The byte range of `part` inside `whole` (`part` is a subslice the container parser returned).
fn range_in(whole: &[u8], part: &[u8]) -> Result<Range<usize>, Error> {
    let start = (part.as_ptr() as usize).wrapping_sub(whole.as_ptr() as usize);
    match start.checked_add(part.len()) {
        Some(end)
            if whole
                .get(start..end)
                .is_some_and(|s| s.as_ptr() == part.as_ptr()) =>
        {
            Ok(start..end)
        }
        _ => Err(invariant("a container slice outside the container")),
    }
}

fn from_metallib(ml: &container::Metallib<'_>) -> Result<Library, Error> {
    let bytes = ml.bytes;
    let mut functions = Vec::with_capacity(ml.functions.len());
    for f in &ml.functions {
        let cnst = &f.public.constants;
        functions.push(FunctionEntry {
            name: f.name.to_owned(),
            stage: f.stage,
            air_hash: f.hash,
            constants: cnst
                .iter()
                .map(|c| FunctionConstant {
                    name: c.name.to_owned(),
                    index: Some(u32::from(c.index)),
                    // A value outside the known set stays unknown: compile's container
                    // cross-check refuses that function, not the whole library.
                    ty: metal::DataType::try_from(u32::from(c.data_type)).ok(),
                    raw_flag: Some(c.flags),
                })
                .collect(),
            bitcode: range_in(bytes, f.bitcode)?,
            source: EntrySource::Metallib(MetallibTags {
                air_version: f.air_version,
                language_version: f.language_version,
                constants: cnst
                    .iter()
                    .map(|c| (c.name.to_owned(), c.data_type, c.index, c.flags))
                    .collect(),
                reflection: f.raw_tags.iter().any(|t| &t.fourcc == b"RFLT"),
                vertex_attributes: f
                    .public
                    .vertex_attributes
                    .iter()
                    .map(|a| (a.name.to_owned(), a.packed))
                    .collect(),
                vertex_attribute_types: f.public.vertex_attribute_types.clone(),
                tessellation: f
                    .raw_tags
                    .iter()
                    .find(|t| &t.fourcc == b"TESS")
                    .map(|t| t.bytes.into()),
            }),
        });
    }
    let library_type = ml.library_type();
    let dynamic = Dynamic {
        library_type,
        install_name: match library_type {
            LibraryType::Dynamic => ml.linkage.install_name.map(str::to_owned),
            _ => None,
        },
        dependencies: ml
            .linkage
            .dependencies
            .iter()
            .map(|d| (*d).to_owned())
            .collect(),
        module: ml
            .linkage
            .module
            .map(|module| range_in(bytes, module))
            .transpose()?,
    };
    Ok(Library(Arc::new(LibraryInner {
        bytes: bytes.into(),
        content_hash: container::sha256(bytes),
        functions,
        dynamic,
    })))
}

fn from_module(bytes: &[u8], hash: Hash256) -> Result<Library, Error> {
    let mut m = Meter::new(&Budget::standard());
    let decoded = crate::air::decode(bytes, &mut m)?;
    let abi = &decoded.abi;
    // The metadata lists function constants per module, so every entry carries all of them. It
    // spells a constant's type only as an MSL name, so `ty` stays unknown.
    let constants: Vec<FunctionConstant> = abi
        .function_constants
        .iter()
        .map(|c| FunctionConstant {
            name: c.name.to_string(),
            index: Some(c.index),
            ty: c.data_type,
            raw_flag: c.flag.map(u8::from),
        })
        .collect();
    let mut functions = Vec::with_capacity(abi.entries.len() + abi.visible_functions.len());
    for (index, e) in (0u32..).zip(&abi.entries) {
        functions.push(FunctionEntry {
            name: e.name.to_string(),
            stage: match e.stage {
                AbiStage::Kernel { .. } => FunctionStage::Kernel,
                AbiStage::Vertex { .. } => FunctionStage::Vertex,
                AbiStage::Fragment { .. } => FunctionStage::Fragment,
                AbiStage::Object { .. } => FunctionStage::Object,
                AbiStage::Mesh { .. } => FunctionStage::Mesh,
            },
            air_hash: hash,
            constants: constants.clone(),
            bitcode: 0..bytes.len(),
            source: EntrySource::Module { index },
        });
    }
    // The visible functions, by their symbol: the name a visible-function reference links.
    for (index, v) in (0u32..).zip(&abi.visible_functions) {
        let symbol = decoded
            .module
            .func(v.func)
            .and_then(|f| decoded.module.symbols().get(f.name))
            .ok_or_else(|| invariant("a visible function outside its module"))?;
        functions.push(FunctionEntry {
            name: String::from_utf8_lossy(symbol).into_owned(),
            stage: FunctionStage::Visible,
            air_hash: hash,
            constants: constants.clone(),
            bitcode: 0..bytes.len(),
            source: EntrySource::Visible { index },
        });
    }
    Ok(Library(Arc::new(LibraryInner {
        bytes: bytes.into(),
        content_hash: hash,
        functions,
        dynamic: Dynamic::none(),
    })))
}

impl FunctionEntry {
    fn info(&self) -> FunctionInfo<'_> {
        FunctionInfo {
            name: &self.name,
            stage: self.stage,
            air_hash: self.air_hash,
            constants: &self.constants,
        }
    }
}

/// One function of a [`Library`]: the library and the function's table index. Only
/// [`Library::function`] and [`Library::function_at`] make one, so the index is in the table.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Function {
    library: Library,
    index: u32,
}

impl Function {
    /// The library the function belongs to.
    pub fn library(&self) -> &Library {
        &self.library
    }

    /// The function's index in the library's function table.
    pub fn index(&self) -> u32 {
        self.index
    }

    /// What the library says about the function, without decoding its bitcode.
    pub fn info(&self) -> FunctionInfo<'_> {
        match self.library.entry(self.index) {
            Some(e) => e.info(),
            // Unreachable: a `Function` is only made for an index in its library's table.
            None => FunctionInfo {
                name: "",
                stage: FunctionStage::Other(u8::MAX),
                air_hash: [0; 32],
                constants: &[],
            },
        }
    }

    /// The layout of the argument buffer the function takes at buffer index `index`, with the
    /// function constants `constants`: what `newArgumentEncoderWithBufferIndex:` encodes into
    /// (the layout rules are [`ArgumentBufferLayout`]'s). It decodes the function's AIR under
    /// `budget`, and specializes it with `constants` only where a function constant decides the
    /// argument buffer.
    ///
    /// Refused ([`ArgumentLayoutError`]): no argument buffer at the index (a plain buffer, or
    /// nothing), an array of arrays, for which Metal makes no encoder, and the decode's or
    /// specialization's own errors (`Compile`).
    pub fn argument_buffer_layout(
        &self,
        index: u32,
        constants: &FunctionConstantValues,
        budget: &Budget,
    ) -> Result<ArgumentBufferLayout, ArgumentLayoutError> {
        let mut m = Meter::new(budget);
        crate::pipeline::compile::argument_buffer_layout(self, constants, index, &mut m)
    }

    /// The function's reflection, as `-[MTLLibrary reflectionForFunctionWithName:]` answers it
    /// (the rules are [`reflection`](crate::reflection)'s). `None` for a metallib function without
    /// the reflection Metal answers from (no `RFLT` tag), as Metal answers nil. A function of a
    /// bitcode wrapper or raw module answers from its AIR, which Metal cannot load. It decodes the
    /// function's AIR under `budget`.
    ///
    /// Refused: an argument whose answer Metal's has not been measured for, and an index a
    /// function constant computes, for which Metal's library states no value the constant gives
    /// (both `Unsupported` with `Feature::Reflection`), and the decode's own errors.
    pub fn reflection(&self, budget: &Budget) -> Result<Option<FunctionReflection>, Error> {
        let e = self.entry()?;
        if let EntrySource::Metallib(tags) = &e.source {
            if !tags.reflection {
                return Ok(None);
            }
        }
        let mut m = Meter::new(budget);
        crate::reflection::function_reflection(self, &mut m).map(Some)
    }

    /// What `MTLFunction` states about the function beyond its name and bindings (the rules are
    /// [`FunctionProperties`]'): its `functionType`, `patchType`, `patchControlPointCount`,
    /// `vertexAttributes` and `stageInputAttributes`. A metallib function answers from its
    /// container tags, as Metal does, decoding nothing; a function of a bitcode wrapper or raw
    /// module answers from its AIR, which Metal cannot load, decoding it under `budget`.
    ///
    /// Refused (`Unsupported` with `Feature::Reflection`): a tag value Metal's answer has not
    /// been measured for, and an attribute index a function constant computes; and the decode's
    /// own errors.
    pub fn properties(&self, budget: &Budget) -> Result<FunctionProperties, Error> {
        let e = self.entry()?;
        match &e.source {
            EntrySource::Metallib(tags) => crate::reflection::metallib_properties(
                e.stage,
                &tags.vertex_attributes,
                &tags.vertex_attribute_types,
                tags.tessellation.as_deref(),
            ),
            EntrySource::Visible { .. } => Ok(crate::reflection::visible_properties()),
            EntrySource::Module { index } => {
                let bitcode = self.bitcode()?;
                let mut m = Meter::new(budget);
                crate::pipeline::compile::guarded(|| {
                    let decoded = crate::air::decode(bitcode, &mut m)?;
                    let ep = usize::try_from(*index)
                        .ok()
                        .and_then(|i| decoded.abi.entries.get(i))
                        .ok_or_else(|| invariant("a library entry outside its module"))?;
                    crate::reflection::entry_properties(ep)
                })
            }
        }
    }

    /// What `MTLFunction` states about the function made with the constant values `constants`
    /// (`newFunctionWithName:constantValues:error:`), as [`Function::properties`] states about
    /// the function itself, except that an attribute a constant gates off is left out, as Metal
    /// leaves it out: those are answered from the AIR specialized with `constants` (decoded under
    /// `budget`), for a metallib function too. A function made so has no constants left in its
    /// `functionConstantsDictionary`, which Metal answers empty.
    ///
    /// Refused as [`Function::properties`] refuses, and with the specialization's own errors
    /// (a constant value of the wrong type, a required constant without one).
    pub fn specialized_properties(
        &self,
        constants: &FunctionConstantValues,
        budget: &Budget,
    ) -> Result<FunctionProperties, Error> {
        let properties = self.properties(budget)?;
        // Constants decide only stage-input attributes among the properties.
        if properties.stage_input_attributes().is_none() {
            return Ok(properties);
        }
        let mut m = Meter::new(budget);
        crate::pipeline::compile::specialized_properties(self, constants, &mut m)
    }

    /// `functionConstantsDictionary`: the function constants the function reads, by name, each
    /// required unless the function asks whether it is defined (`is_function_constant_defined`),
    /// as measured. A metallib function answers from its `CNST` tag, as Metal does. A bitcode
    /// wrapper's or raw module's AIR lists its constants per module; such a function answers
    /// those its code, its callees, and its entry's metadata read, directly or through the
    /// globals the module's static initializers compute from them, as the metallib linker
    /// writes them into `CNST`, decoded under the standard budget.
    ///
    /// Refused (`Unsupported` with `Feature::Reflection`): a `CNST` type or flag Metal's answer
    /// has not been measured for; a module function whose reads its AIR leaves unknown (a static
    /// initializer that branches on a constant or stores a pointer, a function address its code
    /// takes); and the decode's own errors.
    pub fn function_constants_dictionary(
        &self,
    ) -> Result<BTreeMap<String, ReflectedConstant>, Error> {
        let e = self.entry()?;
        let index = match &e.source {
            EntrySource::Metallib(tags) => {
                return crate::reflection::metallib_constants(&tags.constants);
            }
            _ if e.constants.is_empty() => return Ok(BTreeMap::new()),
            EntrySource::Module { index } | EntrySource::Visible { index } => *index,
        };
        let bitcode = self.bitcode()?;
        let mut m = Meter::new(&Budget::standard());
        crate::pipeline::compile::guarded(|| {
            let decoded = crate::air::decode(bitcode, &mut m)?;
            let abi = &decoded.abi;
            let i = usize::try_from(index).ok();
            let func = match &e.source {
                EntrySource::Visible { .. } => {
                    i.and_then(|i| abi.visible_functions.get(i)).map(|v| v.func)
                }
                _ => i.and_then(|i| abi.entries.get(i)).map(|ep| ep.func),
            }
            .ok_or_else(|| invariant("a library function outside its module"))?;
            crate::reflection::module_constants(&decoded.module, abi, func, &e.name, &mut m)
        })
    }

    /// The function's row of its library's table.
    fn entry(&self) -> Result<&FunctionEntry, Error> {
        self.library
            .entry(self.index)
            .ok_or_else(|| invariant("a function index outside its library"))
    }

    /// The function's bitcode and how compile selects and cross-checks its entry.
    pub(crate) fn source(&self) -> Result<FunctionSource<'_>, Error> {
        let e = self.entry()?;
        let bitcode = self
            .library
            .bytes()
            .get(e.bitcode.clone())
            .ok_or_else(|| invariant("a function's bitcode outside its library"))?;
        Ok(match &e.source {
            // A metallib's visible function is no stage, as a module's is not; nor is a function
            // a dynamic library exports, or a Core Image kernel library's extern function.
            EntrySource::Metallib(_)
                if matches!(
                    e.stage,
                    FunctionStage::Visible | FunctionStage::Exported | FunctionStage::Extern
                ) =>
            {
                FunctionSource::Visible
            }
            EntrySource::Metallib(tags) => FunctionSource::Metallib(Box::new(LibraryFunction {
                name: &e.name,
                stage: e.stage,
                hash: e.air_hash,
                air_version: tags.air_version,
                language_version: tags.language_version,
                bitcode,
                public: container::FunctionPublicMetadata {
                    constants: tags
                        .constants
                        .iter()
                        .map(|(name, data_type, index, flags)| ContainerConstant {
                            name,
                            data_type: *data_type,
                            index: *index,
                            flags: *flags,
                        })
                        .collect(),
                    ..container::FunctionPublicMetadata::default()
                },
                private: Vec::new(),
                raw_tags: Vec::new(),
            })),
            EntrySource::Module { index } => FunctionSource::Module {
                bitcode,
                index: *index,
            },
            EntrySource::Visible { .. } => FunctionSource::Visible,
        })
    }

    /// The bitcode module that defines the function, as a pipeline links it: a metallib function's
    /// own module, or the whole wrapper or raw module.
    pub(crate) fn bitcode(&self) -> Result<&[u8], Error> {
        let e = self.entry()?;
        self.library
            .bytes()
            .get(e.bitcode.clone())
            .ok_or_else(|| invariant("a function's bitcode outside its library"))
    }
}

/// A function's bitcode as compile consumes it.
pub(crate) enum FunctionSource<'a> {
    /// A metallib function: its own wrapped module (`bitcode`) and the container tags compile
    /// cross-checks.
    Metallib(Box<LibraryFunction<'a>>),
    /// Entry `index` of a wrapper or raw module.
    Module { bitcode: &'a [u8], index: u32 },
    /// A visible function, a function a dynamic library exports, or a Core Image kernel
    /// library's extern function: not a stage.
    Visible,
}

/// What a library says about one function without decoding its bitcode.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FunctionInfo<'a> {
    pub name: &'a str,
    pub stage: FunctionStage,
    /// SHA-256 of the function's bitcode (its content key): a metallib function's verified
    /// `HASH`, or the hash of the whole wrapper or raw module its entries share.
    pub air_hash: Hash256,
    /// Function constants from the metallib `CNST` tag, or from the module's AIR metadata.
    pub constants: &'a [FunctionConstant],
}

/// A function constant the function declares.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FunctionConstant {
    pub name: String,
    /// The `[[function_constant(index)]]` index, when the constant has one.
    pub index: Option<u32>,
    /// The constant's `MTLDataType`, when the container states a known one. `None` for a wrapper
    /// or raw module, whose AIR metadata spells the type only as an MSL name, and for a `CNST`
    /// value outside the known set.
    pub ty: Option<metal::DataType>,
    /// The raw flag byte of the `CNST` entry, which states Metal's `MTLFunctionConstant.required`
    /// (1 true, 0 false, as measured: [`Function::function_constants_dictionary`]), or the
    /// optional fifth operand of the AIR metadata entry (0 or 1), whose meaning is open: recorded,
    /// never interpreted.
    pub raw_flag: Option<u8>,
}

/// Function-constant values supplied with a function (`MTLFunctionConstantValues`), by index or by
/// name. A constant with no value is "not defined": `is_function_constant_defined` folds to false
/// and arguments gated on it are absent; a surviving value use is
/// `Error::NeedsInput(FunctionConstant)`.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FunctionConstantValues {
    pub by_index: BTreeMap<u32, FunctionConstantValue>,
    pub by_name: BTreeMap<String, FunctionConstantValue>,
}

impl FunctionConstantValues {
    /// No values: every constant is undefined (Metal semantics, and what PVG guests send).
    pub fn defaults() -> FunctionConstantValues {
        FunctionConstantValues::default()
    }

    /// Whether no value is supplied.
    pub fn is_empty(&self) -> bool {
        self.by_index.is_empty() && self.by_name.is_empty()
    }

    /// Supplies the value of the constant at `index` (`setConstantValue:type:atIndex:`).
    pub fn with_index(mut self, index: u32, value: FunctionConstantValue) -> Self {
        self.by_index.insert(index, value);
        self
    }

    /// Supplies the value of the constant named `name` (`setConstantValue:type:withName:`).
    pub fn with_name(mut self, name: impl Into<String>, value: FunctionConstantValue) -> Self {
        self.by_name.insert(name.into(), value);
        self
    }
}

/// One function-constant value: its Metal data type and its bytes, exactly as Metal's
/// `setConstantValue:type:` takes them. Whether the bytes fit the type and the declared constant is
/// checked at compile (`DescriptorError::FunctionConstantType`).
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FunctionConstantValue {
    pub ty: metal::DataType,
    pub bytes: Box<[u8]>,
}

impl FunctionConstantValue {
    /// A value of type `ty` with these bytes.
    pub fn new(ty: metal::DataType, bytes: impl Into<Box<[u8]>>) -> FunctionConstantValue {
        FunctionConstantValue {
            ty,
            bytes: bytes.into(),
        }
    }
}
