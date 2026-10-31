//! `Library` and `Function` (`MTLLibrary`, `MTLFunction`) and `Device::new_library_with_data`.
//!
//! A runtime library is the compile layer's [`library::Library`](metal2vulkan::library::Library) held by
//! its device: the AIR input, parsed once and decoded only when a pipeline state compiles one of
//! its functions. `new_library_with_data` takes a metallib alone, as Metal does; bare AIR bitcode
//! comes in parsed by the translator (`new_library_with_compile_library`). Libraries of every
//! platform parse alike. A function carries the constant values it was made with
//! (`newFunctionWithName:constantValues:`), which every pipeline state compiled from it specializes
//! it with, as its stage or as a function the stage links ([`Function::linked_function`],
//! `MTLLinkedFunctions`), and the options it was made with (`newFunctionWithDescriptor:`): one
//! compiled to a binary is what Metal takes in a stage's `binaryFunctions` and adds to a made
//! pipeline state.

use super::device::device_error;
use super::error::{invalid, Problem, RuntimeGap};
use super::Device;
use super::Error;
use crate::derived::{Derived, DerivedMap};
use crate::resource_ids::{function_name, FunctionHolds, LibraryHolds};
use metal2vulkan::bind::ArgumentBufferLayout;
use metal2vulkan::library::{FunctionConstantValues, FunctionStage, LibraryType};
use metal2vulkan::metal::{FunctionOptions, FunctionType, PatchType};
use metal2vulkan::pipeline::{FunctionDescriptor as StageFunction, LinkedFunction};
use metal2vulkan::reflection::{
    Attribute, FunctionConstant, FunctionProperties, FunctionReflection,
};
use metal2vulkan::Budget;
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

/// A Metal library (`MTLLibrary`). Cheap to clone; `Send + Sync`.
#[derive(Clone, Debug)]
pub struct Library {
    device: Device,
    library: metal2vulkan::library::Library,
    /// What the library holds of the sampler IDs its functions' pipeline states numbered, shared
    /// with every live library of the same content (`resource_ids/numbered.rs`).
    holds: Arc<LibraryHolds>,
    /// The reflections of the library's functions by name, derived when
    /// [`reflection_for_function_with_name`](Library::reflection_for_function_with_name) first
    /// asks for one.
    ///
    /// Deriving one for a function whose library is a module decodes that function's AIR
    /// ([`metal2vulkan::library::Function::reflection`]), so the getter decoded the same bitcode
    /// once per call, where `MTLLibrary` answers from the library it already compiled. Every
    /// input is fixed when the library is made -- its bytes are immutable behind an `Arc` and the
    /// standard budget never changes -- so the memo is sound, and it is the library's rather than
    /// a function object's because the getter takes a name and makes no function object.
    reflections: Arc<Mutex<DerivedMap<String, Option<FunctionReflection>>>>,
}

/// A Metal function of a library (`MTLFunction`), with the constant values it was made with.
/// Cheap to clone; `Send + Sync`.
#[derive(Clone, Debug)]
pub struct Function {
    device: Device,
    function: metal2vulkan::library::Function,
    constants: FunctionConstantValues,
    /// Whether the function was made with constant values (`newFunctionWithName:constantValues:`),
    /// whose `MTLFunction` properties are the specialized function's.
    specialized: bool,
    /// The layouts of the argument buffers the function takes, by buffer index, decoded when an
    /// argument encoder first asks (`argument_encoder.rs`).
    argument_layouts: Arc<Mutex<DerivedMap<u32, Arc<ArgumentBufferLayout>>>>,
    /// The function's `MTLFunction` properties, computed when a getter first asks.
    ///
    /// All five property getters answer one field each of a **single** [`FunctionProperties`],
    /// and computing it for a function whose library is a module decodes the function's AIR
    /// ([`metal2vulkan::library::Function::properties`]). Computing it per getter therefore
    /// decodes the same bitcode once per getter to answer one `MTLFunction`, where Metal answers
    /// each getter from the object it already made; a specialized function pays twice over,
    /// because [`metal2vulkan::library::Function::specialized_properties`] computes the
    /// unspecialized properties first and then specializes the AIR again.
    ///
    /// The memo is sound because every input is fixed when the function is made: the library's
    /// bytes are immutable behind an `Arc`, and `constants` and `specialized` never change. Only
    /// a success is kept -- a refusal under the standard budget is deterministic, so recomputing
    /// one answers the same refusal -- which keeps the error path exactly as it was.
    properties: Arc<Mutex<Derived<FunctionProperties>>>,
    /// The function's `functionConstantsDictionary`, derived when the getter first asks.
    ///
    /// Deriving it for a function whose library is a module decodes the function's AIR
    /// ([`metal2vulkan::library::Function::function_constants_dictionary`]) -- the sixth decode a
    /// reflection call made, beside the five the property getters made -- where Metal answers
    /// from the `MTLFunction` it already made. Sound for the same reasons as `properties`: every
    /// input is fixed when the function is made.
    constants_dictionary: Arc<Mutex<Derived<BTreeMap<String, FunctionConstant>>>>,
    /// The options the function was made with (`newFunctionWithDescriptor:`); none for the
    /// other forms.
    options: FunctionOptions,
    /// What the function object holds of the sampler IDs its function's pipeline states
    /// numbered, shared with every function object of the same function of its library
    /// (`resource_ids/numbered.rs`); held for its drop.
    _holds: Arc<FunctionHolds>,
}

/// `MTLFunctionDescriptor`: the function [`Library::new_function_with_descriptor`] makes.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct FunctionDescriptor {
    /// `name`.
    pub name: String,
    /// `constantValues`: `None` for nil, the function as `newFunctionWithName:` makes it.
    pub constant_values: Option<FunctionConstantValues>,
    /// `options`. `COMPILE_TO_BINARY` makes a function a stage's `binary_functions` and a made
    /// state's additions take; the other three change nothing a program sees, and every
    /// combination of the four is taken and answered back
    /// ([`Library::new_function_with_descriptor`]).
    pub options: FunctionOptions,
}

impl FunctionDescriptor {
    /// The descriptor of the function named `name`, with no constant values and no options.
    pub fn new(name: &str) -> FunctionDescriptor {
        FunctionDescriptor {
            name: name.to_string(),
            constant_values: None,
            options: FunctionOptions::NONE,
        }
    }
}

impl Device {
    /// `newLibraryWithData:error:`: the library of a metallib. Bytes that are no metallib are
    /// refused (`Compile(InvalidInput)`), bare AIR bitcode among them, raw or in a bitcode
    /// wrapper (the `.air` file `metal -c` writes), as Metal refuses each: nil, with
    /// `MTLLibraryErrorDomain` code 1 ("Invalid library file") alike for AIR and for any other
    /// bytes that are no metallib, on an Apple M4. AIR goes in through the translator
    /// ([`Device::new_library_with_compile_library`]).
    pub fn new_library_with_data(&self, data: &[u8]) -> Result<Library, Error> {
        let library = metal2vulkan::library::Library::from_metallib(data).map_err(device_error)?;
        Ok(self.new_library_with_compile_library(library))
    }

    /// `newLibraryWithSource:options:error:` (and its completion-handler form): a library
    /// compiled from Metal Shading Language source. Not built: the runtime compiles AIR, not
    /// MSL, so every source is refused, `NotImplemented(LibrarySource)`
    /// ([`RuntimeGap::LibrarySource`](crate::RuntimeGap::LibrarySource)), where Metal compiles it.
    /// `MTLCompileOptions` has no counterpart: no option changes the answer. Compile the source
    /// ahead of time with Apple's `metal` tool and load the metallib with
    /// [`new_library_with_data`](Self::new_library_with_data).
    pub fn new_library_with_source(&self, _source: &str) -> Result<Library, Error> {
        Err(Error::NotImplemented(RuntimeGap::LibrarySource))
    }

    /// The library the translator parsed, held by this device: a metallib, or bare AIR bitcode
    /// (raw or wrapped), which only the translator takes
    /// ([`Library::parse`](metal2vulkan::library::Library::parse),
    /// [`Library::from_air`](metal2vulkan::library::Library::from_air)). No Metal counterpart:
    /// Metal's libraries come from metallibs alone.
    pub fn new_library_with_compile_library(
        &self,
        library: metal2vulkan::library::Library,
    ) -> Library {
        let holds = LibraryHolds::of(&self.shared().resource_ids, *library.content_hash());
        Library {
            device: self.clone(),
            library,
            holds,
            reflections: Arc::default(),
        }
    }
}

impl Library {
    /// `device`.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// `functionNames`, in the library's table order. A Core Image kernel library's extern
    /// functions are not among them, as Metal leaves them out
    /// ([`extern_function_names`](Self::extern_function_names) lists them).
    pub fn function_names(&self) -> Vec<String> {
        self.library
            .functions()
            .filter(|info| info.stage != FunctionStage::Extern)
            .map(|info| info.name.to_string())
            .collect()
    }

    /// `externFunctionNames` (private to Metal's own library class): the extern functions of a
    /// Core Image kernel library (`metal -fcikernel`, `metallib -cikernel`), the kernels Core Image
    /// takes with [`new_extern_function_with_name`](Self::new_extern_function_with_name) and links
    /// into a kernel of its own; none for any other library. In the reverse of the library's table
    /// order, as Metal answers (libraries of two and of four kernels, an M3 Max under macOS
    /// 27.0.1).
    pub fn extern_function_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self
            .library
            .functions()
            .filter(|info| info.stage == FunctionStage::Extern)
            .map(|info| info.name.to_string())
            .collect();
        names.reverse();
        names
    }

    /// `type`: [`LibraryType::Dynamic`] for a dynamic library's metallib (or the universal file
    /// `DynamicLibrary::serialize_to_url` writes), which `Device::new_dynamic_library` takes,
    /// else [`LibraryType::Executable`].
    pub fn library_type(&self) -> LibraryType {
        self.library.library_type()
    }

    /// `installName`: a dynamic library's install name; `None` for an executable library.
    pub fn install_name(&self) -> Option<&str> {
        self.library.install_name()
    }

    /// The compile layer's library.
    pub(crate) fn compile_library(&self) -> &metal2vulkan::library::Library {
        &self.library
    }

    /// `newFunctionWithName:`: the function named `name`, with no constant values (each
    /// constant it declares is undefined). A name the library lacks, or names more than once, is
    /// refused (`Compile(InvalidDescriptor)`), where Metal answers nil. Every name of a dynamic
    /// library is refused (`Invalid(DynamicLibraryFunction)`): Metal stops at an assertion for
    /// each, its functions being no stage. A Core Image kernel library's extern function is made,
    /// as Metal makes it ([`new_extern_function_with_name`](Self::new_extern_function_with_name)).
    pub fn new_function_with_name(&self, name: &str) -> Result<Function, Error> {
        self.function(
            "new_function_with_name",
            name,
            &FunctionConstantValues::defaults(),
            false,
        )
    }

    /// `newExternFunctionWithName:` (private to Metal's own library class): the function named
    /// `name`, as [`new_function_with_name`](Self::new_function_with_name) makes and refuses it,
    /// which is how Metal answers both for any library (measured with an executable library's
    /// kernel and a Core Image kernel library's extern functions, an M3 Max under macOS 27.0.1).
    /// A Core Image kernel library's function is an extern function
    /// ([`Function::function_type`] answers [`FunctionType::Extern`]): a pipeline state made from
    /// it is refused (`Compile(InvalidDescriptor)`), as Metal fails it ("Function ... was not
    /// found in the module").
    pub fn new_extern_function_with_name(&self, name: &str) -> Result<Function, Error> {
        self.function(
            "new_extern_function_with_name",
            name,
            &FunctionConstantValues::defaults(),
            false,
        )
    }

    /// `newFunctionWithName:constantValues:error:`: the function named `name`, specialized with
    /// `constants` (by index or by name, `MTLFunctionConstantValues`); a constant without a value
    /// is undefined. A name the library lacks, or names more than once, is refused
    /// (`Compile(InvalidDescriptor)`). A value for a constant the function does not declare is
    /// ignored, as Metal ignores it. The others are checked against the constants the function
    /// declares when a pipeline state compiles it, where Metal checks them here: a value of
    /// another type, or a used constant without a value, refuses that pipeline state
    /// (`Compile`). A dynamic library's function is refused as
    /// [`new_function_with_name`](Self::new_function_with_name) refuses it.
    pub fn new_function_with_name_constant_values(
        &self,
        name: &str,
        constants: &FunctionConstantValues,
    ) -> Result<Function, Error> {
        self.function(
            "new_function_with_name_constant_values",
            name,
            constants,
            true,
        )
    }

    /// `newFunctionWithDescriptor:error:`: the function `descriptor` names, made as
    /// [`new_function_with_name_constant_values`](Self::new_function_with_name_constant_values)
    /// makes it where the descriptor has constant values and as
    /// [`new_function_with_name`](Self::new_function_with_name) where it has none, refused as
    /// they refuse; with the descriptor's options ([`Function::options`]). Made with
    /// `COMPILE_TO_BINARY`, its [`linked_function`](Function::linked_function) is precompiled, as
    /// a stage's `binary_functions` and a made state's added binary functions must be (Metal's
    /// API validation stops another, and its creation fails without validation; measured on an
    /// M3 Max and an M4). Every other bit is taken and kept for [`Function::options`] to answer,
    /// as plain Metal takes it: `newFunctionWithDescriptor:error:` over a descriptor that lists
    /// no binary archive answers a function for each of the sixteen combinations of the four
    /// `MTLFunctionOptions` bits and answers `options` back verbatim, 0x4 and 0x8 among them
    /// (measured on an M3 Max under macOS 27.0.1). `FAIL_ON_BINARY_ARCHIVE_MISS` has no archive
    /// to miss: a [`FunctionDescriptor`] lists none, there being no `binaryArchives` on it.
    /// `PIPELINE_INDEPENDENT` changes nothing the runtime makes of the function: what it asks of
    /// Metal is that a compiled binary not be specialized to one pipeline, and the runtime
    /// compiles no Metal binary. Metal's own API validation stops `PIPELINE_INDEPENDENT` without
    /// `COMPILE_TO_BINARY` ("MTLFunctionOptionPipelineIndependent requires
    /// MTLFunctionOptionCompileToBinary"), where plain Metal, the oracle, answers the call.
    pub fn new_function_with_descriptor(
        &self,
        descriptor: &FunctionDescriptor,
    ) -> Result<Function, Error> {
        let call = "new_function_with_descriptor";
        let mut function = match &descriptor.constant_values {
            Some(constants) => self.function(call, &descriptor.name, constants, true)?,
            None => self.function(
                call,
                &descriptor.name,
                &FunctionConstantValues::defaults(),
                false,
            )?,
        };
        function.options = descriptor.options;
        Ok(function)
    }

    /// The function named `name` with `constants`, made with constant values when `specialized`,
    /// in `call`; refused for a dynamic library.
    fn function(
        &self,
        call: &'static str,
        name: &str,
        constants: &FunctionConstantValues,
        specialized: bool,
    ) -> Result<Function, Error> {
        if self.library.library_type() == LibraryType::Dynamic {
            return Err(invalid(call, Problem::DynamicLibraryFunction));
        }
        let function = self.library.function(name).map_err(device_error)?;
        let holds = self.holds.function(function_name(&function));
        Ok(Function {
            device: self.device.clone(),
            function,
            constants: constants.clone(),
            specialized,
            argument_layouts: Arc::default(),
            properties: Arc::default(),
            constants_dictionary: Arc::default(),
            options: FunctionOptions::NONE,
            _holds: holds,
        })
    }

    /// `reflectionForFunctionWithName:`: the reflection of the function named `name`, as the
    /// compile layer's [`Function::reflection`](metal2vulkan::library::Function::reflection) answers it
    /// under the standard budget. `None` for a metallib function whose library carries no
    /// reflection, as Metal answers nil; a bitcode wrapper's or raw module's function answers from
    /// its AIR, which Metal cannot load. A name the library lacks, or names more than once, is
    /// refused (`Compile(InvalidDescriptor)`), as `new_function_with_name` refuses it; an argument
    /// whose answer Metal's has not been measured for is `Unsupported`.
    pub fn reflection_for_function_with_name(
        &self,
        name: &str,
    ) -> Result<Option<FunctionReflection>, Error> {
        let function = self.library.function(name).map_err(device_error)?;
        DerivedMap::or_derive(&self.reflections, name.to_string(), || {
            function
                .reflection(&Budget::standard())
                .map_err(device_error)
        })
    }

    /// How many of its functions' reflections the library has derived (`tests::derived_once`).
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) fn reflections_derived(&self) -> u32 {
        DerivedMap::derived(&self.reflections)
    }
}

impl Function {
    /// `device`.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// `name`.
    pub fn name(&self) -> &str {
        self.function.info().name
    }

    /// `options`: the options a [`FunctionDescriptor`] made the function with; none for a
    /// function made by name.
    pub fn options(&self) -> FunctionOptions {
        self.options
    }

    /// Whether the function was made compiled to a binary (`COMPILE_TO_BINARY`).
    pub(crate) fn precompiled(&self) -> bool {
        self.options.contains(FunctionOptions::COMPILE_TO_BINARY)
    }

    /// The function's `MTLFunction` properties under the standard budget: the compile layer's
    /// [`Function::properties`](metal2vulkan::library::Function::properties), or for a function made with
    /// constant values its
    /// [`specialized_properties`](metal2vulkan::library::Function::specialized_properties).
    fn properties(&self) -> Result<FunctionProperties, Error> {
        Derived::or_derive(&self.properties, || {
            let budget = Budget::standard();
            if self.specialized {
                self.function
                    .specialized_properties(&self.constants, &budget)
                    .map_err(device_error)
            } else {
                self.function.properties(&budget).map_err(device_error)
            }
        })
    }

    /// How many times the function has derived its properties, for the check that reading all
    /// five getters derives them once (`tests::derived_once`).
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) fn properties_derived(&self) -> u32 {
        Derived::derived(&self.properties)
    }

    /// How many times the function has derived its `functionConstantsDictionary`
    /// (`tests::derived_once`).
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) fn constants_dictionary_derived(&self) -> u32 {
        Derived::derived(&self.constants_dictionary)
    }

    /// How many argument-buffer layouts the function has derived (`tests::derived_once`).
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) fn argument_layouts_derived(&self) -> u32 {
        DerivedMap::derived(&self.argument_layouts)
    }

    /// `functionType`. A metallib function answers from its container tags; a bitcode wrapper's
    /// or raw module's from its AIR, which it decodes. Refused where Metal's answer is unmeasured
    /// (`Unsupported`; [`FunctionProperties`] lists the rules).
    pub fn function_type(&self) -> Result<FunctionType, Error> {
        self.properties().map(|p| p.function_type())
    }

    /// `patchType`, answered and refused as [`Function::function_type`].
    pub fn patch_type(&self) -> Result<PatchType, Error> {
        self.properties().map(|p| p.patch_type())
    }

    /// `patchControlPointCount`: `None` where Metal answers -1 (no patch, or a patch that states
    /// no count). Answered and refused as [`Function::function_type`].
    pub fn patch_control_point_count(&self) -> Result<Option<u32>, Error> {
        self.properties().map(|p| p.patch_control_point_count())
    }

    /// `vertexAttributes`: `None` (nil) for a function other than a vertex function, or one
    /// without attributes. A function made with constant values leaves out an attribute they
    /// gate off, as Metal does. Answered and refused as [`Function::function_type`].
    pub fn vertex_attributes(&self) -> Result<Option<Vec<Attribute>>, Error> {
        self.properties()
            .map(|p| p.vertex_attributes().map(<[Attribute]>::to_vec))
    }

    /// `stageInputAttributes`: a vertex function's or kernel's, `None` (nil) for another
    /// function or one without attributes; as [`Function::vertex_attributes`] otherwise.
    pub fn stage_input_attributes(&self) -> Result<Option<Vec<Attribute>>, Error> {
        self.properties()
            .map(|p| p.stage_input_attributes().map(<[Attribute]>::to_vec))
    }

    /// `functionConstantsDictionary`, by name: the library's function's
    /// ([`Function::function_constants_dictionary`](metal2vulkan::library::Function::function_constants_dictionary),
    /// refused as it refuses), and none for a function made with constant values, as Metal
    /// answers.
    pub fn function_constants_dictionary(
        &self,
    ) -> Result<BTreeMap<String, FunctionConstant>, Error> {
        if self.specialized {
            return Ok(BTreeMap::new());
        }
        Derived::or_derive(&self.constants_dictionary, || {
            self.function
                .function_constants_dictionary()
                .map_err(device_error)
        })
    }

    /// The compile layer's function.
    pub(super) fn compile_function(&self) -> &metal2vulkan::library::Function {
        &self.function
    }

    /// The constant values the function was made with.
    pub(super) fn constants(&self) -> &FunctionConstantValues {
        &self.constants
    }

    /// The argument-buffer layouts the function holds, by buffer index: derived when an argument
    /// encoder first asks for one (`argument_encoder.rs`).
    pub(super) fn argument_layouts(&self) -> &Mutex<DerivedMap<u32, Arc<ArgumentBufferLayout>>> {
        &self.argument_layouts
    }

    /// The compile layer's descriptor of the function, with its constant values and without
    /// linked functions (`FunctionDescriptor::with_linked` adds them): a stage of a
    /// [`RenderPipelineDescriptor`] or the kernel of a [`ComputePipelineDescriptor`] as
    /// `StageSource::Function` (`MTLRenderPipelineDescriptor.vertexFunction` and
    /// `.fragmentFunction`, `MTLComputePipelineDescriptor.computeFunction`).
    ///
    /// [`RenderPipelineDescriptor`]: metal2vulkan::pipeline::RenderPipelineDescriptor
    /// [`ComputePipelineDescriptor`]: metal2vulkan::pipeline::ComputePipelineDescriptor
    pub fn function_descriptor(&self) -> StageFunction {
        let mut descriptor = StageFunction::new(self.function.clone());
        descriptor.constants = self.constants.clone();
        descriptor
    }

    /// The function as a stage links it, with its constant values, and precompiled where it was
    /// made compiled to a binary: an entry of [`LinkedFunctions`] (`MTLLinkedFunctions.functions`,
    /// `.privateFunctions`, and the rest), which a stage's
    /// [`FunctionDescriptor`](metal2vulkan::pipeline::FunctionDescriptor) carries
    /// (`with_linked`).
    ///
    /// [`LinkedFunctions`]: metal2vulkan::pipeline::LinkedFunctions
    pub fn linked_function(&self) -> LinkedFunction {
        LinkedFunction::new(self.function.clone())
            .with_constants(self.constants.clone())
            .with_precompiled(self.precompiled())
    }
}
