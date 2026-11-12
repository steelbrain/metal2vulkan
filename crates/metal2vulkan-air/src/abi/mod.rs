//! AIR metadata ABI model and decoders.
//!
//! [`decode`] reads the module's named metadata once into the typed [`AirAbi`]; afterwards no
//! code reads the raw metadata graph. The grammar is strict and closed:
//!
//! - The named metadata set is closed. Any other `air.*`/`llvm.*` list, an unknown `air.compile.*`
//!   option, or an unknown module flag is [`AbiError::UnknownKey`].
//! - Every key of an argument, output, struct member, or entry attribute has a fixed arity and
//!   value type, and is accepted only by the roles that define it. An unknown key, a key the role
//!   does not accept, a repeated key, a missing required key, or a wrong value type is an
//!   [`AbiError`] (`UnknownKey`, `BadValue`, `MissingKey`). An argument-buffer member's
//!   `air.indirect_argument` belongs to an `air.indirect_buffer`'s layout, so an `air.buffer`
//!   (a parameter or a member) whose layout holds one at any depth is `UnknownKey`.
//! - Names (`air.arg_name`, `air.arg_type_name`, struct member and type names, constant names)
//!   are kept as [`DiagName`]s for diagnostics and never decide anything. The varying user key
//!   (`generated(..)`, `user(..)`) is the Metal interface-matching ABI and is compared for
//!   equality only ([`UserKey`]).
//!
//! Decoding also cross-checks the metadata against the IR: every entry parameter has exactly one
//! binding, each role agrees with its parameter's type and address space (a buffer, texture, or
//! sampler table passed as an `array_ref` is a thread-space pointer, and a read-only `device`
//! buffer may be a constant-space pointer), and the outputs agree with
//! the return type; likewise each `!air.visible` function binds every parameter once and has an
//! output exactly when it returns a value, and each `!air.ci` Core Image kernel also binds every
//! parameter once, with a role that agrees with the parameter's type, and has a scalar or vector
//! output exactly when it returns a value. For typed-pointer bitcode, the `air.arg_type_size` of an
//! argument buffer, a threadgroup argument, a buffer with a struct layout or a Core Image pointer
//! equals the allocation size of its pointee under the DataLayout; a plain buffer's states its
//! declared type, which may differ from the pointee (Apple's OpenGL front end declares byte buffers
//! behind typed pointers). A buffer's `air.buffer_size` is a multiple of its declared type's size
//! (a reference to an array), and every `air.struct_type_info` member starts inside its container,
//! with arrays and nested structs wholly inside. `air.arg_type_align_size` is Metal's alignment, which the LLVM type does not carry
//! (clang lowers many structs packed), so it is only required to be a power of two. A metallib
//! function's container metadata is cross-checked by [`AirAbi::check_container`].
//!
//! Resource-handle seeds for memory's provenance lattice are exposed by [`ParamRole::handle`] and
//! [`Member::handle`]. Texture shapes are not part of the ABI model: the only metadata that spells
//! them is the MSL type name (a [`DiagName`]) and, in typed-pointer bitcode only, the pointee
//! struct (erased), so a texture's [`ImageShape`] comes from the typed operand positions of the
//! intrinsics that use it.
//!
//! [`ImageShape`]: crate::intrinsics::ImageShape
//! [`AbiError`]: crate::error::AbiError
//! [`AbiError::UnknownKey`]: crate::error::AbiError::UnknownKey

mod args;
mod check;
mod core_image;
mod keys;
mod md;
mod module;
mod sampler;
mod structs;
#[cfg(test)]
mod tests;
mod visible;

pub use check::data_type_is;
pub use sampler::Half;
pub use sampler::{
    AddressMode, BorderColor, CompareFunction, Coordinates, Filter, MipFilter, Reduction,
    SamplerState,
};

use crate::budget::Meter;
use crate::container::LibraryFunction;
use crate::error::DiagName;
use crate::error::Error;
use crate::ir::{AddrSpace, FuncId, GlobalId, Module, TypeId, TypeLayouts};

use super::bitcode::PointeeTable;

/// Decode the AIR metadata ABI of `module` and cross-check it against the IR. `pointees` are
/// the decode-only typed-pointer pointees; for opaque-pointer bitcode it is empty and the pointee
/// checks are skipped.
pub fn decode(
    module: &Module,
    pointees: &PointeeTable,
    meter: &mut Meter,
) -> Result<AirAbi, Error> {
    // Struct field offsets, which the decode needs to select an argument-buffer member's field
    // and the checks need for the pointee comparisons. Both reach a holding type only through a
    // recorded pointee, so with none there is nothing to lay out.
    let layouts = if pointees.is_empty() {
        None
    } else {
        Some(TypeLayouts::new(
            module.data_layout(),
            module.types(),
            meter,
        )?)
    };
    let abi = module::decode(module, pointees, layouts.as_ref(), meter)?;
    check::module(module, pointees, layouts.as_ref(), &abi, meter)?;
    Ok(abi)
}

/// The decoded AIR metadata ABI of one module.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AirAbi {
    /// `!air.version` (`None`: the module does not state it).
    pub air_version: Option<Version>,
    /// `!air.language_version` (`None`: the module does not state it).
    pub language: Option<LanguageVersion>,
    /// `!air.compile_options`.
    pub compile: CompileOptions,
    /// The `air.max_*` entries of `!llvm.module.flags`.
    pub limits: ModuleLimits,
    /// `!air.kernel`, `!air.vertex`, `!air.fragment`, `!air.object`, `!air.mesh`, in that
    /// order, each in list order.
    pub entries: Vec<EntryPoint>,
    /// `!air.function_constants`.
    pub function_constants: Vec<FunctionConstantDecl>,
    /// `!air.sampler_states`: constexpr samplers.
    pub static_samplers: Vec<StaticSampler>,
    /// `!air.visible_function_references`.
    pub visible_function_refs: Vec<VisibleFunctionRef>,
    /// `!air.visible`: the functions this module exports to the Metal linker, in list order.
    pub visible_functions: Vec<VisibleFunction>,
    /// `!air.ci`: the Core Image kernels this module exports, in list order.
    pub core_image_kernels: Vec<CoreImageKernel>,
    /// `!air.signature`: stitching or linked-constant signatures (how the module was produced;
    /// see [`LinkageSignature`]).
    pub linkage_signature: Option<LinkageSignature>,
    /// Every `air.struct_type_info` layout, shared by the [`StructRef`]s that name it.
    pub structs: Vec<StructInfo>,
    /// `!llvm.ident`, for diagnostics.
    pub producer: Option<DiagName>,
}

impl AirAbi {
    /// The layout `r` names.
    pub fn struct_info(&self, r: StructRef) -> Option<&StructInfo> {
        self.structs.get(r.0 as usize)
    }

    /// Whether entry `ep` can call through a visible-function table: one it takes, or one a buffer
    /// it takes holds, at any depth of the buffer's `air.struct_type_info` layouts (a member whose
    /// role is `air.visible_function_table`). Linear in the layouts.
    pub fn reaches_function_tables(&self, ep: &EntryPoint) -> bool {
        ep.reaches_role(&self.structs, |r| matches!(r, ParamRole::FunctionTable(_)))
    }

    /// The entry point for function `func`.
    #[cfg(test)]
    pub fn entry(&self, func: FuncId) -> Option<&EntryPoint> {
        self.entries.iter().find(|e| e.func == func)
    }

    /// The `!air.visible` declaration of function `func`.
    #[cfg(test)]
    pub fn visible_function(&self, func: FuncId) -> Option<&VisibleFunction> {
        self.visible_functions.iter().find(|v| v.func == func)
    }

    /// Every visible-function-reference stub: a function declaration of this module that stands
    /// for the linked visible function whose symbol it names. A reference through a global
    /// ([`VisibleTarget::Global`]) is not a callable stub and is skipped.
    pub fn visible_function_stubs(&self) -> impl Iterator<Item = (FuncId, &VisibleSymbol)> + '_ {
        self.visible_function_refs
            .iter()
            .filter_map(|r| match r.target {
                VisibleTarget::Function(f) => Some((f, &r.symbol)),
                VisibleTarget::Global(_) => None,
            })
    }

    /// Every post-tessellation control-point fetch declaration an entry names
    /// (`air.patch_control_point_function`): Metal synthesizes its body from the pipeline's
    /// vertex descriptor.
    pub fn control_point_fetches(&self) -> impl Iterator<Item = FuncId> + '_ {
        self.entries
            .iter()
            .flat_map(|e| e.params.iter())
            .filter_map(|p| match &p.role {
                ParamRole::PatchInput(PatchInput::ControlPoints { function, .. }) => {
                    Some(*function)
                }
                _ => None,
            })
    }

    /// Cross-check this ABI against the container metadata of the metallib function `f` whose
    /// bitcode it was decoded from: the bitcode must declare exactly one entry point (a
    /// metallib function is its own module holding only that entry), the function's `TYPE` stage
    /// must be that entry's stage, and every `CNST` entry must name a declared function constant by
    /// index with an `MTLDataType` whose stored form is the value type of the constant's
    /// initializer global in `m` (the module this ABI was decoded from). Records each constant's
    /// `MTLDataType` from the container ([`FunctionConstantDecl::data_type`]). A disagreement is
    /// [`AbiError::ContainerMismatch`], and leaves `self` unchanged; a stage the model does not
    /// cover (a `TYPE` other than vertex, fragment, kernel, mesh and object) is
    /// `Unsupported(Feature::Stage)`.
    ///
    /// [`AbiError::ContainerMismatch`]: crate::error::AbiError::ContainerMismatch
    pub fn check_container(
        &mut self,
        m: &Module,
        f: &LibraryFunction<'_>,
        meter: &mut Meter,
    ) -> Result<(), Error> {
        check::container(self, m, f, meter)
    }
}

/// A three-part version.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Version {
    /// The first part (the `2` of 2.6.0).
    pub major: u32,
    /// The second part (the `6` of 2.6.0).
    pub minor: u32,
    /// The third part (the `0` of 2.6.0).
    pub patch: u32,
}

/// The Metal shading language version the module was compiled for.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LanguageVersion(pub Version);

/// `!air.compile_options`. Each option is `None` when the module does not state it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct CompileOptions {
    /// `air.compile.fast_math_enable` / `_disable`.
    pub fast_math: Option<bool>,
    /// `air.compile.denorms_disable`.
    pub denorms_disable: bool,
    /// `air.compile.framebuffer_fetch_enable` / `_disable`.
    pub framebuffer_fetch: Option<bool>,
}

/// The `air.max_*` module flags: the argument-table sizes the module was compiled against.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct ModuleLimits {
    /// `air.max_device_buffers`: device-space buffer entries (`None`: not stated).
    pub max_device_buffers: Option<u32>,
    /// `air.max_constant_buffers`: constant-space buffer entries (`None`: not stated).
    pub max_constant_buffers: Option<u32>,
    /// `air.max_threadgroup_buffers`: threadgroup memory entries (`None`: not stated).
    pub max_threadgroup_buffers: Option<u32>,
    /// `air.max_textures`: texture entries (`None`: not stated).
    pub max_textures: Option<u32>,
    /// `air.max_read_write_textures`: read-write texture entries (`None`: not stated).
    pub max_read_write_textures: Option<u32>,
    /// `air.max_samplers`: sampler entries (`None`: not stated).
    pub max_samplers: Option<u32>,
}

/// An integer the metadata states either directly or as the value of a global that the static
/// initializer computes (for example an index that depends on a function constant). The global
/// form is resolved by canonicalization, after static-initializer evaluation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum AbiInt {
    /// The integer, stated directly.
    Const(u32),
    /// The integer global whose static-initializer value is the integer; canonicalization
    /// replaces it with `Const`.
    Global(GlobalId),
}

/// A Metal argument-table slot: `air.location_index, <index>, <count>`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Slot {
    /// The first slot's index (the `n` of `[[buffer(n)]]`, `[[texture(n)]]`,
    /// `[[attribute(n)]]`, ...).
    pub index: AbiInt,
    /// Array length (1 for a single argument).
    pub count: AbiInt,
}

/// An entry point.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EntryPoint {
    /// The entry's function.
    pub func: FuncId,
    /// The stage of the list that names it, with its stage attributes.
    pub stage: Stage,
    /// One binding per LLVM parameter, in parameter order.
    pub params: Vec<ParamBinding>,
    /// Stage outputs, in the order of the return value's fields, each naming its field
    /// ([`OutputBinding::field`]). Canonicalization removes the outputs a false predicate gates
    /// off, so a specialized entry's fields need not be dense.
    pub outputs: Vec<OutputBinding>,
    /// The function's name, for diagnostics.
    pub name: DiagName,
    /// `air.user_annotation`: the string `[[user_annotation("...")]]` passes, which Metal answers
    /// as the function's `MTLFunctionReflection.userAnnotation`; it changes nothing the function
    /// computes.
    pub user_annotation: Option<DiagName>,
}

impl EntryPoint {
    /// Whether the entry takes an indirect command buffer (`air.command_buffer`): as a parameter,
    /// or as a member of a buffer it takes, at any depth of the buffer's `air.struct_type_info`
    /// layouts `structs` (an argument buffer's `command_buffer` member). It is what the metadata
    /// states, whatever the body does with it; a body that reaches a command buffer through a
    /// pointer it casts, which no layout of a buffer it takes describes, is not counted here:
    /// [`crate::canon::Entry::may_encode_indirect_commands`] counts it too. Linear in the layouts.
    pub fn reaches_command_buffers(&self, structs: &[StructInfo]) -> bool {
        self.reaches_role(structs, |r| matches!(r, ParamRole::CommandBuffer(_)))
    }

    /// Whether the entry takes a texture it may write: one declared `access::write` or
    /// `access::read_write` (`air.write`, `air.read_write`), as a parameter or as a member of a
    /// buffer it takes, at any depth of the buffer's `air.struct_type_info` layouts `structs` (an
    /// argument buffer's texture member, an array of them included). It is what the metadata
    /// states, whatever the body does with the texture; a body that writes a texture whose handle
    /// no metadata names (one it loads from bytes no layout describes, a resource ID) is not
    /// counted here: [`crate::canon::Entry::may_write_textures`] counts it too. Linear in the
    /// layouts.
    pub fn takes_writable_textures(&self, structs: &[StructInfo]) -> bool {
        self.reaches_role(structs, |r| {
            matches!(
                r,
                ParamRole::Texture(TextureArg {
                    access: Access::Write | Access::ReadWrite,
                    ..
                })
            )
        })
    }

    /// Whether a parameter of the entry, or a member of a buffer it takes at any depth of the
    /// buffer's layouts `structs` (an `air.indirect_argument` member's role), has a role `wanted`
    /// accepts. Each layout is visited once.
    fn reaches_role(&self, structs: &[StructInfo], wanted: impl Fn(&ParamRole) -> bool) -> bool {
        let layout = |role: &ParamRole| match role {
            ParamRole::Buffer(b) | ParamRole::ArgumentBuffer(b) | ParamRole::Threadgroup(b) => {
                b.layout
            }
            _ => None,
        };
        if self.params.iter().any(|p| wanted(&p.role)) {
            return true;
        }
        let mut seen = vec![false; structs.len()];
        let mut stack: Vec<StructRef> =
            self.params.iter().filter_map(|p| layout(&p.role)).collect();
        while let Some(r) = stack.pop() {
            match seen.get_mut(r.0 as usize) {
                Some(s) if !*s => *s = true,
                _ => continue,
            }
            for member in structs.get(r.0 as usize).map_or(&[][..], |i| &i.members) {
                if let MemberKind::Indirect(arg) = &member.kind {
                    if wanted(&arg.role) {
                        return true;
                    }
                    stack.extend(layout(&arg.role));
                }
                stack.extend(member.layout);
            }
        }
        false
    }

    /// Whether `f` is the control-point function of one of this post-tessellation vertex
    /// function's control-point inputs (`air.patch_control_point_function`).
    pub fn is_control_point_function(&self, f: FuncId) -> bool {
        matches!(self.stage, Stage::Vertex { patch: Some(_) })
            && self.params.iter().any(|b| {
                matches!(&b.role, ParamRole::PatchInput(PatchInput::ControlPoints { function, .. })
                    if *function == f)
            })
    }
}

/// An entry point's stage and its stage attributes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Stage {
    /// A kernel (`!air.kernel`), tile functions included.
    Kernel {
        /// `air.max_work_group_size` (`[[max_total_threads_per_threadgroup]]`).
        max_total_threads: Option<u32>,
        /// `air.work_group_size`: the threadgroup size an OpenCL C kernel requires
        /// (`reqd_work_group_size(x, y, z)`). Metal caps the pipeline's threadgroup at its
        /// `x * y * z` threads.
        work_group_size: Option<[u32; 3]>,
    },
    /// A vertex function (`!air.vertex`).
    Vertex {
        /// `Some`: a post-tessellation vertex function.
        patch: Option<PatchInfo>,
    },
    /// A fragment function (`!air.fragment`).
    Fragment {
        /// `early_fragment_tests`.
        early_fragment_tests: bool,
    },
    /// An object function (`!air.object`): it launches a grid of mesh threadgroups per threadgroup
    /// and fills their payload.
    Object {
        /// `air.max_work_group_size` (`[[max_total_threads_per_threadgroup]]`).
        max_total_threads: Option<u32>,
        /// `air.max_mesh_work_groups` (`[[max_total_threadgroups_per_mesh_grid]]`).
        max_mesh_threadgroups: Option<u32>,
    },
    /// A mesh function (`!air.mesh`): each threadgroup writes one chunk of geometry.
    Mesh {
        /// `air.max_work_group_size` (`[[max_total_threads_per_threadgroup]]`).
        max_total_threads: Option<u32>,
    },
}

impl Stage {
    /// `air.work_group_size`, a kernel's.
    pub fn work_group_size(&self) -> Option<[u32; 3]> {
        match *self {
            Stage::Kernel {
                work_group_size, ..
            } => work_group_size,
            _ => None,
        }
    }

    /// `air.max_work_group_size`, the stages that state it: kernels, object and mesh functions.
    pub fn max_total_threads(&self) -> Option<u32> {
        match *self {
            Stage::Kernel {
                max_total_threads, ..
            }
            | Stage::Object {
                max_total_threads, ..
            }
            | Stage::Mesh { max_total_threads } => max_total_threads,
            Stage::Vertex { .. } | Stage::Fragment { .. } => None,
        }
    }
}

/// `air.patch, <kind>[, air.patch_control_point, <n>]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PatchInfo {
    /// The patch type (`[[patch(kind, ..)]]`).
    pub kind: PatchKind,
    /// The control points per patch (`[[patch(.., n)]]`); `None` for `[[patch(kind)]]`, whose
    /// node states no count (a patch of any count, the draw's).
    pub control_points: Option<u32>,
}

/// The patch type of a post-tessellation vertex function.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PatchKind {
    /// `triangle` (`[[patch(triangle, n)]]`).
    Triangle,
    /// `quad` (`[[patch(quad, n)]]`).
    Quad,
}

/// A function-constant predicate (`air.function_constant, !{<global>, <type>, <name>}`): the
/// argument or output is gated by the value the static initializer stores in that integer global.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Predicate {
    /// The integer global: the argument or output exists where its value is nonzero.
    pub global: GlobalId,
    /// The function constant's name, for diagnostics.
    pub name: DiagName,
}

/// Size, alignment, and type name of an argument, as the metadata states them.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct ArgTypeInfo {
    /// `air.arg_type_size`: the size of the pointee (pointer arguments) in bytes.
    pub size: Option<u32>,
    /// `air.arg_type_align_size`.
    pub align: Option<u32>,
    /// `air.arg_type_name`, for diagnostics.
    pub name: Option<DiagName>,
}

/// One entry parameter's binding.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParamBinding {
    /// The LLVM parameter index.
    pub param: u32,
    /// What the parameter is: its role key and role attributes.
    pub role: ParamRole,
    /// `air.arg_unused`.
    pub unused: bool,
    /// `air.function_constant` (`[[function_constant(..)]]`): the argument exists only where
    /// its predicate holds (`None`: always).
    pub predicate: Option<Predicate>,
    /// `air.arg_type_size`, `air.arg_type_align_size`, and `air.arg_type_name`, as stated.
    pub ty: ArgTypeInfo,
    /// `air.arg_name`, for diagnostics.
    pub name: Option<DiagName>,
}

/// Declared access of a buffer, texture, or table argument.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Access {
    /// `air.read`.
    Read,
    /// `air.write`.
    Write,
    /// `air.read_write`.
    ReadWrite,
    /// `air.sample` (textures only).
    Sample,
}

/// A buffer argument (`air.buffer`, `air.indirect_buffer`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BufferArg {
    /// `air.location_index`: the `[[buffer(n)]]` (threadgroup memory: `[[threadgroup(n)]]`)
    /// slot; in an argument buffer, the member's own part of its `[[id(n)]]`.
    pub location: Slot,
    /// `air.read`, `air.write`, or `air.read_write`.
    pub access: Access,
    /// `air.address_space`.
    pub space: AddrSpace,
    /// `air.buffer_size`: the size of a sized pointee.
    pub buffer_size: Option<u32>,
    /// `air.struct_type_info`.
    pub layout: Option<StructRef>,
    /// `air.raster_order_group`.
    pub raster_order_group: Option<u32>,
}

/// A texture argument (`air.texture`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TextureArg {
    /// `air.location_index`: the `[[texture(n)]]` slot (in an argument buffer, the member's own
    /// part of its `[[id(n)]]`).
    pub location: Slot,
    /// `air.sample`, `air.read`, `air.write`, or `air.read_write`.
    pub access: Access,
    /// `air.raster_order_group`.
    pub raster_order_group: Option<u32>,
}

/// An argument-table object other than a buffer, texture, or sampler.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ObjectArg {
    /// `air.location_index`: the `[[buffer(n)]]` slot (in an argument buffer, the member's own
    /// part of its `[[id(n)]]`).
    pub location: Slot,
    /// `air.read`, `air.write`, or `air.read_write` (`None`: not stated).
    pub access: Option<Access>,
}

/// The kind of an acceleration structure argument.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AccelerationStructureKind {
    /// `air.instance_acceleration_structure`.
    Instance,
    /// `air.primitive_acceleration_structure`.
    Primitive,
}

/// Interpolation of a fragment input (`air.center|centroid|sample` with
/// `air.perspective|no_perspective`, `air.interpolation_function` with
/// `air.perspective|no_perspective`, or `air.flat`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Interpolation {
    /// `air.flat`: not interpolated; the provoking vertex's value.
    Flat,
    /// Interpolated across the primitive.
    Smooth {
        /// Where the input is sampled.
        sampling: Sampling,
        /// `air.perspective` (`true`) or `air.no_perspective` (`false`).
        perspective: bool,
    },
}

/// Where a smooth fragment input is sampled.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Sampling {
    /// `air.center`: at the pixel's center.
    Center,
    /// `air.centroid`: at a covered location within the pixel.
    Centroid,
    /// `air.sample`: at each sample's location (per-sample shading).
    Sample,
    /// `air.interpolation_function`: an `interpolant` (Metal's pull-model interpolation), a
    /// handle (`ptr addrspace(1)`) read only through the `air.interpolate_*` calls, each at the
    /// location it names.
    Interpolant,
}

impl Interpolation {
    /// Whether the input is an `interpolant` ([`Sampling::Interpolant`]).
    pub fn is_interpolant(self) -> bool {
        matches!(
            self,
            Interpolation::Smooth {
                sampling: Sampling::Interpolant,
                ..
            }
        )
    }
}

/// A varying's AIR user key (`generated(..)` / `user(..)`): the Metal interface-matching ABI,
/// compared for equality only.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct UserKey(pub Box<[u8]>);

/// A fragment input varying.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Varying {
    /// The user key of the vertex (or mesh) output it receives.
    pub key: UserKey,
    /// How the input is interpolated.
    pub interpolation: Interpolation,
}

/// The kind of an imageblock argument.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ImageblockKind {
    /// `air.imageblock, "explicit"` (kernel).
    Explicit,
    /// `air.imageblock, "implicit"` (kernel).
    Implicit,
    /// `air.imageblock_data` (fragment input or output).
    Data,
}

/// An imageblock argument or output.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ImageblockArg {
    /// Explicit, implicit, or imageblock data.
    pub kind: ImageblockKind,
    /// `air.imageblock_data_size`.
    pub data_size: Option<u32>,
    /// `air.struct_type_info`: the imageblock cell layout.
    pub layout: Option<StructRef>,
    /// `air.imageblock_master`: the layout of the whole imageblock this data views.
    pub master: Option<StructRef>,
    /// `air.alias_implicit_imageblock`.
    pub alias_implicit: bool,
}

/// One field of a post-tessellation control-point input: member `k` of the struct its
/// control-point function returns is field `k`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ControlPointField {
    /// `air.location_index`: the `[[attribute(n)]]` the field is fetched from.
    pub location: Slot,
    /// `air.function_constant`: the field exists only where its predicate holds (`None`:
    /// always). Canonicalization resolves it into `gated_off`.
    pub predicate: Option<Predicate>,
    /// `air.arg_type_name` (a control-point field states no size or alignment).
    pub ty: ArgTypeInfo,
    /// `air.arg_name`, for diagnostics.
    pub name: Option<DiagName>,
    /// `air.arg_unused`: the function reads no control point's value of this field.
    pub unused: bool,
    /// Its function-constant predicate is false for the pipeline (set when canonicalization
    /// resolves the entry's ABI): Metal fetches nothing for it.
    pub gated_off: bool,
}

/// A post-tessellation patch input.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum PatchInput {
    /// `air.patch_control_point_input`: the control-point function and the fields it reads.
    ControlPoints {
        /// `air.patch_control_point_function`: the declaration whose calls fetch a control
        /// point.
        function: FuncId,
        /// One per member of the struct that function returns, in member order.
        fields: Vec<ControlPointField>,
    },
    /// `air.patch_input` with its attribute location.
    Patch {
        /// `air.location_index`: the per-patch attribute's `[[attribute(n)]]`.
        location: Slot,
    },
}

/// The role of an entry parameter or argument-buffer member.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum ParamRole {
    /// `air.buffer` in the device or constant address space.
    Buffer(BufferArg),
    /// `air.indirect_buffer`: an argument buffer.
    ArgumentBuffer(BufferArg),
    /// `air.buffer` in the threadgroup address space.
    Threadgroup(BufferArg),
    /// `air.texture`.
    Texture(TextureArg),
    /// `air.sampler`.
    Sampler {
        /// `air.location_index`: the `[[sampler(n)]]` slot (in an argument buffer, the
        /// member's own part of its `[[id(n)]]`).
        location: Slot,
    },
    /// A stage-input or dispatch builtin.
    Builtin(Builtin),
    /// Kernel `[[stage_in]]` (`air.stage_in`).
    StageIn {
        /// `air.location_index`: the `[[attribute(n)]]` of the stage-in member this parameter
        /// holds.
        location: Slot,
    },
    /// `air.vertex_input`.
    VertexInput {
        /// `air.location_index`: the vertex attribute's `[[attribute(n)]]`.
        location: Slot,
    },
    /// `air.fragment_input`.
    FragmentInput(Varying),
    /// Framebuffer fetch: `air.render_target` as a fragment input.
    ColorInput {
        /// The color attachment read (`[[color(n)]]`).
        index: AbiInt,
        /// `air.raster_order_group` (`[[raster_order_group(n)]]`).
        raster_order_group: Option<u32>,
    },
    /// A post-tessellation patch input.
    PatchInput(PatchInput),
    /// `air.imageblock` / `air.imageblock_data`.
    Imageblock(ImageblockArg),
    /// `air.instance_acceleration_structure` / `air.primitive_acceleration_structure`.
    AccelerationStructure {
        /// Whether it is an instance or a primitive acceleration structure.
        kind: AccelerationStructureKind,
        /// Its slot and declared access.
        arg: ObjectArg,
    },
    /// `air.visible_function_table`.
    FunctionTable(ObjectArg),
    /// `air.intersection_function_table`.
    IntersectionTable(ObjectArg),
    /// `air.indirect_constant`: a plain value in an argument buffer.
    IndirectConstant {
        /// `air.location_index`: the member's own part of its `[[id(n)]]`.
        location: Slot,
    },
    /// `air.command_buffer`: an indirect command buffer in an argument buffer.
    CommandBuffer(ObjectArg),
    /// `air.compute_pipeline_state` in an argument buffer.
    ComputePipelineState(ObjectArg),
    /// `air.render_pipeline_state` in an argument buffer.
    RenderPipelineState(ObjectArg),
    /// `air.depth_stencil_state` in an argument buffer: the state a render command's
    /// `set_depth_stencil_state` sets (MSL 4.0; Metal's Apple10 family alone runs it).
    DepthStencilState(ObjectArg),
    /// `air.payload`: an object function's payload, written by the object function and read by
    /// the mesh functions it launches; a pointer in the `object_data` address space. Its size and
    /// alignment are the binding's `air.arg_type_size` and `air.arg_type_align_size`.
    Payload {
        /// `air.struct_type_info`.
        layout: Option<StructRef>,
    },
    /// `air.mesh_grid_properties`: an object function's handle to the mesh grid it launches.
    MeshGridProperties,
    /// `air.mesh`: a mesh function's output handle and its type, `mesh<V, P, NV, NP, t>`.
    Mesh(Box<MeshOutputs>),
}

/// A mesh function's output type (`air.mesh_type_info`): the vertex and primitive fields, the most
/// vertices and primitives one threadgroup writes, and the topology.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct MeshOutputs {
    /// The vertex struct's fields, in declaration order.
    pub vertex: Vec<MeshField>,
    /// The primitive struct's fields, in declaration order (none for a `void` primitive type).
    pub primitive: Vec<MeshField>,
    /// `NV`.
    pub max_vertices: u32,
    /// `NP`.
    pub max_primitives: u32,
    /// `t`: the primitive kind.
    pub topology: MeshTopology,
}

/// The primitive kind a mesh function writes (`air.point`, `air.line`, `air.triangle`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MeshTopology {
    /// `air.point` (`topology::point`).
    Point,
    /// `air.line` (`topology::line`).
    Line,
    /// `air.triangle` (`topology::triangle`).
    Triangle,
}

/// One field of a mesh function's vertex or primitive struct.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct MeshField {
    /// What the field is: a builtin or a user data field.
    pub role: MeshFieldRole,
    /// `air.function_constant`: the field exists only where its predicate holds
    /// (`[[function_constant(..)]]` on the member).
    pub predicate: Option<Predicate>,
    /// `air.arg_type_name` (a mesh field states no size or alignment).
    pub ty: ArgTypeInfo,
    /// `air.arg_name`, for diagnostics.
    pub name: Option<DiagName>,
    /// Its function-constant predicate is false for the pipeline (set when canonicalization
    /// resolves the entry's ABI): the mesh type has no such field, and a write to it stores
    /// nothing.
    pub gated_off: bool,
}

/// What a mesh output field is. The data fields carry the slot the `air.set_*_data_mesh`
/// operations name and the user key a fragment input matches.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum MeshFieldRole {
    // ---- Vertex fields ----
    /// `air.position`.
    Position {
        /// `air.invariant`.
        invariant: bool,
    },
    /// `air.point_size`.
    PointSize,
    /// `air.clip_distance`.
    ClipDistance {
        /// `air.clip_distance_array_size`.
        array_size: Option<u32>,
    },
    /// `air.mesh_vertex_data, <slot>, <user key>`.
    VertexData {
        /// The slot the `air.set_vertex_data_mesh` calls that write the field name.
        slot: u32,
        /// The user key of the fragment input the field feeds.
        key: UserKey,
    },
    // ---- Primitive fields ----
    /// `air.primitive_culled`.
    PrimitiveCulled,
    /// `air.primitive_id`.
    PrimitiveId,
    /// `air.render_target_array_index`.
    RenderTargetArrayIndex,
    /// `air.viewport_array_index`.
    ViewportArrayIndex,
    /// `air.mesh_primitive_data, <slot>, <user key>`.
    PrimitiveData {
        /// The slot the `air.set_primitive_data_mesh` calls that write the field name.
        slot: u32,
        /// The user key of the fragment input the field feeds.
        key: UserKey,
    },
}

/// The class of a resource handle an ABI role seeds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AbiHandle {
    /// A texture (`air.texture`), with its declared access.
    Texture(Access),
    /// A sampler (`air.sampler`).
    Sampler,
    /// An acceleration structure of the given kind.
    AccelerationStructure(AccelerationStructureKind),
    /// A visible-function table (`air.visible_function_table`).
    FunctionTable,
    /// An intersection-function table (`air.intersection_function_table`).
    IntersectionTable,
    /// An indirect command buffer (`air.command_buffer`).
    CommandBuffer,
    /// A compute pipeline state (`air.compute_pipeline_state`).
    ComputePipelineState,
    /// A render pipeline state (`air.render_pipeline_state`).
    RenderPipelineState,
    /// A depth/stencil state (`air.depth_stencil_state`).
    DepthStencilState,
    /// An object function's mesh grid (`air.mesh_grid_properties`).
    MeshGridProperties,
    /// A mesh function's output (`air.mesh`).
    Mesh,
    /// A fragment input that is an `interpolant` (`air.interpolation_function`).
    Interpolant,
}

impl ParamRole {
    /// The handle class this role seeds, if it is a resource handle rather than bytes.
    pub fn handle(&self) -> Option<AbiHandle> {
        match self {
            ParamRole::Texture(t) => Some(AbiHandle::Texture(t.access)),
            ParamRole::Sampler { .. } => Some(AbiHandle::Sampler),
            ParamRole::AccelerationStructure { kind, .. } => {
                Some(AbiHandle::AccelerationStructure(*kind))
            }
            ParamRole::FunctionTable(_) => Some(AbiHandle::FunctionTable),
            ParamRole::IntersectionTable(_) => Some(AbiHandle::IntersectionTable),
            ParamRole::CommandBuffer(_) => Some(AbiHandle::CommandBuffer),
            ParamRole::ComputePipelineState(_) => Some(AbiHandle::ComputePipelineState),
            ParamRole::RenderPipelineState(_) => Some(AbiHandle::RenderPipelineState),
            ParamRole::DepthStencilState(_) => Some(AbiHandle::DepthStencilState),
            ParamRole::MeshGridProperties => Some(AbiHandle::MeshGridProperties),
            ParamRole::Mesh(_) => Some(AbiHandle::Mesh),
            ParamRole::FragmentInput(v) if v.interpolation.is_interpolant() => {
                Some(AbiHandle::Interpolant)
            }
            ParamRole::Payload { .. }
            | ParamRole::Buffer(_)
            | ParamRole::ArgumentBuffer(_)
            | ParamRole::Threadgroup(_)
            | ParamRole::Builtin(_)
            | ParamRole::StageIn { .. }
            | ParamRole::VertexInput { .. }
            | ParamRole::FragmentInput(_)
            | ParamRole::ColorInput { .. }
            | ParamRole::PatchInput(_)
            | ParamRole::Imageblock(_)
            | ParamRole::IndirectConstant { .. } => None,
        }
    }
}

/// A builtin input (closed set).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Builtin {
    // ---- Dispatch (kernel; some also in other stages) ----
    /// `[[thread_position_in_grid]]`: the thread's position in the grid.
    ThreadPositionInGrid,
    /// `[[thread_position_in_threadgroup]]`: the thread's position in its threadgroup.
    ThreadPositionInThreadgroup,
    /// `[[threadgroup_position_in_grid]]`: the threadgroup's position in the grid.
    ThreadgroupPositionInGrid,
    /// `[[threads_per_threadgroup]]`: this threadgroup's size in threads (smaller at the edge
    /// of a non-uniform grid).
    ThreadsPerThreadgroup,
    /// `[[threads_per_grid]]`: the grid's size in threads.
    ThreadsPerGrid,
    /// `[[threadgroups_per_grid]]`: the grid's size in threadgroups.
    ThreadgroupsPerGrid,
    /// `[[thread_index_in_threadgroup]]`: the thread's linear index in its threadgroup.
    ThreadIndexInThreadgroup,
    /// `[[thread_index_in_simdgroup]]`: the thread's lane in its SIMD-group.
    ThreadIndexInSimdgroup,
    /// `[[thread_index_in_quadgroup]]`: the thread's lane in its quad-group.
    ThreadIndexInQuadgroup,
    /// `[[simdgroup_index_in_threadgroup]]`: the SIMD-group's index in its threadgroup.
    SimdgroupIndexInThreadgroup,
    /// `[[quadgroup_index_in_threadgroup]]`: the quad-group's index in its threadgroup.
    QuadgroupIndexInThreadgroup,
    /// `[[threads_per_simdgroup]]`: the SIMD-group width.
    ThreadsPerSimdgroup,
    /// `[[simdgroups_per_threadgroup]]`: the number of SIMD-groups in the threadgroup.
    SimdgroupsPerThreadgroup,
    /// `[[quadgroups_per_threadgroup]]`: the number of quad-groups in the threadgroup.
    QuadgroupsPerThreadgroup,
    /// `[[dispatch_threads_per_threadgroup]]`: the threadgroup size the dispatch states,
    /// whatever this threadgroup's own size.
    DispatchThreadsPerThreadgroup,
    /// `[[dispatch_simdgroups_per_threadgroup]]`: the number of SIMD-groups in a threadgroup of
    /// the size the dispatch states.
    DispatchSimdgroupsPerThreadgroup,
    /// `[[grid_size]]` (`air.stage_in_grid_size`): the stage-in region's size.
    StageInGridSize,
    /// `[[grid_origin]]` (`air.stage_in_grid_origin`): the stage-in region's origin.
    StageInGridOrigin,
    // ---- Vertex ----
    /// `[[vertex_id]]`: the vertex's index.
    VertexId,
    /// `[[instance_id]]`: the instance's index.
    InstanceId,
    /// `[[base_vertex]]`: the draw's base vertex, its `vertexStart` for a draw without an index
    /// buffer and its `baseVertex` for an indexed one (a vertex function only).
    BaseVertex,
    /// `[[base_instance]]`: the draw's `baseInstance` (a vertex or post-tessellation vertex
    /// function).
    BaseInstance,
    /// `[[amplification_id]]`: the vertex-amplification view being processed.
    AmplificationId,
    /// `[[amplification_count]]`: the number of vertex-amplification views.
    AmplificationCount,
    /// `[[position_in_patch]]`: a post-tessellation vertex's coordinates in its patch.
    PositionInPatch,
    /// `[[patch_id]]`: a post-tessellation vertex's patch index.
    PatchId,
    // ---- Fragment ----
    /// `[[position]]`: the fragment's window-space position, with its interpolation when the
    /// metadata states one.
    Position(Option<Interpolation>),
    /// `[[front_facing]]`: whether the fragment's primitive faces the viewer.
    FrontFacing,
    /// `[[point_coord]]`: the fragment's position within its point.
    PointCoord,
    /// `[[primitive_id]]`: the index of the fragment's primitive.
    PrimitiveId,
    /// `[[sample_id]]`: the sample being shaded.
    SampleId,
    /// `[[sample_mask]]` as an input (`air.sample_mask_in`): the fragment's coverage mask.
    SampleMaskIn,
    /// `[[sample_mask, post_depth_coverage]]` (`air.sample_mask_in` with
    /// `air.post_depth_coverage`): the fragment's coverage mask after the depth and stencil tests,
    /// in a function Metal's compiler requires to declare `[[early_fragment_tests]]`.
    PostDepthCoverage,
    /// `[[barycentric_coord]]`: the fragment's barycentric coordinates, with their
    /// interpolation.
    BarycentricCoord(Interpolation),
    /// `[[render_target_array_index]]`: the layer rendered to (fragment and tile functions).
    RenderTargetArrayIndex,
    /// `[[viewport_array_index]]`: the viewport rendered to (fragment and tile functions).
    ViewportArrayIndex,
}

/// One stage output.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OutputBinding {
    /// The field of the returned struct that holds the output (0 when the entry returns the
    /// output itself): its position among the decoded outputs.
    pub field: u32,
    /// What the output is: its role key and role attributes.
    pub role: OutputRole,
    /// `air.shared`: the output is shared by every vertex-amplification view.
    pub shared: bool,
    /// `air.function_constant`: the output exists only where its predicate holds (`None`:
    /// always).
    pub predicate: Option<Predicate>,
    /// `air.arg_type_name`, and the size and alignment where the role takes them.
    pub ty: ArgTypeInfo,
    /// `air.arg_name`, for diagnostics.
    pub name: Option<DiagName>,
}

/// The role of a stage output.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum OutputRole {
    /// `air.position` (`[[position]]`): the vertex's clip-space position.
    Position {
        /// `air.invariant`.
        invariant: bool,
    },
    /// `air.vertex_output, <user key>`.
    VertexOutput(UserKey),
    /// `air.point_size` (`[[point_size]]`).
    PointSize,
    /// `air.clip_distance` (`[[clip_distance]]`).
    ClipDistance {
        /// `air.clip_distance_array_size`.
        array_size: Option<u32>,
    },
    /// `air.render_target_array_index` (`[[render_target_array_index]]`): the layer rendered to.
    RenderTargetArrayIndex,
    /// `air.viewport_array_index` (`[[viewport_array_index]]`): the viewport rendered to.
    ViewportArrayIndex,
    /// `air.render_target, <attachment>, <index>`.
    RenderTarget {
        /// The color attachment written (`[[color(n)]]`).
        attachment: AbiInt,
        /// The dual-source blending index.
        index: u32,
        /// `air.raster_order_group` (`[[raster_order_group(n)]]`).
        raster_order_group: Option<u32>,
    },
    /// `air.depth` (`[[depth(..)]]`), with its `air.depth_qualifier`.
    Depth(DepthQualifier),
    /// `air.sample_mask` (`[[sample_mask]]`): the fragment's output coverage mask.
    SampleMask,
    /// `air.stencil` (`[[stencil]]`): the fragment's stencil reference value.
    Stencil,
    /// `air.imageblock_data`: a fragment function's imageblock output.
    ImageblockData(ImageblockArg),
}

/// `air.depth_qualifier`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DepthQualifier {
    /// `air.any` (`[[depth(any)]]`): the written depth may be any value.
    Any,
    /// `air.less` (`[[depth(less)]]`): the written depth is at most the fragment's own.
    Less,
    /// `air.greater` (`[[depth(greater)]]`): the written depth is at least the fragment's own.
    Greater,
}

/// An index into [`AirAbi::structs`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct StructRef(pub u32);

/// A decoded `air.struct_type_info`: the members of a struct in declaration order.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct StructInfo {
    /// The struct's members, in declaration order.
    pub members: Vec<Member>,
}

/// One `air.struct_type_info` member.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Member {
    /// The member's byte offset in its struct.
    pub offset: u32,
    /// Size of one element in bytes.
    pub size: u32,
    /// Array length (0 for a non-array member).
    pub array_len: u32,
    /// The member's own layout, when it is a struct.
    pub layout: Option<StructRef>,
    /// What the member binds in an argument buffer (`Plain` without `air.indirect_argument`).
    pub kind: MemberKind,
    /// `air.render_target, <n>` (imageblock members).
    pub render_target: Option<u32>,
    /// `air.raster_order_group, <n>` (imageblock members).
    pub raster_order_group: Option<u32>,
    /// The member's MSL type name, for diagnostics.
    pub type_name: DiagName,
    /// The member's name, for diagnostics.
    pub name: DiagName,
    /// The member's name string as bytes: for an imageblock member, its user key
    /// (`user(..)`, else the declared name), by which a view's member names the master member it
    /// views (MSL §4.7.1). The Metal matching ABI, compared for equality only.
    pub key: UserKey,
}

impl Member {
    /// The handle class of an argument-buffer member, if it is a resource handle.
    pub fn handle(&self) -> Option<AbiHandle> {
        match &self.kind {
            MemberKind::Indirect(arg) => arg.role.handle(),
            MemberKind::Plain | MemberKind::IndirectStruct { .. } => None,
        }
    }
}

/// What an argument-buffer member binds.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum MemberKind {
    /// Plain bytes.
    Plain,
    /// `air.indirect_argument, !{..}`: the member's own argument binding.
    Indirect(IndirectArg),
    /// `air.indirect_argument, <first id>`: a struct member whose members' Metal argument ids
    /// (`[[id(n)]]`) are offset by `first_id`. Ids add down the nesting path: a member's `[[id]]`
    /// is the sum of the `first_id`s of the structs that enclose it plus its role's own
    /// `air.location_index` (plus the element, for an array member).
    IndirectStruct {
        /// The id the struct's members' ids are offset by.
        first_id: u32,
    },
}

/// The binding of one argument-buffer member.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct IndirectArg {
    /// The leading integer of the member's node: its ordinal in the layout that holds it (in
    /// every corpus member, its position in [`StructInfo::members`], the LLVM field index). It is
    /// not the member's Metal `[[id(n)]]`, which comes from the role's `air.location_index` and
    /// the enclosing structs' first ids ([`MemberKind::IndirectStruct`]).
    pub ordinal: u32,
    /// The member's role, decoded as an entry parameter's is.
    pub role: ParamRole,
    /// The member's `air.arg_type_size`, `air.arg_type_align_size`, and `air.arg_type_name`.
    pub ty: ArgTypeInfo,
    /// `air.arg_name`, for diagnostics.
    pub name: Option<DiagName>,
}

/// `!air.function_constants`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FunctionConstantDecl {
    /// The constant's index (`[[function_constant(index)]]`), by which a request supplies it.
    pub index: u32,
    /// The initializer global (section `air.fc_initializer`); the constant's identity.
    pub global: GlobalId,
    /// The global's value type.
    pub ty: TypeId,
    /// The fifth operand, when present. Its meaning is open; it is recorded, not interpreted.
    pub flag: Option<bool>,
    /// The `MTLDataType` the metallib `CNST` tag states, once [`AirAbi::check_container`] ran.
    pub data_type: Option<crate::metal::DataType>,
    /// The constant's name, for diagnostics.
    pub name: DiagName,
    /// The constant's type as the metadata spells it: an MSL type name (`bool`, `uint`,
    /// `float4`), the one place AIR distinguishes a signed integer from an unsigned one.
    pub type_name: DiagName,
}

/// A constexpr sampler: its global and decoded state.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct StaticSampler {
    /// The constant global whose initializer holds the sampler words (`air.sampler_state`).
    pub global: GlobalId,
    /// The state those words decode to.
    pub state: SamplerState,
}

/// What an `air.visible_function_reference` names.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum VisibleTarget {
    /// A global: not a callable stub.
    Global(GlobalId),
    /// A function declaration that stands for the linked visible function (a callable stub).
    Function(FuncId),
}

/// The symbol a visible-function reference names: the Metal linker resolves it to the linked
/// visible function of that name. Like [`UserKey`], it is linkage ABI, compared for equality only.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct VisibleSymbol(pub Box<[u8]>);

/// `!air.visible_function_references`: `!{!"air.visible_function_reference", ptr @stub,
/// !"<symbol>"}`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VisibleFunctionRef {
    /// The operand the reference names (`ptr @stub`).
    pub target: VisibleTarget,
    /// The symbol of the linked visible function it stands for.
    pub symbol: VisibleSymbol,
}

/// One `!air.visible` entry: a function the module exports to the Metal linker (a
/// `[[visible]]` or `[[stitchable]]` function).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VisibleFunction {
    /// The exported function.
    pub func: FuncId,
    /// One binding per LLVM parameter, in parameter order.
    pub params: Vec<VisibleParam>,
    /// The return value (`air.visible_output`): one for a non-`void` function, none otherwise.
    pub outputs: Vec<VisibleOutput>,
    /// `air.stitching_info`: present for a `[[stitchable]]` function.
    pub stitching: Option<Stitching>,
    /// The function's name, for diagnostics.
    pub name: DiagName,
}

/// One parameter of a visible function.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VisibleParam {
    /// The LLVM parameter index.
    pub param: u32,
    /// Whether the parameter is an argument of the MSL signature or padding.
    pub kind: VisibleParamKind,
}

/// What a visible function's parameter is.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VisibleParamKind {
    /// `air.visible_input`: an argument of the MSL signature.
    Input {
        /// `air.arg_type_name`, for diagnostics.
        type_name: Option<DiagName>,
        /// `air.arg_name`, for diagnostics.
        name: Option<DiagName>,
    },
    /// `air.visible_padding`: a parameter the MSL signature does not have.
    Padding,
}

/// A visible function's return value (`air.visible_output`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VisibleOutput {
    /// `air.arg_type_name`, for diagnostics.
    pub type_name: Option<DiagName>,
}

/// `air.stitching_info` of a `[[stitchable]]` function: how a stitching graph passes its values.
/// Only the structure and the helper functions are decoded; the MSL type descriptors are not
/// (no translation reads them).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stitching {
    /// The return value's stitching type (`None`: a `void` function).
    pub result: Option<StitchingType>,
    /// One per stitched argument, in order.
    pub arguments: Vec<StitchingArgument>,
}

/// One `air.stitching_argument`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StitchingArgument {
    /// The argument's `air.stitching_type`.
    pub ty: StitchingType,
    /// The argument's name, for diagnostics.
    pub name: Option<DiagName>,
}

/// One `air.stitching_type`: the four helper-function operands in metadata order (`None`: a
/// `null` operand). Their roles are not established; they are recorded, not interpreted.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct StitchingType {
    /// The four helper functions, in metadata order (`None`: a `null` operand).
    pub helpers: [Option<FuncId>; 4],
}

/// One `!air.ci` entry: a function the module exports as a Core Image kernel. The declaration
/// describes how Core Image passes the kernel its arguments; no translation of this module reads
/// it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CoreImageKernel {
    /// The exported kernel function.
    pub func: FuncId,
    /// One binding per LLVM parameter, in parameter order.
    pub params: Vec<CoreImageParam>,
    /// The return value: `Some` exactly for a non-`void` function.
    pub output: Option<CoreImageOutput>,
    /// The function's name, for diagnostics.
    pub name: DiagName,
}

/// One parameter of a Core Image kernel: `!{i32 <param>, !"air.ci_<role>", <attr>*}`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CoreImageParam {
    /// The LLVM parameter index.
    pub param: u32,
    /// The parameter's `air.ci_<role>` and its role attributes.
    pub role: CoreImageRole,
    /// `air.arg_type_name`, for diagnostics.
    pub type_name: Option<DiagName>,
    /// `air.arg_name`, for diagnostics.
    pub name: Option<DiagName>,
}

/// The role of a Core Image kernel parameter (closed set). Each role is cross-checked against
/// the parameter's IR type (`abi::check`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CoreImageRole {
    /// `air.ci_builtin`: a scalar or vector number.
    Builtin,
    /// `air.ci_matrix`: a matrix, as an array of vectors or the only field of a struct.
    Matrix,
    /// `air.ci_texture`: a texture handle, a device-space pointer passed bare or as the only
    /// field of a struct.
    Texture {
        /// `air.location_count`, as stated (1 in every observed module); recorded, not
        /// interpreted.
        location_count: u32,
        /// `air.sample`, `air.read`, `air.write`, or `air.read_write`.
        access: Access,
    },
    /// `air.ci_sampler`: a sampler handle, a constant-space pointer passed bare or as the only
    /// field of a struct.
    Sampler,
    /// `air.ci_pointer`: a pointer to `size` bytes.
    Pointer {
        /// `air.read`, `air.write`, or `air.read_write`.
        access: Access,
        /// `air.arg_type_size`: the size of the pointee in bytes.
        size: u32,
        /// `air.arg_type_align_size`: Metal's alignment of the pointee.
        align: u32,
    },
}

/// A Core Image kernel's return value: `!{!"air.ci_builtin", <attr>*}`, a scalar or vector
/// number.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CoreImageOutput {
    /// `air.arg_type_name`, for diagnostics.
    pub type_name: Option<DiagName>,
}

/// The kind of an `!air.signature`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LinkageSignatureKind {
    /// `air.linked_constants_signature`.
    LinkedConstants,
    /// `air.stitching_modules_signature`.
    StitchingModules,
}

/// `!air.signature`. Only its kind and operand count are decoded. It describes how a closed
/// module was produced (the constants a linked-constants module was specialized with, the graph
/// a stitched module was built from); the module's own code is what runs, whether the module is
/// the entry's or a linked function's. A linked-constants module that still declares function
/// constants is refused (`Feature::Linking`). An `air.rename_qualified_function_signature` (a
/// function renamed to its qualified name after it was produced) is read as the signature it
/// wraps.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct LinkageSignature {
    /// Which signature the node's leading string names.
    pub kind: LinkageSignatureKind,
    /// The number of operands after that string (not decoded).
    pub operands: u32,
}
