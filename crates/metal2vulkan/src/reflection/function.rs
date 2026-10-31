//! What `MTLFunction` states about a function beyond its name and bindings: `functionType`,
//! `patchType`, `patchControlPointCount`, `vertexAttributes`, `stageInputAttributes` and
//! `functionConstantsDictionary`.
//!
//! The rules, and how they were measured, are [`FunctionProperties`]'.

use std::collections::BTreeMap;

use super::types::data_type;
use crate::air::abi::{AbiInt, ArgTypeInfo, EntryPoint, ParamRole, PatchInput, PatchKind, Stage};
use crate::air::error::{DiagName, Location};
use crate::error::{Error, Feature, Unsupported};
use crate::library::FunctionStage;
use crate::metal::{DataType, FunctionType, PatchType};

/// `MTLAttribute` (`MTLVertexAttribute` states the same): one stage-input attribute of a vertex
/// function or a kernel.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Attribute {
    name: String,
    attribute_index: u32,
    attribute_type: DataType,
    active: bool,
    patch_data: bool,
    patch_control_point_data: bool,
}

impl Attribute {
    /// `name`.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// `attributeIndex`: its `[[attribute(n)]]`.
    pub fn attribute_index(&self) -> u32 {
        self.attribute_index
    }

    /// `attributeType`.
    pub fn attribute_type(&self) -> DataType {
        self.attribute_type
    }

    /// `isActive`: whether the function reads the attribute.
    pub fn is_active(&self) -> bool {
        self.active
    }

    /// `isPatchData`: a post-tessellation vertex function's per-patch input.
    pub fn is_patch_data(&self) -> bool {
        self.patch_data
    }

    /// `isPatchControlPointData`: a field of a post-tessellation vertex function's control
    /// points.
    pub fn is_patch_control_point_data(&self) -> bool {
        self.patch_control_point_data
    }
}

/// `MTLFunctionConstant`: a function constant a function reads.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FunctionConstant {
    name: String,
    data_type: DataType,
    index: u32,
    required: bool,
}

impl FunctionConstant {
    /// `name`.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// `type`.
    pub fn data_type(&self) -> DataType {
        self.data_type
    }

    /// `index`: its `[[function_constant(n)]]`.
    pub fn index(&self) -> u32 {
        self.index
    }

    /// `required`: whether a function made from it needs a value for the constant; false
    /// exactly when the function asks `is_function_constant_defined` about it.
    pub fn is_required(&self) -> bool {
        self.required
    }
}

/// What `MTLFunction` states about a function's kind and inputs: its type, its patch, and its
/// stage-input attributes. With `functionConstantsDictionary`
/// ([`Function::function_constants_dictionary`](crate::library::Function::function_constants_dictionary)),
/// Metal answers them by these rules.
///
/// A metallib function answers from the function tags Metal answers from, decoding nothing; a
/// function of a bitcode wrapper or raw module, which Metal cannot load, answers the same from its
/// AIR. What the tags state was measured on an M4 (macOS 27.0.1) over the reflection fixtures
/// (`validation/fixtures/public/reflection/`: `metallib-tags.txt` holds the tags,
/// `metal-answers.txt` Metal's answers):
///
/// - **`TYPE`** is the function type: 0 vertex, 1 fragment, 2 kernel (a tile function too), 4
///   visible, 6 intersection, 7 mesh, 8 object.
/// - **`TESS`** is one byte for a post-tessellation vertex function: the patch type in its low two
///   bits (1 triangle, 2 quad) and the control-point count above them, 63 when the patch states
///   none (`patchControlPointCount` is then -1, as it is without a patch).
/// - **`VATT`** lists the stage-input attributes in declaration order, each a name and a 16-bit
///   word: the attribute index in its low five bits, bit 15 set when the function reads the
///   attribute (`isActive`), bit 14 for a control-point field and bit 13 for a per-patch input.
///   **`VATY`** gives each one's `MTLDataType`. A vertex function answers them as both its vertex
///   and its stage-input attributes, a kernel as its stage-input attributes only; a function
///   without them answers nil for both.
/// - **`CNST`** lists the constants the function reads, each with its `MTLDataType`, index, and a
///   flag byte that is `required`: 0 exactly for a constant the function asks
///   `is_function_constant_defined` about, 1 for every other, one it reads only as an argument's
///   predicate included. A module function answers the same from its AIR: the constants its
///   code, its callees and its entry's metadata name, directly or through a global the module's
///   static initializers compute from them, with the type its metadata spells.
/// - **A function made with constant values** (`newFunctionWithName:constantValues:error:`) is
///   the function specialized: it leaves out an attribute its constants gate off and keeps the
///   others as they were, and answers no function constants. Its attributes come from the AIR
///   specialized with its constants, for a metallib function too.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FunctionProperties {
    function_type: FunctionType,
    patch_type: PatchType,
    patch_control_point_count: Option<u32>,
    vertex_attributes: Option<Vec<Attribute>>,
    stage_input_attributes: Option<Vec<Attribute>>,
}

impl FunctionProperties {
    /// `functionType`.
    pub fn function_type(&self) -> FunctionType {
        self.function_type
    }

    /// `patchType`: a post-tessellation vertex function's patch, else `None`.
    pub fn patch_type(&self) -> PatchType {
        self.patch_type
    }

    /// `patchControlPointCount`: the control points a post-tessellation vertex function's patch
    /// states; `None` where Metal answers -1 (no patch, or a patch that states no count).
    pub fn patch_control_point_count(&self) -> Option<u32> {
        self.patch_control_point_count
    }

    /// `vertexAttributes`: a vertex function's stage-input attributes in declaration order;
    /// `None` (nil) for another function, or one without attributes.
    pub fn vertex_attributes(&self) -> Option<&[Attribute]> {
        self.vertex_attributes.as_deref()
    }

    /// `stageInputAttributes`: a vertex function's or kernel's stage-input attributes in
    /// declaration order; `None` (nil) for another function, or one without attributes.
    pub fn stage_input_attributes(&self) -> Option<&[Attribute]> {
        self.stage_input_attributes.as_deref()
    }
}

/// A value Metal's answer was not measured for: refused, typed.
fn unmeasured() -> Error {
    Error::Unsupported(Unsupported::not_implemented(Feature::Reflection))
}

/// The properties of a function of type `function_type` with patch `patch` and stage-input
/// attributes `attributes`.
fn properties(
    function_type: FunctionType,
    patch: Option<(PatchType, Option<u32>)>,
    attributes: Option<Vec<Attribute>>,
) -> FunctionProperties {
    let (patch_type, patch_control_point_count) = patch.unwrap_or((PatchType::None, None));
    FunctionProperties {
        function_type,
        patch_type,
        patch_control_point_count,
        vertex_attributes: attributes
            .clone()
            .filter(|_| function_type == FunctionType::Vertex),
        stage_input_attributes: attributes,
    }
}

/// The properties of a metallib function from its `TYPE` stage, its `VATT` attributes
/// (`(name, packed index and flags)`), their `VATY` types and its `TESS` bytes.
pub(crate) fn metallib_properties(
    stage: FunctionStage,
    vatt: &[(String, u16)],
    vaty: &[u8],
    tess: Option<&[u8]>,
) -> Result<FunctionProperties, Error> {
    const ACTIVE: u16 = 0x8000;
    const CONTROL_POINT: u16 = 0x4000;
    const PATCH: u16 = 0x2000;
    const INDEX: u16 = 0x1f;
    let function_type = match stage {
        FunctionStage::Vertex => FunctionType::Vertex,
        FunctionStage::Fragment => FunctionType::Fragment,
        FunctionStage::Kernel => FunctionType::Kernel,
        FunctionStage::Visible => FunctionType::Visible,
        FunctionStage::Extern => FunctionType::Extern,
        FunctionStage::Intersection => FunctionType::Intersection,
        FunctionStage::Mesh => FunctionType::Mesh,
        FunctionStage::Object => FunctionType::Object,
        FunctionStage::Other(_) => return Err(unmeasured()),
        // A stage the front end's crate names beyond these (it is `#[non_exhaustive]`): no
        // measured `MTLFunctionType` answer either.
        _ => return Err(unmeasured()),
    };
    let patch = match tess {
        None => None,
        Some(&[byte]) => {
            let kind = match byte & 3 {
                1 => PatchType::Triangle,
                2 => PatchType::Quad,
                _ => return Err(unmeasured()),
            };
            let count = match byte >> 2 {
                63 => None,
                n @ 1..=32 => Some(u32::from(n)),
                _ => return Err(unmeasured()),
            };
            Some((kind, count))
        }
        Some(_) => return Err(unmeasured()),
    };
    if vatt.len() != vaty.len() {
        return Err(unmeasured());
    }
    let attributes = (!vatt.is_empty())
        .then(|| {
            vatt.iter()
                .zip(vaty)
                .map(|((name, packed), &ty)| {
                    if packed & !(ACTIVE | CONTROL_POINT | PATCH | INDEX) != 0 {
                        return Err(unmeasured());
                    }
                    Ok(Attribute {
                        name: name.clone(),
                        attribute_index: u32::from(packed & INDEX),
                        attribute_type: DataType::try_from(u32::from(ty))
                            .map_err(|_| unmeasured())?,
                        active: packed & ACTIVE != 0,
                        patch_data: packed & PATCH != 0,
                        patch_control_point_data: packed & CONTROL_POINT != 0,
                    })
                })
                .collect::<Result<Vec<_>, Error>>()
        })
        .transpose()?;
    Ok(properties(function_type, patch, attributes))
}

/// The properties of a visible function of a bitcode wrapper or raw module.
pub(crate) fn visible_properties() -> FunctionProperties {
    properties(FunctionType::Visible, None, None)
}

/// The properties of entry `ep`, from its AIR: its stage, its `air.patch`, and its stage-input
/// parameters (`air.vertex_input`, `air.stage_in`, and a post-tessellation function's
/// control-point fields and per-patch inputs) in declaration order, each active unless
/// `air.arg_unused`. A specialized entry has left out those its constants gate off. A function's
/// control points are one argument: every field is active where the function reads any of them,
/// one the AIR marks `air.arg_unused` too, and inactive where it reads none (the same on the M3 Max
/// and the M4).
pub(crate) fn entry_properties(ep: &EntryPoint) -> Result<FunctionProperties, Error> {
    let mut attributes = Vec::new();
    for p in &ep.params {
        match &p.role {
            ParamRole::VertexInput { location } | ParamRole::StageIn { location } => {
                attributes.push(attribute(
                    ep,
                    p.param,
                    p.name.as_ref(),
                    location.index,
                    &p.ty,
                    p.unused,
                    Input::Vertex,
                )?);
            }
            ParamRole::PatchInput(PatchInput::ControlPoints { fields, .. }) => {
                // A specialized entry marks the fields its constants gate off.
                let present = || fields.iter().filter(|f| !f.gated_off);
                let unread = p.unused || present().all(|f| f.unused);
                for f in present() {
                    attributes.push(attribute(
                        ep,
                        p.param,
                        f.name.as_ref(),
                        f.location.index,
                        &f.ty,
                        unread,
                        Input::ControlPoint,
                    )?);
                }
            }
            ParamRole::PatchInput(PatchInput::Patch { location }) => {
                attributes.push(attribute(
                    ep,
                    p.param,
                    p.name.as_ref(),
                    location.index,
                    &p.ty,
                    p.unused,
                    Input::Patch,
                )?);
            }
            _ => {}
        }
    }
    let (function_type, patch) = match ep.stage {
        Stage::Kernel { .. } => (FunctionType::Kernel, None),
        Stage::Fragment { .. } => (FunctionType::Fragment, None),
        Stage::Object { .. } => (FunctionType::Object, None),
        Stage::Mesh { .. } => (FunctionType::Mesh, None),
        Stage::Vertex { patch } => (
            FunctionType::Vertex,
            patch.map(|p| {
                let kind = match p.kind {
                    PatchKind::Triangle => PatchType::Triangle,
                    PatchKind::Quad => PatchType::Quad,
                };
                (kind, p.control_points)
            }),
        ),
    };
    Ok(properties(
        function_type,
        patch,
        (!attributes.is_empty()).then_some(attributes),
    ))
}

/// What a stage-input attribute is to its function.
#[derive(Clone, Copy)]
enum Input {
    /// A vertex function's or kernel's own attribute.
    Vertex,
    /// A field of a post-tessellation vertex function's control points.
    ControlPoint,
    /// A post-tessellation vertex function's per-patch input.
    Patch,
}

/// The attribute of parameter `param` of entry `ep` (or of one of its control-point fields):
/// `name`, at `index`, of type `ty`, which the function reads unless `unused`.
fn attribute(
    ep: &EntryPoint,
    param: u32,
    name: Option<&DiagName>,
    index: AbiInt,
    ty: &ArgTypeInfo,
    unused: bool,
    input: Input,
) -> Result<Attribute, Error> {
    let gap = |key: &'static str, ty: Option<&DiagName>| {
        let mut u = Unsupported::not_implemented(Feature::Reflection).at(Location::metadata(
            Some(ep.name.clone()),
            Some(param),
            Some(key),
        ));
        if let Some(ty) = ty {
            u = u.naming(ty.clone());
        }
        Error::Unsupported(u)
    };
    let AbiInt::Const(attribute_index) = index else {
        return Err(gap("air.location_index", None));
    };
    let attribute_type = ty
        .name
        .as_ref()
        .and_then(|n| data_type(&n.to_string()))
        .ok_or_else(|| gap("air.arg_type_name", ty.name.as_ref()))?;
    let name = name.ok_or_else(|| gap("air.arg_name", None))?;
    Ok(Attribute {
        name: name.to_string(),
        attribute_index,
        attribute_type,
        active: !unused,
        patch_data: matches!(input, Input::Patch),
        patch_control_point_data: matches!(input, Input::ControlPoint),
    })
}

/// The `functionConstantsDictionary` of a metallib function from its `CNST` entries
/// (`(name, raw MTLDataType, index, flag byte)`).
pub(crate) fn metallib_constants(
    cnst: &[(String, u8, u16, u8)],
) -> Result<BTreeMap<String, FunctionConstant>, Error> {
    cnst.iter()
        .map(|(name, ty, index, flag)| {
            let required = match flag {
                0 => false,
                1 => true,
                _ => return Err(unmeasured()),
            };
            let constant = FunctionConstant {
                name: name.clone(),
                data_type: DataType::try_from(u32::from(*ty)).map_err(|_| unmeasured())?,
                index: u32::from(*index),
                required,
            };
            Ok((name.clone(), constant))
        })
        .collect()
}

/// The `functionConstantsDictionary` of function `func`, named `name`, of a bitcode wrapper or
/// raw module (`module`, whose ABI is `abi`): the constants it reads, each required unless it
/// asks whether the constant is defined, as the metallib linker writes them into the function's
/// `CNST` tag (`air::canon::constants_read`). Each constant's type is the `MTLDataType` of the MSL
/// type the module's metadata spells.
///
/// Refused, naming the function, where the module leaves that unknown (a static initializer the
/// analysis does not follow, any function of a module whose static initializer branches on a
/// constant, a function address the function's code takes) and for a type outside the MSL type
/// grammar.
pub(crate) fn module_constants(
    module: &crate::air::ir::Module,
    abi: &crate::air::abi::AirAbi,
    func: crate::air::ir::FuncId,
    name: &str,
    m: &mut crate::air::budget::Meter,
) -> Result<BTreeMap<String, FunctionConstant>, Error> {
    let refused = || {
        Error::Unsupported(
            Unsupported::not_implemented(Feature::Reflection).naming(DiagName::new(name)),
        )
    };
    let reads = crate::air::canon::constants_read(module, abi, func, m)?.ok_or_else(refused)?;
    reads
        .into_iter()
        .map(|read| {
            let decl = abi
                .function_constants
                .iter()
                .find(|c| c.index == read.index)
                .ok_or_else(refused)?;
            let constant = FunctionConstant {
                name: decl.name.to_string(),
                data_type: data_type(&decl.type_name.to_string()).ok_or_else(refused)?,
                index: read.index,
                required: !read.asked,
            };
            Ok((constant.name.clone(), constant))
        })
        .collect()
}
