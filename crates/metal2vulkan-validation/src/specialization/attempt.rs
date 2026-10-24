//! One specialization audit attempt: a row's compile under one value set, answering each
//! pipeline input the compile asks for that the audit can derive from the row's AIR.
//!
//! The compile runs exactly as the plain census's ([`crate::engine::translate_with`]: the same
//! meter, deadline, allocator measure, standalone link, relocation, and L3), with the set's
//! function-constant values. When it stops at `NeedsInput`, the audit derives that input from the
//! row and compiles again on a fresh meter under the same deadline, as the plain census does for
//! a kernel's stage-input descriptor. It derives:
//! - a kernel's stage-input descriptor: the one its interface implies (`stage_inputs::implied`), as
//!   the plain census does, here under the set's constants;
//! - a post-tessellation vertex function's vertex descriptor: the L4 lane's
//!   (`acceptance::row::vertex_row_of`), every attribute the specialized entry reads pulled as
//!   the compile's own vertex stage pulls it (`DeviceProfile::vertex_entry_inputs`);
//! - a tile function's attachment formats: each implicit imageblock member's, the unnormalized
//!   format of the type its `air.struct_type_info` entry states (or the pixel format a
//!   color-attachment type names);
//! - a storage texture slot's format on a profile that cannot access it without one: the widest
//!   unnormalized format of the slot's texel class (`RGBA32Float`, `RGBA32Sint`, `RGBA32Uint`),
//!   as `storage_image_choice` declares it for the profile;
//! - a visible-function reference: the module the row's authored cases link the symbol from
//!   ([`AuthoredLinks`], a real pipeline's linked function); else the one retained module of the
//!   row's parent libraries that defines it (`library_module::indexed_symbol_definers`), or the
//!   row's own module when it alone defines it (a linked function's reference back into the
//!   row). The compile asks for each symbol a linked module references in turn;
//! - a visible-function table's candidates: every function the authored cases link that the
//!   attempt has not linked yet (Metal's `MTLLinkedFunctions`, the functions a pipeline can place
//!   in a table). A call no linked function of its type answers asks again, and stays the input.
//!
//! Each kind of input is derived at most once per slot or symbol, and each table stop links at
//! least one more authored function, so the attempt's compiles are bounded by the row's slots,
//! references, and authored functions. A constant the set leaves undefined, and every other
//! input, stay what the compile says, with the reason the audit supplies none. An index the audit
//! cannot read makes the attempt a transient `tooling` failure, never a recorded reason.

use crate::acceptance::row::vertex_row_of;
use crate::check::interface::{AirInterface, Root, TexelClass};
use crate::check::loop_free::DecodedModule;
use crate::engine::PendingL3;
use crate::engine::{bounded_detail, translate_unvalidated, Attempt, FailureKind, Outcome};
use crate::executor_contract::stage_inputs;
use crate::specialization::AuthoredLinks;
use metal2vulkan::device::DeviceProfile;
use metal2vulkan::error::{Error, RuntimeInput};
use metal2vulkan::library::FunctionConstantValues;
use metal2vulkan::metal::format::storage_image_choice;
use metal2vulkan::metal::PixelFormat;
use metal2vulkan::pipeline::{EntryDescriptor, EntryRef, LinkedSource, LoweringInputs};
use metal2vulkan::pipeline::{SlotClass, SlotRef};
use metal2vulkan::Budget;
use metal2vulkan_air::abi::{ImageblockKind, ParamRole};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::rc::Rc;
use std::time::Instant;

/// An input the audit derived and the compile used.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "input")]
pub enum Supplied {
    StageInputDescriptor,
    VertexDescriptor,
    /// Each color attachment's format, in color order. A list, not a map: an internally tagged
    /// enum cannot read a JSON object's keys back as integers.
    TileAttachments {
        formats: Vec<(u32, String)>,
    },
    StorageTextureFormat {
        slot: u32,
        format: String,
    },
    /// The module that defines `symbol`.
    LinkedFunction {
        symbol: String,
        source: LinkSource,
    },
}

/// Where a linked function's module came from.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "from")]
pub enum LinkSource {
    /// The retained module the row's authored cases link it from.
    Authored { module: String },
    /// The one retained module of the row's parent libraries that defines it.
    Library { module: String },
    /// The row's own module, the only one that defines it.
    Row,
}

impl Supplied {
    /// The input kind, as the outcome key names it (`needs_input <kind>`).
    pub fn kind(&self) -> &'static str {
        match self {
            Supplied::StageInputDescriptor => "StageInputDescriptor",
            Supplied::VertexDescriptor => "VertexDescriptor",
            Supplied::TileAttachments { .. } => "TileAttachments",
            Supplied::StorageTextureFormat { .. } => "StorageTextureFormat",
            Supplied::LinkedFunction { .. } => "LinkedFunction",
        }
    }
}

/// Where the attempt finds a visible function's module: the functions the row's authored cases
/// link and the row's parent libraries, both read through the index (`None`: the attempt links
/// nothing, as in a test without an index).
#[derive(Clone, Copy, Debug)]
pub struct Linkage<'a> {
    pub corpus_root: &'a Path,
    pub index: &'a Path,
    pub parent_libraries: &'a [String],
    pub authored: &'a AuthoredLinks,
}

/// One attempt's result: its outcome and cost, the inputs it derived, and, when it still ends
/// at `NeedsInput`, why the audit supplied none.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpecAttempt {
    pub attempt: Attempt,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub supplied: Vec<Supplied>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub unsupplied: Option<String>,
    /// The refusal is of the ray-tracing family ([`is_ray_tracing`]), out of scope for the
    /// launch: reported apart.
    #[serde(default)]
    pub ray_tracing: bool,
}

/// What the attempt has derived so far and what it asked of the row's AIR.
struct Derivation<'a> {
    bitcode: &'a [u8],
    entry: &'a str,
    constants: &'a FunctionConstantValues,
    profile: &'a DeviceProfile,
    linkage: Option<Linkage<'a>>,
    /// The attempt's wall deadline, which a derivation that decodes the row through the product
    /// (`DeviceProfile::vertex_entry_inputs`) also runs under.
    deadline: Option<Instant>,
    inputs: LoweringInputs,
    /// (symbol, its retained module's bitcode, or `None` for the row's own) of every linked
    /// function, in resolution order.
    linked: Vec<(String, Option<Rc<[u8]>>)>,
    /// Each retained module read, by hash, read once however many functions it defines.
    modules: BTreeMap<String, Rc<[u8]>>,
    /// Each derived input's identity (kind and slot or symbol), so none is derived twice.
    derived: BTreeSet<String>,
    supplied: Vec<Supplied>,
    /// The row's decoded module, on first need.
    module: Option<Result<DecodedModule, String>>,
    /// The row's interface under the set's constants, on first need.
    interface: Option<Result<AirInterface, String>>,
}

/// Compile entry `entry` of `bitcode` for `profile` under `constants`, deriving each input the
/// compile asks for that the module states (see the module docs), then link and relocate as the
/// plain census does, up to L3's `spirv-val`: what that still has to run on a module the last
/// compile linked comes back beside the attempt (the specialization worker runs it in its L3
/// stage, [`crate::worker::l3`]). `deadline` bounds every compile of the attempt together.
///
/// Each compile is its own [`translate_unvalidated`]: its own meter and its own allocator window,
/// so the attempt's cost is its last compile's, the ledger and the allocator peak over the same
/// scope. A derivation runs between compiles, outside every window: what it holds (the row's
/// decoded interface, a linked module's bitcode) is an input of the next compile, as the row's own
/// bitcode is.
pub fn translate_specialized(
    bitcode: &[u8],
    entry: &str,
    constants: &FunctionConstantValues,
    profile: &DeviceProfile,
    linkage: Option<Linkage<'_>>,
    deadline: Option<Instant>,
) -> (SpecAttempt, Option<PendingL3>) {
    let (replay, pending) =
        replay_unvalidated(bitcode, entry, constants, profile, linkage, deadline);
    (replay.attempt, pending)
}

/// An attempt, and what its last compile took besides the row and the set: the lowering inputs
/// and the linked functions the attempt derived.
pub struct Replay<'a> {
    pub attempt: SpecAttempt,
    pub inputs: LoweringInputs,
    row: &'a [u8],
    linked: Vec<(String, Option<Rc<[u8]>>)>,
}

impl Replay<'_> {
    /// Each linked function as the last compile took it, with no constants of its own
    /// (`unlinked`).
    pub fn sources<'s>(&'s self, unlinked: &'s FunctionConstantValues) -> Vec<LinkedSource<'s>> {
        sources(&self.linked, self.row, unlinked)
    }
}

/// Each linked function of `linked` as a compile takes it: its retained module, or the row's own
/// (`row`).
fn sources<'s>(
    linked: &'s [(String, Option<Rc<[u8]>>)],
    row: &'s [u8],
    unlinked: &'s FunctionConstantValues,
) -> Vec<LinkedSource<'s>> {
    linked
        .iter()
        .map(|(symbol, module)| {
            LinkedSource::new(symbol, module.as_deref().unwrap_or(row), unlinked)
        })
        .collect()
}

/// [`translate_specialized`] with L3 run in this process, keeping what its last compile took
/// ([`Replay`]).
pub fn replay_specialized<'a>(
    bitcode: &'a [u8],
    entry: &'a str,
    constants: &'a FunctionConstantValues,
    profile: &'a DeviceProfile,
    linkage: Option<Linkage<'a>>,
    deadline: Option<Instant>,
) -> Replay<'a> {
    let (mut replay, pending) =
        replay_unvalidated(bitcode, entry, constants, profile, linkage, deadline);
    if let Some(pending) = pending {
        replay.attempt.attempt.settle_l3(pending.check());
    }
    replay
}

/// [`translate_specialized`], keeping what its last compile took ([`Replay`]).
fn replay_unvalidated<'a>(
    bitcode: &'a [u8],
    entry: &'a str,
    constants: &'a FunctionConstantValues,
    profile: &'a DeviceProfile,
    linkage: Option<Linkage<'a>>,
    deadline: Option<Instant>,
) -> (Replay<'a>, Option<PendingL3>) {
    let mut d = Derivation {
        bitcode,
        entry,
        constants,
        profile,
        linkage,
        deadline,
        inputs: LoweringInputs::new(),
        linked: Vec::new(),
        modules: BTreeMap::new(),
        derived: BTreeSet::new(),
        supplied: Vec::new(),
        module: None,
        interface: None,
    };
    loop {
        let mut asked = None;
        let mut ray_tracing = false;
        let (attempt, pending) = translate_unvalidated(profile, deadline, |budget| {
            let unlinked = FunctionConstantValues::defaults();
            let result = d
                .profile
                .compile_entry(&d.descriptor(&unlinked), &d.inputs, budget);
            match &result.0 {
                Err(Error::NeedsInput(input)) => asked = Some(input.clone()),
                Err(error) => ray_tracing = is_ray_tracing(error),
                Ok(_) => {}
            }
            result
        });
        let Some(input) = asked else {
            return (d.finish(attempt, None, ray_tracing), pending);
        };
        match d.derive(&input) {
            Ok(()) => {}
            Err(Unsupplied::Why(why)) => return (d.finish(attempt, Some(why), ray_tracing), None),
            Err(Unsupplied::Tooling(why)) => {
                let attempt = Attempt {
                    outcome: Outcome::Failed {
                        failure: FailureKind::Tooling,
                    },
                    detail: Some(bounded_detail(&format!(
                        "the audit could not read the index: {why}"
                    ))),
                    ..attempt
                };
                return (d.finish(attempt, None, ray_tracing), None);
            }
        }
    }
}

/// Whether `error` refuses a ray-tracing construct: the ray-tracing family or its lowering, an
/// intersection through intersection functions, or a device capability that only ray queries
/// give. Ray tracing is out of scope for the launch (user, 2026-10-03), so the report lists these
/// apart.
pub fn is_ray_tracing(error: &Error) -> bool {
    use metal2vulkan::device::{VkExtension, VkFeature};
    use metal2vulkan::error::{Capability, Feature, LowerFamily, Needs};
    let Error::Unsupported(u) = error else {
        return false;
    };
    let feature = matches!(
        u.feature,
        Feature::RayTracing
            | Feature::IntersectionFunctionTable
            | Feature::Lowering(LowerFamily::RayTracing)
    );
    let capability = match &u.needs {
        Needs::AnyOf(alternatives) => alternatives.iter().flatten().any(|c| {
            matches!(
                c,
                Capability::Feature(VkFeature::RayQuery)
                    | Capability::Extension(VkExtension::KhrRayQuery)
            )
        }),
        _ => false,
    };
    feature || capability
}

/// Why the audit supplies no input: a property of the row (recorded with the attempt), or an
/// index it could not read (the attempt is then a transient `tooling` failure).
enum Unsupplied {
    Why(String),
    Tooling(String),
}

impl From<String> for Unsupplied {
    fn from(why: String) -> Unsupplied {
        Unsupplied::Why(why)
    }
}

impl From<&str> for Unsupplied {
    fn from(why: &str) -> Unsupplied {
        Unsupplied::Why(why.to_string())
    }
}

impl<'a> Derivation<'a> {
    /// The row's entry with the set's constants and the linked functions derived so far, each
    /// with no constants of its own (`unlinked`).
    fn descriptor<'s>(&'s self, unlinked: &'s FunctionConstantValues) -> EntryDescriptor<'s> {
        let mut descriptor = EntryDescriptor::new(self.bitcode, EntryRef::Name(self.entry));
        descriptor.constants = self.constants.clone();
        descriptor.linked = sources(&self.linked, self.bitcode, unlinked);
        descriptor
    }

    /// The attempt that ended with `attempt`, with what its last compile took.
    fn finish(self, attempt: Attempt, unsupplied: Option<String>, ray_tracing: bool) -> Replay<'a> {
        Replay {
            attempt: SpecAttempt {
                attempt,
                supplied: self.supplied,
                unsupplied,
                ray_tracing,
            },
            inputs: self.inputs,
            row: self.bitcode,
            linked: self.linked,
        }
    }

    /// Derive `input` into the next compile's inputs, or say why the audit supplies none.
    fn derive(&mut self, input: &RuntimeInput) -> Result<(), Unsupplied> {
        let identity = match input {
            RuntimeInput::StorageTextureFormat { slot } => Some(format!("{input:?}{}", slot.index)),
            RuntimeInput::LinkedFunction { symbol } => Some(format!("LinkedFunction{symbol}")),
            _ => Some(crate::engine::bounded_detail(&format!(
                "{:?}",
                Outcome::from(&Error::NeedsInput(input.clone()))
            ))),
        };
        if identity.is_some_and(|identity| !self.derived.insert(identity)) {
            return Err("the compile asked again for an input the audit derived".into());
        }
        let supplied = match input {
            RuntimeInput::FunctionConstant { index } => {
                return Err(Unsupplied::Why(format!(
                    "constant {index} is undefined in this set: the set is what the audit \
                     measures"
                )))
            }
            RuntimeInput::StageInputDescriptor { .. } => {
                let (_, interface) = self.interface()?;
                let fetches = stage_inputs::implied(interface)?;
                let descriptor = stage_inputs::descriptor(&fetches)
                    .ok_or("the kernel declares no stage input")?;
                self.inputs.stage_input = Some(descriptor);
                Supplied::StageInputDescriptor
            }
            RuntimeInput::VertexDescriptor { .. } => {
                let (decoded, interface) = self.interface()?;
                let row = vertex_row_of(decoded.module(), interface)
                    .map_err(|unmodelled| unmodelled.to_string())?;
                let descriptor = row
                    .template
                    .vertex_descriptor
                    .ok_or("the function reads no attribute a descriptor lays out")?;
                // The fetches the compile's own vertex stage takes from the descriptor: a
                // post-tessellation function's patch fetches of the attributes it reads.
                let unlinked = FunctionConstantValues::defaults();
                let inputs = self
                    .profile
                    .vertex_entry_inputs(
                        &self.descriptor(&unlinked),
                        Some(&descriptor),
                        &Budget::with_deadline(self.deadline),
                    )
                    .map_err(|error| error.to_string())?;
                self.inputs.vertex_fetch = inputs.vertex_fetch;
                Supplied::VertexDescriptor
            }
            RuntimeInput::TileAttachments => {
                let formats = self.tile_formats()?;
                let names = formats
                    .iter()
                    .map(|(&color, format)| (color, format.name().to_string()))
                    .collect();
                self.inputs.tile_attachments = Some(formats);
                Supplied::TileAttachments { formats: names }
            }
            RuntimeInput::StorageTextureFormat { slot } => {
                let format = self.storage_format(*slot)?;
                let choice =
                    storage_image_choice(format, self.profile).map_err(|e| e.to_string())?;
                self.inputs.storage_images.insert(*slot, choice);
                Supplied::StorageTextureFormat {
                    slot: slot.index,
                    format: format.name().to_string(),
                }
            }
            RuntimeInput::LinkedFunction { symbol } => self.linked_function(&symbol.to_string())?,
            other => {
                return Err(Unsupplied::Why(format!(
                    "{:?} is not an input the audit derives",
                    Outcome::from(&Error::NeedsInput(other.clone())).key()
                )))
            }
        };
        self.supplied.push(supplied);
        Ok(())
    }

    /// The module `symbol` is linked from: the authored cases' module, else the one retained
    /// module of the parent libraries that defines it, else the row's own module when it alone
    /// defines it.
    fn linked_function(&mut self, symbol: &str) -> Result<Supplied, Unsupplied> {
        let symbol = symbol.to_string();
        let linkage = self
            .linkage
            .ok_or("the attempt has no index to resolve a linked function in")?;
        if let Some(module) = linkage.authored.functions.get(&symbol) {
            let bitcode = self.retained(linkage, module)?;
            self.linked.push((symbol.clone(), Some(bitcode)));
            return Ok(Supplied::LinkedFunction {
                symbol,
                source: LinkSource::Authored {
                    module: module.clone(),
                },
            });
        }
        if linkage.authored.conflicts.contains(&symbol) {
            return Err(Unsupplied::Why(format!(
                "the row's authored cases link {symbol:?} from more than one module"
            )));
        }
        let (definers, module) = crate::library_module::indexed_symbol_definers(
            linkage.corpus_root,
            linkage.index,
            &symbol,
            linkage.parent_libraries,
        )
        .map_err(Unsupplied::Tooling)?;
        let own = self.defines(&symbol)?;
        match (module, own) {
            (Some(module), false) => {
                let bitcode = match self.modules.get(&module.module_sha256) {
                    Some(bitcode) => Rc::clone(bitcode),
                    None => {
                        let bitcode = Rc::<[u8]>::from(module.bitcode()?);
                        self.modules
                            .insert(module.module_sha256.clone(), Rc::clone(&bitcode));
                        bitcode
                    }
                };
                self.linked.push((symbol.clone(), Some(bitcode)));
                Ok(Supplied::LinkedFunction {
                    symbol,
                    source: LinkSource::Library {
                        module: module.module_sha256,
                    },
                })
            }
            (None, true) if definers.is_empty() => {
                self.linked.push((symbol.clone(), None));
                Ok(Supplied::LinkedFunction {
                    symbol,
                    source: LinkSource::Row,
                })
            }
            _ => Err(Unsupplied::Why(format!(
                "{} retained modules of the {} parent libraries define {symbol:?}{}",
                definers.len(),
                linkage.parent_libraries.len(),
                if own {
                    ", and so does the row's own module"
                } else {
                    ""
                }
            ))),
        }
    }

    /// The bitcode of retained module `hash`, read from its indexed byte range on first need.
    fn retained(&mut self, linkage: Linkage<'_>, hash: &str) -> Result<Rc<[u8]>, Unsupplied> {
        if let Some(bitcode) = self.modules.get(hash) {
            return Ok(Rc::clone(bitcode));
        }
        let mut read = None;
        // A public linked module is no indexed row: its text is assembled.
        if let Some(module) =
            crate::library_module::find_public_library_module(hash).map_err(Unsupplied::Tooling)?
        {
            read = Some(module.bitcode().map_err(Unsupplied::Tooling)?);
        } else {
            crate::library_module::for_each_indexed_library_module(
                linkage.corpus_root,
                linkage.index,
                &[hash.to_string()],
                |module| {
                    read = Some(module.bitcode()?);
                    Ok(())
                },
            )
            .map_err(Unsupplied::Tooling)?;
        }
        let bitcode =
            Rc::<[u8]>::from(read.ok_or_else(|| {
                Unsupplied::Tooling(format!("library module {hash} was not read"))
            })?);
        self.modules.insert(hash.to_string(), Rc::clone(&bitcode));
        Ok(bitcode)
    }

    /// Whether the row's own module defines a function named `symbol` (with a body). It reads
    /// the decoded module alone: no slot of the entry's interface needs to evaluate.
    fn defines(&mut self, symbol: &str) -> Result<bool, String> {
        let module = self.module()?.module();
        let Some(name) = module.symbols().lookup(symbol.as_bytes()) else {
            return Ok(false);
        };
        Ok(module
            .parts()
            .funcs
            .values()
            .any(|function| function.name == name && function.body.is_some()))
    }

    /// The row's decoded module.
    fn module(&mut self) -> Result<&DecodedModule, String> {
        let bitcode = self.bitcode;
        let entry = self.entry;
        match self.module.get_or_insert_with(|| {
            DecodedModule::from_bitcode(entry, bitcode)
                .map_err(|evidence| format!("the AIR does not decode: {evidence}"))
        }) {
            Ok(decoded) => Ok(decoded),
            Err(why) => Err(why.clone()),
        }
    }

    /// The row's decoded module and its interface under the set's constants.
    fn interface(&mut self) -> Result<(&DecodedModule, &AirInterface), String> {
        self.module()?;
        let Some(Ok(decoded)) = &self.module else {
            return Err("the row's module was not decoded".into());
        };
        let (entry, constants) = (self.entry, self.constants);
        let interface = self
            .interface
            .get_or_insert_with(|| interface_of(decoded, entry, constants));
        match interface {
            Ok(interface) => Ok((decoded, interface)),
            Err(why) => Err(why.clone()),
        }
    }

    /// Each implicit imageblock member's color index and the format its type states.
    fn tile_formats(&mut self) -> Result<BTreeMap<u32, PixelFormat>, String> {
        let (_, interface) = self.interface()?;
        let mut formats = BTreeMap::new();
        for param in &interface.params {
            let ParamRole::Imageblock(arg) = &param.binding.role else {
                continue;
            };
            let aliased = match arg.kind {
                ImageblockKind::Implicit => false,
                ImageblockKind::Explicit if arg.alias_implicit => true,
                _ => continue,
            };
            let layout = arg
                .layout
                .and_then(|r| interface.structs.get(usize::try_from(r.0).ok()?))
                .ok_or("an implicit imageblock without a layout")?;
            for (i, member) in layout.members.iter().enumerate() {
                let color = if aliased {
                    u32::try_from(i).map_err(|_| "a member index past u32")?
                } else {
                    member
                        .render_target
                        .ok_or("an implicit imageblock member without a render target")?
                };
                let name = member.type_name.to_string();
                let format = tile_format(&name).ok_or_else(|| {
                    format!("implicit imageblock member {color} of type {name:?} names no format")
                })?;
                if formats.insert(color, format).is_some_and(|f| f != format) {
                    return Err(format!("color {color} is two formats"));
                }
            }
        }
        if formats.is_empty() {
            return Err("no implicit imageblock member".into());
        }
        Ok(formats)
    }

    /// The widest unnormalized format of texture slot `slot`'s texel class.
    fn storage_format(&mut self, slot: SlotRef) -> Result<PixelFormat, String> {
        if slot.class != SlotClass::Texture {
            return Err(format!("slot {slot:?} is not a texture slot"));
        }
        let (_, interface) = self.interface()?;
        let mut classes = BTreeSet::new();
        for param in &interface.params {
            if !matches!(param.binding.role, ParamRole::Texture(_)) {
                continue;
            }
            let Some((first, count)) = param.slots() else {
                continue;
            };
            if !(first..first.saturating_add(count)).contains(&slot.index) {
                continue;
            }
            let root = param.binding.param;
            let uses = if param.table {
                // A table (`array_ref`) of textures: its element's handle, 8 bytes per slot.
                interface
                    .loaded_uses(root, u64::from(slot.index - first) * 8)
                    .collect::<Vec<_>>()
            } else {
                interface.texture_uses(Root::Param(root)).collect()
            };
            classes.extend(uses.into_iter().filter_map(|texture| texture.texel));
        }
        // Metal binds a `ulong` texture to `RG32Uint` alone.
        if classes.contains(&TexelClass::Ulong) {
            return if classes.len() == 1 {
                Ok(PixelFormat::RG32Uint)
            } else {
                Err(format!(
                    "texture slot {} is used as more than one texel class",
                    slot.index
                ))
            };
        }
        let sint = classes.contains(&TexelClass::Sint);
        let uint = classes.contains(&TexelClass::Uint) || classes.contains(&TexelClass::Int);
        match (classes.contains(&TexelClass::Float), sint, uint) {
            (true, false, false) => Ok(PixelFormat::RGBA32Float),
            (false, true, _) if !classes.contains(&TexelClass::Uint) => Ok(PixelFormat::RGBA32Sint),
            (false, false, true) => Ok(PixelFormat::RGBA32Uint),
            (false, false, false) => Err(format!(
                "no operation on texture slot {} states its texel class",
                slot.index
            )),
            _ => Err(format!(
                "texture slot {} is used as more than one texel class",
                slot.index
            )),
        }
    }
}

/// The interface of `entry` of `decoded` under `constants`.
fn interface_of(
    decoded: &DecodedModule,
    entry: &str,
    constants: &FunctionConstantValues,
) -> Result<AirInterface, String> {
    let abi = decoded
        .abi()
        .map_err(|error| format!("the AIR metadata ABI does not decode: {error}"))?;
    AirInterface::new(
        decoded.module(),
        abi,
        entry,
        &crate::check::interface::bound_constants(constants),
    )
}

/// The pixel format an implicit imageblock member of MSL type `name` holds: the unnormalized
/// format of a scalar or 2- or 4-lane vector type, or the format a color-attachment type
/// (`rgba8unorm<half4>`) names.
pub fn tile_format(name: &str) -> Option<PixelFormat> {
    use PixelFormat as P;
    let packed = name.split_once('<').map(|(head, _)| head).unwrap_or(name);
    Some(match packed {
        "half" => P::R16Float,
        "half2" => P::RG16Float,
        "half4" => P::RGBA16Float,
        "float" => P::R32Float,
        "float2" => P::RG32Float,
        "float4" => P::RGBA32Float,
        "char" => P::R8Sint,
        "char2" => P::RG8Sint,
        "char4" => P::RGBA8Sint,
        "uchar" => P::R8Uint,
        "uchar2" => P::RG8Uint,
        "uchar4" => P::RGBA8Uint,
        "short" => P::R16Sint,
        "short2" => P::RG16Sint,
        "short4" => P::RGBA16Sint,
        "ushort" => P::R16Uint,
        "ushort2" => P::RG16Uint,
        "ushort4" => P::RGBA16Uint,
        "int" => P::R32Sint,
        "int2" => P::RG32Sint,
        "int4" => P::RGBA32Sint,
        "uint" => P::R32Uint,
        "uint2" => P::RG32Uint,
        "uint4" => P::RGBA32Uint,
        "r8unorm" => P::R8Unorm,
        "r8snorm" => P::R8Snorm,
        "r16unorm" => P::R16Unorm,
        "r16snorm" => P::R16Snorm,
        "rg8unorm" => P::RG8Unorm,
        "rg8snorm" => P::RG8Snorm,
        "rg16unorm" => P::RG16Unorm,
        "rg16snorm" => P::RG16Snorm,
        "rgba8unorm" => P::RGBA8Unorm,
        "srgba8unorm" => P::RGBA8UnormSRGB,
        "rgba8snorm" => P::RGBA8Snorm,
        "rgba16unorm" => P::RGBA16Unorm,
        "rgba16snorm" => P::RGBA16Snorm,
        "rgb10a2" => P::RGB10A2Unorm,
        "rg11b10f" => P::RG11B10Float,
        "rgb9e5" => P::RGB9E5Float,
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{load_profile, BitcodeError, RowSource};
    use crate::specialization::{plan, RowPlan, SpecKind};
    use metal2vulkan::Budget;
    use metal2vulkan_air::budget::Meter;
    use metal2vulkan_devtools::{engine_dir, fixtures_dir};
    use std::path::PathBuf;

    /// The bitcode of public fixture `path`, or `None` (a visible skip) without `llvm-as`.
    fn bitcode(path: PathBuf) -> Option<Vec<u8>> {
        let text = std::fs::read_to_string(&path).unwrap();
        match RowSource::of(None, text).bitcode() {
            Ok(bytes) => Some(bytes),
            Err(BitcodeError::Tooling(why)) => {
                eprintln!("SKIPPED: {why}");
                None
            }
            Err(other) => panic!("{}: {other}", path.display()),
        }
    }

    fn attempt(
        bitcode: &[u8],
        entry: &str,
        constants: &FunctionConstantValues,
        profile: &str,
    ) -> SpecAttempt {
        let profile = load_profile(profile).unwrap();
        replay_specialized(bitcode, entry, constants, &profile, None, None).attempt
    }

    fn row_plan(bitcode: &[u8], entry: &str) -> RowPlan {
        let mut meter = Meter::new(&Budget::standard());
        let decoded = metal2vulkan_air::decode(bitcode, &mut meter).unwrap();
        RowPlan::of(&decoded, entry, &[])
    }

    /// A kernel whose code uses a required constant's value stops at it undefined, with the
    /// reason; every planned set supplies it, and the set of ones translates.
    #[test]
    fn a_required_constant_is_supplied_by_every_planned_set() {
        let Some(bitcode) = bitcode(fixtures_dir().join("kernel_function_constant_gate_fold.ll"))
        else {
            return;
        };
        let undefined = attempt(
            &bitcode,
            "fcfold",
            &FunctionConstantValues::defaults(),
            "vk12-floor",
        );
        assert_eq!(
            undefined.attempt.outcome.key(),
            "function_constant",
            "{undefined:?}"
        );
        assert!(undefined
            .unsupplied
            .as_deref()
            .is_some_and(|why| why.contains("constant 0 is undefined in this set")));
        let row = row_plan(&bitcode, "fcfold");
        assert_eq!(row.facts.declared, 1);
        assert_eq!(row.facts.optional, 0);
        assert!(row.refused.is_none());
        // One required constant: 0 and 1, each also its defined and covering set.
        assert_eq!(row.specializations.len(), 2, "{row:#?}");
        for set in &row.specializations {
            let result = attempt(&bitcode, "fcfold", &set.values(), "vk12-floor");
            assert_ne!(
                result.attempt.outcome.key(),
                "function_constant",
                "{result:?}"
            );
            if set.kinds.contains(&SpecKind::One) {
                assert_eq!(result.attempt.outcome, Outcome::Translated, "{result:?}");
            }
        }
    }

    /// The planner reads the declarations of a module with several constants: each optional
    /// one is named by an `air.is_function_constant_defined` call, and an entry without the fifth
    /// operand counts as absent.
    #[test]
    fn the_planner_reads_optional_constants_from_the_defined_queries() {
        let Some(bitcode) = bitcode(engine_dir().join("function_constant_interface.ll")) else {
            return;
        };
        let mut meter = Meter::new(&Budget::standard());
        let decoded = metal2vulkan_air::decode(&bitcode, &mut meter).unwrap();
        let constants = crate::specialization::declared_constants(&decoded).unwrap();
        assert_eq!(constants.len(), 6);
        assert!(constants.iter().any(|c| c.optional), "{constants:?}");
        assert!(constants.iter().all(|c| c.flag.is_none()));
        assert!(constants
            .iter()
            .all(|c| c.data_type == metal2vulkan::metal::DataType::UChar));
        let sets = plan(&constants, &[]);
        assert!(!sets.is_empty() && sets.len() <= crate::specialization::MAX_SPECIALIZATIONS);
    }

    /// A kernel reading stage inputs is supplied the descriptor its interface implies.
    #[test]
    fn a_stage_input_kernel_is_supplied_its_implied_descriptor() {
        let Some(bitcode) = bitcode(engine_dir().join("kernel_stage_in.ll")) else {
            return;
        };
        let result = attempt(
            &bitcode,
            "stage_in_fetch",
            &FunctionConstantValues::defaults(),
            "moltenvk-m3",
        );
        assert_eq!(result.attempt.outcome, Outcome::Translated, "{result:?}");
        assert_eq!(result.supplied, vec![Supplied::StageInputDescriptor]);
    }

    /// A tile function is supplied each implicit imageblock member's format.
    #[test]
    fn a_tile_function_is_supplied_its_members_formats() {
        let Some(bitcode) = bitcode(engine_dir().join("k10ts_tile_order.ll")) else {
            return;
        };
        let result = attempt(
            &bitcode,
            "k10ts_tile_order",
            &FunctionConstantValues::defaults(),
            "moltenvk-m3",
        );
        assert_eq!(
            result.supplied,
            vec![Supplied::TileAttachments {
                formats: vec![(0, PixelFormat::R32Float.name().to_string())]
            }],
            "{result:?}"
        );
        assert_eq!(result.attempt.outcome, Outcome::Translated, "{result:?}");
    }

    /// Every derived input crosses the worker boundary and the index as JSON unchanged.
    #[test]
    fn every_supplied_input_survives_json() {
        let attempt = SpecAttempt {
            attempt: Attempt::uncosted(Outcome::Translated, None),
            supplied: vec![
                Supplied::StageInputDescriptor,
                Supplied::VertexDescriptor,
                Supplied::TileAttachments {
                    formats: vec![(0, "R32Float".into()), (3, "RGBA16Float".into())],
                },
                Supplied::StorageTextureFormat {
                    slot: 2,
                    format: "RGBA32Float".into(),
                },
                Supplied::LinkedFunction {
                    symbol: "f".into(),
                    source: LinkSource::Library {
                        module: "ab".repeat(32),
                    },
                },
                Supplied::LinkedFunction {
                    symbol: "h".into(),
                    source: LinkSource::Authored {
                        module: "cd".repeat(32),
                    },
                },
                Supplied::LinkedFunction {
                    symbol: "g".into(),
                    source: LinkSource::Row,
                },
            ],
            unsupplied: Some("why".into()),
            ray_tracing: true,
        };
        let json = serde_json::to_string(&attempt).unwrap();
        assert_eq!(serde_json::from_str::<SpecAttempt>(&json).unwrap(), attempt);
        let json = serde_json::to_string(&attempt.supplied).unwrap();
        assert_eq!(
            serde_json::from_str::<Vec<Supplied>>(&json).unwrap(),
            attempt.supplied
        );
    }

    /// A post-tessellation vertex function is supplied the L4 lane's vertex descriptor.
    #[test]
    fn a_post_tessellation_vertex_function_is_supplied_its_descriptor() {
        let Some(bitcode) = bitcode(engine_dir().join("k7_tess_quad_domain.ll")) else {
            return;
        };
        let result = attempt(
            &bitcode,
            "k7_tess_quad_domain",
            &FunctionConstantValues::defaults(),
            "moltenvk-m3",
        );
        assert_eq!(
            result.supplied,
            vec![Supplied::VertexDescriptor],
            "{result:?}"
        );
        assert_eq!(result.attempt.outcome, Outcome::Translated, "{result:?}");
    }

    /// A storage texture on a profile without formatless storage access is supplied the widest
    /// format of its texel class.
    #[test]
    fn a_storage_texture_is_supplied_the_widest_format_of_its_class() {
        let Some(bitcode) =
            bitcode(fixtures_dir().join("kernel_write_one_dimensional_textures.ll"))
        else {
            return;
        };
        let result = attempt(
            &bitcode,
            "kernel_write_one_dimensional_textures",
            &FunctionConstantValues::defaults(),
            "roadmap-2022",
        );
        assert!(
            result
                .supplied
                .iter()
                .any(|s| matches!(s, Supplied::StorageTextureFormat { .. })),
            "{result:?}"
        );
        assert_ne!(
            result.attempt.outcome.key(),
            "StorageTextureFormat",
            "{result:?}"
        );
    }

    /// Without an index a linked function stays the compile's input and says why; a
    /// visible-function table is no input at all, and the attempt translates.
    #[test]
    fn a_linked_function_says_why_it_stays_an_input() {
        let Some(bitcode) = bitcode(engine_dir().join("visible_table_caller.ll")) else {
            return;
        };
        let constants = FunctionConstantValues::defaults();
        let linked = attempt(
            &bitcode,
            "visible_pointer_dispatch",
            &constants,
            "moltenvk-m3",
        );
        assert_eq!(linked.attempt.outcome.key(), "LinkedFunction", "{linked:?}");
        assert!(linked
            .unsupplied
            .as_deref()
            .is_some_and(|why| why.contains("no index")));
        let table = attempt(
            &bitcode,
            "visible_table_dispatch",
            &constants,
            "moltenvk-m3",
        );
        assert_eq!(table.attempt.outcome, Outcome::Translated, "{table:?}");
        assert_eq!(table.unsupplied, None, "{table:?}");
    }

    /// A linked function is supplied the one retained module of the row's libraries that defines
    /// it, and a reference back into the row its own module; an index the audit cannot read makes
    /// the attempt a transient tooling failure with no recorded reason.
    #[test]
    fn a_linked_function_is_supplied_its_one_defining_module() {
        let caller_path = engine_dir().join("linked_visible_caller.ll");
        let functions_path = engine_dir().join("linked_visible_functions.ll");
        let (Some(caller), Some(functions)) =
            (bitcode(caller_path), bitcode(functions_path.clone()))
        else {
            return;
        };
        let scratch = crate::ScratchDir::new("specialization-linked").unwrap();
        let functions_ll = std::fs::read_to_string(&functions_path).unwrap();
        let module_sha256 = crate::hash::sha256_bytes(functions_ll.as_bytes());
        let library = "11".repeat(32);
        crate::library_module::merge_library_module_shards(
            scratch.path(),
            [crate::library_module::LibraryModuleRow {
                module_sha256: module_sha256.clone(),
                air_ll: functions_ll,
                blob_b64: base64::Engine::encode(
                    &base64::engine::general_purpose::STANDARD,
                    &functions,
                ),
                lib_sha256s: vec![library.clone()],
                label: "local/linked_visible_functions.ll".into(),
            }],
        )
        .unwrap();
        let index = scratch.path().join("index.sqlite");
        crate::library_module::sync_library_module_index(scratch.path(), &index).unwrap();
        let profile = load_profile("moltenvk-m3").unwrap();
        let libraries = [library];
        let none = AuthoredLinks::default();
        let run = |index: &Path| {
            replay_specialized(
                &caller,
                "linked_visible_caller",
                &FunctionConstantValues::defaults(),
                &profile,
                Some(Linkage {
                    corpus_root: scratch.path(),
                    index,
                    parent_libraries: &libraries,
                    authored: &none,
                }),
                None,
            )
            .attempt
        };
        let linked = run(&index);
        assert!(
            linked.supplied.contains(&Supplied::LinkedFunction {
                symbol: "linked_scale".into(),
                source: LinkSource::Library {
                    module: module_sha256
                },
            }),
            "{linked:?}"
        );
        assert_eq!(linked.attempt.outcome, Outcome::Translated, "{linked:?}");
        // The attempt's cost is its last compile's: the allocator window is that compile's meter
        // scope, never the derivations between compiles (the floor holds).
        let cost = linked.attempt.cost.as_ref().unwrap();
        if let Some(allocator_peak) = cost.allocator_peak {
            assert!(
                crate::budgets::ledger_covers(cost.ledger_peak, allocator_peak),
                "ledger {} against allocator {allocator_peak}",
                cost.ledger_peak
            );
        }
        let unreadable = scratch.path().join("no-library-tables.sqlite");
        rusqlite::Connection::open(&unreadable).unwrap();
        let tooling = run(&unreadable);
        assert_eq!(tooling.attempt.outcome.key(), "tooling", "{tooling:?}");
        assert!(tooling.attempt.outcome.is_transient());
        assert_eq!(tooling.unsupplied, None);
    }

    /// The table fixture's functions module, retained in a scratch corpus under a library that is
    /// none of the caller's parents, and its hash.
    fn retained_table_functions(scratch: &crate::ScratchDir) -> Option<(PathBuf, String)> {
        let path = engine_dir().join("visible_table_functions.ll");
        let functions = bitcode(path.clone())?;
        let ll = std::fs::read_to_string(&path).unwrap();
        let module_sha256 = crate::hash::sha256_bytes(ll.as_bytes());
        crate::library_module::merge_library_module_shards(
            scratch.path(),
            [crate::library_module::LibraryModuleRow {
                module_sha256: module_sha256.clone(),
                air_ll: ll,
                blob_b64: base64::Engine::encode(
                    &base64::engine::general_purpose::STANDARD,
                    &functions,
                ),
                lib_sha256s: vec!["22".repeat(32)],
                label: "local/visible_table_functions.ll".into(),
            }],
        )
        .unwrap();
        let index = scratch.path().join("index.sqlite");
        crate::library_module::sync_library_module_index(scratch.path(), &index).unwrap();
        Some((index, module_sha256))
    }

    /// A reference the authored cases link is linked from their module, which no parent library
    /// of the row retains; a table call asks for nothing, because with no handle to hold its
    /// entries are null and its switch is unreachable.
    #[test]
    fn authored_functions_answer_references_and_tables() {
        let Some(caller) = bitcode(engine_dir().join("visible_table_caller.ll")) else {
            return;
        };
        let scratch = crate::ScratchDir::new("specialization-authored").unwrap();
        let Some((index, module)) = retained_table_functions(&scratch) else {
            return;
        };
        let names = ["table_add", "table_half", "table_mul", "table_sub"];
        let authored = AuthoredLinks {
            functions: names
                .iter()
                .map(|name| (name.to_string(), module.clone()))
                .collect(),
            conflicts: BTreeSet::new(),
        };
        let profile = load_profile("moltenvk-m3").unwrap();
        let libraries = ["11".repeat(32)];
        let run = |entry: &str, authored: &AuthoredLinks| {
            replay_specialized(
                &caller,
                entry,
                &FunctionConstantValues::defaults(),
                &profile,
                Some(Linkage {
                    corpus_root: scratch.path(),
                    index: &index,
                    parent_libraries: &libraries,
                    authored,
                }),
                None,
            )
            .attempt
        };
        let pointer = run("visible_pointer_dispatch", &authored);
        assert_eq!(pointer.attempt.outcome, Outcome::Translated, "{pointer:?}");
        assert!(
            pointer.supplied.contains(&Supplied::LinkedFunction {
                symbol: "table_add".into(),
                source: LinkSource::Authored {
                    module: module.clone()
                },
            }),
            "{pointer:?}"
        );
        let table = run("visible_table_dispatch", &authored);
        assert_eq!(table.attempt.outcome, Outcome::Translated, "{table:?}");
        assert_eq!(table.supplied, Vec::new(), "{table:?}");
        // Only a function of another type than the table's calls: still no input, and the call
        // takes the entries' null words.
        let half = AuthoredLinks {
            functions: BTreeMap::from([("table_half".to_string(), module.clone())]),
            conflicts: BTreeSet::new(),
        };
        let untyped = run("visible_table_dispatch", &half);
        assert_eq!(untyped.attempt.outcome, Outcome::Translated, "{untyped:?}");
        assert_eq!(untyped.supplied, Vec::new(), "{untyped:?}");
        // A name the cases link from two modules is linked by none of them.
        let conflicted = AuthoredLinks {
            functions: BTreeMap::new(),
            conflicts: BTreeSet::from(["table_add".to_string()]),
        };
        let refused = run("visible_pointer_dispatch", &conflicted);
        assert!(
            refused
                .unsupplied
                .as_deref()
                .is_some_and(|why| why.contains("more than one module")),
            "{refused:?}"
        );
    }

    #[test]
    fn tile_formats_follow_the_member_type() {
        assert_eq!(tile_format("half4"), Some(PixelFormat::RGBA16Float));
        assert_eq!(tile_format("float"), Some(PixelFormat::R32Float));
        assert_eq!(tile_format("uint2"), Some(PixelFormat::RG32Uint));
        assert_eq!(
            tile_format("rgba8unorm<half4>"),
            Some(PixelFormat::RGBA8Unorm)
        );
        assert_eq!(tile_format("half3"), None);
        assert_eq!(tile_format("bool"), None);
    }

    #[test]
    fn ray_tracing_refusals_are_typed() {
        use metal2vulkan::device::VkFeature;
        use metal2vulkan::error::{Capability, Feature, LowerFamily, Needs, Unsupported};
        let refused = |feature| Error::from(Unsupported::not_implemented(feature));
        assert!(is_ray_tracing(&refused(Feature::RayTracing)));
        assert!(is_ray_tracing(&refused(Feature::IntersectionFunctionTable)));
        assert!(is_ray_tracing(&refused(Feature::Lowering(
            LowerFamily::RayTracing
        ))));
        assert!(!is_ray_tracing(&refused(Feature::Linking)));
        let mut needs = Unsupported::not_implemented(Feature::DynamicCall);
        needs.needs = Needs::AnyOf(vec![vec![Capability::Feature(VkFeature::RayQuery)]]);
        assert!(is_ray_tracing(&Error::from(needs)));
        assert!(!is_ray_tracing(&Error::NeedsInput(
            RuntimeInput::TileAttachments
        )));
    }
}
