pub mod buffer_slots;
pub mod depth_range;
pub mod depth_stencil;
pub mod mesh;
pub mod min_lod_clamp;
pub mod multisample;
pub mod out_of_bounds;
pub mod rate_map;
pub mod render_target_array;
pub mod sample_rate;
pub mod sparse;
pub mod stage_inputs;
pub mod tessellation;

pub use depth_stencil::{depth_stencil_test, DepthStencilTest, DrawDepthStencil};

use crate::air::reflect::{
    ResourceKind, ShaderInterface, ShaderStage, TextureDimension,
    TextureFormat as ReflectedTextureFormat,
};
use crate::case::{
    AuthoredCase, DeviceBufferArrayResource, OutputSelection, Primitive, Stage, TextureFormat,
    ThreadgroupMemoryResource,
};
use crate::requirement::{Executor, Limit, ToolingRequirement, UnobservedEffect};
use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet, HashSet};

pub fn air_call_counts(ll: &str) -> BTreeMap<String, usize> {
    crate::air::calls::air_call_counts(ll)
}

/// The census's intrinsic contract: the calls of `calls` whose `air.*` name the engine's
/// intrinsic table does not know ([`air_intrinsic_is_known`]), with their counts. A row with one is
/// unresolved: no executor contract describes the call.
pub fn unrecognized_air_intrinsics(calls: &BTreeMap<String, usize>) -> BTreeMap<String, usize> {
    calls
        .iter()
        .filter(|(name, _)| !air_intrinsic_is_known(name))
        .map(|(name, count)| (name.clone(), *count))
        .collect()
}

/// Whether the engine's intrinsic table (`metal2vulkan_air::intrinsics`, the product's decode
/// of `air.*` calls) knows the `air.*` name `name`: its name grammar decodes it into a family, or
/// the table lists it as a known family it does not decode (`UNDECODED`) or as a typed feature not
/// implemented yet (`undecoded_feature`). A name of neither kind stays `Callee::UnknownIntrinsic`
/// at decode and is refused as `UnknownIntrinsic` if it survives specialization.
///
/// The table classifies a declaration by its name and its function type
/// (`air::intrinsics::classify`); the census has only the call's name. A name outside the decoded
/// families is `DeclClass::Unknown` whatever the type, and a decoded family's name is
/// `DeclClass::Air` or, against a type that disagrees with its signature, `DeclClass::Mismatch`.
/// So the name is classified against the probe type `void ()`: anything but `Unknown` is a
/// decoded family. Which declared type a corpus row uses is the decoder's question (a mismatch is
/// malformed input, refused at decode), as is an immediate outside a family's enumerators, which
/// keeps one call unknown (`air::intrinsics::bind`); neither is a question about the name.
pub fn air_intrinsic_is_known(name: &str) -> bool {
    use metal2vulkan::Budget;
    use metal2vulkan_air::budget::Meter;
    use metal2vulkan_air::intrinsics::{classify, undecoded_family, undecoded_feature, DeclClass};
    use metal2vulkan_air::ir::{Symbols, Type, TypeTable};
    if !name.starts_with("air.") {
        return false;
    }
    if undecoded_family(name).is_some() || undecoded_feature(name.as_bytes()).is_some() {
        return true;
    }
    let mut meter = Meter::new(&Budget::default());
    let mut types = TypeTable::new();
    let mut symbols = Symbols::new();
    let probe = types.intern(Type::Void, &mut meter).and_then(|void| {
        let probe = types.intern(
            Type::Function {
                ret: void,
                params: Box::new([]),
                vararg: false,
            },
            &mut meter,
        )?;
        Ok((probe, symbols.intern(name.as_bytes(), &mut meter)?))
    });
    // A name the default budget cannot hold is not known: the census keeps its row unresolved.
    probe.is_ok_and(|(probe, symbol)| {
        !matches!(
            classify(name.as_bytes(), symbol, probe, &types),
            None | Some(DeclClass::Unknown)
        )
    })
}

/// Descriptive command-buffer use, independent of whether executors currently support it: the
/// module declares an `air.command_buffer` handle or calls a command-encoder operation.
pub fn uses_indirect_commands(source_ll: &str, calls: &BTreeMap<String, usize>) -> bool {
    encodes_indirect_commands(calls) || source_ll.contains("!\"air.command_buffer\"")
}

/// The module writes command state: it calls at least one command-encoder operation. A module that
/// only declares a command-buffer handle leaves every command as allocated, so the buffer is an
/// inert resource both executors supply.
pub fn encodes_indirect_commands(calls: &BTreeMap<String, usize>) -> bool {
    calls
        .keys()
        .any(|name| crate::air::calls::CommandEncoderOperation::from_symbol(name).is_some())
}

/// Each call whose effect both executors run exactly but no observation captures, with the typed
/// effect it belongs to. Persisted beside the requirements and left out of symbol coverage, so no
/// report counts these calls' semantics as verified. (A case that executes the commands observes
/// them; this fact of the source alone does not know a case, so it stays the narrower claim.)
pub fn unobserved_calls(calls: &BTreeMap<String, usize>) -> BTreeMap<String, UnobservedEffect> {
    calls
        .keys()
        .filter(|name| crate::air::calls::CommandEncoderOperation::from_symbol(name).is_some())
        .map(|name| (name.clone(), UnobservedEffect::IndirectCommandContents))
        .collect()
}

/// Whether both executors can run `operation` as a producer. The Metal oracle guards each native
/// setter, and the engine's lowering writes the record; both check the same `CommandDescriptor`
/// policy before writing and fail a run whose encoding faulted, and both execute what was written:
/// an indexed draw as a draw of primitives is, over the index buffer its record names, and a patch
/// draw as the encoder's patch draw is, over the patch index and factor buffers its record names,
/// its factors scaled as Metal's, an indexed patch draw as a patch draw is, over the control-point
/// index buffer its record names too, a mesh draw as the encoder's mesh draw of its grid form is,
/// under the mesh render pipeline it runs, with the object and mesh buffers its record names (a
/// render descriptor counts them), a reset as the empty command it leaves, which runs nothing, a
/// stage-in region or an imageblock size as the command's dispatch takes the encoder's, and a copy
/// as the command it copies; a copy whose result is Metal's private record layout faults on both
/// sides. A buffer setter, an indexed draw, a patch draw or an indexed patch draw with an
/// unimplemented pointer encoding has no producer lowering on either side; and an operation neither
/// models (a render state setter, an object threadgroup's memory length, among others) has no case
/// schema or guard on either side.
fn encoder_operation_is_executable(operation: crate::air::calls::CommandEncoderOperation) -> bool {
    use crate::air::calls::CommandEncoderOperation as Operation;
    match operation {
        Operation::UnsupportedBufferEncoding | Operation::Unmodeled => false,
        Operation::Buffer { .. }
        | Operation::Dispatch { .. }
        | Operation::Pipeline { .. }
        | Operation::Barrier { .. }
        | Operation::ThreadgroupMemory
        | Operation::Draw
        | Operation::DrawIndexed { .. }
        | Operation::DrawPatches { .. }
        | Operation::DrawIndexedPatches { .. }
        | Operation::DrawMesh { .. }
        | Operation::Reset { .. }
        | Operation::StageInRegion
        | Operation::ImageblockSize
        | Operation::Copy => true,
    }
}

/// The requirements a module's `air.*` calls state by their names alone: an implicit imageblock
/// texture whose format no literal names, and an indirect command no executor encodes.
pub fn unsupported_air_requirements(
    calls: &BTreeMap<String, usize>,
) -> BTreeSet<ToolingRequirement> {
    let mut requirements = BTreeSet::new();
    if calls
        .keys()
        .any(|name| crate::air::meta::implicit_imageblock_texture_format(name).is_err())
    {
        requirements.insert(ToolingRequirement::ImplicitImageblockLiteral);
    }
    if calls.keys().any(|name| {
        crate::air::calls::CommandEncoderOperation::from_symbol(name)
            .is_some_and(|operation| !encoder_operation_is_executable(operation))
    }) {
        requirements.insert(ToolingRequirement::IndirectCommandBuffer);
    }
    requirements
}

/// The module's `air.*` texture operations on a Metal `ulong` texture: those the product's
/// intrinsic table decodes with a 64-bit unsigned texel (`air::intrinsics::Texture::ulong`). A
/// `ulong` texture that is only queried spells no texel, so no call names it. Descriptive: every
/// executor runs the family, and the census selects it by the `ulong_texture` fact.
pub fn ulong_texture_calls(calls: &BTreeMap<String, usize>) -> impl Iterator<Item = &str> {
    use metal2vulkan_air::intrinsics::AirIntrinsic;
    calls.keys().map(String::as_str).filter(|name| {
        matches!(AirIntrinsic::parse(name), Some(AirIntrinsic::Texture(texture)) if texture.ulong())
    })
}

/// Whether the module calls a texture operation on a Metal `ulong` texture
/// ([`ulong_texture_calls`]).
pub fn calls_ulong_texture(calls: &BTreeMap<String, usize>) -> bool {
    ulong_texture_calls(calls).next().is_some()
}

/// How a module whose only effect is encoding indirect commands can be observed: it encodes, and
/// has no other effect a case can observe ([`observable_beyond_indirect_commands`]), so its only
/// evidence is replay.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CommandsOnly {
    /// Replaying the commands observes them (`replay_observes_commands`): a case of the module must
    /// execute them ([`require_reflection`]).
    Replayed,
    /// Nothing observes them: [`ToolingRequirement::IndirectCommandBuffer`].
    Unobserved,
}

/// Whether the module's only effect is the indirect commands it encodes, and if so how they can be
/// observed ([`CommandsOnly`]); `None` for a module that encodes none or has another observable
/// effect.
///
/// `reflection` is `None` when the product does not reflect the entry (inspection failed, or the
/// stage has no reflected interface). Stage outputs then count as absent, and so does the kernel
/// stage replay needs, so a module is never resolved because it could not be inspected; a kernel's
/// observable effects are all memory writes, which need no reflection.
pub fn commands_only(
    source_ll: &str,
    calls: &BTreeMap<String, usize>,
    reflection: Option<&ShaderInterface>,
) -> Option<CommandsOnly> {
    if !encodes_indirect_commands(calls)
        || observable_beyond_indirect_commands(source_ll, reflection)
    {
        return None;
    }
    Some(if replay_observes_commands(calls, reflection) {
        CommandsOnly::Replayed
    } else {
        CommandsOnly::Unobserved
    })
}

/// The requirement of a module whose only effect is encoding indirect commands that no replay
/// observes ([`CommandsOnly::Unobserved`]): [`ToolingRequirement::IndirectCommandBuffer`].
pub fn indirect_command_requirement(
    source_ll: &str,
    calls: &BTreeMap<String, usize>,
    reflection: Option<&ShaderInterface>,
) -> Option<ToolingRequirement> {
    (commands_only(source_ll, calls, reflection) == Some(CommandsOnly::Unobserved))
        .then_some(ToolingRequirement::IndirectCommandBuffer)
}

/// Whether replaying the commands a module encodes observes them. Both executors execute a case's
/// compute commands and its render commands after its producer: a kernel's dispatch or a vertex or
/// fragment function's draw, never a tile function's (a render pass's tile dispatch, whose commands
/// neither models). So the stage is one of those, every operation it encodes is a compute or a
/// render command's, and one of them selects what a replayed command writes: a kernel buffer, a
/// dispatch, or a compute pipeline for a replayed dispatch; a vertex buffer, a draw, or a render
/// pipeline for a replayed draw, whose template is a rasterization-disabled vertex function, an
/// indexed draw, whose indices are the vertex IDs that template writes by, or a patch draw or an
/// indexed patch draw, whose patches, instances and factors are the domain points a
/// rasterization-disabled post-tessellation template writes by; an object or mesh buffer or a mesh
/// draw, whose grid and threadgroups are the positions a rasterization-disabled mesh template
/// writes by, for a replayed mesh draw; a reset, which leaves a command the host encoded running
/// nothing; and a copy, which makes a command the one it copies. A barrier, a threadgroup length,
/// a stage-in region, an imageblock size, or a fragment buffer alone changes no value a replayed
/// command writes unless its template reads it, which the module does not decide.
fn replay_observes_commands(
    calls: &BTreeMap<String, usize>,
    reflection: Option<&ShaderInterface>,
) -> bool {
    use crate::air::calls::{BufferStage, CommandEncoderOperation as Operation};
    let operations = calls
        .keys()
        .filter_map(|name| Operation::from_symbol(name))
        .collect::<Vec<_>>();
    let selects = |operation: &Operation| {
        matches!(
            operation,
            Operation::Buffer {
                stage: BufferStage::Kernel
                    | BufferStage::Vertex
                    | BufferStage::Object
                    | BufferStage::Mesh,
                ..
            } | Operation::Dispatch { .. }
                | Operation::Pipeline { .. }
                | Operation::Draw
                | Operation::DrawIndexed { .. }
                | Operation::DrawPatches { .. }
                | Operation::DrawIndexedPatches { .. }
                | Operation::DrawMesh { .. }
                | Operation::Reset { .. }
                | Operation::Copy
        )
    };
    let carried = |operation: &Operation| {
        matches!(
            operation,
            Operation::Buffer {
                stage: BufferStage::Fragment,
                ..
            } | Operation::Barrier { .. }
                | Operation::ThreadgroupMemory
                | Operation::StageInRegion
                | Operation::ImageblockSize
        )
    };
    let producer = reflection.is_some_and(|reflection| match reflection.stage {
        ShaderStage::Kernel => reflection.implicit_imageblock_attachments.is_empty(),
        ShaderStage::Vertex | ShaderStage::TessellationEvaluation | ShaderStage::Fragment => true,
        // No executor encodes or replays an object or mesh function's commands.
        ShaderStage::Object | ShaderStage::Mesh => false,
    });
    producer
        && operations
            .iter()
            .all(|operation| selects(operation) || carried(operation))
        && operations.iter().any(selects)
}

/// Whether the module has an effect a case can observe besides the commands it encodes: a write
/// of a comparable value into observable memory (`writes_observable_memory`) or a stage output the
/// executors capture (a vertex or mesh vertex position, which the rasterized attachments record,
/// or a fragment color, depth, or stencil output). Without a reflection, stage outputs count as
/// absent.
///
/// A module that encodes commands and has neither has only [`UnobservedEffect`]s, so no case can
/// observe a value that depends on it: that keeps [`ToolingRequirement::IndirectCommandBuffer`].
pub fn observable_beyond_indirect_commands(
    source_ll: &str,
    reflection: Option<&ShaderInterface>,
) -> bool {
    let stage_output = reflection.is_some_and(|reflection| match reflection.stage {
        ShaderStage::Vertex | ShaderStage::TessellationEvaluation => reflection
            .vertex_builtins
            .is_some_and(|builtins| builtins.writes_position),
        ShaderStage::Fragment => {
            !reflection.render_targets.is_empty()
                || !reflection.depth_members.is_empty()
                || !reflection.stencil_members.is_empty()
        }
        ShaderStage::Mesh => reflection
            .mesh_stage
            .as_ref()
            .is_some_and(crate::air::reflect::MeshStageInterface::writes_position),
        ShaderStage::Kernel | ShaderStage::Object => false,
    });
    stage_output || writes_observable_memory(source_ll)
}

/// Whether any instruction writes a comparable value into memory a case can select as output,
/// other than through a command-encoder operation: a `store`, `atomicrmw`, or `cmpxchg` into device
/// memory (AIR address space 1), a device `llvm.mem{cpy,move,set}`, a device atomic other than a
/// load, or a texture write or atomic other than a load.
///
/// A device address is not comparable across the two APIs, so a written value that is a pointer, or
/// that derives from `ptrtoint` within its function (`address_valued_names`), does not count.
///
/// The scan is structural and errs both ways. It misses a write it does not list, which keeps an
/// encoding row's requirement. It also counts a listed write that no case may be able to observe:
/// an integer loaded from memory that holds an address, `llvm.memcpy` bytes that carry one, a write
/// on a path never taken. For a row it closes, the evidence is its authored case, whose selected
/// output the checker requires the shader to write.
fn writes_observable_memory(source_ll: &str) -> bool {
    function_bodies(source_ll).iter().any(|body| {
        let addresses = address_valued_names(body);
        body.iter()
            .filter_map(|line| instruction(line))
            .any(|(_, instruction)| writes_comparable_value(instruction, &addresses))
    })
}

fn writes_comparable_value(instruction: &str, addresses: &HashSet<&str>) -> bool {
    let carries_address = |text: &str| {
        text.contains("ptrtoint") || ssa_names(text).any(|name| addresses.contains(name))
    };
    // A pointer-typed value: `ptr`, a vector of `ptr`, or a typed pointer (`T addrspace(n)*`).
    let is_address = |operand: &str| {
        let ty = operand.split(['%', '@']).next().unwrap_or_default();
        ty.contains("ptr") || ty.contains('*') || carries_address(operand)
    };
    if let Some(rest) = instruction.strip_prefix("store ") {
        let rest = rest.strip_prefix("volatile ").unwrap_or(rest);
        let operands = split_top_level(rest);
        return operands
            .get(1)
            .and_then(|pointer| pointer.rsplit_once(' '))
            .is_some_and(|(ty, _)| device_pointer_type(ty.trim()))
            && operands.first().is_some_and(|value| !is_address(value));
    }
    if instruction.starts_with("atomicrmw ") || instruction.starts_with("cmpxchg ") {
        let operands = split_top_level(instruction);
        return operands
            .first()
            .is_some_and(|pointer| device_pointer_operand(pointer))
            && operands.iter().skip(1).all(|value| !is_address(value));
    }
    if !instruction.contains("call ") || carries_address(instruction) {
        return false;
    }
    ["@llvm.memcpy.p1", "@llvm.memmove.p1", "@llvm.memset.p1"]
        .iter()
        .any(|callee| instruction.contains(callee))
        || crate::air::calls::air_call_counts(instruction)
            .keys()
            .any(|name| air_call_writes_observable_memory(name))
}

/// The instruction lines of each function body, from its `define` line to its closing `}`. SSA
/// names are function-local, so each body is analyzed on its own.
fn function_bodies(source_ll: &str) -> Vec<Vec<&str>> {
    let lines = source_ll.lines().collect::<Vec<_>>();
    let mut bodies = Vec::new();
    let mut first = None;
    for (index, line) in lines.iter().enumerate() {
        if line.starts_with("define ") {
            first = Some(index + 1);
        } else if line.starts_with('}') {
            if let Some(first) = first {
                bodies.push(lines[first..index].to_vec());
            }
            first = None;
        }
    }
    bodies
}

/// An instruction line's result name (if any) and its text after `%result = `, without comments
/// (a `;` outside a quoted name or string). Instructions are indented; declarations, metadata, and
/// labels are not.
fn instruction(line: &str) -> Option<(Option<&str>, &str)> {
    if !line.starts_with([' ', '\t']) {
        return None;
    }
    let mut quoted = false;
    let end = line
        .char_indices()
        .find(|&(_, ch)| {
            quoted ^= ch == '"';
            ch == ';' && !quoted
        })
        .map_or(line.len(), |(index, _)| index);
    let text = line[..end].trim();
    Some(match text.split_once(" = ") {
        Some((result, rest)) if result.starts_with('%') => (Some(result), rest.trim_start()),
        _ => (None, text),
    })
}

/// The local SSA names (`%name`, or `%"name"` with its quotes) a piece of instruction text
/// mentions, outside quoted names and strings.
fn ssa_names(text: &str) -> impl Iterator<Item = &str> {
    let mut quoted = false;
    let mut skip = 0;
    text.char_indices().filter_map(move |(start, ch)| {
        if start < skip {
            return None;
        }
        if ch == '"' {
            quoted = !quoted;
            return None;
        }
        if quoted || ch != '%' {
            return None;
        }
        let after = &text[start + 1..];
        let length = match after.strip_prefix('"') {
            Some(name) => name.find('"').map_or(after.len(), |end| end + 2),
            None => after
                .find(|ch: char| {
                    !(ch.is_ascii_alphanumeric() || matches!(ch, '.' | '_' | '$' | '-'))
                })
                .unwrap_or(after.len()),
        };
        skip = start + 1 + length;
        Some(&text[start..start + 1 + length])
    })
}

/// The values of one function body that carry a device address as an integer: `ptrtoint` results,
/// and every value computed from one (arithmetic, casts, selects, phis, calls), to a fixed point.
/// Loading through a pointer, or turning an integer back into one (`inttoptr`, `getelementptr`),
/// yields a value that is not itself an address, or a pointer whose writes are checked by type.
fn address_valued_names<'a>(body: &[&'a str]) -> HashSet<&'a str> {
    let mut addresses = HashSet::new();
    loop {
        let before = addresses.len();
        for (result, text) in body.iter().filter_map(|line| instruction(line)) {
            let Some(result) = result else { continue };
            if addresses.contains(result) {
                continue;
            }
            let derived = match text.split_whitespace().next() {
                Some("ptrtoint") => true,
                Some("load" | "inttoptr" | "getelementptr" | "alloca") => false,
                _ => {
                    text.contains("ptrtoint")
                        || ssa_names(text).any(|name| addresses.contains(name))
                }
            };
            if derived {
                addresses.insert(result);
            }
        }
        if addresses.len() == before {
            return addresses;
        }
    }
}

fn air_call_writes_observable_memory(name: &str) -> bool {
    if let Some(rest) = name.strip_prefix("air.atomic.global.") {
        return !rest.starts_with("load.");
    }
    name.starts_with("air.write_texture")
        || name.starts_with("air.write_imageblock_slice_to_texture")
        || (name.starts_with("air.atomic_")
            && name.contains("_texture")
            && !name.starts_with("air.atomic_load"))
}

fn device_pointer_type(ty: &str) -> bool {
    ty == "ptr addrspace(1)" || ty.ends_with(" addrspace(1)*")
}

/// The pointer operand of an `atomicrmw`/`cmpxchg`, which leads with its keywords and ordering.
fn device_pointer_operand(operand: &str) -> bool {
    operand.contains("ptr addrspace(1) ") || operand.contains(" addrspace(1)* ")
}

/// Split LLVM operands on commas outside `()`, `[]`, `{}`, `<>`, and quoted names and strings.
fn split_top_level(text: &str) -> Vec<&str> {
    let mut depth = 0usize;
    let mut quoted = false;
    let mut start = 0;
    let mut parts = Vec::new();
    for (index, ch) in text.char_indices() {
        match ch {
            '"' => quoted = !quoted,
            _ if quoted => {}
            '(' | '[' | '{' | '<' => depth += 1,
            ')' | ']' | '}' | '>' => depth = depth.saturating_sub(1),
            ',' if depth == 0 => {
                parts.push(text[start..index].trim());
                start = index + 1;
            }
            _ => {}
        }
    }
    parts.push(text[start..].trim());
    parts
}

/// The comparison of a draw's depth test.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DepthCompare {
    Always,
    Less,
    Greater,
}

/// Structural reflection requirements that the shared Metal/Vulkan execution contract cannot yet
/// run. Keeping this inventory beside [`require_case`] prevents triage, checking, and the candidate
/// descriptor path from silently describing different capability boundaries.
pub fn unsupported_reflection_requirements(
    reflection: &ShaderInterface,
) -> BTreeSet<ToolingRequirement> {
    let mut requirements = BTreeSet::new();
    if reflection.function_constants.iter().any(|constant| {
        crate::case::ScalarType::from_metal_abi_type_encoding(&constant.abi_type_encoding)
            .is_none_or(|(scalar, _)| !scalar.supports_metal_function_constant())
    }) {
        requirements.insert(ToolingRequirement::FunctionConstantLiteral);
    }
    if reflection.vertex_attributes.iter().any(|attribute| {
        attribute
            .type_name
            .as_deref()
            .and_then(crate::case::AttributeFormat::from_air_type_name)
            .is_none()
    }) {
        requirements.insert(ToolingRequirement::VertexAttributeLiteral);
    }
    // A control-point or patch attribute is authored in the format its AIR type names; one no
    // format names is not authorable (an 8-bit one is).
    if reflection.tessellation.iter().any(|interface| {
        interface
            .control_point_attributes
            .iter()
            .chain(&interface.patch_attributes)
            .any(|attribute| {
                attribute
                    .type_name
                    .as_deref()
                    .and_then(crate::case::AttributeFormat::from_air_type_name)
                    .is_none()
            })
    }) {
        requirements.insert(ToolingRequirement::TessellationAttributeLiteral);
    }
    if reflection.tessellation.as_ref().is_some_and(|interface| {
        [
            interface.instance_id.as_ref(),
            interface.amplification_id.as_ref(),
            interface.amplification_count.as_ref(),
        ]
        .into_iter()
        .flatten()
        .any(|attribute| {
            attribute
                .type_name
                .as_deref()
                .and_then(crate::case::AttributeFormat::from_air_type_name)
                .is_none_or(|format| !format.supports_tessellation_system_value())
        })
    }) {
        requirements.insert(ToolingRequirement::TessellationSystemInputLiteral);
    }
    // A fragment imageblock member is authored in the format its AIR type names; one no format
    // names is not authorable (a `float` one is, and a `uint` one).
    if reflection.fragment_imageblock.iter().any(|imageblock| {
        imageblock.members.iter().any(|member| {
            crate::case::FragmentImageblockFormat::from_air_type(&member.type_name, member.size)
                .is_none()
        })
    }) {
        requirements.insert(ToolingRequirement::FragmentImageblockMemberLiteral);
    }
    // A descriptor the reflection synthesized (a null texture, a read sampler) has no case
    // resource behind it and is no executor's gap: only the old engine's candidate bound those
    // descriptors, and it is retired.
    for binding in &reflection.bindings {
        if binding.kind == ResourceKind::KernelStageInput
            && binding
                .type_name
                .as_deref()
                .and_then(crate::case::AttributeFormat::from_air_type_name)
                .is_none()
        {
            requirements.insert(ToolingRequirement::KernelStageInputLiteral);
        }
        if let Some(shape) = binding.texture_shape {
            if shape.dimension == TextureDimension::Buffer
                && !matches!(
                    binding.kind,
                    ResourceKind::Texture | ResourceKind::StorageImage
                )
            {
                requirements.insert(ToolingRequirement::TextureBufferLiteral);
            }
            if shape.storage_format.is_some_and(|format| {
                !matches!(
                    format,
                    ReflectedTextureFormat::R8
                        | ReflectedTextureFormat::Rgba8
                        | ReflectedTextureFormat::R16f
                        | ReflectedTextureFormat::R16ui
                        | ReflectedTextureFormat::Rg16f
                        | ReflectedTextureFormat::Rg32f
                        | ReflectedTextureFormat::R32i
                        | ReflectedTextureFormat::R32f
                        | ReflectedTextureFormat::R32ui
                        | ReflectedTextureFormat::Rgba32i
                        | ReflectedTextureFormat::Rgba32ui
                        | ReflectedTextureFormat::Rgba32f
                        | ReflectedTextureFormat::Rgba16f
                        | ReflectedTextureFormat::Rgba8ui
                        | ReflectedTextureFormat::Rgba16ui
                        | ReflectedTextureFormat::Rgba8i
                        | ReflectedTextureFormat::Rgba16i
                )
            }) {
                requirements.insert(ToolingRequirement::StorageTextureFormatLiteral);
            }
        }
    }
    requirements
}

/// Source-aware executor requirements.
///
/// Indirect commands: both executors run a module that encodes commands, but the commands are an
/// [`UnobservedEffect`] until a case executes them. A module whose only effect is encoding
/// therefore offers a value a case can observe only through replay
/// ([`indirect_command_requirement`]).
///
/// The engine's candidate: the families of the reflected interface it does not run
/// ([`engine_source_requirements`]), and rasterization-rate map data the literal cannot place
/// ([`rate_map::rate_map_placement_requirement`]).
///
/// The Metal oracle: a compute kernel that samples with a `min_lod_clamp`, whose pipeline Apple's
/// compiler cannot build ([`min_lod_clamp::kernel_min_lod_clamp_requirement`]).
pub fn unsupported_source_requirements(
    source_ll: &str,
    reflection: &ShaderInterface,
) -> BTreeSet<ToolingRequirement> {
    let mut requirements = unsupported_reflection_requirements(reflection);
    requirements.extend(engine_source_requirements(reflection));
    requirements.extend(rate_map::rate_map_placement_requirement(
        source_ll, reflection,
    ));
    requirements.extend(min_lod_clamp::kernel_min_lod_clamp_requirement(
        source_ll, reflection,
    ));
    let calls = air_call_counts(source_ll);
    requirements.extend(unsupported_air_requirements(&calls));
    requirements.extend(indirect_command_requirement(
        source_ll,
        &calls,
        Some(reflection),
    ));
    requirements
}

/// The families of a reflected interface that every case of it authors and that the engine's
/// candidate does not run, each one the checker requires a case to author whenever the interface
/// has it:
///
/// - [`ToolingRequirement::RenderDeviceBufferArray`]: a vertex, fragment, object or mesh
///   function's buffer argument that is an array of device buffers (`array_ref<void>`, which the
///   checker requires authored as a device-buffer array);
/// - [`ToolingRequirement::RenderAccelerationStructure`]: a vertex or fragment function's
///   acceleration-structure or intersection-function-table argument (the checker requires every
///   bound one).
///
/// A post-tessellation vertex function is none of them: the candidate draws its patches as the
/// oracle does (`tessellation`); nor is a fragment function's color input, whose render target the
/// candidate attaches as the input attachment the plan reads, nor its explicit imageblock, whose
/// cells the candidate holds in the storage image the plan's imageblock write names, defined as the
/// oracle's initializer defines them; nor is an argument-buffer member holding a compute or render
/// pipeline state, which the candidate writes into its owner as an opaque id that no translated
/// pipeline reads; nor a kernel's implicit imageblock attachments, which the candidate runs as the
/// oracle's tile pipeline over the render area's tiles. The engine's other gaps are a case's
/// choice, not the source's ([`case_requirements`]): an argument-buffer intersection-function table
/// (the checker does not require one), a linked intersection function (the case's acceleration
/// structures are opaque triangles, which call none), and a vertex case's depth/stencil
/// attachment or a rasterization-disabled vertex case's render targets.
pub fn engine_source_requirements(reflection: &ShaderInterface) -> BTreeSet<ToolingRequirement> {
    let render = matches!(
        reflection.stage,
        ShaderStage::Vertex | ShaderStage::TessellationEvaluation | ShaderStage::Fragment
    );
    // An object or mesh function binds through its render encoder's setters too. Metal's encoder
    // has no object or mesh setter for an acceleration structure or a function table, so no case
    // authors one at those stages (`case::MeshDraw`).
    let drawn = render || matches!(reflection.stage, ShaderStage::Object | ShaderStage::Mesh);
    let device_buffer_array = reflection.bindings.iter().any(|binding| {
        binding.kind == ResourceKind::Buffer
            && binding
                .type_name
                .as_deref()
                .is_some_and(crate::air::meta::is_device_buffer_array_type_name)
    });
    let acceleration_structure = reflection.bindings.iter().any(|binding| {
        matches!(
            binding.kind,
            ResourceKind::PrimitiveAccelerationStructure
                | ResourceKind::AccelerationStructureShadow
                | ResourceKind::IntersectionFunctionTable
        )
    });
    [
        (
            drawn && device_buffer_array,
            ToolingRequirement::RenderDeviceBufferArray,
        ),
        (
            render && acceleration_structure,
            ToolingRequirement::RenderAccelerationStructure,
        ),
    ]
    .into_iter()
    .filter_map(|(present, requirement)| present.then_some(requirement))
    .collect()
}

/// The families of an authored case's own literals that an executor does not run: what the new
/// engine's candidate refuses before it maps a case. The source's families are
/// [`engine_source_requirements`]'s and an object function's threadgroup memory
/// ([`mesh::mesh_requirements`]); each of them a case authors is also named here, so a case that
/// was checked against an older reflection is refused all the same.
pub fn case_requirements(case: &AuthoredCase) -> BTreeSet<ToolingRequirement> {
    let render = matches!(case.stage, Stage::Vertex | Stage::Fragment);
    let drawn = render || case.stage.is_mesh_pipeline_stage();
    let linked = |entries: &[crate::case::IntersectionFunctionTableEntry]| {
        entries.iter().any(|entry| {
            matches!(
                entry,
                crate::case::IntersectionFunctionTableEntry::Linked { .. }
            )
        })
    };
    [
        (
            !case.argument_buffer_intersection_function_tables.is_empty(),
            ToolingRequirement::ArgumentBufferIntersectionFunctionTable,
        ),
        (
            case.intersection_function_tables
                .iter()
                .any(|table| linked(&table.entries))
                || case
                    .argument_buffer_intersection_function_tables
                    .iter()
                    .any(|table| linked(&table.entries)),
            ToolingRequirement::LinkedIntersectionFunction,
        ),
        (
            drawn && !case.device_buffer_arrays.is_empty(),
            ToolingRequirement::RenderDeviceBufferArray,
        ),
        (
            render
                && (!case.acceleration_structures.is_empty()
                    || !case.intersection_function_tables.is_empty()),
            ToolingRequirement::RenderAccelerationStructure,
        ),
        (
            case.stage == Stage::Vertex && case.depth_stencil.is_some(),
            ToolingRequirement::VertexDepthStencilAttachment,
        ),
        (
            case.is_rasterization_disabled_vertex() && !case.render_targets.is_empty(),
            ToolingRequirement::RasterizationDisabledRenderTarget,
        ),
    ]
    .into_iter()
    .filter_map(|(present, requirement)| present.then_some(requirement))
    .collect()
}

/// Every requirement of `case` against its source: what the source's structure raises
/// ([`unsupported_source_requirements`]), what its mesh draw needs ([`mesh::mesh_requirements`]),
/// what its stage's observation needs ([`unsupported_observation_requirements`]), and the families
/// of the case's own literals ([`case_requirements`]). The checker computes it once per checked
/// shader; each executor refuses the ones that stop it.
pub fn checked_requirements(
    case: &AuthoredCase,
    source_ll: &str,
    reflection: &ShaderInterface,
) -> BTreeSet<ToolingRequirement> {
    let mut requirements = unsupported_source_requirements(source_ll, reflection);
    requirements.extend(mesh::mesh_requirements(
        source_ll,
        reflection,
        case.mesh_draw.as_ref(),
    ));
    requirements.extend(unsupported_observation_requirements(case.stage, reflection));
    requirements.extend(case_requirements(case));
    requirements
}

/// Why the executor pair cannot evidence a case with `requirements`, or `None`: one that stops the
/// Metal oracle leaves no observation to compare with, and one that stops the candidate leaves
/// nothing to compare. A case both run is checkable. A requirement whose limit is Metal's own is
/// named as such: no tooling can give the oracle that case.
pub fn evidence_gap(requirements: &BTreeSet<ToolingRequirement>) -> Option<String> {
    fn joined(requirements: impl IntoIterator<Item = ToolingRequirement>) -> String {
        requirements
            .into_iter()
            .map(ToolingRequirement::as_str)
            .collect::<Vec<_>>()
            .join(",")
    }
    let names = |executor| {
        joined(ToolingRequirement::stopping(
            requirements.iter().copied(),
            executor,
        ))
    };
    let (metal_limit, tooling): (Vec<_>, Vec<_>) =
        ToolingRequirement::stopping(requirements.iter().copied(), Executor::MetalOracle)
            .into_iter()
            .partition(|requirement| requirement.limit() == Limit::Metal);
    let metal = [
        (!metal_limit.is_empty()).then(|| {
            format!(
                "{} (Metal cannot build the case's pipeline, so it has no reference)",
                joined(metal_limit)
            )
        }),
        (!tooling.is_empty())
            .then(|| format!("{} (the Metal oracle cannot run the case)", joined(tooling))),
    ]
    .into_iter()
    .flatten()
    .collect::<Vec<_>>();
    if !metal.is_empty() {
        return Some(metal.join("; "));
    }
    let candidate = Executor::EngineCandidate;
    let stopped = names(candidate);
    (!stopped.is_empty()).then(|| format!("{candidate}: {stopped} (the candidate is stopped)"))
}

/// The requirements of `requirements` that stop `executor`, as the error its admission returns, or
/// `Ok` when none does.
pub fn require_executor(
    requirements: &BTreeSet<ToolingRequirement>,
    executor: Executor,
) -> Result<(), BTreeSet<ToolingRequirement>> {
    let stopped = ToolingRequirement::stopping(requirements.iter().copied(), executor);
    if stopped.is_empty() {
        Ok(())
    } else {
        Err(stopped)
    }
}

/// Observation gaps shared by corpus classification and the authored graphics executors.
///
/// These are not shader-translation gaps: they mean the validation harness cannot yet construct
/// an observable experiment for an otherwise reflected interface.
pub fn unsupported_observation_requirements(
    stage: Stage,
    reflection: &ShaderInterface,
) -> BTreeSet<ToolingRequirement> {
    let mut requirements = BTreeSet::new();
    match stage {
        Stage::Vertex => {
            let writes_position = reflection
                .vertex_builtins
                .is_some_and(|builtins| builtins.writes_position);
            let has_shader_resource_output = reflection.bindings.iter().any(|binding| {
                matches!(
                    binding.kind,
                    ResourceKind::Buffer
                        | ResourceKind::StorageImage
                        | ResourceKind::EmbeddedArgBufferBuffer
                        | ResourceKind::EmbeddedArgBufferTexture
                ) && !matches!(
                    binding.access,
                    Some(crate::air::reflect::ResourceAccess::ReadOnly)
                )
            });
            if !writes_position && !has_shader_resource_output {
                requirements.insert(ToolingRequirement::VertexSideEffectObservation);
            }
        }
        Stage::Fragment => {
            for varying in &reflection.varyings {
                if varying
                    .type_name
                    .as_deref()
                    .and_then(crate::observation_contract::ObservationType::parse)
                    .is_none()
                {
                    requirements.insert(ToolingRequirement::FragmentVaryingObservationType);
                }
            }
            if reflection.render_targets.iter().any(|target| {
                target
                    .type_name
                    .as_deref()
                    .and_then(crate::observation_contract::ObservationType::parse)
                    .is_none()
            }) {
                requirements.insert(ToolingRequirement::FragmentOutputObservationType);
            }
        }
        // An object or mesh function's observation is its mesh draw's stages'
        // ([`mesh::mesh_requirements`]).
        Stage::Kernel | Stage::Object | Stage::Mesh => {}
    }
    requirements
}

/// The checker's executor-contract checks of `case` against its source, returning its
/// [`checked_requirements`] when some executor pair can evidence it ([`evidence_gap`]).
pub fn require_reflection(
    case: &AuthoredCase,
    source_ll: &str,
    reflection: &ShaderInterface,
    executor: &str,
) -> Result<BTreeSet<ToolingRequirement>, String> {
    let requirements = checked_requirements(case, source_ll, reflection);
    if let Some(gap) = evidence_gap(&requirements) {
        return Err(format!("{executor} cannot evidence the case: {gap}"));
    }
    // A module whose only effect is the commands it encodes is observed only by executing them.
    if commands_only(source_ll, &air_call_counts(source_ll), Some(reflection))
        == Some(CommandsOnly::Replayed)
        && case
            .command_buffers
            .iter()
            .all(|resource| resource.execute.is_none())
    {
        return Err(format!(
            "{executor} observes a module whose only effect is the commands it encodes only by \
             executing them, and the case executes none"
        ));
    }
    let authored_depth = case
        .depth_stencil
        .as_ref()
        .is_some_and(|attachment| attachment.initial_depth_b64.is_some());
    let authored_stencil = case
        .depth_stencil
        .as_ref()
        .is_some_and(|attachment| attachment.initial_stencil_b64.is_some());
    if !reflection.depth_members.is_empty() && !authored_depth {
        return Err(format!(
            "{executor} fragment depth output requires an authored depth attachment"
        ));
    }
    if !reflection.stencil_members.is_empty() && !authored_stencil {
        return Err(format!(
            "{executor} fragment stencil output requires an authored stencil attachment"
        ));
    }
    if reflection.fragment_imageblock.is_some() && case.fragment_imageblock.is_none() {
        return Err(format!(
            "{executor} custom fragment imageblock requires authored fragment_imageblock planes"
        ));
    }
    let color_inputs = reflection
        .bindings
        .iter()
        .filter(|binding| binding.kind == ResourceKind::ColorInput)
        .collect::<Vec<_>>();
    for binding in &color_inputs {
        if !case
            .render_targets
            .iter()
            .any(|target| target.index == binding.metal_index)
        {
            return Err(format!(
                "{executor} framebuffer-fetch input {} requires an authored render target at the same index",
                binding.metal_index
            ));
        }
    }
    // The generated companion's fullscreen triangle covers every pixel once per instance, so `n`
    // instances are `n` ordered read-modify-writes of each pixel.
    if !color_inputs.is_empty()
        && !case.draw.as_ref().is_some_and(|draw| {
            draw.primitive == Primitive::Triangle
                && draw.vertex_start == 0
                && draw.vertex_count == 3
                && draw.instance_count >= 1
        })
    {
        return Err(format!(
            "{executor} framebuffer-fetch execution requires the authored fullscreen triangle (triangle, vertex_start=0, vertex_count=3, instance_count>=1)"
        ));
    }
    Ok(requirements)
}

/// Metal's granularity for a threadgroup-memory allocation, in bytes.
///
/// `MTLComputeCommandEncoder`'s `setThreadgroupMemoryLength:atIndex:` asserts on anything else --
/// measured, not assumed: running an authored case that declares 4 bytes under
/// `METAL_DEVICE_WRAPPER_TYPE=1` fails with `-[MTLDebugComputeCommandEncoder
/// setThreadgroupMemoryLength:atIndex:]:799: failed assertion 'length(4) must be a multiple of 16
/// bytes.'`. Without the validation layer the driver accepts the call, so an authored length that
/// Metal cannot allocate executes and records evidence anyway.
///
/// The Metal executor passes the authored length straight to that API, so the length a manifest
/// declares IS the length Metal is asked for and has to be one Metal can allocate. Rounding it up
/// inside the executor instead would hand Metal a different size from the one the manifest states
/// and from the one the Vulkan candidate reads.
pub const METAL_THREADGROUP_MEMORY_GRANULARITY: u32 = 16;

/// Why a threadgroup-memory declaration is not one a Metal compute encoder can bind, or `None`.
///
/// One rule, consulted by [`require_case`] at both executors' entry and by the authored-case
/// checker before install, so an unallocatable length cannot reach the store from either side.
pub fn threadgroup_memory_error(
    resource: &ThreadgroupMemoryResource,
    context: &str,
) -> Option<String> {
    if resource.binding >= crate::air::reflect::THREADGROUP_BUFFER_ARGUMENT_COUNT {
        return Some(format!(
            "{context} threadgroup memory index {} exceeds the argument table (indices 0..={})",
            resource.binding,
            crate::air::reflect::THREADGROUP_BUFFER_ARGUMENT_COUNT - 1
        ));
    }
    if resource.length == 0 {
        return Some(format!(
            "{context} threadgroup memory {} declares zero bytes, which satisfies the reflected requirement without allocating anything",
            resource.binding
        ));
    }
    if !resource
        .length
        .is_multiple_of(METAL_THREADGROUP_MEMORY_GRANULARITY)
    {
        return Some(format!(
            "{context} threadgroup memory {} declares {} bytes, which Metal cannot allocate: a threadgroup-memory length must be a multiple of {METAL_THREADGROUP_MEMORY_GRANULARITY} bytes",
            resource.binding, resource.length
        ));
    }
    None
}

/// Why a device-buffer array does not author every element it declares, or `None`.
///
/// Metal binds element `i` of the array at buffer index `binding + i` and requires all `length` of
/// them: a slot the manifest leaves out is not bound at all, and
/// `validateComputeFunctionArguments` refuses the dispatch with `missing Buffer binding at index 2
/// for buffers.2[0]`. Without `MTL_DEBUG_LAYER=1` the driver dispatches anyway and the shader
/// reads undefined memory through the empty slot while the case still reports `qualified`.
///
/// There is no honest value to substitute for an unauthored operand, so this refuses. Contrast
/// [`buffer_bytes_padded_to_declared_size`], which pads a *declared size*: that padding lies past
/// everything the manifest describes, while a missing array element is a resource the manifest
/// claims exists and never says anything about.
///
/// One rule, consulted by [`require_case`] at both executors' entry and by the authored-case
/// checker before install, so an incomplete array cannot reach the store from either side.
pub fn device_buffer_array_completeness_error(
    resource: &DeviceBufferArrayResource,
    context: &str,
) -> Option<String> {
    let authored = resource
        .elements
        .iter()
        .map(|element| element.index)
        .filter(|&index| index < resource.length)
        .collect::<BTreeSet<_>>();
    let missing_count = u64::from(resource.length) - authored.len() as u64;
    if missing_count == 0 {
        return None;
    }
    let missing = (0..resource.length)
        .filter(|index| !authored.contains(index))
        .take(16)
        .map(|index| index.to_string())
        .collect::<Vec<_>>();
    let omitted = missing_count - missing.len() as u64;
    let suffix = if omitted == 0 {
        String::new()
    } else {
        format!(", ... ({omitted} more missing)")
    };
    Some(format!(
        "{context} device-buffer array {} declares {} elements but authors no bytes for element{} {}{suffix}",
        resource.binding,
        resource.length,
        if missing.len() == 1 { "" } else { "s" },
        missing.join(", ")
    ))
}

/// The bytes to bind for a buffer argument, padded to the size AIR declares for it.
///
/// A case authors only the bytes its shader reads, which is routinely far fewer than the
/// argument's declared struct -- one byte against the 264 of `params`, sixteen against the 1179628
/// of `spanTable`. Both APIs size a binding from the declaration rather than from the use, and
/// Metal's validation layer refuses the short one outright:
///
/// ```text
/// argument params[0] from Buffer(29) with offset(0) and length(1) has space for 1 bytes,
/// but argument has a length(264).
/// ```
///
/// Without that layer it binds the short buffer anyway and any read past the end is undefined.
/// Padding the *binding* leaves the authored bytes at the front and does not author the padding:
/// a case whose result depends on it is reading bytes it never declared, and now reads a stable
/// zero on both executors rather than whatever followed each allocation.
///
/// Never shortens. A runtime `array_ref` declares no size at all, and a case may legitimately
/// author more than the declaration.
pub fn buffer_bytes_padded_to_declared_size(
    authored: &[u8],
    declared_size: Option<usize>,
) -> Cow<'_, [u8]> {
    match declared_size {
        Some(size) if size > authored.len() => {
            let mut padded = Vec::with_capacity(size);
            padded.extend_from_slice(authored);
            padded.resize(size, 0);
            Cow::Owned(padded)
        }
        _ => Cow::Borrowed(authored),
    }
}

/// The declared byte size of the `[[buffer(n)]]` argument at `metal_index`, as AIR states it
/// through `air.arg_type_size`. `None` when the entry declares no buffer there, or declares one
/// whose extent is not a fixed object.
pub fn reflected_buffer_declared_size(
    reflection: &ShaderInterface,
    metal_index: u32,
) -> Option<usize> {
    reflection
        .bindings
        .iter()
        .find(|binding| binding.kind == ResourceKind::Buffer && binding.metal_index == metal_index)
        .and_then(|binding| binding.declared_size)
        .map(|size| size as usize)
}

/// Shared capability boundary for the literal Metal and Vulkan executors.
///
/// Unsupported manifest resources are rejected as a whole; no runner may drop them or synthesize
/// defaults while executing the supported literal-resource subset.
///
/// Command-buffer and pipeline-state handles are supplied by both executors: the Metal oracle
/// allocates each command buffer from its authored descriptor, encodes it and every pipeline
/// template into its argument-buffer field, guards the native setters, and checks the policy
/// table after completion; the old-engine candidate allocates `HostRecordsV1` records and pipeline
/// tokens and validates the records after completion. Whether a source's encoding has anything
/// observable besides the commands is a source question, answered by
/// [`unsupported_source_requirements`] in the checker before either executor runs.
pub fn require_case(case: &AuthoredCase, executor: &str) -> Result<(), String> {
    for resource in &case.device_buffer_arrays {
        if let Some(error) = device_buffer_array_completeness_error(resource, executor) {
            return Err(error);
        }
    }
    match case.stage {
        Stage::Kernel => {
            for resource in &case.threadgroup_memory {
                if let Some(error) = threadgroup_memory_error(resource, executor) {
                    return Err(error);
                }
            }
            if !case.vertex_inputs.is_empty() {
                return Err(format!(
                    "{executor} kernel execution does not accept vertex inputs"
                ));
            }
            if !case.render_targets.is_empty() && case.imageblock.is_none() {
                return Err(format!(
                    "{executor} kernel render targets require an authored imageblock"
                ));
            }
            if !case.render_targets.is_empty()
                && case
                    .imageblock
                    .as_ref()
                    .is_some_and(|imageblock| imageblock.implicit_coverage.is_none())
            {
                return Err(format!(
                    "{executor} kernel render targets require authored implicit imageblock coverage"
                ));
            }
            if matches!(case.output, OutputSelection::RenderTarget { .. })
                && case.imageblock.is_none()
            {
                return Err(format!(
                    "{executor} kernel render-target output requires an authored imageblock"
                ));
            }
        }
        Stage::Fragment | Stage::Vertex => {
            // Both executors support a 1x1 attachmentless fragment pass, including buffer
            // side effects. Literal validation separately checks that observed resources exist.
            if case.render_targets.is_empty()
                && case.depth_stencil.is_none()
                && case.fragment_imageblock.is_none()
                && case.stage != Stage::Fragment
                && !case.is_rasterization_disabled_vertex()
            {
                return Err(format!(
                    "{executor} graphics execution requires at least one attachment"
                ));
            }
            if !case.kernel_stage_inputs.is_empty()
                || !case.threadgroup_memory.is_empty()
                || case.imageblock.is_some()
            {
                return Err(format!(
                    "{executor} graphics execution does not accept kernel stage inputs, threadgroup memory, or imageblocks"
                ));
            }
            if case.fragment_imageblock.is_some() && case.stage != Stage::Fragment {
                return Err(format!(
                    "{executor} custom fragment imageblocks are valid only for fragment execution"
                ));
            }
            if case.stage == Stage::Fragment && !case.vertex_inputs.is_empty() {
                return Err(format!(
                    "{executor} fragment execution does not accept authored vertex inputs"
                ));
            }
            let dimensions = case
                .render_targets
                .first()
                .map(|target| target.dimensions)
                .or_else(|| {
                    case.depth_stencil
                        .as_ref()
                        .map(|attachment| attachment.dimensions)
                })
                .unwrap_or([1, 1]);
            // The imageblock is sized by the pass, not the other way round. Without this an
            // attachment-less case could author a plane larger than the 1x1 region both executors
            // raster and compare bytes the shader never covered.
            if case
                .fragment_imageblock
                .as_ref()
                .is_some_and(|imageblock| imageblock.dimensions != dimensions)
            {
                return Err(format!(
                    "{executor} custom fragment imageblock dimensions must match the pass attachments"
                ));
            }
            for target in &case.render_targets {
                if target.dimensions != dimensions {
                    return Err(format!(
                        "{executor} graphics render targets must have identical dimensions"
                    ));
                }
                if target.format == TextureFormat::Depth32Float {
                    return Err(format!(
                        "{executor} color render target {} cannot use depth32_float",
                        target.index
                    ));
                }
            }
            if case
                .depth_stencil
                .as_ref()
                .is_some_and(|attachment| attachment.dimensions != dimensions)
            {
                return Err(format!(
                    "{executor} graphics attachments must have identical dimensions"
                ));
            }
        }
        Stage::Object | Stage::Mesh => {
            // The case schema admits threadgroup memory only for an object case, whose setter
            // `setObjectThreadgroupMemoryLength:atIndex:` is.
            for resource in &case.threadgroup_memory {
                if let Some(error) = threadgroup_memory_error(resource, executor) {
                    return Err(error);
                }
            }
            if !case.vertex_inputs.is_empty()
                || !case.kernel_stage_inputs.is_empty()
                || case.imageblock.is_some()
                || case.fragment_imageblock.is_some()
            {
                return Err(format!(
                    "{executor} mesh execution does not accept vertex or kernel stage inputs, or \
                     imageblocks"
                ));
            }
            let mut dimensions = case.render_targets.iter().map(|target| target.dimensions);
            if let Some(first) = dimensions.next() {
                if dimensions.any(|other| other != first) {
                    return Err(format!(
                        "{executor} mesh render targets must have identical dimensions"
                    ));
                }
            }
            if let Some(target) = case
                .render_targets
                .iter()
                .find(|target| target.format == TextureFormat::Depth32Float)
            {
                return Err(format!(
                    "{executor} color render target {} cannot use depth32_float",
                    target.index
                ));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn device_buffer_array(
        binding: u32,
        length: u32,
        indices: &[u32],
    ) -> DeviceBufferArrayResource {
        DeviceBufferArrayResource {
            binding,
            length,
            elements: indices
                .iter()
                .map(|index| crate::case::DeviceBufferArrayElementResource {
                    index: *index,
                    role: crate::case::ResourceRole::Input,
                    bytes_b64: Some("AAAAAA==".into()),
                    initial_bytes_b64: None,
                })
                .collect(),
        }
    }

    /// Metal requires every element of a device-buffer array bound, so a manifest that declares
    /// more elements than it authors has to be refused rather than left to read undefined memory.
    #[test]
    fn a_device_buffer_array_must_author_every_element_it_declares() {
        let complete = |length, indices: &[u32]| {
            device_buffer_array_completeness_error(
                &device_buffer_array(0, length, indices),
                "manifest",
            )
        };
        assert_eq!(complete(3, &[0, 1, 2]), None);
        assert_eq!(complete(0, &[]), None);
        assert_eq!(complete(2, &[1, 0]), None);
        let one = complete(3, &[0, 1]).expect("element 2 is unauthored");
        assert!(one.ends_with("authors no bytes for element 2"), "{one}");
        let two =
            device_buffer_array_completeness_error(&device_buffer_array(4, 4, &[1]), "manifest")
                .expect("elements 0, 2 and 3 are unauthored");
        assert!(
            two.starts_with("manifest device-buffer array 4 declares 4 elements"),
            "{two}"
        );
        assert!(two.ends_with("no bytes for elements 0, 2, 3"), "{two}");
    }

    #[test]
    fn device_buffer_array_missing_diagnostic_is_bounded_by_authored_input() {
        let error = device_buffer_array_completeness_error(
            &device_buffer_array(0, u32::MAX, &[0, 0, u32::MAX]),
            "manifest",
        )
        .unwrap();
        assert!(
            error.ends_with("16, ... (4294967278 more missing)"),
            "{error}"
        );
        assert!(error.len() < 256);
    }

    /// Both APIs size a buffer binding from the argument's declaration, not from the bytes a case
    /// chose to author. Padding must extend the binding without disturbing the authored prefix,
    /// and must never shorten it: a runtime `array_ref` has no declared size at all, and a case
    /// may author more than the declaration.
    #[test]
    fn a_buffer_binding_is_padded_to_the_declared_size_and_never_shortened() {
        let padded = buffer_bytes_padded_to_declared_size;
        assert_eq!(padded(&[1, 2, 3], Some(6)).as_ref(), &[1, 2, 3, 0, 0, 0]);
        assert_eq!(padded(&[1, 2, 3], Some(3)).as_ref(), &[1, 2, 3]);
        assert_eq!(padded(&[1, 2, 3], Some(2)).as_ref(), &[1, 2, 3]);
        assert_eq!(padded(&[1, 2, 3], None).as_ref(), &[1, 2, 3]);
        assert!(matches!(padded(&[1, 2, 3], Some(3)), Cow::Borrowed(_)));
        assert!(matches!(padded(&[1, 2, 3], Some(4)), Cow::Owned(_)));
    }
    use crate::case::{
        BufferResource, Comparison, DepthStencilResource, Dispatch, Draw, ExecutionSafety,
        FragmentImageblockFormat, FragmentImageblockMemberResource, FragmentImageblockResource,
        FunctionConstant, OutputSelection, Primitive, RenderTargetResource, ResourceRole,
        ScalarType, TextureFormat, TextureResource, TextureType, VertexObservation,
    };

    /// A one-command compute command buffer at `(0, 0)` that executes as `execute` states.
    fn command_buffer(
        execute: Option<crate::case::indirect_commands::Execution>,
    ) -> crate::case::indirect_commands::CommandBufferResource {
        crate::case::indirect_commands::CommandBufferResource {
            buffer_binding: 0,
            field_offset: 0,
            capacity: 1,
            execution_range: [0, 1],
            inherit_pipeline_state: false,
            inherit_buffers: false,
            descriptor: crate::case::indirect_commands::CommandDescriptor::Compute {
                dispatch: crate::case::indirect_commands::ComputeDispatch::Both,
                max_buffer_bind_count: 1,
                max_threadgroup_memory_bind_count: 0,
                support_dynamic_attribute_stride: false,
            },
            commands: vec![],
            execute,
        }
    }

    fn case() -> AuthoredCase {
        AuthoredCase {
            air_sha256: "11".repeat(32),
            case_id: "22".repeat(32),
            name: "literal".into(),
            entry: "main".into(),
            stage: Stage::Kernel,
            buffers: vec![BufferResource {
                binding: 0,
                role: ResourceRole::Output,
                bytes_b64: None,
                initial_bytes_b64: Some("q6urqw==".into()),
                rasterization_rate_map: None,
            }],
            argument_buffer_buffers: vec![],
            device_buffer_arrays: vec![],
            command_buffers: vec![],
            pipeline_states: vec![],
            threadgroup_memory: vec![],
            imageblock: None,
            fragment_imageblock: None,
            acceleration_structures: vec![],
            visible_function_references: vec![],
            visible_function_tables: vec![],
            intersection_function_tables: vec![],
            argument_buffer_intersection_function_tables: vec![],
            textures: vec![],
            texture_arrays: vec![],
            argument_buffer_textures: vec![],
            argument_buffer_samplers: vec![],
            samplers: vec![],
            render_targets: vec![],
            depth_stencil: None,
            vertex_inputs: vec![],
            vertex_observation: None,
            varying_values: Vec::new(),
            kernel_stage_inputs: vec![],
            function_constants: vec![],
            dispatch: Some(Dispatch {
                grid: [1, 1, 1],
                threads_per_threadgroup: [1, 1, 1],
                stage_in_region: None,
            }),
            draw: None,
            tessellation: None,
            mesh_draw: None,
            output: OutputSelection::Buffer {
                binding: 0,
                offset: 0,
                length: 4,
            },
            compare: Comparison::Exact,
            execution_safety: ExecutionSafety::LoopFree,
            fuel: None,
            out_of_bounds: None,
            rationale: None,
            authored_by: None,
            render_pass_rate_map: None,
            raster_sample_count: None,
            public_source: None,
            no_effect: None,
            creation_bound: None,
        }
    }

    #[test]
    fn unsupported_manifest_data_is_rejected_not_ignored_or_defaulted() {
        let mut case = case();
        assert!(require_case(&case, "test executor").is_ok());
        case.function_constants.push(FunctionConstant {
            index: 0,
            scalar_type: ScalarType::U32,
            lanes: 1,
            bytes_b64: "AQAAAA==".into(),
        });
        assert!(require_case(&case, "test executor").is_ok());
        case.textures.push(TextureResource {
            binding: 1,
            role: ResourceRole::Output,
            texture_type: TextureType::D2,
            format: TextureFormat::Rgba32Float,
            dimensions: [1, 1, 1],
            sample_count: 1,
            bytes_b64: None,
            initial_bytes_b64: Some("qqqqqqqqqqqqqqqqqqqqqg==".into()),
            sparse: None,
            mip_levels: 1,
        });
        case.output = OutputSelection::Texture {
            binding: 1,
            origin: [0, 0, 0],
            dimensions: [1, 1, 1],
        };
        assert!(require_case(&case, "test executor").is_ok());
    }

    #[test]
    fn a_threadgroup_memory_length_metal_cannot_allocate_is_refused_at_the_executor() {
        let mut case = case();
        case.threadgroup_memory = vec![crate::case::ThreadgroupMemoryResource {
            binding: 0,
            length: METAL_THREADGROUP_MEMORY_GRANULARITY,
        }];
        assert!(require_case(&case, "test executor").is_ok());

        // Measured against Metal's own validation layer: the encoder asserts on a length that is
        // not a multiple of sixteen, so no runner may pass one through.
        case.threadgroup_memory[0].length = METAL_THREADGROUP_MEMORY_GRANULARITY - 1;
        assert!(require_case(&case, "test executor")
            .unwrap_err()
            .contains("must be a multiple of 16 bytes"));

        case.threadgroup_memory[0].length = 0;
        assert!(require_case(&case, "test executor")
            .unwrap_err()
            .contains("zero bytes"));

        case.threadgroup_memory[0].length = METAL_THREADGROUP_MEMORY_GRANULARITY;
        case.threadgroup_memory[0].binding = 30;
        assert!(require_case(&case, "test executor").is_ok());
        for index in [31, 32, u32::MAX] {
            case.threadgroup_memory[0].binding = index;
            assert!(require_case(&case, "test executor")
                .unwrap_err()
                .contains("exceeds the argument table"));
        }
    }

    #[test]
    fn source_and_observation_requirements_are_enforced_by_the_shared_executor_gate() {
        let reflection = crate::air::reflect::interface_from_kernel(
            &crate::air::meta::KernMeta::default(),
            Some("kernel"),
        );
        for (source_ll, requirement) in [
            (
                "call void @air.store.implicit_imageblock.v3f16()",
                ToolingRequirement::ImplicitImageblockLiteral,
            ),
            (
                "  call void @air.set_object_threadgroup_memory_length_render_command()",
                ToolingRequirement::IndirectCommandBuffer,
            ),
        ] {
            assert!(
                unsupported_air_requirements(&air_call_counts(source_ll)).contains(&requirement)
            );
            assert!(
                require_reflection(&case(), source_ll, &reflection, "test executor")
                    .unwrap_err()
                    .contains(requirement.as_str())
            );
        }
        // Encoding with nothing else to observe is a source requirement the gate enforces too:
        // a fragment buffer, which no replayed command reads, stops every executor; a kernel's
        // compute and render commands are observed only by a case that executes them.
        let encoding_only = "  call void @air.set_fragment_buffer_render_command.p1i8()";
        assert!(unsupported_air_requirements(&air_call_counts(encoding_only)).is_empty());
        assert!(
            require_reflection(&case(), encoding_only, &reflection, "test executor")
                .unwrap_err()
                .contains(ToolingRequirement::IndirectCommandBuffer.as_str())
        );
        for encoding_only in [
            "  call void @air.set_vertex_buffer_render_command.p1i8()",
            "  call void @air.set_kernel_buffer_compute_command.p1i8()",
        ] {
            assert!(
                require_reflection(&case(), encoding_only, &reflection, "test executor")
                    .unwrap_err()
                    .contains("the case executes none"),
                "{encoding_only}"
            );
        }
        let encoding_only = "  call void @air.set_kernel_buffer_compute_command.p1i8()";
        let mut executing = case();
        executing.command_buffers.push(command_buffer(Some(
            crate::case::indirect_commands::Execution {
                pipeline: None,
                buffers: vec![],
                vertex_buffers: vec![],
                fragment_buffers: vec![],
                object_buffers: vec![],
                mesh_buffers: vec![],
            },
        )));
        assert_eq!(
            require_reflection(&executing, encoding_only, &reflection, "test executor"),
            Ok(BTreeSet::new())
        );

        let vertex = crate::air::reflect::interface_from_vertex(
            &crate::air::meta::VertMeta::default(),
            Some("vertex"),
        );
        assert!(unsupported_observation_requirements(Stage::Vertex, &vertex)
            .contains(&ToolingRequirement::VertexSideEffectObservation));
        let mut authored = case();
        authored.stage = Stage::Vertex;
        assert!(require_reflection(&authored, "", &vertex, "test executor")
            .unwrap_err()
            .contains(ToolingRequirement::VertexSideEffectObservation.as_str()));
    }

    /// The indirect-command contract, row by row: a declared handle is inert; encoding is run
    /// by both executors but its commands are unobserved, so it needs another observable effect;
    /// setters no descriptor admits, and pointer encodings without a lowering, stay requirements.
    #[test]
    fn indirect_commands_close_only_where_another_effect_is_observable() {
        let kernel = crate::air::reflect::interface_from_kernel(
            &crate::air::meta::KernMeta::default(),
            Some("kernel"),
        );
        let declared = "!0 = !{i32 0, !\"air.command_buffer\", !\"air.location_index\", i32 0}";
        let encode = "  tail call void @air.set_kernel_buffer_compute_command.p1i8(ptr addrspace(1) %c, i32 %i, ptr addrspace(1) %b, i64 -1, i32 0)";
        let device_store = "  store i32 %v, ptr addrspace(1) %out, align 4, !tbaa !7";
        // Instruction lines are analyzed per function body.
        let function = |lines: &str| format!("define void @main() {{\n{lines}\n}}");
        // Encoding-only: the module's only effect is its commands, which a case observes only by
        // replaying them (a kernel's compute commands) or not at all (the command-buffer
        // requirement).
        let requirement = |ll: &str| {
            let ll = function(ll);
            commands_only(&ll, &air_call_counts(&ll), Some(&kernel)).is_some()
                || unsupported_source_requirements(&ll, &kernel)
                    .contains(&ToolingRequirement::IndirectCommandBuffer)
        };

        // Declared but never encoded: inert, supplied by both executors, nothing unobserved.
        let calls = air_call_counts(declared);
        assert!(uses_indirect_commands(declared, &calls));
        assert!(!encodes_indirect_commands(&calls));
        assert!(unobserved_calls(&calls).is_empty());
        assert!(!requirement(declared));

        // Encoding whose only effect is the commands keeps the requirement.
        let calls = air_call_counts(encode);
        assert!(encodes_indirect_commands(&calls));
        assert_eq!(
            unobserved_calls(&calls),
            BTreeMap::from([(
                "air.set_kernel_buffer_compute_command.p1i8".to_string(),
                UnobservedEffect::IndirectCommandContents
            )])
        );
        assert!(requirement(encode));
        assert_eq!(
            commands_only(&function(encode), &calls, Some(&kernel)),
            Some(CommandsOnly::Replayed)
        );
        assert_eq!(
            indirect_command_requirement(&function(encode), &calls, Some(&kernel)),
            None
        );
        assert!(require_reflection(&case(), &function(encode), &kernel, "test executor").is_err());
        // Without a reflection (the product could not inspect the entry) the row is not resolved.
        assert_eq!(
            indirect_command_requirement(&function(encode), &calls, None),
            Some(ToolingRequirement::IndirectCommandBuffer)
        );

        // A device write beside it is observable: authorable, with the commands still unobserved.
        let both = format!("{encode}\n{device_store}");
        assert!(!requirement(&both));
        assert!(require_reflection(&case(), &function(&both), &kernel, "test executor").is_ok());
        assert_eq!(
            indirect_command_requirement(&function(&both), &calls, None),
            None
        );
        for write in [
            "  %r = atomicrmw add ptr addrspace(1) %p, i32 1 seq_cst, align 4",
            "  %r = cmpxchg ptr addrspace(1) %p, i32 0, i32 1 seq_cst seq_cst, align 4",
            "  store volatile <2 x i32> <i32 1, i32 2>, ptr addrspace(1) %p, align 8",
            "  store i32 1, i32 addrspace(1)* %p, align 4",
            "  call void @llvm.memset.p1.i64(ptr addrspace(1) %p, i8 0, i64 4, i1 false)",
            "  %r = call i32 @air.atomic.global.add.u.i32(ptr addrspace(1) %p, i32 1, i32 0, i32 2, i1 true)",
            "  call void @air.write_texture_2d.v4f32(ptr addrspace(1) %t, <2 x i32> %c, <4 x float> %v, i32 0, i32 2)",
        ] {
            assert!(!requirement(&format!("{encode}\n{write}")), "{write}");
        }
        // Reads, private or threadgroup writes, stored pointers, and declarations are not
        // observable writes.
        for other in [
            "  store ptr addrspace(1) %b, ptr %slot, align 8",
            "  store i32 1, ptr addrspace(3) %shared, align 4",
            "  %r = call i32 @air.atomic.global.load.i32(ptr addrspace(1) %p, i32 0, i32 2, i1 true)",
            "  %r = atomicrmw add ptr addrspace(3) %p, i32 1 seq_cst, align 4",
            "  %v = load i32, ptr addrspace(1) %p, align 4",
            "declare void @llvm.memset.p1.i64(ptr addrspace(1), i8, i64, i1)",
        ] {
            assert!(requirement(&format!("{encode}\n{other}")), "{other}");
        }
        // A written device address is not comparable across the APIs: a pointer value, or a value
        // derived from `ptrtoint` in the same function, is not an observable write.
        for address in [
            "  store ptr addrspace(1) %b, ptr addrspace(1) %out, align 8",
            "  store i8 addrspace(1)* %b, i8 addrspace(1)* addrspace(1)* %out, align 8",
            "  store <2 x ptr addrspace(1)> %pair, ptr addrspace(1) %out, align 16",
            "  store i64 ptrtoint (ptr addrspace(1) @g to i64), ptr addrspace(1) %out, align 8",
            "  %a = ptrtoint ptr addrspace(1) %b to i64\n  %o = add i64 %a, 16\n  store i64 %o, ptr addrspace(1) %out, align 8",
            "  store i64 %o, ptr addrspace(1) %out, align 8\n  %o = or i64 %a, 1\n  %a = ptrtoint ptr addrspace(1) %b to i64",
            "  %r = atomicrmw xchg ptr addrspace(1) %p, ptr addrspace(1) %b seq_cst, align 8",
            "  %a = ptrtoint ptr addrspace(1) %b to i32\n  %r = call i32 @air.atomic.global.add.u.i32(ptr addrspace(1) %p, i32 %a, i32 0, i32 2, i1 true)",
        ] {
            assert!(requirement(&format!("{encode}\n{address}")), "{address}");
        }
        // Data stays data: a name that merely shares a prefix with an address, a value loaded
        // through an address, and an address in another function.
        for data in [
            "  %a = ptrtoint ptr addrspace(1) %b to i64\n  store i64 %ab, ptr addrspace(1) %out, align 8",
            "  %a = ptrtoint ptr addrspace(1) %b to i64\n  %q = inttoptr i64 %a to ptr addrspace(1)\n  %v = load i32, ptr addrspace(1) %q, align 4\n  store i32 %v, ptr addrspace(1) %out, align 4",
            "  ret void\n}\ndefine void @helper() {\n  %a = ptrtoint ptr addrspace(1) %b to i64\n  ret void\n}\ndefine void @other() {\n  store i64 %a, ptr addrspace(1) %out, align 8",
        ] {
            assert!(!requirement(&format!("{encode}\n{data}")), "{data}");
        }

        // Stage outputs the executors capture count as observable effects.
        let mut vertex = crate::air::reflect::interface_from_vertex(
            &crate::air::meta::VertMeta::default(),
            Some("vertex"),
        );
        let render = function("  tail call void @air.set_vertex_buffer_render_command.p1i8(ptr addrspace(1) %c, i32 %i, ptr addrspace(1) %b, i64 -1, i32 0)");
        let render_calls = air_call_counts(&render);
        // A rasterization-disabled vertex function whose only effect is a vertex buffer it
        // encodes is observed by executing its render commands.
        assert_eq!(
            commands_only(&render, &render_calls, Some(&vertex)),
            Some(CommandsOnly::Replayed)
        );
        vertex.vertex_builtins = Some(crate::air::reflect::VertexBuiltins {
            writes_position: true,
        });
        let requirements = unsupported_source_requirements(&render, &vertex);
        assert!(!requirements.contains(&ToolingRequirement::IndirectCommandBuffer));
        assert_eq!(commands_only(&render, &render_calls, Some(&vertex)), None);
        // A stage output nobody reflected is absent.
        assert_eq!(
            indirect_command_requirement(&render, &render_calls, None),
            Some(ToolingRequirement::IndirectCommandBuffer)
        );

        // An unimplemented pointer encoding has no producer lowering, and neither executor models
        // the other encoder operations: requirements whatever else the module writes.
        for unsupported in [
            "  call void @air.set_object_buffer_render_command.p3i8()",
            "  call void @air.set_mesh_buffer_render_command()",
            "  call void @air.set_object_threadgroup_memory_length_render_command()",
            "  call void @air.set_kernel_buffer_compute_command.p3i8()",
            "  call void @air.set_kernel_buffer_compute_command()",
            "  call void @air.draw_indexed_primitives_render_command.p3i16()",
            "  call void @air.draw_indexed_primitives_render_command.p1i8()",
            "  call void @air.draw_patches_render_command.p3i32.p1i8()",
            "  call void @air.draw_patches_render_command.p1i16.p1i8()",
            "  call void @air.draw_patches_render_command.p1i32()",
            "  call void @air.draw_indexed_patches_render_command.p1i32.p1i8()",
            "  call void @air.draw_indexed_patches_render_command.p1i32.p3i8.p1i8()",
            "  call void @air.set_cull_mode_render_command()",
        ] {
            let ll = format!("{unsupported}\n{device_store}");
            assert!(
                unsupported_air_requirements(&air_call_counts(&ll))
                    .contains(&ToolingRequirement::IndirectCommandBuffer),
                "{unsupported}"
            );
            assert!(requirement(&ll));
        }
        // Both executors run an indexed draw of either index type and space, a patch draw and an
        // indexed patch draw of either space for each buffer, a mesh draw of either grid form, an
        // object or a mesh buffer of either space, a reset of either family, a stage-in region,
        // an imageblock size and a copy: no requirement beside an observable write.
        for modeled in [
            "  call void @air.draw_mesh_threadgroups_render_command()",
            "  call void @air.draw_mesh_threads_render_command()",
            "  call void @air.set_object_buffer_render_command.p1i8()",
            "  call void @air.set_object_buffer_render_command.p2i8()",
            "  call void @air.set_mesh_buffer_render_command.p1i8()",
            "  call void @air.set_mesh_buffer_render_command.p2i8()",
            "  call void @air.draw_indexed_primitives_render_command.p1i16()",
            "  call void @air.draw_indexed_primitives_render_command.p1i32()",
            "  call void @air.draw_indexed_primitives_render_command.p2i16()",
            "  call void @air.draw_indexed_primitives_render_command.p2i32()",
            "  call void @air.draw_patches_render_command.p1i32.p1i8()",
            "  call void @air.draw_patches_render_command.p2i32.p2i8()",
            "  call void @air.draw_patches_render_command.p1i32.p2i8()",
            "  call void @air.draw_indexed_patches_render_command.p1i32.p1i8.p1i8()",
            "  call void @air.draw_indexed_patches_render_command.p2i32.p2i8.p2i8()",
            "  call void @air.draw_indexed_patches_render_command.p1i32.p2i8.p1i8()",
            "  call void @air.reset_render_command()",
            "  call void @air.reset_compute_command()",
            "  call void @air.set_stage_in_region_compute_command()",
            "  call void @air.set_imageblock_size_compute_command()",
            "  call void @air.copy_compute_command()",
        ] {
            let ll = format!("{modeled}\n{device_store}");
            assert!(
                unsupported_air_requirements(&air_call_counts(&ll)).is_empty(),
                "{modeled}"
            );
            assert!(!requirement(&ll), "{modeled}");
        }

        // The executors' shared gate supplies authored command-buffer and pipeline handles.
        let mut authored = case();
        authored.command_buffers.push(command_buffer(None));
        assert!(require_case(&authored, "test executor").is_ok());
        assert!(case_requirements(&authored).is_empty());
        // Host-encoded or executed commands are no executor's gap.
        let mut host = command_buffer(None);
        host.commands
            .push(crate::case::indirect_commands::HostCommand {
                index: 0,
                action: None,
                barrier: true,
                pipeline: None,
                buffers: vec![],
                threadgroup_memory_lengths: vec![],
                stage_in_region: None,
                vertex_buffers: vec![],
                fragment_buffers: vec![],
            });
        for resource in [
            host,
            command_buffer(Some(crate::case::indirect_commands::Execution {
                pipeline: None,
                buffers: vec![],
                vertex_buffers: vec![],
                fragment_buffers: vec![],
                object_buffers: vec![],
                mesh_buffers: vec![],
            })),
        ] {
            let mut authored = case();
            authored.command_buffers.push(resource);
            assert!(case_requirements(&authored).is_empty());
        }
    }

    /// Replay observes a module whose only effect is its commands only where both executors
    /// execute them: the compute and render commands of a kernel's dispatch or a vertex or fragment
    /// function's draw, never a tile function's, one of which selects what a replayed command
    /// writes.
    #[test]
    fn encoding_only_modules_are_observed_through_replay() {
        use crate::air::reflect::{ImplicitImageblockAttachment, ResourceAccess};
        let kernel = crate::air::reflect::interface_from_kernel(
            &crate::air::meta::KernMeta::default(),
            Some("kernel"),
        );
        let vertex = crate::air::reflect::interface_from_vertex(
            &crate::air::meta::VertMeta::default(),
            Some("vertex"),
        );
        let mut fragment = vertex.clone();
        fragment.stage = ShaderStage::Fragment;
        let mut patches = vertex.clone();
        patches.stage = ShaderStage::TessellationEvaluation;
        let mut tile = kernel.clone();
        tile.implicit_imageblock_attachments
            .push(ImplicitImageblockAttachment {
                attachment: 0,
                data_rate: 0,
                binding: 0,
                format: crate::air::reflect::TextureFormat::Rgba8,
                access: ResourceAccess::ReadWrite,
            });
        let classify = |calls: &[&str], reflection: Option<&ShaderInterface>| {
            let ll = calls
                .iter()
                .map(|callee| format!("  call void @{callee}()"))
                .collect::<Vec<_>>()
                .join("\n");
            let ll = format!("define void @main() {{\n{ll}\n}}");
            commands_only(&ll, &air_call_counts(&ll), reflection)
        };
        let replay = Some(CommandsOnly::Replayed);
        let buffer = Some(CommandsOnly::Unobserved);
        let producers = [&kernel, &vertex, &fragment, &patches];
        // What selects a replayed dispatch's or draw's writes, and what is carried beside it.
        let carried = [
            "air.set_barrier_compute_command",
            "air.clear_barrier_compute_command",
            "air.set_threadgroup_memory_length_compute_command",
            "air.set_stage_in_region_compute_command",
            "air.set_imageblock_size_compute_command",
            "air.set_fragment_buffer_render_command.p1i8",
            "air.set_fragment_buffer_render_command.p2i8",
            "air.set_barrier_render_command",
            "air.clear_barrier_render_command",
        ];
        for selecting in [
            "air.set_kernel_buffer_compute_command.p1i8",
            "air.set_kernel_buffer_compute_command.p2i8",
            "air.concurrent_dispatch_threadgroups_compute_command",
            "air.concurrent_dispatch_threads_compute_command",
            "air.set_pipeline_state_compute_command",
            "air.set_vertex_buffer_render_command.p1i8",
            "air.set_vertex_buffer_render_command.p2i8",
            "air.set_pipeline_state_render_command",
            "air.draw_primitives_render_command",
            "air.draw_indexed_primitives_render_command.p1i16",
            "air.draw_indexed_primitives_render_command.p2i32",
            "air.draw_patches_render_command.p1i32.p1i8",
            "air.draw_patches_render_command.p2i32.p2i8",
            "air.draw_indexed_patches_render_command.p1i32.p1i8.p1i8",
            "air.draw_indexed_patches_render_command.p2i32.p2i8.p2i8",
            "air.set_object_buffer_render_command.p1i8",
            "air.set_object_buffer_render_command.p2i8",
            "air.set_mesh_buffer_render_command.p1i8",
            "air.set_mesh_buffer_render_command.p2i8",
            "air.draw_mesh_threadgroups_render_command",
            "air.draw_mesh_threads_render_command",
            "air.reset_compute_command",
            "air.reset_render_command",
            "air.copy_compute_command",
        ] {
            for producer in producers {
                assert_eq!(
                    classify(&[selecting], Some(producer)),
                    replay,
                    "{selecting} {:?}",
                    producer.stage
                );
                let mut beside = carried.to_vec();
                beside.push(selecting);
                assert_eq!(
                    classify(&beside, Some(producer)),
                    replay,
                    "{selecting} {:?}",
                    producer.stage
                );
            }
            // A tile function's commands, or a module nobody reflected: no case executes them.
            assert_eq!(classify(&[selecting], Some(&tile)), buffer, "{selecting}");
            assert_eq!(classify(&[selecting], None), buffer, "{selecting}");
            // An object threadgroup's memory length beside it, which neither executor models: no
            // case executes it.
            assert_eq!(
                classify(
                    &[
                        selecting,
                        "air.set_object_threadgroup_memory_length_render_command"
                    ],
                    Some(&kernel)
                ),
                buffer,
                "{selecting}"
            );
        }
        // Compute and render commands of one module are both replayed.
        assert_eq!(
            classify(
                &[
                    "air.set_kernel_buffer_compute_command.p1i8",
                    "air.draw_primitives_render_command",
                ],
                Some(&vertex)
            ),
            replay
        );
        // A barrier, a threadgroup length, a stage-in region, an imageblock size, or a fragment
        // buffer alone changes no replayed value the module decides: a replayed draw's template is
        // a rasterization-disabled vertex function, and whether a dispatch's reads a region or an
        // imageblock is its template's.
        for alone in carried {
            for producer in producers {
                assert_eq!(classify(&[alone], Some(producer)), buffer, "{alone}");
            }
        }
        // No encoding, no requirement.
        assert_eq!(classify(&[], Some(&kernel)), None);
    }

    /// Each texture operation on a `ulong` texel, in every shape the public fixtures spell, is a
    /// `ulong` texture call, and no executor's gap. The same operations on a 32-bit texel are no
    /// `ulong` texture call.
    #[test]
    fn ulong_texture_operations_are_named_by_their_texel() {
        let fixtures = [
            include_str!("../fixtures/public/engine/unk_ulong_rw.ll"),
            include_str!("../fixtures/public/engine/unk_ulong_atomics.ll"),
        ];
        let mut operations = 0;
        for source in fixtures {
            let calls = air_call_counts(source);
            let ulong = ulong_texture_calls(&calls).collect::<Vec<_>>();
            assert!(
                ulong.iter().all(|name| name.ends_with(".u.v4i64")),
                "{ulong:?}"
            );
            assert_eq!(
                ulong.len(),
                calls
                    .keys()
                    .filter(|name| name.ends_with(".u.v4i64"))
                    .count()
            );
            operations += ulong.len();
            assert!(unsupported_air_requirements(&calls).is_empty());
            let narrow = calls
                .keys()
                .map(|name| (name.replace(".u.v4i64", ".u.v4i32"), 1))
                .collect::<BTreeMap<_, _>>();
            assert!(!calls_ulong_texture(&narrow));
        }
        // The distinct operations the two modules spell: reads, writes, and atomic minima and
        // maxima of the six shapes, the 2D ones at 32- and 16-bit coordinates.
        assert_eq!(operations, 28);
    }

    /// A tessellation control-point or patch attribute and a fragment imageblock member are
    /// authorable in every format the literal names, an 8-bit attribute and a `float` member among
    /// them; a type no format names stays the literal's requirement, which stops every executor.
    #[test]
    fn tessellation_attributes_and_imageblock_members_classify_by_their_format() {
        use crate::air::reflect::{
            FragmentImageblock, FragmentImageblockMember, PatchDomain, ResourceAccess,
            TessellationAttribute, TessellationInterface,
        };
        let watched = [
            ToolingRequirement::TessellationAttributeLiteral,
            ToolingRequirement::FragmentImageblockMemberLiteral,
        ];
        let classified = |reflection: &ShaderInterface| {
            unsupported_reflection_requirements(reflection)
                .into_iter()
                .filter(|requirement| watched.contains(requirement))
                .collect::<Vec<_>>()
        };
        let base = crate::air::reflect::interface_from_vertex(
            &crate::air::meta::VertMeta::default(),
            Some("vertex"),
        );
        let attribute = |type_name: Option<&str>| TessellationAttribute {
            location: 2,
            type_name: type_name.map(Into::into),
        };
        let tessellated = |control_point: Option<&str>, patch: Option<&str>| {
            let mut reflection = base.clone();
            reflection.tessellation = Some(TessellationInterface {
                domain: PatchDomain::Triangle,
                control_point_count: 3,
                control_point_attributes: vec![attribute(Some("float4")), attribute(control_point)],
                patch_attributes: vec![attribute(patch)],
                instance_id: None,
                amplification_id: None,
                amplification_count: None,
            });
            classified(&reflection)
        };
        let literal = [ToolingRequirement::TessellationAttributeLiteral];
        for name in [
            "char", "char2", "char3", "char4", "uchar", "uchar2", "uchar3", "uchar4", "short",
            "ushort4", "half", "half2", "float", "float4", "int3", "uint",
        ] {
            assert_eq!(tessellated(Some(name), Some("float")), [], "{name}");
            assert_eq!(tessellated(Some("float"), Some(name)), [], "{name}");
        }
        assert_eq!(tessellated(Some("ulong"), Some("float")), literal);
        assert_eq!(tessellated(Some("float"), None), literal);
        assert_eq!(tessellated(Some("ulong"), Some("uchar4")), literal);

        let imageblock = |members: &[(&str, u32)]| {
            let mut reflection = base.clone();
            reflection.fragment_imageblock = Some(FragmentImageblock {
                sample_size: 16,
                members: members
                    .iter()
                    .enumerate()
                    .map(|(index, (type_name, size))| FragmentImageblockMember {
                        offset: 4 * index as u32,
                        size: *size,
                        type_name: (*type_name).into(),
                        semantic: format!("m{index}"),
                        raster_order_group: 0,
                        binding: Some(index as u32),
                        access: ResourceAccess::ReadWrite,
                    })
                    .collect(),
                inputs: vec![],
                outputs: vec![],
            });
            classified(&reflection)
        };
        assert_eq!(
            imageblock(&[("half", 2), ("half4", 8), ("uchar4", 4), ("ushort", 2)]),
            []
        );
        assert_eq!(imageblock(&[("half", 2), ("float", 4)]), []);
        assert_eq!(imageblock(&[("uint", 4), ("uint", 4)]), []);
        for unknown in [("float2", 8), ("float", 8), ("uint", 8), ("int", 4)] {
            assert_eq!(
                imageblock(&[unknown]),
                [ToolingRequirement::FragmentImageblockMemberLiteral],
                "{unknown:?}"
            );
        }
    }

    #[test]
    fn reflected_stage_input_types_use_the_authored_attribute_contract() {
        use crate::air::meta::{KernMeta, KernRole, VertMeta, VertRole};

        let mut vertex_meta = VertMeta {
            roles: vec![(0, VertRole::VertexInput(3))],
            ..Default::default()
        };
        vertex_meta.vertex_input_types.insert(3, "uchar4".into());
        let vertex = crate::air::reflect::interface_from_vertex(&vertex_meta, Some("vertex"));
        assert!(!unsupported_reflection_requirements(&vertex)
            .contains(&ToolingRequirement::VertexAttributeLiteral));
        vertex_meta.vertex_input_types.insert(3, "ulong".into());
        let vertex = crate::air::reflect::interface_from_vertex(&vertex_meta, Some("vertex"));
        assert!(unsupported_reflection_requirements(&vertex)
            .contains(&ToolingRequirement::VertexAttributeLiteral));

        let mut kernel_meta = KernMeta {
            roles: vec![(0, KernRole::StageInput(2))],
            ..Default::default()
        };
        kernel_meta
            .stage_input_type_names
            .insert(0, "float3".into());
        let supported = crate::air::reflect::interface_from_kernel(&kernel_meta, Some("kernel"));
        assert!(!unsupported_reflection_requirements(&supported)
            .contains(&ToolingRequirement::KernelStageInputLiteral));

        kernel_meta.stage_input_type_names.insert(0, "ulong".into());
        let unsupported = crate::air::reflect::interface_from_kernel(&kernel_meta, Some("kernel"));
        assert!(unsupported_reflection_requirements(&unsupported)
            .contains(&ToolingRequirement::KernelStageInputLiteral));
    }

    #[test]
    fn fragment_cases_use_the_shared_render_contract() {
        let mut case = case();
        case.stage = Stage::Fragment;
        case.buffers.clear();
        case.dispatch = None;
        case.draw = Some(Draw {
            primitive: Primitive::Triangle,
            vertex_start: 0,
            vertex_count: 3,
            instance_count: 1,
            index_buffer: None,
            base_vertex: 0,
            base_instance: 0,
        });
        case.render_targets.push(RenderTargetResource {
            index: 0,
            format: TextureFormat::Rgba32Float,
            dimensions: [2, 2],
            initial_bytes_b64: "q6urq6urq6urq6urq6urq6urq6urq6urq6urq6urq6urq6urq6urq6urq6urq6urq6urq6urq6urq6urq6urqw==".into(),
        });
        case.output = OutputSelection::RenderTarget {
            index: 0,
            origin: [0, 0],
            dimensions: [2, 2],
        };
        assert!(require_case(&case, "test executor").is_ok());

        let mut attachmentless = case.clone();
        attachmentless.render_targets.clear();
        attachmentless.output = OutputSelection::None;
        assert!(require_case(&attachmentless, "test executor").is_ok());
        attachmentless.output = OutputSelection::Buffer {
            binding: 0,
            offset: 0,
            length: 4,
        };
        attachmentless.buffers.push(BufferResource {
            binding: 0,
            role: ResourceRole::Output,
            bytes_b64: None,
            initial_bytes_b64: Some("q6urqw==".into()),
            rasterization_rate_map: None,
        });
        assert!(attachmentless.validate_literal_resources().is_ok());
        assert!(require_case(&attachmentless, "test executor").is_ok());

        case.render_targets.push(RenderTargetResource {
            index: 1,
            format: TextureFormat::Rgba32Float,
            dimensions: [1, 2],
            initial_bytes_b64: "q6urq6urq6urq6urq6urq6urq6urq6urq6urq6urqw==".into(),
        });
        assert!(require_case(&case, "test executor")
            .unwrap_err()
            .contains("identical dimensions"));

        case.render_targets.pop();
        case.stage = Stage::Vertex;
        case.vertex_observation = Some(VertexObservation::Position);
        assert!(require_case(&case, "test executor").is_ok());
    }

    #[test]
    fn framebuffer_fetch_uses_the_authored_render_target_contract() {
        let ll = r#"define <4 x float> @frag(<4 x float> %color) { ret <4 x float> %color }
!air.fragment = !{!0}
!0 = !{ptr @frag, !1, !3}
!1 = !{!2}
!2 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}
!3 = !{!4}
!4 = !{i32 0, !"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}
"#;
        let reflection =
            crate::air::reflect::inspect_stage(ll, crate::case::Stage::Fragment).unwrap();
        assert_eq!(reflection.bindings[0].type_name.as_deref(), Some("float4"));
        let mut case = case();
        case.stage = Stage::Fragment;
        assert!(require_reflection(&case, ll, &reflection, "test executor")
            .unwrap_err()
            .contains("same index"));
        case.render_targets.push(RenderTargetResource {
            index: 0,
            format: TextureFormat::Rgba32Float,
            dimensions: [1, 1],
            initial_bytes_b64: "q6urq6urq6urq6urq6urqw==".into(),
        });
        case.dispatch = None;
        case.draw = Some(Draw {
            primitive: Primitive::Triangle,
            vertex_start: 0,
            vertex_count: 3,
            instance_count: 1,
            index_buffer: None,
            base_vertex: 0,
            base_instance: 0,
        });
        assert!(require_reflection(&case, ll, &reflection, "test executor")
            .unwrap()
            .is_empty());
        // More instances of the fullscreen triangle are as many ordered read-modify-writes of every
        // pixel, which the engine's candidate draws.
        case.draw.as_mut().unwrap().instance_count = 3;
        assert!(require_reflection(&case, ll, &reflection, "test executor")
            .unwrap()
            .is_empty());
        // Any other draw covers no pixel the authored triangle does not, or misses some.
        case.draw.as_mut().unwrap().vertex_count = 6;
        assert!(require_reflection(&case, ll, &reflection, "test executor")
            .unwrap_err()
            .contains("fullscreen triangle"));
    }

    #[test]
    fn texture_shape_requirements_match_literal_executor_boundaries() {
        use crate::air::meta::{KernMeta, KernRole};

        let names = [
            "texture2d_ms<float, read>",
            "texture_buffer<uint, read>",
            "texture2d<uint, write>",
        ];
        let mut meta = KernMeta {
            roles: names
                .iter()
                .enumerate()
                .map(|(index, _)| (index as u32, KernRole::Texture(index as u32)))
                .collect(),
            ..Default::default()
        };
        for (index, name) in names.into_iter().enumerate() {
            meta.texture_type_names.insert(index as u32, name.into());
        }
        let reflection = crate::air::reflect::interface_from_kernel(&meta, Some("k"));
        assert!(unsupported_reflection_requirements(&reflection).is_empty());
    }

    #[test]
    fn custom_fragment_imageblock_accepts_every_product_storage_format() {
        let ll = include_str!("../fixtures/public/fragment_custom_imageblock.ll");
        let mut reflection =
            crate::air::reflect::inspect_stage(ll, crate::case::Stage::Fragment).unwrap();
        {
            let members = &mut reflection.fragment_imageblock.as_mut().unwrap().members;
            for (member, (type_name, size)) in
                members
                    .iter_mut()
                    .zip([("half", 2), ("half4", 8), ("uchar4", 4), ("ushort", 2)])
            {
                member.type_name = type_name.into();
                member.size = size;
            }
        }
        assert!(!unsupported_reflection_requirements(&reflection)
            .contains(&ToolingRequirement::FragmentImageblockMemberLiteral));
        // A 32-bit `float` or `uint` member is authorable too.
        for (type_name, size) in [("float", 4), ("uint", 4)] {
            let member = &mut reflection.fragment_imageblock.as_mut().unwrap().members[0];
            member.type_name = type_name.into();
            member.size = size;
            assert!(
                !unsupported_reflection_requirements(&reflection)
                    .contains(&ToolingRequirement::FragmentImageblockMemberLiteral),
                "{type_name}"
            );
        }

        let member = &mut reflection.fragment_imageblock.as_mut().unwrap().members[0];
        member.type_name = "float4".into();
        member.size = 16;
        assert!(unsupported_reflection_requirements(&reflection)
            .contains(&ToolingRequirement::FragmentImageblockMemberLiteral));
    }

    #[test]
    fn fragment_depth_output_uses_the_shared_authored_attachment_contract() {
        let ll = r#"
define float @frag() { ret float 5.000000e-01 }
!air.fragment = !{!0}
!0 = !{ptr @frag, !1, !2}
!1 = !{!3}
!2 = !{}
!3 = !{!"air.depth", !"air.depth_qualifier", !"air.less", !"air.arg_type_name", !"float"}
"#;
        let reflection =
            crate::air::reflect::inspect_stage(ll, crate::case::Stage::Fragment).unwrap();
        assert_eq!(
            reflection.depth_qualifier,
            Some(crate::air::reflect::DepthQualifier::Less)
        );
        assert!(unsupported_source_requirements(ll, &reflection).is_empty());
        let mut case = case();
        case.stage = Stage::Fragment;
        assert!(require_reflection(&case, ll, &reflection, "test executor")
            .unwrap_err()
            .contains("authored depth attachment"));
        case.depth_stencil = Some(DepthStencilResource {
            dimensions: [1, 1],
            initial_depth_b64: Some("AAAAAA==".into()),
            initial_stencil_b64: None,
            state: None,
        });
        assert!(require_reflection(&case, ll, &reflection, "test executor").is_ok());

        let stencil_ll = ll.replace(
            "!\"air.depth\", !\"air.depth_qualifier\", !\"air.less\", !\"air.arg_type_name\", !\"float\"",
            "!\"air.stencil\", !\"air.arg_type_name\", !\"uint\"",
        );
        let stencil_reflection =
            crate::air::reflect::inspect_stage(stencil_ll.as_str(), crate::case::Stage::Fragment)
                .unwrap();
        assert!(unsupported_source_requirements(&stencil_ll, &stencil_reflection).is_empty());
        case.depth_stencil = Some(DepthStencilResource {
            dimensions: [1, 1],
            initial_depth_b64: None,
            initial_stencil_b64: Some("AA==".into()),
            state: None,
        });
        assert!(
            require_reflection(&case, &stencil_ll, &stencil_reflection, "test executor").is_ok()
        );
    }

    /// A fragment entry whose only output is a custom imageblock has no attachment to author: AIR
    /// declares no render target, and the resource contract rejects one the reflection does not
    /// name. Both executors already raster an attachment-free fragment pass, so the imageblock is
    /// the attachment -- and it is sized by the pass, not the other way round.
    #[test]
    fn a_custom_fragment_imageblock_is_the_attachment_and_takes_the_pass_dimensions() {
        let imageblock = |dimensions: [u32; 2]| FragmentImageblockResource {
            dimensions,
            members: vec![FragmentImageblockMemberResource {
                semantic: "color".into(),
                format: FragmentImageblockFormat::Half4,
                role: ResourceRole::InOut,
                bytes_b64: None,
                initial_bytes_b64: Some("AAAAAAAAAAA=".into()),
            }],
        };
        let mut case = case();
        case.stage = Stage::Fragment;
        case.buffers = vec![];
        case.render_targets = vec![];
        case.output = OutputSelection::FragmentImageblock {
            semantic: "color".into(),
            origin: [0, 0],
            dimensions: [1, 1],
        };
        assert!(
            case.validate_literal_resources()
                .unwrap_err()
                .iter()
                .any(|error| error.contains(
                    "fragment imageblock output requires a fragment_imageblock resource"
                )),
            "an imageblock observation requires its resource"
        );
        case.fragment_imageblock = Some(imageblock([1, 1]));
        require_case(&case, "test executor")
            .expect("a custom fragment imageblock is an attachment");
        // The attachment-free pass rasters 1x1, so a larger plane would compare bytes no fragment
        // ever covered.
        case.fragment_imageblock = Some(imageblock([2, 2]));
        assert!(
            require_case(&case, "test executor")
                .unwrap_err()
                .contains("imageblock dimensions must match"),
            "a plane larger than the pass is refused"
        );
    }

    /// The census's intrinsic contract is the engine's table. A family it decodes that the old
    /// product's table did not list (`air.simdgroup_matrix_8x8_init_filled`, which the frontend
    /// emits for `make_filled_simdgroup_matrix`) is known; so are the families the table lists as
    /// not decoded and the ray-tracing names it refuses as a typed feature. A name of neither kind,
    /// and anything outside `air.`, is not.
    #[test]
    fn the_intrinsic_contract_is_the_engines_table() {
        let filled = "air.simdgroup_matrix_8x8_init_filled.v64f32.f32";
        let unknown = "air.future_tensor.i32";
        let calls = BTreeMap::from([(filled.to_string(), 1), (unknown.to_string(), 2)]);
        assert_eq!(
            unrecognized_air_intrinsics(&calls),
            BTreeMap::from([(unknown.to_string(), 2)])
        );
        for known in [
            filled,
            "air.simdgroup_matrix_8x8_init_diag.v64f32.f32",
            "air.fast_sin.f32",
            // Listed as not decoded (`UNDECODED`), and refused as a typed feature.
            "air.function_constant_predicate",
            "air.set_pipeline_state_render_command",
            // A ray-tracing name the grammar does not decode (`undecoded_feature`).
            "air.next_intersection_query",
        ] {
            assert!(air_intrinsic_is_known(known), "{known}");
        }
        for unknown in [unknown, "air.", "llvm.fabs.f32", "metal2vulkan.helper"] {
            assert!(!air_intrinsic_is_known(unknown), "{unknown}");
        }
    }

    /// A sampler state held in an argument buffer stops no executor, as its reflected member or as
    /// its authored literal: the Metal oracle encodes the sampler and the engine's candidate
    /// binds it.
    #[test]
    fn an_argument_buffer_sampler_stops_no_executor() {
        use crate::air::reflect::{ArgumentBufferResourceKind, EmbeddedArgBuffer};
        use crate::case::{
            ArgumentBufferSamplerResource, SamplerAddressMode as CaseAddressMode,
            SamplerCompareFunction, SamplerFilter as CaseFilter, SamplerMipFilter as CaseMipFilter,
        };
        let field = |offset: u32, kind| EmbeddedArgBuffer {
            buffer_param_index: 0,
            buffer_index: 0,
            field_offset: offset,
            field_ordinal: offset / 8,
            argument_index: offset / 8,
            resource_kind: Some(kind),
        };
        let mut texture = kernel_reflection();
        texture
            .argument_buffer_fields
            .push(field(0, ArgumentBufferResourceKind::Texture));
        assert!(unsupported_reflection_requirements(&texture).is_empty());
        let mut material = texture.clone();
        material
            .argument_buffer_fields
            .push(field(8, ArgumentBufferResourceKind::Sampler));
        assert!(unsupported_reflection_requirements(&material).is_empty());
        assert!(unsupported_source_requirements("", &material).is_empty());
        assert!(engine_source_requirements(&material).is_empty());

        let mut authored = case();
        authored
            .argument_buffer_samplers
            .push(ArgumentBufferSamplerResource {
                buffer_binding: 0,
                field_offset: 8,
                address_mode: CaseAddressMode::ClampToEdge,
                min_filter: CaseFilter::Linear,
                mag_filter: CaseFilter::Linear,
                mip_filter: CaseMipFilter::NotMipmapped,
                normalized_coordinates: true,
                compare_function: Some(SamplerCompareFunction::Less),
            });
        assert!(case_requirements(&authored).is_empty());
    }

    /// A kernel with one device buffer, whose reflected binding the family tests below reshape.
    fn kernel_reflection() -> ShaderInterface {
        let ll = r#"
define void @k(ptr addrspace(1) %out) {
entry:
  store i32 1, ptr addrspace(1) %out, align 4
  ret void
}
!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 0, !"air.buffer", !"air.buffer_size", i32 4, !"air.location_index", i32 0, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
"#;
        crate::air::reflect::inspect_stage(ll, crate::case::Stage::Kernel).unwrap()
    }

    /// Each family of a reflected interface the checker requires a case to author and the new
    /// engine's candidate does not run is a source requirement that stops that candidate alone; the
    /// render families only in a vertex or fragment function, and a device-buffer array in an
    /// object or mesh function too.
    #[test]
    fn engine_source_families_follow_the_reflected_interface() {
        use crate::air::reflect::{
            EmbeddedArgBuffer, ImplicitImageblockAttachment, PatchDomain, ResourceAccess,
            TessellationInterface,
        };
        let kernel = kernel_reflection();
        assert!(engine_source_requirements(&kernel).is_empty());
        let only = |reflection: &ShaderInterface, requirement| {
            assert_eq!(
                engine_source_requirements(reflection),
                BTreeSet::from([requirement]),
                "{requirement}"
            );
            assert_eq!(requirement.stops(), [Executor::EngineCandidate]);
            assert!(unsupported_source_requirements("", reflection).contains(&requirement));
        };

        let mut pipelines = kernel.clone();
        for (offset, kind) in [
            (0, crate::air::reflect::ArgumentBufferResourceKind::Buffer),
            (
                8,
                crate::air::reflect::ArgumentBufferResourceKind::ComputePipelineState,
            ),
        ] {
            pipelines.argument_buffer_fields.push(EmbeddedArgBuffer {
                buffer_param_index: 0,
                buffer_index: 0,
                field_offset: offset,
                field_ordinal: offset / 8,
                argument_index: offset / 8,
                resource_kind: Some(kind),
            });
        }
        // A pipeline-state member is an opaque id in its owner.
        assert!(engine_source_requirements(&pipelines).is_empty());

        let mut tile = kernel.clone();
        tile.implicit_imageblock_attachments
            .push(ImplicitImageblockAttachment {
                attachment: 0,
                data_rate: 0,
                binding: 0,
                format: crate::air::reflect::TextureFormat::Rgba8,
                access: ResourceAccess::ReadWrite,
            });
        // The candidate runs a kernel's implicit imageblock as a tile pipeline.
        assert!(engine_source_requirements(&tile).is_empty());

        let ll = include_str!("../fixtures/public/fragment_custom_imageblock.ll");
        let imageblock =
            crate::air::reflect::inspect_stage(ll, crate::case::Stage::Fragment).unwrap();
        // The candidate runs a fragment function's explicit imageblock.
        assert!(imageblock.fragment_imageblock.is_some());
        assert!(engine_source_requirements(&imageblock).is_empty());

        // A device-buffer array and an acceleration structure are render families only.
        let buffer = kernel.bindings[0].clone();
        let mut vertex = crate::air::reflect::interface_from_vertex(
            &crate::air::meta::VertMeta::default(),
            Some("vertex"),
        );
        assert!(engine_source_requirements(&vertex).is_empty());
        for (kind, type_name, requirement) in [
            (
                ResourceKind::Buffer,
                "array_ref<void>",
                ToolingRequirement::RenderDeviceBufferArray,
            ),
            (
                ResourceKind::PrimitiveAccelerationStructure,
                "primitive_acceleration_structure",
                ToolingRequirement::RenderAccelerationStructure,
            ),
            (
                ResourceKind::IntersectionFunctionTable,
                "intersection_function_table<triangle_data>",
                ToolingRequirement::RenderAccelerationStructure,
            ),
        ] {
            let mut binding = buffer.clone();
            binding.kind = kind;
            binding.type_name = Some(type_name.into());
            let mut render = vertex.clone();
            render.bindings.push(binding.clone());
            only(&render, requirement);
            // An object or mesh function's device-buffer array is a render family too; no case
            // binds an acceleration structure or a function table at those stages.
            let mut mesh = render.clone();
            mesh.stage = ShaderStage::Mesh;
            assert_eq!(
                engine_source_requirements(&mesh),
                BTreeSet::from_iter(
                    (requirement == ToolingRequirement::RenderDeviceBufferArray)
                        .then_some(requirement)
                ),
                "{kind:?}"
            );
            let mut compute = kernel.clone();
            compute.bindings.push(binding);
            assert!(engine_source_requirements(&compute).is_empty(), "{kind:?}");
        }

        vertex.tessellation = Some(TessellationInterface {
            domain: PatchDomain::Triangle,
            control_point_count: 3,
            control_point_attributes: vec![],
            patch_attributes: vec![],
            instance_id: None,
            amplification_id: None,
            amplification_count: None,
        });
        // The engine's candidate draws patches.
        assert!(engine_source_requirements(&vertex).is_empty());

        // A color input is a fragment function's framebuffer fetch.
        let fetch = r#"define <4 x float> @frag(<4 x float> %color) { ret <4 x float> %color }
!air.fragment = !{!0}
!0 = !{ptr @frag, !1, !3}
!1 = !{!2}
!2 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}
!3 = !{!4}
!4 = !{i32 0, !"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}
"#;
        let fetch =
            crate::air::reflect::inspect_stage(fetch, crate::case::Stage::Fragment).unwrap();
        // The engine's candidate reads it through an input attachment.
        assert!(engine_source_requirements(&fetch).is_empty());
    }

    /// Each family of a case's literals the engine's candidate does not run is a case
    /// requirement, the render families only in a vertex or fragment case, and a device-buffer
    /// array in an object or mesh case too.
    #[test]
    fn engine_case_families_follow_the_case_literals() {
        use crate::case::{
            AccelerationStructureKind, AccelerationStructureResource,
            ArgumentBufferIntersectionFunctionTableResource, ImageblockResource,
            ImplicitImageblockCoverage, IntersectionFunctionTableEntry,
            IntersectionFunctionTableResource,
        };
        let kernel = case();
        assert!(case_requirements(&kernel).is_empty());
        let only = |case: &AuthoredCase, requirement| {
            assert_eq!(
                case_requirements(case),
                BTreeSet::from([requirement]),
                "{requirement}"
            );
            assert_eq!(requirement.stops(), [Executor::EngineCandidate]);
        };

        let mut pipelines = kernel.clone();
        pipelines
            .pipeline_states
            .push(crate::case::indirect_commands::PipelineResource {
                buffer_binding: 0,
                field_offset: 8,
                token: 1,
                air_sha256: "33".repeat(32),
                case_id: "44".repeat(32),
            });
        // A pipeline-state literal is an opaque id in its owner.
        assert!(case_requirements(&pipelines).is_empty());

        let mut tile = kernel.clone();
        tile.imageblock = Some(ImageblockResource {
            dimensions: [1, 1],
            implicit_coverage: Some(ImplicitImageblockCoverage::FullSingleSample),
        });
        // The candidate runs a tile function over any grid of tiles: one tile of one thread per
        // cell, a render area of more than one tile, a threadgroup that is not the tile's, or a
        // tile past 2x2.
        for (dimensions, grid, threads) in [
            ([1, 1], [1, 1, 1], [1, 1, 1]),
            ([2, 2], [2, 2, 1], [2, 2, 1]),
            ([1, 1], [2, 1, 1], [1, 1, 1]),
            ([1, 1], [1, 1, 1], [2, 1, 1]),
            ([1, 1], [1, 1, 2], [1, 1, 1]),
            ([1, 2], [1, 2, 1], [1, 2, 1]),
            ([4, 4], [4, 4, 1], [4, 4, 1]),
            ([16, 16], [16, 16, 1], [16, 16, 1]),
        ] {
            let mut grid_case = tile.clone();
            grid_case.imageblock.as_mut().unwrap().dimensions = dimensions;
            grid_case.dispatch = Some(Dispatch {
                grid,
                threads_per_threadgroup: threads,
                stage_in_region: None,
            });
            assert!(
                case_requirements(&grid_case).is_empty(),
                "{dimensions:?} {grid:?} {threads:?}"
            );
        }
        // An explicit imageblock is no tile function.
        tile.imageblock.as_mut().unwrap().implicit_coverage = None;
        tile.dispatch = Some(Dispatch {
            grid: [2, 1, 1],
            threads_per_threadgroup: [1, 1, 1],
            stage_in_region: None,
        });
        assert!(case_requirements(&tile).is_empty());

        let opaque = IntersectionFunctionTableEntry::OpaqueTriangle {
            index: 0,
            signature: vec![],
        };
        let linked = IntersectionFunctionTableEntry::Linked {
            index: 0,
            module_sha256: "55".repeat(32),
            function: "f".into(),
        };
        let mut encoded = kernel.clone();
        encoded.argument_buffer_intersection_function_tables =
            vec![ArgumentBufferIntersectionFunctionTableResource {
                buffer_binding: 0,
                field_offset: 0,
                size: 1,
                entries: vec![opaque.clone()],
            }];
        only(
            &encoded,
            ToolingRequirement::ArgumentBufferIntersectionFunctionTable,
        );
        let table = |entry| IntersectionFunctionTableResource {
            binding: 2,
            size: 1,
            entries: vec![entry],
        };
        let mut traced = kernel.clone();
        traced.intersection_function_tables = vec![table(opaque)];
        traced.acceleration_structures = vec![AccelerationStructureResource {
            binding: 3,
            kind: AccelerationStructureKind::Instance,
            primitive_triangles_b64: None,
            child_references: vec![0],
        }];
        assert!(case_requirements(&traced).is_empty());
        let mut linking = traced.clone();
        linking.intersection_function_tables = vec![table(linked)];
        only(&linking, ToolingRequirement::LinkedIntersectionFunction);

        let mut fragment = traced.clone();
        fragment.stage = Stage::Fragment;
        only(&fragment, ToolingRequirement::RenderAccelerationStructure);
        let mut arrays = kernel.clone();
        arrays.device_buffer_arrays = vec![device_buffer_array(0, 1, &[0])];
        assert!(case_requirements(&arrays).is_empty());
        let mut mesh = arrays.clone();
        mesh.stage = Stage::Mesh;
        only(&mesh, ToolingRequirement::RenderDeviceBufferArray);
        // An object case's threadgroup memory, which both executors set.
        let mut object = kernel.clone();
        object.threadgroup_memory = vec![crate::case::ThreadgroupMemoryResource {
            binding: 0,
            length: 16,
        }];
        object.stage = Stage::Object;
        assert!(case_requirements(&object).is_empty());
        arrays.stage = Stage::Vertex;
        arrays.vertex_observation = Some(VertexObservation::Position);
        only(&arrays, ToolingRequirement::RenderDeviceBufferArray);

        let mut vertex = kernel.clone();
        vertex.stage = Stage::Vertex;
        vertex.vertex_observation = Some(VertexObservation::Position);
        assert!(case_requirements(&vertex).is_empty());
        vertex.depth_stencil = Some(DepthStencilResource {
            dimensions: [1, 1],
            initial_depth_b64: Some("AAAAAA==".into()),
            initial_stencil_b64: None,
            state: None,
        });
        only(&vertex, ToolingRequirement::VertexDepthStencilAttachment);
        let mut disabled = kernel.clone();
        disabled.stage = Stage::Vertex;
        disabled.render_targets.push(RenderTargetResource {
            index: 0,
            format: TextureFormat::Rgba32Float,
            dimensions: [1, 1],
            initial_bytes_b64: "q6urq6urq6urq6urq6urqw==".into(),
        });
        only(
            &disabled,
            ToolingRequirement::RasterizationDisabledRenderTarget,
        );

        // The engine's candidate draws a tessellation in any authored state.
        let mut patches = vertex.clone();
        patches.depth_stencil = None;
        patches.tessellation = Some(crate::case::TessellationDraw {
            factors: vec![crate::case::TessellationFactors {
                edge_f16: vec![0x3c00; 3],
                inside_f16: vec![0x3c00],
            }],
            instance_count: 1,
            amplification_count: 1,
            control_points: vec![],
            patch_inputs: vec![],
            partition_mode: Default::default(),
            output_winding: Default::default(),
            cull_mode: Default::default(),
        });
        assert!(case_requirements(&patches).is_empty());
        let mut culled = patches.clone();
        culled.tessellation.as_mut().unwrap().factors[0].edge_f16[1] = 0;
        let mut back = patches.clone();
        back.tessellation.as_mut().unwrap().cull_mode = crate::case::CullMode::Back;
        // The same draw with rasterization disabled (no vertex observation, a buffer output) is
        // drawn by the Metal oracle and the engine only.
        let mut unrasterized = patches.clone();
        unrasterized.vertex_observation = None;
        assert!(unrasterized.is_rasterization_disabled_vertex());
        for case in [&culled, &back, &unrasterized] {
            assert!(case_requirements(case).is_empty());
        }

        // A draw of indices from an index buffer stops no executor.
        let mut drawn = kernel.clone();
        drawn.stage = Stage::Vertex;
        drawn.vertex_observation = Some(VertexObservation::Position);
        drawn.draw = Some(crate::case::Draw {
            primitive: crate::case::Primitive::Point,
            vertex_start: 0,
            vertex_count: 1,
            instance_count: 1,
            index_buffer: None,
            base_vertex: 0,
            base_instance: 0,
        });
        assert!(case_requirements(&drawn).is_empty());
        let mut indexed = drawn.clone();
        indexed.draw.as_mut().unwrap().index_buffer = Some(crate::case::IndexBufferLiteral {
            index_type: crate::case::IndexType::Uint32,
            offset: 0,
            bytes_b64: "AAAAAA==".into(),
        });
        assert!(case_requirements(&indexed).is_empty());
    }

    /// The checker accepts a case the Metal oracle and the engine's candidate run, and refuses
    /// one either cannot. Each executor then refuses exactly the requirements that stop it.
    #[test]
    fn a_case_is_checkable_while_the_metal_oracle_and_the_candidate_run_it() {
        use ToolingRequirement as R;
        let set = |requirements: &[R]| requirements.iter().copied().collect::<BTreeSet<_>>();
        assert_eq!(evidence_gap(&set(&[])), None);
        assert_eq!(
            evidence_gap(&set(&[R::RenderDeviceBufferArray])).as_deref(),
            Some("engine_candidate: render_device_buffer_array (the candidate is stopped)")
        );
        let metal = evidence_gap(&set(&[
            R::IndirectCommandBuffer,
            R::RenderDeviceBufferArray,
        ]))
        .unwrap();
        assert!(
            metal.starts_with("indirect_command_buffer (the Metal oracle"),
            "{metal}"
        );
        // Metal's own limit is named as such, beside a tooling gap of the oracle.
        assert_eq!(
            evidence_gap(&set(&[R::KernelMinLodClamp, R::IndirectCommandBuffer])).as_deref(),
            Some(
                "kernel_min_lod_clamp (Metal cannot build the case's pipeline, so it has no \
                 reference); indirect_command_buffer (the Metal oracle cannot run the case)"
            )
        );

        let requirements = set(&[R::RenderDeviceBufferArray]);
        assert_eq!(
            require_executor(&requirements, Executor::EngineCandidate),
            Err(set(&[R::RenderDeviceBufferArray]))
        );
        assert_eq!(
            require_executor(&requirements, Executor::MetalOracle),
            Ok(())
        );
    }
}
