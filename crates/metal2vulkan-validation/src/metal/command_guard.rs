//! The Metal oracle's native command-buffer guard: the source rewrite that hands every native
//! command setter the policy table it must check before it encodes ([`native_guarded_source`]), and
//! the table itself ([`NativeCommandPolicies`]), with the LLVM text front end it parses with
//! (`llvm`), pruned to what the guard reaches.

use crate::air::calls::CommandEncoderOperation;
use crate::air::text::{matching_paren, split_top_level, strip_comment};
use llvm::{
    parse_declaration, parse_function_header, parse_global_symbol_with_params,
    parse_quoted_global_symbol, resolve_call, LlType,
};
use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write;

mod encoding;
mod layout;
mod llvm;
mod policy;
pub use policy::{NativeCommandPolicies, NativeCommandPolicy};

/// Guard native setters without changing command handles or original resource layouts.
///
/// `policy` supplies `(registration_count, policy_buffer_index)`. Without a policy, setter
/// references are refused; unused declarations and metadata alone need no transport.
/// Each producer entry receives the selected device policy-table parameter.
/// Internal direct calls forward that parameter. Address-taken producers are refused because
/// their callable ABI cannot be changed locally. The caller must retain the initialized table
/// and validate its entire readback before replay. This does not instrument source loops or
/// establish GPU execution admission or shader/host resource-ID agreement.
pub fn native_guarded_source(
    source: &str,
    policy: Option<(u32, u32)>,
) -> Result<Cow<'_, str>, String> {
    if !native_setter_uses(source)? {
        return Ok(Cow::Borrowed(source));
    }
    let (registration_count, policy_buffer_index) =
        policy.ok_or("native command setters require a policy binding")?;
    let mut declarations = BTreeMap::new();
    let mut definitions = BTreeMap::new();
    let mut nodes = BTreeMap::new();
    let mut callers: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    let mut needed = BTreeSet::new();
    let mut locals = BTreeSet::new();
    let mut current = None;
    let mut max_metadata = 0u32;
    for raw in source.lines() {
        let line = strip_comment(raw).trim();
        locals.extend(local_symbols(line)?.into_iter().map(|(_, name)| name));
        if line.starts_with("define ") {
            let function = parse_function_header(line)?;
            current = Some(function.name.clone());
            if definitions.insert(function.name, line).is_some() {
                return Err("native module repeats a function definition".into());
            }
        } else if line == "}" {
            current = None;
        } else if line.starts_with("declare ") {
            let declaration = parse_declaration(line)?;
            if declarations.insert(declaration.name, line).is_some() {
                return Err("native module repeats an ABI declaration".into());
            }
        } else if let Some(call) = resolve_call(line) {
            let caller = current
                .as_ref()
                .ok_or("native call is outside a function")?;
            callers
                .entry(call.callee.clone())
                .or_default()
                .insert(caller.clone());
            if CommandEncoderOperation::from_symbol(&call.callee).is_some() {
                needed.insert(caller.clone());
            }
        }
        if let Some((name, body)) = line.split_once(" = ") {
            if let Some(id) = name.strip_prefix('!') {
                if let Ok(id) = id.parse::<u32>() {
                    max_metadata = max_metadata.max(id);
                }
                let body = body.strip_prefix("distinct ").unwrap_or(body);
                if let Some(body) = body.strip_prefix("!{").and_then(|s| s.strip_suffix('}')) {
                    if nodes.insert(name, body).is_some() {
                        return Err("native module repeats a metadata node".into());
                    }
                }
            }
        }
    }
    if policy_buffer_index >= 31 {
        return Err("native policy buffer index exceeds the Metal buffer namespace".into());
    }
    let data_layout = layout::AirDataLayout::from_ir(source)?.unwrap_or_default();
    if data_layout.pointer_storage_size_align(1) != (8, 8)
        || layout::raw_size_align(&LlType::Int(32), &|ty| Ok(ty.clone()), Some(&data_layout))?
            != (4, 4)
    {
        return Err(
            "native policy table requires eight-byte device pointers and four-byte words".into(),
        );
    }
    let mut pending = needed.iter().cloned().collect::<Vec<_>>();
    while let Some(callee) = pending.pop() {
        for caller in callers.get(&callee).into_iter().flatten() {
            if needed.insert(caller.clone()) {
                pending.push(caller.clone());
            }
        }
    }
    let mut occupied = global_names(source)?;
    let mut serial = 0;
    let mut fresh = || loop {
        let name = format!("__m2v_native_command_{serial}");
        serial += 1;
        if occupied.insert(name.clone()) {
            break name;
        }
    };
    let lookup_name = fresh();
    let parameter = (0..)
        .map(|index| format!("__m2v_policy_{index}"))
        .find(|name| !locals.contains(&format!("%\"{name}\"")))
        .unwrap();
    let parameter = format!("%{parameter}");
    let mut helpers = BTreeMap::new();
    for symbol in callers
        .keys()
        .filter(|symbol| CommandEncoderOperation::from_symbol(symbol).is_some())
    {
        if definitions.contains_key(symbol) {
            return Err("native command ABI symbol has a function body".into());
        }
        let declaration = declarations
            .get(symbol)
            .ok_or("native command use has no declaration")?;
        let name = fresh();
        let guard = native_command_guard(declaration, &name, &lookup_name)?;
        helpers.insert(symbol.clone(), (name, guard));
    }
    let fault = "declare void @air.atomic.global.store.i32(i32 addrspace(1)*, i32, i32, i32, i1)\n";
    if definitions.contains_key("air.atomic.global.store.i32") {
        return Err("native command fault atomic has a function body".into());
    }
    if let Some(existing) = declarations.get("air.atomic.global.store.i32") {
        let at = existing
            .find('@')
            .ok_or("native fault atomic has no symbol")?;
        let (_, open) = parse_global_symbol_with_params(existing, at, "native fault atomic")?;
        let close = matching_paren(existing, open)
            .ok_or("native fault atomic has unbalanced parameters")?;
        if existing[..open].trim() != "declare void @air.atomic.global.store.i32"
            || split_top_level(&existing[open + 1..close], ',')
                != ["i32 addrspace(1)*", "i32", "i32", "i32", "i1"]
        {
            return Err("native command fault atomic has an incompatible declaration".into());
        }
    }
    let mut rewritten_roots = BTreeMap::new();
    let mut extra_metadata = String::new();
    let mut entries = BTreeSet::new();
    for stage in ["!air.kernel", "!air.vertex", "!air.fragment"] {
        for root in nodes
            .get(stage)
            .into_iter()
            .flat_map(|body| split_top_level(body, ','))
        {
            let body = nodes.get(root).ok_or("native stage root is missing")?;
            let mut fields = split_top_level(body, ',')
                .into_iter()
                .map(str::to_string)
                .collect::<Vec<_>>();
            let first = fields.first().ok_or("native stage root has no function")?;
            let globals = global_names(first)?;
            if globals.len() != 1 {
                return Err("native stage root must name one function".into());
            }
            let name = globals.first().unwrap();
            if !needed.contains(name) {
                continue;
            }
            let header = definitions
                .get(name)
                .ok_or("native producer has no definition")?;
            let function = parse_function_header(header)?;
            let argument_list = fields
                .get(2)
                .ok_or("native stage root has no argument list")?;
            let arguments = nodes
                .get(argument_list.as_str())
                .ok_or("native entry argument list is missing")?;
            for argument in split_top_level(arguments, ',')
                .into_iter()
                .filter(|s| !s.is_empty())
            {
                let info = split_top_level(
                    nodes
                        .get(argument)
                        .ok_or("native argument metadata is missing")?,
                    ',',
                );
                let buffer = [
                    "!\"air.buffer\"",
                    "!\"air.indirect_buffer\"",
                    "!\"air.visible_function_table\"",
                    "!\"air.intersection_function_table\"",
                    "!\"air.acceleration_structure\"",
                ]
                .iter()
                .any(|role| info.contains(role));
                if buffer {
                    let location = info
                        .iter()
                        .position(|field| *field == "!\"air.location_index\"")
                        .ok_or("native buffer has no location index")?;
                    let integer = |index: usize| {
                        info.get(index)
                            .and_then(|field| field.strip_prefix("i32 "))
                            .and_then(|value| value.parse::<u32>().ok())
                    };
                    let base = integer(location + 1)
                        .ok_or("native buffer location is not an unsigned integer")?;
                    let count =
                        integer(location + 2).ok_or("native buffer location range is missing")?;
                    if u64::from(base) <= u64::from(policy_buffer_index)
                        && u64::from(policy_buffer_index) < u64::from(base) + u64::from(count)
                    {
                        return Err(
                            "native policy buffer binding collides with an existing resource"
                                .into(),
                        );
                    }
                }
            }
            let argument = max_metadata
                .checked_add(1)
                .ok_or("native metadata identifier overflow")?;
            let list = argument
                .checked_add(1)
                .ok_or("native metadata identifier overflow")?;
            max_metadata = list;
            let open = first
                .find('(')
                .ok_or("native entry metadata requires a typed function pointer")?;
            fields[0] = append_native_argument(first, open, "i32 addrspace(1)*")?;
            fields[2] = format!("!{list}");
            rewritten_roots.insert(root, format!("{root} = !{{{}}}", fields.join(", ")));
            let separator = if arguments.trim().is_empty() {
                ""
            } else {
                ", "
            };
            writeln!(extra_metadata, "!{list} = !{{{arguments}{separator}!{argument}}}\n!{argument} = !{{i32 {}, !\"air.buffer\", !\"air.location_index\", i32 {policy_buffer_index}, i32 1, !\"air.read_write\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"native_policy\"}}", function.params.len()).unwrap();
            entries.insert(name.clone());
        }
    }
    if entries.is_empty() {
        return Err("native command producer has no stage entry metadata".into());
    }
    let mut output = String::with_capacity(source.len());
    current = None;
    for raw in source.split_inclusive('\n') {
        let line = strip_comment(raw).trim();
        if let Some((node, _)) = line.split_once(" = ") {
            if let Some(rewritten) = rewritten_roots.get(node) {
                writeln!(output, "{rewritten}").unwrap();
                continue;
            }
        }
        let globals = global_names(line)?;
        if line.starts_with("define ") {
            let function = parse_function_header(line)?;
            current = Some(function.name.clone());
            if needed.contains(&function.name) {
                let (_, open) = parse_global_symbol_with_params(
                    line,
                    line.find('@').unwrap(),
                    "native definition",
                )?;
                let rewritten =
                    append_native_argument(line, open, &format!("i32 addrspace(1)* {parameter}"))?;
                writeln!(output, "{}", without_memory_contracts(&rewritten)?).unwrap();
                continue;
            }
        } else if line == "}" {
            current = None;
        }
        if let Some(call) = resolve_call(line) {
            if needed.contains(&call.callee) || helpers.contains_key(&call.callee) {
                if !current.as_ref().is_some_and(|name| needed.contains(name)) {
                    return Err("native policy call has no producer parameter".into());
                }
                let at = line
                    .find('@')
                    .ok_or("native policy call must have a direct callee")?;
                let (_, open) = parse_global_symbol_with_params(line, at, "native policy call")?;
                let close = matching_paren(line, open)
                    .ok_or("native policy call has unbalanced arguments")?;
                if global_names(&line[open + 1..close])?
                    .iter()
                    .any(|name| needed.contains(name) || helpers.contains_key(name))
                {
                    return Err("native producer or setter escaped through a call argument".into());
                }
                if let Some((name, guard)) = helpers.get(&call.callee) {
                    let prefix = line[..at].trim();
                    if !matches!(prefix, "call void" | "tail call void" | "notail call void")
                        || globals.len() != 1
                    {
                        return Err("native setter requires an ordinary direct void call without symbol operands".into());
                    }
                    let arguments = split_top_level(&line[open + 1..close], ',');
                    if arguments.len() != guard.parameters.len() {
                        return Err("native setter call differs from its declared ABI".into());
                    }
                    let mut operands = Vec::new();
                    for (operand, expected) in arguments.into_iter().zip(&guard.parameters) {
                        let (_, spelling) = llvm::parse_type_prefix(operand)?;
                        if canonical_type(&spelling)? != canonical_type(expected)? {
                            return Err(
                                "native setter call has an incompatible typed pointer".into()
                            );
                        }
                        let words = crate::air::text::split_top_level_whitespace(operand);
                        let mut value =
                            crate::air::text::split_top_level_whitespace(&spelling).len();
                        while words
                            .get(value)
                            .is_some_and(|attribute| parameter_hint(attribute))
                        {
                            value += 1;
                        }
                        let value = words[value..].join(" ");
                        llvm::parse_value(&value)?;
                        operands.push(format!("{spelling} {value}"));
                    }
                    if line[close + 1..].split_whitespace().any(|word| {
                        word != "nounwind"
                            && !word.strip_prefix('#').is_some_and(|n| {
                                !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit())
                            })
                    }) {
                        return Err(
                            "native setter has unsupported trailing attributes or operand bundles"
                                .into(),
                        );
                    }
                    operands.push(format!("i32 addrspace(1)* {parameter}"));
                    writeln!(output, "  call void @{name}({})", operands.join(", ")).unwrap();
                } else {
                    if line[..at].split_whitespace().any(|word| word == "musttail") {
                        return Err(
                            "native policy threading does not support musttail calls".into()
                        );
                    }
                    let rewritten = append_native_argument(
                        line,
                        open,
                        &format!("i32 addrspace(1)* {parameter}"),
                    )?;
                    writeln!(output, "  {}", without_memory_contracts(&rewritten)?).unwrap();
                }
                continue;
            }
        }
        if !line.starts_with("define ") && globals.iter().any(|name| needed.contains(name)) {
            return Err("native producer escaped a direct call or stage entry".into());
        }
        if line.starts_with("attributes ")
            || line.starts_with("declare ")
            || line.starts_with("define ")
            || resolve_call(line).is_some()
        {
            writeln!(output, "{}", without_memory_contracts(line)?).unwrap();
        } else {
            output.push_str(raw);
        }
    }
    if !output.ends_with('\n') {
        output.push('\n');
    }
    if !declarations.contains_key("air.atomic.global.store.i32") {
        output.push_str(fault);
    }
    output.push_str(&NativeCommandPolicies::lookup_source(
        &lookup_name,
        registration_count,
    ));
    for (_, guard) in helpers.into_values() {
        output.push_str(&guard.source);
    }
    output.push_str(&extra_metadata);
    Ok(Cow::Owned(output))
}

/// Check references before parsing unrelated native types or constructing call-graph state.
fn native_setter_uses(source: &str) -> Result<bool, String> {
    let mut used = false;
    for raw in source.lines() {
        let line = strip_comment(raw).trim();
        if line.starts_with("declare ") {
            continue;
        }
        let setters = global_names(line)?
            .into_iter()
            .filter(|name| CommandEncoderOperation::from_symbol(name).is_some())
            .collect::<BTreeSet<_>>();
        if !setters.is_empty() {
            let call = resolve_call(line).ok_or("native setter escaped a direct call")?;
            if setters.len() != 1 || !setters.contains(&call.callee) {
                return Err("native setter escaped a direct call".into());
            }
            used = true;
        }
    }
    Ok(used)
}

fn append_native_argument(line: &str, open: usize, argument: &str) -> Result<String, String> {
    let close = matching_paren(line, open).ok_or("native parameter list is unbalanced")?;
    if split_top_level(&line[open + 1..close], ',').contains(&"...") {
        return Err("native policy threading does not support variadic functions".into());
    }
    let separator = if line[open + 1..close].trim().is_empty() {
        ""
    } else {
        ", "
    };
    Ok(format!(
        "{}{separator}{argument}{}",
        &line[..close],
        &line[close..]
    ))
}

fn local_symbols(line: &str) -> Result<Vec<(std::ops::Range<usize>, String)>, String> {
    let mut out = Vec::new();
    let mut offset = 0;
    let bytes = line.as_bytes();
    while offset < bytes.len() {
        let start = offset;
        if bytes[offset] == b'%' {
            offset += 1;
            if bytes.get(offset) == Some(&b'"') {
                let (_, end) = parse_quoted_global_symbol(line, offset)?;
                offset = end;
            } else {
                while bytes
                    .get(offset)
                    .is_some_and(|b| b.is_ascii_alphanumeric() || b"-.$_".contains(b))
                {
                    offset += 1;
                }
            }
            let raw = &line[start + 1..offset];
            let mut name = String::from("%\"");
            let mut cursor = 0;
            let bytes = raw
                .strip_prefix('"')
                .and_then(|s| s.strip_suffix('"'))
                .unwrap_or(raw)
                .as_bytes();
            while cursor < bytes.len() {
                let byte = if bytes[cursor] == b'\\' {
                    let digits = bytes
                        .get(cursor + 1..cursor + 3)
                        .ok_or("truncated native type escape")?;
                    let digits =
                        std::str::from_utf8(digits).map_err(|_| "invalid native type escape")?;
                    cursor += 3;
                    u8::from_str_radix(digits, 16).map_err(|_| "invalid native type escape")?
                } else {
                    cursor += 1;
                    bytes[cursor - 1]
                };
                if (32..127).contains(&byte) && !matches!(byte, b'"' | b'\\') {
                    name.push(char::from(byte));
                } else {
                    write!(name, "\\{byte:02X}").unwrap();
                }
            }
            name.push('"');
            out.push((start..offset, name));
        } else if bytes[offset] == b'"' {
            let (_, end) = parse_quoted_global_symbol(line, offset)?;
            offset = end;
        } else {
            offset += 1;
        }
    }
    Ok(out)
}

/// Guarded callees access policy memory in addition to the original native setter effects.
/// Keep target/ABI attributes, but weaken memory and speculation promises on all callers too.
fn without_memory_contracts(line: &str) -> Result<Cow<'_, str>, String> {
    let bytes = line.as_bytes();
    let mut cursor = 0;
    let mut retained = 0;
    let mut out = String::new();
    while cursor < bytes.len() {
        let start = cursor;
        if bytes[cursor] == b'"' {
            cursor = parse_quoted_global_symbol(line, cursor)?.1;
        } else if matches!(bytes[cursor], b'%' | b'@') {
            cursor += 1;
            if bytes.get(cursor) == Some(&b'"') {
                cursor = parse_quoted_global_symbol(line, cursor)?.1;
            } else {
                while bytes
                    .get(cursor)
                    .is_some_and(|b| b.is_ascii_alphanumeric() || b"_.$-".contains(b))
                {
                    cursor += 1;
                }
            }
        } else if bytes[cursor].is_ascii_alphabetic() {
            while bytes
                .get(cursor)
                .is_some_and(|b| b.is_ascii_alphanumeric() || *b == b'_')
            {
                cursor += 1;
            }
            let word = &line[start..cursor];
            let remove = matches!(
                word,
                "readnone"
                    | "readonly"
                    | "writeonly"
                    | "argmemonly"
                    | "inaccessiblememonly"
                    | "inaccessiblemem_or_argmemonly"
                    | "nosync"
                    | "speculatable"
            );
            if word == "memory" && bytes.get(cursor) == Some(&b'(') {
                cursor = matching_paren(line, cursor)
                    .ok_or("native memory attribute is unbalanced")?
                    + 1;
            } else if !remove {
                continue;
            }
            out.push_str(&line[retained..start]);
            retained = cursor;
        } else {
            cursor += 1;
        }
    }
    if retained == 0 {
        return Ok(Cow::Borrowed(line));
    }
    out.push_str(&line[retained..]);
    Ok(Cow::Owned(out))
}

fn parameter_hint(attribute: &str) -> bool {
    matches!(
        attribute,
        "nocapture" | "readonly" | "writeonly" | "readnone" | "nonnull" | "noundef" | "noalias"
    )
}

fn canonical_type(spelling: &str) -> Result<String, String> {
    let mut result = String::new();
    let mut cursor = 0;
    for (range, name) in local_symbols(spelling)? {
        result.push_str(&spelling[cursor..range.start]);
        result.push_str(&name);
        cursor = range.end;
    }
    result.push_str(&spelling[cursor..]);
    Ok(result)
}

/// One native helper and the source operand types used to validate calls before extraction.
#[derive(Debug)]
struct NativeCommandGuard {
    source: String,
    parameters: Vec<String>,
}

/// Build a typed native helper with the same descriptor checks as host-record producers.
/// The module planner owns fresh global names and exact wrapper type spelling.
fn native_command_guard(
    declaration: &str,
    name: &str,
    lookup_name: &str,
) -> Result<NativeCommandGuard, String> {
    use crate::air::text::split_top_level_whitespace;
    use llvm::parse_type_prefix;
    use std::fmt::Write;
    let text = strip_comment(declaration).trim();
    if !text.starts_with("declare ") {
        return Err("native command guard requires an AIR declaration".into());
    }
    let at = text
        .find('@')
        .ok_or("native command declaration has no symbol")?;
    if text[..at].trim() != "declare void" {
        return Err("native command guard requires the default AIR calling convention".into());
    }
    let parsed = parse_declaration(text)?;
    let operation = CommandEncoderOperation::from_symbol(&parsed.name)
        .ok_or("native guard requires a recognized command operation")?;
    let expected = parse_function_header(
        &operation
            .signature(name)
            .ok_or("native guard cannot represent this pointer encoding")?,
    )?;
    if parsed.ret != LlType::Void
        || parsed
            .params
            .iter()
            .ne(expected.params.iter().map(|(_, ty)| ty))
    {
        return Err("native command declaration does not match its AIR ABI signature".into());
    }
    let (_, open) = parse_global_symbol_with_params(text, text.find('@').unwrap(), "native guard")?;
    let close = matching_paren(text, open).ok_or("unmatched native guard declaration")?;
    let types = split_top_level(&text[open + 1..close], ',')
        .into_iter()
        .map(|param| {
            let (_, spelling) = parse_type_prefix(param)?;
            let type_words = split_top_level_whitespace(&spelling).len();
            if split_top_level_whitespace(param)
                .into_iter()
                .skip(type_words)
                .any(|attribute| !parameter_hint(attribute))
            {
                return Err("native command parameter has an unsupported ABI attribute".into());
            }
            Ok(spelling)
        })
        .collect::<Result<Vec<_>, String>>()?;
    if parsed
        .params
        .iter()
        .zip(&types)
        .any(|(ty, spelling)| matches!(ty, LlType::Ptr(_)) && !spelling.ends_with('*'))
    {
        return Err("native command guards require typed pointer declarations".into());
    }
    if operation
        .typed_pointers()
        .into_iter()
        .any(|(operand, spelling)| types.get(operand) != Some(&spelling))
    {
        return Err("native command pointer operand differs from its typed ABI suffix".into());
    }
    let native_pointer_type = types[0].clone();
    let tail = types
        .iter()
        .zip(&expected.params)
        .skip(1)
        .map(|(ty, (parameter, _))| format!(", {ty} {parameter}"))
        .collect::<String>();
    let mut source = format!("define internal void @{name}({native_pointer_type} %native{tail}, i32 addrspace(1)* %table) alwaysinline nounwind {{\nentry:\n  %key = ptrtoint {native_pointer_type} %native to i64\n  %records = call i32 addrspace(1)* @{lookup_name}(i32 addrspace(1)* %table, i64 %key)\n  %found = icmp ne i32 addrspace(1)* %records, null\n  br i1 %found, label %policy_check, label %missing\nmissing:\n  ret void\npolicy_check:\n");
    // A copy looks its source command buffer up too: one the table does not hold faults the table,
    // as the command's does.
    if operation == CommandEncoderOperation::Copy {
        if types.get(2) != Some(&native_pointer_type) {
            return Err("a copy's source command buffer is not of its command's type".into());
        }
        writeln!(source, "  %source_key = ptrtoint {native_pointer_type} %source to i64\n  %source_records = call i32 addrspace(1)* @{lookup_name}(i32 addrspace(1)* %table, i64 %source_key)\n  %source_found = icmp ne i32 addrspace(1)* %source_records, null\n  br i1 %source_found, label %copy_check, label %missing\ncopy_check:").unwrap();
    }
    operation
        .emit_policy_checks(&mut source, true)
        .ok_or("native guard cannot represent this operation")?;
    writeln!(
        source,
        "  call void @{}({native_pointer_type} %native{tail})\n  ret void\n}}",
        parsed.name
    )
    .unwrap();
    Ok(NativeCommandGuard {
        source,
        parameters: types,
    })
}

/// Include definitions, declarations, and referenced globals. Ignore comments and ordinary string
/// literals, but decode quoted global identifiers using the same rules as the native parser.
fn global_names(source: &str) -> Result<BTreeSet<String>, String> {
    let mut names = BTreeSet::new();
    for raw in source.lines() {
        let line = strip_comment(raw);
        let bytes = line.as_bytes();
        let mut offset = 0;
        while offset < bytes.len() {
            match bytes[offset] {
                b'@' if bytes.get(offset + 1) == Some(&b'"') => {
                    let (name, end) = parse_quoted_global_symbol(line, offset + 1)?;
                    names.insert(name);
                    offset = end;
                }
                b'@' => {
                    let start = offset + 1;
                    offset = start;
                    while bytes
                        .get(offset)
                        .is_some_and(|byte| byte.is_ascii_alphanumeric() || b"-.$_".contains(byte))
                    {
                        offset += 1;
                    }
                    names.insert(line[start..offset].to_string());
                }
                b'"' => {
                    offset += 1;
                    while offset < bytes.len() {
                        match bytes[offset] {
                            b'\\' => offset = (offset + 2).min(bytes.len()),
                            b'"' => {
                                offset += 1;
                                break;
                            }
                            _ => offset += 1,
                        }
                    }
                }
                _ => offset += 1,
            }
        }
    }
    Ok(names)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SOURCE: &str = r#"%Command = type opaque
%Owner = type { %Command addrspace(1)* }
declare void @air.set_barrier_compute_command(%Command addrspace(1)*, i32)
define void @producer(%Owner addrspace(1)* %owner) {
entry:
  %field = getelementptr %Owner, %Owner addrspace(1)* %owner, i64 0, i32 0
  %command = load %Command addrspace(1)*, %Command addrspace(1)* addrspace(1)* %field, align 8
  call void @air.set_barrier_compute_command(%Command addrspace(1)* %command, i32 0)
  ret void
}
!air.kernel = !{!0}
!0 = !{void (%Owner addrspace(1)*)* @producer, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 0, !"air.indirect_buffer", !"air.buffer_size", i32 8, !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.struct_type_info", !4, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"Owner", !"air.arg_name", !"owner"}
!4 = !{i32 0, i32 8, i32 0, !"command_buffer", !"command", !"air.indirect_argument", !5}
!5 = !{i32 0, !"air.command_buffer", !"air.location_index", i32 0, i32 1, !"air.arg_type_name", !"command_buffer", !"air.arg_name", !"command"}
"#;

    #[test]
    fn absent_policy_refuses_setter_uses_but_not_unused_declarations() {
        assert!(native_guarded_source(SOURCE, None)
            .unwrap_err()
            .contains("policy binding"));
        let unused = SOURCE
            .lines()
            .filter(|line| !line.trim().starts_with("call void @air."))
            .collect::<Vec<_>>()
            .join("\n");
        assert!(matches!(
            native_guarded_source(&unused, None).unwrap(),
            Cow::Borrowed(_)
        ));
        let escaped = format!("@saved = global void (%Command addrspace(1)*, i32)* @air.set_barrier_compute_command\n{unused}");
        assert!(native_guarded_source(&escaped, None)
            .unwrap_err()
            .contains("escaped"));
        let linked = SOURCE.replace("!air.kernel = !{!0}", "");
        assert!(native_guarded_source(&linked, None)
            .unwrap_err()
            .contains("policy binding"));
    }

    #[test]
    fn original_resource_bytes_and_metadata_are_preserved() {
        let call = "  call void @air.set_barrier_compute_command(%Command addrspace(1)* %command, i32 0)\n";
        let source = SOURCE.replace(call, &call.repeat(512));
        let guarded = native_guarded_source(&source, Some((1, 1))).unwrap();
        assert_eq!(
            guarded
                .matches("define internal void @__m2v_native_command_")
                .count(),
            1
        );
        assert_eq!(
            guarded.matches("call void @__m2v_native_command_").count(),
            512
        );
        assert!(guarded.contains("%Command = type opaque"));
        assert!(guarded.contains("%Owner = type { %Command addrspace(1)* }"));
        assert!(guarded.contains("%owner, i32 addrspace(1)* %__m2v_policy_0)"));
        assert!(guarded.contains("void (%Owner addrspace(1)*, i32 addrspace(1)*)* @producer"));
        for line in SOURCE.lines().filter(|line| {
            line.starts_with("!3 =") || line.starts_with("!4 =") || line.starts_with("!5 =")
        }) {
            assert!(guarded.lines().any(|actual| actual == line));
        }
        let erased = SOURCE.replace("  ret void", "  %raw = bitcast %Owner addrspace(1)* %owner to i64 addrspace(1)*\n  %value = load i64, i64 addrspace(1)* %raw\n  ret void");
        assert!(native_guarded_source(&erased, Some((1, 1)))
            .unwrap()
            .contains("%value = load i64, i64 addrspace(1)* %raw"));
    }

    #[test]
    fn direct_calls_forward_policy_and_address_escapes_are_refused() {
        let call =
            "call void @air.set_barrier_compute_command(%Command addrspace(1)* %command, i32 0)";
        let source = format!("{}\ndefine internal void @helper(%Command addrspace(1)* %command) readonly {{\n  {call}\n  ret void\n}}\n", SOURCE.replace(call, "call void @helper(%Command addrspace(1)* %command)"));
        let guarded = native_guarded_source(&source, Some((1, 1))).unwrap();
        assert!(guarded.contains(
            "@helper(%Command addrspace(1)* %command, i32 addrspace(1)* %__m2v_policy_0)"
        ));
        assert!(!guarded.contains("readonly"));
        for escaped in [
            format!("@saved = global void (%Command addrspace(1)*)* @helper\n{source}"),
            source.replace("call void @helper(", "musttail call void @helper("),
            source.replace("%command) readonly", "%command, ...) readonly"),
        ] {
            assert!(native_guarded_source(&escaped, Some((1, 1))).is_err());
        }
    }

    #[test]
    fn policy_bindings_and_names_cannot_collide() {
        assert!(native_guarded_source(SOURCE, Some((1, 0)))
            .unwrap_err()
            .contains("collides"));
        assert!(native_guarded_source(SOURCE, Some((1, 31))).is_err());
        let source = SOURCE.replace("%owner", "%__m2v_policy_0");
        assert!(native_guarded_source(&source, Some((1, 1)))
            .unwrap()
            .contains("i32 addrspace(1)* %__m2v_policy_1"));
        let quoted = SOURCE
            .replace("%Command", "%\"command type\"")
            .replace("@producer", "@\"producer name\"");
        assert!(native_guarded_source(&quoted, Some((1, 1)))
            .unwrap()
            .contains("%\"command type\" = type opaque"));
        assert_eq!(
            append_native_argument("define void @empty() {", 18, "i32 %extra").unwrap(),
            "define void @empty(i32 %extra) {"
        );
    }

    #[test]
    fn malformed_setter_calls_and_atomic_contracts_remain_refusals() {
        for source in [
            SOURCE.replace("%command, i32 0)", "%command, i64 0)"),
            SOURCE.replace("%command, i32 0)", "%command, i32 byval(i32) 0)"),
            SOURCE.replace("%command, i32 0)", "%command, i32 0) [ \"deopt\"(i32 0) ]"),
            SOURCE.replace("call void @air.", "musttail call void @air."),
            format!("@saved = global void (%Command addrspace(1)*, i32)* @air.set_barrier_compute_command\n{SOURCE}"),
            format!("declare void @air.atomic.global.store.i32(i64 addrspace(1)*, i32, i32, i32, i1)\n{SOURCE}"),
            format!("define void @air.atomic.global.store.i32(i32 addrspace(1)* %p, i32 %v, i32 %o, i32 %s, i1 %f) {{\n ret void\n}}\n{SOURCE}"),
            format!("target datalayout = \"e-p1:32:32\"\n{SOURCE}"),
            SOURCE.replace("!air.kernel = !{!0}", "!air.kernel = !{!99}"),
            SOURCE.replace("!air.kernel = !{!0}", ""),
            SOURCE.replace("!\"air.location_index\", i32 0, i32 1", "!\"air.location_index\", i32 0"),
        ] { assert!(native_guarded_source(&source, Some((1, 1))).is_err(), "{source}"); }
    }

    #[test]
    fn metadata_only_modules_do_not_need_policy_transport() {
        let source = SOURCE
            .lines()
            .filter(|line| !line.contains("@air.set_barrier_compute_command"))
            .collect::<Vec<_>>()
            .join("\n");
        assert!(matches!(
            native_guarded_source(&source, Some((1, 1))).unwrap(),
            Cow::Borrowed(_)
        ));
        assert_eq!(
            native_guarded_source(&source, Some((1, 1))).unwrap(),
            source
        );
    }

    #[test]
    fn native_helpers_preserve_typed_pointers_and_reject_incompatible_abis() {
        let declaration = r#"declare void @air.set_kernel_buffer_compute_command.p2i8(%"native command" addrspace(1)* nocapture, i32, i8 addrspace(2)* readonly, i64, i32)"#;
        let guard = native_command_guard(declaration, "guard", "lookup").unwrap();
        assert!(guard
            .source
            .contains(r#"%"native command" addrspace(1)* %native"#));
        assert!(guard
            .source
            .contains("i8 addrspace(2)* %buffer, i64 %stride, i32 %binding"));
        assert!(guard.source.contains("alwaysinline nounwind"));
        assert!(guard.source.contains("%stride_valid"));
        for invalid in [
            declaration.replace("addrspace(2)", "addrspace(1)"),
            declaration.replace("i64", "i32"),
            declaration.replace("declare void", "declare i32"),
            declaration.replace("declare void", "declare fastcc void"),
            declaration.replace("nocapture", "byval(i32)"),
            declaration.replace(r#"%"native command" addrspace(1)*"#, "ptr addrspace(1)"),
        ] {
            assert!(native_command_guard(&invalid, "guard", "lookup").is_err());
        }
    }

    /// An indexed draw's guard forwards every operand, its index buffer of the overload's index
    /// type and space, and checks the header for `RENDER` and one of `DRAW_INDEXED`, `DRAW_PATCHES`
    /// and `DRAW_INDEXED_PATCHES`, whose records hold it; a declaration whose index pointer differs
    /// from the overload is refused.
    #[test]
    fn indexed_draw_guards_check_their_draw_kind_and_index_type() {
        let declaration = "declare void @air.draw_indexed_primitives_render_command.p1i16(%Command addrspace(1)* nocapture, i32, i32, i32, i16 addrspace(1)*, i32, i32, i32)";
        let guard = native_command_guard(declaration, "guard", "lookup").unwrap();
        assert!(guard.source.contains(
            "i32 %primitive, i32 %count, i16 addrspace(1)* %indices, i32 %instances, i32 \
             %base_vertex, i32 %base_instance"
        ));
        assert!(guard.source.contains("%policy_bits = and i32 %policy, 2"));
        assert!(guard
            .source
            .contains("%policy_valid = icmp eq i32 %policy_bits, 2"));
        assert!(guard.source.contains("%held_kinds = and i32 %policy, 1792"));
        assert!(guard.source.contains(
            "call void @air.draw_indexed_primitives_render_command.p1i16(%Command addrspace(1)* \
             %native, i32 %command, i32 %primitive, i32 %count, i16 addrspace(1)* %indices, i32 \
             %instances, i32 %base_vertex, i32 %base_instance)"
        ));
        let constant = declaration
            .replace("p1i16", "p2i32")
            .replace("i16 addrspace(1)*", "i32 addrspace(2)*");
        assert!(native_command_guard(&constant, "guard", "lookup").is_ok());
        for invalid in [
            declaration.replace("i16 addrspace(1)*", "i32 addrspace(1)*"),
            declaration.replace("i16 addrspace(1)*", "i16 addrspace(2)*"),
            declaration.replace("i16 addrspace(1)*", "i8 addrspace(1)*"),
            declaration.replace(", i32, i32, i32)", ", i32, i32)"),
        ] {
            assert!(
                native_command_guard(&invalid, "guard", "lookup").is_err(),
                "{invalid}"
            );
        }
    }

    /// A reset's guard checks the header for its family alone and forwards the command and its
    /// index.
    #[test]
    fn reset_guards_check_their_family() {
        for (family, bits) in [("compute", 1), ("render", 2)] {
            let declaration =
                format!("declare void @air.reset_{family}_command(%Command addrspace(1)*, i32)");
            let guard = native_command_guard(&declaration, "guard", "lookup").unwrap();
            assert!(guard
                .source
                .contains(&format!("%policy_bits = and i32 %policy, {bits}")));
            assert!(guard.source.contains(&format!(
                "call void @air.reset_{family}_command(%Command addrspace(1)* %native, i32 \
                 %command)"
            )));
        }
    }

    /// A stage-in region's and an imageblock size's guards check the header for compute commands
    /// and forward their vector operands.
    #[test]
    fn region_and_imageblock_guards_forward_their_vectors() {
        for (declaration, forwarded) in [
            (
                "declare void @air.set_stage_in_region_compute_command(%Command addrspace(1)*, i32, <3 x i32>, <3 x i32>)",
                "i32 %command, <3 x i32> %origin, <3 x i32> %size)",
            ),
            (
                "declare void @air.set_imageblock_size_compute_command(%Command addrspace(1)*, i32, <2 x i16>)",
                "i32 %command, <2 x i16> %size)",
            ),
        ] {
            let guard = native_command_guard(declaration, "guard", "lookup").unwrap();
            assert!(guard.source.contains("%policy_bits = and i32 %policy, 1"));
            assert!(guard.source.contains(forwarded), "{}", guard.source);
        }
        assert!(native_command_guard(
            "declare void @air.set_imageblock_size_compute_command(%Command addrspace(1)*, i32, <2 x i32>)",
            "guard",
            "lookup"
        )
        .is_err());
    }

    /// A copy's guard looks its source command buffer up in the table too, checks the destination
    /// for compute commands, the source index against the source's capacity, the two headers'
    /// descriptors and (into an inheriting buffer) the destination index against 0 as the product's
    /// lowering does, and forwards both commands; a source of another type than the command's is
    /// refused.
    #[test]
    fn copy_guards_look_their_source_up_and_compare_descriptors() {
        let declaration = "declare void @air.copy_compute_command(%Command addrspace(1)* nocapture, i32, %Command addrspace(1)* nocapture readonly, i32)";
        let guard = native_command_guard(declaration, "guard", "lookup").unwrap();
        for line in [
            "%source_key = ptrtoint %Command addrspace(1)* %source to i64",
            "%source_records = call i32 addrspace(1)* @lookup(i32 addrspace(1)* %table, i64 %source_key)",
            "br i1 %source_found, label %copy_check, label %missing",
            "%policy_bits = and i32 %policy, 1",
            "%source_present = icmp ult i32 %source_command, %source_capacity",
            "%equal = and i1 %same_policy, %same_counts",
            "%inheriting_policy = or i32 %source_policy, 192",
            "%not_first = icmp ne i32 %command, 0",
            "%valid4 = and i1 %valid0, %copy_valid",
            "br i1 %valid4, label %write, label %invalid",
            "call void @air.copy_compute_command(%Command addrspace(1)* %native, i32 %command, %Command addrspace(1)* %source, i32 %source_command)",
        ] {
            assert!(guard.source.contains(line), "{line}\n{}", guard.source);
        }
        for invalid in [
            declaration.replace("nocapture readonly, i32)", "nocapture readonly, i64)"),
            declaration.replace(
                "%Command addrspace(1)* nocapture readonly",
                "%Other addrspace(1)*",
            ),
            declaration.replace(
                "%Command addrspace(1)* nocapture readonly",
                "i8 addrspace(1)*",
            ),
        ] {
            assert!(
                native_command_guard(&invalid, "guard", "lookup").is_err(),
                "{invalid}"
            );
        }
    }

    /// A patch draw's guard forwards its patch index buffer, its factor buffer and its scale,
    /// checks the header for `RENDER` and one of `DRAW_PATCHES` and `DRAW_INDEXED_PATCHES`, and
    /// refuses a declaration whose pointers are not the overload's; a call with a null patch index
    /// buffer and a scale is rewritten to the guard.
    #[test]
    fn patch_draw_guards_check_their_draw_kind_and_buffers() {
        let declaration = "declare void @air.draw_patches_render_command.p1i32.p2i8(%Command addrspace(1)* nocapture, i32, i32, i32, i32, i32 addrspace(1)*, i32, i32, i8 addrspace(2)*, i32, float)";
        let guard = native_command_guard(declaration, "guard", "lookup").unwrap();
        assert!(guard.source.contains(
            "i32 addrspace(1)* %patch_indices, i32 %instances, i32 %base_instance, i8 \
             addrspace(2)* %factors, i32 %instance_stride, float %scale"
        ));
        assert!(guard.source.contains("%policy_bits = and i32 %policy, 2"));
        assert!(guard.source.contains("%held_kinds = and i32 %policy, 1536"));
        for invalid in [
            declaration.replace("i32 addrspace(1)*", "i16 addrspace(1)*"),
            declaration.replace("i8 addrspace(2)*", "i8 addrspace(1)*"),
            declaration.replace(", i32, float)", ", i32, double)"),
            declaration.replace(", i32, float)", ", i32)"),
        ] {
            assert!(
                native_command_guard(&invalid, "guard", "lookup").is_err(),
                "{invalid}"
            );
        }
        let call = "  call void @air.set_barrier_compute_command(%Command addrspace(1)* %command, i32 0)\n";
        let source = SOURCE
            .replace(
                "declare void @air.set_barrier_compute_command(%Command addrspace(1)*, i32)",
                "declare void @air.draw_patches_render_command.p1i32.p1i8(%Command addrspace(1)*, i32, i32, i32, i32, i32 addrspace(1)*, i32, i32, i8 addrspace(1)*, i32, float)",
            )
            .replace(
                call,
                "  call void @air.draw_patches_render_command.p1i32.p1i8(%Command addrspace(1)* %command, i32 0, i32 4, i32 0, i32 2, i32 addrspace(1)* null, i32 1, i32 0, i8 addrspace(1)* null, i32 0, float 5.000000e-01)\n",
            );
        let guarded = native_guarded_source(&source, Some((1, 1))).unwrap();
        assert!(guarded.contains(
            "(%Command addrspace(1)* %command, i32 0, i32 4, i32 0, i32 2, i32 addrspace(1)* \
             null, i32 1, i32 0, i8 addrspace(1)* null, i32 0, float 5.000000e-01, i32 \
             addrspace(1)* %__m2v_policy_0)"
        ));
    }
    /// An indexed patch draw's guard forwards its patch index, control-point index and factor
    /// buffers and its scale, checks the header for `RENDER` and one of `DRAW_PATCHES` and
    /// `DRAW_INDEXED_PATCHES`, and refuses a declaration whose pointers are not the overload's; a
    /// call with null patch and control-point index buffers is rewritten to the guard.
    #[test]
    fn indexed_patch_draw_guards_check_their_draw_kind_and_buffers() {
        let declaration = "declare void @air.draw_indexed_patches_render_command.p1i32.p2i8.p1i8(%Command addrspace(1)* nocapture, i32, i32, i32, i32, i32 addrspace(1)*, i8 addrspace(2)*, i32, i32, i8 addrspace(1)*, i32, float)";
        let guard = native_command_guard(declaration, "guard", "lookup").unwrap();
        assert!(guard.source.contains(
            "i32 addrspace(1)* %patch_indices, i8 addrspace(2)* %control_point_indices, i32 \
             %instances, i32 %base_instance, i8 addrspace(1)* %factors, i32 %instance_stride, \
             float %scale"
        ));
        assert!(guard.source.contains("%policy_bits = and i32 %policy, 2"));
        assert!(guard.source.contains("%held_kinds = and i32 %policy, 1536"));
        for invalid in [
            declaration.replace("i32 addrspace(1)*", "i16 addrspace(1)*"),
            declaration.replace("i8 addrspace(2)*", "i8 addrspace(1)*"),
            declaration.replace("i8 addrspace(2)*", "i16 addrspace(2)*"),
            declaration.replace(
                "i8 addrspace(1)*, i32, float",
                "i8 addrspace(2)*, i32, float",
            ),
            declaration.replace(", i32, float)", ", i32)"),
        ] {
            assert!(
                native_command_guard(&invalid, "guard", "lookup").is_err(),
                "{invalid}"
            );
        }
        let call = "  call void @air.set_barrier_compute_command(%Command addrspace(1)* %command, i32 0)\n";
        let source = SOURCE
            .replace(
                "declare void @air.set_barrier_compute_command(%Command addrspace(1)*, i32)",
                "declare void @air.draw_indexed_patches_render_command.p1i32.p1i8.p1i8(%Command addrspace(1)*, i32, i32, i32, i32, i32 addrspace(1)*, i8 addrspace(1)*, i32, i32, i8 addrspace(1)*, i32, float)",
            )
            .replace(
                call,
                "  call void @air.draw_indexed_patches_render_command.p1i32.p1i8.p1i8(%Command addrspace(1)* %command, i32 0, i32 3, i32 0, i32 2, i32 addrspace(1)* null, i8 addrspace(1)* null, i32 1, i32 0, i8 addrspace(1)* null, i32 0, float 1.000000e+00)\n",
            );
        let guarded = native_guarded_source(&source, Some((1, 1))).unwrap();
        assert!(guarded.contains(
            "(%Command addrspace(1)* %command, i32 0, i32 3, i32 0, i32 2, i32 addrspace(1)* \
             null, i8 addrspace(1)* null, i32 1, i32 0, i8 addrspace(1)* null, i32 0, float \
             1.000000e+00, i32 addrspace(1)* %__m2v_policy_0)"
        ));
    }
}
