//! A function's own encoders (`newArgumentEncoderWithBufferIndex:`) against Metal's:
//! `metal-answers.txt`'s `encoder_` sections over the `argument_buffers` fixture, recorded by
//! `probe.swift` on an M3 Max and an M4 (where the two differ, the M3 Max's).

use std::collections::{BTreeMap, BTreeSet};

use super::{diff, fixture};
use crate::bind::{ArgumentBufferLayout, ArgumentLayoutError, ArgumentType};
use crate::budget::Budget;
use crate::error::{DiagName, Error, Feature};
use crate::library::{FunctionConstantValues, Library};
use crate::metal::{BindingAccess, DataType};
use crate::pipeline::testing::{fixtures_dir, AssembleError};
use crate::pipeline::{ComputePipelineDescriptor, FunctionDescriptor, StageSource};
use crate::reflection::{
    Binding, BufferBinding, PointerType, StructMember, StructType, TypeDetail,
};

const FUNCTION: &str = "reflection_argument_buffers";

/// The recorded `encoder_<case>` sections, by case.
fn encoder_answers() -> BTreeMap<String, Vec<String>> {
    let path = fixtures_dir().join("reflection/metal-answers.txt");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path:?}: {e}"));
    let mut out: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut case: Option<&str> = None;
    for line in text.lines().filter(|l| !l.starts_with('#')) {
        if let Some(name) = line.strip_prefix("== ") {
            case = name.strip_prefix("encoder_");
            continue;
        }
        if let Some(name) = case {
            out.entry(name.to_owned())
                .or_default()
                .push(line.to_owned());
        }
    }
    out
}

/// The function's encoder of its buffer `index`; `None` without `llvm-as`.
fn function_encoder(index: u32) -> Option<ArgumentBufferLayout> {
    let lib = Library::from_air(&fixture("argument_buffers")?).expect("library");
    let function = lib
        .function(FUNCTION)
        .expect("function")
        .argument_buffer_layout(
            index,
            &FunctionConstantValues::defaults(),
            &Budget::standard(),
        )
        .expect("the function's layout");
    Some(function)
}

/// The lines of `lines` under the probe's `function` encoder: its line and the indented ones
/// after it.
fn function_lines(lines: &[String]) -> Vec<String> {
    let mut out = Vec::new();
    let mut open = false;
    for line in lines {
        if !line.starts_with(' ') {
            open = line.starts_with("function ");
        }
        if open {
            out.push(line.clone());
        }
    }
    out
}

/// The probe's lines for an encoder: its length and alignment, the offset `constantDataAtIndex:`
/// answers for each of `constants` (the argument's offset, 0 for an index no argument has), and
/// each of `nested`'s nested encoder or nil.
fn encoder_lines(
    label: &str,
    l: &ArgumentBufferLayout,
    constants: &[u32],
    nested: &[u32],
) -> Vec<String> {
    let mut out = vec![format!(
        "{label} encodedLength={} alignment={}",
        l.encoded_length, l.alignment
    )];
    for &i in constants {
        let offset = l.argument(i).map_or(0, |p| p.offset);
        out.push(format!("  constant index={i} offset={offset}"));
    }
    for &i in nested {
        out.push(match l.argument(i).and_then(|p| p.nested) {
            Some(n) => format!(
                "  nested index={i} encodedLength={} alignment={}",
                n.encoded_length, n.alignment
            ),
            None => format!("  nested index={i} nil"),
        });
    }
    out
}

/// The function's encoders: lengths, alignments, constant offsets and nested encoders as Metal's.
#[test]
fn the_function_s_encoders_match_metal() {
    let answers = encoder_answers();
    let mut mismatches = Vec::new();
    for (case, index, name, constants, nested) in [
        ("args", 0, "args", &[7, 8, 20, 21][..], &[2, 12, 15][..]),
        ("imp", 1, "imp", &[3][..], &[0][..]),
    ] {
        let Some(l) = function_encoder(index) else {
            return;
        };
        let got = encoder_lines("function", &l, constants, nested);
        let want = function_lines(&answers[case]);
        mismatches.extend(diff(&format!("encoder_{case} {name}"), &got, &want));
    }
    assert!(mismatches.is_empty(), "{}", mismatches.join("\n"));
}

/// The arguments the probe sets, in its order: textures, then samplers, then buffers.
const PLACES: [(ArgumentType, &[u32]); 3] = [
    (ArgumentType::Texture, &[0, 3, 4, 5, 22, 24, 26]),
    (ArgumentType::Sampler, &[1, 16, 17, 23, 25, 27]),
    (ArgumentType::Buffer, &[2, 18, 19]),
];

/// Where each argument's value is found in a zeroed 512-byte argument buffer after the probe's
/// writes, each argument a distinct value at its place (a later write over an earlier one wins),
/// searched as the probe searches: 8 bytes at every multiple of 4.
fn placements(l: &ArgumentBufferLayout) -> BTreeMap<u32, BTreeSet<u64>> {
    let mut bytes = [0u8; 512];
    let mut values = Vec::new();
    for (ty, indices) in PLACES {
        for &i in indices {
            let place = l.argument(i).unwrap_or_else(|| panic!("no argument {i}"));
            assert_eq!(place.ty, ty, "argument {i}");
            let value = 0x1000 + u64::from(i);
            let at = place.offset as usize;
            bytes[at..at + 8].copy_from_slice(&value.to_le_bytes());
            values.push((i, value));
        }
    }
    values
        .into_iter()
        .map(|(i, value)| {
            let found = (0..512 - 7)
                .step_by(4)
                .filter(|&at| bytes[at..at + 8] == value.to_le_bytes())
                .map(|at| at as u64)
                .collect();
            (i, found)
        })
        .collect()
}

/// Every element and struct member lands where Metal's encoder of the function's argument buffer
/// puts it, at its layout's place. Metal's lists can also hold another argument's place (a
/// texture's and a sampler's resource IDs can be equal), so each list must be this crate's place
/// for the argument plus only places of other arguments.
#[test]
fn the_function_s_encoder_places_arguments_as_metal() {
    let answers = encoder_answers();
    let Some(function) = function_encoder(0) else {
        return;
    };
    let encoders = [("function", function)];
    let mut recorded: BTreeMap<&str, (String, BTreeMap<u32, BTreeSet<u64>>)> = BTreeMap::new();
    let mut label = "";
    for line in &answers["args_places"] {
        if let Some(rest) = line.strip_prefix("  index=") {
            let (index, list) = rest.split_once(" offsets=").expect("an offsets list");
            let offsets = list
                .trim_matches(|c| c == '[' || c == ']')
                .split(", ")
                .filter(|o| !o.is_empty())
                .map(|o| o.parse().expect("an offset"))
                .collect();
            let index = index.parse().expect("an index");
            recorded.get_mut(label).unwrap().1.insert(index, offsets);
        } else {
            label = ["function", "library", "pipeline"]
                .into_iter()
                .find(|l| line.starts_with(&format!("{l} ")))
                .expect("an encoder line");
            recorded.insert(label, (line.clone(), BTreeMap::new()));
        }
    }
    for (label, l) in &encoders {
        let (line, metal) = &recorded[label];
        assert_eq!(line, &format!("{label} encodedLength={}", l.encoded_length));
        let ours = placements(l);
        assert_eq!(
            ours.keys().collect::<Vec<_>>(),
            metal.keys().collect::<Vec<_>>()
        );
        for (i, found) in &ours {
            let others: BTreeSet<u64> = ours
                .iter()
                .filter(|(j, _)| *j != i)
                .flat_map(|(_, f)| f.iter().copied())
                .collect();
            assert!(
                found.is_subset(&metal[i])
                    && metal[i].difference(found).all(|o| others.contains(o)),
                "{label} index {i}: ours {found:?}, Metal's {:?}",
                metal[i]
            );
        }
    }
}

/// An argument buffer's member the AIR gives no argument index (a plain member of its struct
/// type info), which Apple's compiler emitted in no measured shape: reflection refuses it,
/// naming the member's type, rather than guess what Metal answers.
#[test]
fn an_argument_buffer_member_without_an_index_is_refused() {
    let path = fixtures_dir().join("reflection/encoders.ll");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path:?}: {e}"));
    let from = r#"i32 8, i32 4, i32 0, !"float", !"f", !"air.indirect_argument", !68}"#;
    assert_eq!(text.matches(from).count(), 1);
    let text = text.replace(from, r#"i32 8, i32 4, i32 0, !"float", !"f"}"#);
    let bc = match metal2vulkan_devtools::llvm_assemble_text(&text) {
        Ok(b) => b,
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            return;
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as: {why}"),
    };
    let lib = Library::from_air(&bc).expect("library");
    let f = lib.function("encoders_args").expect("function");
    match f.reflection(&Budget::standard()) {
        Err(Error::Unsupported(u)) => {
            assert_eq!(u.feature, Feature::Reflection);
            assert_eq!(u.symbol, Some(DiagName::new("float")));
        }
        other => panic!("{other:?}"),
    }
}

/// A pipeline's stride of an argument buffer's array of structs whose members are no constant
/// but hold a struct is unmeasured (encoder_arrays' `a7` holds a constant beside its struct): its
/// pipeline's reflection is refused, naming the struct, while its library's states the element
/// size.
#[test]
fn a_pipeline_array_of_structs_of_structs_is_refused() {
    let path = fixtures_dir().join("reflection/encoder_arrays.ll");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path:?}: {e}"));
    let from = r#"!"ArrTex", !"in", !"air.indirect_argument", i32 0, i32 8, i32 4, i32 0, !"float", !"c", !"air.indirect_argument", !42}"#;
    assert_eq!(text.matches(from).count(), 1);
    let text = text.replace(
        from,
        r#"!"ArrTex", !"in", !"air.indirect_argument", i32 0}"#,
    );
    let bc = match metal2vulkan_devtools::llvm_assemble_text(&text) {
        Ok(b) => b,
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            return;
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as: {why}"),
    };
    let lib = Library::from_air(&bc).expect("library");
    let f = lib.function("encoder_arrays").expect("function");
    let budget = Budget::standard();
    let library = f.reflection(&budget).expect("reflection").expect("present");
    let a7 = library
        .bindings()
        .iter()
        .find(|b| b.name() == "a7")
        .and_then(Binding::buffer)
        .and_then(|b| b.buffer_struct_type())
        .expect("a7's struct");
    let array = a7.members()[0].array_type().expect("an array");
    assert_eq!((array.array_length(), array.stride()), (2, 16));
    match ComputePipelineDescriptor::new(StageSource::Function(FunctionDescriptor::new(f)))
        .reflection(&budget)
    {
        Err(Error::Unsupported(u)) => {
            assert_eq!(u.feature, Feature::Reflection);
            assert_eq!(u.symbol, Some(DiagName::new("ArrNestTex")));
        }
        other => panic!("{other:?}"),
    }
}
