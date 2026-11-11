//! Reflection against Metal's recorded answers: the MSL type grammar, and each fixture function's
//! library reflection and each pipeline case's reflection line for line against
//! `metal-answers.txt` (`validation/fixtures/public/reflection/`, recorded by `probe.swift` there
//! on an M3 Max and an M4), rendered in the probe's format.

mod encoder;

use std::collections::BTreeMap;

use super::types::{array_element, data_type, texture, Texture};
use super::{
    ArrayType, Attribute, Binding, ComputePipelineReflection, FunctionConstant, FunctionProperties,
    FunctionReflection, PointerType, RenderPipelineReflection, StructType, TextureReferenceType,
};
use crate::budget::Budget;
use crate::error::{DescriptorError, DiagName, Error, Feature, Needs};
use crate::library::{FunctionConstantValue, FunctionConstantValues, Library};
use crate::metal::{
    AttributeFormat, DataType, PixelFormat, StepFunction, TextureType, VertexFormat,
    VertexStepFunction,
};
use crate::pipeline::testing::{fixtures_dir, llvm_assemble, AssembleError};
use crate::pipeline::{
    ColorAttachmentDescriptor, ComputePipelineDescriptor, FunctionDescriptor,
    MeshPipelineDescriptor, RenderPipelineDescriptor, StageInputAttributeDescriptor,
    StageInputDescriptor, StageInputLayoutDescriptor, StageSource, TessellationDescriptor,
    TileRenderPipelineDescriptor, VertexAttributeDescriptor, VertexBufferLayoutDescriptor,
    VertexDescriptor,
};
use metal2vulkan_devtools::metallib::{metallib, wrap, MtlbFunction};

#[test]
fn the_type_grammar_names_metal_data_types() {
    for (name, ty) in [
        ("float", Some(DataType::Float)),
        ("uint4", Some(DataType::UInt4)),
        ("half3x3", Some(DataType::Half3x3)),
        ("float2x4", Some(DataType::Float2x4)),
        ("packed_float3", Some(DataType::Float3)),
        ("bfloat4", Some(DataType::BFloat4)),
        ("bool2", Some(DataType::Bool2)),
        ("char", Some(DataType::Char)),
        ("uchar2", Some(DataType::UChar2)),
        ("ulong", Some(DataType::ULong)),
        // Matrices are float and half only; widths are 2 to 4.
        ("int2x2", None),
        ("float5", None),
        ("float1", None),
        ("ReflectionArgs", None),
        ("", None),
    ] {
        assert_eq!(data_type(name), ty, "{name}");
    }
}

#[test]
fn the_type_grammar_names_textures() {
    let t = |texture_type, data_type, depth| {
        Some(Texture {
            texture_type,
            data_type,
            depth,
        })
    };
    for (name, want) in [
        (
            "texture2d<float, sample>",
            t(TextureType::Type2D, DataType::Float, false),
        ),
        (
            "depth2d<float, sample>",
            t(TextureType::Type2D, DataType::Float, true),
        ),
        (
            "texture_buffer<float, read>",
            t(TextureType::TextureBuffer, DataType::Float, false),
        ),
        (
            "texturecube_array<half, sample>",
            t(TextureType::CubeArray, DataType::Half, false),
        ),
        (
            "texture3d<int, read>",
            t(TextureType::Type3D, DataType::Int, false),
        ),
        (
            "depth2d_ms_array<float, read>",
            t(TextureType::Type2DMultisampleArray, DataType::Float, true),
        ),
        // An OpenCL C image's: AIR names no component, Metal reflects none.
        (
            "texture1d<unknown, read>",
            t(TextureType::Type1D, DataType::None, false),
        ),
        (
            "texture2d_array<unknown, write>",
            t(TextureType::Type2DArray, DataType::None, false),
        ),
        ("texture4d<float, read>", None),
        ("texture2d<Opaque, read>", None),
        ("sampler", None),
    ] {
        assert_eq!(texture(name), want, "{name}");
    }
    assert_eq!(
        array_element("array<texture2d<float, read>, 3>"),
        "texture2d<float, read>"
    );
    assert_eq!(array_element("array<sampler, 2>"), "sampler");
    assert_eq!(
        array_element("texture2d<float, sample>"),
        "texture2d<float, sample>"
    );
}

/// Metal's library answer for each function of a fixture, by fixture and function: the
/// `library ...` line and its binding lines.
fn library_answers() -> BTreeMap<(String, String), Vec<String>> {
    let path = fixtures_dir().join("reflection/metal-answers.txt");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path:?}: {e}"));
    let mut out = BTreeMap::new();
    let mut fixture: Option<&str> = None;
    let mut function: Option<&str> = None;
    let mut current: Option<Vec<String>> = None;
    let mut finish = |fixture: Option<&str>, function: Option<&str>, lines: Option<Vec<String>>| {
        if let (Some(fx), Some(f), Some(lines)) = (fixture, function, lines) {
            out.insert((fx.to_owned(), f.to_owned()), lines);
        }
    };
    for line in text.lines().filter(|l| !l.starts_with('#')) {
        if let Some(case) = line.strip_prefix("== ") {
            finish(fixture, function, current.take());
            fixture = case.strip_prefix("functions_");
            function = None;
        } else if fixture.is_none() {
            // A pipeline case.
        } else if let Some(rest) = line.strip_prefix("function ") {
            finish(fixture, function, current.take());
            function = rest.split(' ').next();
        } else if line.starts_with("library ") {
            current = Some(vec![line.to_owned()]);
        } else if line.starts_with("  name=") {
            if let Some(lines) = current.as_mut() {
                lines.push(line.to_owned());
            }
        }
    }
    finish(fixture, function, current.take());
    out
}

/// The fixture whose AIR the translator does not decode (its mesh, object and intersection
/// functions): only its metallib's answers are compared, and its source
/// (`reflection/metallib-only/`) has no `.ll` companions.
const METALLIB_ONLY: &str = "function_types";

/// A fixture's opaque-pointer bitcode (`llvm-as` of `reflection/<stem>.ll`); `None` (skipped
/// visibly) without `llvm-as`.
fn fixture(stem: &str) -> Option<Vec<u8>> {
    match llvm_assemble(&fixtures_dir().join(format!("reflection/{stem}.ll"))) {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected {stem}.ll: {why}"),
    }
}

/// `r` in the probe's format: the `library` line and one line per binding.
fn render(r: &FunctionReflection) -> Vec<String> {
    let mut lines = vec![format!(
        "library userAnnotation={} count={}",
        r.user_annotation().unwrap_or("nil"),
        r.bindings().len()
    )];
    lines.extend(r.bindings().iter().map(|b| format!("  {}", binding(b))));
    lines
}

fn binding(b: &Binding) -> String {
    let mut p = vec![
        format!("name={}", b.name()),
        format!("type={}", b.binding_type().as_raw()),
        format!("access={}", b.access().as_raw()),
        format!("index={}", b.index()),
        format!("used={}", b.is_used()),
        format!("argument={}", b.is_argument()),
    ];
    if let Some(x) = b.buffer() {
        p.push(format!("bufferAlignment={}", x.buffer_alignment()));
        p.push(format!("bufferDataSize={}", ns(x.buffer_data_size())));
        p.push(format!("bufferDataType={}", x.buffer_data_type().as_raw()));
        p.push(format!(
            "bufferStructType={}",
            x.buffer_struct_type().map_or("nil".to_owned(), structure)
        ));
        p.push(format!(
            "bufferPointerType={}",
            x.buffer_pointer_type().map_or("nil".to_owned(), pointer)
        ));
    }
    if let Some(x) = b.texture() {
        p.push(format!("textureType={}", x.texture_type().as_raw()));
        p.push(format!(
            "textureDataType={}",
            x.texture_data_type().as_raw()
        ));
        p.push(format!("depth={}", x.is_depth_texture()));
        p.push(format!("arrayLength={}", x.array_length()));
    }
    if let Some(x) = b.threadgroup() {
        p.push(format!(
            "threadgroupMemoryAlignment={}",
            x.threadgroup_memory_alignment()
        ));
        p.push(format!(
            "threadgroupMemoryDataSize={}",
            x.threadgroup_memory_data_size()
        ));
    }
    if let Some(x) = b.object_payload() {
        p.push(format!(
            "objectPayloadAlignment={}",
            x.object_payload_alignment()
        ));
        p.push(format!(
            "objectPayloadDataSize={}",
            x.object_payload_data_size()
        ));
    }
    p.join(" ")
}

/// An `NSUInteger` as the probe prints it, through Swift's `Int` (`NSUIntegerMax` is -1).
fn ns(v: u64) -> i64 {
    i64::from_ne_bytes(v.to_ne_bytes())
}

fn structure(s: &StructType) -> String {
    let members: Vec<String> = s
        .members()
        .iter()
        .map(|m| {
            let mut p = vec![
                format!("name={}", m.name()),
                format!("offset={}", m.offset()),
                format!("dataType={}", m.data_type().as_raw()),
                format!("argumentIndex={}", m.argument_index()),
            ];
            if let Some(s) = m.struct_type() {
                p.push(format!("struct={}", structure(s)));
            }
            if let Some(a) = m.array_type() {
                p.push(format!("array={}", array(a)));
            }
            if let Some(t) = m.texture_reference_type() {
                p.push(format!("texref={}", texref(t)));
            }
            if let Some(q) = m.pointer_type() {
                p.push(format!("pointer={}", pointer(q)));
            }
            format!("({})", p.join(" "))
        })
        .collect();
    format!("{{{}}}", members.join(","))
}

fn array(a: &ArrayType) -> String {
    let mut p = vec![
        format!("elementType={}", a.element_type().as_raw()),
        format!("length={}", a.array_length()),
        format!("stride={}", a.stride()),
        format!("argumentIndexStride={}", a.argument_index_stride()),
    ];
    if let Some(s) = a.element_struct_type() {
        p.push(format!("struct={}", structure(s)));
    }
    if let Some(e) = a.element_array_type() {
        p.push(format!("array={}", array(e)));
    }
    if let Some(t) = a.element_texture_reference_type() {
        p.push(format!("texref={}", texref(t)));
    }
    if let Some(q) = a.element_pointer_type() {
        p.push(format!("pointer={}", pointer(q)));
    }
    format!("[{}]", p.join(" "))
}

fn texref(t: &TextureReferenceType) -> String {
    format!(
        "<dataType={} type={} access={} depth={}>",
        t.texture_data_type().as_raw(),
        t.texture_type().as_raw(),
        t.access().as_raw(),
        t.is_depth_texture()
    )
}

fn pointer(q: &PointerType) -> String {
    let mut p = vec![
        format!("elementType={}", q.element_type().as_raw()),
        format!("access={}", q.access().as_raw()),
        format!("alignment={}", q.alignment()),
        format!("dataSize={}", ns(q.data_size())),
        format!("argumentBuffer={}", q.element_is_argument_buffer()),
    ];
    if let Some(s) = q.element_struct_type() {
        p.push(format!("struct={}", structure(s)));
    }
    if let Some(a) = q.element_array_type() {
        p.push(format!("array={}", array(a)));
    }
    format!("*<{}>", p.join(" "))
}

/// The mismatches between `got` and Metal's `want`, line by line.
fn diff(what: &str, got: &[String], want: &[String]) -> Vec<String> {
    let mut out = Vec::new();
    for i in 0..got.len().max(want.len()) {
        let (g, w) = (got.get(i), want.get(i));
        if g != w {
            out.push(format!(
                "{what} line {i}:\n  ours:  {}\n  Metal: {}",
                g.map_or("(none)", String::as_str),
                w.map_or("(none)", String::as_str)
            ));
        }
    }
    out
}

/// Every fixture function's library reflection equals Metal's, line for line: the order and
/// presence of bindings (gated arguments listed, unreferenced ones unused), the access table,
/// buffer data types (structs, atomics as `__s`, packed vectors, matrices), every buffer as a
/// pointer, texture types and array lengths, threadgroup memory, a tile function's imageblock,
/// function tables, and argument buffers' member offsets, ids, arrays and strides.
#[test]
fn library_reflection_matches_metal() {
    let answers = library_answers();
    assert!(answers.len() >= 14, "{} answers", answers.len());
    let mut libraries: BTreeMap<String, Library> = BTreeMap::new();
    let mut mismatches = Vec::new();
    let mut refused = 0;
    for ((stem, name), want) in &answers {
        // Its AIR uses entry kinds the translator does not decode (mesh, object, intersection).
        if stem == METALLIB_ONLY {
            continue;
        }
        if !libraries.contains_key(stem) {
            let Some(bc) = fixture(stem) else {
                return;
            };
            let lib = Library::from_air(&bc).unwrap_or_else(|e| panic!("{stem}: {e:?}"));
            libraries.insert(stem.clone(), lib);
        }
        let f = libraries[stem]
            .function(name)
            .unwrap_or_else(|e| panic!("{stem} {name}: {e:?}"));
        let got = match f.reflection(&Budget::standard()) {
            Ok(Some(r)) => render(&r),
            // Indices a function constant computes: Metal's library states 28, 4294967236 and
            // 18 for `base`, `base + 2` and `base * 2`, no value the constant gives, on both
            // chips. The crate refuses them, naming the index.
            Err(Error::Unsupported(u)) if COMPUTED_INDICES.contains(&(stem, name)) => {
                assert_eq!(u.feature, Feature::Reflection, "{stem} {name}");
                refused += 1;
                continue;
            }
            other => panic!("{stem} {name}: {other:?}"),
        };
        mismatches.extend(diff(&format!("{stem} {name}"), &got, want));
    }
    assert_eq!(
        refused,
        COMPUTED_INDICES.len(),
        "each computed-index function refused"
    );
    assert!(mismatches.is_empty(), "{}", mismatches.join("\n"));
}

/// The fixture functions whose arguments' indices a function constant computes, by fixture and
/// function.
const COMPUTED_INDICES: [(&str, &str); 1] = [("leftovers", "reflection_computed_indices")];

/// A metallib function answers only with its `RFLT` tag, as Metal answers nil without it; a
/// metallib's visible function answers no bindings.
#[test]
fn a_metallib_function_answers_with_its_reflection_tag_only() {
    let Some(bc) = fixture("tables") else {
        return;
    };
    let answers = library_answers();
    let make = |tags: &[([u8; 4], Vec<u8>)]| {
        let bytes = metallib(&[
            MtlbFunction {
                name: "reflection_tables",
                type_code: 2,
                bitcode: wrap(&bc),
                constants: vec![],
                public: vec![],
                tags: tags.to_vec(),
            },
            // `TYPE` 4: a visible function, as the compiler driver tags one.
            MtlbFunction {
                name: "reflection_visible",
                type_code: 4,
                bitcode: wrap(&bc),
                constants: vec![],
                public: vec![],
                tags: tags.to_vec(),
            },
        ]);
        Library::parse(&bytes).expect("metallib")
    };
    let tagged = make(&[(*b"RFLT", 0u64.to_le_bytes().to_vec())]);
    let reflect = |lib: &Library, name: &str| {
        lib.function(name)
            .expect("function")
            .reflection(&Budget::standard())
            .unwrap_or_else(|e| panic!("{name}: {e:?}"))
    };
    let kernel = reflect(&tagged, "reflection_tables").expect("a tagged kernel answers");
    let key = ("tables".to_owned(), "reflection_tables".to_owned());
    assert_eq!(render(&kernel), answers[&key]);
    let visible = reflect(&tagged, "reflection_visible").expect("a tagged visible function");
    assert!(visible.bindings().is_empty(), "{visible:?}");

    let untagged = make(&[]);
    assert_eq!(reflect(&untagged, "reflection_tables"), None);
    assert_eq!(reflect(&untagged, "reflection_visible"), None);
}

/// An argument of a type the MSL type grammar does not cover is refused, naming the type, never
/// guessed.
#[test]
fn an_unknown_type_is_refused() {
    let path = fixtures_dir().join("reflection/stage_in.ll");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path:?}: {e}"));
    let from = r#"!"air.arg_type_name", !"float4", !"air.arg_name", !"out""#;
    assert_eq!(text.matches(from).count(), 1);
    let text = text.replace(
        from,
        r#"!"air.arg_type_name", !"quaternion", !"air.arg_name", !"out""#,
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
    let f = lib.function("reflection_stage_in").expect("function");
    match f.reflection(&Budget::standard()) {
        Err(Error::Unsupported(u)) => {
            assert_eq!(u.feature, Feature::Reflection);
            assert_eq!(u.symbol, Some(DiagName::new("quaternion")));
        }
        other => panic!("{other:?}"),
    }
}

/// Metal's pipeline answers by case: the stage lines under `MTLPipelineOption` 1, and the lines
/// of options 0, 2 and 3.
fn pipeline_answers() -> BTreeMap<String, (Vec<String>, Vec<String>)> {
    let path = fixtures_dir().join("reflection/metal-answers.txt");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path:?}: {e}"));
    let mut out: BTreeMap<String, (Vec<String>, Vec<String>)> = BTreeMap::new();
    let mut case: Option<&str> = None;
    for line in text.lines().filter(|l| !l.starts_with('#')) {
        if let Some(name) = line.strip_prefix("== ") {
            case = (!name.starts_with("functions_")
                && !name.starts_with("specialized_")
                && !name.starts_with("encoder_")
                && !name.starts_with("enc:"))
            .then_some(name);
            continue;
        }
        let Some(name) = case else {
            continue;
        };
        let (stages, options) = out.entry(name.to_owned()).or_default();
        if line.starts_with("options=") {
            options.push(line.to_owned());
        } else {
            stages.push(line.to_owned());
        }
    }
    out
}

/// The probe's stage lines of `bindings` under `label`.
fn stage_lines(label: &str, bindings: &[Binding]) -> Vec<String> {
    let mut lines = vec![format!("{label} count={}", bindings.len())];
    lines.extend(bindings.iter().map(|b| format!("  {}", binding(b))));
    lines
}

fn compute_lines(r: &ComputePipelineReflection) -> Vec<String> {
    stage_lines("compute", r.bindings())
}

fn render_lines(r: &RenderPipelineReflection) -> Vec<String> {
    [
        stage_lines("vertex", r.vertex_bindings()),
        stage_lines("fragment", r.fragment_bindings()),
        stage_lines("tile", r.tile_bindings()),
        stage_lines("object", r.object_bindings()),
        stage_lines("mesh", r.mesh_bindings()),
    ]
    .concat()
}

/// The fixtures' libraries, by stem; `None` without `llvm-as`.
fn libraries() -> Option<BTreeMap<&'static str, Library>> {
    let mut out = BTreeMap::new();
    for stem in [
        "buffers",
        "textures",
        "argument_buffers",
        "encoders",
        "encoder_arrays",
        "tables",
        "stage_in",
        "render",
        "tile",
        "tessellation",
        "vertex_inputs",
        "leftovers",
        "tables_in_arguments",
        "annotations",
        "specialized_reads",
        "mesh_functions",
        "specialized_inputs",
    ] {
        let bc = fixture(stem)?;
        let lib = Library::from_air(&bc).unwrap_or_else(|e| panic!("{stem}: {e:?}"));
        out.insert(stem, lib);
    }
    Some(out)
}

/// A pipeline case of the probe, as the compile layer describes it.
enum Case {
    Compute(Box<ComputePipelineDescriptor>),
    Render(Box<RenderPipelineDescriptor>),
    Tile(Box<TileRenderPipelineDescriptor>),
    Mesh(Box<MeshPipelineDescriptor>),
}

/// Every pipeline case of the probe (`probe.swift`'s `cases`), described as it describes them.
fn pipeline_cases(libs: &BTreeMap<&'static str, Library>) -> Vec<(&'static str, Case)> {
    let function = |stem: &str, name: &str, constants: &[(u32, DataType, Vec<u8>)]| {
        let f = libs[stem]
            .function(name)
            .unwrap_or_else(|e| panic!("{stem} {name}: {e:?}"));
        let mut fd = FunctionDescriptor::new(f);
        for (index, ty, bytes) in constants {
            fd.constants = fd
                .constants
                .with_index(*index, FunctionConstantValue::new(*ty, bytes.clone()));
        }
        StageSource::Function(fd)
    };
    let kernel = |stem: &str, name: &str, constants: &[(u32, DataType, Vec<u8>)]| {
        Case::Compute(Box::new(ComputePipelineDescriptor::new(function(
            stem, name, constants,
        ))))
    };
    let buffers = |extra: bool| {
        vec![
            (0, DataType::Bool, vec![u8::from(extra)]),
            (1, DataType::Int, 3i32.to_le_bytes().to_vec()),
        ]
    };
    let stage_in_of = |f: StageSource,
                       attributes: &[(u32, AttributeFormat, u32)],
                       layouts: &[(u32, Option<u32>)]| {
        let mut s = StageInputDescriptor::new();
        for &(i, format, buffer) in attributes {
            s.attributes
                .insert(i, StageInputAttributeDescriptor::new(format, 0, buffer));
        }
        for &(buffer, stride) in layouts {
            let mut layout = StageInputLayoutDescriptor::new(stride);
            layout.step_function = StepFunction::ThreadPositionInGridX;
            s.layouts.insert(buffer, layout);
        }
        let mut d = ComputePipelineDescriptor::new(f);
        d.stage_input = Some(s);
        Case::Compute(Box::new(d))
    };
    let stage_in = |attributes: &[(u32, AttributeFormat, u32)], layouts: &[(u32, Option<u32>)]| {
        stage_in_of(
            function("stage_in", "reflection_stage_in", &[]),
            attributes,
            layouts,
        )
    };
    let vertex_descriptor = |attributes: &[(u32, VertexFormat, u32, u32)],
                             layouts: &[(u32, Option<u32>)]| {
        let mut vd = VertexDescriptor::new();
        for &(i, format, offset, buffer) in attributes {
            vd.attributes
                .insert(i, VertexAttributeDescriptor::new(format, offset, buffer));
        }
        for &(buffer, stride) in layouts {
            vd.layouts
                .insert(buffer, VertexBufferLayoutDescriptor::new(stride));
        }
        vd
    };
    let render = |vertex: StageSource, fragment: StageSource, vd: Option<VertexDescriptor>| {
        let mut d = RenderPipelineDescriptor::new(vertex);
        d.fragment = Some(fragment);
        d.vertex_descriptor = vd;
        d.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
        d
    };
    let base = [
        (0, VertexFormat::Float4, 0, 0),
        (1, VertexFormat::Float2, 16, 0),
        (2, VertexFormat::Half4, 0, 2),
    ];
    let two = [
        (0, VertexFormat::Float4, 0, 0),
        (1, VertexFormat::Float2, 0, 1),
    ];
    let drawing = |vd: VertexDescriptor| {
        Case::Render(Box::new(render(
            function("render", "reflection_vertex", &[]),
            function("render", "reflection_fragment", &[]),
            Some(vd),
        )))
    };
    let inputs = |name: &str, constants: &[(u32, DataType, Vec<u8>)]| {
        Case::Render(Box::new(render(
            function("vertex_inputs", name, constants),
            function("vertex_inputs", "reflection_solid", &[]),
            Some(vertex_descriptor(&two, &[(0, Some(16)), (1, Some(8))])),
        )))
    };
    let use_uv = |v: bool| vec![(0, DataType::Bool, vec![u8::from(v)])];
    let meshes = |object: bool| {
        let mut d = MeshPipelineDescriptor::new(function("mesh_functions", "reflection_mesh", &[]));
        if object {
            d.object = Some(function("mesh_functions", "reflection_object", &[]));
        }
        d.fragment = Some(function("mesh_functions", "reflection_mesh_fragment", &[]));
        d.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
        Case::Mesh(Box::new(d))
    };
    let branch = |read: bool| {
        Case::Render(Box::new(render(
            function("specialized_reads", "reflection_branch_read", &use_uv(read)),
            function("specialized_reads", "reflection_branch_solid", &[]),
            Some(vertex_descriptor(&two, &[(0, Some(16)), (1, Some(8))])),
        )))
    };
    // A kernel's stage inputs, a post-tessellation function's inputs, and bindings that the
    // constant leaves unread.
    let gated_stage_in = |read: bool| {
        stage_in_of(
            function(
                "specialized_inputs",
                "reflection_gated_stage_in",
                &use_uv(read),
            ),
            &[
                (0, AttributeFormat::Float4, 1),
                (1, AttributeFormat::UInt, 2),
            ],
            &[(1, Some(16)), (2, Some(4))],
        )
    };
    let gated_tessellation = |read: bool| {
        let mut vd = vertex_descriptor(
            &[
                (0, VertexFormat::Float4, 0, 0),
                (1, VertexFormat::Float4, 16, 0),
                (3, VertexFormat::Float4, 0, 1),
                (4, VertexFormat::Float4, 16, 1),
            ],
            &[(0, Some(32)), (1, Some(32))],
        );
        for (buffer, step) in [
            (0, VertexStepFunction::PerPatchControlPoint),
            (1, VertexStepFunction::PerPatch),
        ] {
            if let Some(l) = vd.layouts.get_mut(&buffer) {
                l.step_function = step;
            }
        }
        let mut d = render(
            function(
                "specialized_inputs",
                "reflection_gated_tessellation",
                &use_uv(read),
            ),
            function("specialized_inputs", "reflection_gated_solid", &[]),
            Some(vd),
        );
        d.tessellation = Some(TessellationDescriptor::default());
        Case::Render(Box::new(d))
    };
    let gated_resources = |read: bool| {
        kernel(
            "specialized_inputs",
            "reflection_gated_resources",
            &use_uv(read),
        )
    };
    let mut instance = vertex_descriptor(&base, &[(0, Some(24)), (2, Some(8))]);
    if let Some(l) = instance.layouts.get_mut(&2) {
        l.step_function = VertexStepFunction::PerInstance;
    }
    let mut patches = vertex_descriptor(
        &[
            (0, VertexFormat::Float4, 0, 0),
            (3, VertexFormat::Float4, 0, 1),
        ],
        &[(0, Some(16)), (1, Some(16))],
    );
    for (buffer, step) in [
        (0, VertexStepFunction::PerPatchControlPoint),
        (1, VertexStepFunction::PerPatch),
    ] {
        if let Some(l) = patches.layouts.get_mut(&buffer) {
            l.step_function = step;
        }
    }
    let mut tessellated = render(
        function("tessellation", "reflection_tessellation", &[]),
        function("render", "reflection_fragment_solid", &[]),
        Some(patches),
    );
    tessellated.tessellation = Some(TessellationDescriptor::default());
    let tile_of = |stem: &str, name: &str| {
        let mut tile = TileRenderPipelineDescriptor::new(function(stem, name, &[]));
        tile.color_attachment_formats[0] = Some(PixelFormat::RGBA8Unorm);
        tile.threadgroup_size_matches_tile_size = true;
        tile
    };
    let tile = tile_of("tile", "reflection_tile");
    let tile_explicit = tile_of("leftovers", "reflection_tile_explicit");
    let base_at = |base: i32| vec![(1, DataType::Int, base.to_le_bytes().to_vec())];
    vec![
        (
            "buffers_extra_false",
            kernel("buffers", "reflection_buffers", &buffers(false)),
        ),
        (
            "buffers_extra_true",
            kernel("buffers", "reflection_buffers", &buffers(true)),
        ),
        ("order", kernel("buffers", "reflection_order", &[])),
        (
            "threadgroup",
            kernel("buffers", "reflection_threadgroup", &[]),
        ),
        ("textures", kernel("textures", "reflection_textures", &[])),
        (
            "argument_buffers",
            kernel("argument_buffers", "reflection_argument_buffers", &[]),
        ),
        ("encoders_args", kernel("encoders", "encoders_args", &[])),
        (
            "struct_arrays",
            kernel("encoder_arrays", "encoder_arrays", &[]),
        ),
        ("tables", kernel("tables", "reflection_tables", &[])),
        (
            "stage_in",
            stage_in(
                &[
                    (0, AttributeFormat::Float4, 1),
                    (1, AttributeFormat::UInt, 2),
                ],
                &[(1, Some(16)), (2, Some(4))],
            ),
        ),
        (
            "render_vertex_descriptor",
            drawing(vertex_descriptor(&base, &[(0, Some(24)), (2, Some(8))])),
        ),
        (
            "render_plain",
            Case::Render(Box::new(render(
                function("render", "reflection_vertex_plain", &[]),
                function("render", "reflection_fragment_solid", &[]),
                None,
            ))),
        ),
        ("tile", Case::Tile(Box::new(tile))),
        (
            "vertex_undeclared",
            drawing(vertex_descriptor(
                &[base[0], base[1], base[2], (5, VertexFormat::Float, 0, 4)],
                &[(0, Some(24)), (2, Some(8)), (4, Some(4))],
            )),
        ),
        (
            "vertex_shared_mixed",
            drawing(vertex_descriptor(
                &[base[0], base[1], base[2], (5, VertexFormat::Float, 20, 0)],
                &[(0, Some(24)), (2, Some(8))],
            )),
        ),
        (
            "vertex_buffer_order",
            drawing(vertex_descriptor(
                &[
                    (0, VertexFormat::Float4, 0, 6),
                    (1, VertexFormat::Float2, 0, 3),
                    (2, VertexFormat::Half4, 8, 3),
                ],
                &[(6, Some(16)), (3, Some(16))],
            )),
        ),
        (
            "vertex_dynamic_stride",
            drawing(vertex_descriptor(&base, &[(0, None), (2, Some(8))])),
        ),
        ("vertex_instance_step", drawing(instance)),
        ("vertex_unread", inputs("reflection_unread", &[])),
        (
            "vertex_gated_false",
            inputs("reflection_gated", &use_uv(false)),
        ),
        (
            "vertex_gated_true",
            inputs("reflection_gated", &use_uv(true)),
        ),
        ("vertex_tessellation", Case::Render(Box::new(tessellated))),
        (
            "stage_in_undeclared",
            stage_in(
                &[
                    (0, AttributeFormat::Float4, 1),
                    (1, AttributeFormat::UInt, 2),
                    (4, AttributeFormat::Float, 5),
                ],
                &[(1, Some(16)), (2, Some(4)), (5, Some(4))],
            ),
        ),
        (
            "stage_in_buffer_order",
            stage_in(
                &[
                    (0, AttributeFormat::Float4, 7),
                    (1, AttributeFormat::UInt, 3),
                ],
                &[(7, Some(16)), (3, Some(4))],
            ),
        ),
        (
            "stage_in_dynamic_stride",
            stage_in(
                &[
                    (0, AttributeFormat::Float4, 1),
                    (1, AttributeFormat::UInt, 2),
                ],
                &[(1, None), (2, Some(4))],
            ),
        ),
        ("tile_explicit", Case::Tile(Box::new(tile_explicit))),
        (
            "kernel_explicit",
            kernel("leftovers", "reflection_kernel_explicit", &[]),
        ),
        (
            "fragment_data",
            Case::Render(Box::new(render(
                function("leftovers", "reflection_cover", &[]),
                function("leftovers", "reflection_fragment_data", &[]),
                None,
            ))),
        ),
        (
            "pipeline_states",
            kernel("leftovers", "reflection_pipeline_states", &[]),
        ),
        (
            "gated_indices_false",
            kernel("leftovers", "reflection_gated_indices", &use_uv(false)),
        ),
        (
            "gated_indices_true",
            kernel("leftovers", "reflection_gated_indices", &use_uv(true)),
        ),
        (
            "computed_indices_0",
            kernel("leftovers", "reflection_computed_indices", &base_at(0)),
        ),
        (
            "computed_indices_3",
            kernel("leftovers", "reflection_computed_indices", &base_at(3)),
        ),
        (
            "vertex_no_layout",
            drawing(vertex_descriptor(&base, &[(0, Some(24))])),
        ),
        (
            "vertex_no_layout_unread",
            Case::Render(Box::new(render(
                function("vertex_inputs", "reflection_unread", &[]),
                function("vertex_inputs", "reflection_solid", &[]),
                Some(vertex_descriptor(&two, &[(0, Some(16))])),
            ))),
        ),
        (
            "stage_in_no_layout",
            stage_in(
                &[
                    (0, AttributeFormat::Float4, 1),
                    (1, AttributeFormat::UInt, 2),
                ],
                &[(1, Some(16))],
            ),
        ),
        (
            "tables_in_arguments",
            kernel("tables_in_arguments", "reflection_tables_in_arguments", &[]),
        ),
        (
            "annotated",
            kernel("annotations", "reflection_annotated", &[]),
        ),
        ("render_branch_false", branch(false)),
        ("render_branch_true", branch(true)),
        ("mesh_pipeline", meshes(true)),
        ("mesh_pipeline_no_object", meshes(false)),
        ("stage_in_gated_false", gated_stage_in(false)),
        ("stage_in_gated_true", gated_stage_in(true)),
        ("tessellation_gated_false", gated_tessellation(false)),
        ("tessellation_gated_true", gated_tessellation(true)),
        ("resources_gated_false", gated_resources(false)),
        ("resources_gated_true", gated_resources(true)),
    ]
}

/// Every pipeline case's reflection equals Metal's, line for line: a pipeline leaves out the
/// arguments its constants gate off, answers unused those they leave unread (a binding of each
/// kind), states an argument buffer's arrays of resources with stride 0, and lists a vertex or
/// stage input descriptor's buffers first, in buffer index order, sized by their layouts' strides
/// (a dynamic stride as `NSUIntegerMax`), used when an attribute the stage reads is in them (not
/// one it leaves unread, a gated-off one, an undeclared one, or a post-tessellation function's
/// patch input). Metal's answer under options 2 and 3 is option 1's, and under 0 nil (the runtime's
/// lane checks the creation forms).
#[test]
fn pipeline_reflection_matches_metal() {
    let Some(libs) = libraries() else {
        return;
    };
    let answers = pipeline_answers();
    let cases = pipeline_cases(&libs);
    let names: Vec<&str> = cases.iter().map(|(name, _)| *name).collect();
    assert_eq!(
        answers.keys().map(String::as_str).collect::<Vec<_>>(),
        {
            let mut sorted = names.clone();
            sorted.sort_unstable();
            sorted
        },
        "every recorded case is described here"
    );
    let mut mismatches = Vec::new();
    let mut stopped = 0;
    for (name, case) in &cases {
        let (want, options) = &answers[*name];
        let got = case_lines(case, &Budget::standard());
        // Metal stopped at an assertion naming the buffer without a layout: the crate's answer
        // is the descriptor's error naming it.
        if let Some(message) = want.first().and_then(|l| l.strip_prefix("stopped: ")) {
            assert!(options.is_empty() && want.len() == 1, "{name}");
            let buffer: u32 = message
                .split("references a buffer at index ")
                .nth(1)
                .and_then(|rest| rest.split(' ').next())
                .and_then(|n| n.parse().ok())
                .unwrap_or_else(|| panic!("{name}: {message}"));
            match got {
                Err(Error::InvalidDescriptor(
                    DescriptorError::VertexLayout { layout }
                    | DescriptorError::StageInputLayout { layout },
                )) if layout == buffer => stopped += 1,
                other => mismatches.push(format!("{name}: {other:?}, Metal: {message}")),
            }
            continue;
        }
        assert_eq!(
            options,
            &[
                "options=1 present",
                "options=0 nil",
                "options=2 same",
                "options=3 same"
            ],
            "{name}"
        );
        let got = got.unwrap_or_else(|e| panic!("{name}: {e:?}"));
        mismatches.extend(diff(name, &got, want));
    }
    assert_eq!(stopped, 3, "the descriptors Metal stops at");
    assert!(mismatches.is_empty(), "{}", mismatches.join("\n"));
}

/// The probe's lines of `case`'s pipeline reflection, or why the crate refuses it.
fn case_lines(case: &Case, budget: &Budget) -> Result<Vec<String>, Error> {
    match case {
        Case::Compute(d) => d.reflection(budget).map(|r| compute_lines(&r)),
        Case::Render(d) => d.reflection(budget).map(|r| render_lines(&r)),
        Case::Tile(d) => d.reflection(budget).map(|r| render_lines(&r)),
        Case::Mesh(d) => d.reflection(budget).map(|r| render_lines(&r)),
    }
}

/// The compile layer's device of the lavapipe profile.
fn lavapipe() -> crate::device::Device {
    crate::device::Device::new(
        crate::device::DeviceDescription::from_khronos_json(
            &std::fs::read_to_string(
                std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("../metal2vulkan-validation/profiles/lavapipe.json"),
            )
            .expect("profile"),
            None,
        )
        .expect("description"),
        crate::device::DevicePolicy::default(),
    )
    .expect("device")
}

/// Pipeline creation stops where Metal stops: each recorded case whose descriptor has an
/// attribute naming a buffer without a layout (read, as `vertex_no_layout`'s and
/// `stage_in_no_layout`'s, or declared unread, as `vertex_no_layout_unread`'s) is refused as the
/// descriptor's `InvalidDescriptor` naming that buffer, as its reflection is.
#[test]
fn pipeline_creation_stops_at_an_attribute_without_a_layout() {
    let Some(libs) = libraries() else {
        return;
    };
    let device = lavapipe();
    let answers = pipeline_answers();
    let budget = Budget::standard();
    let mut stopped = 0;
    for (name, case) in pipeline_cases(&libs) {
        let Some(message) = answers[name]
            .0
            .first()
            .and_then(|l| l.strip_prefix("stopped: "))
        else {
            continue;
        };
        let buffer: u32 = message
            .split("references a buffer at index ")
            .nth(1)
            .and_then(|rest| rest.split(' ').next())
            .and_then(|n| n.parse().ok())
            .unwrap_or_else(|| panic!("{name}: {message}"));
        let got = match &case {
            Case::Compute(d) => device.compile_compute_pipeline(d, &budget).map(|_| ()),
            Case::Render(d) => device.compile_render_pipeline(d, &budget).map(|_| ()),
            Case::Tile(d) => device.compile_tile_pipeline(d, &budget).map(|_| ()),
            Case::Mesh(d) => device.compile_mesh_pipeline(d, &budget).map(|_| ()),
        };
        match got {
            Err(Error::InvalidDescriptor(
                DescriptorError::VertexLayout { layout }
                | DescriptorError::StageInputLayout { layout },
            )) if layout == buffer => stopped += 1,
            other => panic!("{name}: {other:?}, Metal: {message}"),
        }
    }
    assert_eq!(stopped, 3, "the descriptors Metal stops at");
}

/// An attribute of format `Invalid` is unset, whatever buffer it names: added to each recorded
/// case Metal accepts that has a vertex or stage input descriptor, naming a buffer without a
/// layout, it changes neither the case's acceptance nor its reflection (Metal accepts it: its
/// reflection lists only the set attributes' buffers).
#[test]
fn an_invalid_attribute_changes_nothing() {
    let Some(libs) = libraries() else {
        return;
    };
    let answers = pipeline_answers();
    let budget = Budget::standard();
    let mut checked = 0;
    for (name, case) in pipeline_cases(&libs) {
        if answers[name]
            .0
            .first()
            .is_some_and(|l| l.starts_with("stopped: "))
        {
            continue;
        }
        let unset = match &case {
            Case::Compute(d) if d.stage_input.is_some() => {
                let mut d = d.clone();
                if let Some(s) = d.stage_input.as_mut() {
                    s.attributes.insert(
                        30,
                        StageInputAttributeDescriptor::new(AttributeFormat::Invalid, 0, 29),
                    );
                }
                Case::Compute(d)
            }
            Case::Render(d) if d.vertex_descriptor.is_some() => {
                let mut d = d.clone();
                if let Some(vd) = d.vertex_descriptor.as_mut() {
                    vd.attributes.insert(
                        30,
                        VertexAttributeDescriptor::new(VertexFormat::Invalid, 0, 29),
                    );
                }
                Case::Render(d)
            }
            Case::Compute(_) | Case::Render(_) | Case::Tile(_) | Case::Mesh(_) => continue,
        };
        assert_eq!(
            case_lines(&unset, &budget),
            case_lines(&case, &budget),
            "{name}"
        );
        checked += 1;
    }
    assert!(checked >= 10, "{checked} cases with a descriptor");
}

/// A descriptor whose every attribute names a laid-out buffer passes; one with an attribute no
/// function declares naming a buffer without a layout is refused, naming the first such
/// attribute's buffer, as Metal stops for it (an undeclared attribute of a vertex descriptor and of
/// a stage input descriptor), and so is a layout of a nonzero or dynamic stride that no set
/// attribute names; an `Invalid` attribute is unset.
#[test]
fn every_attribute_needs_its_buffers_layout() {
    let mut vd = VertexDescriptor::new();
    vd.attributes.insert(
        0,
        VertexAttributeDescriptor::new(VertexFormat::Float4, 0, 0),
    );
    vd.layouts
        .insert(0, VertexBufferLayoutDescriptor::new(Some(16)));
    assert_eq!(vd.check_layouts(), Ok(()));
    vd.attributes
        .insert(5, VertexAttributeDescriptor::new(VertexFormat::Float, 0, 4));
    vd.attributes
        .insert(7, VertexAttributeDescriptor::new(VertexFormat::Float, 0, 2));
    assert_eq!(
        vd.check_layouts(),
        Err(Error::InvalidDescriptor(DescriptorError::VertexLayout {
            layout: 4
        }))
    );
    let mut s = StageInputDescriptor::new();
    s.attributes.insert(
        0,
        StageInputAttributeDescriptor::new(AttributeFormat::Float4, 0, 0),
    );
    let mut layout = StageInputLayoutDescriptor::new(Some(16));
    layout.step_function = StepFunction::ThreadPositionInGridX;
    s.layouts.insert(0, layout);
    assert_eq!(s.check_layouts(), Ok(()));
    s.attributes.insert(
        4,
        StageInputAttributeDescriptor::new(AttributeFormat::Float, 0, 3),
    );
    assert_eq!(
        s.check_layouts(),
        Err(Error::InvalidDescriptor(
            DescriptorError::StageInputLayout { layout: 3 }
        ))
    );
    // An attribute of format `Invalid` is unset, whatever buffer it names.
    s.attributes.insert(
        4,
        StageInputAttributeDescriptor::new(AttributeFormat::Invalid, 0, 3),
    );
    assert_eq!(s.check_layouts(), Ok(()));
    vd.attributes.insert(
        5,
        VertexAttributeDescriptor::new(VertexFormat::Invalid, 0, 4),
    );
    vd.attributes.insert(
        7,
        VertexAttributeDescriptor::new(VertexFormat::Invalid, 0, 2),
    );
    assert_eq!(vd.check_layouts(), Ok(()));
    // A layout no set attribute names stops Metal at a nonzero or dynamic stride, whatever its
    // step, an `Invalid` attribute naming it or not; at stride 0 it passes.
    let lonely = |stride: Option<u32>, step: VertexStepFunction| {
        let mut d = vd.clone();
        let mut layout = VertexBufferLayoutDescriptor::new(stride);
        layout.step_function = step;
        d.layouts.insert(2, layout);
        d.check_layouts()
    };
    let stopped = Err(Error::InvalidDescriptor(DescriptorError::VertexLayout {
        layout: 2,
    }));
    assert_eq!(lonely(Some(32), VertexStepFunction::PerVertex), stopped);
    assert_eq!(lonely(None, VertexStepFunction::PerVertex), stopped);
    assert_eq!(lonely(Some(0), VertexStepFunction::PerInstance), Ok(()));
    assert_eq!(lonely(Some(0), VertexStepFunction::Constant), Ok(()));
    let mut lonely = s.clone();
    lonely
        .layouts
        .insert(3, StageInputLayoutDescriptor::new(Some(32)));
    assert_eq!(
        lonely.check_layouts(),
        Err(Error::InvalidDescriptor(
            DescriptorError::StageInputLayout { layout: 3 }
        ))
    );
    lonely
        .layouts
        .insert(3, StageInputLayoutDescriptor::new(Some(0)));
    assert_eq!(lonely.check_layouts(), Ok(()));
    // A layout's step Metal's compute pipeline creation stops at, with and without API
    // validation: a vertex stage's step function, the default per-vertex step of
    // a fresh layout among them, and a constant step at a rate other than 0. Metal's unset layout
    // (stride 0, per vertex, rate 1) passes, as layout 3 just did.
    let stepped = |step: StepFunction, rate: u32| {
        let mut d = s.clone();
        let layout = d.layouts.get_mut(&0).expect("layout 0");
        (layout.step_function, layout.step_rate) = (step, rate);
        d.check_layouts()
    };
    for step in [StepFunction::PerVertex, StepFunction::PerInstance] {
        assert_eq!(
            stepped(step, 1),
            Err(Error::InvalidDescriptor(
                DescriptorError::StageInputStepFunction {
                    layout: 0,
                    step_function: step
                }
            ))
        );
    }
    assert_eq!(
        stepped(StepFunction::Constant, 1),
        Err(Error::InvalidDescriptor(
            DescriptorError::StageInputStepRate {
                layout: 0,
                step_rate: 1
            }
        ))
    );
    assert_eq!(stepped(StepFunction::Constant, 0), Ok(()));
    assert_eq!(
        StageInputLayoutDescriptor::new(Some(0)).step_function,
        StepFunction::PerVertex
    );
}

/// A stage the compile layer already compiled answers what its compile kept, through the cache's
/// encoding: the same lines as the function it was compiled from, over every pipeline case, each
/// stage that compiles on lavapipe compiled (a vertex function under no vertex inputs that cannot
/// stays a function).
#[test]
fn a_compiled_stage_answers_as_its_function() {
    let Some(libs) = libraries() else {
        return;
    };
    let device = lavapipe();
    let compiled = std::cell::Cell::new(0);
    let compile = |source: &StageSource| match source {
        StageSource::Function(fd) => device
            .compile_function(
                fd,
                &crate::pipeline::LoweringInputs::new(),
                &Budget::standard(),
            )
            .ok()
            .map_or_else(
                || source.clone(),
                |c| {
                    compiled.set(compiled.get() + 1);
                    let back = crate::library::CompiledFunction::from_bytes(&c.to_bytes(), &device)
                        .expect("decode");
                    StageSource::Compiled(back)
                },
            ),
        StageSource::Compiled(_) => source.clone(),
    };
    let mut mismatches = Vec::new();
    for (name, case) in pipeline_cases(&libs) {
        let swapped = match &case {
            Case::Compute(d) => {
                let mut c = (**d).clone();
                c.kernel = compile(&d.kernel);
                Case::Compute(Box::new(c))
            }
            Case::Render(d) => {
                let mut c = (**d).clone();
                c.vertex = compile(&d.vertex);
                c.fragment = d.fragment.as_ref().map(compile);
                Case::Render(Box::new(c))
            }
            Case::Tile(d) => {
                let mut c = (**d).clone();
                c.tile = compile(&d.tile);
                Case::Tile(Box::new(c))
            }
            Case::Mesh(d) => {
                let mut c = (**d).clone();
                c.object = d.object.as_ref().map(compile);
                c.mesh = compile(&d.mesh);
                c.fragment = d.fragment.as_ref().map(compile);
                Case::Mesh(Box::new(c))
            }
        };
        let budget = Budget::standard();
        let (want, got) = (case_lines(&case, &budget), case_lines(&swapped, &budget));
        if want != got {
            mismatches.push(format!(
                "{name}:\n  function: {want:?}\n  compiled: {got:?}"
            ));
        }
    }
    assert!(compiled.get() >= 30, "{} stages compiled", compiled.get());
    assert!(mismatches.is_empty(), "{}", mismatches.join("\n"));
}

/// Metal's `MTLFunction` property lines: per `functions_<fixture>` function, keyed by fixture and
/// function, and per `specialized_<case>`, keyed `("specialized", case)`. Each is the `function`
/// line, the attribute lines and the function-constant lines.
fn property_answers() -> BTreeMap<(String, String), Vec<String>> {
    let path = fixtures_dir().join("reflection/metal-answers.txt");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path:?}: {e}"));
    let mut out: BTreeMap<(String, String), Vec<String>> = BTreeMap::new();
    let mut section: Option<(&str, Option<&str>)> = None;
    let mut key: Option<(String, String)> = None;
    for line in text.lines().filter(|l| !l.starts_with('#')) {
        if let Some(name) = line.strip_prefix("== ") {
            section = if let Some(fixture) = name.strip_prefix("functions_") {
                Some((fixture, None))
            } else {
                name.strip_prefix("specialized_")
                    .map(|case| ("specialized", Some(case)))
            };
            key = None;
            continue;
        }
        let Some((fixture, case)) = section else {
            continue;
        };
        if let Some(rest) = line.strip_prefix("function ") {
            let name = case.or_else(|| rest.split(' ').next()).unwrap_or_default();
            key = Some((fixture.to_owned(), name.to_owned()));
        }
        let property = [
            "function ",
            "vertexAttributes ",
            "  vertexAttribute ",
            "stageInputAttributes ",
            "  stageInputAttribute ",
            "functionConstants ",
            "  functionConstant ",
        ]
        .iter()
        .any(|p| line.starts_with(p));
        if let (Some(k), true) = (&key, property) {
            out.entry(k.clone()).or_default().push(line.to_owned());
        }
    }
    out
}

/// The probe's lines for function `name`'s properties `p` and constants `constants` (`None`
/// leaves the constant lines out).
fn property_lines(
    name: &str,
    p: &FunctionProperties,
    constants: Option<&BTreeMap<String, FunctionConstant>>,
) -> Vec<String> {
    let count = p.patch_control_point_count().map_or(-1, i64::from);
    let mut lines = vec![format!(
        "function {name} functionType={} patchType={} patchControlPointCount={count}",
        p.function_type().as_raw(),
        p.patch_type().as_raw(),
    )];
    let attributes = |lines: &mut Vec<String>, label: &str, list: Option<&[Attribute]>| {
        let Some(list) = list else {
            lines.push(format!("{label}s nil"));
            return;
        };
        lines.push(format!("{label}s count={}", list.len()));
        for a in list {
            lines.push(format!(
                "  {label} name={} index={} type={} active={} patch={} controlPoint={}",
                a.name(),
                a.attribute_index(),
                a.attribute_type().as_raw(),
                a.is_active(),
                a.is_patch_data(),
                a.is_patch_control_point_data()
            ));
        }
    };
    attributes(&mut lines, "vertexAttribute", p.vertex_attributes());
    attributes(
        &mut lines,
        "stageInputAttribute",
        p.stage_input_attributes(),
    );
    if let Some(constants) = constants {
        lines.push(format!("functionConstants count={}", constants.len()));
        for (key, c) in constants {
            lines.push(format!(
                "  functionConstant key={key} name={} type={} index={} required={}",
                c.name(),
                c.data_type().as_raw(),
                c.index(),
                c.is_required()
            ));
        }
    }
    lines
}

/// One function's measured metallib tags: its name, `TYPE`, and each other kept tag's fourcc and
/// bytes.
struct TaggedFunction {
    name: String,
    type_code: u8,
    tags: Vec<([u8; 4], Vec<u8>)>,
}

/// The measured metallib tags of each fixture's functions (`metallib-tags.txt`), by fixture, in
/// function-list order.
fn metallib_tags() -> BTreeMap<String, Vec<TaggedFunction>> {
    let path = fixtures_dir().join("reflection/metallib-tags.txt");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path:?}: {e}"));
    let hex = |h: &str| -> Vec<u8> {
        (0..h.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&h[i..i + 2], 16).expect("hex"))
            .collect()
    };
    let mut out: BTreeMap<String, Vec<_>> = BTreeMap::new();
    let mut fixture = String::new();
    for line in text.lines().filter(|l| !l.starts_with('#')) {
        if let Some(name) = line.strip_prefix("== ") {
            fixture = name.to_owned();
        } else if let Some(rest) = line.strip_prefix("function ") {
            let mut words = rest.split(' ');
            let name = words.next().expect("name").to_owned();
            let mut ty = None;
            let mut tags = Vec::new();
            for w in words {
                let (k, v) = w.split_once('=').expect("tag=value");
                match k {
                    "TYPE" => ty = hex(v).first().copied(),
                    "other" => {}
                    _ => tags.push((<[u8; 4]>::try_from(k.as_bytes()).expect("fourcc"), hex(v))),
                }
            }
            out.entry(fixture.clone())
                .or_default()
                .push(TaggedFunction {
                    name,
                    type_code: ty.expect("TYPE"),
                    tags,
                });
        }
    }
    out
}

/// Every fixture function's `MTLFunction` properties and function constants equal Metal's, line
/// for line, as a metallib carrying the measured tags answers them: `TYPE` as the function type
/// (vertex, fragment, kernel, visible, intersection, mesh, object), `TESS` as the patch type and
/// count (63 for none), `VATT` and `VATY` as the attributes (declaration order, the index, active,
/// patch and control-point bits, the type), a kernel's as stage-input attributes only, and `CNST`
/// as the constants (the flag as `required`). Nothing is decoded.
#[test]
fn metallib_function_properties_match_metal() {
    let answers = property_answers();
    let tags = metallib_tags();
    assert!(tags.len() >= 11, "{} fixtures", tags.len());
    let mut compared = 0;
    let mut mismatches = Vec::new();
    for (fixture, functions) in &tags {
        let bytes = metallib(
            &functions
                .iter()
                .map(|f| {
                    // `TESS` is a function-list tag; the others are public metadata.
                    let (public, list): (Vec<_>, Vec<_>) = f
                        .tags
                        .iter()
                        .cloned()
                        .partition(|(fourcc, _)| fourcc != b"TESS");
                    MtlbFunction {
                        name: &f.name,
                        type_code: f.type_code,
                        bitcode: wrap(&[0x42, 0x43, 0xc0, 0xde]),
                        constants: vec![],
                        public,
                        tags: list,
                    }
                })
                .collect::<Vec<_>>(),
        );
        let lib = Library::parse(&bytes).unwrap_or_else(|e| panic!("{fixture}: {e:?}"));
        for TaggedFunction { name, .. } in functions {
            let f = lib.function(name).expect("function");
            let p = f
                .properties(&Budget::standard())
                .unwrap_or_else(|e| panic!("{fixture} {name}: {e:?}"));
            let c = f
                .function_constants_dictionary()
                .unwrap_or_else(|e| panic!("{fixture} {name}: {e:?}"));
            let want = &answers[&(fixture.clone(), name.clone())];
            mismatches.extend(diff(
                &format!("{fixture} {name}"),
                &property_lines(name, &p, Some(&c)),
                want,
            ));
            compared += 1;
        }
    }
    assert_eq!(compared, 78, "every measured function");
    assert!(mismatches.is_empty(), "{}", mismatches.join("\n"));
}

/// A bitcode wrapper's or raw module's function answers the same properties from its AIR: the
/// entry's stage, `air.patch`, and stage-input parameters, active unless `air.arg_unused`; and
/// the same function constants, those its code and metadata read (directly, through a static
/// initializer's copy, as an argument's or attribute's predicate, or as a computed index), each
/// required unless it asks `is_function_constant_defined` about it.
#[test]
fn module_function_properties_match_metal() {
    let answers = property_answers();
    let mut libraries: BTreeMap<String, Library> = BTreeMap::new();
    let mut mismatches = Vec::new();
    let mut read = 0;
    for ((stem, name), want) in &answers {
        if stem == "specialized" || stem == METALLIB_ONLY {
            continue;
        }
        if !libraries.contains_key(stem) {
            let Some(bc) = fixture(stem) else {
                return;
            };
            let lib = Library::from_air(&bc).unwrap_or_else(|e| panic!("{stem}: {e:?}"));
            libraries.insert(stem.clone(), lib);
        }
        let f = libraries[stem].function(name).expect("function");
        let p = f
            .properties(&Budget::standard())
            .unwrap_or_else(|e| panic!("{stem} {name}: {e:?}"));
        let constants = f
            .function_constants_dictionary()
            .unwrap_or_else(|e| panic!("{stem} {name}: {e:?}"));
        if !constants.is_empty() {
            read += 1;
        }
        let got = property_lines(name, &p, Some(&constants));
        mismatches.extend(diff(&format!("{stem} {name}"), &got, want));
    }
    // The functions whose metallib's `CNST` lists constants (`metallib-tags.txt`).
    assert_eq!(read, 28, "functions that read constants");
    assert!(mismatches.is_empty(), "{}", mismatches.join("\n"));
}

/// A raw module's post-tessellation function whose patch states no count (`[[patch(quad)]]`,
/// `patch_any`, `types_quad_any` word for word) answers its metallib's properties from its AIR: a
/// quad patch of no count. Its compile refuses `Tessellation` `NotImplemented`: the control stage
/// compiles each patch's count in, and no pipeline is keyed on the draw's.
#[test]
fn a_module_patch_without_a_count_answers_as_its_metallib() {
    let Some(bc) = fixture("patch_any") else {
        return;
    };
    let lib = Library::from_air(&bc).expect("a count-less patch decodes");
    let f = lib.function("types_quad_any").expect("function");
    let p = f.properties(&Budget::standard()).expect("properties");
    let want = &property_answers()[&(METALLIB_ONLY.to_owned(), "types_quad_any".to_owned())];
    let (properties, constants) = want.split_at(
        want.iter()
            .position(|l| l.starts_with("functionConstants "))
            .expect("a functionConstants line"),
    );
    let got = property_lines("types_quad_any", &p, None);
    let mismatches = diff("patch_any types_quad_any", &got, properties);
    assert!(mismatches.is_empty(), "{}", mismatches.join("\n"));
    assert_eq!(constants, ["functionConstants count=0"]);
    assert!(f.function_constants_dictionary().expect("none").is_empty());
    let device = lavapipe();
    match device.compile_function(
        &FunctionDescriptor::new(f),
        &crate::pipeline::LoweringInputs::new(),
        &Budget::standard(),
    ) {
        Err(Error::Unsupported(u)) => assert_eq!(
            (u.feature, u.needs),
            (Feature::Tessellation, Needs::NotImplemented)
        ),
        other => panic!("{:?}", other.map(|_| ())),
    }
}

/// A function made with constant values answers Metal's properties for it: an attribute its
/// constants gate off left out, a stage input they leave unread inactive (a vertex input, a
/// kernel's stage input, a per-patch input, every field of unread control points), the others as
/// they were (`specialized_<case>`).
#[test]
fn specialized_function_properties_match_metal() {
    let answers = property_answers();
    let (Some(inputs), Some(kinds), Some(reads), Some(gated)) = (
        fixture("vertex_inputs"),
        fixture("function_kinds"),
        fixture("specialized_reads"),
        fixture("specialized_inputs"),
    ) else {
        return;
    };
    let inputs = Library::from_air(&inputs).expect("vertex_inputs");
    let kinds = Library::from_air(&kinds).expect("function_kinds");
    let reads = Library::from_air(&reads).expect("specialized_reads");
    let gated = Library::from_air(&gated).expect("specialized_inputs");
    let boolean = |v: bool| FunctionConstantValue::new(DataType::Bool, vec![u8::from(v)]);
    let cases = [
        (
            "gated_false",
            &inputs,
            "reflection_gated",
            FunctionConstantValues::defaults().with_index(0, boolean(false)),
        ),
        (
            "gated_true",
            &inputs,
            "reflection_gated",
            FunctionConstantValues::defaults().with_index(0, boolean(true)),
        ),
        (
            "constants_gate_false",
            &kinds,
            "kinds_constants",
            FunctionConstantValues::defaults()
                .with_index(1, boolean(false))
                .with_index(0, boolean(true)),
        ),
        (
            "constants_undefined",
            &kinds,
            "kinds_constants",
            FunctionConstantValues::defaults().with_index(1, boolean(true)),
        ),
        (
            "values",
            &kinds,
            "kinds_values",
            FunctionConstantValues::defaults().with_index(
                2,
                FunctionConstantValue::new(DataType::Int, 5i32.to_le_bytes().to_vec()),
            ),
        ),
        (
            "none",
            &kinds,
            "kinds_none",
            FunctionConstantValues::defaults(),
        ),
        // A vertex input the constant leaves unread is inactive.
        (
            "branch_false",
            &reads,
            "reflection_branch_read",
            FunctionConstantValues::defaults().with_index(0, boolean(false)),
        ),
        (
            "branch_true",
            &reads,
            "reflection_branch_read",
            FunctionConstantValues::defaults().with_index(0, boolean(true)),
        ),
        // A kernel's stage input and a post-tessellation function's per-patch input the constant
        // leaves unread are inactive; its control points, one argument, are all inactive unread
        // and all active read.
        (
            "gated_kernel_false",
            &gated,
            "reflection_gated_stage_in",
            FunctionConstantValues::defaults().with_index(0, boolean(false)),
        ),
        (
            "gated_kernel_true",
            &gated,
            "reflection_gated_stage_in",
            FunctionConstantValues::defaults().with_index(0, boolean(true)),
        ),
        (
            "gated_tessellation_false",
            &gated,
            "reflection_gated_tessellation",
            FunctionConstantValues::defaults().with_index(0, boolean(false)),
        ),
        (
            "gated_tessellation_true",
            &gated,
            "reflection_gated_tessellation",
            FunctionConstantValues::defaults().with_index(0, boolean(true)),
        ),
        (
            "gated_control_points_false",
            &gated,
            "reflection_gated_control_points",
            FunctionConstantValues::defaults().with_index(0, boolean(false)),
        ),
        (
            "gated_control_points_true",
            &gated,
            "reflection_gated_control_points",
            FunctionConstantValues::defaults().with_index(0, boolean(true)),
        ),
    ];
    let recorded = answers.keys().filter(|(s, _)| s == "specialized").count();
    assert_eq!(
        recorded,
        cases.len(),
        "every recorded case is described here"
    );
    let mut mismatches = Vec::new();
    for (case, lib, name, constants) in &cases {
        let f = lib.function(name).expect("function");
        let p = f
            .specialized_properties(constants, &Budget::standard())
            .unwrap_or_else(|e| panic!("{case}: {e:?}"));
        // A function made with constant values answers no constants (the runtime's `Function`).
        let got = property_lines(name, &p, Some(&BTreeMap::new()));
        let want = &answers[&("specialized".to_owned(), (*case).to_owned())];
        mismatches.extend(diff(case, &got, want));
    }
    assert!(mismatches.is_empty(), "{}", mismatches.join("\n"));
}

/// A synthetic module for the static-initializer cases of a module function's constants: `a`
/// (`bool`, 0), `b` (`int`, 1) and `c` (`uint`, 2), and the kernels `reads_both` (a global the
/// initializer selects from `a` and `b`), `reads_c` (a global it copies `c` into), `reads_has_c`
/// (a global it stores whether `c` is defined into) and `reads_none`. `branch` makes the
/// initializer branch on `a` before it copies `c`.
fn initializer_module(branch: bool) -> String {
    let copy = if branch {
        "  br i1 %3, label %copy, label %done\ncopy:\n  %5 = load i32, ptr addrspace(2) @c.MTL_FC_INIT_2_j, align 4\n  store i32 %5, ptr addrspace(2) @copied, align 4\n  br label %done\ndone:\n"
    } else {
        "  %5 = load i32, ptr addrspace(2) @c.MTL_FC_INIT_2_j, align 4\n  store i32 %5, ptr addrspace(2) @copied, align 4\n"
    };
    let kernel = |name: &str, global: Option<&str>| {
        let value = match global {
            Some(g) => format!("  %3 = load i32, ptr addrspace(2) @{g}, align 4\n  store i32 %3, ptr addrspace(1) %0, align 4\n"),
            None => "  store i32 0, ptr addrspace(1) %0, align 4\n".to_owned(),
        };
        format!("define void @{name}(ptr addrspace(1) noundef %0, i32 noundef %1) {{\n{value}  ret void\n}}\n")
    };
    format!(
        r#"target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

@a.MTL_FC_INIT_0_b = linkonce_odr hidden addrspace(2) externally_initialized constant i8 undef, section "air.fc_initializer", align 1
@b.MTL_FC_INIT_1_i = linkonce_odr hidden addrspace(2) externally_initialized constant i32 undef, section "air.fc_initializer", align 4
@c.MTL_FC_INIT_2_j = linkonce_odr hidden addrspace(2) externally_initialized constant i32 undef, section "air.fc_initializer", align 4
@selected = internal addrspace(2) global i32 undef, align 4
@copied = internal addrspace(2) global i32 undef, align 4
@has_c = internal addrspace(2) global i8 0, align 1

{both}{copy_kernel}{none}
define void @reads_has_c(ptr addrspace(1) noundef %0, i32 noundef %1) {{
  %3 = load i8, ptr addrspace(2) @has_c, align 1
  %4 = zext i8 %3 to i32
  store i32 %4, ptr addrspace(1) %0, align 4
  ret void
}}

declare i1 @air.is_function_constant_defined(ptr addrspace(2))

define internal void @init() section "air.static_init" {{
  %1 = load i8, ptr addrspace(2) @a.MTL_FC_INIT_0_b, align 1
  %2 = load i32, ptr addrspace(2) @b.MTL_FC_INIT_1_i, align 4
  %3 = trunc i8 %1 to i1
  %4 = select i1 %3, i32 %2, i32 7
  store i32 %4, ptr addrspace(2) @selected, align 4
  %defined = call i1 @air.is_function_constant_defined(ptr addrspace(2) @c.MTL_FC_INIT_2_j)
  %byte = zext i1 %defined to i8
  store i8 %byte, ptr addrspace(2) @has_c, align 1
{copy}  ret void
}}

!air.kernel = !{{!0, !4, !5, !12}}
!air.function_constants = !{{!6, !7, !8}}
!air.version = !{{!9}}
!air.language_version = !{{!10}}

!0 = !{{ptr @reads_both, !1, !2}}
!1 = !{{}}
!2 = !{{!3, !11}}
!3 = !{{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"int", !"air.arg_name", !"out"}}
!4 = !{{ptr @reads_c, !1, !2}}
!5 = !{{ptr @reads_none, !1, !2}}
!6 = !{{ptr addrspace(2) @a.MTL_FC_INIT_0_b, !"bool", !"a", i32 0}}
!7 = !{{ptr addrspace(2) @b.MTL_FC_INIT_1_i, !"int", !"b", i32 1}}
!8 = !{{ptr addrspace(2) @c.MTL_FC_INIT_2_j, !"uint", !"c", i32 2}}
!9 = !{{i32 2, i32 8, i32 0}}
!10 = !{{!"Metal", i32 4, i32 0, i32 0}}
!11 = !{{i32 1, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"tid"}}
!12 = !{{ptr @reads_has_c, !1, !2}}
"#,
        both = kernel("reads_both", Some("selected")),
        copy_kernel = kernel("reads_c", Some("copied")),
        none = kernel("reads_none", None),
    )
}

/// A module function's constants follow the static initializer's dependencies: a global it
/// selects from two constants reads both, each required, with the type the metadata spells
/// (`int` and `uint` alike `i32` in the IR); a global it stores a definedness query into reads
/// the constant, optional; a function that names no variable global reads none. A static
/// initializer whose store a branch decides (Metal's answer unmeasured) leaves every function's
/// dictionary unknown, refused naming it, one that names no variable global too.
#[test]
fn a_module_functions_constants_follow_its_static_initializer() {
    let library =
        |branch| match metal2vulkan_devtools::llvm_assemble_text(&initializer_module(branch)) {
            Ok(bitcode) => Some(Library::from_air(&bitcode).expect("the module decodes")),
            Err(metal2vulkan_devtools::AssembleError::Missing(why)) => {
                eprintln!("SKIPPED: {why}");
                None
            }
            Err(metal2vulkan_devtools::AssembleError::Rejected(why)) => {
                panic!("llvm-as rejected the module: {why}")
            }
        };
    let Some(straight) = library(false) else {
        return;
    };
    let read = |lib: &Library, name: &str| {
        lib.function(name)
            .expect("function")
            .function_constants_dictionary()
            .map(|c| {
                c.values()
                    .map(|c| {
                        (
                            c.name().to_owned(),
                            c.data_type(),
                            c.index(),
                            c.is_required(),
                        )
                    })
                    .collect::<Vec<_>>()
            })
    };
    assert_eq!(
        read(&straight, "reads_both").expect("known"),
        [
            ("a".to_owned(), DataType::Bool, 0, true),
            ("b".to_owned(), DataType::Int, 1, true),
        ]
    );
    assert_eq!(
        read(&straight, "reads_c").expect("known"),
        [("c".to_owned(), DataType::UInt, 2, true)]
    );
    assert_eq!(read(&straight, "reads_none").expect("known"), []);
    // A stored definedness query: its reader lists the constant, optional (Metal's answer for
    // the fixture `reflection/constant_initializers`' `ci_has_opt`).
    assert_eq!(
        read(&straight, "reads_has_c").expect("known"),
        [("c".to_owned(), DataType::UInt, 2, false)]
    );
    // A store a branch on `a` decides: unmeasured, so every function's dictionary is refused.
    let branching = library(true).expect("assembled once already");
    for name in ["reads_both", "reads_c", "reads_has_c", "reads_none"] {
        match read(&branching, name) {
            Err(Error::Unsupported(u)) => {
                assert_eq!(u.feature, Feature::Reflection);
                assert_eq!(u.symbol, Some(DiagName::new(name)));
            }
            other => panic!("{name}: {other:?}"),
        }
    }
}

/// A synthetic module whose static initializer stores pointers `a` (`bool`, 0) chooses: into two
/// literal rows (`to_rows`), into a row or a global it writes from `c` (`uint`, 2; `to_vals`), into
/// a row or the constant `b` itself (`int`, 1; `to_constant`), and into a row or a table that
/// holds `c`'s address (`to_table`). Each kernel `reads_<x>` loads its pointer and reads through it.
fn pointer_module() -> String {
    let kernel = |name: &str| {
        format!(
            "define void @reads_{name}(ptr addrspace(1) noundef %0, i32 noundef %1) {{\n  %3 = load ptr addrspace(2), ptr addrspace(2) @to_{name}, align 8\n  %4 = load i32, ptr addrspace(2) %3, align 4\n  store i32 %4, ptr addrspace(1) %0, align 4\n  ret void\n}}\n"
        )
    };
    format!(
        r#"target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

@a.MTL_FC_INIT_0_b = linkonce_odr hidden addrspace(2) externally_initialized constant i8 undef, section "air.fc_initializer", align 1
@b.MTL_FC_INIT_1_i = linkonce_odr hidden addrspace(2) externally_initialized constant i32 undef, section "air.fc_initializer", align 4
@c.MTL_FC_INIT_2_j = linkonce_odr hidden addrspace(2) externally_initialized constant i32 undef, section "air.fc_initializer", align 4
@row0 = internal addrspace(2) constant [2 x i32] [i32 1, i32 2], align 4
@row1 = internal addrspace(2) constant [2 x i32] [i32 3, i32 4], align 4
@vals = internal addrspace(2) global [2 x i32] zeroinitializer, align 4
@table = internal addrspace(2) constant ptr addrspace(2) @c.MTL_FC_INIT_2_j, align 8
@to_rows = internal addrspace(2) global ptr addrspace(2) null, align 8
@to_vals = internal addrspace(2) global ptr addrspace(2) null, align 8
@to_constant = internal addrspace(2) global ptr addrspace(2) null, align 8
@to_table = internal addrspace(2) global ptr addrspace(2) null, align 8

{rows}{vals}{constant}{table}
define internal void @init() section "air.static_init" {{
  %1 = load i8, ptr addrspace(2) @a.MTL_FC_INIT_0_b, align 1
  %2 = trunc i8 %1 to i1
  %3 = select i1 %2, ptr addrspace(2) @row0, ptr addrspace(2) @row1
  store ptr addrspace(2) %3, ptr addrspace(2) @to_rows, align 8
  %4 = load i32, ptr addrspace(2) @c.MTL_FC_INIT_2_j, align 4
  store i32 %4, ptr addrspace(2) @vals, align 4
  %5 = select i1 %2, ptr addrspace(2) @row0, ptr addrspace(2) @vals
  store ptr addrspace(2) %5, ptr addrspace(2) @to_vals, align 8
  %6 = select i1 %2, ptr addrspace(2) @row0, ptr addrspace(2) @b.MTL_FC_INIT_1_i
  store ptr addrspace(2) %6, ptr addrspace(2) @to_constant, align 8
  %7 = select i1 %2, ptr addrspace(2) @row0, ptr addrspace(2) @table
  store ptr addrspace(2) %7, ptr addrspace(2) @to_table, align 8
  ret void
}}

!air.kernel = !{{!0, !4, !5, !12}}
!air.function_constants = !{{!6, !7, !8}}
!air.version = !{{!9}}
!air.language_version = !{{!10}}

!0 = !{{ptr @reads_rows, !1, !2}}
!1 = !{{}}
!2 = !{{!3, !11}}
!3 = !{{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"int", !"air.arg_name", !"out"}}
!4 = !{{ptr @reads_vals, !1, !2}}
!5 = !{{ptr @reads_constant, !1, !2}}
!6 = !{{ptr addrspace(2) @a.MTL_FC_INIT_0_b, !"bool", !"a", i32 0}}
!7 = !{{ptr addrspace(2) @b.MTL_FC_INIT_1_i, !"int", !"b", i32 1}}
!8 = !{{ptr addrspace(2) @c.MTL_FC_INIT_2_j, !"uint", !"c", i32 2}}
!9 = !{{i32 2, i32 8, i32 0}}
!10 = !{{!"Metal", i32 4, i32 0, i32 0}}
!11 = !{{i32 1, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"tid"}}
!12 = !{{ptr @reads_table, !1, !2}}
"#,
        rows = kernel("rows"),
        vals = kernel("vals"),
        constant = kernel("constant"),
        table = kernel("table"),
    )
}

/// A pointer the static initializer stores answers the constants that chose it where what it
/// points into holds nothing a constant decides (`reads_rows`: `a`, as Metal answers the fixture
/// `reflection/constant_pointers`). Where it may point into a global the initializer writes, a
/// declared constant, or a table holding a constant's address, what the function reads through
/// it was never measured against Metal: refused, naming the function.
#[test]
fn a_stored_pointer_answers_only_where_it_points_into_nothing_a_constant_decides() {
    let library = match metal2vulkan_devtools::llvm_assemble_text(&pointer_module()) {
        Ok(bitcode) => Library::from_air(&bitcode).expect("the module decodes"),
        Err(metal2vulkan_devtools::AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            return;
        }
        Err(metal2vulkan_devtools::AssembleError::Rejected(why)) => {
            panic!("llvm-as rejected the module: {why}")
        }
    };
    let read = |name: &str| {
        library
            .function(name)
            .expect("function")
            .function_constants_dictionary()
            .map(|c| {
                c.values()
                    .map(|c| {
                        (
                            c.name().to_owned(),
                            c.data_type(),
                            c.index(),
                            c.is_required(),
                        )
                    })
                    .collect::<Vec<_>>()
            })
    };
    assert_eq!(
        read("reads_rows").expect("known"),
        [("a".to_owned(), DataType::Bool, 0, true)]
    );
    for name in ["reads_vals", "reads_constant", "reads_table"] {
        match read(name) {
            Err(Error::Unsupported(u)) => {
                assert_eq!(u.feature, Feature::Reflection);
                assert_eq!(u.symbol, Some(DiagName::new(name)));
            }
            other => panic!("{name}: {other:?}"),
        }
    }
}

/// A static initializer whose if/else on `a` (`bool`, 0) holds another on `b` (`bool`, 1), merged
/// inside the outer arm, before it stores the merged value into `g`; the kernels `reads_g` and
/// `reads_none`.
fn nested_merge_module() -> String {
    r#"target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

@a.MTL_FC_INIT_0_b = linkonce_odr hidden addrspace(2) externally_initialized constant i8 undef, section "air.fc_initializer", align 1
@b.MTL_FC_INIT_1_b = linkonce_odr hidden addrspace(2) externally_initialized constant i8 undef, section "air.fc_initializer", align 1
@c.MTL_FC_INIT_2_i = linkonce_odr hidden addrspace(2) externally_initialized constant i32 undef, section "air.fc_initializer", align 4
@g = internal addrspace(2) global i32 0, align 4

define void @reads_g(ptr addrspace(1) noundef %0, i32 noundef %1) {
  %3 = load i32, ptr addrspace(2) @g, align 4
  store i32 %3, ptr addrspace(1) %0, align 4
  ret void
}

define void @reads_none(ptr addrspace(1) noundef %0, i32 noundef %1) {
  store i32 0, ptr addrspace(1) %0, align 4
  ret void
}

define internal void @init() section "air.static_init" {
  %1 = load i8, ptr addrspace(2) @a.MTL_FC_INIT_0_b, align 1
  %2 = trunc i8 %1 to i1
  br i1 %2, label %outer, label %done
outer:
  %3 = load i8, ptr addrspace(2) @b.MTL_FC_INIT_1_b, align 1
  %4 = trunc i8 %3 to i1
  %5 = load i32, ptr addrspace(2) @c.MTL_FC_INIT_2_i, align 4
  br i1 %4, label %halve, label %inner
halve:
  %6 = sdiv i32 %5, 2
  br label %inner
inner:
  %7 = phi i32 [ %6, %halve ], [ %5, %outer ]
  br label %done
done:
  %8 = phi i32 [ %7, %inner ], [ 0, %0 ]
  store i32 %8, ptr addrspace(2) @g, align 4
  ret void
}

!air.kernel = !{!0, !4}
!air.function_constants = !{!5, !6, !7}
!air.version = !{!8}
!air.language_version = !{!9}

!0 = !{ptr @reads_g, !1, !2}
!1 = !{}
!2 = !{!3, !10}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"int", !"air.arg_name", !"out"}
!4 = !{ptr @reads_none, !1, !2}
!5 = !{ptr addrspace(2) @a.MTL_FC_INIT_0_b, !"bool", !"a", i32 0}
!6 = !{ptr addrspace(2) @b.MTL_FC_INIT_1_b, !"bool", !"b", i32 1}
!7 = !{ptr addrspace(2) @c.MTL_FC_INIT_2_i, !"int", !"c", i32 2}
!8 = !{i32 2, i32 8, i32 0}
!9 = !{!"Metal", i32 4, i32 0, i32 0}
!10 = !{i32 1, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"tid"}
"#
    .to_owned()
}

/// A merge that a branch decides outside a loop (an if/else nested in another's arm): Metal's
/// answer is unmeasured (the fixture `reflection/constant_branches` measured one if/else, whose
/// merge no branch decides), so every function's dictionary is refused, naming it, one that
/// names no global too.
#[test]
fn a_merge_a_branch_decides_leaves_every_function_unknown() {
    let library = match metal2vulkan_devtools::llvm_assemble_text(&nested_merge_module()) {
        Ok(bitcode) => Library::from_air(&bitcode).expect("the module decodes"),
        Err(metal2vulkan_devtools::AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            return;
        }
        Err(metal2vulkan_devtools::AssembleError::Rejected(why)) => {
            panic!("llvm-as rejected the module: {why}")
        }
    };
    for name in ["reads_g", "reads_none"] {
        match library
            .function(name)
            .expect("function")
            .function_constants_dictionary()
        {
            Err(Error::Unsupported(u)) => {
                assert_eq!(u.feature, Feature::Reflection);
                assert_eq!(u.symbol, Some(DiagName::new(name)));
            }
            other => panic!("{name}: {other:?}"),
        }
    }
}
