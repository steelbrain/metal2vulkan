//! Companion derivation from the AIR and the companion's AIR text: each generated module assembles
//! with `llvm-as`, decodes through the product's ABI decoder with the user keys and IR types of the
//! stage it completes, and links with that stage in a render pipeline of the engine; on macOS
//! the typed spelling also passes the Metal toolchain at full verification (host tools only, no
//! device).

use super::*;
use crate::case::{attachmentless_fragment_test_case, RenderTargetResource, VaryingValue};
use crate::check::loop_free::DecodedModule;
use metal2vulkan_air::abi::{Interpolation, Sampling};
use metal2vulkan_devtools::{engine_dir, fixtures_dir, llvm_assemble, AssembleError};
use std::collections::BTreeMap;
use std::path::Path;

/// `text` assembled with `llvm-as`, or `None` (a visible skip) without it.
fn assembled(test: &str, text: &str) -> Option<Vec<u8>> {
    let scratch = crate::ScratchDir::new("companion-test").unwrap();
    let path = scratch.path().join("module.ll");
    std::fs::write(&path, text).unwrap();
    bitcode_of(test, &path)
}

fn bitcode_of(test: &str, path: &Path) -> Option<Vec<u8>> {
    match llvm_assemble(path) {
        Ok(bitcode) => Some(bitcode),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED {test}: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected {}: {why}", path.display()),
    }
}

/// A case of `stage` for `entry` with a 1x1 `format` render target 0.
pub(crate) fn case(stage: Stage, entry: &str, format: TextureFormat) -> AuthoredCase {
    let mut case = attachmentless_fragment_test_case(String::new(), entry.into());
    case.stage = stage;
    case.render_targets = vec![RenderTargetResource {
        index: 0,
        format,
        dimensions: [1, 1],
        initial_bytes_b64: String::new(),
    }];
    case
}

/// The companion of `case` over the module `bitcode`.
pub(crate) fn derived(case: &AuthoredCase, bitcode: &[u8]) -> Result<Option<Companion>, String> {
    let decoded = DecodedModule::from_bitcode("test", bitcode).unwrap();
    let abi = decoded.abi().unwrap();
    let interface =
        AirInterface::new(decoded.module(), abi, &case.entry, &BTreeMap::new()).unwrap();
    derive(case, decoded.module(), abi, &interface)
}

/// The decoded module of a companion's opaque spelling, or `None` without `llvm-as`.
fn companion_module(test: &str, companion: &Companion) -> Option<DecodedModule> {
    let bitcode = assembled(test, &companion.air_text(Spelling::Opaque))?;
    Some(DecodedModule::from_bitcode("companion", &bitcode).unwrap())
}

/// The single entry point of `decoded`, which is `entry`.
fn only_entry<'a>(decoded: &'a DecodedModule, entry: &str) -> &'a EntryPoint {
    let abi = decoded.abi().unwrap();
    assert_eq!(abi.entries.len(), 1);
    let ep = &abi.entries[0];
    let module = decoded.module();
    let name = module.func(ep.func).map(|f| module.symbols().get(f.name));
    assert_eq!(name, Some(Some(entry.as_bytes())));
    ep
}

/// Each fragment input of `entry` in `decoded`: its key and IR type.
fn fragment_inputs(decoded: &DecodedModule, entry: &str) -> Vec<(UserKey, TypeId)> {
    let ep = entry_point(decoded.module(), decoded.abi().unwrap(), entry).unwrap();
    let signature = signature(decoded.module(), ep).unwrap();
    ep.params
        .iter()
        .filter_map(|binding| match &binding.role {
            ParamRole::FragmentInput(varying) => Some((
                varying.key.clone(),
                signature.params[binding.param as usize],
            )),
            _ => None,
        })
        .collect()
}

/// Whether the IR types `a` of module `m` and `b` of module `n` are the same type.
fn same_type(m: &Module, a: TypeId, n: &Module, b: TypeId) -> bool {
    VaryingType::of(m, a).is_some() && VaryingType::of(m, a) == VaryingType::of(n, b)
}

#[test]
fn a_fragment_case_draws_through_a_passthrough_of_every_varying_it_reads() {
    let test = "a_fragment_case_draws_through_a_passthrough_of_every_varying_it_reads";
    let Some(bitcode) = bitcode_of(test, &engine_dir().join("stage_io.ll")) else {
        return;
    };
    let case = case(
        Stage::Fragment,
        "stage_io_fragment",
        TextureFormat::Rgba32Float,
    );
    let Some(Companion::PassthroughVertex(vertex)) = derived(&case, &bitcode).unwrap() else {
        panic!("a fragment case has a passthrough vertex");
    };
    // `uv` (half2), `tint` (float4, centroid), `id` (uint, flat), in parameter order; the
    // position, front-facing, and sample-mask inputs are no varyings.
    let values = vertex
        .varyings
        .iter()
        .map(|varying| (varying.ty, varying.type_name.as_str(), varying.value))
        .collect::<Vec<_>>();
    assert_eq!(
        values,
        [
            (
                VaryingType {
                    scalar: Scalar::F16,
                    lanes: 2
                },
                "half2",
                PassthroughValue::Uv
            ),
            (
                VaryingType {
                    scalar: Scalar::F32,
                    lanes: 4
                },
                "float4",
                PassthroughValue::UvHalfOne
            ),
            (
                VaryingType {
                    scalar: Scalar::I32,
                    lanes: 1
                },
                "uint",
                PassthroughValue::Counting
            ),
        ]
    );
    assert!(!vertex.viewport_array_index && !vertex.render_target_array_index);

    // The generated vertex function writes the position and exactly the fragment's varyings,
    // under their keys and at their IR types.
    let companion = Companion::PassthroughVertex(vertex);
    let Some(generated) = companion_module(test, &companion) else {
        return;
    };
    let fragment = DecodedModule::from_bitcode("stage_io", &bitcode).unwrap();
    let ep = only_entry(&generated, PASSTHROUGH_VERTEX_ENTRY);
    assert!(matches!(
        ep.stage,
        metal2vulkan_air::abi::Stage::Vertex { patch: None }
    ));
    let outputs = signature(generated.module(), ep).unwrap().outputs;
    assert!(matches!(ep.outputs[0].role, OutputRole::Position { .. }));
    let inputs = fragment_inputs(&fragment, "stage_io_fragment");
    assert_eq!(ep.outputs.len(), inputs.len() + 1);
    for ((output, &ty), (key, input_ty)) in ep.outputs[1..].iter().zip(&outputs[1..]).zip(&inputs) {
        assert_eq!(output.role, OutputRole::VertexOutput(key.clone()));
        assert!(same_type(
            generated.module(),
            ty,
            fragment.module(),
            *input_ty
        ));
    }
}

/// Floats, a pair of three-lane floats whose difference the fragment normalizes, a boolean, a
/// short vector, both layer indices, and a key that needs escaping.
const FRAGMENT: &str = r#"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

define <4 x float> @normals(<4 x float> %position, float %a, <3 x float> %n0, <3 x float> %n1, i1 %flag, <2 x i16> %s, i32 %viewport, i32 %layer) {
  %d = fsub fast <3 x float> %n0, %n1
  %x = extractelement <3 x float> %d, i32 0
  %r = call fast float @air.fast_rsqrt.f32(float %x)
  %c = insertelement <4 x float> zeroinitializer, float %r, i32 0
  ret <4 x float> %c
}

declare float @air.fast_rsqrt.f32(float)

!air.fragment = !{!0}
!air.version = !{!20}
!air.language_version = !{!21}

!0 = !{ptr @normals, !1, !3}
!1 = !{!2}
!2 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}
!3 = !{!4, !5, !6, !7, !8, !9, !10, !11}
!4 = !{i32 0, !"air.position", !"air.center", !"air.no_perspective", !"air.arg_type_name", !"float4"}
!5 = !{i32 1, !"air.fragment_input", !"generated(1af)", !"air.centroid", !"air.no_perspective", !"air.arg_type_name", !"float"}
!6 = !{i32 2, !"air.fragment_input", !"generated(2n0Dv3_f)", !"air.center", !"air.perspective", !"air.arg_type_name", !"float3"}
!7 = !{i32 3, !"air.fragment_input", !"user(\22odd\5C)", !"air.center", !"air.perspective", !"air.arg_type_name", !"float3"}
!8 = !{i32 4, !"air.fragment_input", !"generated(4flagb)", !"air.flat", !"air.arg_type_name", !"bool"}
!9 = !{i32 5, !"air.fragment_input", !"generated(1sDv2_s)", !"air.flat", !"air.arg_type_name", !"short2"}
!10 = !{i32 6, !"air.viewport_array_index", !"air.arg_type_name", !"uint"}
!11 = !{i32 7, !"air.render_target_array_index", !"air.arg_type_name", !"uint"}
!20 = !{i32 2, i32 8, i32 0}
!21 = !{!"Metal", i32 4, i32 0, i32 0}
"#;

#[test]
fn passthrough_values_follow_the_oracles_rules_and_keys_survive_the_text() {
    let test = "passthrough_values_follow_the_oracles_rules_and_keys_survive_the_text";
    let Some(bitcode) = assembled(test, FRAGMENT) else {
        return;
    };
    let case = case(Stage::Fragment, "normals", TextureFormat::Rgba32Float);
    let Some(Companion::PassthroughVertex(vertex)) = derived(&case, &bitcode).unwrap() else {
        panic!("a fragment case has a passthrough vertex");
    };
    let values = vertex
        .varyings
        .iter()
        .map(|varying| varying.value)
        .collect::<Vec<_>>();
    // The module normalizes the difference of two float3s, so each float3 after the first
    // varying (here both: the float `a` comes first) takes z = 1, as the oracle's MSL did.
    assert_eq!(
        values,
        [
            PassthroughValue::Quarter,
            PassthroughValue::UvZ { z_one: true },
            PassthroughValue::UvZ { z_one: true },
            PassthroughValue::Alternating,
            PassthroughValue::Counting,
        ]
    );
    assert!(vertex.viewport_array_index && vertex.render_target_array_index);
    assert_eq!(vertex.varyings[2].key.0.as_ref(), b"user(\"odd\\)");

    // Without the reciprocal square root the three-lane floats stay alike.
    let plain = FRAGMENT
        .replace(
            "call fast float @air.fast_rsqrt.f32(float %x)",
            "fmul fast float %x, %x",
        )
        .replace("declare float @air.fast_rsqrt.f32(float)", "");
    let bitcode_plain = assembled(test, &plain).unwrap();
    let Some(Companion::PassthroughVertex(alike)) = derived(&case, &bitcode_plain).unwrap() else {
        panic!("a fragment case has a passthrough vertex");
    };
    assert_eq!(
        (alike.varyings[1].value, alike.varyings[2].value),
        (
            PassthroughValue::UvZ { z_one: false },
            PassthroughValue::UvZ { z_one: false }
        )
    );

    let companion = Companion::PassthroughVertex(vertex);
    let Some(generated) = companion_module(test, &companion) else {
        return;
    };
    let fragment = DecodedModule::from_bitcode("fragment", &bitcode).unwrap();
    let ep = only_entry(&generated, PASSTHROUGH_VERTEX_ENTRY);
    let roles = ep
        .outputs
        .iter()
        .map(|output| output.role.clone())
        .collect::<Vec<_>>();
    let keys = fragment_inputs(&fragment, "normals")
        .into_iter()
        .map(|(key, _)| OutputRole::VertexOutput(key));
    let expected = [
        OutputRole::Position { invariant: false },
        OutputRole::ViewportArrayIndex,
        OutputRole::RenderTargetArrayIndex,
    ]
    .into_iter()
    .chain(keys)
    .collect::<Vec<_>>();
    assert_eq!(roles, expected);
}

/// A case of FRAGMENT's `normals` that authors four of its five varyings: the float `a` 2.0; the
/// float3 under the escaped key 1.0, a quiet NaN with a payload, and the negative of the smallest
/// subnormal; the boolean false; the short2 -1 and 7. `n0` keeps its default.
fn authored_normals() -> AuthoredCase {
    let mut case = case(Stage::Fragment, "normals", TextureFormat::Rgba32Float);
    let value = |key: &str, lanes: &[u32]| VaryingValue {
        key: key.into(),
        lanes: lanes.to_vec(),
    };
    case.varying_values = vec![
        value("generated(1af)", &[0x4000_0000]),
        value("user(\"odd\\)", &[0x3F80_0000, 0x7FC0_0001, 0x8000_0001]),
        value("generated(4flagb)", &[0]),
        value("generated(1sDv2_s)", &[0xFFFF, 7]),
    ];
    case
}

#[test]
fn authored_varying_values_replace_the_defaults_of_their_keys_only() {
    let test = "authored_varying_values_replace_the_defaults_of_their_keys_only";
    let Some(bitcode) = assembled(test, FRAGMENT) else {
        return;
    };
    let Some(Companion::PassthroughVertex(vertex)) =
        derived(&authored_normals(), &bitcode).unwrap()
    else {
        panic!("a fragment case has a passthrough vertex");
    };
    let values = vertex
        .varyings
        .iter()
        .map(|varying| varying.value)
        .collect::<Vec<_>>();
    assert_eq!(
        values,
        [
            PassthroughValue::Authored {
                lanes: [0x4000_0000, 0, 0, 0]
            },
            PassthroughValue::UvZ { z_one: true },
            PassthroughValue::Authored {
                lanes: [0x3F80_0000, 0x7FC0_0001, 0x8000_0001, 0]
            },
            PassthroughValue::Authored { lanes: [0; 4] },
            PassthroughValue::Authored {
                lanes: [0xFFFF, 7, 0, 0]
            },
        ]
    );

    // Each lane is spelled with its exact bits: a float as the double of its value (a NaN's
    // payload moved to the double's top bits), an integer as its signed value.
    let companion = Companion::PassthroughVertex(vertex);
    let text = companion.air_text(Spelling::Opaque);
    for constant in [
        "float 0x4000000000000000,",
        "<3 x float> <float 0x3FF0000000000000, float 0x7FF8000020000000, \
         float 0xB6A0000000000000>",
        "i1 false,",
        "<2 x i16> <i16 -1, i16 7>",
    ] {
        assert!(text.contains(constant), "{constant} in\n{text}");
    }
    let Some(generated) = companion_module(test, &companion) else {
        return;
    };
    let ep = only_entry(&generated, PASSTHROUGH_VERTEX_ENTRY);
    assert_eq!(ep.outputs.len(), 8);
}

#[test]
fn an_authored_varying_value_must_fit_a_varying_the_fragment_reads() {
    let test = "an_authored_varying_value_must_fit_a_varying_the_fragment_reads";
    let Some(bitcode) = assembled(test, FRAGMENT) else {
        return;
    };
    for (key, lanes, refusal) in [
        (
            "generated(9missingf)",
            vec![0],
            "names no varying the fragment function reads",
        ),
        (
            "generated(2n0Dv3_f)",
            vec![0, 0],
            "has 2 lanes for a 3-lane float3 varying",
        ),
        ("generated(4flagb)", vec![2], "wider than its bool"),
        (
            "generated(1sDv2_s)",
            vec![0x1_0000, 0],
            "wider than its short2",
        ),
    ] {
        let mut case = case(Stage::Fragment, "normals", TextureFormat::Rgba32Float);
        case.varying_values = vec![VaryingValue {
            key: key.into(),
            lanes,
        }];
        let error = derived(&case, &bitcode).unwrap_err();
        assert!(error.contains(refusal), "{key}: {error}");
    }
}

#[test]
fn the_two_spellings_differ_only_in_how_metadata_names_the_entry() {
    let vertex = Companion::PassthroughVertex(PassthroughVertex {
        varyings: vec![PassthroughVarying {
            key: UserKey(b"generated(2uvDv2_f)".as_slice().into()),
            ty: VaryingType {
                scalar: Scalar::F32,
                lanes: 2,
            },
            type_name: "float2".into(),
            value: PassthroughValue::Uv,
        }],
        viewport_array_index: false,
        render_target_array_index: false,
    });
    let typed = vertex.air_text(Spelling::Typed);
    let opaque = vertex.air_text(Spelling::Opaque);
    let differing = typed
        .lines()
        .zip(opaque.lines())
        .filter(|(a, b)| a != b)
        .collect::<Vec<_>>();
    assert_eq!(typed.lines().count(), opaque.lines().count());
    assert_eq!(differing.len(), 1, "{differing:?}");
    let (typed_line, opaque_line) = differing[0];
    assert!(
        typed_line
            .contains("{ <4 x float>, <2 x float> }> (i32)* @metal2vulkan_passthrough_vertex"),
        "{typed_line}"
    );
    assert!(
        opaque_line.contains("!{ptr @metal2vulkan_passthrough_vertex, "),
        "{opaque_line}"
    );
    assert!(!typed.contains(" ptr ") && !opaque.contains(")*"));
}

#[test]
fn an_observed_vertex_case_draws_into_an_observer_of_its_value() {
    let test = "an_observed_vertex_case_draws_into_an_observer_of_its_value";
    let Some(triangle) = bitcode_of(test, &fixtures_dir().join("vertex_varying_triangle.ll"))
    else {
        return;
    };
    let mut observed = case(
        Stage::Vertex,
        "vertex_varying_triangle",
        TextureFormat::Rgba32Float,
    );
    observed.vertex_observation = Some(VertexObservation::Varying { location: 0 });
    let Some(Companion::ObserverFragment(observer)) = derived(&observed, &triangle).unwrap() else {
        panic!("an observed vertex case has an observer fragment");
    };
    assert_eq!(observer.output, ObserverOutput::Float);
    let ObservedValue::Varying {
        key, ty, type_name, ..
    } = &observer.input
    else {
        panic!("{observer:?}");
    };
    assert_eq!(key.0.as_ref(), b"generated(2uvDv2_f)");
    assert_eq!(
        (*ty, type_name.as_str()),
        (
            VaryingType {
                scalar: Scalar::F32,
                lanes: 2
            },
            "float2"
        )
    );
    let companion = Companion::ObserverFragment(observer.clone());
    let Some(generated) = companion_module(test, &companion) else {
        return;
    };
    let ep = only_entry(&generated, OBSERVER_FRAGMENT_ENTRY);
    assert!(matches!(
        ep.stage,
        metal2vulkan_air::abi::Stage::Fragment { .. }
    ));
    assert!(matches!(
        &ep.params[0].role,
        ParamRole::FragmentInput(varying) if varying.key == *key
            && varying.interpolation == Interpolation::Smooth {
                sampling: Sampling::Center,
                perspective: true,
            }
    ));
    assert!(matches!(
        ep.outputs[0].role,
        OutputRole::RenderTarget { index: 0, .. }
    ));

    // The position.
    observed.vertex_observation = Some(VertexObservation::Position);
    assert_eq!(
        derived(&observed, &triangle).unwrap(),
        Some(Companion::ObserverFragment(ObserverFragment {
            input: ObservedValue::Position,
            output: ObserverOutput::Float,
        }))
    );
    // What no observer can read, or no target can receive.
    observed.vertex_observation = Some(VertexObservation::Varying { location: 1 });
    assert_eq!(
        derived(&observed, &triangle).unwrap_err(),
        "vertex observation selects undeclared varying location 1"
    );
    observed.vertex_observation = Some(VertexObservation::Varying { location: 0 });
    observed.render_targets[0].format = TextureFormat::Rgba32Uint;
    assert_eq!(
        derived(&observed, &triangle).unwrap_err(),
        "vertex observation requires render target format Rgba32Float, got Rgba32Uint"
    );
    // A case with neither a fragment stage nor an observation has no companion.
    observed.vertex_observation = None;
    assert_eq!(derived(&observed, &triangle).unwrap(), None);
}

/// A mesh case's observation reads its mesh function's output through the same observer fragment:
/// its position, or a vertex or primitive data field under its user key and the type the mesh
/// function writes it with (`air.set_primitive_data_mesh.v4i32`), a primitive data field read flat
/// (`mesh_draws_grid` writes two `uint4` primitive fields and a position).
#[test]
fn an_observed_mesh_case_draws_into_an_observer_of_its_output() {
    let test = "an_observed_mesh_case_draws_into_an_observer_of_its_output";
    let Some(bitcode) = bitcode_of(test, &engine_dir().join("mesh_draws.ll")) else {
        return;
    };
    let mut observed = case(Stage::Mesh, "mesh_draws_grid", TextureFormat::Rgba32Uint);
    let observe = |case: &mut AuthoredCase, observation| {
        case.mesh_draw = Some(crate::case::MeshDraw {
            object: None,
            mesh: None,
            fragment: None,
            grid: crate::case::MeshGrid::Threadgroups([1, 1, 1]),
            threads_per_object_threadgroup: [1, 1, 1],
            threads_per_mesh_threadgroup: [1, 1, 1],
            rasterization: true,
            observation: Some(observation),
            object_launch: None,
        });
    };
    observe(&mut observed, MeshObservation::PrimitiveData { index: 1 });
    let Some(Companion::MeshStages(MeshStages {
        object: None,
        mesh: None,
        fragment: Some(observer),
    })) = derived(&observed, &bitcode).unwrap()
    else {
        panic!("an observed mesh case has an observer fragment");
    };
    let key = UserKey(b"generated(1bDv4_j)".as_slice().into());
    assert_eq!(
        observer,
        ObserverFragment {
            input: ObservedValue::Varying {
                key: key.clone(),
                ty: VaryingType {
                    scalar: Scalar::I32,
                    lanes: 4
                },
                type_name: "uint4".into(),
                per_primitive: true,
            },
            output: ObserverOutput::Uint,
        }
    );
    // The generated fragment reads the field flat, under its key.
    let companion = Companion::MeshStages(MeshStages {
        object: None,
        mesh: None,
        fragment: Some(observer),
    });
    if let Some(generated) = companion_module(test, &companion) {
        let ep = only_entry(&generated, OBSERVER_FRAGMENT_ENTRY);
        assert!(matches!(
            &ep.params[0].role,
            ParamRole::FragmentInput(varying) if varying.key == key
                && varying.interpolation == Interpolation::Flat
        ));
    }

    // The position.
    observed.render_targets[0].format = TextureFormat::Rgba32Float;
    observe(&mut observed, MeshObservation::Position);
    assert_eq!(
        derived(&observed, &bitcode).unwrap(),
        Some(Companion::MeshStages(MeshStages {
            object: None,
            mesh: None,
            fragment: Some(ObserverFragment {
                input: ObservedValue::Position,
                output: ObserverOutput::Float,
            }),
        }))
    );
    // What the mesh function does not declare, or no target can receive.
    observe(&mut observed, MeshObservation::VertexData { index: 0 });
    assert_eq!(
        derived(&observed, &bitcode).unwrap_err(),
        "mesh observation selects undeclared vertex data field 0"
    );
    observe(&mut observed, MeshObservation::PrimitiveData { index: 2 });
    assert_eq!(
        derived(&observed, &bitcode).unwrap_err(),
        "mesh observation selects undeclared primitive data field 2"
    );
    observe(&mut observed, MeshObservation::PrimitiveData { index: 0 });
    assert_eq!(
        derived(&observed, &bitcode).unwrap_err(),
        "mesh observation requires render target format Rgba32Uint, got Rgba32Float"
    );
    // A mesh case that names its fragment partner has no companion.
    observed.mesh_draw.as_mut().unwrap().observation = None;
    observed.mesh_draw.as_mut().unwrap().fragment = Some("mesh_draws_fragment".into());
    assert_eq!(derived(&observed, &bitcode).unwrap(), None);
}

/// An object case's launch observation: a mesh function in its mesh partner's place that
/// declares the object function's payload at its size and alignment, and draws one triangle a
/// pixel with one `uint4` primitive data field, which the observer fragment reads flat under the
/// same key; a payload needs whole words and a target wide enough for a row.
#[test]
fn an_object_case_s_launch_draws_through_a_generated_mesh_function() {
    let test = "an_object_case_s_launch_draws_through_a_generated_mesh_function";
    let Some(bitcode) = bitcode_of(test, &engine_dir().join("mesh_draws.ll")) else {
        return;
    };
    let mut launched = case(
        Stage::Object,
        "mesh_draws_object",
        TextureFormat::Rgba32Uint,
    );
    launched.render_targets[0].dimensions = [4, 8];
    launched.mesh_draw = Some(crate::case::MeshDraw {
        object: None,
        mesh: None,
        fragment: None,
        grid: crate::case::MeshGrid::Threadgroups([1, 1, 1]),
        threads_per_object_threadgroup: [4, 1, 1],
        threads_per_mesh_threadgroup: [2, 1, 1],
        rasterization: true,
        observation: Some(MeshObservation::Launch),
        object_launch: None,
    });
    let Some(Companion::MeshStages(stages)) = derived(&launched, &bitcode).unwrap() else {
        panic!("a launch observation has generated mesh stages");
    };
    // `DrawsPayload` is two `uint4`s: 32 bytes, aligned to 16.
    let observer = LaunchObserver {
        payload: Some(LaunchPayload {
            words: 8,
            align: 16,
        }),
        target: [4, 8],
    };
    assert_eq!(stages.mesh, Some(observer.clone()));
    assert_eq!(observer.pixels(), 3);
    let key = UserKey(LAUNCH_KEY.into());
    assert_eq!(
        stages.fragment,
        Some(ObserverFragment {
            input: ObservedValue::Varying {
                key: key.clone(),
                ty: VaryingType {
                    scalar: Scalar::I32,
                    lanes: 4
                },
                type_name: "uint4".into(),
                per_primitive: true,
            },
            output: ObserverOutput::Uint,
        })
    );
    let companion = Companion::MeshStages(stages);
    assert_eq!(companion.entry(), OBSERVER_MESH_ENTRY);
    assert_eq!(
        [Stage::Object, Stage::Mesh, Stage::Fragment].map(|stage| companion.stage_entry(stage)),
        [
            None,
            Some(OBSERVER_MESH_ENTRY),
            Some(OBSERVER_FRAGMENT_ENTRY)
        ]
    );
    if let Some(generated) = companion_module(test, &companion) {
        let abi = generated.abi().unwrap();
        assert_eq!(abi.entries.len(), 2);
        let mesh = entry_point(generated.module(), abi, OBSERVER_MESH_ENTRY).unwrap();
        let outputs = mesh
            .params
            .iter()
            .find_map(|binding| match &binding.role {
                ParamRole::Mesh(outputs) => Some(outputs.as_ref()),
                _ => None,
            })
            .unwrap();
        // A triangle for the grid's pixel and each of the payload's two pixels.
        assert_eq!(outputs.vertex.len(), 1);
        assert!(matches!(
            outputs.primitive.as_slice(),
            [field] if matches!(&field.role, MeshFieldRole::PrimitiveData { slot: 0, key: k } if *k == key)
        ));
        let payload = mesh
            .params
            .iter()
            .find(|binding| matches!(binding.role, ParamRole::Payload { .. }))
            .unwrap();
        assert_eq!((payload.ty.size, payload.ty.align), (Some(32), Some(16)));
        let writes = MeshWrites::of(generated.module());
        assert!(writes.position);
        assert_eq!(
            writes.data_type(0, true).map(mesh_data_type),
            Ok(Some(VaryingType {
                scalar: Scalar::I32,
                lanes: 4
            }))
        );
        let ep = entry_point(generated.module(), abi, OBSERVER_FRAGMENT_ENTRY).unwrap();
        assert!(matches!(
            &ep.params[0].role,
            ParamRole::FragmentInput(varying) if varying.key == key
                && varying.interpolation == Interpolation::Flat
        ));
        // The typed spelling differs only in its pointers.
        let typed = companion.air_text(Spelling::Typed);
        assert!(typed.contains("%struct._mesh_t addrspace(7)* %out"));
        assert!(typed.contains("%struct.metal2vulkan_payload addrspace(6)* %payload"));
    }

    // What the observer cannot draw.
    launched.render_targets[0].dimensions = [2, 8];
    assert_eq!(
        derived(&launched, &bitcode).unwrap_err(),
        "launch observation draws 3 pixels a row, and render target 0 is 2 wide"
    );
    launched.render_targets[0].format = TextureFormat::Rgba32Float;
    assert_eq!(
        derived(&launched, &bitcode).unwrap_err(),
        "launch observation requires render target format Rgba32Uint, got Rgba32Float"
    );
    // An object case that names its mesh and fragment partners has no companion.
    let draw = launched.mesh_draw.as_mut().unwrap();
    draw.observation = None;
    draw.mesh = Some("mesh_draws_mesh".into());
    draw.fragment = Some("mesh_draws_fragment".into());
    assert_eq!(derived(&launched, &bitcode).unwrap(), None);
}

/// The draw of `mesh_draws_mesh` with its fragment partner and an object launch: a 32-byte payload
/// (`DrawsPayload`'s `object` words 11 to 14, then `grid.x` 1) and 2 by 1 mesh threadgroups.
fn payload_draw() -> crate::case::MeshDraw {
    crate::case::MeshDraw {
        object: None,
        mesh: None,
        fragment: Some("mesh_draws_fragment".into()),
        grid: crate::case::MeshGrid::Threadgroups([1, 1, 1]),
        threads_per_object_threadgroup: [1, 1, 1],
        threads_per_mesh_threadgroup: [2, 1, 1],
        rasterization: true,
        observation: None,
        object_launch: Some(crate::case::ObjectLaunch {
            payload_b64: base64::Engine::encode(
                &base64::engine::general_purpose::STANDARD,
                [11u32, 12, 13, 14, 1, 0, 0, 0]
                    .map(u32::to_le_bytes)
                    .concat(),
            ),
            mesh_threadgroups: [2, 1, 1],
        }),
    }
}

/// A mesh case's object launch: an object function in its object partner's place that declares
/// the payload at the authored length and the mesh function's alignment, stores its words and
/// sets the authored grid; the launch holds at least what the mesh function reads.
#[test]
fn a_mesh_case_s_object_launch_draws_through_a_generated_object_function() {
    let test = "a_mesh_case_s_object_launch_draws_through_a_generated_object_function";
    let Some(bitcode) = bitcode_of(test, &engine_dir().join("mesh_draws.ll")) else {
        return;
    };
    let mut payload = case(Stage::Mesh, "mesh_draws_mesh", TextureFormat::Rgba32Uint);
    payload.mesh_draw = Some(payload_draw());
    let Some(Companion::MeshStages(stages)) = derived(&payload, &bitcode).unwrap() else {
        panic!("an object launch has generated mesh stages");
    };
    let object = PayloadObject {
        words: vec![11, 12, 13, 14, 1, 0, 0, 0],
        align: 16,
        mesh_threadgroups: [2, 1, 1],
    };
    assert_eq!(
        stages,
        MeshStages {
            object: Some(object),
            mesh: None,
            fragment: None,
        }
    );
    let companion = Companion::MeshStages(stages);
    assert_eq!(companion.entry(), PAYLOAD_OBJECT_ENTRY);
    assert_eq!(
        [Stage::Object, Stage::Mesh, Stage::Fragment].map(|stage| companion.stage_entry(stage)),
        [Some(PAYLOAD_OBJECT_ENTRY), None, None]
    );
    if let Some(generated) = companion_module(test, &companion) {
        let ep = only_entry(&generated, PAYLOAD_OBJECT_ENTRY);
        let payload = ep
            .params
            .iter()
            .find(|binding| matches!(binding.role, ParamRole::Payload { .. }))
            .unwrap();
        assert_eq!((payload.ty.size, payload.ty.align), (Some(32), Some(16)));
        assert!(ep
            .params
            .iter()
            .any(|binding| matches!(binding.role, ParamRole::MeshGridProperties)));
        let typed = companion.air_text(Spelling::Typed);
        assert!(typed.contains("%struct._mesh_grid_properties_t addrspace(3)* %grid"));
        assert!(typed.contains("%struct.metal2vulkan_payload addrspace(6)* %payload"));
    }

    // With an observation in its fragment partner's place, the companion holds both.
    let draw = payload.mesh_draw.as_mut().unwrap();
    draw.fragment = None;
    draw.observation = Some(MeshObservation::PrimitiveData { index: 0 });
    let Some(Companion::MeshStages(stages)) = derived(&payload, &bitcode).unwrap() else {
        panic!("an object launch has generated mesh stages");
    };
    assert!(stages.object.is_some() && stages.fragment.is_some());
    // A launch that holds less than the mesh function reads.
    payload.mesh_draw.as_mut().unwrap().object_launch = Some(crate::case::ObjectLaunch {
        payload_b64: base64::Engine::encode(&base64::engine::general_purpose::STANDARD, [0u8; 16]),
        mesh_threadgroups: [1, 1, 1],
    });
    assert_eq!(
        derived(&payload, &bitcode).unwrap_err(),
        "the object launch's payload holds 16 bytes, and the mesh function reads 32"
    );
}

/// The IR carries no integer sign: the observed target's format states it.
#[test]
fn an_integer_observation_takes_its_sign_from_the_target() {
    let test = "an_integer_observation_takes_its_sign_from_the_target";
    let Some(bitcode) = bitcode_of(test, &engine_dir().join("stage_io.ll")) else {
        return;
    };
    // `stage_io_vertex`'s varyings: `uv` (half2), `tint` (float4), `id` (uint).
    let mut observed = case(Stage::Vertex, "stage_io_vertex", TextureFormat::Rgba32Sint);
    observed.vertex_observation = Some(VertexObservation::Varying { location: 2 });
    for (format, output) in [
        (TextureFormat::Rgba32Sint, ObserverOutput::Sint),
        (TextureFormat::Rgba32Uint, ObserverOutput::Uint),
    ] {
        observed.render_targets[0].format = format;
        let Some(Companion::ObserverFragment(observer)) = derived(&observed, &bitcode).unwrap()
        else {
            panic!("an observed vertex case has an observer fragment");
        };
        assert_eq!(observer.output, output);
        let Some(generated) = companion_module(test, &Companion::ObserverFragment(observer)) else {
            return;
        };
        let ep = only_entry(&generated, OBSERVER_FRAGMENT_ENTRY);
        assert!(matches!(
            &ep.params[0].role,
            ParamRole::FragmentInput(varying) if varying.interpolation == Interpolation::Flat
        ));
    }
    observed.render_targets[0].format = TextureFormat::Rgba32Float;
    assert!(derived(&observed, &bitcode).is_err());
    // A half varying widens into a float target.
    observed.vertex_observation = Some(VertexObservation::Varying { location: 0 });
    let Some(Companion::ObserverFragment(observer)) = derived(&observed, &bitcode).unwrap() else {
        panic!("an observed vertex case has an observer fragment");
    };
    assert!(companion_module(test, &Companion::ObserverFragment(observer)).is_some());
}

/// Every companion links with the stage it completes in one engine render pipeline: by user
/// key, at the types the two sides state.
#[test]
fn generated_companions_link_with_their_cases_in_the_engine() {
    use metal2vulkan::library::Library;
    use metal2vulkan::pipeline::{
        FunctionDescriptor, RenderPipelineDescriptor, StageSource, VertexAttributeDescriptor,
        VertexBufferLayoutDescriptor, VertexDescriptor,
    };
    let test = "generated_companions_link_with_their_cases_in_the_engine";
    let (Some(fragment), Some(triangle)) = (
        bitcode_of(test, &fixtures_dir().join("fragment_varying_color.ll")),
        bitcode_of(test, &fixtures_dir().join("vertex_varying_triangle.ll")),
    ) else {
        return;
    };
    let device = crate::engine::load_device("lavapipe").unwrap();
    let function = |bitcode: &[u8], entry: &str| {
        FunctionDescriptor::new(Library::from_air(bitcode).unwrap().function(entry).unwrap())
    };
    let pipeline = |vertex, fragment, layout: Option<VertexDescriptor>| {
        let mut descriptor = RenderPipelineDescriptor::new(StageSource::Function(vertex));
        descriptor.fragment = Some(StageSource::Function(fragment));
        descriptor.vertex_descriptor = layout;
        descriptor.color_attachments[0] =
            Some(metal2vulkan::pipeline::ColorAttachmentDescriptor::new(
                metal2vulkan::metal::PixelFormat::RGBA32Float,
            ));
        device.compile_render_pipeline(&descriptor, &metal2vulkan::Budget::standard())
    };

    let case_of = case(
        Stage::Fragment,
        "fragment_varying_color",
        TextureFormat::Rgba32Float,
    );
    let passthrough = derived(&case_of, &fragment).unwrap().unwrap();
    let companion = passthrough.opaque_bitcode().unwrap();
    // The module names its own source, so two assemblies (each from its own scratch file) are
    // the same bitcode and hash to the same pipeline key.
    assert_eq!(passthrough.opaque_bitcode().unwrap(), companion);
    let linked = pipeline(
        function(&companion, passthrough.entry()),
        function(&fragment, "fragment_varying_color"),
        None,
    )
    .unwrap();
    assert_eq!(linked.stages.len(), 2);

    let mut observed = case(
        Stage::Vertex,
        "vertex_varying_triangle",
        TextureFormat::Rgba32Float,
    );
    let mut layout = VertexDescriptor::new();
    for location in 0..2 {
        layout
            .layouts
            .insert(30 - location, VertexBufferLayoutDescriptor::new(Some(8)));
        layout.attributes.insert(
            location,
            VertexAttributeDescriptor::new(
                metal2vulkan::metal::VertexFormat::Float2,
                0,
                30 - location,
            ),
        );
    }
    for observation in [
        VertexObservation::Varying { location: 0 },
        VertexObservation::Position,
    ] {
        observed.vertex_observation = Some(observation);
        let observer = derived(&observed, &triangle).unwrap().unwrap();
        let companion = observer.opaque_bitcode().unwrap();
        let linked = pipeline(
            function(&triangle, "vertex_varying_triangle"),
            function(&companion, observer.entry()),
            Some(layout.clone()),
        )
        .unwrap();
        assert_eq!(linked.stages.len(), 2);
    }

    // The launch observer's mesh function under the object function whose payload it reads, and
    // its observer fragment of the primitive data it writes.
    let Some(draws) = bitcode_of(test, &engine_dir().join("mesh_draws.ll")) else {
        return;
    };
    let mut launched = case(
        Stage::Object,
        "mesh_draws_object",
        TextureFormat::Rgba32Uint,
    );
    launched.render_targets[0].dimensions = [4, 8];
    launched.mesh_draw = Some(crate::case::MeshDraw {
        object: None,
        mesh: None,
        fragment: None,
        grid: crate::case::MeshGrid::Threadgroups([1, 1, 1]),
        threads_per_object_threadgroup: [4, 1, 1],
        threads_per_mesh_threadgroup: [2, 1, 1],
        rasterization: true,
        observation: Some(MeshObservation::Launch),
        object_launch: None,
    });
    let stages = derived(&launched, &draws).unwrap().unwrap();
    let companion = stages.opaque_bitcode().unwrap();
    let mut descriptor = metal2vulkan::pipeline::MeshPipelineDescriptor::new(
        StageSource::Function(function(&companion, OBSERVER_MESH_ENTRY)),
    );
    descriptor.object = Some(StageSource::Function(function(&draws, "mesh_draws_object")));
    descriptor.fragment = Some(StageSource::Function(function(
        &companion,
        OBSERVER_FRAGMENT_ENTRY,
    )));
    descriptor.color_attachments[0] = Some(metal2vulkan::pipeline::ColorAttachmentDescriptor::new(
        metal2vulkan::metal::PixelFormat::RGBA32Uint,
    ));
    let linked = device
        .compile_mesh_pipeline(&descriptor, &metal2vulkan::Budget::standard())
        .unwrap();
    // The object, mesh and fragment stages (with the mesh emulation's, where the device has no
    // mesh shaders).
    assert!(linked.stages.len() >= 3, "{}", linked.stages.len());

    // The payload object under the mesh function that reads its payload, and that mesh
    // function's fragment partner.
    let mut payload = case(Stage::Mesh, "mesh_draws_mesh", TextureFormat::Rgba32Uint);
    payload.mesh_draw = Some(payload_draw());
    let stages = derived(&payload, &draws).unwrap().unwrap();
    let companion = stages.opaque_bitcode().unwrap();
    let mut descriptor = metal2vulkan::pipeline::MeshPipelineDescriptor::new(
        StageSource::Function(function(&draws, "mesh_draws_mesh")),
    );
    descriptor.object = Some(StageSource::Function(function(
        &companion,
        PAYLOAD_OBJECT_ENTRY,
    )));
    descriptor.fragment = Some(StageSource::Function(function(
        &draws,
        "mesh_draws_fragment",
    )));
    for slot in descriptor.color_attachments.iter_mut().take(2) {
        *slot = Some(metal2vulkan::pipeline::ColorAttachmentDescriptor::new(
            metal2vulkan::metal::PixelFormat::RGBA32Uint,
        ));
    }
    let linked = device
        .compile_mesh_pipeline(&descriptor, &metal2vulkan::Budget::standard())
        .unwrap();
    assert!(linked.stages.len() >= 3, "{}", linked.stages.len());
}

/// The typed spelling is what the Metal oracle loads: `metal-as` assembles it, `metal-opt`
/// verifies it at the strictest level (`full`, which compares every AIR declaration with the
/// toolchain's table), and `metallib` packages it. Host tools only; no device.
#[cfg(target_os = "macos")]
#[test]
fn the_typed_spelling_passes_the_metal_toolchain_at_full_verification() {
    use crate::gpu::compile::{has_tool, xcrun};
    let test = "the_typed_spelling_passes_the_metal_toolchain_at_full_verification";
    if !has_tool("metal-opt") {
        eprintln!("SKIPPED {test}: xcrun metal-opt not found");
        return;
    }
    let (Some(stage_io), Some(normals), Some(triangle)) = (
        bitcode_of(test, &engine_dir().join("stage_io.ll")),
        assembled(test, FRAGMENT),
        bitcode_of(test, &fixtures_dir().join("vertex_varying_triangle.ll")),
    ) else {
        return;
    };
    let mut companions = vec![
        derived(
            &case(
                Stage::Fragment,
                "stage_io_fragment",
                TextureFormat::Rgba32Float,
            ),
            &stage_io,
        ),
        derived(
            &case(Stage::Fragment, "normals", TextureFormat::Rgba32Float),
            &normals,
        ),
        derived(&authored_normals(), &normals),
    ];
    for (bitcode, entry, location, format) in [
        (
            &triangle,
            "vertex_varying_triangle",
            Some(0),
            TextureFormat::Rgba32Float,
        ),
        (
            &triangle,
            "vertex_varying_triangle",
            None,
            TextureFormat::Rgba32Float,
        ),
        (
            &stage_io,
            "stage_io_vertex",
            Some(0),
            TextureFormat::Rgba32Float,
        ),
        (
            &stage_io,
            "stage_io_vertex",
            Some(2),
            TextureFormat::Rgba32Sint,
        ),
        (
            &stage_io,
            "stage_io_vertex",
            Some(2),
            TextureFormat::Rgba32Uint,
        ),
    ] {
        let mut observed = case(Stage::Vertex, entry, format);
        observed.vertex_observation = Some(match location {
            Some(location) => VertexObservation::Varying { location },
            None => VertexObservation::Position,
        });
        companions.push(derived(&observed, bitcode));
    }
    // An object case's launch observer, of `mesh_draws_object`'s 32-byte payload.
    let Some(draws) = bitcode_of(test, &engine_dir().join("mesh_draws.ll")) else {
        return;
    };
    let mut launched = case(
        Stage::Object,
        "mesh_draws_object",
        TextureFormat::Rgba32Uint,
    );
    launched.render_targets[0].dimensions = [4, 8];
    launched.mesh_draw = Some(crate::case::MeshDraw {
        object: None,
        mesh: None,
        fragment: None,
        grid: crate::case::MeshGrid::Threadgroups([1, 1, 1]),
        threads_per_object_threadgroup: [1, 1, 1],
        threads_per_mesh_threadgroup: [1, 1, 1],
        rasterization: true,
        observation: Some(MeshObservation::Launch),
        object_launch: None,
    });
    companions.push(derived(&launched, &draws));
    // A mesh case's payload object, of `mesh_draws_mesh`'s 32-byte payload.
    let mut payload = case(Stage::Mesh, "mesh_draws_mesh", TextureFormat::Rgba32Uint);
    payload.mesh_draw = Some(payload_draw());
    companions.push(derived(&payload, &draws));
    let scratch = crate::ScratchDir::new("companion-metal").unwrap();
    for (index, companion) in companions.into_iter().enumerate() {
        let companion = companion.unwrap().unwrap();
        let text = scratch.path().join(format!("{index}.air.ll"));
        let air = scratch.path().join(format!("{index}.air"));
        let verified = scratch.path().join(format!("{index}.verified.ll"));
        let metallib = scratch.path().join(format!("{index}.metallib"));
        std::fs::write(&text, companion.air_text(Spelling::Typed)).unwrap();
        for (tool, args) in [
            (
                "metal-as",
                vec![text.as_os_str(), "-o".as_ref(), air.as_os_str()],
            ),
            (
                "metal-opt",
                vec![
                    "-S".as_ref(),
                    "-passes=verify".as_ref(),
                    "--air-verification-level=full".as_ref(),
                    air.as_os_str(),
                    "-o".as_ref(),
                    verified.as_os_str(),
                ],
            ),
            (
                "metallib",
                vec![air.as_os_str(), "-o".as_ref(), metallib.as_os_str()],
            ),
        ] {
            let output = xcrun(tool, args).unwrap();
            assert!(
                output.success,
                "{tool} refused companion {index} ({companion:?}): {}{}",
                output.stdout, output.stderr
            );
        }
    }
}
