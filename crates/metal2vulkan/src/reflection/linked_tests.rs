//! The linked-use answers against Metal's, for the owned fixture `examples/icb_linked_arguments`
//! (its functions' bodies are the probes', on an M4 under macOS 27.0.1): which stages of a pipeline
//! use a direct texture or sampler once linked, and whether a vertex function returns a value. The
//! runtime lane's test (`indirect_command_buffers.rs`) makes the same states.

use super::{DirectArgumentUse, LinkedUseGap};
use crate::budget::Budget;
use crate::library::Library;
use crate::metal::{BlendFactor, ColorWriteMask, PixelFormat};
use crate::pipeline::testing::{fixtures_dir, llvm_assemble, AssembleError};
use crate::pipeline::{
    ColorAttachmentDescriptor, FunctionDescriptor, MeshPipelineDescriptor,
    RenderPipelineDescriptor, Stage, StageEntries, StageSource,
};

/// The fixture's library; `None` (skipped visibly) without `llvm-as`.
fn library() -> Option<Library> {
    match llvm_assemble(&fixtures_dir().join("examples/icb_linked_arguments.ll")) {
        Ok(bitcode) => Some(Library::from_air(&bitcode).expect("the fixture decodes")),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected the fixture: {why}"),
    }
}

fn source(library: &Library, name: &str) -> StageSource {
    StageSource::Function(FunctionDescriptor::new(
        library.function(name).expect("function"),
    ))
}

/// Color attachment 0 of `RGBA8Unorm`, as Metal's probes attach.
fn color() -> Option<ColorAttachmentDescriptor> {
    Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm))
}

/// Each use's stage, and its gap where it is unproven.
fn answer(
    uses: Result<Vec<DirectArgumentUse>, crate::error::Error>,
) -> Vec<(Stage, Option<LinkedUseGap>)> {
    uses.expect("an answer")
        .into_iter()
        .map(|used| (used.stage, used.gap))
        .collect()
}

/// The vertex function's texture is used where Metal refuses the state with indirect command
/// buffers, and only there (one row per row Metal answered); the fragment functions use none.
#[test]
fn a_vertex_function_uses_its_texture_where_metal_keeps_it_once_linked() {
    let Some(library) = library() else {
        return;
    };
    let vertex = [(Stage::Vertex, None)].as_slice();
    let none: &[(Stage, Option<LinkedUseGap>)] = &[];
    for (vertex_function, fragment, raster, stages) in [
        ("varying_vertex", Some("varying_fragment"), true, vertex),
        (
            "sampled_varying_vertex",
            Some("varying_fragment"),
            true,
            vertex,
        ),
        ("varying_vertex", Some("declaring_fragment"), true, none),
        ("varying_vertex", Some("inputless_fragment"), true, none),
        ("varying_vertex", None, true, none),
        (
            "sampled_varying_vertex",
            Some("declaring_fragment"),
            true,
            none,
        ),
        ("sampled_varying_vertex", None, true, none),
        ("depth_vertex", Some("declaring_fragment"), true, vertex),
        ("depth_vertex", None, true, vertex),
        ("position_only_vertex", Some("depth_fragment"), true, vertex),
        ("position_only_vertex", None, true, vertex),
        ("store_vertex", Some("declaring_fragment"), true, vertex),
        ("store_vertex", None, true, vertex),
        ("void_store_vertex", None, false, vertex),
        ("discarded_vertex", Some("varying_fragment"), true, none),
        ("plain_vertex", Some("varying_fragment"), true, none),
        // A fragment function a pipeline that does not rasterize never runs.
        ("varying_vertex", Some("varying_fragment"), false, none),
        ("branch_vertex", Some("declaring_fragment"), true, vertex),
        ("branch_vertex", None, true, vertex),
        (
            "guarded_store_vertex",
            Some("declaring_fragment"),
            true,
            vertex,
        ),
        ("guarded_store_vertex", None, true, vertex),
        ("trip_count_vertex", Some("varying_fragment"), true, vertex),
        (
            "trip_count_vertex",
            Some("declaring_fragment"),
            true,
            vertex,
        ),
        ("trip_count_vertex", None, true, vertex),
        ("select_vertex", Some("varying_fragment"), true, vertex),
        ("select_vertex", Some("declaring_fragment"), true, none),
        ("select_vertex", None, true, none),
        ("point_size_vertex", Some("point_fragment"), true, vertex),
        ("point_size_vertex", None, true, vertex),
        (
            "other_varying_vertex",
            Some("first_varying_fragment"),
            true,
            none,
        ),
        ("other_varying_vertex", Some("varying_fragment"), true, none),
        ("dead_branch_vertex", Some("varying_fragment"), true, vertex),
        ("dead_branch_vertex", Some("declaring_fragment"), true, none),
        ("dead_branch_vertex", None, true, none),
        ("merged_vertex", Some("varying_fragment"), true, vertex),
        ("merged_vertex", Some("declaring_fragment"), true, none),
        ("merged_vertex", None, true, none),
        ("switch_vertex", Some("varying_fragment"), true, vertex),
        ("switch_vertex", Some("declaring_fragment"), true, vertex),
        ("switch_vertex", None, true, vertex),
        ("loop_body_vertex", Some("varying_fragment"), true, vertex),
        ("loop_body_vertex", Some("declaring_fragment"), true, none),
        ("loop_body_vertex", None, true, none),
        (
            "storing_loop_vertex",
            Some("varying_fragment"),
            true,
            vertex,
        ),
        (
            "storing_loop_vertex",
            Some("declaring_fragment"),
            true,
            none,
        ),
        ("storing_loop_vertex", None, true, none),
    ] {
        let mut descriptor = RenderPipelineDescriptor::new(source(&library, vertex_function));
        descriptor.fragment = fragment.map(|f| source(&library, f));
        descriptor.rasterization_enabled = raster;
        descriptor.color_attachments[0] = color();
        assert_eq!(
            answer(
                descriptor.stages_using_direct_arguments(&Budget::standard(), &StageEntries::new())
            ),
            stages,
            "{vertex_function} {fragment:?} rasterized {raster}"
        );
    }
}

/// The mesh function's texture is used as Metal answers, and an object function's wherever its
/// reflection lists it.
#[test]
fn a_mesh_function_uses_its_texture_where_metal_keeps_it_once_linked() {
    let Some(library) = library() else {
        return;
    };
    for (object, mesh, fragment, stages) in [
        (
            None,
            "varying_mesh",
            "mesh_varying_fragment",
            vec![Stage::Mesh],
        ),
        (None, "varying_mesh", "mesh_declaring_fragment", vec![]),
        (
            None,
            "depth_mesh",
            "mesh_declaring_fragment",
            vec![Stage::Mesh],
        ),
        (None, "plain_mesh", "mesh_varying_fragment", vec![]),
        (
            Some("texture_object"),
            "payload_mesh",
            "mesh_declaring_fragment",
            vec![Stage::Object],
        ),
        (
            Some("texture_object"),
            "varying_mesh",
            "mesh_varying_fragment",
            vec![Stage::Object, Stage::Mesh],
        ),
        (
            Some("plain_object"),
            "payload_mesh",
            "mesh_varying_fragment",
            vec![],
        ),
        (
            Some("plain_object"),
            "varying_mesh",
            "mesh_varying_fragment",
            vec![Stage::Mesh],
        ),
        (
            Some("plain_object"),
            "varying_mesh",
            "mesh_declaring_fragment",
            vec![],
        ),
    ] {
        let mut descriptor = MeshPipelineDescriptor::new(source(&library, mesh));
        descriptor.object = object.map(|o| source(&library, o));
        descriptor.fragment = Some(source(&library, fragment));
        descriptor.color_attachments[0] = color();
        let stages: Vec<_> = stages.into_iter().map(|stage| (stage, None)).collect();
        assert_eq!(
            answer(
                descriptor.stages_using_direct_arguments(&Budget::standard(), &StageEntries::new())
            ),
            stages,
            "{object:?} {mesh} {fragment}"
        );
    }
}

/// A vertex function returns a value where its return type is not `void`: a struct of varyings,
/// a struct of its position alone, or a bare position.
#[test]
fn a_vertex_function_returns_a_value_where_its_type_is_not_void() {
    let Some(library) = library() else {
        return;
    };
    for (vertex_function, returns) in [
        ("depth_vertex", true),
        ("position_only_vertex", true),
        ("plain_vertex", true),
        ("position_value_vertex", true),
        ("void_store_vertex", false),
        // A post-tessellation vertex function returning its position.
        ("post_tess_vertex", true),
    ] {
        let mut descriptor = RenderPipelineDescriptor::new(source(&library, vertex_function));
        descriptor.rasterization_enabled = false;
        // Twice through one object: the second ask reads the answer the first read, and the two
        // agree, as a creation of the same function after another gets it.
        let entries = StageEntries::new();
        for _ in 0..2 {
            assert_eq!(
                descriptor.vertex_returns_value(&Budget::standard(), &entries),
                Ok(returns),
                "{vertex_function}"
            );
        }
        assert_eq!(
            format!("{entries:?}"),
            "StageEntries { made: 0 }",
            "a plain answer keeps no specialized entry"
        );
    }
}

/// A rasterizing render pipeline whose vertex function returns `void` is refused at its link,
/// with or without a fragment function, as Metal refuses it (`RasterizedVoidVertex`): the compile
/// reads it from the vertex entry it specializes anyway, so a creation asks no question for it.
/// With rasterization disabled it links, and so does every vertex function returning a value. On
/// a device whose vertex stage stores to buffers (lavapipe's), as the `void` function does.
#[test]
fn a_rasterizing_void_vertex_function_is_refused_at_its_link() {
    use crate::device::{Device, DeviceDescription, DevicePolicy};
    use crate::error::{DescriptorError, Error};
    let Some(library) = library() else {
        return;
    };
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles/lavapipe.json");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let description = DeviceDescription::from_khronos_json(&text, None).expect("profile");
    let device = Device::new(description, DevicePolicy::default()).expect("device");
    let refused = Err(Error::InvalidDescriptor(
        DescriptorError::RasterizedVoidVertex,
    ));
    for fragment in [None, Some("inputless_fragment")] {
        let mut descriptor = RenderPipelineDescriptor::new(source(&library, "void_store_vertex"));
        descriptor.fragment = fragment.map(|name| source(&library, name));
        descriptor.color_attachments[0] = color();
        let linked = device.compile_render_pipeline(&descriptor, &Budget::standard());
        assert_eq!(linked.map(|_| ()), refused, "{fragment:?} rasterizing");
        descriptor.rasterization_enabled = false;
        descriptor.fragment = None;
        let unrasterized = device.compile_render_pipeline(&descriptor, &Budget::standard());
        assert!(unrasterized.is_ok(), "{fragment:?}: {unrasterized:?}");
    }
    for vertex_function in [
        "position_only_vertex",
        "plain_vertex",
        "position_value_vertex",
    ] {
        let mut descriptor = RenderPipelineDescriptor::new(source(&library, vertex_function));
        descriptor.fragment = Some(source(&library, "inputless_fragment"));
        descriptor.color_attachments[0] = color();
        let linked = device.compile_render_pipeline(&descriptor, &Budget::standard());
        assert!(linked.is_ok(), "{vertex_function}: {linked:?}");
    }
}

/// A mesh pipeline's fragment function writes depth without a depth attachment where it has a
/// `[[depth(...)]]` output and the pipeline no depth format: with or without a
/// color attachment; not with a depth attachment, nor for a fragment function without one.
#[test]
fn a_mesh_pipelines_fragment_writes_depth_unattached_as_a_render_pipelines() {
    let Some(library) = library() else {
        return;
    };
    for (fragment, depth, colored, writes) in [
        ("depth_writing_fragment", None, true, true),
        ("depth_writing_fragment", None, false, true),
        (
            "depth_writing_fragment",
            Some(PixelFormat::Depth32Float),
            true,
            false,
        ),
        ("mesh_varying_fragment", None, true, false),
    ] {
        let mut meshed = MeshPipelineDescriptor::new(source(&library, "position_mesh"));
        meshed.fragment = Some(source(&library, fragment));
        meshed.depth_format = depth;
        if colored {
            meshed.color_attachments[0] = color();
        }
        assert_eq!(
            meshed.fragment_writes_depth_unattached(&Budget::standard(), &StageEntries::new()),
            Ok(writes),
            "{fragment} {depth:?} {colored}"
        );
    }
}

/// A pipeline configuration of Metal's fragment rows.
#[derive(Clone, Copy, Debug)]
enum Setup {
    /// Color 0 of `RGBA8Unorm`, every channel written.
    Color,
    /// No color attachment.
    NoColor,
    /// No color attachment, a depth attachment.
    NoColorDepth,
    /// Color 0 of no format, color 1 of `RGBA8Unorm`.
    SecondColor,
    /// Color 0 under write mask `mask`.
    Mask(ColorWriteMask),
    /// Color 0 of `format` under write mask `mask`.
    Format(PixelFormat, ColorWriteMask),
    /// Color 0 and rasterization disabled (the vertex function `void_plain_vertex`).
    RasterOff,
    /// No color attachment and rasterization disabled.
    RasterOffNoColor,
    /// Color 0 and a depth attachment.
    Depth,
    /// Color 0 blending a second source.
    SecondSource,
}

/// The render descriptor of `vertex` and `fragment` under `setup`.
fn setup(
    library: &Library,
    vertex: &str,
    fragment: &str,
    setup: Setup,
) -> RenderPipelineDescriptor {
    let raster_off = matches!(setup, Setup::RasterOff | Setup::RasterOffNoColor);
    let vertex = if raster_off {
        "void_plain_vertex"
    } else {
        vertex
    };
    let mut d = RenderPipelineDescriptor::new(source(library, vertex));
    d.fragment = Some(source(library, fragment));
    d.rasterization_enabled = !raster_off;
    d.color_attachments[0] = match setup {
        Setup::NoColor | Setup::NoColorDepth | Setup::SecondColor | Setup::RasterOffNoColor => None,
        _ => color(),
    };
    match setup {
        Setup::SecondColor => d.color_attachments[1] = color(),
        Setup::Mask(mask) => {
            if let Some(a) = &mut d.color_attachments[0] {
                a.write_mask = mask;
            }
        }
        Setup::Format(format, mask) => {
            let mut a = ColorAttachmentDescriptor::new(format);
            a.write_mask = mask;
            d.color_attachments[0] = Some(a);
        }
        Setup::NoColorDepth | Setup::Depth => d.depth_format = Some(PixelFormat::Depth32Float),
        Setup::SecondSource => {
            if let Some(a) = &mut d.color_attachments[0] {
                a.blending_enabled = true;
                a.source_rgb_blend_factor = BlendFactor::Source1Color;
                a.destination_rgb_blend_factor = BlendFactor::OneMinusSource1Color;
            }
        }
        _ => {}
    }
    d
}

/// A fragment function's texture is used once linked where Metal keeps it: a texel that reaches
/// a side effect, a depth, stencil or sample mask output, or a color its attachment writes, with or
/// without rasterization; not one that reaches only a color the pipeline has no attachment for,
/// writes no channel of, or a dual-source second color without blending (one row per row of our
/// probes on an M4 under macOS 27.0.1). A color under a partial write mask
/// is kept channel by channel on Metal (an alpha the mask leaves out is dropped, a red it writes
/// kept), which the analysis does not track: unproven, `UnmeasuredColorWrite`, whatever Metal
/// answers.
#[test]
fn a_fragment_function_uses_its_texture_where_metal_keeps_it_once_linked() {
    let Some(library) = library() else {
        return;
    };
    let rgb = ColorWriteMask::RED
        .union(ColorWriteMask::GREEN)
        .union(ColorWriteMask::BLUE);
    let gba = ColorWriteMask::GREEN
        .union(ColorWriteMask::BLUE)
        .union(ColorWriteMask::ALPHA);
    let proven = Some(None);
    let partial = Some(Some(LinkedUseGap::UnmeasuredColorWrite));
    for (fragment, configuration, used) in [
        ("texel_fragment", Setup::Color, proven),
        ("texel_fragment", Setup::NoColor, None),
        ("texel_fragment", Setup::NoColorDepth, None),
        ("texel_fragment", Setup::SecondColor, None),
        ("texel_fragment", Setup::Mask(ColorWriteMask::NONE), None),
        ("texel_fragment", Setup::RasterOff, proven),
        ("texel_fragment", Setup::RasterOffNoColor, None),
        ("discarding_fragment", Setup::Color, proven),
        ("discarding_fragment", Setup::NoColor, proven),
        (
            "discarding_fragment",
            Setup::Mask(ColorWriteMask::NONE),
            proven,
        ),
        ("discarding_fragment", Setup::RasterOffNoColor, proven),
        ("storing_fragment", Setup::Color, proven),
        ("storing_fragment", Setup::NoColor, proven),
        (
            "storing_fragment",
            Setup::Mask(ColorWriteMask::NONE),
            proven,
        ),
        ("storing_fragment", Setup::RasterOffNoColor, proven),
        ("second_color_fragment", Setup::Color, None),
        ("second_color_fragment", Setup::NoColor, None),
        ("second_color_fragment", Setup::SecondColor, proven),
        ("second_color_fragment", Setup::RasterOff, None),
        // Metal refuses, refuses, makes, refuses, makes, refuses.
        ("texel_fragment", Setup::Mask(rgb), partial),
        ("alpha_fragment", Setup::Color, proven),
        ("alpha_fragment", Setup::Mask(rgb), partial),
        (
            "alpha_fragment",
            Setup::Mask(ColorWriteMask::ALPHA),
            partial,
        ),
        ("red_fragment", Setup::Mask(gba), partial),
        ("red_fragment", Setup::Mask(ColorWriteMask::RED), partial),
        ("depth_writing_fragment", Setup::Depth, proven),
        ("depth_writing_fragment", Setup::NoColorDepth, proven),
        ("mask_fragment", Setup::Color, proven),
        ("mask_fragment", Setup::NoColor, proven),
        ("stencil_fragment", Setup::Color, proven),
        ("dual_source_fragment", Setup::Color, None),
        ("dual_source_fragment", Setup::SecondSource, proven),
        ("dual_source_fragment", Setup::NoColor, None),
        // The consumer's row on an M3 Max under macOS 26.6.2: a mask holding none of the
        // format's channels writes nothing, made. One holding them writes a color in part.
        (
            "texel_fragment",
            Setup::Format(PixelFormat::R8Unorm, ColorWriteMask::ALPHA),
            None,
        ),
        (
            "alpha_fragment",
            Setup::Format(PixelFormat::R8Unorm, ColorWriteMask::ALPHA),
            None,
        ),
        (
            "texel_fragment",
            Setup::Format(PixelFormat::R8Unorm, ColorWriteMask::ALL),
            partial,
        ),
        (
            "texel_fragment",
            Setup::Format(PixelFormat::RG8Unorm, ColorWriteMask::BLUE),
            None,
        ),
    ] {
        let descriptor = setup(&library, "plain_vertex", fragment, configuration);
        let expected: Vec<_> = used.into_iter().map(|gap| (Stage::Fragment, gap)).collect();
        assert_eq!(
            answer(
                descriptor.stages_using_direct_arguments(&Budget::standard(), &StageEntries::new())
            ),
            expected,
            "{fragment} {configuration:?}"
        );
    }
}

/// A vertex function's texel that reaches a varying the fragment function reads only into a color
/// the pipeline has no attachment for, or writes no channel of, is dropped (Metal's two measured
/// rules composed); one the fragment function reads into a color under a partial write mask is
/// unproven (`UnmeasuredColorWrite`).
#[test]
fn a_vertex_function_s_texture_reaching_a_color_written_in_part_is_unproven() {
    let Some(library) = library() else {
        return;
    };
    let rgb = ColorWriteMask::RED
        .union(ColorWriteMask::GREEN)
        .union(ColorWriteMask::BLUE);
    for (configuration, expected) in [
        (Setup::Color, vec![(Stage::Vertex, None)]),
        (Setup::NoColor, vec![]),
        (Setup::Mask(ColorWriteMask::NONE), vec![]),
        (
            Setup::Mask(rgb),
            vec![(Stage::Vertex, Some(LinkedUseGap::UnmeasuredColorWrite))],
        ),
    ] {
        let descriptor = setup(
            &library,
            "varying_vertex",
            "varying_fragment",
            configuration,
        );
        assert_eq!(
            answer(
                descriptor.stages_using_direct_arguments(&Budget::standard(), &StageEntries::new())
            ),
            expected,
            "{configuration:?}"
        );
    }
}

/// The forward analysis is the backward one's dual, the check a compile makes before it reads a
/// stage input the linking drops as zero (`dropped_fragment_inputs`) resting on it: in every
/// function of the fixture, with nothing and everything of its returned value kept and with the
/// rules kept beyond Metal off and on, a parameter's value reaches what the linking keeps (its
/// effects, its returned value, a loop, a branch that decides either) exactly where an
/// instruction the linking keeps reads the parameter.
#[test]
fn values_followed_forward_reach_what_is_kept_exactly_where_the_linking_keeps_their_reads() {
    use super::linked::Rules;
    use super::live::{forward_reach, kept_parameters, Reach, Returned, Roots};
    use std::collections::BTreeSet;
    let Some(library) = library() else {
        return;
    };
    let names: Vec<String> = library.functions().map(|f| f.name.to_string()).collect();
    let mut checked = 0usize;
    for name in &names {
        let descriptor = FunctionDescriptor::new(library.function(name).expect("function"));
        let mut m = crate::budget::Meter::new(&Budget::standard());
        crate::pipeline::compile::stage_entry(
            &descriptor,
            0,
            &StageEntries::new(),
            &mut m,
            |entry, m| {
                for returned in [Returned::Nothing, Returned::All] {
                    for rules in [Rules::NONE, Rules::ALL] {
                        let roots = Roots {
                            returned: returned.clone(),
                            mesh: None,
                            rules,
                        };
                        let kept = kept_parameters(entry, &roots, m)?;
                        for p in &entry.ep.params {
                            let reach =
                                forward_reach(entry, &BTreeSet::from([p.param]), &roots, m)?;
                            assert_eq!(
                                reach == Reach::Kept,
                                kept[p.param as usize],
                                "{name}: parameter {} ({:?}), {returned:?}, {rules:?}",
                                p.param,
                                p.role
                            );
                            checked += 1;
                        }
                    }
                }
                Ok(())
            },
        )
        .unwrap_or_else(|e| panic!("{name}: {e:?}"));
    }
    assert!(checked > 100, "only {checked} parameters checked");
}

/// The `Location`s of the `Input` variables of SPIR-V module `words` (built-ins carry none).
fn input_locations(words: &[u32]) -> Vec<u32> {
    const OP_VARIABLE: u32 = 59;
    const OP_DECORATE: u32 = 71;
    const LOCATION: u32 = 30;
    const INPUT: u32 = 1;
    let mut inputs = Vec::new();
    let mut decorated = Vec::new();
    let mut at = 5;
    while let Some(&first) = words.get(at) {
        let length = (first >> 16) as usize;
        assert!(length > 0, "an instruction of zero words at {at}");
        let word = |k: usize| words.get(at + k).copied().unwrap_or(u32::MAX);
        match first & 0xffff {
            OP_VARIABLE if word(3) == INPUT => inputs.push(word(2)),
            OP_DECORATE if word(2) == LOCATION => decorated.push((word(1), word(3))),
            _ => {}
        }
        at += length;
    }
    let mut out: Vec<u32> = decorated
        .into_iter()
        .filter(|(id, _)| inputs.contains(id))
        .map(|(_, location)| location)
        .collect();
    out.sort_unstable();
    out
}

/// A fragment function's stage input that the vertex function writes no output of, and that the
/// linking drops, is absent from the compiled fragment module: it has no input variable and is
/// read as zero. Metal makes such a state (the probe's `vc`+`fd` without a color
/// attachment, on an M4 under macOS 27.0.1). This holds where the input reaches only a color the
/// pipeline drops, by its value or by a branch on it. An input the linking keeps, by its value
/// or by a branch whose arms make a kept color, keeps its variable, and the link refuses the
/// pair; the runtime refuses it first, with Metal's text (`unwritten_fragment_inputs`). An input
/// written to a color under a partial write mask is kept with the unmeasured rule on (unproven).
#[test]
fn a_dropped_unwritten_fragment_input_is_absent_from_the_compiled_fragment() {
    use crate::device::{Device, DeviceDescription, DevicePolicy};
    let Some(library) = library() else {
        return;
    };
    let rgb = ColorWriteMask::RED
        .union(ColorWriteMask::GREEN)
        .union(ColorWriteMask::BLUE);
    let devices: Vec<(&str, Device)> = ["moltenvk-m3", "vk12-floor"]
        .into_iter()
        .map(|stem| {
            let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../metal2vulkan-validation/profiles")
                .join(format!("{stem}.json"));
            let text = std::fs::read_to_string(&path)
                .unwrap_or_else(|e| panic!("{}: {e}", path.display()));
            let description = DeviceDescription::from_khronos_json(&text, None).expect(stem);
            (
                stem,
                Device::new(description, DevicePolicy::default()).expect(stem),
            )
        })
        .collect();
    let color = crate::air::abi::UserKey(b"generated(5colorDv4_f)".to_vec().into());
    for (fragment, configuration, absent) in [
        ("varying_fragment", Setup::Color, false),
        ("varying_fragment", Setup::NoColor, true),
        ("varying_fragment", Setup::Mask(ColorWriteMask::NONE), true),
        ("varying_fragment", Setup::Mask(rgb), false),
        (
            "varying_fragment",
            Setup::Format(PixelFormat::R8Unorm, ColorWriteMask::ALPHA),
            true,
        ),
        ("branching_fragment", Setup::Color, false),
        ("branching_fragment", Setup::NoColor, true),
        ("varying_first_color_fragment", Setup::Color, false),
        ("varying_first_color_fragment", Setup::SecondColor, true),
    ] {
        let row = format!("{fragment} {configuration:?}");
        let descriptor = setup(&library, "position_value_vertex", fragment, configuration);
        let Some(StageSource::Function(f)) = &descriptor.fragment else {
            panic!("{row}: a fragment function");
        };
        let mut m = crate::budget::Meter::new(&Budget::standard());
        let dropped = crate::pipeline::compile::stage_entry(
            f,
            0,
            &StageEntries::new(),
            &mut m,
            |entry, m| super::dropped_fragment_inputs(entry, &descriptor.color_attachments, m),
        )
        .unwrap_or_else(|e| panic!("{row}: {e:?}"));
        assert_eq!(dropped.contains(&color), absent, "{row}: {dropped:?}");
        for (stem, device) in &devices {
            match device.compile_render_pipeline(&descriptor, &Budget::standard()) {
                Ok(pipeline) => {
                    assert!(absent, "{stem} {row}: links though the input is kept");
                    let fragment_module = pipeline.stages.last().expect("a fragment stage");
                    assert_eq!(
                        input_locations(&fragment_module.spirv),
                        Vec::<u32>::new(),
                        "{stem} {row}: the absent input has a variable"
                    );
                }
                Err(e) => assert!(!absent, "{stem} {row}: refused: {e:?}"),
            }
        }
    }
}

/// A post-tessellation vertex function's and a mesh function's unwritten fragment input are
/// answered as a vertex function's. Metal refuses with color 0 attached and makes without (on an M4
/// under macOS 27.0.1). Kept, the descriptors name the input and the link
/// refuses the pair; dropped, the compile reads it as zero (no input variable) and links.
#[test]
fn tessellated_and_mesh_pipelines_answer_an_unwritten_fragment_input_as_a_vertex_function_s() {
    use crate::device::{Device, DeviceDescription, DevicePolicy};
    let Some(library) = library() else {
        return;
    };
    // A profile with tessellation and mesh shading.
    let stem = "nvidia-rtx5080-laptop";
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles")
        .join(format!("{stem}.json"));
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let description = DeviceDescription::from_khronos_json(&text, None).expect(stem);
    let device = Device::new(description, DevicePolicy::default()).expect(stem);
    let budget = Budget::standard();
    for attached in [true, false] {
        let named = attached.then(|| (vec!["color".to_string()], None));
        let mut tessellated = RenderPipelineDescriptor::new(source(&library, "post_tess_vertex"));
        tessellated.fragment = Some(source(&library, "varying_fragment"));
        let mut meshed = MeshPipelineDescriptor::new(source(&library, "position_mesh"));
        meshed.fragment = Some(source(&library, "mesh_varying_fragment"));
        if attached {
            tessellated.color_attachments[0] = color();
            meshed.color_attachments[0] = color();
        }
        let answer = |u: Option<super::UnwrittenFragmentInputs>| u.map(|u| (u.names, u.gap));
        assert_eq!(
            answer(
                tessellated
                    .unwritten_fragment_inputs(&budget, &StageEntries::new())
                    .expect("an answer")
            ),
            named,
            "tessellated, color 0 attached: {attached}"
        );
        assert_eq!(
            answer(
                meshed
                    .unwritten_fragment_inputs(&budget, &StageEntries::new())
                    .expect("an answer")
            ),
            named,
            "mesh, color 0 attached: {attached}"
        );
        let modules = [
            device
                .compile_render_pipeline(&tessellated, &budget)
                .map(|p| p.stages.last().map(|s| s.spirv.clone())),
            device
                .compile_mesh_pipeline(&meshed, &budget)
                .map(|p| p.stages.last().map(|s| s.spirv.clone())),
        ];
        for (pipeline, compiled) in ["tessellated", "mesh"].into_iter().zip(modules) {
            match (attached, compiled) {
                (true, compiled) => assert!(compiled.is_err(), "{pipeline}: links though kept"),
                (false, Ok(Some(spirv))) => assert_eq!(
                    input_locations(&spirv),
                    Vec::<u32>::new(),
                    "{pipeline}: the absent input has a variable"
                ),
                (false, other) => panic!("{pipeline}: {other:?}"),
            }
        }
    }
}

/// A creation's checks and key read each stage's one specialized entry: given one
/// `StageEntries`, a render pipeline's unwritten-input and depth checks and its variant key
/// specialize the vertex and the fragment function once each (two entries made, where each of
/// the three questions alone specializes its stages again), and answer as each answers alone.
#[test]
fn one_creation_specializes_each_stage_once() {
    use crate::cache::VariantKey;
    use crate::device::{Device, DeviceDescription, DevicePolicy};
    let Some(library) = library() else {
        return;
    };
    let stem = "lavapipe";
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles")
        .join(format!("{stem}.json"));
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let description = DeviceDescription::from_khronos_json(&text, None).expect(stem);
    let device = Device::new(description, DevicePolicy::default()).expect(stem);
    let budget = Budget::standard();
    let mut descriptor = RenderPipelineDescriptor::new(source(&library, "varying_vertex"));
    descriptor.fragment = Some(source(&library, "depth_writing_fragment"));
    descriptor.color_attachments[0] = color();
    let shared = StageEntries::new();
    let depth = descriptor.fragment_writes_depth_unattached(&budget, &shared);
    let unwritten = descriptor.unwritten_fragment_inputs(&budget, &shared);
    let keys = device.render_variant_key(&descriptor, &VariantKey::new(), &budget, &shared);
    assert_eq!(format!("{shared:?}"), "StageEntries { made: 2 }");
    assert_eq!(
        depth,
        descriptor.fragment_writes_depth_unattached(&budget, &StageEntries::new())
    );
    assert_eq!(
        unwritten,
        descriptor.unwritten_fragment_inputs(&budget, &StageEntries::new())
    );
    assert_eq!(keys, device.render_pipeline_key(&descriptor, &budget));
    assert_eq!(depth, Ok(true), "the fragment function writes depth");
    keys.expect("the keys");
}

/// Whether a stage uses a direct texture or sampler is decided from its arguments' roles and use
/// alone: the owned fixture `examples/icb_direct_arguments.ll` with the kernels' output buffer and
/// the fragment functions' texture of types the reflection refuses (a `quaternion` buffer, a
/// `texture2d<quaternion, sample>`; Metal's answer for neither is measured) answers as the
/// fixture does, while the reflection of each refuses the type.
#[test]
fn direct_argument_use_is_decided_where_the_reflection_refuses_a_type() {
    use crate::error::{DiagName, Error, Feature};
    use crate::pipeline::ComputePipelineDescriptor;
    let path = fixtures_dir().join("examples/icb_direct_arguments.ll");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path:?}: {e}"));
    const BUFFER: &str = "quaternion";
    const TEXTURE: &str = "texture2d<quaternion, sample>";
    let mut unmeasured = text.clone();
    for (from, to) in [
        (
            r#"!"float4", !"air.arg_name", !"out"}"#,
            format!(r#"!"{BUFFER}", !"air.arg_name", !"out"}}"#),
        ),
        (
            r#"!"texture2d<float, sample>", !"air.arg_name", !"source"}"#,
            format!(r#"!"{TEXTURE}", !"air.arg_name", !"source"}}"#),
        ),
    ] {
        assert_eq!(unmeasured.matches(from).count(), 1, "{from}");
        unmeasured = unmeasured.replace(from, &to);
    }
    let libraries = [text, unmeasured].map(|text| match metal2vulkan_devtools::llvm_assemble_text(
        &text,
    ) {
        Ok(bitcode) => Some(Library::from_air(&bitcode).expect("the fixture decodes")),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as: {why}"),
    });
    let [Some(measured), Some(unmeasured)] = libraries else {
        return;
    };
    let budget = Budget::standard();
    let refused_type = |answer: Result<(), Error>, symbol: &str, at: &str| match answer {
        Err(Error::Unsupported(unsupported)) => {
            assert_eq!(unsupported.feature, Feature::Reflection, "{at}");
            assert_eq!(unsupported.symbol, Some(DiagName::new(symbol)), "{at}");
        }
        other => panic!("{at}: {other:?}"),
    };
    for (kernel, uses) in [("buffer_kernel", false), ("texture_kernel", true)] {
        let descriptor =
            |library: &Library| ComputePipelineDescriptor::new(source(library, kernel));
        refused_type(
            descriptor(&unmeasured).reflection(&budget).map(drop),
            BUFFER,
            kernel,
        );
        for library in [&measured, &unmeasured] {
            assert_eq!(
                descriptor(library).uses_direct_arguments(&budget),
                Ok(uses),
                "{kernel}"
            );
        }
    }
    for (fragment, uses) in [
        ("plain_fragment", vec![]),
        ("texture_fragment", vec![(Stage::Fragment, None)]),
        ("sampler_fragment", vec![(Stage::Fragment, None)]),
        ("unused_arguments_fragment", vec![]),
    ] {
        let render = |library: &Library| {
            let mut descriptor = RenderPipelineDescriptor::new(source(library, "plain_vertex"));
            descriptor.fragment = Some(source(library, fragment));
            descriptor.color_attachments[0] = color();
            descriptor
        };
        let mesh = |library: &Library| {
            let mut descriptor = MeshPipelineDescriptor::new(source(library, "plain_mesh"));
            descriptor.fragment = Some(source(library, fragment));
            descriptor.color_attachments[0] = color();
            descriptor
        };
        if matches!(fragment, "texture_fragment" | "sampler_fragment") {
            refused_type(
                render(&unmeasured).reflection(&budget).map(drop),
                TEXTURE,
                fragment,
            );
        }
        for library in [&measured, &unmeasured] {
            let entries = StageEntries::new();
            assert_eq!(
                answer(render(library).stages_using_direct_arguments(&budget, &entries)),
                uses,
                "{fragment}"
            );
            assert_eq!(
                answer(mesh(library).stages_using_direct_arguments(&budget, &entries)),
                uses,
                "{fragment} (mesh)"
            );
        }
    }
}
