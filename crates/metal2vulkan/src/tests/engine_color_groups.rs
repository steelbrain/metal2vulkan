//! Render pipelines whose color attachments a device draws in groups
//! (`pipeline::compile::color_group`): the owned fixtures `engine/color_groups`,
//! `engine/color_groups_fetch`, `engine/dfk_dead_group` and `engine/dfk_dead_rog` on `lavapipe`
//! with four color attachments and four fragment output locations (Vulkan's floor, the device of
//! the groups' lane), on `moltenvk-m3` likewise, on `vk12-floor`, and on `lavapipe` as it is
//! (eight), where nothing groups. `llvm-as` assembles the fixtures; a missing tool skips visibly.

use std::collections::BTreeSet;

use crate::budget::Budget;
use crate::cache::VariantKey;
use crate::device::requirements::VkLimit;
use crate::device::{Device, DevicePolicy};
use crate::error::{Capability, Error, Feature, InternalKind, Needs, RuntimeInput};
use crate::library::{FunctionConstantValue, FunctionConstantValues, Library};
use crate::metal::{DataType, PixelFormat};
use crate::pipeline::descriptor::{
    ColorAttachmentDescriptor, FunctionDescriptor, RenderPipelineDescriptor, StageSource,
};
use crate::pipeline::testing::{description, engine_dir, llvm_assemble, AssembleError};
use crate::pipeline::{ColorGroup, PrecisionNote, RenderPipeline, StageModule};
use crate::spirv::testing::check_words_valid;
use crate::spv::BuiltIn;
use crate::vk;

fn fixture(name: &str) -> Option<Vec<u8>> {
    match llvm_assemble(&engine_dir().join(format!("{name}.ll"))) {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected {name}.ll: {why}"),
    }
}

/// `lavapipe` drawing to four color attachments at once.
fn four() -> Device {
    narrowed("lavapipe")
}

/// The committed profile `stem` drawing to four color attachments at once.
fn narrowed(stem: &str) -> Device {
    let mut desc = description(stem).expect("profile");
    desc.limits.max_color_attachments = 4;
    desc.limits.max_fragment_output_attachments = 4;
    Device::new(desc, DevicePolicy::default()).expect("device")
}

fn device(stem: &str) -> Device {
    Device::new(description(stem).expect("profile"), DevicePolicy::default()).expect("device")
}

/// A different format at each color index, so an attachment's place in a group shows.
const FORMATS: [PixelFormat; 8] = [
    PixelFormat::RGBA8Unorm,
    PixelFormat::RGBA16Float,
    PixelFormat::RGBA32Float,
    PixelFormat::BGRA8Unorm,
    PixelFormat::RG16Float,
    PixelFormat::RGBA8UnormSRGB,
    PixelFormat::R32Float,
    PixelFormat::RG32Float,
];

/// The pipeline of `cg_cover` and fragment function `fragment` of module `fs`, with attachments
/// at `colors`.
fn pipeline(vs: &[u8], fs: &[u8], fragment: &str, colors: &[u32]) -> RenderPipelineDescriptor {
    let function = |bc: &[u8], name: &str| {
        FunctionDescriptor::new(
            Library::from_air(bc)
                .expect("library")
                .function(name)
                .expect("function"),
        )
    };
    let mut d = RenderPipelineDescriptor::new(StageSource::Function(function(vs, "cg_cover")));
    d.fragment = Some(StageSource::Function(function(fs, fragment)));
    for &c in colors {
        d.color_attachments[c as usize] = Some(ColorAttachmentDescriptor::new(FORMATS[c as usize]));
    }
    d
}

fn compile(
    dev: &Device,
    d: &RenderPipelineDescriptor,
    group: u32,
) -> Result<RenderPipeline, Error> {
    let mut v = VariantKey::new();
    v.color_group = group;
    dev.compile_render_variant(d, &v, &Budget::standard())
}

fn stage(p: &RenderPipeline, s: vk::ShaderStage) -> &StageModule {
    p.stages.iter().find(|m| m.stage == s).expect("stage")
}

/// Each instruction of `words` past the header: its opcode and operands.
fn instructions(words: &[u32]) -> Vec<(u32, &[u32])> {
    let mut out = Vec::new();
    let mut at = 5;
    while at < words.len() {
        let len = (words[at] >> 16) as usize;
        out.push((words[at] & 0xffff, &words[at + 1..at + len]));
        at += len.max(1);
    }
    out
}

const OP_VARIABLE: u32 = 59;
const OP_DECORATE: u32 = 71;
const STORAGE_OUTPUT: u32 = 3;
const DECORATION_BUILTIN: u32 = 11;
const DECORATION_INVARIANT: u32 = 18;
const DECORATION_LOCATION: u32 = 30;

/// The locations of `words`' output variables.
fn output_locations(words: &[u32]) -> BTreeSet<u32> {
    let insts = instructions(words);
    let outputs: BTreeSet<u32> = insts
        .iter()
        .filter(|(op, o)| *op == OP_VARIABLE && o.get(2) == Some(&STORAGE_OUTPUT))
        .map(|(_, o)| o[1])
        .collect();
    insts
        .iter()
        .filter(|(op, o)| *op == OP_DECORATE && o.get(1) == Some(&DECORATION_LOCATION))
        .filter(|(_, o)| outputs.contains(&o[0]))
        .map(|(_, o)| o[2])
        .collect()
}

/// The builtins `words` decorates `Invariant`.
fn invariant_builtins(words: &[u32]) -> BTreeSet<u32> {
    let insts = instructions(words);
    let invariant: BTreeSet<u32> = insts
        .iter()
        .filter(|(op, o)| *op == OP_DECORATE && o.get(1) == Some(&DECORATION_INVARIANT))
        .map(|(_, o)| o[0])
        .collect();
    insts
        .iter()
        .filter(|(op, o)| *op == OP_DECORATE && o.get(1) == Some(&DECORATION_BUILTIN))
        .filter(|(_, o)| invariant.contains(&o[0]))
        .map(|(_, o)| o[2])
        .collect()
}

/// Every stage of `p` passes L3 on `dev`.
fn validate(p: &RenderPipeline, dev: &Device) {
    for m in &p.stages {
        check_words_valid(&m.spirv, &p.requirements, dev.profile())
            .unwrap_or_else(|e| panic!("{:?}: L3: {e}", m.stage));
    }
}

/// The formats of `p`'s color attachments, in order.
fn formats(p: &RenderPipeline) -> Vec<Option<vk::Format>> {
    p.graphics
        .color_attachments
        .iter()
        .map(|a| a.as_ref().map(|a| a.format))
        .collect()
}

/// The formats `p`'s attachments at Metal's indices `colors` take on `lavapipe`, which draws
/// every one at its own index: what a group of them holds, in order.
fn reference(d: &RenderPipelineDescriptor, colors: &[u32]) -> Vec<Option<vk::Format>> {
    let whole = formats(&compile(&device("lavapipe"), d, 0).expect("lavapipe"));
    colors.iter().map(|&c| whole[c as usize]).collect()
}

/// The limit refusal of a pipeline whose attachments reach index `required - 1` on a device of four.
fn is_limit(e: &Error, required: u64) -> bool {
    matches!(e, Error::Unsupported(u) if u.feature == Feature::InterfaceComponents
    && u.needs == Needs::AnyOf(vec![vec![Capability::Limit {
        limit: VkLimit::MaxColorAttachments,
        required,
        available: 4,
    }]]))
}

/// Five colors on a device of four draw in two groups: the first four attachments, then the
/// fifth, each variant's fragment stage writing its group's alone at locations `0..` in a state of
/// its group's attachments alone, every variant's vertex stage the same module, and each variant
/// noting the groups. A variant past the last group is no plan's. On `vk12-floor` too; on
/// `lavapipe` as it is, one pipeline of five, no note.
#[test]
fn five_colors_on_a_device_of_four_draw_in_two_groups() {
    let Some(bc) = fixture("color_groups") else {
        return;
    };
    let d = pipeline(&bc, &bc, "cg_five", &[0, 1, 2, 3, 4]);
    for (stem, dev) in [("four", four()), ("vk12-floor", device("vk12-floor"))] {
        let groups: Vec<RenderPipeline> = (0..2)
            .map(|g| compile(&dev, &d, g).unwrap_or_else(|e| panic!("{stem} group {g}: {e}")))
            .collect();
        for (g, (p, colors)) in groups.iter().zip([vec![0, 1, 2, 3], vec![4]]).enumerate() {
            let group = p.graphics.color_group.as_ref().expect("a group");
            assert_eq!(group.colors(), colors, "{stem} group {g}");
            assert_eq!(group.groups(), 2, "{stem} group {g}");
            if stem == "four" {
                assert_eq!(formats(p), reference(&d, &colors), "{stem} group {g}");
            }
            assert_eq!(formats(p).len(), colors.len(), "{stem} group {g}");
            let fragment = stage(p, vk::ShaderStage::FRAGMENT);
            assert_eq!(
                output_locations(&fragment.spirv),
                (0..colors.len() as u32).collect(),
                "{stem} group {g}"
            );
            assert!(
                fragment
                    .precision_notes
                    .contains(&PrecisionNote::ColorAttachmentGroups),
                "{stem} group {g}: {:?}",
                fragment.precision_notes
            );
            validate(p, &dev);
        }
        assert_eq!(
            stage(&groups[0], vk::ShaderStage::VERTEX).spirv,
            stage(&groups[1], vk::ShaderStage::VERTEX).spirv,
            "{stem}: the vertex stage differs between groups"
        );
        assert_ne!(groups[0].key, groups[1].key, "{stem}");
        match compile(&dev, &d, 2) {
            Err(Error::Internal(e)) => assert_eq!(e.kind, InternalKind::Invariant, "{stem}"),
            other => panic!("{stem}: {other:?}"),
        }
    }
    let dev = device("lavapipe");
    let p = compile(&dev, &d, 0).expect("lavapipe");
    assert_eq!(p.graphics.color_group, None);
    assert_eq!(formats(&p).len(), 5);
    let fragment = stage(&p, vk::ShaderStage::FRAGMENT);
    assert_eq!(output_locations(&fragment.spirv), (0..5).collect());
    assert!(!fragment
        .precision_notes
        .contains(&PrecisionNote::ColorAttachmentGroups));
    assert!(
        compile(&dev, &d, 1).is_err(),
        "lavapipe draws no second group"
    );
}

/// Seven colors take two groups, the second of three; colors 0 and 5 take one group, renumbered
/// below four with no note, as the colors present are numbered whatever their indices; a pipeline
/// whose colors are all below four takes none.
#[test]
fn groups_number_the_attachments_present_densely() {
    let Some(bc) = fixture("color_groups") else {
        return;
    };
    let dev = four();
    let seven = pipeline(&bc, &bc, "cg_seven", &[0, 1, 2, 3, 4, 5, 6]);
    let p = compile(&dev, &seven, 1).expect("seven, group 1");
    let group = p.graphics.color_group.as_ref().expect("a group");
    assert_eq!((group.colors(), group.groups()), (&[4, 5, 6][..], 2));
    assert_eq!(formats(&p), reference(&seven, &[4, 5, 6]));
    assert_eq!(
        output_locations(&stage(&p, vk::ShaderStage::FRAGMENT).spirv),
        BTreeSet::from([0, 1, 2])
    );
    validate(&p, &dev);

    let sparse = pipeline(&bc, &bc, "cg_sparse", &[0, 5]);
    let p = compile(&dev, &sparse, 0).expect("sparse");
    assert_eq!(
        p.graphics.color_group,
        Some(ColorGroup {
            present: vec![0, 5],
            width: 4,
            group: 0,
        })
    );
    assert_eq!(formats(&p), reference(&sparse, &[0, 5]));
    let fragment = stage(&p, vk::ShaderStage::FRAGMENT);
    assert_eq!(output_locations(&fragment.spirv), BTreeSet::from([0, 1]));
    assert!(!fragment
        .precision_notes
        .contains(&PrecisionNote::ColorAttachmentGroups));
    assert!(invariant_builtins(&fragment.spirv).is_empty());
    validate(&p, &dev);
    assert!(matches!(
        compile(&dev, &sparse, 1),
        Err(Error::Internal(e)) if e.kind == InternalKind::Invariant
    ));

    let low = pipeline(&bc, &bc, "cg_sparse", &[0, 3]);
    let p = compile(&dev, &low, 0).expect("colors 0 and 3");
    assert_eq!(p.graphics.color_group, None);
}

/// A variant of a pipeline of more than one group decorates its fragment's `FragDepth` and
/// `SampleMask` `Invariant`, beside the vertex stage's `Position`; a pipeline of one group, or
/// none, decorates the position alone.
#[test]
fn a_grouped_fragment_s_per_fragment_outputs_are_invariant() {
    let Some(bc) = fixture("color_groups") else {
        return;
    };
    let d = pipeline(&bc, &bc, "cg_depth", &[0, 1, 2, 3, 4]);
    let per_fragment = BTreeSet::from([BuiltIn::FragDepth as u32, BuiltIn::SampleMask as u32]);
    let dev = four();
    for g in 0..2 {
        let p = compile(&dev, &d, g).unwrap_or_else(|e| panic!("group {g}: {e}"));
        let fragment = stage(&p, vk::ShaderStage::FRAGMENT);
        assert_eq!(
            invariant_builtins(&fragment.spirv),
            per_fragment,
            "group {g}"
        );
        assert_eq!(
            invariant_builtins(&stage(&p, vk::ShaderStage::VERTEX).spirv),
            BTreeSet::from([BuiltIn::Position as u32]),
            "group {g}"
        );
        validate(&p, &dev);
    }
    let p = compile(&device("lavapipe"), &d, 0).expect("lavapipe");
    assert!(invariant_builtins(&stage(&p, vk::ShaderStage::FRAGMENT).spirv).is_empty());
}

/// What each group's draw would run again, or read from another group, is the pipeline's limit
/// refusal: a fragment function that writes memory, alpha to coverage, and a read of a color of
/// another group. A read of a color in a pipeline of one group is that group's own, renumbered with
/// it. A fragment compiled for another group than the variant's is `NeedsInput(ColorWrites)`.
#[test]
fn what_each_group_would_run_again_is_the_limit_refusal() {
    let (Some(bc), Some(fetch)) = (fixture("color_groups"), fixture("color_groups_fetch")) else {
        return;
    };
    let dev = four();
    let counts = pipeline(&bc, &bc, "cg_counts", &[0, 1, 2, 3, 4]);
    let e = compile(&dev, &counts, 0).expect_err("a fragment that writes memory");
    assert!(is_limit(&e, 5), "{e:?}");

    let mut covered = pipeline(&bc, &bc, "cg_five", &[0, 1, 2, 3, 4]);
    covered.alpha_to_coverage = true;
    let e = compile(&dev, &covered, 1).expect_err("alpha to coverage");
    assert!(is_limit(&e, 5), "{e:?}");

    let fetched = pipeline(&bc, &fetch, "cg_fetch", &[0, 1, 2, 3, 4]);
    let e = compile(&dev, &fetched, 0).expect_err("a read of color 4 in group 0");
    assert!(is_limit(&e, 5), "{e:?}");

    let sparse = pipeline(&bc, &fetch, "cg_fetch_sparse", &[0, 5]);
    let p = compile(&dev, &sparse, 0).expect("a read of color 5 in its own group");
    assert_eq!(formats(&p), reference(&sparse, &[0, 5]));
    validate(&p, &dev);

    // Group 0's fragment, compiled alone, in group 1's variant.
    let d = pipeline(&bc, &bc, "cg_five", &[0, 1, 2, 3, 4]);
    let keys = dev
        .render_variant_key(
            &d,
            &VariantKey::new(),
            &Budget::standard(),
            &crate::pipeline::compile::StageEntries::new(),
        )
        .expect("keys");
    let Some(StageSource::Function(f)) = &d.fragment else {
        unreachable!()
    };
    let inputs = keys.fragment_inputs.expect("the fragment's inputs");
    assert!(inputs.color_group.is_some());
    let first = dev
        .compile_function(f, &inputs, &Budget::standard())
        .expect("group 0's fragment");
    let mut mixed = d.clone();
    mixed.fragment = Some(StageSource::Compiled(first));
    compile(&dev, &mixed, 0).expect("group 0's fragment in group 0");
    assert!(matches!(
        compile(&dev, &mixed, 1),
        Err(Error::NeedsInput(RuntimeInput::ColorWrites))
    ));
}

/// [`pipeline`] of `cg_cover` and `fragment` of `fs` at `colors`, the fragment function
/// specialized with its `bool` function constant 0 `on`.
fn specialized(
    vs: &[u8],
    fs: &[u8],
    fragment: &str,
    colors: &[u32],
    on: bool,
) -> RenderPipelineDescriptor {
    let mut d = pipeline(vs, fs, fragment, colors);
    if let Some(StageSource::Function(f)) = d.fragment.as_mut() {
        f.constants = FunctionConstantValues::defaults().with_index(
            0,
            FunctionConstantValue::new(DataType::Bool, [u8::from(on)]),
        );
        f.specialized = true;
    }
    d
}

/// A read of a color of another group that the function's constants make dead reads nothing
/// (`dfk_dead_group`: `dfg_branch`, the read of color 4 in the arm of a branch on constant 0, and
/// `dfg_gated`, whose color 4 input the constant gates). Five colors on a device of four draw in two
/// groups, color 4 in the second; on `lavapipe` of four (input attachments), `moltenvk-m3` of four
/// (the pixel interlock) and `vk12-floor` (no fetch form), specialized with the read dead, each
/// group's variant compiles as a pipeline of its group's attachments that reads none (no input or
/// storage attachment, every color a color attachment), passing L3. Specialized live, group 0's
/// variant, which would read color 4 of group 1, is the pipeline's limit refusal. The misses are
/// counted by kind before the test fails.
#[test]
fn a_read_of_another_group_dead_after_specialization_is_no_refusal() {
    let (Some(bc), Some(fs)) = (fixture("color_groups"), fixture("dfk_dead_group")) else {
        return;
    };
    let colors = [0, 1, 2, 3, 4];
    let devices = [
        ("lavapipe of four", four()),
        ("moltenvk-m3 of four", narrowed("moltenvk-m3")),
        ("vk12-floor", device("vk12-floor")),
    ];
    let mut misses = std::collections::BTreeMap::<String, Vec<String>>::new();
    let mut checked = 0;
    for (name, dev) in &devices {
        for entry in ["dfg_branch", "dfg_gated"] {
            let dead = specialized(&bc, &fs, entry, &colors, false);
            for (group, held) in [(0u32, &colors[..4]), (1, &colors[4..])] {
                let what = format!("{name} {entry} dead, group {group}");
                let p = match compile(dev, &dead, group) {
                    Ok(p) => p,
                    Err(e) => {
                        let kind = match is_limit(&e, 5) {
                            true => "refused at the limit".to_owned(),
                            false => format!("refused otherwise: {e}"),
                        };
                        misses.entry(kind).or_default().push(what);
                        continue;
                    }
                };
                assert_eq!(formats(&p), reference(&dead, held), "{what}");
                assert!(p.graphics.input_attachments.is_empty(), "{what}");
                assert!(p.graphics.storage_attachments.is_empty(), "{what}");
                validate(&p, dev);
                checked += 1;
            }
            let live = specialized(&bc, &fs, entry, &colors, true);
            let e = compile(dev, &live, 0).expect_err("a read of color 4 in group 0");
            assert!(is_limit(&e, 5), "{name} {entry} live, group 0: {e:?}");
        }
    }
    for (kind, whats) in &misses {
        eprintln!("{}: {kind} (first: {})", whats.len(), whats[0]);
    }
    assert!(misses.is_empty(), "{misses:#?}");
    assert_eq!(checked, devices.len() * 2 * 2);
}

/// Whether `words` interlocks: an interlock execution mode (`PixelInterlockOrderedEXT` 5366 to
/// `ShadingRateInterlockUnorderedEXT` 5371) or an `OpBeginInvocationInterlockEXT` (5364).
fn interlocks(words: &[u32]) -> bool {
    const OP_EXECUTION_MODE: u32 = 16;
    const OP_BEGIN_INTERLOCK: u32 = 5364;
    instructions(words).iter().any(|(op, o)| {
        *op == OP_BEGIN_INTERLOCK
            || (*op == OP_EXECUTION_MODE && o.get(1).is_some_and(|m| (5366..=5371).contains(m)))
    })
}

/// A raster order group on an input the function's constants make dead orders nothing, in a
/// pipeline of groups as in any (`dfk_dead_rog`: `dfr_color`, a color 0 input in group 0, and
/// `dfr_buffer`, a buffer in group 0, each read in the arm of a branch on constant 0; plain Metal
/// makes and draws both, dead and live). Five colors on a device of four draw in two groups; on
/// `lavapipe` and `moltenvk-m3` of four (each with the pixel interlock) and `vk12-floor` (without),
/// specialized with the read dead, each group's variant compiles as a pipeline of its group's
/// attachments without an interlock, an input or a storage attachment, passing L3. Specialized
/// live, group 0's variant, which would order its read in each group's draw, is the pipeline's limit
/// refusal. The misses are counted by kind before the test fails.
#[test]
fn a_raster_order_group_dead_after_specialization_is_no_refusal_in_groups() {
    let (Some(bc), Some(fs)) = (fixture("color_groups"), fixture("dfk_dead_rog")) else {
        return;
    };
    let colors = [0, 1, 2, 3, 4];
    let devices = [
        ("lavapipe of four", four()),
        ("moltenvk-m3 of four", narrowed("moltenvk-m3")),
        ("vk12-floor", device("vk12-floor")),
    ];
    let mut misses = std::collections::BTreeMap::<String, Vec<String>>::new();
    let mut checked = 0;
    for (name, dev) in &devices {
        for entry in ["dfr_color", "dfr_buffer"] {
            let dead = specialized(&bc, &fs, entry, &colors, false);
            for (group, held) in [(0u32, &colors[..4]), (1, &colors[4..])] {
                let what = format!("{name} {entry} dead, group {group}");
                let p = match compile(dev, &dead, group) {
                    Ok(p) => p,
                    Err(e) => {
                        let kind = match is_limit(&e, 5) {
                            true => "refused at the limit".to_owned(),
                            false => format!("refused otherwise: {e}"),
                        };
                        misses.entry(kind).or_default().push(what);
                        continue;
                    }
                };
                if interlocks(&stage(&p, vk::ShaderStage::FRAGMENT).spirv) {
                    misses.entry("interlocked".into()).or_default().push(what);
                    continue;
                }
                assert_eq!(formats(&p), reference(&dead, held), "{what}");
                assert!(p.graphics.input_attachments.is_empty(), "{what}");
                assert!(p.graphics.storage_attachments.is_empty(), "{what}");
                validate(&p, dev);
                checked += 1;
            }
            let live = specialized(&bc, &fs, entry, &colors, true);
            let e = compile(dev, &live, 0).expect_err("a raster order group in group 0");
            assert!(is_limit(&e, 5), "{name} {entry} live, group 0: {e:?}");
        }
    }
    for (kind, whats) in &misses {
        eprintln!("{}: {kind} (first: {})", whats.len(), whats[0]);
    }
    assert!(misses.is_empty(), "{misses:#?}");
    assert_eq!(checked, devices.len() * 2 * 2);
}
