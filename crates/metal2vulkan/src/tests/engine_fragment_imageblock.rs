//! Fragment imageblocks through a coherent storage image and the ordered pixel interlock, over the
//! owned fixture `engine/fragment_imageblock_views` and the public
//! `fragment_custom_imageblock`:
//!
//! - Where the device has `fragmentShaderPixelInterlock` (and `fragmentStoresAndAtomics` for a
//!   function that writes its imageblock), each function runs its body inside
//!   `PixelInterlockOrderedEXT`, reaches the imageblock through one `Coherent` `R32ui` 2D-array
//!   storage image whose layer `w` is word `w` of the pixel's cell, reads its input view and writes
//!   its output view at the master layout's offsets, notes `RuntimeFact::FragmentImageblock`, and
//!   passes L3. Elsewhere it is refused, naming every capability it lacks.
//! - A function that writes its imageblock and returns a sample mask writes only where the mask
//!   covers the pixel's sample (bit 0), and views of two structurally different masters each reach
//!   their own master's bytes of a cell as large as the larger master.
//! - Not implemented: an imageblock aliasing the implicit one (an aliased input is refused as tile
//!   shading), a member that is an array or a struct or names a color attachment, a view without
//!   a data size, and a function under the Vulkan memory model.
//! - A function that reaches no raster order group runs outside the interlock, whatever other
//!   layouts of its module name.
//! - A discarded fragment writes nothing: a demoted one is a helper, whose stores have no effect,
//!   and a predicated one skips the stores.
//! - A function whose only effects are its outputs and its imageblock write declares early tests
//!   (a fragment its tests fail drops its write, Metal's answer); one that discards states
//!   `RuntimeFact::TestedStorageWrites`, whose draws' plans choose its variant.
//! - Compiled for a multisampled pipeline, a function runs once per sample (`SampleId`) and reaches
//!   its sample's cell at layers `sample * words` on; on a driver that drops an early-tests
//!   shader's image writes at more than one sample, none declares early tests, and one whose Metal
//!   function does is refused as that defect.
//! - A render pipeline's `GraphicsState::fragment_imageblock` records the storage and its master's
//!   fingerprint (equal for views of one master, different for another); link refuses a write
//!   alpha to coverage can drop after the function made it, and links beside a depth or stencil
//!   attachment and at four samples.
//! - The binder writes the storage to the fragment's synthetic binding and refuses a layered or
//!   amplified pass.
//!
//! `llvm-as` assembles the fixtures and `spirv-dis` disassembles the modules; a missing tool skips
//! visibly. No GPU is used.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use crate::bind::{BindState, VertexAmplification, ViewMapping, WriteSource};
use crate::budget::Budget;
use crate::device::requirements::VkFeature;
use crate::device::{Device, DeviceDescription, DevicePolicy, DeviceProfile};
use crate::error::{BindError, Capability, Error, Feature, Needs};
use crate::library::Library;
use crate::lower::fragment::DiscardForm;
use crate::metal::PixelFormat;
use crate::pipeline::descriptor::{
    ColorAttachmentDescriptor, FunctionDescriptor, RenderPipelineDescriptor, StageSource,
};
use crate::pipeline::facts::RuntimeFact;
use crate::pipeline::layout::BindingSource;
use crate::pipeline::sym::{Stage, SyntheticBinding};
use crate::pipeline::testing::{
    compile, engine_dir, fixtures_dir, interlocked_coherence, link_and_validate, llvm_assemble,
    typed_module, AssembleError, Scratch,
};
use crate::pipeline::{ImageblockMechanism, RenderPipeline};
use crate::spirv::ir::{ExecMode, Inst, Terminator, TypedModule};
use crate::spirv::testing::{committed_profiles, spirv_dis};
use crate::vk;

fn assemble(path: &Path) -> Option<Vec<u8>> {
    match llvm_assemble(path) {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected {}: {why}", path.display()),
    }
}

fn views() -> Option<Vec<u8>> {
    assemble(&engine_dir().join("fragment_imageblock_views.ll"))
}

/// How the depth and stencil tests bear on a function's imageblock write.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Tests {
    /// It writes none.
    None,
    /// The module declares early fragment tests.
    Early,
    /// It writes before Vulkan's tests and states `RuntimeFact::TestedStorageWrites`.
    Tested { early_variant: bool },
}

/// The tests of compiled function `a` whose typed module is `m`.
fn tests(a: &crate::library::FunctionArtifact, m: &TypedModule) -> Tests {
    let tested = a.facts.iter().find_map(|f| match *f {
        RuntimeFact::TestedStorageWrites {
            early_tests: false,
            early_variant,
        } => Some(early_variant),
        _ => None,
    });
    match (
        m.entry().modes.contains(&ExecMode::EarlyFragmentTests),
        tested,
    ) {
        (true, None) => Tests::Early,
        (false, Some(early_variant)) => Tests::Tested { early_variant },
        (false, None) => Tests::None,
        (true, Some(_)) => panic!("early tests beside a tested write"),
    }
}

/// One fragment function's imageblock: whether it writes it, how the tests bear on the write at
/// one sample, its cell's words and bytes, its views' alignment, and the cell words it reads and
/// writes (the layers of its image reads and writes; a write of part of a word reads that word
/// first).
struct Expect {
    entry: &'static str,
    writes: bool,
    tests: Tests,
    words: u32,
    data_size: u32,
    alignment: u32,
    read_layers: &'static [u32],
    written_layers: &'static [u32],
}

/// The views fixture's fragment functions. `imageblock_transition` reads `b` (the high half of
/// word 0) and `sum` (word 2) and writes `a` (the low half of word 0), `tag` (word 1), and `n`
/// (the low half of word 3), each at its master offset, through views aligned to 4 bytes; the
/// `Cell` functions, whose layout is aligned to 8, hold `color` in words 0 and 1 and `depth` in
/// the low half of word 2. Each cell is 16 bytes.
const VIEWS: [Expect; 4] = [
    Expect {
        entry: "imageblock_transition",
        writes: true,
        tests: Tests::Early,
        words: 4,
        data_size: 16,
        alignment: 4,
        read_layers: &[0, 2, 3],
        written_layers: &[0, 1, 3],
    },
    Expect {
        entry: "imageblock_accumulate",
        writes: true,
        tests: Tests::Early,
        words: 4,
        data_size: 16,
        alignment: 8,
        read_layers: &[0, 1, 2],
        written_layers: &[0, 1, 2],
    },
    Expect {
        entry: "imageblock_clear_odd",
        writes: true,
        tests: Tests::Tested {
            early_variant: true,
        },
        words: 4,
        data_size: 16,
        alignment: 8,
        read_layers: &[2],
        written_layers: &[0, 1, 2],
    },
    Expect {
        entry: "imageblock_resolve",
        writes: false,
        tests: Tests::None,
        words: 4,
        data_size: 16,
        alignment: 8,
        read_layers: &[0, 1, 2],
        written_layers: &[],
    },
];

fn description(stem: &str) -> DeviceDescription {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles")
        .join(format!("{stem}.json"));
    let text = std::fs::read_to_string(path).expect("profile");
    DeviceDescription::from_khronos_json(&text, None).expect("description")
}

fn device(stem: &str) -> Device {
    Device::from_queries(description(stem), DevicePolicy::default()).expect("device")
}

/// `moltenvk-m3` without `features`.
fn moltenvk_without(features: &[VkFeature]) -> DeviceProfile {
    let mut desc = description("moltenvk-m3");
    for f in features {
        assert!(desc.features.remove(f), "moltenvk-m3 has {f:?}");
    }
    Device::from_queries(desc, DevicePolicy::default())
        .expect("device")
        .profile()
        .clone()
}

/// The `FragmentImageblock` fact of an artifact's facts: its words, whether the function writes
/// the imageblock, its cell's bytes, and its views' alignment.
fn fact(facts: &[RuntimeFact]) -> Option<(u32, bool, u32, u32)> {
    facts.iter().find_map(|f| match *f {
        RuntimeFact::FragmentImageblock {
            words,
            writes,
            data_size,
            alignment,
            ..
        } => Some((words, writes, data_size, alignment)),
        _ => None,
    })
}

/// Whether compiling `entry` of `bc` on `p` is refused `Unsupported{Imageblocks, NotImplemented}`.
fn not_implemented(bc: &[u8], entry: &str, p: &DeviceProfile) {
    match compile(bc, entry, p) {
        Err(Error::Unsupported(u)) => assert_eq!(
            (u.feature, u.needs),
            (Feature::Imageblocks, Needs::NotImplemented),
            "{entry}"
        ),
        other => panic!("{entry}: {:?}", other.map(|_| ())),
    }
}

/// The layers of the image reads and writes of disassembled module `dis`, and whether it
/// declares exactly one image, a `Coherent` `R32ui` 2D-array storage image.
fn cell_words(dis: &str) -> (BTreeSet<u32>, BTreeSet<u32>) {
    let images: Vec<&str> = dis.lines().filter(|l| l.contains("OpTypeImage")).collect();
    assert_eq!(images.len(), 1, "one image type: {images:?}");
    assert!(
        images[0].ends_with("OpTypeImage %uint 2D 0 1 0 2 R32ui"),
        "{}",
        images[0]
    );
    assert_eq!(
        dis.lines().filter(|l| l.ends_with(" Coherent")).count(),
        1,
        "one coherent variable"
    );
    // `%c = OpCompositeConstruct %v3uint %x %y %uint_<layer>`
    let mut layer = BTreeMap::new();
    for l in dis.lines() {
        let t: Vec<&str> = l.split_whitespace().collect();
        if let [id, "=", "OpCompositeConstruct", "%v3uint", _, _, w] = t[..] {
            if let Some(n) = w.strip_prefix("%uint_") {
                layer.insert(id.to_owned(), n.parse::<u32>().expect("layer"));
            }
        }
    }
    let at = |coord: &str| {
        *layer
            .get(coord)
            .unwrap_or_else(|| panic!("layer of {coord}"))
    };
    let (mut reads, mut writes) = (BTreeSet::new(), BTreeSet::new());
    for l in dis.lines() {
        let t: Vec<&str> = l.split_whitespace().collect();
        match t[..] {
            [_, "=", "OpImageRead", _, _, coord, ..] => {
                reads.insert(at(coord));
            }
            ["OpImageWrite", _, coord, ..] => {
                writes.insert(at(coord));
            }
            _ => {}
        }
    }
    (reads, writes)
}

/// The refusal of an imageblock function on a device that lacks `missing`.
fn lacks(missing: &[VkFeature]) -> Needs {
    Needs::AnyOf(vec![missing
        .iter()
        .map(|f| Capability::Feature(*f))
        .collect()])
}

/// On every committed profile with pixel interlock (and fragment stores for a writer) each function
/// compiles inside the ordered interlock, notes its imageblock, and passes L3; on `moltenvk-m3` its
/// image reads and writes reach exactly the words of its views at their master offsets through one
/// coherent word-plane image. Every other profile refuses it, naming each capability it lacks.
#[test]
fn each_view_reaches_its_master_words_inside_the_interlock() {
    let Some(bc) = views() else {
        return;
    };
    let (mut compiled, mut refused) = (0, 0);
    for (stem, p) in committed_profiles().expect("profiles") {
        let f = p.fields();
        for e in &VIEWS {
            let what = format!("{} on {stem}", e.entry);
            let mut missing = Vec::new();
            if !f.fragment.pixel_interlock {
                missing.push(VkFeature::FragmentShaderPixelInterlock);
            }
            if e.writes && !f.leaf.fragment_stores_and_atomics {
                missing.push(VkFeature::FragmentStoresAndAtomics);
            }
            if !missing.is_empty() {
                match compile(&bc, e.entry, &p) {
                    Err(Error::Unsupported(u)) => {
                        assert_eq!(
                            (u.feature, u.needs),
                            (Feature::Imageblocks, lacks(&missing)),
                            "{what}"
                        );
                    }
                    other => panic!("{what}: {:?}", other.map(|_| ())),
                }
                refused += 1;
                continue;
            }
            let (_, m) = typed_module(&bc, e.entry, &p).unwrap_or_else(|e| panic!("{what}: {e}"));
            assert!(
                m.entry().modes.contains(&ExecMode::PixelInterlockOrdered),
                "{what}"
            );
            let a = compile(&bc, e.entry, &p).unwrap_or_else(|e| panic!("{what}: {e}"));
            assert_eq!(
                fact(&a.facts),
                Some((e.words, e.writes, e.data_size, e.alignment)),
                "{what}"
            );
            assert_eq!(tests(&a, &m), e.tests, "{what}");
            let (_, words) = link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: {e}"));
            if stem == "moltenvk-m3" {
                let (reads, writes) = cell_words(&spirv_dis(&words));
                assert_eq!(
                    reads,
                    e.read_layers.iter().copied().collect(),
                    "{what}: reads"
                );
                assert_eq!(
                    writes,
                    e.written_layers.iter().copied().collect(),
                    "{what}: writes"
                );
            }
            compiled += 1;
        }
    }
    assert!(
        compiled > 0 && refused > 0,
        "compiled {compiled}, refused {refused}"
    );
    // A device that lacks both names both in one alternative; a reader needs only the interlock.
    let p = moltenvk_without(&[
        VkFeature::FragmentShaderPixelInterlock,
        VkFeature::FragmentStoresAndAtomics,
    ]);
    for e in &VIEWS {
        let missing: &[VkFeature] = if e.writes {
            &[
                VkFeature::FragmentShaderPixelInterlock,
                VkFeature::FragmentStoresAndAtomics,
            ]
        } else {
            &[VkFeature::FragmentShaderPixelInterlock]
        };
        match compile(&bc, e.entry, &p) {
            Err(Error::Unsupported(u)) => {
                assert_eq!((u.feature, u.needs), (Feature::Imageblocks, lacks(missing)))
            }
            other => panic!("{}: {:?}", e.entry, other.map(|_| ())),
        }
    }
}

/// A function that writes its imageblock and returns a sample mask keeps its write only where the
/// mask covers the pixel's one sample: Metal drops the fragment, and its write, where bit 0 is
/// clear (measured on the M3 Max and an M4, whatever the other bits). The plan names the mask's
/// field, every image write runs on the branch the bit leaves, and the pipeline links
/// single-sampled (link refuses a multisampled one).
#[test]
fn a_write_beside_a_sample_mask_is_kept_where_the_mask_covers_the_sample() {
    let Some(bc) = views() else {
        return;
    };
    let p = moltenvk_without(&[]);
    let entry = "imageblock_masked";
    let (plan, m) = typed_module(&bc, entry, &p).expect("typed module");
    let ib = plan.families.fragment.imageblock.as_ref().expect("plan");
    assert_eq!(ib.mask, Some(1));
    let writing = writing_blocks(&m);
    assert!(!writing.is_empty());
    let guarded = |b: &usize| {
        m.blocks().iter().any(|h| {
            matches!(h.term, Terminator::CondBranch { t, f, .. }
                if t.index() == *b || f.index() == *b)
        })
    };
    assert!(writing.iter().all(guarded), "{writing:?}");
    let a = compile(&bc, entry, &p).expect("compile");
    assert_eq!(fact(&a.facts), Some((4, true, 16, 8)));
    // A returned mask keeps the write from early tests.
    assert_eq!(
        tests(&a, &m),
        Tests::Tested {
            early_variant: false
        }
    );
    link_and_validate(&a, &p).expect("L3");
    let dev = device("moltenvk-m3");
    dev.compile_render_pipeline(&render(&bc, entry), &Budget::standard())
        .expect("pipeline");
    // A function that only reads its imageblock and returns a mask plans no guard.
    let (plan, _) = typed_module(&bc, "imageblock_transition", &p).expect("typed module");
    assert_eq!(plan.families.fragment.imageblock.expect("plan").mask, None);
}

/// An imageblock aliasing the implicit one is the color attachments' own storage, which the side
/// image is not: a function that only writes it is not implemented (`Imageblocks`), and one that
/// reads it is refused as tile shading with every other aliased imageblock parameter
/// (`select::param_family`).
#[test]
fn an_imageblock_aliasing_the_implicit_one_is_not_implemented() {
    let Some(bc) = views() else {
        return;
    };
    let p = moltenvk_without(&[]);
    not_implemented(&bc, "imageblock_aliased", &p);
    match compile(&bc, "imageblock_aliased_in", &p) {
        Err(Error::Unsupported(u)) => assert_eq!(
            (u.feature, u.needs),
            (Feature::TileShading, Needs::NotImplemented)
        ),
        other => panic!("{:?}", other.map(|_| ())),
    }
}

/// `imageblock_two_masters` reads `a` through `Master` (byte 0) and writes it through `Swapped`
/// (byte 2). Each view reaches its own master's bytes of the one cell, as Metal's do (measured on
/// the M3 Max and an M4): the input reads word 0's low half and the output replaces its high half.
/// The cell spans the larger master, `Master`'s 16 bytes at its alignment, and the fact names
/// `Master`, which `imageblock_transition` views too.
#[test]
fn views_of_two_masters_reach_each_masters_bytes() {
    let Some(bc) = views() else {
        return;
    };
    let p = moltenvk_without(&[]);
    let entry = "imageblock_two_masters";
    let (plan, _) = typed_module(&bc, entry, &p).expect("typed module");
    let ib = plan.families.fragment.imageblock.expect("plan");
    let (input, output) = (ib.input.expect("input").1, ib.output.expect("output").1);
    let input: Vec<u32> = input.members.iter().map(|m| m.offset).collect();
    let output: Vec<u32> = output.members.iter().map(|m| m.offset).collect();
    assert_eq!((input, output), (vec![0], vec![2]));
    let a = compile(&bc, entry, &p).expect("compile");
    assert_eq!(fact(&a.facts), Some((4, true, 16, 4)));
    let (_, words) = link_and_validate(&a, &p).expect("L3");
    let (reads, writes) = cell_words(&spirv_dis(&words));
    assert_eq!((reads, writes), ([0].into(), [0].into()));
    let dev = device("moltenvk-m3");
    let master = |entry: &str| {
        dev.compile_render_pipeline(&render(&bc, entry), &Budget::standard())
            .unwrap_or_else(|e| panic!("{entry}: {e}"))
            .graphics
            .fragment_imageblock
            .expect("imageblock")
            .master
    };
    assert_eq!(master(entry), master("imageblock_transition"));
}

/// A function that reaches no raster order group runs outside the interlock, and so compiles on
/// every committed profile, pixel interlock or not, although other layouts of its module (the
/// imageblock masters) name groups.
#[test]
fn a_function_without_a_group_runs_outside_the_interlock() {
    let Some(bc) = views() else {
        return;
    };
    for (stem, p) in committed_profiles().expect("profiles") {
        let (_, m) = typed_module(&bc, "plain_color", &p).unwrap_or_else(|e| panic!("{stem}: {e}"));
        assert!(
            !m.entry().modes.contains(&ExecMode::PixelInterlockOrdered),
            "{stem}"
        );
        let a = compile(&bc, "plain_color", &p).unwrap_or_else(|e| panic!("{stem}: {e}"));
        link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{stem}: L3: {e}"));
    }
}

/// A fragment function `probe` whose imageblock input is struct `Cell = { fields }` of layout
/// `members`, with `size` the `air.imageblock_data_size` operands (or none), a device buffer at
/// index 0, and `body` computing its `<4 x half>` result `%r`.
fn probe(fields: &str, members: &str, size: &str, body: &str, decls: &str) -> String {
    format!(
        r#"target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

%struct.Inner = type {{ half, half }}
%struct.Cell = type {{ {fields} }}

define <4 x half> @probe(%struct.Cell %0, ptr addrspace(1) %1) {{
{body}
  ret <4 x half> %r
}}
{decls}
!air.fragment = !{{!0}}
!air.version = !{{!10}}
!air.language_version = !{{!11}}
!0 = !{{ptr @probe, !1, !2}}
!1 = !{{!3}}
!3 = !{{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"half4"}}
!2 = !{{!4, !5}}
!4 = !{{i32 0, !"air.imageblock_data", {size}!"air.struct_type_info", !6, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"Cell", !"air.arg_name", !"in"}}
!5 = !{{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"buf"}}
!6 = !{{{members}}}
!7 = !{{i32 0, i32 2, i32 0, !"half", !"x", i32 2, i32 2, i32 0, !"half", !"y"}}
!10 = !{{i32 2, i32 8, i32 0}}
!11 = !{{!"Metal", i32 4, i32 0, i32 0}}
"#
    )
}

/// Assemble `text` through a scratch file named after `name`, removed after.
fn assemble_text(name: &str, text: &str) -> Option<Vec<u8>> {
    let scratch = Scratch::new(&format!("m2v-fragment-imageblock-{name}"));
    let dir = scratch.path();
    let path = dir.join(format!("{name}.ll"));
    std::fs::write(&path, text).expect("write scratch .ll");
    assemble(&path)
}

const CELL: &str = "<4 x half>, half";
const CELL_MEMBERS: &str = r#"i32 0, i32 8, i32 0, !"half4", !"color", !"air.raster_order_group", i32 0, i32 8, i32 2, i32 0, !"half", !"depth", !"air.raster_order_group", i32 0"#;
const SIZE: &str = "!\"air.imageblock_data_size\", i32 16, ";
const READ_COLOR: &str = "  %r = extractvalue %struct.Cell %0, 0";

/// What a view's structure puts out of reach is not implemented: a member that is an array or a
/// struct, a member naming a color attachment (crafted: an explicit layout cannot hold one, so it
/// is the implicit imageblock's), and a view without a data size. The control, the same function
/// without any of them, compiles, and so does one under the Vulkan memory model (a coherent buffer
/// load selects it), whose imageblock accesses then carry `MakeTexel*` at device scope where the
/// control's image is `Coherent`.
#[test]
fn unmodelled_views_are_not_implemented() {
    let coherent = format!(
        "{READ_COLOR}\n  %v = call i32 @air.load.system_coherent.volatile.i32.p1i32(ptr addrspace(1) %1)\n  %h = trunc i32 %v to i16\n  %f = bitcast i16 %h to half\n  %r = insertelement <4 x half> %c, half %f, i32 3"
    )
    .replace("%r = extractvalue", "%c = extractvalue");
    let cases = [
        ("control", CELL, CELL_MEMBERS, SIZE, READ_COLOR.to_owned(), "", true),
        (
            "coherent",
            CELL,
            CELL_MEMBERS,
            SIZE,
            coherent,
            "declare i32 @air.load.system_coherent.volatile.i32.p1i32(ptr addrspace(1))",
            true,
        ),
        ("unsized", CELL, CELL_MEMBERS, "", READ_COLOR.to_owned(), "", false),
        (
            "color",
            CELL,
            r#"i32 0, i32 8, i32 0, !"half4", !"color", !"air.render_target", i32 0, i32 8, i32 2, i32 0, !"half", !"depth", !"air.raster_order_group", i32 0"#,
            SIZE,
            READ_COLOR.to_owned(),
            "",
            false,
        ),
        (
            "array",
            "[2 x half]",
            r#"i32 0, i32 2, i32 2, !"half", !"pair", !"air.raster_order_group", i32 0"#,
            "!\"air.imageblock_data_size\", i32 4, ",
            "  %e = extractvalue %struct.Cell %0, 0, 1\n  %r = insertelement <4 x half> zeroinitializer, half %e, i32 0".to_owned(),
            "",
            false,
        ),
        (
            "struct",
            "%struct.Inner",
            r#"!"air.struct_type_info", !7, i32 0, i32 4, i32 0, !"Inner", !"inner", !"air.raster_order_group", i32 0"#,
            "!\"air.imageblock_data_size\", i32 4, ",
            "  %e = extractvalue %struct.Cell %0, 0, 1\n  %r = insertelement <4 x half> zeroinitializer, half %e, i32 0".to_owned(),
            "",
            false,
        ),
    ];
    let mut profiles = committed_profiles().expect("profiles");
    profiles.retain(|(_, p)| p.fields().fragment.pixel_interlock && p.fields().vulkan_memory_model);
    assert!(!profiles.is_empty());
    for (name, fields, members, size, body, decls, ok) in cases {
        let Some(bc) = assemble_text(name, &probe(fields, members, size, &body, decls)) else {
            return;
        };
        for (stem, p) in &profiles {
            if ok {
                let a = compile(&bc, "probe", p).unwrap_or_else(|e| panic!("{name} {stem}: {e}"));
                let (_, words) =
                    link_and_validate(&a, p).unwrap_or_else(|e| panic!("{name} {stem}: L3: {e}"));
                let n =
                    interlocked_coherence(&words).unwrap_or_else(|e| panic!("{name} {stem}: {e}"));
                assert_eq!(n.vulkan, name == "coherent", "{name} {stem}");
                assert!(n.storage_image > 0, "{name} {stem}: {n:?}");
            } else {
                not_implemented(&bc, "probe", p);
            }
        }
    }
}

/// The public fixture's imageblock (read and written, no master) compiles the same way.
#[test]
fn the_public_imageblock_fixture_compiles_inside_the_interlock() {
    let Some(bc) = assemble(&fixtures_dir().join("fragment_custom_imageblock.ll")) else {
        return;
    };
    let p = moltenvk_without(&[]);
    let a = compile(&bc, "fragment_custom_imageblock", &p).expect("compile");
    assert!(
        matches!(fact(&a.facts), Some((_, true, ..))),
        "{:?}",
        a.facts
    );
    let (_, words) = link_and_validate(&a, &p).expect("L3");
    let (reads, writes) = cell_words(&spirv_dis(&words));
    assert!(!reads.is_empty() && !writes.is_empty());
}

/// The blocks of `m`'s functions that write an image.
fn writing_blocks(m: &TypedModule) -> Vec<usize> {
    m.blocks()
        .iter()
        .enumerate()
        .filter(|(_, b)| {
            b.insts
                .iter()
                .any(|n| matches!(n.inst, Inst::ImageWrite { .. }))
        })
        .map(|(i, _)| i)
        .collect()
}

/// A discarded fragment writes nothing to its imageblock. Where the device demotes, the discarded
/// fragment is a helper, whose stores have no effect, so the writes stand unguarded; where it
/// predicates, the writes run only on the branch the predicate leaves.
#[test]
fn a_discarded_fragment_writes_nothing() {
    let Some(bc) = views() else {
        return;
    };
    let entry = "imageblock_clear_odd";
    for (p, form) in [
        (moltenvk_without(&[]), DiscardForm::Demote),
        (
            moltenvk_without(&[VkFeature::ShaderDemoteToHelperInvocation]),
            DiscardForm::Predicate,
        ),
    ] {
        let (plan, m) = typed_module(&bc, entry, &p).unwrap_or_else(|e| panic!("{form:?}: {e}"));
        let forms: Vec<DiscardForm> = plan.families.fragment.discards.values().copied().collect();
        assert_eq!(forms, [form]);
        let writing = writing_blocks(&m);
        assert!(!writing.is_empty(), "{form:?}");
        let guarded = |b: &usize| {
            m.blocks().iter().any(|h| {
                matches!(h.term, Terminator::CondBranch { t, f, .. }
                    if t.index() == *b || f.index() == *b)
            })
        };
        match form {
            DiscardForm::Demote => assert!(!writing.iter().any(guarded), "{form:?}"),
            _ => assert!(writing.iter().all(guarded), "{form:?}"),
        }
        let a = compile(&bc, entry, &p).unwrap_or_else(|e| panic!("{form:?}: {e}"));
        link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{form:?}: L3: {e}"));
    }
}

/// A render pipeline of `imageblock_vertex` and `fragment` with one `RGBA16Float` color
/// attachment.
fn render(bc: &[u8], fragment: &str) -> RenderPipelineDescriptor {
    let lib = Library::from_air(bc).expect("library");
    let stage = |name: &str| {
        StageSource::Function(FunctionDescriptor::new(
            lib.function(name).expect("function"),
        ))
    };
    let mut d = RenderPipelineDescriptor::new(stage("imageblock_vertex"));
    d.fragment = Some(stage(fragment));
    d.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA16Float));
    d
}

fn refused_not_implemented(r: Result<RenderPipeline, Error>, what: &str) {
    match r {
        Err(Error::Unsupported(u)) => assert_eq!(
            (u.feature, u.needs),
            (Feature::Imageblocks, Needs::NotImplemented),
            "{what}"
        ),
        other => panic!("{what}: {:?}", other.map(|_| ())),
    }
}

/// Link records the pass's imageblock storage in the graphics state, links beside a depth or
/// stencil attachment (the tests run before each write: early, or as each draw's plan chooses),
/// and refuses a write alpha to coverage can drop after the function made it.
#[test]
fn link_records_the_imageblock_and_refuses_what_it_does_not_model() {
    let Some(bc) = views() else {
        return;
    };
    let dev = device("moltenvk-m3");
    let budget = Budget::standard();
    let mut masters = BTreeMap::new();
    for e in &VIEWS {
        let p = dev
            .compile_render_pipeline(&render(&bc, e.entry), &budget)
            .unwrap_or_else(|err| panic!("{}: {err}", e.entry));
        let fi = p.graphics.fragment_imageblock.expect("imageblock");
        assert_eq!(
            (fi.mechanism, fi.words),
            (ImageblockMechanism::StorageImageInterlock, e.words),
            "{}",
            e.entry
        );
        masters.insert(e.entry, fi.master);
        for depth in [
            (Some(PixelFormat::Depth32Float), None),
            (None, Some(PixelFormat::Stencil8)),
        ] {
            let mut d = render(&bc, e.entry);
            (d.depth_format, d.stencil_format) = depth;
            dev.compile_render_pipeline(&d, &budget)
                .unwrap_or_else(|err| panic!("{} with {depth:?}: {err}", e.entry));
        }
        let mut d = render(&bc, e.entry);
        d.alpha_to_coverage = true;
        let r = dev.compile_render_pipeline(&d, &budget);
        if e.writes {
            refused_not_implemented(r, &format!("{} alpha to coverage", e.entry));
        } else {
            r.unwrap_or_else(|err| panic!("{} alpha to coverage: {err}", e.entry));
        }
    }
    // The `Cell` functions view one layout (no master); `imageblock_transition` views `Master`.
    let cell = masters["imageblock_accumulate"];
    for entry in ["imageblock_clear_odd", "imageblock_resolve"] {
        assert_eq!(masters[entry], cell, "{entry}");
    }
    assert_ne!(masters["imageblock_transition"], cell);
}

/// The binder writes the pass's imageblock storage to the fragment's storage-image binding and
/// refuses a pass of more than one layer and an amplified draw.
#[test]
fn the_binder_writes_the_storage_and_refuses_layered_passes() {
    let Some(bc) = views() else {
        return;
    };
    let dev = device("moltenvk-m3");
    let mut d = render(&bc, "imageblock_transition");
    d.max_vertex_amplification_count = 2;
    let p = dev
        .compile_render_pipeline(&d, &Budget::standard())
        .expect("pipeline");
    let plan = p.plan(&dev, &BindState::default().shape()).expect("plan");
    let imageblock: Vec<_> = plan
        .descriptors
        .iter()
        .filter(|w| w.source == WriteSource::Imageblock)
        .collect();
    assert_eq!(imageblock.len(), 1, "{:?}", plan.descriptors);
    let b = p
        .layout
        .sets
        .iter()
        .flat_map(|s| &s.bindings)
        .find(|b| b.binding == imageblock[0].binding)
        .expect("binding");
    assert_eq!(b.ty, vk::DescriptorType::STORAGE_IMAGE);
    assert_eq!(
        b.source,
        BindingSource::Synthetic {
            stage: Stage::Fragment,
            binding: SyntheticBinding::Imageblock
        }
    );
    let layered = BindState {
        render_target_array_length: 2,
        ..Default::default()
    };
    // Two views over the two layers of a layered pass: multiview the binder plans.
    let amplified = BindState {
        vertex_amplification: VertexAmplification::new(
            2,
            vec![ViewMapping::new(0, 0), ViewMapping::new(0, 1)],
        ),
        render_target_array_length: 2,
        ..Default::default()
    };
    let single = BindState {
        vertex_amplification: VertexAmplification::new(1, Vec::new()),
        ..Default::default()
    };
    p.plan(&dev, &single.shape()).expect("one view");
    for (what, s) in [("layered", layered), ("amplified", amplified)] {
        match p.plan(&dev, &s.shape()) {
            Err(BindError::Unsupported(u)) => assert_eq!(
                (u.feature, u.needs),
                (Feature::Imageblocks, Needs::NotImplemented),
                "{what}"
            ),
            other => panic!("{what}: {:?}", other.map(|_| ())),
        }
    }
}

/// At four samples each function runs once per sample (`SampleId`) and reaches its sample's cell,
/// layers `sample * words + w`; the pipeline links, and the graphics state records the storage. A
/// function whose only effects are its outputs and its write declares early tests as at one
/// sample, on MoltenVK on the M3 Max and on RADV alike; a function that states
/// `TestedStorageWrites` needs, for a draw whose depth test acts, the variant that tests depth in
/// the shader, before the write, whose plan is refused as not implemented at the depth attachment:
/// no profile states which depth a sample-rate `FragCoord` holds
/// (`VkProperty::SampleFragCoordDepth`, `CenterFragCoordDepth`).
#[test]
fn a_multisampled_pipeline_reaches_each_sample_s_cell() {
    use crate::bind::FragmentTests;
    let Some(bc) = views() else {
        return;
    };
    let budget = Budget::standard();
    for stem in ["moltenvk-m3", "amd-780m-radv"] {
        let dev = device(stem);
        for e in &VIEWS {
            let what = format!("{} on {stem}", e.entry);
            let mut d = render(&bc, e.entry);
            d.raster_sample_count = 4;
            d.depth_format = Some(PixelFormat::Depth32Float);
            let p = dev
                .compile_render_pipeline(&d, &budget)
                .unwrap_or_else(|err| panic!("{what}: {err}"));
            let fi = p.graphics.fragment_imageblock.expect("imageblock");
            assert_eq!(fi.words, e.words, "{what}");
            let fragment = (p.stages.iter())
                .find(|s| s.stage == vk::ShaderStage::FRAGMENT)
                .expect("a fragment stage");
            let dis = spirv_dis(&fragment.spirv);
            assert!(dis.contains("BuiltIn SampleId"), "{what}");
            assert!(dis.contains("OpIMul"), "{what}");
            let early = dis.contains("EarlyFragmentTests");
            let tested = p.dependencies.iter().find_map(|x| match x.fact {
                RuntimeFact::TestedStorageWrites {
                    early_tests: false,
                    early_variant,
                } => Some(early_variant),
                _ => None,
            });
            let want = e.tests;
            let got = match (early, tested) {
                (true, None) => Tests::Early,
                (false, Some(early_variant)) => Tests::Tested { early_variant },
                (false, None) => Tests::None,
                (true, Some(_)) => panic!("{what}: early tests beside a tested write"),
            };
            assert_eq!(got, want, "{what}");
            // A draw whose depth test acts runs the in-shader test before the write.
            if let Tests::Tested { .. } = got {
                let shape = BindState {
                    fragment_tests: FragmentTests::Depth { writes: true },
                    ..Default::default()
                }
                .shape();
                match p.plan(&dev, &shape) {
                    Err(BindError::Unsupported(u)) => assert_eq!(
                        (u.feature, u.needs),
                        (Feature::FramebufferFetch, Needs::NotImplemented),
                        "{what}"
                    ),
                    other => panic!("{what}: the plan: {:?}", other.map(|_| ())),
                }
            }
        }
    }
}
