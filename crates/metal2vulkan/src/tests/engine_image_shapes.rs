//! The image a texture slot's binding declares is the image its view shows: the engine's
//! guard over its final modules, as `tests/reflection_describes_the_image_it_binds.rs` was for the
//! old engine's reflected `texture_shape` until 0.2.0 deleted both.
//!
//! Vulkan requires an image view's type to match the `Dim` and `Arrayed` of the image variable a
//! descriptor is read through, and its sample count the variable's `MS`. The binder makes each
//! view from what the interface reports: the slot's Metal texture type, and the pipeline's
//! `ImageViewShape` fact for a cube viewed as the 2D array it is declared as (SPIR-V has no texel
//! read of a cube). `bind::plan::resources`' own tests pin the binder's mapping from those to a
//! view. This file pins the other half: an independent walk of the final words finds every
//! variable a texture slot's binding names and requires its image to be the one that mapping
//! gives, so a module never declares an image its view cannot show.
//!
//! The mapping, written out again here: 1D, 2D and 3D types are their dimension, an array type
//! is arrayed, a multisample type is multisampled, a cube is a cube, a cube array an arrayed
//! cube, and a texture buffer a buffer image. `ImageViewShape` makes a cube or a cube array an
//! arrayed 2D image in the slot's own binding, which texel reads go through; a
//! combined image sampler's or a separate texture's binding samples, and keeps the cube. Each
//! binding that holds a texture slot's image is checked.
//!
//! The sweep compiles every entry point of every committed opaque-pointer fixture on four
//! committed profiles, each linked alone. `llvm-as` assembles the fixtures; without it the test
//! skips visibly.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use crate::device::{Device, DeviceDescription, DevicePolicy};
use crate::library::{FunctionStage, Library};
use crate::metal::TextureType;
use crate::pipeline::sym::{ResourceKind, ResourceSlot};
use crate::pipeline::testing::{
    engine_dir, fixtures_dir, link_standalone, llvm_assemble, AssembleError,
};
use crate::pipeline::{
    FunctionDescriptor, LoweringInputs, RuntimeFact, SlotClass, SlotInterface, SlotRef,
};
use crate::Budget;

const OP_TYPE_IMAGE: u32 = 25;
const OP_TYPE_SAMPLED_IMAGE: u32 = 27;
const OP_TYPE_ARRAY: u32 = 28;
const OP_TYPE_RUNTIME_ARRAY: u32 = 29;
const OP_TYPE_POINTER: u32 = 32;
const OP_VARIABLE: u32 = 59;
const OP_DECORATE: u32 = 71;
const DECORATION_BINDING: u32 = 33;
const DECORATION_DESCRIPTOR_SET: u32 = 34;

/// SPIR-V `Dim`.
const DIM_1D: u32 = 0;
const DIM_2D: u32 = 1;
const DIM_3D: u32 = 2;
const DIM_CUBE: u32 = 3;
const DIM_BUFFER: u32 = 5;

/// An image's `Dim`, `Arrayed` and `MS`.
type Shape = (u32, bool, bool);

/// The instructions of `words`: each opcode with its operand words.
fn instructions(words: &[u32]) -> Vec<(u32, &[u32])> {
    const HEADER_WORDS: usize = 5;
    let mut out = Vec::new();
    let mut index = HEADER_WORDS;
    while index < words.len() {
        let length = (words[index] >> 16) as usize;
        assert!(
            length > 0 && index + length <= words.len(),
            "the instruction at word {index} declares {length} words"
        );
        out.push((words[index] & 0xffff, &words[index + 1..index + length]));
        index += length;
    }
    out
}

/// The image shape of every variable of `words` with a descriptor binding, by `(set, binding)`:
/// the image its pointee is, through sampled images and arrays. A variable whose pointee holds no
/// image (a buffer, a sampler) is left out.
fn images(words: &[u32]) -> BTreeMap<(u32, u32), Vec<Shape>> {
    let mut image = BTreeMap::new();
    let mut wraps = BTreeMap::new();
    let mut pointee = BTreeMap::new();
    let mut variables = BTreeMap::new();
    let mut set = BTreeMap::new();
    let mut binding = BTreeMap::new();
    for (opcode, ops) in instructions(words) {
        match opcode {
            OP_TYPE_IMAGE => {
                image.insert(ops[0], (ops[2], ops[4] == 1, ops[5] == 1));
            }
            OP_TYPE_SAMPLED_IMAGE | OP_TYPE_ARRAY | OP_TYPE_RUNTIME_ARRAY => {
                wraps.insert(ops[0], ops[1]);
            }
            OP_TYPE_POINTER => {
                pointee.insert(ops[0], ops[2]);
            }
            OP_VARIABLE => {
                variables.insert(ops[1], ops[0]);
            }
            OP_DECORATE if ops[1] == DECORATION_DESCRIPTOR_SET => {
                set.insert(ops[0], ops[2]);
            }
            OP_DECORATE if ops[1] == DECORATION_BINDING => {
                binding.insert(ops[0], ops[2]);
            }
            _ => {}
        }
    }
    let mut out: BTreeMap<(u32, u32), Vec<Shape>> = BTreeMap::new();
    for (var, ty) in variables {
        let (Some(&s), Some(&b), Some(&p)) = (set.get(&var), binding.get(&var), pointee.get(&ty))
        else {
            continue;
        };
        let mut t = p;
        while let Some(&inner) = wraps.get(&t) {
            t = inner;
        }
        if let Some(&shape) = image.get(&t) {
            out.entry((s, b)).or_default().push(shape);
        }
    }
    out
}

/// The image a slot of `texture_type` is declared as, viewed as the 2D array when `cube_as_array`
/// (the `ImageViewShape` fact).
fn expected(texture_type: TextureType, cube_as_array: bool) -> Shape {
    match texture_type {
        TextureType::Type1D => (DIM_1D, false, false),
        TextureType::Type1DArray => (DIM_1D, true, false),
        TextureType::Type2D => (DIM_2D, false, false),
        TextureType::Type2DArray => (DIM_2D, true, false),
        TextureType::Type2DMultisample => (DIM_2D, false, true),
        TextureType::Type2DMultisampleArray => (DIM_2D, true, true),
        TextureType::Cube | TextureType::CubeArray if cube_as_array => (DIM_2D, true, false),
        TextureType::Cube => (DIM_CUBE, false, false),
        TextureType::CubeArray => (DIM_CUBE, true, false),
        TextureType::Type3D => (DIM_3D, false, false),
        TextureType::TextureBuffer => (DIM_BUFFER, false, false),
    }
}

/// The texture slot index whose image binding `resource` holds, and whether the binding samples
/// it (a combined image sampler's or a separate texture's), or `None` for a binding that holds no
/// texture slot's image.
fn texture_slot(resource: &ResourceSlot) -> Option<(u32, bool)> {
    match resource.kind {
        ResourceKind::Texture => Some((resource.index, false)),
        ResourceKind::CombinedTextureSampler(_) => Some((resource.index, true)),
        ResourceKind::SeparateTexture(_) => resource.separate_texture_of().map(|(t, _)| (t, true)),
        _ => None,
    }
}

/// Profiles that bind textures by descriptor and differ in what they support: robust image
/// access and its absence, formatless storage images, and the floor.
const PROFILES: [&str; 4] = [
    "lunarg-desktop-baseline-2023",
    "vk12-floor",
    "moltenvk-m3",
    "nvidia-rtx5080-laptop",
];

fn device(stem: &str) -> Device {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles")
        .join(format!("{stem}.json"));
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let description =
        DeviceDescription::from_khronos_json(&text, None).unwrap_or_else(|e| panic!("{stem}: {e}"));
    Device::from_queries(description, DevicePolicy::default())
        .unwrap_or_else(|e| panic!("{stem}: {e}"))
}

fn fixtures() -> Vec<PathBuf> {
    let mut paths = Vec::new();
    for dir in [fixtures_dir(), engine_dir()] {
        for entry in std::fs::read_dir(&dir).unwrap_or_else(|e| panic!("{}: {e}", dir.display())) {
            let path = entry.expect("fixture directory entry").path();
            let name = path.file_name().map(|n| n.to_string_lossy().into_owned());
            if name.is_some_and(|n| n.ends_with(".ll") && !n.ends_with(".air.ll")) {
                paths.push(path);
            }
        }
    }
    paths.sort();
    paths
}

/// Every image binding of every texture slot of every committed fixture's entry points, linked
/// alone on each profile, declares the image the slot's view shows.
#[test]
fn every_texture_binding_declares_the_image_its_view_shows() {
    let devices: Vec<(&str, Device)> = PROFILES.iter().map(|&s| (s, device(s))).collect();
    let budget = Budget::default();
    let mut modules = 0usize;
    let mut checked: BTreeMap<Shape, usize> = BTreeMap::new();
    let mut wrong = Vec::new();
    for path in fixtures() {
        let label = path
            .strip_prefix(fixtures_dir())
            .unwrap_or(&path)
            .display()
            .to_string();
        let bitcode = match llvm_assemble(&path) {
            Ok(bitcode) => bitcode,
            Err(AssembleError::Missing(why)) => {
                eprintln!("SKIPPED: {why}");
                return;
            }
            Err(AssembleError::Rejected(why)) => {
                eprintln!("{label}: this llvm-as rejects it, so it is not checked: {why}");
                continue;
            }
        };
        let Ok(library) = Library::from_air(&bitcode) else {
            continue;
        };
        for (stem, device) in &devices {
            for (index, info) in (0u32..).zip(library.functions()) {
                if info.stage == FunctionStage::Visible {
                    continue;
                }
                let Ok(function) = library.function_at(index) else {
                    continue;
                };
                let function = FunctionDescriptor::new(function);
                let Ok(compiled) =
                    device.compile_function(&function, &LoweringInputs::new(), &budget)
                else {
                    continue;
                };
                let Ok((table, words)) = link_standalone(compiled.artifact(), device.profile())
                else {
                    continue;
                };
                modules += 1;
                let artifact = compiled.artifact();
                let declared = images(&words);
                for (resource, bound) in &table.bindings {
                    let Some((texture, sampled)) = texture_slot(resource) else {
                        continue;
                    };
                    let slot = SlotRef {
                        stage: resource.stage,
                        class: SlotClass::Texture,
                        index: texture,
                    };
                    let Some(SlotInterface::Texture(t)) = artifact.interface.slots.get(&slot)
                    else {
                        panic!(
                            "{label} {} on {stem}: {resource:?} is bound but {slot:?} is no \
                             texture slot",
                            info.name
                        );
                    };
                    let cube_as_array = !sampled
                        && artifact
                            .facts
                            .contains(&RuntimeFact::ImageViewShape { slot });
                    let want = expected(t.texture_type, cube_as_array);
                    let Some(shapes) = declared.get(bound) else {
                        panic!(
                            "{label} {} on {stem}: {resource:?} at {bound:?} declares no image",
                            info.name
                        );
                    };
                    for &shape in shapes {
                        *checked.entry(shape).or_default() += 1;
                        if shape != want {
                            wrong.push(format!(
                                "{label} {} on {stem}: {resource:?} at {bound:?} declares \
                                 (Dim, Arrayed, MS) {shape:?}, but its {:?} slot{} is viewed as \
                                 {want:?}",
                                info.name,
                                t.texture_type,
                                if cube_as_array {
                                    " with ImageViewShape"
                                } else {
                                    ""
                                },
                            ));
                        }
                    }
                }
            }
        }
    }
    eprintln!("{modules} modules checked; image bindings by (Dim, Arrayed, MS): {checked:?}");
    assert!(
        wrong.is_empty(),
        "{} image bindings declare an image their view does not show:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
    let total: usize = checked.values().sum();
    assert!(
        modules >= 1000 && total >= 500 && checked.len() >= 6,
        "the sweep checked far fewer than expected: {total} image bindings of {} shapes in \
         {modules} modules",
        checked.len()
    );
}
