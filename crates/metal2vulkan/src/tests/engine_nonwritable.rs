//! A `NonWritable` storage resource is one the module cannot write: the engine's guard over
//! its final modules.
//!
//! Vulkan reads the absence of the decoration as a demand: a graphics-stage module that declares
//! an undecorated storage buffer, storage image, or storage texel buffer requires its consumer to
//! enable `fragmentStoresAndAtomics` or `vertexPipelineStoresAndAtomics`, whether or not any store
//! exists (`VUID-RuntimeSpirv-NonWritable-06340`, `-06341`). The decoration is also an assertion
//! the driver may act on (read-only memory, a skipped barrier), and `spirv-val` does not check it
//! against the module's writes, so a wrong one is silent all the way to the device.
//!
//! This file supplies the check `spirv-val` does not, as `tests/nonwritable_covers_the_module.rs`
//! did for the old engine until 0.2.0 deleted both: an independent walk of the final words that roots every derived
//! pointer and image at the variables it may come from, and fails when a store, a copy, an atomic,
//! or an image write may land on a decorated variable, or on another variable of its descriptor
//! binding (views of one binding are one memory). A written pointer or image the walk cannot root
//! counts as a write to every variable of its storage class. On a profile without the two features
//! it also requires the other direction, which the VUIDs make a validity rule there: every storage
//! resource of a compiled graphics module is decorated.
//!
//! The same walk also holds the access a compiled function reports for each slot to what its
//! module does, as `tests/reflection_access_covers_the_module.rs` did for the old engine's
//! reflection until 0.2.0 deleted both. A consumer reads the reported access to decide what to stage, what to read back
//! and which barriers to place, so an access narrower than the module's is a missing barrier, or a
//! write to memory placed read-only. The interface takes it from what the body does
//! (`memory::plan`), with what loaded pointers may do to a slot whose address escapes, and from the
//! stores emission merged into their words; the sweep checks it against the final words: a buffer
//! slot whose variables a store, copy or atomic may write reports `write`, one a load, copy or
//! atomic may read reports `read` (every atomic read-modify-write reads its word, a sub-word
//! store's two included), and a texture slot reports `write` for its image writes, `read` for its
//! storage image reads, and `read` or `sample` for its samples, fetches and gathers (a sampled
//! texture's `read` is a fetch, and so is a bicubic sample). The other direction is not checked: a
//! reported access wider than the module's is legal.
//!
//! The authored inputs separate what AIR declares from what the body does. The sweeps compile
//! every entry point of every committed opaque-pointer fixture on three committed profiles (the
//! access sweep on four).

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::device::requirements::VkFeature;
use crate::device::{Device, DeviceDescription, DevicePolicy};
use crate::library::{FunctionStage, Library};
use crate::metal::format::storage_image_choice;
use crate::metal::PixelFormat;
use crate::pipeline::sym::{ResourceKind, ResourceSlot};
use crate::pipeline::testing::{
    engine_dir, fixtures_dir, link_standalone, llvm_assemble, AssembleError,
};
use crate::pipeline::{
    AccessSet, ComputePipelineDescriptor, FunctionDescriptor, LoweringInputs, SlotClass,
    SlotInterface, SlotRef, Stage, StageSource,
};
use crate::Budget;

// ---- The independent walk ----

const OP_ENTRY_POINT: u32 = 15;
const OP_TYPE_IMAGE: u32 = 25;
const OP_TYPE_SAMPLED_IMAGE: u32 = 27;
const OP_TYPE_ARRAY: u32 = 28;
const OP_TYPE_RUNTIME_ARRAY: u32 = 29;
const OP_TYPE_POINTER: u32 = 32;
const OP_FUNCTION: u32 = 54;
const OP_FUNCTION_PARAMETER: u32 = 55;
const OP_UNDEF: u32 = 1;
const OP_FUNCTION_CALL: u32 = 57;
const OP_VARIABLE: u32 = 59;
const OP_IMAGE_TEXEL_POINTER: u32 = 60;
const OP_LOAD: u32 = 61;
/// `OpImageSampleImplicitLod` through `OpImageSampleProjDrefExplicitLod`, `OpImageFetch`,
/// `OpImageGather` and `OpImageDrefGather`, and their sparse forms: each reads texels through a
/// sampled image (operand 2), as a sample or a sampled texture's `read` does (a bicubic sample is
/// fetches).
const OP_IMAGE_SAMPLES: [u32; 22] = [
    87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314,
    315,
];
/// `OpImageRead` and `OpImageSparseRead`: each reads texels of its storage image (operand 2).
const OP_IMAGE_READS: [u32; 2] = [98, 320];
/// `OpAtomicLoad`: reads its pointer (operand 2).
const OP_ATOMIC_LOAD: u32 = 227;
const OP_STORE: u32 = 62;
const OP_COPY_MEMORY: u32 = 63;
const OP_COPY_MEMORY_SIZED: u32 = 64;
const OP_ACCESS_CHAIN: u32 = 65;
const OP_IN_BOUNDS_ACCESS_CHAIN: u32 = 66;
const OP_PTR_ACCESS_CHAIN: u32 = 67;
const OP_IN_BOUNDS_PTR_ACCESS_CHAIN: u32 = 70;
const OP_DECORATE: u32 = 71;
const OP_COPY_OBJECT: u32 = 83;
const OP_SAMPLED_IMAGE: u32 = 86;
const OP_IMAGE_WRITE: u32 = 99;
const OP_IMAGE: u32 = 100;
const OP_CONVERT_U_TO_PTR: u32 = 120;
const OP_BITCAST: u32 = 124;
const OP_SELECT: u32 = 169;
const OP_ATOMIC_STORE: u32 = 228;
const OP_PHI: u32 = 245;
const OP_ATOMIC_FLAG_TEST_AND_SET: u32 = 318;
const OP_ATOMIC_FLAG_CLEAR: u32 = 319;
const OP_COPY_LOGICAL: u32 = 400;
/// `OpAtomicExchange` through `OpAtomicXor`: each reads and writes its pointer (operand 2).
const OP_ATOMIC_READ_MODIFY_WRITE: std::ops::RangeInclusive<u32> = 229..=242;
/// `OpAtomicFMinEXT`, `OpAtomicFMaxEXT`, `OpAtomicFAddEXT`: the same layout.
const OP_ATOMIC_FLOAT: [u32; 3] = [5614, 5615, 6035];

const DECORATION_BUFFER_BLOCK: u32 = 3;
const DECORATION_NON_WRITABLE: u32 = 24;
const DECORATION_BINDING: u32 = 33;
const DECORATION_DESCRIPTOR_SET: u32 = 34;

const STORAGE_CLASS_UNIFORM_CONSTANT: u32 = 0;
const STORAGE_CLASS_UNIFORM: u32 = 2;
const STORAGE_CLASS_STORAGE_BUFFER: u32 = 12;

/// `Vertex`, `TessellationControl`, `TessellationEvaluation`, `Geometry`, `Fragment`: the stages
/// the two VUIDs name.
const GRAPHICS_MODELS: std::ops::RangeInclusive<u32> = 0..=4;

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

/// What the walk learns of one module.
#[derive(Debug, Default)]
struct Walk {
    /// Whether an entry point is a graphics stage.
    graphics: bool,
    /// The storage buffers, storage images and storage texel buffers, by storage class.
    storage: BTreeMap<u32, u32>,
    /// The variables decorated `NonWritable`.
    decorated: BTreeSet<u32>,
    /// Each variable's `(set, binding)`.
    binding: BTreeMap<u32, (u32, u32)>,
    /// The variables a write may land on.
    written: BTreeSet<u32>,
    /// The buffer variables a load, an atomic or a copy's source may read.
    read: BTreeSet<u32>,
    /// The storage image variables an image read may read.
    image_read: BTreeSet<u32>,
    /// The image variables a sample, fetch or gather may read.
    image_sampled: BTreeSet<u32>,
}

/// What an access does to the memory or image it reaches.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Effect {
    /// A store, a copy's target, an atomic: `true` for an image write.
    Write { image: bool },
    /// A load, an atomic, a copy's source.
    Read,
    /// A storage image read.
    ImageRead,
    /// A sample, fetch or gather.
    Sample,
}

/// Walk `words` (see the file docs). Panics on a written value whose origin the walk does not
/// model, rather than guessing.
fn walk(words: &[u32]) -> Walk {
    let instructions = instructions(words);
    let mut out = Walk::default();
    // Types: pointer (class, pointee), arrays' elements, images' `Sampled` operand.
    let mut pointer: BTreeMap<u32, (u32, u32)> = BTreeMap::new();
    let mut element: BTreeMap<u32, u32> = BTreeMap::new();
    let mut image_sampled: BTreeMap<u32, u32> = BTreeMap::new();
    let mut sampled_image: BTreeSet<u32> = BTreeSet::new();
    let mut buffer_block: BTreeSet<u32> = BTreeSet::new();
    // Values: each result id's type and defining opcode, what it derives from, the parameters of
    // each function.
    let mut type_of: BTreeMap<u32, u32> = BTreeMap::new();
    let mut opcode_of: BTreeMap<u32, u32> = BTreeMap::new();
    let mut from: BTreeMap<u32, Vec<u32>> = BTreeMap::new();
    let mut parameters: BTreeMap<u32, Vec<u32>> = BTreeMap::new();
    let mut calls: Vec<(u32, Vec<u32>)> = Vec::new();
    let mut variables: BTreeMap<u32, u32> = BTreeMap::new();
    let mut targets: Vec<(u32, Effect)> = Vec::new();
    let mut function = None;
    for &(opcode, ops) in &instructions {
        match opcode {
            OP_ENTRY_POINT => out.graphics |= GRAPHICS_MODELS.contains(&ops[0]),
            OP_TYPE_POINTER => {
                pointer.insert(ops[0], (ops[1], ops[2]));
            }
            OP_TYPE_ARRAY | OP_TYPE_RUNTIME_ARRAY => {
                element.insert(ops[0], ops[1]);
            }
            OP_TYPE_IMAGE => {
                image_sampled.insert(ops[0], ops[6]);
            }
            OP_TYPE_SAMPLED_IMAGE => {
                sampled_image.insert(ops[0]);
            }
            OP_DECORATE => match ops[1] {
                DECORATION_NON_WRITABLE => {
                    out.decorated.insert(ops[0]);
                }
                DECORATION_BUFFER_BLOCK => {
                    buffer_block.insert(ops[0]);
                }
                DECORATION_DESCRIPTOR_SET => {
                    out.binding.entry(ops[0]).or_default().0 = ops[2];
                }
                DECORATION_BINDING => {
                    out.binding.entry(ops[0]).or_default().1 = ops[2];
                }
                _ => {}
            },
            OP_FUNCTION => function = Some(ops[1]),
            OP_FUNCTION_PARAMETER => {
                let f = function.expect("a parameter outside a function");
                parameters.entry(f).or_default().push(ops[1]);
            }
            OP_FUNCTION_CALL => calls.push((ops[2], ops[3..].to_vec())),
            OP_VARIABLE => {
                variables.insert(ops[1], ops[2]);
            }
            OP_STORE | OP_ATOMIC_STORE | OP_ATOMIC_FLAG_CLEAR => {
                targets.push((ops[0], Effect::Write { image: false }))
            }
            OP_COPY_MEMORY | OP_COPY_MEMORY_SIZED => {
                targets.push((ops[0], Effect::Write { image: false }));
                targets.push((ops[1], Effect::Read));
            }
            OP_IMAGE_WRITE => targets.push((ops[0], Effect::Write { image: true })),
            OP_LOAD | OP_ATOMIC_LOAD => targets.push((ops[2], Effect::Read)),
            _ if OP_ATOMIC_READ_MODIFY_WRITE.contains(&opcode)
                || OP_ATOMIC_FLOAT.contains(&opcode)
                || opcode == OP_ATOMIC_FLAG_TEST_AND_SET =>
            {
                targets.push((ops[2], Effect::Write { image: false }));
                targets.push((ops[2], Effect::Read));
            }
            _ if OP_IMAGE_READS.contains(&opcode) => targets.push((ops[2], Effect::ImageRead)),
            _ if OP_IMAGE_SAMPLES.contains(&opcode) => targets.push((ops[2], Effect::Sample)),
            _ => {}
        }
        // The value instructions a pointer or an image passes through, each `[type, result, ..]`.
        let derives: Option<Vec<u32>> = match opcode {
            OP_ACCESS_CHAIN
            | OP_IN_BOUNDS_ACCESS_CHAIN
            | OP_PTR_ACCESS_CHAIN
            | OP_IN_BOUNDS_PTR_ACCESS_CHAIN
            | OP_COPY_OBJECT
            | OP_COPY_LOGICAL
            | OP_BITCAST
            | OP_IMAGE_TEXEL_POINTER
            | OP_SAMPLED_IMAGE
            | OP_IMAGE => Some(vec![ops[2]]),
            OP_SELECT => Some(vec![ops[3], ops[4]]),
            OP_PHI => Some(ops[2..].iter().step_by(2).copied().collect()),
            // Loading an image object from its variable. A loaded pointer is not followed: its
            // origin is whatever was stored, so a write through one is unrooted.
            OP_LOAD => Some(vec![ops[2]]),
            _ => None,
        };
        if let Some(sources) = derives {
            type_of.insert(ops[1], ops[0]);
            opcode_of.insert(ops[1], opcode);
            from.insert(ops[1], sources);
        } else if matches!(
            opcode,
            OP_VARIABLE | OP_FUNCTION_PARAMETER | OP_FUNCTION_CALL | OP_CONVERT_U_TO_PTR | OP_UNDEF
        ) {
            // Values with a type and no source the walk follows: a written one is unrooted.
            type_of.insert(ops[1], ops[0]);
            opcode_of.insert(ops[1], opcode);
        }
    }
    let is_image = |ty: u32| image_sampled.contains_key(&ty) || sampled_image.contains(&ty);
    // A load follows its pointer only when it loads an image object.
    from.retain(|id, _| {
        opcode_of.get(id) != Some(&OP_LOAD) || type_of.get(id).is_some_and(|t| is_image(*t))
    });
    for (callee, args) in calls {
        for (parameter, arg) in parameters.get(&callee).into_iter().flatten().zip(args) {
            from.entry(*parameter).or_default().push(arg);
        }
    }
    // The storage resources.
    let innermost = |mut ty: u32| {
        while let Some(&e) = element.get(&ty) {
            ty = e;
        }
        ty
    };
    for (&var, &class) in &variables {
        let Some(&(_, pointee)) = type_of.get(&var).and_then(|t| pointer.get(t)) else {
            continue;
        };
        let storage = match class {
            STORAGE_CLASS_STORAGE_BUFFER => true,
            STORAGE_CLASS_UNIFORM => buffer_block.contains(&innermost(pointee)),
            STORAGE_CLASS_UNIFORM_CONSTANT => image_sampled.get(&innermost(pointee)) == Some(&2),
            _ => false,
        };
        if storage {
            out.storage.insert(var, class);
        }
    }
    // The buffer variables (storage and uniform) and the image variables, by storage class: what
    // an unrooted access may reach.
    let buffers: BTreeMap<u32, u32> = variables
        .iter()
        .filter(|&(_, &class)| {
            matches!(class, STORAGE_CLASS_STORAGE_BUFFER | STORAGE_CLASS_UNIFORM)
        })
        .map(|(&var, &class)| (var, class))
        .collect();
    let images: BTreeSet<u32> = variables
        .iter()
        .filter(|&(&var, &class)| {
            class == STORAGE_CLASS_UNIFORM_CONSTANT
                && type_of
                    .get(&var)
                    .and_then(|t| pointer.get(t))
                    .is_some_and(|&(_, pointee)| is_image(innermost(pointee)))
        })
        .map(|(&var, _)| var)
        .collect();
    // Root each access.
    for (target, effect) in targets {
        let mut roots = BTreeSet::new();
        let mut unrooted = Vec::new();
        let mut stack = vec![target];
        let mut seen = BTreeSet::new();
        while let Some(id) = stack.pop() {
            if !seen.insert(id) {
                continue;
            }
            if variables.contains_key(&id) {
                roots.insert(id);
            } else if let Some(sources) = from.get(&id).filter(|s| !s.is_empty()) {
                stack.extend(sources);
            } else {
                unrooted.push(id);
            }
        }
        let reached = match effect {
            Effect::Write { .. } => None,
            Effect::Read => Some((&mut out.read, false)),
            Effect::ImageRead => Some((&mut out.image_read, true)),
            Effect::Sample => Some((&mut out.image_sampled, true)),
        };
        if let Some((set, image)) = reached {
            if image {
                set.extend(roots.iter().filter(|var| images.contains(var)));
            } else {
                set.extend(roots.iter().filter(|var| buffers.contains_key(var)));
            }
            for id in unrooted {
                // An unrooted access reaches every variable of its kind: an image access every
                // image, a load every buffer of its pointer's storage class.
                if image {
                    set.extend(images.iter().copied());
                } else if let Some(&(class, _)) = type_of.get(&id).and_then(|ty| pointer.get(ty)) {
                    set.extend(
                        buffers
                            .iter()
                            .filter(|&(_, &c)| c == class)
                            .map(|(&var, _)| var),
                    );
                }
            }
            continue;
        }
        let image_write = effect == (Effect::Write { image: true });
        out.written.extend(roots);
        for id in unrooted {
            // An unrooted written value writes every storage resource of its kind.
            let class = if image_write {
                Some(STORAGE_CLASS_UNIFORM_CONSTANT)
            } else {
                let ty = type_of.get(&id).unwrap_or_else(|| {
                    panic!(
                        "a write through %{id}, whose defining instruction the walk does not \
                         model:\n{}",
                        instructions
                            .iter()
                            .find(|(_, ops)| ops.get(1) == Some(&id))
                            .map_or("not found".to_owned(), |(op, ops)| format!("{op} {ops:?}"))
                    )
                });
                pointer.get(ty).map(|&(class, _)| class)
            };
            if let Some(class) = class {
                out.written.extend(
                    out.storage
                        .iter()
                        .filter(|&(_, &c)| c == class)
                        .map(|(&var, _)| var),
                );
            }
        }
    }
    out
}

/// What [`check`] counted in one module.
#[derive(Debug, Default)]
struct Counts {
    graphics: usize,
    decorated: usize,
    written_storage: usize,
    /// Graphics modules with a storage resource, compiled for a profile without the stores.
    without_stores: usize,
}

/// The two rules over one module's words (see the file docs). `stores` says whether the profile
/// enables the graphics stages' stores and atomics.
fn check(label: &str, words: &[u32], stores: bool) -> Counts {
    let w = walk(words);
    // Every variable of a written variable's binding is written.
    let written_bindings: BTreeSet<(u32, u32)> = w
        .written
        .iter()
        .filter_map(|var| w.binding.get(var).copied())
        .collect();
    for &var in &w.decorated {
        let shares = w
            .binding
            .get(&var)
            .is_some_and(|b| written_bindings.contains(b));
        assert!(
            !w.written.contains(&var) && !shares,
            "{label}: %{var} is decorated NonWritable, but a write may land on it or on another \
             variable of its binding {:?}",
            w.binding.get(&var)
        );
    }
    if w.graphics && !stores {
        for &var in w.storage.keys() {
            assert!(
                w.decorated.contains(&var),
                "{label}: a graphics module compiled for a profile without the graphics stages' \
                 stores and atomics declares the storage resource %{var} (binding {:?}) without \
                 NonWritable, which VUID-RuntimeSpirv-NonWritable-06340/06341 forbid",
                w.binding.get(&var)
            );
        }
    }
    Counts {
        graphics: usize::from(w.graphics),
        decorated: w.decorated.len(),
        written_storage: w
            .written
            .intersection(&w.storage.keys().copied().collect())
            .count(),
        without_stores: usize::from(w.graphics && !stores && !w.storage.is_empty()),
    }
}

// ---- Compiling ----

/// The committed profiles every input compiles on, and whether each enables
/// `fragmentStoresAndAtomics` and `vertexPipelineStoresAndAtomics` (their JSON): a device that
/// reaches buffers by device address, one that binds them, and the Vulkan 1.2 floor, which binds
/// them and has neither feature.
const PROFILES: [(&str, bool); 3] = [
    ("nvidia-rtx5080-laptop", true),
    ("lunarg-desktop-baseline-2023", true),
    ("vk12-floor", false),
];

fn description(stem: &str) -> DeviceDescription {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles")
        .join(format!("{stem}.json"));
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    DeviceDescription::from_khronos_json(&text, None).unwrap_or_else(|e| panic!("{stem}: {e}"))
}

fn device(stem: &str) -> Device {
    Device::from_queries(description(stem), DevicePolicy::default())
        .unwrap_or_else(|e| panic!("{stem}: {e}"))
}

/// The name [`floor_with_16_bit`]'s device goes by in the access sweep.
const FLOOR_WITH_16_BIT: &str = "vk12-floor+16-bit";

/// `vk12-floor` with `shaderInt16` and `storageBuffer16BitAccess`: a slot's views hold the 16-bit
/// unit and not the 8-bit one.
fn floor_with_16_bit() -> Device {
    let mut desc = description("vk12-floor");
    desc.features
        .extend([VkFeature::ShaderInt16, VkFeature::StorageBuffer16BitAccess]);
    Device::from_queries(desc, DevicePolicy::default())
        .unwrap_or_else(|e| panic!("{FLOOR_WITH_16_BIT}: {e}"))
}

/// `llvm-as` of `text`, through a scratch file that is removed at once.
fn assemble(text: &str) -> Result<Vec<u8>, AssembleError> {
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    let path = std::env::temp_dir().join(format!(
        "m2v-nonwritable-{}-{}.ll",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::write(&path, text).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let assembled = llvm_assemble(&path);
    std::fs::remove_file(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    assembled
}

/// Each entry point of `library` that compiles on `device`, with its final words. A kernel becomes
/// a compute pipeline, any other stage is linked alone (as `src/tests/golden_hashes.rs` does).
/// Refusals are skipped: the suites that own those inputs report on them.
fn compiled(library: &Library, device: &Device) -> Vec<(String, Vec<u32>)> {
    let budget = Budget::default();
    let mut out = Vec::new();
    for (index, info) in (0u32..).zip(library.functions()) {
        if info.stage == FunctionStage::Visible {
            continue;
        }
        let Ok(function) = library.function_at(index) else {
            continue;
        };
        let function = FunctionDescriptor::new(function);
        let Ok(compiled) = device.compile_function(&function, &LoweringInputs::new(), &budget)
        else {
            continue;
        };
        let module = if info.stage == FunctionStage::Kernel {
            let descriptor = ComputePipelineDescriptor::new(StageSource::Compiled(compiled));
            device
                .compile_compute_pipeline(&descriptor, &budget)
                .ok()
                .map(|p| p.stage.spirv.to_vec())
        } else {
            link_standalone(compiled.artifact(), device.profile())
                .ok()
                .map(|(_, words)| words)
        };
        if let Some(words) = module {
            out.push((info.name.to_owned(), words));
        }
    }
    out
}

// ---- Authored inputs ----

/// The `(set, binding)` link gave each resource of a stage linked alone.
type Bindings = BTreeMap<ResourceSlot, (u32, u32)>;

/// The module header and trailer of the authored inputs: Metal 4's target and the AIR version
/// metadata a compiled `.air` carries.
fn authored(body: &str) -> String {
    format!(
        r#"target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

{body}
!air.version = !{{!93}}
!air.language_version = !{{!94}}
!93 = !{{i32 2, i32 8, i32 0}}
!94 = !{{!"Metal", i32 4, i32 0, i32 0}}
"#
    )
}

/// A fragment function whose three buffers separate what AIR declares from what the body does.
///
/// `read_only` is declared `air.read` and only loaded. `declared_read` is declared `air.read` and
/// stored through anyway (AIR's declared access is not a guarantee about the body), the case a
/// metadata-driven decoration would get wrong. `written` is declared `air.read_write` and stored.
const MIXED_BUFFER_ACCESS: &str = r#"define <4 x float> @mixed_buffer_access(<4 x float> noundef %position, ptr addrspace(1) noundef %read_only, ptr addrspace(1) noundef %declared_read, ptr addrspace(1) noundef %written) local_unnamed_addr #0 {
  %v = load i32, ptr addrspace(1) %read_only, align 4
  store i32 %v, ptr addrspace(1) %declared_read, align 4
  store i32 %v, ptr addrspace(1) %written, align 4
  ret <4 x float> %position
}

attributes #0 = { nounwind }

!air.fragment = !{!0}
!0 = !{ptr @mixed_buffer_access, !1, !3}
!1 = !{!2}
!2 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}
!3 = !{!4, !5, !6, !7}
!4 = !{i32 0, !"air.position", !"air.center", !"air.no_perspective", !"air.arg_type_name", !"float4", !"air.arg_name", !"position"}
!5 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"read_only"}
!6 = !{i32 2, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"declared_read"}
!7 = !{i32 3, !"air.buffer", !"air.location_index", i32 2, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"written"}
"#;

/// A fragment function whose three storage textures separate what AIR declares from what the body
/// does. Only `access::write` and `access::read_write` make a storage image, so the split is drawn
/// inside `read_write`: `never_written` is declared `read_write` and only read,
/// `read_then_written` is declared `read_write` and written, and `written` is declared `write`.
const MIXED_IMAGE_ACCESS: &str = r#"define <4 x float> @mixed_image_access(<4 x float> noundef %position, ptr addrspace(1) %never_written, ptr addrspace(1) %read_then_written, ptr addrspace(1) %written) local_unnamed_addr #0 {
  %s = tail call ptr addrspace(2) @air.get_read_sampler() #1
  %a = tail call { <4 x float>, i8 } @air.read_texture_2d.v4f32(ptr addrspace(1) readonly captures(none) %never_written, ptr addrspace(2) %s, <2 x i32> zeroinitializer, <2 x i32> zeroinitializer, i32 0, i32 3) #2
  %av = extractvalue { <4 x float>, i8 } %a, 0
  %b = tail call { <4 x float>, i8 } @air.read_texture_2d.v4f32(ptr addrspace(1) readonly captures(none) %read_then_written, ptr addrspace(2) %s, <2 x i32> zeroinitializer, <2 x i32> zeroinitializer, i32 0, i32 3) #2
  %bv = extractvalue { <4 x float>, i8 } %b, 0
  %sum = fadd <4 x float> %av, %bv
  tail call void @air.write_texture_2d.v4f32(ptr addrspace(1) captures(none) %read_then_written, <2 x i32> zeroinitializer, <4 x float> %sum, i32 0, i32 3) #3
  tail call void @air.write_texture_2d.v4f32(ptr addrspace(1) captures(none) %written, <2 x i32> zeroinitializer, <4 x float> %av, i32 0, i32 3) #3
  ret <4 x float> %sum
}

declare ptr addrspace(2) @air.get_read_sampler() local_unnamed_addr #1
declare { <4 x float>, i8 } @air.read_texture_2d.v4f32(ptr addrspace(1) readonly captures(none), ptr addrspace(2), <2 x i32>, <2 x i32>, i32, i32) local_unnamed_addr #2
declare void @air.write_texture_2d.v4f32(ptr addrspace(1) captures(none), <2 x i32>, <4 x float>, i32, i32) local_unnamed_addr #3

attributes #0 = { nounwind }
attributes #1 = { nounwind willreturn memory(inaccessiblemem: read) }
attributes #2 = { nounwind willreturn memory(argmem: read) }
attributes #3 = { nounwind willreturn memory(argmem: readwrite) }

!air.fragment = !{!0}
!0 = !{ptr @mixed_image_access, !1, !3}
!1 = !{!2}
!2 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}
!3 = !{!4, !5, !6, !7}
!4 = !{i32 0, !"air.position", !"air.center", !"air.no_perspective", !"air.arg_type_name", !"float4", !"air.arg_name", !"position"}
!5 = !{i32 1, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.arg_type_name", !"texture2d<float, read_write>", !"air.arg_name", !"never_written"}
!6 = !{i32 2, !"air.texture", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.arg_type_name", !"texture2d<float, read_write>", !"air.arg_name", !"read_then_written"}
!7 = !{i32 3, !"air.texture", !"air.location_index", i32 2, i32 1, !"air.write", !"air.arg_type_name", !"texture2d<float, write>", !"air.arg_name", !"written"}
"#;

/// A fragment function that only reads its buffer and its `read_write` texture: on a device
/// without `fragmentStoresAndAtomics` it must compile, with both decorated.
const READ_ONLY: &str = r#"define <4 x float> @read_only(<4 x float> noundef %position, ptr addrspace(1) noundef %buffer, ptr addrspace(1) %texture) local_unnamed_addr #0 {
  %s = tail call ptr addrspace(2) @air.get_read_sampler() #1
  %t = tail call { <4 x float>, i8 } @air.read_texture_2d.v4f32(ptr addrspace(1) readonly captures(none) %texture, ptr addrspace(2) %s, <2 x i32> zeroinitializer, <2 x i32> zeroinitializer, i32 0, i32 3) #2
  %tv = extractvalue { <4 x float>, i8 } %t, 0
  %b = load float, ptr addrspace(1) %buffer, align 4
  %bv = insertelement <4 x float> %tv, float %b, i64 0
  ret <4 x float> %bv
}

declare ptr addrspace(2) @air.get_read_sampler() local_unnamed_addr #1
declare { <4 x float>, i8 } @air.read_texture_2d.v4f32(ptr addrspace(1) readonly captures(none), ptr addrspace(2), <2 x i32>, <2 x i32>, i32, i32) local_unnamed_addr #2

attributes #0 = { nounwind }
attributes #1 = { nounwind willreturn memory(inaccessiblemem: read) }
attributes #2 = { nounwind willreturn memory(argmem: read) }

!air.fragment = !{!0}
!0 = !{ptr @read_only, !1, !3}
!1 = !{!2}
!2 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}
!3 = !{!4, !5, !6}
!4 = !{i32 0, !"air.position", !"air.center", !"air.no_perspective", !"air.arg_type_name", !"float4", !"air.arg_name", !"position"}
!5 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"buffer"}
!6 = !{i32 2, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.arg_type_name", !"texture2d<float, read_write>", !"air.arg_name", !"texture"}
"#;

/// The one entry point of `body` compiled on profile `stem`, checked by both rules, with the
/// `(set, binding)` of each of its resources. Each fragment texture slot in `storage` is a storage
/// texture of `R32Float`, under the choice `storage_image_choice` makes on the profile. `None` when
/// `llvm-as` is not installed.
fn authored_module(
    label: &str,
    body: &str,
    stem: &str,
    storage: &[u32],
) -> Option<(Vec<u32>, Bindings)> {
    let bitcode = match assemble(&authored(body)) {
        Ok(bitcode) => bitcode,
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            return None;
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejects {label}: {why}"),
    };
    let library = Library::from_air(&bitcode).unwrap_or_else(|e| panic!("{label}: {e}"));
    let device = device(stem);
    let function = FunctionDescriptor::new(library.function_at(0).expect("the entry point"));
    let mut inputs = LoweringInputs::new();
    for &index in storage {
        let choice = storage_image_choice(PixelFormat::R32Float, device.profile())
            .unwrap_or_else(|e| panic!("{label} on {stem}: R32Float storage: {e}"));
        let slot = SlotRef {
            stage: Stage::Fragment,
            class: SlotClass::Texture,
            index,
        };
        inputs.storage_images.insert(slot, choice);
    }
    let compiled = device
        .compile_function(&function, &inputs, &Budget::default())
        .unwrap_or_else(|e| panic!("{label} on {stem}: {e}"));
    let (table, words) = link_standalone(compiled.artifact(), device.profile())
        .unwrap_or_else(|e| panic!("{label} on {stem}: link: {e}"));
    let stores = PROFILES
        .iter()
        .find(|(s, _)| *s == stem)
        .map(|&(_, stores)| stores)
        .expect("a listed profile");
    check(label, &words, stores);
    Some((words, table.bindings))
}

/// The `(set, binding)`s of the variables of `words` decorated `NonWritable`.
fn decorated_bindings(words: &[u32]) -> BTreeSet<(u32, u32)> {
    let w = walk(words);
    w.decorated
        .iter()
        .map(|var| {
            *w.binding
                .get(var)
                .unwrap_or_else(|| panic!("%{var} is decorated NonWritable and has no binding"))
        })
        .collect()
}

/// `(set, binding)` of the fragment stage's resource `kind` at Metal index `index`.
fn bound(bindings: &Bindings, kind: ResourceKind, index: u32) -> (u32, u32) {
    let slot = ResourceSlot {
        stage: Stage::Fragment,
        kind,
        index,
    };
    *bindings
        .get(&slot)
        .unwrap_or_else(|| panic!("no binding for {slot:?} in {bindings:?}"))
}

/// The profile that binds buffers and enables the fragment stage's stores.
const BOUND_WITH_STORES: &str = "lunarg-desktop-baseline-2023";

#[test]
fn only_the_buffer_the_body_never_writes_is_decorated() {
    let Some((words, bindings)) = authored_module(
        "the mixed-buffer-access fragment function",
        MIXED_BUFFER_ACCESS,
        BOUND_WITH_STORES,
        &[],
    ) else {
        return;
    };
    let decorated = decorated_bindings(&words);
    for (index, name, expected) in [
        (0, "read_only", true),
        (1, "declared_read", false),
        (2, "written", false),
    ] {
        let binding = bound(&bindings, ResourceKind::Buffer, index);
        assert_eq!(
            decorated.contains(&binding),
            expected,
            "buffer {index} ({name}) at {binding:?}: decorated NonWritable is {}, expected {expected}",
            decorated.contains(&binding)
        );
    }
}

#[test]
fn only_the_storage_image_the_body_never_writes_is_decorated() {
    let Some((words, bindings)) = authored_module(
        "the mixed-image-access fragment function",
        MIXED_IMAGE_ACCESS,
        BOUND_WITH_STORES,
        &[0, 1, 2],
    ) else {
        return;
    };
    let decorated = decorated_bindings(&words);
    for (index, name, expected) in [
        (0, "never_written", true),
        (1, "read_then_written", false),
        (2, "written", false),
    ] {
        let binding = bound(&bindings, ResourceKind::Texture, index);
        assert_eq!(
            decorated.contains(&binding),
            expected,
            "texture {index} ({name}) at {binding:?}: decorated NonWritable is {}, expected \
             {expected}",
            decorated.contains(&binding)
        );
    }
}

/// The walk is not vacuous: decorating the written buffer of the mixed-access module by hand is
/// caught.
#[test]
fn the_walk_catches_a_decoration_on_a_written_buffer() {
    let Some((words, bindings)) = authored_module(
        "the mixed-buffer-access fragment function",
        MIXED_BUFFER_ACCESS,
        BOUND_WITH_STORES,
        &[],
    ) else {
        return;
    };
    let written = bound(&bindings, ResourceKind::Buffer, 2);
    let w = walk(&words);
    let var = *w
        .binding
        .iter()
        .find(|&(_, &b)| b == written)
        .map(|(var, _)| var)
        .expect("the written buffer's variable");
    // `OpDecorate %var NonWritable`, placed before the module's first decoration.
    let first = instructions(&words)
        .iter()
        .scan(5usize, |at, &(opcode, ops)| {
            let here = *at;
            *at += ops.len() + 1;
            Some((here, opcode))
        })
        .find(|&(_, opcode)| opcode == OP_DECORATE)
        .map(|(at, _)| at)
        .expect("a decoration");
    let mut forged = words[..first].to_vec();
    forged.extend([(3 << 16) | OP_DECORATE, var, DECORATION_NON_WRITABLE]);
    forged.extend_from_slice(&words[first..]);
    let caught = std::panic::catch_unwind(|| check("the forged module", &forged, true));
    assert!(
        caught.is_err(),
        "the walk accepted NonWritable on the buffer the module stores through"
    );
}

#[test]
fn a_fragment_function_that_only_reads_compiles_without_the_stores_feature() {
    let Some((words, bindings)) = authored_module(
        "the read-only fragment function",
        READ_ONLY,
        "vk12-floor",
        &[0],
    ) else {
        return;
    };
    let decorated = decorated_bindings(&words);
    for (kind, name) in [
        (ResourceKind::Buffer, "buffer"),
        (ResourceKind::Texture, "texture"),
    ] {
        let binding = bound(&bindings, kind, 0);
        assert!(
            decorated.contains(&binding),
            "the {name} at {binding:?} is only read, so it is decorated NonWritable"
        );
    }
}

// ---- The fixture sweep ----

/// The committed opaque-pointer fixtures: the `.ll` files of `validation/fixtures/public` and its
/// `engine/` directory, without the typed-pointer `.air.ll` ones (`xcrun metal-as`).
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

/// Every entry point of every committed fixture that compiles, on each profile, keeps both rules.
#[test]
fn every_public_fixture_decorates_only_what_it_never_writes() {
    let devices: Vec<(&str, bool, Device)> = PROFILES
        .iter()
        .map(|&(stem, stores)| (stem, stores, device(stem)))
        .collect();
    let mut total = Counts::default();
    let mut modules = 0usize;
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
        for (stem, stores, device) in &devices {
            for (entry, words) in compiled(&library, device) {
                let counts = check(&format!("{label} {entry} on {stem}"), &words, *stores);
                modules += 1;
                total.graphics += counts.graphics;
                total.decorated += counts.decorated;
                total.written_storage += counts.written_storage;
                total.without_stores += counts.without_stores;
            }
        }
    }
    eprintln!(
        "{modules} modules checked: {} graphics, {} NonWritable variables, {} written storage \
         resources; {} graphics modules with a storage resource on a profile without the stores",
        total.graphics, total.decorated, total.written_storage, total.without_stores
    );
    // 1538 modules, 268 graphics, 48 decorated variables at this guard's introduction; the floors
    // leave room for fixtures to come and go.
    assert!(
        modules >= 1000 && total.graphics >= 200 && total.decorated >= 30,
        "the sweep checked far fewer modules than it did: {modules} modules, {} graphics, {} \
         decorated variables",
        total.graphics,
        total.decorated
    );
}

// ---- The reported access ----

/// What [`check_access`] counted in one module.
#[derive(Debug, Default)]
struct AccessCounts {
    buffers: usize,
    textures: usize,
    written: usize,
    read: usize,
    sampled: usize,
    /// One line per slot whose reported access is narrower than its module's.
    narrower: Vec<String>,
}

/// Every buffer and texture slot `bindings` binds in a module, against what the module does
/// through that binding (see the file docs).
fn check_access(
    label: &str,
    words: &[u32],
    bindings: &Bindings,
    slots: &BTreeMap<SlotRef, SlotInterface>,
) -> AccessCounts {
    let w = walk(words);
    let mut counts = AccessCounts::default();
    for (resource, &bound) in bindings {
        let class = match resource.kind {
            ResourceKind::Buffer => SlotClass::Buffer,
            ResourceKind::Texture | ResourceKind::CombinedTextureSampler(_) => SlotClass::Texture,
            _ => continue,
        };
        let slot = SlotRef {
            stage: resource.stage,
            class,
            index: resource.index,
        };
        let interface = slots
            .get(&slot)
            .unwrap_or_else(|| panic!("{label}: {resource:?} is bound but {slot:?} is not a slot"));
        let vars: BTreeSet<u32> = w
            .binding
            .iter()
            .filter(|&(_, &b)| b == bound)
            .map(|(&var, _)| var)
            .collect();
        let any = |set: &BTreeSet<u32>| !set.is_disjoint(&vars);
        let (written, read, image_read, sampled) = (
            any(&w.written),
            any(&w.read),
            any(&w.image_read),
            any(&w.image_sampled),
        );
        let narrower = match interface {
            SlotInterface::Buffer(b) => {
                counts.buffers += 1;
                (written && !b.access.write) || (read && !b.access.read)
            }
            SlotInterface::Texture(t) => {
                counts.textures += 1;
                (written && !t.access.write)
                    || (image_read && !t.access.read)
                    || (sampled && !t.access.read && !t.access.sample)
            }
            // A sampler has no memory; any other slot form is not this check's.
            _ => continue,
        };
        if narrower {
            counts.narrower.push(format!(
                "{label}: {slot:?} reports {interface:?}, but its module may write it: {written}, \
                 read it: {}, sample it: {sampled} (binding {bound:?})",
                read || image_read
            ));
        }
        counts.written += usize::from(written);
        counts.read += usize::from(read || image_read);
        counts.sampled += usize::from(sampled);
    }
    counts
}

/// The profiles of the access sweep, beside [`floor_with_16_bit`]: each binds buffers and textures
/// by descriptor, so every slot's memory is a variable of the module. They differ in which narrow
/// units a slot's views hold, and so in which stores merge into their words: `vk12-floor` has no
/// 8- or 16-bit storage access, the floor with 16-bit access no `shaderInt8`,
/// `lunarg-desktop-baseline-2023` no `shaderInt16`, and `moltenvk-m3` both widths.
const BOUND_PROFILES: [&str; 3] = ["lunarg-desktop-baseline-2023", "vk12-floor", "moltenvk-m3"];

/// A kernel whose four buffers AIR declares `air.write` (the most storage buffers `vk12-floor`
/// gives a stage): a byte at `gid` of `bytes`, a `ushort` at `gid` of `halves`, a `uint` at `gid`
/// of `words`, and a byte at `gid` through a pointer that selects `bytes` or `other` by the
/// thread's parity (a store with two candidate slots, dispatched by root, `other`'s only store).
/// A byte or `ushort` store merges into its word through the sub-word store's two atomics, which
/// read the word, unless the slot's narrow views hold its unit; the `uint` store never does.
const MERGED_STORES: &str = r#"define void @merged_stores(ptr addrspace(1) noundef %bytes, ptr addrspace(1) noundef %halves, ptr addrspace(1) noundef %words, ptr addrspace(1) noundef %other, i32 noundef %gid) local_unnamed_addr #0 {
  %i = zext i32 %gid to i64
  %b = trunc i32 %gid to i8
  %pb = getelementptr inbounds i8, ptr addrspace(1) %bytes, i64 %i
  store i8 %b, ptr addrspace(1) %pb, align 1
  %h = trunc i32 %gid to i16
  %ph = getelementptr inbounds i16, ptr addrspace(1) %halves, i64 %i
  store i16 %h, ptr addrspace(1) %ph, align 2
  %pw = getelementptr inbounds i32, ptr addrspace(1) %words, i64 %i
  store i32 %gid, ptr addrspace(1) %pw, align 4
  %odd = trunc i32 %gid to i1
  %side = select i1 %odd, ptr addrspace(1) %bytes, ptr addrspace(1) %other
  %ps = getelementptr inbounds i8, ptr addrspace(1) %side, i64 %i
  store i8 %b, ptr addrspace(1) %ps, align 1
  ret void
}

attributes #0 = { nounwind }

!air.kernel = !{!0}
!0 = !{ptr @merged_stores, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5, !6, !7}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_size", i32 1, !"air.arg_type_align_size", i32 1, !"air.arg_type_name", !"uchar", !"air.arg_name", !"bytes"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_size", i32 2, !"air.arg_type_align_size", i32 2, !"air.arg_type_name", !"ushort", !"air.arg_name", !"halves"}
!5 = !{i32 2, !"air.buffer", !"air.location_index", i32 2, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"words"}
!6 = !{i32 3, !"air.buffer", !"air.location_index", i32 3, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_size", i32 1, !"air.arg_type_align_size", i32 1, !"air.arg_type_name", !"uchar", !"air.arg_name", !"other"}
!7 = !{i32 4, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"gid"}
"#;

/// `NonReadable` (SPIR-V `Decoration` 25).
const DECORATION_NON_READABLE: u32 = 25;

/// The variables of `words` decorated `decoration`.
fn decorated_with(words: &[u32], decoration: u32) -> BTreeSet<u32> {
    instructions(words)
        .into_iter()
        .filter(|&(opcode, ops)| opcode == OP_DECORATE && ops.get(1) == Some(&decoration))
        .map(|(_, ops)| ops[0])
        .collect()
}

/// A store merged into its word reads the word, so its slot reports a read, and nothing declares
/// the slot's variable unreadable. On `vk12-floor`, with no narrow storage access, every byte and
/// `ushort` store merges: their slots, `other` through the root dispatch, report `read` and
/// `write`, and the module's atomics read each of them. On `moltenvk-m3`, whose slots hold both
/// narrow units, each such store is one plain unit store and every slot reports `write` alone, as
/// the `uint` slot does on both.
#[test]
fn a_store_merged_into_its_word_reports_a_read_of_its_slot() {
    let bitcode = match assemble(&authored(MERGED_STORES)) {
        Ok(bitcode) => bitcode,
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            return;
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejects the kernel: {why}"),
    };
    let library = Library::from_air(&bitcode).expect("the kernel decodes");
    // (Metal index, name, whether its store is narrower than a word)
    let buffers = [
        (0, "bytes", true),
        (1, "halves", true),
        (2, "words", false),
        (3, "other", true),
    ];
    for (stem, narrow_units) in [("vk12-floor", false), ("moltenvk-m3", true)] {
        let device = device(stem);
        let function = FunctionDescriptor::new(library.function_at(0).expect("the kernel"));
        let compiled = device
            .compile_function(&function, &LoweringInputs::new(), &Budget::default())
            .unwrap_or_else(|e| panic!("{stem}: {e}"));
        let (table, words) = link_standalone(compiled.artifact(), device.profile())
            .unwrap_or_else(|e| panic!("{stem}: link: {e}"));
        let slots = &compiled.artifact().interface.slots;
        let w = walk(&words);
        let unreadable = decorated_with(&words, DECORATION_NON_READABLE);
        for (index, name, narrow) in buffers {
            let merged = narrow && !narrow_units;
            let slot = SlotRef {
                stage: Stage::Kernel,
                class: SlotClass::Buffer,
                index,
            };
            let Some(SlotInterface::Buffer(buffer)) = slots.get(&slot) else {
                panic!("{stem}: {name} is not a buffer slot: {slots:?}");
            };
            let expected = if merged {
                AccessSet::READ_WRITE
            } else {
                AccessSet::WRITE
            };
            assert_eq!(buffer.access, expected, "{stem}: {name}'s reported access");
            let resource = ResourceSlot {
                stage: Stage::Kernel,
                kind: ResourceKind::Buffer,
                index,
            };
            let bound = *table
                .bindings
                .get(&resource)
                .unwrap_or_else(|| panic!("{stem}: {name} is not bound"));
            let vars: BTreeSet<u32> = w
                .binding
                .iter()
                .filter(|&(_, &b)| b == bound)
                .map(|(&var, _)| var)
                .collect();
            assert!(!vars.is_empty(), "{stem}: {name} has no variable");
            // The kernel loads nothing, so only a merged store's atomics read a slot.
            assert_eq!(
                !w.read.is_disjoint(&vars),
                merged,
                "{stem}: whether the module reads {name}"
            );
            assert!(
                !w.written.is_disjoint(&vars),
                "{stem}: the module writes {name}"
            );
            assert!(
                unreadable.is_disjoint(&vars),
                "{stem}: {name}'s variable is decorated NonReadable"
            );
        }
        let counts = check_access("merged stores", &words, &table.bindings, slots);
        assert!(counts.narrower.is_empty(), "{:?}", counts.narrower);
    }
}

/// Two kernels whose slot `out` escapes (each thread stores its element's address into
/// `table`) and is then reached only through a pointer loaded back from a neighbour's entry: the
/// first stores through it, the second loads through it into `quiet`. `quiet` never escapes.
const ESCAPED_SLOTS: &str = r#"define void @escaped_written(ptr addrspace(1) noundef %out, ptr addrspace(1) noundef %table, ptr addrspace(1) noundef %quiet, i32 noundef %gid) local_unnamed_addr #0 {
  %i = zext i32 %gid to i64
  %entry = getelementptr inbounds ptr addrspace(1), ptr addrspace(1) %table, i64 %i
  %mine = getelementptr inbounds i32, ptr addrspace(1) %out, i64 %i
  store ptr addrspace(1) %mine, ptr addrspace(1) %entry, align 8
  %j = xor i64 %i, 1
  %theirs = getelementptr inbounds ptr addrspace(1), ptr addrspace(1) %table, i64 %j
  %p = load ptr addrspace(1), ptr addrspace(1) %theirs, align 8
  store i32 %gid, ptr addrspace(1) %p, align 4
  %q = getelementptr inbounds i32, ptr addrspace(1) %quiet, i64 %i
  store i32 %gid, ptr addrspace(1) %q, align 4
  ret void
}

define void @escaped_read(ptr addrspace(1) noundef %out, ptr addrspace(1) noundef %table, ptr addrspace(1) noundef %quiet, i32 noundef %gid) local_unnamed_addr #0 {
  %i = zext i32 %gid to i64
  %entry = getelementptr inbounds ptr addrspace(1), ptr addrspace(1) %table, i64 %i
  %mine = getelementptr inbounds i32, ptr addrspace(1) %out, i64 %i
  store ptr addrspace(1) %mine, ptr addrspace(1) %entry, align 8
  %j = xor i64 %i, 1
  %theirs = getelementptr inbounds ptr addrspace(1), ptr addrspace(1) %table, i64 %j
  %p = load ptr addrspace(1), ptr addrspace(1) %theirs, align 8
  %v = load i32, ptr addrspace(1) %p, align 4
  %q = getelementptr inbounds i32, ptr addrspace(1) %quiet, i64 %i
  store i32 %v, ptr addrspace(1) %q, align 4
  ret void
}

attributes #0 = { nounwind }

!air.kernel = !{!0, !10}
!0 = !{ptr @escaped_written, !1, !2}
!10 = !{ptr @escaped_read, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5, !6}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"uint*", !"air.arg_name", !"table"}
!5 = !{i32 2, !"air.buffer", !"air.location_index", i32 2, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"quiet"}
!6 = !{i32 3, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"gid"}
"#;

/// A slot whose address escapes is in the reach of every access through a loaded pointer: on a
/// profile with device addresses, `out`, which the function reaches only through a pointer it loads
/// back, reports a write where some such access stores (and a read: a store through a device
/// address may merge its words) and a read where one only loads. Before, it reported neither, and a
/// consumer placed no barrier for it. `quiet`, which does not escape, reports its own accesses
/// alone, and `table` its pointer stores and loads. The accesses the module makes through the
/// loaded pointer are device-address accesses, which no descriptor binding of the walk sees, so the
/// walk's coverage check does not reach them.
#[test]
fn an_escaped_slot_reports_what_loaded_pointers_may_do_to_it() {
    let bitcode = match assemble(&authored(ESCAPED_SLOTS)) {
        Ok(bitcode) => bitcode,
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            return;
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejects the kernels: {why}"),
    };
    let library = Library::from_air(&bitcode).expect("the kernels decode");
    let slot = |index| SlotRef {
        stage: Stage::Kernel,
        class: SlotClass::Buffer,
        index,
    };
    // (function index, name, out's expected access, quiet's)
    let kernels = [
        (
            0,
            "escaped_written",
            AccessSet::READ_WRITE,
            AccessSet::WRITE,
        ),
        (1, "escaped_read", AccessSet::READ, AccessSet::WRITE),
    ];
    for stem in ["moltenvk-m3", "nvidia-rtx5080-laptop"] {
        let device = device(stem);
        for (index, name, out, quiet) in kernels {
            let function = FunctionDescriptor::new(library.function_at(index).expect(name));
            let compiled = device
                .compile_function(&function, &LoweringInputs::new(), &Budget::default())
                .unwrap_or_else(|e| panic!("{stem}: {name}: {e}"));
            let slots = &compiled.artifact().interface.slots;
            for (index, slot_name, expected) in [
                (0, "out", out),
                (1, "table", AccessSet::READ_WRITE),
                (2, "quiet", quiet),
            ] {
                let Some(SlotInterface::Buffer(buffer)) = slots.get(&slot(index)) else {
                    panic!("{stem}: {name}: {slot_name} is not a buffer slot: {slots:?}");
                };
                assert_eq!(
                    buffer.access, expected,
                    "{stem}: {name}: {slot_name}'s reported access"
                );
            }
        }
    }
}

/// Every entry point of every committed fixture that compiles, linked alone on each bound
/// profile, reports an access that covers its module's.
#[test]
fn every_public_fixture_reports_an_access_covering_its_module() {
    let mut devices: Vec<(&str, Device)> = BOUND_PROFILES
        .iter()
        .map(|&stem| (stem, device(stem)))
        .collect();
    devices.push((FLOOR_WITH_16_BIT, floor_with_16_bit()));
    let budget = Budget::default();
    let mut total = AccessCounts::default();
    let mut modules = 0usize;
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
                let counts = check_access(
                    &format!("{label} {} on {stem}", info.name),
                    &words,
                    &table.bindings,
                    &compiled.artifact().interface.slots,
                );
                modules += 1;
                total.buffers += counts.buffers;
                total.textures += counts.textures;
                total.written += counts.written;
                total.read += counts.read;
                total.sampled += counts.sampled;
                total.narrower.extend(counts.narrower);
            }
        }
    }
    eprintln!(
        "{modules} modules checked: {} buffer and {} texture slots; {} written, {} read, {} \
         sampled",
        total.buffers, total.textures, total.written, total.read, total.sampled
    );
    assert!(
        total.narrower.is_empty(),
        "{} slots report an access narrower than their module's:\n{}",
        total.narrower.len(),
        total.narrower.join("\n")
    );
    assert!(
        modules >= 1000 && total.buffers >= 1000 && total.textures >= 300,
        "the sweep checked far fewer than expected: {total:?} in {modules} modules"
    );
}
