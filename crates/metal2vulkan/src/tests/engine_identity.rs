//! The identity domains (`PointerDomain::Identity` with `HandleDomain::Identity`) in the lowering.
//!
//! - A kernel over buffers alone (`engine/a1_bool_from_float`) compiles with each slot reached
//!   through its own device address, pushed (`SlotGuestBase`), with no root table and no
//!   descriptor. It passes L3, and on the CPU interpreter, with its buffers above 4 GiB and bound
//!   at offsets that are not 0, it computes the program's answer and writes nothing else.
//! - A kernel whose slot addresses do not fit the 128-byte window (`many`, generated: 20 read
//!   buffers and an output) pushes its parameter block's address first
//!   (`ArgumentTableAddress`), the addresses that fit after it, and reads the rest from the block
//!   at their placed offsets, with no descriptor; it passes L3 and computes its answer on the CPU
//!   interpreter. On `Device::from_queries`' profile, whose kernel window is 256 bytes, the same
//!   kernel pushes all 21 addresses with no block, and one of 31 slots, Metal's most, all 31.
//! - A kernel that reads grid builtins (`grid`, generated) declares no `MultiRegion` constant and
//!   reads Metal's grid from its pushed `DispatchRegion` on every dispatch: whole, split in two
//!   pieces at a base, and indirect (a zero grid, its grid `NumWorkgroups · L`) store the same
//!   answers.
//! - A kernel's partial-threadgroups variant (`partial`, generated) dispatched at its whole
//!   threadgroup size over a ragged grid, its pushed region flagged `NONUNIFORM`, answers each
//!   edge threadgroup's own size, and its invocations past the grid store nothing.
//! - A kernel's texture slots (`tex`, generated: a read, a write, a width and an `is_null_texture`)
//!   are the elements of the two fixed set-1 arrays at the indices their pushed resource IDs hold,
//!   with no descriptor of their own. It passes L3 and, on the CPU interpreter, reads and writes
//!   the textures there, answers the width and `is_null_texture` by the index (0 for nil), and,
//!   where the shader checks reads (no `robustImageAccess2`), takes a format's missing alpha out of
//!   bounds from the word, as `tests/resource_words.tsv` gives it.
//! - The same kernel with its textures in an argument buffer (`tex` with `ab`) reads their
//!   resource IDs from the buffer's members, at its pushed address, and answers the same at the
//!   same elements, with no handle table and no descriptor.
//! - A kernel's slots that nothing writes (`read_only`, generated: an unaliased `device` buffer it
//!   only reads and a `constant` one) are loaded through `NonWritable` device-address places; a
//!   written slot, and a `device` slot the AIR does not declare unaliased, are not. It passes L3
//!   and computes its answer on the CPU interpreter.
//! - A kernel sampling a texture slot through a runtime sampler slot (the harness's
//!   `sample_down`) takes the texture and the sampler from set 1's arrays at the indices their
//!   pushed resource IDs hold, the sampler from the plain array, with no descriptor, no
//!   specialization constant and no fact of the bound sampler; it passes L3.
//! - A kernel sampling through a constexpr sampler (the harness's `sample_up`, and the same
//!   kernel through a pixel sampler) takes the sampler from the plain array at the index the
//!   consumer pushes for it (`PushField::ConstexprSampler`), whose device sampler is normalized;
//!   it passes L3.
//! - A kernel writing integer texels to a storage slot without a format (`ic`, generated)
//!   saturates each lane to the channel widths the consumer pushes
//!   (`PushField::TextureIntegerChannels`), with no fact, on the CPU interpreter.
//! - A kernel that samples through a bicubic constexpr sampler (`engine/a2_bicubic_weights`) is
//!   still refused as `Unsupported{IdentityDomains}`, not implemented, where the default domains
//!   compile it.
//! - A kernel's emulated-simdgroup variant (`engine/gsm2_simd_products`'s `simd_float_products`)
//!   opens every barrier of its exchanges under a branch on `1 < N`, which the default domains do
//!   not; on the CPU interpreter a threadgroup of one thread passes no barrier, and threadgroups of
//!   one and of 80 threads (a partial simdgroup of 16) answer Metal's products.
//!
//! `llvm-as` assembles the kernels; without it the tests skip visibly.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::device::profile::{DeviceProfile, HandleDomain, PointerDomain};
use crate::device::requirements::VkFeature;
use crate::device::{Device, DeviceDescription, DevicePolicy};
use crate::error::{Error, Feature, Needs};
use crate::lower::dispatch::{region_bytes, NONUNIFORM};
use crate::lower::select::{LowerInputs, Plan};
use crate::memory::handles::Lookup;
use crate::metal::resource_word::TextureWord;
use crate::metal::{PixelFormat, TextureSwizzleChannels};
use crate::pipeline::facts::RuntimeFact;
use crate::pipeline::interface::PushLoc;
use crate::pipeline::layout::BindingSource;
use crate::pipeline::link::{link, LinkInputs};
use crate::pipeline::sym::{Axis, PushField, ResourceSlot, SlotClass, SlotRef, SpecKey, Stage};
use crate::pipeline::testing::{
    compile, compile_kept, compile_with, engine_dir, link_and_validate, llvm_assemble, profile,
    typed_module_with, AssembleError,
};
use crate::spirv::interp::{
    run, run_with_images, BarrierRule, Buffer, Config, Image, Order, Resources,
};
use crate::spirv::testing::spirv_dis;
use crate::vk;

/// Profiles with `bufferDeviceAddress` and `shaderInt64`.
const PROFILES: [&str; 2] = ["lavapipe", "moltenvk-m3"];

fn identity() -> DevicePolicy {
    DevicePolicy {
        pointer_domain: PointerDomain::Identity,
        handle_domain: HandleDomain::Identity,
        ..DevicePolicy::default()
    }
}

fn assemble(name: &str) -> Option<Vec<u8>> {
    assembled(
        name,
        llvm_assemble(&engine_dir().join(format!("{name}.ll"))),
    )
}

fn assembled(name: &str, r: Result<Vec<u8>, AssembleError>) -> Option<Vec<u8>> {
    match r {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(e) => panic!("{name}: {e:?}"),
    }
}

/// `text` assembled in a scratch directory of its own, removed once `llvm-as` is done.
fn assemble_text(name: &str, text: &str) -> Option<Vec<u8>> {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let dir = std::env::temp_dir().join(format!(
        "m2v-identity-{}-{}-{name}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&dir).expect("scratch");
    let ll = dir.join("k.ll");
    std::fs::write(&ll, text).expect("write");
    let r = llvm_assemble(&ll);
    let _ = std::fs::remove_dir_all(&dir);
    assembled(name, r)
}

/// The spec map of a dispatch of `group` threads a threadgroup, in one dimension.
fn local_size(group: u32) -> BTreeMap<SpecKey, u64> {
    local_size3([group, 1, 1])
}

/// The spec map of a dispatch of threadgroups of size `l`.
fn local_size3(l: [u32; 3]) -> BTreeMap<SpecKey, u64> {
    BTreeMap::from([
        (SpecKey::LocalSize(Axis::X), u64::from(l[0])),
        (SpecKey::LocalSize(Axis::Y), u64::from(l[1])),
        (SpecKey::LocalSize(Axis::Z), u64::from(l[2])),
        (SpecKey::FlatLocalSize, u64::from(l[0] * l[1] * l[2])),
    ])
}

fn slot(index: u32) -> SlotRef {
    SlotRef {
        stage: Stage::Kernel,
        class: SlotClass::Buffer,
        index,
    }
}

/// `float` patterns none of which is a NaN or a denormal (the fixture compiles with fast math).
const FLOATS: [f32; 6] = [0.0, -0.0, 0.5, 2.0, 1.0, -3.0];
/// The same values' `half` bits.
const HALVES: [u16; 6] = [0x0000, 0x8000, 0x3800, 0x4000, 0x3c00, 0xc200];

/// Thread `i`'s input lanes: indices into [`FLOATS`] and [`HALVES`].
fn lanes(i: usize) -> ([usize; 4], [usize; 4]) {
    (
        std::array::from_fn(|j| (i + j) % FLOATS.len()),
        std::array::from_fn(|j| (3 * i + j) % HALVES.len()),
    )
}

/// What `a1_bool_from_float` writes for thread `i`: each lane's `x != 0`, the `float` lanes in
/// bits 0-3 and 4-7, the `half` lanes in bits 8-11 and 12-15.
fn want(i: usize) -> u32 {
    let (f, h) = lanes(i);
    let mut r = 0u32;
    for (j, (&fi, &hi)) in f.iter().zip(&h).enumerate() {
        let fb = u32::from(FLOATS[fi] != 0.0);
        let hb = u32::from(HALVES[hi] & 0x7fff != 0);
        r |= fb << j | fb << (4 + j) | hb << (8 + j) | hb << (12 + j);
    }
    r
}

/// A kernel over buffers alone compiles under the identity domains to a module that reads each
/// slot's device address from its own push field and nothing else, and computes the program's
/// answer there.
#[test]
fn a_buffer_kernel_reads_each_slot_at_its_pushed_device_address() {
    let Some(bc) = assemble("a1_bool_from_float") else {
        return;
    };
    const GROUP: u32 = 8;
    const GROUPS: u32 = 4;
    let threads = (GROUP * GROUPS) as usize;
    // (the buffer's device address, the bind offset): above 4 GiB, so a dropped high word shows.
    let at = [
        (0x1_0000_0000u64, 32u64),
        (0x2_0000_1000, 16),
        (0x3_0000_2000, 4),
    ];
    let mut bits = vec![0u8; at[0].1 as usize];
    let mut halves = vec![0u8; at[1].1 as usize];
    for i in 0..threads {
        let (f, h) = lanes(i);
        for (&fi, &hi) in f.iter().zip(&h) {
            bits.extend_from_slice(&FLOATS[fi].to_bits().to_le_bytes());
            halves.extend_from_slice(&HALVES[hi].to_le_bytes());
        }
    }
    // One spare word past the output's end, which the kernel must not write.
    let out = vec![0u8; at[2].1 as usize + 4 * threads + 4];
    for stem in PROFILES {
        let p = profile(stem, identity()).unwrap_or_else(|e| panic!("{e}"));
        let (a, plan, module) =
            compile_kept(&bc, "a1_bool_from_float", &p).unwrap_or_else(|e| panic!("{stem}: {e}"));
        assert!(a.interface.descriptors.is_empty(), "{stem}");
        assert!(!a.interface.needs_pointer_translation, "{stem}");
        assert_eq!(plan.push.argument_table_size, 0, "{stem}");
        link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{stem}: L3: {e}"));
        let mut push = vec![0u8; plan.push.push_range.map_or(0, |(_, end)| end as usize)];
        let mut read = Vec::new();
        // The grid builtin's region, as the identity domains fill it: base 0, the threadgroup
        // size, and the grid.
        let region = region_bytes([0; 3], [GROUP * GROUPS, 1, 1], [GROUP, 1, 1]).expect("packs");
        for f in &plan.push.fields {
            let o = match f.loc {
                PushLoc::Push { offset } => offset as usize,
                PushLoc::ArgumentTable { .. } => panic!("{stem}: {f:?} spilled"),
            };
            match f.field {
                PushField::SlotGuestBase(s) => {
                    let (base, bind) = at[s.index as usize];
                    push[o..o + 8].copy_from_slice(&(base + bind).to_le_bytes());
                    read.push(s);
                }
                PushField::DispatchRegion => push[o..o + 32].copy_from_slice(&region),
                _ => panic!("{stem}: unexpected push field {f:?}"),
            }
        }
        read.sort();
        assert_eq!(read, [slot(0), slot(1), slot(2)], "{stem}");
        let buffer = |(address, _): (u64, u64), bytes: &Vec<u8>| Buffer {
            slot: None,
            address: Some(address),
            bytes: bytes.clone(),
        };
        let mut expected = vec![0u8; at[2].1 as usize];
        for i in 0..threads {
            expected.extend_from_slice(&want(i).to_le_bytes());
        }
        expected.extend_from_slice(&[0; 4]);
        let res = Resources {
            buffers: vec![
                buffer(at[0], &bits),
                buffer(at[1], &halves),
                buffer(at[2], &out),
            ],
            push,
            spec: local_size(GROUP),
        };
        let cfg = Config {
            workgroups: [GROUPS, 1, 1],
            ..Config::default()
        };
        let got = run(&module, &cfg, res).unwrap_or_else(|e| panic!("{stem}: {e}"));
        assert_eq!(got.buffers[2], expected, "{stem}: out");
        assert_eq!(got.buffers[0], bits, "{stem}: bits");
        assert_eq!(got.buffers[1], halves, "{stem}: halves");
    }
}

/// The harness's sampling kernels (`fixtures/public/harness/workloads.ll`, our own), assembled.
fn workloads() -> Option<Vec<u8>> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/fixtures/public/harness/workloads.ll");
    assembled("workloads", llvm_assemble(&path))
}

/// A kernel sampling a texture slot through a runtime sampler slot (`sample_down`: a 2D `float`
/// sample, a size query and a write) takes the sampled texture from set 1's sampled-image array
/// and the sampler from its plain sampler array, each at the index its pushed resource ID holds,
/// with no descriptor, no specialization constant and no fact of the bound sampler: the word the
/// sample reads is the resource ID's. It passes L3 on each profile.
#[test]
fn a_runtime_sampler_is_the_set_1_element_its_resource_id_names() {
    let Some(bc) = workloads() else {
        return;
    };
    let mut profiles: Vec<(String, DeviceProfile)> = PROFILES
        .iter()
        .map(|stem| {
            let p = profile(stem, identity()).unwrap_or_else(|e| panic!("{e}"));
            (stem.to_string(), p)
        })
        .collect();
    profiles.push((
        "moltenvk-m3 without robustImageAccess2".to_string(),
        identity_without("moltenvk-m3", &[VkFeature::RobustImageAccess2]),
    ));
    for (stem, p) in &profiles {
        let (a, plan, _) =
            compile_kept(&bc, "sample_down", p).unwrap_or_else(|e| panic!("{stem}: {e}"));
        link_and_validate(&a, p).unwrap_or_else(|e| panic!("{stem}: L3: {e}"));
        assert!(
            (a.interface.spec.iter()).all(|c| matches!(c.key, SpecKey::LocalSize(_))),
            "{stem}: {:?}",
            a.interface.spec
        );
        assert!(
            !plan.facts.iter().any(|f| matches!(
                f,
                RuntimeFact::SamplerCoordinates { .. }
                    | RuntimeFact::SamplerReduction { .. }
                    | RuntimeFact::CubicSampler { .. }
            )),
            "{stem}: {:?}",
            plan.facts
        );
        let mut slots: Vec<_> = (plan.push.fields.iter())
            .filter_map(|f| match f.field {
                PushField::SlotGuestBase(s) => Some((s.class, s.index)),
                _ => None,
            })
            .collect();
        slots.sort();
        assert_eq!(
            slots,
            [
                (SlotClass::Texture, 0),
                (SlotClass::Texture, 1),
                (SlotClass::Sampler, 0)
            ],
            "{stem}"
        );
        let linked = link(&[&a], &LinkInputs::standalone(), p).unwrap_or_else(|e| panic!("{e}"));
        let sets: Vec<_> = (linked.layout.sets.iter())
            .map(|s| {
                let b: Vec<_> = (s.bindings.iter())
                    .map(|b| (b.binding, b.ty, b.count))
                    .collect();
                (s.set, b)
            })
            .collect();
        assert_eq!(sets, [(0, vec![]), (1, identity_arrays())], "{stem}");
        let mut used: Vec<u32> = (linked.table.bindings.values())
            .filter(|(set, _)| *set == 1)
            .map(|(_, binding)| *binding)
            .collect();
        used.sort();
        // The sampled source, the written destination, and the plain sampler.
        assert_eq!(used, [0, 1, 2], "{stem}");
    }
}

/// The words of `sample_up`'s constexpr sampler (`fixtures/public/harness/workloads.ll`).
const SAMPLE_UP_WORDS: [u64; 2] = [34_901_797_601_020_489, 0];

/// A kernel sampling a texture slot through a constexpr sampler (`sample_up`, a 2D `float` sample
/// of normalized coordinates, and the same kernel through a pixel sampler that the default domains
/// hand the device's unnormalized sampler) takes the texture from set 1's sampled-image array at
/// the index its pushed resource ID holds and the sampler from the plain sampler array at the
/// index the consumer pushes for it (`PushField::ConstexprSampler`, a `u32`), with no descriptor
/// and no specialization constant. The field's sampler is the constexpr one in normalized
/// coordinates, whose device sampler (`CombinedSampler::constexpr_sampler`) is normalized: the
/// shader scales the linear pixel sampler's coordinate and records
/// `PrecisionNote::PixelSamplerScaling`, which the default domains do not. It passes L3 on each
/// profile.
#[test]
fn a_constexpr_sampler_is_the_set_1_element_its_pushed_index_names() {
    use crate::air::abi::{AddressMode, Coordinates, Filter, MipFilter, SamplerState};
    use crate::pipeline::artifact::PrecisionNote;
    use crate::pipeline::sym::{CombinedSampler, SamplerRef};
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/fixtures/public/harness/workloads.ll");
    let text = std::fs::read_to_string(path).expect("workloads.ll");
    let state = |w: [u64; 2]| format!("[2 x i64] [i64 {}, i64 {}]", w[0], w[1]);
    assert_eq!(text.matches(&state(SAMPLE_UP_WORDS)).count(), 1);
    let s = crate::lower::sampler::state_of(SAMPLE_UP_WORDS).expect("state");
    let pixel = crate::lower::sampler::state_words(&SamplerState {
        coordinates: Coordinates::Pixel,
        min_filter: Filter::Linear,
        mag_filter: Filter::Linear,
        mip_filter: MipFilter::None,
        max_anisotropy: 1,
        address_s: AddressMode::ClampToEdge,
        address_t: AddressMode::ClampToEdge,
        ..s
    });
    let pixel_text = text.replace(&state(SAMPLE_UP_WORDS), &state(pixel));
    let (Some(normalized), Some(pixel_bc)) = (
        assemble_text("sample_up", &text),
        assemble_text("sample_up_pixel", &pixel_text),
    ) else {
        return;
    };
    let mut profiles: Vec<(String, DeviceProfile)> = PROFILES
        .iter()
        .map(|stem| {
            let p = profile(stem, identity()).unwrap_or_else(|e| panic!("{e}"));
            (stem.to_string(), p)
        })
        .collect();
    profiles.push((
        "moltenvk-m3 without robustImageAccess2".to_string(),
        identity_without("moltenvk-m3", &[VkFeature::RobustImageAccess2]),
    ));
    for (bc, words) in [(&normalized, SAMPLE_UP_WORDS), (&pixel_bc, pixel)] {
        for stem in PROFILES {
            // The default domains read the linear pixel sampler's coordinate through the
            // device's unnormalized sampler, with no note, so the identity form below scales it
            // in its place.
            let default = profile(stem, DevicePolicy::default()).unwrap_or_else(|e| panic!("{e}"));
            let (a, plan, _) = compile_kept(bc, "sample_up", &default)
                .unwrap_or_else(|e| panic!("{stem}: default domains: {e}"));
            let unnormalized: Vec<bool> = (plan.families.sampler.decls.values())
                .map(|d| d.sampler.unnormalized)
                .collect();
            assert_eq!(unnormalized, [words == pixel], "{stem}: default domains");
            assert!(
                !a.precision.contains(&PrecisionNote::PixelSamplerScaling),
                "{stem}: default domains"
            );
        }
        for (stem, p) in &profiles {
            let (a, plan, _) =
                compile_kept(bc, "sample_up", p).unwrap_or_else(|e| panic!("{stem}: {e}"));
            link_and_validate(&a, p).unwrap_or_else(|e| panic!("{stem}: L3: {e}"));
            assert!(
                (a.interface.spec.iter()).all(|c| matches!(c.key, SpecKey::LocalSize(_))),
                "{stem}: {:?}",
                a.interface.spec
            );
            // The sampler's own address modes: no texture's map, so no variant fact.
            assert!(
                !(plan.facts.iter()).any(|f| matches!(f, RuntimeFact::ConstexprSampled { .. })),
                "{stem}: {:?}",
                plan.facts
            );
            let c = CombinedSampler::new(SamplerRef::Constexpr(words), false, false);
            let mut fields: Vec<_> = (plan.push.fields.iter())
                .map(|f| f.field)
                .filter(|f| *f != PushField::DispatchRegion)
                .collect();
            fields.sort();
            assert_eq!(
                fields,
                [
                    PushField::SlotGuestBase(SlotRef {
                        stage: Stage::Kernel,
                        class: SlotClass::Texture,
                        index: 0
                    }),
                    PushField::SlotGuestBase(SlotRef {
                        stage: Stage::Kernel,
                        class: SlotClass::Texture,
                        index: 1
                    }),
                    PushField::ConstexprSampler(c),
                ],
                "{stem}"
            );
            let desc = (c.constexpr_sampler(p))
                .unwrap_or_else(|e| panic!("{stem}: {e}"))
                .expect("a constexpr sampler");
            assert!(!desc.unnormalized, "{stem}");
            // The scaled coordinate's static note.
            assert_eq!(
                a.precision.contains(&PrecisionNote::PixelSamplerScaling),
                words == pixel,
                "{stem}"
            );
            let linked =
                link(&[&a], &LinkInputs::standalone(), p).unwrap_or_else(|e| panic!("{e}"));
            assert!(
                (linked.layout.sets.iter())
                    .flat_map(|s| &s.bindings)
                    .all(|b| b.immutable_sampler.is_none() && b.immutable_samplers.is_empty()),
                "{stem}"
            );
            let mut used: Vec<u32> = (linked.table.bindings.values())
                .filter(|(set, _)| *set == 1)
                .map(|(_, binding)| *binding)
                .collect();
            used.sort();
            assert_eq!(used, [0, 1, 2], "{stem}");
        }
    }
}

/// A kernel writing `uint4(300, 70000, 5, 2^31)` to its `texture2d<uint, write>` slot 0 at its
/// position, over a `2 × 1` grid.
fn channels_kernel() -> String {
    "define void @ic(ptr addrspace(1) %t, <2 x i32> noundef %tid) {\n\
entry:\n\
  tail call void @air.write_texture_2d.u.v4i32(ptr addrspace(1) %t, <2 x i32> %tid, <4 x i32> <i32 300, i32 70000, i32 5, i32 -2147483648>, i32 0, i32 2)\n\
  ret void\n\
}\n\n\
declare void @air.write_texture_2d.u.v4i32(ptr addrspace(1), <2 x i32>, <4 x i32>, i32, i32)\n\n\
!air.kernel = !{!0}\n!air.version = !{!4}\n!air.language_version = !{!5}\n\
!0 = !{ptr @ic, !1, !2}\n!1 = !{}\n!2 = !{!6, !3}\n\
!6 = !{i32 0, !\"air.texture\", !\"air.location_index\", i32 0, i32 1, !\"air.write\", !\"air.arg_type_name\", !\"texture2d<uint, write>\", !\"air.arg_name\", !\"t\"}\n\
!3 = !{i32 1, !\"air.thread_position_in_grid\", !\"air.arg_type_name\", !\"uint2\", !\"air.arg_name\", !\"tid\"}\n\
!4 = !{i32 2, i32 8, i32 0}\n!5 = !{!\"Metal\", i32 4, i32 0, i32 0}\n"
        .to_string()
}

/// An integer write to a storage slot without a format saturates each lane to the bound format's
/// channel width, which the consumer pushes (`PushField::TextureIntegerChannels`, the push word
/// of the contract, with no fact): into `RGBA8Uint` the kernel's 300, 70000 and 2^31 store 255
/// and its 5 stays, and a word of 0 (32-bit channels) stores every lane as written.
#[test]
fn an_integer_write_saturates_to_the_pushed_channel_widths() {
    let Some(bc) = assemble_text("ic", &channels_kernel()) else {
        return;
    };
    const L: [u32; 3] = [2, 1, 1];
    for stem in PROFILES {
        let p = profile(stem, identity()).unwrap_or_else(|e| panic!("{e}"));
        let (a, plan, module) =
            compile_kept(&bc, "ic", &p).unwrap_or_else(|e| panic!("{stem}: {e}"));
        link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{stem}: L3: {e}"));
        assert!(
            !(plan.facts.iter())
                .any(|f| matches!(f, RuntimeFact::StorageTextureIntegerChannels { .. })),
            "{stem}"
        );
        let linked = link(&[&a], &LinkInputs::standalone(), &p).unwrap_or_else(|e| panic!("{e}"));
        let (storage, _) = (linked.table.bindings.iter())
            .find(|(_, b)| **b == (1, 1))
            .unwrap_or_else(|| panic!("{stem}: no storage class"));
        let rgba8 = crate::metal::format::integer_channels_word(PixelFormat::RGBA8Uint);
        for (word, want) in [
            (rgba8, [255, 255, 5, 255]),
            (0, [300, 70_000, 5, 0x8000_0000]),
        ] {
            let mut push = vec![0u8; plan.push.push_range.map_or(0, |(_, end)| end as usize)];
            for field in &plan.push.fields {
                let PushLoc::Push { offset } = field.loc else {
                    panic!("{stem}: {field:?} spilled");
                };
                let o = offset as usize;
                match field.field {
                    PushField::SlotGuestBase(_) => {
                        push[o..o + 8].copy_from_slice(&resource_id(2, 0).to_le_bytes())
                    }
                    PushField::TextureIntegerChannels(_) => {
                        push[o..o + 4].copy_from_slice(&word.to_le_bytes())
                    }
                    PushField::DispatchRegion => {
                        let region = region_bytes([0; 3], L, L).expect("packs");
                        push[o..o + 32].copy_from_slice(&region);
                    }
                    other => panic!("{stem}: unexpected push field {other:?}"),
                }
            }
            let mut images: Vec<Image> = (0..2)
                .map(|e| texture(*storage, e, [1, 1], |_, _| [0; 4], None))
                .collect();
            images.push(texture(*storage, 2, [2, 1], |_, _| [0; 4], None));
            let res = Resources {
                buffers: Vec::new(),
                push,
                spec: local_size3(L),
            };
            let (_, images) = run_with_images(&module, &Config::default(), res, images)
                .unwrap_or_else(|e| panic!("{stem}: {e}"));
            assert_eq!(images[2].texels, [want, want], "{stem}: word {word:#x}");
        }
    }
}

/// Set 1's fixed arrays under the identity domains: binding, descriptor type and count.
fn identity_arrays() -> Vec<(u32, vk::DescriptorType, u32)> {
    vec![
        (0, vk::DescriptorType::SAMPLED_IMAGE, 65_536),
        (1, vk::DescriptorType::STORAGE_IMAGE, 4_096),
        (2, vk::DescriptorType::SAMPLER, 4_000),
        (3, vk::DescriptorType::SAMPLER, 4_000),
        (4, vk::DescriptorType::SAMPLER, 4_000),
    ]
}

/// A kernel that samples a texture through a bicubic constexpr sampler is not lowered under the
/// identity domains yet.
#[test]
fn a_bicubic_kernel_is_refused_under_the_identity_domains() {
    let Some(bc) = assemble("a2_bicubic_weights") else {
        return;
    };
    for stem in PROFILES {
        let default = profile(stem, DevicePolicy::default()).unwrap_or_else(|e| panic!("{e}"));
        compile(&bc, "a2_bicubic_weights", &default)
            .unwrap_or_else(|e| panic!("{stem}: default domains: {e}"));
        let p = profile(stem, identity()).unwrap_or_else(|e| panic!("{e}"));
        match compile(&bc, "a2_bicubic_weights", &p) {
            Err(Error::Unsupported(u)) => {
                assert_eq!(
                    (u.feature, u.needs),
                    (Feature::IdentityDomains, Needs::NotImplemented),
                    "{stem}"
                );
            }
            other => panic!("{stem}: {other:?}"),
        }
    }
}

/// The `gpuResourceID` high word of a texture of pixel format `pf` with the identity swizzle, from the test vectors (`docs/RESOURCE-ID.md`), checked against
/// `TextureWord::of`: the word the runtime writes.
fn vector_word(pf: PixelFormat) -> u32 {
    let format = pf.as_raw();
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/resource_words.tsv");
    let text = std::fs::read_to_string(path).expect("resource_words.tsv");
    let row = format!("texture\t{format}\t2\t3\t4\t5\t");
    let line = text
        .lines()
        .find(|l| l.starts_with(&row))
        .unwrap_or_else(|| panic!("no vector for format {format}"));
    let word = u32::from_str_radix(line[row.len()..].trim_start_matches("0x"), 16).expect("hex");
    let of = TextureWord::of(pf, TextureSwizzleChannels::DEFAULT).expect("a word");
    assert_eq!(of, TextureWord(word), "format {format}");
    word
}

/// A kernel over a `5 × 2` grid that reads `src` at its position, writes the texel to `dst`
/// there, and stores, at word `2 · (y · 5 + x)` of `out`, `src`'s width with
/// `is_null_texture(other)` in bit 16, and the texel's alpha bits in the next word. Its textures
/// are slots 0 to 2 and `out` buffer 0, or, with `ab`, the members of the argument buffer at
/// buffer 0 (`src`, `dst` and `other`, 8 bytes each) and `out` buffer 1.
fn texture_kernel(ab: bool) -> String {
    let texture = |member: u32, access: &str, name: &str| {
        format!(
            "!{{i32 {member}, !\"air.texture\", !\"air.location_index\", i32 {member}, i32 1, !\"air.{access}\", !\"air.arg_type_name\", !\"texture2d<float, {access}>\", !\"air.arg_name\", !\"{name}\"}}"
        )
    };
    let textures = [
        texture(0, "read", "src"),
        texture(1, "write", "dst"),
        texture(2, "read", "other"),
    ];
    let out = |arg: u32, index: u32| {
        format!(
            "!{{i32 {arg}, !\"air.buffer\", !\"air.location_index\", i32 {index}, i32 1, !\"air.read_write\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"out\"}}"
        )
    };
    let tid = |arg: u32| {
        format!(
            "!{{i32 {arg}, !\"air.thread_position_in_grid\", !\"air.arg_type_name\", !\"uint2\", !\"air.arg_name\", !\"tid\"}}"
        )
    };
    let tail = "ptr addrspace(1) noundef \"air-buffer-no-alias\" %out, <2 x i32> noundef %tid";
    let (types, params, loads, args) = match ab {
        false => (
            String::new(),
            format!("ptr addrspace(1) %src, ptr addrspace(1) %dst, ptr addrspace(1) %other, {tail}"),
            String::new(),
            format!(
                "!11 = !{{!12, !13, !14, !15, !16}}\n!12 = {}\n!13 = {}\n!14 = {}\n!15 = {}\n!16 = {}\n",
                textures[0],
                textures[1],
                textures[2],
                out(3, 0),
                tid(4)
            ),
        ),
        true => (
            "%struct.T = type { ptr addrspace(1) }\n%struct.R = type { %struct.T, %struct.T, %struct.T }\n\n"
                .to_string(),
            format!("ptr addrspace(2) %r, {tail}"),
            ["src", "dst", "other"]
                .iter()
                .enumerate()
                .map(|(i, name)| {
                    format!(
                        "  %p{name} = getelementptr inbounds %struct.R, ptr addrspace(2) %r, i64 0, i32 {i}, i32 0\n  %{name} = load ptr addrspace(1), ptr addrspace(2) %p{name}, align 8\n"
                    )
                })
                .collect(),
            format!(
                "!11 = !{{!19, !15, !16}}\n!12 = {}\n!13 = {}\n!14 = {}\n!15 = {}\n!16 = {}\n\
!19 = !{{i32 0, !\"air.indirect_buffer\", !\"air.buffer_size\", i32 24, !\"air.location_index\", i32 0, i32 1, !\"air.read\", !\"air.address_space\", i32 2, !\"air.struct_type_info\", !20, !\"air.arg_type_size\", i32 24, !\"air.arg_type_align_size\", i32 8, !\"air.arg_type_name\", !\"R\", !\"air.arg_name\", !\"r\"}}\n\
!20 = !{{i32 0, i32 8, i32 0, !\"texture2d<float, read>\", !\"src\", !\"air.indirect_argument\", !12, i32 8, i32 8, i32 0, !\"texture2d<float, write>\", !\"dst\", !\"air.indirect_argument\", !13, i32 16, i32 8, i32 0, !\"texture2d<float, read>\", !\"other\", !\"air.indirect_argument\", !14}}\n",
                textures[0],
                textures[1],
                textures[2],
                out(1, 1),
                tid(2)
            ),
        ),
    };
    format!(
        "target datalayout = \"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32\"\n\
target triple = \"air64_v28-apple-macosx26.0.0\"\n\n\
{types}\
define void @tex({params}) {{\n\
entry:\n\
{loads}\
  %s = tail call ptr addrspace(2) @air.get_read_sampler()\n\
  %r0 = tail call {{ <4 x float>, i8 }} @air.read_texture_2d.v4f32(ptr addrspace(1) %src, ptr addrspace(2) %s, <2 x i32> %tid, <2 x i32> zeroinitializer, i32 0, i32 1)\n\
  %v = extractvalue {{ <4 x float>, i8 }} %r0, 0\n\
  tail call void @air.write_texture_2d.v4f32(ptr addrspace(1) %dst, <2 x i32> %tid, <4 x float> %v, i32 0, i32 2)\n\
  %w = tail call i32 @air.get_width_texture_2d(ptr addrspace(1) %src, i32 0)\n\
  %n = tail call i1 @air.is_null_texture_2d(ptr addrspace(1) %other)\n\
  %nz = zext i1 %n to i32\n\
  %ns = shl i32 %nz, 16\n\
  %q = or i32 %w, %ns\n\
  %a = extractelement <4 x float> %v, i64 3\n\
  %ab = bitcast float %a to i32\n\
  %tx = extractelement <2 x i32> %tid, i64 0\n\
  %ty = extractelement <2 x i32> %tid, i64 1\n\
  %row = mul i32 %ty, 5\n\
  %idx = add i32 %row, %tx\n\
  %i0 = shl i32 %idx, 1\n\
  %i1 = or i32 %i0, 1\n\
  %e0 = zext i32 %i0 to i64\n\
  %e1 = zext i32 %i1 to i64\n\
  %p0 = getelementptr inbounds i32, ptr addrspace(1) %out, i64 %e0\n\
  %p1 = getelementptr inbounds i32, ptr addrspace(1) %out, i64 %e1\n\
  store i32 %q, ptr addrspace(1) %p0, align 4\n\
  store i32 %ab, ptr addrspace(1) %p1, align 4\n\
  ret void\n\
}}\n\n\
declare ptr addrspace(2) @air.get_read_sampler()\n\
declare {{ <4 x float>, i8 }} @air.read_texture_2d.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x i32>, <2 x i32>, i32, i32)\n\
declare void @air.write_texture_2d.v4f32(ptr addrspace(1), <2 x i32>, <4 x float>, i32, i32)\n\
declare i32 @air.get_width_texture_2d(ptr addrspace(1), i32)\n\
declare i1 @air.is_null_texture_2d(ptr addrspace(1))\n\n\
!air.kernel = !{{!9}}\n!air.version = !{{!17}}\n!air.language_version = !{{!18}}\n\
!9 = !{{ptr @tex, !10, !11}}\n!10 = !{{}}\n{args}\
!17 = !{{i32 2, i32 8, i32 0}}\n!18 = !{{!\"Metal\", i32 4, i32 0, i32 0}}\n"
    )
}

/// `stem`'s committed profile under the identity domains, without `drop`.
fn identity_without(stem: &str, drop: &[VkFeature]) -> DeviceProfile {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles")
        .join(format!("{stem}.json"));
    let text = std::fs::read_to_string(path).expect("profile");
    let mut desc = DeviceDescription::from_khronos_json(&text, None).expect("description");
    for f in drop {
        desc.features.remove(f);
    }
    Device::new(desc, identity())
        .expect("device")
        .profile()
        .clone()
}

/// A texture slot's push field: its resource ID, the index in the low word and the word in the
/// high one.
fn resource_id(index: u32, word: u32) -> u64 {
    u64::from(index) | u64::from(word) << 32
}

/// A `w × h` image of `slot` at element `element`, its texels `texel(x, y)`, and robust with
/// `robust`.
fn texture(
    slot: ResourceSlot,
    element: u32,
    [w, h]: [u32; 2],
    texel: impl Fn(u32, u32) -> [u32; 4],
    robust: Option<[u32; 4]>,
) -> Image {
    let mut img = Image::new(slot, [w, h, 1], 1, 1, 1);
    img.element = element;
    img.texels = (0..h)
        .flat_map(|y| (0..w).map(move |x| (x, y)))
        .map(|(x, y)| texel(x, y))
        .collect();
    img.robust = robust;
    img
}

/// A kernel's texture slots under the identity domains are elements of the two fixed set-1
/// arrays at the indices their pushed resource IDs hold: it reads and writes them there, queries
/// their size, answers `is_null_texture` by the index, and takes a format's missing alpha out of
/// bounds from the word, as the test vectors give it.
#[test]
fn texture_slots_are_the_set_1_elements_their_pushed_resource_ids_name() {
    textures_at_their_resource_ids(false);
}

/// An argument buffer's textures under the identity domains are the same elements, at the
/// indices of the resource IDs its members hold, with no handle table and no descriptor of the
/// argument buffer's own.
#[test]
fn argument_buffer_textures_are_the_set_1_elements_their_resource_ids_name() {
    textures_at_their_resource_ids(true);
}

/// The address of the argument buffer of [`texture_kernel`]`(true)`.
const AB: u64 = 0x4_0000_0100;

/// [`texture_kernel`]`(ab)` under the identity domains, on the CPU interpreter: its textures are
/// the set-1 elements their resource IDs name, pushed or, with `ab`, in the argument buffer.
fn textures_at_their_resource_ids(ab: bool) {
    let Some(bc) = assemble_text("tex", &texture_kernel(ab)) else {
        return;
    };
    const L: [u32; 3] = [5, 2, 1];
    // `src` at index 3 of the sampled array, `dst` at index 2 of the storage array, `other` at
    // index 1 or nil; the elements below theirs are other textures.
    const SRC: u32 = 3;
    const DST: u32 = 2;
    let rgba = vector_word(PixelFormat::RGBA8Unorm);
    let red = vector_word(PixelFormat::R8Unorm);
    let f = |v: f32| v.to_bits();
    let src_texel = |x: u32, y: u32| {
        [
            f((x + 1) as f32),
            f((y + 1) as f32),
            f(0.5),
            f((y * 4 + x + 1) as f32 / 8.0),
        ]
    };
    let filler = |_: u32, _: u32| [0xdead_beef; 4];
    let mut profiles: Vec<(String, DeviceProfile)> = PROFILES
        .iter()
        .map(|stem| {
            let p = profile(stem, identity()).unwrap_or_else(|e| panic!("{e}"));
            (stem.to_string(), p)
        })
        .collect();
    // Without `robustImageAccess2` the shader clamps the read and selects Metal's out-of-bounds
    // texel, whose alpha the word says; the images then trap on any access outside them.
    profiles.push((
        "moltenvk-m3 without robustImageAccess2".to_string(),
        identity_without("moltenvk-m3", &[VkFeature::RobustImageAccess2]),
    ));
    for (stem, p) in &profiles {
        let checked = !p.fields().images.robust_image_access2;
        let (a, plan, module) =
            compile_kept(&bc, "tex", p).unwrap_or_else(|e| panic!("{stem}: {e}"));
        link_and_validate(&a, p).unwrap_or_else(|e| panic!("{stem}: L3: {e}"));
        // An argument buffer's handle is its ID's index, with the word's missing-alpha bit where
        // the shader checks the read, and nothing else.
        let handles = match (ab, checked) {
            (false, _) => None,
            (true, false) => Some(Lookup::Element),
            (true, true) => Some(Lookup::Flagged {
                alpha: true,
                class: false,
                red_blue: false,
                ycbcr: false,
                unreadable: false,
            }),
        };
        let used = !plan.families.texture.heap.is_empty();
        assert_eq!(
            used.then(|| plan.families.access.lookup()),
            handles,
            "{stem}"
        );
        assert!(
            a.interface
                .spec
                .iter()
                .all(|c| !matches!(c.key, SpecKey::Presence(_) | SpecKey::OutOfBoundsAlpha(_))),
            "{stem}"
        );
        assert!(
            !plan.facts.iter().any(|f| matches!(
                f,
                RuntimeFact::ResourcePresence { .. } | RuntimeFact::OutOfBoundsAlpha { .. }
            )),
            "{stem}"
        );
        // Set 1 is the fixed pair of arrays; each class the kernel declares is a variable of its
        // descriptor type's binding, and set 0 holds nothing.
        let linked = link(&[&a], &LinkInputs::standalone(), p).unwrap_or_else(|e| panic!("{e}"));
        let sets: Vec<_> = (linked.layout.sets.iter())
            .map(|s| {
                let b: Vec<_> = (s.bindings.iter())
                    .map(|b| (b.binding, b.ty, b.count, b.source))
                    .collect();
                (s.set, b)
            })
            .collect();
        let arrays: Vec<_> = (identity_arrays().into_iter())
            .map(|(binding, ty, count)| (binding, ty, count, BindingSource::IdentityArray))
            .collect();
        assert_eq!(sets, [(0, vec![]), (1, arrays)], "{stem}");
        let class = |binding: u32| {
            let mut s = (linked.table.bindings.iter()).filter(|(_, b)| **b == (1, binding));
            let (slot, _) = s
                .next()
                .unwrap_or_else(|| panic!("{stem}: no class at {binding}"));
            assert!(s.next().is_none(), "{stem}: two classes at {binding}");
            *slot
        };
        let (sampled, storage) = (class(0), class(1));
        // Two dispatches: `src` of a format without alpha and `other` nil, then `src` with alpha
        // and `other` bound.
        for (word, other, alpha) in [(red, 0, 1.0f32), (rgba, resource_id(1, rgba), 0.0)] {
            let oob = [0, 0, 0, f(alpha)];
            let robust = (!checked).then_some(oob);
            let ids = [resource_id(SRC, word), resource_id(DST, rgba), other];
            let mut push = vec![0u8; plan.push.push_range.map_or(0, |(_, end)| end as usize)];
            let mut fields = Vec::new();
            for field in &plan.push.fields {
                let PushLoc::Push { offset } = field.loc else {
                    panic!("{stem}: {field:?} spilled");
                };
                let o = offset as usize;
                let bytes = match field.field {
                    PushField::SlotGuestBase(s) => match (ab, s.class, s.index) {
                        (false, SlotClass::Buffer, 0) | (true, SlotClass::Buffer, 1) => OUT,
                        (false, SlotClass::Texture, i @ 0..=2) => ids[i as usize],
                        (true, SlotClass::Buffer, 0) => AB,
                        _ => panic!("{stem}: unexpected slot {s:?}"),
                    },
                    PushField::DispatchRegion => {
                        let region = region_bytes([0; 3], L, L).expect("packs");
                        push[o..o + 32].copy_from_slice(&region);
                        fields.push(field.field);
                        continue;
                    }
                    _ => panic!("{stem}: unexpected push field {field:?}"),
                };
                push[o..o + 8].copy_from_slice(&bytes.to_le_bytes());
                fields.push(field.field);
            }
            assert_eq!(fields.len(), [5, 3][usize::from(ab)], "{stem}: {fields:?}");
            let mut images: Vec<Image> = (0..SRC)
                .map(|e| texture(sampled, e, [1, 1], filler, None))
                .collect();
            images.push(texture(sampled, SRC, [4, 2], src_texel, robust));
            images.extend((0..DST).map(|e| texture(storage, e, [1, 1], filler, None)));
            images.push(texture(storage, DST, [5, 2], |_, _| [0; 4], None));
            let mut buffers = vec![Buffer {
                slot: None,
                address: Some(OUT),
                bytes: vec![0; 4 * 21],
            }];
            if ab {
                buffers.push(Buffer {
                    slot: None,
                    address: Some(AB),
                    bytes: ids.iter().flat_map(|id| id.to_le_bytes()).collect(),
                });
            }
            let res = Resources {
                buffers,
                push,
                spec: local_size3(L),
            };
            let (got, images) = run_with_images(&module, &Config::default(), res, images)
                .unwrap_or_else(|e| panic!("{stem}: {e}"));
            let read = |x: u32, y: u32| match x < 4 {
                true => src_texel(x, y),
                false => oob,
            };
            let mut want = Vec::new();
            for y in 0..2 {
                for x in 0..5 {
                    let null = u32::from(other == 0);
                    want.extend_from_slice(&(4 | null << 16).to_le_bytes());
                    want.extend_from_slice(&read(x, y)[3].to_le_bytes());
                }
            }
            want.extend_from_slice(&[0; 4]);
            assert_eq!(got.buffers[0], want, "{stem}: out, alpha {alpha}");
            let dst = &images[(SRC + 1 + DST) as usize];
            let written: Vec<[u32; 4]> = (0..2)
                .flat_map(|y| (0..5).map(move |x| (x, y)))
                .map(|(x, y)| read(x, y))
                .collect();
            assert_eq!(dst.texels, written, "{stem}: dst, alpha {alpha}");
        }
    }
}

/// The read buffers of [`many`]: with its output, 21 slot addresses of 8 bytes, past the window.
const READS: u32 = 20;

/// [`many_of`] [`READS`] buffers.
fn many() -> String {
    many_of(READS)
}

/// A kernel reading one `uint` from each of `reads` buffers and storing their ordered fold
/// (`s · 3 + v`, so two swapped addresses change it) to the next buffer.
fn many_of(reads: u32) -> String {
    let mut params = Vec::new();
    let mut body = String::new();
    let mut meta = Vec::new();
    for k in 0..=reads {
        params.push(format!(
            "ptr addrspace(1) noundef \"air-buffer-no-alias\" %{k}"
        ));
        let (access, name) = match k == reads {
            true => ("air.read_write", "out".to_string()),
            false => ("air.read", format!("b{k}")),
        };
        meta.push(format!(
            "!{} = !{{i32 {k}, !\"air.buffer\", !\"air.location_index\", i32 {k}, i32 1, !\"{access}\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"{name}\"}}",
            100 + k
        ));
    }
    body.push_str("  %s0 = load i32, ptr addrspace(1) %0, align 4\n");
    for k in 1..reads {
        body.push_str(&format!(
            "  %v{k} = load i32, ptr addrspace(1) %{k}, align 4\n  %m{k} = mul i32 %s{}, 3\n  %s{k} = add i32 %m{k}, %v{k}\n",
            k - 1
        ));
    }
    let last = reads - 1;
    let args: Vec<String> = (0..=reads).map(|k| format!("!{}", 100 + k)).collect();
    format!(
        "target datalayout = \"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32\"\n\
target triple = \"air64_v28-apple-macosx26.0.0\"\n\n\
define void @many({}) {{\nentry:\n{body}  store i32 %s{last}, ptr addrspace(1) %{reads}, align 4\n  ret void\n}}\n\n\
!air.kernel = !{{!9}}\n!air.version = !{{!17}}\n!air.language_version = !{{!18}}\n\
!9 = !{{ptr @many, !10, !11}}\n!10 = !{{}}\n!11 = !{{{}}}\n{}\n\
!17 = !{{i32 2, i32 8, i32 0}}\n!18 = !{{!\"Metal\", i32 4, i32 0, i32 0}}\n",
        params.join(", "),
        args.join(", "),
        meta.join("\n"),
    )
}

/// What [`many_of`]`(reads)`'s placement on `p` pushes and spills: its first push field, the
/// slots whose addresses it pushes and those it reads from the block, and the block's length.
/// Checks the module passes L3, needs no descriptor, and computes its answer on the CPU
/// interpreter, its buffers above 4 GiB and its block where the pushed address says.
fn many_placed(
    stem: &str,
    p: &DeviceProfile,
    bc: &[u8],
    reads: u32,
) -> (
    Option<(PushField, PushLoc)>,
    Vec<SlotRef>,
    Vec<SlotRef>,
    u32,
) {
    // Buffer `k` at `BASE + k · STRIDE`, holding one word; the block at `BLOCK`.
    const BASE: u64 = 0x5_0000_0000;
    const STRIDE: u64 = 0x100;
    const BLOCK: u64 = 0x7_0000_0000;
    let value = |k: u32| 0x0101_0101u32.wrapping_mul(k + 1) ^ 0x5a5a;
    let want = (1..reads).fold(value(0), |s, k| s.wrapping_mul(3).wrapping_add(value(k)));
    let (a, plan, module) = compile_kept(bc, "many", p).unwrap_or_else(|e| panic!("{stem}: {e}"));
    assert!(a.interface.descriptors.is_empty(), "{stem}");
    assert!(!a.interface.needs_pointer_translation, "{stem}");
    link_and_validate(&a, p).unwrap_or_else(|e| panic!("{stem}: L3: {e}"));
    let first = plan.push.fields.first().map(|f| (f.field, f.loc));
    let mut push = vec![0u8; plan.push.push_range.map_or(0, |(_, end)| end as usize)];
    let mut block = vec![0u8; plan.push.argument_table_size as usize];
    let (mut pushed, mut spilled) = (Vec::new(), Vec::new());
    for f in &plan.push.fields {
        let (bytes, into, offset) = match (f.field, f.loc) {
            (PushField::ArgumentTableAddress, PushLoc::Push { offset }) => {
                (BLOCK, &mut push, offset)
            }
            (PushField::SlotGuestBase(s), PushLoc::Push { offset }) => {
                pushed.push(s);
                (BASE + u64::from(s.index) * STRIDE, &mut push, offset)
            }
            (PushField::SlotGuestBase(s), PushLoc::ArgumentTable { offset }) => {
                spilled.push(s);
                (BASE + u64::from(s.index) * STRIDE, &mut block, offset)
            }
            other => panic!("{stem}: unexpected push field {other:?}"),
        };
        let o = offset as usize;
        into[o..o + 8].copy_from_slice(&bytes.to_le_bytes());
    }
    let mut buffers: Vec<Buffer> = (0..=reads)
        .map(|k| Buffer {
            slot: None,
            address: Some(BASE + u64::from(k) * STRIDE),
            bytes: match k == reads {
                true => vec![0; 4],
                false => value(k).to_le_bytes().to_vec(),
            },
        })
        .collect();
    buffers.push(Buffer {
        slot: None,
        address: Some(BLOCK),
        bytes: block,
    });
    let res = Resources {
        buffers,
        push,
        spec: local_size(1),
    };
    let got = run(&module, &Config::default(), res).unwrap_or_else(|e| panic!("{stem}: {e}"));
    assert_eq!(
        got.buffers[reads as usize],
        want.to_le_bytes(),
        "{stem}: out"
    );
    (first, pushed, spilled, plan.push.argument_table_size)
}

/// A kernel whose slot addresses do not fit the push window reads the rest from its parameter
/// block, whose address the window holds first, and computes its answer there.
#[test]
fn slot_addresses_past_the_window_are_read_from_the_pushed_parameter_block() {
    let Some(bc) = assemble_text("many", &many()) else {
        return;
    };
    for stem in PROFILES {
        let p = profile(stem, identity()).unwrap_or_else(|e| panic!("{e}"));
        let (first, pushed, spilled, table) = many_placed(stem, &p, &bc, READS);
        assert_eq!(
            first,
            Some((PushField::ArgumentTableAddress, PushLoc::Push { offset: 0 })),
            "{stem}"
        );
        // The window's 120 bytes past the block address hold 15 slot addresses.
        assert_eq!(pushed.len(), 15, "{stem}");
        assert_eq!(spilled.len(), (READS + 1) as usize - 15, "{stem}");
        assert_eq!(table, 48, "{stem}");
    }
}

/// On `Device::from_queries`' profile a kernel's push window is 256 bytes (`PushWindow::Floor`):
/// the same kernel's 21 slot addresses are all pushed, with no block, and so are a kernel's 31,
/// Metal's most buffer slots (248 bytes). Each computes its answer.
#[test]
fn the_floor_window_pushes_every_slot_address_of_a_kernel_without_a_region() {
    for reads in [READS, 30] {
        let Some(bc) = assemble_text("many", &many_of(reads)) else {
            return;
        };
        for stem in PROFILES {
            let desc =
                crate::pipeline::testing::description(stem).unwrap_or_else(|e| panic!("{e}"));
            let device = Device::from_queries(desc, identity()).unwrap_or_else(|e| panic!("{e}"));
            let (first, pushed, spilled, table) = many_placed(stem, device.profile(), &bc, reads);
            let what = format!("{stem}, {} slots", reads + 1);
            assert_eq!(pushed.len(), (reads + 1) as usize, "{what}");
            assert!(spilled.is_empty(), "{what}");
            assert_eq!(table, 0, "{what}");
            assert_ne!(
                first.map(|(f, _)| f),
                Some(PushField::ArgumentTableAddress),
                "{what}"
            );
        }
    }
}

/// A kernel storing, at its position in the grid `y · grid.x + x`, the grid's threads along `x` and
/// `y` (bits 0-7, 8-15) and its threadgroups along `x` and `y` (bits 16-23, 24-31).
fn grid_kernel(name: &str) -> String {
    format!(
        "target datalayout = \"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32\"\n\
target triple = \"air64_v28-apple-macosx26.0.0\"\n\n\
define void @{name}(ptr addrspace(1) noundef \"air-buffer-no-alias\" %0, <3 x i32> noundef %1, <3 x i32> noundef %2, <3 x i32> noundef %3) {{\n\
entry:\n\
  %tx = extractelement <3 x i32> %1, i64 0\n\
  %ty = extractelement <3 x i32> %1, i64 1\n\
  %gx = extractelement <3 x i32> %2, i64 0\n\
  %gy = extractelement <3 x i32> %2, i64 1\n\
  %row = mul i32 %ty, %gx\n\
  %idx = add i32 %row, %tx\n\
  %hy = shl i32 %gy, 8\n\
  %v0 = or i32 %gx, %hy\n\
  %nx = extractelement <3 x i32> %3, i64 0\n\
  %ny = extractelement <3 x i32> %3, i64 1\n\
  %hx = shl i32 %nx, 16\n\
  %hn = shl i32 %ny, 24\n\
  %v1 = or i32 %v0, %hx\n\
  %v = or i32 %v1, %hn\n\
  %i = zext i32 %idx to i64\n\
  %p = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %i\n\
  store i32 %v, ptr addrspace(1) %p, align 4\n\
  ret void\n\
}}\n\n\
!air.kernel = !{{!9}}\n!air.version = !{{!17}}\n!air.language_version = !{{!18}}\n\
!9 = !{{ptr @{name}, !10, !11}}\n!10 = !{{}}\n!11 = !{{!12, !13, !14, !15}}\n\
!12 = !{{i32 0, !\"air.buffer\", !\"air.location_index\", i32 0, i32 1, !\"air.read_write\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"out\"}}\n\
!13 = !{{i32 1, !\"air.thread_position_in_grid\", !\"air.arg_type_name\", !\"uint3\", !\"air.arg_name\", !\"tid\"}}\n\
!14 = !{{i32 2, !\"air.threads_per_grid\", !\"air.arg_type_name\", !\"uint3\", !\"air.arg_name\", !\"grid\"}}\n\
!15 = !{{i32 3, !\"air.threadgroups_per_grid\", !\"air.arg_type_name\", !\"uint3\", !\"air.arg_name\", !\"groups\"}}\n\
!17 = !{{i32 2, i32 8, i32 0}}\n!18 = !{{!\"Metal\", i32 4, i32 0, i32 0}}\n"
    )
}

/// The push bytes of a [`grid_kernel`]: its output at `out` and the region `region`.
fn grid_push(plan: &Plan, out: u64, region: [u8; 32]) -> Vec<u8> {
    let mut push = vec![0u8; plan.push.push_range.map_or(0, |(_, end)| end as usize)];
    for f in &plan.push.fields {
        let PushLoc::Push { offset } = f.loc else {
            panic!("{f:?} spilled");
        };
        let o = offset as usize;
        match f.field {
            PushField::SlotGuestBase(s) if s == slot(0) => {
                push[o..o + 8].copy_from_slice(&out.to_le_bytes())
            }
            PushField::DispatchRegion => push[o..o + 32].copy_from_slice(&region),
            _ => panic!("unexpected push field {f:?}"),
        }
    }
    push
}

/// The output's address in the grid tests.
const OUT: u64 = 0x6_0000_0000;

/// A kernel's grid builtins read Metal's grid from the pushed region on every dispatch: whole,
/// split in two pieces, and indirect.
#[test]
fn every_dispatch_reads_metals_grid_from_its_pushed_region() {
    let Some(bc) = assemble_text("grid", &grid_kernel("grid")) else {
        return;
    };
    const L: [u32; 3] = [4, 2, 1];
    const GRID: [u32; 3] = [8, 6, 1];
    const GROUPS: [u32; 3] = [2, 3, 1];
    let threads = (GRID[0] * GRID[1]) as usize;
    let value = GRID[0] | GRID[1] << 8 | GROUPS[0] << 16 | GROUPS[1] << 24;
    let want: Vec<u8> = (0..threads).flat_map(|_| value.to_le_bytes()).collect();
    // Each form's dispatches: (region base, region grid, workgroups).
    type Dispatch = ([u32; 3], [u32; 3], [u32; 3]);
    let forms: [(&str, &[Dispatch]); 3] = [
        ("whole", &[([0; 3], GRID, GROUPS)]),
        (
            "split",
            &[([0; 3], GRID, [2, 1, 1]), ([0, 1, 0], GRID, [2, 2, 1])],
        ),
        ("indirect", &[([0; 3], [0; 3], GROUPS)]),
    ];
    for stem in PROFILES {
        let p = profile(stem, identity()).unwrap_or_else(|e| panic!("{e}"));
        let (a, plan, module) =
            compile_kept(&bc, "grid", &p).unwrap_or_else(|e| panic!("{stem}: {e}"));
        assert!(
            a.interface
                .spec
                .iter()
                .all(|c| c.key != SpecKey::MultiRegion),
            "{stem}"
        );
        link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{stem}: L3: {e}"));
        for (what, dispatches) in forms {
            let mut out = vec![0u8; 4 * threads];
            for &(base, grid, workgroups) in dispatches {
                let region = region_bytes(base, grid, L).expect("packs");
                let res = Resources {
                    buffers: vec![Buffer {
                        slot: None,
                        address: Some(OUT),
                        bytes: out,
                    }],
                    push: grid_push(&plan, OUT, region),
                    spec: local_size3(L),
                };
                let cfg = Config {
                    workgroups,
                    ..Config::default()
                };
                let got = run(&module, &cfg, res).unwrap_or_else(|e| panic!("{stem} {what}: {e}"));
                out = got.buffers.into_iter().next().expect("out");
            }
            assert_eq!(out, want, "{stem} {what}");
        }
    }
}

/// A kernel storing, at its position in the grid `y · 7 + x` (the grid is 7 wide), its
/// threadgroup's threads along `x` and `y` (bits 0-7, 8-15).
fn partial_kernel() -> String {
    "target datalayout = \"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32\"\n\
target triple = \"air64_v28-apple-macosx26.0.0\"\n\n\
define void @partial(ptr addrspace(1) noundef \"air-buffer-no-alias\" %0, <3 x i32> noundef %1, <3 x i32> noundef %2) {\n\
entry:\n\
  %tx = extractelement <3 x i32> %1, i64 0\n\
  %ty = extractelement <3 x i32> %1, i64 1\n\
  %row = mul i32 %ty, 7\n\
  %idx = add i32 %row, %tx\n\
  %sx = extractelement <3 x i32> %2, i64 0\n\
  %sy = extractelement <3 x i32> %2, i64 1\n\
  %hy = shl i32 %sy, 8\n\
  %v = or i32 %sx, %hy\n\
  %i = zext i32 %idx to i64\n\
  %p = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %i\n\
  store i32 %v, ptr addrspace(1) %p, align 4\n\
  ret void\n\
}\n\n\
!air.kernel = !{!9}\n!air.version = !{!17}\n!air.language_version = !{!18}\n\
!9 = !{ptr @partial, !10, !11}\n!10 = !{}\n!11 = !{!12, !13, !14}\n\
!12 = !{i32 0, !\"air.buffer\", !\"air.location_index\", i32 0, i32 1, !\"air.read_write\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"out\"}\n\
!13 = !{i32 1, !\"air.thread_position_in_grid\", !\"air.arg_type_name\", !\"uint3\", !\"air.arg_name\", !\"tid\"}\n\
!14 = !{i32 2, !\"air.threads_per_threadgroup\", !\"air.arg_type_name\", !\"uint3\", !\"air.arg_name\", !\"size\"}\n\
!17 = !{i32 2, i32 8, i32 0}\n!18 = !{!\"Metal\", i32 4, i32 0, i32 0}\n"
        .to_string()
}

/// A partial-threadgroups variant over a ragged grid, its region flagged `NONUNIFORM`, answers
/// each edge threadgroup's own size, and its invocations past the grid store nothing.
#[test]
fn a_partial_threadgroups_variant_answers_the_edge_threadgroups_of_a_ragged_grid() {
    let Some(bc) = assemble_text("partial", &partial_kernel()) else {
        return;
    };
    const L: [u32; 3] = [4, 2, 1];
    const GRID: [u32; 3] = [7, 5, 1];
    // 2 x 3 threadgroups of 8: 48 invocations over 35 threads, whose stores past the grid along
    // `x` would land on the next row's first word, and along `y` past the 35 words.
    let words = 48usize;
    let mut want = vec![0u8; 4 * words];
    for y in 0..GRID[1] {
        for x in 0..GRID[0] {
            let sx = L[0].min(GRID[0] - x / L[0] * L[0]);
            let sy = L[1].min(GRID[1] - y / L[1] * L[1]);
            let at = 4 * (y * GRID[0] + x) as usize;
            want[at..at + 4].copy_from_slice(&(sx | sy << 8).to_le_bytes());
        }
    }
    let mut inputs = LowerInputs::new();
    inputs.partial_threadgroups = true;
    for stem in PROFILES {
        let p = profile(stem, identity()).unwrap_or_else(|e| panic!("{e}"));
        let a = compile_with(&bc, "partial", &p, &inputs).unwrap_or_else(|e| panic!("{stem}: {e}"));
        link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{stem}: L3: {e}"));
        let (plan, module) = typed_module_with(&bc, "partial", &p, &inputs)
            .unwrap_or_else(|e| panic!("{stem}: {e}"));
        let mut region = region_bytes([0; 3], GRID, L).expect("packs");
        region[28..32].copy_from_slice(&NONUNIFORM.to_le_bytes());
        let res = Resources {
            buffers: vec![Buffer {
                slot: None,
                address: Some(OUT),
                bytes: vec![0; 4 * words],
            }],
            push: grid_push(&plan, OUT, region),
            spec: local_size3(L),
        };
        let cfg = Config {
            workgroups: [2, 3, 1],
            ..Config::default()
        };
        let got = run(&module, &cfg, res).unwrap_or_else(|e| panic!("{stem}: {e}"));
        assert_eq!(got.buffers[0], want, "{stem}");
    }
}

/// The `OpControlBarrier`s of disassembly `text`: how many open a block that only a branch on
/// `1 < N` (an `OpSpecConstantOp ULessThan` of the constant 1) enters, and how many others.
fn barrier_guards(text: &str) -> (usize, usize) {
    let mut conditions = BTreeSet::new();
    let mut guarded = BTreeSet::new();
    let mut block = "";
    let mut first = false;
    let (mut inside, mut outside) = (0, 0);
    for line in text.lines() {
        let words: Vec<&str> = line.split_whitespace().collect();
        match words[..] {
            [id, "=", "OpSpecConstantOp", "%bool", "ULessThan", "%uint_1", _] => {
                conditions.insert(id);
            }
            [label, "=", "OpLabel"] => {
                block = label;
                first = true;
                continue;
            }
            ["OpBranchConditional", condition, then, _] if conditions.contains(condition) => {
                guarded.insert(then);
            }
            ["OpControlBarrier", ..] if first && guarded.contains(block) => inside += 1,
            ["OpControlBarrier", ..] => outside += 1,
            _ => {}
        }
        first = false;
    }
    (inside, outside)
}

/// The input of thread `t` of the simdgroup products: 2, 0.5 and 1 in both signs, so every
/// product is exact in any order.
fn simd_input(t: u32) -> f32 {
    let lane = t % 32;
    let sign = if lane % 2 == 0 { 1.0 } else { -1.0 };
    match lane {
        0..8 => 2.0 * sign,
        8..14 => 0.5 * sign,
        _ if lane % 3 == 0 => -1.0,
        _ => 1.0,
    }
}

/// Under the identity domains the emulated tier's barriers run only in a threadgroup of more
/// than one thread: each opens a block a branch on `1 < N` enters, a threadgroup of one passes
/// none, and threadgroups of one and of 80 threads answer Metal's products, the last simdgroup of
/// 80 holding 16 threads. The default domains keep every barrier unconditional.
#[test]
fn the_emulated_tiers_barriers_run_only_in_a_threadgroup_of_several_threads() {
    let Some(bc) = assemble("gsm2_simd_products") else {
        return;
    };
    const ENTRY: &str = "simd_float_products";
    const INPUT: u64 = 0x5_0000_0000;
    let mut inputs = LowerInputs::new();
    inputs.emulated_simdgroups = true;
    for stem in PROFILES {
        let plain = profile(stem, DevicePolicy::default()).unwrap_or_else(|e| panic!("{e}"));
        let a = compile_with(&bc, ENTRY, &plain, &inputs).unwrap_or_else(|e| panic!("{stem}: {e}"));
        let (_, words) = link_and_validate(&a, &plain).unwrap_or_else(|e| panic!("{stem}: {e}"));
        let (inside, outside) = barrier_guards(&spirv_dis(&words));
        assert!(
            inside == 0 && outside > 0,
            "{stem}: the default domains guard {inside} barriers and leave {outside}"
        );
        let p = profile(stem, identity()).unwrap_or_else(|e| panic!("{e}"));
        let a = compile_with(&bc, ENTRY, &p, &inputs).unwrap_or_else(|e| panic!("{stem}: {e}"));
        let (_, words) = link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{stem}: L3: {e}"));
        let text = spirv_dis(&words);
        let (inside, outside) = barrier_guards(&text);
        assert!(
            inside > 0 && outside == 0,
            "{stem}: {inside} barriers guarded, {outside} not\n{text}"
        );
        let (plan, module) =
            typed_module_with(&bc, ENTRY, &p, &inputs).unwrap_or_else(|e| panic!("{stem}: {e}"));
        for n in [1u32, 80] {
            let input: Vec<u8> = (0..n).flat_map(|t| simd_input(t).to_le_bytes()).collect();
            let mut want = Vec::new();
            for t in 0..n {
                let base = t - t % 32;
                let group = (base..n.min(base + 32)).map(simd_input);
                let product: f32 = group.clone().product();
                let inclusive: f32 = group.clone().take((t - base + 1) as usize).product();
                let exclusive: f32 = group.clone().take((t - base) as usize).product();
                let negated: f32 = group.map(|x| -x).product();
                want.extend([product, inclusive, exclusive, negated].map(f32::to_bits));
            }
            let region = region_bytes([0; 3], [n, 1, 1], [n, 1, 1]).expect("packs");
            let mut push = vec![0u8; plan.push.push_range.map_or(0, |(_, end)| end as usize)];
            for f in &plan.push.fields {
                let PushLoc::Push { offset } = f.loc else {
                    panic!("{f:?} spilled");
                };
                let o = offset as usize;
                match f.field {
                    PushField::SlotGuestBase(s) if s == slot(0) => {
                        push[o..o + 8].copy_from_slice(&INPUT.to_le_bytes())
                    }
                    PushField::SlotGuestBase(s) if s == slot(1) => {
                        push[o..o + 8].copy_from_slice(&OUT.to_le_bytes())
                    }
                    PushField::DispatchRegion => push[o..o + 32].copy_from_slice(&region),
                    _ => panic!("unexpected push field {f:?}"),
                }
            }
            let res = Resources {
                buffers: vec![
                    Buffer {
                        slot: None,
                        address: Some(INPUT),
                        bytes: input,
                    },
                    Buffer {
                        slot: None,
                        address: Some(OUT),
                        bytes: vec![0; 16 * n as usize],
                    },
                ],
                push,
                spec: local_size(n),
            };
            for order in [Order::Forward, Order::Reverse] {
                let cfg = Config {
                    order,
                    barriers: BarrierRule::Vulkan,
                    ..Config::default()
                };
                let got = run(&module, &cfg, res.clone())
                    .unwrap_or_else(|e| panic!("{stem} {n} {order:?}: {e}"));
                let words: Vec<u32> = got.buffers[1]
                    .chunks(4)
                    .map(|w| u32::from_le_bytes(w.try_into().expect("a word")))
                    .collect();
                assert_eq!(words, want, "{stem}: a threadgroup of {n} {order:?}");
                let passed = got.invocations.iter().map(|i| i.barriers).max();
                match n {
                    1 => assert_eq!(passed, Some(0), "{stem}: one thread passes no barrier"),
                    _ => assert!(passed > Some(0), "{stem}: {n} threads exchange"),
                }
            }
        }
    }
}

/// A kernel adding the `constant` word `n` (slot 2) to the word at `data + n` (slot 0) and storing
/// the sum to `out` (slot 1); `data` carries `air-buffer-no-alias` where `unaliased`, and the sum
/// is stored back to `data` too where `data_written`. `n` carries no attribute.
fn read_only_kernel(unaliased: bool, data_written: bool) -> String {
    let attribute = if unaliased {
        " \"air-buffer-no-alias\""
    } else {
        ""
    };
    let write_back = if data_written {
        "  store i32 %s, ptr addrspace(1) %0, align 4\n"
    } else {
        ""
    };
    format!(
        "target datalayout = \"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32\"\n\
target triple = \"air64_v28-apple-macosx26.0.0\"\n\n\
define void @read_only(ptr addrspace(1) noundef{attribute} %0, ptr addrspace(1) noundef \"air-buffer-no-alias\" %1, ptr addrspace(2) noundef %2) {{\n\
entry:\n\
  %n = load i32, ptr addrspace(2) %2, align 4\n\
  %p = getelementptr i8, ptr addrspace(1) %0, i32 %n\n\
  %v = load i32, ptr addrspace(1) %p, align 4\n\
  %s = add i32 %v, %n\n\
  store i32 %s, ptr addrspace(1) %1, align 4\n\
{write_back}  ret void\n\
}}\n\n\
!air.kernel = !{{!9}}\n!air.version = !{{!17}}\n!air.language_version = !{{!18}}\n\
!9 = !{{ptr @read_only, !10, !11}}\n!10 = !{{}}\n!11 = !{{!12, !13, !14}}\n\
!12 = !{{i32 0, !\"air.buffer\", !\"air.location_index\", i32 0, i32 1, !\"air.read_write\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"data\"}}\n\
!13 = !{{i32 1, !\"air.buffer\", !\"air.location_index\", i32 1, i32 1, !\"air.read_write\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"out\"}}\n\
!14 = !{{i32 2, !\"air.buffer\", !\"air.buffer_size\", i32 4, !\"air.location_index\", i32 2, i32 1, !\"air.read\", !\"air.address_space\", i32 2, !\"air.arg_type_size\", i32 4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"n\"}}\n\
!17 = !{{i32 2, i32 8, i32 0}}\n!18 = !{{!\"Metal\", i32 4, i32 0, i32 0}}\n"
    )
}

/// The device-address loads of serialized module `words`: how many go through a `NonWritable`
/// place (a member of a block decorated `NonWritable`) and how many through a writable one, and
/// how many stores go through a `NonWritable` place.
fn device_address_loads(words: &[u32]) -> (usize, usize, usize) {
    const MEMBER_DECORATE: u32 = 72;
    const NON_WRITABLE: u32 = 24;
    const TYPE_POINTER: u32 = 32;
    const PHYSICAL_STORAGE_BUFFER: u32 = 5349;
    const CONVERT_U_TO_PTR: u32 = 120;
    const ACCESS_CHAIN: u32 = 65;
    const LOAD: u32 = 61;
    const STORE: u32 = 62;
    let insts = || {
        let mut at = 5;
        std::iter::from_fn(move || {
            let w = *words.get(at)?;
            let len = ((w >> 16) as usize).max(1);
            let inst = &words[at..(at + len).min(words.len())];
            at += len;
            Some((w & 0xFFFF, inst))
        })
    };
    let read_only_blocks: Vec<u32> = insts()
        .filter(|(op, i)| *op == MEMBER_DECORATE && i.get(3) == Some(&NON_WRITABLE))
        .map(|(_, i)| i[1])
        .collect();
    // Each device-address pointer type, whether its block is `NonWritable`.
    let pointers: BTreeMap<u32, bool> = insts()
        .filter(|(op, i)| *op == TYPE_POINTER && i.get(2) == Some(&PHYSICAL_STORAGE_BUFFER))
        .map(|(_, i)| (i[1], read_only_blocks.contains(&i[3])))
        .collect();
    let mut bases = BTreeMap::new();
    let mut places = BTreeMap::new();
    let (mut read_only, mut writable, mut stores) = (0, 0, 0);
    for (op, i) in insts() {
        match op {
            CONVERT_U_TO_PTR => {
                if let Some(&ro) = pointers.get(&i[1]) {
                    bases.insert(i[2], ro);
                }
            }
            ACCESS_CHAIN => {
                if let Some(&ro) = bases.get(&i[3]) {
                    places.insert(i[2], ro);
                }
            }
            LOAD => match places.get(&i[3]) {
                Some(true) => read_only += 1,
                Some(false) => writable += 1,
                None => {}
            },
            STORE => stores += usize::from(places.get(&i[1]) == Some(&true)),
            _ => {}
        }
    }
    (read_only, writable, stores)
}

/// Under the identity domains a slot nothing in the kernel writes is loaded through
/// `NonWritable` device-address places, so the device may read it as read-only memory (RADV reads
/// a uniform address of it with a scalar load): a `constant` slot, aliased or not, and a `device`
/// slot the AIR declares unaliased that no store reaches. A `device` slot without the attribute,
/// or one the kernel also writes, is loaded through a writable place, and no store goes through a
/// `NonWritable` one. Each kernel passes L3 and computes its sum on the CPU interpreter.
#[test]
fn unwritten_slots_are_loaded_through_non_writable_places() {
    // (case, data unaliased, data written, the unwritten slots, read-only and writable loads)
    let cases = [
        ("read_only", true, false, vec![0u32, 2], (2, 0)),
        ("data_aliased", false, false, vec![2], (1, 1)),
        ("data_written", true, true, vec![2], (1, 1)),
    ];
    const N: u32 = 8;
    const AT: [u64; 3] = [0x1_0000_0000, 0x2_0000_1000, 0x3_0000_2000];
    let mut data = vec![0u8; 16];
    data[8..12].copy_from_slice(&0x1234_5678u32.to_le_bytes());
    let want = 0x1234_5678u32 + N;
    for (case, unaliased, written, slots, loads) in cases {
        let Some(bc) = assemble_text(case, &read_only_kernel(unaliased, written)) else {
            return;
        };
        for stem in PROFILES {
            let what = format!("{stem} {case}");
            let p = profile(stem, identity()).unwrap_or_else(|e| panic!("{e}"));
            let (a, plan, module) =
                compile_kept(&bc, "read_only", &p).unwrap_or_else(|e| panic!("{what}: {e}"));
            link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
            let unwritten: Vec<u32> = plan
                .families
                .access
                .unwritten
                .iter()
                .filter_map(|&i| plan.slots.get(i).copied().flatten())
                .map(|c| c.slot.index)
                .collect();
            assert_eq!(unwritten, slots, "{what}: unwritten slots");
            let (read_only, writable, stores) = device_address_loads(&a.module.words);
            assert_eq!((read_only, writable), loads, "{what}: loads");
            assert_eq!(stores, 0, "{what}: a store through a NonWritable place");
            let mut push = vec![0u8; plan.push.push_range.map_or(0, |(_, end)| end as usize)];
            for f in &plan.push.fields {
                let PushLoc::Push { offset } = f.loc else {
                    panic!("{what}: {f:?} spilled");
                };
                let o = offset as usize;
                match f.field {
                    PushField::SlotGuestBase(s) => {
                        push[o..o + 8].copy_from_slice(&AT[s.index as usize].to_le_bytes());
                    }
                    PushField::DispatchRegion => {
                        let region = region_bytes([0; 3], [1, 1, 1], [1, 1, 1]).expect("packs");
                        push[o..o + 32].copy_from_slice(&region);
                    }
                    _ => panic!("{what}: unexpected push field {f:?}"),
                }
            }
            let buffer = |at: u64, bytes: Vec<u8>| Buffer {
                slot: None,
                address: Some(at),
                bytes,
            };
            let res = Resources {
                buffers: vec![
                    buffer(AT[0], data.clone()),
                    buffer(AT[1], vec![0; 4]),
                    buffer(AT[2], N.to_le_bytes().to_vec()),
                ],
                push,
                spec: local_size(1),
            };
            let got =
                run(&module, &Config::default(), res).unwrap_or_else(|e| panic!("{what}: {e}"));
            assert_eq!(got.buffers[1], want.to_le_bytes(), "{what}: out");
        }
    }
}
