//! The identity domains (`PointerDomain::Identity` with `HandleDomain::Identity`) in the lowering.
//!
//! - A kernel over buffers alone (`engine/a1_bool_from_float`) compiles with each slot reached
//!   through its own device address, pushed (`SlotGuestBase`), with no root table and no
//!   descriptor. It passes L3, and on the CPU interpreter, with its buffers above 4 GiB and bound
//!   at offsets that are not 0, it computes the program's answer and writes nothing else.
//! - In the kernel window of 256 bytes a kernel of 21 slots (`many`, generated: 20 read buffers
//!   and an output) pushes all 21 addresses with no block, below its tail word; one of 31 slots,
//!   Metal's most, pushes its parameter block's address (`ArgumentTableAddress`) and 27 slot
//!   addresses below the tail word at 248, and reads the other 4 from the block at their placed
//!   offsets, with no descriptor. In the window of 128 bytes (`runtime-floor`) one of 12 slots
//!   pushes all 12, and ones of 21 and 31 push the block's address and 11 below the word at 120.
//!   Each passes L3 and computes its answer on the CPU interpreter.
//! - A kernel that reads grid builtins (`grid`, generated) declares `MultiRegion`, default 0, and
//!   reads Metal's grid from its pushed `DispatchRegion` and its base from every region: at 0 a
//!   whole dispatch, one split in two pieces at a base, and an indirect one (a zero grid, its grid
//!   `NumWorkgroups · L`) store the same answers, and a ragged grid split into a region of its
//!   threadgroup size and an edge region of fewer threads, at 1 with Metal's threadgroup size in
//!   the region, stores Metal's answers.
//! - A kernel's partial-threadgroups variant (`partial`, generated) dispatched at its whole
//!   threadgroup size over a ragged grid, its pushed region flagged `NONUNIFORM`, answers each
//!   edge threadgroup's own size, and its invocations past the grid store nothing.
//! - The partial-threadgroups variant of a kernel that does not observe its threadgroup partition
//!   (`edge`, generated) runs its grid in tiles (`RuntimeFact::GridTiles`), its module holds
//!   no division by a value the driver does not know, and over grids ragged along every axis,
//!   along one, and whole, in `grid_tile`'s tiles, it stores each thread's Metal position once and
//!   nothing past the grid; an indirect dispatch runs at its own size.
//!   `Device::kernel_observes_partition` answers what its compile does.
//! - A kernel's texture slots (`tex`, generated: a read, a write, a width and an `is_null_texture`)
//!   are the elements of the two fixed set-1 arrays at the indices their pushed resource IDs hold,
//!   with no descriptor of their own. It passes L3 and, on the CPU interpreter, reads and writes
//!   the textures there, answers the width and `is_null_texture` by the index (0 for nil), and,
//!   where the shader checks reads (no `robustImageAccess2`), takes a format's missing alpha out of
//!   bounds from the word, as `tests/resource_words.tsv` gives it.
//! - A kernel's texture buffers (`engine/ob1_texel_buffer`, `engine/texture_ops`' `texel_buffers`,
//!   and an argument buffer's in `engine/rlm_runtime_limits`' `rlm_member_texture_buffer`) are the
//!   elements of set 1's texel-buffer arrays at the indices their resource IDs hold, a read one
//!   the uniform array's and a written one the storage array's, with no fact of the bound texture
//!   buffer. Each passes L3.
//! - A kernel's texture atomic (`ta`, generated: `atomic_fetch_add` on a `texture2d<uint,
//!   access::read_write>`) declares its slot `R32ui`, a class whose variable is the storage-image
//!   array's. It passes L3 and, on the CPU interpreter, adds to the texels of the element its
//!   pushed resource ID names and answers the texels they held, guarded by the element's size on
//!   a device without robust image access.
//! - A kernel's plain `ulong` texture slots (`engine/unk_ulong_rw`'s `unk_ulong_access`: two
//!   reads and a write) are `RG32` unsigned images, the sampled array's unsigned class and the
//!   storage array's declared `Rg32ui`. It passes L3 and, on the CPU interpreter, joins each
//!   read's two words and splits the write's at the elements their pushed resource IDs name.
//! - A slot's read through a runtime sampler that clamps to its border color
//!   (`engine/kernel_read_through`'s `rt_arg`) takes the border texels from bits 0-11 of the
//!   slot's resource ID's word, with no `BorderTexels` specialization or fact. It passes L3 and,
//!   on the CPU interpreter, reads Metal's border texel of `R32Float`, `RGBA8Unorm`, `A8Unorm`
//!   and a swizzled `RGBA8Unorm` under each border color, and the texel inside the texture.
//! - The same kernel with its textures in an argument buffer (`tex` with `ab`) reads their
//!   resource IDs from the buffer's members, at its pushed address, and answers the same at the
//!   same elements, with no handle table and no descriptor; IDs whose indices are 65,536 past
//!   the arrays read the elements at their low 16 bits.
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
//! - A storage slot read without a format (`engine/rrp_swizzled_storage`'s `rrp_sw_rwread_f`) on
//!   the Iris Xe, which reads storage so by the formats' own bits alone, is declared without a
//!   format where every read-write tier format has the read bit, and refused where one lacks it;
//!   the resident-table domains' answer there is unchanged.
//! - A kernel that samples 2D textures through a bicubic constexpr sampler at normalized
//!   coordinates (`engine/bcu_formats`, `float` and `half` results) reads how the filter treats
//!   each slot's bound format from the slot's pushed field (`PushField::BicubicTexels`): no
//!   specialization constant and no fact, where the resident-table domains specialize
//!   `SpecKey::BicubicTexels` and state `RuntimeFact::BicubicTexels` as before. It passes L3. So
//!   does one that samples at pixel coordinates too (`engine/bcu_address`). A bicubic sample of a
//!   1D texture (`engine/bcu_forms`) is still refused as `Unsupported{IdentityDomains}`, not
//!   implemented.
//! - A kernel that samples, compares or fetches through a pixel sampler a texture its argument
//!   buffer holds, through a runtime sampler slot or a constexpr sampler (`engine/handles`' heap
//!   samplings, `examples/heap_samples`' `hs_sample`, and a generated comparison, alone and beside
//!   a sample of the same depth texture without comparison), takes the
//!   texture from its class's set-1 array at the index the member's resource ID holds and the
//!   sampler from the set-1 sampler array of its use at the index the slot's pushed resource ID or
//!   the constexpr sampler's pushed index names, with no heap sampler binding and nothing in set
//!   0; each passes L3. Constexpr samplers a lane chooses among, where they are one array of
//!   samplers, are still refused as `Unsupported{IdentityDomains}`, not implemented, naming that
//!   form.
//! - A kernel that samples an argument buffer's texture through the sampler the argument buffer
//!   holds beside it (`engine/tail_argument_samplers`, `engine/tail_argument_cube_gradient`) takes
//!   the sampler's resource ID as its handle: the element of the set-1 sampler array of its use at
//!   the ID's index, element 0 where the index is past the arrays, and the word's fields from the
//!   ID's high word, with no sampler table, no execution status address and no cubic test; each
//!   passes L3.
//! - A kernel that encodes compute commands into an indirect command buffer it reaches through an
//!   argument buffer (`engine/icb_replay_encode`) is refused as
//!   `Unsupported{GpuEncodedIndirectCommands}`, not implemented: the feature it uses, not the
//!   domains' generic refusal.
//! - A kernel that reads the compute encoder's stage-in region (`engine/kernel_stage_in`'s
//!   `stage_in_region_short`: `[[grid_origin]]` and `[[grid_size]]`, with no `[[stage_in]]`)
//!   reads it from its pushed field (`PushField::StageInRegion`), with no descriptor; it passes
//!   L3 and, on the CPU interpreter, every thread stores the region's origin and size.
//! - A kernel's emulated-simdgroup variant (`engine/gsm2_simd_products`'s `simd_float_products`)
//!   opens every barrier of its exchanges under a branch on `1 < N`, which the resident-table
//!   domains do not; on the CPU interpreter a threadgroup of one thread passes no barrier, and
//!   threadgroups of one and of 80 threads (a partial simdgroup of 16) answer Metal's products.
//! - A kernel that reads program-scope constants through memory (`blob_words`, generated: a
//!   `ushort3` constant read as `half3`, as `as_type` folds into the load, and a word table read
//!   as 2-byte units at offsets only the run knows) holds the constant blob as the module's own
//!   `Private` words: no descriptor, no blob in the artifact, the words counted in its private
//!   bytes, and on the CPU interpreter each value read, and zero for a read past the words. The
//!   resident-table domains bind the same blob as before. A blob past `memory::blob::MODULE_BYTES`
//!   (`blob_past`, generated) is refused as `Unsupported{IdentityDomains}` naming its size.
//!
//! `llvm-as` assembles the kernels; without it the tests skip visibly.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicU64, Ordering};

use super::engine_word_phases::{bounds_word, slot_word, ZERO_UNMAPPED};
use crate::budget::Budget;
use crate::device::profile::{DeviceProfile, HandleDomain, PointerDomain};
use crate::device::requirements::VkFeature;
use crate::device::{Device, DeviceDescription, DevicePolicy};
use crate::error::{Error, Feature, Needs};
use crate::library::Library;
use crate::lower::dispatch::{nonuniform_region_bytes, region_bytes, NONUNIFORM};
use crate::lower::select::{LowerInputs, Plan, BICUBIC_LINE_FORM};
use crate::memory::handles::Lookup;
use crate::metal::resource_word::TextureWord;
use crate::metal::{PixelFormat, TextureSwizzleChannels};
use crate::pipeline::facts::RuntimeFact;
use crate::pipeline::interface::PushLoc;
use crate::pipeline::layout::BindingSource;
use crate::pipeline::link::{link, LinkInputs};
use crate::pipeline::sym::{
    Axis, HandleClass, ImageAccess, ImageDim, PushField, ResourceKind, ResourceSlot, SampledType,
    SlotClass, SlotRef, SpecKey, Stage, StorageFormatClass,
};
use crate::pipeline::testing::{
    compile, compile_kept, compile_with, engine_dir, link_and_validate, llvm_assemble, profile,
    typed_module_with, AssembleError,
};
use crate::pipeline::{ComputePipelineDescriptor, FunctionDescriptor, StageSource};
use crate::spirv::ids::ConstId;
use crate::spirv::interp::{
    run, run_with_images, BarrierRule, Buffer, Config, Image, Order, Resources,
};
use crate::spirv::ir::{ConstKind, TypedModule};
use crate::spirv::testing::spirv_dis;
use crate::vk;

/// Profiles with `bufferDeviceAddress` and `shaderInt64`.
const PROFILES: [&str; 2] = ["lavapipe", "moltenvk-m3"];

fn identity() -> DevicePolicy {
    DevicePolicy {
        pointer_domain: PointerDomain::Identity,
        handle_domain: HandleDomain::Identity,
        ..DevicePolicy::RESIDENT_TABLES
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
                    push[o..o + 8].copy_from_slice(&slot_word(base + bind).to_le_bytes());
                    read.push(s);
                }
                PushField::SlotBounds(s) => {
                    let (_, bind) = at[s.index as usize];
                    let bytes = [bits.len(), halves.len(), out.len()][s.index as usize];
                    push[o..o + 8].copy_from_slice(&bounds_word(bytes, bind));
                }
                PushField::ZeroAddress => {
                    push[o..o + 8].copy_from_slice(&ZERO_UNMAPPED.to_le_bytes())
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

/// A slot pointer used as a value under the identity domains is its Metal address, though the
/// runtime pushes each buffer slot's address less 2^31 (`lower::access::SLOT_ADDRESS_BIAS`): two
/// address compares add the bias back, and the loads beside them, one 4 bytes below its slot's
/// bind address, add their offsets plus the bias to the word. The two slots lie on either side of
/// 2^31, so their words lie on either side of 0 and a compare of the words would answer the other
/// way.
#[test]
fn a_slot_pointer_used_as_a_value_is_its_metal_address() {
    use super::engine_pointer_memory::{assemble, kernel};
    let body = "  %p = getelementptr inbounds i8, ptr addrspace(1) %b0, i64 20
  %q = getelementptr inbounds i8, ptr addrspace(1) %b1, i64 -4
  %lt = icmp ult ptr addrspace(1) %b0, %b1
  %z = zext i1 %lt to i32
  store i32 %z, ptr addrspace(1) %b2, align 4
  %gt = icmp ugt ptr addrspace(1) %p, %q
  %y = zext i1 %gt to i32
  %o1 = getelementptr inbounds i32, ptr addrspace(1) %b2, i64 1
  store i32 %y, ptr addrspace(1) %o1, align 4
  %w = load i32, ptr addrspace(1) %q, align 4
  %o2 = getelementptr inbounds i32, ptr addrspace(1) %b2, i64 2
  store i32 %w, ptr addrspace(1) %o2, align 4
  %v = load i32, ptr addrspace(1) %p, align 4
  %o3 = getelementptr inbounds i32, ptr addrspace(1) %b2, i64 3
  store i32 %v, ptr addrspace(1) %o3, align 4
";
    let name = "slot_values";
    let Some(bc) = assemble(name, &kernel(name, 3, body, "")) else {
        return;
    };
    // (the buffer's device address, the bind offset): slot 0 just below 2^31 and slot 1 just
    // above it, the output above 4 GiB, so a dropped high word shows.
    let at = [
        (0x7FFF_FF00u64, 0u64),
        (0x8000_0100, 16),
        (0x3_0000_2000, 0),
    ];
    let words =
        |base: u32| -> Vec<u8> { (0..16u32).flat_map(|i| (base + i).to_le_bytes()).collect() };
    let bytes = [words(0x100), words(0x200), vec![0xEE; 20]];
    let slot_address = |i: usize| at[i].0 + at[i].1;
    let mut want = 1u32.to_le_bytes().to_vec();
    want.extend_from_slice(&0u32.to_le_bytes());
    want.extend_from_slice(&bytes[1][12..16]);
    want.extend_from_slice(&bytes[0][20..24]);
    want.extend_from_slice(&[0xEE; 4]);
    for stem in PROFILES {
        let p = profile(stem, identity()).unwrap_or_else(|e| panic!("{e}"));
        let (a, plan, module) =
            compile_kept(&bc, name, &p).unwrap_or_else(|e| panic!("{stem}: {e}"));
        assert!(a.interface.descriptors.is_empty(), "{stem}");
        link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{stem}: L3: {e}"));
        let mut push = vec![0u8; plan.push.push_range.map_or(0, |(_, end)| end as usize)];
        let region = region_bytes([0; 3], [1, 1, 1], [1, 1, 1]).expect("packs");
        for f in &plan.push.fields {
            let PushLoc::Push { offset } = f.loc else {
                panic!("{stem}: {f:?} spilled");
            };
            let o = offset as usize;
            match f.field {
                PushField::SlotGuestBase(s) => {
                    let word = slot_word(slot_address(s.index as usize));
                    push[o..o + 8].copy_from_slice(&word.to_le_bytes());
                }
                PushField::DispatchRegion => push[o..o + 32].copy_from_slice(&region),
                _ => panic!("{stem}: unexpected push field {f:?}"),
            }
        }
        let buffers = (0..3)
            .map(|i| Buffer {
                slot: None,
                address: Some(at[i].0),
                bytes: bytes[i].clone(),
            })
            .collect();
        let res = Resources {
            buffers,
            push,
            spec: local_size(1),
        };
        let got = run(&module, &Config::default(), res).unwrap_or_else(|e| panic!("{stem}: {e}"));
        assert_eq!(got.buffers[2], want, "{stem}: out");
    }
}

/// The harness's sampling kernels (`fixtures/public/harness/workloads.ll`, our own), assembled.
fn workloads() -> Option<Vec<u8>> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/fixtures/public/harness/workloads.ll");
    assembled("workloads", llvm_assemble(&path))
}

/// Whether `key` is one of a dispatch's own specialization constants: its local size, its
/// `MultiRegion` and its `DeepThreadgroup`.
fn dispatch_key(key: SpecKey) -> bool {
    matches!(
        key,
        SpecKey::LocalSize(_) | SpecKey::MultiRegion | SpecKey::DeepThreadgroup
    )
}

/// A kernel sampling a texture slot through a runtime sampler slot (`sample_down`: a 2D `float`
/// sample, a size query and a write) takes the sampled texture from set 1's sampled-image array
/// and the sampler from its plain sampler array, each at the index its pushed resource ID holds,
/// with no descriptor, no specialization constant but the dispatch's ([`dispatch_key`]) and no
/// fact of the bound sampler: the word the sample reads is the resource ID's. It passes L3 on each
/// profile.
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
            (a.interface.spec.iter()).all(|c| dispatch_key(c.key)),
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

/// A kernel that reads one texture buffer and writes another (`ob1_texel_buffer`, of `uint`
/// texels, and `texture_ops`' `texel_buffers`, of `float` ones), and one whose argument buffer
/// holds one (`rlm_runtime_limits`' `rlm_member_texture_buffer`), take each from set 1's texel
/// buffer arrays at the index its resource ID holds: a read one from the uniform array (binding 5)
/// and a written one from the storage array (binding 6), with no fact of the bound texture buffer
/// (no `TexelBufferRange`: the runtime's buffer view is the texture buffer, its width its texels).
/// A slot's ID is its pushed field. Each passes L3 on each profile.
#[test]
fn a_texture_buffer_is_the_texel_buffer_element_its_resource_id_names() {
    let kernels = [
        ("ob1_texel_buffer", "ob1_texel_buffer", vec![5, 6]),
        ("texture_ops", "texel_buffers", vec![5, 6]),
        ("rlm_runtime_limits", "rlm_member_texture_buffer", vec![5]),
    ];
    for (fixture, entry, bindings) in kernels {
        let Some(bc) = assemble(fixture) else {
            return;
        };
        for stem in PROFILES {
            let p = profile(stem, identity()).unwrap_or_else(|e| panic!("{e}"));
            let (a, plan, _) =
                compile_kept(&bc, entry, &p).unwrap_or_else(|e| panic!("{stem} {entry}: {e}"));
            link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{stem} {entry}: L3: {e}"));
            assert!(
                !(plan.facts.iter()).any(|f| matches!(f, RuntimeFact::TexelBufferRange { .. })),
                "{stem} {entry}: {:?}",
                plan.facts
            );
            let linked =
                link(&[&a], &LinkInputs::standalone(), &p).unwrap_or_else(|e| panic!("{e}"));
            let mut used: Vec<u32> = (linked.table.bindings.values())
                .filter(|(set, _)| *set == 1)
                .map(|(_, binding)| *binding)
                .collect();
            used.sort();
            used.dedup();
            assert_eq!(used, bindings, "{stem} {entry}");
        }
    }
}

/// The words of `sample_up`'s constexpr sampler (`fixtures/public/harness/workloads.ll`).
const SAMPLE_UP_WORDS: [u64; 2] = [34_901_797_601_020_489, 0];

/// A kernel sampling a texture slot through a constexpr sampler (`sample_up`, a 2D `float` sample
/// of normalized coordinates, and the same kernel through a pixel sampler that the resident-table
/// domains hand the device's unnormalized sampler) takes the texture from set 1's sampled-image
/// array at the index its pushed resource ID holds and the sampler from the plain sampler array at
/// the index the consumer pushes for it (`PushField::ConstexprSampler`, a `u32`), with no
/// descriptor and no specialization constant but the dispatch's ([`dispatch_key`]). The field's
/// sampler is the constexpr one in normalized coordinates, whose device sampler
/// (`CombinedSampler::constexpr_sampler`) is normalized: the shader scales the linear pixel
/// sampler's coordinate and records `PrecisionNote::PixelSamplerScaling`, which the resident-table
/// domains do not. It passes L3 on each profile.
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
            // The resident-table domains read the linear pixel sampler's coordinate through the
            // device's unnormalized sampler, with no note, so the identity form below scales it
            // in its place.
            let default =
                profile(stem, DevicePolicy::RESIDENT_TABLES).unwrap_or_else(|e| panic!("{e}"));
            let (a, plan, _) = compile_kept(bc, "sample_up", &default)
                .unwrap_or_else(|e| panic!("{stem}: resident-table domains: {e}"));
            let unnormalized: Vec<bool> = (plan.families.sampler.decls.values())
                .map(|d| d.sampler.unnormalized)
                .collect();
            assert_eq!(
                unnormalized,
                [words == pixel],
                "{stem}: resident-table domains"
            );
            assert!(
                !a.precision.contains(&PrecisionNote::PixelSamplerScaling),
                "{stem}: resident-table domains"
            );
        }
        for (stem, p) in &profiles {
            let (a, plan, _) =
                compile_kept(bc, "sample_up", p).unwrap_or_else(|e| panic!("{stem}: {e}"));
            link_and_validate(&a, p).unwrap_or_else(|e| panic!("{stem}: L3: {e}"));
            assert!(
                (a.interface.spec.iter()).all(|c| dispatch_key(c.key)),
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

/// A kernel over a `2 × 2 × 6` grid writing `(x, y, face, 7)` at texel `(x, y)` of face `z` of
/// its `texturecube<uint, write>` slot 0 and of cube 1 of its `texturecube_array<uint, write>`
/// slot 1, and storing slot 0's width and slot 1's cube count to words 0 and 1 of `out`
/// (buffer 2).
const CUBE_FACES_KERNEL: &str = "define void @cf(ptr addrspace(1) %c, ptr addrspace(1) %a, ptr addrspace(1) noundef \"air-buffer-no-alias\" %out, <3 x i32> noundef %tid) {\n\
entry:\n\
  %xy = shufflevector <3 x i32> %tid, <3 x i32> poison, <2 x i32> <i32 0, i32 1>\n\
  %x = extractelement <3 x i32> %tid, i64 0\n\
  %y = extractelement <3 x i32> %tid, i64 1\n\
  %f = extractelement <3 x i32> %tid, i64 2\n\
  %v0 = insertelement <4 x i32> <i32 0, i32 0, i32 0, i32 7>, i32 %x, i64 0\n\
  %v1 = insertelement <4 x i32> %v0, i32 %y, i64 1\n\
  %v = insertelement <4 x i32> %v1, i32 %f, i64 2\n\
  tail call void @air.write_texture_cube.u.v4i32(ptr addrspace(1) %c, <2 x i32> %xy, i32 %f, <4 x i32> %v, i32 0, i32 2)\n\
  tail call void @air.write_texture_cube_array.u.v4i32(ptr addrspace(1) %a, <2 x i32> %xy, i32 %f, i32 1, <4 x i32> %v, i32 0, i32 2)\n\
  %w = tail call i32 @air.get_width_texture_cube(ptr addrspace(1) %c, i32 0)\n\
  %n = tail call i32 @air.get_array_size_texture_cube_array(ptr addrspace(1) %a)\n\
  store i32 %w, ptr addrspace(1) %out, align 4\n\
  %p = getelementptr inbounds i32, ptr addrspace(1) %out, i64 1\n\
  store i32 %n, ptr addrspace(1) %p, align 4\n\
  ret void\n\
}\n\n\
declare void @air.write_texture_cube.u.v4i32(ptr addrspace(1), <2 x i32>, i32, <4 x i32>, i32, i32)\n\
declare void @air.write_texture_cube_array.u.v4i32(ptr addrspace(1), <2 x i32>, i32, i32, <4 x i32>, i32, i32)\n\
declare i32 @air.get_width_texture_cube(ptr addrspace(1), i32)\n\
declare i32 @air.get_array_size_texture_cube_array(ptr addrspace(1))\n\n\
!air.kernel = !{!0}\n!air.version = !{!4}\n!air.language_version = !{!5}\n\
!0 = !{ptr @cf, !1, !2}\n!1 = !{}\n!2 = !{!6, !7, !8, !3}\n\
!6 = !{i32 0, !\"air.texture\", !\"air.location_index\", i32 0, i32 1, !\"air.write\", !\"air.arg_type_name\", !\"texturecube<uint, write>\", !\"air.arg_name\", !\"c\"}\n\
!7 = !{i32 1, !\"air.texture\", !\"air.location_index\", i32 1, i32 1, !\"air.write\", !\"air.arg_type_name\", !\"texturecube_array<uint, write>\", !\"air.arg_name\", !\"a\"}\n\
!8 = !{i32 2, !\"air.buffer\", !\"air.location_index\", i32 2, i32 1, !\"air.read_write\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"out\"}\n\
!3 = !{i32 3, !\"air.thread_position_in_grid\", !\"air.arg_type_name\", !\"uint3\", !\"air.arg_name\", !\"tid\"}\n\
!4 = !{i32 2, i32 8, i32 0}\n!5 = !{!\"Metal\", i32 4, i32 0, i32 0}\n";

/// A storage cube or cube array slot written by face is the element of the 2D-array storage
/// class at its resource ID's index, which the runtime holds as the 2D-array view of the
/// texture's faces (layer `6 · cube + face`): no view shape is stated, the writes land on those
/// layers, and the width and the cube count are the view's.
#[test]
fn a_storage_cube_written_by_face_is_the_2d_array_element_of_its_faces() {
    let Some(bc) = assemble_text("cf", CUBE_FACES_KERNEL) else {
        return;
    };
    const L: [u32; 3] = [2, 2, 6];
    // `c` at index 2 and `a` at index 3 of the storage array; the elements below are others.
    const C: u32 = 2;
    const A: u32 = 3;
    for stem in PROFILES {
        let p = profile(stem, identity()).unwrap_or_else(|e| panic!("{e}"));
        let (a, plan, module) =
            compile_kept(&bc, "cf", &p).unwrap_or_else(|e| panic!("{stem}: {e}"));
        link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{stem}: L3: {e}"));
        assert!(
            !(plan.facts.iter()).any(|f| matches!(f, RuntimeFact::ImageViewShape { .. })),
            "{stem}"
        );
        let linked = link(&[&a], &LinkInputs::standalone(), &p).unwrap_or_else(|e| panic!("{e}"));
        let mut classes = (linked.table.bindings.iter()).filter(|(_, b)| **b == (1, 1));
        let (storage, _) = classes
            .next()
            .unwrap_or_else(|| panic!("{stem}: no storage class"));
        assert!(classes.next().is_none(), "{stem}: two storage classes");
        let mut push = vec![0u8; plan.push.push_range.map_or(0, |(_, end)| end as usize)];
        for field in &plan.push.fields {
            let PushLoc::Push { offset } = field.loc else {
                panic!("{stem}: {field:?} spilled");
            };
            let o = offset as usize;
            match field.field {
                PushField::SlotGuestBase(s) => {
                    let bytes = match (s.class, s.index) {
                        (SlotClass::Texture, 0) => resource_id(C, 0),
                        (SlotClass::Texture, 1) => resource_id(A, 0),
                        (SlotClass::Buffer, 2) => slot_word(OUT),
                        _ => panic!("{stem}: unexpected slot {s:?}"),
                    };
                    push[o..o + 8].copy_from_slice(&bytes.to_le_bytes());
                }
                PushField::SlotBounds(_) => push[o..o + 8].copy_from_slice(&bounds_word(8, 0)),
                PushField::ZeroAddress => {
                    push[o..o + 8].copy_from_slice(&ZERO_UNMAPPED.to_le_bytes())
                }
                PushField::DispatchRegion => {
                    let region = region_bytes([0; 3], L, L).expect("packs");
                    push[o..o + 32].copy_from_slice(&region);
                }
                other => panic!("{stem}: unexpected push field {other:?}"),
            }
        }
        let mut images: Vec<Image> = (0..C)
            .map(|e| texture(*storage, e, [1, 1], |_, _| [0; 4], None))
            .collect();
        images.push(Image::new(*storage, [2, 2, 1], 6, 1, 1).at_element(C));
        images.push(Image::new(*storage, [2, 2, 1], 12, 1, 1).at_element(A));
        let res = Resources {
            buffers: vec![Buffer {
                slot: None,
                address: Some(OUT),
                bytes: vec![0; 8],
            }],
            push,
            spec: local_size3(L),
        };
        let (got, images) = run_with_images(&module, &Config::default(), res, images)
            .unwrap_or_else(|e| panic!("{stem}: {e}"));
        let faces: Vec<[u32; 4]> = (0..6)
            .flat_map(|f| (0..2).flat_map(move |y| (0..2).map(move |x| [x, y, f, 7])))
            .collect();
        assert_eq!(images[C as usize].texels, faces, "{stem}: the cube");
        let cubes: Vec<[u32; 4]> = std::iter::repeat_n([0; 4], 24).chain(faces).collect();
        assert_eq!(images[A as usize].texels, cubes, "{stem}: the cube array");
        let words: Vec<u8> = [2u32, 2].iter().flat_map(|w| w.to_le_bytes()).collect();
        assert_eq!(got.buffers[0], words, "{stem}: width and cubes");
    }
}

/// Set 1's fixed arrays under the identity domains: binding, descriptor type and count.
fn identity_arrays() -> Vec<(u32, vk::DescriptorType, u32)> {
    vec![
        (0, vk::DescriptorType::SAMPLED_IMAGE, 65_536),
        (1, vk::DescriptorType::STORAGE_IMAGE, 65_536),
        (2, vk::DescriptorType::SAMPLER, 4_000),
        (3, vk::DescriptorType::SAMPLER, 4_000),
        (4, vk::DescriptorType::SAMPLER, 4_000),
        (5, vk::DescriptorType::UNIFORM_TEXEL_BUFFER, 65_536),
        (6, vk::DescriptorType::STORAGE_TEXEL_BUFFER, 65_536),
    ]
}

/// `bcu_formats`' texture slots: ten 2D textures, the last two sampled with `half` results.
fn bicubic_slots() -> BTreeSet<SlotRef> {
    (0..10)
        .map(|index| SlotRef {
            stage: Stage::Kernel,
            class: SlotClass::Texture,
            index,
        })
        .collect()
}

/// A kernel that samples 2D textures through a bicubic constexpr sampler at normalized
/// coordinates reads how the filter treats each slot's bound format from the slot's push field
/// under the identity domains, where the resident-table domains specialize it and state its fact,
/// and so does one that samples at pixel coordinates too (`bcu_address`); a bicubic sample of a 1D
/// texture is still refused there ([`BICUBIC_LINE_FORM`]).
#[test]
fn a_bicubic_kernel_reads_each_slots_format_treatment_from_its_pushed_field() {
    use crate::pipeline::artifact::PrecisionNote;
    let (Some(formats), Some(forms), Some(address)) = (
        assemble("bcu_formats"),
        assemble("bcu_forms"),
        assemble("bcu_address"),
    ) else {
        return;
    };
    let slots = bicubic_slots();
    for stem in PROFILES {
        let default =
            profile(stem, DevicePolicy::RESIDENT_TABLES).unwrap_or_else(|e| panic!("{e}"));
        let (a, plan, _) = compile_kept(&formats, "bcu_formats", &default)
            .unwrap_or_else(|e| panic!("{stem}: resident-table domains: {e}"));
        let facts: BTreeSet<SlotRef> = (plan.facts.iter())
            .filter_map(|f| match f {
                RuntimeFact::BicubicTexels { slot } => Some(*slot),
                _ => None,
            })
            .collect();
        let specialized: BTreeSet<SlotRef> = (a.interface.spec.iter())
            .filter_map(|c| match c.key {
                SpecKey::BicubicTexels(slot) => Some(slot),
                _ => None,
            })
            .collect();
        assert_eq!(
            (&facts, &specialized),
            (&slots, &slots),
            "{stem}: resident-table domains"
        );
        assert!(
            !(plan.push.fields.iter()).any(|f| matches!(f.field, PushField::BicubicTexels(_))),
            "{stem}: resident-table domains"
        );
        let p = profile(stem, identity()).unwrap_or_else(|e| panic!("{e}"));
        let (a, plan, _) =
            compile_kept(&formats, "bcu_formats", &p).unwrap_or_else(|e| panic!("{stem}: {e}"));
        link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{stem}: L3: {e}"));
        let pushed: BTreeSet<SlotRef> = (plan.push.fields.iter())
            .filter_map(|f| match f.field {
                PushField::BicubicTexels(slot) => Some(slot),
                _ => None,
            })
            .collect();
        assert_eq!(pushed, slots, "{stem}");
        assert!(
            !(a.interface.spec.iter()).any(|c| matches!(c.key, SpecKey::BicubicTexels(_))),
            "{stem}: {:?}",
            a.interface.spec
        );
        assert!(
            !(plan.facts.iter()).any(|f| matches!(f, RuntimeFact::BicubicTexels { .. })),
            "{stem}: {:?}",
            plan.facts
        );
        assert!(
            a.precision
                .contains(&PrecisionNote::BicubicFilterArithmetic),
            "{stem}"
        );
        // `bcu_address` samples its 2D texture at normalized and at pixel coordinates: its one
        // slot's field, no specialization constant and no fact, and it passes L3.
        let (a, plan, _) = compile_kept(&address, "bcu_address", &p)
            .unwrap_or_else(|e| panic!("{stem}: bcu_address: {e}"));
        link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{stem}: bcu_address: L3: {e}"));
        let pushed: Vec<SlotRef> = (plan.push.fields.iter())
            .filter_map(|f| match f.field {
                PushField::BicubicTexels(slot) => Some(slot),
                _ => None,
            })
            .collect();
        assert_eq!(
            pushed,
            bicubic_slots().into_iter().take(1).collect::<Vec<_>>(),
            "{stem}: bcu_address"
        );
        assert!(
            !(a.interface.spec.iter()).any(|c| matches!(c.key, SpecKey::BicubicTexels(_))),
            "{stem}: bcu_address"
        );
        // `bcu_forms` samples a 1D texture at normalized coordinates: refused with its form.
        compile(&forms, "bcu_forms", &default)
            .unwrap_or_else(|e| panic!("{stem}: bcu_forms: resident-table domains: {e}"));
        match compile(&forms, "bcu_forms", &p) {
            Err(Error::Unsupported(u)) => {
                assert_eq!(
                    (u.feature, u.needs),
                    (Feature::IdentityDomains, Needs::NotImplemented),
                    "{stem}: bcu_forms"
                );
                assert_eq!(
                    u.symbol.map(|f| f.to_string()).as_deref(),
                    Some(BICUBIC_LINE_FORM),
                    "{stem}: bcu_forms"
                );
            }
            other => panic!("{stem}: bcu_forms: {other:?}"),
        }
    }
}

/// A constexpr sampler's words: nearest filters, no mip filter, normalized coordinates clamped to
/// the edge, comparing under `less`.
const NEAREST_LESS: u64 = 0x007b_ff00_0001_0049;

/// A kernel `k` that samples the `depth2d<float>` its argument buffer (buffer 0) holds at
/// `(0.25, 0.25)` with comparison against 0.5 through a constexpr [`NEAREST_LESS`] sampler and,
/// where `plain`, also without comparison through a constexpr nearest one, and writes the answers
/// to the first lanes of buffer 1's `float4` at lane `gid`.
fn heap_depth_kernel(plain: bool) -> String {
    let nearest = super::engine_handles::CHOSEN_SAMPLERS[0];
    let (global, sample, store, declare, states, state) = match plain {
        true => (
            format!("@plain = internal addrspace(2) constant [2 x i64] [i64 {nearest}, i64 0], align 8\n"),
            "  %q = call { float, i8 } @air.sample_depth_2d.f32(ptr addrspace(1) %t, ptr addrspace(2) @plain, i32 1, <2 x float> <float 0.25, float 0.25>, i1 true, <2 x i32> zeroinitializer, i1 true, float 0.0, float 0.0, i32 0)\n  %y = extractvalue { float, i8 } %q, 0\n  %v = insertelement <4 x float> %w, float %y, i64 1\n",
            "%v",
            "declare { float, i8 } @air.sample_depth_2d.f32(ptr addrspace(1), ptr addrspace(2), i32, <2 x float>, i1, <2 x i32>, i1, float, float, i32)\n",
            "!20, !21",
            "!21 = !{!\"air.sampler_state\", ptr addrspace(2) @plain}\n",
        ),
        false => (String::new(), "", "%w", "", "!20", ""),
    };
    format!(
        r#"target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

%struct.M = type {{ ptr addrspace(1) }}
@less = internal addrspace(2) constant [2 x i64] [i64 {NEAREST_LESS}, i64 0], align 8
{global}
define void @k(ptr addrspace(2) %m, ptr addrspace(1) %out, i32 %gid) {{
  %pt = getelementptr inbounds %struct.M, ptr addrspace(2) %m, i64 0, i32 0
  %t = load ptr addrspace(1), ptr addrspace(2) %pt, align 8
  %r = call {{ float, i8 }} @air.sample_compare_depth_2d.f32(ptr addrspace(1) %t, ptr addrspace(2) @less, i32 1, <2 x float> <float 0.25, float 0.25>, float 0.5, i1 true, <2 x i32> zeroinitializer, i1 true, float 0.0, float 0.0, i32 0)
  %x = extractvalue {{ float, i8 }} %r, 0
  %w = insertelement <4 x float> zeroinitializer, float %x, i64 0
{sample}  %o = getelementptr inbounds <4 x float>, ptr addrspace(1) %out, i32 %gid
  store <4 x float> {store}, ptr addrspace(1) %o, align 16
  ret void
}}
declare {{ float, i8 }} @air.sample_compare_depth_2d.f32(ptr addrspace(1), ptr addrspace(2), i32, <2 x float>, float, i1, <2 x i32>, i1, float, float, i32)
{declare}!air.kernel = !{{!0}}
!air.version = !{{!11}}
!air.language_version = !{{!12}}
!air.sampler_states = !{{{states}}}
!0 = !{{ptr @k, !1, !2}}
!1 = !{{}}
!2 = !{{!3, !6, !8}}
!3 = !{{i32 0, !"air.indirect_buffer", !"air.buffer_size", i32 8, !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.struct_type_info", !4, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"M", !"air.arg_name", !"m"}}
!4 = !{{i32 0, i32 8, i32 0, !"depth2d<float, sample>", !"d", !"air.indirect_argument", !5}}
!5 = !{{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"depth2d<float, sample>", !"air.arg_name", !"d"}}
!6 = !{{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4", !"air.arg_name", !"out"}}
!8 = !{{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"gid"}}
!11 = !{{i32 2, i32 8, i32 0}}
!12 = !{{!"Metal", i32 4, i32 0, i32 0}}
!20 = !{{!"air.sampler_state", ptr addrspace(2) @less}}
{state}"#
    )
}

/// Under the identity domains a kernel that samples, compares or fetches through a pixel sampler a
/// texture its argument buffer holds, through a runtime sampler slot or a constexpr sampler
/// (`engine/handles`' heap samplings, `examples/heap_samples`' `hs_sample`, and
/// [`heap_depth_kernel`]'s comparison, alone and beside a sample of the same depth texture
/// without comparison), takes the texture as the element of its class's set-1
/// array that the member's resource ID names and the sampler as the element of the set-1 sampler
/// array of its use that the slot's pushed resource ID or the constexpr sampler's pushed index
/// (`PushField::ConstexprSampler`) names: no heap sampler binding and nothing in set 0. Each passes
/// L3 on every profile; the resident-table domains compile each as before.
///
/// The depth texture's sample without comparison reads the same element of its depth-comparison
/// class (`Depth` 1) as its comparison, through the plain sampler array (binding 2). SPIR-V 1.6
/// (revision 8) leaves the comparison to the opcode, not the image type: `OpTypeImage`'s "Depth
/// is whether or not this image is a depth image. (Note that whether or not depth comparisons are
/// actually done is a property of the sampling opcode, not of this type declaration.)", and
/// `OpImageSampleImplicitLod` and `OpImageSampleExplicitLod` constrain only its `Dim` and `MS`.
/// Vulkan 1.4.366's "Instruction/Format/Sampler/Image View Validation" makes an instruction that is
/// not `OpImage*Dref*` poison only through a sampler whose `compareEnable` is `VK_TRUE`, which no
/// element of the plain array has, and its image-view compatibility list does not name `Depth`.
#[test]
fn an_argument_buffer_textures_sample_takes_its_sampler_from_the_identity_arrays() {
    let (Some(handles), Some(examples), Some(depth), Some(both)) = (
        assemble("handles"),
        assembled(
            "heap_samples",
            llvm_assemble(&engine_dir().join("../examples/heap_samples.ll")),
        ),
        assemble_text("heap_depth", &heap_depth_kernel(false)),
        assemble_text("heap_depth_both", &heap_depth_kernel(true)),
    ) else {
        return;
    };
    // Each entry, the set-1 bindings it uses, how many sampler slots' resource IDs it pushes, and
    // each constexpr sampler's pushed use (comparison, integer texture).
    type Case<'a> = (&'a [u8], &'a str, &'a [u32], usize, &'a [(bool, bool)]);
    let cases: [Case<'_>; 8] = [
        // A linear sample, and a nearest pixel sample's exact fetch, written to a member.
        (
            &handles,
            "argument_buffer_sample",
            &[0, 1, 2],
            0,
            &[(false, false); 2],
        ),
        (
            &handles,
            "argument_buffer_sample_int",
            &[0, 1, 4],
            0,
            &[(false, true)],
        ),
        (&handles, "argument_buffer_runtime_sampler", &[0, 2], 1, &[]),
        (
            &handles,
            "argument_buffer_sample_compare",
            &[0, 3],
            0,
            &[(true, false)],
        ),
        (
            &handles,
            "argument_buffer_sample_border",
            &[0, 2],
            0,
            &[(false, false)],
        ),
        (&examples, "hs_sample", &[0, 2], 1, &[(false, false)]),
        (&depth, "k", &[0, 3], 0, &[(true, false)]),
        (&both, "k", &[0, 2, 3], 0, &[(false, false), (true, false)]),
    ];
    for stem in PROFILES {
        let default =
            profile(stem, DevicePolicy::RESIDENT_TABLES).unwrap_or_else(|e| panic!("{e}"));
        let p = profile(stem, identity()).unwrap_or_else(|e| panic!("{e}"));
        for (bc, entry, bindings, slots, constexpr) in cases {
            let what = format!("{stem}: {entry}");
            compile(bc, entry, &default)
                .unwrap_or_else(|e| panic!("{what}: resident-table domains: {e}"));
            let (a, plan, _) =
                compile_kept(bc, entry, &p).unwrap_or_else(|e| panic!("{what}: {e}"));
            link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
            let pushed_slots = (plan.push.fields.iter())
                .filter(|f| {
                    matches!(f.field, PushField::SlotGuestBase(s) if s.class == SlotClass::Sampler)
                })
                .count();
            let mut uses: Vec<(bool, bool)> = (plan.push.fields.iter())
                .filter_map(|f| match f.field {
                    PushField::ConstexprSampler(c) => Some((c.compare, c.integer)),
                    _ => None,
                })
                .collect();
            uses.sort();
            assert_eq!((pushed_slots, &uses[..]), (slots, constexpr), "{what}");
            let linked = link(&[&a], &LinkInputs::standalone(), &p)
                .unwrap_or_else(|e| panic!("{what}: {e}"));
            let sets: Vec<_> = (linked.layout.sets.iter())
                .map(|s| {
                    let b: Vec<_> = (s.bindings.iter())
                        .map(|b| (b.binding, b.ty, b.count))
                        .collect();
                    (s.set, b)
                })
                .collect();
            assert_eq!(sets, [(0, vec![]), (1, identity_arrays())], "{what}");
            let mut used: Vec<u32> = (linked.table.bindings.values())
                .filter(|(set, _)| *set == 1)
                .map(|(_, binding)| *binding)
                .collect();
            used.sort();
            used.dedup();
            assert_eq!(used, bindings, "{what}");
        }
    }
}

/// What [`chosen_heap`] does with the texture its argument buffer holds, through the constexpr
/// sampler its lane chooses.
#[derive(Clone, Copy, Debug)]
enum Chosen {
    /// A kernel's sample of a `texture2d<float>` at level 0.
    Float,
    /// A kernel's sample of a `texture2d<uint>` at level 0.
    Integer,
    /// A kernel's `sample_compare` of a `depth2d<float>` against 0.5 at level 0.
    Compare,
    /// A fragment's sample of a `texture2d<float>` at its implicit level, at its position over 16,
    /// its lane its position's `x`.
    Fragment,
}

/// A function `chosen` that does `chosen` with the texture its argument buffer (buffer 0) holds,
/// at `(1.375, 0.5)` in a kernel, through the constexpr sampler of words `words[lane % n]` (each a
/// constexpr sampler of the module, listed in order, so their handles are consecutive), and
/// writes the result to buffer 1's lane (a kernel's `float4` or `uint4`) or its color attachment.
fn chosen_heap(words: &[u64], chosen: Chosen) -> String {
    let n = words.len();
    let mut choose = format!("  %sel = urem i32 %lane, {n}\n");
    let mut sampler = format!("@s{}", n - 1);
    for k in (0..n - 1).rev() {
        choose += &format!(
            "  %c{k} = icmp eq i32 %sel, {k}\n  %p{k} = select i1 %c{k}, ptr addrspace(2) @s{k}, ptr addrspace(2) {sampler}\n"
        );
        sampler = format!("%p{k}");
    }
    let globals: String = (words.iter().enumerate())
        .map(|(i, w)| {
            format!("@s{i} = internal addrspace(2) constant [2 x i64] [i64 {w}, i64 0], align 8\n")
        })
        .collect();
    let list: Vec<String> = (0..n).map(|i| format!("!{}", 20 + i)).collect();
    let states: String = (0..n)
        .map(|i| {
            format!(
                "!{} = !{{!\"air.sampler_state\", ptr addrspace(2) @s{i}}}\n",
                20 + i
            )
        })
        .collect();
    let at_zero = "<2 x float> <float 1.375, float 0.5>, i1 true, <2 x i32> zeroinitializer, i1 true, float 0.0, float 0.0, i32 0";
    let float = "declare { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i1, float, float, i32)";
    let (texture, elem, sample, declare) = match chosen {
        Chosen::Float => (
            "texture2d<float, sample>",
            ("float", "float"),
            format!(
                "  %r = call {{ <4 x float>, i8 }} @air.sample_texture_2d.v4f32(ptr addrspace(1) %t, ptr addrspace(2) {sampler}, {at_zero})\n  %v = extractvalue {{ <4 x float>, i8 }} %r, 0\n"
            ),
            float,
        ),
        Chosen::Integer => (
            "texture2d<uint, sample>",
            ("i32", "uint"),
            format!(
                "  %r = call {{ <4 x i32>, i8 }} @air.sample_texture_2d.u.v4i32(ptr addrspace(1) %t, ptr addrspace(2) {sampler}, {at_zero})\n  %v = extractvalue {{ <4 x i32>, i8 }} %r, 0\n"
            ),
            "declare { <4 x i32>, i8 } @air.sample_texture_2d.u.v4i32(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i1, float, float, i32)",
        ),
        Chosen::Compare => (
            "depth2d<float, sample>",
            ("float", "float"),
            format!(
                "  %r = call {{ float, i8 }} @air.sample_compare_depth_2d.f32(ptr addrspace(1) %t, ptr addrspace(2) {sampler}, i32 1, <2 x float> <float 1.375, float 0.5>, float 0.5, i1 true, <2 x i32> zeroinitializer, i1 true, float 0.0, float 0.0, i32 0)\n  %x = extractvalue {{ float, i8 }} %r, 0\n  %v = insertelement <4 x float> zeroinitializer, float %x, i64 0\n"
            ),
            "declare { float, i8 } @air.sample_compare_depth_2d.f32(ptr addrspace(1), ptr addrspace(2), i32, <2 x float>, float, i1, <2 x i32>, i1, float, float, i32)",
        ),
        Chosen::Fragment => (
            "texture2d<float, sample>",
            ("float", "float"),
            format!(
                "  %xy = shufflevector <4 x float> %pos, <4 x float> poison, <2 x i32> <i32 0, i32 1>\n  %uv = fmul <2 x float> %xy, splat (float 6.250000e-02)\n  %r = call {{ <4 x float>, i8 }} @air.sample_texture_2d.v4f32(ptr addrspace(1) %t, ptr addrspace(2) {sampler}, <2 x float> %uv, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.0, float 0.0, i32 0)\n  %v = extractvalue {{ <4 x float>, i8 }} %r, 0\n"
            ),
            float,
        ),
    };
    let held = format!(
        r#"!3 = !{{i32 {index}, !"air.indirect_buffer", !"air.buffer_size", i32 8, !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.struct_type_info", !4, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"M", !"air.arg_name", !"m"}}
!4 = !{{i32 0, i32 8, i32 0, !"{texture}", !"t", !"air.indirect_argument", !5}}
!5 = !{{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"{texture}", !"air.arg_name", !"t"}}"#,
        index = u32::from(matches!(chosen, Chosen::Fragment)),
    );
    let (ty, msl) = elem;
    let (function, stage) = match chosen {
        Chosen::Fragment => (
            format!(
                r#"define <4 x float> @chosen(<4 x float> %pos, ptr addrspace(2) %m) {{
  %x0 = extractelement <4 x float> %pos, i64 0
  %lane = fptoui float %x0 to i32
  %pt = getelementptr inbounds %struct.M, ptr addrspace(2) %m, i64 0, i32 0
  %t = load ptr addrspace(1), ptr addrspace(2) %pt, align 8
{choose}{sample}  ret <4 x float> %v
}}"#
            ),
            r#"!air.fragment = !{!0}
!0 = !{ptr @chosen, !1, !2}
!1 = !{!9}
!2 = !{!8, !3}
!8 = !{i32 0, !"air.position", !"air.center", !"air.no_perspective", !"air.arg_type_name", !"float4", !"air.arg_name", !"pos"}
!9 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}"#
                .to_owned(),
        ),
        _ => (
            format!(
                r#"define void @chosen(ptr addrspace(2) %m, ptr addrspace(1) %out, i32 %lane) {{
  %pt = getelementptr inbounds %struct.M, ptr addrspace(2) %m, i64 0, i32 0
  %t = load ptr addrspace(1), ptr addrspace(2) %pt, align 8
{choose}{sample}  %o = getelementptr inbounds <4 x {ty}>, ptr addrspace(1) %out, i32 %lane
  store <4 x {ty}> %v, ptr addrspace(1) %o, align 16
  ret void
}}"#
            ),
            format!(
                r#"!air.kernel = !{{!0}}
!0 = !{{ptr @chosen, !1, !2}}
!1 = !{{}}
!2 = !{{!3, !6, !8}}
!6 = !{{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"{msl}4", !"air.arg_name", !"out"}}
!8 = !{{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"lane"}}"#
            ),
        ),
    };
    format!(
        r#"target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

%struct.M = type {{ ptr addrspace(1) }}
{globals}
{function}
{declare}
{stage}
!air.version = !{{!11}}
!air.language_version = !{{!12}}
!air.sampler_states = !{{{}}}
{held}
!11 = !{{i32 2, i32 8, i32 0}}
!12 = !{{!"Metal", i32 4, i32 0, i32 0}}
{states}"#,
        list.join(", ")
    )
}

/// Under the identity domains constexpr samplers of consecutive handles that a lane chooses among
/// beside an argument buffer's texture ([`chosen_heap`]) are one sample where their samplings
/// differ only in the sampler: a kernel's sample of a float class through [`CHOSEN_SAMPLERS`]
/// (`engine_handles`) reads the set-1 sampler array of its use, `NonUniform`, at the pushed index
/// of the candidate the handle names (`sampler.sampler_array.sample`): one sampled image and one
/// sample, no switch, one `PushField::ConstexprSampler` per candidate. The switch over the
/// candidates stays, a sampled image per candidate, for pixel coordinates (two linear pixel
/// samplers that differ in their border color), an integer texture, a comparison and a fragment's
/// implicit level. The resident-table domains take their set-0 array for all but the fragment's,
/// so the candidates' samplings are alike there and only the identity domains' rule keeps the
/// switch. Each compiles and passes L3 on every committed profile.
#[test]
fn chosen_constexpr_samplers_beside_an_argument_buffer_texture_are_one_sample_where_alike() {
    use super::engine_handles::{
        chosen_sampler, count, uses, CHOSEN_SAMPLERS, OP_IMAGE_SAMPLE_DREF_EXPLICIT_LOD,
        OP_IMAGE_SAMPLE_EXPLICIT_LOD, OP_SAMPLED_IMAGE, OP_SWITCH,
    };
    use crate::spirv::testing::committed_profiles_with;
    const PIXEL: u64 = 1 << 15;
    const OPAQUE_WHITE: u64 = 2 << 56;
    // `less` in place of the words' `never`.
    let compare = |w: u64| (w & !(0xf << 16)) | 1 << 16;
    let nearest = [
        chosen_sampler(1, 0),
        chosen_sampler(2, 0),
        chosen_sampler(3, 0),
    ];
    // Each case: its name, what it samples, its candidates, and whether it is one sample.
    let cases: [(&str, Chosen, Vec<u64>, bool); 5] = [
        ("float", Chosen::Float, CHOSEN_SAMPLERS.to_vec(), true),
        (
            "pixel",
            Chosen::Float,
            vec![
                chosen_sampler(0, 1) | PIXEL,
                chosen_sampler(0, 1) | PIXEL | OPAQUE_WHITE,
            ],
            false,
        ),
        ("integer", Chosen::Integer, nearest.to_vec(), false),
        (
            "compare",
            Chosen::Compare,
            nearest.iter().map(|w| compare(*w)).collect(),
            false,
        ),
        (
            "fragment",
            Chosen::Fragment,
            CHOSEN_SAMPLERS.to_vec(),
            false,
        ),
    ];
    let profiles = committed_profiles_with(DevicePolicy::default()).expect("profiles");
    for (name, chosen, words, one) in cases {
        let Some(bc) = assemble_text(name, &chosen_heap(&words, chosen)) else {
            return;
        };
        let n = words.len();
        let sample = match chosen {
            Chosen::Compare => OP_IMAGE_SAMPLE_DREF_EXPLICIT_LOD,
            _ => OP_IMAGE_SAMPLE_EXPLICIT_LOD,
        };
        for stem in PROFILES {
            let default =
                profile(stem, DevicePolicy::RESIDENT_TABLES).unwrap_or_else(|e| panic!("{e}"));
            let a = compile(&bc, "chosen", &default)
                .unwrap_or_else(|e| panic!("{stem}: {name}: resident-table domains: {e}"));
            let array = !matches!(chosen, Chosen::Fragment);
            assert_eq!(
                uses(&a, "sampler.sampler_array.sample"),
                u32::from(array),
                "{stem}: {name}: resident-table domains"
            );
        }
        for (stem, p) in &profiles {
            let what = format!("{stem}: {name}");
            let (a, plan, _) =
                compile_kept(&bc, "chosen", p).unwrap_or_else(|e| panic!("{what}: {e}"));
            link_and_validate(&a, p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
            let pushed = (plan.push.fields.iter())
                .filter(|f| matches!(f.field, PushField::ConstexprSampler(_)))
                .count();
            assert_eq!(pushed, n, "{what}");
            let arrays = uses(&a, "sampler.sampler_array.sample");
            let (images, switches) = (count(&a, OP_SAMPLED_IMAGE), count(&a, OP_SWITCH));
            if one {
                assert_eq!((arrays, images, switches), (1, 1, 0), "{what}");
                assert_eq!(count(&a, sample), 1, "{what}");
            } else {
                assert_eq!((arrays, images), (0, n), "{what}");
                assert!(switches > 0, "{what}");
            }
        }
    }
}

/// Under the identity domains a kernel that samples, compares or fetches an argument buffer's
/// texture through the sampler the argument buffer holds beside it (`engine/tail_argument_samplers`'
/// kernels and `engine/tail_argument_cube_gradient`'s) takes the sampler's resource ID from the
/// member's 8 bytes as its handle (`access::handle::emit_identity_sampler`): the sampler is the
/// element of the set-1 sampler array of its use that the ID's guarded index names, and its word's
/// fields are the ID's high word. No sampler table, nothing in set 0, no sampler slot's ID, constexpr
/// sampler index or execution status address pushed, and no cubic test
/// (`RuntimeFact::CubicArgumentSampler`). Each passes L3 on every
/// profile; the resident-table domains compile each as before.
#[test]
fn an_argument_buffers_sampler_samples_through_the_identity_arrays_by_its_resource_id() {
    let (Some(samplers), Some(cube)) = (
        assemble("tail_argument_samplers"),
        assemble("tail_argument_cube_gradient"),
    ) else {
        return;
    };
    // Each entry and the set-1 bindings it uses.
    let cases: [(&[u8], &str, &[u32]); 5] = [
        (&samplers, "argument_sampler_sample", &[0, 2]),
        (&samplers, "argument_sampler_edges", &[0, 2]),
        (&samplers, "argument_sampler_compare", &[0, 3]),
        (&samplers, "argument_sampler_integer", &[0, 4]),
        (&cube, "argument_sampler_cube_gradient", &[0, 2]),
    ];
    for stem in PROFILES {
        let default =
            profile(stem, DevicePolicy::RESIDENT_TABLES).unwrap_or_else(|e| panic!("{e}"));
        let p = profile(stem, identity()).unwrap_or_else(|e| panic!("{e}"));
        for (bc, entry, bindings) in cases {
            let what = format!("{stem}: {entry}");
            compile(bc, entry, &default)
                .unwrap_or_else(|e| panic!("{what}: resident-table domains: {e}"));
            let (a, plan, _) =
                compile_kept(bc, entry, &p).unwrap_or_else(|e| panic!("{what}: {e}"));
            link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
            let pushed: Vec<_> = (plan.push.fields.iter())
                .map(|f| f.field)
                .filter(|f| match f {
                    PushField::SlotGuestBase(s) => s.class == SlotClass::Sampler,
                    PushField::ExecutionStatusAddress | PushField::ConstexprSampler(_) => true,
                    _ => false,
                })
                .collect();
            assert_eq!(pushed, [], "{what}");
            assert!(
                !(a.facts.iter()).any(|f| matches!(f, RuntimeFact::CubicArgumentSampler { .. })),
                "{what}"
            );
            let linked = link(&[&a], &LinkInputs::standalone(), &p)
                .unwrap_or_else(|e| panic!("{what}: {e}"));
            let sets: Vec<_> = (linked.layout.sets.iter())
                .map(|s| {
                    let b: Vec<_> = (s.bindings.iter())
                        .map(|b| (b.binding, b.ty, b.count))
                        .collect();
                    (s.set, b)
                })
                .collect();
            assert_eq!(sets, [(0, vec![]), (1, identity_arrays())], "{what}");
            let mut used: Vec<u32> = (linked.table.bindings.values())
                .filter(|(set, _)| *set == 1)
                .map(|(_, binding)| *binding)
                .collect();
            used.sort();
            used.dedup();
            assert_eq!(used, bindings, "{what}");
        }
    }
}

/// Under the identity domains a kernel that encodes indirect commands is refused for that feature,
/// `Unsupported{GpuEncodedIndirectCommands}`, not implemented, on every profile.
#[test]
fn a_gpu_encoding_kernel_is_refused_as_gpu_encoded_commands_under_the_identity_domains() {
    let Some(bc) = assemble("icb_replay_encode") else {
        return;
    };
    for stem in PROFILES {
        let p = profile(stem, identity()).unwrap_or_else(|e| panic!("{e}"));
        match compile(&bc, "icb_replay_encode", &p) {
            Err(Error::Unsupported(u)) => {
                assert_eq!(
                    (u.feature, u.needs),
                    (Feature::GpuEncodedIndirectCommands, Needs::NotImplemented),
                    "{stem}"
                );
            }
            other => panic!("{stem}: {other:?}"),
        }
    }
}

/// A kernel that reads the stage-in region (`stage_in_region_short`: `out[t] = (origin.xy,
/// size.xy)` as `ushort`s) reads it from its pushed `StageInRegion` field, two `uvec4`s of the
/// origin and the size as the binder writes them, beside its output's address and the dispatch's
/// region, with no descriptor; every thread of the grid stores the region's origin and size.
#[test]
fn the_stage_in_region_is_read_from_its_pushed_field() {
    let Some(bc) = assemble("kernel_stage_in") else {
        return;
    };
    const ENTRY: &str = "stage_in_region_short";
    const GROUP: u32 = 4;
    const GROUPS: u32 = 2;
    const ORIGIN: [u32; 3] = [3, 5, 7];
    const SIZE: [u32; 3] = [11, 13, 1];
    let threads = (GROUP * GROUPS) as usize;
    let mut region_in = Vec::new();
    for v in [ORIGIN, SIZE] {
        for word in v.into_iter().chain([0]) {
            region_in.extend_from_slice(&word.to_le_bytes());
        }
    }
    let lane = |x: u32| u16::try_from(x).expect("a ushort");
    let value = [ORIGIN[0], ORIGIN[1], SIZE[0], SIZE[1]].map(lane);
    let mut expected: Vec<u8> = (0..threads)
        .flat_map(|_| value.iter().flat_map(|half| half.to_le_bytes()))
        .collect();
    // One spare element past the grid, which the kernel must not write.
    expected.extend_from_slice(&[0; 8]);
    for stem in PROFILES {
        let p = profile(stem, identity()).unwrap_or_else(|e| panic!("{e}"));
        let (a, plan, module) =
            compile_kept(&bc, ENTRY, &p).unwrap_or_else(|e| panic!("{stem}: {e}"));
        assert!(a.interface.descriptors.is_empty(), "{stem}");
        link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{stem}: L3: {e}"));
        let mut push = vec![0u8; plan.push.push_range.map_or(0, |(_, end)| end as usize)];
        let region = region_bytes([0; 3], [GROUP * GROUPS, 1, 1], [GROUP, 1, 1]).expect("packs");
        let mut read = BTreeSet::new();
        for f in &plan.push.fields {
            let PushLoc::Push { offset } = f.loc else {
                panic!("{stem}: {f:?} spilled");
            };
            let o = offset as usize;
            match f.field {
                PushField::SlotGuestBase(s) if s == slot(0) => {
                    push[o..o + 8].copy_from_slice(&slot_word(OUT).to_le_bytes())
                }
                PushField::SlotBounds(s) if s == slot(0) => {
                    push[o..o + 8].copy_from_slice(&bounds_word(expected.len(), 0))
                }
                PushField::ZeroAddress => {
                    push[o..o + 8].copy_from_slice(&ZERO_UNMAPPED.to_le_bytes())
                }
                PushField::DispatchRegion => push[o..o + 32].copy_from_slice(&region),
                PushField::StageInRegion => push[o..o + 32].copy_from_slice(&region_in),
                _ => panic!("{stem}: unexpected push field {f:?}"),
            }
            read.insert(f.field);
        }
        assert!(read.contains(&PushField::StageInRegion), "{stem}: {read:?}");
        let res = Resources {
            buffers: vec![Buffer {
                slot: None,
                address: Some(OUT),
                bytes: vec![0u8; expected.len()],
            }],
            push,
            spec: local_size(GROUP),
        };
        let cfg = Config {
            workgroups: [GROUPS, 1, 1],
            ..Config::default()
        };
        let got = run(&module, &cfg, res).unwrap_or_else(|e| panic!("{stem}: {e}"));
        assert_eq!(got.buffers[0], expected, "{stem}: out");
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
    Device::from_queries(desc, identity())
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
/// argument buffer's own, and an ID's index past the arrays is masked to its low 16 bits.
#[test]
fn argument_buffer_textures_are_the_set_1_elements_their_resource_ids_name() {
    textures_at_their_resource_ids(true);
}

/// The address of the argument buffer of [`texture_kernel`]`(true)`.
const AB: u64 = 0x4_0000_0100;

/// An index past the set-1 image arrays by their length: masked to its low 16 bits, index `i +
/// PAST` is element `i`.
const PAST: u32 = 0x1_0000;

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
        // and `other` bound. With `ab`, a third holds the second's IDs with their indices
        // `PAST` higher, past the arrays: the shader masks an argument buffer's index to its low
        // 16 bits, so they read the same elements.
        let dispatches = [(red, 0, 1.0f32, 0), (rgba, 1, 0.0, 0), (rgba, 1, 0.0, PAST)];
        for &(word, other, alpha, past) in &dispatches[..2 + usize::from(ab)] {
            let oob = [0, 0, 0, f(alpha)];
            let robust = (!checked).then_some(oob);
            let other = match other {
                0 => 0,
                index => resource_id(index + past, rgba),
            };
            let ids = [
                resource_id(SRC + past, word),
                resource_id(DST + past, rgba),
                other,
            ];
            let mut push = vec![0u8; plan.push.push_range.map_or(0, |(_, end)| end as usize)];
            let mut fields = Vec::new();
            for field in &plan.push.fields {
                let PushLoc::Push { offset } = field.loc else {
                    panic!("{stem}: {field:?} spilled");
                };
                let o = offset as usize;
                let bytes = match field.field {
                    PushField::SlotGuestBase(s) => match (ab, s.class, s.index) {
                        (false, SlotClass::Buffer, 0) | (true, SlotClass::Buffer, 1) => {
                            slot_word(OUT)
                        }
                        (false, SlotClass::Texture, i @ 0..=2) => ids[i as usize],
                        (true, SlotClass::Buffer, 0) => slot_word(AB),
                        _ => panic!("{stem}: unexpected slot {s:?}"),
                    },
                    PushField::DispatchRegion => {
                        let region = region_bytes([0; 3], L, L).expect("packs");
                        push[o..o + 32].copy_from_slice(&region);
                        fields.push(field.field);
                        continue;
                    }
                    // The check's fields, which the count below leaves out: the output's
                    // length, the argument buffer's, and the zero range.
                    PushField::SlotBounds(s) => {
                        let bytes = match (ab, s.index) {
                            (false, 0) | (true, 1) => 4 * 21,
                            _ => 8 * ids.len(),
                        };
                        push[o..o + 8].copy_from_slice(&bounds_word(bytes, 0));
                        continue;
                    }
                    PushField::ZeroAddress => ZERO_UNMAPPED,
                    _ => panic!("{stem}: unexpected push field {field:?}"),
                };
                push[o..o + 8].copy_from_slice(&bytes.to_le_bytes());
                if field.field != PushField::ZeroAddress {
                    fields.push(field.field);
                }
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
            assert_eq!(
                got.buffers[0], want,
                "{stem}: out, alpha {alpha}, indices {past:#x} past"
            );
            let dst = &images[(SRC + 1 + DST) as usize];
            let written: Vec<[u32; 4]> = (0..2)
                .flat_map(|y| (0..5).map(move |x| (x, y)))
                .map(|(x, y)| read(x, y))
                .collect();
            assert_eq!(
                dst.texels, written,
                "{stem}: dst, alpha {alpha}, indices {past:#x} past"
            );
        }
    }
}

/// A kernel over a `4 × 2` grid that adds 5 to the texel at its position of `tex` (a
/// `texture2d<uint, access::read_write>`, slot 0) by `atomic_fetch_add`, and stores the texel's
/// old first component at word `y · 4 + x` of `out` (buffer 0).
const ATOMIC_KERNEL: &str = "target datalayout = \"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32\"\n\
target triple = \"air64_v28-apple-macosx26.0.0\"\n\n\
define void @ta(ptr addrspace(1) %tex, ptr addrspace(1) noundef \"air-buffer-no-alias\" %out, <2 x i32> noundef %tid) {\n\
entry:\n\
  %old = tail call <4 x i32> @air.atomic_fetch_add_explicit_texture_2d.u.v4i32(ptr addrspace(1) %tex, <2 x i32> %tid, <2 x i32> zeroinitializer, <4 x i32> <i32 5, i32 5, i32 5, i32 5>, i32 0, i32 3)\n\
  %x = extractelement <4 x i32> %old, i64 0\n\
  %tx = extractelement <2 x i32> %tid, i64 0\n\
  %ty = extractelement <2 x i32> %tid, i64 1\n\
  %row = mul i32 %ty, 4\n\
  %idx = add i32 %row, %tx\n\
  %e = zext i32 %idx to i64\n\
  %p = getelementptr inbounds i32, ptr addrspace(1) %out, i64 %e\n\
  store i32 %x, ptr addrspace(1) %p, align 4\n\
  ret void\n\
}\n\n\
declare <4 x i32> @air.atomic_fetch_add_explicit_texture_2d.u.v4i32(ptr addrspace(1), <2 x i32>, <2 x i32>, <4 x i32>, i32, i32)\n\n\
!air.kernel = !{!9}\n!air.version = !{!17}\n!air.language_version = !{!18}\n\
!9 = !{ptr @ta, !10, !11}\n!10 = !{}\n!11 = !{!12, !13, !14}\n\
!12 = !{i32 0, !\"air.texture\", !\"air.location_index\", i32 0, i32 1, !\"air.read_write\", !\"air.arg_type_name\", !\"texture2d<uint, read_write>\", !\"air.arg_name\", !\"tex\"}\n\
!13 = !{i32 1, !\"air.buffer\", !\"air.location_index\", i32 0, i32 1, !\"air.read_write\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"out\"}\n\
!14 = !{i32 2, !\"air.thread_position_in_grid\", !\"air.arg_type_name\", !\"uint2\", !\"air.arg_name\", !\"tid\"}\n\
!17 = !{i32 2, i32 8, i32 0}\n!18 = !{!\"Metal\", i32 4, i32 0, i32 0}\n";

/// A texture atomic's slot under the identity domains ([`ATOMIC_KERNEL`]: `atomic_fetch_add` on a
/// `texture2d<uint, access::read_write>`) is declared `R32ui`, a class of its own whose variable is
/// the storage-image array's (set 1, binding 1). On the CPU interpreter each thread adds to its
/// texel of the element the slot's pushed resource ID names and answers the texel it held, with the
/// atomic left to the device where it drops one outside the image, and guarded by the image's size
/// where it does not (no `robustImageAccess`); no other element changes.
#[test]
fn a_texture_atomic_is_the_r32_storage_element_its_resource_id_names() {
    let Some(bc) = assemble_text("ta", ATOMIC_KERNEL) else {
        return;
    };
    const L: [u32; 3] = [4, 2, 1];
    // The texture at index 2 of the storage array; the elements below it are other textures.
    const TEX: u32 = 2;
    let word = vector_word(PixelFormat::R32Uint);
    let held = |x: u32, y: u32| [0x100 * (y * 4 + x + 1), 0, 0, 1];
    let filler = |_: u32, _: u32| [0xdead_beef; 4];
    let mut profiles: Vec<(String, DeviceProfile)> = PROFILES
        .iter()
        .map(|stem| {
            let p = profile(stem, identity()).unwrap_or_else(|e| panic!("{e}"));
            (stem.to_string(), p)
        })
        .collect();
    profiles.push((
        "moltenvk-m3 without robust image access".to_string(),
        identity_without(
            "moltenvk-m3",
            &[VkFeature::RobustImageAccess, VkFeature::RobustImageAccess2],
        ),
    ));
    for (stem, p) in &profiles {
        let (a, plan, module) =
            compile_kept(&bc, "ta", p).unwrap_or_else(|e| panic!("{stem}: {e}"));
        link_and_validate(&a, p).unwrap_or_else(|e| panic!("{stem}: L3: {e}"));
        let linked = link(&[&a], &LinkInputs::standalone(), p).unwrap_or_else(|e| panic!("{e}"));
        let classes: Vec<_> = (linked.table.bindings.iter())
            .filter_map(|(slot, at)| match slot.kind {
                ResourceKind::Heap(HandleClass::Image(c)) => Some((*slot, c, *at)),
                _ => None,
            })
            .collect();
        let [(storage, class, at)] = classes[..] else {
            panic!("{stem}: {classes:?}");
        };
        assert_eq!(
            (class.access, class.sampled_type, class.format, at),
            (
                ImageAccess::Storage,
                SampledType::Uint,
                StorageFormatClass::Format(vk::Format::R32_UINT),
                (1, 1)
            ),
            "{stem}"
        );
        // The guard reads the element's size; a device that drops the atomic outside has none.
        let queries = (module.functions().iter())
            .flat_map(|f| f.blocks.iter())
            .flat_map(|b| module.blocks()[b.index()].insts.iter())
            .filter(|n| matches!(n.inst, crate::spirv::ir::Inst::ImageQuery { .. }))
            .count();
        let guarded = !p.fields().images.robust_image_access;
        assert_eq!(queries > 0, guarded, "{stem}: {queries} size queries");
        let mut push = vec![0u8; plan.push.push_range.map_or(0, |(_, end)| end as usize)];
        for field in &plan.push.fields {
            let PushLoc::Push { offset } = field.loc else {
                panic!("{stem}: {field:?} spilled");
            };
            let o = offset as usize;
            let bytes = match field.field {
                PushField::SlotGuestBase(s) if s.class == SlotClass::Buffer => slot_word(OUT),
                PushField::SlotGuestBase(s) if s.class == SlotClass::Texture => {
                    resource_id(TEX, word)
                }
                PushField::DispatchRegion => {
                    let region = region_bytes([0; 3], L, L).expect("packs");
                    push[o..o + 32].copy_from_slice(&region);
                    continue;
                }
                PushField::SlotBounds(_) => {
                    push[o..o + 8].copy_from_slice(&bounds_word(4 * 8, 0));
                    continue;
                }
                PushField::ZeroAddress => ZERO_UNMAPPED,
                _ => panic!("{stem}: unexpected push field {field:?}"),
            };
            push[o..o + 8].copy_from_slice(&bytes.to_le_bytes());
        }
        let mut images: Vec<Image> = (0..TEX)
            .map(|e| texture(storage, e, [1, 1], filler, None))
            .collect();
        images.push(texture(storage, TEX, [4, 2], held, None));
        let res = Resources {
            buffers: vec![Buffer {
                slot: None,
                address: Some(OUT),
                bytes: vec![0; 4 * 8],
            }],
            push,
            spec: local_size3(L),
        };
        let (got, images) = run_with_images(&module, &Config::default(), res, images)
            .unwrap_or_else(|e| panic!("{stem}: {e}"));
        let texels = || (0..2).flat_map(|y| (0..4).map(move |x| (x, y)));
        let old: Vec<u8> = texels()
            .flat_map(|(x, y)| held(x, y)[0].to_le_bytes())
            .collect();
        assert_eq!(got.buffers[0], old, "{stem}: out");
        let added: Vec<[u32; 4]> = texels()
            .map(|(x, y)| {
                let [r, g, b, a] = held(x, y);
                [r + 5, g, b, a]
            })
            .collect();
        assert_eq!(images[TEX as usize].texels, added, "{stem}: the texture");
        for other in &images[..TEX as usize] {
            assert_eq!(
                other.texels,
                [[0xdead_beef; 4]],
                "{stem}: element {}",
                other.element
            );
        }
    }
}

/// The `RG32Uint` words of texel `n` of element `e` and the `ulong` they hold (low word first).
fn ulong_texel(e: u32, n: u32) -> ([u32; 4], u64) {
    let lo = 0xA000_0000 | e << 20 | n;
    let hi = 0x0B00_0000 | e << 20 | n << 4;
    ([lo, hi, 0, 1], u64::from(hi) << 32 | u64::from(lo))
}

/// A kernel's plain `ulong` texture slots under the identity domains (the owned fixture
/// `unk_ulong_access`: a sample-access and a read-access `texture2d<ulong>`, and a write-access
/// `texture3d<ulong>`) are `RG32` unsigned images, elements of the set-1 arrays at the indices
/// their pushed resource IDs hold: the two read slots the sampled array's unsigned 2D class, the
/// written one the storage array's 3D class declared `Rg32ui`. It passes L3 and, on the CPU
/// interpreter over 20 threads, each read joins its element's two words into `(v, 0, 0, 1)`,
/// `(0, 0, 0, 1)` outside the texture (threads 16 and on, rows 2 and 3), and the write stores the
/// sum's two words, inside the texture alone.
#[test]
fn ulong_texture_slots_are_rg32_elements_their_resource_ids_name() {
    let Some(bc) = assemble("unk_ulong_rw") else {
        return;
    };
    const LANES: u32 = 20;
    const L: [u32; 3] = [LANES, 1, 1];
    // The sample-access texture at index 1 and the read-access one at 2 of the sampled array,
    // the written one at 3 of the storage array.
    const S: u32 = 1;
    const R: u32 = 2;
    const W: u32 = 3;
    let word = vector_word(PixelFormat::RG32Uint);
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
        let (a, plan, module) =
            compile_kept(&bc, "unk_ulong_access", p).unwrap_or_else(|e| panic!("{stem}: {e}"));
        link_and_validate(&a, p).unwrap_or_else(|e| panic!("{stem}: L3: {e}"));
        let linked = link(&[&a], &LinkInputs::standalone(), p).unwrap_or_else(|e| panic!("{e}"));
        let mut classes: Vec<_> = (linked.table.bindings.iter())
            .filter_map(|(slot, at)| match slot.kind {
                ResourceKind::Heap(HandleClass::Image(c)) => Some((*at, *slot, c)),
                _ => None,
            })
            .collect();
        classes.sort_by_key(|(at, _, _)| *at);
        let [(sampled_at, sampled, s_class), (storage_at, storage, w_class)] = classes[..] else {
            panic!("{stem}: {classes:?}");
        };
        assert_eq!(
            (
                (
                    sampled_at,
                    s_class.dim,
                    s_class.access,
                    s_class.sampled_type,
                    s_class.format
                ),
                (
                    storage_at,
                    w_class.dim,
                    w_class.access,
                    w_class.sampled_type,
                    w_class.format
                ),
            ),
            (
                (
                    (1, 0),
                    ImageDim::D2,
                    ImageAccess::Sampled,
                    SampledType::Uint,
                    StorageFormatClass::NotStorage
                ),
                (
                    (1, 1),
                    ImageDim::D3,
                    ImageAccess::Storage,
                    SampledType::Uint,
                    StorageFormatClass::Format(vk::Format::R32G32_UINT)
                ),
            ),
            "{stem}"
        );
        let out_bytes = 2 * 32 * LANES as usize;
        let mut push = vec![0u8; plan.push.push_range.map_or(0, |(_, end)| end as usize)];
        for field in &plan.push.fields {
            let PushLoc::Push { offset } = field.loc else {
                panic!("{stem}: {field:?} spilled");
            };
            let o = offset as usize;
            let bytes = match field.field {
                PushField::SlotGuestBase(s) => match (s.class, s.index) {
                    (SlotClass::Buffer, 0) => slot_word(OUT),
                    (SlotClass::Texture, i @ 0..=2) => resource_id([S, R, W][i as usize], word),
                    _ => panic!("{stem}: unexpected slot {s:?}"),
                },
                PushField::DispatchRegion => {
                    let region = region_bytes([0; 3], L, L).expect("packs");
                    push[o..o + 32].copy_from_slice(&region);
                    continue;
                }
                PushField::SlotBounds(_) => {
                    push[o..o + 8].copy_from_slice(&bounds_word(out_bytes, 0));
                    continue;
                }
                PushField::ZeroAddress => ZERO_UNMAPPED,
                _ => panic!("{stem}: unexpected push field {field:?}"),
            };
            push[o..o + 8].copy_from_slice(&bytes.to_le_bytes());
        }
        // An `RG32` view's read outside the image is the device's under `robustImageAccess2`;
        // without it the shader checks every read, and the read images trap on one outside them.
        // A write outside the image is the device's to drop under `robustImageAccess` at either
        // level (`contain::device_drops_texel_writes`), and the shader's to guard without it, so
        // the written image traps on one only there.
        let robustness = &p.fields().images;
        let reads = robustness.robust_image_access2.then_some([0, 0, 0, 1]);
        let writes = robustness.robust_image_access.then_some([0, 0, 0, 1]);
        let image = |slot, e: u32, robust| {
            texture(slot, e, [8, 2], |x, y| ulong_texel(e, y * 8 + x).0, robust)
        };
        // An array's elements are bound from 0: those below the named ones, which no thread
        // reaches, hold one texel.
        let filler = |slot, e: u32| texture(slot, e, [1, 1], |_, _| [0xdead_beef; 4], None);
        let mut images = vec![
            filler(sampled, 0),
            image(sampled, S, reads),
            image(sampled, R, reads),
        ];
        images.extend((0..W).map(|e| filler(storage, e)));
        images.push(image(storage, W, writes));
        let written = images.len() - 1;
        let res = Resources {
            buffers: vec![Buffer {
                slot: None,
                address: Some(OUT),
                bytes: vec![0xAB; out_bytes],
            }],
            push,
            spec: local_size3(L),
        };
        let (got, after) = run_with_images(&module, &Config::default(), res, images)
            .unwrap_or_else(|e| panic!("{stem}: {e}"));
        let out: Vec<u64> = (got.buffers[0].as_chunks::<8>().0.iter())
            .map(|c| u64::from_le_bytes(*c))
            .collect();
        for t in 0..LANES {
            let n = (t & 7) + 8 * ((t >> 3) & 1);
            let at = 8 * t as usize;
            if t < 16 {
                let (vs, vr) = (ulong_texel(S, n).1, ulong_texel(R, n).1);
                assert_eq!(out[at..at + 4], [vs, 0, 0, 1], "{stem}: lane {t} sampled");
                assert_eq!(out[at + 4..at + 8], [vr, 0, 0, 1], "{stem}: lane {t} read");
                let sum = vs.wrapping_add(vr);
                let stored = after[written].texels[n as usize];
                assert_eq!(
                    [stored[0], stored[1]],
                    [sum as u32, (sum >> 32) as u32],
                    "{stem}: lane {t} written"
                );
            } else {
                assert_eq!(
                    out[at..at + 8],
                    [0, 0, 0, 1, 0, 0, 0, 1],
                    "{stem}: lane {t}"
                );
            }
        }
    }
}

/// The address of the positions [`a_read_through_a_sampler_takes_its_border_texels_from_the_word`]
/// pushes for `rt_arg`'s buffer 0.
const POSITIONS: u64 = 0x5_0000_0000;

/// A slot's read through a runtime sampler that clamps every axis to its border color
/// (`engine/kernel_read_through`'s `rt_arg`: `texture2d<float>`, offset `(1, -1)`) under the
/// identity domains takes the texel an index outside the texture reads from bits 0-11 of the slot's
/// resource ID's word, for the color the sampler's word names: no specialization constant but the
/// dispatch's, and no `BorderTexels` fact. On the CPU interpreter the border texels of
/// `R32Float`, `RGBA8Unorm`, `A8Unorm` and an `RGBA8Unorm` swizzled `(a, b, 0, r)` match Metal's
/// ([`border_texel`](crate::metal::sampler::border_texel)) under each border color, and a read
/// inside the texture reads its texel.
#[test]
fn a_read_through_a_sampler_takes_its_border_texels_from_the_word() {
    use crate::metal::sampler::{border_texel, SamplerWord, BORDER_COLORS};
    use crate::metal::{SamplerAddressMode, SamplerBorderColor, TextureSwizzle};
    let Some(bc) = assemble("kernel_read_through") else {
        return;
    };
    // Thread `i` reads at `P[i] + (1, -1)`: inside at (1, 0) and (1, 1), and past the right, left
    // and top edges of the 2 × 2 texture.
    const P: [[i32; 4]; 5] = [
        [0, 1, 0, 0],
        [1, 2, 0, 0],
        [-2, 1, 0, 0],
        [0, 0, 0, 0],
        [0, 2, 0, 0],
    ];
    const INSIDE: [Option<(u32, u32)>; 5] = [Some((1, 0)), None, None, None, Some((1, 1))];
    const L: [u32; 3] = [5, 1, 1];
    // The texture at index 3 of the sampled array, the sampler at index 1 of its arrays.
    const TEX: u32 = 3;
    const SAMPLER: u32 = 1;
    let swizzled = TextureSwizzleChannels::new(
        TextureSwizzle::Alpha,
        TextureSwizzle::Blue,
        TextureSwizzle::Zero,
        TextureSwizzle::Red,
    );
    let formats = [
        (PixelFormat::R32Float, TextureSwizzleChannels::DEFAULT),
        (PixelFormat::RGBA8Unorm, TextureSwizzleChannels::DEFAULT),
        (PixelFormat::A8Unorm, TextureSwizzleChannels::DEFAULT),
        (PixelFormat::RGBA8Unorm, swizzled),
    ];
    let colors = [
        SamplerBorderColor::TransparentBlack,
        SamplerBorderColor::OpaqueBlack,
        SamplerBorderColor::OpaqueWhite,
    ];
    let f = |v: f32| v.to_bits();
    let held = |x: u32, y: u32| [f(x as f32 + 0.25), f(y as f32 + 0.5), f(0.125), f(0.75)];
    let filler = |_: u32, _: u32| [0xdead_beef; 4];
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
        let (a, plan, module) =
            compile_kept(&bc, "rt_arg", p).unwrap_or_else(|e| panic!("{stem}: {e}"));
        link_and_validate(&a, p).unwrap_or_else(|e| panic!("{stem}: L3: {e}"));
        assert!(
            (a.interface.spec.iter()).all(|c| dispatch_key(c.key)),
            "{stem}: {:?}",
            a.interface.spec
        );
        assert!(
            !(plan.facts.iter()).any(|f| matches!(f, RuntimeFact::BorderTexels { .. })),
            "{stem}: {:?}",
            plan.facts
        );
        let linked = link(&[&a], &LinkInputs::standalone(), p).unwrap_or_else(|e| panic!("{e}"));
        let sampled: Vec<ResourceSlot> = (linked.table.bindings.iter())
            .filter(|(slot, _)| matches!(slot.kind, ResourceKind::Heap(HandleClass::Image(_))))
            .map(|(slot, _)| *slot)
            .collect();
        let [sampled] = sampled[..] else {
            panic!("{stem}: {sampled:?}");
        };
        for (pf, swizzle) in formats {
            let word = TextureWord::of(pf, swizzle).expect("a word").0;
            let present = crate::metal::format::channels(pf).expect("channels");
            for (color, rgba) in colors.into_iter().zip(BORDER_COLORS) {
                let mut desc = crate::bind::SamplerDescriptor::new();
                desc.s_address_mode = SamplerAddressMode::ClampToBorderColor;
                desc.t_address_mode = SamplerAddressMode::ClampToBorderColor;
                desc.r_address_mode = SamplerAddressMode::ClampToBorderColor;
                desc.border_color = color;
                let sampler = SamplerWord::of(&desc).0;
                let mut push = vec![0u8; plan.push.push_range.map_or(0, |(_, end)| end as usize)];
                for field in &plan.push.fields {
                    let PushLoc::Push { offset } = field.loc else {
                        panic!("{stem}: {field:?} spilled");
                    };
                    let o = offset as usize;
                    let bytes = match field.field {
                        PushField::SlotGuestBase(s) => match (s.class, s.index) {
                            (SlotClass::Buffer, 0) => slot_word(POSITIONS),
                            (SlotClass::Buffer, 1) => slot_word(OUT),
                            (SlotClass::Texture, 0) => resource_id(TEX, word),
                            (SlotClass::Sampler, 0) => resource_id(SAMPLER, sampler),
                            _ => panic!("{stem}: unexpected slot {s:?}"),
                        },
                        PushField::DispatchRegion => {
                            let region = region_bytes([0; 3], L, L).expect("packs");
                            push[o..o + 32].copy_from_slice(&region);
                            continue;
                        }
                        PushField::SlotBounds(s) => {
                            push[o..o + 8].copy_from_slice(&bounds_word(16 * P.len(), 0));
                            assert_eq!(s.class, SlotClass::Buffer, "{stem}");
                            continue;
                        }
                        PushField::ZeroAddress => ZERO_UNMAPPED,
                        _ => panic!("{stem}: unexpected push field {field:?}"),
                    };
                    push[o..o + 8].copy_from_slice(&bytes.to_le_bytes());
                }
                let mut images: Vec<Image> = (0..TEX)
                    .map(|e| texture(sampled, e, [1, 1], filler, None))
                    .collect();
                images.push(texture(sampled, TEX, [2, 2], held, None));
                let positions = P.iter().flatten().flat_map(|v| v.to_le_bytes()).collect();
                let res = Resources {
                    buffers: vec![
                        Buffer {
                            slot: None,
                            address: Some(POSITIONS),
                            bytes: positions,
                        },
                        Buffer {
                            slot: None,
                            address: Some(OUT),
                            bytes: vec![0; 16 * P.len()],
                        },
                    ],
                    push,
                    spec: local_size3(L),
                };
                let (got, _) = run_with_images(&module, &Config::default(), res, images)
                    .unwrap_or_else(|e| panic!("{stem}: {e}"));
                let border =
                    border_texel(pf, present, swizzle, rgba).map(|one| f(f32::from(u8::from(one))));
                let want: Vec<u8> = (INSIDE.iter())
                    .flat_map(|at| match at {
                        Some((x, y)) => held(*x, *y),
                        None => border,
                    })
                    .flat_map(|lane| lane.to_le_bytes())
                    .collect();
                assert_eq!(got.buffers[1], want, "{stem}: {pf:?} {swizzle:?} {color:?}");
            }
        }
    }
}

/// The read buffers of a kernel of [`many_of`]: with its output, 21 slot addresses of 8 bytes.
const READS: u32 = 20;

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
                (
                    slot_word(BASE + u64::from(s.index) * STRIDE),
                    &mut push,
                    offset,
                )
            }
            (PushField::SlotGuestBase(s), PushLoc::ArgumentTable { offset }) => {
                spilled.push(s);
                (
                    slot_word(BASE + u64::from(s.index) * STRIDE),
                    &mut block,
                    offset,
                )
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

/// On `Device::from_queries`' profile a kernel's push window is the device's, 256 bytes on
/// `PROFILES`: the same kernel's 21 slot addresses are all pushed, with no block, below its tail
/// word 24 bytes above them (`PushPlacement::tail_word`). A kernel's 31, Metal's most buffer slots,
/// would end past the window with the word: its block's address and 27 slot addresses end at 224,
/// the word is at 248, and the other 4 are read from the block. In `runtime-floor`'s window of 128
/// bytes a kernel's 12 are all pushed, and of its 21 and its 31 the block's address and 11 end at
/// 96, the word is at 120, and the others are read from the block. Each computes its answer.
#[test]
fn the_window_pushes_a_kernel_s_slot_addresses_below_its_tail_word() {
    let wide: [(u32, usize); 2] = [(READS, READS as usize + 1), (30, 27)];
    let least: [(u32, usize); 3] = [(11, 12), (READS, 11), (30, 11)];
    let windows: [(&[&str], u32, &[(u32, usize)]); 2] =
        [(&PROFILES, 256, &wide), (&["runtime-floor"], 128, &least)];
    for (stems, window, cases) in windows {
        for &(reads, in_window) in cases {
            let Some(bc) = assemble_text("many", &many_of(reads)) else {
                return;
            };
            for &stem in stems {
                let desc =
                    crate::pipeline::testing::description(stem).unwrap_or_else(|e| panic!("{e}"));
                let device =
                    Device::from_queries(desc, identity()).unwrap_or_else(|e| panic!("{e}"));
                assert_eq!(device.profile().push_window(), window, "{stem}");
                let (first, pushed, spilled, table) =
                    many_placed(stem, device.profile(), &bc, reads);
                let what = format!("{stem}, window {window}, {} slots", reads + 1);
                assert_eq!(pushed.len(), in_window, "{what}");
                assert_eq!(spilled.len(), (reads + 1) as usize - in_window, "{what}");
                let block = (8 * spilled.len() as u32).next_multiple_of(16);
                assert_eq!(table, block, "{what}");
                assert_eq!(
                    first.map(|(f, _)| f) == Some(PushField::ArgumentTableAddress),
                    !spilled.is_empty(),
                    "{what}"
                );
                println!("{what}: {in_window} pushed, {} in the block", spilled.len());
            }
        }
    }
}

/// On `PROFILES` and in `runtime-floor`'s window, [`many_of`]`(3)` pushes its 4 slot addresses
/// below its tail word, and its straight-line module returns once: the tail's test, a compare of
/// the word's load, is the condition of one select per other pushed field and of no branch
/// (`EmitCx::fold_tail`), so the body depends on the tail through those values alone. AMD's open
/// driver (2025.Q2.1) dies making a fragment stage whose body runs under a never-taken return
/// there and holds a loop of convergent instructions under `MaximallyReconvergesKHR`.
#[test]
fn the_floor_s_tail_folds_into_the_other_pushed_fields_with_no_branch() {
    use crate::spirv::ir::{BinOp, Inst, Terminator, Type};
    let reads = 3;
    let Some(bc) = assemble_text("many", &many_of(reads)) else {
        return;
    };
    for stem in PROFILES.iter().copied().chain(["runtime-floor"]) {
        let desc = crate::pipeline::testing::description(stem).unwrap_or_else(|e| panic!("{e}"));
        let device = Device::from_queries(desc, identity()).unwrap_or_else(|e| panic!("{e}"));
        let p = device.profile();
        let (a, plan, m) = compile_kept(&bc, "many", p).unwrap_or_else(|e| panic!("{stem}: {e}"));
        link_and_validate(&a, p).unwrap_or_else(|e| panic!("{stem}: L3: {e}"));
        assert!(plan.push.tail_word.is_some(), "{stem}: no tail word");
        let pushed = (plan.push.fields.iter())
            .filter(|f| matches!(f.loc, PushLoc::Push { .. }))
            .count();
        assert_eq!(pushed, (reads + 1) as usize, "{stem}");
        let insts: Vec<_> = m.blocks().iter().flat_map(|b| &b.insts).collect();
        let push_load = |v: crate::spirv::ids::Value| {
            insts.iter().any(|n| match (&n.inst, n.result) {
                (Inst::Load { ptr, .. }, Some(r)) if r == v => {
                    matches!(
                        m.value(*ptr).and_then(|d| m.types().get(d.ty)),
                        Ok(Type::Pointer {
                            class: crate::spirv::ir::StorageClass::PushConstant,
                            ..
                        })
                    )
                }
                _ => false,
            })
        };
        // The tail's tests: a compare of a push load that some select takes as its condition.
        let tests: Vec<_> = (insts.iter())
            .filter_map(|n| match (&n.inst, n.result) {
                (
                    Inst::Binary {
                        op: BinOp::INotEqual,
                        a,
                        ..
                    },
                    Some(r),
                ) if push_load(*a) => Some(r),
                _ => None,
            })
            .filter(|c| {
                (insts.iter()).any(|n| matches!(n.inst, Inst::Select { cond, .. } if cond == *c))
            })
            .collect();
        assert_eq!(tests.len(), 1, "{stem}: the tail's tests");
        let test = tests[0];
        let selects = (insts.iter())
            .filter(|n| matches!(n.inst, Inst::Select { cond, .. } if cond == test))
            .count();
        assert_eq!(selects, pushed, "{stem}: one select per other pushed field");
        let branches = (m.blocks().iter())
            .filter(|b| matches!(b.term, Terminator::CondBranch { cond, .. } if cond == test))
            .count();
        assert_eq!(branches, 0, "{stem}: a branch on the tail");
        let returns = (m.blocks().iter())
            .filter(|b| matches!(b.term, Terminator::Return))
            .count();
        assert_eq!(returns, 1, "{stem}: the module's returns");
        println!("{stem}: the tail folds into {selects} pushed fields; one return");
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

/// The push bytes of a [`grid_kernel`]: its output at `out.0`, `out.1` bytes long, and the region
/// `region`.
fn grid_push(plan: &Plan, out: (u64, usize), region: [u8; 32]) -> Vec<u8> {
    let mut push = vec![0u8; plan.push.push_range.map_or(0, |(_, end)| end as usize)];
    for f in &plan.push.fields {
        let PushLoc::Push { offset } = f.loc else {
            panic!("{f:?} spilled");
        };
        let o = offset as usize;
        match f.field {
            PushField::SlotGuestBase(s) if s == slot(0) => {
                push[o..o + 8].copy_from_slice(&slot_word(out.0).to_le_bytes())
            }
            PushField::SlotBounds(s) if s == slot(0) => {
                push[o..o + 8].copy_from_slice(&bounds_word(out.1, 0))
            }
            PushField::ZeroAddress => push[o..o + 8].copy_from_slice(&ZERO_UNMAPPED.to_le_bytes()),
            PushField::DispatchRegion => push[o..o + 32].copy_from_slice(&region),
            _ => panic!("unexpected push field {f:?}"),
        }
    }
    push
}

/// The output's address in the grid tests.
const OUT: u64 = 0x6_0000_0000;

/// A kernel's grid builtins ([`grid_kernel`]) read Metal's grid from the pushed region
/// (`IdentityRegion::Sized`): the kernel declares `MultiRegion`, default 0, and reads its base
/// from every region and its threadgroup size from its pipeline unless `MultiRegion` is 1. At 0 a
/// whole dispatch, one split in two pieces and an indirect one store Metal's answers; a ragged
/// grid split as the binder splits it, a region of the threadgroup size at 0 and an edge region
/// of fewer threads at 1, stores Metal's answers for the ragged grid.
#[test]
fn from_queries_reads_every_regions_base_and_its_size_by_multi_region() {
    let Some(bc) = assemble_text("grid", &grid_kernel("grid")) else {
        return;
    };
    const L: [u32; 3] = [4, 2, 1];
    // Each form's grid and dispatches: (region base, region grid, local size, `MultiRegion`,
    // workgroups).
    type Dispatch = ([u32; 3], [u32; 3], [u32; 3], u64, [u32; 3]);
    const GRID: [u32; 3] = [8, 6, 1];
    const RAGGED: [u32; 3] = [7, 6, 1];
    let forms: [(&str, [u32; 3], &[Dispatch]); 4] = [
        ("whole", GRID, &[([0; 3], GRID, L, 0, [2, 3, 1])]),
        (
            "split",
            GRID,
            &[
                ([0; 3], GRID, L, 0, [2, 1, 1]),
                ([0, 1, 0], GRID, L, 0, [2, 2, 1]),
            ],
        ),
        ("indirect", GRID, &[([0; 3], [0; 3], L, 0, [2, 3, 1])]),
        (
            "edge",
            RAGGED,
            &[
                ([0; 3], RAGGED, L, 0, [1, 3, 1]),
                ([1, 0, 0], RAGGED, [3, 2, 1], 1, [1, 3, 1]),
            ],
        ),
    ];
    for stem in PROFILES {
        let desc = crate::pipeline::testing::description(stem).unwrap_or_else(|e| panic!("{e}"));
        let device = Device::from_queries(desc, identity()).unwrap_or_else(|e| panic!("{e}"));
        let p = device.profile();
        let (a, plan, module) =
            compile_kept(&bc, "grid", p).unwrap_or_else(|e| panic!("{stem}: {e}"));
        let declared: Vec<u64> = (a.interface.spec.iter())
            .filter(|c| c.key == SpecKey::MultiRegion)
            .map(|c| c.default)
            .collect();
        assert_eq!(declared, [0], "{stem}");
        link_and_validate(&a, p).unwrap_or_else(|e| panic!("{stem}: L3: {e}"));
        for (what, grid, dispatches) in forms {
            let groups = [grid[0].div_ceil(L[0]), grid[1].div_ceil(L[1])];
            let value = grid[0] | grid[1] << 8 | groups[0] << 16 | groups[1] << 24;
            let threads = (grid[0] * grid[1]) as usize;
            let want: Vec<u8> = (0..threads).flat_map(|_| value.to_le_bytes()).collect();
            let mut out = vec![0u8; 4 * threads];
            for &(base, region_grid, local, multi, workgroups) in dispatches {
                let region = region_bytes(base, region_grid, L).expect("packs");
                let mut spec = local_size3(local);
                spec.insert(SpecKey::MultiRegion, multi);
                let res = Resources {
                    push: grid_push(&plan, (OUT, out.len()), region),
                    buffers: vec![Buffer {
                        slot: None,
                        address: Some(OUT),
                        bytes: out,
                    }],
                    spec,
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

/// A kernel that stores, for its thread `t` of threadgroup `g` along `x`, two words at `2 · (g ·
/// n + t)` of a threadgroup of `n` threads: its position in the threadgroup `x | y << 10 | z << 20`,
/// and the index of thread `n - 1 - t`, which it reads from threadgroup memory that thread wrote
/// before a threadgroup barrier.
fn deep_kernel() -> String {
    "target datalayout = \"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32\"\n\
target triple = \"air64_v28-apple-macosx26.0.0\"\n\n\
@s = internal addrspace(3) global [1024 x i32] undef, align 4\n\n\
define void @deep(ptr addrspace(1) noundef \"air-buffer-no-alias\" %0, <3 x i32> noundef %1, i32 noundef %2, <3 x i32> noundef %3, <3 x i32> noundef %4) {\n\
entry:\n\
  %ps = getelementptr inbounds [1024 x i32], ptr addrspace(3) @s, i32 0, i32 %2\n\
  store i32 %2, ptr addrspace(3) %ps, align 4\n\
  tail call void @air.wg.barrier(i32 2, i32 1)\n\
  %sx = extractelement <3 x i32> %3, i64 0\n\
  %sy = extractelement <3 x i32> %3, i64 1\n\
  %sz = extractelement <3 x i32> %3, i64 2\n\
  %sxy = mul i32 %sx, %sy\n\
  %n = mul i32 %sxy, %sz\n\
  %last = sub i32 %n, 1\n\
  %r = sub i32 %last, %2\n\
  %pr = getelementptr inbounds [1024 x i32], ptr addrspace(3) @s, i32 0, i32 %r\n\
  %v = load i32, ptr addrspace(3) %pr, align 4\n\
  %x = extractelement <3 x i32> %1, i64 0\n\
  %y = extractelement <3 x i32> %1, i64 1\n\
  %z = extractelement <3 x i32> %1, i64 2\n\
  %y10 = shl i32 %y, 10\n\
  %z20 = shl i32 %z, 20\n\
  %xy = or i32 %x, %y10\n\
  %w = or i32 %xy, %z20\n\
  %g = extractelement <3 x i32> %4, i64 0\n\
  %gn = mul i32 %g, %n\n\
  %idx = add i32 %gn, %2\n\
  %i2 = shl i32 %idx, 1\n\
  %i = zext i32 %i2 to i64\n\
  %p = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %i\n\
  store i32 %w, ptr addrspace(1) %p, align 4\n\
  %q = getelementptr inbounds i32, ptr addrspace(1) %p, i64 1\n\
  store i32 %v, ptr addrspace(1) %q, align 4\n\
  ret void\n\
}\n\n\
declare void @air.wg.barrier(i32, i32)\n\n\
!air.kernel = !{!9}\n!air.version = !{!17}\n!air.language_version = !{!18}\n\
!9 = !{ptr @deep, !10, !11}\n!10 = !{}\n!11 = !{!12, !13, !14, !15, !16}\n\
!12 = !{i32 0, !\"air.buffer\", !\"air.location_index\", i32 0, i32 1, !\"air.read_write\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"out\"}\n\
!13 = !{i32 1, !\"air.thread_position_in_threadgroup\", !\"air.arg_type_name\", !\"uint3\", !\"air.arg_name\", !\"pos\"}\n\
!14 = !{i32 2, !\"air.thread_index_in_threadgroup\", !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"t\"}\n\
!15 = !{i32 3, !\"air.threads_per_threadgroup\", !\"air.arg_type_name\", !\"uint3\", !\"air.arg_name\", !\"size\"}\n\
!16 = !{i32 4, !\"air.threadgroup_position_in_grid\", !\"air.arg_type_name\", !\"uint3\", !\"air.arg_name\", !\"group\"}\n\
!17 = !{i32 2, i32 8, i32 0}\n!18 = !{!\"Metal\", i32 4, i32 0, i32 0}\n"
        .to_string()
}

/// The workgroup `module` declares under `spec`: its `WorkgroupSize` composite, each part a
/// constant, a specialization constant or an operation of them, at those values.
fn workgroup_of(module: &TypedModule, spec: &BTreeMap<SpecKey, u64>) -> [u32; 3] {
    fn value(module: &TypedModule, spec: &BTreeMap<SpecKey, u64>, c: ConstId) -> u64 {
        match &module.consts().get(c).expect("a constant").kind {
            ConstKind::Plain(crate::spirv::ir::Constant::Scalar { bits, .. }) => bits.0,
            ConstKind::Spec { key, default } => spec.get(key).copied().unwrap_or(default.0),
            ConstKind::SpecOp { op, operands } => {
                let values: Vec<u64> = (operands.iter()).map(|&o| value(module, spec, o)).collect();
                op.eval(32, &values).expect("an operation of its arity")
            }
            other => panic!("a workgroup size part {other:?}"),
        }
    }
    let size = module
        .consts()
        .workgroup_size_const()
        .expect("a WorkgroupSize");
    let ConstKind::WorkgroupSize { parts } = module.consts().get(size).expect("it").kind else {
        panic!("not a WorkgroupSize");
    };
    parts.map(|part| value(module, spec, part) as u32)
}

/// A kernel whose workgroup is its threadgroup declares `SpecKey::DeepThreadgroup`, default false.
/// Set for a threadgroup deeper than `DEEP_THREADGROUP`, its workgroup is one row of the
/// threadgroup's threads and its threads' positions, linear indices and threadgroup memory are
/// Metal's: [`deep_kernel`] over two threadgroups of 1 by 1 by 320, 2 by 3 by 100 and 1 by 4 by 256
/// stores what Metal's do, and so does one of 4 by 4 by 4, at its own workgroup with the key unset.
/// The device reports Metal's 1,024 threads along every axis, on a profile of 64 along `z` too.
#[test]
fn a_deep_threadgroup_runs_as_one_row_on_from_queries() {
    use crate::pipeline::DEEP_THREADGROUP;
    let Some(bc) = assemble_text("deep", &deep_kernel()) else {
        return;
    };
    for stem in PROFILES {
        let desc = crate::pipeline::testing::description(stem).unwrap_or_else(|e| panic!("{e}"));
        let device = Device::from_queries(desc, identity()).unwrap_or_else(|e| panic!("{e}"));
        let p = device.profile();
        let (a, plan, module) =
            compile_kept(&bc, "deep", p).unwrap_or_else(|e| panic!("{stem}: {e}"));
        let declared: Vec<u64> = (a.interface.spec.iter())
            .filter(|c| c.key == SpecKey::DeepThreadgroup)
            .map(|c| c.default)
            .collect();
        assert_eq!(declared, [0], "{stem}");
        link_and_validate(&a, p).unwrap_or_else(|e| panic!("{stem}: L3: {e}"));
        for size in [[1, 1, 320], [2, 3, 100], [1, 4, 256], [4, 4, 4]] {
            let n = size.iter().product::<u32>();
            let deep = size[2] > DEEP_THREADGROUP;
            let mut spec = local_size3(size);
            spec.insert(SpecKey::DeepThreadgroup, u64::from(deep));
            let want_group = match deep {
                true => [n, 1, 1],
                false => size,
            };
            assert_eq!(workgroup_of(&module, &spec), want_group, "{stem} {size:?}");
            let mut want = Vec::new();
            for _ in 0..2 {
                for t in 0..n {
                    let (x, y, z) = (t % size[0], t / size[0] % size[1], t / (size[0] * size[1]));
                    want.extend((x | y << 10 | z << 20).to_le_bytes());
                    want.extend((n - 1 - t).to_le_bytes());
                }
            }
            let grid = [2 * size[0], size[1], size[2]];
            let region = region_bytes([0; 3], grid, size).expect("packs");
            let res = Resources {
                buffers: vec![Buffer {
                    slot: None,
                    address: Some(OUT),
                    bytes: vec![0; want.len()],
                }],
                push: grid_push(&plan, (OUT, want.len()), region),
                spec,
            };
            let cfg = Config {
                workgroups: [2, 1, 1],
                ..Config::default()
            };
            let got = run(&module, &cfg, res).unwrap_or_else(|e| panic!("{stem} {size:?}: {e}"));
            assert_eq!(got.buffers[0], want, "{stem} {size:?}");
        }
    }
    let desc = crate::pipeline::testing::description("nvidia-rtx50-linux")
        .unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(desc.limits.max_compute_work_group_size[2], 64);
    let queries = Device::from_queries(desc, identity()).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(
        queries.metal_properties().max_threads_per_threadgroup,
        [1024; 3]
    );
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
            push: grid_push(&plan, (OUT, 4 * words), region),
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

/// A kernel that does not observe its threadgroup partition, storing at its position in the grid
/// `(x, y, z)` of a grid `(gx, gy, gz)` the word `1 + x + (y << 10) + (z << 20)`, at
/// `(z · gy + y) · gx + x`.
fn edge_kernel() -> String {
    "target datalayout = \"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32\"\n\
target triple = \"air64_v28-apple-macosx26.0.0\"\n\n\
define void @edge(ptr addrspace(1) noundef \"air-buffer-no-alias\" %0, <3 x i32> noundef %1, <3 x i32> noundef %2) {\n\
entry:\n\
  %tx = extractelement <3 x i32> %1, i64 0\n\
  %ty = extractelement <3 x i32> %1, i64 1\n\
  %tz = extractelement <3 x i32> %1, i64 2\n\
  %gx = extractelement <3 x i32> %2, i64 0\n\
  %gy = extractelement <3 x i32> %2, i64 1\n\
  %plane = mul i32 %tz, %gy\n\
  %rows = add i32 %plane, %ty\n\
  %row = mul i32 %rows, %gx\n\
  %idx = add i32 %row, %tx\n\
  %hy = shl i32 %ty, 10\n\
  %hz = shl i32 %tz, 20\n\
  %xy = or i32 %tx, %hy\n\
  %xyz = or i32 %xy, %hz\n\
  %v = add i32 %xyz, 1\n\
  %i = zext i32 %idx to i64\n\
  %p = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %i\n\
  store i32 %v, ptr addrspace(1) %p, align 4\n\
  ret void\n\
}\n\n\
!air.kernel = !{!9}\n!air.version = !{!17}\n!air.language_version = !{!18}\n\
!9 = !{ptr @edge, !10, !11}\n!10 = !{}\n!11 = !{!12, !13, !14}\n\
!12 = !{i32 0, !\"air.buffer\", !\"air.location_index\", i32 0, i32 1, !\"air.read_write\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"out\"}\n\
!13 = !{i32 1, !\"air.thread_position_in_grid\", !\"air.arg_type_name\", !\"uint3\", !\"air.arg_name\", !\"tid\"}\n\
!14 = !{i32 2, !\"air.threads_per_grid\", !\"air.arg_type_name\", !\"uint3\", !\"air.arg_name\", !\"grid\"}\n\
!17 = !{i32 2, i32 8, i32 0}\n!18 = !{!\"Metal\", i32 4, i32 0, i32 0}\n"
        .to_string()
}

/// The `OpUDiv`s and `OpUMod`s of disassembly `text` whose divisor is not a constant or a
/// specialization constant: a division the driver keeps once it specializes the module.
fn runtime_divisions(text: &str) -> usize {
    let mut constants = BTreeSet::new();
    let mut count = 0;
    for line in text.lines() {
        let words: Vec<&str> = line.split_whitespace().collect();
        match words[..] {
            [id, "=", op, ..]
                if op.starts_with("OpConstant") || op.starts_with("OpSpecConstant") =>
            {
                constants.insert(id);
            }
            [_, "=", "OpUDiv" | "OpUMod", _, _, divisor] if !constants.contains(divisor) => {
                count += 1
            }
            _ => {}
        }
    }
    count
}

/// The words [`edge_kernel`] stores over `grid` where each of its threads runs once: each
/// position's word at its index, and `len` words in all.
fn edge_words(grid: [u32; 3], len: usize) -> Vec<u8> {
    let mut want = vec![0u8; 4 * len];
    for z in 0..grid[2] {
        for y in 0..grid[1] {
            for x in 0..grid[0] {
                let at = 4 * ((z * grid[1] + y) * grid[0] + x) as usize;
                let v = 1 + x + (y << 10) + (z << 20);
                want[at..at + 4].copy_from_slice(&v.to_le_bytes());
            }
        }
    }
    want
}

/// The partial-threadgroups variant of a kernel that does not observe its threadgroup partition
/// runs its grid in tiles (`RuntimeFact::GridTiles`): over
/// grids ragged along every axis, along one, and whole, each in `grid_tile`'s tile, one workgroup
/// of `GRID_TILE_THREADS` with the tile as the region's nominal size at `MultiRegion` 1, the
/// region flagged [`NONUNIFORM`], each of the grid's threads stores Metal's position once and no
/// invocation past the grid stores anything; unflagged, a whole grid does the same and a ragged
/// one's invocations past the grid store too, so the flag alone guards the test of the grid's
/// edge; an indirect dispatch at `MultiRegion` 0, its region stating no
/// grid, runs at its own size, every invocation storing. Its module divides by no value the driver
/// does not know when it specializes it.
#[test]
fn an_oblivious_kernels_partial_variant_runs_its_grid_in_tiles() {
    use crate::lower::dispatch::{grid_tile, GRID_TILE_THREADS};
    let Some(bc) = assemble_text("edge", &edge_kernel()) else {
        return;
    };
    let mut inputs = LowerInputs::new();
    inputs.partial_threadgroups = true;
    for stem in PROFILES {
        let desc = crate::pipeline::testing::description(stem).unwrap_or_else(|e| panic!("{e}"));
        let device = Device::from_queries(desc, identity()).unwrap_or_else(|e| panic!("{e}"));
        let p = device.profile();
        let a = compile_with(&bc, "edge", p, &inputs).unwrap_or_else(|e| panic!("{stem}: {e}"));
        assert!(a.facts.contains(&RuntimeFact::GridTiles), "{stem}");
        // The tile's workgroup takes its position from the region only where `MultiRegion` is 1,
        // so a runtime running the tiles must be able to set it.
        assert!(
            a.interface
                .spec
                .iter()
                .any(|c| c.key == SpecKey::MultiRegion),
            "{stem}: GridTiles without MultiRegion"
        );
        let (_, module_words) =
            link_and_validate(&a, p).unwrap_or_else(|e| panic!("{stem}: L3: {e}"));
        let text = spirv_dis(&module_words);
        assert_eq!(runtime_divisions(&text), 0, "{stem}: from_queries\n{text}");
        let (plan, module) =
            typed_module_with(&bc, "edge", p, &inputs).unwrap_or_else(|e| panic!("{stem}: {e}"));
        // Ragged along every axis (each odd, so the tile takes no side for free), a grid of 8x8
        // threadgroups whole in 8x8x4, ragged along `z` alone, a row, and a pillar.
        for grid in [[7, 5, 3], [24, 24, 8], [8, 8, 22], [300, 1, 1], [1, 1, 300]] {
            let tile = grid_tile(grid);
            let groups: [u32; 3] = std::array::from_fn(|i| grid[i].div_ceil(tile[i]));
            // Every word a launched invocation could store at, so a store past the grid shows.
            let len = ((groups[2] * tile[2] * grid[1] + groups[1] * tile[1]) * grid[0]
                + groups[0] * tile[0]) as usize;
            let whole = (0..3).all(|i| grid[i].is_multiple_of(tile[i]));
            for flagged in [true, false] {
                let region = match flagged {
                    true => nonuniform_region_bytes([0; 3], grid, tile),
                    false => region_bytes([0; 3], grid, tile),
                }
                .expect("packs");
                let mut spec = local_size3([GRID_TILE_THREADS, 1, 1]);
                spec.insert(SpecKey::MultiRegion, 1);
                let res = Resources {
                    buffers: vec![Buffer {
                        slot: None,
                        address: Some(OUT),
                        bytes: vec![0; 4 * len],
                    }],
                    push: grid_push(&plan, (OUT, 4 * len), region),
                    spec,
                };
                let cfg = Config {
                    workgroups: groups,
                    ..Config::default()
                };
                let got = run(&module, &cfg, res)
                    .unwrap_or_else(|e| panic!("{stem} {grid:?} {flagged}: {e}"));
                assert_eq!(
                    got.buffers[0] == edge_words(grid, len),
                    flagged || whole,
                    "{stem} {grid:?} {tile:?} flagged {flagged}"
                );
            }
        }
        // An indirect dispatch: 2 x 3 threadgroups of 4 x 2, its grid theirs.
        let res = Resources {
            buffers: vec![Buffer {
                slot: None,
                address: Some(OUT),
                bytes: vec![0; 4 * 48],
            }],
            push: grid_push(
                &plan,
                (OUT, 4 * 48),
                region_bytes([0; 3], [0; 3], [4, 2, 1]).expect("packs"),
            ),
            spec: local_size3([4, 2, 1]),
        };
        let cfg = Config {
            workgroups: [2, 3, 1],
            ..Config::default()
        };
        let got = run(&module, &cfg, res).unwrap_or_else(|e| panic!("{stem} indirect: {e}"));
        assert_eq!(got.buffers[0], edge_words([8, 6, 1], 48), "{stem} indirect");
    }
}

/// `Device::kernel_observes_partition` answers, lowering nothing, what the compile of the same
/// kernel under the same inputs answers (`CompiledFunction::observes_threadgroup_partition`): no
/// for a kernel of the grid alone, yes for one that reads its threadgroup's position, on both
/// profiles.
#[test]
fn the_partition_query_answers_what_the_compile_does() {
    let kernels = [
        ("edge", edge_kernel(), false),
        ("grid", grid_kernel("grid"), true),
    ];
    for (name, text, observes) in kernels {
        let Some(bc) = assemble_text(name, &text) else {
            return;
        };
        for stem in PROFILES {
            let desc =
                crate::pipeline::testing::description(stem).unwrap_or_else(|e| panic!("{e}"));
            for dev in [
                Device::from_queries(desc.clone(), identity()),
                Device::from_queries(desc, identity()),
            ] {
                let dev = dev.unwrap_or_else(|e| panic!("{e}"));
                let lib = Library::from_air(&bc).unwrap_or_else(|e| panic!("{stem}: {e}"));
                let f =
                    FunctionDescriptor::new(lib.function(name).unwrap_or_else(|e| panic!("{e}")));
                let inputs = LowerInputs::new();
                let entries = crate::pipeline::StageEntries::new();
                let answer = dev
                    .kernel_observes_partition(&f, &inputs, &Budget::standard(), &entries)
                    .unwrap_or_else(|e| panic!("{stem} {name}: {e}"));
                let compiled = dev
                    .compile_function(&f, &inputs, &Budget::standard())
                    .unwrap_or_else(|e| panic!("{stem} {name}: {e}"));
                assert_eq!(
                    answer,
                    compiled.observes_threadgroup_partition(),
                    "{stem} {name}"
                );
                assert_eq!(answer, observes, "{stem} {name}");
            }
        }
    }
}

/// On `Device::from_queries`' profile the module of a kernel's grid tiles runs a direct dispatch
/// at a size its runtime picks, at `MultiRegion` 0: over grids ragged along every axis, along
/// one, and whole, in workgroups of 16x16x1 and of the grid's tile, the region flagged
/// [`NONUNIFORM`], each of the grid's threads stores Metal's position once and no invocation past
/// the grid stores anything; unflagged, a whole grid does the same and a ragged one's invocations
/// past the grid store too.
#[test]
fn an_oblivious_kernel_runs_at_a_chosen_size() {
    use crate::lower::dispatch::grid_tile;
    let Some(bc) = assemble_text("edge", &edge_kernel()) else {
        return;
    };
    let mut inputs = LowerInputs::new();
    inputs.partial_threadgroups = true;
    for stem in PROFILES {
        let desc = crate::pipeline::testing::description(stem).unwrap_or_else(|e| panic!("{e}"));
        let device = Device::from_queries(desc, identity()).unwrap_or_else(|e| panic!("{e}"));
        let p = device.profile();
        let (plan, module) =
            typed_module_with(&bc, "edge", p, &inputs).unwrap_or_else(|e| panic!("{stem}: {e}"));
        for grid in [
            [7, 5, 3],
            [20, 40, 2],
            [300, 1, 1],
            [32, 16, 1],
            [24, 24, 8],
        ] {
            for tile in [[16, 16, 1], grid_tile(grid)] {
                let groups: [u32; 3] = std::array::from_fn(|i| grid[i].div_ceil(tile[i]));
                // Every word a launched invocation could store at, so a store past the grid shows.
                let len = ((groups[2] * tile[2] * grid[1] + groups[1] * tile[1]) * grid[0]
                    + groups[0] * tile[0]) as usize;
                let whole = (0..3).all(|i| grid[i].is_multiple_of(tile[i]));
                for flagged in [true, false] {
                    let region = match flagged {
                        true => nonuniform_region_bytes([0; 3], grid, tile),
                        false => region_bytes([0; 3], grid, tile),
                    }
                    .expect("packs");
                    let res = Resources {
                        buffers: vec![Buffer {
                            slot: None,
                            address: Some(OUT),
                            bytes: vec![0; 4 * len],
                        }],
                        push: grid_push(&plan, (OUT, 4 * len), region),
                        spec: local_size3(tile),
                    };
                    let cfg = Config {
                        workgroups: groups,
                        ..Config::default()
                    };
                    let got = run(&module, &cfg, res)
                        .unwrap_or_else(|e| panic!("{stem} {grid:?} {tile:?}: {e}"));
                    assert_eq!(
                        got.buffers[0] == edge_words(grid, len),
                        flagged || whole,
                        "{stem} {grid:?} {tile:?} flagged {flagged}"
                    );
                }
            }
        }
    }
}

/// Whether a kernel is image-only (`KernelFacts::image_only`), on the harness's kernels
/// (`fixtures/public/harness/workloads.ll`): one that samples a texture into another through a
/// runtime sampler (`sample_down`) and one that does through a constexpr sampler (`sample_up`)
/// are; one that reads a texture into a `device` buffer (`texels_out`) and one that writes a
/// texture from one (`texels_in`) are not.
#[test]
fn a_kernel_whose_memory_work_is_images_alone_is_image_only() {
    let Some(bc) = workloads() else {
        return;
    };
    let p = profile("lavapipe", identity()).unwrap_or_else(|e| panic!("{e}"));
    for (entry, want) in [
        ("sample_down", true),
        ("sample_up", true),
        ("texels_out", false),
        ("texels_in", false),
    ] {
        let a = compile(&bc, entry, &p).unwrap_or_else(|e| panic!("{entry}: {e}"));
        assert_eq!(a.kernel.image_only, want, "{entry}");
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
/// 80 holding 16 threads. The resident-table domains keep every barrier unconditional.
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
        let plain = profile(stem, DevicePolicy::RESIDENT_TABLES).unwrap_or_else(|e| panic!("{e}"));
        let a = compile_with(&bc, ENTRY, &plain, &inputs).unwrap_or_else(|e| panic!("{stem}: {e}"));
        let (_, words) = link_and_validate(&a, &plain).unwrap_or_else(|e| panic!("{stem}: {e}"));
        let (inside, outside) = barrier_guards(&spirv_dis(&words));
        assert!(
            inside == 0 && outside > 0,
            "{stem}: the resident-table domains guard {inside} barriers and leave {outside}"
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
                        push[o..o + 8].copy_from_slice(&slot_word(INPUT).to_le_bytes())
                    }
                    PushField::SlotGuestBase(s) if s == slot(1) => {
                        push[o..o + 8].copy_from_slice(&slot_word(OUT).to_le_bytes())
                    }
                    PushField::SlotBounds(s) if s == slot(0) => {
                        push[o..o + 8].copy_from_slice(&bounds_word(input.len(), 0))
                    }
                    PushField::SlotBounds(s) if s == slot(1) => {
                        push[o..o + 8].copy_from_slice(&bounds_word(16 * n as usize, 0))
                    }
                    PushField::ZeroAddress => {
                        push[o..o + 8].copy_from_slice(&ZERO_UNMAPPED.to_le_bytes())
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
                        push[o..o + 8]
                            .copy_from_slice(&slot_word(AT[s.index as usize]).to_le_bytes());
                    }
                    // `data`'s 16 bytes, `out`'s word and the count's.
                    PushField::SlotBounds(s) => {
                        let bytes = [data.len(), 4, 4][s.index as usize];
                        push[o..o + 8].copy_from_slice(&bounds_word(bytes, 0));
                    }
                    PushField::ZeroAddress => {
                        push[o..o + 8].copy_from_slice(&ZERO_UNMAPPED.to_le_bytes())
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

/// A storage slot's read without a format that no input names (`engine/rrp_swizzled_storage`'s
/// `rrp_sw_rwread_f`), under the identity domains on a device that reads storage so by the formats'
/// own bits alone (`DeviceProfile::storage_reads_by_format_bits`: the Iris Xe, which lacks
/// `shaderStorageImageReadWithoutFormat` and sets the read bit on every read-write tier format), is
/// declared without a format, as a heap storage class's read is: the pipeline requires the SPIR-V
/// capability and not the feature. With one tier format's bit cleared (`RGBA16Float`) the read is
/// still refused as `Unsupported{IdentityDomains}`. The resident-table domains' answer on the Iris
/// Xe, the format as the binder's input (`NeedsInput{StorageTextureFormat}`), does not change.
#[test]
fn a_storage_read_the_formats_bits_grant_is_formatless_under_the_identity_domains() {
    use crate::device::requirements::SpirvCapability;
    use crate::error::RuntimeInput;
    const IRIS: &str = "intel-irisxe-rplp-anv";
    const ENTRY: &str = "rrp_sw_rwread_f";
    let Some(bc) = assemble("rrp_swizzled_storage") else {
        return;
    };
    let compiled = |device: &Device| {
        let lib = Library::from_air(&bc).expect("library");
        let f = lib.function(ENTRY).expect("entry");
        let d = ComputePipelineDescriptor::new(StageSource::Function(FunctionDescriptor::new(f)));
        device.compile_compute_pipeline(&d, &Budget::standard())
    };
    let desc = crate::pipeline::testing::description(IRIS).unwrap_or_else(|e| panic!("{e}"));
    let iris = Device::from_queries(desc.clone(), identity()).unwrap_or_else(|e| panic!("{e}"));
    assert!(iris.profile().storage_reads_by_format_bits());
    let pipe = compiled(&iris).unwrap_or_else(|e| panic!("{IRIS}: {e}"));
    let capability = SpirvCapability::of(crate::spv::Capability::StorageImageReadWithoutFormat);
    assert!(pipe.requirements.spirv_capabilities.contains(&capability));
    assert!(!pipe
        .requirements
        .features
        .contains(&VkFeature::ShaderStorageImageReadWithoutFormat));

    let mut cleared = desc.clone();
    let props = (cleared.formats.formats)
        .get_mut(&vk::Format::R16G16B16A16_SFLOAT)
        .expect("RGBA16F");
    props.optimal_tiling = props
        .optimal_tiling
        .difference(vk::FormatFeatureFlags2::STORAGE_READ_WITHOUT_FORMAT);
    let cleared = Device::from_queries(cleared, identity()).unwrap_or_else(|e| panic!("{e}"));
    assert!(cleared.profile().storage_reads_by_format_bits());
    match compiled(&cleared) {
        Err(Error::Unsupported(u)) => assert_eq!(
            (u.feature, u.needs),
            (Feature::IdentityDomains, Needs::NotImplemented)
        ),
        other => panic!("cleared: {:?}", other.map(|_| ())),
    }

    let default =
        Device::from_queries(desc, DevicePolicy::RESIDENT_TABLES).unwrap_or_else(|e| panic!("{e}"));
    match compiled(&default) {
        Err(Error::NeedsInput(RuntimeInput::StorageTextureFormat { slot })) => {
            assert_eq!((slot.class, slot.index), (SlotClass::Texture, 0));
        }
        other => panic!("the resident-table domains: {:?}", other.map(|_| ())),
    }
}

/// The 32-bit floats `blob_words` reads of its `ushort3` constant as `half3`: 1.0, 2.5 and -0.75.
const BLOB_LANES: [f32; 3] = [1.0, 2.5, -0.75];

/// Under the identity domains the constant blob is the module's own words. `blob_words` reads a
/// `ushort3` constant as `half3` (the load `as_type` folds into, the records' form) and a word
/// table as 2-byte units at offsets only the run knows, one past the table's words: it binds no
/// descriptor and its artifact carries no blob, its private bytes count the blob's 32 padded bytes,
/// and on the CPU interpreter each thread stores its lane's float and its unit, and zero for the
/// read past the words. The resident-table domains compile the same kernel with the blob bound, its
/// 24 bytes in the artifact.
#[test]
fn the_constant_blob_is_the_modules_words_under_the_identity_domains() {
    use super::engine_pointer_memory::{assemble, kernel};
    let decls = "@k = internal addrspace(2) constant <3 x i16> <i16 15360, i16 16640, i16 -17920>, align 8
@w = internal addrspace(2) constant [4 x i32] [i32 131073, i32 262147, i32 393221, i32 524295], align 4
";
    let body = "  %h = load <3 x half>, ptr addrspace(2) @k, align 8
  %l = urem i32 %t, 3
  %e = extractelement <3 x half> %h, i32 %l
  %f = fpext half %e to float
  %fb = bitcast float %f to i32
  %m = and i32 %t, 7
  %off = shl i32 %m, 1
  %far = icmp uge i32 %t, 8
  %o = select i1 %far, i32 4096, i32 %off
  %p = getelementptr i8, ptr addrspace(2) @w, i32 %o
  %v = load i16, ptr addrspace(2) %p, align 2
  %vz = zext i16 %v to i32
  %i = shl i32 %t, 1
  %q0 = getelementptr inbounds i32, ptr addrspace(1) %b0, i32 %i
  store i32 %fb, ptr addrspace(1) %q0, align 4
  %j = or i32 %i, 1
  %q1 = getelementptr inbounds i32, ptr addrspace(1) %b0, i32 %j
  store i32 %vz, ptr addrspace(1) %q1, align 4
";
    let name = "blob_words";
    let Some(bc) = assemble(name, &kernel(name, 1, body, decls)) else {
        return;
    };
    const THREADS: u32 = 9;
    let address = 0x3_0000_2000u64;
    let out = vec![0xEEu8; 8 * THREADS as usize];
    let mut want = Vec::new();
    for t in 0..THREADS {
        want.extend_from_slice(&BLOB_LANES[(t % 3) as usize].to_bits().to_le_bytes());
        let unit = if t < 8 { (t & 7) + 1 } else { 0 };
        want.extend_from_slice(&unit.to_le_bytes());
    }
    for stem in PROFILES {
        let default =
            profile(stem, DevicePolicy::RESIDENT_TABLES).unwrap_or_else(|e| panic!("{e}"));
        let (bound, bound_plan, _) =
            compile_kept(&bc, name, &default).unwrap_or_else(|e| panic!("{stem}: default: {e}"));
        assert_eq!(bound.blob.as_deref().map(<[u8]>::len), Some(24), "{stem}");
        let p = profile(stem, identity()).unwrap_or_else(|e| panic!("{e}"));
        let (a, plan, module) =
            compile_kept(&bc, name, &p).unwrap_or_else(|e| panic!("{stem}: {e}"));
        assert!(a.interface.descriptors.is_empty(), "{stem}");
        assert_eq!(a.blob, None, "{stem}");
        assert_eq!(plan.private_bytes, bound_plan.private_bytes + 32, "{stem}");
        link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{stem}: L3: {e}"));
        let mut push = vec![0u8; plan.push.push_range.map_or(0, |(_, end)| end as usize)];
        let region = region_bytes([0; 3], [THREADS, 1, 1], [THREADS, 1, 1]).expect("packs");
        for f in &plan.push.fields {
            let PushLoc::Push { offset } = f.loc else {
                panic!("{stem}: {f:?} spilled");
            };
            let o = offset as usize;
            match f.field {
                PushField::SlotGuestBase(_) => {
                    push[o..o + 8].copy_from_slice(&slot_word(address).to_le_bytes());
                }
                PushField::SlotBounds(_) => {
                    push[o..o + 8].copy_from_slice(&bounds_word(out.len(), 0));
                }
                PushField::ZeroAddress => {
                    push[o..o + 8].copy_from_slice(&ZERO_UNMAPPED.to_le_bytes())
                }
                PushField::DispatchRegion => push[o..o + 32].copy_from_slice(&region),
                _ => panic!("{stem}: unexpected push field {f:?}"),
            }
        }
        let res = Resources {
            buffers: vec![Buffer {
                slot: None,
                address: Some(address),
                bytes: out.clone(),
            }],
            push,
            spec: local_size(THREADS),
        };
        let got = run(&module, &Config::default(), res).unwrap_or_else(|e| panic!("{stem}: {e}"));
        assert_eq!(got.buffers[0], want, "{stem}: out");
    }
}

/// Under the identity domains a constant blob past `memory::blob::MODULE_BYTES` is refused as
/// `Unsupported{IdentityDomains}`, not implemented, naming the constant data's size, where the
/// resident-table domains bind it.
#[test]
fn a_constant_blob_past_the_modules_bound_is_refused_naming_its_size() {
    use super::engine_pointer_memory::{assemble, kernel};
    let decls = "@big = internal addrspace(2) constant [16400 x i32] zeroinitializer, align 4\n";
    let body = "  %p = getelementptr i8, ptr addrspace(2) @big, i32 6
  %v = load i16, ptr addrspace(2) %p, align 2
  %vz = zext i16 %v to i32
  store i32 %vz, ptr addrspace(1) %b0, align 4
";
    let name = "blob_past";
    let Some(bc) = assemble(name, &kernel(name, 1, body, decls)) else {
        return;
    };
    let bytes = 16400 * 4;
    assert!(bytes > crate::memory::blob::MODULE_BYTES);
    for stem in PROFILES {
        let default =
            profile(stem, DevicePolicy::RESIDENT_TABLES).unwrap_or_else(|e| panic!("{e}"));
        let (bound, _, _) =
            compile_kept(&bc, name, &default).unwrap_or_else(|e| panic!("{stem}: default: {e}"));
        assert_eq!(
            bound.blob.as_deref().map(|b| b.len() as u64),
            Some(bytes),
            "{stem}"
        );
        let p = profile(stem, identity()).unwrap_or_else(|e| panic!("{e}"));
        let e = match compile(&bc, name, &p) {
            Err(e) => e,
            Ok(_) => panic!("{stem}: compiled"),
        };
        let text = e.to_string();
        let words = format!("constant data of {bytes} bytes read through memory");
        assert!(text.contains(&words), "{stem}: {text}");
        let Error::Unsupported(u) = e else {
            panic!("{stem}: {text}");
        };
        assert_eq!(
            (u.feature, u.needs),
            (Feature::IdentityDomains, Needs::NotImplemented),
            "{stem}"
        );
    }
}

/// A kernel whose device-scope `seq_cst` fence orders its two stores (`a4_device_fence`) compiles
/// under the resident-table domains, its fence ordered through the command buffer's
/// `ExecutionStatus` word, and is refused under the identity domains, which reach no such word,
/// naming the fence ([`SEQ_CST_FORM`](crate::lower::sync::barrier::SEQ_CST_FORM)).
#[test]
fn a_seq_cst_fence_is_refused_naming_its_form_under_the_identity_domains() {
    let Some(bc) = assemble("a4_device_fence") else {
        return;
    };
    for stem in PROFILES {
        let default =
            profile(stem, DevicePolicy::RESIDENT_TABLES).unwrap_or_else(|e| panic!("{e}"));
        let a = compile(&bc, "a4_device_fence", &default)
            .unwrap_or_else(|e| panic!("{stem}: resident-table domains: {e}"));
        assert!(a.facts.contains(&RuntimeFact::ExecutionStatus), "{stem}");
        let p = profile(stem, identity()).unwrap_or_else(|e| panic!("{e}"));
        match compile(&bc, "a4_device_fence", &p) {
            Err(Error::Unsupported(u)) => {
                assert_eq!(
                    (u.feature, u.needs),
                    (Feature::IdentityDomains, Needs::NotImplemented),
                    "{stem}"
                );
                assert_eq!(
                    u.symbol.map(|f| f.to_string()).as_deref(),
                    Some(crate::lower::sync::barrier::SEQ_CST_FORM),
                    "{stem}"
                );
            }
            other => panic!("{stem}: {:?}", other.map(|_| ())),
        }
    }
}

/// The golden table's storage pair under the identity domains (owned fixture
/// `storage_texture_write`, a write through `texture2d<float, write>`, given each format's storage
/// choice as its input). A choice the identity arrays hold an element of, the bound format's own
/// image declared without a format, compiles. Any other is refused naming the slot's form ("texture
/// argument ... beyond its resource ID"):
/// - bgra8 where it is not storage-capable, whose choice swizzles red and blue
///   (`StorageImageClass::SwizzleBgra`), state of the bound texture beyond its resource ID;
/// - rgba8 where the device reads storage images of it only through its format, whose choice
///   declares that format, a storage image the identity arrays hold no element of.
///
/// The runtime gives no such choice: no input names the bound format, and the slot stays the
/// write the floor's `shaderStorageImageWriteWithoutFormat` takes.
#[test]
fn a_storage_choice_the_identity_arrays_hold_no_element_of_is_refused_naming_the_texture_argument()
{
    use crate::pipeline::StorageImageClass;
    let Some(bc) = assemble("storage_texture_write") else {
        return;
    };
    let (mut compiled, mut swizzled, mut declared) = (0, 0, 0);
    let profiles = crate::spirv::testing::committed_profiles_with(identity())
        .unwrap_or_else(|e| panic!("{e}"));
    for (stem, p) in &profiles {
        for format in [PixelFormat::BGRA8Unorm, PixelFormat::RGBA8Unorm] {
            let Ok(choice) = crate::metal::format::storage_image_choice(format, p) else {
                continue;
            };
            let held = choice.class == StorageImageClass::Identity
                && choice.vk_format == vk::Format::UNDEFINED;
            match (held, choice.class) {
                (true, _) => compiled += 1,
                (false, StorageImageClass::SwizzleBgra) => swizzled += 1,
                (false, StorageImageClass::Identity) => declared += 1,
                (false, class) => panic!("{stem}: {format:?}'s choice {class:?}"),
            }
            let mut inputs = LowerInputs::new();
            inputs.storage_images.insert(
                SlotRef {
                    stage: Stage::Kernel,
                    class: SlotClass::Texture,
                    index: 0,
                },
                choice,
            );
            match (held, compile_with(&bc, "storage_texture_write", p, &inputs)) {
                (true, Ok(_)) => {}
                (false, Err(Error::Unsupported(u))) => {
                    assert_eq!(
                        (u.feature, u.needs),
                        (Feature::IdentityDomains, Needs::NotImplemented),
                        "{stem} {format:?}"
                    );
                    let symbol = u.symbol.map(|f| f.to_string()).unwrap_or_default();
                    assert!(
                        symbol.starts_with("texture argument")
                            && symbol.contains("beyond its resource ID"),
                        "{stem} {format:?}: {symbol}"
                    );
                }
                (_, other) => panic!("{stem} {format:?}: {:?}", other.map(|_| ())),
            }
        }
    }
    assert!(
        compiled > 0 && swizzled > 0 && declared > 0,
        "compiled {compiled}, swizzled {swizzled}, declared {declared}: a case of each"
    );
}
