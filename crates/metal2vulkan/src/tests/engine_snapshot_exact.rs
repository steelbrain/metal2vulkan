//! A texture read through a snapshot at a slot whose reads may take any bound format.
//!
//! The copy that fills a snapshot (`lower::fragment`'s `snapshot`) fetches through one view, and
//! the read the snapshot stands for must be the fetch through that same view: the slot's own image,
//! whose bound view the binder names the snapshot's form by.
//!
//! - Every form the binder can name for a `texture2d<float, access::read>` slot — the bound
//!   view's codes, or the fetched lanes' bits — translates and passes L3 under the resident-table
//!   domains. An 8-bit sRGB texture, whose view decodes its codes, keeps the
//!   fetched lanes' bits.
//! - The copy pairs the form with its fetch's lanes where the invariant lives: a form whose codes
//!   those lanes do not hold is named, not left to the builder to refuse (a float code form over
//!   integer lanes walks into `FClamp`, and an integer form over float lanes bitcasts the value).
//!   Every one of the ten forms, over float lanes and over unsigned integer lanes.
//!
//! `llvm-as` assembles the fixtures; a missing tool skips a test visibly.

use std::sync::atomic::{AtomicU64, Ordering};

use crate::bind::{BindState, BoundTexture, TextureFacts};
use crate::budget::Budget;
use crate::device::{Device, DevicePolicy};
use crate::library::Library;
use crate::metal::{PixelFormat, TextureType, TextureUsage};
use crate::pipeline::descriptor::{
    ColorAttachmentDescriptor, FunctionDescriptor, RenderPipelineDescriptor, StageSource,
};
use crate::pipeline::layout::BindingSource;
use crate::pipeline::sym::{SlotClass, SlotRef, SnapshotForm, Stage};
use crate::pipeline::testing::{engine_dir, llvm_assemble, AssembleError};
use crate::spirv::testing::check_words_valid;

/// The snapshot slot under test.
const SLOT: SlotRef = SlotRef {
    stage: Stage::Fragment,
    class: SlotClass::Texture,
    index: 0,
};

/// Every form a snapshot can keep its texels in, in declaration order.
const FORMS: [SnapshotForm; 10] = [
    SnapshotForm::Bits,
    SnapshotForm::Unorm8,
    SnapshotForm::Snorm8,
    SnapshotForm::Unorm16,
    SnapshotForm::Snorm16,
    SnapshotForm::Half,
    SnapshotForm::Uint8,
    SnapshotForm::Sint8,
    SnapshotForm::Uint16,
    SnapshotForm::Sint16,
];

/// Whether `form`'s codes hold every lane an image whose sampled type is `f32` (`float_lanes`)
/// or `u32` returns: a float lane's every form but an integer one's codes, an unsigned integer
/// lane's the bits and the unsigned integer codes. The bits hold any lanes.
fn holds(form: SnapshotForm, float_lanes: bool) -> bool {
    match float_lanes {
        true => !matches!(
            form,
            SnapshotForm::Uint8 | SnapshotForm::Uint16 | SnapshotForm::Sint8 | SnapshotForm::Sint16
        ),
        false => matches!(
            form,
            SnapshotForm::Bits | SnapshotForm::Uint8 | SnapshotForm::Uint16
        ),
    }
}

/// The profiles whose draws can take a snapshot (the pixel interlock, fragment stores, the
/// feedback loop layout and dynamic color write masks); `vk12-floor` has none, and is left out.
const PROFILES: [&str; 4] = [
    "intel-irisxe-anv",
    "amd-780m-radv",
    "nvidia-rtx5080-laptop",
    "lavapipe",
];

const HEADER: &str = "target datalayout = \"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32\"\ntarget triple = \"air64_v28-apple-macosx26.0.0\"\n\n";

/// A fragment function returning the `uint4` texel of a `texture2d<uint, access::read>` slot under
/// its own pixel, the read an integer-lane snapshot slot stands for.
const UINT_READ: &str = r#"define <4 x i32> @sx_read_u(<4 x float> %0, ptr addrspace(1) %1) {
  %3 = shufflevector <4 x float> %0, <4 x float> poison, <2 x i32> <i32 0, i32 1>
  %4 = call <2 x i32> @air.convert.u.v2i32.f.v2f32(<2 x float> %3)
  %5 = call ptr addrspace(2) @air.get_read_sampler()
  %6 = call { <4 x i32>, i8 } @air.read_texture_2d.u.v4i32(ptr addrspace(1) %1, ptr addrspace(2) %5, <2 x i32> %4, <2 x i32> zeroinitializer, i32 0, i32 1)
  %7 = extractvalue { <4 x i32>, i8 } %6, 0
  ret <4 x i32> %7
}

declare <2 x i32> @air.convert.u.v2i32.f.v2f32(<2 x float>)
declare ptr addrspace(2) @air.get_read_sampler()
declare { <4 x i32>, i8 } @air.read_texture_2d.u.v4i32(ptr addrspace(1), ptr addrspace(2), <2 x i32>, <2 x i32>, i32, i32)

!air.fragment = !{!0}
!0 = !{ptr @sx_read_u, !1, !3}
!1 = !{!2}
!2 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"uint4"}
!3 = !{!4, !5}
!4 = !{i32 0, !"air.position", !"air.center", !"air.no_perspective", !"air.arg_type_name", !"float4", !"air.arg_name", !"position"}
!5 = !{i32 1, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.read", !"air.arg_type_name", !"texture2d<uint, read>", !"air.arg_name", !"t"}
"#;

/// Bitcode of the owned fixture `name`, or `None` (skipped visibly) without `llvm-as`.
fn fixture(name: &str) -> Option<Vec<u8>> {
    match llvm_assemble(&engine_dir().join(format!("{name}.ll"))) {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected {name}: {why}"),
    }
}

/// Bitcode of the module `text`, or `None` (skipped visibly) without `llvm-as`.
fn assemble(text: &str) -> Option<Vec<u8>> {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let dir = std::env::temp_dir().join(format!(
        "m2v-snapshot-exact-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&dir).expect("scratch");
    let ll = dir.join("f.ll");
    std::fs::write(&ll, format!("{HEADER}{text}")).expect("write");
    let r = llvm_assemble(&ll);
    let _ = std::fs::remove_dir_all(&dir);
    match r {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected the fragment: {why}"),
    }
}

/// Function `name` of `bc` as a stage source.
fn function(bc: &[u8], name: &str) -> StageSource {
    StageSource::Function(FunctionDescriptor::new(
        Library::from_air(bc)
            .expect("library")
            .function(name)
            .expect("function"),
    ))
}

/// A draw state that asks for a snapshot of texture slot 0 with `pixel_format` bound there.
fn state(pixel_format: Option<PixelFormat>) -> BindState {
    let mut s = BindState {
        snapshot_textures: [0].into(),
        ..Default::default()
    };
    if let Some(pf) = pixel_format {
        let mut facts = TextureFacts::new(TextureType::Type2D, pf);
        facts.usage = TextureUsage::SHADER_READ | TextureUsage::RENDER_TARGET;
        s.stages
            .entry(Stage::Fragment)
            .or_default()
            .textures
            .insert(
                0,
                BoundTexture::Texture {
                    facts,
                    resource_id: None,
                },
            );
    }
    s
}

/// A `texture2d<float, access::read>` slot the draw reads through a snapshot translates under
/// every form the binder names for it under the resident-table domains. An 8-bit sRGB texture,
/// whose view decodes its codes, keeps the fetched bits. Each variant passes
/// L3 and binds the snapshot as a storage image.
#[test]
fn a_snapshot_slot_translates_under_every_form() {
    let Some(bc) = fixture("tob_renders") else {
        return;
    };
    let mut d = RenderPipelineDescriptor::new(function(&bc, "tob_rect"));
    d.fragment = Some(function(&bc, "tob_read"));
    d.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
    // Each form `SnapshotForm::of` names for a float slot, by a texture the binder names it for.
    let bound = [
        (PixelFormat::RGBA8Unorm, SnapshotForm::Unorm8),
        (PixelFormat::RGBA8Snorm, SnapshotForm::Snorm8),
        (PixelFormat::RGBA16Unorm, SnapshotForm::Unorm16),
        (PixelFormat::RGBA16Snorm, SnapshotForm::Snorm16),
        (PixelFormat::RGBA16Float, SnapshotForm::Half),
        (PixelFormat::R16Float, SnapshotForm::Half),
        (PixelFormat::RGBA32Float, SnapshotForm::Bits),
        (PixelFormat::RGBA8UnormSRGB, SnapshotForm::Bits),
    ];
    for stem in PROFILES {
        {
            let (label, policy) = ("the default", DevicePolicy::RESIDENT_TABLES);
            let desc = crate::pipeline::testing::description(stem).expect("description");
            let dev = Device::from_queries(desc, policy).expect("device");
            let base = dev
                .compile_render_pipeline(&d, &Budget::standard())
                .unwrap_or_else(|e| panic!("{stem}/{label}: {e}"));
            for (pf, form) in bound {
                let plan = base
                    .plan(&dev, &state(Some(pf)).shape())
                    .unwrap_or_else(|e| panic!("{stem}/{label}/{pf:?}: plan: {e}"));
                assert_eq!(
                    plan.variant.snapshot_textures,
                    [(SLOT, form)].into(),
                    "{stem}/{label}/{pf:?}"
                );
                let variant = dev
                    .compile_render_variant(&d, &plan.variant, &Budget::standard())
                    .unwrap_or_else(|e| panic!("{stem}/{label}/{pf:?}: {e}"));
                assert!(
                    variant
                        .layout
                        .sets
                        .iter()
                        .flat_map(|s| &s.bindings)
                        .any(|b| b.source == BindingSource::SnapshotTexels(0)),
                    "{stem}/{label}/{pf:?}: no snapshot binding"
                );
                for m in &variant.stages {
                    check_words_valid(&m.spirv, &variant.requirements, dev.profile())
                        .unwrap_or_else(|e| panic!("{stem}/{label}/{pf:?}: L3: {e}"));
                }
            }
        }
    }
}

/// The copy of a snapshot slot names a form whose codes its fetch's lanes do not hold, where that
/// invariant lives, instead of leaving the builder to refuse the encode: over float lanes every
/// integer form, and over unsigned integer lanes every float code form (the pairing that otherwise
/// reaches `FClamp`). The binder never names such a pairing (`SnapshotForm::of`), so the key is
/// forged. On ANV's Iris Xe profile.
#[test]
fn the_copy_of_a_snapshot_slot_refuses_a_form_its_fetch_s_lanes_do_not_hold() {
    let (Some(bc), Some(uint)) = (fixture("tob_renders"), assemble(UINT_READ)) else {
        return;
    };
    let stem = "intel-irisxe-anv";
    let desc = crate::pipeline::testing::description(stem).expect("description");
    let dev = Device::from_queries(desc, DevicePolicy::RESIDENT_TABLES).expect("device");
    // The float slot's fragment, and the unsigned integer slot's into an integer attachment.
    let legs = [
        ("uint", "sx_read_u", &uint, PixelFormat::RGBA8Uint, false),
        ("float", "tob_read", &bc, PixelFormat::RGBA8Unorm, true),
    ];
    for (leg, entry, module, attachment, float_lanes) in legs {
        let mut d = RenderPipelineDescriptor::new(function(&bc, "tob_rect"));
        d.fragment = Some(function(module, entry));
        d.color_attachments[0] = Some(ColorAttachmentDescriptor::new(attachment));
        let base = dev
            .compile_render_pipeline(&d, &Budget::standard())
            .unwrap_or_else(|e| panic!("{leg}: {e}"));
        let plan = base
            .plan(&dev, &state(None).shape())
            .unwrap_or_else(|e| panic!("{leg}: plan: {e}"));
        assert!(
            !plan.variant.snapshot_textures.is_empty(),
            "{leg}: no snapshot variant"
        );
        for form in FORMS {
            let mut key = plan.variant.clone();
            key.snapshot_textures = [(SLOT, form)].into();
            let r = dev.compile_render_variant(&d, &key, &Budget::standard());
            if holds(form, float_lanes) {
                r.unwrap_or_else(|e| panic!("{leg}/{form:?}: {e}"));
                continue;
            }
            let e = r.err().map(|e| e.to_string()).unwrap_or_default();
            assert!(
                e.contains(&format!(
                    "the copy of snapshot slot 0 fetches lanes its form {form:?} does not hold"
                )),
                "{leg}/{form:?}: {e}"
            );
        }
    }
}
