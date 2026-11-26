//! A kernel's buffer arguments as storage buffers of set 0 under the identity domains, on every
//! committed profile `Device::from_queries` derives there that answers the bound form's query
//! (`DeviceProfile::binds_kernel_buffers`), with the vector word-phase fixture
//! (`engine/vector_word_phases`: `vwp_words` over two `uint` buffers, `vwp_vec16` over two
//! `float4` ones).
//!
//! - Each kernel's two slots are storage views of set 0, one storage buffer each, reached at their
//!   pushed bind offsets (`PushField::SlotBindOffset`): no access goes through a device address,
//!   no slot pushes its address or its bounds (`PushField::SlotGuestBase`,
//!   `PushField::SlotBounds`), and no load reads the zero range (`PushField::ZeroAddress`). The
//!   module states `RuntimeFact::RobustStorageBuffers` and each slot's alignment
//!   (`RuntimeFact::AddressAlignment`: 4 for a `uint` buffer, 16 for a `float4` one), requires the
//!   pipeline-robustness feature (`VkFeature::PipelineRobustness`), and passes L3.
//! - A slot the dispatch binds at a multiple of 4 below its argument's alignment
//!   (`LoweringInputs::word_bound_buffers`) keeps its device address, held to 4: a view's `uvec4`
//!   units take the bind offset to be a multiple of the argument's alignment.
//! - lavapipe, whose `maxStorageBufferRange` is below 2 GiB, answers the query otherwise.
//!
//! `llvm-as` assembles the fixture; without it the tests skip visibly.

use std::collections::BTreeSet;

use super::engine_word_phases::{bound_identity_profiles, device_alignments};
use crate::device::requirements::VkFeature;
use crate::lower::select::LowerInputs;
use crate::pipeline::facts::RuntimeFact;
use crate::pipeline::sym::{PushField, ResourceKind, SlotClass, SlotRef, Stage};
use crate::pipeline::testing::{
    compile_with, engine_dir, link_and_validate, llvm_assemble, AssembleError,
};
use crate::vk;
use metal2vulkan_devtools::vector_word_phases::{FIXTURE, VEC16, WORDS};

fn bitcode() -> Option<Vec<u8>> {
    match llvm_assemble(&engine_dir().join(format!("{FIXTURE}.ll"))) {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(e) => panic!("{FIXTURE}: {e:?}"),
    }
}

fn slot(index: u32) -> SlotRef {
    SlotRef {
        stage: Stage::Kernel,
        class: SlotClass::Buffer,
        index,
    }
}

/// The committed identity profiles that bind a kernel's buffers, by stem; lavapipe's is not one.
fn bound_profiles() -> Vec<(String, crate::device::profile::DeviceProfile)> {
    let profiles = bound_identity_profiles();
    let lavapipe = profiles.iter().find(|(stem, _)| stem == "lavapipe");
    assert!(
        lavapipe.is_some_and(|(_, p)| !p.binds_kernel_buffers()),
        "lavapipe answers the query otherwise"
    );
    let bound: Vec<_> = (profiles.into_iter())
        .filter(|(_, p)| p.binds_kernel_buffers())
        .collect();
    for stem in ["amd-780m-radv", "intel-irisxe-anv", "nvidia-rtx50-linux"] {
        assert!(
            bound.iter().any(|(s, _)| s == stem),
            "{stem} binds a kernel's buffers"
        );
    }
    bound
}

#[test]
fn a_kernel_reaches_its_buffers_through_set_0_views_at_their_bind_offsets() {
    let Some(bc) = bitcode() else {
        return;
    };
    for (stem, p) in &bound_profiles() {
        for (name, align) in [(WORDS, 4), (VEC16, 16)] {
            let what = format!("{name} on {stem}");
            let a = compile_with(&bc, name, p, &LowerInputs::new())
                .unwrap_or_else(|e| panic!("{what}: {e}"));
            link_and_validate(&a, p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
            assert!(
                device_alignments(&a.module.words).is_empty(),
                "{what}: a device-address access"
            );
            let views: BTreeSet<u32> = (a.interface.descriptors.iter())
                .map(|d| {
                    assert_eq!(d.ty, vk::DescriptorType::STORAGE_BUFFER, "{what}: {d:?}");
                    assert_eq!(d.count, 1, "{what}: {d:?}");
                    assert!(matches!(d.slot.kind, ResourceKind::Buffer), "{what}: {d:?}");
                    d.slot.index
                })
                .collect();
            assert_eq!(views, BTreeSet::from([0, 1]), "{what}: the views");
            let fields: Vec<PushField> =
                (a.interface.push.fields.iter()).map(|f| f.field).collect();
            for index in [0, 1] {
                assert!(
                    fields.contains(&PushField::SlotBindOffset(slot(index))),
                    "{what}: {fields:?}"
                );
            }
            assert!(
                !fields.iter().any(|f| matches!(
                    f,
                    PushField::SlotGuestBase(_) | PushField::SlotBounds(_) | PushField::ZeroAddress
                )),
                "{what}: {fields:?}"
            );
            assert!(
                a.facts.contains(&RuntimeFact::RobustStorageBuffers),
                "{what}: {:?}",
                a.facts
            );
            assert!(
                (a.requirements.features).contains(&VkFeature::PipelineRobustness),
                "{what}: {:?}",
                a.requirements.features
            );
            let stated: Vec<u32> = (a.facts.iter())
                .filter_map(|f| match f {
                    RuntimeFact::AddressAlignment { align, .. } => Some(*align),
                    _ => None,
                })
                .collect();
            assert_eq!(stated, [align; 2], "{what}: stated alignments");
        }
    }
}

#[test]
fn a_word_bound_slot_keeps_its_device_address() {
    let Some(bc) = bitcode() else {
        return;
    };
    let mut inputs = LowerInputs::new();
    inputs.word_bound_buffers = [slot(0), slot(1)].into();
    for (stem, p) in &bound_profiles() {
        let what = format!("{VEC16} word-bound on {stem}");
        let a = compile_with(&bc, VEC16, p, &inputs).unwrap_or_else(|e| panic!("{what}: {e}"));
        link_and_validate(&a, p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
        let claimed = device_alignments(&a.module.words);
        assert!(!claimed.is_empty(), "{what}: no device-address access");
        assert!(
            claimed.iter().all(|&c| c == 4),
            "{what}: claimed {claimed:?}"
        );
        assert!(a.interface.descriptors.is_empty(), "{what}: a descriptor");
        assert!(
            !a.facts.contains(&RuntimeFact::RobustStorageBuffers),
            "{what}: {:?}",
            a.facts
        );
        assert!(
            !(a.requirements.features).contains(&VkFeature::PipelineRobustness),
            "{what}: {:?}",
            a.requirements.features
        );
    }
}
