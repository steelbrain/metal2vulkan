//! A key's general form (`pipeline/render/residual.rs`): each constant with an arm that serves
//! every binding at that arm, the others as the key has them.

use crate::pipeline::general_form;
use metal2vulkan::cache::SpecializationKey;
use metal2vulkan::pipeline::{SlotClass, SlotRef, SpecKey, Stage};
use metal2vulkan::vk::SpecializationMapEntry;

/// Fragment slot `index` of `class`.
fn slot(class: SlotClass, index: u32) -> SlotRef {
    SlotRef {
        stage: Stage::Fragment,
        class,
        index,
    }
}

/// The constants of a fragment function that reads a uniform view at buffer 0, converts texture
/// 0 to `half` (both its booleans), and samples it through sampler 0 and a bicubic filter: IDs 0
/// to 4.
fn constants() -> Vec<(u32, SpecKey)> {
    let texture = slot(SlotClass::Texture, 0);
    vec![
        (0, SpecKey::RangeAtBindOffset(slot(SlotClass::Buffer, 0))),
        (1, SpecKey::PlainHalfTexels(texture)),
        (2, SpecKey::UnmeasuredTexels(texture)),
        (
            3,
            SpecKey::SamplerState {
                slot: slot(SlotClass::Sampler, 0),
                fields: u32::MAX,
            },
        ),
        (4, SpecKey::BicubicTexels(texture)),
    ]
}

/// A key giving IDs 0 to 4 `values`, each four bytes, packed in order.
fn key(values: [u32; 5]) -> SpecializationKey {
    let mut key = SpecializationKey::new();
    for (id, value) in (0..).zip(values) {
        key.entries.push(SpecializationMapEntry {
            constant_id: id,
            offset: key.data.len() as u32,
            size: 4,
        });
        key.data.extend_from_slice(&value.to_le_bytes());
    }
    key
}

#[test]
fn the_general_form_zeroes_the_constants_whose_zero_serves_every_binding() {
    let general = general_form(constants().into_iter(), &key([1, 1, 1, 5, 3]));
    // The sampler's word and the bicubic class keep their values: no arm of theirs serves every
    // binding.
    assert_eq!(general, Some(key([0, 0, 0, 5, 3])));
    // A plain-class or unbound texture's key and a measured class's alike.
    let general = general_form(constants().into_iter(), &key([0, 1, 0, 5, 3]));
    assert_eq!(general, Some(key([0, 0, 0, 5, 3])));
}

#[test]
fn a_key_in_its_general_form_has_none_of_its_own() {
    assert_eq!(
        general_form(constants().into_iter(), &key([0, 0, 0, 5, 3])),
        None
    );
    // A constant the key's stages do not declare is not the general form's to set.
    assert_eq!(
        general_form(std::iter::empty(), &key([1, 1, 1, 5, 3])),
        None
    );
}
