//! A key's general form (`pipeline/render/residual.rs`): each constant with an arm that serves
//! every binding at that arm, the others as the key has them. A texture slot's plain `half` arm
//! keeps its value: it serves the plain classes alone, and a creation predicts it. And the key a
//! plan takes where the device reads its uniform views at their residuals: those views' constants
//! alone at 0.

use crate::pipeline::{general_form, residual_form_of, residual_ids, residual_in_place};
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
    // The plain arm, the sampler's word and the bicubic class keep their values: no arm of theirs
    // serves every binding the key's does not.
    assert_eq!(general, Some(key([0, 1, 0, 5, 3])));
    // An unmeasured plain class's key: its general form is the key of an unbound or measured
    // plain class, the one a creation predicts.
    let general = general_form(constants().into_iter(), &key([0, 1, 1, 5, 3]));
    assert_eq!(general, Some(key([0, 1, 0, 5, 3])));
    // A class the plain arm does not serve keeps the routine's arm.
    let general = general_form(constants().into_iter(), &key([1, 0, 1, 5, 3]));
    assert_eq!(general, Some(key([0, 0, 0, 5, 3])));
}

#[test]
fn a_key_in_its_general_form_has_none_of_its_own() {
    // An unbound or measured plain class's key, which a creation predicts, and one of the
    // routine's arm.
    for values in [[0, 1, 0, 5, 3], [0, 0, 0, 5, 3]] {
        assert_eq!(general_form(constants().into_iter(), &key(values)), None);
    }
    // A constant the key's stages do not declare is not the general form's to set.
    assert_eq!(
        general_form(std::iter::empty(), &key([1, 1, 1, 5, 3])),
        None
    );
}

/// Where the device reads its uniform views at their residuals, a plan's key takes the
/// `RangeAtBindOffset` constant of each slot read through a uniform view at 0
/// (`uniform_residual_form`, here as the slots it names), and keeps a storage view's, which a
/// profile taking `DriverWorkaround::PushReadsLengthenThreadPayload` declares and sets at bind
/// offset 0; the residual form, which serves every binding, takes both at 0. Constants of no slot
/// keep their values in either.
#[test]
fn a_plans_key_zeroes_its_uniform_views_and_keeps_its_storage_views() {
    let uniform = slot(SlotClass::Buffer, 0);
    let texture = slot(SlotClass::Texture, 0);
    let constants = || {
        vec![
            (0, SpecKey::RangeAtBindOffset(uniform)),
            (1, SpecKey::RangeAtBindOffset(slot(SlotClass::Buffer, 1))),
            (2, SpecKey::UnmeasuredTexels(texture)),
            (3, SpecKey::PlainHalfTexels(texture)),
            (4, SpecKey::BicubicTexels(texture)),
        ]
        .into_iter()
    };
    let uniform_views = |at: SlotRef| at == uniform;
    let at_bind = key([1, 1, 1, 1, 3]);
    assert_eq!(
        residual_form_of(constants(), uniform_views, &at_bind),
        Some(key([0, 1, 1, 1, 3]))
    );
    assert_eq!(
        residual_form_of(constants(), |_| true, &at_bind),
        Some(key([0, 0, 1, 1, 3]))
    );
    // A key whose uniform view is read at its residual already is the plan's own.
    let kept = key([0, 1, 1, 1, 3]);
    assert_eq!(residual_form_of(constants(), uniform_views, &kept), None);
}

/// A dispatch's regions take that key in their own bytes (`Variant::keyed_regions`): zeroing the
/// constants `residual_ids` lists in place gives `residual_form_of`'s key, or the key as it was
/// where that answers `None`, in the storage the key held, for the uniform views alone and for
/// every view.
#[test]
fn a_key_takes_its_residual_form_in_its_own_bytes() {
    let uniform = slot(SlotClass::Buffer, 0);
    let texture = slot(SlotClass::Texture, 0);
    let constants = || {
        vec![
            (0, SpecKey::RangeAtBindOffset(uniform)),
            (1, SpecKey::RangeAtBindOffset(slot(SlotClass::Buffer, 1))),
            (2, SpecKey::UnmeasuredTexels(texture)),
            (3, SpecKey::PlainHalfTexels(texture)),
            (4, SpecKey::BicubicTexels(texture)),
        ]
        .into_iter()
    };
    let uniform_views = |at: SlotRef| at == uniform;
    let every_view = |_: SlotRef| true;
    let views: [&dyn Fn(SlotRef) -> bool; 2] = [&uniform_views, &every_view];
    assert_eq!(residual_ids(constants(), uniform_views), vec![0]);
    assert_eq!(residual_ids(constants(), every_view), vec![0, 1]);
    for (n, views) in views.into_iter().enumerate() {
        for values in [[1, 1, 1, 1, 3], [0, 1, 1, 1, 3], [0, 0, 0, 0, 0]] {
            let was = key(values);
            let made = residual_form_of(constants(), views, &was).unwrap_or_else(|| was.clone());
            let mut in_place = was.clone();
            let storage = (in_place.data.as_ptr(), in_place.entries.as_ptr());
            residual_in_place(&mut in_place, &residual_ids(constants(), views));
            assert_eq!(in_place, made, "views {n}, {values:?}");
            let kept = (in_place.data.as_ptr(), in_place.entries.as_ptr());
            assert_eq!(kept, storage, "views {n}, {values:?}: the key's storage");
        }
    }
}
