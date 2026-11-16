//! A key's general form (`pipeline/render/residual.rs`): each constant with an arm that serves
//! every binding at that arm, the others as the key has them. A texture slot's plain `half` arm
//! keeps its value: it serves the plain classes alone, and a creation predicts it. And the key a
//! plan takes where the device reads its uniform views at their residuals: those views' constants
//! alone at 0.

use crate::pipeline::{general_form, residual_form_of, uniform_residual_form_of};
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

/// A kernel slot's uniform view with a storage twin (`DescriptorView::UniformWithStorageTwin`)
/// keeps its plan's `RangeAtBindOffset` where every other one of the key holds, its twin read at
/// its bind offset then leaving no residual word to read; where one is 0, as a storage view bound
/// past offset 0 is, the twin's takes 0 too, and the kernel reads the uniform view at its residual
/// (`uniform_residual_form_of`). A uniform view of no twin takes 0 in either, and a constant of no
/// slot keeps its value.
#[test]
fn a_twin_keeps_its_key_only_where_no_other_view_reads_its_residual() {
    let kernel = |index| SlotRef {
        stage: Stage::Kernel,
        class: SlotClass::Buffer,
        index,
    };
    let (storage, moved, twin, other_twin) = (kernel(0), kernel(1), kernel(2), kernel(3));
    let constants = || {
        vec![
            (0, SpecKey::RangeAtBindOffset(storage)),
            (1, SpecKey::RangeAtBindOffset(moved)),
            (2, SpecKey::RangeAtBindOffset(twin)),
            (3, SpecKey::RangeAtBindOffset(other_twin)),
            (4, SpecKey::BicubicTexels(slot(SlotClass::Texture, 0))),
        ]
        .into_iter()
    };
    let no_uniform = |_: SlotRef| false;
    let twins = |at: SlotRef| at == twin || at == other_twin;
    // Every storage view at offset 0 and the constants inline or aligned: the plan's own key.
    let at_bind = key([1, 1, 1, 1, 3]);
    assert_eq!(
        uniform_residual_form_of(constants(), no_uniform, twins, &at_bind),
        None
    );
    // A storage view bound past offset 0 reads its residual: both twins read their uniform views.
    let moved_off = key([1, 0, 1, 1, 3]);
    assert_eq!(
        uniform_residual_form_of(constants(), no_uniform, twins, &moved_off),
        Some(key([1, 0, 0, 0, 3]))
    );
    // A twin bound at an offset no alignment divides reads its uniform view's residual, and so
    // does the other.
    let unaligned = key([1, 1, 0, 1, 3]);
    assert_eq!(
        uniform_residual_form_of(constants(), no_uniform, twins, &unaligned),
        Some(key([1, 1, 0, 0, 3]))
    );
    // A uniform view of no twin takes its residual form, which reads its residual word.
    let uniform = |at: SlotRef| at == moved;
    assert_eq!(
        uniform_residual_form_of(constants(), uniform, twins, &at_bind),
        Some(key([1, 0, 0, 0, 3]))
    );
    // Off the profile that declares twins this is the uniform views' residual form alone.
    let none = |_: SlotRef| false;
    assert_eq!(
        uniform_residual_form_of(constants(), uniform, none, &at_bind),
        residual_form_of(constants(), uniform, &at_bind)
    );
}
