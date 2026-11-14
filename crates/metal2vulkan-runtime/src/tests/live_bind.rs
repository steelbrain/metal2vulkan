//! What a kept bind state's [`Reshape`] names as the change that first told it its shape may have
//! changed (`live_bind.rs`), which a draw rebound that stops reshaped is counted under
//! (`WorkCounts::draws_rebind_reshaped_*`): the first slot change, of its class and kind, and none
//! where a field beside the stage tables came first.

use super::super::live_bind::Reshape;
use super::super::work_counts::ReshapedBy;
use metal2vulkan::pipeline::SlotClass;
use std::collections::BTreeMap;

/// A slot's binding: its shape is its tens.
fn shape(binding: &u32) -> u32 {
    binding / 10
}

/// A slot bound where none was, a slot bound to a binding of another shape and a slot unbound
/// each name their class and kind; a binding of the same shape, or the same binding, names none;
/// and the first change is kept whatever comes after it.
#[test]
fn a_reshape_names_the_first_slot_change() {
    let mut slots = BTreeMap::new();
    let mut reshape = Reshape::default();
    assert!(!reshape.may_reshape());
    reshape.slot(SlotClass::Texture, &mut slots, 0, 10, shape);
    assert_eq!(reshape.reshaped_by(), Some(ReshapedBy::TextureBound));
    // Later changes keep the first.
    reshape.slot(SlotClass::Sampler, &mut slots, 0, 20, shape);
    reshape.unbound(SlotClass::Buffer);
    assert_eq!(reshape.reshaped_by(), Some(ReshapedBy::TextureBound));
    let mut reshape = Reshape::default();
    // The same binding, then one of the same shape: no change of shape.
    reshape.slot(SlotClass::Buffer, &mut slots, 0, 20, shape);
    reshape.slot(SlotClass::Buffer, &mut slots, 0, 21, shape);
    assert!(!reshape.may_reshape());
    assert_eq!(slots[&0], 21, "the binding stored");
    reshape.slot(SlotClass::Sampler, &mut slots, 0, 30, shape);
    assert_eq!(reshape.reshaped_by(), Some(ReshapedBy::SamplerShape));
    let mut reshape = Reshape::default();
    reshape.unbound(SlotClass::Texture);
    assert_eq!(reshape.reshaped_by(), Some(ReshapedBy::TextureUnbound));
    let mut reshape = Reshape::default();
    let mut slot = 30;
    reshape.replace_by(ReshapedBy::BufferOffset, &mut slot, 31, |_, _| true);
    assert!(!reshape.may_reshape(), "a binding of the same shape");
    reshape.replace_by(ReshapedBy::BufferOffset, &mut slot, 40, |_, _| false);
    assert_eq!(reshape.reshaped_by(), Some(ReshapedBy::BufferOffset));
    assert_eq!(slot, 40, "the binding stored");
}

/// A field beside the stage tables that changed first reshapes, but names no slot change, though a
/// slot changed after it.
#[test]
fn a_field_changed_first_names_no_slot() {
    let mut slots = BTreeMap::new();
    let mut reshape = Reshape::default();
    let mut field = 1u32;
    reshape.field(&mut field, 1);
    assert!(!reshape.may_reshape(), "a field kept");
    reshape.field(&mut field, 2);
    reshape.slot(SlotClass::Buffer, &mut slots, 0, 10, shape);
    assert!(reshape.may_reshape());
    assert_eq!(reshape.reshaped_by(), None);
}
