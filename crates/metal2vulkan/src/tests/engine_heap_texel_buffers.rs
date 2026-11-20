//! Texture buffers read through argument buffers: the owned fixture `abt_descriptor_sets`, in the
//! shape SPIRV-Cross writes for descriptor sets a Vulkan layer keeps in tier-2 argument buffers,
//! whose texel buffers are `texture_buffer` members. Such a member's handle is a set-1 element of
//! a texel-buffer heap class (`lower::texture`'s heap handles).

use crate::device::{Device, DevicePolicy, VkFeature};
use crate::error::{Capability, Error, Feature, Needs};
use crate::library::FunctionArtifact;
use crate::pipeline::sym::{HandleClass, ImageDim, ResourceKind};
use crate::pipeline::testing::{
    compile, description, engine_dir, link_and_validate, llvm_assemble, AssembleError,
};
use crate::spirv::testing::committed_profiles;
use crate::vk;

fn fixture() -> Option<Vec<u8>> {
    match llvm_assemble(&engine_dir().join("abt_descriptor_sets.ll")) {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected the fixture: {why}"),
    }
}

/// The heap arrays `a` declares: each class with its descriptor type.
fn heap(a: &FunctionArtifact) -> Vec<(HandleClass, vk::DescriptorType)> {
    (a.interface.descriptors.iter())
        .filter_map(|d| match d.slot.kind {
            ResourceKind::Heap(class) => Some((class, d.ty)),
            _ => None,
        })
        .collect()
}

/// `abt_frag_texel_buffer` reads a texture buffer member of its argument buffer. On every committed
/// profile with what a uniform texel-buffer heap array needs (`runtimeDescriptorArray`,
/// `descriptorBindingPartiallyBound`, `shaderUniformTexelBufferArrayNonUniformIndexing`,
/// `descriptorBindingUniformTexelBufferUpdateAfterBind`) it compiles to one set-1 uniform texel
/// buffer array of a `Buffer` class and passes L3; on one without, it is refused naming what lacks,
/// never as not implemented. A device described without the update-after-bind feature alone is
/// refused naming exactly it.
#[test]
fn a_texture_buffer_argument_is_a_texel_buffer_heap_class() {
    let Some(bc) = fixture() else {
        return;
    };
    let entry = "abt_frag_texel_buffer";
    let mut compiled = 0;
    for (stem, p) in committed_profiles().expect("profiles") {
        let d = &p.fields().descriptors;
        let held = d.runtime_descriptor_array
            && d.partially_bound
            && d.non_uniform_uniform_texel_buffer
            && d.update_after_bind_uniform_texel_buffer;
        match compile(&bc, entry, &p) {
            Ok(a) => {
                assert!(held, "{stem}");
                let arrays = heap(&a);
                assert_eq!(arrays.len(), 1, "{stem}: {arrays:?}");
                assert!(
                    matches!(
                        arrays[0],
                        (HandleClass::Image(c), vk::DescriptorType::UNIFORM_TEXEL_BUFFER)
                            if c.dim == ImageDim::Buffer
                    ),
                    "{stem}: {arrays:?}"
                );
                link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{stem}: L3: {e}"));
                compiled += 1;
            }
            Err(Error::Unsupported(u)) => {
                assert!(!held, "{stem}: {u:?}");
                assert_eq!(u.feature, Feature::ArgumentBufferTextures, "{stem}");
                assert!(matches!(u.needs, Needs::AnyOf(_)), "{stem}: {u:?}");
            }
            Err(e) => panic!("{stem}: {e}"),
        }
    }
    assert!(
        compiled > 0,
        "no committed profile holds a texel-buffer heap"
    );

    let mut desc = description("nvidia-rtx50-linux").expect("description");
    desc.features
        .remove(&VkFeature::DescriptorBindingUniformTexelBufferUpdateAfterBind);
    let device = Device::from_queries(desc, DevicePolicy::default()).expect("device");
    match compile(&bc, entry, device.profile()) {
        Err(Error::Unsupported(u)) => {
            assert_eq!(u.feature, Feature::ArgumentBufferTextures);
            assert_eq!(
                u.needs,
                Needs::AnyOf(vec![vec![Capability::Feature(
                    VkFeature::DescriptorBindingUniformTexelBufferUpdateAfterBind
                )]])
            );
        }
        other => panic!("{:?}", other.map(|_| ())),
    }
}
