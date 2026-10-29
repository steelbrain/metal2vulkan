//! A crate-owned kernel whose set 0 is a push descriptor set's (`owned_kernel.rs`): Vulkan
//! forbids a set from such a layout (`VkDescriptorSetAllocateInfo`), so its dispatches bind
//! through `OwnedBinds`, which pushes them, and `OwnedSets` refuses the kernel rather than
//! allocate one (the shadow merge's kernel, which a linear attachment's merge dispatches too,
//! `linear_merge.rs`).
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects.

use super::blit_holds::{device, Scratch};
use crate::owned_kernel::{OwnedBinds, OwnedModule, OwnedSet, OwnedSets};
use crate::ResourceOptions;
use metal2vulkan::bind::ShadowMergeStores;

#[test]
fn a_pushed_kernel_s_sets_are_pushed_and_never_allocated() {
    let scratch = Scratch::new("owned-pushes");
    let device = device(&scratch.0);
    let pushes = device.shared().vk().push_descriptor().is_some();
    let buffer = device.new_buffer(4096, ResourceOptions::private()).unwrap();
    let (raw, at) = buffer.at(0);
    for module in [
        OwnedModule::ShadowMerge(ShadowMergeStores::Words),
        OwnedModule::ShadowMerge(ShadowMergeStores::Blocks),
        OwnedModule::RowsCopy,
    ] {
        let kernel = device.owned_kernel(module).unwrap();
        assert_eq!(kernel.pushed, pushes, "{module:?}");
        let set = || OwnedSet {
            images: Vec::new(),
            storage_images: Vec::new(),
            buffers: vec![(raw, at, 256); kernel.buffers as usize],
        };
        let binds = OwnedBinds::new(&device, &kernel, &[set()]).unwrap();
        assert_eq!(matches!(binds, OwnedBinds::Pushed(_)), pushes, "{module:?}");
        let sets = OwnedSets::write_each(&device, &[kernel], &[set()]);
        assert_eq!(
            sets.is_err(),
            pushes,
            "{module:?}: a set from a pushed layout"
        );
    }
}
