//! The unit sweep over every committed profile: slot counts up to the Metal maxima (31 buffers, 128
//! textures, 16 samplers per stage) plus synthetic bindings, layouts far above 32 descriptors,
//! compute and render pipelines. Link either produces a dense, exact layout within every
//! `maxDescriptorSet*` limit, or refuses naming a limit the pipeline really exceeds.

use super::*;

const SYNTHETIC: [(SyntheticBinding, vk::DescriptorType); 3] = [
    (
        SyntheticBinding::ArgumentTable,
        vk::DescriptorType::UNIFORM_BUFFER,
    ),
    (
        SyntheticBinding::ExecutionStatus,
        vk::DescriptorType::STORAGE_BUFFER,
    ),
    (
        SyntheticBinding::ConstantBlob,
        vk::DescriptorType::STORAGE_BUFFER,
    ),
];

/// A stage with `buffers` storage buffers, `textures` sampled images, `samplers` samplers, and
/// the synthetic bindings.
fn stage(stage: Stage, buffers: u32, textures: u32, samplers: u32) -> StageInterface {
    use vk::DescriptorType as T;
    let mut d = Vec::new();
    for i in 0..buffers {
        d.push(StageDescriptor::new(
            res(stage, ResourceKind::Buffer, i),
            T::STORAGE_BUFFER,
            1,
        ));
    }
    for i in 0..textures {
        d.push(StageDescriptor::new(
            res(stage, ResourceKind::Texture, i),
            T::SAMPLED_IMAGE,
            1,
        ));
    }
    for i in 0..samplers {
        d.push(StageDescriptor::new(
            res(stage, ResourceKind::Sampler, i),
            T::SAMPLER,
            1,
        ));
    }
    for (b, ty) in SYNTHETIC {
        d.push(StageDescriptor::new(
            res(stage, ResourceKind::Synthetic(b), 0),
            ty,
            1,
        ));
    }
    d.sort();
    iface(stage, d, vec![])
}

/// Independent per-type sums of the stages' descriptors.
fn sums(stages: &[&StageInterface]) -> BTreeMap<VkLimit, u64> {
    let mut n = BTreeMap::new();
    for s in stages {
        for d in &s.descriptors {
            let limit = match d.ty {
                vk::DescriptorType::STORAGE_BUFFER => VkLimit::MaxDescriptorSetStorageBuffers,
                vk::DescriptorType::UNIFORM_BUFFER => VkLimit::MaxDescriptorSetUniformBuffers,
                vk::DescriptorType::SAMPLED_IMAGE => VkLimit::MaxDescriptorSetSampledImages,
                vk::DescriptorType::SAMPLER => VkLimit::MaxDescriptorSetSamplers,
                other => panic!("{other:?}"),
            };
            *n.entry(limit).or_insert(0) += u64::from(d.count);
        }
    }
    n
}

fn available(p: &DeviceProfile, limit: VkLimit) -> u64 {
    let l = &p.fields().limits;
    u64::from(match limit {
        VkLimit::MaxDescriptorSetStorageBuffers => l.max_descriptor_set_storage_buffers,
        VkLimit::MaxDescriptorSetUniformBuffers => l.max_descriptor_set_uniform_buffers,
        VkLimit::MaxDescriptorSetSampledImages => l.max_descriptor_set_sampled_images,
        VkLimit::MaxDescriptorSetSamplers => l.max_descriptor_set_samplers,
        other => panic!("{other:?}"),
    })
}

fn check(p: &DeviceProfile, stem: &str, stages: &[&StageInterface], inputs: &LinkInputs) {
    let arts: Vec<FunctionArtifact> = stages
        .iter()
        .map(|s| artifact((*s).clone(), None))
        .collect();
    let refs: Vec<&FunctionArtifact> = arts.iter().collect();
    let need = sums(stages);
    let breached: Vec<VkLimit> = need
        .iter()
        .filter(|(l, n)| **n > available(p, **l))
        .map(|(l, _)| *l)
        .collect();
    let what = format!("{stem} {need:?}");
    match link(&refs, inputs, p) {
        Ok(l) => {
            assert!(breached.is_empty(), "{what}: linked over {breached:?}");
            let set = &l.layout.sets[0];
            let total: usize = stages.iter().map(|s| s.descriptors.len()).sum();
            assert_eq!(set.bindings.len(), total, "{what}");
            for (n, b) in set.bindings.iter().enumerate() {
                assert_eq!(b.binding, n as u32, "{what}: dense");
                assert_eq!(b.count, 1);
            }
            for s in stages {
                for d in &s.descriptors {
                    let b = &set.bindings[l.table.bindings[&d.slot].1 as usize];
                    assert_eq!((b.ty, b.stages), (d.ty, stage_of(d)), "{what}");
                }
            }
        }
        Err(Error::Unsupported(u)) => {
            assert_eq!(u.feature, Feature::DescriptorCount, "{what}");
            match u.needs {
                crate::error::Needs::AnyOf(alts) => match alts.as_slice() {
                    [one] => match one.as_slice() {
                        [Capability::Limit {
                            limit,
                            required,
                            available: max,
                        }] => {
                            assert!(breached.contains(limit), "{what}: {limit:?}");
                            assert_eq!(*required, need[limit], "{what}");
                            assert_eq!(*max, available(p, *limit), "{what}");
                        }
                        other => panic!("{what}: {other:?}"),
                    },
                    other => panic!("{what}: {other:?}"),
                },
                other => panic!("{what}: {other:?}"),
            }
        }
        Err(e) => panic!("{what}: {e}"),
    }
}

#[test]
fn layouts_up_to_the_metal_maxima_link_or_refuse_exactly_on_every_committed_profile() {
    let profiles = crate::spirv::testing::committed_profiles().expect("profiles");
    assert!(profiles.len() >= 10, "the committed profiles");
    let counts = [
        (1, 0, 0),
        (8, 16, 2),
        (31, 0, 0),
        (31, 128, 16),
        (0, 128, 16),
    ];
    let mut linked = 0;
    let mut refused = 0;
    for (stem, p) in &profiles {
        for (b, t, s) in counts {
            let k = stage(Stage::Kernel, b, t, s);
            check(p, stem, &[&k], &LinkInputs::compute());
            let v = stage(Stage::Vertex, b, t, s);
            let f = stage(Stage::Fragment, b, t, s);
            check(p, stem, &[&v, &f], &render());
            let refs = [&k];
            match link(
                &[&artifact(refs[0].clone(), None)],
                &LinkInputs::compute(),
                p,
            ) {
                Ok(_) => linked += 1,
                Err(_) => refused += 1,
            }
        }
    }
    // The maxima exceed the floor profiles' limits and fit the desktop ones: both outcomes occur.
    assert!(
        linked > 0 && refused > 0,
        "linked {linked}, refused {refused}"
    );
}
