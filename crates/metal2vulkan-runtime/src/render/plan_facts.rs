//! What a render draw asks of its plan's descriptor writes, worked out once as its pipeline state
//! settles the plan for a bind shape (`GraphicsState::settle_shape`, kept with the settle as
//! [`Settled::facts`](crate::pipeline::Settled)) rather than at each draw. A plan's writes number
//! its heap's members, one for each texture of the device in each heap image array of set 1
//! (`metal2vulkan::bind::BindingPlan::heap_elements`), so a draw that walked them for each of
//! these paid every texture the device holds. Each fact is the answer the walk gave, over the
//! same writes.

use metal2vulkan::bind::{BindingPlan, ImageSource, WriteSource};
use metal2vulkan::pipeline::SlotRef;
use metal2vulkan::vk::DescriptorType;

/// What a render draw asks of its plan's descriptor writes (the module docs).
pub(crate) struct PlanFacts {
    /// The slots of the encoder's tables whose textures a write samples: a texture's or a
    /// combined image sampler's image, other than as a storage image.
    sampled: Vec<SlotRef>,
    /// The slots of the encoder's tables whose textures a write binds as a storage image.
    stored: Vec<SlotRef>,
    /// Whether a write is a handle table's: the plan's functions read texture handles.
    reads_handles: bool,
    /// The bytes of the pass's tile memory a write binds, the first such write's.
    tile_memory: Option<u64>,
    /// The colors whose extended-range shadows a write binds, ascending, each once.
    extended: Vec<u32>,
    /// Whether a write binds a YCbCr texture in place or its planes' image.
    ycbcr: bool,
}

impl PlanFacts {
    /// Those of `plan`, from one walk of its writes.
    pub(crate) fn of(plan: &BindingPlan) -> PlanFacts {
        #[cfg(test)]
        PLAN_FACTS_MADE.with(|made| made.set(made.get() + 1));
        let mut facts = PlanFacts {
            sampled: Vec::new(),
            stored: Vec::new(),
            reads_handles: false,
            tile_memory: None,
            extended: Vec::new(),
            ycbcr: false,
        };
        for write in &plan.descriptors {
            let storage = write.ty == DescriptorType::STORAGE_IMAGE;
            match &write.source {
                WriteSource::Texture { slot, .. } if storage => facts.stored.push(*slot),
                WriteSource::Texture { slot, .. }
                | WriteSource::CombinedImageSampler {
                    image: ImageSource::Texture { slot, .. },
                    ..
                } if !storage => facts.sampled.push(*slot),
                WriteSource::HandleTable { .. } => facts.reads_handles = true,
                WriteSource::TileMemory { bytes } => {
                    facts.tile_memory = facts.tile_memory.or(Some(*bytes));
                }
                WriteSource::ExtendedShadow { color_index } => facts.extended.push(*color_index),
                WriteSource::YCbCrInPlace { .. } | WriteSource::YCbCrPlanesImage { .. } => {
                    facts.ycbcr = true;
                }
                _ => {}
            }
        }
        facts.extended.sort_unstable();
        facts.extended.dedup();
        facts
    }

    /// The slots whose textures a write samples, not as a storage image.
    pub(crate) fn sampled(&self) -> &[SlotRef] {
        &self.sampled
    }

    /// The slots whose textures a write binds as a storage image.
    pub(crate) fn stored(&self) -> &[SlotRef] {
        &self.stored
    }

    /// Whether the plan's functions read texture handles: a write is a handle table's.
    pub(crate) fn reads_handles(&self) -> bool {
        self.reads_handles
    }

    /// The bytes of the pass's tile memory the plan binds, where it binds it.
    pub(crate) fn tile_memory(&self) -> Option<u64> {
        self.tile_memory
    }

    /// The colors whose extended-range shadows the plan binds, ascending.
    pub(crate) fn extended(&self) -> &[u32] {
        &self.extended
    }

    /// Whether the plan binds a YCbCr texture in place or its planes' image.
    pub(crate) fn ycbcr(&self) -> bool {
        self.ycbcr
    }
}

#[cfg(test)]
thread_local! {
    /// The plans whose facts this thread worked out ([`PlanFacts::of`]), in tests.
    static PLAN_FACTS_MADE: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// How many plans' facts this thread worked out so far ([`PlanFacts::of`]).
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
pub(crate) fn plan_facts_made() -> u64 {
    PLAN_FACTS_MADE.with(std::cell::Cell::get)
}
