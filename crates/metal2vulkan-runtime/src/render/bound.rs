//! What a draw's secondary command buffer has bound, so a draw leaves out a state command that
//! sets what it already holds.
//!
//! A secondary command buffer begins with nothing bound and inherits nothing from the primary.
//! [`Bound`] follows what the draws recorded into one set there, under Vulkan's rules for what a
//! later command keeps or disturbs:
//!
//! - **The pipeline.** Binding the same `VkPipeline` again is left out. Binding another one
//!   invalidates every dynamic state it does not hold dynamic (and, where it does not make the
//!   vertex strides dynamic, the strides the vertex buffers were bound with), so those are set
//!   again by the next pipeline that holds them. A pipeline holds the dynamic states of the parts
//!   it has: one that discards rasterization has no fragment shader or fragment output state, so
//!   its bind leaves the depth, stencil and blend-constant states unknown too, whatever its
//!   `pDynamicStates` lists, and the next pipeline that holds them sets them again (the draw passes
//!   the states its pipeline holds, `render::fragment_state`).
//! - **Descriptor sets and push constants.** Kept while they are bound or pushed with the same
//!   `VkPipelineLayout` (the same layout is compatible for every set and for the push constants);
//!   a bind or push with another layout forgets what the earlier one held. A draw's push
//!   constants carry, of the same writes in order as the last push's (each write's stages, offset
//!   and length), only those whose bytes differ, and each later one that overlaps one of those
//!   for a stage both name; of other writes, every one ([`Bound::pushes`]).
//! - **A pushed set 0** (`pipeline::objects::pushes_set_zero`). Kept while pushed with the same
//!   layout: a push of the same bindings and elements, in order, carries only the writes that
//!   differ from those set 0 holds (none: left out), as the descriptors it does not write keep
//!   theirs; a bind of set 0 or a push or bind with another layout forgets it. The Vulkan
//!   specification, `vkCmdPushDescriptorSet`: "Push descriptors can be updated incrementally and
//!   cause shaders to use the updated descriptors for subsequent bound pipeline commands with the
//!   pipeline type set by pipelineBindPoint until the descriptor is overwritten, or else until
//!   the set is disturbed as described in Pipeline Layout Compatibility. When the set is
//!   disturbed or push descriptors with a different descriptor set layout are set, all push
//!   descriptors are undefined." One layout's set 0 is one set layout. Set 1 bound with the same
//!   layout disturbs no set 0, and a pipeline bound between draws disturbs none (Pipeline Layout
//!   Compatibility): "When binding a descriptor set (see Descriptor Set Binding) to set number N,
//!   a previously bound descriptor set bound with lower index M than N is disturbed if the
//!   pipeline layouts for set M and N are not compatible for set M. Otherwise, the bound
//!   descriptor set in M is not disturbed.", and "When binding a pipeline, the pipeline can
//!   correctly access any previously bound descriptor set N if it was bound with a compatible
//!   pipeline layout for set N, and it was not disturbed." A pipeline of another layout is drawn
//!   with that layout, whose push or bind forgets what set 0 held. Variants whose layouts are
//!   identically defined share one (`pipeline/shared_layouts.rs`), so a pipeline bind between
//!   their draws keeps set 0 and the push constants too. Where the driver's pushed descriptors go
//!   stale across a pipeline bind
//!   (`RuntimeDriverDefect::PushedDescriptorsStaleAfterPipelineBinds`), the push after every
//!   pipeline bind carries at least one write: the writes that changed, or else set 0's first, as
//!   it holds it. Any push there makes the driver take the bound pipeline's descriptors again;
//!   what the push leaves out stays as set 0 holds it.
//! - **Equal shapes, measured.** Two layouts that are not one are not compatible here, even
//!   where their set 0 and push-constant ranges have the same shape
//!   (`VariantObjects::set_zero_shape`) and Vulkan would hold them compatible for set 0 (Pipeline
//!   Layout Compatibility: "compatible for set N" if created "with identically defined descriptor
//!   set layouts for sets zero through N", the same
//!   `VK_PIPELINE_LAYOUT_CREATE_INDEPENDENT_SETS_BIT_EXT` and "identical push constant ranges"):
//!   a push "with a different descriptor set layout" leaves every pushed descriptor undefined,
//!   and the specification does not say whether an identically defined layout is a different
//!   one. So a draw of another layout of the same shape (one that differs in a later set, or
//!   bakes a host step's own samplers) pushes set 0 whole. [`Bound::pushed_set`] counts such
//!   pushes and the writes they would carry if those layouts were one
//!   (`WorkCounts::set_zero_equal_shape_switches` and kin): what a push carries is unchanged.
//! - **Vertex and index buffers.** Each binding is kept until bound again; binding a pipeline
//!   disturbs none.
//! - **Dynamic states.** Each is kept, at the value it was set to, until a pipeline that makes it
//!   static is bound. Values compare as the commands carry them: floats bit for bit, so a zero
//!   of the other sign is set again.
//!
//! Only draws record these commands into the command buffer; its other commands (barriers and
//! queries) disturb none of them. A record follows one command buffer: a draw into another
//! begins it again, empty.
//!
//! A record counts the recordings recorded through it and its beginnings again
//! ([`Bound::records`]): where the count is what it was just after a draw's recording, nothing
//! was recorded into the command buffer through it since, so it holds that draw's pipeline and
//! dynamic states as the draw set them. A draw rebound after it, of the same pipeline and states,
//! then compares none of the states a value here holds (`render/draw_again/rebound.rs`), each of
//! whose compares would answer that it holds them.

use ash::vk;

use super::{DrawState, IndexBind};
use crate::binding::{PushWrite, SetWrite};
use crate::pipeline::FixedState;
use crate::render_state::{DepthBiasSetting, ViewportList};

/// What draws' recordings recorded (`render::Recording::record`): their state commands, the
/// writes their set 0's pushes carried, and their pushes across equal shapes.
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct Recorded {
    pub(super) states: u64,
    pub(super) writes_pushed: u64,
    pub(super) equal_shapes: EqualShapes,
}

/// Draws' pushes that carried set 0 whole although set 0 held the pushed descriptors of another
/// layout of the same shape ([`Bound::equal_shape_writes`]): how many, the writes they carried,
/// and the writes they would carry if the layouts were one.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct EqualShapes {
    pub(super) switches: u64,
    pub(super) carried: u64,
    pub(super) if_shared: u64,
}

impl std::ops::AddAssign for Recorded {
    fn add_assign(&mut self, other: Recorded) {
        self.states += other.states;
        self.writes_pushed += other.writes_pushed;
        self.equal_shapes.switches += other.equal_shapes.switches;
        self.equal_shapes.carried += other.equal_shapes.carried;
        self.equal_shapes.if_shared += other.equal_shapes.if_shared;
    }
}

impl Recorded {
    /// Whether the draw recorded no state command and pushed no write: its Vulkan draw alone.
    pub(super) fn draw_alone(&self) -> bool {
        self.states == 0 && self.writes_pushed == 0 && self.equal_shapes == EqualShapes::default()
    }

    /// Add them to `counts` (`WorkCounts::draw_states_recorded`, `set_zero_writes_pushed`,
    /// `set_zero_equal_shape_switches` and kin).
    pub(super) fn count(self, counts: crate::work_counts::Counting<'_>) {
        counts.draw_states_recorded(self.states);
        counts.set_zero_writes_pushed(self.writes_pushed);
        self.count_equal_shapes(counts);
    }

    /// Add its pushes that carried set 0 whole to `counts` (`set_zero_equal_shape_switches` and
    /// kin): what [`count`](Self::count) adds but its states and writes pushed.
    pub(super) fn count_equal_shapes(self, counts: crate::work_counts::Counting<'_>) {
        let EqualShapes {
            switches,
            carried,
            if_shared,
        } = self.equal_shapes;
        counts.set_zero_equal_shape_switches(switches, carried, if_shared);
    }
}

/// The state commands recorded into one secondary command buffer that still hold (the module
/// docs).
#[derive(Default)]
pub(super) struct Bound {
    /// The command buffer: null before its first draw.
    command_buffer: vk::CommandBuffer,
    /// The bound pipeline: null before the first.
    pipeline: vk::Pipeline,
    /// The layout the descriptor sets below were bound with, and each set number's set.
    sets_layout: vk::PipelineLayout,
    sets: Vec<Option<vk::DescriptorSet>>,
    /// The dynamic offsets set 0 was bound with, where `sets` holds it (`binding::dynamic_writes`).
    offsets: Vec<u32>,
    /// The descriptors pushed as set 0 with `sets_layout`, where set 0 holds pushed ones.
    pushed: Vec<SetWrite>,
    /// The shape of `sets_layout`'s set 0 and push-constant ranges, as the last push gave it
    /// (`VariantObjects::set_zero_shape`).
    pushed_shape: u64,
    /// A pipeline bind since the last push, where the driver's pushed descriptors go stale across
    /// one (a driver's defect: the module docs): the next push carries at least one write.
    forgotten: bool,
    /// The writes the last push that set 0 held the bindings of carried ([`Bound::pushed_set`]).
    changed: Vec<SetWrite>,
    /// The layout the last push constants were pushed with, and their writes: each write's
    /// stages, offset and length, its bytes one after another in `push_bytes`.
    push_layout: vk::PipelineLayout,
    pushes: Vec<(vk::ShaderStageFlags, u32, usize)>,
    push_bytes: Vec<u8>,
    /// `push_bytes` before the last push (builds with debug assertions: [`check_pushed`]), its
    /// storage kept from push to push, so a draw allocates as many times as without them.
    #[cfg(debug_assertions)]
    push_before: Vec<u8>,
    /// Each vertex buffer binding: its buffer, offset and, bound with the strides dynamic, its
    /// stride.
    vertex: Vec<(u32, vk::Buffer, u64, Option<u64>)>,
    /// The index buffer: its buffer, offset, index type and, bound with a size, the size.
    index: Option<(vk::Buffer, u64, vk::IndexType, Option<u64>)>,
    /// The dynamic states set that still hold, and their values.
    set: States,
    values: Option<Values>,
    /// The recordings recorded through it and its beginnings again (the module docs), from its
    /// first; only ever counted up.
    records: u64,
}

#[cfg(debug_assertions)]
thread_local! {
    /// The compares of a dynamic state's value with a record's this thread made
    /// ([`Bound::dynamic`]), in builds with debug assertions.
    static DYNAMIC_COMPARES: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// How many dynamic states' values this thread compared with a record's so far
/// ([`Bound::dynamic`]).
#[cfg(all(test, debug_assertions, any(m2v_gpu, m2v_cpu_icd)))]
pub(crate) fn dynamic_compares() -> u64 {
    DYNAMIC_COMPARES.with(std::cell::Cell::get)
}

/// Dynamic states whose values [`Values`] holds, a bit each: a draw tests and notes each of its
/// pipeline's at every draw, which a bit does without a search, and a record holds them with no
/// storage of its own.
#[derive(Clone, Copy, Default)]
struct States(u16);

impl States {
    /// The bit of `state`, where [`Values`] holds its value.
    #[inline]
    fn bit(state: vk::DynamicState) -> Option<u16> {
        let at = match state {
            vk::DynamicState::VIEWPORT => 0,
            vk::DynamicState::SCISSOR => 1,
            vk::DynamicState::DEPTH_BIAS => 2,
            vk::DynamicState::BLEND_CONSTANTS => 3,
            vk::DynamicState::STENCIL_COMPARE_MASK => 4,
            vk::DynamicState::STENCIL_WRITE_MASK => 5,
            vk::DynamicState::STENCIL_REFERENCE => 6,
            vk::DynamicState::CULL_MODE => 7,
            vk::DynamicState::FRONT_FACE => 8,
            vk::DynamicState::PRIMITIVE_TOPOLOGY => 9,
            vk::DynamicState::PRIMITIVE_RESTART_ENABLE => 10,
            vk::DynamicState::DEPTH_TEST_ENABLE => 11,
            vk::DynamicState::DEPTH_WRITE_ENABLE => 12,
            vk::DynamicState::DEPTH_COMPARE_OP => 13,
            vk::DynamicState::STENCIL_TEST_ENABLE => 14,
            vk::DynamicState::STENCIL_OP => 15,
            _ => return None,
        };
        Some(1 << at)
    }

    /// Those of `states` [`Values`] holds.
    #[inline]
    fn of(states: &[vk::DynamicState]) -> States {
        States(
            states
                .iter()
                .filter_map(|&s| States::bit(s))
                .fold(0, |all, bit| all | bit),
        )
    }
}

/// The values a draw sets its dynamic states to: the encoder's state at the draw and its input
/// assembly.
pub(super) struct Values {
    viewports: ViewportList<vk::Viewport>,
    scissors: ViewportList<vk::Rect2D>,
    fixed: FixedState,
    stencil_reference: [u32; 2],
    depth_bias: DepthBiasSetting,
    blend_color: [f32; 4],
    topology: metal2vulkan::vk::PrimitiveTopology,
    primitive_restart: bool,
}

/// A draw's dynamic state values, borrowed: the encoder's state and the input assembly.
#[derive(Clone, Copy)]
pub(super) struct Now<'a> {
    pub(super) state: &'a DrawState,
    pub(super) topology: metal2vulkan::vk::PrimitiveTopology,
    pub(super) primitive_restart: bool,
}

// The checks a draw makes here, at every draw, are inline: they join `Recording::record` in
// whichever codegen unit it lands in, as the state they test is matched there too.
/// Which writes of a draw's push constants its push carries ([`Bound::pushes`]): by index.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Pushed(u64);

impl Pushed {
    /// Every write.
    const ALL: Pushed = Pushed(u64::MAX);
    /// The most writes a push leaves any out of: of more, every one is pushed.
    const MOST: usize = 64;

    /// Whether the push carries any write.
    pub(super) fn any(self) -> bool {
        self.0 != 0
    }

    /// Whether the push carries write `at`.
    pub(super) fn has(self, at: usize) -> bool {
        self == Pushed::ALL || self.0 >> at & 1 != 0
    }
}

/// That pushing `pushed` of `writes` over push constants that hold the last push's writes, whose
/// bytes were `before` (one after another, each write's stages, offset and length those of
/// `writes`), leaves every stage reading what pushing every write leaves it: at each byte a write
/// sets for a stage, the last pushed write that sets it, else the last push's, is the last write
/// that sets it. Nothing allocated.
#[cfg(debug_assertions)]
fn check_pushed(writes: &[PushWrite], before: &[u8], pushed: Pushed) {
    let sets = |write: &PushWrite, bit: u32, byte: usize| {
        let start = write.1 as usize;
        write.0.as_raw() >> bit & 1 != 0 && start <= byte && byte < start + write.2.len()
    };
    for write in writes {
        for bit in (0..32).filter(|bit| write.0.as_raw() >> bit & 1 != 0) {
            for byte in write.1 as usize..write.1 as usize + write.2.len() {
                let last = (writes.iter().rev()).find(|w| sets(w, bit, byte));
                let every = last.map(|w| w.2[byte - w.1 as usize]);
                let carried = match (writes.iter().enumerate().rev())
                    .find(|&(at, w)| pushed.has(at) && sets(w, bit, byte))
                {
                    Some((_, w)) => Some(w.2[byte - w.1 as usize]),
                    None => {
                        let (mut at, mut held) = (0, None);
                        for w in writes {
                            if sets(w, bit, byte) {
                                held = Some(before[at + byte - w.1 as usize]);
                            }
                            at += w.2.len();
                        }
                        held
                    }
                };
                assert_eq!(
                    every, carried,
                    "a push of some of a draw's push-constant writes leaves stage bit {bit} \
                     reading another byte {byte} than a push of all of them"
                );
            }
        }
    }
}

impl Bound {
    /// Follow `command_buffer`: where the record is another's, it begins again, empty.
    #[inline]
    pub(super) fn follow(&mut self, command_buffer: vk::CommandBuffer) {
        if self.command_buffer != command_buffer {
            *self = Bound {
                command_buffer,
                records: self.records + 1,
                ..Bound::default()
            };
        }
    }

    /// The recordings recorded through it and its beginnings again so far (the module docs).
    #[inline]
    pub(super) fn records(&self) -> u64 {
        self.records
    }

    /// Count a recording recorded through it (`render::Recording::record`).
    #[inline]
    pub(super) fn recorded(&mut self) {
        self.records += 1;
    }

    /// Whether a value here holds dynamic `state` ([`dynamic`](Self::dynamic) compares it), else
    /// a draw that makes it dynamic sets it at every draw.
    #[inline]
    pub(super) fn holds_value(state: vk::DynamicState) -> bool {
        States::bit(state).is_some()
    }

    /// Whether binding `pipeline`, which holds the dynamic states `dynamic` (the module docs),
    /// sets anything: noted as bound where it does, the next push of set 0 then carrying at least
    /// one write where `forgets_pushed` (a driver's defect: the module docs).
    #[inline]
    pub(super) fn pipeline(
        &mut self,
        pipeline: vk::Pipeline,
        dynamic: &[vk::DynamicState],
        forgets_pushed: bool,
    ) -> bool {
        if self.pipeline == pipeline {
            return false;
        }
        self.pipeline = pipeline;
        if forgets_pushed {
            self.forgotten = true;
        }
        self.set.0 &= States::of(dynamic).0;
        if !dynamic.contains(&vk::DynamicState::VERTEX_INPUT_BINDING_STRIDE) {
            self.vertex.retain(|&(.., stride)| stride.is_none());
        }
        true
    }

    /// Whether the record holds `sets` bound from set number `first` with `layout`, set 0 with the
    /// dynamic offsets `offsets`: binding them again sets nothing ([`sets`](Self::sets)).
    #[inline]
    pub(super) fn holds_sets(
        &self,
        layout: vk::PipelineLayout,
        first: u32,
        (sets, offsets): (&[vk::DescriptorSet], &[u32]),
    ) -> bool {
        let first = first as usize;
        let held = |at: usize| self.sets.get(at).copied().flatten();
        self.sets_layout == layout
            && sets
                .iter()
                .enumerate()
                .all(|(i, &set)| held(first + i) == Some(set))
            && (first > 0 || self.offsets == offsets)
    }

    /// Whether binding `sets` from set number `first` with `layout` and the dynamic offsets
    /// `offsets` sets anything: noted as bound where it does. A set holds its dynamic offsets as
    /// it was bound with them, so set 0 bound again with other offsets sets them. Only set 0
    /// holds dynamic buffers (`objects::dynamic_bindings`).
    #[inline]
    pub(super) fn sets(
        &mut self,
        layout: vk::PipelineLayout,
        first: u32,
        (sets, offsets): (&[vk::DescriptorSet], &[u32]),
    ) -> bool {
        if self.holds_sets(layout, first, (sets, offsets)) {
            return false;
        }
        let first = first as usize;
        if first == 0 {
            self.offsets.clear();
            self.offsets.extend_from_slice(offsets);
        }
        if self.sets_layout != layout {
            self.sets_layout = layout;
            self.sets.clear();
            self.pushed.clear();
        }
        if first == 0 {
            self.pushed.clear();
        }
        if self.sets.len() < first + sets.len() {
            self.sets.resize(first + sets.len(), None);
        }
        for (i, &set) in sets.iter().enumerate() {
            self.sets[first + i] = Some(set);
        }
        true
    }

    /// The writes of `writes`, set 0 with `layout`, whose set 0 and push-constant ranges have
    /// `shape`, that a push carries, noted as pushed: where set 0 holds pushed descriptors of the
    /// same bindings and elements, in order, with `layout`, those that differ from them (none
    /// where all are the same, but the first where a pipeline bind since the last push leaves
    /// the driver's pushed descriptors stale: the module docs); else all of them. A push leaves
    /// the later sets bound with the same layout as they were. With them, where the push carries
    /// all of them across a switch between equal shapes, the writes it would carry if those
    /// layouts were one ([`equal_shape_writes`](Self::equal_shape_writes)), worked out only then:
    /// a push that holds set 0 counts nothing.
    pub(super) fn pushed_set<'a>(
        &'a mut self,
        layout: vk::PipelineLayout,
        shape: u64,
        writes: &'a [SetWrite],
    ) -> (&'a [SetWrite], Option<u64>) {
        let held = self.sets_layout == layout
            && !self.pushed.is_empty()
            && self.pushed.len() == writes.len()
            && (self.pushed.iter().zip(writes))
                .all(|(a, b)| (a.binding, a.element, a.ty) == (b.binding, b.element, b.ty));
        if held {
            self.changed.clear();
            for (old, new) in self.pushed.iter_mut().zip(writes) {
                if !old.same(new) {
                    *old = *new;
                    self.changed.push(*new);
                }
            }
            if self.forgotten {
                // The driver takes the bound pipeline's descriptors again at any push: set 0's
                // first, as it holds it, where none changed.
                if let (true, Some(&first)) = (self.changed.is_empty(), writes.first()) {
                    self.changed.push(first);
                }
                self.forgotten = false;
            }
            return (&self.changed, None);
        }
        let if_shared = self.equal_shape_writes(layout, shape, writes);
        if self.sets_layout != layout {
            self.sets_layout = layout;
            self.sets.clear();
        }
        match self.sets.first_mut() {
            Some(first) => *first = None,
            None => self.sets.push(None),
        }
        self.pushed.clear();
        self.pushed.extend_from_slice(writes);
        self.pushed_shape = shape;
        self.forgotten = false;
        (writes, if_shared)
    }

    /// Where a push of `writes` with `layout`, whose set 0 and push-constant ranges have `shape`,
    /// carries set 0 whole although set 0 holds the pushed descriptors of another layout of the
    /// same shape: the writes it would carry if the layouts were one. Those that differ from what
    /// set 0 holds (all of them where its bindings and elements differ), and at least one after a
    /// pipeline bind where the driver's pushed descriptors go stale across one, as the push of a
    /// held set 0 carries there (`WorkCounts::set_zero_writes_if_shapes_shared`). `None` for any
    /// other push. Measured only: [`Bound::pushed_set`] carries what it carried.
    fn equal_shape_writes(
        &self,
        layout: vk::PipelineLayout,
        shape: u64,
        writes: &[SetWrite],
    ) -> Option<u64> {
        if self.sets_layout == layout || self.pushed.is_empty() || shape != self.pushed_shape {
            return None;
        }
        let same_bindings = self.pushed.len() == writes.len()
            && (self.pushed.iter().zip(writes))
                .all(|(a, b)| (a.binding, a.element, a.ty) == (b.binding, b.element, b.ty));
        if !same_bindings {
            return Some(writes.len() as u64);
        }
        let changed = (self.pushed.iter().zip(writes))
            .filter(|(old, new)| !old.same(new))
            .count() as u64;
        Some(changed.max(u64::from(self.forgotten)))
    }

    /// Which of `writes`, a draw's push-constant writes with `layout`, its push carries: noted as
    /// pushed. All of them where the last push was of another layout or of other writes (their
    /// count, or a write's stages, offset or length); else those whose bytes differ from the last
    /// push's, and each write after one of those that shares a stage and a byte with it, which a
    /// push of all of them would leave its own bytes over. Push constants are updated
    /// incrementally (`vkCmdPushConstants`: "causing shader stages in stageFlags to read the new
    /// data from pValues for push constants modified by this command, while still reading the
    /// previous data for push constants not modified by this command"), so the stages read what
    /// a push of every write would leave them (checked in builds with debug assertions).
    #[inline]
    pub(super) fn pushes(&mut self, layout: vk::PipelineLayout, writes: &[PushWrite]) -> Pushed {
        let same_writes = self.push_layout == layout
            && self.pushes.len() == writes.len()
            && writes.len() <= Pushed::MOST
            && (self.pushes.iter().zip(writes)).all(|(&(stages, offset, length), write)| {
                (stages, offset, length) == (write.0, write.1, write.2.len())
            });
        if !same_writes {
            self.push_layout = layout;
            self.pushes.clear();
            self.push_bytes.clear();
            for (stages, offset, bytes) in writes {
                self.pushes.push((*stages, *offset, bytes.len()));
                self.push_bytes.extend_from_slice(bytes);
            }
            return Pushed::ALL;
        }
        #[cfg(debug_assertions)]
        {
            self.push_before.clear();
            self.push_before.extend_from_slice(&self.push_bytes);
        }
        let mut changed = 0u64;
        let mut at = 0;
        for (i, write) in writes.iter().enumerate() {
            let held = &mut self.push_bytes[at..at + write.2.len()];
            if *held != write.2[..] {
                held.copy_from_slice(&write.2);
                changed |= 1 << i;
            }
            at += write.2.len();
        }
        // A later write a changed one overlaps for a stage both name, pushed again after it.
        let overlaps = |a: &PushWrite, b: &PushWrite| {
            let (a_end, b_end) = (a.1 as usize + a.2.len(), b.1 as usize + b.2.len());
            a.0.intersects(b.0) && (a.1 as usize) < b_end && (b.1 as usize) < a_end
        };
        let mut pushed = changed;
        for i in 0..writes.len() {
            if pushed >> i & 1 == 0 {
                continue;
            }
            for (j, later) in writes.iter().enumerate().skip(i + 1) {
                if overlaps(&writes[i], later) {
                    pushed |= 1 << j;
                }
            }
        }
        #[cfg(debug_assertions)]
        check_pushed(writes, &self.push_before, Pushed(pushed));
        Pushed(pushed)
    }

    /// Whether binding `buffer` at `offset` to vertex buffer `binding` (with `stride`, where the
    /// strides are dynamic) sets anything: noted as bound where it does.
    #[inline]
    pub(super) fn vertex_buffer(
        &mut self,
        binding: u32,
        buffer: vk::Buffer,
        offset: u64,
        stride: Option<u64>,
    ) -> bool {
        let bound = (binding, buffer, offset, stride);
        match self.vertex.iter_mut().find(|held| held.0 == binding) {
            Some(held) if *held == bound => false,
            Some(held) => {
                *held = bound;
                true
            }
            None => {
                self.vertex.push(bound);
                true
            }
        }
    }

    /// Whether binding `index` sets anything: noted as bound where it does.
    #[inline]
    pub(super) fn index_buffer(&mut self, index: &IndexBind) -> bool {
        let bound = (
            index.buffer,
            index.offset,
            index.index_type,
            index.sized.map(|(_, size)| size),
        );
        if self.index == Some(bound) {
            return false;
        }
        self.index = Some(bound);
        true
    }

    /// Whether setting dynamic `state` to its value `now` sets anything: noted as set where it
    /// does. A state no value here holds is always set.
    #[inline]
    pub(super) fn dynamic(&mut self, state: vk::DynamicState, now: Now<'_>) -> bool {
        let Some(bit) = States::bit(state) else {
            return true;
        };
        #[cfg(debug_assertions)]
        DYNAMIC_COMPARES.with(|compares| compares.set(compares.get() + 1));
        let values = self.values.get_or_insert_with(|| Values::of(now));
        let held = self.set.0 & bit != 0;
        if held && values.holds(state, now) == Some(true) {
            return false;
        }
        if values.take(state, now) {
            self.set.0 |= bit;
        }
        true
    }
}

impl Values {
    fn of(now: Now<'_>) -> Values {
        let s = now.state;
        Values {
            viewports: s.viewports.clone(),
            scissors: s.scissors.clone(),
            fixed: s.fixed,
            stencil_reference: s.stencil_reference,
            depth_bias: s.depth_bias.setting(),
            blend_color: s.blend_color,
            topology: now.topology,
            primitive_restart: now.primitive_restart,
        }
    }

    /// Whether `state` holds the value `now` sets it to; `None` for a state no value here holds.
    #[inline]
    fn holds(&self, state: vk::DynamicState, now: Now<'_>) -> Option<bool> {
        let s = now.state;
        let (ds, now_ds) = (&self.fixed.depth_stencil, &s.fixed.depth_stencil);
        let faces = |f: fn(&metal2vulkan::metal::state::StencilOpState) -> u32| {
            [f(&ds.front), f(&ds.back)] == [f(&now_ds.front), f(&now_ds.back)]
        };
        Some(match state {
            vk::DynamicState::VIEWPORT => {
                self.viewports.len() == s.viewports.len()
                    && self
                        .viewports
                        .iter()
                        .zip(s.viewports.iter())
                        .all(|(a, b)| viewport_bits(a) == viewport_bits(b))
            }
            vk::DynamicState::SCISSOR => {
                self.scissors.len() == s.scissors.len()
                    && self
                        .scissors
                        .iter()
                        .zip(s.scissors.iter())
                        .all(|(a, b)| rect(a) == rect(b))
            }
            vk::DynamicState::DEPTH_BIAS => self.depth_bias == s.depth_bias.setting(),
            vk::DynamicState::BLEND_CONSTANTS => {
                self.blend_color.map(f32::to_bits) == s.blend_color.map(f32::to_bits)
            }
            vk::DynamicState::STENCIL_COMPARE_MASK => faces(|face| face.compare_mask),
            vk::DynamicState::STENCIL_WRITE_MASK => faces(|face| face.write_mask),
            vk::DynamicState::STENCIL_REFERENCE => self.stencil_reference == s.stencil_reference,
            vk::DynamicState::CULL_MODE => self.fixed.cull_mode == s.fixed.cull_mode,
            vk::DynamicState::FRONT_FACE => self.fixed.front_face == s.fixed.front_face,
            vk::DynamicState::PRIMITIVE_TOPOLOGY => self.topology == now.topology,
            vk::DynamicState::PRIMITIVE_RESTART_ENABLE => {
                self.primitive_restart == now.primitive_restart
            }
            vk::DynamicState::DEPTH_TEST_ENABLE => ds.depth_test_enable == now_ds.depth_test_enable,
            vk::DynamicState::DEPTH_WRITE_ENABLE => {
                ds.depth_write_enable == now_ds.depth_write_enable
            }
            vk::DynamicState::DEPTH_COMPARE_OP => ds.depth_compare_op == now_ds.depth_compare_op,
            vk::DynamicState::STENCIL_TEST_ENABLE => {
                ds.stencil_test_enable == now_ds.stencil_test_enable
            }
            vk::DynamicState::STENCIL_OP => {
                let ops = |face: &metal2vulkan::metal::state::StencilOpState| {
                    (
                        face.fail_op,
                        face.pass_op,
                        face.depth_fail_op,
                        face.compare_op,
                    )
                };
                [ops(&ds.front), ops(&ds.back)] == [ops(&now_ds.front), ops(&now_ds.back)]
            }
            _ => return None,
        })
    }

    /// Take `state`'s value from `now`; `false` for a state no value here holds.
    #[inline]
    fn take(&mut self, state: vk::DynamicState, now: Now<'_>) -> bool {
        let s = now.state;
        let (ds, now_ds) = (&mut self.fixed.depth_stencil, &s.fixed.depth_stencil);
        match state {
            vk::DynamicState::VIEWPORT => self.viewports.clone_from(&s.viewports),
            vk::DynamicState::SCISSOR => self.scissors.clone_from(&s.scissors),
            vk::DynamicState::DEPTH_BIAS => self.depth_bias = s.depth_bias.setting(),
            vk::DynamicState::BLEND_CONSTANTS => self.blend_color = s.blend_color,
            vk::DynamicState::STENCIL_COMPARE_MASK => {
                ds.front.compare_mask = now_ds.front.compare_mask;
                ds.back.compare_mask = now_ds.back.compare_mask;
            }
            vk::DynamicState::STENCIL_WRITE_MASK => {
                ds.front.write_mask = now_ds.front.write_mask;
                ds.back.write_mask = now_ds.back.write_mask;
            }
            vk::DynamicState::STENCIL_REFERENCE => self.stencil_reference = s.stencil_reference,
            vk::DynamicState::CULL_MODE => self.fixed.cull_mode = s.fixed.cull_mode,
            vk::DynamicState::FRONT_FACE => self.fixed.front_face = s.fixed.front_face,
            vk::DynamicState::PRIMITIVE_TOPOLOGY => self.topology = now.topology,
            vk::DynamicState::PRIMITIVE_RESTART_ENABLE => {
                self.primitive_restart = now.primitive_restart
            }
            vk::DynamicState::DEPTH_TEST_ENABLE => ds.depth_test_enable = now_ds.depth_test_enable,
            vk::DynamicState::DEPTH_WRITE_ENABLE => {
                ds.depth_write_enable = now_ds.depth_write_enable
            }
            vk::DynamicState::DEPTH_COMPARE_OP => ds.depth_compare_op = now_ds.depth_compare_op,
            vk::DynamicState::STENCIL_TEST_ENABLE => {
                ds.stencil_test_enable = now_ds.stencil_test_enable
            }
            vk::DynamicState::STENCIL_OP => {
                for (face, now_face) in
                    [(&mut ds.front, &now_ds.front), (&mut ds.back, &now_ds.back)]
                {
                    face.fail_op = now_face.fail_op;
                    face.pass_op = now_face.pass_op;
                    face.depth_fail_op = now_face.depth_fail_op;
                    face.compare_op = now_face.compare_op;
                }
            }
            _ => return false,
        }
        true
    }
}

/// A viewport's fields as the command carries them, bit for bit.
#[inline]
fn viewport_bits(viewport: &vk::Viewport) -> [u32; 6] {
    [
        viewport.x,
        viewport.y,
        viewport.width,
        viewport.height,
        viewport.min_depth,
        viewport.max_depth,
    ]
    .map(f32::to_bits)
}

/// A scissor's fields.
#[inline]
fn rect(rect: &vk::Rect2D) -> [i64; 4] {
    [
        i64::from(rect.offset.x),
        i64::from(rect.offset.y),
        i64::from(rect.extent.width),
        i64::from(rect.extent.height),
    ]
}

#[cfg(test)]
mod tests {
    use super::{Bound, Pushed};
    use crate::binding::{Info, SetWrite};
    use ash::vk::{self, Handle};

    /// A uniform buffer write at binding `binding` of the buffer whose handle is `buffer`.
    fn uniform(binding: u32, buffer: u64) -> SetWrite {
        SetWrite {
            binding,
            element: 0,
            ty: vk::DescriptorType::UNIFORM_BUFFER,
            info: Info::Buffer(vk::DescriptorBufferInfo {
                buffer: vk::Buffer::from_raw(buffer),
                offset: 0,
                range: 64,
            }),
        }
    }

    /// A record counts the recordings through it and its beginnings again: following the
    /// command buffer it follows counts nothing, following another begins it again, empty, and
    /// counts that.
    #[test]
    fn a_record_counts_its_recordings_and_its_beginnings_again() {
        let mut bound = Bound::default();
        assert_eq!(bound.records(), 0);
        bound.follow(vk::CommandBuffer::from_raw(2));
        assert_eq!(bound.records(), 1);
        assert!(bound.pipeline(vk::Pipeline::from_raw(3), &[], false));
        bound.recorded();
        bound.follow(vk::CommandBuffer::from_raw(2));
        assert_eq!(bound.records(), 2);
        assert!(!bound.pipeline(vk::Pipeline::from_raw(3), &[], false));
        bound.follow(vk::CommandBuffer::from_raw(4));
        assert_eq!(bound.records(), 3);
        assert!(
            bound.pipeline(vk::Pipeline::from_raw(3), &[], false),
            "begun again"
        );
    }

    /// A push of the same writes as the last one carries those whose bytes changed, and each
    /// later one that overlaps one of those for a shared stage; a push of another layout or of
    /// other writes carries every one.
    #[test]
    fn a_push_carries_the_writes_that_changed() {
        let (vertex, fragment) = (vk::ShaderStageFlags::VERTEX, vk::ShaderStageFlags::FRAGMENT);
        let (a, b) = (
            vk::PipelineLayout::from_raw(1),
            vk::PipelineLayout::from_raw(2),
        );
        let carried =
            |pushed: Pushed, n: usize| (0..n).filter(|&at| pushed.has(at)).collect::<Vec<_>>();
        let mut bound = Bound::default();
        bound.follow(vk::CommandBuffer::from_raw(2));
        let apart = |v: u8, f: u8| vec![(vertex, 0, vec![v; 16]), (fragment, 16, vec![f; 16])];
        assert_eq!(carried(bound.pushes(a, &apart(1, 2)), 2), [0, 1]);
        assert!(!bound.pushes(a, &apart(1, 2)).any(), "the same bytes");
        assert_eq!(
            carried(bound.pushes(a, &apart(3, 2)), 2),
            [0],
            "the vertex write"
        );
        assert_eq!(
            carried(bound.pushes(a, &apart(3, 4)), 2),
            [1],
            "the fragment write"
        );
        assert_eq!(
            carried(bound.pushes(b, &apart(3, 4)), 2),
            [0, 1],
            "another layout"
        );
        let longer = vec![(vertex, 0, vec![3; 32]), (fragment, 32, vec![4; 16])];
        assert_eq!(
            carried(bound.pushes(b, &longer), 2),
            [0, 1],
            "other lengths"
        );
        // A write of both stages, then one of the fragment stage over its second half.
        let both = vertex | fragment;
        let over = |w: u8, f: u8| vec![(both, 0, vec![w; 16]), (fragment, 8, vec![f; 8])];
        assert_eq!(carried(bound.pushes(a, &over(1, 2)), 2), [0, 1]);
        assert_eq!(
            carried(bound.pushes(a, &over(5, 2)), 2),
            [0, 1],
            "pushed again after"
        );
        assert_eq!(carried(bound.pushes(a, &over(5, 6)), 2), [1]);
    }

    /// A push after a pipeline bind of the same layout carries only the writes that changed, the
    /// bind disturbing no set 0; where the driver's pushed descriptors go stale across the bind (a
    /// driver's defect), it carries at least one: set 0's first, as held, where none changed.
    #[test]
    fn a_push_after_a_pipeline_bind_carries_what_changed_and_one_where_asked() {
        let layout = vk::PipelineLayout::from_raw(1);
        let writes = [uniform(0, 7), uniform(1, 8)];
        let changed = [uniform(0, 7), uniform(1, 9)];
        for forgets in [false, true] {
            let mut bound = Bound::default();
            bound.follow(vk::CommandBuffer::from_raw(2));
            assert!(bound.pipeline(vk::Pipeline::from_raw(3), &[], forgets));
            assert_eq!(bound.pushed_set(layout, 5, &writes).0.len(), 2);
            assert!(bound.pipeline(vk::Pipeline::from_raw(4), &[], forgets));
            let carried = bound.pushed_set(layout, 5, &changed).0;
            let only = |write: &SetWrite| carried.len() == 1 && carried[0].same(write);
            assert!(only(&changed[1]), "forgets {forgets}");
            // Another pipeline, the same writes.
            assert!(bound.pipeline(vk::Pipeline::from_raw(3), &[], forgets));
            let carried = bound.pushed_set(layout, 5, &changed).0;
            match forgets {
                true => assert!(carried.len() == 1 && carried[0].same(&changed[0])),
                false => assert!(carried.is_empty()),
            }
            // The same pipeline again binds nothing, and the push carries nothing.
            assert!(!bound.pipeline(vk::Pipeline::from_raw(3), &[], forgets));
            assert_eq!(bound.pushed_set(layout, 5, &changed).0.len(), 0);
        }
    }

    /// A push of another layout of the same shape carries set 0 whole, and counts the writes
    /// it would carry if the two were one layout: those that changed, at least one where the
    /// driver's pushed descriptors go stale across a pipeline bind; one of another shape, or of
    /// the same layout, counts nothing; what each push carries is unchanged.
    #[test]
    fn a_push_across_equal_shapes_counts_what_one_layout_would_carry() {
        let (a, b) = (
            vk::PipelineLayout::from_raw(1),
            vk::PipelineLayout::from_raw(2),
        );
        let writes = [uniform(0, 7), uniform(1, 8)];
        let changed = [uniform(0, 7), uniform(1, 9)];
        for forgets in [false, true] {
            let mut bound = Bound::default();
            bound.follow(vk::CommandBuffer::from_raw(3));
            bound.pipeline(vk::Pipeline::from_raw(4), &[], forgets);
            let (carried, counted) = bound.pushed_set(a, 5, &writes);
            assert_eq!((carried.len(), counted), (2, None), "nothing pushed yet");
            // Another layout of the same shape, the same writes.
            bound.pipeline(vk::Pipeline::from_raw(5), &[], forgets);
            let none_changed = u64::from(forgets);
            let (carried, counted) = bound.pushed_set(b, 5, &writes);
            assert_eq!(
                (carried.len(), counted),
                (2, Some(none_changed)),
                "carried whole"
            );
            // Back to the first layout, one write changed.
            bound.pipeline(vk::Pipeline::from_raw(4), &[], forgets);
            let (carried, counted) = bound.pushed_set(a, 5, &changed);
            assert_eq!((carried.len(), counted), (2, Some(1)));
            // A layout of another shape.
            bound.pipeline(vk::Pipeline::from_raw(6), &[], forgets);
            let (carried, counted) = bound.pushed_set(b, 6, &changed);
            assert_eq!((carried.len(), counted), (2, None));
            // The same pipeline and layout again: held, nothing counted.
            let (carried, counted) = bound.pushed_set(b, 6, &changed);
            assert_eq!((carried.len(), counted), (0, None));
            // The same layout through another pipeline: held, nothing counted, one write carried
            // where the bind leaves the driver's pushed descriptors stale.
            bound.pipeline(vk::Pipeline::from_raw(7), &[], forgets);
            let (carried, counted) = bound.pushed_set(b, 6, &changed);
            let carried = carried.len();
            assert_eq!(
                (carried, counted),
                (usize::from(forgets), None),
                "forgets {forgets}"
            );
        }
    }
}
