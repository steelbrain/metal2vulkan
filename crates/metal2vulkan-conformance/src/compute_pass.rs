//! What a case asks an `MTLComputeCommandEncoder` to do: one pass, its commands stated in order.
//!
//! [`Side::dispatch`](crate::case::Side::dispatch) and
//! [`Side::dispatch_threads`](crate::case::Side::dispatch_threads) are a **fixed** compute pass:
//! the pipeline, the bindings at 0 up per kind, one dispatch, commit. Four of
//! `MTLComputeCommandEncoder`'s thirty-six items are reachable that way and the rest are not
//! reachable at all — not the inline bytes, not the plural setters, not the LOD-clamped sampler
//! bindings, not the dynamic threadgroup allocation, not the residency declarations, not the
//! barriers, not the threadgroup-grid dispatch and not the indirect one.
//!
//! So a case states the pass's commands the way [`crate::render::RenderStep`] states a render
//! pass's: a list, in order, each naming the Metal call it makes. Three things follow from the
//! shape rather than from care:
//!
//! - **Every command is encoded, whatever an earlier one answered**, as the render and mesh passes
//!   do, and the answer says which command refused ([`ComputePassOutcome::refused`], one entry per
//!   step). A refusal of ours then compares against Metal's answer at that step and reads as a
//!   refusal rather than as a shape mismatch.
//! - **One command buffer, one encoder, one commit.** Several dispatches of a case go in the same
//!   encoder, so a case that chains one dispatch's output into the next reads what *these*
//!   dispatches wrote. A committed command buffer between the halves would give each half its own
//!   run, and a second half that read the first run's leftovers would round-trip bit-exactly with
//!   the first half doing nothing at all.
//! - **Residency is the case's**, never the side's. A side here makes nothing resident on its own:
//!   a case that reaches a resource other than through a bound slot says
//!   [`ComputeStep::UseResource`] or [`ComputeStep::UseResources`], which is both what Metal
//!   requires and how those two items get exercised.
//!
//! Nothing here reads a device name, a vendor or an environment variable: a venue that refuses one
//! of these commands refuses it typed, at its own step.

use crate::case::{Failure, Handle};
use crate::transcript::{Node, NodeName, Observed, Value};

/// How a pass's encoder is opened: `computeCommandEncoder`, or
/// `computeCommandEncoderWithDispatchType:` where a case names the type.
///
/// It is part of the pass rather than a step, because Metal decides it as the encoder is made and
/// a pass cannot change it afterwards. `dispatchType` is then a getter both sides answer, so what
/// a case states here is also something it observes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dispatching {
    /// `MTLDispatchTypeSerial`: `computeCommandEncoder`, whose dispatches run in the order
    /// they were encoded.
    Serial,
    /// `MTLDispatchTypeConcurrent`: `computeCommandEncoderWithDispatchType:`, where the
    /// dispatches may overlap and a case that chains one into the next **must** put a
    /// barrier between them.
    Concurrent,
}

impl Dispatching {
    /// The raw `MTLDispatchType`.
    pub const fn as_raw(self) -> u64 {
        match self {
            Dispatching::Serial => 0,
            Dispatching::Concurrent => 1,
        }
    }

    /// The call the encoder is made by, for the site a refusal is recorded at.
    pub const fn call(self) -> &'static str {
        match self {
            Dispatching::Serial => "computeCommandEncoder",
            Dispatching::Concurrent => "computeCommandEncoderWithDispatchType",
        }
    }
}

/// A resource a residency declaration or a resource barrier names: a buffer or a texture, as
/// Metal's own `id<MTLResource>` is either.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PassResource {
    Buffer(Handle),
    Texture(Handle),
}

/// `MTLResourceUsageRead`.
pub const USAGE_READ: u64 = 1 << 0;

/// `MTLResourceUsageWrite`.
pub const USAGE_WRITE: u64 = 1 << 1;

/// `MTLBarrierScopeBuffers`.
pub const SCOPE_BUFFERS: u64 = 1 << 0;

/// `MTLAttributeStrideStatic`: the `attributeStride` argument that sets **no** per-bind stride, so
/// the index is read as the plain setters leave it.
///
/// It is one of the four values the stride setters are documented to treat apart — 0, a stride of
/// its own, this one, and a stride past 32 bits, which Metal takes modulo `2^32` — and it is the
/// only one of them that is a named constant rather than a number.
pub const ATTRIBUTE_STRIDE_STATIC: u64 = u64::MAX;

/// One command a case encodes into a compute pass, in the order it lists them.
#[derive(Clone, Debug, PartialEq)]
pub enum ComputeStep {
    /// `setComputePipelineState:`.
    Pipeline(Handle),
    /// `setBuffer:offset:atIndex:`.
    Buffer {
        buffer: Handle,
        offset: u64,
        index: u32,
    },
    /// `setBufferOffset:atIndex:`: a new offset for the buffer already bound at that index, which
    /// is only legal once one is.
    BufferOffset { offset: u64, index: u32 },
    /// `setBuffers:offsets:withRange:`: several buffers and their offsets in one call, from
    /// `first`.
    Buffers {
        buffers: Vec<(Handle, u64)>,
        first: u32,
    },
    /// `setBytes:length:atIndex:`: a binding whose bytes the call copies, with no buffer of the
    /// case's own behind it.
    Bytes { bytes: Vec<u8>, index: u32 },
    /// `setBuffer:offset:attributeStride:atIndex:`: the buffer bound with the per-bind stride a
    /// `[[stage_in]]` layout of `MTLBufferLayoutStrideDynamic` reads its elements at.
    ///
    /// Only such a layout reads the stride at all, so a case over this binds a pipeline whose
    /// stage-input descriptor states one; against a layout of a fixed stride the four stride
    /// setters are the plain setters with one more argument that reaches nothing.
    BufferWithStride {
        buffer: Handle,
        offset: u64,
        stride: u64,
        index: u32,
    },
    /// `setBuffers:offsets:attributeStrides:withRange:`: the plural form of
    /// [`ComputeStep::BufferWithStride`], each buffer with its own offset and stride.
    BuffersWithStrides {
        buffers: Vec<(Handle, u64, u64)>,
        first: u32,
    },
    /// `setBufferOffset:attributeStride:atIndex:`: a new offset **and** a new stride for the
    /// buffer already bound at that index, which is only legal once one is.
    ///
    /// [`ATTRIBUTE_STRIDE_STATIC`] here *clears* the index's stride rather than leaving it, which
    /// is the one place a case can tell that value from a stride of 0 by what it replaced.
    BufferOffsetWithStride {
        offset: u64,
        stride: u64,
        index: u32,
    },
    /// `setBytes:length:attributeStride:atIndex:`: [`ComputeStep::Bytes`] with the per-bind stride
    /// a dynamic-stride layout reads the copied bytes at.
    BytesWithStride {
        bytes: Vec<u8>,
        stride: u64,
        index: u32,
    },
    /// `setStageInRegion:`: the region a `[[stage_in]]` fetch's elements are taken from and
    /// clamped to, which Metal requires of every dispatch whose kernel has a stage input.
    StageInRegion { origin: [u32; 3], size: [u32; 3] },
    /// `setTexture:atIndex:`.
    Texture { texture: Handle, index: u32 },
    /// `setTextures:withRange:`: several textures in one call, from `first`.
    Textures { textures: Vec<Handle>, first: u32 },
    /// `setSamplerState:atIndex:`.
    Sampler { sampler: Handle, index: u32 },
    /// `setSamplerStates:withRange:`: several sampler states in one call, from `first`.
    Samplers { samplers: Vec<Handle>, first: u32 },
    /// `setSamplerState:lodMinClamp:lodMaxClamp:atIndex:`: the state with the level range this
    /// binding clamps a sample to.
    ClampedSampler {
        sampler: Handle,
        lod_min: f32,
        lod_max: f32,
        index: u32,
    },
    /// `setSamplerStates:lodMinClamps:lodMaxClamps:withRange:`: the plural form of
    /// [`ComputeStep::ClampedSampler`], each state with its own clamps.
    ClampedSamplers {
        samplers: Vec<(Handle, f32, f32)>,
        first: u32,
    },
    /// `setThreadgroupMemoryLength:atIndex:`: the bytes the kernel's `[[threadgroup(n)]]`
    /// allocation takes, which the kernel itself does not declare.
    ThreadgroupMemory { length: u32, index: u32 },
    /// `useResource:usage:`.
    UseResource { resource: PassResource, usage: u64 },
    /// `useResources:count:usage:`.
    UseResources {
        resources: Vec<PassResource>,
        usage: u64,
    },
    /// `useHeap:`: every resource of the heap is resident for this encoder's dispatches.
    ///
    /// A bound slot is resident without being declared, so this changes nothing a case reading its
    /// own bindings can observe; what a case over it holds is that a dispatch through a
    /// heap-placed buffer answers Metal's own words with the heap declared. The observable a
    /// declaration *would* have is a resource reached off-slot, which is undefined behaviour on a
    /// shared GPU and no case here probes.
    UseHeap { heap: Handle },
    /// `useHeaps:count:`: the plural form of [`ComputeStep::UseHeap`].
    UseHeaps { heaps: Vec<Handle> },
    /// `memoryBarrierWithScope:`: the earlier dispatches' writes to that scope are visible to the
    /// later ones.
    BarrierScope { scope: u64 },
    /// `memoryBarrierWithResources:count:`: the same for the resources named.
    BarrierResources { resources: Vec<PassResource> },
    /// `dispatchThreadgroups:threadsPerThreadgroup:`.
    DispatchThreadgroups { groups: [u32; 3], threads: [u32; 3] },
    /// `dispatchThreadgroupsWithIndirectBuffer:indirectBufferOffset:threadsPerThreadgroup:`,
    /// whose buffer holds the grid's three 32-bit words at `offset`.
    DispatchThreadgroupsIndirect {
        buffer: Handle,
        offset: u64,
        threads: [u32; 3],
    },
}

impl ComputeStep {
    /// The Metal call the step makes, which is the site its answer is emitted at, so a refusal
    /// reads as the call we refused.
    pub fn call(&self) -> &'static str {
        match self {
            ComputeStep::Pipeline(_) => "setComputePipelineState",
            ComputeStep::Buffer { .. } => "setBuffer",
            ComputeStep::BufferOffset { .. } => "setBufferOffset",
            ComputeStep::Buffers { .. } => "setBuffers",
            ComputeStep::Bytes { .. } => "setBytes",
            ComputeStep::BufferWithStride { .. } => "setBufferWithAttributeStride",
            ComputeStep::BuffersWithStrides { .. } => "setBuffersWithAttributeStrides",
            ComputeStep::BufferOffsetWithStride { .. } => "setBufferOffsetWithAttributeStride",
            ComputeStep::BytesWithStride { .. } => "setBytesWithAttributeStride",
            ComputeStep::StageInRegion { .. } => "setStageInRegion",
            ComputeStep::Texture { .. } => "setTexture",
            ComputeStep::Textures { .. } => "setTextures",
            ComputeStep::Sampler { .. } => "setSamplerState",
            ComputeStep::Samplers { .. } => "setSamplerStates",
            ComputeStep::ClampedSampler { .. } => "setSamplerStateWithLodClamps",
            ComputeStep::ClampedSamplers { .. } => "setSamplerStatesWithLodClamps",
            ComputeStep::ThreadgroupMemory { .. } => "setThreadgroupMemoryLength",
            ComputeStep::UseResource { .. } => "useResource",
            ComputeStep::UseResources { .. } => "useResources",
            ComputeStep::UseHeap { .. } => "useHeap",
            ComputeStep::UseHeaps { .. } => "useHeaps",
            ComputeStep::BarrierScope { .. } => "memoryBarrierWithScope",
            ComputeStep::BarrierResources { .. } => "memoryBarrierWithResources",
            ComputeStep::DispatchThreadgroups { .. } => "dispatchThreadgroups",
            ComputeStep::DispatchThreadgroupsIndirect { .. } => {
                "dispatchThreadgroupsWithIndirectBuffer"
            }
        }
    }

    /// The parameters the step's own call carries, for a case's input fingerprint: everything but
    /// the handles, whose numbers are a side's own.
    ///
    /// A buffer, a texture or a sampler a step names is fingerprinted by the case through its own
    /// bytes, texels or descriptor words, so only the call's numbers are here. Inline bytes are
    /// the case's own value rather than a handle, so they are covered whole.
    pub fn parameters(&self) -> Vec<u8> {
        let mut words = self.call().as_bytes().to_vec();
        match self {
            ComputeStep::Pipeline(_) => {}
            ComputeStep::Buffer { offset, index, .. } => {
                words.extend(offset.to_le_bytes());
                words.extend(index.to_le_bytes());
            }
            ComputeStep::BufferOffset { offset, index } => {
                words.extend(offset.to_le_bytes());
                words.extend(index.to_le_bytes());
            }
            ComputeStep::Buffers { buffers, first } => {
                words.extend(first.to_le_bytes());
                for (_, offset) in buffers {
                    words.extend(offset.to_le_bytes());
                }
            }
            ComputeStep::Bytes { bytes, index } => {
                words.extend(bytes);
                words.extend(index.to_le_bytes());
            }
            ComputeStep::BufferWithStride {
                offset,
                stride,
                index,
                ..
            } => {
                words.extend(offset.to_le_bytes());
                words.extend(stride.to_le_bytes());
                words.extend(index.to_le_bytes());
            }
            ComputeStep::BuffersWithStrides { buffers, first } => {
                words.extend(first.to_le_bytes());
                for (_, offset, stride) in buffers {
                    words.extend(offset.to_le_bytes());
                    words.extend(stride.to_le_bytes());
                }
            }
            ComputeStep::BufferOffsetWithStride {
                offset,
                stride,
                index,
            } => {
                words.extend(offset.to_le_bytes());
                words.extend(stride.to_le_bytes());
                words.extend(index.to_le_bytes());
            }
            ComputeStep::BytesWithStride {
                bytes,
                stride,
                index,
            } => {
                words.extend(bytes);
                words.extend(stride.to_le_bytes());
                words.extend(index.to_le_bytes());
            }
            ComputeStep::StageInRegion { origin, size } => {
                for axis in origin.iter().chain(size) {
                    words.extend(axis.to_le_bytes());
                }
            }
            ComputeStep::Texture { index, .. } | ComputeStep::Sampler { index, .. } => {
                words.extend(index.to_le_bytes());
            }
            ComputeStep::Textures { textures, first } => {
                words.extend(first.to_le_bytes());
                words.extend((textures.len() as u32).to_le_bytes());
            }
            ComputeStep::Samplers { samplers, first } => {
                words.extend(first.to_le_bytes());
                words.extend((samplers.len() as u32).to_le_bytes());
            }
            ComputeStep::ClampedSampler {
                lod_min,
                lod_max,
                index,
                ..
            } => {
                words.extend(lod_min.to_le_bytes());
                words.extend(lod_max.to_le_bytes());
                words.extend(index.to_le_bytes());
            }
            ComputeStep::ClampedSamplers { samplers, first } => {
                words.extend(first.to_le_bytes());
                for (_, lod_min, lod_max) in samplers {
                    words.extend(lod_min.to_le_bytes());
                    words.extend(lod_max.to_le_bytes());
                }
            }
            ComputeStep::ThreadgroupMemory { length, index } => {
                words.extend(length.to_le_bytes());
                words.extend(index.to_le_bytes());
            }
            ComputeStep::UseResource { usage, .. } => words.extend(usage.to_le_bytes()),
            ComputeStep::UseResources { resources, usage } => {
                words.extend(usage.to_le_bytes());
                words.extend((resources.len() as u32).to_le_bytes());
            }
            // A heap a step names is fingerprinted by the case through its own descriptor, so
            // only the count of them is here, as for the plural resource declaration.
            ComputeStep::UseHeap { .. } => {}
            ComputeStep::UseHeaps { heaps } => {
                words.extend((heaps.len() as u32).to_le_bytes());
            }
            ComputeStep::BarrierScope { scope } => words.extend(scope.to_le_bytes()),
            ComputeStep::BarrierResources { resources } => {
                words.extend((resources.len() as u32).to_le_bytes());
            }
            ComputeStep::DispatchThreadgroups { groups, threads } => {
                for axis in groups.iter().chain(threads) {
                    words.extend(axis.to_le_bytes());
                }
            }
            ComputeStep::DispatchThreadgroupsIndirect {
                offset, threads, ..
            } => {
                words.extend(offset.to_le_bytes());
                for axis in threads {
                    words.extend(axis.to_le_bytes());
                }
            }
        }
        words
    }

    /// Whether the step is a dispatch, which is what a guard that bounds a case's grids reads.
    pub fn dispatches(&self) -> bool {
        matches!(
            self,
            ComputeStep::DispatchThreadgroups { .. }
                | ComputeStep::DispatchThreadgroupsIndirect { .. }
        )
    }
}

/// One compute pass a case runs: how its encoder is opened, and the commands it holds.
#[derive(Clone, Debug, PartialEq)]
pub struct ComputePass<'a> {
    pub dispatching: Dispatching,
    pub steps: &'a [ComputeStep],
}

impl ComputePass<'_> {
    /// The words a case's input fingerprint covers: the dispatch type, then every step's own
    /// parameters, so a pass whose commands or whose order moved is a change to the case's inputs.
    pub fn words(&self) -> Vec<u8> {
        let mut words = self.dispatching.as_raw().to_le_bytes().to_vec();
        for step in self.steps {
            words.extend(step.parameters());
        }
        words
    }
}

/// What one compute pass came to.
#[derive(Clone, Debug, PartialEq)]
pub struct ComputePassOutcome {
    /// What each step answered, in the order the case gave them: `None` where the call answered,
    /// the typed error where it did not. A side whose encoder methods return nothing — Metal's do
    /// — answers `None` throughout, so a refusal of ours compares against Metal's answer and reads
    /// as a refusal rather than as a divergence.
    pub refused: Vec<Option<Failure>>,
    /// `dispatchType` of the encoder, as the variant of its name: what Metal says the encoder it
    /// made is, which a case states and then observes.
    pub dispatch_type: Value,
    /// The command buffer's status after the bounded wait, as the variant of its name.
    pub status: Value,
    /// The command buffer's error after it completed, where it reported one.
    pub error: Option<Value>,
}

impl ComputePassOutcome {
    /// The outcome as the sites a case emits under one node, so both sides emit one shape.
    ///
    /// Each step's site is named by its **position and its call**, so two steps that make the same
    /// call are two sites rather than one name twice: siblings sharing a name is how a comparison
    /// once reported six cases divergent having compared no bytes of either side.
    pub fn node(&self, name: impl Into<NodeName>, steps: &[ComputeStep]) -> Node {
        let mut node = Node::branch(name, Vec::new())
            .with(Node::leaf(
                "dispatch_type",
                Observed::value(self.dispatch_type.clone()),
            ))
            .with(Node::leaf("status", Observed::value(self.status.clone())))
            .with(Node::leaf(
                "error",
                Observed::value(match &self.error {
                    Some(error) => Value::Variant {
                        name: "Some".to_owned(),
                        payload: Box::new(error.clone()),
                    },
                    None => Value::unit_variant("None"),
                }),
            ));
        for (at, (step, refused)) in steps.iter().zip(&self.refused).enumerate() {
            node = node.with(Node::leaf(
                format!("step/{at}/{}", step.call()),
                Observed::value(match refused {
                    None => Value::ok(Value::Absent),
                    Some(failure) => failure.recorded(),
                }),
            ));
        }
        node
    }

    /// The first step the pass refused, with its position, where one did.
    pub fn first_refusal(&self) -> Option<(usize, &Failure)> {
        self.refused
            .iter()
            .enumerate()
            .find_map(|(at, refused)| refused.as_ref().map(|failure| (at, failure)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    /// A handle to name in a step. Which handle a step holds never reaches the fingerprint, so any
    /// one will do here.
    fn handle() -> Handle {
        Handle(0)
    }

    /// Every kind of one vocabulary, from `first` through the successor each kind names.
    ///
    /// This is where the exhaustiveness comes from: `after` is a `match` over every variant of its
    /// enum with **no wildcard arm**, so a variant added to that enum does not compile until it is
    /// given its own arm, and the arm of the kind it is placed after is what hands it to this
    /// walk. The guards then read the walk rather than an array written out beside the enum, which
    /// is the thing a new variant can be added without touching.
    ///
    /// It is [`crate::render`]'s mechanism: a hand-written list feeding three guards would let a
    /// variant added to [`ComputeStep`] escape all three in silence.
    ///
    /// The walk is checked to be a list and not a cycle, so an arm naming a kind already walked
    /// fails here, by name, instead of running forever.
    fn every_kind<T: std::fmt::Debug>(first: T, after: impl Fn(&T) -> Option<T>) -> Vec<T> {
        let mut kinds = vec![first];
        let mut walked = vec![std::mem::discriminant(&kinds[0])];
        while let Some(next) = after(kinds.last().expect("the walk starts at its first kind")) {
            assert!(
                !walked.contains(&std::mem::discriminant(&next)),
                "the walk of {} returns to {next:?}, so it is a cycle rather than each kind once",
                std::any::type_name::<T>()
            );
            walked.push(std::mem::discriminant(&next));
            kinds.push(next);
        }
        kinds
    }

    /// One step of every kind, in declaration order, through the compiler-kept walk.
    fn one_of_each() -> Vec<ComputeStep> {
        every_kind(ComputeStep::Pipeline(handle()), step_after)
    }

    /// The step that follows `step` in declaration order, or `None` after the last.
    ///
    /// One arm per variant and no wildcard: see [`every_kind`] for why that is the whole point.
    fn step_after(step: &ComputeStep) -> Option<ComputeStep> {
        Some(match step {
            ComputeStep::Pipeline(_) => ComputeStep::Buffer {
                buffer: handle(),
                offset: 0,
                index: 0,
            },
            ComputeStep::Buffer { .. } => ComputeStep::BufferOffset {
                offset: 4,
                index: 0,
            },
            ComputeStep::BufferOffset { .. } => ComputeStep::Buffers {
                buffers: vec![(handle(), 0)],
                first: 0,
            },
            ComputeStep::Buffers { .. } => ComputeStep::Bytes {
                bytes: vec![1, 2, 3],
                index: 1,
            },
            ComputeStep::Bytes { .. } => ComputeStep::BufferWithStride {
                buffer: handle(),
                offset: 0,
                stride: 4,
                index: 1,
            },
            ComputeStep::BufferWithStride { .. } => ComputeStep::BuffersWithStrides {
                buffers: vec![(handle(), 0, 8)],
                first: 1,
            },
            ComputeStep::BuffersWithStrides { .. } => ComputeStep::BufferOffsetWithStride {
                offset: 4,
                stride: ATTRIBUTE_STRIDE_STATIC,
                index: 1,
            },
            ComputeStep::BufferOffsetWithStride { .. } => ComputeStep::BytesWithStride {
                bytes: vec![4, 5, 6],
                stride: 1 << 32,
                index: 1,
            },
            ComputeStep::BytesWithStride { .. } => ComputeStep::StageInRegion {
                origin: [0, 0, 0],
                size: [8, 1, 1],
            },
            ComputeStep::StageInRegion { .. } => ComputeStep::Texture {
                texture: handle(),
                index: 0,
            },
            ComputeStep::Texture { .. } => ComputeStep::Textures {
                textures: vec![handle()],
                first: 0,
            },
            ComputeStep::Textures { .. } => ComputeStep::Sampler {
                sampler: handle(),
                index: 0,
            },
            ComputeStep::Sampler { .. } => ComputeStep::Samplers {
                samplers: vec![handle()],
                first: 0,
            },
            ComputeStep::Samplers { .. } => ComputeStep::ClampedSampler {
                sampler: handle(),
                lod_min: 0.0,
                lod_max: 1.0,
                index: 0,
            },
            ComputeStep::ClampedSampler { .. } => ComputeStep::ClampedSamplers {
                samplers: vec![(handle(), 0.0, 1.0)],
                first: 0,
            },
            ComputeStep::ClampedSamplers { .. } => ComputeStep::ThreadgroupMemory {
                length: 16,
                index: 0,
            },
            ComputeStep::ThreadgroupMemory { .. } => ComputeStep::UseResource {
                resource: PassResource::Buffer(handle()),
                usage: USAGE_READ,
            },
            ComputeStep::UseResource { .. } => ComputeStep::UseResources {
                resources: vec![PassResource::Texture(handle())],
                usage: USAGE_READ,
            },
            ComputeStep::UseResources { .. } => ComputeStep::UseHeap { heap: handle() },
            ComputeStep::UseHeap { .. } => ComputeStep::UseHeaps {
                heaps: vec![handle()],
            },
            ComputeStep::UseHeaps { .. } => ComputeStep::BarrierScope {
                scope: SCOPE_BUFFERS,
            },
            ComputeStep::BarrierScope { .. } => ComputeStep::BarrierResources {
                resources: vec![PassResource::Buffer(handle())],
            },
            ComputeStep::BarrierResources { .. } => ComputeStep::DispatchThreadgroups {
                groups: [1, 1, 1],
                threads: [8, 1, 1],
            },
            ComputeStep::DispatchThreadgroups { .. } => ComputeStep::DispatchThreadgroupsIndirect {
                buffer: handle(),
                offset: 0,
                threads: [8, 1, 1],
            },
            ComputeStep::DispatchThreadgroupsIndirect { .. } => return None,
        })
    }

    /// One pass of `steps`, the way a case states one.
    fn pass<'a>(dispatching: Dispatching, steps: &'a [ComputeStep]) -> ComputePass<'a> {
        ComputePass { dispatching, steps }
    }

    /// Every step names a Metal call no other step names, so a refusal reads as the call we
    /// refused and the fingerprint keys each step by its own name.
    ///
    /// It reads the compiler-kept walk, so a variant added to [`ComputeStep`] is in this check —
    /// and in the two below it — the moment it compiles.
    #[test]
    fn every_step_names_a_call_of_its_own() {
        let steps = one_of_each();
        let named: BTreeSet<&str> = steps.iter().map(ComputeStep::call).collect();
        assert_eq!(named.len(), steps.len(), "{named:?}");
    }

    /// Two steps that make the same call hold two sites, not one name twice.
    ///
    /// Siblings sharing a name make a comparison report cases divergent having compared no bytes
    /// of either side. A pass of four identical dispatches is exactly that shape, so it is the case
    /// stated here.
    #[test]
    fn repeated_steps_hold_sites_of_their_own() {
        let steps = vec![
            ComputeStep::DispatchThreadgroups {
                groups: [1, 1, 1],
                threads: [8, 1, 1],
            };
            4
        ];
        let outcome = ComputePassOutcome {
            refused: vec![None; steps.len()],
            dispatch_type: Value::unit_variant("Serial"),
            status: Value::unit_variant("Completed"),
            error: None,
        };
        let node = outcome.node("pass", &steps);
        let names: BTreeSet<String> = node
            .children
            .iter()
            .map(|child| child.name.to_string())
            .collect();
        assert_eq!(
            names.len(),
            node.children.len(),
            "two of the pass's sites share a name: {names:?}"
        );
        assert_eq!(node.children.len(), 3 + steps.len());
    }

    /// Two passes differing in one call's own numbers fingerprint apart, and the same pass twice
    /// fingerprints the same.
    ///
    /// Without the first half a recording would stay the answer to a pass whose offsets had moved.
    #[test]
    fn a_pass_fingerprints_by_its_calls() {
        let steps = one_of_each();
        assert_eq!(
            pass(Dispatching::Serial, &steps).words(),
            pass(Dispatching::Serial, &steps).words()
        );
        assert_ne!(
            pass(Dispatching::Serial, &steps).words(),
            pass(Dispatching::Concurrent, &steps).words(),
            "the dispatch type does not reach the fingerprint"
        );
        let mut moved = steps.clone();
        moved[2] = ComputeStep::BufferOffset {
            offset: 8,
            index: 0,
        };
        assert_ne!(
            pass(Dispatching::Serial, &steps).words(),
            pass(Dispatching::Serial, &moved).words(),
            "a buffer offset does not reach the fingerprint"
        );
        // The steps' order is part of the pass, so reversing them is a different program.
        let mut reversed = steps.clone();
        reversed.reverse();
        assert_ne!(
            pass(Dispatching::Serial, &steps).words(),
            pass(Dispatching::Serial, &reversed).words()
        );
    }

    /// Both dispatch steps say they dispatch and no other step does, so a guard that bounds a
    /// case's grids reads every dispatch and nothing else.
    #[test]
    fn only_the_dispatch_steps_dispatch() {
        let dispatching: Vec<&'static str> = one_of_each()
            .iter()
            .filter(|step| step.dispatches())
            .map(ComputeStep::call)
            .collect();
        assert_eq!(
            dispatching,
            [
                "dispatchThreadgroups",
                "dispatchThreadgroupsWithIndirectBuffer"
            ]
        );
    }
}
