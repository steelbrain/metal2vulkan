//! `DepthStencilState` (`MTLDepthStencilState`) and `Device::new_depth_stencil_state`
//! (step 6's second part).
//!
//! A depth/stencil state is its descriptor (the compile layer's [`DepthStencilDescriptor`], Metal's
//! properties) and the Vulkan depth/stencil state it stands for
//! ([`metal2vulkan::metal::state::depth_stencil_state`]): a render encoder's draws take it as the
//! pipeline's static depth/stencil state, or as dynamic state where the pipeline makes it dynamic
//! (`render.rs`). Metal makes a state of any descriptor, masks wider than the 8-bit stencil
//! included (measured on an M4), so creation refuses nothing.
//!
//! A state's `gpuResourceID` below Apple10 is the release's answer, not a refusal: Metal's API
//! validation stops the call ("Obtaining the resourceID of a DepthStencilState object requires
//! support for MTLGPUFamilyApple10", an M4 under macOS 27.0.1), but plain Metal is what the
//! runtime answers as, and the two differ. The plain call of the release every device answers as
//! answers 0 for every state asked below Apple10 (our probe on an M3 Max, and an earlier run of
//! three states over two runs; `metal_release.rs`). What the ID names on Apple10, a depth/stencil
//! state an indirect render command sets, is not built.

use super::error::{Error, RuntimeGap};
use super::families::GpuFamily;
use super::metal_release::DEPTH_STENCIL_RESOURCE_ID;
use super::resource_ids::ResourceId;
use super::Device;
use metal2vulkan::metal::state::{depth_stencil_state, DepthStencilDescriptor, StencilDescriptor};
use metal2vulkan::metal::{CompareFunction, StencilOperation};
use std::sync::Arc;

/// `MTLDepthStencilState`: cheap to clone, `Send` and `Sync`.
#[derive(Clone)]
pub struct DepthStencilState(Arc<Shared>);

struct Shared {
    device: Device,
    descriptor: DepthStencilDescriptor,
    vulkan: metal2vulkan::metal::state::DepthStencilState,
}

impl std::fmt::Debug for DepthStencilState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DepthStencilState")
            .field("descriptor", &self.0.descriptor)
            .finish()
    }
}

impl PartialEq for DepthStencilState {
    /// The same state object (Metal's identity), not an equal descriptor.
    fn eq(&self, other: &DepthStencilState) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl Device {
    /// `newDepthStencilStateWithDescriptor:`: a state of `descriptor` (the module docs).
    pub fn new_depth_stencil_state(
        &self,
        descriptor: &DepthStencilDescriptor,
    ) -> Result<DepthStencilState, Error> {
        self.current().new_depth_stencil_state_now(descriptor)
    }

    /// [`Device::new_depth_stencil_state`] on the generation that serves (`device/line.rs`).
    pub(crate) fn new_depth_stencil_state_now(
        &self,
        descriptor: &DepthStencilDescriptor,
    ) -> Result<DepthStencilState, Error> {
        Ok(DepthStencilState(Arc::new(Shared {
            device: self.clone(),
            descriptor: *descriptor,
            vulkan: depth_stencil_state(descriptor),
        })))
    }
}

impl DepthStencilState {
    /// `device`: the device that made the state.
    pub fn device(&self) -> &Device {
        &self.0.device
    }

    /// The descriptor the state was made with. No Metal counterpart: Metal's state keeps none.
    pub fn descriptor(&self) -> &DepthStencilDescriptor {
        &self.0.descriptor
    }

    /// `gpuResourceID` (the module docs): on a device that does not claim Apple10
    /// ([`Device::supports_family`]), what plain Metal answers there, 0. Refused
    /// `NotImplemented(IndirectRenderCommands)` on a device that claims Apple10: indirect render
    /// commands that set a depth/stencil state are not built.
    pub fn gpu_resource_id(&self) -> Result<ResourceId, Error> {
        if self.0.device.supports_family_now(GpuFamily::Apple10.raw()) {
            return Err(Error::NotImplemented(RuntimeGap::IndirectRenderCommands));
        }
        Ok(DEPTH_STENCIL_RESOURCE_ID)
    }

    /// The Vulkan depth/stencil state the state stands for.
    pub(super) fn vulkan(&self) -> metal2vulkan::metal::state::DepthStencilState {
        self.0.vulkan
    }

    /// What the state does with depth in a pass without a depth attachment (measured on an M4):
    /// Metal's validation stops a draw that writes depth or compares other than `Always` and
    /// `Never`, which plain Metal tests against the pass's hidden depth. Under `Never` every
    /// fragment fails, where Vulkan, with no attachment to test, would pass it
    /// ([`DepthStencilState::failing_depth`]).
    pub(super) fn depth_without_attachment(&self) -> DepthUse {
        let descriptor = &self.0.descriptor;
        match descriptor.depth_compare_function {
            _ if descriptor.depth_write_enabled => DepthUse::Stops,
            CompareFunction::Always => DepthUse::None,
            CompareFunction::Never => DepthUse::FailsAll,
            _ => DepthUse::Stops,
        }
    }

    /// The descriptor that stands for the state where every fragment fails Metal's depth test
    /// ([`DepthUse::FailsAll`]) and the fragment function runs (probes): no depth test, and each
    /// face's pass operation its depth-failure operation, since a fragment that passes the stencil
    /// test then fails the depth test. A fragment that fails the stencil test takes its failure
    /// operation, as before, and one the function drops takes none.
    pub(super) fn failing_depth(&self) -> DepthStencilDescriptor {
        let mut descriptor = self.0.descriptor;
        descriptor.depth_compare_function = CompareFunction::Always;
        descriptor.depth_write_enabled = false;
        for face in [
            &mut descriptor.front_face_stencil,
            &mut descriptor.back_face_stencil,
        ] {
            face.depth_stencil_pass_operation = face.depth_failure_operation;
        }
        descriptor
    }

    /// How a draw under the state runs where every fragment fails Metal's depth test
    /// ([`FailingDraw`]), in a pass with (or without) a stencil attachment (`stencil`), of a
    /// pipeline whose fragment function declares `[[early_fragment_tests]]` (`early`), writes
    /// attachments or an imageblock in its body (`body_writes`: color attachments as storage
    /// images, or the pass's imageblock, which no write mask stops), or writes memory
    /// (`writes_memory`: `GraphicsState::fragment_writes_memory`). Under early tests, where
    /// the stencil operation depends on the stored value ([`DepthStencilState::failing_early`]),
    /// a function that writes neither runs as one without early tests does
    /// ([`FailingDraw::Runs`]): Vulkan's early tests take each fragment's operation before the
    /// function as Metal's do, and the run, its colors masked, leaves nothing Metal's never-run
    /// function would not. Refused `NotImplemented(DepthNeverWithoutAttachment)` for a function
    /// that runs and writes in its body, and for one under early tests that writes there or
    /// writes memory whose stencil operation depends on the stored value.
    pub(super) fn failing_draw(
        &self,
        early: bool,
        stencil: bool,
        (body_writes, writes_memory): (bool, bool),
    ) -> Result<FailingDraw, Error> {
        let refused = Error::NotImplemented(RuntimeGap::DepthNeverWithoutAttachment);
        match early {
            true => match self.failing_early(stencil) {
                Some(draw) => Ok(draw),
                None if !body_writes && !writes_memory => {
                    Ok(FailingDraw::Runs(self.failing_depth()))
                }
                None => Err(refused),
            },
            false if body_writes => Err(refused),
            false => Ok(FailingDraw::Runs(self.failing_depth())),
        }
    }

    /// How a draw under the state runs where every fragment fails Metal's depth test and the
    /// fragment function declares `[[early_fragment_tests]]`, which Metal then never runs
    /// (probes), in a pass with (or without) a stencil attachment (`stencil`): each fragment still
    /// takes the face's depth-failure operation where it passes the stencil test and its failure
    /// operation where it fails it. `None` where that depends on the stored value (a comparison
    /// other than `Always` and `Never` whose two operations differ and write), which no Vulkan test
    /// that every fragment fails gives.
    fn failing_early(&self, stencil: bool) -> Option<FailingDraw> {
        let descriptor = &self.0.descriptor;
        if !stencil {
            return Some(FailingDraw::Untouched);
        }
        let front = early_operation(&descriptor.front_face_stencil)?;
        let back = early_operation(&descriptor.back_face_stencil)?;
        if front == StencilOperation::Keep && back == StencilOperation::Keep {
            return Some(FailingDraw::Untouched);
        }
        let mut failing = *descriptor;
        failing.depth_compare_function = CompareFunction::Always;
        failing.depth_write_enabled = false;
        for (face, operation) in [
            (&mut failing.front_face_stencil, front),
            (&mut failing.back_face_stencil, back),
        ] {
            face.stencil_compare_function = CompareFunction::Never;
            face.stencil_failure_operation = operation;
            face.depth_failure_operation = StencilOperation::Keep;
            face.depth_stencil_pass_operation = StencilOperation::Keep;
        }
        Some(FailingDraw::Fails(failing))
    }

    /// Whether the state uses the stencil on either face: a draw under it needs a stencil
    /// attachment (measured: Metal's validation stops such a draw in a pass without one, which
    /// plain Metal tests against the pass's hidden stencil).
    pub(super) fn uses_stencil(&self) -> bool {
        let descriptor = &self.0.descriptor;
        uses(&descriptor.front_face_stencil) || uses(&descriptor.back_face_stencil)
    }
}

/// What a depth/stencil state does with depth in a pass without a depth attachment.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum DepthUse {
    /// Nothing: the default compare, `Always`, without writes.
    None,
    /// Metal's validation stops the draw; plain Metal tests it against the pass's hidden depth.
    Stops,
    /// Every fragment fails (`Never`): Metal writes no attachment, takes each face's
    /// depth-failure operation where the stencil test passes, and counts no sample; it runs the
    /// fragment function unless the function declares `[[early_fragment_tests]]`.
    FailsAll,
}

/// How a draw runs where Metal's depth test fails every fragment, a compare of `Never` in a
/// pass without a depth attachment (probes, the M3 Max and an M4 alike): Metal writes no
/// attachment, counts no sample, and runs the vertex function; each fragment takes the face's
/// depth-failure operation where it passes the stencil test and its failure operation where it
/// fails it, and one the fragment function drops takes none.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum FailingDraw {
    /// The fragment function runs for every fragment (it does not declare
    /// `[[early_fragment_tests]]`, or it does and writes nothing): under this descriptor's state
    /// ([`DepthStencilState::failing_depth`]), every color write mask empty, no visibility query
    /// counting it.
    Runs(DepthStencilDescriptor),
    /// The function declares early tests and runs for no fragment: under this descriptor's
    /// state, whose stencil test every fragment fails, each taking the operation Metal takes
    /// before the function.
    Fails(DepthStencilDescriptor),
    /// The function declares early tests and no fragment changes the stencil: the draw reaches
    /// no fragment (its scissors are empty).
    Untouched,
}

/// The one stencil operation every fragment of a face takes where every fragment fails Metal's
/// depth test before the function runs: the depth-failure operation where the face's stencil
/// test passes, the failure operation where it fails, `Keep` where the write mask leaves the
/// 8-bit stencil alone. `None` where the two can both apply and differ.
fn early_operation(face: &StencilDescriptor) -> Option<StencilOperation> {
    let (fail, depth_fail) = (face.stencil_failure_operation, face.depth_failure_operation);
    match face.stencil_compare_function {
        _ if face.write_mask & 0xff == 0 => Some(StencilOperation::Keep),
        CompareFunction::Never => Some(fail),
        CompareFunction::Always => Some(depth_fail),
        _ => (fail == depth_fail).then_some(fail),
    }
}

/// Whether a face's stencil descriptor tests or writes (measured: Metal's validation stops it
/// without a stencil attachment when it compares other than `Always`, or its depth-failure or
/// pass operation is other than `Keep`; masks, and a failure operation under `Always`, which
/// never fails, are free).
fn uses(face: &StencilDescriptor) -> bool {
    face.stencil_compare_function != CompareFunction::Always
        || face.depth_failure_operation != StencilOperation::Keep
        || face.depth_stencil_pass_operation != StencilOperation::Keep
}
