//! Depth/stencil states: a descriptor's depth and stencil tests in Vulkan's terms, which a render
//! encoder sets as dynamic state before its draws.

use crate::device::{Core, Device};
use ash::vk;
use metal2vulkan::metal::state::{depth_stencil_state, DepthStencilDescriptor, StencilOpState};
use std::sync::Arc;

/// `MTLDepthStencilState`.
///
/// A cheap handle: clones share one state. It holds no Vulkan object: a draw under it sets its
/// tests as dynamic state.
#[derive(Clone)]
pub struct DepthStencilState {
    inner: Arc<Inner>,
}

struct Inner {
    device: Device,
    descriptor: DepthStencilDescriptor,
    tests: Tests,
}

/// A depth/stencil descriptor's tests as Vulkan's dynamic state takes them.
#[derive(Clone, Copy)]
pub(crate) struct Tests {
    /// Whether the depth test compares or writes (`depth_stencil_state`).
    pub(crate) depth: bool,
    /// Whether the stencil test can fail or change the stencil.
    pub(crate) stencil: bool,
    /// The front face's and the back face's stencil operations, compare operation and masks.
    pub(crate) faces: [Face; 2],
}

/// One face's stencil state.
#[derive(Clone, Copy)]
pub(crate) struct Face {
    pub(crate) fail: vk::StencilOp,
    pub(crate) pass: vk::StencilOp,
    pub(crate) depth_fail: vk::StencilOp,
    pub(crate) compare: vk::CompareOp,
    pub(crate) compare_mask: u32,
    pub(crate) write_mask: u32,
}

impl Tests {
    /// The tests of `descriptor`, by the translator's mapping.
    pub(crate) fn of(descriptor: &DepthStencilDescriptor) -> Tests {
        let state = depth_stencil_state(descriptor);
        let face = |s: &StencilOpState| Face {
            fail: vk::StencilOp::from_raw(s.fail_op.as_raw()),
            pass: vk::StencilOp::from_raw(s.pass_op.as_raw()),
            depth_fail: vk::StencilOp::from_raw(s.depth_fail_op.as_raw()),
            compare: vk::CompareOp::from_raw(s.compare_op.as_raw()),
            compare_mask: s.compare_mask,
            write_mask: s.write_mask,
        };
        Tests {
            depth: state.depth_test_enable || state.depth_write_enable,
            stencil: state.stencil_test_enable,
            faces: [face(&state.front), face(&state.back)],
        }
    }
}

impl Device {
    /// `newDepthStencilStateWithDescriptor:`.
    pub fn new_depth_stencil_state(
        &self,
        descriptor: &DepthStencilDescriptor,
    ) -> DepthStencilState {
        DepthStencilState {
            inner: Arc::new(Inner {
                device: self.clone(),
                descriptor: *descriptor,
                tests: Tests::of(descriptor),
            }),
        }
    }
}

impl DepthStencilState {
    /// The descriptor it was made with.
    pub fn descriptor(&self) -> &DepthStencilDescriptor {
        &self.inner.descriptor
    }

    /// Its tests.
    pub(crate) fn tests(&self) -> Tests {
        self.inner.tests
    }

    /// Whether `core` is the device this state was made on.
    pub(crate) fn is_of(&self, core: &Arc<Core>) -> bool {
        Arc::ptr_eq(self.inner.device.core(), core)
    }
}
