//! Depth/stencil states: a descriptor's depth and stencil tests in Vulkan's terms, which a render
//! encoder sets as dynamic state before its draws.

use crate::device::{Core, Device};
use ash::vk;
use metal2vulkan::metal::state::{depth_stencil_state, DepthStencilDescriptor, StencilOpState};
use metal2vulkan::pipeline::PrecisionNote;
use std::sync::Arc;

/// `MTLDepthStencilState`.
///
/// A cheap handle: clones share one state, and a handle holds the device. It holds no Vulkan
/// object: a draw under it sets its tests as dynamic state.
#[derive(Clone)]
pub struct DepthStencilState {
    inner: Arc<Inner>,
    device: Device,
}

struct Inner {
    descriptor: DepthStencilDescriptor,
    tests: Tests,
}

/// A depth/stencil descriptor's tests as Vulkan's dynamic state takes them.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) struct Tests {
    /// Whether the depth test runs: it compares by other than `Always` or writes
    /// (`depth_stencil_state`).
    pub(crate) depth_test: bool,
    /// Whether a fragment that passes writes its depth.
    pub(crate) depth_write: bool,
    /// The depth compare operation.
    pub(crate) depth_compare: vk::CompareOp,
    /// Whether the stencil test can fail or change the stencil.
    pub(crate) stencil: bool,
    /// The front face's and the back face's stencil operations, compare operation and masks.
    pub(crate) faces: [Face; 2],
}

/// One face's stencil state.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) struct Face {
    pub(crate) fail: vk::StencilOp,
    pub(crate) pass: vk::StencilOp,
    pub(crate) depth_fail: vk::StencilOp,
    pub(crate) compare: vk::CompareOp,
    pub(crate) compare_mask: u32,
    pub(crate) write_mask: u32,
}

impl Tests {
    /// The depth test's part: whether it runs, whether it writes, and its compare operation.
    pub(crate) fn depth(&self) -> (bool, bool, vk::CompareOp) {
        (self.depth_test, self.depth_write, self.depth_compare)
    }

    /// The stencil test's part: whether it runs, and each face's state.
    pub(crate) fn stencil(&self) -> (bool, [Face; 2]) {
        (self.stencil, self.faces)
    }

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
            depth_test: state.depth_test_enable,
            depth_write: state.depth_write_enable,
            depth_compare: vk::CompareOp::from_raw(state.depth_compare_op.as_raw()),
            stencil: state.stencil_test_enable,
            faces: [face(&state.front), face(&state.back)],
        }
    }
}

/// The bit a draw through a pipeline whose fragment function's stencil output is drawn by bits
/// pushes to keep every fragment (`PushField::StencilBit` at 8 or more).
pub(crate) const EVERY_FRAGMENT: u32 = 8;

/// The most draws one draw takes: its first, one per stencil bit, and its depth's.
const MOST_DRAWS: usize = 10;

/// One of the draws a draw takes through a pipeline whose fragment function's stencil output is
/// drawn by bits (`RuntimeFact::StencilBitDraws`, on a device without
/// `VK_EXT_shader_stencil_export`): its tests, its stencil reference of each face, and the bit its
/// function keeps (`PushField::StencilBit`).
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) struct BitDraw {
    pub(crate) tests: Tests,
    pub(crate) reference: [u32; 2],
    pub(crate) bit: u32,
}

/// The draws one draw takes through such a pipeline, in order ([`BitDraws::of`]).
pub(crate) struct BitDraws {
    draws: [BitDraw; MOST_DRAWS],
    len: usize,
}

impl BitDraws {
    /// The draws of a draw under the encoder's `tests` and `reference`. Where the stencil test
    /// replaces no bit (it is off, no operation of a face is `Replace`, or those faces' write
    /// masks are 0), the draw alone, every fragment kept: its output is no reference anything
    /// reads. Otherwise, with `w` the low 8 bits of the write masks of the faces with a `Replace`
    /// operation:
    ///
    /// 1. every fragment kept, under the tests but the reference 0 on both faces, so `Replace`
    ///    zeroes the bits of `w` and every other operation runs once;
    /// 2. for each bit `k` of `w`, the fragments whose output holds bit `k`, under the reference
    ///    255, each face's write mask cut to bit `k`, its `Replace` operations kept and its others
    ///    `Keep`, and no depth write, so `Replace` sets bit `k` where the output holds it;
    /// 3. where the tests write depth under a compare other than `Always` and `Never`, the first
    ///    draw writes none, and a last one, every fragment kept, writes it under the tests with
    ///    stencil write masks of 0: each draw before it tests against the depth from before the
    ///    draw, as Metal's draw does.
    ///
    /// Each draw takes the same fragments through the same tests where the compare is `Always`
    /// or `Never` on both faces, which the reference does not change, so a sample one fragment
    /// reaches stores Metal's value.
    ///
    /// `None` for a stencil test that compares by another function on a face, whose answer the
    /// reference decides.
    pub(crate) fn of(tests: Tests, reference: [u32; 2]) -> Option<BitDraws> {
        let own = BitDraw {
            tests,
            reference,
            bit: EVERY_FRAGMENT,
        };
        let mut draws = BitDraws {
            draws: [own; MOST_DRAWS],
            len: 1,
        };
        if !tests.stencil {
            return Some(draws);
        }
        let fixed = |compare| matches!(compare, vk::CompareOp::ALWAYS | vk::CompareOp::NEVER);
        if !tests.faces.iter().all(|face| fixed(face.compare)) {
            return None;
        }
        let replaces = |f: &&Face| [f.fail, f.pass, f.depth_fail].contains(&vk::StencilOp::REPLACE);
        let bits = (tests.faces.iter().filter(replaces)).fold(0, |bits, f| bits | f.write_mask);
        let bits = bits & 0xff;
        if bits == 0 {
            return Some(draws);
        }
        let last = tests.depth_write && !fixed(tests.depth_compare);
        draws.draws[0] = BitDraw {
            tests: Tests {
                depth_write: tests.depth_write && !last,
                ..tests
            },
            reference: [0; 2],
            bit: EVERY_FRAGMENT,
        };
        let kept = |op| match op {
            vk::StencilOp::REPLACE => op,
            _ => vk::StencilOp::KEEP,
        };
        for k in (0..8).filter(|k| bits & (1 << k) != 0) {
            let face = |f: Face| Face {
                fail: kept(f.fail),
                pass: kept(f.pass),
                depth_fail: kept(f.depth_fail),
                write_mask: f.write_mask & (1 << k),
                ..f
            };
            draws.push(BitDraw {
                tests: Tests {
                    depth_write: false,
                    faces: tests.faces.map(face),
                    ..tests
                },
                reference: [255; 2],
                bit: k,
            });
        }
        if last {
            let unwritten = |f: Face| Face { write_mask: 0, ..f };
            draws.push(BitDraw {
                tests: Tests {
                    faces: tests.faces.map(unwritten),
                    ..tests
                },
                reference,
                bit: EVERY_FRAGMENT,
            });
        }
        Some(draws)
    }

    fn push(&mut self, draw: BitDraw) {
        self.draws[self.len] = draw;
        self.len += 1;
    }

    /// The draws, in order: at least one.
    pub(crate) fn draws(&self) -> &[BitDraw] {
        &self.draws[..self.len]
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
                descriptor: *descriptor,
                tests: Tests::of(descriptor),
            }),
            device: self.clone(),
        }
    }
}

impl DepthStencilState {
    /// The descriptor it was made with.
    pub fn descriptor(&self) -> &DepthStencilDescriptor {
        &self.inner.descriptor
    }

    /// No Metal counterpart: the static note of a draw under this state, where it tests or writes
    /// depth ([`PrecisionNote::DepthDenormals`]: a denormal depth is the device's); `None` where it
    /// does neither.
    ///
    /// A negative or NaN depth, which Metal's test compares by its magnitude, takes no note: the
    /// runtime enables no `VK_EXT_depth_range_unrestricted` and refuses a clear depth outside
    /// `[0, 1]` (`RenderPassKind::ClearDepth`) and a viewport's depth range past it
    /// (`DrawKind::Viewport`), so a depth attachment holds such a value only where Vulkan leaves a
    /// depth past the viewport's range undefined, or where an upload put one there, which is the
    /// device's under the upload's own note
    /// ([`Texture::FLOAT_DEPTH_UPLOAD_NOTE`](crate::Texture::FLOAT_DEPTH_UPLOAD_NOTE)).
    pub fn precision_note(&self) -> Option<PrecisionNote> {
        let tests = self.inner.tests;
        (tests.depth_test || tests.depth_write).then_some(PrecisionNote::DepthDenormals)
    }

    /// Its tests.
    pub(crate) fn tests(&self) -> Tests {
        self.inner.tests
    }

    /// Whether `core` is the device this state was made on.
    pub(crate) fn is_of(&self, core: &Arc<Core>) -> bool {
        Arc::ptr_eq(self.device.core(), core)
    }
}
