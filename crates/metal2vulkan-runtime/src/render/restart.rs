//! Primitive restart at a chosen index: Apple's `setPrimitiveRestartEnabled:index:`.
//!
//! Measured on an M4 with macOS 27.0.1, with and without API validation, for a call with index `k`:
//!
//! - An indexed line or triangle strip drawn after the call restarts at `k'` alone, and the index
//!   type's all-ones index draws as a vertex unless it is `k'`. `enabled` changes nothing.
//! - `k'` is `k mod 2^32` for 32-bit indices, and the lesser of `k mod 2^32` and `0xffff` for
//!   16-bit ones.
//! - `k'` is compared with the index as the buffer holds it, before the base vertex.
//! - A triangle's winding parity restarts with each piece, and `primitive_id` counts the drawn
//!   primitives on across a restart, as it does across the all-ones restart.
//! - Lists and points draw every index as a vertex, and a draw of no indices has none to restart
//!   at: the call changes none of them. Of Metal's private types, the quad strip restarts as the
//!   strips do, at `k'` after the call (16-bit, on the M3 Max with macOS 27.0.1); the line loop,
//!   fan and quads never restart, drawing `k'` and the all-ones index as vertices.
//! - An indexed indirect draw follows the call; an execution of indirect render commands does
//!   not, its strips restarting at the all-ones index.
//! - The call applies to the encoder's draws after it, whether before or after its pipeline
//!   state is set; the last call wins; a draw before it keeps the all-ones index; a new encoder
//!   restarts at the all-ones index.
//!
//! Where `k'` is the index type's all-ones index, Vulkan's `primitiveRestartEnable` restarts
//! there as Metal does, and the draw stays as it is. Elsewhere it is drawn as the list the index
//! rewrite pass writes (`index_rewrite.rs`), with the class's list topology.
//!
//! Metal's private primitive types 5 to 8 (`DrawPrimitiveType`), which no Vulkan topology draws as
//! Metal does, are drawn as lists too, in Metal's order: a direct draw without indices as the list
//! the CPU writes into the command buffer's upload arena (its positions are its vertices, so the
//! list follows from its count), and a draw over indices or from indirect arguments as the list
//! the index rewrite pass writes. Of them only the quad strip restarts, at the all-ones index of
//! its index type where the encoder made no call, at `k'` after one.
//!
//! Under Apple's last provoking vertex, a device without it draws a line or triangle draw whose
//! pipeline reads its primitives' vertex order as a list too, of a strip's pieces as above
//! (`render/provoking.rs`); so is a line strip drawn as bands at a width other than 1
//! (`render/line_width.rs`). Every list is written in the draw's order: Metal's at the encoder's
//! provoking vertex mode, or led by each primitive's last vertex.

use super::provoking::Provoking;
use super::RenderCall;
use super::RenderCommandEncoder;
use super::{Assembly, RenderLowering};
use crate::error::RuntimeGap;
use metal2vulkan::bind::{DrawCall, RewriteOrder, RewriteProgram};
use metal2vulkan::metal::PrimitiveType;

/// A draw drawn as a list: its program, `k'`, its index size (`None` for a draw without indices,
/// whose positions are its vertices), the order of each primitive's vertices, and the gap a list
/// the runtime cannot draw as Metal draws refuses with. A direct draw without indices draws the
/// list the CPU writes; any other the list the index rewrite pass writes.
#[derive(Clone, Copy)]
pub(super) struct Restarted {
    pub(super) program: RewriteProgram,
    pub(super) restart: u32,
    pub(super) index_size: Option<u64>,
    pub(super) order: RewriteOrder,
    pub(super) gap: ListGap,
}

/// Why a draw is drawn as a list: the gap that refuses one the runtime cannot draw so.
#[derive(Clone, Copy)]
pub(super) enum ListGap {
    RestartedStrip,
    PrivatePrimitives,
    ProvokingVertex,
    WideLines,
}

impl ListGap {
    /// The gap a list the runtime cannot draw as Metal draws refuses with.
    pub(super) fn gap(self) -> RuntimeGap {
        match self {
            ListGap::RestartedStrip => RuntimeGap::RestartedStrip,
            ListGap::PrivatePrimitives => RuntimeGap::PrivatePrimitives,
            ListGap::ProvokingVertex => RuntimeGap::ProvokingVertex,
            ListGap::WideLines => RuntimeGap::WideLines,
        }
    }
}

impl Restarted {
    /// The list a draw of the program draws: of lines where its primitives have two vertices,
    /// else of triangles.
    pub(super) fn assembly(self) -> Assembly {
        Assembly::Primitives(match self.program.vertices() {
            2 => PrimitiveType::Line,
            _ => PrimitiveType::Triangle,
        })
    }
}

impl RenderLowering {
    /// The lowering of [`RenderCommandEncoder::set_primitive_restart_enabled`].
    pub(crate) fn set_primitive_restart_enabled(&mut self, _enabled: bool, index: u64) {
        self.log(|_| RenderCall::PrimitiveRestartEnabled(_enabled, index));
        self.restart = Some(index);
    }

    /// Where `draw` of `assembly`, which takes the encoder's provoking vertex as `provoking`
    /// says, is drawn as a list: a draw of a private type, which restarts (a quad strip over
    /// indices) at the encoder's `k'`, or its index type's all-ones index without a call; an
    /// indexed strip the encoder's call restarts at another index than Vulkan would, `k'` not its
    /// index type's all-ones index; a draw of lines or triangles `provoking` lists; and a line
    /// strip drawn as bands (`band`, `render/line_width.rs`), whose patches are each line's two
    /// vertices. `None` for every other draw.
    pub(super) fn restarted(
        &self,
        assembly: Assembly,
        draw: &DrawCall,
        (provoking, band): (Provoking, bool),
    ) -> Option<Restarted> {
        let wide = match *draw {
            DrawCall::Indexed { index_type, .. }
            | DrawCall::Indirect {
                indexed: Some((index_type, ..)),
                ..
            } => Some(index_type == metal2vulkan::vk::IndexType::UINT32),
            _ => None,
        };
        let index_size = wide.map(|wide| if wide { 4 } else { 2 });
        let order = provoking.order();
        // Where a restarting program over indices restarts: at `k'` after a call, else at its
        // index type's all-ones index.
        let restart = match wide {
            Some(wide) => match self.restart {
                Some(k) => restart_index(k, wide).0,
                None => restart_index(u64::MAX, wide).1,
            },
            None => 0,
        };
        let listed = |program, gap| {
            Some(Restarted {
                program,
                restart,
                index_size,
                order,
                gap,
            })
        };
        if let Assembly::Private(program) = assembly {
            return listed(program, ListGap::PrivatePrimitives);
        }
        let program = assembly.program()?;
        if let Provoking::Listed(_) = provoking {
            return listed(program, ListGap::ProvokingVertex);
        }
        if band && program == RewriteProgram::LineStrip {
            return listed(program, ListGap::WideLines);
        }
        if !program.restarts() {
            return None;
        }
        let (k, wide) = (self.restart?, wide?);
        if restart == restart_index(k, wide).1 {
            return None;
        }
        listed(program, ListGap::RestartedStrip)
    }
}

impl RenderCommandEncoder {
    /// `setPrimitiveRestartEnabled:index:` (SPI): the encoder's indexed line and triangle strips,
    /// and quad strips ([`DrawPrimitiveType::QuadStrip`](crate::DrawPrimitiveType)), drawn after
    /// the call restart at `index` as Metal takes it for their index type (`k mod 2^32` for 32-bit
    /// indices, the lesser of that and `0xffff` for 16-bit ones), and only there: the all-ones
    /// index then draws as a vertex unless it is that index. `_enabled` changes nothing, as on
    /// Metal. Lists, points, line loops, fans, quads, draws without indices and executions of
    /// indirect render commands are as before; the last call wins until the encoder ends (an M4). A
    /// strip restarting elsewhere than at the all-ones index is drawn as a list a GPU pass writes
    /// before the pass; a draw it cannot draw so is refused (`NotImplemented(RestartedStrip)`).
    pub fn set_primitive_restart_enabled(&mut self, _enabled: bool, index: u64) {
        match self.lowers_at_call() {
            true => self
                .at_call_unit(|lowering| lowering.set_primitive_restart_enabled(_enabled, index)),
            false => self.defer_unit(|_| RenderCall::PrimitiveRestartEnabled(_enabled, index)),
        }
    }
}

/// `k'` for a call with `k` over 32-bit indices (`wide`) or 16-bit ones, and the index type's
/// all-ones index.
fn restart_index(k: u64, wide: bool) -> (u32, u32) {
    // `k mod 2^32`.
    let low = k as u32;
    match wide {
        true => (low, u32::MAX),
        false => (low.min(0xffff), 0xffff),
    }
}

#[cfg(test)]
mod tests {
    use super::restart_index;

    #[test]
    fn the_restart_index_is_metal_s_measured_one() {
        // 32-bit indices take `k mod 2^32`, 16-bit ones the lesser of that and 0xffff.
        let wide = [
            (3, 3),
            (0xffff, 0xffff),
            (0x1_0000_0003, 3),
            (u64::MAX, u32::MAX),
        ];
        for (k, expected) in wide {
            assert_eq!(restart_index(k, true), (expected, u32::MAX), "{k:#x}");
        }
        let narrow = [
            (3, 3),
            (0xfffe, 0xfffe),
            (0x1_0000, 0xffff),
            (0xffff_ffff, 0xffff),
            (0x1_0000_0000, 0),
            (0x1_0000_0003, 3),
            (0x1_0000_00ff, 0xff),
            (0x1_0000_ffff, 0xffff),
            (0x2_0001_0003, 0xffff),
            (u64::MAX, 0xffff),
        ];
        for (k, expected) in narrow {
            assert_eq!(restart_index(k, false), (expected, 0xffff), "{k:#x}");
        }
    }
}
