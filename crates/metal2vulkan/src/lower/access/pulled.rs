//! The source buffers of the pulled vertex fetch: their slot forms, which [`super::finish_slots`]
//! chooses together with the function's own slots, and the word memory the fetch
//! (`lower::stage_io::fetch`) reads them through. A kernel's `[[stage_in]]` buffers and index
//! buffer are source buffers of the same fetch in the kernel's buffer table (`ResourceSlot{Kernel,
//! Buffer, n}`): Metal binds one buffer at one offset per kernel buffer index, which a stage-input
//! layout and a `[[buffer(n)]]` argument both read, so every rule below holds for them as written.
//!
//! A pulled attribute reads its source buffer `ResourceSlot{Vertex, Buffer, n}` at any offset at
//! or above the bind offset (the element index is a runtime value): its window is `Whole`. Metal
//! binds one buffer at one offset per vertex-buffer index, and a vertex descriptor layout and a
//! `[[buffer(n)]]` argument of that index both read that binding. So:
//!
//! - **Shared.** When an accessed buffer slot of the function (a buffer argument or a
//!   buffer-table element) has the source buffer's Metal index, the fetch reads through that
//!   slot. Its window widens to `Whole`, and it is a storage slot (a uniform view's declared size
//!   is a bounded range and has no `OpArrayLength` for the in-shader check) whose access is the
//!   function's plus the fetch's read. The function may write it or reach it by atomics: the fetch
//!   runs once, before any instruction of the body, so it reads what Metal's attribute holds, an
//!   input the body's writes do not change. One shape has no sharing and is
//!   `Unsupported{Lowering(StageIo), NotImplemented}` (`share`): a window that is not within the
//!   first 2³² bytes at or above the bind offset, the reach of a `Whole` window (a `u32` Metal
//!   displacement from the bind offset, through a view or a device address).
//! - **Own.** Otherwise the source buffer is a read-only `Whole` slot of its own.
//!
//! Either way it is a storage slot like any other: a `Whole` range is never provably within
//! `maxStorageBufferRange`, so it declares `BufferRangeLimit`; a `BufferRangeLimit` variant
//! (`LowerInputs.psb_slots`) or Profile A's per-stage descriptor-limit fallback moves it to its
//! device address, and a shared slot takes the device address the argument's wide pointers or typed
//! atomics need. The fetch reads a storage view at the bind residual plus its Metal displacement,
//! contained by the profile's encoding, and a device address through the slot's root entry (`psb`,
//! window start 0), always contained unless containment is off. Reads are plain: a vertex function
//! has no barrier a coherence could order them with.

use crate::error::{Error, Feature, LowerFamily, Unsupported};
use crate::lower::emit::EmitCx;
use crate::lower::stage_io::fetch;
use crate::memory::contain::{self, Check};
use crate::memory::leaf::{self, k32, Leaf, WordMem};
use crate::memory::views::{self, View, ViewKind};
use crate::pipeline::interface::{
    AccessSet, BufferForm, BufferKind, BufferSlot, SlotInterface, Window,
};
use crate::pipeline::sym::{PushField, SlotRef};
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
use crate::spirv::error::{BResult, BuildError, Dangling};
use crate::spirv::ids::Value as SValue;
use crate::spirv::ir::BinOp;

use super::psb::{self, PsbMem};
use super::{descriptor_view, slot_root, AccessPlan, SlotLowering};

/// How the fetch reaches a pulled source buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Reach {
    /// Through memory-plan slot `slot`, an accessed slot of the same Metal index.
    Shared { slot: usize },
    /// Through a read-only `Whole` slot of its own.
    Own(SlotLowering),
}

/// One pulled source buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PulledSlot {
    /// The Metal slot: a vertex buffer, or a kernel buffer.
    pub slot: SlotRef,
    pub reach: Reach,
}

/// The window of an accessed slot of window `window` that the fetch shares (see the module docs):
/// `Whole`, which covers every window in the first 2³² bytes at or above the bind offset (the
/// proven window bounds Metal-defined accesses only, and those lie in the allocation). The slot's
/// access does not matter: the fetch precedes every access of the body. A window starting below the
/// bind offset or ending past 2³² bytes above it is refused, never truncated.
pub(super) fn share(window: Window) -> Result<Window, Error> {
    let refuse = || -> Error {
        Unsupported::not_implemented(Feature::Lowering(LowerFamily::StageIo)).into()
    };
    match window {
        Window::Whole => Ok(Window::Whole),
        Window::Bounded { start, len }
            if start >= 0 && i128::from(start) + i128::from(len) <= 1i128 << 32 =>
        {
            Ok(Window::Whole)
        }
        Window::Bounded { .. } => Err(refuse()),
    }
}

/// The access of a shared slot the function accesses with `access`: the function's, plus the
/// fetch's read.
pub(super) fn shared_access(access: AccessSet) -> AccessSet {
    AccessSet {
        read: true,
        ..access
    }
}

/// The artifact's statement of every source buffer with a slot of its own: a read-only `Whole` slot
/// of its Metal vertex-buffer index, in its chosen form, which the binder binds like any other
/// buffer slot of the stage. A shared source buffer is its slot's
/// [`crate::lower::select::SlotChoice`].
pub fn slot_interfaces(plan: &AccessPlan) -> impl Iterator<Item = (SlotRef, SlotInterface)> + '_ {
    plan.pulled.iter().filter_map(|p| match p.reach {
        Reach::Own(l) => Some((
            p.slot,
            SlotInterface::Buffer(BufferSlot {
                access: AccessSet::READ,
                window: Window::Whole,
                form: match l {
                    SlotLowering::View(kind) => BufferForm::Descriptor(descriptor_view(kind)),
                    SlotLowering::Psb { .. } => BufferForm::Psb,
                },
                kind: BufferKind::Buffer,
            }),
        )),
        Reach::Shared { .. } => None,
    })
}

/// Declare the storage view of every source buffer with a descriptor slot of its own.
pub(super) fn declare(mb: &mut ModuleBuilder, plan: &AccessPlan) -> BResult<Vec<(u32, View)>> {
    let mut out = Vec::new();
    for p in &plan.pulled {
        if let Reach::Own(SlotLowering::View(kind)) = p.reach {
            out.push((p.slot.index, views::declare(mb, p.slot, kind)?));
        }
    }
    Ok(out)
}

/// The read-only word memory of a source buffer: a storage view, or the words at a device
/// address.
#[derive(Clone, Copy, Debug)]
pub struct SourceMem(Mem);

#[derive(Clone, Copy, Debug)]
enum Mem {
    View(View),
    Psb(PsbMem),
}

impl WordMem for SourceMem {
    fn read(&mut self, fb: &mut FunctionBuilder<'_>, word: SValue) -> BResult<SValue> {
        match &mut self.0 {
            Mem::View(v) => views::read_word(fb, *v, word),
            Mem::Psb(m) => m.read(fb, word),
        }
    }

    fn write(&mut self, _: &mut FunctionBuilder<'_>, _: SValue, value: SValue) -> BResult<()> {
        Err(BuildError::Dangling(Dangling::Value(value)))
    }

    fn write_bits(
        &mut self,
        _: &mut FunctionBuilder<'_>,
        _: SValue,
        _: SValue,
        bits: SValue,
    ) -> BResult<()> {
        Err(BuildError::Dangling(Dangling::Value(bits)))
    }
}

/// A source-buffer read prepared for the leaf: the word memory, the byte offset in it, and the
/// in-bounds condition when the read is contained.
#[derive(Clone, Copy, Debug)]
pub struct SourceRead {
    pub mem: SourceMem,
    pub off: SValue,
    pub inb: Option<SValue>,
}

impl SourceRead {
    /// Load `leaf`, contained: the value, or zero out of bounds.
    pub fn load(mut self, fb: &mut FunctionBuilder<'_>, leaf: &Leaf) -> BResult<SValue> {
        let v = leaf::load(fb, &mut self.mem, leaf, self.off)?;
        match self.inb {
            None => Ok(v),
            Some(inb) => {
                let z = leaf::zero(fb, leaf)?;
                fb.select(inb, v, z)
            }
        }
    }
}

/// An `n`-byte read of storage view `view` at the bind residual `residual` plus the Metal
/// displacement `m` (wrapping `u32`), contained by `check`: out of bounds it reads at offset 0 (a
/// bound range is never empty and is padded to 16 bytes, so the at most 16 bytes of a vertex format
/// exist there) and yields zero.
pub fn view_read(
    fb: &mut FunctionBuilder<'_>,
    view: View,
    check: Check,
    residual: SValue,
    m: SValue,
    n: u32,
) -> BResult<SourceRead> {
    let off = fb.binary(BinOp::IAdd, residual, m)?;
    view_read_at(fb, view, check, off, n)
}

/// [`view_read`] at the view's own byte offset `off`.
fn view_read_at(
    fb: &mut FunctionBuilder<'_>,
    view: View,
    check: Check,
    off: SValue,
    n: u32,
) -> BResult<SourceRead> {
    let inb = match check {
        Check::None => None,
        Check::InShader => match views::storage_bytes(fb, view)? {
            Some(size) => Some(contain::in_bounds(fb, off, size, u64::from(n))?),
            None => return Err(BuildError::Dangling(Dangling::Global(view.global))),
        },
    };
    let off = match inb {
        Some(inb) => {
            let zero = k32(fb, 0)?;
            fb.select(inb, off, zero)?
        }
        None => off,
    };
    Ok(SourceRead {
        mem: SourceMem(Mem::View(view)),
        off,
        inb,
    })
}

/// An `n`-byte read of `[[buffer(buffer)]]` by device address, at the `u32` Metal displacement `m`
/// from its bind offset in a slot whose window starts at `start`, through the root table at
/// `table` (a `u64`), contained when `contained` (the PSB leaf, `psb`).
pub fn psb_read(
    fb: &mut FunctionBuilder<'_>,
    table: SValue,
    buffer: u32,
    m: SValue,
    start: i64,
    n: u32,
    contained: bool,
) -> BResult<SourceRead> {
    let a = psb::prepare(fb, table, slot_root(buffer), m, start, n, contained)?;
    Ok(SourceRead {
        mem: SourceMem(Mem::Psb(a.mem)),
        off: a.off,
        inb: a.inb,
    })
}

/// Prepare an `n`-byte read of source buffer `buffer` at the `u32` Metal displacement `m` from
/// its bind offset, in the form [`super::finish_slots`] chose; a view is contained by `check`.
pub fn prepare(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    buffer: u32,
    m: SValue,
    n: u32,
    check: Check,
) -> BResult<SourceRead> {
    match source(cx, fb, buffer)? {
        Source::View { view, residual } => view_read(fb, view, check, residual, m, n),
        Source::Psb {
            table,
            start,
            contained,
        } => psb_read(fb, table, buffer, m, start, n, contained),
    }
}

/// [`prepare`] of a read whose `u32` Metal displacement `m` is Metal's own only where `fits` (a
/// bool) holds, the read's `n` bytes then lying in the first `2^32` bytes from the bind offset.
/// Where it does not, and in a view where the bind residual plus `m` and its `n` bytes run past
/// `2^32` (the view's own offset would wrap), the read lies past the first `2^32` bytes of the
/// slot: outside every buffer a command that may read past them is not refused for (the
/// runtime's `StageInFetchPast4GiB`), and outside every view's range. It then reads at the bind
/// offset and yields zero bytes, Metal's answer for a fetch outside its buffer there, on every
/// device, its robustness aside: a wrapped offset could land inside the buffer.
pub fn prepare_within(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    buffer: u32,
    m: SValue,
    fits: SValue,
    n: u32,
    check: Check,
) -> BResult<SourceRead> {
    let (read, fits) = match source(cx, fb, buffer)? {
        Source::View { view, residual } => {
            // `residual + m + n <= 2^32`: the sum did not wrap, and its end fits.
            let off = fb.binary(BinOp::IAdd, residual, m)?;
            let kept = fb.binary(BinOp::UGreaterThanEqual, off, residual)?;
            let room = k32(fb, u32::MAX - n.saturating_sub(1))?;
            let ends = fb.binary(BinOp::ULessThanEqual, off, room)?;
            let view_fits = fb.binary(BinOp::LogicalAnd, kept, ends)?;
            let fits = fb.binary(BinOp::LogicalAnd, fits, view_fits)?;
            let off = fb.select(fits, off, residual)?;
            (view_read_at(fb, view, check, off, n)?, fits)
        }
        Source::Psb {
            table,
            start,
            contained,
        } => {
            let zero = k32(fb, 0)?;
            let m = fb.select(fits, m, zero)?;
            (psb_read(fb, table, buffer, m, start, n, contained)?, fits)
        }
    };
    let inb = match read.inb {
        Some(inb) => fb.binary(BinOp::LogicalAnd, inb, fits)?,
        None => fits,
    };
    Ok(SourceRead {
        inb: Some(inb),
        ..read
    })
}

/// Where source buffer `buffer` is read, in the form [`super::finish_slots`] chose.
enum Source {
    /// Its storage view, at the bind residual.
    View { view: View, residual: SValue },
    /// Its device address, through the root table at `table`, in a slot whose window starts at
    /// `start`, contained when `contained`.
    Psb {
        table: SValue,
        start: i64,
        contained: bool,
    },
}

/// The form source buffer `buffer` is read in.
fn source(cx: &mut EmitCx<'_>, fb: &mut FunctionBuilder<'_>, buffer: u32) -> BResult<Source> {
    let access = &cx.plan.families.access;
    let slot = fetch::slot(cx.plan.stage, buffer);
    let reach = access
        .pulled
        .iter()
        .find(|p| p.slot == slot)
        .map(|p| p.reach);
    // The form is `AccessPlan::source_lowering`'s, which the census reads too.
    let lowering = access.source_lowering(slot);
    let view = match reach {
        Some(Reach::Shared { slot }) => cx.decl.families.access.views.get(slot).copied().flatten(),
        Some(Reach::Own(_)) => cx
            .decl
            .families
            .access
            .pulled
            .iter()
            .find(|(b, _)| *b == buffer)
            .map(|(_, v)| *v),
        None => return Err(cx.fail(format!("source buffer {slot:?} has no planned slot"))),
    };
    match (lowering, view) {
        (Some(SlotLowering::View(ViewKind::Storage)), Some(view)) => {
            let residual = cx.push_field(fb, PushField::SlotBindOffset(slot))?;
            Ok(Source::View { view, residual })
        }
        (Some(SlotLowering::Psb { start }), _) => {
            let table = cx.push_field(fb, PushField::RootTableAddress)?;
            let contained = cx.plan.families.access.psb_contained;
            Ok(Source::Psb {
                table,
                start,
                contained,
            })
        }
        _ => Err(cx.fail(format!(
            "source buffer {slot:?} is neither a declared storage view nor a device address"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Needs;

    /// A slot's window widens to `Whole` from any window in the first 2³² bytes at or above the
    /// bind offset, whatever the function's access (the fetch precedes the body's writes and
    /// atomics, so they never reach it); the shared slot's access adds the fetch's read. A window
    /// starting below the bind offset or ending past 2³² bytes above it (which a `u32` displacement
    /// cannot reach) is refused, never truncated.
    #[test]
    fn a_shared_window_widens_within_the_first_four_gigabytes() {
        let four = 1u64 << 32;
        for w in [
            Window::Whole,
            Window::Bounded { start: 0, len: 16 },
            Window::Bounded {
                start: 0,
                len: four,
            },
            Window::Bounded {
                start: (four - 16) as i64,
                len: 16,
            },
        ] {
            assert_eq!(share(w), Ok(Window::Whole), "{w:?}");
        }
        let refused = |r: Result<Window, Error>| match r {
            Err(Error::Unsupported(u)) => {
                u.feature == Feature::Lowering(LowerFamily::StageIo)
                    && u.needs == Needs::NotImplemented
            }
            _ => false,
        };
        for w in [
            Window::Bounded { start: -4, len: 64 },
            Window::Bounded {
                start: (four - 16) as i64,
                len: 20,
            },
            Window::Bounded {
                start: (four + 16) as i64,
                len: 4,
            },
            Window::Bounded {
                start: 4096,
                len: 1 << 40,
            },
            Window::Bounded {
                start: i64::MAX,
                len: u64::MAX,
            },
        ] {
            assert!(refused(share(w)), "{w:?}");
        }
        for (access, shared) in [
            (AccessSet::READ, AccessSet::READ),
            (AccessSet::WRITE, AccessSet::READ_WRITE),
            (AccessSet::READ_WRITE, AccessSet::READ_WRITE),
            (AccessSet::UNUSED, AccessSet::READ),
        ] {
            assert_eq!(shared_access(access), shared, "{access:?}");
        }
    }
}
