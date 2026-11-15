//! When a build with debug assertions checks a draw's kept value against the one it would work out
//! whole: a draw's used resources (`RenderLowering::refresh_bind_state`), its bind shape
//! (`LiveBind::shape`), its uses of what its encoder made resident (`binding::command_uses_into`)
//! and its heap part (`render/heap_part.rs`). Each is a walk of everything the encoder made
//! resident to work out, so a check at every draw made a recording of `n` draws over `r` resident
//! resources walk them `n` times in those builds alone: a host step of 512 draws reading textures
//! through the heap over 2048 resident resources spent about 95% of its time in those checks.
//!
//! - **Its first uses, then powers of two.** A kept value is checked at its first [`FIRST`] uses
//!   after it was worked out, and then at each use whose count is a power of two, so a recording
//!   walks the resident resources `FIRST + log2(n)` times for each value it keeps.
//! - **Still caught.** A kept value that is not its draw's own stays so until a change works it
//!   out again, so one that goes wrong after its first uses is caught within twice the uses it
//!   took, where the value is still kept then.
//! - Builds without debug assertions check none of them, and count nothing.

/// The uses after a value is worked out that each check it.
const FIRST: u64 = 8;

/// A kept value's uses since it was worked out (the module docs).
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct KeptChecks(u64);

impl KeptChecks {
    /// The value was worked out again: its uses count from none.
    pub(crate) fn worked_out(&mut self) {
        self.0 = 0;
    }

    /// Count a use of the value: whether this one checks it (the module docs).
    pub(crate) fn due(&mut self) -> bool {
        let uses = self.0;
        self.0 += 1;
        uses < FIRST || uses.is_power_of_two()
    }
}
