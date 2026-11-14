//! Protection options (`setProtectionOptions:`, SPI) as Metal's GPU answers them, measured by a
//! consumer on an M3 Max (macOS 26.7) over every option below 0x10, 0x10 to 0x100, 0x11, 0x13,
//! `u32::MAX`, `1 << 63`, `u64::MAX - 1` and `u64::MAX`. The runtime has no protected content;
//! what Metal does with the options, it does.
//!
//! - **An encoder made under options other than 0** runs nothing, nor does any encoder after it,
//!   and the command buffer fails with [`CommandBufferError::ProtectedEncoder`]: code 16
//!   ("Protection Violation") below 0x10, 1 (`MTLCommandBufferErrorInternal`, "Invalid Input")
//!   from 0x10 up, none of its event signals firing. An encoder made before the options runs.
//! - **Options from 0x10 up** fail it so with no encoder made under them too, every encoder
//!   running.
//! - **Options 1, 3 and 7** end the process where a compute or render encoder is made under them,
//!   and at a blit encoder's first command (SIGABRT, no message): `Invalid`,
//!   [`Problem::ProtectedEncoder`].
//! - **Refused**: the call while an encoder is open, as Metal asserts
//!   (`exec_protection_envelope_open_encoder`; `Problem::EncoderOpen`), and after the commit;
//!   options changed once an encoder was made under some, or after options from 0x10 up, and a
//!   resource state encoder made under 1, 3 or 7, none measured
//!   ([`AppleOnly::ProtectedContent`]).

use super::{CommandBuffer, FailedEncoders, State};
use crate::apple_only::AppleOnly;
use crate::error::{invalid, CommandBufferError, Error, Problem};
use std::collections::BTreeSet;

/// What `setProtectionOptions:` left on a command buffer.
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct Protection {
    /// The options set last: 0 until one is set.
    options: u64,
    /// The place among the ended encoders of the first encoder made under options other than 0.
    first: Option<usize>,
}

/// Whether `options` end the process where an encoder is made under them: a compute or render
/// encoder's creation, a blit encoder's first command.
fn aborts(options: u64) -> bool {
    matches!(options, 1 | 3 | 7)
}

impl Protection {
    /// The options set last.
    pub(super) fn options(&self) -> u64 {
        self.options
    }

    /// Set `options`; refused where an encoder was made under options already, or options from
    /// 0x10 up were set, and `options` differ: not measured.
    fn set(&mut self, options: u64) -> Result<(), AppleOnly> {
        let kept = self.first.is_some() || self.options >= 0x10;
        if kept && options != self.options {
            return Err(AppleOnly::ProtectedContent);
        }
        self.options = options;
        Ok(())
    }

    /// An encoder was made at `place` among the ended encoders, under the options set now.
    fn made(&mut self, place: usize) {
        if self.options != 0 {
            self.first.get_or_insert(place);
        }
    }

    /// How the options fail the command buffer at its commit: the encoders that run nothing, and
    /// its error. `None` where they fail nothing.
    pub(super) fn failure(&self) -> Option<(FailedEncoders, CommandBufferError)> {
        let encoders = match self.first {
            Some(first) => FailedEncoders::From(first),
            None if self.options >= 0x10 => FailedEncoders::Set(BTreeSet::new()),
            None => return None,
        };
        Some((encoders, CommandBufferError::ProtectedEncoder(self.options)))
    }
}

impl State {
    /// An encoder opened now, at its place among the ended encoders, under the options set now.
    pub(crate) fn protected_made(&mut self) {
        let place = self.open_encoder_place();
        self.protection.made(place);
    }

    /// The options set now where they end Metal's process where an encoder is made under them (a
    /// compute or render encoder's creation, a blit encoder's first command); `None` for any
    /// others.
    pub(crate) fn protection_aborts(&self) -> Option<u64> {
        let options = self.protection.options();
        aborts(options).then_some(options)
    }

    /// Refuse `call`, a compute or render encoder's creation, where the options set now end the
    /// process there (`Invalid`, [`Problem::ProtectedEncoder`]), as Metal's do.
    pub(crate) fn refuse_protected(&self, call: &'static str) -> Result<(), Error> {
        match self.protection_aborts() {
            Some(options) => Err(invalid(call, Problem::ProtectedEncoder { options })),
            None => Ok(()),
        }
    }
}

impl CommandBuffer {
    /// `setProtectionOptions:` (SPI): the options the encoders made after it are made under, which
    /// fail the command buffer as Metal's GPU fails it, or end the process where Metal's does
    /// (`command_buffer/protection.rs`: an encoder made under options other than 0 runs nothing,
    /// nor does any after it, and the command buffer fails with
    /// [`CommandBufferError::ProtectedEncoder`], none of its event signals firing; under options
    /// from 0x10 up it fails so with no such encoder too). Refused: while an encoder is open
    /// (`Invalid`, `Problem::EncoderOpen`, as Metal asserts) and after the commit
    /// (`Problem::AlreadyCommitted`); options changed once an encoder was made under some, or
    /// after options from 0x10 up, which nothing measured ([`AppleOnly::ProtectedContent`]).
    pub fn set_protection_options(&self, options: u64) -> Result<(), Error> {
        self.settle_end();
        let mut state = self.state();
        if let Some(problem) = state.refusal() {
            return Err(invalid("set_protection_options", problem));
        }
        state.protection.set(options).map_err(Error::AppleOnly)
    }

    /// Refuse `call`, a compute or render encoder's creation, where the protection options end
    /// the process there (`Invalid`, [`Problem::ProtectedEncoder`]), as Metal's do.
    pub(crate) fn refuse_protected_encoder(&self, call: &'static str) -> Result<(), Error> {
        self.state().refuse_protected(call)
    }

    /// Refuse a resource state encoder's creation under options that end the process where a
    /// compute or render encoder is made: what Metal does there is not measured
    /// ([`AppleOnly::ProtectedContent`]).
    pub(crate) fn refuse_protected_state_encoder(&self) -> Result<(), Error> {
        match self.state().protection_aborts() {
            Some(_) => Err(Error::AppleOnly(AppleOnly::ProtectedContent)),
            None => Ok(()),
        }
    }
}
