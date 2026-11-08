//! Where a command buffer's first submission found a use rebound
//! (`CommandBufferError::EncodedUseRebound`, `first_submission.rs`), and why no rebind encoded or
//! recorded it again (`reencode.rs`): the notice names it (`Device::degrade_encoded_use_rebound`),
//! the first one found, as the host step's notice names its rule.

use crate::used_resources::EncoderKind;
use std::fmt;

/// Why an ended encoder kept no log a rebind encodes it again from (`replay.rs`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Unlogged {
    /// It logged no calls: the device had no buffer over host pages as it opened, or it is of a
    /// kind that logs none (a resource state encoder's).
    NoCalls,
    /// It left work for later that an encoding again does not record at once: a host step's (a
    /// GPU-written indirect dispatch's, a scatter into a guest RAM staged through the CPU's,
    /// `host_ram/copy.rs`), or a scrub's (`Segment::records_again`).
    Segments,
    /// The command buffer's failure fails it, or a fault of its own, as it commits.
    Fails,
    /// A compute encoder of a command buffer that used command predication whose unit laid a
    /// step before it opened, or one of its own that the unit's later encoders read, or a loop,
    /// or left work out (`Predication::alone`): a command encoded again after the commit would
    /// run outside its block, or leave a command of the unit's reading a gate nothing writes.
    Predicated,
}

/// How a segment's recording the first submission judged was made (`encoder.rs`'s segments).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Recording {
    /// At the commit, on the GPU alone, for good (`Deferred::record_on_gpu`).
    GpuAlone,
    /// At the commit, from the CPU's records or as a dispatch whose pipelines were made.
    Commit,
    /// By a host step.
    HostStep,
}

/// Which of an encoder's recordings held the rebound use.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum ReboundAt {
    /// A command of the encoder's own named the buffer, and the encoder kept no log: why, where
    /// known.
    Unlogged(Option<Unlogged>),
    /// A command of the encoder's own named it, and its encoding again was refused (a call that
    /// answered otherwise than at its encoding, a recording unlike the first, work no longer on the
    /// GPU alone): the refusal.
    Refused(String),
    /// A command of the encoder's own named it; the encoder kept its log, and no rebind encoded it
    /// again.
    NotEncodedAgain,
    /// Work it left for later named it.
    Segment(Recording),
}

/// Where the first submission found a use rebound: the command buffer's ended encoder `encoder`,
/// of `kind`, and which of its recordings.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ReboundUse {
    pub(crate) encoder: usize,
    pub(crate) kind: EncoderKind,
    pub(crate) at: ReboundAt,
}

impl fmt::Display for Unlogged {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Unlogged::NoCalls => {
                "it logged no calls (no buffer over host pages as it opened, or a kind that \
                 logs none)"
            }
            Unlogged::Segments => "it left work for later an encoding again does not record",
            Unlogged::Fails => "its command buffer fails it as it commits",
            Unlogged::Predicated => {
                "its unit of command predication laid a step before it opened, or one of its \
                 own that the unit's later encoders read, or a loop, or left work out"
            }
        })
    }
}

impl fmt::Display for ReboundUse {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let kind = match self.kind {
            EncoderKind::Blit => "blit",
            EncoderKind::Compute => "compute",
            EncoderKind::Render => "render",
            EncoderKind::ResourceState => "resource state",
        };
        write!(f, "found in its {kind} encoder {}: ", self.encoder)?;
        match &self.at {
            ReboundAt::Unlogged(Some(why)) => {
                write!(
                    f,
                    "a command of its own named it, not encoded again as {why}"
                )
            }
            ReboundAt::Unlogged(None) => {
                f.write_str("a command of its own named it, and it kept no log")
            }
            ReboundAt::Refused(refusal) => write!(
                f,
                "a command of its own named it, and its encoding again was refused: {refusal}"
            ),
            ReboundAt::NotEncodedAgain => {
                f.write_str("a command of its own named it; its log was kept, no rebind encoded it")
            }
            ReboundAt::Segment(Recording::GpuAlone) => f.write_str(
                "work its commit recorded on the GPU alone named it, and no rebind recorded it \
                 again",
            ),
            ReboundAt::Segment(Recording::Commit) => {
                f.write_str("work its commit recorded named it, which a rebind does not record")
            }
            ReboundAt::Segment(Recording::HostStep) => {
                f.write_str("work a host step recorded named it, which a rebind does not record")
            }
        }
    }
}
