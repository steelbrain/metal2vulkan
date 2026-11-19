//! Replay streams (`docs/REPLAY-STREAM.md`): their reader and writer, the address map the player
//! moves address arguments by, the player, and its backends over the old runtime and the new one.

mod addresses;
mod args;
mod compare;
mod covers;
mod dumps;
mod heaps;
mod new_runtime;
mod old_runtime;
mod pixels;
mod player;
mod range;
mod stale;
mod stream;
mod texwrites;
mod translator;
mod value;

pub use compare::{compare_plays, shown_differences};
pub use covers::SlotStage;
pub use dumps::{compare_dumps, Comparison, Difference};
pub use heaps::HeapSplit;
pub use new_runtime::{judged_records, Ended, ManagedCopies, NewRuntime, VideoHandle};
pub use old_runtime::OldRuntime;
pub use pixels::Shape;
pub use player::WAIT_COMPLETED;
pub use player::{
    Backend, Check, Counters, Counts, End, IdDiffers, Made, ManagedReads, PlayError, Player,
    PrivateBytes, ReadClass, Relaid, Report, Shown, TextureWrites,
};
pub use range::{
    display_library, recording_range, sidecar, RangeRow, RangeSource, RecordingRange,
    RECORDED_RANGES,
};
pub use stream::{
    segments, Call, StreamError, StreamReader, StreamWriter, FLAG_ANNOUNCED_BYTES_RECORDED,
    FLAG_COMPLETION_BYTES_RECORDED,
};
pub use value::{hex, Hash, Name, Value};

#[cfg(test)]
mod tests;
