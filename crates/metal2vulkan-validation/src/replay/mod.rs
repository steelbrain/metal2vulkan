//! Replay streams (`docs/REPLAY-STREAM.md`): their reader and writer, the address map the player
//! relocates by, the player, and its backends over the old runtime and the new one.

mod addresses;
mod args;
mod new_runtime;
mod old_runtime;
mod player;
mod range;
mod stream;
mod texwrites;
mod translator;
mod value;

pub use new_runtime::{ManagedCopies, NewRuntime};
pub use old_runtime::OldRuntime;
pub use player::WAIT_COMPLETED;
pub use player::{
    Backend, Check, Counters, Counts, End, IdDiffers, Made, PlayError, Player, PrivateBytes,
    Relaid, Report, TextureWrites,
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
