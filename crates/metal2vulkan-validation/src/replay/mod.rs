//! Replay streams (`docs/REPLAY-STREAM.md`): their reader and writer, the address map the player
//! relocates by, the player, and its backend over the old runtime.

mod addresses;
mod old_runtime;
mod player;
mod stream;
mod value;

pub use old_runtime::OldRuntime;
pub use player::WAIT_COMPLETED;
pub use player::{Backend, Check, Counters, Made, PlayError, Player, Report};
pub use stream::{segments, Call, StreamError, StreamReader, StreamWriter};
pub use value::{Hash, Name, Value};

#[cfg(test)]
mod tests;
