//! A workload as data: what [`Harness::run`](crate::run::Harness::run) makes, the calls of one run, and what
//! the first run must leave in the buffers.

use crate::{Result, Size, Storage};

/// One workload.
#[derive(Clone, Debug)]
pub struct Plan {
    /// Its name in reports.
    pub name: String,
    /// The buffers it makes, in order; dispatches and expectations name them by index.
    pub buffers: Vec<BufferPlan>,
    /// The dispatches of one run, in order, all in one compute encoder of one command buffer.
    pub dispatches: Vec<Dispatch>,
    /// What the first run must leave in its buffers.
    pub expect: Vec<Expect>,
}

/// One buffer of a plan.
#[derive(Clone, Debug)]
pub struct BufferPlan {
    /// Its length in bytes: a multiple of 16, the harness's transfer unit.
    pub length: u64,
    /// Its storage mode.
    pub storage: Storage,
    /// The bytes it starts with, `length` of them: for `Shared` storage written through
    /// `contents`, for `Private` copied in from a `Shared` buffer by the harness's own kernel.
    /// `None`: as `newBufferWithLength:` made it.
    pub contents: Option<Vec<u8>>,
}

/// One dispatch: the calls an encoder makes for it. The pipeline state is set where it differs
/// from the dispatch before's; bindings stay set for the dispatches after, as Metal's do.
#[derive(Clone, Debug)]
pub struct Dispatch {
    /// The kernel's name in the plan's library.
    pub function: String,
    /// The binding calls before the dispatch, in order.
    pub bindings: Vec<Binding>,
    /// The dispatch call.
    pub grid: Grid,
}

/// A binding call.
#[derive(Clone, Debug)]
pub enum Binding {
    /// `setBuffer:offset:atIndex:` of the plan's buffer `buffer`.
    Buffer {
        /// The argument table index.
        index: u32,
        /// The plan's buffer.
        buffer: usize,
        /// The offset in it.
        offset: u64,
    },
    /// `setBytes:length:atIndex:`.
    Bytes {
        /// The argument table index.
        index: u32,
        /// The bytes.
        bytes: Vec<u8>,
    },
}

/// A dispatch call.
#[derive(Clone, Copy, Debug)]
pub enum Grid {
    /// `dispatchThreadgroups:threadsPerThreadgroup:`.
    Threadgroups {
        /// Threadgroups per grid.
        groups: Size,
        /// Threads per threadgroup.
        threads: Size,
    },
    /// `dispatchThreads:threadsPerThreadgroup:`.
    Threads {
        /// Threads per grid.
        threads: Size,
        /// Threads per threadgroup.
        per_group: Size,
    },
}

impl Grid {
    /// Threadgroups of 256 threads in one dimension.
    pub fn groups_of_256(groups: u64) -> Grid {
        Grid::Threadgroups {
            groups: [groups, 1, 1],
            threads: [256, 1, 1],
        }
    }
}

/// What a buffer must hold after the first run.
#[derive(Clone, Debug)]
pub struct Expect {
    /// The plan's buffer.
    pub buffer: usize,
    /// Its contents from its start.
    pub values: Values,
}

/// Expected contents.
#[derive(Clone, Debug)]
pub enum Values {
    /// 32-bit words, each exactly.
    Words(Vec<u32>),
    /// `float`s, each within `tolerance` of its value: the device's own arithmetic may differ
    /// from the CPU's in its last bits.
    Floats {
        /// The values.
        values: Vec<f32>,
        /// The largest difference allowed.
        tolerance: f32,
    },
}

impl Values {
    /// The bytes the values cover.
    pub fn length(&self) -> u64 {
        let count = match self {
            Values::Words(words) => words.len(),
            Values::Floats { values, .. } => values.len(),
        };
        4 * count as u64
    }

    /// `Ok` where `bytes` (little-endian) holds the values; else how many differ and the first.
    pub fn compare(&self, bytes: &[u8]) -> Result<()> {
        let words = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|w| u32::from_le_bytes(*w));
        let (mut differ, mut first) = (0usize, None);
        let mut note = |index: usize, got: String, want: String| {
            differ += 1;
            first.get_or_insert((index, got, want));
        };
        match self {
            Values::Words(want) => {
                for (index, (got, &want)) in words.zip(want).enumerate() {
                    if got != want {
                        note(index, format!("{got:#x}"), format!("{want:#x}"));
                    }
                }
            }
            Values::Floats { values, tolerance } => {
                for (index, (got, &want)) in words.map(f32::from_bits).zip(values).enumerate() {
                    // False for a NaN on either side, which then differs.
                    let within = (got - want).abs() <= *tolerance;
                    if !within {
                        note(index, format!("{got}"), format!("{want} +- {tolerance}"));
                    }
                }
            }
        }
        match first {
            None => Ok(()),
            Some((index, got, want)) => Err(format!(
                "{differ} values differ; the first, [{index}], is {got}, not {want}"
            )
            .into()),
        }
    }
}

impl Plan {
    /// `Ok` where every index it names exists and every length is whole and agrees.
    pub fn validate(&self) -> Result<()> {
        let name = &self.name;
        for (i, buffer) in self.buffers.iter().enumerate() {
            if buffer.length == 0 || buffer.length % 16 != 0 {
                return Err(format!(
                    "{name}: buffer {i}'s length {} is not a positive multiple of 16",
                    buffer.length
                )
                .into());
            }
            if let Some(contents) = &buffer.contents {
                if contents.len() as u64 != buffer.length {
                    return Err(format!(
                        "{name}: buffer {i} has {} bytes of contents for {}",
                        contents.len(),
                        buffer.length
                    )
                    .into());
                }
            }
        }
        let known = |buffer: usize| buffer < self.buffers.len();
        for (i, dispatch) in self.dispatches.iter().enumerate() {
            for binding in &dispatch.bindings {
                if let Binding::Buffer { buffer, .. } = binding {
                    if !known(*buffer) {
                        return Err(format!(
                            "{name}: dispatch {i} binds buffer {buffer}, which the plan lacks"
                        )
                        .into());
                    }
                }
            }
        }
        for expect in &self.expect {
            if !known(expect.buffer) {
                return Err(format!(
                    "{name}: an expectation names buffer {}, which the plan lacks",
                    expect.buffer
                )
                .into());
            }
            if expect.values.length() > self.buffers[expect.buffer].length {
                return Err(format!(
                    "{name}: buffer {} is shorter than its expected values",
                    expect.buffer
                )
                .into());
            }
        }
        Ok(())
    }
}
