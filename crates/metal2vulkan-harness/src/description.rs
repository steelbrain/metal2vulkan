//! Workloads described in a text file, over metallibs in a directory: the harness's input for
//! kernels it holds no source of. Nothing in the repository names such a workload; the
//! description and the metallibs are the command's arguments.
//!
//! One directive a line; `#` starts a comment. A `workload` line starts a workload, and the
//! lines after it, to the next `workload`, describe it:
//!
//! ```text
//! workload NAME METALLIB                 the metallib, a file in the directory
//! buffer LENGTH shared|private FILL      the next buffer, numbered from 0
//! set-buffer INDEX BUFFER [OFFSET]       setBuffer:offset:atIndex: before the next dispatch
//! set-bytes INDEX LENGTH FILL            setBytes:length:atIndex: before the next dispatch
//! constant INDEX TYPE HEX                a function constant of the next dispatch's kernel: its
//!                                        index, raw MTLDataType and value's bytes
//! dispatch FUNCTION threadgroups GX GY GZ TX TY TZ [repeat N]
//! dispatch FUNCTION threads X Y Z TX TY TZ [repeat N]
//! check BUFFER exact                     an output: each word equal between runtimes
//! check BUFFER float TOLERANCE           an output: each word a float, within TOLERANCE
//! ```
//!
//! `FILL` is `zero`, `words SEED` (SplitMix64's words from SEED), `unit SEED` (`float`s in
//! [0, 1) from the same words) or `hex HEX` (exactly LENGTH bytes). A buffer filled `zero` is
//! left as it was made, which Metal makes zero, so the harness holds no copy of it. A buffer's
//! length is a multiple of 16 and a `unit` or `words` fill's a multiple of 4. `repeat N` makes the
//! dispatch N times, its bindings set before the first. Every runtime runs a workload from the
//! same starting contents, and its outputs after the first run must match the first runtime's.

use crate::plan::{
    seeded_words, Binding, BufferPlan, Comparison, Constant, Dispatch, Grid, Output, Plan, Source,
};
use crate::{Result, Size, Storage};
use std::path::{Path, PathBuf};

/// A workload a description names: its lines, made into its plan when it runs, since a plan
/// holds every buffer's starting contents.
#[derive(Clone, Debug)]
pub struct Described {
    /// Its name.
    pub name: String,
    /// Its metallib.
    metallib: PathBuf,
    /// Its lines after the `workload` line: each one's number and its words.
    lines: Vec<(usize, Vec<String>)>,
}

/// The workloads `text` describes, their metallibs in `directory`. Each one's plan is made once
/// here, to check it, and dropped.
pub fn parse(text: &str, directory: &Path) -> Result<Vec<Described>> {
    let mut described: Vec<Described> = Vec::new();
    for (index, content) in text.lines().enumerate() {
        let number = index + 1;
        let words: Vec<String> = (content.split('#').next().unwrap_or("").split_whitespace())
            .map(str::to_owned)
            .collect();
        match words.as_slice() {
            [] => {}
            [workload, name, metallib] if workload == "workload" => {
                if described.iter().any(|known| known.name == *name) {
                    return Err(format!("line {number}: a second workload {name}").into());
                }
                described.push(Described {
                    name: name.clone(),
                    metallib: directory.join(metallib),
                    lines: Vec::new(),
                });
            }
            [workload, ..] if workload == "workload" => {
                return Err(format!("line {number}: workload takes NAME METALLIB").into());
            }
            [directive, ..] => match described.last_mut() {
                Some(workload) => workload.lines.push((number, words.clone())),
                None => return Err(format!("line {number}: {directive} before a workload").into()),
            },
        }
    }
    for workload in &described {
        workload.plan()?;
    }
    Ok(described)
}

impl Described {
    /// Its plan, every buffer's starting contents made.
    pub fn plan(&self) -> Result<Plan> {
        let mut plan = Plan {
            name: self.name.clone(),
            library: Source::Metallib(self.metallib.clone()),
            buffers: Vec::new(),
            dispatches: Vec::new(),
            expect: Vec::new(),
            textures: Vec::new(),
            expect_textures: Vec::new(),
            outputs: Vec::new(),
            per_command_buffer: None,
        };
        let mut lines = self.lines.iter().map(|(number, words)| {
            let words: Vec<&str> = words.iter().map(String::as_str).collect();
            line(&words).map_err(|why| format!("line {number}: {why}"))
        });
        // A dispatch at a time: the lines up to it, its bindings and constants among them.
        loop {
            let (mut bindings, mut constants) = (Vec::new(), Vec::new());
            let dispatch = loop {
                match lines.next().transpose()? {
                    None => break None,
                    Some(Line::Buffer(buffer)) => plan.buffers.push(buffer),
                    Some(Line::Binding(binding)) => bindings.push(binding),
                    Some(Line::Constant(constant)) => constants.push(constant),
                    Some(Line::Output(output)) => plan.outputs.push(output),
                    Some(Line::Dispatch {
                        function,
                        grid,
                        repeat,
                    }) => break Some((function, grid, repeat)),
                }
            };
            let Some((function, grid, repeat)) = dispatch else {
                if !(bindings.is_empty() && constants.is_empty()) && !plan.dispatches.is_empty() {
                    let name = &self.name;
                    return Err(format!(
                        "workload {name} sets a binding or a constant after its last dispatch"
                    )
                    .into());
                }
                break;
            };
            plan.dispatches.push(Dispatch {
                function: function.clone(),
                constants: constants.clone(),
                bindings,
                grid,
            });
            for _ in 1..repeat {
                plan.dispatches.push(Dispatch {
                    function: function.clone(),
                    constants: constants.clone(),
                    bindings: Vec::new(),
                    grid,
                });
            }
        }
        let name = &self.name;
        if plan.dispatches.is_empty() {
            return Err(format!("workload {name} has no dispatch").into());
        }
        plan.validate()?;
        Ok(plan)
    }
}

/// What a line inside a workload adds to it.
enum Line {
    Buffer(BufferPlan),
    Binding(Binding),
    Constant(Constant),
    Dispatch {
        function: String,
        grid: Grid,
        repeat: u64,
    },
    Output(Output),
}

/// The line of `words`, a directive and its arguments.
fn line(words: &[&str]) -> std::result::Result<Line, String> {
    Ok(match words {
        ["buffer", length, storage, fill @ ..] => {
            let length = number(length)?;
            let storage = match *storage {
                "shared" => Storage::Shared,
                "private" => Storage::Private,
                other => return Err(format!("storage {other} is neither shared nor private")),
            };
            Line::Buffer(BufferPlan {
                length,
                storage,
                contents: match fill {
                    ["zero"] => None,
                    fill => Some(filled(length, fill)?),
                },
            })
        }
        ["set-buffer", index, buffer, offset @ ..] => {
            let offset = match offset {
                [] => 0,
                [offset] => number(offset)?,
                _ => return Err("set-buffer takes INDEX BUFFER [OFFSET]".to_owned()),
            };
            Line::Binding(Binding::Buffer {
                index: number(index)?,
                buffer: number(buffer)?,
                offset,
            })
        }
        ["set-bytes", index, length, fill @ ..] => Line::Binding(Binding::Bytes {
            index: number(index)?,
            bytes: filled(number(length)?, fill)?,
        }),
        ["constant", index, data_type, hex] => Line::Constant(Constant {
            index: number(index)?,
            data_type: number(data_type)?,
            value: bytes_of(hex)?,
        }),
        ["dispatch", function, kind, sizes @ ..] => {
            let (sizes, repeat) = match sizes {
                [sizes @ .., "repeat", count] => (sizes, number(count)?),
                sizes => (sizes, 1),
            };
            if repeat == 0 {
                return Err("repeat 0 dispatches nothing".to_owned());
            }
            let [x, y, z, tx, ty, tz] = sizes else {
                return Err("dispatch takes six sizes".to_owned());
            };
            let grid: Size = [number(x)?, number(y)?, number(z)?];
            let threads: Size = [number(tx)?, number(ty)?, number(tz)?];
            let grid = match *kind {
                "threadgroups" => Grid::Threadgroups {
                    groups: grid,
                    threads,
                },
                "threads" => Grid::Threads {
                    threads: grid,
                    per_group: threads,
                },
                other => {
                    return Err(format!(
                        "dispatch kind {other} is neither threadgroups nor threads"
                    ))
                }
            };
            Line::Dispatch {
                function: (*function).to_owned(),
                grid,
                repeat,
            }
        }
        ["check", buffer, "exact"] => Line::Output(Output {
            buffer: number(buffer)?,
            comparison: Comparison::Exact,
        }),
        ["check", buffer, "float", tolerance] => Line::Output(Output {
            buffer: number(buffer)?,
            comparison: Comparison::Floats {
                tolerance: (tolerance.parse())
                    .map_err(|_| format!("tolerance {tolerance} is not a number"))?,
            },
        }),
        _ => return Err(format!("{} is no directive", words.join(" "))),
    })
}

/// A whole number.
fn number<T: std::str::FromStr>(word: &str) -> std::result::Result<T, String> {
    word.parse()
        .map_err(|_| format!("{word} is not a whole number"))
}

/// `length` bytes of the fill `fill`.
fn filled(length: u64, fill: &[&str]) -> std::result::Result<Vec<u8>, String> {
    let length = usize::try_from(length).map_err(|_| format!("{length} bytes do not fit"))?;
    let words = |seed: &str| -> std::result::Result<Vec<u32>, String> {
        if length % 4 != 0 {
            return Err(format!(
                "a fill of words needs a multiple of 4 bytes, not {length}"
            ));
        }
        Ok(seeded_words(number(seed)?, length / 4))
    };
    let bytes: Vec<u8> = match fill {
        ["zero"] => vec![0; length],
        ["words", seed] => (words(seed)?.iter())
            .flat_map(|word| word.to_le_bytes())
            .collect(),
        ["unit", seed] => (words(seed)?.iter())
            .flat_map(|word| ((word >> 8) as f32 / (1u32 << 24) as f32).to_le_bytes())
            .collect(),
        ["hex", hex] => bytes_of(hex)?,
        _ => {
            return Err(format!(
                "fill {} is not zero, words, unit or hex",
                fill.join(" ")
            ))
        }
    };
    if bytes.len() != length {
        return Err(format!("the fill has {} bytes, not {length}", bytes.len()));
    }
    Ok(bytes)
}

/// The bytes `hex` spells, two digits each.
fn bytes_of(hex: &str) -> std::result::Result<Vec<u8>, String> {
    (0..hex.len())
        .step_by(2)
        .map(|at| {
            (hex.get(at..at + 2))
                .and_then(|pair| u8::from_str_radix(pair, 16).ok())
                .ok_or_else(|| format!("{hex} is not whole bytes of hex"))
        })
        .collect()
}
