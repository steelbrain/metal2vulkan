//! Workloads described in a text file, over metallibs in a directory: the harness's input for
//! kernels it holds no source of. Nothing in the repository names such a workload; the
//! description and the metallibs are the command's arguments.
//!
//! One directive a line; `#` starts a comment. A `workload` line starts a workload, and the
//! lines after it, to the next `workload`, describe it:
//!
//! ```text
//! workload NAME METALLIB                 the metallib, a file in the directory
//! buffer LENGTH STORAGE FILL             the next buffer, numbered from 0, its storage shared,
//!                                        managed or private
//! modify BUFFER OFFSET LENGTH            before each run, the CPU writes LENGTH bytes of BUFFER
//!                                        at OFFSET again, its starting bytes there, and marks
//!                                        them with didModifyRange: where managed; a shared or
//!                                        managed BUFFER alone
//! set-buffer INDEX BUFFER [OFFSET]       setBuffer:offset:atIndex: before the next dispatch
//! set-bytes INDEX LENGTH FILL            setBytes:length:atIndex: before the next dispatch
//! threadgroup-length INDEX LENGTH        setThreadgroupMemoryLength:atIndex: before the next
//!                                        dispatch; LENGTH a multiple of 16, as Metal requires
//! linked private FUNCTION...             the next dispatch's pipeline links FUNCTIONs, functions
//!                                        of the metallib, as private functions, each made with
//!                                        the dispatch's constants
//! texture TYPE FORMAT WIDTH HEIGHT ARRAY USAGE STORAGE SOURCE FILL
//!                                        the next texture, numbered from 0: its raw
//!                                        MTLTextureType (2D or 2D array), MTLPixelFormat and
//!                                        MTLTextureUsage, its array length (1 for 2D), its
//!                                        storage (shared, managed or private) and what makes it
//!                                        (plain: the device; heap: a heap of its own; over-buffer:
//!                                        a buffer of its own, 2D alone)
//! set-texture INDEX TEXTURE              setTexture:atIndex: before the next dispatch
//! set-sampler INDEX FILTER ADDRESS COORDS
//!                                        setSamplerState:atIndex: before the next dispatch, of a
//!                                        sampler state of FILTER (nearest or linear) as its min
//!                                        and mag filter, ADDRESS (clamp-to-edge or clamp-to-zero)
//!                                        on every axis, and COORDS normalized or pixel
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
//!
//! A texture has one level and one sample, of a format whose element size the harness knows
//! ([`element_bytes`](crate::plan::element_bytes)). Its `FILL` is `undefined`, `zero` or
//! `words SEED` (SplitMix64's words from SEED, cut to the texture's bytes). `undefined` leaves
//! it as made: Metal leaves a new texture's contents undefined, and so do both runtimes, so a
//! workload whose outputs read such texels before writing them can differ between runtimes. The
//! others are written by `replaceRegion:` slice by slice, rows tightly packed, which the CPU does
//! for a shared or managed texture alone: a private texture takes `undefined` alone. A texture
//! Metal made over an IOSurface is described as a plain shared or managed one; the harness makes
//! none over an IOSurface. The new runtime refuses a heap's texture and one over a buffer: it
//! makes neither yet.

use crate::plan::{
    seeded_words, Binding, BufferPlan, Comparison, Constant, Dispatch, Filter, Grid, Output, Plan,
    SamplerPlan, Source, TexturePlan, TextureSource,
};
use crate::{Result, Size, Storage};
use metal2vulkan::metal::{PixelFormat, SamplerAddressMode, TextureType, TextureUsage};
use std::ops::Range;
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
            per_encoder: None,
            commit_to_complete: false,
            scaled: None,
        };
        let mut lines = self.lines.iter().map(|(number, words)| {
            let words: Vec<&str> = words.iter().map(String::as_str).collect();
            (line(&words).map(|line| (*number, line)))
                .map_err(|why| format!("line {number}: {why}"))
        });
        // A dispatch at a time: the lines up to it, its bindings and constants among them.
        loop {
            let (mut bindings, mut constants, mut linked) = (Vec::new(), Vec::new(), Vec::new());
            let dispatch = loop {
                let Some((number, line)) = lines.next().transpose()? else {
                    break None;
                };
                match line {
                    Line::Buffer(buffer) => plan.buffers.push(buffer),
                    Line::Modify { buffer, range } => {
                        let Some(made) = plan.buffers.get_mut(buffer) else {
                            return Err(format!(
                                "line {number}: modify names buffer {buffer}, which no line \
                                 before it makes"
                            )
                            .into());
                        };
                        made.modified = Some(range);
                        if let Some(problem) = made.modify_problem() {
                            return Err(format!("line {number}: buffer {buffer}: {problem}").into());
                        }
                    }
                    Line::Texture(texture) => plan.textures.push(texture),
                    Line::Binding(binding) => bindings.push(binding),
                    Line::Constant(constant) => constants.push(constant),
                    Line::Linked(functions) => linked.extend(functions),
                    Line::Output(output) => plan.outputs.push(output),
                    Line::Dispatch {
                        function,
                        grid,
                        repeat,
                    } => break Some((function, grid, repeat)),
                }
            };
            let Some((function, grid, repeat)) = dispatch else {
                if !(bindings.is_empty() && constants.is_empty() && linked.is_empty())
                    && !plan.dispatches.is_empty()
                {
                    let name = &self.name;
                    return Err(format!(
                        "workload {name} sets a binding, a constant or a linked function after \
                         its last dispatch"
                    )
                    .into());
                }
                break;
            };
            plan.dispatches.push(Dispatch {
                function: function.clone(),
                linked: linked.clone(),
                constants: constants.clone(),
                bindings,
                grid,
            });
            for _ in 1..repeat {
                plan.dispatches.push(Dispatch {
                    function: function.clone(),
                    linked: linked.clone(),
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
    Modify {
        buffer: usize,
        range: Range<u64>,
    },
    Texture(TexturePlan),
    Binding(Binding),
    Constant(Constant),
    Linked(Vec<String>),
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
                "managed" => Storage::Managed,
                "private" => Storage::Private,
                other => return Err(format!("storage {other} is not shared, managed or private")),
            };
            Line::Buffer(BufferPlan {
                length,
                storage,
                modified: None,
                contents: match fill {
                    ["zero"] => None,
                    fill => Some(filled(length, fill)?),
                },
            })
        }
        ["modify", buffer, offset, length] => {
            let (offset, length): (u64, u64) = (number(offset)?, number(length)?);
            Line::Modify {
                buffer: number(buffer)?,
                range: offset..offset.checked_add(length).ok_or("the range does not fit")?,
            }
        }
        ["modify", ..] => return Err("modify takes BUFFER OFFSET LENGTH".to_owned()),
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
        ["texture", kind, format, width, height, slices, usage, storage, source, fill @ ..]
            if !fill.is_empty() =>
        {
            Line::Texture(texture(
                [kind, format, usage],
                [width, height, slices],
                storage,
                source,
                fill,
            )?)
        }
        ["texture", ..] => {
            return Err(
                "texture takes TYPE FORMAT WIDTH HEIGHT ARRAY USAGE STORAGE SOURCE FILL".to_owned(),
            )
        }
        ["set-texture", index, texture] => Line::Binding(Binding::Texture {
            index: number(index)?,
            texture: number(texture)?,
        }),
        ["set-texture", ..] => return Err("set-texture takes INDEX TEXTURE".to_owned()),
        ["set-sampler", index, filter, address, coordinates] => Line::Binding(Binding::Sampler {
            index: number(index)?,
            sampler: SamplerPlan {
                filter: match *filter {
                    "nearest" => Filter::Nearest,
                    "linear" => Filter::Linear,
                    other => return Err(format!("filter {other} is neither nearest nor linear")),
                },
                address: match *address {
                    "clamp-to-edge" => SamplerAddressMode::ClampToEdge,
                    "clamp-to-zero" => SamplerAddressMode::ClampToZero,
                    other => {
                        return Err(format!(
                            "address mode {other} is neither clamp-to-edge nor clamp-to-zero"
                        ))
                    }
                },
                normalized: match *coordinates {
                    "normalized" => true,
                    "pixel" => false,
                    other => {
                        return Err(format!(
                            "coordinates {other} are neither normalized nor pixel"
                        ))
                    }
                },
            },
        }),
        ["set-sampler", ..] => {
            return Err("set-sampler takes INDEX FILTER ADDRESS COORDS".to_owned())
        }
        ["set-bytes", index, length, fill @ ..] => Line::Binding(Binding::Bytes {
            index: number(index)?,
            bytes: filled(number(length)?, fill)?,
        }),
        ["threadgroup-length", index, length] => Line::Binding(Binding::ThreadgroupMemory {
            index: number(index)?,
            length: match number(length)? {
                length if length % 16 == 0 => length,
                length => {
                    return Err(format!(
                        "threadgroup length {length} is not a multiple of 16"
                    ))
                }
            },
        }),
        ["linked", "private", functions @ ..] if !functions.is_empty() => Line::Linked(
            functions
                .iter()
                .map(|&function| function.to_owned())
                .collect(),
        ),
        ["linked", ..] => return Err("linked takes private FUNCTION...".to_owned()),
        ["threadgroup-length", ..] => {
            return Err("threadgroup-length takes INDEX LENGTH".to_owned())
        }
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

/// The texture of a `texture` line: its raw type, format and usage, its size, storage, source
/// and fill.
fn texture(
    [texture_type, pixel_format, usage]: [&str; 3],
    [width, height, array_length]: [&str; 3],
    storage: &str,
    source: &str,
    fill: &[&str],
) -> std::result::Result<TexturePlan, String> {
    let raw = |word: &str| number::<u32>(word);
    let mut texture = TexturePlan {
        texture_type: TextureType::try_from(raw(texture_type)?).map_err(|why| why.to_string())?,
        pixel_format: PixelFormat::try_from(raw(pixel_format)?).map_err(|why| why.to_string())?,
        width: number(width)?,
        height: number(height)?,
        array_length: number(array_length)?,
        usage: TextureUsage::try_from(raw(usage)?).map_err(|why| why.to_string())?,
        storage: match storage {
            "shared" => Storage::Shared,
            "managed" => Storage::Managed,
            "private" => Storage::Private,
            other => return Err(format!("storage {other} is not shared, managed or private")),
        },
        source: match source {
            "plain" => TextureSource::Plain,
            "heap" => TextureSource::Heap,
            "over-buffer" => TextureSource::OverBuffer,
            other => return Err(format!("source {other} is not plain, heap or over-buffer")),
        },
        texels: None,
        bytes: None,
    };
    if let Some(problem) = texture.problem(false) {
        return Err(format!("the texture: {problem}"));
    }
    let length = (texture.byte_length())
        .and_then(|length| usize::try_from(length).ok())
        .ok_or("the texture's bytes do not fit")?;
    texture.bytes = match fill {
        ["undefined"] => None,
        _ if texture.storage == Storage::Private => {
            return Err("a private texture takes undefined alone: the CPU writes none".to_owned())
        }
        ["zero"] => Some(vec![0; length]),
        ["words", seed] => Some(
            (seeded_words(number(seed)?, length.div_ceil(4)).iter())
                .flat_map(|word| word.to_le_bytes())
                .take(length)
                .collect(),
        ),
        _ => {
            return Err(format!(
                "fill {} is not undefined, zero or words SEED",
                fill.join(" ")
            ))
        }
    };
    Ok(texture)
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
