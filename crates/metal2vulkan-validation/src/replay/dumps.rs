//! The per-command-buffer compare of two plays (`docs/REPLAY-STREAM.md`, section 10).
//!
//! A play asked to dump command buffers ([`Player::dumping`](super::player::Player::dumping))
//! writes, once each listed command buffer completed, every buffer its encoders named, every
//! colour attachment its render passes stored and every texture its compute encoders bound
//! ([`Stores`]), as these kinds of file in its directory:
//! - `cb<C>-obj<O>.bin`: the buffer's bytes; a texture's, the parts of it the command buffer's
//!   passes stored (a level, a slice and a depth plane each; a layered pass's each slice it
//!   renders), in the order of [`Part`], and a compute-bound texture's first level and slice,
//!   each laid out as a check reads a texture's first level and slice;
//! - `cb<C>-obj<O>.cover`: the ranges of it the player's cover rule (`covers.rs`) held covered
//!   then, as `start..end` byte offsets joined by commas, or `-` for none. A texture's is the
//!   whole of its bytes, `0..N`, where the rule held any byte of it covered (`Covers::covered`),
//!   and `-` otherwise: the rule's ranges are the texture's bytes in its heap, which its parts
//!   here are not laid out as;
//! - `cb<C>-obj<O>.unread`: where the backend could not read the buffer or texture, why, in place
//!   of its bytes (a multisample texture a pass stores without resolving it, for one);
//! - `cb<C>-obj<O>.copy`: for a buffer the play keeps two copies of, which one the bytes are
//!   ([`ReadCopy::label`]): its CPU copy, or its device copy, read back.
//!
//! [`compare_dumps`] reads two plays' dumps and compares each buffer's bytes word by word (4 bytes).
//! It leaves out a word that lies wholly inside a covered range of either play, so all of a
//! texture that either play's rule held covered: the cover rule leaves those bytes undefined, and
//! a backend that answers no texture writes covers less (`Backend::texture_writes`). Every other
//! differing word is reported, and so is a dump that only one play has, a buffer that either play
//! could not read, and a difference in length. So is a buffer the two plays read different copies
//! of: the CPU's bytes of one and the device's of the other are no compare of either. The summary
//! counts the buffers compared by the copy both read.
//! A whole-object checksum is no such compare: one word under a cover would make a buffer differ,
//! and one word outside it would hide behind a cover elsewhere in the buffer.
//!
//! A dump holds colour attachments and compute-bound textures only. A depth or stencil attachment
//! is none: a texture of a
//! combined depth and stencil format reads back only one aspect at a time, through a copy with
//! that aspect's blit option, which the dump's read of a texture part does not make.

use super::player::ReadCopy;
use super::stream::Call;
use super::value::{Name, Value};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::io;
use std::path::{Path, PathBuf};

/// The command buffers a play dumps ([`Player::dumping`](super::player::Player::dumping)), by
/// stream id: what `--dump-command-buffers` lists.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CommandBufferList {
    /// Each range's first id and its last, ranges that overlap or touch merged into one.
    ranges: BTreeMap<u64, u64>,
}

impl CommandBufferList {
    /// The list `listed` names: items joined by commas, in any order, each `FIRST..LAST` (both
    /// ends held, `FIRST` at most `LAST`) or one id. `None` where any item is neither.
    pub fn parse(listed: &str) -> Option<CommandBufferList> {
        let mut list = CommandBufferList::default();
        for item in listed.split(',') {
            let (first, last) = match item.split_once("..") {
                Some((first, last)) => (first.parse().ok()?, last.parse().ok()?),
                None => {
                    let id = item.parse().ok()?;
                    (id, id)
                }
            };
            if first > last {
                return None;
            }
            list.add(first, last);
        }
        Some(list)
    }

    /// Whether the list holds command buffer `id`.
    pub fn contains(&self, id: u64) -> bool {
        (self.ranges.range(..=id).next_back()).is_some_and(|(_, &last)| id <= last)
    }

    /// Hold `first..=last`, merged with the ranges it overlaps or touches.
    fn add(&mut self, mut first: u64, mut last: u64) {
        let merged: Vec<u64> = (self.ranges.range(..=last.saturating_add(1)).rev())
            .take_while(|&(_, &end)| end.saturating_add(1) >= first)
            .map(|(&start, _)| start)
            .collect();
        for start in merged {
            let end = self.ranges.remove(&start).unwrap_or(start);
            first = first.min(start);
            last = last.max(end);
        }
        self.ranges.insert(first, last);
    }
}

/// The dump of buffer `object` named by command buffer `command_buffer`: its bytes, or why the
/// backend could not read them, which of its two copies they are where it has two, and the ranges
/// of it covered then, written into `dir`.
pub(super) fn write(
    dir: &Path,
    command_buffer: u64,
    object: u64,
    bytes: Result<&[u8], String>,
    covered: &[(u64, u64)],
    copy: Option<ReadCopy>,
) -> io::Result<()> {
    let base = dir.join(format!("cb{command_buffer}-obj{object}"));
    match bytes {
        Ok(bytes) => std::fs::write(base.with_extension("bin"), bytes)?,
        Err(why) => std::fs::write(base.with_extension("unread"), why)?,
    }
    if let Some(copy) = copy {
        std::fs::write(base.with_extension("copy"), copy.label())?;
    }
    let text: Vec<String> = covered.iter().map(|(s, e)| format!("{s}..{e}")).collect();
    let text = match text.is_empty() {
        true => "-".to_owned(),
        false => text.join(","),
    };
    std::fs::write(base.with_extension("cover"), text)
}

/// A level, slice and depth plane of a texture: the part of it a render pass attachment names
/// (`level`, `slice` and `depthPlane`, or `resolveLevel`, `resolveSlice` and
/// `resolveDepthPlane`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Part {
    pub level: u64,
    pub slice: u64,
    pub plane: u64,
}

/// A render pass descriptor's colour attachment, as its setter records set it: its texture and
/// part, its resolve texture and part (0 for none), and its store action (`MTLStoreAction`'s raw
/// value; 0, `MTLStoreActionDontCare`, until set).
#[derive(Clone, Copy, Debug, Default)]
struct Colour {
    texture: u64,
    part: Part,
    resolve: u64,
    resolve_part: Part,
    store: u64,
}

impl Colour {
    /// The texture parts the attachment stores under store action `store` in a pass of `layers`
    /// layers (its `renderTargetArrayLength`): its texture's under `MTLStoreActionStore` and
    /// `MTLStoreActionStoreAndMultisampleResolve`, its resolve texture's under
    /// `MTLStoreActionMultisampleResolve` and that one; each `layers` slices from the part's own
    /// slice, its slice alone at 0 and 1, as Metal renders them.
    fn stored(&self, store: u64, layers: u64, out: &mut Vec<(u64, Part)>) {
        let mut each = |texture: u64, part: Part| {
            let slices = (0..layers.max(1)).map(|layer| Part {
                slice: part.slice.saturating_add(layer),
                ..part
            });
            out.extend(slices.map(|part| (texture, part)));
        };
        if matches!(store, 1 | 3) && self.texture != 0 {
            each(self.texture, self.part);
        }
        if matches!(store, 2 | 3) && self.resolve != 0 {
            each(self.resolve, self.resolve_part);
        }
    }

    /// `f`, a `m2v.setColorAttachment` record's map, set on the attachment: each property it
    /// names; a nil one keeps what the attachment had.
    fn set(&mut self, f: &[(Name, Value)]) {
        for (key, value) in f {
            let (object, uint) = (value.as_object(), value.as_uint());
            match (&**key, object, uint) {
                ("texture", Some(id), _) => self.texture = id,
                ("resolveTexture", Some(id), _) => self.resolve = id,
                ("level", _, Some(v)) => self.part.level = v,
                ("slice", _, Some(v)) => self.part.slice = v,
                ("depthPlane", _, Some(v)) => self.part.plane = v,
                ("resolveLevel", _, Some(v)) => self.resolve_part.level = v,
                ("resolveSlice", _, Some(v)) => self.resolve_part.slice = v,
                ("resolveDepthPlane", _, Some(v)) => self.resolve_part.plane = v,
                ("storeAction", _, Some(v)) => self.store = v,
                _ => {}
            }
        }
    }
}

/// The colour attachments a play's render passes store: each render pass descriptor's, as its
/// `m2v.setColorAttachment` records set them (`docs/REPLAY-STREAM.md`, section 9), with the
/// layers its `m2v.setPass` records set (`renderTargetArrayLength`), and each render encoder's
/// whose store action is `MTLStoreActionUnknown`, with its pass's layers, until the encoder's
/// `setColorStoreAction:atIndex:` names the action; and the compute encoders, whose bound
/// textures a dump reads too.
#[derive(Default)]
pub(super) struct Stores {
    passes: HashMap<u64, BTreeMap<u64, Colour>>,
    layers: HashMap<u64, u64>,
    unknown: HashMap<u64, BTreeMap<u64, (Colour, u64)>>,
    /// The compute encoders made and not yet ended.
    computes: HashSet<u64>,
}

impl Stores {
    /// `call`, played: a colour attachment's setter record on a render pass descriptor, and the
    /// layers a pass's own setter record sets, kept. The texture parts a render pass stores where
    /// `call` makes its encoder (`renderCommandEncoderWithDescriptor:`) or names an attachment's
    /// store action left unknown (`setColorStoreAction:atIndex:`), into `out`; and each texture a compute encoder binds
    /// (`setTexture:atIndex:` and `setTextures:withRange:`), as its first level and slice
    /// ([`Part::default`]), a nil one left out.
    pub(super) fn saw(&mut self, call: &Call, out: &mut Vec<(u64, Part)>) {
        let uint = |at: usize| call.args.get(at).and_then(Value::as_uint);
        match &*call.selector {
            "m2v.setColorAttachment" => {
                let (Some(at), Some(f)) = (uint(0), call.args.get(1).and_then(Value::as_map))
                else {
                    return;
                };
                let pass = self.passes.entry(call.receiver).or_default();
                pass.entry(at).or_default().set(f);
            }
            "m2v.setPass" => {
                let Some(f) = call.args.first().and_then(Value::as_map) else {
                    return;
                };
                let set = f
                    .iter()
                    .find(|(key, _)| &**key == "renderTargetArrayLength");
                if let Some(layers) = set.and_then(|(_, value)| value.as_uint()) {
                    self.layers.insert(call.receiver, layers);
                }
            }
            "renderCommandEncoderWithDescriptor:" if call.result != 0 => {
                let Some(pass) = call.args.first().and_then(Value::as_object) else {
                    return;
                };
                let Some(colours) = self.passes.get(&pass) else {
                    return;
                };
                let layers = self.layers.get(&pass).copied().unwrap_or(0);
                for (&at, colour) in colours {
                    match colour.store {
                        4 => {
                            let unknown = self.unknown.entry(call.result).or_default();
                            unknown.insert(at, (*colour, layers));
                        }
                        store => colour.stored(store, layers, out),
                    }
                }
            }
            "setColorStoreAction:atIndex:" => {
                let (Some(store), Some(at)) = (uint(0), uint(1)) else {
                    return;
                };
                let unknown = self.unknown.get_mut(&call.receiver);
                if let Some((colour, layers)) = unknown.and_then(|unknown| unknown.remove(&at)) {
                    colour.stored(store, layers, out);
                }
            }
            "computeCommandEncoder"
            | "computeCommandEncoderWithDispatchType:"
            | "computeCommandEncoderWithDescriptor:"
                if call.result != 0 =>
            {
                self.computes.insert(call.result);
            }
            "setTexture:atIndex:" | "setTextures:withRange:"
                if self.computes.contains(&call.receiver) =>
            {
                let bound = match call.args.first() {
                    Some(Value::List(textures)) => textures.iter().collect(),
                    first => Vec::from_iter(first),
                };
                let bound = bound.into_iter().filter_map(Value::as_object);
                out.extend(bound.filter(|&t| t != 0).map(|t| (t, Part::default())));
            }
            "endEncoding" => {
                self.unknown.remove(&call.receiver);
                self.computes.remove(&call.receiver);
            }
            _ => {}
        }
    }

    /// Object `id` released: a render pass descriptor's attachments and layers are dropped.
    pub(super) fn released(&mut self, id: u64) {
        self.passes.remove(&id);
        self.layers.remove(&id);
    }
}

/// One buffer's dump in two plays that differ outside every cover, or in length.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Difference {
    pub command_buffer: u64,
    pub object: u64,
    /// The words that differ, those past the shorter dump's end included.
    pub differing: u64,
    /// Of them, those wholly inside a range either play held covered.
    pub covered: u64,
    /// The first differing word outside every cover, by its index (its byte offset over 4).
    pub first_uncovered: Option<u64>,
    /// That word in each play's dump, little-endian, a short end's bytes read with zeros above
    /// them: none past that dump's end, or where no word differs outside every cover.
    pub first_values: (Option<u32>, Option<u32>),
    /// The two dumps' lengths in bytes.
    pub lengths: (u64, u64),
}

impl Difference {
    /// The differing words outside every cover.
    pub fn uncovered(&self) -> u64 {
        self.differing - self.covered
    }
}

/// Two plays' dumps compared ([`compare_dumps`]).
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Comparison {
    /// The buffers both plays dumped and read, each compared word by word.
    pub compared: u64,
    /// Every compared buffer that differs, covered words or not, by command buffer and buffer.
    pub differences: Vec<Difference>,
    /// The dumps only one play has (`cb<C>-obj<O>`, and `a` or `b`).
    pub only_one: Vec<(String, char)>,
    /// The dumps a play could not read (`cb<C>-obj<O>`, the play, and why).
    pub unread: Vec<(String, char, String)>,
    /// The buffers the two plays read different copies of (`cb<C>-obj<O>`, then the copy each
    /// read: a `.copy` file's words, or `one copy` where it has none), not compared.
    pub copies: Vec<(String, String, String)>,
    /// Of the buffers compared, those both plays read the CPU copy of, and those both read the
    /// device copy of, read back.
    pub cpu_copies: u64,
    pub device_copies: u64,
}

impl Comparison {
    /// The differing words outside every cover, in all.
    pub fn uncovered(&self) -> u64 {
        self.differences.iter().map(Difference::uncovered).sum()
    }

    /// Whether the plays agree: no differing word outside every cover, every dump in both plays,
    /// each read by both, and of the same copy.
    pub fn agrees(&self) -> bool {
        self.uncovered() == 0
            && self.only_one.is_empty()
            && self.unread.is_empty()
            && self.copies.is_empty()
    }

    /// The comparison's lines: one per buffer that differs outside every cover, one per dump only
    /// one play has or a play could not read, then the summary.
    pub fn lines(&self) -> Vec<String> {
        let mut lines = Vec::new();
        for d in self.differences.iter().filter(|d| d.uncovered() > 0) {
            let value = |v: Option<u32>| v.map_or("none".to_owned(), |v| format!("{v:#010x}"));
            let first = d.first_uncovered.map_or("-".to_owned(), |w| {
                let (a, b) = d.first_values;
                format!("{w} (a {}, b {})", value(a), value(b))
            });
            lines.push(format!(
                "differs cb #{} obj #{}: {} words outside every cover, the first word {first}; \
                 {} covered; lengths {} and {}",
                d.command_buffer,
                d.object,
                d.uncovered(),
                d.covered,
                d.lengths.0,
                d.lengths.1
            ));
        }
        for (dump, play) in &self.only_one {
            lines.push(format!("only in {play}: {dump}"));
        }
        for (dump, play, why) in &self.unread {
            lines.push(format!("unread in {play}: {dump}: {why}"));
        }
        for (dump, a, b) in &self.copies {
            lines.push(format!("copies differ: {dump}: a read its {a}, b its {b}"));
        }
        let covered: u64 = self.differences.iter().map(|d| d.covered).sum();
        lines.push(format!(
            "compared {} buffers ({} as both plays' CPU copies, {} as their device copies, read \
             back): {} differ, {} words outside every cover, {covered} covered; {} dumps in one \
             play only, {} unread, {} of different copies: {}",
            self.compared,
            self.cpu_copies,
            self.device_copies,
            self.differences.len(),
            self.uncovered(),
            self.only_one.len(),
            self.unread.len(),
            self.copies.len(),
            match self.agrees() {
                true => "agree",
                false => "DIFFER",
            }
        ));
        lines
    }
}

/// What one play's directory holds of one dump. The bytes stay in their file until the compare
/// reads them: two plays' dumps of a long range can be far larger than memory.
#[derive(Default)]
struct Held {
    bytes: Option<PathBuf>,
    unread: Option<String>,
    covered: Vec<(u64, u64)>,
    /// The `.copy` file's words: which of its two copies the bytes are.
    copy: Option<String>,
}

/// The dumps in `dir`, by command buffer and buffer.
fn read_dir(dir: &Path) -> io::Result<BTreeMap<(u64, u64), Held>> {
    let mut out: BTreeMap<(u64, u64), Held> = BTreeMap::new();
    for entry in std::fs::read_dir(dir)? {
        let path = entry?.path();
        let (Some(stem), Some(extension)) = (
            path.file_stem().and_then(|s| s.to_str()),
            path.extension().and_then(|s| s.to_str()),
        ) else {
            continue;
        };
        let Some(key) = key(stem) else {
            continue;
        };
        let held = out.entry(key).or_default();
        match extension {
            "bin" => held.bytes = Some(path),
            "unread" => held.unread = Some(std::fs::read_to_string(&path)?),
            "cover" => held.covered = ranges(&std::fs::read_to_string(&path)?, &path)?,
            "copy" => held.copy = Some(std::fs::read_to_string(&path)?.trim().to_owned()),
            _ => {}
        }
    }
    Ok(out)
}

/// The command buffer and buffer a dump's file stem `cb<C>-obj<O>` names.
fn key(stem: &str) -> Option<(u64, u64)> {
    let (command_buffer, object) = stem.strip_prefix("cb")?.split_once("-obj")?;
    Some((command_buffer.parse().ok()?, object.parse().ok()?))
}

/// The covered ranges a `.cover` file's `text` holds.
fn ranges(text: &str, path: &Path) -> io::Result<Vec<(u64, u64)>> {
    let text = text.trim();
    if text == "-" {
        return Ok(Vec::new());
    }
    let bad = || io::Error::new(io::ErrorKind::InvalidData, format!("{}", path.display()));
    let range = |range: &str| -> io::Result<(u64, u64)> {
        let (start, end) = range.split_once("..").ok_or_else(bad)?;
        Ok((
            start.parse().map_err(|_| bad())?,
            end.parse().map_err(|_| bad())?,
        ))
    };
    text.split(',').map(range).collect()
}

/// `a` against `b` word by word, a word wholly inside one of `covered` left out: the differing
/// words, those covered, and the first differing word outside every cover.
pub(super) fn words(a: &[u8], b: &[u8], covered: &[(u64, u64)]) -> (u64, u64, Option<u64>) {
    let (mut differing, mut under, mut first) = (0, 0, None);
    if a == b {
        return (differing, under, first);
    }
    /// The word of `bytes` at byte `at`: none past its end, fewer than 4 bytes at a short end.
    fn side(bytes: &[u8], at: usize) -> Option<&[u8]> {
        bytes.get(at..(at + 4).min(bytes.len()))
    }
    let longest = a.len().max(b.len()).div_ceil(4);
    for word in 0..longest {
        let at = word * 4;
        if side(a, at) == side(b, at) {
            continue;
        }
        differing += 1;
        let (start, end) = (at as u64, at as u64 + 4);
        match covered.iter().any(|&(s, e)| s <= start && end <= e) {
            true => under += 1,
            false => {
                first.get_or_insert(word as u64);
            }
        }
    }
    (differing, under, first)
}

/// Word `index` of `bytes`, little-endian, a short end's bytes read with zeros above them: none
/// past its end.
fn word_at(bytes: &[u8], index: u64) -> Option<u32> {
    let at = usize::try_from(index).ok()?.checked_mul(4)?;
    let word = bytes
        .get(at..(at + 4).min(bytes.len()))
        .filter(|w| !w.is_empty())?;
    let mut four = [0; 4];
    four[..word.len()].copy_from_slice(word);
    Some(u32::from_le_bytes(four))
}

/// The dumps of two plays, in `a` and `b`, compared.
pub fn compare_dumps(a: &Path, b: &Path) -> io::Result<Comparison> {
    let (mut a, mut b) = (read_dir(a)?, read_dir(b)?);
    let mut out = Comparison::default();
    let keys: std::collections::BTreeSet<(u64, u64)> = a.keys().chain(b.keys()).copied().collect();
    for key in keys {
        let name = format!("cb{}-obj{}", key.0, key.1);
        let (x, y) = match (a.remove(&key), b.remove(&key)) {
            (Some(x), Some(y)) => (x, y),
            (x, _) => {
                out.only_one
                    .push((name, if x.is_some() { 'a' } else { 'b' }));
                continue;
            }
        };
        for (held, play) in [(&x, 'a'), (&y, 'b')] {
            if held.bytes.is_none() {
                let why = held.unread.as_deref().map_or("no bytes", str::trim);
                out.unread.push((name.clone(), play, why.to_owned()));
            }
        }
        let (Some(bx), Some(by)) = (&x.bytes, &y.bytes) else {
            continue;
        };
        let (bx, by) = (std::fs::read(bx)?, std::fs::read(by)?);
        if x.copy != y.copy {
            let copy = |held: &Held| held.copy.clone().unwrap_or_else(|| "one copy".to_owned());
            out.copies.push((name, copy(&x), copy(&y)));
            continue;
        }
        out.compared += 1;
        match x.copy.as_deref() {
            Some(copy) if copy == ReadCopy::Cpu.label() => out.cpu_copies += 1,
            Some(copy) if copy == ReadCopy::Device.label() => out.device_copies += 1,
            _ => {}
        }
        let covered = [&x.covered[..], &y.covered[..]].concat();
        let (differing, under, first) = words(&bx, &by, &covered);
        if differing > 0 {
            out.differences.push(Difference {
                command_buffer: key.0,
                object: key.1,
                differing,
                covered: under,
                first_uncovered: first,
                first_values: first.map_or((None, None), |w| (word_at(&bx, w), word_at(&by, w))),
                lengths: (bx.len() as u64, by.len() as u64),
            });
        }
    }
    Ok(out)
}
