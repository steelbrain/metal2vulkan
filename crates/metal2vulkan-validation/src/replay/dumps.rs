//! The per-command-buffer compare of two plays (`docs/REPLAY-STREAM.md`, section 10).
//!
//! A play asked to dump command buffers ([`Player::dumping`](super::player::Player::dumping))
//! writes, once each listed command buffer completed, every buffer its encoders named, as three
//! kinds of file in its directory:
//! - `cb<C>-obj<O>.bin`: the buffer's bytes;
//! - `cb<C>-obj<O>.cover`: the ranges of it the player's cover rule (`covers.rs`) held covered
//!   then, as `start..end` byte offsets joined by commas, or `-` for none;
//! - `cb<C>-obj<O>.unread`: where the backend could not read the buffer, why, in place of its
//!   bytes.
//!
//! [`compare_dumps`] reads two plays' dumps and compares each buffer's bytes word by word (4 bytes).
//! It leaves out a word that lies wholly inside a covered range of either play: the cover rule
//! leaves those bytes undefined, and a backend that answers no texture writes covers less
//! (`Backend::texture_writes`). Every other differing word is reported, and so is a dump that
//! only one play has, a buffer that either play could not read, and a difference in length.
//! A whole-object checksum is no such compare: one word under a cover would make a buffer differ,
//! and one word outside it would hide behind a cover elsewhere in the buffer.

use std::collections::BTreeMap;
use std::io;
use std::path::Path;

/// The dump of buffer `object` named by command buffer `command_buffer`: its bytes, or why the
/// backend could not read them, and the ranges of it covered then, written into `dir`.
pub(super) fn write(
    dir: &Path,
    command_buffer: u64,
    object: u64,
    bytes: Result<&[u8], String>,
    covered: &[(u64, u64)],
) -> io::Result<()> {
    let base = dir.join(format!("cb{command_buffer}-obj{object}"));
    match bytes {
        Ok(bytes) => std::fs::write(base.with_extension("bin"), bytes)?,
        Err(why) => std::fs::write(base.with_extension("unread"), why)?,
    }
    let text: Vec<String> = covered.iter().map(|(s, e)| format!("{s}..{e}")).collect();
    let text = match text.is_empty() {
        true => "-".to_owned(),
        false => text.join(","),
    };
    std::fs::write(base.with_extension("cover"), text)
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
}

impl Comparison {
    /// The differing words outside every cover, in all.
    pub fn uncovered(&self) -> u64 {
        self.differences.iter().map(Difference::uncovered).sum()
    }

    /// Whether the plays agree: no differing word outside every cover, every dump in both plays,
    /// and each read by both.
    pub fn agrees(&self) -> bool {
        self.uncovered() == 0 && self.only_one.is_empty() && self.unread.is_empty()
    }

    /// The comparison's lines: one per buffer that differs outside every cover, one per dump only
    /// one play has or a play could not read, then the summary.
    pub fn lines(&self) -> Vec<String> {
        let mut lines = Vec::new();
        for d in self.differences.iter().filter(|d| d.uncovered() > 0) {
            let first = d.first_uncovered.map_or("-".to_owned(), |w| w.to_string());
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
        let covered: u64 = self.differences.iter().map(|d| d.covered).sum();
        lines.push(format!(
            "compared {} buffers: {} differ, {} words outside every cover, {covered} covered; \
             {} dumps in one play only, {} unread: {}",
            self.compared,
            self.differences.len(),
            self.uncovered(),
            self.only_one.len(),
            self.unread.len(),
            match self.agrees() {
                true => "agree",
                false => "DIFFER",
            }
        ));
        lines
    }
}

/// What one play's directory holds of one dump.
#[derive(Default)]
struct Held {
    bytes: Option<Vec<u8>>,
    unread: Option<String>,
    covered: Vec<(u64, u64)>,
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
            "bin" => held.bytes = Some(std::fs::read(&path)?),
            "unread" => held.unread = Some(std::fs::read_to_string(&path)?),
            "cover" => held.covered = ranges(&std::fs::read_to_string(&path)?, &path)?,
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
    let longest = a.len().max(b.len()).div_ceil(4);
    let (mut differing, mut under, mut first) = (0, 0, None);
    for word in 0..longest {
        let at = word * 4;
        let side = |bytes: &[u8]| bytes.get(at..(at + 4).min(bytes.len())).map(<[u8]>::to_vec);
        if side(a) == side(b) {
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
        out.compared += 1;
        let covered = [&x.covered[..], &y.covered[..]].concat();
        let (differing, under, first) = words(bx, by, &covered);
        if differing > 0 {
            out.differences.push(Difference {
                command_buffer: key.0,
                object: key.1,
                differing,
                covered: under,
                first_uncovered: first,
                lengths: (bx.len() as u64, by.len() as u64),
            });
        }
    }
    Ok(out)
}
