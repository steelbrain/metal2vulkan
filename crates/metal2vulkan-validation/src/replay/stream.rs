//! The container (`docs/REPLAY-STREAM.md`, sections 1 and 3): a header, then name, blob and call
//! records. [`StreamWriter`] appends them; [`StreamReader`] reads a stream's segments in order and
//! keeps only where each blob lies, so a long stream's data is read when a call needs it.

use super::value::{put_value, put_varint, Bad, Hash, Name, Payload, Value};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::fs::File;
use std::io::{self, BufReader, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// The header's magic.
pub const MAGIC: &[u8; 8] = b"M2VRSTRM";
/// The format version this crate reads and writes.
pub const VERSION: u32 = 2;
const HEADER_LEN: u64 = 16;

const KIND_NAME: u8 = 1;
const KIND_BLOB: u8 = 2;
const KIND_CALL: u8 = 3;

/// One call record, its names resolved.
#[derive(Clone, Debug, PartialEq)]
pub struct Call {
    pub selector: Name,
    /// The receiver's id, 0 for none.
    pub receiver: u64,
    /// The id of the object the call makes, 0 for none.
    pub result: u64,
    pub args: Vec<Value>,
    pub answers: Vec<(Name, Value)>,
}

impl Call {
    /// The answer named `key`.
    pub fn answer(&self, key: &str) -> Option<&Value> {
        self.answers
            .iter()
            .find(|(k, _)| &**k == key)
            .map(|(_, v)| v)
    }
}

impl fmt::Display for Call {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.result != 0 {
            write!(f, "#{} = ", self.result)?;
        }
        if self.receiver != 0 {
            write!(f, "#{} ", self.receiver)?;
        }
        write!(f, "{}", self.selector)?;
        for arg in &self.args {
            write!(f, " {arg}")?;
        }
        for (key, value) in &self.answers {
            write!(f, " -> {key}: {value}")?;
        }
        Ok(())
    }
}

/// Appends records to `W`: each name once, each blob once.
pub struct StreamWriter<W: Write> {
    out: W,
    names: HashMap<String, u64>,
    blobs: HashSet<Hash>,
}

impl<W: Write> StreamWriter<W> {
    /// A stream on `out`, its header written.
    pub fn new(mut out: W) -> io::Result<Self> {
        out.write_all(MAGIC)?;
        out.write_all(&VERSION.to_le_bytes())?;
        out.write_all(&0u32.to_le_bytes())?;
        Ok(StreamWriter {
            out,
            names: HashMap::new(),
            blobs: HashSet::new(),
        })
    }

    fn record(&mut self, kind: u8, payload: &[u8]) -> io::Result<()> {
        let mut head = vec![kind];
        put_varint(&mut head, payload.len() as u64);
        self.out.write_all(&head)?;
        self.out.write_all(payload)
    }

    /// `data`'s hash, its blob record written unless the stream holds it.
    pub fn blob(&mut self, data: &[u8]) -> io::Result<Hash> {
        let hash: Hash = Sha256::digest(data).into();
        if self.blobs.insert(hash) {
            let mut payload = hash.to_vec();
            payload.extend_from_slice(data);
            self.record(KIND_BLOB, &payload)?;
        }
        Ok(hash)
    }

    /// One call record, the name records it needs first.
    pub fn call(
        &mut self,
        selector: &str,
        receiver: u64,
        result: u64,
        args: &[Value],
        answers: &[(&str, Value)],
    ) -> io::Result<()> {
        let mut fresh = Vec::new();
        let names = &mut self.names;
        let mut intern = |name: &str| -> u64 {
            let next = names.len() as u64 + 1;
            *names.entry(name.to_string()).or_insert_with(|| {
                fresh.push(name.to_string());
                next
            })
        };
        let mut payload = Vec::new();
        put_varint(&mut payload, intern(selector));
        put_varint(&mut payload, receiver);
        put_varint(&mut payload, result);
        put_varint(&mut payload, args.len() as u64);
        for arg in args {
            put_value(&mut payload, arg, &mut intern);
        }
        put_varint(&mut payload, answers.len() as u64);
        for (key, value) in answers {
            put_varint(&mut payload, intern(key));
            put_value(&mut payload, value, &mut intern);
        }
        for name in fresh {
            self.record(KIND_NAME, name.as_bytes())?;
        }
        self.record(KIND_CALL, &payload)
    }

    /// The writer, flushed.
    pub fn finish(mut self) -> io::Result<W> {
        self.out.flush()?;
        Ok(self.out)
    }
}

/// Why a stream cannot be read.
#[derive(Debug)]
pub enum StreamError {
    Io(PathBuf, io::Error),
    /// A segment whose header is not this version's.
    Header(PathBuf),
    /// A record cut short in a segment that is not the last.
    Cut {
        segment: PathBuf,
        offset: u64,
    },
    UnknownKind {
        segment: PathBuf,
        offset: u64,
        kind: u8,
    },
    /// A record that does not decode.
    Bad {
        segment: PathBuf,
        offset: u64,
        why: Bad,
    },
    /// A blob no record and no blob directory holds.
    MissingBlob(Hash),
    /// A file of the blob directory whose bytes are not the blob its name hashes.
    WrongBlob(PathBuf),
}

impl fmt::Display for StreamError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StreamError::Io(path, error) => write!(f, "{}: {error}", path.display()),
            StreamError::Header(path) => {
                write!(f, "{}: not a version {VERSION} stream", path.display())
            }
            StreamError::Cut { segment, offset } => write!(
                f,
                "{}: a record cut short at {offset}, in a segment a later one follows",
                segment.display()
            ),
            StreamError::UnknownKind {
                segment,
                offset,
                kind,
            } => write!(f, "{}: record kind {kind} at {offset}", segment.display()),
            StreamError::Bad {
                segment,
                offset,
                why,
            } => write!(f, "{}: the record at {offset}: {why}", segment.display()),
            StreamError::MissingBlob(hash) => {
                write!(
                    f,
                    "blob {} is in no record and no blob directory",
                    super::value::hex(hash)
                )
            }
            StreamError::WrongBlob(path) => {
                write!(f, "{}: not the blob its name hashes", path.display())
            }
        }
    }
}

/// Where a blob's data lies.
#[derive(Clone, Copy)]
struct BlobAt {
    segment: usize,
    offset: u64,
    length: u64,
}

struct Segment {
    path: PathBuf,
    reader: BufReader<File>,
    length: u64,
}

/// Reads a stream's segments in order.
pub struct StreamReader {
    segments: Vec<Segment>,
    at: usize,
    offset: u64,
    names: Vec<Name>,
    blobs: HashMap<Hash, BlobAt>,
    blob_dir: Option<PathBuf>,
    truncated: Option<(PathBuf, u64)>,
}

/// The segments of the stream whose first file is `first`: `first` alone, or, where it is named
/// `NAME.0000.m2vr`, it and each `NAME.NNNN.m2vr` after it with no number missing.
pub fn segments(first: &Path) -> Vec<PathBuf> {
    let name = first.file_name().and_then(|n| n.to_str()).unwrap_or("");
    let Some(stem) = name.strip_suffix(".0000.m2vr") else {
        return vec![first.to_path_buf()];
    };
    let mut found = vec![first.to_path_buf()];
    for number in 1.. {
        let next = first.with_file_name(format!("{stem}.{number:04}.m2vr"));
        if !next.is_file() {
            break;
        }
        found.push(next);
    }
    found
}

impl StreamReader {
    /// A reader of `segments`, in order, with each header checked; a blob none of them holds is
    /// read from `blob_dir`.
    pub fn open(segments: &[PathBuf], blob_dir: Option<PathBuf>) -> Result<Self, StreamError> {
        let mut opened = Vec::new();
        for path in segments {
            let io = |error| StreamError::Io(path.clone(), error);
            let file = File::open(path).map_err(io)?;
            let length = file.metadata().map_err(io)?.len();
            let mut reader = BufReader::new(file);
            let mut header = [0u8; HEADER_LEN as usize];
            if reader.read_exact(&mut header).is_err()
                || &header[..8] != MAGIC
                || header[8..12] != VERSION.to_le_bytes()
            {
                return Err(StreamError::Header(path.clone()));
            }
            opened.push(Segment {
                path: path.clone(),
                reader,
                length,
            });
        }
        Ok(StreamReader {
            segments: opened,
            at: 0,
            offset: HEADER_LEN,
            names: Vec::new(),
            blobs: HashMap::new(),
            blob_dir,
            truncated: None,
        })
    }

    /// Where the last segment's final record was cut short, if it was.
    pub fn truncated(&self) -> Option<&(PathBuf, u64)> {
        self.truncated.as_ref()
    }

    /// The next call record, or `None` at the stream's end. Name and blob records are taken on
    /// the way.
    pub fn next_call(&mut self) -> Result<Option<Call>, StreamError> {
        loop {
            let last = self.at + 1 == self.segments.len();
            let Some(segment) = self.segments.get_mut(self.at) else {
                return Ok(None);
            };
            let start = self.offset;
            if start == segment.length {
                self.at += 1;
                self.offset = HEADER_LEN;
                continue;
            }
            let (kind, length, head) = match record_head(segment, start) {
                Ok(Some(found)) => found,
                Ok(None) => return self.cut(start, last),
                Err(error) => return Err(StreamError::Io(segment.path.clone(), error)),
            };
            let body = start + head;
            if length > segment.length - body {
                return self.cut(start, last);
            }
            let segment = &mut self.segments[self.at];
            let io = |error| StreamError::Io(segment.path.clone(), error);
            self.offset = body + length;
            match kind {
                KIND_NAME => {
                    let text = read_vec(&mut segment.reader, length).map_err(io)?;
                    let text = String::from_utf8(text).map_err(|e| StreamError::Bad {
                        segment: segment.path.clone(),
                        offset: start,
                        why: e.to_string(),
                    })?;
                    self.names.push(Arc::from(text));
                }
                KIND_BLOB => {
                    let bad = |why: &str| StreamError::Bad {
                        segment: segment.path.clone(),
                        offset: start,
                        why: why.to_string(),
                    };
                    if length < 32 {
                        return Err(bad("a blob shorter than its hash"));
                    }
                    let mut hash = [0u8; 32];
                    segment.reader.read_exact(&mut hash).map_err(io)?;
                    let at = BlobAt {
                        segment: self.at,
                        offset: body + 32,
                        length: length - 32,
                    };
                    if self.blobs.insert(hash, at).is_some() {
                        return Err(bad("a blob written twice"));
                    }
                    segment
                        .reader
                        .seek(SeekFrom::Start(self.offset))
                        .map_err(io)?;
                }
                KIND_CALL => {
                    let payload = read_vec(&mut segment.reader, length).map_err(io)?;
                    let call = self.decode(&payload).map_err(|why| StreamError::Bad {
                        segment: self.segments[self.at].path.clone(),
                        offset: start,
                        why,
                    })?;
                    return Ok(Some(call));
                }
                kind => {
                    return Err(StreamError::UnknownKind {
                        segment: segment.path.clone(),
                        offset: start,
                        kind,
                    })
                }
            }
        }
    }

    fn cut(&mut self, offset: u64, last: bool) -> Result<Option<Call>, StreamError> {
        let segment = self.segments[self.at].path.clone();
        if !last {
            return Err(StreamError::Cut { segment, offset });
        }
        self.truncated = Some((segment, offset));
        self.at = self.segments.len();
        Ok(None)
    }

    fn decode(&self, payload: &[u8]) -> Result<Call, Bad> {
        let name = |id: u64| -> Result<Name, Bad> {
            let index = usize::try_from(id).ok().and_then(|id| id.checked_sub(1));
            index
                .and_then(|index| self.names.get(index))
                .cloned()
                .ok_or_else(|| format!("name {id} is not defined yet"))
        };
        let mut p = Payload::new(payload);
        let selector = name(p.varint()?)?;
        let receiver = p.varint()?;
        let result = p.varint()?;
        let count = p.varint()?;
        let mut args = Vec::new();
        for _ in 0..count {
            args.push(p.value(&name)?);
        }
        let count = p.varint()?;
        let mut answers = Vec::new();
        for _ in 0..count {
            let key = name(p.varint()?)?;
            answers.push((key, p.value(&name)?));
        }
        if !p.done() {
            return Err("bytes past the call's end".into());
        }
        Ok(Call {
            selector,
            receiver,
            result,
            args,
            answers,
        })
    }

    /// The data of blob `hash`: from the stream where a record held it, else from the blob
    /// directory.
    pub fn blob(&mut self, hash: &Hash) -> Result<Vec<u8>, StreamError> {
        if let Some(at) = self.blobs.get(hash).copied() {
            let segment = &mut self.segments[at.segment];
            let io = |error| StreamError::Io(segment.path.clone(), error);
            let back = segment.reader.stream_position().map_err(io)?;
            segment
                .reader
                .seek(SeekFrom::Start(at.offset))
                .map_err(io)?;
            let data = read_vec(&mut segment.reader, at.length).map_err(io)?;
            segment.reader.seek(SeekFrom::Start(back)).map_err(io)?;
            return Ok(data);
        }
        let Some(dir) = &self.blob_dir else {
            return Err(StreamError::MissingBlob(*hash));
        };
        let path = dir.join(super::value::hex(hash));
        match std::fs::read(&path) {
            Ok(data) if Sha256::digest(&data)[..] == hash[..] => Ok(data),
            Ok(_) => Err(StreamError::WrongBlob(path)),
            Err(error) if error.kind() == io::ErrorKind::NotFound => {
                Err(StreamError::MissingBlob(*hash))
            }
            Err(error) => Err(StreamError::Io(path, error)),
        }
    }
}

/// The kind, the payload's length and the head's own length of the record at `offset`, or `None`
/// where the segment ends inside the head.
fn record_head(segment: &mut Segment, offset: u64) -> io::Result<Option<(u8, u64, u64)>> {
    let room = (segment.length - offset).min(11) as usize;
    let mut head = vec![0u8; room];
    segment.reader.read_exact(&mut head)?;
    segment.reader.seek(SeekFrom::Start(offset))?;
    let mut p = Payload::new(&head[1..]);
    let length = match p.varint() {
        Ok(length) => length,
        // The segment ends inside the length.
        Err(_) if room < 11 => return Ok(None),
        Err(why) => return Err(io::Error::new(io::ErrorKind::InvalidData, why)),
    };
    let used = 1 + (head.len() - 1 - p.remaining()) as u64;
    segment.reader.seek(SeekFrom::Start(offset + used))?;
    Ok(Some((head[0], length, used)))
}

fn read_vec(reader: &mut impl Read, length: u64) -> io::Result<Vec<u8>> {
    let mut data = Vec::new();
    reader.take(length).read_to_end(&mut data)?;
    if data.len() as u64 != length {
        return Err(io::ErrorKind::UnexpectedEof.into());
    }
    Ok(data)
}
